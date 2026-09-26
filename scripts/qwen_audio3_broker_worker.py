"""Single-file SecretRef transport for local audio chunks sent to DashScope.

The caller owns model choice, audio preparation, expiry and review policy. This
worker owns only request-root validation, fixed provider endpoints, transport,
and raw provider receipts. It never imports the ChineseASR package or config.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import (HTTPRedirectHandler, ProxyHandler, Request,
                            build_opener)
import uuid
import wave


REQUEST_SCHEMA = "zh_asr.dashscope_transport_request.v1"
RESULT_SCHEMA = "zh_asr.dashscope_transport_result.v1"
HTTP_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
WEBSOCKET_ENDPOINT = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_REQUEST_BYTES = 1024 * 1024
MAX_CHUNK_BYTES = 10 * 1024 * 1024
MAX_CHUNKS = 2000
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class WorkerPolicyError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class ProviderError(RuntimeError):
    def __init__(self, code: str, *, http_status: int | None = None,
                 request_id: str = "", message: str = "", credential_result: str = "Provider-Unavailable"):
        super().__init__(code)
        self.code = code
        self.http_status = http_status
        self.request_id = request_id
        self.message = message
        self.credential_result = credential_result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _scrub(value: Any, api_key: str) -> Any:
    if isinstance(value, str):
        return value.replace(api_key, "[secret-removed]") if api_key else value
    if isinstance(value, list):
        return [_scrub(item, api_key) for item in value]
    if isinstance(value, dict):
        return {_scrub(str(key), api_key): _scrub(item, api_key)
                for key, item in value.items()}
    return value


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise WorkerPolicyError("request_root_escape")
    return resolved


def _parse_deadline(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise WorkerPolicyError("send_deadline_invalid") from exc
    if parsed.tzinfo is None:
        raise WorkerPolicyError("send_deadline_invalid")
    return parsed


def _text_context(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for message in value:
        if (not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), list)):
            return False
        for item in message["content"]:
            if (not isinstance(item, dict) or item.get("type") not in {"input_text", "text"}
                    or not isinstance(item.get("text"), str)
                    or set(item) != {"type", "text"}):
                return False
    return True


def _has_external_media_pointer(value: Any) -> bool:
    if isinstance(value, dict):
        forbidden = {"input_audio", "audio_url", "file_url", "file_urls", "data_uri"}
        if any(str(key).casefold() in forbidden for key in value):
            return True
        return any(_has_external_media_pointer(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_external_media_pointer(item) for item in value)
    return False


def _check_deadline(deadline: datetime | None) -> None:
    if deadline is not None and datetime.now(timezone.utc) >= deadline:
        raise WorkerPolicyError("request_send_deadline_passed")


def _load_request(path: Path, root: Path) -> dict[str, Any]:
    request_path = _inside(path, root)
    if not request_path.is_file() or request_path.stat().st_size > MAX_REQUEST_BYTES:
        raise WorkerPolicyError("request_file_invalid")
    try:
        value = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise WorkerPolicyError("request_json_invalid") from exc
    if not isinstance(value, dict) or value.get("schema") != REQUEST_SCHEMA:
        raise WorkerPolicyError("request_schema_invalid")
    if value.get("cloud_upload_authorized") is not True:
        raise WorkerPolicyError("cloud_upload_authorization_required")
    try:
        job_id = str(uuid.UUID(str(value["job_id"])))
    except (KeyError, ValueError) as exc:
        raise WorkerPolicyError("job_id_invalid") from exc
    if not request_path.name.startswith(job_id + "."):
        raise WorkerPolicyError("request_job_binding_invalid")
    model = value.get("model")
    if not isinstance(model, str) or MODEL_NAME.fullmatch(model) is None:
        raise WorkerPolicyError("model_name_invalid")
    protocol = value.get("protocol")
    if protocol not in {"http", "websocket"}:
        raise WorkerPolicyError("protocol_unsupported")
    if not isinstance(value.get("parameters"), dict):
        raise WorkerPolicyError("parameters_invalid")
    if _has_external_media_pointer(value["parameters"]):
        raise WorkerPolicyError("external_audio_input_forbidden")
    if protocol == "http" and not _text_context(value.get("messages_prefix", [])):
        raise WorkerPolicyError("input_invalid")
    if protocol == "websocket" and (
        not isinstance(value.get("input", {}), dict)
        or not isinstance(value.get("ws_task", {}), dict)
        or any(not isinstance(value["ws_task"].get(key), str)
               for key in ("task_group", "task", "function"))
    ):
        raise WorkerPolicyError("input_invalid")
    if protocol == "websocket":
        incoming = value.get("input", {})
        if _has_external_media_pointer(incoming):
            raise WorkerPolicyError("external_audio_input_forbidden")
        if "context" in incoming and not _text_context(incoming["context"]):
            raise WorkerPolicyError("input_invalid")
    source_hash = value.get("source_audio_sha256")
    if not isinstance(source_hash, str) or SHA256.fullmatch(source_hash) is None:
        raise WorkerPolicyError("source_hash_invalid")
    if type(value.get("source_audio_bytes")) is not int or value["source_audio_bytes"] <= 0:
        raise WorkerPolicyError("source_size_invalid")
    chunks = value.get("chunks")
    if not isinstance(chunks, list) or not 1 <= len(chunks) <= MAX_CHUNKS:
        raise WorkerPolicyError("chunks_invalid")
    parsed_chunks = []
    previous_end = -1
    for expected, chunk in enumerate(chunks, 1):
        if not isinstance(chunk, dict) or chunk.get("index") != expected:
            raise WorkerPolicyError("chunk_sequence_invalid")
        start, end = chunk.get("start_ms"), chunk.get("end_ms")
        if (type(start) is not int or type(end) is not int or start < 0 or end <= start
                or (previous_end >= 0 and start > previous_end)):
            raise WorkerPolicyError("chunk_range_invalid")
        raw_path = Path(str(chunk.get("path") or ""))
        if not raw_path.is_absolute():
            raise WorkerPolicyError("chunk_path_invalid")
        audio = _inside(raw_path, root)
        if not audio.is_file() or audio.suffix.lower() != ".wav" or not 44 <= audio.stat().st_size <= MAX_CHUNK_BYTES:
            raise WorkerPolicyError("chunk_file_invalid")
        try:
            with wave.open(str(audio), "rb") as reader:
                if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (16000, 1, 2):
                    raise WorkerPolicyError("chunk_audio_format_invalid")
                actual_ms = round(reader.getnframes() * 1000 / reader.getframerate())
        except (OSError, wave.Error, EOFError) as exc:
            raise WorkerPolicyError("chunk_audio_format_invalid") from exc
        if actual_ms < 1 or abs(actual_ms - (end - start)) > 4:
            raise WorkerPolicyError("chunk_range_invalid")
        actual_hash = _hash_file(audio)
        if actual_hash != chunk.get("audio_sha256"):
            raise WorkerPolicyError("chunk_hash_mismatch")
        parsed_chunks.append({"index": expected, "start_ms": start, "end_ms": end,
                              "path": audio, "audio_sha256": actual_hash})
        previous_end = end
    value["job_id"] = job_id
    value["chunks"] = parsed_chunks
    value["send_before_utc"] = _parse_deadline(value.get("send_before_utc"))
    return value


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _credential_result(status: int | None, code: str) -> str:
    low = code.casefold()
    if status == 401 or low in {"invalidapikey", "invalid_api_key"}:
        return "Invalid"
    if status == 403:
        return "Permission-Denied"
    if status == 429:
        return "Rate-Limited"
    if status is not None and status >= 500:
        return "Provider-5xx"
    return "Scope-Error"


def _post_http(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = Request(HTTP_ENDPOINT, data=body, method="POST", headers={
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json", "X-DashScope-SSE": "disable"})
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=180) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            status = response.status
    except HTTPError as exc:
        raw = exc.read(256 * 1024)
        try:
            error = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            error = {}
        code = str(error.get("code") or f"http_{exc.code}")
        raise ProviderError(code, http_status=exc.code,
            request_id=str(error.get("request_id") or ""),
            message=str(error.get("message") or ""),
            credential_result=_credential_result(exc.code, code)) from exc
    except (URLError, OSError, TimeoutError) as exc:
        raise ProviderError("network_failure", credential_result="Network-Failure") from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ProviderError("provider_response_too_large")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ProviderError("provider_response_invalid") from exc
    if not isinstance(parsed, dict):
        raise ProviderError("provider_response_invalid")
    return {"http_status": status, "raw_response": parsed,
            "provider_request_id": parsed.get("request_id"),
            "usage": parsed.get("usage")}


def _post_websocket(model: str, chunk: Path, request: dict[str, Any],
                    api_key: str, deadline: datetime | None) -> dict[str, Any]:
    # The stdlib has no WebSocket client. This installed library supplies the
    # HTTPS-verified handshake and binary framing; the URI is fixed above.
    from websockets.sync.client import connect

    task_id = str(uuid.uuid4())
    task = request["ws_task"]
    run = {"header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
           "payload": {"task_group": task["task_group"], "task": task["task"],
                       "function": task["function"], "model": model,
                       "parameters": request["parameters"], "input": request.get("input", {})}}
    events: list[dict[str, Any]] = []

    def receive(socket, timeout: float) -> str:
        try:
            event = json.loads(socket.recv(timeout=timeout))
            kind = str(event["header"]["event"])
        except (ValueError, TypeError, KeyError) as exc:
            raise ProviderError("provider_event_invalid") from exc
        if event["header"].get("task_id") != task_id:
            raise ProviderError("provider_task_mismatch")
        events.append(event)
        if kind == "task-failed":
            header = event["header"]
            raise ProviderError(str(header.get("error_code") or "provider_task_failed"),
                                request_id=task_id,
                                message=str(header.get("error_message") or ""))
        return kind

    try:
        with connect(WEBSOCKET_ENDPOINT,
                     additional_headers={"Authorization": "Bearer " + api_key},
                     proxy=None, open_timeout=30, close_timeout=10,
                     max_size=MAX_RESPONSE_BYTES) as socket:
            _check_deadline(deadline)
            socket.send(json.dumps(run, ensure_ascii=False))
            if receive(socket, 30) != "task-started":
                raise ProviderError("provider_task_not_started")
            with wave.open(str(chunk), "rb") as reader:
                while block := reader.readframes(1600):
                    _check_deadline(deadline)
                    socket.send(block)
                    try:
                        receive(socket, 0.005)
                    except TimeoutError:
                        pass
                    time.sleep(len(block) / 32000)
            socket.send(json.dumps({"header": {"action": "finish-task", "task_id": task_id,
                "streaming": "duplex"}, "payload": {"input": {}}}, ensure_ascii=False))
            end = time.monotonic() + 90
            while time.monotonic() < end:
                try:
                    if receive(socket, 5) == "task-finished":
                        break
                except TimeoutError:
                    continue
            else:
                raise ProviderError("provider_task_timeout", request_id=task_id)
    except WorkerPolicyError:
        raise
    except ProviderError:
        raise
    except Exception as exc:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        close = getattr(exc, "rcvd", None)
        if (getattr(close, "code", None) == 1007 and
                "model not found" in str(getattr(close, "reason", "")).casefold()):
            code = "model_not_found"
        elif getattr(close, "code", None) == 1007:
            code = "provider_close_1007"
        else:
            code = f"http_{status}" if type(status) is int else "network_failure"
        raise ProviderError(code, http_status=status if type(status) is int else None,
                            credential_result=_credential_result(status, code) if type(status) is int
                            else "Network-Failure") from exc
    final_usage = next((item.get("payload", {}).get("usage") for item in reversed(events)
                        if item.get("header", {}).get("event") == "task-finished"), None)
    return {"http_status": 101, "raw_events": events,
            "provider_request_id": task_id, "usage": final_usage}


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cached(path: Path, identity: dict[str, Any], retry_uncertain: bool) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != "zh_asr.dashscope_chunk_checkpoint.v1" or value.get("identity") != identity:
            raise WorkerPolicyError("cloud_checkpoint_invalid")
        if value.get("status") in {"in_flight", "outcome_unknown"}:
            if retry_uncertain:
                return None
            raise WorkerPolicyError("cloud_chunk_outcome_unknown_explicit_retry_required")
        if value.get("status") != "succeeded":
            return None
        record = value.get("result")
        if not isinstance(record, dict) or _json_hash(record) != value.get("result_sha256"):
            raise WorkerPolicyError("cloud_checkpoint_invalid")
        return record
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError) as exc:
        raise WorkerPolicyError("cloud_checkpoint_invalid") from exc


def process_request_file(request_path: Path, *, api_key: str, request_root: Path | None = None,
                         http_transport: Callable[..., dict[str, Any]] = _post_http,
                         websocket_transport: Callable[..., dict[str, Any]] = _post_websocket) -> dict[str, Any]:
    root = (request_root or request_path.parent).resolve()
    result: dict[str, Any] = {"schema": RESULT_SCHEMA, "job_id": "", "status": "failed",
        "error_code": "", "provider_error_code": "", "provider_error_message": "",
        "http_status": None, "credential_result": "Provider-Unavailable",
        "cloud_upload_performed": False, "chunks": [], "started_utc": _utc_now(),
        "completed_utc": "", "plaintext_returned": False, "secret_returned": False}
    try:
        request = _load_request(request_path, root)
        result.update(job_id=request["job_id"], model=request["model"],
            protocol=request["protocol"], provider="aliyun-bailian",
            provider_endpoint=HTTP_ENDPOINT if request["protocol"] == "http" else WEBSOCKET_ENDPOINT,
            source_audio_path=str(request.get("source_audio_path") or ""),
            source_audio_sha256=request["source_audio_sha256"],
            source_audio_bytes=request["source_audio_bytes"],
            selected_channel=request.get("selected_channel"),
            billing_warning=str(request.get("billing_warning") or ""),
            automatic_review=request.get("automatic_review") is True,
            completion="partial", reused_chunks=0, new_requests=0)
        if not api_key or "\x00" in api_key:
            raise WorkerPolicyError("api_key_missing")
        work = _inside(root / (request["job_id"] + ".work"), root)
        work.mkdir(parents=True, exist_ok=True)
        for chunk in request["chunks"]:
            identity = {"model": request["model"], "protocol": request["protocol"],
                "audio_sha256": chunk["audio_sha256"], "index": chunk["index"],
                "start_ms": chunk["start_ms"], "end_ms": chunk["end_ms"],
                "parameters_sha256": _json_hash(request["parameters"]),
                "input_sha256": _json_hash({"input": request.get("input", {}),
                    "messages_prefix": request.get("messages_prefix", []),
                    "ws_task": request.get("ws_task", {})})}
            checkpoint = work / f"provider-chunk-{chunk['index']:06d}.json"
            record = _cached(checkpoint, identity, request.get("retry_uncertain_chunks") is True)
            if record is None:
                _check_deadline(request["send_before_utc"])
                marker = {"schema": "zh_asr.dashscope_chunk_checkpoint.v1",
                          "identity": identity, "status": "in_flight", "started_utc": _utc_now()}
                _write_json_atomic(checkpoint, marker)
                result["cloud_upload_performed"] = True
                result["new_requests"] += 1
                try:
                    if request["protocol"] == "http":
                        audio = "data:audio/wav;base64," + base64.b64encode(
                            chunk["path"].read_bytes()).decode("ascii")
                        messages = list(request.get("messages_prefix", [])) + [{"role": "user",
                            "content": [{"type": "input_audio", "input_audio": {"data": audio}}]}]
                        payload = {"model": request["model"], "input": {"messages": messages},
                                   "parameters": request["parameters"]}
                        response = http_transport(payload, api_key)
                    else:
                        response = websocket_transport(request["model"], chunk["path"],
                            request, api_key, request["send_before_utc"])
                    record = {"index": chunk["index"], "start_ms": chunk["start_ms"],
                        "end_ms": chunk["end_ms"], "audio_sha256": chunk["audio_sha256"],
                        **_scrub(response, api_key)}
                except Exception as exc:
                    ambiguous = not isinstance(exc, (ProviderError, WorkerPolicyError)) or (
                        isinstance(exc, ProviderError) and exc.code in {
                            "network_failure", "provider_response_invalid", "provider_task_timeout"})
                    marker.update(status="outcome_unknown" if ambiguous else "failed",
                                  error_code=_scrub(exc.code, api_key) if isinstance(exc, (ProviderError, WorkerPolicyError))
                                  else type(exc).__name__, completed_utc=_utc_now())
                    _write_json_atomic(checkpoint, marker)
                    raise
                marker.update(status="succeeded", result=record,
                              result_sha256=_json_hash(record), completed_utc=_utc_now())
                _write_json_atomic(checkpoint, marker)
            else:
                result["reused_chunks"] += 1
            result["chunks"].append(record)
            _write_json_atomic(work / "partial-provider-result.json", result)
        result.update(status="succeeded", credential_result="Success", completion="complete")
    except WorkerPolicyError as exc:
        result.update(status="blocked", error_code=exc.code, credential_result="Scope-Error")
    except ProviderError as exc:
        result.update(status="failed", error_code=_scrub(exc.code, api_key),
            provider_error_code=_scrub(exc.code, api_key),
            provider_error_message=_scrub(exc.message[:500], api_key),
            provider_request_id=_scrub(exc.request_id, api_key),
            http_status=exc.http_status, credential_result=exc.credential_result)
    except Exception as exc:
        result.update(status="failed", error_code="worker_internal_error",
                      error_type=type(exc).__name__)
    finally:
        result["completed_utc"] = _utc_now()
    return result


def claim_single_pending_request(request_root: Path) -> Path:
    pending = sorted(request_root.glob("*.pending.json"))
    if len(pending) != 1:
        raise WorkerPolicyError("pending_request_ambiguous")
    source = pending[0]
    target = source.with_name(source.name.replace(".pending.json", ".running.json"))
    if target.exists():
        raise WorkerPolicyError("running_request_exists")
    source.replace(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.request_root).resolve()
    canonical = (Path(__file__).resolve().parents[1] / "outputs" / "cloud-jobs").resolve()
    if root != canonical:
        return 20
    try:
        request_path = claim_single_pending_request(root)
    except WorkerPolicyError:
        return 20
    api_key = os.environ.pop("DASHSCOPE_API_KEY", "")
    try:
        result = process_request_file(request_path, api_key=api_key, request_root=root)
    finally:
        api_key = ""
    job_id = result.get("job_id") or request_path.name.replace(".running.json", "")
    _write_json_atomic(root / (job_id + ".provider.json"), result)
    request_path.replace(request_path.with_name(request_path.name.replace(".running.json", ".done.json")))
    return 0 if result["status"] == "succeeded" else 3


if __name__ == "__main__":
    raise SystemExit(main())

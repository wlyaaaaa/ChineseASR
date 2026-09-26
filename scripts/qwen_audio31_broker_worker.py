"""Hash-pinned SecretRef worker for Qwen Audio 3.x review.

The only production invocation is a managed Password Center target with
--request-root. No API key is accepted on the command line or stored on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable
import uuid
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import qwen_audio3_broker_worker as legacy
from zh_asr.cloud_review import (
    CloudReviewError, auto_cloud_status, compare_text, load_cloud_config,
    local_review_signals, pause_cloud, select_model, stopping_error,
)


REQUEST_SCHEMA = "chineseasr.qwen-audio31-request.v1"
CONFIG_PATH = ROOT / "configs" / "models.yaml"
CANONICAL_ROOT = ROOT / "outputs" / "cloud-jobs"
AUTO_STATE_PATH = CANONICAL_ROOT / "auto-cloud-state.json"


class CloudProviderFailure(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(code)
        self.code = code
        self.detail = detail


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_request(path: Path) -> dict[str, Any]:
    try:
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise CloudReviewError("request_file_invalid")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request is not an object")
        return value
    except (OSError, UnicodeError, ValueError) as exc:
        raise CloudReviewError("request_json_invalid") from exc


def _validated_request(path: Path) -> dict[str, Any]:
    raw = _load_request(path)
    if raw.get("schema") != REQUEST_SCHEMA:
        raise CloudReviewError("request_schema_invalid")
    if raw.get("cloud_upload_authorized") is not True:
        raise CloudReviewError("cloud_upload_authorization_required")
    if raw.get("purpose") not in {"important_evidence", "quality_review"}:
        raise CloudReviewError("request_purpose_invalid")
    if raw["purpose"] == "important_evidence" and raw.get("importance") != "important":
        raise CloudReviewError("importance_required")
    if raw["purpose"] == "quality_review" and "importance" in raw:
        raise CloudReviewError("quality_review_importance_forbidden")
    try:
        raw["job_id"] = str(uuid.UUID(str(raw["job_id"])))
    except (KeyError, ValueError) as exc:
        raise CloudReviewError("job_id_invalid") from exc
    audio = Path(str(raw.get("audio_path") or ""))
    if not audio.is_absolute() or not audio.is_file():
        raise CloudReviewError("audio_file_invalid")
    raw["audio_path"] = audio.resolve()
    local = raw.get("local_out_dir")
    if local:
        local_path = Path(str(local))
        if not local_path.is_absolute() or not local_path.is_dir():
            raise CloudReviewError("local_result_invalid")
        raw["local_out_dir"] = local_path.resolve()
    hotwords = raw.get("hotwords") or {}
    if not isinstance(hotwords, dict) or len(hotwords) > 2000:
        raise CloudReviewError("hotwords_invalid")
    for word, weight in hotwords.items():
        if not isinstance(word, str) or not word.strip() or len(word) > 128 or type(weight) is not int or weight not in (1, 2, 3, 4, 5, 50):
            raise CloudReviewError("hotwords_invalid")
    if sum(weight == 50 for weight in hotwords.values()) > 50:
        raise CloudReviewError("hotwords_invalid")
    raw["hotwords"] = hotwords
    channel = raw.get("channel_index")
    if channel is not None and (type(channel) is not int or channel < 0):
        raise CloudReviewError("channel_index_invalid")
    for flag in ("automatic_review", "speaker_diarization", "keep_dialect"):
        if type(raw.get(flag, False)) is not bool:
            raise CloudReviewError("request_flags_invalid")
    try:
        chunk_sec = int(raw.get("chunk_sec", 180))
        overlap_sec = int(raw.get("overlap_sec", 1))
    except (TypeError, ValueError) as exc:
        raise CloudReviewError("chunk_policy_invalid") from exc
    if not 1 <= chunk_sec <= 180 or not 0 <= overlap_sec < chunk_sec:
        raise CloudReviewError("chunk_policy_invalid")
    raw["chunk_sec"], raw["overlap_sec"] = chunk_sec, overlap_sec
    return raw


def _select_source_channel(source: Path, work: Path, index: int | None) -> tuple[Path, int | None]:
    if index is None:
        return source, None
    ffprobe, ffmpeg = shutil.which("ffprobe"), shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise CloudReviewError("ffmpeg_required")
    try:
        probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=channels", "-of", "json", str(source)],
            capture_output=True, check=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        channels = int(json.loads(probe.stdout)["streams"][0]["channels"])
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as exc:
        raise CloudReviewError("source_channels_unknown") from exc
    if index >= channels:
        raise CloudReviewError("channel_index_invalid")
    work.mkdir(parents=True, exist_ok=True)
    selected = work / f"selected-channel-{index}.wav"
    try:
        subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-map", "0:a:0", "-af", f"pan=mono|c0=c{index}",
            "-ar", "16000", "-c:a", "pcm_s16le", "-map_metadata", "-1", str(selected)],
            capture_output=True, check=True, timeout=3600,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError) as exc:
        raise CloudReviewError("channel_extraction_failed") from exc
    return selected, channels


def _http_payload(audio_path: Path, model: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {"format": "wav", "sample_rate": "16000"}
    if model["api"] == "http_base64":
        if request.get("speaker_diarization"):
            params["speaker_diarization_enabled"] = True
        else:
            params["keep_dialect"] = bool(request.get("keep_dialect"))
    if request["hotwords"] and len("、".join(request["hotwords"])) <= 350:
        params["vocabulary"] = request["hotwords"]
    messages: list[dict[str, Any]] = []
    if request["hotwords"]:
        messages.append({"role": "user", "content": [{"type": "input_text",
                         "text": "相关词语：" + "、".join(request["hotwords"])}]})
    messages.append({"role": "user", "content": [{"type": "input_audio",
                     "input_audio": {"data": legacy._data_uri(audio_path)}}]})
    return {"model": model["id"], "input": {"messages": messages}, "parameters": params}


def _http_call(model: dict[str, Any], api_key: str, audio_path: Path,
               request: dict[str, Any], transport: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json",
               "X-DashScope-SSE": "disable"}
    response = transport(model["endpoint"], headers,
                         _http_payload(audio_path, model, request), 180)
    text, request_id = legacy._response_text(response)
    return {"text": text, "provider_request_id": request_id,
            "usage": response.get("usage"),
            "sentences": response.get("output", {}).get("sentences", [])}


def _websocket_call(model: dict[str, Any], api_key: str, audio_path: Path,
                    request: dict[str, Any]) -> dict[str, Any]:
    """Replay a bounded local PCM slice through the documented realtime protocol."""
    from websockets.sync.client import connect

    task_id = str(uuid.uuid4())
    params: dict[str, Any] = {"format": "pcm", "sample_rate": 16000,
                              "disfluency_removal_enabled": False,
                              "keep_dialect": bool(request.get("keep_dialect")),
                              "intermediate_result_enabled": False}
    if request["hotwords"]:
        params["vocabulary"] = request["hotwords"]
    context = ({"context": [{"role": "user", "content": [{"type": "input_text",
                "text": "相关词语：" + "、".join(request["hotwords"])}]}]}
               if request["hotwords"] and len("、".join(request["hotwords"])) <= 350 else {})
    run = {"header": {"action": "run-task", "task_id": task_id,
                      "streaming": "duplex"},
           "payload": {"task_group": "audio", "task": "asr", "function": "recognition",
                       "model": model["id"], "parameters": params, "input": context}}
    sentences: dict[int, dict[str, Any]] = {}
    usage = None

    def consume(event: Any) -> str:
        nonlocal usage
        try:
            value = json.loads(event)
            header = value["header"]
            kind = header["event"]
        except (TypeError, ValueError, KeyError) as exc:
            raise CloudReviewError("provider_event_invalid") from exc
        if header.get("task_id") != task_id:
            raise CloudReviewError("provider_task_mismatch")
        if kind == "task-failed":
            raise CloudProviderFailure(str(header.get("error_code") or "provider_task_failed"),
                                       str(header.get("error_message") or ""))
        payload = value.get("payload") or {}
        if kind == "result-generated":
            sentence = (payload.get("output") or {}).get("sentence") or {}
            if sentence.get("sentence_end") and sentence.get("sentence_id"):
                sentences[int(sentence["sentence_id"])] = sentence
            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
        if kind == "task-finished":
            usage = payload.get("usage")
        return kind

    with connect(model["endpoint"], additional_headers={"Authorization": "Bearer " + api_key},
                 open_timeout=30, close_timeout=10, max_size=2 * 1024 * 1024) as socket:
        socket.send(json.dumps(run, ensure_ascii=False))
        if consume(socket.recv(timeout=30)) != "task-started":
            raise CloudReviewError("provider_task_not_started")
        with wave.open(str(audio_path), "rb") as reader:
            if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (16000, 1, 2):
                raise CloudReviewError("prepared_audio_format_invalid")
            while block := reader.readframes(1600):
                socket.send(block)
                try:
                    consume(socket.recv(timeout=0.005))
                except TimeoutError:
                    pass
                time.sleep(len(block) / 32000)
        socket.send(json.dumps({"header": {"action": "finish-task", "task_id": task_id,
                           "streaming": "duplex"}, "payload": {"input": {}}}, ensure_ascii=False))
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                if consume(socket.recv(timeout=5)) == "task-finished":
                    break
            except TimeoutError:
                continue
        else:
            raise CloudReviewError("provider_task_timeout")
    text = "".join(str(sentences[key].get("text") or "") for key in sorted(sentences)).strip()
    if not text:
        raise CloudReviewError("provider_transcript_missing")
    return {"text": text, "provider_request_id": task_id,
            "usage": usage, "sentences": list(sentences.values())}


def process_request_file(path: Path, *, api_key: str,
                         http_transport: Callable[..., dict[str, Any]] = legacy._post_json,
                         websocket_transport: Callable[..., dict[str, Any]] = _websocket_call,
                         config_path: Path = CONFIG_PATH,
                         state_path: Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"schema": legacy.QUALITY_REVIEW_RESULT_SCHEMA,
        "status": "failed", "error_code": "", "cloud_upload_performed": False,
        "credential_result": "Provider-Unavailable", "text": "", "chunks": [],
        "started_utc": legacy._utc_now(), "completed_utc": ""}
    config = None
    model_id = ""
    pause_path = state_path or AUTO_STATE_PATH
    try:
        request = _validated_request(path)
        result.update(job_id=request["job_id"], purpose=request["purpose"],
            important_only=request["purpose"] == "important_evidence",
            schema=(legacy.IMPORTANT_RESULT_SCHEMA if request["purpose"] == "important_evidence"
                    else legacy.QUALITY_REVIEW_RESULT_SCHEMA))
        config = load_cloud_config(config_path)
        result["automatic_review"] = bool(request.get("automatic_review"))
        result["source_audio_path"] = str(request["audio_path"])
        if not api_key or "\x00" in api_key:
            raise CloudReviewError("api_key_missing")
        local_dir = request.get("local_out_dir")
        reasons, local_text = local_review_signals(local_dir,
            evidence_status=str(request.get("evidence_status") or ""),
            important=result["important_only"],
            dialect=bool(request.get("keep_dialect"))) if local_dir else (
                (["important_recording"] if result["important_only"] else []), "")
        if request.get("automatic_review") and not reasons:
            raise CloudReviewError("not_a_difficult_recording")
        if request.get("automatic_review"):
            state = auto_cloud_status(pause_path, config)
            if state["status"] == "paused":
                result["pause_reason"] = state["reason"]
                raise CloudReviewError("auto_cloud_paused")
        source = request["audio_path"]
        result["source_audio_sha256"] = _hash(source)
        result["source_audio_bytes"] = source.stat().st_size
        result["source_audio_path"] = str(source)
        result["channel_policy"] = "source_unchanged_local_mono_copy_for_cloud"
        result["review_reasons"] = reasons
        work = path.parent / (request["job_id"] + ".work31")
        selected_audio, source_channels = _select_source_channel(
            source, work, request.get("channel_index"))
        result["selected_channel"] = request.get("channel_index")
        result["source_channels"] = source_channels
        prepared = legacy._prepare_pcm16_mono(selected_audio, work)
        result["prepared_audio_sha256"] = _hash(prepared)
        with wave.open(str(prepared), "rb") as reader:
            duration_sec = reader.getnframes() / reader.getframerate()
        profile = request.get("model_profile") or select_model(config,
            duration_sec=duration_sec, hotwords=request["hotwords"],
            speaker=bool(request.get("speaker_diarization")),
            dialect=bool(request.get("keep_dialect")))
        if profile not in config["models"]:
            raise CloudReviewError("model_profile_invalid")
        model = config["models"][profile]
        model_id = model["id"]
        if model["api"] not in {"http_base64", "websocket_pcm", "legacy_http_base64"}:
            raise CloudReviewError("model_profile_invalid")
        if request.get("speaker_diarization") and model["api"] != "http_base64":
            raise CloudReviewError("speaker_model_required")
        result.update(model=model["id"], model_profile=profile, provider="aliyun-bailian",
                      provider_endpoint=model["endpoint"], source_audio_seconds=duration_sec)
        chunk_sec = min(request["chunk_sec"], int(model["max_chunk_sec"]))
        if request["overlap_sec"] >= chunk_sec:
            raise CloudReviewError("chunk_policy_invalid")
        chunks = legacy._split_wav(prepared, work / "chunks",
            chunk_sec=chunk_sec, overlap_sec=request["overlap_sec"])
        texts = []
        result.update(chunks=[], completion="partial", uploaded_audio_seconds=0.0)
        for chunk in chunks:
            sent_sec = (chunk["end_ms"] - chunk["start_ms"]) / 1000
            result["cloud_upload_performed"] = True
            if model["api"] in {"http_base64", "legacy_http_base64"}:
                answer = _http_call(model, api_key, chunk["path"], request, http_transport)
            else:
                answer = websocket_transport(model, api_key, chunk["path"], request)
            record = {"index": chunk["index"], "start_ms": chunk["start_ms"],
                      "end_ms": chunk["end_ms"], "audio_sha256": _hash(chunk["path"]),
                      "provider_request_id": answer.get("provider_request_id"),
                      "text": answer["text"], "sentences": answer.get("sentences", []),
                      "usage": answer.get("usage")}
            result["chunks"].append(record)
            texts.append(answer["text"])
            result["uploaded_audio_seconds"] += sent_sec
            result["text"] = "\n".join(texts)
            legacy._write_json_atomic(work / "partial-result.json", result)
        result.update(status="succeeded", completion="complete", credential_result="Success",
                      disagreement=compare_text(local_text, result["text"]) if local_text else [],
                      local_text=local_text)
    except (CloudReviewError, legacy.CloudPolicyError) as exc:
        result.update(status="blocked", error_code=exc.code,
                      credential_result="Scope-Error")
    except legacy.CloudApiError as exc:
        result.update(status="failed", error_code=exc.code,
                      credential_result=exc.credential_result,
                      provider_request_id=exc.request_id)
        reason = stopping_error(exc.code, exc.credential_result, str(exc))
        if reason and config and model_id:
            try:
                pause_cloud(pause_path, config, reason=reason,
                            provider_error=exc.code, model=model_id)
                result["auto_pause_reason"] = reason
            except OSError:
                result["pause_state_write_failed"] = True
    except CloudProviderFailure as exc:
        result.update(status="failed", error_code=exc.code,
                      credential_result="Provider-Unavailable")
        reason = stopping_error(exc.code, provider_message=exc.detail)
        if reason and config and model_id:
            try:
                pause_cloud(pause_path, config, reason=reason,
                            provider_error=exc.code, model=model_id)
                result["auto_pause_reason"] = reason
            except OSError:
                result["pause_state_write_failed"] = True
    except Exception as exc:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        code = f"http_{status_code}" if type(status_code) is int else "worker_internal_error"
        close = getattr(exc, "rcvd", None)
        if (getattr(close, "code", None) == 1007 and
                "model not found" in str(getattr(close, "reason", "")).casefold()):
            code = "model_not_found"
        result.update(status="failed", error_code=code,
                      error_type=type(exc).__name__)
        reason = stopping_error(code)
        if reason and config and model_id:
            try:
                pause_cloud(pause_path, config, reason=reason,
                            provider_error=code, model=model_id)
                result["auto_pause_reason"] = reason
            except OSError:
                result["pause_state_write_failed"] = True
    finally:
        result["completed_utc"] = legacy._utc_now()
    return result


def _claim(root: Path) -> Path:
    pending = list(root.glob("*.pending31.json"))
    if len(pending) != 1:
        raise CloudReviewError("pending_request_ambiguous")
    running = pending[0].with_name(pending[0].name.replace(".pending31.json", ".running31.json"))
    if running.exists():
        raise CloudReviewError("running_request_exists")
    pending[0].replace(running)
    return running


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.request_root).resolve()
    if root != CANONICAL_ROOT.resolve():
        return 20
    try:
        path = _claim(root)
    except CloudReviewError:
        return 20
    api_key = os.environ.pop("DASHSCOPE_API_KEY", "")
    try:
        result = process_request_file(path, api_key=api_key)
    finally:
        api_key = ""
    job = result.get("job_id") or path.name.replace(".running31.json", "")
    local_dir = None
    try:
        source_request = _load_request(path)
        if source_request.get("local_out_dir"):
            local_dir = Path(source_request["local_out_dir"])
    except Exception:
        pass
    legacy._write_json_atomic(root / (job + ".result.json"), result)
    if result["status"] == "succeeded":
        (root / (job + ".transcript.txt")).write_text(result["text"] + "\n", encoding="utf-8")
    if local_dir and local_dir.is_dir():
        try:
            sidecar = {"schema": "zh_asr.cloud_review.v1", "status": result["status"],
                       "model": result.get("model"), "review_reasons": result.get("review_reasons", []),
                       "cloud_job_id": job, "cloud_result_path": str(root / (job + ".result.json")),
                       "source_audio_sha256": result.get("source_audio_sha256"),
                       "selected_channel": result.get("selected_channel"),
                       "local_text": result.get("local_text", ""),
                       "cloud_text": result.get("text", ""),
                       "disagreement": result.get("disagreement", []),
                       "error_code": result.get("error_code", ""),
                       "pause_reason": result.get("pause_reason") or result.get("auto_pause_reason"),
                       "cloud_upload_performed": result.get("cloud_upload_performed", False)}
            legacy._write_json_atomic(local_dir / "cloud.review.json", sidecar)
        except (OSError, ValueError):
            pass
    path.replace(path.with_name(path.name.replace(".running31.json", ".done31.json")))
    return 0 if result["status"] == "succeeded" else 3


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid
import wave


REQUEST_SCHEMA = "chineseasr.qwen-audio3-important-request.v1"
RESULT_SCHEMA = "chineseasr.qwen-audio3-important-result.v1"
MODEL = "qwen-audio-3.0-asr-flash"
DEFAULT_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/"
    "aigc/multimodal-generation/generation"
)
ALLOWED_ENDPOINTS = {DEFAULT_ENDPOINT}
DEFAULT_TIMEOUT_SEC = 180
MAX_DATA_URI_BYTES = 10 * 1024 * 1024
MAX_CHUNK_SEC = 180


Transport = Callable[[str, dict[str, str], dict[str, Any], int], dict[str, Any]]


class CloudPolicyError(RuntimeError):
    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


class CloudApiError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        credential_result: str,
        request_id: str = "",
        message: str = "",
        upload_performed: bool = True,
    ) -> None:
        super().__init__(message or code)
        self.code = code
        self.credential_result = credential_result
        self.request_id = request_id
        self.upload_performed = upload_performed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 64 * 1024:
        raise CloudPolicyError("request_file_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CloudPolicyError("request_json_invalid") from exc
    if not isinstance(value, dict):
        raise CloudPolicyError("request_json_invalid")
    return value


def _validate_request(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("schema") != REQUEST_SCHEMA:
        raise CloudPolicyError("request_schema_invalid")
    if request.get("importance") != "important":
        raise CloudPolicyError("importance_required")
    if request.get("cloud_upload_authorized") is not True:
        raise CloudPolicyError("cloud_upload_authorization_required")
    try:
        job_id = str(uuid.UUID(str(request.get("job_id", ""))))
    except ValueError as exc:
        raise CloudPolicyError("job_id_invalid") from exc
    raw_audio = str(request.get("audio_path", "")).strip()
    if not raw_audio:
        raise CloudPolicyError("audio_path_required")
    audio_path = Path(raw_audio)
    if not audio_path.is_absolute():
        raise CloudPolicyError("audio_path_must_be_absolute")
    try:
        chunk_sec = int(request.get("chunk_sec", MAX_CHUNK_SEC))
        overlap_sec = int(request.get("overlap_sec", 1))
    except (TypeError, ValueError) as exc:
        raise CloudPolicyError("chunk_policy_invalid") from exc
    if not 1 <= chunk_sec <= MAX_CHUNK_SEC:
        raise CloudPolicyError("chunk_policy_invalid")
    if overlap_sec < 0 or overlap_sec >= chunk_sec:
        raise CloudPolicyError("chunk_policy_invalid")
    return {
        "job_id": job_id,
        "audio_path": audio_path.resolve(),
        "chunk_sec": chunk_sec,
        "overlap_sec": overlap_sec,
    }


def _wav_info(path: Path) -> tuple[int, int, int, int] | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            return (
                handle.getframerate(),
                handle.getnchannels(),
                handle.getsampwidth(),
                handle.getnframes(),
            )
    except (EOFError, OSError, wave.Error):
        return None


def _prepare_pcm16_mono(source: Path, work_dir: Path) -> Path:
    if not source.is_file():
        raise CloudPolicyError("audio_file_missing")
    info = _wav_info(source)
    if info is not None and info[:3] == (16_000, 1, 2):
        return source
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise CloudPolicyError("ffmpeg_required")
    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / "prepared.16k-mono.wav"
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map_metadata",
        "-1",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=3600)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CloudPolicyError("audio_conversion_failed") from exc
    except subprocess.TimeoutExpired as exc:
        raise CloudPolicyError("audio_conversion_timeout") from exc
    if _wav_info(target) is None or _wav_info(target)[:3] != (16_000, 1, 2):
        raise CloudPolicyError("audio_conversion_invalid")
    return target


def _split_wav(
    source: Path,
    chunks_dir: Path,
    *,
    chunk_sec: int,
    overlap_sec: int,
) -> list[dict[str, Any]]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as handle:
        frame_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        total_frames = handle.getnframes()
        if (frame_rate, channels, sample_width) != (16_000, 1, 2):
            raise CloudPolicyError("prepared_audio_format_invalid")
        chunk_frames = frame_rate * chunk_sec
        step_frames = frame_rate * (chunk_sec - overlap_sec)
        chunks: list[dict[str, Any]] = []
        start_frame = 0
        index = 1
        while start_frame < total_frames:
            end_frame = min(start_frame + chunk_frames, total_frames)
            handle.setpos(start_frame)
            frames = handle.readframes(end_frame - start_frame)
            target = chunks_dir / f"chunk-{index:06d}.wav"
            with wave.open(str(target), "wb") as output:
                output.setnchannels(channels)
                output.setsampwidth(sample_width)
                output.setframerate(frame_rate)
                output.writeframes(frames)
            chunks.append(
                {
                    "index": index,
                    "start_ms": round(start_frame * 1000 / frame_rate),
                    "end_ms": round(end_frame * 1000 / frame_rate),
                    "path": target,
                }
            )
            if end_frame >= total_frames:
                break
            start_frame += step_frames
            index += 1
    if not chunks:
        raise CloudPolicyError("audio_empty")
    return chunks


def _data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    value = f"data:{mime_type};base64,{encoded}"
    if len(value.encode("ascii")) > MAX_DATA_URI_BYTES:
        raise CloudPolicyError("encoded_chunk_too_large")
    return value


def _payload(path: Path) -> dict[str, Any]:
    return {
        "model": MODEL,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": _data_uri(path)},
                        }
                    ],
                }
            ]
        },
        "parameters": {"format": "wav", "sample_rate": "16000"},
    }


def _bounded_provider_error(value: object) -> str:
    text = str(value or "").strip().replace("\r", " ").replace("\n", " ")
    return text[:500]


def _provider_error_details(body: bytes) -> tuple[str, str, str]:
    try:
        parsed = json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return "", "", ""
    if not isinstance(parsed, dict):
        return "", "", ""
    return (
        _bounded_provider_error(parsed.get("code")),
        _bounded_provider_error(parsed.get("message")),
        _bounded_provider_error(parsed.get("request_id")),
    )


def _credential_result(http_status: int, provider_code: str) -> str:
    normalized = provider_code.casefold()
    if http_status == 401 or normalized in {
        "invalidapikey",
        "invalid_api_key",
        "invalidauthentication",
    }:
        return "Invalid"
    if http_status == 403:
        return "Permission-Denied"
    if http_status == 429:
        return "Rate-Limited"
    if http_status >= 500:
        return "Provider-5xx"
    return "Scope-Error"


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    if url not in ALLOWED_ENDPOINTS:
        raise CloudPolicyError("provider_endpoint_not_allowed")
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(2 * 1024 * 1024)
    except HTTPError as exc:
        raw = exc.read(256 * 1024)
        provider_code, message, request_id = _provider_error_details(raw)
        raise CloudApiError(
            provider_code or f"http_{exc.code}",
            credential_result=_credential_result(exc.code, provider_code),
            request_id=request_id,
            message=message,
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise CloudApiError(
            "network_failure",
            credential_result="Network-Failure",
            message=type(exc).__name__,
        ) from exc
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CloudApiError(
            "provider_response_invalid",
            credential_result="Provider-Unavailable",
        ) from exc
    if not isinstance(parsed, dict):
        raise CloudApiError(
            "provider_response_invalid",
            credential_result="Provider-Unavailable",
        )
    return parsed


def _response_text(response: dict[str, Any]) -> tuple[str, str]:
    output = response.get("output")
    text = ""
    if isinstance(output, dict):
        text = str(output.get("text", "")).strip()
        nested = output.get("output")
        if not text and isinstance(nested, dict):
            sentence = nested.get("sentence")
            if isinstance(sentence, dict):
                text = str(sentence.get("text", "")).strip()
    request_id = _bounded_provider_error(response.get("request_id"))
    if not text:
        raise CloudApiError(
            "provider_transcript_missing",
            credential_result="Provider-Unavailable",
            request_id=request_id,
        )
    return text, request_id


def _merge_texts(texts: list[str]) -> str:
    merged: list[str] = []
    previous = ""
    for raw in texts:
        current = raw.strip()
        overlap = 0
        max_overlap = min(len(previous), len(current), 200)
        for size in range(max_overlap, 1, -1):
            if previous.endswith(current[:size]):
                overlap = size
                break
        remaining = current[overlap:].strip()
        if remaining:
            merged.append(remaining)
        previous = current
    return "\n".join(merged)


def _base_result(job_id: str = "") -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "job_id": job_id,
        "model": MODEL,
        "provider": "aliyun-bailian",
        "provider_endpoint": DEFAULT_ENDPOINT,
        "important_only": True,
        "status": "failed",
        "error_code": "",
        "error_message": "",
        "credential_result": "Provider-Unavailable",
        "cloud_upload_performed": False,
        "text": "",
        "chunks": [],
        "started_utc": _utc_now(),
        "completed_utc": "",
    }


def process_request_file(
    request_path: Path,
    *,
    api_key: str,
    transport: Transport = _post_json,
) -> dict[str, Any]:
    result = _base_result()
    try:
        request = _load_json_object(request_path)
        validated = _validate_request(request)
        result["job_id"] = validated["job_id"]
        if not api_key or "\x00" in api_key:
            raise CloudApiError(
                "api_key_missing",
                credential_result="Permission-Denied",
                upload_performed=False,
            )
        source: Path = validated["audio_path"]
        result["source_audio_sha256"] = _sha256_file(source)
        result["source_audio_bytes"] = source.stat().st_size
        work_dir = request_path.parent / f"{validated['job_id']}.work"
        prepared = _prepare_pcm16_mono(source, work_dir)
        chunks = _split_wav(
            prepared,
            work_dir / "chunks",
            chunk_sec=validated["chunk_sec"],
            overlap_sec=validated["overlap_sec"],
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "disable",
        }
        texts: list[str] = []
        chunk_results: list[dict[str, Any]] = []
        for chunk in chunks:
            result["cloud_upload_performed"] = True
            response = transport(
                DEFAULT_ENDPOINT,
                headers,
                _payload(chunk["path"]),
                DEFAULT_TIMEOUT_SEC,
            )
            text, request_id = _response_text(response)
            texts.append(text)
            chunk_results.append(
                {
                    "index": chunk["index"],
                    "start_ms": chunk["start_ms"],
                    "end_ms": chunk["end_ms"],
                    "audio_sha256": _sha256_file(chunk["path"]),
                    "provider_request_id": request_id,
                    "text": text,
                }
            )
        result.update(
            {
                "status": "succeeded",
                "credential_result": "Success",
                "text": _merge_texts(texts),
                "chunks": chunk_results,
            }
        )
    except CloudPolicyError as exc:
        result.update(
            {
                "status": "blocked",
                "error_code": exc.code,
                "error_message": _bounded_provider_error(exc),
                "credential_result": "Scope-Error",
            }
        )
    except CloudApiError as exc:
        result.update(
            {
                "status": "failed",
                "error_code": exc.code,
                "error_message": _bounded_provider_error(exc),
                "credential_result": exc.credential_result,
                "provider_request_id": exc.request_id,
                "cloud_upload_performed": bool(
                    result["cloud_upload_performed"] or exc.upload_performed
                ),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error_code": "worker_internal_error",
                "error_message": type(exc).__name__,
                "credential_result": "Provider-Unavailable",
            }
        )
    finally:
        result["completed_utc"] = _utc_now()
    return result


def claim_single_pending_request(request_root: Path) -> Path:
    request_root.mkdir(parents=True, exist_ok=True)
    pending = sorted(request_root.glob("*.pending.json"))
    if not pending:
        raise CloudPolicyError("pending_request_missing")
    if len(pending) != 1:
        raise CloudPolicyError("pending_request_ambiguous")
    source = pending[0]
    target = source.with_name(source.name.removesuffix(".pending.json") + ".running.json")
    if target.exists():
        raise CloudPolicyError("running_request_already_exists")
    source.replace(target)
    return target


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-root", required=True)
    args = parser.parse_args(argv)
    request_root = Path(args.request_root).resolve()
    try:
        request_path = claim_single_pending_request(request_root)
    except CloudPolicyError:
        return 20
    api_key = os.environ.pop("DASHSCOPE_API_KEY", "")
    try:
        result = process_request_file(request_path, api_key=api_key)
    finally:
        api_key = ""
    job_id = result.get("job_id") or request_path.name.removesuffix(".running.json")
    result_path = request_root / f"{job_id}.result.json"
    _write_json_atomic(result_path, result)
    if result.get("status") == "succeeded":
        transcript_path = request_root / f"{job_id}.transcript.txt"
        transcript_path.write_text(str(result.get("text", "")) + "\n", encoding="utf-8")
    done_path = request_path.with_name(
        request_path.name.removesuffix(".running.json") + ".done.json"
    )
    request_path.replace(done_path)
    if result.get("status") == "succeeded":
        return 0
    if result.get("status") == "blocked":
        return 3
    return 4


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
from typing import Any, Callable, Sequence
import wave

from zh_asr.config import EngineSpec
from zh_asr.text_normalizer import to_simplified


REQUEST_SCHEMA = "zh_asr.firered_worker.request.v1"
RESPONSE_SCHEMA = "zh_asr.firered_worker.response.v1"
FIRERED_OFFICIAL_MAX_AUDIO_SEC = 40.0


class InvalidFireRedAudio(ValueError):
    """The input cannot be passed safely to FireRedASR2-LLM."""


class FireRedWorkerError(RuntimeError):
    """The isolated FireRed worker failed or broke its protocol."""


class FireRedWorkerTimeout(FireRedWorkerError):
    """The isolated FireRed worker exceeded its configured deadline."""


@dataclass(frozen=True)
class FireRedWaveInfo:
    sample_rate: int
    channels: int
    sample_width_bytes: int
    frame_count: int
    duration_sec: float


def inspect_firered_wav(
    audio_path: Path | str,
    *,
    max_audio_sec: float = FIRERED_OFFICIAL_MAX_AUDIO_SEC,
) -> FireRedWaveInfo:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"FireRed input audio not found: {path}")
    effective_max = _effective_max_audio_sec(max_audio_sec)
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            compression = handle.getcomptype()
    except (EOFError, wave.Error) as exc:
        raise InvalidFireRedAudio(f"FireRed input must be a readable PCM WAV file: {path}: {exc}") from exc

    if compression != "NONE":
        raise InvalidFireRedAudio(f"FireRed input must be uncompressed PCM WAV, got {compression}: {path}")
    if sample_rate != 16_000:
        raise InvalidFireRedAudio(f"FireRed input must be 16000 Hz, got {sample_rate} Hz: {path}")
    if channels != 1:
        raise InvalidFireRedAudio(f"FireRed input must be mono, got {channels} channels: {path}")
    if sample_width != 2:
        raise InvalidFireRedAudio(
            f"FireRed input must be 16-bit PCM, got {sample_width * 8}-bit samples: {path}"
        )

    duration_sec = frame_count / sample_rate
    if duration_sec > effective_max:
        raise InvalidFireRedAudio(
            f"FireRed input duration {duration_sec:.3f}s exceeds the {effective_max:g}s limit: {path}"
        )
    expected_payload_bytes = frame_count * channels * sample_width
    try:
        with wave.open(str(path), "rb") as handle:
            actual_payload_bytes = len(handle.readframes(frame_count))
    except (EOFError, wave.Error) as exc:
        raise InvalidFireRedAudio(f"FireRed PCM payload could not be read: {path}: {exc}") from exc
    if actual_payload_bytes != expected_payload_bytes:
        raise InvalidFireRedAudio(
            f"FireRed PCM payload is truncated: expected {expected_payload_bytes} bytes, "
            f"read {actual_payload_bytes}: {path}"
        )
    return FireRedWaveInfo(
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        frame_count=frame_count,
        duration_sec=duration_sec,
    )


class FireRedWorkerAdapter:
    name = "firered-worker"

    def build_model(
        self,
        spec: EngineSpec,
        device: str,
        cache_dir: Path,
        model_aliases: dict[str, str],
    ) -> "FireRedWorkerModel":
        options = dict(spec.options or {})
        runtime = str(options.get("runtime", "")).strip().lower()
        if runtime not in {"", "native", "wsl"}:
            raise ValueError("FireRed option 'runtime' must be 'native' or 'wsl'.")
        default_path_style = "wsl" if runtime == "wsl" else "native"
        path_style = str(options.get("path_style", default_path_style)).strip().lower()
        if path_style not in {"native", "wsl"}:
            raise ValueError("FireRed option 'path_style' must be 'native' or 'wsl'.")

        command = _worker_command(options, path_style)
        model_dir = _resolve_model_dir(spec, cache_dir, model_aliases, path_style)
        source_dir = _resolve_optional_worker_path(options.get("source_dir"), path_style)
        timeout_sec = _positive_float(options.get("timeout_sec", 600), "timeout_sec")
        max_audio_sec = _effective_max_audio_sec(
            _positive_float(
                options.get("max_audio_sec", FIRERED_OFFICIAL_MAX_AUDIO_SEC),
                "max_audio_sec",
            )
        )
        inference_options = {
            key: options[key]
            for key in (
                "batch_size",
                "beam_size",
                "decode_min_len",
                "repetition_penalty",
                "llm_length_penalty",
                "temperature",
                "use_half",
                "source_revision",
                "model_revision",
            )
            if key in options
        }
        return FireRedWorkerModel(
            command=command,
            model_dir=model_dir,
            source_dir=source_dir,
            device=device,
            timeout_sec=timeout_sec,
            max_audio_sec=max_audio_sec,
            path_style=path_style,
            inference_options=inference_options,
        )


class FireRedWorkerModel:
    def __init__(
        self,
        *,
        command: Sequence[str],
        model_dir: str,
        source_dir: str | None,
        device: str,
        timeout_sec: float,
        max_audio_sec: float,
        path_style: str,
        inference_options: dict[str, Any],
        process_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.command = list(command)
        self.model_dir = model_dir
        self.source_dir = source_dir
        self.device = device
        self.timeout_sec = timeout_sec
        self.max_audio_sec = max_audio_sec
        self.path_style = path_style
        self.inference_options = dict(inference_options)
        self._process_runner = process_runner

    def generate(self, input: str, **_: Any) -> list[dict[str, Any]]:
        return self.generate_many([input])

    def generate_many(self, inputs: Sequence[str]) -> list[dict[str, Any]]:
        if not inputs:
            raise ValueError("FireRed generate_many requires at least one audio path.")
        audio_paths = [Path(value).resolve() for value in inputs]
        infos = [
            inspect_firered_wav(audio, max_audio_sec=self.max_audio_sec)
            for audio in audio_paths
        ]
        audio_requests = [
            {
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "sample_width_bytes": info.sample_width_bytes,
                "frame_count": info.frame_count,
                "duration_sec": info.duration_sec,
                "max_audio_sec": self.max_audio_sec,
            }
            for info in infos
        ]
        request: dict[str, Any] = {
            "schema": REQUEST_SCHEMA,
            "audio_paths": [
                _to_worker_path(str(audio), self.path_style)
                for audio in audio_paths
            ],
            "model_dir": self.model_dir,
            "device": self.device,
            "options": self.inference_options,
            "audios": audio_requests,
        }
        # Keep the v1 single-input fields for older independently installed workers.
        if len(audio_paths) == 1:
            request["audio_path"] = request["audio_paths"][0]
            request["audio"] = audio_requests[0]
        if self.source_dir:
            request["source_dir"] = self.source_dir

        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        if self.path_style == "wsl":
            env.setdefault("WSL_UTF8", "1")
        kwargs: dict[str, Any] = {
            "input": json.dumps(request, ensure_ascii=False),
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "capture_output": True,
            "timeout": self.timeout_sec,
            "check": False,
            "env": env,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        runner = self._process_runner or subprocess.run
        try:
            completed = runner(self.command, **kwargs)
        except subprocess.TimeoutExpired as exc:
            label = (
                audio_paths[0].name
                if len(audio_paths) == 1
                else f"{len(audio_paths)} inputs beginning with {audio_paths[0].name}"
            )
            raise FireRedWorkerTimeout(
                f"FireRed worker timed out after {self.timeout_sec:g}s for {label}."
            ) from exc
        except OSError as exc:
            raise FireRedWorkerError(
                f"Could not start FireRed worker command {self.command!r}: {type(exc).__name__}: {exc}"
            ) from exc

        if completed.returncode != 0:
            detail = _diagnostic_excerpt(completed.stderr or completed.stdout)
            raise FireRedWorkerError(
                f"FireRed worker exited with exit code {completed.returncode}"
                f"{': ' + detail if detail else '.'}"
            )
        response = _parse_worker_response(completed.stdout)
        if not isinstance(response.get("ok"), bool):
            raise FireRedWorkerError(
                f"FireRed worker protocol field 'ok' must be boolean, got "
                f"{type(response.get('ok')).__name__}."
            )
        if response["ok"] is not True:
            error = response.get("error")
            if isinstance(error, dict):
                error_type = str(error.get("type", "WorkerError"))
                message = str(error.get("message", "Unknown FireRed worker error"))
                detail = f"{error_type}: {message}"
            else:
                detail = str(error or "Unknown FireRed worker error")
            raise FireRedWorkerError(f"FireRed worker reported failure: {detail}")
        return _normalize_worker_result(response.get("result"))


def _worker_command(options: dict[str, Any], path_style: str) -> list[str]:
    configured = options.get("worker_command")
    runtime = str(options.get("runtime", "")).strip().lower()
    if configured is None and runtime == "wsl":
        distribution = str(options.get("wsl_distribution", "Ubuntu")).strip()
        python_path = str(options.get("python_path", "python3")).strip()
        if not distribution or not python_path:
            raise ValueError(
                "FireRed WSL options 'wsl_distribution' and 'python_path' must not be empty."
            )
        prefix = ["wsl.exe", "-d", distribution, "--", python_path]
    elif configured is None:
        prefix = [sys.executable]
    elif isinstance(configured, (list, tuple)):
        prefix = [str(item) for item in configured if str(item)]
    elif isinstance(configured, str):
        prefix = shlex.split(configured, posix=os.name != "nt")
    else:
        raise ValueError("FireRed option 'worker_command' must be a non-empty string or list.")
    if not prefix:
        raise ValueError("FireRed option 'worker_command' must not be empty.")

    worker_value = options.get("worker_script")
    if worker_value is None:
        worker_script = _project_root() / "runtime" / "firered_worker.py"
        translated = _to_worker_path(str(worker_script.resolve()), path_style)
    else:
        raw_worker_script = str(worker_value).strip()
        if path_style == "wsl" and raw_worker_script.startswith("/"):
            translated = raw_worker_script
            worker_script = None
        else:
            worker_script = Path(raw_worker_script)
            if not worker_script.is_absolute():
                worker_script = _project_root() / worker_script
            translated = _to_worker_path(str(worker_script.resolve()), path_style)
    if worker_script is not None and path_style == "native" and not worker_script.is_file():
        raise FileNotFoundError(f"FireRed worker script not found: {worker_script}")
    if any("{worker_script}" in token for token in prefix):
        return [token.replace("{worker_script}", translated) for token in prefix]
    return [*prefix, translated]


def _resolve_model_dir(
    spec: EngineSpec,
    cache_dir: Path,
    model_aliases: dict[str, str],
    path_style: str,
) -> str:
    options = spec.options or {}
    configured = str(options.get("model_dir", "")).strip()
    raw = configured or model_aliases.get(spec.model, spec.model)
    if path_style == "wsl" and raw.startswith("/"):
        return raw

    direct = Path(raw).expanduser()
    candidates: list[Path] = []
    if direct.is_absolute():
        candidates.append(direct)
    else:
        candidates.extend(
            [
                _project_root() / direct,
                cache_dir / Path(*raw.replace("\\", "/").split("/")),
            ]
        )
    for candidate in candidates:
        if candidate.is_dir():
            return _to_worker_path(str(candidate.resolve()), path_style)
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"FireRed model directory is not available locally. Checked: {rendered}. "
        "Download weights separately, then set engine options.model_dir; the adapter never downloads models."
    )


def _resolve_optional_worker_path(value: Any, path_style: str) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    raw = str(value).strip()
    if path_style == "wsl" and raw.startswith("/"):
        return raw
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _project_root() / path
    if not path.is_dir():
        raise FileNotFoundError(f"FireRed source directory not found: {path}")
    return _to_worker_path(str(path.resolve()), path_style)


def _to_worker_path(value: str, path_style: str) -> str:
    if path_style == "native":
        return value
    if path_style != "wsl":
        raise ValueError(f"Unsupported FireRed path style: {path_style}")
    normalized = value.replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", normalized)
    if match:
        drive, tail = match.groups()
        return str(PurePosixPath("/mnt", drive.lower(), tail))
    if normalized.startswith("//"):
        raise ValueError(f"UNC paths are not supported by the FireRed WSL worker: {value}")
    if normalized.startswith("/"):
        return normalized
    raise ValueError(f"FireRed WSL paths must be absolute Windows or POSIX paths: {value}")


def _parse_worker_response(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise FireRedWorkerError("FireRed worker returned empty stdout.")
    candidates = [text, *reversed([line for line in text.splitlines() if line.strip()])]
    response: Any = None
    parse_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        try:
            response = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            parse_error = exc
    if not isinstance(response, dict):
        excerpt = _diagnostic_excerpt(text)
        raise FireRedWorkerError(
            f"FireRed worker returned invalid JSON: {parse_error}. stdout={excerpt!r}"
        )
    if response.get("schema") != RESPONSE_SCHEMA:
        raise FireRedWorkerError(
            f"FireRed worker protocol mismatch: expected {RESPONSE_SCHEMA}, got {response.get('schema')!r}."
        )
    return response


def _normalize_worker_result(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        items: list[Any] = [{"text": value}]
    elif isinstance(value, dict):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        raise FireRedWorkerError(
            f"FireRed worker result must be a string, object, or list, got {type(value).__name__}."
        )

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            current: dict[str, Any] = {"text": item}
        elif isinstance(item, dict):
            current = dict(item)
        else:
            raise FireRedWorkerError(
                f"FireRed worker result item {index} must be a string or object, "
                f"got {type(item).__name__}."
            )
        original_text = str(current.get("text", ""))
        text = to_simplified(original_text)
        current["text"] = text
        if original_text != text:
            current["original_text"] = original_text
        normalized.append(current)
    return normalized


def _positive_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"FireRed option '{label}' must be a positive number, got {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"FireRed option '{label}' must be positive, got {parsed:g}.")
    if not math.isfinite(parsed):
        raise ValueError(f"FireRed option '{label}' must be finite, got {parsed!r}.")
    return parsed


def _effective_max_audio_sec(configured: float) -> float:
    try:
        value = float(configured)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"FireRed max audio duration must be a positive number, got {configured!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"FireRed max audio duration must be finite, got {value!r}.")
    if value <= 0:
        raise ValueError(f"FireRed max audio duration must be positive, got {value:g}.")
    return min(value, FIRERED_OFFICIAL_MAX_AUDIO_SEC)


def _diagnostic_excerpt(value: str, limit: int = 2_000) -> str:
    compact = " ".join(str(value).strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]

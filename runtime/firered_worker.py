from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import traceback
from typing import Any, Callable
import wave


REQUEST_SCHEMA = "zh_asr.firered_worker.request.v1"
RESPONSE_SCHEMA = "zh_asr.firered_worker.response.v1"
OFFICIAL_MAX_AUDIO_SEC = 40.0
MODEL_RECEIPT_SCHEMA = "zh_asr.model_receipt.v1"
MODEL_REPOSITORY = "FireRedTeam/FireRedASR2-LLM"
MODEL_REQUIRED_FILES = (
    "asr_encoder.pth.tar",
    "cmvn.ark",
    "model.pth.tar",
    "Qwen2-7B-Instruct/config.json",
    "Qwen2-7B-Instruct/generation_config.json",
    "Qwen2-7B-Instruct/merges.txt",
    "Qwen2-7B-Instruct/model.safetensors.index.json",
    "Qwen2-7B-Instruct/model-00001-of-00004.safetensors",
    "Qwen2-7B-Instruct/model-00002-of-00004.safetensors",
    "Qwen2-7B-Instruct/model-00003-of-00004.safetensors",
    "Qwen2-7B-Instruct/model-00004-of-00004.safetensors",
    "Qwen2-7B-Instruct/tokenizer_config.json",
    "Qwen2-7B-Instruct/tokenizer.json",
    "Qwen2-7B-Instruct/vocab.json",
)
MODEL_RECEIPT_FIELDS = frozenset(
    {"schema", "repository", "revision", "created_utc", "files"}
)
MODEL_FILE_RECORD_FIELDS = frozenset({"path", "bytes", "sha256"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIB = 1024**3
HALF_MIN_MEMORY_BYTES = 28 * GIB
HALF_MIN_COMBINED_BYTES = 34 * GIB
HALF_MIN_AVAILABLE_MEMORY_BYTES = 18 * GIB
HALF_MIN_AVAILABLE_COMBINED_BYTES = 22 * GIB
FP32_MIN_MEMORY_BYTES = 40 * GIB
FP32_MIN_COMBINED_BYTES = 48 * GIB
FP32_MIN_AVAILABLE_MEMORY_BYTES = 36 * GIB
FP32_MIN_AVAILABLE_COMBINED_BYTES = 44 * GIB


def run_request(
    request: dict[str, Any],
    *,
    model_loader: Callable[[str, str, dict[str, Any], str | None], Any] | None = None,
) -> dict[str, Any]:
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError(
            f"FireRed request schema mismatch: expected {REQUEST_SCHEMA}, got {request.get('schema')!r}."
        )
    audio_paths = _request_audio_paths(request)
    model_dir = _required_path(request, "model_dir", directory=True)
    source_dir = _optional_directory(request.get("source_dir"))
    device = str(request.get("device", "cuda:0")).strip() or "cuda:0"
    options = request.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("FireRed request options must be an object.")

    audio_request = request.get("audio")
    if audio_request is None:
        audio_requests = request.get("audios", [])
        if isinstance(audio_requests, list) and audio_requests:
            audio_request = audio_requests[0]
        else:
            audio_request = {}
    if not isinstance(audio_request, dict):
        raise ValueError("FireRed request audio metadata must be an object.")
    requested_max = audio_request.get("max_audio_sec", OFFICIAL_MAX_AUDIO_SEC)
    try:
        effective_max = min(float(requested_max), OFFICIAL_MAX_AUDIO_SEC)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"FireRed max_audio_sec must be numeric, got {requested_max!r}.") from exc
    if not math.isfinite(effective_max):
        raise ValueError(f"FireRed max_audio_sec must be finite, got {effective_max!r}.")
    if effective_max <= 0:
        raise ValueError(f"FireRed max_audio_sec must be positive, got {effective_max:g}.")
    infos = [
        inspect_wav(audio_path, max_audio_sec=effective_max)
        for audio_path in audio_paths
    ]
    batch_size = options.get("batch_size", 1)
    try:
        parsed_batch_size = float(batch_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"FireRedASR2-LLM isolated worker requires batch_size=1, got {batch_size!r}."
        ) from exc
    if isinstance(batch_size, bool) or parsed_batch_size != 1:
        raise ValueError(
            f"FireRedASR2-LLM isolated worker requires batch_size=1, got {batch_size!r}."
        )
    loader = model_loader or _load_model
    model = loader(str(model_dir), device, dict(options), str(source_dir) if source_dir else None)
    diagnostics = _model_runtime_diagnostics(model)
    try:
        result: list[Any] = []
        for audio_path in audio_paths:
            uttid = audio_path.stem or "audio"
            current = model.transcribe([uttid], [str(audio_path)])
            if isinstance(current, list):
                result.extend(current)
            else:
                result.append(current)
    finally:
        del model
        gc.collect()
        _empty_cuda_cache(device)
    if not isinstance(result, (list, dict, str)):
        raise TypeError(
            f"FireRed transcribe returned unsupported result type: {type(result).__name__}."
        )
    response = {
        "schema": RESPONSE_SCHEMA,
        "ok": True,
        "result": result,
        "audio": infos[0],
        "audios": infos,
    }
    if diagnostics:
        response["diagnostics"] = diagnostics
        _attach_runtime_diagnostics(result, diagnostics)
    return response


def inspect_wav(path: Path, *, max_audio_sec: float = OFFICIAL_MAX_AUDIO_SEC) -> dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            compression = handle.getcomptype()
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"FireRed input must be a readable PCM WAV file: {path}: {exc}") from exc
    if compression != "NONE":
        raise ValueError(f"FireRed input must be uncompressed PCM WAV, got {compression}: {path}")
    if sample_rate != 16_000:
        raise ValueError(f"FireRed input must be 16000 Hz, got {sample_rate} Hz: {path}")
    if channels != 1:
        raise ValueError(f"FireRed input must be mono, got {channels} channels: {path}")
    if sample_width != 2:
        raise ValueError(f"FireRed input must be 16-bit PCM, got {sample_width * 8}-bit: {path}")
    duration_sec = frame_count / sample_rate
    if duration_sec > max_audio_sec:
        raise ValueError(
            f"FireRed input duration {duration_sec:.3f}s exceeds the {max_audio_sec:g}s limit: {path}"
        )
    expected_payload_bytes = frame_count * channels * sample_width
    try:
        with wave.open(str(path), "rb") as handle:
            actual_payload_bytes = len(handle.readframes(frame_count))
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"FireRed PCM payload could not be read: {path}: {exc}") from exc
    if actual_payload_bytes != expected_payload_bytes:
        raise ValueError(
            f"FireRed PCM payload is truncated: expected {expected_payload_bytes} bytes, "
            f"read {actual_payload_bytes}: {path}"
        )
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }


def _load_model(
    model_dir: str,
    device: str,
    options: dict[str, Any],
    source_dir: str | None,
) -> Any:
    use_half = bool(options.get("use_half", False))
    _preflight_wsl_memory(device=device, use_half=use_half)
    _verify_model_revision(Path(model_dir), options.get("model_revision"))
    if not source_dir:
        raise RuntimeError(
            "FireRedASR2S source_dir is required so the pinned source revision "
            "and clean working tree can be verified."
        )
    _verify_source_revision(Path(source_dir), options.get("source_revision"))
    source = str(Path(source_dir).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
    except ImportError as exc:
        raise RuntimeError(
            "FireRedASR2S source is unavailable. Set options.source_dir to the local "
            "FireRedASR2S checkout and run scripts/setup-firered.ps1."
        ) from exc

    config_kwargs: dict[str, Any] = {
        "use_gpu": device.lower().startswith("cuda"),
        "beam_size": int(options.get("beam_size", 3)),
        "decode_min_len": options.get("decode_min_len", 0),
        "repetition_penalty": options.get("repetition_penalty", 3.0),
        "llm_length_penalty": options.get("llm_length_penalty", 1.0),
        "temperature": options.get("temperature", 1.0),
    }
    if "use_half" in options:
        config_kwargs["use_half"] = bool(options["use_half"])
    config = FireRedAsr2Config(**config_kwargs)
    with _temporary_initial_llm_load_dtype(
        device=device,
        use_half=use_half,
    ) as load_dtype_name:
        model = FireRedAsr2.from_pretrained("llm", model_dir, config)
    if load_dtype_name:
        setattr(model, "_zh_asr_llm_load_dtype", load_dtype_name)
        print(
            f"ChineseASR FireRed initial LLM load dtype: {load_dtype_name}",
            file=sys.stderr,
            flush=True,
        )
    return model


def _preflight_wsl_memory(
    *,
    device: str,
    use_half: bool,
    meminfo_path: Path = Path("/proc/meminfo"),
) -> None:
    if not device.lower().startswith("cuda") or sys.platform != "linux":
        return
    try:
        text = meminfo_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            "FireRed WSL memory preflight could not read configured and currently "
            f"available memory information at {meminfo_path}; refusing the CUDA "
            "load. Run `wsl -d Ubuntu -- free -h`, restore access to /proc/meminfo, "
            "and retry."
        ) from exc
    try:
        memory = _parse_proc_meminfo(text)
    except ValueError as exc:
        raise RuntimeError(
            f"FireRed could not parse WSL memory information at {meminfo_path}: {exc}"
        ) from exc

    mem_total = memory["MemTotal"]
    mem_available = memory["MemAvailable"]
    swap_total = memory["SwapTotal"]
    swap_free = memory["SwapFree"]
    configured_combined = mem_total + swap_total
    available_combined = mem_available + swap_free
    if mem_available > mem_total or swap_free > swap_total:
        raise RuntimeError(
            "FireRed WSL memory preflight found inconsistent currently available "
            f"capacity: MemAvailable={mem_available / GIB:.1f} GiB of "
            f"MemTotal={mem_total / GIB:.1f} GiB, SwapFree={swap_free / GIB:.1f} "
            f"GiB of SwapTotal={swap_total / GIB:.1f} GiB. Run "
            "`wsl -d Ubuntu -- free -h`, then `wsl --shutdown` if the values remain "
            "inconsistent."
        )
    if use_half:
        min_memory = HALF_MIN_MEMORY_BYTES
        min_combined = HALF_MIN_COMBINED_BYTES
        min_available_memory = HALF_MIN_AVAILABLE_MEMORY_BYTES
        min_available_combined = HALF_MIN_AVAILABLE_COMBINED_BYTES
        mode = "half-precision"
    else:
        min_memory = FP32_MIN_MEMORY_BYTES
        min_combined = FP32_MIN_COMBINED_BYTES
        min_available_memory = FP32_MIN_AVAILABLE_MEMORY_BYTES
        min_available_combined = FP32_MIN_AVAILABLE_COMBINED_BYTES
        mode = "FP32"

    guidance = (
        "Configure `%UserProfile%\\.wslconfig` with `[wsl2] memory=32GB "
        "swap=8GB`, then run `wsl --shutdown` and retry."
    )
    if not use_half:
        guidance += (
            " FP32 needs more memory; prefer options.use_half=true, or provision "
            "at least 48 GiB RAM plus swap."
        )
    if mem_total < min_memory or configured_combined < min_combined:
        raise RuntimeError(
            f"FireRed WSL memory preflight failed for {mode} CUDA load: configured "
            f"capacity is insufficient: MemTotal={mem_total / GIB:.1f} GiB, "
            f"SwapTotal={swap_total / GIB:.1f} GiB, configured combined="
            f"{configured_combined / GIB:.1f} GiB; requires at least "
            f"{min_memory / GIB:.0f} GiB RAM and {min_combined / GIB:.0f} GiB "
            f"configured RAM+swap. {guidance}"
        )
    if (
        mem_available < min_available_memory
        or available_combined < min_available_combined
    ):
        raise RuntimeError(
            f"FireRed WSL memory preflight failed for {mode} CUDA load: currently "
            f"available capacity is insufficient: MemAvailable="
            f"{mem_available / GIB:.1f} GiB, SwapFree={swap_free / GIB:.1f} GiB, "
            f"currently available combined={available_combined / GIB:.1f} GiB; "
            f"requires at least {min_available_memory / GIB:.0f} GiB available RAM "
            f"and {min_available_combined / GIB:.0f} GiB available RAM+free swap. "
            f"The configured capacity is sufficient (MemTotal={mem_total / GIB:.1f} "
            f"GiB, SwapTotal={swap_total / GIB:.1f} GiB). Stop other WSL workloads, "
            "confirm recovery with `wsl -d Ubuntu -- free -h`, and retry; if memory "
            "remains occupied, run `wsl --shutdown` first."
        )


def _parse_proc_meminfo(text: str) -> dict[str, int]:
    wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    parsed: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        label, raw_value = line.split(":", 1)
        label = label.strip()
        if label not in wanted:
            continue
        parts = raw_value.split()
        if len(parts) != 2:
            raise ValueError(f"{label} must contain an integer value and kB unit.")
        value, unit = parts
        if unit != "kB":
            raise ValueError(
                f"{label} uses unsupported unit {unit!r}; expected the /proc kB unit."
            )
        try:
            kib = int(value)
        except ValueError as exc:
            raise ValueError(f"{label} value is not an integer: {value!r}.") from exc
        if kib < 0:
            raise ValueError(f"{label} value must not be negative.")
        parsed[label] = kib * 1024
    for required in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
        if required not in parsed:
            raise ValueError(f"{required} is missing.")
    return parsed


@contextlib.contextmanager
def _temporary_initial_llm_load_dtype(*, device: str, use_half: bool):
    """Force the pinned official LLM loader to materialize directly in half precision."""
    if not use_half or not device.lower().startswith("cuda"):
        yield None
        return

    import torch
    from fireredasr2s.fireredasr2.models import fireredasr_llm

    try:
        bf16_supported = bool(torch.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        bf16_supported = False
    if bf16_supported:
        load_dtype = torch.bfloat16
        load_dtype_name = "bfloat16"
    else:
        load_dtype = torch.float16
        load_dtype_name = "float16"

    original_auto_model = fireredasr_llm.AutoModelForCausalLM

    class _InitialDtypeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            kwargs["torch_dtype"] = load_dtype
            return original_auto_model.from_pretrained(*args, **kwargs)

    fireredasr_llm.AutoModelForCausalLM = _InitialDtypeAutoModelForCausalLM
    try:
        yield load_dtype_name
    finally:
        fireredasr_llm.AutoModelForCausalLM = original_auto_model


def _model_runtime_diagnostics(model: Any) -> dict[str, str]:
    load_dtype = getattr(model, "_zh_asr_llm_load_dtype", None)
    if not isinstance(load_dtype, str) or not load_dtype:
        return {}
    return {"llm_initial_load_dtype": load_dtype}


def _attach_runtime_diagnostics(
    result: list[Any] | dict[str, Any] | str,
    diagnostics: dict[str, str],
) -> None:
    if isinstance(result, dict):
        result["_zh_asr_runtime"] = dict(diagnostics)
        return
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                item["_zh_asr_runtime"] = dict(diagnostics)


def _verify_model_revision(model_dir: Path, expected: Any) -> None:
    expected_revision = str(expected or "").strip()
    if not expected_revision:
        raise RuntimeError(
            "FireRed expected model revision is missing; refusing to load an unpinned model."
        )
    receipt_path = model_dir / "MODEL_RECEIPT.json"
    if not receipt_path.is_file():
        raise RuntimeError(
            f"FireRed pinned model receipt is missing: {receipt_path}. "
            "Run scripts/download-models.ps1 -Engine fireredasr2-llm."
        )
    try:
        receipt = json.loads(
            receipt_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"FireRed model receipt is unreadable: {receipt_path}: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError(f"FireRed model receipt is invalid: {receipt_path}: {exc}") from exc
    if not isinstance(receipt, dict):
        raise RuntimeError("FireRed model receipt must be a JSON object.")
    fields = frozenset(receipt)
    if fields != MODEL_RECEIPT_FIELDS:
        missing = sorted(MODEL_RECEIPT_FIELDS - fields)
        extra = sorted(fields - MODEL_RECEIPT_FIELDS)
        raise RuntimeError(
            "FireRed model receipt top-level fields mismatch: "
            f"missing={missing or '<none>'}, extra={extra or '<none>'}."
        )
    if receipt.get("schema") != MODEL_RECEIPT_SCHEMA:
        raise RuntimeError(
            "FireRed model receipt schema mismatch: "
            f"expected {MODEL_RECEIPT_SCHEMA}, got {receipt.get('schema')!r}."
        )
    if receipt.get("repository") != MODEL_REPOSITORY:
        raise RuntimeError(
            "FireRed model receipt repository mismatch: "
            f"expected {MODEL_REPOSITORY}, got {receipt.get('repository')!r}."
        )
    actual_revision = str(receipt.get("revision", "")).strip()
    if actual_revision != expected_revision:
        raise RuntimeError(
            "FireRed model revision mismatch: "
            f"expected {expected_revision}, receipt has {actual_revision or '<missing>'}."
        )
    created_utc = receipt.get("created_utc")
    if not isinstance(created_utc, str) or not created_utc.strip():
        raise RuntimeError("FireRed model receipt created_utc must be a non-empty string.")
    records = receipt.get("files")
    if not isinstance(records, list):
        raise RuntimeError("FireRed model receipt files must be an array.")

    root = model_dir.resolve(strict=True)
    parsed_records: list[tuple[str, int, str, Path]] = []
    seen_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(
                f"FireRed model receipt files[{index}] must be an object."
            )
        record_fields = frozenset(record)
        if record_fields != MODEL_FILE_RECORD_FIELDS:
            raise RuntimeError(
                f"FireRed model receipt files[{index}] fields mismatch."
            )
        relative_path = record.get("path")
        if not isinstance(relative_path, str) or not _is_safe_posix_relative_path(
            relative_path
        ):
            raise RuntimeError(
                f"FireRed model receipt contains unsafe path at files[{index}]: "
                f"{relative_path!r}."
            )
        if relative_path in seen_paths:
            raise RuntimeError(
                f"FireRed model receipt contains duplicate path: {relative_path}."
            )
        seen_paths.add(relative_path)
        expected_bytes = record.get("bytes")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes <= 0
        ):
            raise RuntimeError(
                f"FireRed model receipt has invalid byte size for {relative_path}: "
                f"{expected_bytes!r}."
            )
        expected_sha256 = record.get("sha256")
        if not isinstance(expected_sha256, str) or not SHA256_PATTERN.fullmatch(
            expected_sha256
        ):
            raise RuntimeError(
                f"FireRed model receipt has invalid SHA-256 for {relative_path}."
            )
        candidate = root.joinpath(*PurePosixPath(relative_path).parts)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise RuntimeError(
                f"FireRed required model artifact is missing or escapes model_dir: "
                f"{relative_path}: {exc}"
            ) from exc
        if not resolved.is_file():
            raise RuntimeError(
                f"FireRed required model artifact is not a file: {relative_path}."
            )
        parsed_records.append(
            (relative_path, expected_bytes, expected_sha256, resolved)
        )

    actual_paths = [record[0] for record in parsed_records]
    if actual_paths != list(MODEL_REQUIRED_FILES):
        missing = sorted(set(MODEL_REQUIRED_FILES) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(MODEL_REQUIRED_FILES))
        raise RuntimeError(
            "FireRed model receipt required file list mismatch "
            "(including canonical order): "
            f"missing={missing or '<none>'}, extra={extra or '<none>'}."
        )

    for relative_path, expected_bytes, expected_sha256, resolved in parsed_records:
        actual_bytes = resolved.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"FireRed model artifact size mismatch for {relative_path}: "
                f"expected {expected_bytes}, got {actual_bytes}."
            )
        actual_sha256 = _sha256_file(resolved)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"FireRed model artifact SHA-256 mismatch for {relative_path}: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )


def _verify_source_revision(source_dir: Path, expected: Any) -> None:
    expected_revision = str(expected or "").strip()
    if not expected_revision:
        raise RuntimeError(
            "FireRed expected source revision is missing; refusing to load unpinned source."
        )
    actual_revision = _run_git(
        source_dir,
        "rev-parse",
        "--verify",
        "HEAD",
        purpose="source revision",
    )
    if actual_revision != expected_revision:
        raise RuntimeError(
            "FireRed source revision mismatch: "
            f"expected {expected_revision}, got {actual_revision or '<unavailable>'}."
        )
    dirty = _run_git(
        source_dir,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        purpose="source working tree",
    )
    if dirty:
        detail = "; ".join(line.strip() for line in dirty.splitlines() if line.strip())
        raise RuntimeError(
            "FireRed source working tree is dirty; tracked and untracked changes "
            f"are not allowed: {detail}."
        )


def _run_git(source_dir: Path, *args: str, purpose: str) -> str:
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                "core.autocrlf=true",
                "-C",
                str(source_dir),
                *args,
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"Could not verify FireRed {purpose} at {source_dir}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    value = completed.stdout.strip()
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        raise RuntimeError(
            f"Could not verify FireRed {purpose} at {source_dir}: "
            f"git exit code {completed.returncode}"
            f"{': ' + detail if detail else ''}."
        )
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _is_safe_posix_relative_path(value: str) -> bool:
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        return False
    return all(part not in {"", ".", ".."} for part in path.parts)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _required_path(request: dict[str, Any], key: str, *, directory: bool) -> Path:
    raw = str(request.get(key, "")).strip()
    if not raw:
        raise ValueError(f"FireRed request is missing '{key}'.")
    path = Path(raw).expanduser()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"FireRed {key} {kind} not found: {path}")
    return path.resolve()


def _request_audio_paths(request: dict[str, Any]) -> list[Path]:
    raw_many = request.get("audio_paths")
    if raw_many is None:
        return [_required_path(request, "audio_path", directory=False)]
    if not isinstance(raw_many, list) or not raw_many:
        raise ValueError("FireRed request 'audio_paths' must be a non-empty list.")
    paths: list[Path] = []
    for index, value in enumerate(raw_many):
        raw = str(value).strip()
        if not raw:
            raise ValueError(f"FireRed request audio_paths[{index}] must not be empty.")
        path = Path(raw).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"FireRed audio_paths[{index}] file not found: {path}")
        paths.append(path.resolve())
    return paths


def _optional_directory(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"FireRed source directory not found: {path}")
    return path.resolve()


def _empty_cuda_cache(device: str) -> None:
    if not device.lower().startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _error_response(exc: BaseException) -> dict[str, Any]:
    return {
        "schema": RESPONSE_SCHEMA,
        "ok": False,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("FireRed worker request must be a JSON object.")
        with contextlib.redirect_stdout(sys.stderr):
            response = run_request(request)
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        response = _error_response(exc)
        print(json.dumps(response, ensure_ascii=False))
        return 1
    print(json.dumps(response, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

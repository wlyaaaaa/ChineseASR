"""Coverage-preserving VAD cuts and explicit channel extraction."""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import wave
from typing import Any

from .config import ModelConfig
from .model_lifecycle import file_hash, write_json_atomic
from .result_writer import canonical_json_sha256


def speech_boundaries(audio: Path, cache_dir: Path, config: ModelConfig, output_dir: Path) -> dict[str, Any]:
    """CPU FSMN-VAD suggests cut positions only; it never removes source intervals."""
    model_ref = config.model_aliases.get("fsmn-vad", "")
    model_dir = cache_dir.joinpath(*model_ref.split("/"))
    result: dict[str, Any] = {"status": "unavailable", "segments": [],
        "excluded_ranges_ms": [], "policy": "cut_hints_only_full_source_coverage"}
    if not model_ref or not model_dir.is_dir():
        return {**result, "reason": "local_vad_not_installed_no_implicit_download"}
    model_files = [p for p in model_dir.iterdir() if p.is_file() and p.name in
                   {"model.pt", "model.pb", "model.onnx", "config.yaml", "configuration.json"}]
    try:
        runtime = importlib.metadata.version("funasr")
    except importlib.metadata.PackageNotFoundError:
        return {**result, "reason": "local_vad_runtime_not_installed"}
    identity = {"model": model_ref, "runtime": runtime,
                "files": {p.name: file_hash(p) for p in model_files}, "audio_sha256": file_hash(audio)}
    key = canonical_json_sha256(identity)
    sidecar = output_dir / (key + ".vad.json")
    if sidecar.is_file():
        try:
            cached = json.loads(sidecar.read_text(encoding="utf-8"))
            if cached.get("identity") == identity and cached.get("status") == "available":
                return cached
        except (ValueError, OSError):
            pass
    try:
        from funasr import AutoModel
        from .adapters.funasr import _normalize_vad_segments
        model = AutoModel(model=str(model_dir), device="cpu", disable_update=True,
                          disable_pbar=True, disable_log=True)
        raw = model.generate(input=str(audio), max_single_segment_time=30000)
        segments = _normalize_vad_segments(raw)
        normalized = sorted([[int(a), int(b)] for a, b in segments if 0 <= a < b])
        result.update(status="available", identity=identity, segments=normalized,
                      reason="vad_zero_segments" if not normalized else "speech_spans_found")
        del model
    except Exception as error:
        result.update(reason="vad_failed_fixed_cut_fallback", error=f"{type(error).__name__}: {error}")
    write_json_atomic(sidecar, result)
    return result


def choose_cut(start_ms: int, target_ms: int, duration_ms: int,
               segments: list[list[int]], min_chunk_ms: int = 20000,
               search_ms: int = 5000) -> int:
    if target_ms >= duration_ms or not segments:
        return min(target_ms, duration_ms)
    lower = max(start_ms + min_chunk_ms, target_ms - search_ms)
    if lower >= target_ms:
        return target_ms
    # Merge overlapping VAD spans before looking for sufficiently wide pauses.
    merged: list[list[int]] = []
    for a, b in sorted(segments):
        a, b = max(0, int(a)), min(duration_ms, int(b))
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    pauses = []
    cursor = 0
    for a, b in merged:
        if a - cursor >= 200:
            pauses.append((cursor, a))
        cursor = max(cursor, b)
    if duration_ms - cursor >= 200:
        pauses.append((cursor, duration_ms))
    candidates = [min(target_ms, b - 80) for a, b in pauses
                  if min(target_ms, b - 80) >= max(lower, a + 80)]
    return max(candidates) if candidates else target_ms


def probe_channels(audio: Path) -> int:
    try:
        with wave.open(str(audio), "rb") as handle:
            return handle.getnchannels()
    except (wave.Error, EOFError):
        pass
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required to inspect non-WAV channels")
    from .process_control import managed_popen_kwargs
    process = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=channels", "-of", "json", str(audio)],
        check=True, capture_output=True, timeout=30, **managed_popen_kwargs())
    streams = json.loads(process.stdout)["streams"]
    if not streams:
        raise ValueError("Input contains no audio stream")
    return int(streams[0]["channels"])


def extract_channel(audio: Path, index: int, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Extract one zero-based channel. Original audio remains immutable."""
    channels = probe_channels(audio)
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < channels:
        raise ValueError(f"Channel must be between 0 and {channels - 1}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for channel extraction")
    from .process_control import managed_popen_kwargs
    source_hash = file_hash(audio)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{source_hash}.channel-{index}.wav"
    temporary = target.with_suffix(".partial.wav")
    try:
        subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(audio), "-map", "0:a:0", "-af", f"pan=mono|c0=c{index}",
            "-ar", "16000", "-c:a", "pcm_s16le", "-map_metadata", "-1", str(temporary)],
            check=True, capture_output=True, timeout=600, **managed_popen_kwargs())
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    identity = {"source": str(audio.resolve()), "source_sha256": source_hash,
        "source_channels": channels, "selected_channel": index,
        "derivative": str(target), "derivative_sha256": file_hash(target),
        "speaker_identity": "not_inferred_from_channel"}
    write_json_atomic(target.with_suffix(".channel.json"), identity)
    return target, identity


def probe_duration_ms(audio: Path) -> int:
    """Read media duration without loading recognition models or changing the source."""
    import math
    try:
        with wave.open(str(audio), "rb") as reader:
            return round(reader.getnframes()*1000/reader.getframerate())
    except (wave.Error, EOFError):
        pass
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required to inspect non-PCM audio duration")
    from .process_control import managed_popen_kwargs
    try:
        process = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
            check=True, capture_output=True, timeout=30, **managed_popen_kwargs())
    except subprocess.SubprocessError as error:
        raise RuntimeError("Media duration probe failed; validity and coverage remain unknown") from error
    seconds = float(process.stdout)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Audio has an invalid duration")
    return round(seconds*1000)

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re
import shutil
import subprocess
import wave


DEFAULT_CONVERSION_TIMEOUT_SEC = 60 * 60


class AudioConversionTimeout(TimeoutError):
    """ffmpeg exceeded the configured conversion deadline."""


@dataclass(frozen=True)
class PreparedAudio:
    source_path: Path
    path: Path
    converted: bool
    source_sha256: str
    derivative_sha256: str
    sample_rate: int
    channels: int
    sample_width_bytes: int
    duration_sec: float
    ffmpeg_version: str = ""
    conversion_command: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_path"] = str(self.source_path)
        payload["path"] = str(self.path)
        payload["conversion_command"] = list(self.conversion_command)
        return payload


def prepare_pcm16_mono(
    source_path: Path,
    derived_dir: Path,
    *,
    sample_rate: int = 16000,
    ffmpeg: str | None = None,
    timeout_sec: float = DEFAULT_CONVERSION_TIMEOUT_SEC,
) -> PreparedAudio:
    source = source_path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Audio file not found: {source}")
    source_sha256 = _sha256_file(source)
    conversion_timeout = _positive_timeout(timeout_sec)

    source_format = _wav_format(source)
    if source_format and source_format[:3] == (sample_rate, 1, 2):
        return PreparedAudio(
            source_path=source,
            path=source,
            converted=False,
            source_sha256=source_sha256,
            derivative_sha256=source_sha256,
            sample_rate=source_format[0],
            channels=source_format[1],
            sample_width_bytes=source_format[2],
            duration_sec=source_format[3],
        )

    executable = ffmpeg or shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError(
            "ffmpeg is required to convert this audio to 16 kHz, 16-bit, mono PCM WAV."
        )

    derived_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("._") or "audio"
    destination = derived_dir / f"{safe_stem}.{source_sha256[:16]}.16k-mono.wav"
    partial = destination.with_suffix(".partial.wav")
    command = (
        str(executable),
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
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(partial),
    )
    if destination.exists():
        existing = _wav_format(destination)
        if existing and existing[:3] == (sample_rate, 1, 2):
            return PreparedAudio(
                source_path=source,
                path=destination,
                converted=True,
                source_sha256=source_sha256,
                derivative_sha256=_sha256_file(destination),
                sample_rate=existing[0],
                channels=existing[1],
                sample_width_bytes=existing[2],
                duration_sec=existing[3],
                ffmpeg_version=_ffmpeg_version(executable),
                conversion_command=command,
            )

    try:
        try:
            subprocess.run(
                list(command),
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=conversion_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioConversionTimeout(
                f"ffmpeg audio conversion timed out after {conversion_timeout:g}s: {source}"
            ) from exc
        converted_format = _wav_format(partial)
        if converted_format is None or converted_format[:3] != (sample_rate, 1, 2):
            raise RuntimeError("ffmpeg output is not 16 kHz, 16-bit, mono PCM WAV.")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)

    converted_format = _wav_format(destination)
    assert converted_format is not None
    return PreparedAudio(
        source_path=source,
        path=destination,
        converted=True,
        source_sha256=source_sha256,
        derivative_sha256=_sha256_file(destination),
        sample_rate=converted_format[0],
        channels=converted_format[1],
        sample_width_bytes=converted_format[2],
        duration_sec=converted_format[3],
        ffmpeg_version=_ffmpeg_version(executable),
        conversion_command=command,
    )


def _wav_format(path: Path) -> tuple[int, int, int, float] | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            duration = handle.getnframes() / frame_rate if frame_rate else 0.0
    except (EOFError, OSError, wave.Error):
        return None
    return frame_rate, channels, sample_width, duration


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffmpeg_version(executable: str) -> str:
    try:
        result = subprocess.run(
            [executable, "-version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except Exception:
        return ""
    return result.stdout.splitlines()[0] if result.stdout else ""


def _positive_timeout(value: float) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_sec must be a positive finite number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_sec must be a positive finite number")
    return timeout

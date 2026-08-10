from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
from typing import Iterator
import wave


DEFAULT_CONVERSION_TIMEOUT_SEC = 60 * 60


class AudioConversionTimeout(TimeoutError):
    """ffmpeg exceeded the configured conversion deadline."""


class PreparedAudioIntegrityError(RuntimeError):
    """Prepared audio no longer matches its in-memory provenance."""


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
        payload["format"] = "wav"
        payload["sample_width"] = self.sample_width_bytes
        return payload


def prepare_pcm16_mono(
    source_path: Path,
    derived_dir: Path,
    *,
    sample_rate: int = 16000,
    ffmpeg: str | None = None,
    timeout_sec: float = DEFAULT_CONVERSION_TIMEOUT_SEC,
    materialize_owner: bool = False,
) -> PreparedAudio:
    source = source_path.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Audio file not found: {source}")
    source_sha256 = _sha256_file(source)
    conversion_timeout = _positive_timeout(timeout_sec)

    source_format = _wav_format(source)
    if source_format and source_format[:3] == (sample_rate, 1, 2) and not materialize_owner:
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

    derived_root = (
        _validated_owner_directory(derived_dir, create=True)
        if materialize_owner
        else derived_dir.resolve()
    )
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("._") or "audio"
    destination = derived_root / f"{safe_stem}.{source_sha256[:16]}.16k-mono.wav"
    partial = destination.with_suffix(".partial.wav")

    if materialize_owner:
        _ensure_owner_slot_is_unambiguous(derived_root, destination)
        _reject_preexisting_partial(derived_root, partial)

    if source_format and source_format[:3] == (sample_rate, 1, 2):
        try:
            with source.open("rb") as source_handle, partial.open("xb") as partial_handle:
                shutil.copyfileobj(source_handle, partial_handle)
            _assert_safe_owner_file(derived_root, partial)
            copied_format = _wav_format(partial)
            if copied_format is None or copied_format[:3] != (sample_rate, 1, 2):
                raise RuntimeError("Copied audio is not 16 kHz, 16-bit, mono PCM WAV.")
            _assert_replace_destination_is_safe(derived_root, destination)
            partial.replace(destination)
            _assert_safe_owner_file(derived_root, destination)
        finally:
            partial.unlink(missing_ok=True)
        prepared = PreparedAudio(
            source_path=source,
            path=destination,
            converted=False,
            source_sha256=source_sha256,
            derivative_sha256=_sha256_file(destination),
            sample_rate=copied_format[0],
            channels=copied_format[1],
            sample_width_bytes=copied_format[2],
            duration_sec=copied_format[3],
        )
        validate_prepared_audio_owner(prepared, derived_root)
        return prepared

    executable = ffmpeg or shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError(
            "ffmpeg is required to convert this audio to 16 kHz, 16-bit, mono PCM WAV."
        )
    if not materialize_owner:
        derived_root.mkdir(parents=True, exist_ok=True)

    command = (
        str(executable),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n" if materialize_owner else "-y",
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
    if destination.exists() and not materialize_owner:
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
        if materialize_owner:
            _assert_safe_owner_file(derived_root, partial)
        converted_format = _wav_format(partial)
        if converted_format is None or converted_format[:3] != (sample_rate, 1, 2):
            raise RuntimeError("ffmpeg output is not 16 kHz, 16-bit, mono PCM WAV.")
        if materialize_owner:
            _assert_replace_destination_is_safe(derived_root, destination)
        partial.replace(destination)
        if materialize_owner:
            _assert_safe_owner_file(derived_root, destination)
    finally:
        partial.unlink(missing_ok=True)

    converted_format = _wav_format(destination)
    assert converted_format is not None
    prepared = PreparedAudio(
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
    if materialize_owner:
        validate_prepared_audio_owner(prepared, derived_root)
    return prepared


def validate_prepared_audio_owner(
    prepared: PreparedAudio,
    derived_dir: Path,
) -> None:
    """Fail closed unless one owner WAV still matches all recorded provenance."""

    derived_root = _validated_owner_directory(derived_dir, create=False)
    source = prepared.source_path.resolve()
    owner = Path(os.path.abspath(prepared.path))
    _assert_child_contained(derived_root, owner)
    if owner.resolve(strict=False) == source:
        raise PreparedAudioIntegrityError(
            "Prepared audio is not an independent file in the expected owner directory."
        )
    if not source.is_file():
        raise PreparedAudioIntegrityError("Prepared audio source is missing.")
    _assert_safe_owner_file(derived_root, owner)

    candidates = list(derived_root.glob("*.wav"))
    for candidate in candidates:
        _assert_safe_owner_file(derived_root, candidate)
    if len(candidates) != 1 or not _same_path(candidates[0], owner):
        raise PreparedAudioIntegrityError(
            "Prepared audio owner directory contains multiple or unexpected derivatives."
        )
    if _sha256_file(source) != prepared.source_sha256:
        raise PreparedAudioIntegrityError("Prepared audio source hash changed.")
    if _sha256_file(owner) != prepared.derivative_sha256:
        raise PreparedAudioIntegrityError("Prepared audio derivative hash changed.")

    actual_format = _wav_format(owner)
    expected_format = (
        prepared.sample_rate,
        prepared.channels,
        prepared.sample_width_bytes,
    )
    if actual_format is None or actual_format[:3] != expected_format:
        raise PreparedAudioIntegrityError("Prepared audio WAV format changed.")
    if expected_format != (16000, 1, 2):
        raise PreparedAudioIntegrityError(
            "Prepared audio provenance is not 16 kHz, 16-bit, mono PCM WAV."
        )
    if not math.isclose(actual_format[3], prepared.duration_sec, abs_tol=1e-9):
        raise PreparedAudioIntegrityError("Prepared audio duration changed.")


@contextmanager
def _locked_prepared_audio_owner(
    prepared: PreparedAudio,
    derived_dir: Path,
) -> Iterator[None]:
    derived_root = _validated_owner_directory(derived_dir, create=False)
    owner = Path(os.path.abspath(prepared.path))
    _assert_safe_owner_file(derived_root, owner)

    if os.name == "nt":
        try:
            lock_handle = _open_windows_owner_read_lock(owner)
        except OSError as exc:
            raise PreparedAudioIntegrityError(
                "Unable to acquire the prepared audio read lock."
            ) from exc
        try:
            validate_prepared_audio_owner(prepared, derived_root)
            try:
                yield
            finally:
                validate_prepared_audio_owner(prepared, derived_root)
        finally:
            _close_windows_handle(lock_handle)
        return

    owner_handle = owner.open("rb")
    try:
        try:
            import fcntl
        except ImportError as exc:
            raise PreparedAudioIntegrityError(
                "Prepared audio read locking is unavailable on this platform."
            ) from exc
        fcntl.flock(owner_handle.fileno(), fcntl.LOCK_SH)
        validate_prepared_audio_owner(prepared, derived_root)
        try:
            yield
        finally:
            validate_prepared_audio_owner(prepared, derived_root)
            fcntl.flock(owner_handle.fileno(), fcntl.LOCK_UN)
    finally:
        owner_handle.close()


def _ensure_owner_slot_is_unambiguous(derived_dir: Path, destination: Path) -> None:
    unexpected: list[Path] = []
    for candidate in derived_dir.glob("*.wav"):
        _assert_safe_owner_file(derived_dir, candidate)
        if not _same_path(candidate, destination):
            unexpected.append(candidate)
    if unexpected:
        raise PreparedAudioIntegrityError(
            "Prepared audio owner directory already contains another derivative."
        )


def _validated_owner_directory(path: Path, *, create: bool) -> Path:
    fixed_root = Path(os.path.abspath(path))
    _reject_reparse_directory_chain(fixed_root, require_exists=False)
    if create:
        fixed_root.mkdir(parents=True, exist_ok=True)
    _reject_reparse_directory_chain(fixed_root, require_exists=True)
    return fixed_root


def _reject_reparse_directory_chain(path: Path, *, require_exists: bool) -> None:
    for component in (*reversed(path.parents), path):
        if not os.path.lexists(component):
            if require_exists:
                raise PreparedAudioIntegrityError(
                    "Prepared audio owner directory chain is incomplete."
                )
            continue
        info = os.lstat(component)
        if _is_reparse_or_symlink(info):
            raise PreparedAudioIntegrityError(
                "Prepared audio owner directory chain contains a reparse point."
            )
        if not stat.S_ISDIR(info.st_mode):
            raise PreparedAudioIntegrityError(
                "Prepared audio owner directory chain contains a non-directory."
            )


def _assert_child_contained(derived_root: Path, child: Path) -> None:
    fixed_child = Path(os.path.abspath(child))
    if not _same_path(fixed_child.parent, derived_root):
        raise PreparedAudioIntegrityError(
            "Prepared audio path is outside the fixed owner directory."
        )
    resolved_root = derived_root.resolve(strict=True)
    resolved_child = fixed_child.resolve(strict=False)
    try:
        relative = resolved_child.relative_to(resolved_root)
    except ValueError as exc:
        raise PreparedAudioIntegrityError(
            "Prepared audio resolved outside the fixed owner directory."
        ) from exc
    if len(relative.parts) != 1:
        raise PreparedAudioIntegrityError(
            "Prepared audio is not a direct child of the fixed owner directory."
        )


def _assert_safe_owner_file(derived_root: Path, path: Path) -> None:
    _assert_child_contained(derived_root, path)
    if not os.path.lexists(path):
        raise PreparedAudioIntegrityError("Prepared audio derivative is missing.")
    info = os.lstat(path)
    if _is_reparse_or_symlink(info):
        raise PreparedAudioIntegrityError(
            "Prepared audio derivative is a reparse point."
        )
    if not stat.S_ISREG(info.st_mode):
        raise PreparedAudioIntegrityError(
            "Prepared audio derivative is not a regular file."
        )
    if info.st_nlink != 1:
        raise PreparedAudioIntegrityError(
            "Prepared audio derivative has multiple filesystem links."
        )


def _reject_preexisting_partial(derived_root: Path, partial: Path) -> None:
    _assert_child_contained(derived_root, partial)
    if os.path.lexists(partial):
        _assert_safe_owner_file(derived_root, partial)
        raise PreparedAudioIntegrityError(
            "Prepared audio partial file already exists."
        )


def _assert_replace_destination_is_safe(
    derived_root: Path,
    destination: Path,
) -> None:
    _assert_child_contained(derived_root, destination)
    if os.path.lexists(destination):
        _assert_safe_owner_file(derived_root, destination)


def _is_reparse_or_symlink(info: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & reparse_flag)


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(
        os.path.abspath(right)
    )


def _open_windows_owner_read_lock(path: Path) -> int:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        0x80000000,
        0x00000001,
        None,
        3,
        0x00000080 | 0x00200000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def _close_windows_handle(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    if not close_handle(handle):
        raise ctypes.WinError(ctypes.get_last_error())


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

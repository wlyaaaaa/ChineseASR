"""On-demand local speaker-verification evidence for the single ``person:self`` anchor.

This module deliberately does not provide a speaker directory, service, queue, or
cross-user database.  A user can enroll one private local reference, or a bounded
set of two or three independently source-bound references, and later compare one
audio interval against the sole resulting profile.  The persistent vector is only
the user's own ``person:self`` profile; per-reference and target embeddings stay in
process memory and are discarded after a centroid or hash-bound score is written.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import wave

from .adapters.funasr import ensure_funasr_available
from .config import ModelConfig, SpeakerVerificationSpec, load_model_config
from .pipeline import default_cache_dir, prepare_model_env, project_root
from .result_writer import canonical_json_sha256, file_sha256


SELF_PERSON_ID = "person:self"
SELF_SPEAKER_PROFILE_SCHEMA = "chinese-asr.person-self-voice-profile.v2"
SELF_SPEAKER_EVIDENCE_SCHEMA = "chinese-asr.person-self-voice-evidence.v1"
SELF_SPEAKER_MULTI_PROFILE_SCHEMA = "chinese-asr.person-self-voice-profile.v3"
SELF_SPEAKER_MULTI_EVIDENCE_SCHEMA = "chinese-asr.person-self-voice-evidence.v2"
SELF_SPEAKER_REFERENCE_SET_SCHEMA = "chinese-asr.person-self-voice-reference-set.v1"
SPEAKER_MODEL_EVIDENCE_SCHEMA = "chinese-asr.speaker-model-evidence.v1"
MAX_SPEAKER_EVIDENCE_DURATION_MS = 120_000
VOICE_SCORE_AMBIGUITY_MARGIN = 0.02
MIN_SELF_SPEAKER_REFERENCES = 2
MAX_SELF_SPEAKER_REFERENCES = 3
CENTROID_AGGREGATION_METHOD = "l2_normalized_centroid.v1"

_CHANNELS = frozenset({"mix", "left", "right"})
_SELECTION_BINDING_KINDS = frozenset(
    {
        "verified_xiaomi_right_channel",
        "dialogue_role",
        "semantic_role",
        "cross_recording_role",
    }
)


class SpeakerEvidenceError(ValueError):
    """Raised for invalid or non-reproducible ``person:self`` evidence."""


def default_self_speaker_profile_path() -> Path:
    """Return the Git-ignored local path for the sole persistent profile."""

    return project_root() / "outputs" / "private" / "person-self.voice-profile.json"


def enroll_self_speaker(
    reference_audio: Path,
    *,
    start_ms: int | float,
    end_ms: int | float,
    channel: str = "mix",
    inference_basis: str,
    profile_path: Path | None = None,
    replace: bool = False,
    device: str = "cuda:0",
    cache_dir: Path | None = None,
    model_config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Create or explicitly replace the one private ``person:self`` profile.

    The original reference audio is never copied into the profile.  Only its
    path/hash/interval provenance, its reversible inferred-identity basis, and
    the model vector are retained locally.  This command cannot create a
    ``confirmed`` identity profile.
    """

    destination = (profile_path or default_self_speaker_profile_path()).resolve()
    if destination.exists() and not replace:
        raise FileExistsError(
            f"A {SELF_PERSON_ID} profile already exists at {destination}. Use --replace to replace it."
        )
    config = model_config or load_model_config()
    cache = _resolve_cache_dir(cache_dir)
    model_evidence = speaker_model_evidence(config, cache)
    reference = _source_reference(reference_audio)
    start, end = _bounded_interval(start_ms, end_ms)
    selected_channel = _channel(channel)
    identity = _inferred_identity(inference_basis)

    with tempfile.TemporaryDirectory(prefix="zh-asr-self-speaker-") as temp_dir:
        clip = Path(temp_dir) / "reference.16k-mono.wav"
        channel_binding = _prepare_segment_wav(
            Path(reference["path"]),
            clip,
            start_ms=start,
            end_ms=end,
            channel=selected_channel,
        )
        embedding = _extract_embedding(clip, model_evidence["local_model_dir"], device)

    profile = {
        "schema": SELF_SPEAKER_PROFILE_SCHEMA,
        "person_id": SELF_PERSON_ID,
        "created_utc": _utc_now(),
        "identity": identity,
        "enrollment_reference": {
            "source": reference,
            "segment": {
                "start_ms": start,
                "end_ms": end,
                "channel": selected_channel,
                "channel_binding": channel_binding,
            },
        },
        "model": model_evidence,
        "embedding": embedding,
    }
    _write_json_atomic(destination, profile)
    return profile


def enroll_self_speaker_reference_set(
    reference_set_path: Path,
    *,
    profile_path: Path | None = None,
    replace: bool = False,
    device: str = "cuda:0",
    cache_dir: Path | None = None,
    model_config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Create one private profile from exactly two or three distinct sources.

    The private input manifest may contain source paths so the original files can
    be opened.  Those paths and the individual embeddings are deliberately absent
    from the persisted profile.  Only source hashes/sizes, exact intervals,
    selection bindings, and one normalized centroid remain.
    """

    destination = (profile_path or default_self_speaker_profile_path()).resolve()
    if destination.exists() and not replace:
        raise FileExistsError(
            f"A {SELF_PERSON_ID} profile already exists at {destination}. Use --replace to replace it."
        )
    manifest = _load_reference_set_manifest(reference_set_path)
    references = _resolve_reference_set(manifest)
    config = model_config or load_model_config()
    cache = _resolve_cache_dir(cache_dir)
    model_evidence = speaker_model_evidence(config, cache)

    persisted_references: list[dict[str, Any]] = []
    normalized_embeddings: list[list[float]] = []
    with tempfile.TemporaryDirectory(prefix="zh-asr-self-speaker-") as temp_dir:
        temp_root = Path(temp_dir)
        for index, reference in enumerate(references):
            clip = temp_root / f"reference-{index}.16k-mono.wav"
            channel_binding = _prepare_segment_wav(
                Path(reference["source_path"]),
                clip,
                start_ms=reference["start_ms"],
                end_ms=reference["end_ms"],
                channel=reference["channel"],
            )
            normalized_embeddings.append(
                _l2_normalize(
                    _extract_embedding(clip, model_evidence["local_model_dir"], device)
                )
            )
            persisted_references.append(
                {
                    "source": {
                        "bytes": reference["source_bytes"],
                        "sha256": reference["source_sha256"],
                    },
                    "segment": {
                        "start_ms": reference["start_ms"],
                        "end_ms": reference["end_ms"],
                        "channel": reference["channel"],
                        "channel_binding": channel_binding,
                    },
                    "inference_basis": reference["inference_basis"],
                    "selection_binding": reference["selection_binding"],
                }
            )

    reference_set_payload = {
        "schema": SELF_SPEAKER_REFERENCE_SET_SCHEMA,
        "references": persisted_references,
    }
    reference_set_sha256 = canonical_json_sha256(reference_set_payload)
    profile = {
        "schema": SELF_SPEAKER_MULTI_PROFILE_SCHEMA,
        "person_id": SELF_PERSON_ID,
        "created_utc": _utc_now(),
        "identity": _inferred_identity(manifest["inference_basis"]),
        "reference_set": {
            **reference_set_payload,
            "sha256": reference_set_sha256,
            "reference_count": len(persisted_references),
        },
        "aggregation": {
            "method": CENTROID_AGGREGATION_METHOD,
            "reference_count": len(persisted_references),
        },
        "model": model_evidence,
        "embedding": _centroid(normalized_embeddings),
    }
    _write_json_atomic(destination, profile)
    return profile


def create_self_speaker_evidence(
    target_audio: Path,
    *,
    start_ms: int | float,
    end_ms: int | float,
    channel: str = "mix",
    profile_path: Path | None = None,
    device: str = "cuda:0",
    cache_dir: Path | None = None,
    model_config: ModelConfig | None = None,
    require_held_out: bool = False,
) -> dict[str, Any]:
    """Compare one bounded target interval with the private ``person:self`` vector.

    Returned evidence is intentionally not an identity confirmation.  It contains
    no target embedding and must be fused with source, channel, contact, dialogue,
    semantic, or cross-recording evidence by ``speaker_attribution``.
    """

    profile_source = profile_path or default_self_speaker_profile_path()
    profile = load_self_speaker_profile(profile_source)
    config = model_config or load_model_config()
    cache = _resolve_cache_dir(cache_dir)
    current_model = speaker_model_evidence(config, cache)
    _validate_profile_for_model(profile, current_model)
    target = _source_reference(target_audio)
    source_relation = (
        "enrollment_source"
        if target["sha256"] in _profile_enrollment_source_hashes(profile)
        else "held_out_source"
    )
    if require_held_out and source_relation == "enrollment_source":
        raise SpeakerEvidenceError(
            "Held-out speaker evidence cannot use any source that contributed to the active profile."
        )
    start, end = _bounded_interval(start_ms, end_ms)
    selected_channel = _channel(channel)

    with tempfile.TemporaryDirectory(prefix="zh-asr-self-speaker-") as temp_dir:
        clip = Path(temp_dir) / "target.16k-mono.wav"
        channel_binding = _prepare_segment_wav(
            Path(target["path"]),
            clip,
            start_ms=start,
            end_ms=end,
            channel=selected_channel,
        )
        target_embedding = _extract_embedding(clip, current_model["local_model_dir"], device)

    similarity = _cosine_similarity(profile["embedding"], target_embedding)
    threshold = float(current_model["threshold"])
    comparison = "above_threshold" if similarity >= threshold else "below_threshold"
    profile_hash = canonical_json_sha256(profile)
    return {
        "schema": (
            SELF_SPEAKER_MULTI_EVIDENCE_SCHEMA
            if profile["schema"] == SELF_SPEAKER_MULTI_PROFILE_SCHEMA
            else SELF_SPEAKER_EVIDENCE_SCHEMA
        ),
        "person_id": SELF_PERSON_ID,
        "generated_utc": _utc_now(),
        "target": {
            "source": target,
            "segment": {
                "start_ms": start,
                "end_ms": end,
                "channel": selected_channel,
                "channel_binding": channel_binding,
            },
        },
        "profile": _evidence_profile_binding(profile, profile_hash),
        "source_relation": source_relation,
        "model": current_model,
        "score": {
            "metric": "cosine_similarity",
            "value": similarity,
            "threshold": threshold,
            "comparison": comparison,
        },
        "identity_status": "unconfirmed",
        "meaning": (
            "目标片段与 enrollment 来自同一原件，保留分数供审计但不作为方向性身份线索。"
            if source_relation == "enrollment_source"
            else
            "相似度接近当前阈值，单独不作为方向性身份线索。"
            if abs(similarity - threshold) <= VOICE_SCORE_AMBIGUITY_MARGIN
            else "相似度高于当前阈值，只是支持本人候选，不能单独确认身份。"
            if similarity > threshold
            else "相似度低于当前阈值，只是弱的非本人线索，不能单独确认他人身份。"
        ),
    }


def write_self_speaker_evidence(
    output_path: Path,
    target_audio: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create one evidence document and atomically write it to *output_path*."""

    evidence = create_self_speaker_evidence(target_audio, **kwargs)
    _write_json_atomic(output_path.resolve(), evidence)
    return evidence


def load_self_speaker_profile(profile_path: Path) -> dict[str, Any]:
    """Read and validate the only allowed persistent speaker profile."""

    path = profile_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{SELF_PERSON_ID} profile not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpeakerEvidenceError(f"Invalid {SELF_PERSON_ID} profile JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SpeakerEvidenceError(f"{SELF_PERSON_ID} profile must be a JSON object.")
    _validate_profile_shape(value)
    return value


def delete_self_speaker_profile(profile_path: Path, *, confirmation: str) -> None:
    """Delete the local self profile only after an explicit fixed confirmation."""

    if confirmation != SELF_PERSON_ID:
        raise SpeakerEvidenceError(
            f"Profile deletion requires --confirm-delete {SELF_PERSON_ID}."
        )
    path = profile_path.resolve()
    profile = load_self_speaker_profile(path)
    if profile["person_id"] != SELF_PERSON_ID:  # Defensive; shape already checks this.
        raise SpeakerEvidenceError("Only the person:self profile can be deleted here.")
    path.unlink()


def speaker_model_evidence(
    model_config: ModelConfig,
    cache_dir: Path,
) -> dict[str, Any]:
    """Return hash-bound local CAM++ model/runtime evidence without loading it."""

    spec = _speaker_verification_spec(model_config)
    model_id = model_config.model_aliases[spec.model_alias]
    model_dir = cache_dir / Path(*model_id.split("/"))
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Local speaker-verification model is not prepared: {model_dir}. "
            "Download the configured CAM++ model before enrolling or comparing audio."
        )
    required_files = (spec.model_file, "configuration.json", "config.yaml")
    files: list[dict[str, Any]] = []
    for relative in required_files:
        path = model_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Local speaker-verification model file is missing: {path}")
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    try:
        runtime_version = importlib.metadata.version("funasr")
    except importlib.metadata.PackageNotFoundError:
        runtime_version = "unavailable"
    return {
        "schema": SPEAKER_MODEL_EVIDENCE_SCHEMA,
        "model_id": model_id,
        "configured_revision": spec.model_revision,
        "threshold": spec.threshold,
        "local_model_dir": str(model_dir.resolve()),
        "registry_sha256": file_sha256(model_config.path),
        "runtime": {"package": "funasr", "version": runtime_version},
        "files": files,
    }


def _speaker_verification_spec(model_config: ModelConfig) -> SpeakerVerificationSpec:
    spec = model_config.speaker_verification
    if spec is None:
        raise SpeakerEvidenceError(
            "speaker_verification is not configured in the active models.yaml."
        )
    return spec


def _resolve_cache_dir(cache_dir: Path | None) -> Path:
    cache = prepare_model_env(cache_dir or default_cache_dir())
    return cache.resolve()


def _source_reference(path: Path) -> dict[str, Any]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }


def _load_reference_set_manifest(path: Path) -> dict[str, Any]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"person:self reference-set manifest not found: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpeakerEvidenceError(f"Invalid person:self reference-set JSON: {source}") from exc
    if not isinstance(value, dict):
        raise SpeakerEvidenceError("person:self reference-set manifest must be a JSON object.")
    if value.get("schema") != SELF_SPEAKER_REFERENCE_SET_SCHEMA:
        raise SpeakerEvidenceError(
            f"Reference-set schema must be {SELF_SPEAKER_REFERENCE_SET_SCHEMA!r}."
        )
    return value


def _resolve_reference_set(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    _single_sentence(manifest.get("inference_basis"), "reference-set inference_basis")
    items = manifest.get("references")
    if not isinstance(items, list) or not (
        MIN_SELF_SPEAKER_REFERENCES <= len(items) <= MAX_SELF_SPEAKER_REFERENCES
    ):
        raise SpeakerEvidenceError(
            f"person:self reference set must contain {MIN_SELF_SPEAKER_REFERENCES} to "
            f"{MAX_SELF_SPEAKER_REFERENCES} references."
        )
    resolved: list[dict[str, Any]] = []
    source_hashes: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise SpeakerEvidenceError(f"Reference {index} must be a JSON object.")
        source_path = item.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            raise SpeakerEvidenceError(f"Reference {index} source_path is invalid.")
        source = _source_reference(Path(source_path))
        if source["sha256"] in source_hashes:
            raise SpeakerEvidenceError("Each person:self enrollment reference must use a distinct source hash.")
        source_hashes.add(source["sha256"])
        start, end = _bounded_interval(item.get("start_ms"), item.get("end_ms"))
        selection_binding = _selection_binding(item.get("selection_binding"), index=index)
        resolved.append(
            {
                "source_path": source["path"],
                "source_bytes": source["bytes"],
                "source_sha256": source["sha256"],
                "start_ms": start,
                "end_ms": end,
                "channel": _channel(item.get("channel", "mix")),
                "inference_basis": _single_sentence(
                    item.get("inference_basis"), f"reference {index} inference_basis"
                ),
                "selection_binding": selection_binding,
            }
        )
    return sorted(resolved, key=_reference_sort_key)


def _selection_binding(value: Any, *, index: int | None = None) -> dict[str, str]:
    label = f"Reference {index} selection_binding" if index is not None else "selection_binding"
    if not isinstance(value, Mapping):
        raise SpeakerEvidenceError(f"{label} must be a JSON object.")
    kind = value.get("kind")
    if kind not in _SELECTION_BINDING_KINDS:
        raise SpeakerEvidenceError(f"{label} kind is invalid.")
    evidence_hash = value.get("evidence_json_sha256")
    _validate_sha256(evidence_hash, f"{label} evidence JSON")
    raw_json_pointer = value.get("raw_json_pointer")
    if (
        not isinstance(raw_json_pointer, str)
        or not raw_json_pointer.strip()
        or "\n" in raw_json_pointer
        or len(raw_json_pointer) > 512
    ):
        raise SpeakerEvidenceError(f"{label} raw_json_pointer is invalid.")
    return {
        "kind": str(kind),
        "evidence_json_sha256": str(evidence_hash).lower(),
        "raw_json_pointer": raw_json_pointer.strip(),
    }


def _reference_sort_key(reference: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        reference["source_sha256"],
        float(reference["start_ms"]),
        float(reference["end_ms"]),
        reference["channel"],
    )


def _bounded_interval(
    start_ms: int | float,
    end_ms: int | float,
) -> tuple[int | float, int | float]:
    start = _finite_milliseconds(start_ms, "start_ms")
    end = _finite_milliseconds(end_ms, "end_ms")
    if end <= start:
        raise SpeakerEvidenceError("end_ms must be greater than start_ms.")
    if end - start > MAX_SPEAKER_EVIDENCE_DURATION_MS:
        raise SpeakerEvidenceError(
            f"One speaker-evidence interval may not exceed {MAX_SPEAKER_EVIDENCE_DURATION_MS} ms."
        )
    return start, end


def _finite_milliseconds(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpeakerEvidenceError(f"{label} must be a non-negative finite millisecond number.")
    if not math.isfinite(float(value)) or value < 0:
        raise SpeakerEvidenceError(f"{label} must be a non-negative finite millisecond number.")
    return int(value) if isinstance(value, int) else float(value)


def _channel(value: str) -> str:
    if value not in _CHANNELS:
        raise SpeakerEvidenceError("channel must be mix, left, or right.")
    return value


def _prepare_segment_wav(
    source: Path,
    destination: Path,
    *,
    start_ms: int | float,
    end_ms: int | float,
    channel: str,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
) -> str:
    """Extract one exact interval without claiming mixed audio is a channel fact."""

    executable = ffmpeg or shutil.which("ffmpeg")
    if not executable:
        raise SpeakerEvidenceError("ffmpeg is required to prepare speaker-verification evidence.")
    channel_binding = "mixed_not_channel_evidence"
    filters: list[str] = []
    if channel != "mix":
        source_channels = _audio_channel_count(source, ffprobe=ffprobe)
        if source_channels != 2:
            raise SpeakerEvidenceError(
                "left/right speaker evidence requires an original stereo (two-channel) source."
            )
        source_index = "0" if channel == "left" else "1"
        filters.append(f"pan=mono|c0=c{source_index}")
        channel_binding = "exact_stereo_channel"

    duration_ms = end_ms - start_ms
    command = [
        str(executable),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ss",
        _ffmpeg_seconds(start_ms),
        "-t",
        _ffmpeg_seconds(duration_ms),
        "-map_metadata",
        "-1",
        "-vn",
    ]
    if filters:
        command.extend(["-af", ",".join(filters)])
    command.extend(["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)])
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise SpeakerEvidenceError("ffmpeg timed out while preparing bounded speaker evidence.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        raise SpeakerEvidenceError(
            f"ffmpeg could not prepare speaker evidence{': ' + detail if detail else ''}"
        ) from exc
    output_format = _wav_format(destination)
    if output_format is None or output_format[:3] != (16000, 1, 2):
        raise SpeakerEvidenceError("Speaker-evidence preparation did not produce 16 kHz mono PCM WAV.")
    return channel_binding


def _audio_channel_count(source: Path, *, ffprobe: str | None = None) -> int:
    wav_format = _wav_format(source)
    if wav_format is not None:
        return wav_format[1]
    executable = ffprobe or shutil.which("ffprobe")
    if not executable:
        raise SpeakerEvidenceError(
            "ffprobe is required to verify an original non-WAV source has two channels."
        )
    try:
        result = subprocess.run(
            [
                str(executable),
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=channels",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        return int(result.stdout.strip())
    except (ValueError, subprocess.SubprocessError) as exc:
        raise SpeakerEvidenceError("Could not verify the original audio channel count.") from exc


def _ffmpeg_seconds(value_ms: int | float) -> str:
    return f"{float(value_ms) / 1000:.6f}"


def _wav_format(path: Path) -> tuple[int, int, int, float] | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            duration = handle.getnframes() / rate if rate else 0.0
    except (EOFError, OSError, wave.Error):
        return None
    return rate, channels, width, duration


def _extract_embedding(clip: Path, model_dir: str, device: str) -> list[float]:
    """Load the local CAM++ model once, return one vector, and release it."""

    ensure_funasr_available()
    from funasr import AutoModel

    model = AutoModel(model=model_dir, device=device, disable_update=True)
    try:
        result = model.generate(input=str(clip), cache={}, is_final=True)
        if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], Mapping):
            raise SpeakerEvidenceError("CAM++ returned no single speaker embedding for the prepared interval.")
        return _embedding_values(result[0].get("spk_embedding"))
    finally:
        del model
        gc.collect()
        _release_cuda_cache(device)


def _embedding_values(value: Any) -> list[float]:
    if value is None:
        raise SpeakerEvidenceError("CAM++ result did not contain spk_embedding.")
    candidate = value
    for method in ("detach", "cpu"):
        operation = getattr(candidate, method, None)
        if callable(operation):
            candidate = operation()
    reshape = getattr(candidate, "reshape", None)
    if callable(reshape):
        candidate = reshape(-1)
    tolist = getattr(candidate, "tolist", None)
    if callable(tolist):
        candidate = tolist()
    if not isinstance(candidate, (list, tuple)):
        raise SpeakerEvidenceError("CAM++ speaker embedding has an unsupported shape.")
    values = [float(item) for item in candidate]
    if not values or not all(math.isfinite(item) for item in values):
        raise SpeakerEvidenceError("CAM++ speaker embedding must contain finite values.")
    return values


def _release_cuda_cache(device: str) -> None:
    if not str(device).lower().startswith(("cuda", "gpu")):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _cosine_similarity(left: Any, right: Any) -> float:
    left_values = _profile_embedding(left)
    right_values = _profile_embedding(right)
    if len(left_values) != len(right_values):
        raise SpeakerEvidenceError("Profile and target speaker embeddings have different dimensions.")
    numerator = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if left_norm == 0 or right_norm == 0:
        raise SpeakerEvidenceError("Speaker embeddings must not be zero vectors.")
    similarity = numerator / (left_norm * right_norm)
    if not math.isfinite(similarity):
        raise SpeakerEvidenceError("Speaker similarity is not finite.")
    return similarity


def _profile_embedding(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        raise SpeakerEvidenceError("person:self profile embedding must be a non-empty numeric list.")
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise SpeakerEvidenceError("person:self profile embedding must be numeric.") from exc
    if not all(math.isfinite(item) for item in values):
        raise SpeakerEvidenceError("person:self profile embedding must be finite.")
    return values


def _l2_normalize(value: Any) -> list[float]:
    values = _profile_embedding(value)
    norm = math.sqrt(sum(item * item for item in values))
    if norm == 0:
        raise SpeakerEvidenceError("Speaker embeddings must not be zero vectors.")
    return [item / norm for item in values]


def _centroid(embeddings: Sequence[Sequence[float]]) -> list[float]:
    if not embeddings:
        raise SpeakerEvidenceError("Cannot create a person:self centroid without embeddings.")
    dimension = len(embeddings[0])
    if dimension == 0 or any(len(item) != dimension for item in embeddings):
        raise SpeakerEvidenceError("Enrollment speaker embeddings have different dimensions.")
    mean = [
        sum(float(embedding[index]) for embedding in embeddings) / len(embeddings)
        for index in range(dimension)
    ]
    return _l2_normalize(mean)


def _inferred_identity(basis: str) -> dict[str, Any]:
    normalized_basis = _single_sentence(basis, "inference_basis")
    return {
        "status": "inferred",
        "basis": normalized_basis,
        "reversible": True,
        "replacement": "可由更强、更新或用户明确确认的本人参考替换；替换后不保留旧向量。",
    }


def _validate_profile_shape(profile: Mapping[str, Any]) -> None:
    schema = profile.get("schema")
    if schema not in {SELF_SPEAKER_PROFILE_SCHEMA, SELF_SPEAKER_MULTI_PROFILE_SCHEMA}:
        raise SpeakerEvidenceError("Profile schema is unsupported.")
    if profile.get("person_id") != SELF_PERSON_ID:
        raise SpeakerEvidenceError("This command only accepts a person:self profile.")
    _validate_created_utc(profile.get("created_utc"))
    identity = profile.get("identity")
    if not isinstance(identity, Mapping):
        raise SpeakerEvidenceError("person:self profile has no reversible inferred identity basis.")
    if identity.get("status") != "inferred" or identity.get("reversible") is not True:
        raise SpeakerEvidenceError("person:self profile identity must be reversible inferred, never confirmed.")
    for field in ("basis", "replacement"):
        value = identity.get(field)
        if not isinstance(value, str) or not value.strip() or "\n" in value:
            raise SpeakerEvidenceError(f"person:self profile identity {field} must be one non-empty sentence.")
    if schema == SELF_SPEAKER_PROFILE_SCHEMA:
        _validate_single_reference_profile(profile)
    else:
        _validate_multi_reference_profile(profile)
    model = profile.get("model")
    if not isinstance(model, Mapping):
        raise SpeakerEvidenceError("person:self profile has no model evidence.")
    _validate_model_evidence(model)
    embedding = _profile_embedding(profile.get("embedding"))
    if schema == SELF_SPEAKER_MULTI_PROFILE_SCHEMA:
        norm = math.sqrt(sum(item * item for item in embedding))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise SpeakerEvidenceError("person:self centroid embedding must be L2-normalized.")


def _validate_single_reference_profile(profile: Mapping[str, Any]) -> None:
    reference = profile.get("enrollment_reference")
    if not isinstance(reference, Mapping):
        raise SpeakerEvidenceError("person:self profile has no enrollment_reference.")
    source = reference.get("source")
    segment = reference.get("segment")
    if not isinstance(source, Mapping) or not isinstance(segment, Mapping):
        raise SpeakerEvidenceError("person:self profile enrollment_reference is malformed.")
    _validate_source_hash(source, "profile enrollment source")
    _validate_segment(segment, "profile enrollment segment")


def _validate_multi_reference_profile(profile: Mapping[str, Any]) -> None:
    reference_set = profile.get("reference_set")
    if not isinstance(reference_set, Mapping):
        raise SpeakerEvidenceError("person:self profile has no reference_set.")
    if reference_set.get("schema") != SELF_SPEAKER_REFERENCE_SET_SCHEMA:
        raise SpeakerEvidenceError("person:self profile reference_set schema is invalid.")
    references = reference_set.get("references")
    if not isinstance(references, list) or not (
        MIN_SELF_SPEAKER_REFERENCES <= len(references) <= MAX_SELF_SPEAKER_REFERENCES
    ):
        raise SpeakerEvidenceError("person:self profile reference count is invalid.")
    if reference_set.get("reference_count") != len(references):
        raise SpeakerEvidenceError("person:self profile reference_count is inconsistent.")
    hashes: list[str] = []
    sort_keys: list[tuple[Any, ...]] = []
    for index, reference in enumerate(references):
        if not isinstance(reference, Mapping):
            raise SpeakerEvidenceError("person:self profile reference is malformed.")
        source = reference.get("source")
        segment = reference.get("segment")
        if not isinstance(source, Mapping) or not isinstance(segment, Mapping):
            raise SpeakerEvidenceError("person:self profile reference source/segment is malformed.")
        _validate_private_source_hash(source, "profile reference source")
        _validate_segment(segment, "profile reference segment")
        _single_sentence(reference.get("inference_basis"), f"profile reference {index} inference_basis")
        _selection_binding(reference.get("selection_binding"))
        hashes.append(str(source["sha256"]).lower())
        sort_keys.append(
            (
                str(source["sha256"]).lower(),
                float(segment["start_ms"]),
                float(segment["end_ms"]),
                segment["channel"],
            )
        )
    if len(set(hashes)) != len(hashes):
        raise SpeakerEvidenceError("person:self profile references must use distinct source hashes.")
    if sort_keys != sorted(sort_keys):
        raise SpeakerEvidenceError("person:self profile references are not in deterministic order.")
    expected_set_hash = canonical_json_sha256(
        {"schema": SELF_SPEAKER_REFERENCE_SET_SCHEMA, "references": references}
    )
    _validate_sha256(reference_set.get("sha256"), "profile reference set")
    if reference_set["sha256"].lower() != expected_set_hash:
        raise SpeakerEvidenceError("person:self profile reference_set hash is inconsistent.")
    aggregation = profile.get("aggregation")
    if not isinstance(aggregation, Mapping):
        raise SpeakerEvidenceError("person:self profile aggregation is missing.")
    if aggregation.get("method") != CENTROID_AGGREGATION_METHOD:
        raise SpeakerEvidenceError("person:self profile aggregation method is invalid.")
    if aggregation.get("reference_count") != len(references):
        raise SpeakerEvidenceError("person:self profile aggregation count is inconsistent.")


def _validate_segment(segment: Mapping[str, Any], label: str) -> None:
    _bounded_interval(segment.get("start_ms"), segment.get("end_ms"))
    channel = _channel(segment.get("channel"))
    binding = segment.get("channel_binding")
    expected = "mixed_not_channel_evidence" if channel == "mix" else "exact_stereo_channel"
    if binding != expected:
        raise SpeakerEvidenceError(f"{label} channel binding is invalid.")


def _validate_profile_for_model(profile: Mapping[str, Any], current_model: Mapping[str, Any]) -> None:
    _validate_profile_shape(profile)
    profile_model = profile["model"]
    if canonical_json_sha256(profile_model) != canonical_json_sha256(current_model):
        raise SpeakerEvidenceError(
            "person:self profile model/runtime/hash evidence changed; re-enroll the reference before comparing audio."
        )


def _validate_model_evidence(model: Mapping[str, Any]) -> None:
    if model.get("schema") != SPEAKER_MODEL_EVIDENCE_SCHEMA:
        raise SpeakerEvidenceError("person:self profile has an unsupported model evidence schema.")
    if not isinstance(model.get("model_id"), str) or not model["model_id"].strip():
        raise SpeakerEvidenceError("person:self profile model_id is invalid.")
    threshold = model.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise SpeakerEvidenceError("person:self profile model threshold is invalid.")
    files = model.get("files")
    if not isinstance(files, list) or not files:
        raise SpeakerEvidenceError("person:self profile model files are invalid.")
    for item in files:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise SpeakerEvidenceError("person:self profile model file record is invalid.")
        _validate_sha256(item.get("sha256"), "profile model file")


def _validate_source_hash(source: Mapping[str, Any], label: str) -> None:
    if not isinstance(source.get("path"), str) or not source["path"].strip():
        raise SpeakerEvidenceError(f"{label} path is invalid.")
    _validate_sha256(source.get("sha256"), label)
    if isinstance(source.get("bytes"), bool) or not isinstance(source.get("bytes"), int) or source["bytes"] < 0:
        raise SpeakerEvidenceError(f"{label} bytes are invalid.")


def _validate_private_source_hash(source: Mapping[str, Any], label: str) -> None:
    if "path" in source:
        raise SpeakerEvidenceError(f"{label} must not persist a source path.")
    _validate_sha256(source.get("sha256"), label)
    if isinstance(source.get("bytes"), bool) or not isinstance(source.get("bytes"), int) or source["bytes"] < 0:
        raise SpeakerEvidenceError(f"{label} bytes are invalid.")


def _profile_enrollment_source_hashes(profile: Mapping[str, Any]) -> set[str]:
    if profile["schema"] == SELF_SPEAKER_PROFILE_SCHEMA:
        return {str(profile["enrollment_reference"]["source"]["sha256"]).lower()}
    return {
        str(reference["source"]["sha256"]).lower()
        for reference in profile["reference_set"]["references"]
    }


def _evidence_profile_binding(profile: Mapping[str, Any], profile_hash: str) -> dict[str, Any]:
    common = {
        "schema": profile["schema"],
        "sha256": profile_hash,
        "identity_status": profile["identity"]["status"],
        "enrollment_basis": profile["identity"]["basis"],
    }
    if profile["schema"] == SELF_SPEAKER_PROFILE_SCHEMA:
        return {
            **common,
            "enrollment_source_sha256": profile["enrollment_reference"]["source"]["sha256"],
        }
    reference_set = profile["reference_set"]
    return {
        **common,
        "reference_set_sha256": reference_set["sha256"],
        "reference_count": reference_set["reference_count"],
        "enrollment_source_sha256s": [
            reference["source"]["sha256"] for reference in reference_set["references"]
        ],
    }


def _single_sentence(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise SpeakerEvidenceError(f"{label} must be one non-empty sentence.")
    return value.strip()


def _validate_created_utc(value: Any) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SpeakerEvidenceError("person:self profile created_utc is invalid.")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SpeakerEvidenceError("person:self profile created_utc is invalid.") from exc


def _validate_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise SpeakerEvidenceError(f"{label} sha256 is invalid.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise SpeakerEvidenceError(f"{label} sha256 is invalid.") from exc


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

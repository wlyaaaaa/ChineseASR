"""Objective audio-result semantics kept separate from ASR execution evidence.

The ASR bundle answers whether the configured engines ran and whether their
artifacts are internally consistent.  This module answers the narrower audio
question: what can be said about an empty (or non-empty) result from the
available, explicitly recorded speech-detection evidence.  In particular,
empty text is never treated as proof of silence.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import wave
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


OBJECTIVE_RESULT_SCHEMA = "media.objective-result.v1"
NEGATIVE_EVIDENCE_SCHEMA = "media.negative-evidence.v1"
OBJECTIVE_PROCESSOR = "pcm-signal-inspection"
OBJECTIVE_PROCESSOR_VERSION = "1"
OBJECTIVE_STATUSES = {
    "speech_transcribed",
    "no_speech_detected",
    "speech_detected_but_not_transcribable",
    "indeterminate",
}
EXECUTION_STATUSES = {"completed", "failed", "unsupported", "corrupt"}
COVERAGE_STATUSES = {"complete", "partial", "unknown"}
QUALITY_STATUSES = {"sufficient", "low_confidence", "unknown"}
RAW_ARTIFACT_REF_SCHEMA = "media.raw-artifact-ref.v1"

_PCM_SAMPLE_RMS_THRESHOLD = 0.01
_PCM_FRAME_MS = 20
_POLICY = {
    "empty_text_never_proves_no_speech": True,
    "zero_pcm_only_is_negative_without_vad": True,
    "vad_requires_complete_coverage": True,
}


def canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_audio(
    audio_path: Path | str,
    *,
    analysis_path: Path | str | None = None,
) -> dict[str, Any]:
    """Return bounded, deterministic signal facts for a readable PCM WAV.

    This deliberately confirms silence only for an exact all-zero PCM payload.
    Non-zero audio is not called speech merely because it has energy; callers
    need a model VAD result for that stronger claim.  Unsupported containers,
    malformed WAVs and zero-byte payloads remain separately observable.
    """

    source = Path(audio_path).expanduser()
    candidate = Path(analysis_path).expanduser() if analysis_path else source
    result: dict[str, Any] = {
        "processor": OBJECTIVE_PROCESSOR,
        "processor_version": OBJECTIVE_PROCESSOR_VERSION,
        "source_path": str(source.resolve()) if source.exists() else str(source),
        "analysis_path": str(candidate.resolve()) if candidate.exists() else str(candidate),
        "source_sha256": _safe_file_sha256(source),
        "source_size_bytes": _safe_file_size(source),
        "status": "unsupported",
        "reason": "pcm_wav_not_available",
        "coverage_complete": False,
        "coverage": {"start_ms": 0, "end_ms": None, "excluded_ranges_ms": []},
        "sample_rate": None,
        "channels": None,
        "sample_width_bytes": None,
        "frame_count": 0,
        "duration_ms": None,
        "nonzero_sample_count": None,
        "peak_normalized": None,
        "rms_normalized": None,
        "active_frame_count": None,
        "thresholds": {
            "sample_rms_normalized": _PCM_SAMPLE_RMS_THRESHOLD,
            "frame_ms": _PCM_FRAME_MS,
        },
    }

    if not source.exists() or not source.is_file():
        result["status"] = "corrupt"
        result["reason"] = "audio_file_missing"
        return result
    if source.stat().st_size == 0:
        result["status"] = "corrupt"
        result["reason"] = "zero_byte_audio"
        return result
    if not candidate.exists() or not candidate.is_file():
        result["reason"] = "analysis_audio_missing"
        return result
    if candidate.stat().st_size == 0:
        result["status"] = "corrupt"
        result["reason"] = "zero_byte_analysis_audio"
        return result
    if candidate.suffix.lower() not in {".wav", ".wave"}:
        result["reason"] = "unsupported_analysis_container"
        return result

    try:
        with wave.open(str(candidate), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            compression = handle.getcomptype()
            if compression != "NONE":
                result["reason"] = "compressed_wav"
                return result
            if channels <= 0 or sample_width not in {1, 2, 4} or sample_rate <= 0:
                result["status"] = "corrupt"
                result["reason"] = "invalid_pcm_format"
                return result
            if frame_count <= 0:
                result["status"] = "corrupt"
                result["reason"] = "empty_pcm_payload"
                return result

            peak = 0
            nonzero = 0
            square_sum = 0.0
            sample_count = 0
            active_frames = 0
            frame_samples = max(1, int(round(sample_rate * _PCM_FRAME_MS / 1000)))
            while True:
                raw = handle.readframes(frame_samples)
                if not raw:
                    break
                values = _decode_pcm_samples(raw, sample_width)
                if not values:
                    continue
                frame_peak = 0
                frame_square_sum = 0.0
                for value in values:
                    magnitude = abs(value)
                    frame_peak = max(frame_peak, magnitude)
                    peak = max(peak, magnitude)
                    if magnitude:
                        nonzero += 1
                    normalized = magnitude / _pcm_full_scale(sample_width)
                    frame_square_sum += normalized * normalized
                    square_sum += normalized * normalized
                    sample_count += 1
                frame_rms = math.sqrt(frame_square_sum / len(values))
                if frame_rms >= _PCM_SAMPLE_RMS_THRESHOLD:
                    active_frames += 1
    except (OSError, EOFError, wave.Error, struct.error, ValueError) as exc:
        result["status"] = "corrupt"
        result["reason"] = f"pcm_read_failed:{type(exc).__name__}"
        return result

    duration_ms = round(frame_count * 1000 / sample_rate)
    peak_normalized = peak / _pcm_full_scale(sample_width)
    rms_normalized = math.sqrt(square_sum / sample_count) if sample_count else 0.0
    result.update(
        {
            "status": "no_speech_detected" if peak == 0 else "indeterminate",
            "reason": "all_pcm_samples_zero" if peak == 0 else "nonzero_audio_without_vad",
            "coverage_complete": True,
            "coverage": {
                "start_ms": 0,
                "end_ms": duration_ms,
                "excluded_ranges_ms": [],
            },
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
            "frame_count": frame_count,
            "duration_ms": duration_ms,
            "nonzero_sample_count": nonzero,
            "peak_normalized": round(peak_normalized, 9),
            "rms_normalized": round(rms_normalized, 9),
            "active_frame_count": active_frames,
        }
    )
    # Energy alone is intentionally not a speech detector.  A noisy room,
    # music, or a clipped/corrupt stream must remain indeterminate until an
    # engine-owned VAD supplies speech intervals.
    return result


def classify_objective_outcome(
    *,
    primary_text: str,
    secondary_text: str = "",
    primary_error: str | None = None,
    secondary_error: str | None = None,
    primary_result: Any = None,
    secondary_result: Any = None,
    audio_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify result semantics without upgrading an empty transcript."""

    errors = [error for error in (primary_error, secondary_error) if error]
    texts = [str(primary_text or "").strip(), str(secondary_text or "").strip()]
    detection = _merge_detection_hints(primary_result, secondary_result)
    if errors:
        failure_class = _failure_class(errors)
        if any(texts):
            return {
                "objective_outcome": "speech_transcribed",
                "empty_semantics": "uncertain_empty",
                "confidence": "low",
                "execution_status": _execution_status_for_failure(failure_class),
                "coverage_status": "unknown",
                "quality_status": "low_confidence",
                "failure_class": failure_class,
                "reason": "text_returned_with_engine_failure",
                "transcript_available": True,
                "qualifiers": ["engine_failure"],
                "detection": detection,
            }
        return {
            "objective_outcome": "indeterminate",
            "empty_semantics": "unknown",
            "confidence": "unknown",
            "execution_status": _execution_status_for_failure(failure_class),
            "coverage_status": "unknown",
            "quality_status": "unknown",
            "failure_class": failure_class,
            "reason": "one_or_more_engines_failed",
            "transcript_available": bool(any(texts)),
            "detection": detection,
        }

    if any(texts):
        qualifiers: list[str] = []
        if not all(texts):
            qualifiers.append("single_engine_empty")
        return {
            "objective_outcome": "speech_transcribed",
            "empty_semantics": "uncertain_empty" if qualifiers else None,
            "confidence": "low" if qualifiers else "observed",
            "execution_status": "completed",
            "coverage_status": "complete",
            "quality_status": "low_confidence" if qualifiers else "sufficient",
            "failure_class": None,
            "reason": "at_least_one_engine_returned_text",
            "transcript_available": True,
            "qualifiers": qualifiers,
            "detection": detection,
        }

    explicit_status = str(detection.get("status") or "")
    explicit_complete = bool(detection.get("coverage_complete"))
    if explicit_status == "no_speech_detected" and explicit_complete:
        return {
            "objective_outcome": "no_speech_detected",
            "empty_semantics": "confirmed_no_speech",
            "confidence": "high",
            "execution_status": "completed",
            "coverage_status": "complete",
            "quality_status": "sufficient",
            "failure_class": None,
            "reason": "complete_speech_detection_returned_zero_segments",
            "transcript_available": False,
            "detection": detection,
        }
    if explicit_status in {"speech_detected", "speech_detected_but_not_transcribable"} and explicit_complete:
        return {
            "objective_outcome": "speech_detected_but_not_transcribable",
            "empty_semantics": "uncertain_empty",
            "confidence": "deferred",
            "execution_status": "completed",
            "coverage_status": "complete",
            "quality_status": "unknown",
            "failure_class": None,
            "reason": "speech_detection_found_segments_but_transcript_is_empty",
            "transcript_available": False,
            "detection": detection,
        }
    if explicit_status in {"speech_detected", "speech_detected_but_not_transcribable"}:
        return {
            "objective_outcome": "indeterminate",
            "empty_semantics": "uncertain_empty",
            "confidence": "unknown",
            "execution_status": "completed",
            "coverage_status": "partial" if detection.get("segments") else "unknown",
            "quality_status": "unknown",
            "failure_class": None,
            "reason": "speech_detection_coverage_is_incomplete",
            "transcript_available": False,
            "detection": detection,
        }

    analysis_status = str((audio_analysis or {}).get("status") or "")
    analysis_complete = bool((audio_analysis or {}).get("coverage_complete"))
    if analysis_status == "no_speech_detected" and analysis_complete:
        return {
            "objective_outcome": "no_speech_detected",
            "empty_semantics": "confirmed_no_speech",
            "confidence": "high",
            "execution_status": "completed",
            "coverage_status": "complete",
            "quality_status": "sufficient",
            "failure_class": None,
            "reason": "complete_zero_pcm_negative_evidence",
            "transcript_available": False,
            "detection": detection,
        }
    if analysis_status == "speech_detected_but_not_transcribable":
        return {
            "objective_outcome": "speech_detected_but_not_transcribable",
            "empty_semantics": "uncertain_empty",
            "confidence": "deferred",
            "execution_status": "completed",
            "coverage_status": "complete",
            "quality_status": "unknown",
            "failure_class": None,
            "reason": "signal_activity_without_transcript",
            "transcript_available": False,
            "detection": detection,
        }
    if analysis_status in {"unsupported", "corrupt"}:
        return {
            "objective_outcome": "indeterminate",
            "empty_semantics": "uncertain_empty",
            "confidence": "unknown",
            "execution_status": analysis_status,
            "coverage_status": "unknown",
            "quality_status": "unknown",
            "failure_class": "audio_analysis_unavailable",
            "reason": str((audio_analysis or {}).get("reason") or analysis_status),
            "transcript_available": False,
            "detection": detection,
        }
    return {
        "objective_outcome": "indeterminate",
        "empty_semantics": "uncertain_empty",
        "confidence": "unknown",
        "execution_status": "completed",
        "coverage_status": "unknown",
        "quality_status": "unknown",
        "failure_class": None,
        "reason": "empty_transcript_without_complete_speech_detection",
        "transcript_available": False,
        "detection": detection,
    }


def build_objective_result(
    *,
    audio_path: Path | str,
    mode: str,
    engines: Sequence[str],
    primary_text: str,
    secondary_text: str = "",
    primary_result: Any = None,
    secondary_result: Any = None,
    primary_error: str | None = None,
    secondary_error: str | None = None,
    primary_provenance: Mapping[str, Any] | None = None,
    secondary_provenance: Mapping[str, Any] | None = None,
    raw_artifacts: Sequence[Mapping[str, Any]] = (),
    strict_receipt: Mapping[str, Any] | None = None,
    request: Mapping[str, Any] | None = None,
    caller_binding: Mapping[str, Any] | None = None,
    coverage: Mapping[str, Any] | None = None,
    audio_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(audio_path).expanduser()
    analysis_path = _provenance_audio_path(primary_provenance) or _provenance_audio_path(
        secondary_provenance
    )
    analysis = dict(audio_analysis or analyze_audio(source, analysis_path=analysis_path))
    classification = classify_objective_outcome(
        primary_text=primary_text,
        secondary_text=secondary_text,
        primary_error=primary_error,
        secondary_error=secondary_error,
        primary_result=primary_result,
        secondary_result=secondary_result,
        audio_analysis=analysis,
    )
    request_payload = {
        "mode": mode,
        "engines": list(engines),
        **dict(request or {}),
    }
    if caller_binding is not None:
        request_payload["caller_binding"] = dict(caller_binding)
    request_hash = canonical_json_sha256(request_payload)
    detector_config = {
        "processor": OBJECTIVE_PROCESSOR,
        "processor_version": OBJECTIVE_PROCESSOR_VERSION,
        "thresholds": analysis.get("thresholds", {}),
        "policy": _POLICY,
    }
    processor_config_hash = canonical_json_sha256(detector_config)
    policy_hash = canonical_json_sha256(_POLICY)
    coverage_payload = _coverage_payload(analysis, coverage)
    raw_payloads = [dict(item) for item in raw_artifacts]
    detection_payload = dict(classification.get("detection") or {})
    detection_hash = canonical_json_sha256(detection_payload)
    execution_status = _coerce_execution_status(
        classification.get("execution_status"),
    )
    coverage_status = _coverage_status_from_payload(
        coverage_payload,
        preferred=classification.get("coverage_status"),
    )
    quality_status = _coerce_quality_status(classification.get("quality_status"))
    compatibility = {
        "execution_status": _legacy_execution_status(execution_status),
        "coverage_status": _legacy_coverage_status(coverage_status),
        "quality_status": _legacy_quality_status(
            quality_status,
            confidence=classification.get("confidence"),
        ),
    }
    model_payload = [
        _model_provenance(engines[0] if engines else "", primary_provenance),
        _model_provenance(engines[1] if len(engines) > 1 else "", secondary_provenance),
    ]
    basis = {
        "source_audio_sha256": analysis.get("source_sha256") or _safe_file_sha256(source),
        "request_sha256": request_hash,
        "processor_config_sha256": processor_config_hash,
        "policy_sha256": policy_hash,
        "speech_detection_sha256": detection_hash,
        "models": model_payload,
    }
    payload: dict[str, Any] = {
        "schema": OBJECTIVE_RESULT_SCHEMA,
        "schema_version": 1,
        "media_kind": "audio",
        "objective_outcome": classification["objective_outcome"],
        "result_status": classification["objective_outcome"],
        "audio_result_status": classification["objective_outcome"],
        "empty_semantics": classification.get("empty_semantics"),
        "confidence": classification.get("confidence"),
        "reason": classification.get("reason"),
        "failure_class": classification.get("failure_class"),
        "execution": {
            "status": execution_status,
        },
        "coverage": {
            "status": coverage_status,
            **coverage_payload,
        },
        "quality": {
            "status": quality_status,
        },
        # These aliases keep early sidecar consumers readable while the
        # nested fields above are the versioned contract.
        "compatibility": compatibility,
        "execution_status": compatibility["execution_status"],
        "coverage_status": compatibility["coverage_status"],
        "quality_status": compatibility["quality_status"],
        "failure": (
            {"class": classification.get("failure_class"), "reason": classification.get("reason")}
            if classification.get("failure_class")
            else None
        ),
        "qualifiers": list(classification.get("qualifiers", [])),
        "transcript_available": bool(classification.get("transcript_available")),
        "audio": {
            "raw_sha256": basis["source_audio_sha256"],
            "size_bytes": analysis.get("source_size_bytes"),
            "analysis_path": analysis.get("analysis_path"),
            "analysis_sha256": _safe_file_sha256(Path(str(analysis.get("analysis_path"))))
            if analysis.get("analysis_path")
            else "",
            "coverage": coverage_payload,
            "exclusions_ms": list(coverage_payload.get("excluded_ranges_ms", [])),
        },
        "processor": detector_config,
        "processor_config_sha256": processor_config_hash,
        "policy_sha256": policy_hash,
        "request": {**request_payload, "sha256": request_hash},
        "models": model_payload,
        "analysis": analysis,
        "raw_artifacts": raw_payloads,
        "strict_receipt": dict(strict_receipt or {}),
        "detection": detection_payload,
        "idempotency_basis": basis,
        "idempotency_key": canonical_json_sha256(basis),
    }
    if caller_binding is not None:
        payload["caller_binding"] = dict(caller_binding)
    if payload["objective_outcome"] == "no_speech_detected":
        negative = {
            "schema": NEGATIVE_EVIDENCE_SCHEMA,
            "schema_version": 1,
            "kind": "complete_zero_pcm_or_vad_zero_segments",
            "non_empty": True,
            "coverage": coverage_payload,
            "excluded_ranges_ms": list(coverage_payload.get("excluded_ranges_ms", [])),
            "thresholds": analysis.get("thresholds", {}),
            "detectors": {
                "pcm_signal_inspection": detector_config,
                "speech_detection": dict(classification.get("detection") or {}),
            },
            "detection": dict(classification.get("detection") or {}),
            "observed": {
                "status": analysis.get("status"),
                "reason": analysis.get("reason"),
                "frame_count": analysis.get("frame_count"),
                "nonzero_sample_count": analysis.get("nonzero_sample_count"),
                "segments": (classification.get("detection") or {}).get("segments", []),
                "speech_detection": dict(classification.get("detection") or {}),
            },
        }
        payload["negative_evidence"] = _hashed_negative_evidence(negative)
    else:
        payload["negative_evidence"] = None
    return payload


def write_objective_result(path: Path | str, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_objective_result(path: Path | str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def validate_objective_result(
    payload: Mapping[str, Any] | None,
    *,
    raw_artifacts: Sequence[Mapping[str, Any]] | None = None,
    strict_receipt: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate sidecar identity without requiring it for old strict bundles."""

    if not isinstance(payload, Mapping):
        return ["objective result sidecar is missing or not an object"]
    failures: list[str] = []
    if payload.get("schema") != OBJECTIVE_RESULT_SCHEMA:
        failures.append("objective result schema is unsupported")
    if payload.get("media_kind") != "audio":
        failures.append("objective result media_kind must be audio")
    status = str(payload.get("objective_outcome") or "")
    if status not in OBJECTIVE_STATUSES:
        failures.append("objective result status is invalid")
    audio = payload.get("audio")
    if not isinstance(audio, Mapping) or not _is_sha256(audio.get("raw_sha256")):
        failures.append("objective result audio.raw_sha256 is missing or invalid")
    execution = payload.get("execution")
    if not isinstance(execution, Mapping) or execution.get("status") not in EXECUTION_STATUSES:
        failures.append("objective result execution.status is invalid")
    coverage = payload.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("status") not in COVERAGE_STATUSES:
        failures.append("objective result coverage.status is invalid")
    quality = payload.get("quality")
    if not isinstance(quality, Mapping) or quality.get("status") not in QUALITY_STATUSES:
        failures.append("objective result quality.status is invalid")
    if status == "no_speech_detected":
        if not (
            isinstance(execution, Mapping)
            and execution.get("status") == "completed"
            and isinstance(coverage, Mapping)
            and coverage.get("status") == "complete"
            and isinstance(quality, Mapping)
            and quality.get("status") == "sufficient"
        ):
            failures.append(
                "no_speech_detected requires completed execution, complete coverage, "
                "and sufficient quality"
            )
    basis = payload.get("idempotency_basis")
    if not isinstance(basis, Mapping) or payload.get("idempotency_key") != canonical_json_sha256(basis):
        failures.append("objective result idempotency key is invalid")
    request = payload.get("request")
    if not isinstance(request, Mapping) or not _is_sha256(request.get("sha256")):
        failures.append("objective result request hash is missing or invalid")
    elif canonical_json_sha256({key: value for key, value in request.items() if key != "sha256"}) != request.get("sha256"):
        failures.append("objective result request hash does not match request")
    elif not isinstance(basis, Mapping) or basis.get("request_sha256") != request.get("sha256"):
        failures.append("objective result idempotency basis request hash does not match")
    processor = payload.get("processor")
    if not isinstance(processor, Mapping):
        failures.append("objective result processor is missing")
    else:
        expected_hash = canonical_json_sha256(dict(processor))
        if payload.get("processor_config_sha256") != expected_hash:
            failures.append("objective result processor config hash is invalid")
    if not _is_sha256(payload.get("policy_sha256")):
        failures.append("objective result policy hash is missing or invalid")
    negative = payload.get("negative_evidence")
    if status == "no_speech_detected":
        if not isinstance(negative, Mapping):
            failures.append("no_speech_detected requires negative evidence")
        else:
            if negative.get("schema") != NEGATIVE_EVIDENCE_SCHEMA:
                failures.append("negative evidence schema is unsupported")
            if not isinstance(negative.get("size_bytes"), int) or negative.get("size_bytes", 0) <= 0:
                failures.append("negative evidence size_bytes must be positive")
            else:
                envelope_without_hash = {
                    key: value for key, value in negative.items() if key != "sha256"
                }
                if negative.get("sha256") != canonical_json_sha256(envelope_without_hash):
                    failures.append("negative evidence hash is invalid")
            artifact = negative.get("artifact")
            if not isinstance(artifact, Mapping) or not artifact.get("non_empty"):
                failures.append("negative evidence artifact must be non-empty")
            elif artifact.get("schema") != NEGATIVE_EVIDENCE_SCHEMA:
                failures.append("negative evidence artifact schema is unsupported")
            elif not isinstance(artifact.get("size_bytes"), int) or artifact.get("size_bytes", 0) <= 0:
                failures.append("negative evidence artifact size_bytes must be positive")
            else:
                artifact_without_hash = {
                    key: value
                    for key, value in artifact.items()
                    if key not in {"sha256", "size_bytes"}
                }
                if artifact.get("sha256") != canonical_json_sha256(artifact_without_hash):
                    failures.append("negative evidence artifact hash is invalid")
    elif negative is not None:
        failures.append("negative evidence is only valid for no_speech_detected")
    declared_raw_artifacts = payload.get("raw_artifacts")
    failures.extend(_validate_declared_artifact_refs(declared_raw_artifacts, "raw"))
    if raw_artifacts is not None:
        failures.extend(_validate_artifact_refs(declared_raw_artifacts, raw_artifacts, "raw"))
    declared = payload.get("strict_receipt")
    if declared:
        failures.extend(
            _validate_declared_artifact_refs(
                [declared],
                "strict receipt",
                schema="media.strict-receipt-ref.v1",
            )
        )
    if strict_receipt is not None:
        if not isinstance(declared, Mapping) or dict(declared) != dict(strict_receipt):
            failures.append("objective result strict receipt reference does not match")
    return failures


def aggregate_objective_result(
    *,
    audio_path: Path | str,
    mode: str,
    engines: Sequence[str],
    children: Iterable[Mapping[str, Any]],
    request: Mapping[str, Any] | None = None,
    caller_binding: Mapping[str, Any] | None = None,
    primary_provenance: Mapping[str, Any] | None = None,
    secondary_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    child_list = [dict(child) for child in children]
    statuses = [str(child.get("objective_outcome") or "indeterminate") for child in child_list]
    executions = [_child_execution_status(child) for child in child_list]
    coverage = _aggregate_coverage(child_list)
    source_analysis = analyze_audio(audio_path)
    source_duration = source_analysis.get("duration_ms")
    if not isinstance(source_duration, (int, float)) and isinstance(request, Mapping):
        source_duration = request.get("duration_ms")
    coverage_complete = _coverage_is_complete(
        coverage,
        duration_ms=source_duration,
    )
    if not child_list:
        aggregate_status = "indeterminate"
    elif any(execution != "completed" for execution in executions):
        aggregate_status = "indeterminate"
    elif not coverage_complete:
        aggregate_status = "indeterminate"
    elif any(status == "speech_transcribed" for status in statuses):
        aggregate_status = "speech_transcribed"
    elif any(status == "speech_detected_but_not_transcribable" for status in statuses):
        aggregate_status = "speech_detected_but_not_transcribable"
    elif any(status == "indeterminate" for status in statuses):
        aggregate_status = "indeterminate"
    elif all(status == "no_speech_detected" for status in statuses):
        aggregate_status = "no_speech_detected"
    else:
        aggregate_status = "indeterminate"

    if child_list and all(item == "completed" for item in executions):
        execution_status = "completed"
    else:
        execution_status = next(
            (item for item in executions if item in {"corrupt", "unsupported", "failed"}),
            "failed",
        )
    aggregate_coverage_status = "complete" if coverage_complete else (
        "partial" if coverage.get("intervals_ms") else "unknown"
    )
    payload = build_objective_result(
        audio_path=audio_path,
        mode=mode,
        engines=engines,
        primary_text="",
        secondary_text="",
        primary_provenance=primary_provenance,
        secondary_provenance=secondary_provenance,
        request=request,
        caller_binding=caller_binding,
        coverage=coverage,
        audio_analysis=source_analysis,
    )
    payload["objective_outcome"] = aggregate_status
    payload["result_status"] = aggregate_status
    payload["audio_result_status"] = aggregate_status
    payload["empty_semantics"] = (
        "confirmed_no_speech" if aggregate_status == "no_speech_detected" else "uncertain_empty"
        if aggregate_status in {"indeterminate", "speech_detected_but_not_transcribable"}
        else None
    )
    payload["confidence"] = "high" if aggregate_status == "no_speech_detected" else "unknown"
    payload["reason"] = "aggregate_chunk_objective_results"
    aggregate_quality_status = (
        "sufficient"
        if aggregate_status in {"speech_transcribed", "no_speech_detected"}
        else "unknown"
    )
    payload["execution"] = {"status": execution_status}
    payload["coverage"] = {
        "status": aggregate_coverage_status,
        **dict(coverage),
    }
    payload["quality"] = {"status": aggregate_quality_status}
    payload["compatibility"] = {
        "execution_status": _legacy_execution_status(execution_status),
        "coverage_status": _legacy_coverage_status(aggregate_coverage_status),
        "quality_status": _legacy_quality_status(
            aggregate_quality_status,
            confidence=payload.get("confidence"),
        ),
    }
    payload["execution_status"] = payload["compatibility"]["execution_status"]
    payload["coverage_status"] = payload["compatibility"]["coverage_status"]
    payload["quality_status"] = payload["compatibility"]["quality_status"]
    payload["children"] = child_list
    payload["idempotency_basis"] = {
        **dict(payload.get("idempotency_basis") or {}),
        "children": [
            {
                "chunk_id": child.get("chunk_id"),
                "objective_outcome": child.get("objective_outcome"),
                "idempotency_key": child.get("idempotency_key"),
            }
            for child in child_list
        ],
    }
    payload["idempotency_key"] = canonical_json_sha256(payload["idempotency_basis"])
    if aggregate_status == "no_speech_detected":
        negative = {
            "schema": NEGATIVE_EVIDENCE_SCHEMA,
            "schema_version": 1,
            "kind": "complete_chunk_vad_zero_segments",
            "non_empty": True,
            "coverage": coverage,
            "excluded_ranges_ms": list(coverage.get("excluded_ranges_ms", [])),
            "children": [
                {
                    "chunk_id": child.get("chunk_id"),
                    "objective_outcome": child.get("objective_outcome"),
                    "idempotency_key": child.get("idempotency_key"),
                    "raw_artifacts": child.get("raw_artifacts", []),
                    "detection": child.get("detection", {}),
                    "detection_global": child.get("detection_global", child.get("detection", {})),
                    "chunk_interval_ms": child.get("chunk_interval_ms"),
                    "chunk_local_coverage": child.get("chunk_local_coverage"),
                    "coverage": (child.get("audio") or {}).get("coverage", {})
                    if isinstance(child.get("audio"), Mapping)
                    else {},
                }
                for child in child_list
            ],
            "source_audio_sha256": (payload.get("audio") or {}).get("raw_sha256"),
        }
        payload["negative_evidence"] = _hashed_negative_evidence(negative)
    else:
        payload["negative_evidence"] = None
    return payload


def _coerce_execution_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "completed": "completed",
        "succeeded": "completed",
        "success": "completed",
        "failed": "failed",
        "failure": "failed",
        "engine_failure": "failed",
        "partial": "failed",
        "unsupported": "unsupported",
        "corrupt": "corrupt",
    }
    return aliases.get(normalized, "failed")


def _coerce_coverage_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "complete": "complete",
        "completed": "complete",
        "partial": "partial",
        "partial_coverage": "partial",
        "incomplete": "unknown",
        "unknown": "unknown",
        "unsupported": "unknown",
        "corrupt": "unknown",
    }
    return aliases.get(normalized, "unknown")


def _coerce_quality_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    aliases = {
        "sufficient": "sufficient",
        "high_confidence": "sufficient",
        "observed": "sufficient",
        "low_confidence": "low_confidence",
        "low": "low_confidence",
        "unknown": "unknown",
        "deferred": "unknown",
    }
    return aliases.get(normalized, "unknown")


def _legacy_execution_status(value: str) -> str:
    return {
        "completed": "succeeded",
        "failed": "engine_failure",
        "unsupported": "unsupported",
        "corrupt": "corrupt",
    }.get(value, "engine_failure")


def _legacy_coverage_status(value: str) -> str:
    return {
        "complete": "complete",
        "partial": "partial_coverage",
        "unknown": "incomplete",
    }.get(value, "incomplete")


def _legacy_quality_status(value: str, *, confidence: Any = None) -> str:
    if value == "sufficient":
        return "high_confidence" if confidence == "high" else "observed"
    return value


def _execution_status_for_failure(failure_class: str) -> str:
    normalized = str(failure_class or "").lower()
    if "unsupported" in normalized:
        return "unsupported"
    if "corrupt" in normalized:
        return "corrupt"
    return "failed"


def _coverage_status_from_payload(
    payload: Mapping[str, Any],
    *,
    preferred: Any = None,
) -> str:
    preferred_status = _coerce_coverage_status(preferred)
    if preferred_status in {"partial", "unknown"}:
        return preferred_status
    if payload.get("excluded_ranges_ms") or payload.get("gap_ms"):
        return "partial"
    if bool(payload.get("complete")):
        return "complete"
    if payload.get("intervals_ms"):
        return "partial"
    return _coerce_coverage_status(preferred)


def _child_execution_status(child: Mapping[str, Any]) -> str:
    execution = child.get("execution")
    if isinstance(execution, Mapping):
        return _coerce_execution_status(execution.get("status"))
    return _coerce_execution_status(child.get("execution_status"))


def _hashed_negative_evidence(core: Mapping[str, Any]) -> dict[str, Any]:
    artifact_core = dict(core)
    artifact_core.pop("sha256", None)
    artifact_core.pop("size_bytes", None)
    artifact = {
        **artifact_core,
        "size_bytes": _canonical_size_bytes(artifact_core),
        "sha256": canonical_json_sha256(artifact_core),
    }
    envelope = {
        "schema": NEGATIVE_EVIDENCE_SCHEMA,
        "schema_version": 1,
        "artifact": artifact,
    }
    envelope["size_bytes"] = _canonical_size_bytes(envelope)
    envelope["sha256"] = canonical_json_sha256(envelope)
    return envelope


def _canonical_size_bytes(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _decode_pcm_samples(raw: bytes, sample_width: int) -> list[int]:
    usable = len(raw) - (len(raw) % sample_width)
    if sample_width == 1:
        return [byte - 128 for byte in raw[:usable]]
    if sample_width == 2:
        return list(struct.unpack("<" + "h" * (usable // 2), raw[:usable]))
    if sample_width == 4:
        return list(struct.unpack("<" + "i" * (usable // 4), raw[:usable]))
    return []


def _pcm_full_scale(sample_width: int) -> float:
    return float({1: 128, 2: 32768, 4: 2147483648}[sample_width])


def _safe_file_sha256(path: Path) -> str:
    try:
        return file_sha256(path)
    except (OSError, ValueError):
        return ""


def _safe_file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _provenance_audio_path(provenance: Mapping[str, Any] | None) -> Path | None:
    if not isinstance(provenance, Mapping):
        return None
    value = provenance.get("audio")
    if not isinstance(value, Mapping):
        return None
    for key in ("path", "derivative_path"):
        candidate = value.get(key)
        if candidate and Path(str(candidate)).exists():
            return Path(str(candidate))
    return None


def _model_provenance(engine: str, provenance: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        return {"engine": engine}
    return {
        "engine": engine,
        "adapter": provenance.get("adapter"),
        "model": provenance.get("model"),
        "runtime_identity": dict(provenance.get("runtime_identity") or {})
        if isinstance(provenance.get("runtime_identity"), Mapping)
        else {},
    }


def _coverage_payload(
    analysis: Mapping[str, Any],
    coverage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if coverage:
        start = coverage.get("start_ms", 0)
        end = coverage.get("end_ms", analysis.get("duration_ms"))
        excluded = list(coverage.get("excluded_ranges_ms", []))
        complete = bool(coverage.get("complete", coverage.get("coverage_complete", False)))
        intervals = list(coverage.get("intervals_ms", []))
        overlap_ms = coverage.get("overlap_ms", 0)
        gap_ms = coverage.get("gap_ms", 0)
    else:
        raw = analysis.get("coverage")
        raw = raw if isinstance(raw, Mapping) else {}
        start = raw.get("start_ms", 0)
        end = raw.get("end_ms", analysis.get("duration_ms"))
        excluded = list(raw.get("excluded_ranges_ms", []))
        complete = bool(analysis.get("coverage_complete"))
        intervals = list(raw.get("intervals_ms", []))
        overlap_ms = raw.get("overlap_ms", 0)
        gap_ms = raw.get("gap_ms", 0)
    return {
        "start_ms": start,
        "end_ms": end,
        "intervals_ms": intervals,
        "excluded_ranges_ms": excluded,
        "overlap_ms": overlap_ms,
        "gap_ms": gap_ms,
        "complete": complete,
    }


def _aggregate_coverage(children: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    intervals: list[tuple[float, float]] = []
    complete = bool(children)
    excluded: list[Any] = []
    for child in children:
        declared_coverage = child.get("coverage")
        if isinstance(declared_coverage, Mapping):
            if declared_coverage.get("status") != "complete":
                complete = False
        elif "coverage_status" in child and _coerce_coverage_status(child.get("coverage_status")) != "complete":
            complete = False
        coverage = child.get("audio", {}).get("coverage", {}) if isinstance(child.get("audio"), Mapping) else {}
        if not isinstance(coverage, Mapping):
            complete = False
            continue
        start = coverage.get("start_ms")
        end = coverage.get("end_ms")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            complete = False
        else:
            intervals.append((float(start), float(end)))
        complete = complete and bool(coverage.get("complete"))
        excluded.extend(list(coverage.get("excluded_ranges_ms", [])))
    if not intervals:
        return {
            "start_ms": 0,
            "end_ms": None,
            "intervals_ms": [],
            "excluded_ranges_ms": excluded,
            "overlap_ms": 0,
            "gap_ms": None,
            "complete": False,
        }
    intervals.sort()
    merged: list[list[float]] = []
    overlap_ms = 0.0
    gap_ms = 0.0
    previous_end: float | None = None
    for start, end in intervals:
        if previous_end is not None:
            if start < previous_end:
                overlap_ms += previous_end - start
            elif start > previous_end:
                gap_ms += start - previous_end
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
        previous_end = max(previous_end or end, end)
    return {
        "start_ms": merged[0][0],
        "end_ms": merged[-1][1],
        "intervals_ms": merged,
        "excluded_ranges_ms": excluded,
        "overlap_ms": round(overlap_ms, 3),
        "gap_ms": round(gap_ms, 3),
        "complete": complete,
    }


def _coverage_is_complete(
    coverage: Mapping[str, Any],
    *,
    duration_ms: Any,
) -> bool:
    if not bool(coverage.get("complete")):
        return False
    if coverage.get("excluded_ranges_ms"):
        return False
    if coverage.get("gap_ms", 0):
        return False
    if not isinstance(duration_ms, (int, float)) or isinstance(duration_ms, bool):
        return False
    start = coverage.get("start_ms")
    end = coverage.get("end_ms")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return False
    return math.isclose(float(start), 0.0, abs_tol=0.001) and math.isclose(
        float(end), float(duration_ms), abs_tol=1.0
    )


def _merge_detection_hints(*results: Any) -> dict[str, Any]:
    hints: list[dict[str, Any]] = []
    for result in results:
        hint = _find_detection_hint(result)
        if hint:
            hints.append(hint)
    if not hints:
        return {"status": "unavailable", "coverage_complete": False, "segments": []}
    segments: list[Any] = []
    statuses = []
    complete = True
    for hint in hints:
        statuses.append(str(hint.get("status") or ""))
        complete = complete and bool(hint.get("coverage_complete"))
        if isinstance(hint.get("segments"), list):
            segments.extend(hint["segments"])
    if any(status in {"speech_detected", "speech_detected_but_not_transcribable"} for status in statuses):
        status = "speech_detected"
    elif all(status == "no_speech_detected" for status in statuses):
        status = "no_speech_detected"
    else:
        status = statuses[0] or "unavailable"
    merged = {
        **hints[0],
        "status": status,
        "coverage_complete": complete,
        "segments": segments,
        "evidence": hints,
    }
    return merged


def _find_detection_hint(value: Any) -> dict[str, Any] | None:
    if isinstance(value, list):
        for item in value:
            hint = _find_detection_hint(item)
            if hint:
                return hint
        return None
    if not isinstance(value, Mapping):
        return None
    for key in ("speech_detection", "speech_activity", "vad", "objective_result"):
        candidate = value.get(key)
        if isinstance(candidate, Mapping):
            candidate_coverage = candidate.get("coverage")
            candidate_coverage = candidate_coverage if isinstance(candidate_coverage, Mapping) else {}
            status = _normalize_detection_status(candidate.get("status") or candidate.get("objective_outcome"))
            if status:
                return {
                    **dict(candidate),
                    "status": status,
                    "coverage_complete": bool(
                        candidate.get("coverage_complete", candidate_coverage.get("complete", False))
                    ),
                }
    for key in ("speech_status", "vad_status", "objective_outcome", "audio_result_status"):
        status = _normalize_detection_status(value.get(key))
        if status:
            return {
                "status": status,
                "segments": list(value.get("segments") or value.get("speech_segments") or []),
                "coverage_complete": bool(value.get("coverage_complete")),
            }
    for key in ("speech_detected", "has_speech", "is_speech"):
        if isinstance(value.get(key), bool):
            return {
                "status": "speech_detected" if value[key] else "no_speech_detected",
                "segments": list(value.get("segments") or []),
                "coverage_complete": bool(value.get("coverage_complete")),
            }
    for item in value.values():
        hint = _find_detection_hint(item)
        if hint:
            return hint
    return None


def _normalize_detection_status(value: Any) -> str | None:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "no_speech": "no_speech_detected",
        "silence": "no_speech_detected",
        "speech": "speech_detected",
        "speech_present": "speech_detected",
        "speech_detected": "speech_detected",
        "speech_detected_but_not_transcribable": "speech_detected_but_not_transcribable",
        "no_speech_detected": "no_speech_detected",
    }
    return aliases.get(normalized)


def _failure_class(errors: Sequence[str]) -> str:
    lowered = " ".join(str(error).lower() for error in errors)
    if any(token in lowered for token in ("preparedaudio", "audio_conversion", "ffmpeg", "decode")):
        return "audio_preprocessing_failure"
    if any(token in lowered for token in ("invalid wav", "corrupt", "zero_byte", "empty pcm")):
        return "corrupt_audio"
    if any(token in lowered for token in ("not found", "unsupported")):
        return "unsupported_audio"
    if any(token in lowered for token in ("timeout", "timed out")):
        return "subprocess_timeout"
    return "engine_failure"


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    if len(text) != 64:
        return False
    try:
        int(text, 16)
    except (TypeError, ValueError):
        return False
    return True


def _validate_declared_artifact_refs(
    declared: Any,
    label: str,
    *,
    schema: str = RAW_ARTIFACT_REF_SCHEMA,
) -> list[str]:
    if not isinstance(declared, list):
        return [f"objective result {label} artifact references are missing"]
    failures: list[str] = []
    for item in declared:
        if not isinstance(item, Mapping):
            failures.append(f"objective result {label} artifact reference is invalid")
            continue
        if item.get("schema") != schema:
            failures.append(f"objective result {label} artifact schema is invalid")
        if not isinstance(item.get("size_bytes"), int) or item.get("size_bytes", 0) <= 0:
            failures.append(f"objective result {label} artifact size_bytes must be positive")
        if not _is_sha256(item.get("sha256")):
            failures.append(f"objective result {label} artifact sha256 is invalid")
    return failures


def _validate_artifact_refs(
    declared: Any,
    actual: Sequence[Mapping[str, Any]],
    label: str,
) -> list[str]:
    if not isinstance(declared, list):
        return [f"objective result {label} artifact references are missing"]
    actual_by_path = {str(item.get("path")): item for item in actual}
    failures: list[str] = []
    for item in declared:
        if not isinstance(item, Mapping):
            failures.append(f"objective result {label} artifact reference is invalid")
            continue
        if item.get("schema") != RAW_ARTIFACT_REF_SCHEMA:
            failures.append(f"objective result {label} artifact schema is invalid")
        if not isinstance(item.get("size_bytes"), int) or item.get("size_bytes", 0) <= 0:
            failures.append(f"objective result {label} artifact size_bytes must be positive")
        if not item.get("sha256"):
            failures.append(f"objective result {label} artifact sha256 is missing")
        path = str(item.get("path") or "")
        expected = actual_by_path.get(path)
        if (
            expected is None
            or item.get("sha256") != expected.get("sha256")
            or item.get("size_bytes") != expected.get("size_bytes")
        ):
            failures.append(f"objective result {label} artifact hash does not match")
    return failures

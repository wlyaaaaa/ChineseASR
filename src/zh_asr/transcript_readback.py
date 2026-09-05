"""Read existing hash-bound ASR artifacts without running a model.

The readback route intentionally consumes only the persisted job snapshot and
the result files named by those jobs.  It never opens the source audio.  This
keeps the media consumer independent from an ASR runtime while still allowing
the owner to verify that a returned transcript belongs to the requested source
bytes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from .audio_outcome import validate_objective_result
from .result_writer import extract_segments


READBACK_SCHEMA = "chinese-asr.transcript-readback.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_RAW_OUTPUT_KEYS = ("raw_json", "primary_raw_json", "secondary_raw_json")
_MAX_REJECTIONS = 32


def read_transcript_readback(
    audio_sha256: str,
    jobs_snapshot: Path | str = Path("outputs") / "api" / "jobs.json",
) -> dict[str, Any]:
    """Read the best existing transcript bundle for one source hash.

    The source hash is the only audio identity accepted by this API.  A
    matching job must carry the same hash in its request and in the objective
    sidecar.  A raw artifact is accepted only when the sidecar declares its
    path, size, and SHA-256 and those bytes still match.
    """

    try:
        requested_hash = _normalize_sha256(audio_sha256)
    except ValueError as exc:
        return _response(
            status="invalid_request",
            source_audio_sha256=str(audio_sha256 or ""),
            gap={"code": "invalid_audio_sha256", "message": str(exc)},
            lookup_scope={"original_audio_read": False, "model_run": False},
        )

    snapshot = Path(jobs_snapshot).expanduser().resolve()
    lookup_scope: dict[str, Any] = {
        "kind": "jobs_snapshot_and_referenced_artifacts",
        "jobs_snapshot": str(snapshot),
        "jobs_examined": 0,
        "matching_jobs": 0,
        "artifact_candidates_examined": 0,
        "valid_candidates": 0,
        "rejected_candidates": 0,
        "original_audio_read": False,
        "model_run": False,
    }

    try:
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _response(
            status="gap",
            source_audio_sha256=requested_hash,
            gap={
                "code": "jobs_snapshot_missing",
                "message": "The owner-managed jobs snapshot is unavailable.",
            },
            lookup_scope=lookup_scope,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _response(
            status="gap",
            source_audio_sha256=requested_hash,
            gap={
                "code": "jobs_snapshot_unreadable",
                "message": f"Could not read the owner-managed jobs snapshot: {type(exc).__name__}.",
            },
            lookup_scope=lookup_scope,
        )

    if not isinstance(payload, Mapping) or not isinstance(payload.get("jobs"), list):
        return _response(
            status="gap",
            source_audio_sha256=requested_hash,
            gap={
                "code": "jobs_snapshot_invalid",
                "message": "The owner-managed jobs snapshot has no jobs list.",
            },
            lookup_scope=lookup_scope,
        )

    jobs = payload["jobs"]
    lookup_scope["jobs_examined"] = len(jobs)
    valid_candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    pending_jobs: list[dict[str, str]] = []

    for job in jobs:
        if not isinstance(job, Mapping):
            continue
        request = job.get("request")
        if not isinstance(request, Mapping):
            continue
        request_hash = str(request.get("audio_sha256") or "").strip().lower()
        if request_hash != requested_hash:
            continue
        lookup_scope["matching_jobs"] += 1

        job_id = str(job.get("job_id") or "")
        if job.get("status") not in (None, "succeeded"):
            if job.get("status") in {"queued", "pending", "running"}:
                pending_jobs.append({"job_id": job_id, "status": str(job["status"])})
            _record_rejection(rejected, job_id, "job_not_succeeded")
            continue

        outputs = job.get("outputs")
        if not isinstance(outputs, Mapping):
            _record_rejection(rejected, job_id, "job_outputs_missing")
            continue
        objective_value = outputs.get("objective_result")
        if not isinstance(objective_value, str) or not objective_value.strip():
            _record_rejection(rejected, job_id, "objective_result_missing")
            continue
        try:
            objective_path = _resolve_path(objective_value)
        except (OSError, RuntimeError, ValueError):
            _record_rejection(rejected, job_id, "objective_result_path_invalid")
            continue

        raw_values = _raw_output_values(outputs)
        if not raw_values:
            _record_rejection(rejected, job_id, "raw_artifact_reference_missing")
            continue

        for output_key, raw_value in raw_values:
            lookup_scope["artifact_candidates_examined"] += 1
            candidate, reason = _inspect_candidate(
                job=job,
                request_hash=request_hash,
                output_key=output_key,
                raw_value=raw_value,
                objective_path=objective_path,
            )
            if candidate is not None:
                valid_candidates.append(candidate)
            else:
                _record_rejection(rejected, job_id, reason, output_key)

    lookup_scope["valid_candidates"] = len(valid_candidates)
    lookup_scope["rejected_candidates"] = len(rejected)
    if rejected:
        lookup_scope["rejections"] = rejected[:_MAX_REJECTIONS]
    if pending_jobs:
        lookup_scope["pending_job_count"] = len(pending_jobs)
        lookup_scope["pending_jobs"] = pending_jobs[:_MAX_REJECTIONS]

    if not valid_candidates:
        if lookup_scope["matching_jobs"] == 0:
            return _response(
                status="not_found",
                source_audio_sha256=requested_hash,
                gap={
                    "code": "source_hash_not_in_jobs_snapshot",
                    "message": "No retained job in the owner snapshot carries this source hash.",
                },
                lookup_scope=lookup_scope,
            )
        return _response(
            status="gap",
            source_audio_sha256=requested_hash,
            gap={
                "code": "job_execution_pending" if pending_jobs else "no_valid_transcript_artifact",
                "message": (
                    "A matching job is still pending; continue that job rather than submit another."
                    if pending_jobs else
                    "Matching jobs exist, but none has a completed transcript artifact passing the required checks; see rejections."
                ),
            },
            lookup_scope=lookup_scope,
        )

    selected = _select_candidate(valid_candidates)
    lookup_scope["selected_job_id"] = selected["job_id"]
    lookup_scope["selected_engine"] = selected["engine"]
    lookup_scope["segments_examined"] = selected["total_segments"]
    lookup_scope["segments_with_valid_timestamps"] = selected["timed_segments"]
    lookup_scope["segments_without_usable_timestamps"] = selected["total_segments"] - selected["timed_segments"]

    if not selected["segments"]:
        return _response(
            status="gap",
            source_audio_sha256=requested_hash,
            artifact=selected["artifact"],
            quality=selected["quality"],
            coverage=selected["coverage"],
            evidence_status=selected["evidence_status"],
            objective_outcome=selected["objective_outcome"],
            selection_reason=selected["selection_reason"],
            gap={
                "code": "timestamps_unavailable",
                "message": "The matching transcript has no valid segment timestamps.",
            },
            lookup_scope=lookup_scope,
        )

    return _response(
        status="ok",
        source_audio_sha256=requested_hash,
        segments=selected["segments"],
        artifact=selected["artifact"],
        quality=selected["quality"],
        coverage=selected["coverage"],
        evidence_status=selected["evidence_status"],
        objective_outcome=selected["objective_outcome"],
        selection_reason=selected["selection_reason"],
        lookup_scope=lookup_scope,
    )


def _response(
    *,
    status: str,
    source_audio_sha256: str,
    segments: list[dict[str, Any]] | None = None,
    artifact: Mapping[str, Any] | None = None,
    quality: Mapping[str, Any] | None = None,
    coverage: Mapping[str, Any] | None = None,
    evidence_status: str | None = None,
    objective_outcome: str | None = None,
    selection_reason: str | None = None,
    gap: Mapping[str, Any] | None = None,
    lookup_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": READBACK_SCHEMA,
        "status": status,
        "source_audio_sha256": source_audio_sha256,
        "segments": list(segments or []),
    }
    if artifact is not None:
        result["artifact"] = copy.deepcopy(dict(artifact))
    if quality is not None:
        result["quality"] = copy.deepcopy(dict(quality))
    if coverage is not None:
        result["coverage"] = copy.deepcopy(dict(coverage))
    if evidence_status is not None:
        result["evidence_status"] = evidence_status
    if objective_outcome is not None:
        result["objective_outcome"] = objective_outcome
    if selection_reason is not None:
        result["selection_reason"] = selection_reason
    if gap is not None:
        result["gap"] = copy.deepcopy(dict(gap))
    result["lookup_scope"] = copy.deepcopy(dict(lookup_scope or {}))
    return result


def _normalize_sha256(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError("audio SHA-256 must be exactly 64 hexadecimal characters")
    return normalized


def _raw_output_values(outputs: Mapping[str, Any]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key in _RAW_OUTPUT_KEYS:
        value = outputs.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path_key = str(Path(value).expanduser())
        normalized = path_key.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        values.append((key, value))
    return values


def _inspect_candidate(
    *,
    job: Mapping[str, Any],
    request_hash: str,
    output_key: str,
    raw_value: str,
    objective_path: Path,
) -> tuple[dict[str, Any] | None, str]:
    job_id = str(job.get("job_id") or "")
    try:
        raw_path = _resolve_path(raw_value)
    except (OSError, RuntimeError, ValueError):
        return None, "raw_artifact_path_invalid"
    if not raw_path.is_file():
        return None, "raw_artifact_missing"
    if not objective_path.is_file():
        return None, "objective_result_missing"

    try:
        objective_bytes = objective_path.read_bytes()
        objective = json.loads(objective_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "objective_result_unreadable"
    if not isinstance(objective, Mapping):
        return None, "objective_result_not_object"

    try:
        objective_failures = validate_objective_result(objective)
    except (TypeError, ValueError, KeyError, AttributeError):
        return None, "objective_sidecar_invalid"
    if objective_failures:
        return None, "objective_sidecar_invalid"
    if objective["execution"]["status"] != "completed":
        return None, "objective_execution_not_completed"

    audio = objective.get("audio")
    if not isinstance(audio, Mapping):
        return None, "objective_source_hash_missing"
    objective_hash = str(audio.get("raw_sha256") or "").strip().lower()
    if objective_hash != request_hash:
        return None, "objective_source_hash_mismatch"
    basis = objective.get("idempotency_basis")
    if not isinstance(basis, Mapping) or str(basis.get("source_audio_sha256") or "").strip().lower() != request_hash:
        return None, "objective_basis_source_hash_mismatch"

    raw_ref, raw_ref_reason = _find_raw_ref(objective, raw_path, objective_path.parent)
    if raw_ref is None:
        return None, raw_ref_reason
    expected_hash = str(raw_ref.get("sha256") or "").strip().lower()
    try:
        raw_bytes = raw_path.read_bytes()
    except OSError:
        return None, "raw_artifact_unreadable"
    actual_hash = hashlib.sha256(raw_bytes).hexdigest()
    if actual_hash != expected_hash:
        return None, "raw_artifact_sha256_mismatch"
    declared_size = raw_ref.get("size_bytes")
    if type(declared_size) is not int or len(raw_bytes) != declared_size:
        return None, "raw_artifact_size_mismatch"

    try:
        raw_payload = json.loads(raw_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "raw_artifact_unreadable"
    segments, total_segments, timed_segments = _normalize_segments(raw_payload)
    request = job.get("request")
    request = request if isinstance(request, Mapping) else {}
    engine = str(raw_ref.get("engine") or request.get("engine") or "unknown")
    quality = objective.get("quality")
    quality_payload = {
        "status": str(quality.get("status") or "unknown")
        if isinstance(quality, Mapping)
        else "unknown"
    }
    if objective.get("confidence") is not None:
        quality_payload["confidence"] = objective.get("confidence")
    coverage = objective.get("coverage")
    coverage_payload = copy.deepcopy(dict(coverage)) if isinstance(coverage, Mapping) else {}
    evidence_status = str(job.get("evidence_status") or "unknown")
    objective_result_hash = hashlib.sha256(objective_bytes).hexdigest()
    objective_request = objective.get("request")
    mode = str(
        request.get("mode")
        or (objective_request.get("mode") if isinstance(objective_request, Mapping) else None)
        or "unknown"
    )
    artifact = {
        "job_id": job_id,
        "engine": engine,
        "mode": mode,
        "raw_artifact_role": str(raw_ref.get("role") or output_key),
        "raw_json": {
            "path": str(raw_path),
            "sha256": actual_hash,
            "size_bytes": len(raw_bytes),
        },
        "objective_result": {
            "path": str(objective_path),
            "sha256": objective_result_hash,
            "size_bytes": len(objective_bytes),
        },
    }
    selection_reason = (
        "Selected a valid timestamped artifact by timestamp availability, evidence, "
        "quality, coverage, and segment count; job freshness is not used as a quality claim."
    )
    return (
        {
            "job_id": job_id,
            "engine": engine,
            "segments": segments,
            "total_segments": total_segments,
            "timed_segments": timed_segments,
            "quality": quality_payload,
            "coverage": coverage_payload,
            "evidence_status": evidence_status,
            "objective_outcome": str(objective.get("objective_outcome") or "indeterminate"),
            "artifact": artifact,
            "selection_reason": selection_reason,
        },
        "",
    )


def _find_raw_ref(
    objective: Mapping[str, Any],
    raw_path: Path,
    objective_parent: Path,
) -> tuple[Mapping[str, Any] | None, str]:
    refs = objective.get("raw_artifacts")
    if not isinstance(refs, list):
        return None, "objective_raw_artifact_refs_missing"
    for ref in refs:
        if not isinstance(ref, Mapping):
            continue
        value = ref.get("path")
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            declared_path = _resolve_path(value, objective_parent)
        except (OSError, RuntimeError, ValueError):
            continue
        if _same_path(declared_path, raw_path):
            return ref, ""
    return None, "raw_artifact_not_declared_by_objective"


def _resolve_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def _same_path(left: Path, right: Path) -> bool:
    return str(left).casefold() == str(right).casefold()


def _normalize_segments(raw_payload: Any) -> tuple[list[dict[str, Any]], int, int]:
    extracted = extract_segments(raw_payload)
    normalized: list[dict[str, Any]] = []
    for segment in extracted:
        start = _valid_ms(segment.start_ms)
        end = _valid_ms(segment.end_ms)
        if start is None or end is None or end <= start:
            continue
        node = _pointer_value(raw_payload, segment.raw_path)
        timestamp = _timestamp_spans(node.get("timestamp")) if isinstance(node, Mapping) else None
        if timestamp and any(left < start or right > end for left, right in timestamp):
            timestamp = None
        granularity = "sentence" if ".sentence_info[" in segment.raw_path else "segment"
        item: dict[str, Any] = {
            "start_ms": start,
            "end_ms": end,
            "text": segment.text,
            "speaker": segment.speaker,
            "timestamp_granularity": f"{granularity}_with_subspans" if timestamp else granularity,
            "raw_path": segment.raw_path,
        }
        if timestamp:
            item["timestamp"] = timestamp
        normalized.append(item)
    return normalized, len(extracted), len(normalized)


def _pointer_value(value: Any, pointer: str) -> Any:
    if pointer == "$":
        return value
    current = value
    cursor = 1
    while cursor < len(pointer):
        if pointer[cursor] == ".":
            end = cursor + 1
            while end < len(pointer) and pointer[end] not in ".[":
                end += 1
            if not isinstance(current, Mapping):
                return None
            current = current.get(pointer[cursor + 1 : end])
            cursor = end
            continue
        if pointer[cursor] == "[":
            end = pointer.find("]", cursor + 1)
            if end < 0 or not isinstance(current, list):
                return None
            try:
                current = current[int(pointer[cursor + 1 : end])]
            except (ValueError, IndexError):
                return None
            cursor = end + 1
            continue
        return None
    return current


def _valid_ms(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or value < 0:
        return None
    return int(value) if float(value).is_integer() else float(value)


def _timestamp_spans(value: Any) -> list[list[int | float]] | None:
    if not isinstance(value, list) or not value:
        return None
    spans: list[list[int | float]] = []
    for item in value:
        if not isinstance(item, list) or len(item) < 2:
            return None
        start = _valid_ms(item[0])
        end = _valid_ms(item[1])
        if start is None or end is None or end <= start:
            return None
        spans.append([start, end])
    return spans


def _select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(candidates, key=_candidate_sort_key)[0]


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(bool(candidate.get("timed_segments"))),
        -_evidence_rank(str(candidate.get("evidence_status") or "")),
        -_quality_rank(candidate.get("quality")),
        -_coverage_rank(candidate.get("coverage")),
        -int(candidate.get("timed_segments") or 0),
        -_engine_rank(str(candidate.get("engine") or "")),
        -int(candidate.get("total_segments") or 0),
        str(candidate.get("job_id") or "").casefold(),
        str((candidate.get("artifact") or {}).get("raw_json", {}).get("path") or "").casefold(),
    )


def _evidence_rank(value: str) -> int:
    return {"verified": 3, "provisional": 2, "not_applicable": 1}.get(value, 0)


def _quality_rank(value: Any) -> int:
    status = value.get("status") if isinstance(value, Mapping) else ""
    return {"sufficient": 2, "low_confidence": 1}.get(str(status), 0)


def _coverage_rank(value: Any) -> int:
    if not isinstance(value, Mapping):
        return 0
    status = str(value.get("status") or "")
    if status == "complete" and value.get("complete") is True:
        return 1 if value.get("gap_ms") or value.get("excluded_ranges_ms") else 2
    return 1 if status == "partial" else 0


def _engine_rank(value: str) -> int:
    return 2 if value.casefold() == "paraformer" else 1


def _record_rejection(
    rejected: list[dict[str, str]],
    job_id: str,
    reason: str,
    output_key: str | None = None,
) -> None:
    item = {"job_id": job_id, "reason": reason}
    if output_key:
        item["artifact_key"] = output_key
    rejected.append(item)

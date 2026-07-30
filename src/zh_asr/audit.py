from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Sequence

from .result_writer import TranscriptSegment, extract_text
from .risk_rules import RuleHit, evaluate_risk_rules
from .text_normalizer import to_simplified


SELECTION_POLICY = "primary_preserving_no_majority_vote_no_semantic_rewrite"
_STRICT_ENGINE_ROLES = (
    ("lexical_primary", "primary_json"),
    ("lexical_verifier", "secondary_json"),
)


@dataclass(frozen=True)
class EngineEvidence:
    engine: str
    role: str
    text: str
    raw_result_reference: str
    execution_status: str
    error: str | None
    segments: tuple[TranscriptSegment, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Disagreement:
    id: str
    scope: str
    primary_segment_index: int | None
    secondary_segment_index: int | None
    primary_text: str
    secondary_text: str
    similarity: float
    reason: str
    review_required: bool
    audio_start_ms: int | float | None = None
    audio_end_ms: int | float | None = None


@dataclass(frozen=True)
class ReviewItem:
    id: str
    kind: str
    reason: str
    disagreement_ids: tuple[str, ...] = ()
    rule_hit_ids: tuple[str, ...] = ()
    audio_start_ms: int | float | None = None
    audio_end_ms: int | float | None = None


@dataclass(frozen=True)
class AuditReport:
    status: str
    evidence_status: str
    evidence_status_rationale: str
    final_text: str
    primary_engine: str
    primary_text: str
    secondary_engine: str
    secondary_text: str
    similarity: float
    needs_review: bool
    flags: tuple[str, ...]
    rule_hits: tuple[RuleHit, ...]
    alternatives: tuple[str, ...]
    rationale: str
    schema_version: str = "2.0"
    selection_policy: str = SELECTION_POLICY
    engine_evidence: tuple[EngineEvidence, ...] = ()
    disagreements: tuple[Disagreement, ...] = ()
    review_items: tuple[ReviewItem, ...] = ()


def build_audit_report(
    primary_engine: str,
    primary_text: str,
    secondary_engine: str,
    secondary_text: str,
    conflict_threshold: float = 0.95,
    expect_empty: bool = False,
    primary_error: str | None = None,
    secondary_error: str | None = None,
    primary_role: str = "lexical_primary",
    secondary_role: str = "lexical_verifier",
    primary_segments: Sequence[TranscriptSegment | Mapping[str, Any]] | None = None,
    secondary_segments: Sequence[TranscriptSegment | Mapping[str, Any]] | None = None,
    primary_raw_result_reference: str = "",
    secondary_raw_result_reference: str = "",
    primary_provenance: Mapping[str, Any] | None = None,
    secondary_provenance: Mapping[str, Any] | None = None,
) -> AuditReport:
    primary = to_simplified(primary_text.strip())
    secondary = to_simplified(secondary_text.strip())
    primary_norm = _normalize_for_compare(primary)
    secondary_norm = _normalize_for_compare(secondary)
    similarity = _similarity(primary_norm, secondary_norm)
    primary_segment_evidence = _coerce_segments(primary_segments, primary)
    secondary_segment_evidence = _coerce_segments(secondary_segments, secondary)
    engine_evidence = (
        EngineEvidence(
            engine=primary_engine,
            role=primary_role,
            text=primary,
            raw_result_reference=primary_raw_result_reference,
            execution_status="failed" if primary_error else "succeeded",
            error=primary_error,
            segments=primary_segment_evidence,
            provenance=dict(primary_provenance or {}),
        ),
        EngineEvidence(
            engine=secondary_engine,
            role=secondary_role,
            text=secondary,
            raw_result_reference=secondary_raw_result_reference,
            execution_status="failed" if secondary_error else "succeeded",
            error=secondary_error,
            segments=secondary_segment_evidence,
            provenance=dict(secondary_provenance or {}),
        ),
    )
    disagreements = _build_disagreements(
        primary_segment_evidence,
        secondary_segment_evidence,
        conflict_threshold,
    )
    error_hits = _engine_error_hits(primary_engine, primary_error, secondary_engine, secondary_error)

    if error_hits:
        chosen = primary or secondary
        alternatives = tuple(text for text in (secondary, primary) if text and text != chosen)
        all_engines_failed = bool(primary_error and secondary_error)
        rule_hits = error_hits + evaluate_risk_rules(
            primary_text=primary,
            secondary_text=secondary,
            final_text=chosen,
            similarity=similarity,
            expect_empty=expect_empty,
        )
        flags = tuple(sorted(hit.id for hit in rule_hits))
        return AuditReport(
            status="engine_failure",
            evidence_status="unavailable" if all_engines_failed else "provisional",
            evidence_status_rationale=(
                (
                    "Both required ASR engines failed; no strict evidence is available."
                    if all_engines_failed
                    else "One required ASR engine failed; the available-engine "
                    "fallback is provisional and must not be treated as dual-engine verification."
                )
            ),
            final_text=f"[疑似]{chosen}" if chosen else "[听不清]",
            primary_engine=primary_engine,
            primary_text=primary,
            secondary_engine=secondary_engine,
            secondary_text=secondary,
            similarity=similarity,
            needs_review=True,
            flags=flags,
            rule_hits=rule_hits,
            alternatives=alternatives,
            rationale="One or more ASR engines failed; final text falls back to available evidence and must be reviewed.",
            engine_evidence=engine_evidence,
            disagreements=disagreements,
            review_items=_build_review_items(disagreements, rule_hits),
        )

    if not primary_norm and not secondary_norm:
        rule_hits = (
            RuleHit(
                id="empty_transcript",
                severity="medium",
                message="Both ASR engines returned empty text.",
                evidence="primary_empty=true, secondary_empty=true",
            ),
        )
        return AuditReport(
            status="unclear",
            evidence_status="verified",
            evidence_status_rationale=(
                "Both required ASR engines completed without an engine failure; "
                "verified describes pipeline completeness, not transcript accuracy."
            ),
            final_text="[听不清]",
            primary_engine=primary_engine,
            primary_text=primary,
            secondary_engine=secondary_engine,
            secondary_text=secondary,
            similarity=1.0,
            needs_review=True,
            flags=("empty_transcript",),
            rule_hits=rule_hits,
            alternatives=(),
            rationale="Both ASR engines returned empty text.",
            engine_evidence=engine_evidence,
            disagreements=disagreements,
            review_items=_build_review_items(disagreements, rule_hits),
        )

    chosen = primary or secondary
    alternatives = tuple(text for text in (secondary, primary) if text and text != chosen)
    rule_hits = evaluate_risk_rules(
        primary_text=primary,
        secondary_text=secondary,
        final_text=chosen,
        similarity=similarity,
        expect_empty=expect_empty,
    )
    flags = tuple(sorted(hit.id for hit in rule_hits))

    if "model_conflict" in flags:
        return AuditReport(
            status="conflict",
            evidence_status="verified",
            evidence_status_rationale=(
                "Both required ASR engines completed without an engine failure; "
                "verified describes pipeline completeness, while audit status still requires review."
            ),
            final_text=f"[疑似]{chosen}",
            primary_engine=primary_engine,
            primary_text=primary,
            secondary_engine=secondary_engine,
            secondary_text=secondary,
            similarity=similarity,
            needs_review=True,
            flags=flags,
            rule_hits=rule_hits,
            alternatives=alternatives,
            rationale="The engines disagree materially; final text is the best current guess and must be checked against audio.",
            engine_evidence=engine_evidence,
            disagreements=disagreements,
            review_items=_build_review_items(disagreements, rule_hits),
        )

    if rule_hits:
        return AuditReport(
            status="suspicious",
            evidence_status="verified",
            evidence_status_rationale=(
                "Both required ASR engines completed without an engine failure; "
                "verified describes pipeline completeness, while audit status still requires review."
            ),
            final_text=f"[疑似]{chosen}",
            primary_engine=primary_engine,
            primary_text=primary,
            secondary_engine=secondary_engine,
            secondary_text=secondary,
            similarity=similarity,
            needs_review=True,
            flags=flags,
            rule_hits=rule_hits,
            alternatives=alternatives,
            rationale="One or more deterministic ASR risk rules matched this transcript.",
            engine_evidence=engine_evidence,
            disagreements=disagreements,
            review_items=_build_review_items(disagreements, rule_hits),
        )

    if primary_norm == secondary_norm:
        return AuditReport(
            status="consistent",
            evidence_status="verified",
            evidence_status_rationale=(
                "Both required ASR engines completed without an engine failure; "
                "verified describes pipeline completeness, not independent proof of transcript accuracy."
            ),
            final_text=chosen,
            primary_engine=primary_engine,
            primary_text=primary,
            secondary_engine=secondary_engine,
            secondary_text=secondary,
            similarity=1.0,
            needs_review=False,
            flags=(),
            rule_hits=(),
            alternatives=alternatives,
            rationale="Both ASR engines agree after punctuation and whitespace normalization.",
            engine_evidence=engine_evidence,
            disagreements=disagreements,
            review_items=(),
        )

    if similarity >= conflict_threshold:
        return AuditReport(
            status="minor_difference",
            evidence_status="verified",
            evidence_status_rationale=(
                "Both required ASR engines completed without an engine failure; "
                "verified describes pipeline completeness, not independent proof of transcript accuracy."
            ),
            final_text=chosen,
            primary_engine=primary_engine,
            primary_text=primary,
            secondary_engine=secondary_engine,
            secondary_text=secondary,
            similarity=similarity,
            needs_review=False,
            flags=(),
            rule_hits=(),
            alternatives=alternatives,
            rationale="The engines differ only slightly; final text follows the primary engine.",
            engine_evidence=engine_evidence,
            disagreements=disagreements,
            review_items=_build_review_items(disagreements, ()),
        )

    conflict_hit = RuleHit(
        id="model_conflict",
        severity="high" if similarity < 0.80 else "medium",
        message="The primary and secondary ASR engines disagree materially.",
        evidence=f"similarity={similarity:.3f}",
    )
    return AuditReport(
        status="conflict",
        evidence_status="verified",
        evidence_status_rationale=(
            "Both required ASR engines completed without an engine failure; "
            "verified describes pipeline completeness, while audit status still requires review."
        ),
        final_text=f"[疑似]{chosen}",
        primary_engine=primary_engine,
        primary_text=primary,
        secondary_engine=secondary_engine,
        secondary_text=secondary,
        similarity=similarity,
        needs_review=True,
        flags=("model_conflict",),
        rule_hits=(conflict_hit,),
        alternatives=alternatives,
        rationale="The engines disagree materially; final text is the best current guess and must be checked against audio.",
        engine_evidence=engine_evidence,
        disagreements=disagreements,
        review_items=_build_review_items(disagreements, (conflict_hit,)),
    )


def validate_strict_artifact_bundle(
    outputs: Mapping[str, Any],
    *,
    expected_primary_engine: str | None = None,
    expected_secondary_engine: str | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Validate the complete persisted strict bundle before assigning evidence status."""
    paths: dict[str, Path] = {}
    for key, aliases, label in (
        ("final", ("final",), "final transcript"),
        ("audit", ("audit",), "audit Markdown"),
        ("audit_json", ("audit_json",), "strict audit JSON"),
        ("primary_json", ("primary_json", "primary_raw_json"), "primary raw JSON"),
        (
            "secondary_json",
            ("secondary_json", "secondary_raw_json"),
            "secondary raw JSON",
        ),
    ):
        value = next((outputs.get(alias) for alias in aliases if outputs.get(alias)), None)
        if not value:
            return "unavailable", [_artifact_failure(f"{label} is missing")]
        path = Path(str(value)).expanduser()
        try:
            if key in {"final", "audit"}:
                if not path.read_text(encoding="utf-8").strip():
                    raise ValueError("file is empty")
        except (OSError, UnicodeError, ValueError) as exc:
            return "unavailable", [
                _artifact_failure(f"{label} is unreadable: {type(exc).__name__}: {exc}")
            ]
        paths[key] = path

    audit, failure = _read_json_artifact(paths["audit_json"], "strict audit JSON")
    if failure:
        return "unavailable", [failure]
    primary_raw, failure = _read_json_artifact(paths["primary_json"], "primary raw JSON")
    if failure:
        return "unavailable", [failure]
    secondary_raw, failure = _read_json_artifact(paths["secondary_json"], "secondary raw JSON")
    if failure:
        return "unavailable", [failure]
    if not isinstance(audit, dict):
        return "unavailable", [_artifact_failure("strict audit JSON is not an object")]

    engine_evidence = audit.get("engine_evidence")
    if not isinstance(engine_evidence, list) or len(engine_evidence) != 2:
        return "unavailable", [
            _artifact_failure("strict audit must contain exactly two engine_evidence records")
        ]
    by_role = {
        str(item.get("role") or ""): item
        for item in engine_evidence
        if isinstance(item, dict)
    }
    if set(by_role) != {role for role, _ in _STRICT_ENGINE_ROLES}:
        return "unavailable", [
            _artifact_failure("strict audit engine role identity is incomplete or duplicated")
        ]

    expected_engines = {
        "lexical_primary": expected_primary_engine,
        "lexical_verifier": expected_secondary_engine,
    }
    raw_by_key = {
        "primary_json": primary_raw,
        "secondary_json": secondary_raw,
    }
    execution_failures: list[dict[str, str]] = []
    execution_statuses: list[str] = []
    for role, raw_key in _STRICT_ENGINE_ROLES:
        item = by_role[role]
        engine = str(item.get("engine") or "")
        expected_engine = expected_engines[role] or engine
        raw_path = paths[raw_key]
        raw = raw_by_key[raw_key]
        identity_error = _strict_identity_error(
            item,
            raw,
            raw_path,
            expected_engine,
        )
        if identity_error:
            return "unavailable", [_artifact_failure(identity_error)]

        execution_status = str(item.get("execution_status") or "")
        audit_error = item.get("error")
        raw_error = _raw_error_text(raw)
        if execution_status == "succeeded":
            if (
                (audit_error is not None and audit_error != "")
                or raw_error is not None
            ):
                return "unavailable", [
                    _artifact_failure(
                        f"{role} execution_status/error does not match successful raw JSON"
                    )
                ]
            if not extract_text(raw).strip():
                return "unavailable", [
                    _artifact_failure(f"{role} successful raw JSON has empty text")
                ]
        elif execution_status == "failed":
            if not audit_error or raw_error is None:
                return "unavailable", [
                    _artifact_failure(
                        f"{role} execution_status/error does not match failed raw JSON"
                    )
                ]
            if _normalize_error(str(audit_error)) != _normalize_error(raw_error):
                return "unavailable", [
                    _artifact_failure(
                        f"{role} audit error does not match failed raw JSON error"
                    )
                ]
            execution_failures.append(
                {
                    "engine": expected_engine,
                    "role": role,
                    "error": str(audit_error),
                }
            )
        else:
            return "unavailable", [
                _artifact_failure(f"{role} has invalid execution_status")
            ]
        execution_statuses.append(execution_status)

    failed_count = execution_statuses.count("failed")
    computed_status = (
        "verified"
        if failed_count == 0
        else "provisional" if failed_count == 1 else "unavailable"
    )
    if audit.get("evidence_status") != computed_status:
        return "unavailable", [
            _artifact_failure(
                "strict audit evidence_status is inconsistent with persisted raw evidence"
            )
        ]
    audit_status = str(audit.get("status") or "")
    if (failed_count > 0) != (audit_status == "engine_failure"):
        return "unavailable", [
            _artifact_failure(
                "strict audit status is inconsistent with engine execution status"
            )
        ]
    return computed_status, execution_failures


def _strict_identity_error(
    item: Mapping[str, Any],
    raw: Any,
    raw_path: Path,
    expected_engine: str,
) -> str | None:
    if not expected_engine or item.get("engine") != expected_engine:
        return "strict audit engine identity does not match the requested engine"
    if not raw_path.name.lower().endswith(f".{expected_engine}.raw.json".lower()):
        return "raw JSON filename engine identity does not match the requested engine"
    reference = item.get("raw_result_reference")
    if not reference or Path(str(reference)).expanduser().resolve() != raw_path.resolve():
        return "strict audit raw_result_reference identity does not match the raw JSON"
    if isinstance(raw, dict):
        raw_engine = raw.get("engine")
        if raw_engine is not None and raw_engine != expected_engine:
            return "raw JSON engine identity does not match the requested engine"
    return None


def _read_json_artifact(
    path: Path,
    label: str,
) -> tuple[Any | None, dict[str, str] | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return None, _artifact_failure(
            f"{label} is unreadable: {type(exc).__name__}: {exc}"
        )


def _raw_error_text(raw: Any) -> str | None:
    if not isinstance(raw, dict):
        return None
    error = raw.get("error")
    if error is None or error == "":
        return None
    if isinstance(error, dict):
        error_type = str(error.get("type") or "").strip()
        message = str(error.get("message") or "").strip()
        if error_type and message:
            return f"{error_type}: {message}"
        return error_type or message or None
    return str(error)


def _normalize_error(value: str) -> str:
    return " ".join(value.split())


def _artifact_failure(error: str) -> dict[str, str]:
    return {"kind": "artifact_failure", "error": error}


def _coerce_segments(
    segments: Sequence[TranscriptSegment | Mapping[str, Any]] | None,
    fallback_text: str,
) -> tuple[TranscriptSegment, ...]:
    if segments is None:
        if not fallback_text:
            return ()
        return (TranscriptSegment(index=0, text=fallback_text, raw_path="$text"),)

    coerced: list[TranscriptSegment] = []
    for fallback_index, segment in enumerate(segments):
        if isinstance(segment, TranscriptSegment):
            coerced.append(segment)
            continue
        if not isinstance(segment, Mapping):
            continue
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        coerced.append(
            TranscriptSegment(
                index=_int_or_fallback(segment.get("index"), fallback_index),
                text=text,
                start_ms=_number_or_none(segment.get("start_ms")),
                end_ms=_number_or_none(segment.get("end_ms")),
                speaker=segment.get("speaker"),
                raw_path=str(segment.get("raw_path") or f"$segments[{fallback_index}]"),
            )
        )
    return tuple(coerced)


def _build_disagreements(
    primary_segments: tuple[TranscriptSegment, ...],
    secondary_segments: tuple[TranscriptSegment, ...],
    conflict_threshold: float,
) -> tuple[Disagreement, ...]:
    records: list[Disagreement] = []
    for primary, secondary in _align_segments(primary_segments, secondary_segments):
        primary_text = primary.text if primary else ""
        secondary_text = secondary.text if secondary else ""
        primary_norm = _normalize_for_compare(to_simplified(primary_text))
        secondary_norm = _normalize_for_compare(to_simplified(secondary_text))
        if primary_norm == secondary_norm:
            continue
        similarity = _similarity(primary_norm, secondary_norm)
        if primary is None:
            reason = "primary_segment_missing"
        elif secondary is None:
            reason = "secondary_segment_missing"
        else:
            reason = "segment_text_mismatch"
        start_ms, end_ms = _combined_audio_span(primary, secondary)
        records.append(
            Disagreement(
                id=f"disagreement-{len(records) + 1:03d}",
                scope="segment",
                primary_segment_index=primary.index if primary else None,
                secondary_segment_index=secondary.index if secondary else None,
                primary_text=primary_text,
                secondary_text=secondary_text,
                similarity=similarity,
                reason=reason,
                review_required=primary is None
                or secondary is None
                or similarity < conflict_threshold,
                audio_start_ms=start_ms,
                audio_end_ms=end_ms,
            )
        )
    return tuple(records)


def _align_segments(
    primary_segments: tuple[TranscriptSegment, ...],
    secondary_segments: tuple[TranscriptSegment, ...],
) -> tuple[tuple[TranscriptSegment | None, TranscriptSegment | None], ...]:
    primary_keys = [_normalize_for_compare(to_simplified(item.text)) for item in primary_segments]
    secondary_keys = [_normalize_for_compare(to_simplified(item.text)) for item in secondary_segments]
    matcher = SequenceMatcher(a=primary_keys, b=secondary_keys, autojunk=False)
    pairs: list[tuple[TranscriptSegment | None, TranscriptSegment | None]] = []

    for tag, p_start, p_end, s_start, s_end in matcher.get_opcodes():
        if tag == "equal":
            pairs.extend(
                zip(
                    primary_segments[p_start:p_end],
                    secondary_segments[s_start:s_end],
                )
            )
            continue
        primary_block = primary_segments[p_start:p_end]
        secondary_block = secondary_segments[s_start:s_end]
        common = min(len(primary_block), len(secondary_block))
        pairs.extend(zip(primary_block[:common], secondary_block[:common]))
        pairs.extend((item, None) for item in primary_block[common:])
        pairs.extend((None, item) for item in secondary_block[common:])
    return tuple(pairs)


def _combined_audio_span(
    primary: TranscriptSegment | None,
    secondary: TranscriptSegment | None,
) -> tuple[int | float | None, int | float | None]:
    starts = [
        segment.start_ms
        for segment in (primary, secondary)
        if segment is not None and segment.start_ms is not None
    ]
    ends = [
        segment.end_ms
        for segment in (primary, secondary)
        if segment is not None and segment.end_ms is not None
    ]
    return (min(starts) if starts else None, max(ends) if ends else None)


def _build_review_items(
    disagreements: tuple[Disagreement, ...],
    rule_hits: tuple[RuleHit, ...],
) -> tuple[ReviewItem, ...]:
    items: list[ReviewItem] = []
    for disagreement in disagreements:
        if not disagreement.review_required:
            continue
        items.append(
            ReviewItem(
                id=f"review-{len(items) + 1:03d}",
                kind="audio_span",
                reason=disagreement.reason,
                disagreement_ids=(disagreement.id,),
                audio_start_ms=disagreement.audio_start_ms,
                audio_end_ms=disagreement.audio_end_ms,
            )
        )

    represented_rule_ids = {
        "model_conflict"
        for disagreement in disagreements
        if disagreement.review_required
    }
    for hit in rule_hits:
        if hit.id in represented_rule_ids:
            continue
        items.append(
            ReviewItem(
                id=f"review-{len(items) + 1:03d}",
                kind="rule_hit",
                reason=hit.message,
                rule_hit_ids=(hit.id,),
            )
        )
    return tuple(items)


def _int_or_fallback(value: Any, fallback: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else fallback


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()\[\]【】<>《》|-]+", "", text).lower()


def _similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(a=left, b=right).ratio()


def _engine_error_hits(
    primary_engine: str,
    primary_error: str | None,
    secondary_engine: str,
    secondary_error: str | None,
) -> tuple[RuleHit, ...]:
    errors = []
    if primary_error:
        errors.append((primary_engine, primary_error))
    if secondary_error:
        errors.append((secondary_engine, secondary_error))
    if not errors:
        return ()
    evidence = "; ".join(f"{engine}: {_clip_error(error)}" for engine, error in errors)
    return (
        RuleHit(
            id="engine_failure",
            severity="high" if len(errors) > 1 else "medium",
            message="One or more ASR engines failed before returning usable text.",
            evidence=evidence,
        ),
    )


def _clip_error(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."

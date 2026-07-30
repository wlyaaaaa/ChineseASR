from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Sequence

from .result_writer import (
    TranscriptSegment,
    canonical_json_sha256,
    extract_text,
    file_sha256,
    text_sha256,
)
from .risk_rules import RuleHit, evaluate_risk_rules
from .text_normalizer import to_simplified


SELECTION_POLICY = "primary_preserving_no_majority_vote_no_semantic_rewrite"
STRICT_BUNDLE_RECEIPT_SCHEMA_VERSION = "zh_asr.strict_bundle_receipt.v1"
STRICT_BUNDLE_ARTIFACT_KEYS = (
    "final",
    "audit",
    "audit_json",
    "review_json",
    "primary_json",
    "secondary_json",
)
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
    """Verify hashes and semantic bindings for one complete persisted strict bundle.

    ``verified`` means all required artifacts are readable, receipt-bound, and
    mutually consistent, in addition to both ASR engines succeeding.
    """
    paths, path_failures = _strict_bundle_paths(outputs)
    if path_failures:
        return "unavailable", path_failures
    if paths["primary_json"].resolve() == paths["secondary_json"].resolve():
        return "unavailable", [
            _artifact_failure(
                "primary and secondary raw JSON artifacts must be distinct"
            )
        ]

    receipt, failure = _read_json_artifact(paths["receipt"], "bundle receipt JSON")
    if failure:
        return "unavailable", [failure]
    receipt_failures = _validate_bundle_receipt(paths, receipt)
    if receipt_failures:
        return "unavailable", receipt_failures

    payloads: dict[str, Any] = {}
    json_failures: list[dict[str, str]] = []
    for key, label in (
        ("audit_json", "strict audit JSON"),
        ("review_json", "review JSON"),
        ("primary_json", "primary raw JSON"),
        ("secondary_json", "secondary raw JSON"),
    ):
        payload, load_failure = _read_json_artifact(paths[key], label)
        if load_failure:
            json_failures.append(load_failure)
        else:
            payloads[key] = payload
    if json_failures:
        return "unavailable", json_failures

    audit = payloads["audit_json"]
    review = payloads["review_json"]
    if not isinstance(audit, dict):
        return "unavailable", [_artifact_failure("strict audit JSON is not an object")]
    if not isinstance(review, dict):
        return "unavailable", [_artifact_failure("review JSON is not an object")]

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
        "primary_json": payloads["primary_json"],
        "secondary_json": payloads["secondary_json"],
    }
    execution_failures: list[dict[str, str]] = []
    execution_statuses: list[str] = []
    binding_failures: list[dict[str, str]] = []
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
            binding_failures.append(_artifact_failure(identity_error))
            continue

        audit_text_value = item.get("text")
        if not isinstance(audit_text_value, str):
            binding_failures.append(
                _artifact_failure(f"{role} engine_evidence text is not a string")
            )
            audit_text = ""
        else:
            audit_text = audit_text_value.strip()
        raw_text = to_simplified(extract_text(raw).strip())
        if raw_text != audit_text:
            binding_failures.append(
                _artifact_failure(
                    f"{role} raw text does not match strict audit engine_evidence text"
                )
            )

        execution_status = str(item.get("execution_status") or "")
        audit_error = item.get("error")
        raw_error = _raw_error_text(raw)
        if execution_status == "succeeded":
            if (audit_error is not None and audit_error != "") or raw_error is not None:
                binding_failures.append(
                    _artifact_failure(
                        f"{role} execution_status/error does not match successful raw JSON"
                    )
                )
        elif execution_status == "failed":
            if not audit_error or raw_error is None:
                binding_failures.append(
                    _artifact_failure(
                        f"{role} execution_status/error does not match failed raw JSON"
                    )
                )
            elif _normalize_error(str(audit_error)) != _normalize_error(raw_error):
                binding_failures.append(
                    _artifact_failure(
                        f"{role} audit error does not match failed raw JSON error"
                    )
                )
            else:
                execution_failures.append(
                    {
                        "engine": expected_engine,
                        "role": role,
                        "error": str(audit_error),
                    }
                )
        else:
            binding_failures.append(
                _artifact_failure(f"{role} has invalid execution_status")
            )
        execution_statuses.append(execution_status)

    binding_failures.extend(
        _validate_audit_projection(
            paths=paths,
            audit=audit,
            review=review,
            by_role=by_role,
            receipt=receipt,
        )
    )
    if binding_failures:
        return "unavailable", binding_failures

    failed_count = execution_statuses.count("failed")
    computed_status = (
        "verified"
        if failed_count == 0
        else "provisional" if failed_count == 1 else "unavailable"
    )
    status_failures: list[dict[str, str]] = []
    if audit.get("evidence_status") != computed_status:
        status_failures.append(
            _artifact_failure(
                "strict audit evidence_status is inconsistent with persisted raw evidence"
            )
        )
    audit_status = str(audit.get("status") or "")
    if (failed_count > 0) != (audit_status == "engine_failure"):
        status_failures.append(
            _artifact_failure(
                "strict audit status is inconsistent with engine execution status"
            )
        )
    if status_failures:
        return "unavailable", status_failures
    return computed_status, execution_failures


def _strict_bundle_paths(
    outputs: Mapping[str, Any],
) -> tuple[dict[str, Path], list[dict[str, str]]]:
    paths: dict[str, Path] = {}
    specs = (
        ("final", ("final",), "final transcript"),
        ("audit", ("audit",), "audit Markdown"),
        ("audit_json", ("audit_json",), "strict audit JSON"),
        ("primary_json", ("primary_json", "primary_raw_json"), "primary raw JSON"),
        (
            "secondary_json",
            ("secondary_json", "secondary_raw_json"),
            "secondary raw JSON",
        ),
    )
    for key, aliases, label in specs:
        value = next((outputs.get(alias) for alias in aliases if outputs.get(alias)), None)
        if not value:
            return paths, [_artifact_failure(f"{label} is missing")]
        path = Path(str(value)).expanduser()
        if not path.is_file():
            return paths, [_artifact_failure(f"{label} is missing: {path}")]
        paths[key] = path

    audit_json_path = paths["audit_json"]
    derived = {
        "review_json": _strict_sibling_path(
            audit_json_path,
            ".strict.audit.json",
            ".strict.review.json",
        ),
        "receipt": _strict_sibling_path(
            audit_json_path,
            ".strict.audit.json",
            ".strict.receipt.json",
        ),
    }
    for key, aliases, label in (
        ("review_json", ("review_json", "review"), "review JSON"),
        ("receipt", ("receipt", "receipt_json", "bundle_receipt"), "bundle receipt JSON"),
    ):
        value = next((outputs.get(alias) for alias in aliases if outputs.get(alias)), None)
        path = Path(str(value)).expanduser() if value else derived[key]
        if not path.is_file():
            return paths, [_artifact_failure(f"{label} is missing: {path}")]
        paths[key] = path
    return paths, []


def _strict_sibling_path(path: Path, old_suffix: str, new_suffix: str) -> Path:
    if path.name.endswith(old_suffix):
        return path.with_name(f"{path.name[:-len(old_suffix)]}{new_suffix}")
    return path.with_name(f"{path.stem}{new_suffix}")


def _validate_bundle_receipt(
    paths: Mapping[str, Path],
    receipt: Any,
) -> list[dict[str, str]]:
    if not isinstance(receipt, dict):
        return [_artifact_failure("bundle receipt JSON is not an object")]
    if receipt.get("schema_version") != STRICT_BUNDLE_RECEIPT_SCHEMA_VERSION:
        return [_artifact_failure("bundle receipt schema_version is unsupported")]

    artifacts = receipt.get("artifacts")
    claims = receipt.get("claims")
    if not isinstance(artifacts, dict) or set(artifacts) != set(
        STRICT_BUNDLE_ARTIFACT_KEYS
    ):
        return [
            _artifact_failure(
                "bundle receipt artifacts must contain every strict artifact exactly once"
            )
        ]
    if not isinstance(claims, dict):
        return [_artifact_failure("bundle receipt claims is not an object")]

    failures: list[dict[str, str]] = []
    expected_bundle_sha = canonical_json_sha256(
        {
            "schema_version": receipt["schema_version"],
            "artifacts": artifacts,
            "claims": claims,
        }
    )
    if receipt.get("bundle_sha256") != expected_bundle_sha:
        failures.append(_artifact_failure("bundle receipt bundle_sha256 mismatch"))

    receipt_root = paths["receipt"].parent.resolve()
    for key in STRICT_BUNDLE_ARTIFACT_KEYS:
        entry = artifacts.get(key)
        if not isinstance(entry, dict):
            failures.append(_artifact_failure(f"{key} receipt entry is not an object"))
            continue
        relative_value = entry.get("path")
        if not isinstance(relative_value, str) or not relative_value:
            failures.append(_artifact_failure(f"{key} receipt path is missing"))
            continue
        relative_path = Path(relative_value)
        if (
            relative_path.is_absolute()
            or relative_path.drive
            or relative_path.root
        ):
            failures.append(_artifact_failure(f"{key} receipt path is not bundle-relative"))
            continue
        receipt_path = _resolve_bundle_reference(
            relative_value,
            receipt_root,
        )
        if receipt_path is None:
            failures.append(_artifact_failure(f"{key} receipt path is not bundle-relative"))
            continue
        artifact_path = paths[key].resolve()
        if receipt_path != artifact_path:
            failures.append(
                _artifact_failure(f"{key} receipt path does not match supplied artifact")
            )
            continue
        try:
            actual_size = paths[key].stat().st_size
            actual_sha = file_sha256(paths[key])
        except OSError as exc:
            failures.append(
                _artifact_failure(
                    f"{key} is unreadable: {type(exc).__name__}: {exc}"
                )
            )
            continue
        if entry.get("size_bytes") != actual_size:
            failures.append(
                _artifact_failure(
                    f"{key} ({_strict_artifact_label(key)}) size_bytes mismatch"
                )
            )
        if entry.get("sha256") != actual_sha:
            failures.append(
                _artifact_failure(
                    f"{key} ({_strict_artifact_label(key)}) SHA-256 mismatch"
                )
            )
    return failures


def _validate_audit_projection(
    *,
    paths: Mapping[str, Path],
    audit: Mapping[str, Any],
    review: Mapping[str, Any],
    by_role: Mapping[str, Mapping[str, Any]],
    receipt: Any,
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    try:
        final_markdown = paths["final"].read_text(encoding="utf-8")
        audit_markdown = paths["audit"].read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [
            _artifact_failure(
                f"Markdown artifact is unreadable: {type(exc).__name__}: {exc}"
            )
        ]

    audit_status = _required_string(audit.get("status"))
    evidence_status = _required_string(audit.get("evidence_status"))
    final_text = _required_string(audit.get("final_text"))
    primary_text = _required_string(audit.get("primary_text"))
    secondary_text = _required_string(audit.get("secondary_text"))
    primary_engine = _required_string(audit.get("primary_engine"))
    secondary_engine = _required_string(audit.get("secondary_engine"))
    if None in (
        audit_status,
        evidence_status,
        final_text,
        primary_text,
        secondary_text,
        primary_engine,
        secondary_engine,
    ):
        failures.append(
            _artifact_failure("strict audit JSON is missing required text projections")
        )
        return failures

    primary_item = by_role["lexical_primary"]
    secondary_item = by_role["lexical_verifier"]
    if primary_engine != primary_item.get("engine"):
        failures.append(
            _artifact_failure("strict audit primary_engine does not match engine_evidence")
        )
    if secondary_engine != secondary_item.get("engine"):
        failures.append(
            _artifact_failure("strict audit secondary_engine does not match engine_evidence")
        )
    if primary_text != primary_item.get("text"):
        failures.append(
            _artifact_failure("strict audit primary_text does not match engine_evidence")
        )
    if secondary_text != secondary_item.get("text"):
        failures.append(
            _artifact_failure("strict audit secondary_text does not match engine_evidence")
        )

    final_status = _markdown_field(final_markdown, "Status")
    final_evidence_status = _markdown_field(final_markdown, "Evidence status")
    final_projection = _markdown_section_to_end(final_markdown, "Transcript")
    if final_status != audit_status:
        failures.append(
            _artifact_failure(
                "final transcript status does not match strict audit status"
            )
        )
    if final_evidence_status != evidence_status:
        failures.append(
            _artifact_failure(
                "final transcript evidence status does not match strict audit evidence_status"
            )
        )
    if final_projection != final_text:
        failures.append(
            _artifact_failure(
                "final transcript text does not match strict audit final_text"
            )
        )

    audit_md_status = _markdown_field(audit_markdown, "Status")
    audit_md_evidence_status = _markdown_field(audit_markdown, "Evidence status")
    audit_md_final = _markdown_between_sections(
        audit_markdown,
        "Final Guess",
        primary_engine,
    )
    audit_md_primary = _markdown_between_sections(
        audit_markdown,
        primary_engine,
        secondary_engine,
    )
    audit_md_secondary = _markdown_between_sections(
        audit_markdown,
        secondary_engine,
        "Alternatives",
    )
    if audit_md_status != audit_status:
        failures.append(
            _artifact_failure(
                "audit Markdown status does not match strict audit JSON status"
            )
        )
    if audit_md_evidence_status != evidence_status:
        failures.append(
            _artifact_failure(
                "audit Markdown evidence status does not match strict audit JSON evidence_status"
            )
        )
    if audit_md_final != final_text:
        failures.append(
            _artifact_failure(
                "audit Markdown final guess does not match strict audit JSON final_text"
            )
        )
    if audit_md_primary != (primary_text or "_Empty_"):
        failures.append(
            _artifact_failure(
                "audit Markdown primary text does not match strict audit JSON"
            )
        )
    if audit_md_secondary != (secondary_text or "_Empty_"):
        failures.append(
            _artifact_failure(
                "audit Markdown secondary text does not match strict audit JSON"
            )
        )

    for key, label in (
        ("status", "status"),
        ("evidence_status", "evidence_status"),
        ("final_text", "final_text"),
        ("primary_text", "primary_text"),
        ("secondary_text", "secondary_text"),
        ("needs_review", "needs_review"),
        ("selection_policy", "selection_policy"),
        ("engine_evidence", "engine_evidence"),
        ("disagreements", "disagreements"),
        ("review_items", "review_items"),
    ):
        if review.get(key) != audit.get(key):
            failures.append(
                _artifact_failure(
                    f"review JSON {label} does not match strict audit JSON {label}"
                )
            )

    receipt_reference = str(paths["receipt"])
    for source, value in (
        ("strict audit JSON", audit.get("bundle_receipt_reference")),
        ("review JSON", review.get("bundle_receipt_reference")),
        ("final transcript", _markdown_field(final_markdown, "Bundle receipt")),
        ("audit Markdown", _markdown_field(audit_markdown, "Bundle receipt")),
    ):
        if (
            not value
            or _resolve_bundle_reference(
                value,
                paths["receipt"].parent,
            )
            != paths["receipt"].resolve()
        ):
            failures.append(
                _artifact_failure(
                    f"{source} bundle receipt reference does not match {receipt_reference}"
                )
            )

    failures.extend(
        _validate_receipt_claims(
            receipt=receipt,
            audit=audit,
            by_role=by_role,
        )
    )
    return failures


def _validate_receipt_claims(
    *,
    receipt: Any,
    audit: Mapping[str, Any],
    by_role: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    if not isinstance(receipt, dict) or not isinstance(receipt.get("claims"), dict):
        return [_artifact_failure("bundle receipt claims is unavailable")]
    claims = receipt["claims"]
    failures: list[dict[str, str]] = []
    if claims.get("status") != audit.get("status"):
        failures.append(
            _artifact_failure("bundle receipt status claim does not match strict audit")
        )
    if claims.get("evidence_status") != audit.get("evidence_status"):
        failures.append(
            _artifact_failure(
                "bundle receipt evidence_status claim does not match strict audit"
            )
        )
    final_text = audit.get("final_text")
    if not isinstance(final_text, str) or claims.get("final_text_sha256") != text_sha256(
        final_text
    ):
        failures.append(
            _artifact_failure(
                "bundle receipt final_text_sha256 claim does not match strict audit"
            )
        )

    claim_items = claims.get("engine_evidence")
    if not isinstance(claim_items, list) or len(claim_items) != 2:
        failures.append(
            _artifact_failure(
                "bundle receipt must contain exactly two engine_evidence claims"
            )
        )
        return failures
    claims_by_role = {
        str(item.get("role") or ""): item
        for item in claim_items
        if isinstance(item, dict)
    }
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        return failures
    for role, raw_key in _STRICT_ENGINE_ROLES:
        item = by_role[role]
        claim = claims_by_role.get(role)
        raw_entry = artifacts.get(raw_key)
        if not isinstance(claim, dict) or not isinstance(raw_entry, dict):
            failures.append(
                _artifact_failure(f"bundle receipt {role} claim is missing")
            )
            continue
        if claim.get("engine") != item.get("engine"):
            failures.append(
                _artifact_failure(
                    f"bundle receipt {role} engine claim does not match strict audit"
                )
            )
        item_text = item.get("text")
        if not isinstance(item_text, str) or claim.get("text_sha256") != text_sha256(
            item_text
        ):
            failures.append(
                _artifact_failure(
                    f"bundle receipt {role} text_sha256 claim does not match strict audit"
                )
            )
        if claim.get("raw_artifact") != raw_key:
            failures.append(
                _artifact_failure(
                    f"bundle receipt {role} raw_artifact claim is incorrect"
                )
            )
        if claim.get("raw_sha256") != raw_entry.get("sha256"):
            failures.append(
                _artifact_failure(
                    f"bundle receipt {role} raw_sha256 claim is incorrect"
                )
            )
    return failures


def _markdown_field(markdown: str, label: str) -> str | None:
    match = re.search(
        rf"^- {re.escape(label)}: `([^`]*)`\s*$",
        markdown,
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def _markdown_section_to_end(markdown: str, heading: str) -> str | None:
    marker = f"## {heading}"
    start = markdown.find(marker)
    if start < 0:
        return None
    return markdown[start + len(marker) :].strip()


def _markdown_between_sections(
    markdown: str,
    start_heading: str,
    end_heading: str,
) -> str | None:
    start_marker = f"## {start_heading}"
    end_marker = f"## {end_heading}"
    start = markdown.find(start_marker)
    if start < 0:
        return None
    start += len(start_marker)
    end = markdown.find(end_marker, start)
    if end < 0:
        return None
    return markdown[start:end].strip()


def _required_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _strict_artifact_label(key: str) -> str:
    return {
        "final": "final transcript",
        "audit": "audit Markdown",
        "audit_json": "strict audit JSON",
        "review_json": "review JSON",
        "primary_json": "primary raw JSON",
        "secondary_json": "secondary raw JSON",
    }.get(key, key)


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
    if (
        not reference
        or _resolve_bundle_reference(reference, raw_path.parent)
        != raw_path.resolve()
    ):
        return "strict audit raw_result_reference identity does not match the raw JSON"
    if isinstance(raw, dict):
        raw_engine = raw.get("engine")
        if raw_engine is not None and raw_engine != expected_engine:
            return "raw JSON engine identity does not match the requested engine"
    return None


def _resolve_bundle_reference(
    value: Any,
    bundle_root: Path,
) -> Path | None:
    """Resolve a bundle-internal reference without allowing relative escape.

    Version 1 artifacts written before portable references used absolute paths;
    those remain valid at their original location. New artifacts use paths
    relative to the bundle directory so the complete directory can be copied
    or archived without invalidating its semantic bindings.
    """
    try:
        reference = Path(str(value)).expanduser()
        root = bundle_root.resolve()
        if reference.is_absolute():
            return reference.resolve()
        if reference.drive or reference.root:
            return None
        resolved = (root / reference).resolve()
        resolved.relative_to(root)
        return resolved
    except (OSError, RuntimeError, TypeError, ValueError):
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

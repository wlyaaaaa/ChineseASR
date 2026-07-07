from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .risk_rules import RuleHit, evaluate_risk_rules
from .text_normalizer import to_simplified


@dataclass(frozen=True)
class AuditReport:
    status: str
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


def build_audit_report(
    primary_engine: str,
    primary_text: str,
    secondary_engine: str,
    secondary_text: str,
    conflict_threshold: float = 0.95,
    expect_empty: bool = False,
    primary_error: str | None = None,
    secondary_error: str | None = None,
) -> AuditReport:
    primary = to_simplified(primary_text.strip())
    secondary = to_simplified(secondary_text.strip())
    primary_norm = _normalize_for_compare(primary)
    secondary_norm = _normalize_for_compare(secondary)
    similarity = _similarity(primary_norm, secondary_norm)
    error_hits = _engine_error_hits(primary_engine, primary_error, secondary_engine, secondary_error)

    if error_hits:
        chosen = primary or secondary
        alternatives = tuple(text for text in (secondary, primary) if text and text != chosen)
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
        )

    if rule_hits:
        return AuditReport(
            status="suspicious",
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
        )

    if primary_norm == secondary_norm:
        return AuditReport(
            status="consistent",
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
        )

    if similarity >= conflict_threshold:
        return AuditReport(
            status="minor_difference",
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
        )

    conflict_hit = RuleHit(
        id="model_conflict",
        severity="high" if similarity < 0.80 else "medium",
        message="The primary and secondary ASR engines disagree materially.",
        evidence=f"similarity={similarity:.3f}",
    )
    return AuditReport(
        status="conflict",
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
    )


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

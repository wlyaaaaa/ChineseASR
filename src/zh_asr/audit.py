from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .text_normalizer import to_simplified


SUSPICIOUS_STOCK_PHRASES = (
    "谢谢观看",
    "感谢观看",
    "字幕",
    "字幕组",
    "订阅",
    "点赞",
)


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
    alternatives: tuple[str, ...]
    rationale: str


def build_audit_report(
    primary_engine: str,
    primary_text: str,
    secondary_engine: str,
    secondary_text: str,
    conflict_threshold: float = 0.95,
) -> AuditReport:
    primary = to_simplified(primary_text.strip())
    secondary = to_simplified(secondary_text.strip())
    primary_norm = _normalize_for_compare(primary)
    secondary_norm = _normalize_for_compare(secondary)
    similarity = _similarity(primary_norm, secondary_norm)
    flags = _flags(primary, secondary)

    if not primary_norm and not secondary_norm:
        return AuditReport(
            status="unclear",
            final_text="[听不清]",
            primary_engine=primary_engine,
            primary_text=primary,
            secondary_engine=secondary_engine,
            secondary_text=secondary,
            similarity=1.0,
            needs_review=True,
            flags=tuple(sorted(set(flags + ["empty_transcript"]))),
            alternatives=(),
            rationale="Both ASR engines returned empty text.",
        )

    chosen = primary or secondary
    alternatives = tuple(text for text in (secondary, primary) if text and text != chosen)

    if flags:
        return AuditReport(
            status="suspicious",
            final_text=f"[疑似]{chosen}",
            primary_engine=primary_engine,
            primary_text=primary,
            secondary_engine=secondary_engine,
            secondary_text=secondary,
            similarity=similarity,
            needs_review=True,
            flags=tuple(sorted(set(flags))),
            alternatives=alternatives,
            rationale="A known hallucination-like stock phrase appeared in at least one engine output.",
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
            alternatives=alternatives,
            rationale="The engines differ only slightly; final text follows the primary engine.",
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


def _flags(primary: str, secondary: str) -> list[str]:
    combined = primary + "\n" + secondary
    flags: list[str] = []
    if any(phrase in combined for phrase in SUSPICIOUS_STOCK_PHRASES):
        flags.append("suspicious_stock_phrase")
    return flags

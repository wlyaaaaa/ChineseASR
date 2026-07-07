from __future__ import annotations

import re
from dataclasses import dataclass

from .text_normalizer import to_simplified


SUSPICIOUS_STOCK_PHRASES = (
    "谢谢观看",
    "感谢观看",
    "字幕",
    "字幕组",
    "订阅",
    "点赞",
)

CONFLICT_THRESHOLD = 0.95
HIGH_CONFLICT_THRESHOLD = 0.80
LONG_UNPUNCTUATED_CJK_THRESHOLD = 80
LONG_UNPUNCTUATED_MAX_PUNCTUATION = 1


@dataclass(frozen=True)
class RuleHit:
    id: str
    severity: str
    message: str
    evidence: str


def evaluate_risk_rules(
    primary_text: str,
    secondary_text: str,
    final_text: str,
    similarity: float,
    expect_empty: bool = False,
) -> tuple[RuleHit, ...]:
    primary = primary_text or ""
    secondary = secondary_text or ""
    final = final_text or ""
    combined = "\n".join(text for text in (primary, secondary, final) if text)
    hits: list[RuleHit] = []

    substantive_len = _substantive_len(final)
    if expect_empty and substantive_len > 0:
        hits.append(
            RuleHit(
                id="empty_audio_hallucination",
                severity="high",
                message="Expected empty or unclear audio produced substantive text.",
                evidence=f"substantive_chars={substantive_len}",
            )
        )

    stock_phrase = _first_stock_phrase(combined)
    if stock_phrase:
        hits.append(
            RuleHit(
                id="suspicious_stock_phrase",
                severity="high",
                message="A known ASR hallucination-like stock phrase appeared.",
                evidence=f"phrase={stock_phrase}",
            )
        )

    repeated_span = _repeated_span(final)
    if repeated_span:
        hits.append(
            RuleHit(
                id="abnormal_repetition",
                severity="medium",
                message="The transcript contains an abnormal repeated span.",
                evidence=f"span={_clip(repeated_span)}",
            )
        )

    if _substantive_len(primary) > 0 and _substantive_len(secondary) > 0 and similarity < CONFLICT_THRESHOLD:
        severity = "high" if similarity < HIGH_CONFLICT_THRESHOLD else "medium"
        hits.append(
            RuleHit(
                id="model_conflict",
                severity=severity,
                message="The primary and secondary ASR engines disagree materially.",
                evidence=f"similarity={similarity:.3f}",
            )
        )

    if final and final != to_simplified(final):
        hits.append(
            RuleHit(
                id="traditional_residue",
                severity="medium",
                message="The final transcript still contains Traditional Chinese residue.",
                evidence="final_text_changed_by_simplification=true",
            )
        )

    cjk_len = _cjk_len(final)
    punctuation_count = _punctuation_count(final)
    if cjk_len >= LONG_UNPUNCTUATED_CJK_THRESHOLD and punctuation_count <= LONG_UNPUNCTUATED_MAX_PUNCTUATION:
        hits.append(
            RuleHit(
                id="long_unpunctuated_text",
                severity="medium",
                message="The transcript is long but has very little punctuation.",
                evidence=f"cjk_chars={cjk_len}, punctuation={punctuation_count}",
            )
        )

    return tuple(_dedupe_hits(hits))


def _first_stock_phrase(text: str) -> str:
    return next((phrase for phrase in SUSPICIOUS_STOCK_PHRASES if phrase in text), "")


def _substantive_len(text: str) -> int:
    normalized = _normalize_for_rules(text)
    normalized = normalized.replace("疑似", "").replace("听不清", "").replace("nospeechtextreturned", "")
    return len(normalized)


def _normalize_for_rules(text: str) -> str:
    simplified = to_simplified(text)
    return re.sub(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()\[\]【】<>《》|-]+", "", simplified).lower()


def _repeated_span(text: str) -> str:
    normalized = _normalize_for_rules(text)
    if len(normalized) < 6:
        return ""
    for size in range(2, min(20, len(normalized) // 3) + 1):
        match = re.search(rf"(.{{{size}}})\1{{2,}}", normalized)
        if match:
            return match.group(1)
    char_match = re.search(r"(.)\1{5,}", normalized)
    if char_match:
        return char_match.group(1)
    return ""


def _cjk_len(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def _punctuation_count(text: str) -> int:
    return len(re.findall(r"[，。！？、,.!?;；:：]", text))


def _clip(text: str, limit: int = 24) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


def _dedupe_hits(hits: list[RuleHit]) -> list[RuleHit]:
    seen: set[str] = set()
    deduped: list[RuleHit] = []
    for hit in hits:
        if hit.id in seen:
            continue
        seen.add(hit.id)
        deduped.append(hit)
    return deduped

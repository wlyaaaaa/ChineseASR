"""Ignore presentation differences without erasing numeric or technical meaning."""
from __future__ import annotations
import re
import unicodedata
from difflib import SequenceMatcher
from .text_normalizer import to_simplified

COMPARISON_POLICY_VERSION = "semantic-v1"
_NUMBER = re.compile(r"[+-]?(?:\d+(?:[.,:/-]\d+)*(?:%|‰)?|[零〇一二两三四五六七八九十百千万亿]+(?:点[零〇一二三四五六七八九]+)?)")
_NEGATION = re.compile(r"不|没|未|无|勿|别|否|禁止|拒绝")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9]*(?:[._:/+#-][a-z0-9]+)*", re.I)
_UNIT = re.compile(r"(?:\d|[一二两三四五六七八九十百千万亿])\s*(万元|亿元|元|毫秒|秒|分钟|小时|天|年|月|日|度|毫米|厘米|公里|米|毫克|千克|公斤|克|吨|GB|MB|KB|ms|kg|mg|km|ml|%|‰)", re.I)

def strip_audit_markers(text: str) -> str:
    return re.sub(r"\[(?:疑似|听不清|边界待复核)\]", "", text)

def normalize_comparison(text: str) -> str:
    text = unicodedata.normalize("NFKC", to_simplified(text)).replace("−", "-").lower()
    result = []
    for i, char in enumerate(text):
        before = text[i - 1] if i else ""
        after = text[i + 1] if i + 1 < len(text) else ""
        if char.isalnum():
            result.append(char)
        elif char in "+-/%‰=<>#":
            result.append(char)
        elif char in ".,:_" and before.isascii() and after.isascii() and before.isalnum() and after.isalnum():
            result.append(char)
    return "".join(result)

def critical_differences(left: str, right: str, terms: tuple[str, ...] = ()) -> tuple[str, ...]:
    left, right = normalize_comparison(left), normalize_comparison(right)
    if left == right:
        return ()
    found = []
    for name, pattern in (("number", _NUMBER), ("negation", _NEGATION), ("identifier", _IDENTIFIER), ("unit", _UNIT)):
        if pattern.findall(left) != pattern.findall(right):
            found.append(name)
    for tag, a, b, c, d in SequenceMatcher(a=left, b=right, autojunk=False).get_opcodes():
        if tag != "equal" and (_NEGATION.search(left[a:b]) or _NEGATION.search(right[c:d])):
            if "negation" not in found:
                found.append("negation")
    # Sequence alignment can move the subject around a shared negation token.
    # Compare local attachment as well as counts, without interpreting intent.
    negation_context = lambda text: [(text[max(0, m.start()-3):m.start()], m.group(), text[m.end():m.end()+3]) for m in _NEGATION.finditer(text)]
    if negation_context(left) != negation_context(right) and "negation" not in found:
        found.append("negation")
    for term in terms:
        key = normalize_comparison(term)
        if key and left.count(key) != right.count(key):
            found.append("configured_term")
            break
    return tuple(found)


def align_segment_evidence(primary_segments, secondary_segments):
    """Align continuous text, mapping changes back to original segment bounds.

    Bounds enclose source segments. They are not claimed word timestamps.
    Different ASR sentence boundaries alone never constitute disagreement.
    """
    from .result_writer import TranscriptSegment

    def flatten(segments):
        text, owners = "", []
        for i, segment in enumerate(segments):
            part = normalize_comparison(segment.text)
            text += part
            owners.extend([i] * len(part))
        return text, owners

    def combine(segments, ids):
        if not ids:
            return None
        parts = [segments[i] for i in ids]
        starts = [x.start_ms for x in parts if x.start_ms is not None]
        ends = [x.end_ms for x in parts if x.end_ms is not None]
        return TranscriptSegment(
            index=parts[0].index, text="".join(x.text for x in parts),
            start_ms=min(starts) if starts else None,
            end_ms=max(ends) if ends else None,
            raw_path=";".join(x.raw_path for x in parts),
        )

    left, lo = flatten(primary_segments)
    right, ro = flatten(secondary_segments)
    if left == right:
        return ()
    groups = []
    for tag, a, b, c, d in SequenceMatcher(a=left, b=right, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        li = tuple(dict.fromkeys(lo[a:b]))
        ri = tuple(dict.fromkeys(ro[c:d]))
        if not li and lo:
            li = (lo[min(a, len(lo) - 1)],)
        if not ri and ro:
            ri = (ro[min(c, len(ro) - 1)],)
        pair = (li, ri)
        if pair not in groups:
            groups.append(pair)
    return tuple((combine(primary_segments, li), combine(secondary_segments, ri)) for li, ri in groups)

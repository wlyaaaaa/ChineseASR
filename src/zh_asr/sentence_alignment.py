"""Keep FunASR ``sentence_info`` text on the token timestamps it describes.

FunASR 1.4.14 cuts Paraformer sentences by zipping per-token punctuation ids
and timestamps with ``text.split()``.  The text it splits is joined by
``_join_vad_texts``, which drops the space between two Chinese VAD chunks, so
every such boundary fuses two tokens and moves all later sentence text one
token earlier than its times.  On long recordings the drift reaches tens of
seconds and the tail becomes punctuation-only sentences.  Punctuation ids,
token timestamps, sentence bounds and ``spk`` labels stay aligned with each
other; only the sentence text is displaced.

The pipeline requests ``return_raw_text``: its space-separated tokens are
one-to-one with ``timestamp``.  ``realign_sentence_info`` rebuilds each
sentence's text from the tokens under its timestamps, and
``sentence_alignment_failures`` lets readers reject sentence text that cannot be
tied to its token timestamps, such as results written before this repair.
"""

from __future__ import annotations

import re
from typing import Any

SENTENCE_ALIGNMENT_SCHEMA = "chinese-asr.sentence-token-alignment.v1"
# FunASR ``timestamp_sentence`` ends every sentence with one of these marks.
_SENTENCE_PUNCTUATION = frozenset("，。？、")
# Older results carry no raw tokens: count one token per non-ASCII character
# and per ASCII word, ignoring whitespace and sentence punctuation.
_TEXT_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9'，。？、,.?!]")


class _Unaligned(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def realign_sentence_info(result: Any) -> Any:
    """Return *result* with FunASR sentence text rebuilt from its timed tokens.

    Sentence bounds, token timestamps and ``spk`` labels are kept.  Tokens left
    after the last closed sentence become one more sentence without a speaker
    label.  When the raw tokens cannot be tied one-to-one to the timestamps,
    the sentences move to ``sentence_info_unaligned`` so no reader takes their
    times as evidence.
    """

    if isinstance(result, list):
        return [realign_sentence_info(item) for item in result]
    if not isinstance(result, dict) or not isinstance(result.get("sentence_info"), list):
        return result
    updated = dict(result)
    try:
        sentences, realigned = _realigned_sentences(result)
    except _Unaligned as exc:
        updated["sentence_info_unaligned"] = updated.pop("sentence_info")
        updated["sentence_info_alignment"] = {
            "schema": SENTENCE_ALIGNMENT_SCHEMA,
            "status": "unaligned",
            "reason": exc.reason,
        }
        return updated
    updated["sentence_info"] = sentences
    updated["sentence_info_alignment"] = {
        "schema": SENTENCE_ALIGNMENT_SCHEMA,
        "status": "aligned",
        "token_source": "raw_text",
        "token_count": len(result["timestamp"]),
        "sentence_count": len(sentences),
        "realigned_sentence_count": realigned,
    }
    return updated


def sentence_alignment_failures(value: Any) -> list[str]:
    """Return why ``sentence_info`` text cannot be tied to its token timestamps.

    Only results that carry the token-level ``timestamp`` list beside
    ``sentence_info`` can be checked; other shapes return no failures.  Each
    sentence must cover the next run of token timestamps and its text must hold
    exactly those tokens, compared with ``raw_text`` tokens when present and by
    counting characters and ASCII words otherwise.
    """

    failures: list[str] = []
    _collect_failures(value, "$", failures)
    return failures


def _realigned_sentences(result: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    raw_text = result.get("raw_text")
    timestamps = result.get("timestamp")
    if not isinstance(raw_text, str) or not _is_span_list(timestamps):
        raise _Unaligned("raw_text_tokens_missing")
    tokens = raw_text.split()
    if len(tokens) != len(timestamps):
        raise _Unaligned("token_timestamp_count_mismatch")

    sentences: list[dict[str, Any]] = []
    realigned = 0
    cursor = 0
    for sentence in result["sentence_info"]:
        spans = sentence.get("timestamp") if isinstance(sentence, dict) else None
        if not isinstance(spans, list) or not spans:
            raise _Unaligned("sentence_timestamps_missing")
        end = cursor + len(spans)
        if spans != timestamps[cursor:end]:
            raise _Unaligned("sentence_timestamps_not_contiguous")
        text = str(sentence.get("text") or "")
        punctuation = text[-1] if text and text[-1] in _SENTENCE_PUNCTUATION else ""
        rebuilt = dict(sentence)
        rebuilt["text"] = _join_tokens(tokens[cursor:end]) + punctuation
        rebuilt["start"] = spans[0][0]
        rebuilt["end"] = spans[-1][1]
        rebuilt["raw_text"] = " ".join(tokens[cursor:end])
        realigned += rebuilt["text"] != text
        sentences.append(rebuilt)
        cursor = end
    if cursor < len(tokens):
        sentences.append(
            {
                "text": _join_tokens(tokens[cursor:]),
                "start": timestamps[cursor][0],
                "end": timestamps[-1][1],
                "timestamp": timestamps[cursor:],
                "raw_text": " ".join(tokens[cursor:]),
            }
        )
        realigned += 1
    return sentences, realigned


def _collect_failures(value: Any, path: str, failures: list[str]) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _collect_failures(item, f"{path}[{index}]", failures)
        return
    if not isinstance(value, dict):
        return
    sentences = value.get("sentence_info")
    timestamps = value.get("timestamp")
    if not isinstance(sentences, list) or not _is_span_list(timestamps):
        return

    raw_text = value.get("raw_text")
    exact = isinstance(raw_text, str) and all(
        isinstance(sentence, dict) and isinstance(sentence.get("raw_text"), str)
        for sentence in sentences
    )
    sentence_tokens: list[str] = []
    cursor = 0
    for index, sentence in enumerate(sentences):
        where = f"{path}.sentence_info[{index}]"
        spans = sentence.get("timestamp") if isinstance(sentence, dict) else None
        if not isinstance(spans, list) or not spans:
            failures.append(f"{where}: no token timestamps")
            return
        end = cursor + len(spans)
        if spans != timestamps[cursor:end]:
            failures.append(f"{where}: timestamps are not the next run of token timestamps")
            return
        if sentence.get("start") != spans[0][0] or sentence.get("end") != spans[-1][1]:
            failures.append(f"{where}: start/end differ from its token timestamps")
            return
        text = str(sentence.get("text") or "")
        if exact:
            tokens = sentence["raw_text"].split()
            if len(tokens) != len(spans) or _surface_letters(text) != "".join(tokens):
                failures.append(f"{where}: text does not hold its {len(spans)} timed tokens")
                return
            sentence_tokens.extend(tokens)
        else:
            count = len(_TEXT_TOKEN_RE.findall(text))
            if count != len(spans):
                failures.append(f"{where}: text holds {count} tokens for {len(spans)} token timestamps")
                return
        cursor = end
    if exact and raw_text.split()[: len(sentence_tokens)] != sentence_tokens:
        failures.append(f"{path}: sentence tokens differ from the raw_text tokens")


def _join_tokens(tokens: list[str]) -> str:
    """Join tokens the way FunASR ``timestamp_sentence`` spaces ASCII words."""

    text = ""
    for token in tokens:
        if text and (_is_ascii_letter(token[0]) or _is_ascii_letter(text[-1])):
            text += " "
        text += token
    return text


def _surface_letters(text: str) -> str:
    return "".join(
        character
        for character in text
        if not character.isspace() and character not in _SENTENCE_PUNCTUATION
    )


def _is_ascii_letter(character: str) -> bool:
    return character.isascii() and character.isalpha()


def _is_span_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(span, (list, tuple)) and len(span) >= 2 for span in value
    )

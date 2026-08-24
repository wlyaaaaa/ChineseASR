"""Small, fail-closed speaker-attribution projection for timestamped ASR text.

This module deliberately does not run diarization, retain voiceprints, or infer
identity from a model speaker number.  It turns already timestamped transcript
segments plus a narrow, source-specific context into the only result a consumer
needs: an anonymous speaker, a possible role, and one concrete reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .result_writer import TranscriptSegment, extract_segments


SPEAKER_ATTRIBUTION_SCHEMA = "chinese-asr.speaker-attribution.v1"
SPEAKER_ATTRIBUTION_CONTEXT_SCHEMA = "chinese-asr.speaker-attribution-context.v1"
XIAOMI_APP_STEREO_COHORT_ID = "xiaomi-app-stereo-2026-08-verified"

_RECORDING_KINDS = frozenset({"mono_call", "xiaomi_app_stereo", "other"})
_ROLES = frozenset({"self", "other"})
_CHANNELS = frozenset({"left", "right", "unknown"})


class SpeakerAttributionError(ValueError):
    """Raised when a caller supplies an ambiguous attribution request."""


def attribute_transcript_result(
    transcript_result: Any,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Attribute segments extracted from one existing ASR raw result.

    The result may contain any of the transcript shapes supported by
    :func:`zh_asr.result_writer.extract_segments`.  This function never loads
    audio or an ASR/voice model; callers obtain timestamped segments first when
    a real question actually needs speaker identity.
    """

    return attribute_segments(extract_segments(transcript_result), context)


def attribute_segments(
    segments: Sequence[TranscriptSegment],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the minimal consumer-facing attribution projection.

    ``source_identity`` is the only confirmed route and must come from a
    source fact.  ``dialogue_role`` and ``semantic_role`` are bounded
    per-segment evidence used only for timestamped calls.  A speaker number is
    anonymized for segmentation, never used as identity evidence.
    """

    recording_kind, cohort_id, evidence_by_index = _parse_context(context, len(segments))
    anonymous_speakers: dict[str, str] = {}
    projected: list[dict[str, Any]] = []
    has_gap = False

    for segment in segments:
        segment_evidence = evidence_by_index.get(segment.index, {})
        status, candidate_role, basis = _attribute_segment(
            segment,
            recording_kind=recording_kind,
            cohort_id=cohort_id,
            evidence=segment_evidence,
        )
        if status == "unknown":
            has_gap = True
        projected.append(
            {
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "text": segment.text,
                "speaker": _anonymous_speaker(segment.speaker, anonymous_speakers),
                "attribution_status": status,
                "candidate_role": candidate_role,
                "basis": basis,
            }
        )

    return {
        "schema": SPEAKER_ATTRIBUTION_SCHEMA,
        "segments": projected,
        "speaker_attribution_gap": has_gap,
    }


def write_speaker_attribution(
    output_path: Path,
    transcript_result: Any,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one JSON projection and return the same in-memory result."""

    payload = attribute_transcript_result(transcript_result, context)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _attribute_segment(
    segment: TranscriptSegment,
    *,
    recording_kind: str,
    cohort_id: str | None,
    evidence: Mapping[str, Any],
) -> tuple[str, str, str]:
    if not _has_complete_time_range(segment):
        return (
            "unknown",
            "unknown",
            "转写没有可用的起止时间，不能把这段话归给任何人。",
        )

    source_identity = _role_evidence(evidence, "source_identity")
    if source_identity is not None:
        role, reason = source_identity
        return "confirmed", role, reason

    dialogue_role = _role_evidence(evidence, "dialogue_role")
    semantic_role = _role_evidence(evidence, "semantic_role")
    inferred_role = _combined_inferred_role(dialogue_role, semantic_role)
    if inferred_role is not None:
        role, reason = inferred_role
        return "inferred", role, reason
    if dialogue_role is not None and semantic_role is not None:
        return (
            "unknown",
            "unknown",
            "对话角色与句义线索相互矛盾，不能安全归属这段话。",
        )

    channel = _channel(evidence)
    if (
        recording_kind == "xiaomi_app_stereo"
        and cohort_id == XIAOMI_APP_STEREO_COHORT_ID
        and channel == "right"
    ):
        return (
            "inferred",
            "self",
            "该段位于已验证小米应用立体声 cohort 的右声道，只能作为本人候选。",
        )

    if recording_kind == "mono_call":
        return (
            "unknown",
            "unknown",
            "这是一段单声道通话，缺少可核对的联系人、对话角色或句义依据。",
        )
    if recording_kind == "xiaomi_app_stereo" and cohort_id != XIAOMI_APP_STEREO_COHORT_ID:
        return (
            "unknown",
            "unknown",
            "该录音不属于已验证的小米立体声 cohort，声道不能用于判断本人。",
        )
    return (
        "unknown",
        "unknown",
        "没有可核对的来源身份、对话角色或句义依据。",
    )


def _parse_context(
    context: Mapping[str, Any],
    segment_count: int,
) -> tuple[str, str | None, dict[int, Mapping[str, Any]]]:
    if not isinstance(context, Mapping):
        raise SpeakerAttributionError("Speaker-attribution context must be a JSON object.")
    if context.get("schema") != SPEAKER_ATTRIBUTION_CONTEXT_SCHEMA:
        raise SpeakerAttributionError(
            f"Context schema must be {SPEAKER_ATTRIBUTION_CONTEXT_SCHEMA!r}."
        )

    recording_kind = context.get("recording_kind")
    if recording_kind not in _RECORDING_KINDS:
        raise SpeakerAttributionError(
            "recording_kind must be one of mono_call, xiaomi_app_stereo, or other."
        )

    cohort_id = context.get("stereo_cohort_id")
    if cohort_id is not None and (not isinstance(cohort_id, str) or not cohort_id.strip()):
        raise SpeakerAttributionError("stereo_cohort_id must be a non-empty string or null.")

    raw_evidence = context.get("segment_evidence", [])
    if not isinstance(raw_evidence, list):
        raise SpeakerAttributionError("segment_evidence must be a list.")

    by_index: dict[int, Mapping[str, Any]] = {}
    for item in raw_evidence:
        if not isinstance(item, Mapping):
            raise SpeakerAttributionError("Each segment_evidence item must be an object.")
        index = item.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < segment_count:
            raise SpeakerAttributionError("segment_evidence index must name an existing segment.")
        if index in by_index:
            raise SpeakerAttributionError("segment_evidence cannot contain the same index twice.")
        _channel(item)
        _role_evidence(item, "source_identity")
        _role_evidence(item, "dialogue_role")
        _role_evidence(item, "semantic_role")
        by_index[index] = item
    return recording_kind, cohort_id, by_index


def _role_evidence(
    evidence: Mapping[str, Any],
    field: str,
) -> tuple[str, str] | None:
    value = evidence.get(field)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SpeakerAttributionError(f"{field} must be an object when present.")
    role = value.get("candidate_role")
    reason = value.get("reason")
    if role not in _ROLES:
        raise SpeakerAttributionError(f"{field}.candidate_role must be self or other.")
    if not isinstance(reason, str) or not reason.strip() or "\n" in reason:
        raise SpeakerAttributionError(f"{field}.reason must be one non-empty sentence.")
    return role, reason.strip()


def _combined_inferred_role(
    dialogue_role: tuple[str, str] | None,
    semantic_role: tuple[str, str] | None,
) -> tuple[str, str] | None:
    if dialogue_role is None:
        return semantic_role
    if semantic_role is None:
        return dialogue_role
    if dialogue_role[0] == semantic_role[0]:
        return dialogue_role
    return None


def _channel(evidence: Mapping[str, Any]) -> str:
    channel = evidence.get("channel", "unknown")
    if channel not in _CHANNELS:
        raise SpeakerAttributionError("channel must be left, right, or unknown.")
    return channel


def _has_complete_time_range(segment: TranscriptSegment) -> bool:
    if segment.start_ms is None or segment.end_ms is None:
        return False
    return segment.end_ms >= segment.start_ms


def _anonymous_speaker(raw_speaker: Any, aliases: dict[str, str]) -> str:
    if raw_speaker is None:
        return "speaker-unknown"
    key = json.dumps(raw_speaker, ensure_ascii=False, sort_keys=True, default=str)
    if key not in aliases:
        aliases[key] = f"speaker-{len(aliases) + 1}"
    return aliases[key]

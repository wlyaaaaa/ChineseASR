"""Reversible speaker-role projection with soft, source-bound evidence fusion.

The projection never treats a diarization cluster, an anonymous ``speaker``
number, a CAM++ score, or a channel by itself as a confirmed identity. It can
use any one usable positive signal to make a reversible ``inferred`` judgement;
only unavailable, genuinely balanced, or conflicting evidence stays ``unknown``.
``confirmed`` remains reserved for an authoritative source reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .result_writer import TranscriptSegment, canonical_json_sha256, extract_segments
from .speaker_evidence import (
    SELF_PERSON_ID,
    SELF_SPEAKER_EVIDENCE_SCHEMA,
    SPEAKER_MODEL_EVIDENCE_SCHEMA,
    VOICE_SCORE_AMBIGUITY_MARGIN,
)


SPEAKER_ATTRIBUTION_SCHEMA = "chinese-asr.speaker-attribution.v2"
SPEAKER_ATTRIBUTION_CONTEXT_SCHEMA = "chinese-asr.speaker-attribution-context.v2"
_LEGACY_CONTEXT_SCHEMA = "chinese-asr.speaker-attribution-context.v1"
XIAOMI_APP_STEREO_COHORT_ID = "xiaomi-app-stereo-2026-08-verified"

_RECORDING_KINDS = frozenset({"mono_call", "xiaomi_app_stereo", "other"})
_ROLES = frozenset({"self", "other"})
_CHANNELS = frozenset({"mix", "left", "right"})
_CHANNEL_BINDINGS = frozenset({"mixed_not_channel_evidence", "exact_stereo_channel"})
_CONTEXTUAL_KINDS = frozenset(
    {
        "source_context",
        "contact_role",
        "dialogue_role",
        "semantic_role",
        "cross_recording_role",
    }
)


class SpeakerAttributionError(ValueError):
    """Raised when an attribution context or evidence document is malformed."""


@dataclass(frozen=True)
class _Context:
    recording_kind: str
    cohort_id: str | None
    recording_source_sha256: str | None
    evidence_by_index: dict[int, Mapping[str, Any]]


def attribute_transcript_result(
    transcript_result: Any,
    context: Mapping[str, Any],
    *,
    voice_evidence: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Attribute timestamped existing transcript segments without loading audio/models."""

    return attribute_segments(
        extract_segments(transcript_result),
        context,
        voice_evidence=voice_evidence,
    )


def attribute_segments(
    segments: Sequence[TranscriptSegment],
    context: Mapping[str, Any],
    *,
    voice_evidence: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return the minimal consumer-facing, evidence-explaining role projection."""

    parsed = _parse_context(context, len(segments))
    parsed_voice_evidence = _parse_voice_evidence(voice_evidence)
    if parsed_voice_evidence and parsed.recording_source_sha256 is None:
        raise SpeakerAttributionError(
            "recording_audio.sha256 is required before voice evidence can bind to transcript segments."
        )

    anonymous_speakers: dict[str, str] = {}
    projected: list[dict[str, Any]] = []
    has_gap = False
    for segment in segments:
        status, candidate_role, basis, evidence = _attribute_segment(
            segment,
            parsed,
            parsed.evidence_by_index.get(segment.index, {}),
            parsed_voice_evidence,
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
                "evidence": evidence,
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
    *,
    voice_evidence: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Write one JSON projection and return the same in-memory result."""

    payload = attribute_transcript_result(
        transcript_result,
        context,
        voice_evidence=voice_evidence,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _attribute_segment(
    segment: TranscriptSegment,
    context: _Context,
    evidence: Mapping[str, Any],
    voice_evidence: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str, list[dict[str, Any]]]:
    if not _has_complete_time_range(segment):
        return (
            "unknown",
            "unknown",
            "转写没有可用的起止时间，不能把这段话与来源、声道或声纹证据精确绑定。",
            [],
        )

    signals = _context_signals(evidence)
    signals.extend(_voice_signals(segment, context, voice_evidence))
    signals.extend(_xiaomi_channel_signals(segment, context, voice_evidence))

    authoritative = [item for item in signals if item["kind"] == "authoritative_source"]
    authoritative_roles = {item["candidate_role"] for item in authoritative}
    if len(authoritative_roles) == 1:
        role = next(iter(authoritative_roles))
        return (
            "confirmed",
            role,
            authoritative[0]["reason"],
            signals,
        )
    if len(authoritative_roles) > 1:
        return (
            "unknown",
            "unknown",
            "多个权威来源引用对这段话给出了相互矛盾的身份结论。",
            signals,
        )

    directional = [item for item in signals if item["candidate_role"] in _ROLES]
    active_roles = {item["candidate_role"] for item in directional}
    if not active_roles:
        return (
            "unknown",
            "unknown",
            _unknown_reason(context, signals),
            signals,
        )
    if len(active_roles) == 1:
        role = next(iter(active_roles))
        return (
            "inferred",
            role,
            _basis_for_role(role, directional),
            signals,
        )

    contextual = [item for item in directional if item["kind"] in _CONTEXTUAL_KINDS]
    contextual_roles = {item["candidate_role"] for item in contextual}
    if len(contextual_roles) == 1:
        role = next(iter(contextual_roles))
        opposing = [item for item in directional if item["candidate_role"] != role]
        return (
            "inferred",
            role,
            _basis_for_context_override(role, contextual, opposing),
            signals,
        )
    if len(contextual_roles) > 1:
        reason = "来源、联系人、对话或句义判断彼此冲突，现有材料无法给出可解释的取舍，保留为 unknown。"
    else:
        reason = "只有相互矛盾的声纹或声道线索，缺少具体的来源、联系人、对话或句义判断来消解，保留为 unknown。"
    return ("unknown", "unknown", reason, signals)


def _parse_context(context: Mapping[str, Any], segment_count: int) -> _Context:
    if not isinstance(context, Mapping):
        raise SpeakerAttributionError("Speaker-attribution context must be a JSON object.")
    if context.get("schema") not in {
        SPEAKER_ATTRIBUTION_CONTEXT_SCHEMA,
        _LEGACY_CONTEXT_SCHEMA,
    }:
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
    recording_source_sha256 = _recording_source_sha256(context.get("recording_audio"))

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
        for field in (
            "source_identity",
            "contact_role",
            "dialogue_role",
            "semantic_role",
            "cross_recording_role",
        ):
            _role_evidence(item, field)
        by_index[index] = item
    return _Context(recording_kind, cohort_id, recording_source_sha256, by_index)


def _recording_source_sha256(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SpeakerAttributionError("recording_audio must be an object when present.")
    source_hash = value.get("sha256")
    _validate_sha256(source_hash, "recording_audio")
    return str(source_hash).lower()


def _context_signals(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    source = _role_evidence(evidence, "source_identity")
    if source is not None:
        if source["authority_ref"] is not None:
            signals.append(
                _signal(
                    "authoritative_source",
                    source["candidate_role"],
                    source["reason"],
                    authority_ref=source["authority_ref"],
                )
            )
        else:
            signals.append(
                _signal(
                    "source_context",
                    source["candidate_role"],
                    source["reason"],
                )
            )
    for field in ("contact_role", "dialogue_role", "semantic_role", "cross_recording_role"):
        item = _role_evidence(evidence, field)
        if item is not None:
            signals.append(
                _signal(
                    field,
                    item["candidate_role"],
                    item["reason"],
                )
            )
    return signals


def _voice_signals(
    segment: TranscriptSegment,
    context: _Context,
    documents: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for document in _matching_voice_evidence(segment, context, documents):
        score = document["score"]
        value = float(score["value"])
        threshold = float(score["threshold"])
        evidence_ref = canonical_json_sha256(document)
        if abs(value - threshold) <= VOICE_SCORE_AMBIGUITY_MARGIN:
            signals.append(
                _signal(
                    "voice_similarity_near_threshold",
                    "unknown",
                    "本段 CAM++ 相似度接近当前阈值，单独不作为方向性身份线索。",
                    evidence_sha256=evidence_ref,
                )
            )
            continue
        if value > threshold:
            signals.append(
                _signal(
                    "voice_similarity",
                    "self",
                    f"本段 CAM++ 余弦相似度 {value:.4f} 高于阈值 {threshold:.4f}，支持本人候选。",
                    evidence_sha256=evidence_ref,
                )
            )
        else:
            signals.append(
                _signal(
                    "voice_similarity",
                    "other",
                    f"本段 CAM++ 余弦相似度 {value:.4f} 低于阈值 {threshold:.4f}，仅是弱的非本人线索。",
                    evidence_sha256=evidence_ref,
                )
            )
    return signals


def _xiaomi_channel_signals(
    segment: TranscriptSegment,
    context: _Context,
    documents: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if (
        context.recording_kind != "xiaomi_app_stereo"
        or context.cohort_id != XIAOMI_APP_STEREO_COHORT_ID
    ):
        return []
    signals: list[dict[str, Any]] = []
    for document in _matching_voice_evidence(segment, context, documents):
        segment_ref = document["target"]["segment"]
        if (
            segment_ref["channel"] == "right"
            and segment_ref["channel_binding"] == "exact_stereo_channel"
        ):
            signals.append(
                _signal(
                    "xiaomi_exact_stereo_channel",
                    "self",
                    "该段从已验证小米应用立体声 cohort 的原始右声道精确提取，只支持本人候选。",
                    evidence_sha256=canonical_json_sha256(document),
                )
            )
    return signals


def _matching_voice_evidence(
    segment: TranscriptSegment,
    context: _Context,
    documents: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if context.recording_source_sha256 is None:
        return []
    matches: list[Mapping[str, Any]] = []
    for document in documents:
        target = document["target"]
        source = target["source"]
        segment_ref = target["segment"]
        if source["sha256"].lower() != context.recording_source_sha256:
            continue
        if (
            segment_ref["start_ms"] == segment.start_ms
            and segment_ref["end_ms"] == segment.end_ms
        ):
            matches.append(document)
    return matches


def _parse_voice_evidence(value: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise SpeakerAttributionError("voice_evidence must be a sequence of JSON objects.")
    documents: list[Mapping[str, Any]] = []
    for document in value:
        if not isinstance(document, Mapping):
            raise SpeakerAttributionError("Each voice_evidence document must be a JSON object.")
        if document.get("schema") != SELF_SPEAKER_EVIDENCE_SCHEMA:
            raise SpeakerAttributionError(
                f"voice_evidence schema must be {SELF_SPEAKER_EVIDENCE_SCHEMA!r}."
            )
        if document.get("person_id") != SELF_PERSON_ID:
            raise SpeakerAttributionError("voice_evidence may only refer to person:self.")
        target = document.get("target")
        if not isinstance(target, Mapping):
            raise SpeakerAttributionError("voice_evidence target must be an object.")
        source = target.get("source")
        segment = target.get("segment")
        if not isinstance(source, Mapping) or not isinstance(segment, Mapping):
            raise SpeakerAttributionError("voice_evidence target source/segment is malformed.")
        _validate_sha256(source.get("sha256"), "voice_evidence target source")
        _finite_time(segment.get("start_ms"), "voice_evidence start_ms")
        _finite_time(segment.get("end_ms"), "voice_evidence end_ms")
        if segment["end_ms"] <= segment["start_ms"]:
            raise SpeakerAttributionError("voice_evidence end_ms must be greater than start_ms.")
        if segment.get("channel") not in _CHANNELS:
            raise SpeakerAttributionError("voice_evidence channel must be mix, left, or right.")
        if segment.get("channel_binding") not in _CHANNEL_BINDINGS:
            raise SpeakerAttributionError("voice_evidence channel binding is invalid.")
        score = document.get("score")
        if not isinstance(score, Mapping) or score.get("metric") != "cosine_similarity":
            raise SpeakerAttributionError("voice_evidence must contain a cosine similarity score.")
        for field in ("value", "threshold"):
            _finite_score(score.get(field), f"voice_evidence score.{field}")
        model = document.get("model")
        if not isinstance(model, Mapping) or model.get("schema") != SPEAKER_MODEL_EVIDENCE_SCHEMA:
            raise SpeakerAttributionError("voice_evidence must contain hash-bound speaker model evidence.")
        documents.append(document)
    return tuple(documents)


def _role_evidence(evidence: Mapping[str, Any], field: str) -> dict[str, str | None] | None:
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
    authority_ref = value.get("authority_ref") if field == "source_identity" else None
    if authority_ref is not None and (
        not isinstance(authority_ref, str) or not authority_ref.strip() or "\n" in authority_ref
    ):
        raise SpeakerAttributionError("source_identity.authority_ref must be one non-empty source reference.")
    return {
        "candidate_role": str(role),
        "reason": reason.strip(),
        "authority_ref": authority_ref.strip() if isinstance(authority_ref, str) else None,
    }


def _signal(
    kind: str,
    candidate_role: str,
    reason: str,
    **reference: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": kind,
        "candidate_role": candidate_role,
        "reason": reason,
    }
    if reference:
        payload["reference"] = reference
    return payload


def _basis_for_role(role: str, signals: Sequence[Mapping[str, Any]]) -> str:
    reasons = [item["reason"] for item in signals if item["candidate_role"] == role]
    return "；".join(reasons)


def _basis_for_context_override(
    role: str,
    contextual: Sequence[Mapping[str, Any]],
    opposing: Sequence[Mapping[str, Any]],
) -> str:
    reasons = _basis_for_role(role, contextual)
    opposing_kinds = "、".join(sorted({str(item["kind"]) for item in opposing}))
    return (
        f"{reasons}；与此相反的{opposing_kinds}只提供可撤销的声学线索，"
        "不能替代这条具体的来源/语义判断，因此暂列 inferred；后续更强原始来源或语义证据可以推翻。"
    )


def _unknown_reason(context: _Context, signals: Sequence[Mapping[str, Any]]) -> str:
    if any(item["kind"] == "voice_similarity_near_threshold" for item in signals):
        return "现有声纹相似度接近阈值，且没有其他可用线索，保留为 unknown。"
    if context.recording_kind == "xiaomi_app_stereo" and context.cohort_id != XIAOMI_APP_STEREO_COHORT_ID:
        return "该录音不属于已验证的小米立体声 cohort，声道不能用于判断本人。"
    if context.recording_kind == "mono_call":
        return "这是一段单声道通话，缺少可核对的联系人、对话角色、句义或声纹依据。"
    return "没有可核对的来源、联系人、对话角色、句义、跨录音或声纹依据。"


def _validate_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise SpeakerAttributionError(f"{label} sha256 is invalid.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise SpeakerAttributionError(f"{label} sha256 is invalid.") from exc


def _finite_time(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SpeakerAttributionError(f"{label} must be a finite number.")


def _finite_score(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SpeakerAttributionError(f"{label} must be a finite number.")


def _has_complete_time_range(segment: TranscriptSegment) -> bool:
    if segment.start_ms is None or segment.end_ms is None:
        return False
    try:
        return (
            math.isfinite(float(segment.start_ms))
            and math.isfinite(float(segment.end_ms))
            and segment.end_ms >= segment.start_ms
        )
    except (TypeError, ValueError):
        return False


def _anonymous_speaker(raw_speaker: Any, aliases: dict[str, str]) -> str:
    if raw_speaker is None:
        return "speaker-unknown"
    key = json.dumps(raw_speaker, ensure_ascii=False, sort_keys=True, default=str)
    if key not in aliases:
        aliases[key] = f"speaker-{len(aliases) + 1}"
    return aliases[key]

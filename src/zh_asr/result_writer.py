from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .audio_outcome import build_objective_result, write_objective_result


@dataclass(frozen=True)
class TranscriptSegment:
    index: int
    text: str
    start_ms: int | float | None = None
    end_ms: int | float | None = None
    speaker: Any = None
    raw_path: str = "$"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of the exact bytes persisted at *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    """Hash UTF-8 text using the bundle's canonical text encoding."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value with deterministic key/order formatting."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_text(result: Any) -> str:
    return "\n".join(segment.text for segment in extract_segments(result)).strip()


def extract_segments(result: Any) -> tuple[TranscriptSegment, ...]:
    """Extract verbatim transcript segments while retaining raw-result pointers."""
    collected: list[TranscriptSegment] = []
    _collect_segments(result, "$", collected)
    return tuple(
        TranscriptSegment(
            index=index,
            text=segment.text,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            speaker=segment.speaker,
            raw_path=segment.raw_path,
        )
        for index, segment in enumerate(collected)
    )


def write_transcript_bundle(
    audio_path: Path,
    result: object,
    out_dir: Path,
    engine: str,
    caller_binding: Mapping[str, Any] | None = None,
    primary_provenance: Mapping[str, Any] | None = None,
    request_options: Mapping[str, Any] | None = None,
) -> dict[str, Path | str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    json_path = out_dir / f"{stem}.{engine}.raw.json"
    markdown_path = out_dir / f"{stem}.{engine}.md"
    objective_path = out_dir / f"{stem}.{engine}.objective-result.json"

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    transcript = extract_text(result)
    objective = build_objective_result(
        audio_path=audio_path,
        mode="quick",
        engines=[engine],
        primary_text=transcript,
        primary_result=result,
        raw_artifacts=[
            {
                "schema": "media.raw-artifact-ref.v1",
                "role": "lexical_primary",
                "engine": engine,
                "path": json_path.name,
                "size_bytes": json_path.stat().st_size,
                "sha256": file_sha256(json_path),
            }
        ],
        request={**dict(request_options or {}), "engine": engine},
        caller_binding=caller_binding,
        primary_provenance=primary_provenance,
    )
    write_objective_result(objective_path, objective)
    markdown = "\n".join(
        [
            f"# {stem} Transcript",
            "",
            f"- Audio: `{audio_path}`",
            f"- Engine: `{engine}`",
            "- Evidence status: `not_applicable`",
            f"- Objective outcome: `{objective['objective_outcome']}`",
            f"- Objective result: `{objective_path.name}`",
            f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
            "",
            "## Transcript",
            "",
            transcript or "_No speech text returned._",
            "",
        ]
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    return {
        "json": json_path,
        "markdown": markdown_path,
        "objective_result": objective_path,
        "objective_outcome": objective["objective_outcome"],
        "audio_result_status": objective["objective_outcome"],
        "evidence_status": "not_applicable",
    }


def _collect_segments(value: Any, raw_path: str, collected: list[TranscriptSegment]) -> None:
    if isinstance(value, str):
        text = _clean_text(value)
        if text:
            collected.append(TranscriptSegment(len(collected), text, raw_path=raw_path))
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _collect_segments(item, f"{raw_path}[{index}]", collected)
        return

    if isinstance(value, dict):
        if isinstance(value.get("sentence_info"), list):
            count_before = len(collected)
            _collect_segment_list(
                value["sentence_info"],
                f"{raw_path}.sentence_info",
                collected,
            )
            if len(collected) > count_before:
                return

        if isinstance(value.get("segments"), list):
            count_before = len(collected)
            _collect_segment_list(value["segments"], f"{raw_path}.segments", collected)
            if len(collected) > count_before:
                return

        text = value.get("text") or value.get("sentence")
        if text:
            cleaned = _clean_text(str(text))
            if cleaned:
                start_ms, end_ms = _extract_times(value)
                collected.append(
                    TranscriptSegment(
                        index=len(collected),
                        text=cleaned,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        speaker=_first_present(value, "spk", "speaker", "speaker_id"),
                        raw_path=raw_path,
                    )
                )


def _collect_segment_list(
    segments: list[Any],
    raw_path: str,
    collected: list[TranscriptSegment],
) -> None:
    for index, segment in enumerate(segments):
        segment_path = f"{raw_path}[{index}]"
        if not isinstance(segment, dict):
            _collect_segments(segment, segment_path, collected)
            continue
        text = segment.get("text") or segment.get("sentence")
        if not text:
            continue
        cleaned = _clean_text(str(text))
        if not cleaned:
            continue
        start_ms, end_ms = _extract_times(segment)
        collected.append(
            TranscriptSegment(
                index=len(collected),
                text=cleaned,
                start_ms=start_ms,
                end_ms=end_ms,
                speaker=_first_present(segment, "spk", "speaker", "speaker_id"),
                raw_path=segment_path,
            )
        )


def _extract_times(value: dict[str, Any]) -> tuple[int | float | None, int | float | None]:
    start = _first_present(value, "start_ms", "start", "begin_ms", "begin")
    end = _first_present(value, "end_ms", "end", "finish_ms", "finish")
    timestamp = value.get("timestamp")
    if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
        start = start if start is not None else timestamp[0]
        end = end if end is not None else timestamp[1]
    return _number_or_none(start), _number_or_none(end)


def _first_present(value: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in value and value[key] is not None:
            return value[key]
    return None


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _clean_text(text: str) -> str:
    return re.sub(r"<\|[^|]+?\|>", "", text).strip()

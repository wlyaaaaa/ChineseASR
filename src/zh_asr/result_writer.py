from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def extract_text(result: Any) -> str:
    if isinstance(result, list):
        return "\n".join(part for item in result for part in _extract_parts(item)).strip()
    return "\n".join(_extract_parts(result)).strip()


def write_transcript_bundle(audio_path: Path, result: object, out_dir: Path, engine: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    json_path = out_dir / f"{stem}.{engine}.raw.json"
    markdown_path = out_dir / f"{stem}.{engine}.md"

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    transcript = extract_text(result)
    markdown = "\n".join(
        [
            f"# {stem} Transcript",
            "",
            f"- Audio: `{audio_path}`",
            f"- Engine: `{engine}`",
            f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
            "",
            "## Transcript",
            "",
            transcript or "_No speech text returned._",
            "",
        ]
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _extract_parts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        if isinstance(value.get("sentence_info"), list):
            parts: list[str] = []
            for segment in value["sentence_info"]:
                if isinstance(segment, dict):
                    text = segment.get("text") or segment.get("sentence")
                    if text:
                        parts.append(_clean_text(str(text)))
            if parts:
                return parts
        text = value.get("text") or value.get("sentence")
        if text:
            return [_clean_text(str(text))]
    return []


def _clean_text(text: str) -> str:
    return re.sub(r"<\|[^|]+?\|>", "", text).strip()

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .audit import AuditReport, build_audit_report
from .result_writer import extract_text


def write_strict_bundle(
    audio_path: Path,
    primary_engine: str,
    primary_result: object,
    secondary_engine: str,
    secondary_result: object,
    out_dir: Path,
    expect_empty: bool = False,
    primary_error: str | None = None,
    secondary_error: str | None = None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    primary_text = extract_text(primary_result)
    secondary_text = extract_text(secondary_result)
    report = build_audit_report(
        primary_engine,
        primary_text,
        secondary_engine,
        secondary_text,
        expect_empty=expect_empty,
        primary_error=primary_error,
        secondary_error=secondary_error,
    )

    final_path = out_dir / f"{stem}.strict.md"
    audit_path = out_dir / f"{stem}.strict.audit.md"
    audit_json_path = out_dir / f"{stem}.strict.audit.json"
    primary_json_path = out_dir / f"{stem}.{primary_engine}.raw.json"
    secondary_json_path = out_dir / f"{stem}.{secondary_engine}.raw.json"

    primary_json_path.write_text(json.dumps(primary_result, ensure_ascii=False, indent=2), encoding="utf-8")
    secondary_json_path.write_text(json.dumps(secondary_result, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_json_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")

    final_path.write_text(_format_final_markdown(audio_path, report), encoding="utf-8")
    audit_path.write_text(_format_audit_markdown(audio_path, report), encoding="utf-8")

    return {
        "final": final_path,
        "audit": audit_path,
        "audit_json": audit_json_path,
        "primary_json": primary_json_path,
        "secondary_json": secondary_json_path,
    }


def _format_final_markdown(audio_path: Path, report: AuditReport) -> str:
    return "\n".join(
        [
            f"# {audio_path.stem} Strict Transcript",
            "",
            f"- Audio: `{audio_path}`",
            f"- Primary: `{report.primary_engine}`",
            f"- Secondary: `{report.secondary_engine}`",
            f"- Status: `{report.status}`",
            f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
            "",
            "## Transcript",
            "",
            report.final_text,
            "",
        ]
    )


def _format_audit_markdown(audio_path: Path, report: AuditReport) -> str:
    alternatives = "\n".join(f"- {text}" for text in report.alternatives) or "- None"
    flags = ", ".join(report.flags) if report.flags else "none"
    rule_hits = "\n".join(
        f"- `{hit.id}` ({hit.severity}): {hit.message} Evidence: `{hit.evidence}`"
        for hit in report.rule_hits
    ) or "- None"
    return "\n".join(
        [
            f"# {audio_path.stem} Strict Audit",
            "",
            f"- Audio: `{audio_path}`",
            f"- Status: `{report.status}`",
            f"- Needs review: `{str(report.needs_review).lower()}`",
            f"- Similarity: `{report.similarity:.3f}`",
            f"- Flags: `{flags}`",
            "",
            "## Rule Hits",
            "",
            rule_hits,
            "",
            "## Final Guess",
            "",
            report.final_text,
            "",
            f"## {report.primary_engine}",
            "",
            report.primary_text or "_Empty_",
            "",
            f"## {report.secondary_engine}",
            "",
            report.secondary_text or "_Empty_",
            "",
            "## Alternatives",
            "",
            alternatives,
            "",
            "## Rationale",
            "",
            report.rationale,
            "",
        ]
    )

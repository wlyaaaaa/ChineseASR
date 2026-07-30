from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .audit import (
    STRICT_BUNDLE_ARTIFACT_KEYS,
    STRICT_BUNDLE_RECEIPT_SCHEMA_VERSION,
    AuditReport,
    build_audit_report,
    validate_strict_artifact_bundle,
)
from .result_writer import (
    canonical_json_sha256,
    extract_segments,
    extract_text,
    file_sha256,
    text_sha256,
)

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
    primary_role: str = "lexical_primary",
    secondary_role: str = "lexical_verifier",
    primary_provenance: Mapping[str, Any] | None = None,
    secondary_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    primary_text = extract_text(primary_result)
    secondary_text = extract_text(secondary_result)

    final_path = out_dir / f"{stem}.strict.md"
    audit_path = out_dir / f"{stem}.strict.audit.md"
    audit_json_path = out_dir / f"{stem}.strict.audit.json"
    review_json_path = out_dir / f"{stem}.strict.review.json"
    receipt_path = out_dir / f"{stem}.strict.receipt.json"
    primary_json_path = out_dir / f"{stem}.{primary_engine}.raw.json"
    secondary_json_path = out_dir / f"{stem}.{secondary_engine}.raw.json"

    report = build_audit_report(
        primary_engine,
        primary_text,
        secondary_engine,
        secondary_text,
        expect_empty=expect_empty,
        primary_error=primary_error,
        secondary_error=secondary_error,
        primary_role=primary_role,
        secondary_role=secondary_role,
        primary_segments=extract_segments(primary_result),
        secondary_segments=extract_segments(secondary_result),
        primary_raw_result_reference=primary_json_path.name,
        secondary_raw_result_reference=secondary_json_path.name,
        primary_provenance=primary_provenance,
        secondary_provenance=secondary_provenance,
    )

    primary_json_path.write_text(
        json.dumps(primary_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    secondary_json_path.write_text(
        json.dumps(secondary_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit_payload = {
        **asdict(report),
        "bundle_receipt_reference": receipt_path.name,
    }
    audit_json_path.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    review_json_path.write_text(
        json.dumps(
            _review_payload(audio_path, report, receipt_path),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    final_path.write_text(
        _format_final_markdown(audio_path, report, receipt_path),
        encoding="utf-8",
    )
    audit_path.write_text(
        _format_audit_markdown(audio_path, report, receipt_path),
        encoding="utf-8",
    )

    artifact_paths = {
        "final": final_path,
        "audit": audit_path,
        "audit_json": audit_json_path,
        "review_json": review_json_path,
        "primary_json": primary_json_path,
        "secondary_json": secondary_json_path,
    }
    receipt_path.write_text(
        json.dumps(
            _strict_bundle_receipt(artifact_paths, report),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    evidence_status, evidence_failures = validate_strict_artifact_bundle(
        {
            **artifact_paths,
            "receipt": receipt_path,
        },
        expected_primary_engine=primary_engine,
        expected_secondary_engine=secondary_engine,
    )
    return {
        **artifact_paths,
        "receipt": receipt_path,
        "evidence_status": evidence_status,
        "evidence_failures": evidence_failures,
    }


def _format_final_markdown(
    audio_path: Path,
    report: AuditReport,
    receipt_path: Path,
) -> str:
    return "\n".join(
        [
            f"# {audio_path.stem} Strict Transcript",
            "",
            f"- Audio: `{audio_path}`",
            f"- Primary: `{report.primary_engine}`",
            f"- Secondary: `{report.secondary_engine}`",
            f"- Status: `{report.status}`",
            f"- Evidence status: `{report.evidence_status}`",
            f"- Bundle receipt: `{receipt_path.name}`",
            f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
            "",
            "## Transcript",
            "",
            report.final_text,
            "",
        ]
    )


def _format_audit_markdown(
    audio_path: Path,
    report: AuditReport,
    receipt_path: Path,
) -> str:
    alternatives = "\n".join(f"- {text}" for text in report.alternatives) or "- None"
    flags = ", ".join(report.flags) if report.flags else "none"
    rule_hits = "\n".join(
        f"- `{hit.id}` ({hit.severity}): {hit.message} Evidence: `{hit.evidence}`"
        for hit in report.rule_hits
    ) or "- None"
    evidence = "\n".join(
        (
            f"- `{item.role}` → `{item.engine}`; raw: "
            f"`{item.raw_result_reference or 'not_recorded'}`; "
            f"segments: `{len(item.segments)}`"
        )
        for item in report.engine_evidence
    ) or "- None"
    disagreements = "\n".join(
        (
            f"- `{item.id}` ({item.scope}, similarity={item.similarity:.3f}, "
            f"review={str(item.review_required).lower()}): "
            f"`{item.primary_text or '[missing]'}` ↔ "
            f"`{item.secondary_text or '[missing]'}`"
        )
        for item in report.disagreements
    ) or "- None"
    review_items = "\n".join(
        (
            f"- `{item.id}` ({item.kind}): {item.reason}; "
            f"audio=`{item.audio_start_ms}`–`{item.audio_end_ms}`"
        )
        for item in report.review_items
    ) or "- None"
    return "\n".join(
        [
            f"# {audio_path.stem} Strict Audit",
            "",
            f"- Audio: `{audio_path}`",
            f"- Status: `{report.status}`",
            f"- Evidence status: `{report.evidence_status}`",
            f"- Needs review: `{str(report.needs_review).lower()}`",
            f"- Similarity: `{report.similarity:.3f}`",
            f"- Flags: `{flags}`",
            f"- Bundle receipt: `{receipt_path.name}`",
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
            "## Selection Policy",
            "",
            f"`{report.selection_policy}`",
            "",
            "## Engine Evidence",
            "",
            evidence,
            "",
            "## Disagreements",
            "",
            disagreements,
            "",
            "## Review Queue",
            "",
            review_items,
            "",
        ]
    )


def _review_payload(
    audio_path: Path,
    report: AuditReport,
    receipt_path: Path,
) -> dict[str, Any]:
    report_dict = asdict(report)
    return {
        "schema_version": report.schema_version,
        "audio": str(audio_path),
        "status": report.status,
        "evidence_status": report.evidence_status,
        "evidence_status_rationale": report.evidence_status_rationale,
        "final_text": report.final_text,
        "primary_text": report.primary_text,
        "secondary_text": report.secondary_text,
        "needs_review": report.needs_review,
        "selection_policy": report.selection_policy,
        "bundle_receipt_reference": receipt_path.name,
        "engine_evidence": report_dict["engine_evidence"],
        "disagreements": report_dict["disagreements"],
        "review_items": report_dict["review_items"],
    }


def _strict_bundle_receipt(
    artifact_paths: Mapping[str, Path],
    report: AuditReport,
) -> dict[str, Any]:
    artifacts = {
        key: {
            "path": artifact_paths[key].name,
            "size_bytes": artifact_paths[key].stat().st_size,
            "sha256": file_sha256(artifact_paths[key]),
        }
        for key in STRICT_BUNDLE_ARTIFACT_KEYS
    }
    raw_artifacts = ("primary_json", "secondary_json")
    engine_claims = [
        {
            "role": item.role,
            "engine": item.engine,
            "text_sha256": text_sha256(item.text),
            "raw_artifact": raw_key,
            "raw_sha256": artifacts[raw_key]["sha256"],
        }
        for item, raw_key in zip(report.engine_evidence, raw_artifacts)
    ]
    claims = {
        "status": report.status,
        "evidence_status": report.evidence_status,
        "final_text_sha256": text_sha256(report.final_text),
        "engine_evidence": engine_claims,
    }
    bundle_payload = {
        "schema_version": STRICT_BUNDLE_RECEIPT_SCHEMA_VERSION,
        "artifacts": artifacts,
        "claims": claims,
    }
    return {
        **bundle_payload,
        "bundle_sha256": canonical_json_sha256(bundle_payload),
    }

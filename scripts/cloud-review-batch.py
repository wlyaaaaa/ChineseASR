"""Backfill existing difficult local ASR jobs through the sole cloud entry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zh_asr.cloud_review import (CloudReviewError, auto_cloud_status,
    compare_text, load_cloud_config, local_review_signals, pause_message)


def _write_sidecar(out_dir: str, payload: dict) -> None:
    target = Path(out_dir) / "cloud.review.json"
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps({"schema": "zh_asr.cloud_review.v1",
        "local_text_rewritten": False, **payload}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    temporary.replace(target)


def _skip_group(group: dict, reason: str) -> None:
    for run in group["local_runs"]:
        _write_sidecar(run["out_dir"], {"status": "skipped",
            "error_code": "auto_cloud_paused", "pause_reason": reason,
            "message": pause_message(reason),
            "source_audio_sha256": group["audio_sha256"],
            "cloud_upload_performed": False})


def _link_result(group: dict, result_path: Path) -> bool:
    try:
        retained = json.loads(result_path.read_text(encoding="utf-8"))
        if (retained.get("status") != "succeeded" or
            retained.get("source_audio_sha256") != group["audio_sha256"] or
            retained.get("selected_channel") != group["channel_index"] or
            (group["important"] and retained.get("purpose") != "important_evidence") or
            not isinstance(retained.get("text"), str)):
            return False
    except (OSError, ValueError, AttributeError):
        return False
    for run in group["local_runs"]:
        reasons, local_text = local_review_signals(Path(run["out_dir"]),
            evidence_status=run["evidence_status"], important=run["important"])
        _write_sidecar(run["out_dir"], {"status": "succeeded",
            "model": retained.get("model"), "review_reasons": reasons,
            "purpose": retained.get("purpose"),
            "cloud_job_id": retained.get("job_id"),
            "cloud_result_path": str(result_path),
            "source_audio_sha256": group["audio_sha256"],
            "selected_channel": group["channel_index"],
            "local_text": local_text, "cloud_text": retained["text"],
            "disagreement": compare_text(local_text, retained["text"]) if local_text else [],
            "cloud_upload_performed": True,
            "reused_cloud_result": run["job_id"] != group["job_id"]})
    return True


def candidates(jobs_file: Path) -> list[dict]:
    data = json.loads(jobs_file.read_text(encoding="utf-8"))
    if data.get("schema") != "zh_asr.jobs.v1" or not isinstance(data.get("jobs"), list):
        raise ValueError("jobs_schema_invalid")
    groups: dict[tuple[str, int | None], dict] = {}
    retained: dict[tuple[str, int | None], tuple[str, str]] = {}
    ordered = sorted(data["jobs"], key=lambda item: (
        item.get("request", {}).get("important") is True,
        item.get("finished_at") or 0), reverse=True)
    for job in ordered:
        request = job.get("request", {})
        important = request.get("important") is True
        if job.get("status") != "succeeded" or (request.get("mode") == "quick" and not important):
            continue
        out_dir = Path(str(job.get("out_dir") or ""))
        audio = Path(str(request.get("audio") or ""))
        if not out_dir.is_dir() or not audio.is_file():
            continue
        audio_hash = str(request.get("audio_sha256") or "")
        key = ((audio_hash or str(audio)).casefold(), request.get("channel_index"))
        sidecar = out_dir / "cloud.review.json"
        if sidecar.is_file():
            try:
                current = json.loads(sidecar.read_text(encoding="utf-8"))
                result_path = Path(str(current.get("cloud_result_path") or ""))
                if current.get("status") == "succeeded" and result_path.is_file():
                    cloud = json.loads(result_path.read_text(encoding="utf-8"))
                    if (cloud.get("status") == "succeeded" and
                        cloud.get("source_audio_sha256") == audio_hash and
                        cloud.get("selected_channel") == request.get("channel_index") and
                        (not important or cloud.get("purpose") == "important_evidence")):
                        retained[key] = (str(result_path), str(cloud.get("purpose") or ""))
                        continue
            except (OSError, ValueError, AttributeError):
                pass
        reasons, _ = local_review_signals(out_dir,
            evidence_status=str(job.get("evidence_status") or ""), important=important)
        if not reasons:
            continue
        run = {"job_id": job.get("job_id"), "out_dir": str(out_dir),
               "evidence_status": str(job.get("evidence_status") or ""),
               "important": important, "reasons": reasons}
        if key in groups:
            groups[key]["local_runs"].append(run)
        else:
            groups[key] = {"job_id": run["job_id"], "audio": str(audio),
                "out_dir": str(out_dir), "evidence_status": run["evidence_status"],
                "important": important, "audio_sha256": audio_hash,
                "channel_index": request.get("channel_index"), "reasons": reasons,
                "local_runs": [run], "existing_result_path": ""}
    for key, group in groups.items():
        existing = retained.get(key)
        if existing and (not group["important"] or existing[1] == "important_evidence"):
            group["existing_result_path"] = existing[0]
    return list(groups.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, default=ROOT / "outputs" / "api" / "jobs.json")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runtime-principal", choices=("Codex", "Claude"), default="Codex",
                        help="real caller checked by the Secret Broker; Claude sessions pass Claude")
    args = parser.parse_args(argv)
    try:
        selected = candidates(args.jobs)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error_code": str(exc)}))
        return 2
    if args.max_files > 0:
        selected = selected[:args.max_files]
    if args.dry_run:
        print(json.dumps({"status": "preview", "count": len(selected),
                          "jobs": selected}, ensure_ascii=False))
        return 0
    try:
        config = load_cloud_config(ROOT / "configs" / "models.yaml")
    except CloudReviewError as exc:
        print(json.dumps({"status": "blocked", "error_code": exc.code}))
        return 2
    state_path = ROOT / "outputs" / "cloud-jobs" / "auto-cloud-state.json"
    state = auto_cloud_status(state_path, config)
    if state["status"] == "paused":
        reused = 0
        for group in selected:
            if group["existing_result_path"] and _link_result(group, Path(group["existing_result_path"])):
                reused += 1
            else:
                _skip_group(group, state["reason"])
        print(json.dumps({"status": "stopped", "selected": len(selected),
            "attempted": 0, "reused": reused,
            "pause_reason": state["reason"]}, ensure_ascii=False))
        return 3
    results = []
    stopped = False
    for position, group in enumerate(selected):
        existing = group["existing_result_path"]
        if existing:
            linked = _link_result(group, Path(existing))
            results.append({"job_id": group["job_id"], "status": "reused" if linked else "failed",
                "error_code": "" if linked else "cloud_result_invalid", "cloud_result_path": existing})
            continue
        command = ["pwsh", "-NoProfile", "-NonInteractive", "-File",
            str(ROOT / "scripts" / "asr-professional-cloud.ps1"),
            "-Audio", group["audio"], "-Important" if group["important"] else "-QualityReview",
            "-AutomaticReview", "-LocalOutDir", group["out_dir"],
            "-EvidenceStatus", group["evidence_status"],
            "-RuntimePrincipal", args.runtime_principal, "-Json"]
        if group["channel_index"] is not None:
            command.extend(["-ChannelIndex", str(group["channel_index"])])
        try:
            done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=86500,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            receipt = json.loads(done.stdout)
            if not isinstance(receipt, dict):
                raise ValueError("cloud_receipt_invalid")
        except (OSError, ValueError, subprocess.TimeoutExpired):
            receipt = {"status": "failed", "error_code": "cloud_receipt_invalid"}
        result_path = str(receipt.get("result_path") or "")
        success = receipt.get("status") == "succeeded" and bool(result_path)
        if success:
            success = _link_result(group, Path(result_path))
        if not success:
            for run in group["local_runs"]:
                existing_sidecar = Path(run["out_dir"]) / "cloud.review.json"
                if existing_sidecar.is_file():
                    try:
                        existing_data = json.loads(existing_sidecar.read_text(encoding="utf-8"))
                        if (existing_data.get("schema") == "zh_asr.cloud_review.v1" and
                                existing_data.get("status") in {"failed", "blocked"} and
                                existing_data.get("error_code")):
                            continue
                    except (OSError, ValueError, AttributeError):
                        pass
                _write_sidecar(run["out_dir"], {"status": "failed",
                    "error_code": str(receipt.get("error_code") or "cloud_result_invalid"),
                    "cloud_result_path": result_path,
                    "source_audio_sha256": group["audio_sha256"],
                    "cloud_upload_performed": bool(receipt.get("cloud_upload_performed"))})
        results.append({"job_id": group["job_id"],
            "local_runs": len(group["local_runs"]),
            "status": "succeeded" if success else "failed",
            "error_code": "" if success else str(receipt.get("error_code") or "cloud_result_invalid"),
            "model": receipt.get("model"), "cloud_result_path": result_path})
        state = auto_cloud_status(state_path, config)
        if state["status"] == "paused":
            for pending in selected[position + 1:]:
                if not (pending["existing_result_path"] and _link_result(
                        pending, Path(pending["existing_result_path"]))):
                    _skip_group(pending, state["reason"])
            stopped = True
            break
    status = ("stopped" if stopped else "completed" if all(
        item["status"] in {"succeeded", "reused"} for item in results)
        else "completed_with_failures")
    print(json.dumps({"status": status, "selected": len(selected),
        "attempted": len(results), "results": results}, ensure_ascii=False))
    return 0 if status == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())

"""Backfill existing difficult local ASR jobs through the sole cloud entry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
CLOUD_RESULTS_ROOT = ROOT / "outputs" / "cloud-jobs"
SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
RESULT_SCHEMAS = {"important_evidence": "chineseasr.qwen-audio3-important-result.v1",
                  "quality_review": "chineseasr.qwen-audio3-quality-review-result.v1"}
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


def _successful_result(result_path: Path) -> dict | None:
    try:
        retained = json.loads(result_path.read_text(encoding="utf-8"))
        purpose = retained.get("purpose")
        if (not isinstance(retained, dict) or
            retained.get("schema") != RESULT_SCHEMAS.get(purpose) or
            retained.get("status") != "succeeded" or
            retained.get("credential_result") != "Success" or
            retained.get("cloud_upload_performed") is not True or
            not isinstance(retained.get("text"), str) or not retained["text"].strip() or
            result_path.name != str(retained.get("job_id")) + ".result.json"):
            return None
        return retained
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def _result_key(audio_hash: str, channel: int | None, purpose: str) -> tuple[str, int | None, str]:
    return audio_hash.casefold(), channel, purpose


def _retained_result_index() -> dict[tuple[str, int | None, str], Path]:
    matches: dict[tuple[str, int | None, str], Path] = {}
    try:
        paths = list(CLOUD_RESULTS_ROOT.glob("*.result.json"))
    except OSError:
        return matches
    dated_paths = []
    for path in paths:
        try:
            dated_paths.append((path.stat().st_mtime_ns, path))
        except OSError:
            continue
    for _, path in sorted(dated_paths):
        retained = _successful_result(path)
        if retained:
            matches[_result_key(str(retained.get("source_audio_sha256") or ""),
                retained.get("selected_channel"), retained["purpose"])] = path
    return matches


def _link_result(group: dict, result_path: Path, *, reused: bool = False) -> bool:
    retained = _successful_result(result_path)
    if (retained is None or
        _result_key(str(retained.get("source_audio_sha256") or ""),
                    retained.get("selected_channel"), retained["purpose"]) !=
        _result_key(group["audio_sha256"], group["channel_index"],
                    "important_evidence" if group["important"] else "quality_review")):
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
            "reused_cloud_result": reused or run["job_id"] != group["job_id"]})
    return True


def _last_json_object(stdout: str) -> dict:
    """Read the last complete object from concatenated process receipts."""
    decoder = json.JSONDecoder()
    position = 0
    last: dict | None = None
    while (start := stdout.find("{", position)) >= 0:
        try:
            value, end = decoder.raw_decode(stdout, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        if isinstance(value, dict):
            last = value
        position = end
    if last is None:
        raise ValueError("cloud_receipt_invalid")
    return last


def _matching_source(path: Path, expected_hash: str, hashes: dict[Path, str]) -> bool:
    if not path.is_file():
        return False
    if not SHA256.fullmatch(expected_hash):
        return True  # Older job snapshots did not always retain a source digest.
    if path not in hashes:
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[path] = digest.hexdigest()
        except OSError:
            return False
    return hashes[path].casefold() == expected_hash.casefold()


def _recover_source(audio: Path, expected_hash: str, roots: list[Path],
                    hashes: dict[Path, str]) -> Path | None:
    if not SHA256.fullmatch(expected_hash) or not audio.name:
        return None
    for root in roots:
        if not root.is_dir():
            continue
        try:
            matches = root.rglob(audio.name)
            for candidate in matches:
                if _matching_source(candidate, expected_hash, hashes):
                    return candidate.resolve()
        except OSError:
            continue
    return None


def candidates(jobs_file: Path, *, recovery_roots: list[Path] | None = None,
               missing_sources: list[dict] | None = None) -> list[dict]:
    data = json.loads(jobs_file.read_text(encoding="utf-8"))
    if data.get("schema") != "zh_asr.jobs.v1" or not isinstance(data.get("jobs"), list):
        raise ValueError("jobs_schema_invalid")
    groups: dict[tuple[str, int | None, str], dict] = {}
    retained = _retained_result_index()
    hashes: dict[Path, str] = {}
    roots = recovery_roots or []
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
        if not out_dir.is_dir():
            continue
        audio_hash = str(request.get("audio_sha256") or "")
        purpose = "important_evidence" if important else "quality_review"
        key = _result_key(audio_hash or str(audio), request.get("channel_index"), purpose)
        sidecar = out_dir / "cloud.review.json"
        if sidecar.is_file():
            try:
                current = json.loads(sidecar.read_text(encoding="utf-8"))
                result_path = Path(str(current.get("cloud_result_path") or ""))
                cloud = _successful_result(result_path) if result_path.is_file() else None
                if (current.get("status") == "succeeded" and cloud and
                    _result_key(str(cloud.get("source_audio_sha256") or ""),
                        cloud.get("selected_channel"), cloud["purpose"]) == key):
                    continue
            except (OSError, ValueError, AttributeError):
                pass
        reasons, _ = local_review_signals(out_dir,
            evidence_status=str(job.get("evidence_status") or ""), important=important)
        if not reasons:
            continue
        recovered = False
        original_audio = str(audio)
        existing = retained.get(key)
        if not existing and not _matching_source(audio, audio_hash, hashes):
            replacement = _recover_source(audio, audio_hash, roots, hashes)
            if replacement is None:
                if missing_sources is not None:
                    missing_sources.append({"job_id": job.get("job_id"),
                        "out_dir": str(out_dir), "original_audio": original_audio,
                        "audio_sha256": audio_hash,
                        "reason": "source_missing_or_hash_mismatch"})
                continue
            audio = replacement
            recovered = True
        run = {"job_id": job.get("job_id"), "out_dir": str(out_dir),
               "evidence_status": str(job.get("evidence_status") or ""),
               "important": important, "reasons": reasons}
        if key in groups:
            groups[key]["local_runs"].append(run)
        else:
            groups[key] = {"job_id": run["job_id"], "audio": str(audio),
                "original_audio": original_audio, "source_recovered": recovered,
                "out_dir": str(out_dir), "evidence_status": run["evidence_status"],
                "important": important, "audio_sha256": audio_hash,
                "channel_index": request.get("channel_index"), "reasons": reasons,
                "local_runs": [run], "existing_result_path": str(existing or "")}
    return list(groups.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, default=ROOT / "outputs" / "api" / "jobs.json")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recover-root", type=Path, action="append",
                        help="search matching filenames here; accept only an exact SHA-256 match")
    parser.add_argument("--runtime-principal", choices=("Codex", "Claude"), default="Codex",
                        help="real caller checked by the Secret Broker; Claude sessions pass Claude")
    args = parser.parse_args(argv)
    try:
        missing_sources: list[dict] = []
        recovery_roots = args.recover_root if args.recover_root is not None else [Path("E:/Music")]
        selected = candidates(args.jobs, recovery_roots=recovery_roots,
                              missing_sources=missing_sources)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error_code": str(exc)}))
        return 2
    if args.max_files > 0:
        selected = selected[:args.max_files]
    if args.dry_run:
        print(json.dumps({"status": "preview", "count": len(selected),
                          "jobs": selected, "missing_source_count": len(missing_sources),
                          "missing_source_jobs": missing_sources,
                          "recovered_source_count": sum(group["source_recovered"] for group in selected)},
                          ensure_ascii=False))
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
            if group["existing_result_path"] and _link_result(group, Path(group["existing_result_path"]), reused=True):
                reused += 1
            else:
                _skip_group(group, state["reason"])
        print(json.dumps({"status": "stopped", "selected": len(selected),
            "attempted": 0, "reused": reused,
            "missing_source_count": len(missing_sources),
            "missing_source_jobs": missing_sources,
            "pause_reason": state["reason"]}, ensure_ascii=False))
        return 3
    results = []
    stopped = False
    for position, group in enumerate(selected):
        existing = group["existing_result_path"]
        if existing:
            linked = _link_result(group, Path(existing), reused=True)
            if linked:
                results.append({"job_id": group["job_id"], "status": "reused",
                    "error_code": "", "cloud_result_path": existing})
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
            receipt = _last_json_object(done.stdout)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            receipt = {"status": "failed", "error_code": "cloud_receipt_invalid"}
        result_path = str(receipt.get("result_path") or "")
        success = receipt.get("status") == "succeeded" and bool(result_path)
        if success:
            success = _link_result(group, Path(result_path))
        recovered_result = False
        if not success:
            key = _result_key(group["audio_sha256"], group["channel_index"],
                "important_evidence" if group["important"] else "quality_review")
            retained_path = _retained_result_index().get(key)
            if retained_path and _link_result(group, retained_path, reused=True):
                success, recovered_result, result_path = True, True, str(retained_path)
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
            "status": "reused" if recovered_result else "succeeded" if success else "failed",
            "error_code": "" if success else str(receipt.get("error_code") or "cloud_result_invalid"),
            "model": receipt.get("model"), "cloud_result_path": result_path})
        state = auto_cloud_status(state_path, config)
        if state["status"] == "paused":
            for pending in selected[position + 1:]:
                if not (pending["existing_result_path"] and _link_result(
                        pending, Path(pending["existing_result_path"]), reused=True)):
                    _skip_group(pending, state["reason"])
            stopped = True
            break
    status = ("stopped" if stopped else "completed" if all(
        item["status"] in {"succeeded", "reused"} for item in results)
        else "completed_with_failures")
    if status == "completed" and missing_sources:
        status = "completed_with_missing_sources"
    print(json.dumps({"status": status, "selected": len(selected),
        "attempted": len(results), "results": results,
        "missing_source_count": len(missing_sources),
        "missing_source_jobs": missing_sources}, ensure_ascii=False))
    return 0 if status == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())

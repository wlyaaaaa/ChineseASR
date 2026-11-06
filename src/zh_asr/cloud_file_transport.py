"""Send a prepared file job at chunk boundaries; credentials stay in the broker."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

from .cloud_review import CloudReviewError, _chunk_completed, _read_json, _write_state
from .process_control import managed_popen_kwargs, terminate_process_tree


def request_digest(request: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "retry_uncertain"},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _broker(request_path: Path, target: str, broker: Path) -> str:
    process = subprocess.Popen(["pwsh", "-NoProfile", "-NonInteractive", "-File", str(broker),
        "-Action", "AgentSecretRef", "-Query", target, "-SecretRefRequestPath", str(request_path), "-Json"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, **managed_popen_kwargs())
    try:
        raw, _ = process.communicate(timeout=240)
    except subprocess.TimeoutExpired:
        terminate_process_tree(process)
        return "cloud_chunk_timeout"
    try:
        receipt = json.loads(raw.decode("utf-8-sig"))
        return str(receipt.get("error") or "") if process.returncode else ""
    except (ValueError, UnicodeError):
        return "cloud_broker_receipt_invalid" if process.returncode else ""


def transmit_cloud_chunks(request_path: Path, intent_path: Path, root: Path, broker: Path, *,
                          target: str, progress_path: Path | None = None, access=None, send=None) -> dict:
    """Reuse completed operations; never repeat an uncertain operation automatically."""
    from .file_job import _access_open
    access, send = access or _access_open, send or _broker
    if not isinstance(target, str) or not target:
        raise CloudReviewError("cloud_credential_route_missing")
    root = root.resolve()
    request_path, intent_path = request_path.resolve(), intent_path.resolve()
    if not request_path.is_relative_to(root) or not intent_path.is_relative_to(root):
        raise CloudReviewError("request_path_escape")
    request = _read_json(request_path) or {}
    intent = _read_json(intent_path) or {}
    full_digest = request_digest(request)
    if full_digest != intent.get("vendor_request_sha256"):
        raise CloudReviewError("vendor_request_binding_invalid")
    job_id = intent["job_id"]
    work = root / (job_id + ".work")
    work.mkdir(parents=True, exist_ok=True)
    result = dict(schema="passwordcenter.vendor-result.v1", vendor="qwen", request_sha256=full_digest,
                  status="failed", error_code="", credential_result="Scope-Error", results=[],
                  new_requests=0, reused_requests=0, secret_returned=False)
    result_path = root / (job_id + ".provider.json")

    def progress(status="running"):
        if progress_path:
            if not progress_path.resolve().is_relative_to(root.parent / "api" / "file-jobs"):
                raise CloudReviewError("progress_path_escape")
            _write_state(progress_path, dict(updated_at=time.time(), job_id=job_id, status=status,
                completed_chunks=len(result["results"]), source_audio_sha256=intent.get("source_audio_sha256")))
        _write_state(result_path, result)

    progress()
    for operation in request["operations"]:
        if not access():
            result.update(status="blocked", error_code="personal_access_protection_required")
            progress("waiting_access")
            return result
        ident = operation["id"]
        if not isinstance(ident, str) or not ident.isdecimal():
            raise CloudReviewError("operation_id_invalid")
        part_path = work / ("part-" + ident + ".request.json")
        provider_path = work / ("part-" + ident + ".provider.json")
        part = {**request, "operations": [operation], "result_path": str(provider_path)}
        part_digest = request_digest(part)
        provider = _read_json(provider_path) or {}
        completed = (provider.get("request_sha256") == part_digest and provider.get("vendor") == "qwen"
                     and provider.get("status") == "succeeded" and len(provider.get("results", [])) == 1
                     and provider["results"][0].get("id") == ident and _chunk_completed(provider["results"][0]))
        if completed:
            result["reused_requests"] += 1
        else:
            checkpoint = _read_json(provider_path.with_name(provider_path.name + ".checkpoints") / (ident + ".json")) or {}
            if checkpoint.get("status") in {"in_flight", "outcome_unknown"}:
                result.update(status="blocked", error_code="previous_outcome_unknown")
                progress("failed")
                return result
            _write_state(part_path, part)
            try:
                error = send(part_path, target, broker)
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                error = "cloud_chunk_" + type(exc).__name__
            provider = _read_json(provider_path) or {}
            if provider.get("request_sha256") != part_digest or provider.get("vendor") != "qwen":
                result.update(status="failed", error_code=error or "cloud_worker_result_missing")
                progress("failed")
                return result
            result["new_requests"] += provider.get("new_requests", 0)
            result["reused_requests"] += provider.get("reused_requests", 0)
            if (provider.get("status") != "succeeded" or len(provider.get("results", [])) != 1
                    or provider["results"][0].get("id") != ident or not _chunk_completed(provider["results"][0])):
                result.update({key: provider[key] for key in ("status", "error_code", "credential_result",
                    "provider_error_code", "provider_error_message", "first_failure", "transport_diagnostic") if key in provider})
                result["error_code"] = result["error_code"] or error or "cloud_chunk_incomplete"
                progress("failed")
                return result
        result["results"].append(provider["results"][0])
        progress()
    result.update(status="succeeded", credential_result="Success", error_code="")
    progress("succeeded")
    return result

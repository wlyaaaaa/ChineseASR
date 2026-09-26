"""Cloud review routing, local audit signals and provider-error pause state.

This module neither counts provider credit nor sends audio. The cloud worker
keeps a visible job record for every attempted upload.
"""
from __future__ import annotations

from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

from .text_comparison import normalize_comparison, strip_audit_markers


class CloudReviewError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


_APIS = {"http_base64", "websocket_pcm", "legacy_http_base64"}
_ENDPOINTS = {
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
    "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
}
STATE_SCHEMA = "zh_asr.auto_cloud_state.v1"


def load_cloud_config(path: Path) -> dict[str, Any]:
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = data["cloud_review"]
        models, routes = config["models"], config["routes"]
        if config["region"] != "cn-beijing" or not isinstance(models, dict):
            raise ValueError("invalid region or models")
        if not isinstance(config["credit_cycle"], str) or not config["credit_cycle"].strip():
            raise ValueError("credit_cycle must identify the current grant")
        for name, model in models.items():
            if not name or not isinstance(model, dict) or model["api"] not in _APIS:
                raise ValueError("invalid cloud model")
            if not isinstance(model["id"], str) or not model["id"].strip():
                raise ValueError("model id missing")
            if model.get("endpoint") not in _ENDPOINTS:
                raise ValueError("unapproved endpoint")
            if not (1 <= int(model["max_chunk_sec"]) <= 180):
                raise ValueError("invalid chunk length")
        if any(value not in models for value in routes.values()):
            raise ValueError("route references unknown model")
        if not {"short", "long", "speaker", "dialect", "hotwords"} <= routes.keys():
            raise ValueError("missing cloud route")
        return config
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise CloudReviewError("cloud_config_invalid") from exc


def select_model(config: dict[str, Any], *, duration_sec: float,
                 hotwords: dict[str, int] | None = None, speaker: bool = False,
                 dialect: bool = False) -> str:
    route = "speaker" if speaker else "dialect" if dialect else (
        "hotwords" if hotwords else "long" if duration_sec > 180 else "short"
    )
    return str(config["routes"][route])


def config_resume_signature(config: dict[str, Any]) -> str:
    """A new model or a newly granted credit cycle reopens a provider pause."""
    selected = {"credit_cycle": config["credit_cycle"],
                "model_ids": sorted({config["models"][profile]["id"]
                                     for profile in config["routes"].values()})}
    data = json.dumps(selected, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, UnicodeError):
        return None


def auto_cloud_status(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {"status": "running", "reason": ""}
    state = _read_json(path)
    if state is None or state.get("schema") != STATE_SCHEMA:
        return {"status": "paused", "reason": "cloud_state_unreadable"}
    if state.get("status") == "running":
        return {"status": "running", "reason": ""}
    if state.get("status") != "paused" or not state.get("reason"):
        return {"status": "paused", "reason": "cloud_state_unreadable"}
    if state.get("config_signature") != config_resume_signature(config):
        return {"status": "running", "reason": "resumed_by_new_model_or_credit_cycle"}
    return {"status": "paused", "reason": str(state["reason"]),
            "provider_error": str(state.get("provider_error") or ""),
            "model": str(state.get("model") or "")}


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def pause_cloud(path: Path, config: dict[str, Any], *, reason: str,
                provider_error: str, model: str) -> None:
    _write_state(path, {"schema": STATE_SCHEMA, "status": "paused", "reason": reason,
        "provider_error": provider_error, "model": model,
        "config_signature": config_resume_signature(config),
        "updated_utc": datetime.now(timezone.utc).isoformat()})


def resume_cloud(path: Path, config: dict[str, Any]) -> None:
    _write_state(path, {"schema": STATE_SCHEMA, "status": "running",
        "reason": "owner_requested_resume", "config_signature": config_resume_signature(config),
        "updated_utc": datetime.now(timezone.utc).isoformat()})


def stopping_error(code: str, credential_result: str = "",
                   provider_message: str = "") -> str | None:
    """Pause only billing, permission and retired/unavailable model failures."""
    normalized = "".join(char for char in code.casefold() if char.isalnum())
    detail = "".join(char for char in provider_message.casefold() if char.isalnum())
    combined = normalized + detail
    if ("freeallocatedquotaexceeded" in detail or
            any(part in combined for part in ("freequota", "freeexhaust", "freetieronly"))):
        return "free_quota_exhausted"
    if any(part in combined for part in ("arrear", "overdue", "outstandingbill", "unpaidfee")):
        return "account_arrears"
    if any(part in combined for part in ("insufficientbalance", "balanceinsufficient",
            "insufficientfund", "paymentrequired", "billingblocked", "budgetlimitexceeded")) or normalized == "http402":
        return "balance_insufficient"
    if any(part in combined for part in ("modelnotfound",
            "modelretired", "modeloffline", "modeldeprecated")) or normalized == "http404":
        return "model_unavailable"
    if ("commoditynotpurchased" in combined or "donothaveaccess" in combined or
            "permissiondenied" in combined or "accessdenied" in combined):
        return "access_denied"
    if credential_result == "Rate-Limited" or normalized.startswith("throttling") or normalized == "http429":
        return None
    if credential_result in {"Invalid", "Revoked", "Expired", "Permission-Denied"} or any(
            part in normalized for part in ("permissiondenied", "accessdenied",
                                       "unauthorized", "forbidden")) or normalized in {"http401", "http403"}:
        return "access_denied"
    return None


def local_review_signals(out_dir: Path, *, evidence_status: str = "",
                         important: bool = False, dialect: bool = False) -> tuple[list[str], str]:
    """Read existing audit evidence; unknown lexical truth is not a clean result."""
    reasons: set[str] = set()
    if important:
        reasons.add("important_recording")
    if dialect:
        reasons.add("dialect")
    if evidence_status == "provisional":
        reasons.add("provisional_evidence")
    review = _read_json(out_dir / "quality.review.json")
    if review and review.get("needs_review") is True:
        reasons.add("quality_needs_review")
    if review:
        for entry in review.get("entries", []):
            if isinstance(entry, dict) and entry.get("reasons"):
                reasons.add("local_review_entry")
    texts: list[str] = []
    audit_paths = sorted(out_dir.glob("*.strict.audit.json"))
    if not audit_paths:
        audit_paths = sorted(out_dir.glob("chunks/*/*.strict.audit.json"))
    for path in audit_paths:
        audit = _read_json(path)
        if not audit:
            continue
        final = str(audit.get("final_text") or "")
        if final:
            texts.append(final)
        if "[疑似]" in final:
            reasons.add("suspected_text")
        if audit.get("needs_review") is True:
            reasons.add("audit_needs_review")
        left, right = str(audit.get("primary_text") or ""), str(audit.get("secondary_text") or "")
        if left and right and left != right:
            reasons.add("engine_disagreement")
        if "engine_failure" in audit.get("flags", []):
            reasons.add("engine_failure")
    manifest = _read_json(out_dir / "manifest.json")
    if manifest and manifest.get("evidence_status") == "provisional":
        reasons.add("provisional_evidence")
    return sorted(reasons), "\n".join(texts)


def compare_text(local: str, cloud: str, *, limit: int = 120) -> list[dict[str, str]]:
    """Flag lexical disagreements without declaring either transcript correct."""
    local = normalize_comparison(strip_audit_markers(local))
    cloud = normalize_comparison(cloud)
    changes = []
    for operation, a, b, c, d in SequenceMatcher(
        a=local, b=cloud, autojunk=False
    ).get_opcodes():
        if operation != "equal":
            changes.append({"kind": operation, "local": local[a:b], "cloud": cloud[c:d],
                "local_context": local[max(0, a-12):min(len(local), b+12)],
                "cloud_context": cloud[max(0, c-12):min(len(cloud), d+12)]})
        if len(changes) >= limit:
            break
    return changes

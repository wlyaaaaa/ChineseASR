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
import wave

from .text_comparison import normalize_comparison, strip_audit_markers


class CloudReviewError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


_APIS = {"http", "websocket"}
STATE_SCHEMA = "zh_asr.auto_cloud_state.v1"


def _free_until(model: dict[str, Any]) -> datetime:
    value = model.get("free_until")
    if not isinstance(value, str):
        raise ValueError("free_until must be an ISO timestamp")
    cutoff = datetime.fromisoformat(value)
    if cutoff.tzinfo is None:
        raise ValueError("free_until needs a timezone")
    return cutoff


def free_period_expired(model: dict[str, Any], *, now: datetime | None = None) -> bool:
    """The configured Beijing-time cutoff is exclusive, with no usage estimate."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now needs a timezone")
    return current >= _free_until(model)


def all_automatic_models_expired(config: dict[str, Any], *,
                                 now: datetime | None = None) -> bool:
    profiles = set(config["routes"].values())
    return all(free_period_expired(config["models"][profile], now=now)
               for profile in profiles)


def pause_message(reason: str) -> str:
    if reason == "free_period_expired":
        return "免费期已到期，云端未跑"
    if reason == "free_quota_exhausted":
        return "云端报免费额度用完，自动上云已停止"
    if reason == "cloud_state_unreadable":
        return "云端停用状态不可读，云端未跑"
    return "自动上云已停止，云端未跑；原因：" + reason


def load_cloud_config(path: Path) -> dict[str, Any]:
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = data["cloud_review"]
        models, routes = config["models"], config["routes"]
        if config["region"] != "cn-beijing" or not isinstance(models, dict):
            raise ValueError("invalid region or models")
        for name in ("secret_ref_target", "credential_ref"):
            value = config.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("cloud credential routing missing")
        if not isinstance(config["credit_cycle"], str) or not config["credit_cycle"].strip():
            raise ValueError("credit_cycle must identify the current grant")
        for name, model in models.items():
            if not name or not isinstance(model, dict) or model["api"] not in _APIS:
                raise ValueError("invalid cloud model")
            if not isinstance(model["id"], str) or not model["id"].strip():
                raise ValueError("model id missing")
            _free_until(model)
            if not (1 <= int(model["max_chunk_sec"]) <= 180):
                raise ValueError("invalid chunk length")
            for capability in ("supports_speaker_diarization", "supports_keep_dialect",
                               "speaker_dialect_exclusive", "supports_vocabulary"):
                if capability in model and type(model[capability]) is not bool:
                    raise ValueError("invalid cloud model capability")
            if not isinstance(model.get("parameters", {}), dict):
                raise ValueError("invalid cloud model parameters")
            if model["api"] == "websocket":
                task = model.get("ws_task")
                if not isinstance(task, dict) or any(not isinstance(task.get(key), str)
                    for key in ("task_group", "task", "function")):
                    raise ValueError("invalid websocket task")
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
    """A new model, new cutoff or new grant reopens a provider pause."""
    selected = {"credit_cycle": config["credit_cycle"],
                "models": sorted({(config["models"][profile]["id"],
                                   config["models"][profile]["free_until"])
                                  for profile in config["routes"].values()})}
    data = json.dumps(selected, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, UnicodeError):
        return None


def auto_cloud_status(path: Path, config: dict[str, Any], *,
                      now: datetime | None = None) -> dict[str, Any]:
    if all_automatic_models_expired(config, now=now):
        return {"status": "paused", "reason": "free_period_expired",
                "message": pause_message("free_period_expired")}
    if not path.exists():
        return {"status": "running", "reason": ""}
    state = _read_json(path)
    if state is None or state.get("schema") != STATE_SCHEMA:
        return {"status": "paused", "reason": "cloud_state_unreadable",
                "message": pause_message("cloud_state_unreadable")}
    if state.get("status") == "running":
        return {"status": "running", "reason": ""}
    if state.get("status") != "paused" or not state.get("reason"):
        return {"status": "paused", "reason": "cloud_state_unreadable",
                "message": pause_message("cloud_state_unreadable")}
    if state.get("config_signature") != config_resume_signature(config):
        return {"status": "running", "reason": "resumed_by_new_model_or_free_until"}
    return {"status": "paused", "reason": str(state["reason"]),
            "message": pause_message(str(state["reason"])),
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


INTENT_SCHEMA = "zh_asr.cloud_review_intent.v1"
TRANSPORT_SCHEMA = "zh_asr.dashscope_transport_request.v1"
PROVIDER_SCHEMA = "zh_asr.dashscope_transport_result.v1"
IMPORTANT_RESULT_SCHEMA = "chineseasr.qwen-audio3-important-result.v1"
QUALITY_RESULT_SCHEMA = "chineseasr.qwen-audio3-quality-review-result.v1"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounded_intent(path: Path, root: Path) -> dict[str, Any]:
    try:
        if not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size > 1024 * 1024:
            raise ValueError("intent outside request root or too large")
        intent = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(intent, dict) or intent.get("schema") != INTENT_SCHEMA:
            raise ValueError("intent schema")
        job_id = str(uuid.UUID(str(intent["job_id"])))
        if path.name != job_id + ".intent.json":
            raise ValueError("intent job binding")
        if intent.get("cloud_upload_authorized") is not True:
            raise ValueError("cloud authorization missing")
        purpose = intent.get("purpose")
        if purpose not in {"important_evidence", "quality_review"}:
            raise ValueError("intent purpose")
        if purpose == "important_evidence" and intent.get("importance") != "important":
            raise ValueError("importance missing")
        if purpose == "quality_review" and "importance" in intent:
            raise ValueError("quality purpose conflict")
        audio = Path(str(intent["audio_path"]))
        if not audio.is_absolute() or not audio.is_file():
            raise ValueError("audio missing")
        local = intent.get("local_out_dir")
        if local and (not Path(str(local)).is_absolute() or not Path(str(local)).is_dir()):
            raise ValueError("local review missing")
        if intent.get("automatic_review") and not local:
            raise ValueError("automatic review needs local result")
        intent["job_id"] = job_id
        intent["audio_path"] = str(audio.resolve())
        return intent
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as exc:
        raise CloudReviewError("cloud_intent_invalid") from exc


def _split_prepared_wav(source: Path, output_dir: Path, *,
                        chunk_sec: int, overlap_sec: int) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as reader:
        if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (16000, 1, 2):
            raise CloudReviewError("prepared_audio_format_invalid")
        total = reader.getnframes()
        if total == 0:
            raise CloudReviewError("audio_empty")
        step = 16000 * (chunk_sec - overlap_sec)
        frames_per_chunk = 16000 * chunk_sec
        chunks = []
        start = 0
        while start < total:
            end = min(start + frames_per_chunk, total)
            reader.setpos(start)
            path = output_dir / f"chunk-{len(chunks)+1:06d}.wav"
            with wave.open(str(path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(reader.readframes(end - start))
            chunks.append({"index": len(chunks) + 1, "start_ms": round(start / 16),
                "end_ms": round(end / 16), "path": str(path.resolve()),
                "audio_sha256": _file_hash(path)})
            if end >= total:
                break
            start += step
    return chunks


def _local_sidecar(intent: dict[str, Any], result: dict[str, Any]) -> None:
    local = intent.get("local_out_dir")
    if not local:
        return
    _, local_text = local_review_signals(Path(local),
        evidence_status=str(intent.get("evidence_status") or ""),
        important=intent["purpose"] == "important_evidence",
        dialect=bool(intent.get("keep_dialect")))
    cloud_text = str(result.get("text") or "")
    payload = {"schema": "zh_asr.cloud_review.v1", "status": result["status"],
        "model": result.get("model"), "purpose": result.get("purpose"),
        "cloud_job_id": intent["job_id"], "cloud_result_path": result.get("result_path", ""),
        "source_audio_sha256": result.get("source_audio_sha256"),
        "selected_channel": intent.get("channel_index"),
        "local_text": local_text, "cloud_text": cloud_text,
        "disagreement": compare_text(local_text, cloud_text) if local_text and cloud_text else [],
        "error_code": result.get("error_code", ""),
        "pause_reason": result.get("pause_reason", ""),
        "message": result.get("message", ""),
        "billing_warning": result.get("billing_warning", ""),
        "cloud_upload_performed": result.get("cloud_upload_performed", False),
        "local_text_rewritten": False}
    _write_state(Path(local) / "cloud.review.json", payload)


def prepare_cloud_request(intent_path: Path, root: Path, config_path: Path, *,
                          state_path: Path | None = None,
                          now: datetime | None = None) -> dict[str, Any]:
    """Prepare all audio and model policy before a SecretRef worker is invoked."""
    from .audio_frontend import prepare_pcm16_mono
    from .audio_quality import extract_channel

    root = root.resolve()
    intent = _bounded_intent(intent_path, root)
    config = load_cloud_config(config_path)
    automatic = intent.get("automatic_review") is True
    local = intent.get("local_out_dir")
    reasons, _ = local_review_signals(Path(local),
        evidence_status=str(intent.get("evidence_status") or ""),
        important=intent["purpose"] == "important_evidence",
        dialect=bool(intent.get("keep_dialect"))) if local else (
            (["important_recording"] if intent["purpose"] == "important_evidence" else []), "")
    if automatic and not reasons:
        return {"status": "blocked", "error_code": "not_a_difficult_recording"}
    state = auto_cloud_status(state_path or root / "auto-cloud-state.json", config, now=now)
    if automatic and state["status"] == "paused":
        response = {"status": "skipped", "error_code": "auto_cloud_paused",
                    "pause_reason": state["reason"], "message": state.get("message") or pause_message(state["reason"]),
                    "cloud_upload_performed": False}
        _local_sidecar(intent, response)
        return response
    source = Path(intent["audio_path"])
    source_hash = _file_hash(source)
    work = root / (intent["job_id"] + ".work")
    work.mkdir(parents=True, exist_ok=True)
    selected = source
    if intent.get("channel_index") is not None:
        selected, _ = extract_channel(source, int(intent["channel_index"]),
                                      work / "selected-channel")
    prepared = prepare_pcm16_mono(selected, work / "prepared", materialize_owner=True)
    duration_sec = prepared.duration_sec
    profile = str(intent.get("model_profile") or select_model(config,
        duration_sec=duration_sec, hotwords=intent.get("hotwords") or {},
        speaker=bool(intent.get("speaker_diarization")),
        dialect=bool(intent.get("keep_dialect"))))
    if profile not in config["models"]:
        raise CloudReviewError("model_profile_invalid")
    model = config["models"][profile]
    protocol = model["api"]
    speaker = bool(intent.get("speaker_diarization"))
    dialect = bool(intent.get("keep_dialect"))
    if speaker and (protocol != "http" or not model.get("supports_speaker_diarization", False)):
        raise CloudReviewError("speaker_model_required")
    if dialect and not model.get("supports_keep_dialect", False):
        raise CloudReviewError("dialect_model_required")
    if speaker and dialect and model.get("speaker_dialect_exclusive", False):
        raise CloudReviewError("speaker_and_dialect_conflict")
    expired = free_period_expired(model, now=now)
    if automatic and expired:
        response = {"status": "skipped", "error_code": "free_period_expired",
                    "pause_reason": "free_period_expired",
                    "message": pause_message("free_period_expired"),
                    "model": model["id"], "source_audio_sha256": source_hash,
                    "cloud_upload_performed": False}
        _local_sidecar(intent, response)
        return response
    billing_warning = ("免费期已到期，本次显式上传可能计费" if expired else "")
    try:
        chunk_sec = min(int(intent.get("chunk_sec", 180)), int(model["max_chunk_sec"]))
        overlap_sec = int(intent.get("overlap_sec", 1))
    except (ValueError, TypeError) as exc:
        raise CloudReviewError("chunk_policy_invalid") from exc
    if chunk_sec < 1 or overlap_sec < 0 or overlap_sec >= chunk_sec:
        raise CloudReviewError("chunk_policy_invalid")
    chunks = _split_prepared_wav(prepared.path, work / "chunks",
        chunk_sec=chunk_sec, overlap_sec=overlap_sec)
    hotwords = intent.get("hotwords") or {}
    if (not isinstance(hotwords, dict) or len(hotwords) > 2000 or
            any(not isinstance(word, str) or not word.strip() or
                type(weight) is not int or weight not in (1, 2, 3, 4, 5, 50)
                for word, weight in hotwords.items())):
        raise CloudReviewError("hotwords_invalid")
    if hotwords and not model.get("supports_vocabulary", False):
        raise CloudReviewError("vocabulary_model_required")
    parameters = dict(model.get("parameters") or {})
    parameters.update({"format": "wav" if protocol == "http" else "pcm",
                       "sample_rate": "16000" if protocol == "http" else 16000})
    if speaker:
        parameters["speaker_diarization_enabled"] = True
    if dialect:
        parameters["keep_dialect"] = True
    if hotwords:
        parameters["vocabulary"] = hotwords
    terms = "、".join(hotwords)
    context_message = {"role": "user", "content": [{"type": "input_text",
                       "text": "相关词语：" + terms}]}
    messages_prefix = [context_message] if hotwords and len(terms) <= 350 else []
    ws_input = {"context": [context_message]} if messages_prefix else {}
    request = {"schema": TRANSPORT_SCHEMA, "job_id": intent["job_id"],
        "purpose": intent["purpose"], "important_only": intent["purpose"] == "important_evidence",
        "cloud_upload_authorized": True, "automatic_review": automatic,
        "model": model["id"], "protocol": protocol,
        "parameters": parameters, "messages_prefix": messages_prefix,
        "input": ws_input, "ws_task": model.get("ws_task", {}), "chunks": chunks,
        "source_audio_path": str(source), "source_audio_sha256": source_hash,
        "source_audio_bytes": source.stat().st_size,
        "selected_channel": intent.get("channel_index"),
        "billing_warning": billing_warning,
        "send_before_utc": model["free_until"] if automatic else None,
        "created_utc": datetime.now(timezone.utc).isoformat()}
    pending = root / (intent["job_id"] + ".pending.json")
    if pending.exists():
        raise CloudReviewError("pending_request_exists")
    intent.update(model=model["id"], model_profile=profile, protocol=protocol,
        source_audio_sha256=source_hash, source_audio_bytes=source.stat().st_size,
        source_audio_seconds=duration_sec, billing_warning=billing_warning,
        review_reasons=reasons, prepared_audio_sha256=_file_hash(prepared.path),
        chunk_bindings=[{key: chunk[key] for key in ("index", "start_ms", "end_ms", "audio_sha256")}
                        for chunk in chunks])
    _write_state(intent_path, intent)
    _write_state(pending, request)
    return {"status": "ready", "job_id": intent["job_id"],
            "model": model["id"], "protocol": protocol,
            "secret_ref_target": config["secret_ref_target"],
            "credential_ref": config["credential_ref"],
            "request_path": str(pending), "billing_warning": billing_warning,
            "source_audio_seconds": duration_sec, "chunk_count": len(chunks)}


def _provider_chunk_text(chunk: dict[str, Any], protocol: str) -> str:
    if protocol == "http":
        response = chunk.get("raw_response") or {}
        output = response.get("output") or {}
        text = str(output.get("text") or "").strip()
        if not text:
            text = str(((output.get("output") or {}).get("sentence") or {}).get("text") or "").strip()
        return text
    sentences: dict[int, str] = {}
    for event in chunk.get("raw_events") or []:
        if (event.get("header") or {}).get("event") != "result-generated":
            continue
        sentence = ((event.get("payload") or {}).get("output") or {}).get("sentence") or {}
        if sentence.get("sentence_end") and type(sentence.get("sentence_id")) is int:
            sentences[sentence["sentence_id"]] = str(sentence.get("text") or "")
    return "".join(sentences[key] for key in sorted(sentences)).strip()


def _provider_chunk_usage(chunk: dict[str, Any], protocol: str) -> dict[str, Any] | None:
    """Select one cumulative usage snapshot for this provider task."""
    if protocol == "websocket":
        events = chunk.get("raw_events") or []
        for kind in ("task-finished", "result-generated"):
            for event in reversed(events):
                if (event.get("header") or {}).get("event") != kind:
                    continue
                usage = (event.get("payload") or {}).get("usage")
                if isinstance(usage, dict) and usage:
                    return usage
    usage = chunk.get("usage")
    return usage if isinstance(usage, dict) and usage else None


def _job_usage(chunks: list[dict[str, Any]], expected_count: int) -> dict[str, int | float] | None:
    """Sum final per-task snapshots only when every chunk has the same numeric fields."""
    if not chunks or len(chunks) != expected_count:
        return None
    usages = [chunk.get("usage") for chunk in chunks]
    if any(not isinstance(usage, dict) or not usage for usage in usages):
        return None
    fields = set(usages[0])
    if any(set(usage) != fields for usage in usages):
        return None
    if any(type(usage[field]) not in (int, float) for usage in usages for field in fields):
        return None
    return {field: sum(usage[field] for usage in usages) for field in sorted(fields)}


def finalize_cloud_result(intent_path: Path, root: Path, *,
                          provider_path: Path | None = None,
                          broker_error: str = "", state_path: Path | None = None,
                          config_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    intent = _bounded_intent(intent_path, root)
    config = load_cloud_config(config_path or root.parents[1] / "configs" / "models.yaml")
    job_id = intent["job_id"]
    result_path = root / (job_id + ".result.json")
    final: dict[str, Any] = {"schema": IMPORTANT_RESULT_SCHEMA if intent["purpose"] == "important_evidence"
        else QUALITY_RESULT_SCHEMA, "job_id": job_id, "purpose": intent["purpose"],
        "important_only": intent["purpose"] == "important_evidence",
        "status": "failed", "error_code": broker_error or "cloud_worker_result_missing",
        "credential_result": "Provider-Unavailable", "cloud_upload_performed": False,
        "provider": "aliyun-bailian", "model": intent.get("model"),
        "source_audio_sha256": intent.get("source_audio_sha256"),
        "source_audio_bytes": intent.get("source_audio_bytes"),
        "selected_channel": intent.get("channel_index"),
        "billing_warning": intent.get("billing_warning", ""),
        "text": "", "chunks": [], "usage": None, "result_path": str(result_path),
        "completed_utc": datetime.now(timezone.utc).isoformat()}
    if provider_path and provider_path.is_file():
        if not provider_path.resolve().is_relative_to(root):
            raise CloudReviewError("provider_result_outside_root")
        try:
            provider = json.loads(provider_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CloudReviewError("provider_result_invalid") from exc
        if (provider.get("schema") != PROVIDER_SCHEMA or
                provider.get("job_id") != job_id or
                provider.get("source_audio_sha256") != intent.get("source_audio_sha256") or
                provider.get("model") != intent.get("model") or
                provider.get("protocol") != intent.get("protocol")):
            raise CloudReviewError("provider_result_invalid")
        raw_chunks = provider.get("chunks") or []
        expected_chunks = intent.get("chunk_bindings") or []
        if (not isinstance(raw_chunks, list) or len(raw_chunks) > len(expected_chunks) or
                (provider.get("status") == "succeeded" and len(raw_chunks) != len(expected_chunks))):
            raise CloudReviewError("provider_result_binding_invalid")
        for actual, expected in zip(raw_chunks, expected_chunks):
            if (not isinstance(actual, dict) or any(actual.get(key) != expected[key]
                    for key in ("index", "start_ms", "end_ms", "audio_sha256"))):
                raise CloudReviewError("provider_result_binding_invalid")
        chunks = []
        for raw in raw_chunks:
            text = _provider_chunk_text(raw, str(provider.get("protocol")))
            chunks.append({"index": raw["index"], "start_ms": raw["start_ms"],
                "end_ms": raw["end_ms"], "audio_sha256": raw["audio_sha256"],
                "provider_request_id": raw.get("provider_request_id"),
                "text": text, "usage": _provider_chunk_usage(raw, str(provider.get("protocol"))),
                "raw_response": raw.get("raw_response"),
                "raw_events": raw.get("raw_events")})
        texts = [chunk["text"] for chunk in chunks if chunk["text"]]
        status = provider.get("status")
        if status == "succeeded" and len(texts) != len(chunks):
            status = "failed"
            error_code = "provider_transcript_missing"
        else:
            error_code = str(provider.get("error_code") or "")
        final.update(status=status, error_code=error_code,
            model=provider.get("model"), protocol=provider.get("protocol"),
            provider_endpoint=provider.get("provider_endpoint"),
            credential_result=provider.get("credential_result"),
            cloud_upload_performed=provider.get("cloud_upload_performed") is True,
            provider_error_code=provider.get("provider_error_code"),
            provider_error_message=provider.get("provider_error_message"),
            http_status=provider.get("http_status"),
            provider_request_id=provider.get("provider_request_id"),
            provider_result_path=str(provider_path),
            chunks=chunks, text="\n".join(texts),
            usage=_job_usage(chunks, len(expected_chunks)),
            started_utc=provider.get("started_utc"),
            completed_utc=provider.get("completed_utc"),
            reused_chunks=provider.get("reused_chunks", 0),
            new_requests=provider.get("new_requests", 0))
        reason = stopping_error(str(provider.get("provider_error_code") or error_code),
            str(provider.get("credential_result") or ""),
            str(provider.get("provider_error_message") or ""))
        if reason:
            pause_cloud(state_path or root / "auto-cloud-state.json", config,
                        reason=reason, provider_error=str(provider.get("provider_error_code") or error_code),
                        model=str(provider.get("model") or ""))
            final["auto_pause_reason"] = reason
            final["pause_reason"] = reason
            final["message"] = pause_message(reason)
        elif error_code == "request_send_deadline_passed":
            final["pause_reason"] = "free_period_expired"
            final["message"] = pause_message("free_period_expired")
    if final.get("status") == "succeeded":
        _, local_text = local_review_signals(Path(intent["local_out_dir"]),
            evidence_status=str(intent.get("evidence_status") or ""),
            important=intent["purpose"] == "important_evidence") if intent.get("local_out_dir") else ([], "")
        final["local_text"] = local_text
        final["disagreement"] = compare_text(local_text, final["text"]) if local_text else []
    _write_state(result_path, final)
    if final["status"] == "succeeded":
        (root / (job_id + ".transcript.txt")).write_text(final["text"] + "\n", encoding="utf-8")
    _local_sidecar(intent, final)
    return final

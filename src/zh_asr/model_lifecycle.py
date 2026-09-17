"""Explicit model maintenance: pinned installation, verification and reversible profiles.

No scheduler, background update, implicit download during ASR or automatic promotion.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
from typing import Any

from .config import ModelConfig, load_model_config
from .qwen_identity import RequiredModelFile, verify_model_receipt, write_model_receipt

ALIGNER_FILES = (
    "chat_template.json", "config.json", "generation_config.json", "merges.txt",
    "model.safetensors", "preprocessor_config.json", "tokenizer_config.json", "vocab.json",
)


def project_path(config: ModelConfig, value: str) -> Path:
    path = Path(value).expanduser()
    root = config.path.resolve().parent
    if root.name.lower() == "configs":
        root = root.parent
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alignment_contract(config: ModelConfig) -> tuple[dict[str, Any], Path, Path]:
    options = config.alignment
    if options.get("adapter") != "qwen-forced-aligner":
        raise ValueError("No supported forced-aligner configured")
    revision = str(options.get("model_revision", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("The aligner requires an immutable 40-character revision")
    for key in ("model", "model_dir", "artifact_lock", "runtime_distribution", "runtime_version"):
        if not str(options.get(key, "")).strip():
            raise ValueError(f"alignment.{key} is required")
    return options, project_path(config, options["model_dir"]), project_path(config, options["artifact_lock"])


def verify_aligner(config: ModelConfig) -> dict[str, Any]:
    options, model_dir, lock_path = alignment_contract(config)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if (lock.get("schema") != "zh_asr.artifact_lock.v1" or
            lock.get("repository") != options["model"] or
            lock.get("revision") != options["model_revision"]):
        raise ValueError("Forced-aligner configuration and artifact lock disagree")
    records = tuple(RequiredModelFile(**record) for record in lock["files"])
    if {record.path for record in records} != set(ALIGNER_FILES):
        raise ValueError("Forced-aligner artifact set is incomplete or unexpected")
    receipt = verify_model_receipt(model_dir, repository=lock["repository"],
        revision=lock["revision"], required_files=records)
    runtime = importlib.metadata.version(options["runtime_distribution"])
    if runtime != options["runtime_version"]:
        raise RuntimeError(f"Aligner runtime mismatch: {runtime} != {options['runtime_version']}")
    return {"model": lock["repository"], "revision": lock["revision"],
        "model_dir": str(model_dir), "artifact_lock_sha256": file_hash(lock_path),
        "receipt_sha256": receipt.sha256, "runtime": runtime, "status": "verified",
        "alignment_code_sha256": file_hash(Path(__file__).with_name("alignment.py")),
        "comparison_code_sha256": file_hash(Path(__file__).with_name("text_comparison.py"))}


def fetch_aligner(config: ModelConfig, download_dir: Path | None = None) -> dict[str, Any]:
    """Install missing pinned files, including on a fresh checkout with an existing lock.

    Existing matching files are reused. Mismatching installed files are reported,
    never accepted by creating a new checksum baseline. Downloaded bytes are
    verified in staging before they enter the installed model directory.
    """
    import os
    import tempfile
    from huggingface_hub import HfApi, hf_hub_download
    options, model_dir, lock_path = alignment_contract(config)
    if importlib.metadata.version(options["runtime_distribution"]) != options["runtime_version"]:
        raise RuntimeError("Install the configured aligner runtime before fetching weights")
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else None
    metadata = None
    if lock is not None:
        if (lock.get("schema") != "zh_asr.artifact_lock.v1" or
                lock.get("repository") != options["model"] or
                lock.get("revision") != options["model_revision"]):
            raise ValueError("Forced-aligner configuration and artifact lock disagree")
        records = tuple(RequiredModelFile(**x) for x in lock["files"])
        if len(records) != len(ALIGNER_FILES) or {x.path for x in records} != set(ALIGNER_FILES):
            raise ValueError("Forced-aligner artifact lock must contain the exact expected file set")
        by_name = {x.path: x for x in records}
        for name, record in by_name.items():
            target = model_dir / name
            if target.exists() and (target.stat().st_size != record.bytes or file_hash(target) != record.sha256):
                raise RuntimeError(f"Installed file differs from the immutable lock; preserve and repair explicitly: {name}")
        missing = [name for name in ALIGNER_FILES if not (model_dir / name).is_file()]
    else:
        info = HfApi().model_info(options["model"], revision=options["model_revision"], files_metadata=True)
        if info.sha != options["model_revision"]:
            raise RuntimeError("Resolved model revision differs from the configured immutable revision")
        metadata = {item.rfilename: item for item in info.siblings}
        if not set(ALIGNER_FILES).issubset(metadata):
            raise RuntimeError("Upstream forced-aligner snapshot is incomplete")
        missing, by_name = list(ALIGNER_FILES), {}
    if missing:
        root = download_dir or (Path("E:/Downloads/ChineseASR") if Path("E:/Downloads").is_dir()
                                else Path.home() / "Downloads" / "ChineseASR")
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="aligner-", dir=root) as staging_name:
            staging = Path(staging_name)
            for name in missing:
                hf_hub_download(repo_id=options["model"], revision=options["model_revision"],
                    filename=name, local_dir=str(staging))
                downloaded = staging / name
                digest, size = file_hash(downloaded), downloaded.stat().st_size
                if lock is not None:
                    expected = by_name[name]
                    if size != expected.bytes or digest != expected.sha256:
                        raise RuntimeError(f"Downloaded file differs from the immutable lock: {name}")
                else:
                    item = metadata[name]
                    if size != item.size:
                        raise RuntimeError(f"Upstream file size differs: {name}")
                    if item.lfs is not None:
                        matches = digest == item.lfs.sha256
                    else:
                        body = downloaded.read_bytes()
                        matches = hashlib.sha1(b"blob " + str(len(body)).encode("ascii") + b"\0" + body).hexdigest() == item.blob_id
                    if not matches:
                        raise RuntimeError(f"Upstream checksum differs: {name}")
                    by_name[name] = RequiredModelFile(name, size, digest)
            # Commit only after every missing file has passed verification.
            model_dir.mkdir(parents=True, exist_ok=True)
            for name in missing:
                target = model_dir / name
                if target.exists():
                    if file_hash(target) != by_name[name].sha256:
                        raise RuntimeError(f"Installation changed while downloading: {name}")
                else:
                    os.replace(staging / name, target)
    records = tuple(by_name[name] for name in ALIGNER_FILES)
    receipt = model_dir / "MODEL_RECEIPT.json"
    if receipt.exists():
        verify_model_receipt(model_dir, repository=options["model"],
            revision=options["model_revision"], required_files=records)
    else:
        write_model_receipt(model_dir, repository=options["model"],
            revision=options["model_revision"], required_files=records)
    if lock is None:
        write_json_atomic(lock_path, {"schema": "zh_asr.artifact_lock.v1",
            "repository": options["model"], "revision": options["model_revision"],
            "upstream": "huggingface_immutable_snapshot", "files": [asdict(x) for x in records]})
    return verify_aligner(config)


def model_status(config: ModelConfig) -> dict[str, Any]:
    from .adapters import ADAPTERS
    from .pipeline import default_cache_dir
    rows = []
    for name, spec in config.engines.items():
        options = spec.options or {}
        implemented = spec.adapter in ADAPTERS and not spec.is_whisper
        canonical = config.model_aliases.get(spec.model, spec.model)
        directory = (project_path(config, options["model_dir"]) if options.get("model_dir")
                     else default_cache_dir().joinpath(*canonical.split("/")))
        rows.append({"engine": name, "adapter": spec.adapter, "model": canonical,
            "configured": True, "implemented": implemented,
            "weights_present": directory.is_dir(), "revision": options.get("model_revision"),
            "readiness": "not_integrated" if not implemented else "present_not_verified" if directory.is_dir() else "not_installed",
            "runtime": options.get("runtime", "windows"),
            "note": "No inference or checksum verification performed by status"})
    options = config.alignment
    alignment = {"configured": bool(options), "model": options.get("model"),
        "revision": options.get("model_revision"), "weights_present": False}
    if options:
        _, directory, lock = alignment_contract(config)
        alignment.update(weights_present=directory.is_dir(), artifact_lock_present=lock.is_file())
    return {"schema": "zh_asr.model_status.v1", "default_engine": config.default_engine,
        "strict": [config.strict_primary_engine, config.strict_secondary_engine],
        "profiles": config.profiles, "engines": rows, "alignment": alignment,
        "update_policy": "explicit_check_pinned_candidate_validate_then_promote_no_scheduler"}


def check_updates(config: ModelConfig) -> dict[str, Any]:
    """Report upstream identities only. A newer revision is not quality evidence."""
    from huggingface_hub import HfApi
    api = HfApi()
    rows = []
    candidates = [(name, spec.model, (spec.options or {}).get("model_revision"),
                   "modelscope" if spec.adapter in {"qwen-asr", "funasr"} else "huggingface")
                  for name, spec in config.engines.items() if not spec.is_whisper]
    if config.alignment:
        candidates.append(("forced-aligner", config.alignment["model"], config.alignment["model_revision"], "huggingface"))
    for name, repository, active, source in candidates:
        row = {"component": name, "repository": repository, "active_revision": active,
               "active_revision_provider": source, "quality_improvement": "not_evaluated"}
        try:
            if source == "huggingface":
                info = api.model_info(repository)
                row.update(latest_revision=info.sha, revision_changed=bool(active and active != info.sha),
                           latest_revision_provider="huggingface")
            else:
                # ModelScope and Hugging Face revision IDs are not interchangeable.
                # Query the actual source rather than announcing a false upgrade.
                from modelscope.hub.api import HubApi
                detail = HubApi().get_valid_revision_detail(repository, revision="master")
                row.update(latest_revision=detail, latest_revision_provider="modelscope",
                           revision_changed="inspect_resolved_revision")
        except Exception as error:
            row.update(status="lookup_failed", error=f"{type(error).__name__}: {error}")
        rows.append(row)
    return {"schema": "zh_asr.update_check.v1", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "components": rows, "defaults_changed": False, "downloads_performed": False}


def compare_evaluations(baseline: Path, candidate: Path) -> dict[str, Any]:
    left = json.loads(baseline.read_text(encoding="utf-8"))
    right = json.loads(candidate.read_text(encoding="utf-8"))
    problems = []
    if left.get("schema_version") != 3 or right.get("schema_version") != 3:
        problems.append("unsupported_evaluation_schema")
    if not left.get("comparison_policy") or left.get("comparison_policy") != right.get("comparison_policy"):
        problems.append("metric_policy_mismatch")
    import math
    for label, payload in (("baseline", left), ("candidate", right)):
        items = payload.get("cases", [])
        if not isinstance(items, list) or any(not isinstance(x, dict) or not x.get("id") for x in items):
            raise ValueError(f"Invalid {label} evaluation cases")
        if len({x["id"] for x in items}) != len(items):
            raise ValueError(f"Duplicate case identifiers in {label}")
        for item in items:
            cer = item.get("cer")
            if cer is not None and (isinstance(cer, bool) or not isinstance(cer, (int, float))
                    or not math.isfinite(cer) or cer < 0):
                raise ValueError(f"Invalid CER in {label}: {item['id']}")
    lc = {item["id"]: item for item in left.get("cases", [])}
    rc = {item["id"]: item for item in right.get("cases", [])}
    if not lc or lc.keys() != rc.keys():
        problems.append("case_set_mismatch_or_empty")
    for key in lc.keys() & rc.keys():
        for field in ("audio_sha256", "truth_sha256"):
            if not lc[key].get(field) or lc[key].get(field) != rc[key].get(field):
                problems.append(f"source_mismatch:{key}:{field}")
        if (lc[key].get("cer") is None) != (rc[key].get("cer") is None):
            problems.append(f"measurement_availability_mismatch:{key}")
        if lc[key].get("kind") != rc[key].get("kind"):
            problems.append(f"case_kind_mismatch:{key}")
        if lc[key].get("skipped") or rc[key].get("skipped"):
            problems.append(f"skipped_case:{key}")
    def score(cases):
        measured = [x for x in cases.values() if x.get("cer") is not None and not x.get("skipped")]
        return {"mean_cer": sum(x["cer"] for x in measured) / max(1, len(measured)),
            "measured_cases": len(measured),
            "critical_error_cases": sum(bool(x.get("critical_errors")) for x in measured),
            "false_confident_cases": sum(bool(x.get("false_confident")) for x in cases.values()),
            "failed_cases": sum(x.get("audit_status") == "engine_failure" for x in cases.values()),
            "review_cases": sum(bool(x.get("audit_needs_review", x.get("needs_review"))) for x in cases.values())}
    ls, rs = score(lc), score(rc)
    for key in ("mean_cer", "critical_error_cases", "false_confident_cases", "failed_cases"):
        if rs[key] > ls[key] + 1e-12:
            problems.append("regression:" + key)
    if rs["measured_cases"] < 5:
        problems.append("insufficient_measured_cases")
    unique_audio = {x.get("audio_sha256") for x in rc.values() if x.get("cer") is not None and not x.get("skipped")}
    if len(unique_audio) < 5:
        problems.append("insufficient_distinct_audio_cases")
    kinds = {x.get("kind", "") for x in rc.values()}
    if not kinds.intersection({"human", "real", "public", "recording"}):
        problems.append("synthetic_only_not_quality_promotion_evidence")
    if not any(rs[k] < ls[k] for k in ("mean_cer", "critical_error_cases", "false_confident_cases")):
        problems.append("no_measured_quality_improvement")
    return {"schema": "zh_asr.promotion_comparison.v1", "eligible": not problems,
        "baseline_sha256": file_hash(baseline), "candidate_sha256": file_hash(candidate),
        "baseline": ls, "candidate": rs, "reasons": problems,
        "limitation": "Eligibility is bounded to the supplied matched evaluation corpus, not universal accuracy."}


def _replace_strict_pair(text: str, pair: tuple[str, str]) -> str:
    """Change only the two strict defaults, preserving comments and other settings."""
    import copy
    import yaml
    before = yaml.safe_load(text)
    match = re.search(r"(?ms)^strict:[^\n]*\n(?P<body>.*?)(?=^\S|\Z)", text)
    if match is None:
        raise ValueError("A top-level strict mapping is required")
    body = match.group("body")
    for key, value in zip(("primary_engine", "secondary_engine"), pair):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError("Engine name cannot be represented as a plain YAML scalar")
        pattern = rf"(?m)^(  {key}:)[^\n#]*(?P<comment>#[^\n]*)?$"
        body, count = re.subn(pattern, lambda m: m[1] + " " + value + (" " + m["comment"] if m["comment"] else ""), body)
        if count != 1:
            raise ValueError(f"Exactly one strict.{key} declaration is required")
    result = text[:match.start("body")] + body + text[match.end("body"):]
    expected = copy.deepcopy(before)
    expected["strict"]["primary_engine"], expected["strict"]["secondary_engine"] = pair
    if yaml.safe_load(result) != expected:
        raise RuntimeError("Changing the strict defaults would alter unrelated configuration")
    return result


def _switch_defaults(config: ModelConfig, pair: tuple[str, str], *, reason: dict[str, Any]) -> dict[str, Any]:
    import os
    import uuid
    before = config.path.read_bytes()
    current = load_model_config(config.path)
    previous = (current.strict_primary_engine, current.strict_secondary_engine)
    if previous == pair:
        return {"status": "unchanged", "pair": list(pair), "defaults_changed": False}
    after = _replace_strict_pair(before.decode("utf-8-sig"), pair).encode("utf-8")
    root = project_path(config, "outputs/model-maintenance") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    (root / "before.yaml").write_bytes(before)
    (root / "after.yaml").write_bytes(after)
    receipt = {"schema": "zh_asr.profile_switch.v1", "status": "prepared", "config": str(config.path.resolve()),
        "before_sha256": hashlib.sha256(before).hexdigest(), "after_sha256": hashlib.sha256(after).hexdigest(),
        "previous_pair": list(previous), "activated_pair": list(pair), "reason": reason,
        "receipt": str((root / "switch.json").resolve()), "defaults_changed": False}
    write_json_atomic(root / "switch.json", receipt)
    temporary = config.path.with_name(config.path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(after)
        load_model_config(temporary)
        if config.path.read_bytes() != before:
            raise RuntimeError("Configuration changed during profile switch; original left untouched")
        os.replace(temporary, config.path)
        if config.path.read_bytes() != after:
            raise RuntimeError("Configuration readback differs after profile switch")
        receipt.update(status="applied", defaults_changed=True)
        write_json_atomic(root / "switch.json", receipt)
    finally:
        temporary.unlink(missing_ok=True)
    return receipt


def activate_profile(config: ModelConfig, profile: str, baseline: Path, candidate: Path) -> dict[str, Any]:
    from .config import resolve_profile
    from .long_audio import _runtime_code_identity
    comparison = compare_evaluations(baseline, candidate)
    if not comparison["eligible"]:
        raise ValueError("Candidate did not pass matched evaluation: " + ", ".join(comparison["reasons"]))
    target = resolve_profile(config, profile)
    left, right = (json.loads(x.read_text(encoding="utf-8")) for x in (baseline, candidate))
    if right.get("model_config", {}).get("sha256") != file_hash(config.path):
        raise ValueError("Candidate was not evaluated against the current model configuration")
    if right.get("runtime_code", {}).get("sha256") != _runtime_code_identity()["sha256"]:
        raise ValueError("Candidate was not evaluated against the current runtime code and model locks")
    for report, expected in ((left, (config.strict_primary_engine, config.strict_secondary_engine)), (right, target)):
        if any((row.get("models", {}).get("primary"), row.get("models", {}).get("secondary")) != expected
               for row in report["cases"]):
            raise ValueError("Evaluation engine pair differs from the baseline or target profile")
    return _switch_defaults(config, target, reason={"profile": profile, "comparison": comparison})


def rollback_profile(config: ModelConfig, receipt_path: Path) -> dict[str, Any]:
    from .config import resolve_profile
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "zh_asr.profile_switch.v1" or receipt.get("config") != str(config.path.resolve()):
        raise ValueError("Rollback receipt does not belong to this configuration")
    previous, activated = tuple(receipt["previous_pair"]), tuple(receipt["activated_pair"])
    if len(previous) != 2 or len(activated) != 2:
        raise ValueError("Invalid rollback engine pairs")
    current = load_model_config(config.path)
    if (current.strict_primary_engine, current.strict_secondary_engine) != activated:
        raise ValueError("Defaults changed after activation; refusing to roll back an unrelated change")
    resolve_profile(current, primary=previous[0], secondary=previous[1])
    # Restore only this switch's defaults; preserve later unrelated config edits.
    return _switch_defaults(current, previous, reason={"rollback_of": str(receipt_path.resolve()),
        "rollback_receipt_sha256": file_hash(receipt_path)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "fetch-aligner", "verify-aligner", "check-updates", "compare", "activate-profile", "rollback-profile"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    config = load_model_config(args.config)
    actions = {"status": model_status, "fetch-aligner": fetch_aligner,
               "verify-aligner": verify_aligner, "check-updates": check_updates}
    if args.command == "activate-profile":
        if not args.profile or not args.baseline or not args.candidate:
            parser.error("activate-profile requires --profile, --baseline and --candidate")
        result = activate_profile(config, args.profile, args.baseline, args.candidate)
    elif args.command == "rollback-profile":
        if not args.receipt:
            parser.error("rollback-profile requires --receipt")
        result = rollback_profile(config, args.receipt)
    elif args.command == "compare":
        if not args.baseline or not args.candidate:
            parser.error("compare requires --baseline and --candidate")
        result = compare_evaluations(args.baseline, args.candidate)
    elif args.command == "fetch-aligner":
        result = fetch_aligner(config, args.download_dir)
    else:
        result = actions[args.command](config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get("eligible") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Content-bound evaluation reuse; an existing filename is never a cache proof."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .audit import validate_strict_artifact_bundle
from .metadata import file_metadata
from .model_lifecycle import write_json_atomic


CACHE_SCHEMA = "zh_asr.evaluation_case_cache.v1"


def observe_case(corpus: Path, case: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Read actual reference/input identity, rejecting stale declared hashes."""
    audio = corpus / str(case["audio"])
    if not audio.is_file():
        raise FileNotFoundError(f"Evaluation audio is missing: {audio}")
    audio_meta = file_metadata(audio)
    truth_path = corpus / str(case["truth"]) if case.get("truth") else None
    if truth_path is not None and truth_path.is_file():
        raw = truth_path.read_bytes()
        text = raw.decode("utf-8-sig").strip()
        truth_meta = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    elif "truth_text" in case:
        text = str(case["truth_text"]).strip()
        raw = text.encode("utf-8")
        truth_meta = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    else:
        raise FileNotFoundError("Evaluation reference text is missing; absence is not an empty reference")
    observed = {**case, "audio_sha256": audio_meta["sha256"],
        "audio_size_bytes": audio_meta["size_bytes"], "truth_sha256": truth_meta["sha256"],
        "truth_size_bytes": truth_meta["size_bytes"]}
    for key in ("audio_sha256", "truth_sha256"):
        if case.get(key) and case[key] != observed[key]:
            raise ValueError(f"Evaluation manifest no longer matches the actual {key}: {case['id']}")
    return observed, text


def read_case_cache(path: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != CACHE_SCHEMA or value.get("identity") != dict(identity):
            return None
        records = value.get("outputs")
        if not isinstance(records, dict) or not records:
            return None
        outputs = {}
        for key, record in records.items():
            output = Path(record["path"])
            if not output.is_file() or file_metadata(output) != record["metadata"]:
                return None
            outputs[key] = output
        status, failures = validate_strict_artifact_bundle(outputs,
            expected_primary_engine=identity["primary_engine"],
            expected_secondary_engine=identity["secondary_engine"])
        if status != "verified" or failures:
            return None
        outputs["timing"] = dict(value.get("timing") or {})
        outputs["evaluation_cache_reused"] = True
        return outputs
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def save_case_cache(path: Path, identity: Mapping[str, Any], outputs: Mapping[str, Any]) -> None:
    """Only complete verified bundles are reusable; failed runs remain observable."""
    records = {key: {"path": str(value.resolve()), "metadata": file_metadata(value)}
               for key, value in outputs.items() if isinstance(value, Path) and value.is_file()}
    status, failures = validate_strict_artifact_bundle(outputs,
        expected_primary_engine=identity["primary_engine"],
        expected_secondary_engine=identity["secondary_engine"])
    if status != "verified" or failures:
        path.unlink(missing_ok=True)
        return
    write_json_atomic(path, {"schema": CACHE_SCHEMA, "identity": dict(identity),
        "outputs": records, "timing": dict(outputs.get("timing") or {})})

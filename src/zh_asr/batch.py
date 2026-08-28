from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from .audit import validate_strict_artifact_bundle
from .audio_outcome import load_objective_result, validate_objective_result
from .config import ModelConfig, load_model_config
from .metadata import file_metadata, snapshot_model_config
from .pipeline import (
    strict_transcribe_audio,
    strict_transcribe_many,
    transcribe_audio,
    transcribe_audio_many,
)
from .result_writer import canonical_json_sha256, file_sha256


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
BATCH_ITEM_STATE_SCHEMA = "zh_asr.batch_item.v1"
BATCH_ITEM_STATE_FILENAME = "batch-item.json"
_STRICT_CACHE_OUTPUTS = (
    "final",
    "audit",
    "audit_json",
    "review_json",
    "receipt",
    "primary_json",
    "secondary_json",
    "objective_result",
)
_QUICK_CACHE_OUTPUTS = ("markdown", "json", "objective_result")


@dataclass(frozen=True)
class BatchItem:
    audio: Path
    out_dir: Path
    status: str
    message: str = ""
    objective_outcome: str = "indeterminate"


@dataclass(frozen=True)
class BatchSummary:
    input_dir: Path
    out_dir: Path
    mode: str
    total: int
    processed: int
    skipped: int
    failed: int
    items: tuple[BatchItem, ...] = field(default_factory=tuple)


StrictFn = Callable[..., dict[str, Path]]
TranscribeFn = Callable[..., dict[str, Path]]
StrictManyFn = Callable[..., list[dict[str, Any]]]
TranscribeManyFn = Callable[..., list[dict[str, Any]]]


def find_audio_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    return sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS),
        key=lambda path: str(path.relative_to(input_dir)).lower(),
    )


def run_batch(
    input_dir: Path,
    out_dir: Path,
    mode: str = "strict",
    device: str = "cuda:0",
    cache_dir: Path | None = None,
    force: bool = False,
    engine: str | None = None,
    primary_engine: str | None = None,
    secondary_engine: str | None = None,
    config: ModelConfig | None = None,
    transcribe_fn: TranscribeFn = transcribe_audio,
    strict_fn: StrictFn = strict_transcribe_audio,
    transcribe_many_fn: TranscribeManyFn = transcribe_audio_many,
    strict_many_fn: StrictManyFn = strict_transcribe_many,
    caller_binding: Mapping[str, Any] | None = None,
) -> BatchSummary:
    mode_key = mode.strip().lower()
    if mode_key not in {"strict", "quick"}:
        raise ValueError("Batch mode must be 'strict' or 'quick'.")

    input_root = input_dir.resolve()
    output_root = out_dir.resolve()
    model_config = config or load_model_config()
    quick_engine = engine or model_config.default_engine
    strict_primary = primary_engine or model_config.strict_primary_engine
    strict_secondary = secondary_engine or model_config.strict_secondary_engine
    output_root.mkdir(parents=True, exist_ok=True)
    failed_path = output_root / "failed.jsonl"
    if not failed_path.exists():
        failed_path.touch()

    audio_files = find_audio_files(input_root)
    items_by_audio: dict[Path, BatchItem] = {}
    pending: list[tuple[Path, Path, dict[str, Any]]] = []
    for audio_path in audio_files:
        item_dir = output_root / _output_dir_name(input_root, audio_path)
        identity = _batch_item_identity(
            audio_path,
            mode=mode_key,
            quick_engine=quick_engine,
            strict_primary=strict_primary,
            strict_secondary=strict_secondary,
            config=model_config,
            device=device,
            cache_dir=cache_dir,
            caller_binding=caller_binding,
        )

        if not force:
            objective_outcome = _cached_item_objective_outcome(
                item_dir,
                identity,
                mode=mode_key,
                primary_engine=strict_primary,
                secondary_engine=strict_secondary,
                quick_engine=quick_engine,
            )
            if objective_outcome is not None:
                items_by_audio[audio_path] = BatchItem(
                    audio=audio_path,
                    out_dir=item_dir,
                    status="skipped",
                    message="already complete",
                    objective_outcome=objective_outcome,
                )
                continue
        pending.append((audio_path, item_dir, identity))

    use_many = (
        mode_key == "strict" and strict_fn is strict_transcribe_audio
    ) or (
        mode_key == "quick" and transcribe_fn is transcribe_audio
    )
    if pending and use_many:
        for _, item_dir, _ in pending:
            item_dir.mkdir(parents=True, exist_ok=True)
        try:
            if mode_key == "strict":
                call_kwargs = {
                    "out_dirs": [item_dir for _, item_dir, _ in pending],
                    "primary_engine": strict_primary,
                    "secondary_engine": strict_secondary,
                    "device": device,
                    "cache_dir": cache_dir,
                    "config": model_config,
                }
                if caller_binding is not None:
                    call_kwargs["caller_binding"] = caller_binding
                outputs_many = strict_many_fn(
                    [audio_path for audio_path, _, _ in pending],
                    **call_kwargs,
                )
            else:
                call_kwargs = {
                    "out_dirs": [item_dir for _, item_dir, _ in pending],
                    "engine": quick_engine,
                    "device": device,
                    "cache_dir": cache_dir,
                    "config": model_config,
                }
                if caller_binding is not None:
                    call_kwargs["caller_binding"] = caller_binding
                outputs_many = transcribe_many_fn(
                    [audio_path for audio_path, _, _ in pending],
                    **call_kwargs,
                )
            if len(outputs_many) != len(pending):
                raise RuntimeError(
                    f"Batch transcription returned {len(outputs_many)} results for {len(pending)} inputs."
                )
            for (audio_path, item_dir, identity), outputs in zip(pending, outputs_many):
                objective_outcome = _result_objective_outcome(
                    outputs,
                    item_dir,
                    audio_path,
                )
                _write_batch_item_state(
                    item_dir,
                    identity,
                    outputs,
                    objective_outcome,
                )
                items_by_audio[audio_path] = BatchItem(
                    audio=audio_path,
                    out_dir=item_dir,
                    status="processed",
                    objective_outcome=objective_outcome,
                )
        except Exception as exc:
            for audio_path, item_dir, _ in pending:
                items_by_audio[audio_path] = BatchItem(
                    audio=audio_path,
                    out_dir=item_dir,
                    status="failed",
                    message=str(exc),
                    objective_outcome="indeterminate",
                )
                _append_failure(failed_path, audio_path, item_dir, exc)
    else:
        for audio_path, item_dir, identity in pending:
            item_dir.mkdir(parents=True, exist_ok=True)
            try:
                if mode_key == "strict":
                    call_kwargs = {
                        "primary_engine": strict_primary,
                        "secondary_engine": strict_secondary,
                        "device": device,
                        "out_dir": item_dir,
                        "cache_dir": cache_dir,
                        "config": model_config,
                    }
                    if caller_binding is not None:
                        call_kwargs["caller_binding"] = caller_binding
                    outputs = strict_fn(audio_path, **call_kwargs)
                else:
                    call_kwargs = {
                        "engine": quick_engine,
                        "device": device,
                        "out_dir": item_dir,
                        "cache_dir": cache_dir,
                        "config": model_config,
                    }
                    if caller_binding is not None:
                        call_kwargs["caller_binding"] = caller_binding
                    outputs = transcribe_fn(audio_path, **call_kwargs)
                objective_outcome = _result_objective_outcome(
                    outputs,
                    item_dir,
                    audio_path,
                )
                _write_batch_item_state(
                    item_dir,
                    identity,
                    outputs,
                    objective_outcome,
                )
                items_by_audio[audio_path] = BatchItem(
                    audio=audio_path,
                    out_dir=item_dir,
                    status="processed",
                    objective_outcome=objective_outcome,
                )
            except Exception as exc:
                items_by_audio[audio_path] = BatchItem(
                    audio=audio_path,
                    out_dir=item_dir,
                    status="failed",
                    message=str(exc),
                    objective_outcome="indeterminate",
                )
                _append_failure(failed_path, audio_path, item_dir, exc)

    items = [items_by_audio[audio_path] for audio_path in audio_files]

    summary = BatchSummary(
        input_dir=input_root,
        out_dir=output_root,
        mode=mode_key,
        total=len(items),
        processed=sum(1 for item in items if item.status == "processed"),
        skipped=sum(1 for item in items if item.status == "skipped"),
        failed=sum(1 for item in items if item.status == "failed"),
        items=tuple(items),
    )
    _write_summary(output_root / "summary.md", summary)
    return summary


def _batch_item_identity(
    audio_path: Path,
    *,
    mode: str,
    quick_engine: str,
    strict_primary: str,
    strict_secondary: str,
    config: ModelConfig,
    device: str,
    cache_dir: Path | None,
    caller_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    selected_engines = (
        (quick_engine,)
        if mode == "quick"
        else (strict_primary, strict_secondary)
    )
    source = file_metadata(audio_path)
    identity: dict[str, Any] = {
        "audio_path": str(audio_path.resolve()),
        "audio_sha256": str(source.get("sha256") or ""),
        "audio_size_bytes": int(source.get("size_bytes") or 0),
        "mode": mode,
        "engine": quick_engine if mode == "quick" else None,
        "primary_engine": strict_primary if mode == "strict" else None,
        "secondary_engine": strict_secondary if mode == "strict" else None,
        "device": device,
        "cache_dir": (
            str(cache_dir.expanduser().resolve()) if cache_dir is not None else None
        ),
        "model_config": snapshot_model_config(config, selected_engines),
        "caller_binding_sha256": (
            canonical_json_sha256(dict(caller_binding))
            if caller_binding is not None
            else ""
        ),
    }
    identity["identity_sha256"] = canonical_json_sha256(identity)
    return identity


def _cached_item_objective_outcome(
    item_dir: Path,
    identity: Mapping[str, Any],
    *,
    mode: str,
    primary_engine: str,
    secondary_engine: str,
    quick_engine: str,
) -> str | None:
    state_path = item_dir / BATCH_ITEM_STATE_FILENAME
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict) or state.get("schema") != BATCH_ITEM_STATE_SCHEMA:
        return None
    saved_identity = state.get("identity")
    if not isinstance(saved_identity, dict) or dict(saved_identity) != dict(identity):
        return None
    saved_identity_hash = saved_identity.get("identity_sha256")
    identity_without_hash = {
        key: value for key, value in saved_identity.items() if key != "identity_sha256"
    }
    if saved_identity_hash != canonical_json_sha256(identity_without_hash):
        return None

    output_records = state.get("outputs")
    required = _QUICK_CACHE_OUTPUTS if mode == "quick" else _STRICT_CACHE_OUTPUTS
    if (
        not isinstance(output_records, dict)
        or not all(key in output_records for key in required)
    ):
        return None

    outputs: dict[str, str] = {}
    for key, record in output_records.items():
        if not isinstance(record, dict):
            return None
        path = Path(str(record.get("path") or "")).expanduser()
        expected_hash = record.get("sha256")
        expected_size = record.get("size_bytes")
        if (
            not path.is_file()
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
        ):
            return None
        try:
            if path.stat().st_size != expected_size or file_sha256(path) != expected_hash:
                return None
        except OSError:
            return None
        outputs[str(key)] = str(path)

    if mode == "strict":
        evidence_status, _ = validate_strict_artifact_bundle(
            outputs,
            expected_primary_engine=primary_engine,
            expected_secondary_engine=secondary_engine,
        )
        # A provisional bundle records a failed required engine.  It remains
        # useful for review, but must not become an "already complete" cache
        # hit for a strict batch rerun.
        if evidence_status != "verified":
            return None

    objective_path = Path(outputs["objective_result"]).expanduser()
    objective = load_objective_result(objective_path)
    if validate_objective_result(objective):
        return None
    audio = objective.get("audio") if isinstance(objective, dict) else None
    if (
        not isinstance(audio, Mapping)
        or str(audio.get("raw_sha256") or "")
        != str(identity.get("audio_sha256") or "")
    ):
        return None

    try:
        actual_refs = []
        if mode == "quick":
            json_path = Path(outputs["json"]).expanduser()
            actual_refs.append(
                {
                    "schema": "media.raw-artifact-ref.v1",
                    "role": "lexical_primary",
                    "engine": quick_engine,
                    "path": json_path.name,
                    "size_bytes": json_path.stat().st_size,
                    "sha256": file_sha256(json_path),
                }
            )
        else:
            for key, role, engine in (
                ("primary_json", "lexical_primary", primary_engine),
                ("secondary_json", "lexical_verifier", secondary_engine),
            ):
                path = Path(outputs[key]).expanduser()
                actual_refs.append(
                    {
                        "schema": "media.raw-artifact-ref.v1",
                        "role": role,
                        "engine": engine,
                        "path": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
        receipt_path = (
            Path(outputs["receipt"]).expanduser()
            if mode == "strict"
            else None
        )
        receipt_ref = (
            {
                "schema": "media.strict-receipt-ref.v1",
                "path": receipt_path.name,
                "size_bytes": receipt_path.stat().st_size,
                "sha256": file_sha256(receipt_path),
            }
            if receipt_path is not None
            else None
        )
    except (OSError, TypeError, ValueError):
        return None
    if validate_objective_result(
        objective,
        raw_artifacts=actual_refs,
        strict_receipt=receipt_ref,
    ):
        return None
    objective_outcome = str(objective.get("objective_outcome") or "indeterminate")
    if str(state.get("objective_outcome") or objective_outcome) != objective_outcome:
        return None
    return objective_outcome


def _write_batch_item_state(
    item_dir: Path,
    identity: Mapping[str, Any],
    outputs: object,
    objective_outcome: str,
) -> None:
    if not isinstance(outputs, dict):
        raise RuntimeError("Batch transcription result must be an object.")
    output_records: dict[str, dict[str, Any]] = {}
    for key, value in outputs.items():
        if isinstance(value, Path):
            path = value.resolve()
        elif isinstance(value, str) and Path(value).is_file():
            path = Path(value).resolve()
        else:
            continue
        output_records[str(key)] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    if not output_records:
        raise RuntimeError("Batch transcription returned no persisted output paths.")
    _write_json_atomic(
        item_dir / BATCH_ITEM_STATE_FILENAME,
        {
            "schema": BATCH_ITEM_STATE_SCHEMA,
            "version": 1,
            "identity": dict(identity),
            "outputs": output_records,
            "objective_outcome": objective_outcome,
        },
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _output_dir_name(input_dir: Path, audio_path: Path) -> str:
    relative = audio_path.relative_to(input_dir).with_suffix("")
    raw = "__".join(relative.parts)
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", raw).strip("._") or audio_path.stem


def _expected_output_path(item_dir: Path, audio_path: Path, mode: str, engine: str | None) -> Path:
    if mode == "strict":
        return item_dir / f"{audio_path.stem}.strict.md"
    engine_name = engine or "sensevoice"
    return item_dir / f"{audio_path.stem}.{engine_name}.md"


def _append_failure(path: Path, audio_path: Path, out_dir: Path, exc: Exception) -> None:
    payload = {
        "audio": str(audio_path),
        "out_dir": str(out_dir),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_summary(path: Path, summary: BatchSummary) -> None:
    lines = [
        "# Batch Transcription Summary",
        "",
        f"- Input: `{summary.input_dir}`",
        f"- Output: `{summary.out_dir}`",
        f"- Mode: `{summary.mode}`",
        f"- Total: {summary.total}",
        f"- Processed: {summary.processed}",
        f"- Skipped: {summary.skipped}",
        f"- Failed: {summary.failed}",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Files",
        "",
    ]
    if not summary.items:
        lines.append("- None")
    else:
        for item in summary.items:
            suffix = f" - {item.message}" if item.message else ""
            suffix += f"; objective={item.objective_outcome}"
            lines.append(f"- `{item.status}` `{item.audio}` -> `{item.out_dir}`{suffix}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_objective_outcome(item_dir: Path, audio_path: Path) -> str:
    matches = sorted(item_dir.glob("*.objective-result.json"))
    if not matches:
        return "indeterminate"
    payload = load_objective_result(matches[0])
    if validate_objective_result(payload):
        return "indeterminate"
    return str(payload.get("objective_outcome") or "indeterminate")


def _result_objective_outcome(outputs: object, item_dir: Path, audio_path: Path) -> str:
    if isinstance(outputs, dict):
        direct = outputs.get("objective_outcome") or outputs.get("audio_result_status")
        if direct:
            return str(direct)
    return _read_objective_outcome(item_dir, audio_path)

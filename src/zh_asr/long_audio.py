from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import re
import uuid
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Any, Mapping

from .arbitration import ArbitrationEvidence
from .audit import validate_strict_artifact_bundle
from .audio_outcome import (
    aggregate_objective_result,
    load_objective_result,
    validate_objective_result,
    write_objective_result,
)
from .audio_frontend import PreparedAudio, prepare_pcm16_mono
from .config import load_model_config
from .metadata import file_metadata, sha256_file
from .pipeline import default_cache_dir, strict_transcribe_many


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: str
    index: int
    start_ms: int
    end_ms: int
    audio_path: Path

    def to_dict(self) -> dict[str, str | int]:
        return {
            "chunk_id": self.chunk_id,
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "audio_path": str(self.audio_path),
        }


@dataclass
class ChunkState:
    spec: ChunkSpec
    status: str = "pending"
    evidence_status: str = "pending"
    evidence_failures: list[dict[str, str]] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    error: str = ""
    arbitration: dict[str, Any] | None = None
    objective_outcome: str = "indeterminate"

    def to_dict(self) -> dict[str, Any]:
        payload = self.spec.to_dict()
        payload.update(
            {
                "status": self.status,
                "evidence_status": self.evidence_status,
                "evidence_failures": self.evidence_failures,
                "outputs": self.outputs,
                "error": self.error,
                "arbitration": self.arbitration,
                "objective_outcome": self.objective_outcome,
            }
        )
        return payload


@dataclass(frozen=True)
class LongRunSummary:
    out_dir: Path
    manifest_path: Path
    transcript_path: Path
    audit_path: Path
    metrics_path: Path
    total: int
    processed: int
    skipped: int
    failed: int
    evidence_status: str
    objective_outcome: str = "indeterminate"


StrictFn = Callable[..., dict[str, Any]]
_DEFAULT_MAX_REQUEST_INPUTS = 16


def plan_chunks(
    audio_path: Path,
    chunk_sec: int = 300,
    overlap_sec: int = 1,
    chunks_dir: Path | None = None,
    *,
    recommended_chunk_sec: int | None = None,
    absolute_max_chunk_sec: int | None = None,
) -> list[ChunkSpec]:
    if chunk_sec <= 0:
        raise ValueError("chunk_sec must be greater than 0")
    if overlap_sec < 0:
        raise ValueError("overlap_sec must be greater than or equal to 0")
    effective_chunk_sec = _effective_chunk_sec(
        chunk_sec,
        recommended_chunk_sec=recommended_chunk_sec,
        absolute_max_chunk_sec=absolute_max_chunk_sec,
    )
    if overlap_sec >= effective_chunk_sec:
        raise ValueError("overlap_sec must be smaller than the effective chunk duration")

    duration_ms = _wav_duration_ms(audio_path)
    chunks_root = chunks_dir or audio_path.parent / "chunks"
    chunk_ms = effective_chunk_sec * 1000
    step_ms = (effective_chunk_sec - overlap_sec) * 1000
    specs: list[ChunkSpec] = []
    start_ms = 0
    index = 1
    while start_ms < duration_ms:
        end_ms = min(start_ms + chunk_ms, duration_ms)
        chunk_id = f"chunk-{index:06d}"
        specs.append(
            ChunkSpec(
                chunk_id=chunk_id,
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                audio_path=chunks_root / f"{chunk_id}.wav",
            )
        )
        if end_ms >= duration_ms:
            break
        start_ms += step_ms
        index += 1
    return specs


def run_long_transcription(
    audio_path: Path,
    out_dir: Path,
    *,
    chunk_sec: int = 300,
    overlap_sec: int = 1,
    primary_engine: str | None = None,
    secondary_engine: str | None = None,
    device: str = "cuda:0",
    cache_dir: Path | None = None,
    force: bool = False,
    strict_fn: StrictFn | None = None,
    arbiter=None,
    caller_binding: Mapping[str, Any] | None = None,
) -> LongRunSummary:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    out_dir = out_dir.resolve()
    chunks_dir = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    prepared_audio = prepare_pcm16_mono(audio_path.resolve(), out_dir / "_derived")
    processing_audio_path = prepared_audio.path
    model_config = load_model_config()
    resolved_cache_dir = (cache_dir or default_cache_dir()).expanduser().resolve()
    resolved_primary_engine = primary_engine or model_config.strict_primary_engine
    resolved_secondary_engine = secondary_engine or model_config.strict_secondary_engine
    runtime_artifact_identity = _runtime_artifact_identity(
        model_config,
        (resolved_primary_engine, resolved_secondary_engine),
        resolved_cache_dir,
    )
    recommended_chunk_sec, absolute_max_chunk_sec = _engine_chunk_limits(
        model_config,
        (resolved_primary_engine, resolved_secondary_engine),
    )
    max_request_inputs = (
        _engine_request_batch_limit(
            model_config,
            (resolved_primary_engine, resolved_secondary_engine),
        )
        if strict_fn is None
        else None
    )
    effective_chunk_sec = _effective_chunk_sec(
        chunk_sec,
        recommended_chunk_sec=recommended_chunk_sec,
        absolute_max_chunk_sec=absolute_max_chunk_sec,
    )
    specs = plan_chunks(
        processing_audio_path,
        chunk_sec=chunk_sec,
        overlap_sec=overlap_sec,
        chunks_dir=chunks_dir,
        recommended_chunk_sec=recommended_chunk_sec,
        absolute_max_chunk_sec=absolute_max_chunk_sec,
    )
    manifest_path = out_dir / "manifest.json"
    run_identity = {
        "requested_chunk_sec": chunk_sec,
        "effective_chunk_sec": effective_chunk_sec,
        "overlap_sec": overlap_sec,
        "explicit_primary_engine": primary_engine,
        "explicit_secondary_engine": secondary_engine,
        "resolved_primary_engine": resolved_primary_engine,
        "resolved_secondary_engine": resolved_secondary_engine,
        "model_config_sha256": sha256_file(model_config.path) if model_config.path.exists() else "",
        "device": device,
        "cache_dir": str(resolved_cache_dir),
        "runtime_versions": _runtime_versions(
            model_config,
            (resolved_primary_engine, resolved_secondary_engine),
        ),
        "runtime_artifact_identity": runtime_artifact_identity,
        "runtime_code_identity": _runtime_code_identity(),
        "recommended_chunk_sec": recommended_chunk_sec,
        "absolute_max_chunk_sec": absolute_max_chunk_sec,
        "prepared_audio_sha256": prepared_audio.derivative_sha256,
        "prepared_audio_sample_rate": prepared_audio.sample_rate,
        "prepared_audio_channels": prepared_audio.channels,
        "prepared_audio_sample_width_bytes": prepared_audio.sample_width_bytes,
    }
    run_fingerprint = _run_fingerprint(audio_path, run_identity)
    existing = _load_manifest(manifest_path)
    states = _build_states(
        specs,
        existing,
        run_fingerprint,
        resolved_primary_engine,
        resolved_secondary_engine,
    )
    manifest = _new_manifest(audio_path, prepared_audio, run_identity, run_fingerprint, states)
    _write_manifest(manifest_path, manifest)

    processed = 0
    skipped = 0
    failed = 0

    def persist_manifest() -> None:
        _write_manifest(
            manifest_path,
            _new_manifest(audio_path, prepared_audio, run_identity, run_fingerprint, states),
        )

    if strict_fn is None:
        assert max_request_inputs is not None
        pending_states: list[ChunkState] = []
        for state in states:
            if _can_skip(
                state,
                force=force,
                primary_engine=resolved_primary_engine,
                secondary_engine=resolved_secondary_engine,
            ):
                skipped += 1
                continue

            pending_states.append(state)

        for batch_start in range(0, len(pending_states), max_request_inputs):
            batch_states = pending_states[batch_start : batch_start + max_request_inputs]
            batch_out_dirs = [chunks_dir / state.spec.chunk_id for state in batch_states]
            for state in batch_states:
                _write_chunk(
                    processing_audio_path,
                    state.spec.audio_path,
                    state.spec.start_ms,
                    state.spec.end_ms,
                )
                state.status = "running"
                state.evidence_status = "pending"
                state.evidence_failures = []
                state.error = ""
            persist_manifest()

            try:
                batch_kwargs = {
                    "out_dirs": batch_out_dirs,
                    "primary_engine": resolved_primary_engine,
                    "secondary_engine": resolved_secondary_engine,
                    "device": device,
                    "cache_dir": resolved_cache_dir,
                    "config": model_config,
                }
                if caller_binding is not None:
                    batch_kwargs["caller_binding"] = caller_binding
                batch_outputs = strict_transcribe_many(
                    [state.spec.audio_path for state in batch_states],
                    **batch_kwargs,
                )
                if len(batch_outputs) != len(batch_states):
                    raise RuntimeError(
                        "strict_transcribe_many returned "
                        f"{len(batch_outputs)} results for {len(batch_states)} chunks"
                    )
            except Exception as exc:
                for state in batch_states:
                    _mark_chunk_failed(state, f"{type(exc).__name__}: {exc}")
                    failed += 1
            else:
                for state, outputs in zip(batch_states, batch_outputs):
                    try:
                        state.outputs = {
                            key: str(path)
                            for key, path in outputs.items()
                            if isinstance(path, Path)
                        }
                        _load_chunk_objective(state)
                        (
                            state.evidence_status,
                            state.evidence_failures,
                        ) = _strict_outputs_evidence(
                            state.outputs,
                            resolved_primary_engine,
                            resolved_secondary_engine,
                        )
                        _maybe_arbitrate(state, arbiter)
                        state.status = "succeeded"
                        processed += 1
                    except Exception as exc:
                        _mark_chunk_failed(state, f"{type(exc).__name__}: {exc}")
                        failed += 1
            persist_manifest()
    else:
        for state in states:
            if _can_skip(
                state,
                force=force,
                primary_engine=resolved_primary_engine,
                secondary_engine=resolved_secondary_engine,
            ):
                skipped += 1
                continue

            _write_chunk(processing_audio_path, state.spec.audio_path, state.spec.start_ms, state.spec.end_ms)
            state.status = "running"
            state.evidence_status = "pending"
            state.evidence_failures = []
            state.error = ""
            persist_manifest()
            chunk_out_dir = chunks_dir / state.spec.chunk_id
            try:
                strict_kwargs = {
                    "primary_engine": resolved_primary_engine,
                    "secondary_engine": resolved_secondary_engine,
                    "device": device,
                    "out_dir": chunk_out_dir,
                    "cache_dir": resolved_cache_dir,
                    "config": model_config,
                }
                if caller_binding is not None:
                    strict_kwargs["caller_binding"] = caller_binding
                outputs = strict_fn(state.spec.audio_path, **strict_kwargs)
                state.outputs = {
                    key: str(path)
                    for key, path in outputs.items()
                    if isinstance(path, Path)
                }
                _load_chunk_objective(state)
                (
                    state.evidence_status,
                    state.evidence_failures,
                ) = _strict_outputs_evidence(
                    state.outputs,
                    resolved_primary_engine,
                    resolved_secondary_engine,
                )
                _maybe_arbitrate(state, arbiter)
                state.status = "succeeded"
                processed += 1
            except Exception as exc:
                _mark_chunk_failed(state, f"{type(exc).__name__}: {exc}")
                failed += 1
            finally:
                persist_manifest()

    transcript_path = out_dir / "transcript.md"
    audit_path = out_dir / "audit.md"
    metrics_path = out_dir / "metrics.json"
    _write_merged_transcript(transcript_path, audio_path, states)
    _write_merged_audit(audit_path, audio_path, states)
    _write_metrics(metrics_path, audio_path, states, processed, skipped, failed)

    objective_children = []
    for state in states:
        child = load_objective_result(state.outputs.get("objective_result"))
        if not isinstance(child, dict):
            child = {
                "schema": "media.objective-result.v1",
                "objective_outcome": "indeterminate",
                "media_kind": "audio",
                "execution": {"status": "failed" if state.status != "succeeded" else "completed"},
                "coverage": {"status": "partial"},
                "quality": {"status": "unknown"},
                "execution_status": "engine_failure" if state.status != "succeeded" else "succeeded",
                "coverage_status": "partial_coverage",
                "quality_status": "unknown",
                "chunk_id": state.spec.chunk_id,
                "idempotency_key": "",
                "audio": {
                    "coverage": {
                        "start_ms": state.spec.start_ms,
                        "end_ms": state.spec.end_ms,
                        "excluded_ranges_ms": [],
                        "complete": False,
                    }
                },
            }
        child = dict(child)
        child["chunk_id"] = state.spec.chunk_id
        child = _globalize_chunk_objective(child, state.spec)
        objective_children.append(child)
    objective_result = aggregate_objective_result(
        audio_path=audio_path,
        mode="long-strict",
        engines=[resolved_primary_engine, resolved_secondary_engine],
        children=objective_children,
        request={
            "requested_chunk_sec": chunk_sec,
            "effective_chunk_sec": effective_chunk_sec,
            "overlap_sec": overlap_sec,
            "duration_ms": round(prepared_audio.duration_sec * 1000),
            "run_fingerprint": run_fingerprint,
        },
        caller_binding=caller_binding,
    )
    objective_path = out_dir / "objective-result.json"
    write_objective_result(objective_path, objective_result)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["objective_outcome"] = objective_result["objective_outcome"]
    manifest_payload["objective_result_reference"] = objective_path.name
    _write_manifest(manifest_path, manifest_payload)
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_payload["objective_outcome"] = objective_result["objective_outcome"]
    metrics_payload["objective_result_reference"] = objective_path.name
    metrics_path.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return LongRunSummary(
        out_dir=out_dir,
        manifest_path=manifest_path,
        transcript_path=transcript_path,
        audit_path=audit_path,
        metrics_path=metrics_path,
        total=len(states),
        processed=processed,
        skipped=skipped,
        failed=failed,
        evidence_status=_aggregate_evidence_status(states),
        objective_outcome=str(objective_result["objective_outcome"]),
    )


def _wav_duration_ms(audio_path: Path) -> int:
    with wave.open(str(audio_path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    return int(round(frames / rate * 1000))


def _write_chunk(source: Path, target: Path, start_ms: int, end_ms: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as src:
        channels = src.getnchannels()
        sampwidth = src.getsampwidth()
        framerate = src.getframerate()
        start_frame = int(start_ms * framerate / 1000)
        end_frame = int(end_ms * framerate / 1000)
        src.setpos(start_frame)
        frames = src.readframes(max(0, end_frame - start_frame))
    with wave.open(str(target), "wb") as dst:
        dst.setnchannels(channels)
        dst.setsampwidth(sampwidth)
        dst.setframerate(framerate)
        dst.writeframes(frames)


def _effective_chunk_sec(
    requested_chunk_sec: int,
    *,
    recommended_chunk_sec: int | None = None,
    absolute_max_chunk_sec: int | None = None,
) -> int:
    candidates = [requested_chunk_sec]
    for label, value in (
        ("recommended_chunk_sec", recommended_chunk_sec),
        ("absolute_max_chunk_sec", absolute_max_chunk_sec),
    ):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
        candidates.append(value)
    return min(candidates)


def _engine_chunk_limits(model_config, engine_names: tuple[str, ...]) -> tuple[int | None, int | None]:
    recommended: list[int] = []
    absolute: list[int] = []
    for engine_name in engine_names:
        spec = model_config.engines.get(engine_name)
        options = getattr(spec, "options", None) or {}
        recommended_value = options.get("recommended_chunk_sec")
        absolute_value = options.get("max_audio_sec")
        if recommended_value is not None:
            recommended.append(_positive_int_option(engine_name, "recommended_chunk_sec", recommended_value))
        if absolute_value is not None:
            absolute.append(_positive_int_option(engine_name, "max_audio_sec", absolute_value))
    return (min(recommended) if recommended else None, min(absolute) if absolute else None)


def _engine_request_batch_limit(
    model_config,
    engine_names: tuple[str, ...],
    default: int = _DEFAULT_MAX_REQUEST_INPUTS,
) -> int:
    limits: list[int] = []
    for engine_name in engine_names:
        spec = model_config.engines.get(engine_name)
        options = getattr(spec, "options", None) or {}
        value = options.get("max_request_inputs", default)
        limits.append(_positive_int_option(engine_name, "max_request_inputs", value))
    return min(limits) if limits else default


def _runtime_artifact_identity(
    model_config,
    engine_names: tuple[str, ...],
    cache_dir: Path,
) -> list[dict[str, str]]:
    identities: list[dict[str, str]] = []
    for engine_name in engine_names:
        spec = model_config.engines.get(engine_name)
        if getattr(spec, "adapter", "") == "qwen-asr":
            from .qwen_identity import qwen_runtime_identity

            identities.append(
                qwen_runtime_identity(
                    spec,
                    cache_dir,
                    dict(getattr(model_config, "model_aliases", {}) or {}),
                )
            )
            continue
        options = getattr(spec, "options", None) or {}
        model_dir_value = options.get("model_dir")
        receipt_path = (
            _resolve_runtime_artifact_path(model_config.path, model_dir_value)
            / "MODEL_RECEIPT.json"
            if model_dir_value
            else None
        )
        receipt_present = bool(receipt_path and receipt_path.is_file())
        identities.append(
            {
                "engine": engine_name,
                "adapter": str(getattr(spec, "adapter", "") or ""),
                "model": str(getattr(spec, "model", "") or ""),
                "runtime_version": (
                    _installed_package_version("funasr")
                    if str(getattr(spec, "adapter", "") or "") == "funasr"
                    else ""
                ),
                "model_revision": str(options.get("model_revision") or ""),
                "source_revision": str(options.get("source_revision") or ""),
                "model_receipt_path": str(receipt_path) if receipt_path else "",
                "model_receipt_status": (
                    "present"
                    if receipt_present
                    else "missing" if receipt_path is not None else "not_configured"
                ),
                "model_receipt_sha256": (
                    sha256_file(receipt_path)
                    if receipt_present and receipt_path is not None
                    else ""
                ),
            }
        )
    return identities


def _runtime_versions(model_config, engine_names: tuple[str, ...]) -> dict[str, str]:
    """Bind cache identity to installed runtimes used by this run."""

    packages = {"modelscope", "torch", "torchaudio"}
    for engine_name in engine_names:
        spec = model_config.engines.get(engine_name)
        adapter = str(getattr(spec, "adapter", "") or "")
        if adapter == "funasr":
            packages.add("funasr")
        elif adapter == "qwen-asr":
            packages.add("qwen-asr")
    return {
        package: _installed_package_version(package)
        for package in sorted(packages)
    }


def _installed_package_version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def _runtime_code_identity(root: Path | None = None) -> dict[str, Any]:
    project = (root or Path(__file__).resolve().parents[2]).resolve()
    candidates = [
        *sorted((project / "src" / "zh_asr").rglob("*.py")),
        *sorted((project / "runtime").rglob("*.py")),
    ]
    records: list[dict[str, str]] = []
    for path in candidates:
        if not path.is_file():
            continue
        records.append(
            {
                "path": path.relative_to(project).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": "zh_asr.runtime_code_identity.v1",
        "file_count": len(records),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _resolve_runtime_artifact_path(config_path: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    resolved_config = config_path.resolve()
    config_parent = resolved_config.parent
    project_root = (
        config_parent.parent
        if config_parent.name.lower() == "configs"
        else config_parent
    )
    return (project_root / path).resolve()


def _positive_int_option(engine_name: str, option_name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Engine '{engine_name}' option '{option_name}' must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Engine '{engine_name}' option '{option_name}' must be a positive integer"
        ) from exc
    if parsed <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"Engine '{engine_name}' option '{option_name}' must be a positive integer")
    return parsed


def _run_fingerprint(audio_path: Path, run_identity: dict[str, Any]) -> str:
    audio_meta = file_metadata(audio_path)
    payload = {
        "audio_sha256": audio_meta["sha256"],
        **run_identity,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        # A torn/legacy manifest is a cache miss.  The explicit caller may
        # safely rebuild it; never treat a partial document as resumable state.
        return None
    return value if isinstance(value, dict) else None


def _build_states(
    specs: list[ChunkSpec],
    existing: dict[str, Any] | None,
    fingerprint: str,
    primary_engine: str,
    secondary_engine: str,
) -> list[ChunkState]:
    existing_chunks = {}
    if (
        existing
        and existing.get("schema_version") == 2
        and existing.get("run_fingerprint") == fingerprint
    ):
        existing_chunks = {chunk["chunk_id"]: chunk for chunk in existing.get("chunks", [])}

    states: list[ChunkState] = []
    for spec in specs:
        saved = existing_chunks.get(spec.chunk_id)
        if saved:
            status = saved.get("status", "pending")
            if status == "running":
                status = "stale"
            outputs = dict(saved.get("outputs") or {})
            if status in {"pending", "running", "stale"}:
                evidence_status, evidence_failures = "pending", []
            elif status == "succeeded":
                evidence_status, evidence_failures = _strict_outputs_evidence(
                    outputs,
                    primary_engine,
                    secondary_engine,
                )
            else:
                evidence_status = "unavailable"
                evidence_failures = [
                    {
                        "kind": "chunk_failure",
                        "error": str(
                            saved.get("error")
                            or "chunk did not produce strict evidence"
                        ),
                    }
                ]
            states.append(
                ChunkState(
                    spec=spec,
                    status=status,
                    evidence_status=evidence_status,
                    evidence_failures=evidence_failures,
                    outputs=outputs,
                    error=str(saved.get("error") or ""),
                    arbitration=saved.get("arbitration"),
                    objective_outcome=str(saved.get("objective_outcome") or "indeterminate"),
                )
            )
        else:
            states.append(ChunkState(spec=spec))
    return states


def _strict_outputs_evidence(
    outputs: dict[str, str],
    primary_engine: str,
    secondary_engine: str,
) -> tuple[str, list[dict[str, str]]]:
    return validate_strict_artifact_bundle(
        outputs,
        expected_primary_engine=primary_engine,
        expected_secondary_engine=secondary_engine,
    )


def _load_chunk_objective(state: ChunkState) -> dict[str, Any] | None:
    payload = load_objective_result(state.outputs.get("objective_result"))
    if not isinstance(payload, dict):
        state.objective_outcome = "indeterminate"
        return None
    state.objective_outcome = str(payload.get("objective_outcome") or "indeterminate")
    return payload


def _globalize_chunk_objective(
    payload: Mapping[str, Any],
    spec: ChunkSpec,
) -> dict[str, Any]:
    """Project a chunk-local sidecar interval into source-audio coordinates.

    Each strict chunk is its own WAV and therefore correctly records coverage
    from zero.  The root long sidecar must reason in source coordinates so an
    intentional overlap is not mistaken for repeated coverage of the first
    chunk only.
    """

    child = dict(payload)
    audio = child.get("audio")
    if not isinstance(audio, Mapping):
        return child
    local_coverage = audio.get("coverage")
    if not isinstance(local_coverage, Mapping):
        return child
    local_start = local_coverage.get("start_ms")
    local_end = local_coverage.get("end_ms")
    if not isinstance(local_start, (int, float)) or not isinstance(local_end, (int, float)):
        return child
    chunk_duration = float(spec.end_ms - spec.start_ms)
    is_local = (
        abs(float(local_start)) <= 1.0
        and float(local_end) <= chunk_duration + 1.0
        and (spec.start_ms != 0 or abs(float(local_end) - chunk_duration) > 1.0)
    )
    if not is_local:
        return child

    global_coverage = dict(local_coverage)
    global_coverage["start_ms"] = float(local_start) + spec.start_ms
    global_coverage["end_ms"] = float(local_end) + spec.start_ms
    intervals = local_coverage.get("intervals_ms")
    if isinstance(intervals, list):
        global_coverage["intervals_ms"] = [
            [float(item[0]) + spec.start_ms, float(item[1]) + spec.start_ms]
            for item in intervals
            if isinstance(item, (list, tuple))
            and len(item) >= 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], (int, float))
        ]
    child["chunk_interval_ms"] = [spec.start_ms, spec.end_ms]
    child["chunk_local_coverage"] = dict(local_coverage)
    child["audio"] = {**dict(audio), "coverage": global_coverage}

    detection = child.get("detection")
    if isinstance(detection, Mapping):
        segments = detection.get("segments")
        if isinstance(segments, list):
            global_segments = [
                [float(item[0]) + spec.start_ms, float(item[1]) + spec.start_ms]
                for item in segments
                if isinstance(item, (list, tuple))
                and len(item) >= 2
                and isinstance(item[0], (int, float))
                and isinstance(item[1], (int, float))
            ]
            child["detection_global"] = {
                **dict(detection),
                "segments": global_segments,
            }
    return child


def _mark_chunk_failed(state: ChunkState, error: str) -> None:
    state.status = "failed"
    state.evidence_status = "unavailable"
    state.evidence_failures = [{"kind": "chunk_failure", "error": error}]
    state.error = error


def _aggregate_evidence_status(states: list[ChunkState]) -> str:
    if not states:
        return "unavailable"
    if any(
        state.status in {"pending", "running", "stale"}
        or state.evidence_status == "pending"
        for state in states
    ):
        return "pending"
    if any(
        state.status != "succeeded" or state.evidence_status == "unavailable"
        for state in states
    ):
        return "unavailable"
    if any(state.evidence_status == "provisional" for state in states):
        return "provisional"
    if all(state.evidence_status == "verified" for state in states):
        return "verified"
    return "unavailable"


def _aggregate_evidence_failures(
    states: list[ChunkState],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    for state in states:
        for item in state.evidence_failures:
            failures.append({"chunk_id": state.spec.chunk_id, **item})
    return failures


def _new_manifest(
    audio_path: Path,
    prepared_audio: PreparedAudio,
    run_identity: dict[str, Any],
    fingerprint: str,
    states: list[ChunkState],
) -> dict[str, Any]:
    meta = file_metadata(audio_path)
    return {
        "schema_version": 2,
        "audio": str(audio_path.resolve()),
        "audio_sha256": meta["sha256"],
        "audio_size_bytes": meta["size_bytes"],
        "prepared_audio": prepared_audio.as_dict(),
        "chunk_sec": run_identity["effective_chunk_sec"],
        **run_identity,
        "run_fingerprint": fingerprint,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evidence_status": _aggregate_evidence_status(states),
        "evidence_failures": _aggregate_evidence_failures(states),
        "chunks": [state.to_dict() for state in states],
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _write_json_atomic(path, manifest)


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


def _can_skip(
    state: ChunkState,
    force: bool,
    primary_engine: str,
    secondary_engine: str,
) -> bool:
    if force or state.status != "succeeded":
        return False
    evidence_status, evidence_failures = _strict_outputs_evidence(
        state.outputs,
        primary_engine,
        secondary_engine,
    )
    state.evidence_status = evidence_status
    state.evidence_failures = evidence_failures
    if evidence_status not in {"verified", "provisional"}:
        state.status = "stale"
        state.error = "Persisted strict evidence failed fresh bundle verification"
        return False
    objective_payload = load_objective_result(state.outputs.get("objective_result"))
    objective_failures = validate_objective_result(objective_payload)
    if objective_failures:
        state.status = "stale"
        state.error = "Persisted objective result failed sidecar verification"
        state.objective_outcome = "indeterminate"
        return False
    state.objective_outcome = str(objective_payload.get("objective_outcome") or "indeterminate")
    return not any(
        failure.get("kind") == "artifact_failure"
        for failure in evidence_failures
    )


def _maybe_arbitrate(state: ChunkState, arbiter) -> None:
    if not arbiter or not state.outputs.get("audit_json"):
        return
    audit_json = Path(state.outputs["audit_json"])
    if not audit_json.exists():
        return
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    if not _needs_arbitration(audit):
        return
    evidence = ArbitrationEvidence.from_audit(
        chunk_id=state.spec.chunk_id,
        time_range=f"{_fmt_ms(state.spec.start_ms)}-{_fmt_ms(state.spec.end_ms)}",
        audit=audit,
    )
    decision = arbiter.arbitrate(evidence)
    if decision:
        state.arbitration = decision.to_dict() if hasattr(decision, "to_dict") else decision


def _needs_arbitration(audit: dict[str, Any]) -> bool:
    flags = set(audit.get("flags", []) or [])
    if flags:
        return True
    if audit.get("needs_review"):
        return True
    return float(audit.get("similarity", 1.0) or 1.0) < 0.72


def _write_merged_transcript(path: Path, audio_path: Path, states: list[ChunkState]) -> None:
    lines = [
        f"# {audio_path.stem} Long Transcript",
        "",
        f"- Audio: `{audio_path}`",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Transcript",
        "",
    ]
    previous_state: ChunkState | None = None
    previous_raw_text = ""
    for state in states:
        raw_text = _chunk_text(state)
        text = raw_text
        removed_overlap_chars = 0
        if (
            previous_state is not None
            and state.spec.start_ms < previous_state.spec.end_ms
            and previous_raw_text
            and raw_text
        ):
            removed_overlap_chars = _exact_boundary_overlap_size(previous_raw_text, raw_text)
            text = _remove_exact_boundary_overlap(
                previous_raw_text,
                raw_text,
                overlap_size=removed_overlap_chars,
            )
        rendered_text = text or (
            "[与上一分块精确重叠]" if raw_text and previous_raw_text else "[听不清]"
        )
        lines.extend(
            [
                f"### {state.spec.chunk_id} [{_fmt_ms(state.spec.start_ms)} - {_fmt_ms(state.spec.end_ms)}]",
                "",
            ]
        )
        if removed_overlap_chars:
            lines.extend(
                [
                    f"<!-- exact-boundary-overlap-removed: {removed_overlap_chars} chars -->",
                    "",
                ]
            )
        lines.extend([rendered_text, ""])
        previous_state = state
        previous_raw_text = raw_text
    path.write_text("\n".join(lines), encoding="utf-8")


def _exact_boundary_overlap_size(previous_text: str, current_text: str, min_chars: int = 2) -> int:
    """Measure only a literal suffix/prefix match, never a fuzzy or semantic match."""
    previous = previous_text.rstrip()
    current = current_text.lstrip()
    max_overlap = min(len(previous), len(current))
    for size in range(max_overlap, min_chars - 1, -1):
        if previous[-size:] == current[:size]:
            return size
    return 0


def _remove_exact_boundary_overlap(
    previous_text: str,
    current_text: str,
    *,
    overlap_size: int | None = None,
) -> str:
    size = (
        _exact_boundary_overlap_size(previous_text, current_text)
        if overlap_size is None
        else overlap_size
    )
    if size:
        return current_text.lstrip()[size:].lstrip()
    return current_text


def _write_merged_audit(path: Path, audio_path: Path, states: list[ChunkState]) -> None:
    lines = [
        f"# {audio_path.stem} Long Audit",
        "",
        f"- Audio: `{audio_path}`",
        f"- Chunks: `{len(states)}`",
        "",
    ]
    for state in states:
        lines.extend(
            [
                f"## {state.spec.chunk_id} [{_fmt_ms(state.spec.start_ms)} - {_fmt_ms(state.spec.end_ms)}]",
                "",
                f"- Status: `{state.status}`",
                f"- Evidence status: `{state.evidence_status}`",
                f"- Error: `{state.error or 'none'}`",
            ]
        )
        if state.arbitration:
            lines.append(f"- Arbitration: `{json.dumps(state.arbitration, ensure_ascii=False)}`")
        audit_path = state.outputs.get("audit")
        if audit_path and Path(audit_path).exists():
            lines.extend(["", Path(audit_path).read_text(encoding="utf-8").strip(), ""])
        else:
            lines.extend(["", "_No audit output_", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_metrics(path: Path, audio_path: Path, states: list[ChunkState], processed: int, skipped: int, failed: int) -> None:
    payload = {
        "schema_version": 1,
        "audio": str(audio_path.resolve()),
        "total": len(states),
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "evidence_status": _aggregate_evidence_status(states),
        "evidence_failures": _aggregate_evidence_failures(states),
        "chunks": [state.to_dict() for state in states],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _chunk_text(state: ChunkState) -> str:
    if state.arbitration and state.arbitration.get("final_text"):
        return str(state.arbitration["final_text"]).strip()
    final = state.outputs.get("final")
    if not final or not Path(final).exists():
        return ""
    raw = Path(final).read_text(encoding="utf-8")
    match = re.search(r"## Transcript\s+(.*)", raw, flags=re.S)
    return (match.group(1) if match else raw).strip()


def _fmt_ms(value: int) -> str:
    total_seconds = value // 1000
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

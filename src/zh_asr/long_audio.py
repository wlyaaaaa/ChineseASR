from __future__ import annotations

import hashlib
import json
import re
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Any

from .arbitration import ArbitrationEvidence
from .config import load_model_config
from .metadata import file_metadata, sha256_file
from .pipeline import default_cache_dir, strict_transcribe_audio


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
    outputs: dict[str, str] = field(default_factory=dict)
    error: str = ""
    arbitration: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = self.spec.to_dict()
        payload.update(
            {
                "status": self.status,
                "outputs": self.outputs,
                "error": self.error,
                "arbitration": self.arbitration,
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


StrictFn = Callable[..., dict[str, Path]]


def plan_chunks(audio_path: Path, chunk_sec: int = 300, overlap_sec: int = 1, chunks_dir: Path | None = None) -> list[ChunkSpec]:
    if chunk_sec <= 0:
        raise ValueError("chunk_sec must be greater than 0")
    if overlap_sec < 0:
        raise ValueError("overlap_sec must be greater than or equal to 0")
    if overlap_sec >= chunk_sec:
        raise ValueError("overlap_sec must be smaller than chunk_sec")

    duration_ms = _wav_duration_ms(audio_path)
    chunks_root = chunks_dir or audio_path.parent / "chunks"
    chunk_ms = chunk_sec * 1000
    step_ms = (chunk_sec - overlap_sec) * 1000
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
    strict_fn: StrictFn = strict_transcribe_audio,
    arbiter=None,
) -> LongRunSummary:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    out_dir = out_dir.resolve()
    chunks_dir = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    specs = plan_chunks(audio_path.resolve(), chunk_sec=chunk_sec, overlap_sec=overlap_sec, chunks_dir=chunks_dir)
    manifest_path = out_dir / "manifest.json"
    model_config = load_model_config()
    run_fingerprint = _run_fingerprint(audio_path, chunk_sec, overlap_sec, model_config.path)
    existing = _load_manifest(manifest_path)
    states = _build_states(specs, existing, run_fingerprint)
    manifest = _new_manifest(audio_path, chunk_sec, overlap_sec, run_fingerprint, states)
    _write_manifest(manifest_path, manifest)

    processed = 0
    skipped = 0
    failed = 0
    for state in states:
        if _can_skip(state, force=force):
            skipped += 1
            continue

        _write_chunk(audio_path, state.spec.audio_path, state.spec.start_ms, state.spec.end_ms)
        state.status = "running"
        state.error = ""
        _write_manifest(manifest_path, _new_manifest(audio_path, chunk_sec, overlap_sec, run_fingerprint, states))
        chunk_out_dir = chunks_dir / state.spec.chunk_id
        try:
            outputs = strict_fn(
                state.spec.audio_path,
                primary_engine=primary_engine or model_config.strict_primary_engine,
                secondary_engine=secondary_engine or model_config.strict_secondary_engine,
                device=device,
                out_dir=chunk_out_dir,
                cache_dir=cache_dir or default_cache_dir(),
                config=model_config,
            )
            state.outputs = {key: str(path) for key, path in outputs.items() if isinstance(path, Path)}
            _maybe_arbitrate(state, arbiter)
            state.status = "succeeded"
            processed += 1
        except Exception as exc:
            state.status = "failed"
            state.error = f"{type(exc).__name__}: {exc}"
            failed += 1
        finally:
            _write_manifest(manifest_path, _new_manifest(audio_path, chunk_sec, overlap_sec, run_fingerprint, states))

    transcript_path = out_dir / "transcript.md"
    audit_path = out_dir / "audit.md"
    metrics_path = out_dir / "metrics.json"
    _write_merged_transcript(transcript_path, audio_path, states)
    _write_merged_audit(audit_path, audio_path, states)
    _write_metrics(metrics_path, audio_path, states, processed, skipped, failed)

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


def _run_fingerprint(audio_path: Path, chunk_sec: int, overlap_sec: int, model_config_path: Path) -> str:
    audio_meta = file_metadata(audio_path)
    payload = {
        "audio_sha256": audio_meta["sha256"],
        "chunk_sec": chunk_sec,
        "overlap_sec": overlap_sec,
        "model_config_sha256": sha256_file(model_config_path) if model_config_path.exists() else "",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_states(specs: list[ChunkSpec], existing: dict[str, Any] | None, fingerprint: str) -> list[ChunkState]:
    existing_chunks = {}
    if existing and existing.get("run_fingerprint") == fingerprint:
        existing_chunks = {chunk["chunk_id"]: chunk for chunk in existing.get("chunks", [])}

    states: list[ChunkState] = []
    for spec in specs:
        saved = existing_chunks.get(spec.chunk_id)
        if saved:
            status = saved.get("status", "pending")
            if status == "running":
                status = "stale"
            states.append(
                ChunkState(
                    spec=spec,
                    status=status,
                    outputs=dict(saved.get("outputs") or {}),
                    error=str(saved.get("error") or ""),
                    arbitration=saved.get("arbitration"),
                )
            )
        else:
            states.append(ChunkState(spec=spec))
    return states


def _new_manifest(audio_path: Path, chunk_sec: int, overlap_sec: int, fingerprint: str, states: list[ChunkState]) -> dict[str, Any]:
    meta = file_metadata(audio_path)
    return {
        "schema_version": 1,
        "audio": str(audio_path.resolve()),
        "audio_sha256": meta["sha256"],
        "audio_size_bytes": meta["size_bytes"],
        "chunk_sec": chunk_sec,
        "overlap_sec": overlap_sec,
        "run_fingerprint": fingerprint,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "chunks": [state.to_dict() for state in states],
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _can_skip(state: ChunkState, force: bool) -> bool:
    if force or state.status != "succeeded":
        return False
    final = state.outputs.get("final")
    audit = state.outputs.get("audit")
    return bool(final and audit and Path(final).exists() and Path(audit).exists())


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
    for state in states:
        text = _chunk_text(state)
        lines.extend(
            [
                f"### {state.spec.chunk_id} [{_fmt_ms(state.spec.start_ms)} - {_fmt_ms(state.spec.end_ms)}]",
                "",
                text or "[听不清]",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


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

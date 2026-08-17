from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from .audio_outcome import load_objective_result, validate_objective_result
from .config import ModelConfig, load_model_config
from .pipeline import strict_transcribe_audio, transcribe_audio


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}


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
    failed_path.write_text("", encoding="utf-8")

    items: list[BatchItem] = []
    for audio_path in find_audio_files(input_root):
        item_dir = output_root / _output_dir_name(input_root, audio_path)
        expected = _expected_output_path(item_dir, audio_path, mode_key, quick_engine)

        if expected.exists() and not force:
            objective_outcome = _read_objective_outcome(item_dir, audio_path)
            items.append(
                BatchItem(
                    audio=audio_path,
                    out_dir=item_dir,
                    status="skipped",
                    message="already complete",
                    objective_outcome=objective_outcome,
                )
            )
            continue

        item_dir.mkdir(parents=True, exist_ok=True)
        try:
            if mode_key == "strict":
                outputs = strict_fn(
                    audio_path,
                    primary_engine=strict_primary,
                    secondary_engine=strict_secondary,
                    device=device,
                    out_dir=item_dir,
                    cache_dir=cache_dir,
                    config=model_config,
                )
            else:
                outputs = transcribe_fn(
                    audio_path,
                    engine=quick_engine,
                    device=device,
                    out_dir=item_dir,
                    cache_dir=cache_dir,
                    config=model_config,
                )
            items.append(
                BatchItem(
                    audio=audio_path,
                    out_dir=item_dir,
                    status="processed",
                    objective_outcome=_result_objective_outcome(outputs, item_dir, audio_path),
                )
            )
        except Exception as exc:
            items.append(
                BatchItem(
                    audio=audio_path,
                    out_dir=item_dir,
                    status="failed",
                    message=str(exc),
                    objective_outcome="indeterminate",
                )
            )
            _append_failure(failed_path, audio_path, item_dir, exc)

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

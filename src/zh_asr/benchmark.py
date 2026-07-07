from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .batch import AUDIO_EXTENSIONS
from .config import ModelConfig
from .eval_pack import EvalSummary, run_evaluation
from .pipeline import strict_transcribe_audio


def build_benchmark_manifest(audio_dir: Path, truth_dir: Path, manifest_dir: Path, force: bool = False) -> Path:
    audio_root = audio_dir.resolve()
    truth_root = truth_dir.resolve()
    manifest_root = manifest_dir.resolve()
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_root}")
    if not audio_root.is_dir():
        raise NotADirectoryError(f"Audio path is not a directory: {audio_root}")
    if not truth_root.exists():
        raise FileNotFoundError(f"Truth directory not found: {truth_root}")
    if not truth_root.is_dir():
        raise NotADirectoryError(f"Truth path is not a directory: {truth_root}")

    if force and manifest_root.exists():
        shutil.rmtree(manifest_root)
    manifest_root.mkdir(parents=True, exist_ok=True)

    cases = []
    for audio_path in _find_audio_files(audio_root):
        case_id = _case_id(audio_root, audio_path)
        truth_path = truth_root / f"{audio_path.stem}.txt"
        case = {
            "id": case_id,
            "category": "benchmark",
            "kind": audio_path.suffix.lower().lstrip("."),
            "audio": str(audio_path),
            "truth": str(truth_path),
            "truth_text": "",
            "expect_empty": False,
            "available": True,
            "notes": "User-provided benchmark audio matched to human truth by filename stem.",
        }
        if truth_path.exists():
            case["truth_text"] = truth_path.read_text(encoding="utf-8").strip()
        else:
            case["available"] = False
            case["error"] = f"Missing truth file: {truth_path}"
        cases.append(case)

    manifest = {
        "version": 1,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "description": "User-provided ChineseASR benchmark manifest.",
        "audio_dir": str(audio_root),
        "truth_dir": str(truth_root),
        "cases": cases,
    }
    manifest_path = manifest_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def run_benchmark(
    audio_dir: Path,
    truth_dir: Path,
    out_dir: Path,
    device: str = "cuda:0",
    cache_dir: Path | None = None,
    force: bool = False,
    primary_engine: str | None = None,
    secondary_engine: str | None = None,
    config: ModelConfig | None = None,
    strict_fn=strict_transcribe_audio,
) -> EvalSummary:
    output_root = out_dir.resolve()
    manifest_dir = output_root / "_manifest"
    manifest_path = build_benchmark_manifest(audio_dir, truth_dir, manifest_dir, force=force)
    summary = run_evaluation(
        corpus_dir=manifest_dir,
        out_dir=output_root,
        device=device,
        cache_dir=cache_dir,
        force=force,
        primary_engine=primary_engine,
        secondary_engine=secondary_engine,
        config=config,
        strict_fn=strict_fn,
    )
    _write_benchmark_json(output_root / "benchmark.json", output_root / "metrics.json", manifest_path)
    return summary


def _find_audio_files(audio_dir: Path) -> list[Path]:
    return sorted(
        (path for path in audio_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS),
        key=lambda path: str(path.relative_to(audio_dir)).lower(),
    )


def _case_id(audio_dir: Path, audio_path: Path) -> str:
    relative = audio_path.relative_to(audio_dir).with_suffix("")
    return "__".join(relative.parts)


def _write_benchmark_json(path: Path, metrics_path: Path, manifest_path: Path) -> None:
    payload: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truth_hash_by_id = {
        str(case["id"]): _sha256(Path(case["truth"])) if Path(case["truth"]).exists() else ""
        for case in manifest.get("cases", [])
    }
    payload["benchmark"] = {
        "audio_dir": manifest.get("audio_dir", ""),
        "truth_dir": manifest.get("truth_dir", ""),
        "manifest": str(manifest_path),
    }
    for case in payload.get("cases", []):
        case_id = str(case.get("id", ""))
        case["truth_sha256"] = truth_hash_by_id.get(case_id, "")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

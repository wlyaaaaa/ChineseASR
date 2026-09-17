"""File-only orchestration: bounded duration, profiles and optional audio review.

Dictation does not import this module. A selected channel retains its original
recording identity; channel numbers do not certify speaker identity.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from .config import ModelConfig, load_model_config, resolve_profile
from .audio_quality import probe_duration_ms


def chunk_budget(config: ModelConfig, primary: str, secondary: str) -> int:
    limits = [int(config.quality.get("max_chunk_sec", 300))]
    if config.quality.get("align") and config.alignment:
        limits.append(int(config.alignment.get("max_audio_sec", 300)))
    for name in (primary, secondary):
        options = config.engines[name].options or {}
        for key in ("recommended_chunk_sec", "max_audio_sec"):
            if options.get(key):
                limits.append(int(options[key]))
    if min(limits) < 1:
        raise ValueError("File chunk limits must be at least one second")
    return min(limits)


def needs_long_route(audio: Path, config: ModelConfig, primary: str, secondary: str) -> bool:
    return probe_duration_ms(audio) > chunk_budget(config, primary, secondary) * 1000


def transcribe_file(audio_path: Path, *, out_dir: Path, primary_engine: str | None = None,
                    secondary_engine: str | None = None, profile: str | None = None,
                    device: str = "cuda:0", cache_dir: Path | None = None,
                    config: ModelConfig | None = None, caller_binding=None,
                    channel_index: int | None = None, force: bool = False,
                    strict_fn=None) -> dict[str, Any]:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    config = config or load_model_config()
    primary, secondary = resolve_profile(config, profile, primary_engine, secondary_engine)
    if needs_long_route(audio_path, config, primary, secondary):
        from .long_audio import run_long_transcription
        summary = run_long_transcription(audio_path, out_dir, chunk_sec=chunk_budget(config, primary, secondary),
            primary_engine=primary, secondary_engine=secondary, device=device,
            cache_dir=cache_dir, force=force, caller_binding=caller_binding,
            config=config, channel_index=channel_index)
        paths = {"final": summary.transcript_path, "transcript": summary.transcript_path,
            "audit": summary.audit_path, "metrics": summary.metrics_path,
            "manifest": summary.manifest_path, "objective_result": out_dir / "objective-result.json",
            "objective_outcome": summary.objective_outcome, "evidence_status": summary.evidence_status,
            "failed_chunks": summary.failed, "resolved_mode": "long-strict"}
    else:
        from .pipeline import strict_transcribe_audio, default_cache_dir
        operation = strict_fn or strict_transcribe_audio
        kwargs = dict(primary_engine=primary, secondary_engine=secondary, device=device,
            out_dir=out_dir, cache_dir=cache_dir, config=config, caller_binding=caller_binding)
        if channel_index is not None:
            kwargs["channel_index"] = channel_index
        paths = operation(audio_path, **kwargs)
        paths["resolved_mode"] = "strict"
        if config.quality and paths.get("audit_json") and Path(paths["audit_json"]).is_file():
            from .quality_review import enhance_single
            review_audio = audio_path
            if channel_index is not None:
                from .audio_quality import extract_channel
                review_audio, _ = extract_channel(audio_path, channel_index, out_dir / "_derived" / "channels")
            enhance_single(review_audio, paths, config=config, device=device,
                cache_dir=cache_dir or default_cache_dir(), output_dir=out_dir)
    for key, filename in (("quality_review", "quality.review.json"), ("review_html", "quality.review.html")):
        if (out_dir / filename).is_file():
            paths[key] = out_dir / filename
    return paths

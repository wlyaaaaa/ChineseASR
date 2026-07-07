from __future__ import annotations

import importlib.util
import gc
import os
import time
from pathlib import Path
from typing import Any

from .adapters import get_adapter
from .adapters.base import MissingDependencyError
from .adapters.funasr import ensure_funasr_available, funasr_kwargs as _funasr_kwargs
from .config import ModelConfig, get_engine_spec, load_model_config
from .proxy_guard import sanitize_current_process_env
from .result_writer import write_transcript_bundle
from .strict_writer import write_strict_bundle


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_cache_dir() -> Path:
    return project_root() / "models" / "modelscope"


def prepare_model_env(cache_dir: Path | None = None) -> Path:
    sanitize_current_process_env()
    cache = cache_dir or default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MODELSCOPE_CACHE"] = str(cache)
    os.environ.setdefault("PYTHONUTF8", "1")
    return cache


def build_model(
    engine: str,
    device: str = "cuda:0",
    cache_dir: Path | None = None,
    config: ModelConfig | None = None,
) -> Any:
    model_config = config or load_model_config()
    spec = get_engine_spec(engine, config=model_config)
    if spec.is_whisper:
        raise ValueError(f"Engine '{engine}' is fallback/comparison only and cannot be used for direct transcription.")
    cache = prepare_model_env(cache_dir)
    if spec.adapter == "funasr":
        ensure_funasr_available()

    adapter = get_adapter(spec.adapter)
    return adapter.build_model(spec, device, cache, model_config.model_aliases)


def transcribe_audio(
    audio_path: Path,
    engine: str | None = None,
    device: str = "cuda:0",
    out_dir: Path | None = None,
    cache_dir: Path | None = None,
    config: ModelConfig | None = None,
) -> dict[str, Path]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    model_config = config or load_model_config()
    engine_name = engine or model_config.default_engine
    model = build_model(engine_name, device=device, cache_dir=cache_dir, config=model_config)
    result = model.generate(input=str(audio_path), batch_size_s=300)
    return write_transcript_bundle(audio_path, result, out_dir or project_root() / "outputs", engine_name)


def strict_transcribe_audio(
    audio_path: Path,
    primary_engine: str | None = None,
    secondary_engine: str | None = None,
    device: str = "cuda:0",
    out_dir: Path | None = None,
    cache_dir: Path | None = None,
    config: ModelConfig | None = None,
) -> dict[str, Any]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    model_config = config or load_model_config()
    primary_name = primary_engine or model_config.strict_primary_engine
    secondary_name = secondary_engine or model_config.strict_secondary_engine
    total_started = time.perf_counter()
    primary_started = time.perf_counter()
    primary_result = _generate_once(audio_path, primary_name, device, cache_dir, model_config)
    primary_sec = time.perf_counter() - primary_started
    secondary_started = time.perf_counter()
    secondary_result = _generate_once(audio_path, secondary_name, device, cache_dir, model_config)
    secondary_sec = time.perf_counter() - secondary_started
    paths = write_strict_bundle(
        audio_path=audio_path,
        primary_engine=primary_name,
        primary_result=primary_result,
        secondary_engine=secondary_name,
        secondary_result=secondary_result,
        out_dir=out_dir or project_root() / "outputs",
    )
    paths["timing"] = {
        "total_sec": time.perf_counter() - total_started,
        "primary_sec": primary_sec,
        "secondary_sec": secondary_sec,
    }
    return paths


def _generate_once(audio_path: Path, engine: str, device: str, cache_dir: Path | None, config: ModelConfig) -> Any:
    model = build_model(engine, device=device, cache_dir=cache_dir, config=config)
    try:
        return model.generate(input=str(audio_path), batch_size_s=300)
    finally:
        del model
        gc.collect()
        _empty_cuda_cache()


def _empty_cuda_cache() -> None:
    if importlib.util.find_spec("torch") is None:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return

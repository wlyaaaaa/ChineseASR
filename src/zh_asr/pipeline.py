from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from .config import EngineSpec, get_engine_spec
from .proxy_guard import sanitize_current_process_env
from .result_writer import write_transcript_bundle


class MissingDependencyError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_cache_dir() -> Path:
    return project_root() / "models" / "modelscope"


def ensure_funasr_available() -> None:
    if os.environ.get("ZH_ASR_TEST_FORCE_MISSING_FUNASR") == "1":
        raise MissingDependencyError(
            "FunASR is not installed. Run scripts\\setup-core.ps1 after installing CUDA PyTorch."
        )
    if importlib.util.find_spec("funasr") is None:
        raise MissingDependencyError(
            "FunASR is not installed. Run scripts\\setup-core.ps1 after installing CUDA PyTorch."
        )


def prepare_model_env(cache_dir: Path | None = None) -> None:
    sanitize_current_process_env()
    cache = cache_dir or default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MODELSCOPE_CACHE"] = str(cache)
    os.environ.setdefault("PYTHONUTF8", "1")


def build_model(engine: str, device: str = "cuda:0", cache_dir: Path | None = None) -> Any:
    spec = get_engine_spec(engine)
    if spec.is_whisper:
        raise ValueError("Whisper is fallback/comparison only in this project; use sensevoice or paraformer.")
    ensure_funasr_available()
    prepare_model_env(cache_dir)

    from funasr import AutoModel

    kwargs = _funasr_kwargs(spec, device)
    return AutoModel(**kwargs)


def transcribe_audio(
    audio_path: Path,
    engine: str = "sensevoice",
    device: str = "cuda:0",
    out_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    model = build_model(engine, device=device, cache_dir=cache_dir)
    result = model.generate(input=str(audio_path), batch_size_s=300)
    return write_transcript_bundle(audio_path, result, out_dir or project_root() / "outputs", engine)


def _funasr_kwargs(spec: EngineSpec, device: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": spec.model,
        "device": device,
        "disable_update": True,
    }
    if spec.vad_model:
        kwargs["vad_model"] = spec.vad_model
        kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
    if spec.punc_model:
        kwargs["punc_model"] = spec.punc_model
    if spec.spk_model:
        kwargs["spk_model"] = spec.spk_model
    return kwargs

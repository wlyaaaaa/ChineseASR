from __future__ import annotations

import importlib.util
import gc
import os
from pathlib import Path
from typing import Any

from .config import EngineSpec, get_engine_spec
from .proxy_guard import sanitize_current_process_env
from .result_writer import write_transcript_bundle
from .strict_writer import write_strict_bundle


class MissingDependencyError(RuntimeError):
    pass


MODEL_ALIASES = {
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "ct-punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    "cam++": "iic/speech_campplus_sv_zh-cn_16k-common",
}


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


def prepare_model_env(cache_dir: Path | None = None) -> Path:
    sanitize_current_process_env()
    cache = cache_dir or default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MODELSCOPE_CACHE"] = str(cache)
    os.environ.setdefault("PYTHONUTF8", "1")
    return cache


def build_model(engine: str, device: str = "cuda:0", cache_dir: Path | None = None) -> Any:
    spec = get_engine_spec(engine)
    if spec.is_whisper:
        raise ValueError("Whisper is fallback/comparison only in this project; use sensevoice or paraformer.")
    ensure_funasr_available()
    cache = prepare_model_env(cache_dir)

    from funasr import AutoModel

    kwargs = _funasr_kwargs(spec, device, cache)
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


def strict_transcribe_audio(
    audio_path: Path,
    primary_engine: str = "sensevoice",
    secondary_engine: str = "paraformer",
    device: str = "cuda:0",
    out_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    primary_result = _generate_once(audio_path, primary_engine, device, cache_dir)
    secondary_result = _generate_once(audio_path, secondary_engine, device, cache_dir)
    return write_strict_bundle(
        audio_path=audio_path,
        primary_engine=primary_engine,
        primary_result=primary_result,
        secondary_engine=secondary_engine,
        secondary_result=secondary_result,
        out_dir=out_dir or project_root() / "outputs",
    )


def _funasr_kwargs(spec: EngineSpec, device: str, cache_dir: Path | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": _resolve_model_ref(spec.model, cache_dir),
        "device": device,
        "disable_update": True,
    }
    if spec.vad_model:
        kwargs["vad_model"] = _resolve_model_ref(spec.vad_model, cache_dir)
        kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
    if spec.punc_model:
        kwargs["punc_model"] = _resolve_model_ref(spec.punc_model, cache_dir)
    if spec.spk_model:
        kwargs["spk_model"] = _resolve_model_ref(spec.spk_model, cache_dir)
    return kwargs


def _resolve_model_ref(model_ref: str, cache_dir: Path | None) -> str:
    canonical = MODEL_ALIASES.get(model_ref, model_ref)
    if cache_dir and "/" in canonical:
        local = cache_dir / Path(*canonical.split("/"))
        if local.exists():
            return str(local)
    return model_ref


def _generate_once(audio_path: Path, engine: str, device: str, cache_dir: Path | None) -> Any:
    model = build_model(engine, device=device, cache_dir=cache_dir)
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

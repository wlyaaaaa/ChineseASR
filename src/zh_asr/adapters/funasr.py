from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from zh_asr.config import EngineSpec
from zh_asr.adapters.base import MissingDependencyError


_MODEL_OPTION_KEYS = {
    "hub",
    "model_revision",
    "remote_code",
    "trust_remote_code",
}


class FunASRAdapter:
    name = "funasr"

    def build_model(
        self,
        spec: EngineSpec,
        device: str,
        cache_dir: Path,
        model_aliases: dict[str, str],
    ) -> Any:
        from funasr import AutoModel

        return AutoModel(**funasr_kwargs(spec, device, cache_dir, model_aliases))


def ensure_funasr_available() -> None:
    if os.environ.get("ZH_ASR_TEST_FORCE_MISSING_FUNASR") == "1":
        raise MissingDependencyError(
            "FunASR is not installed. Run scripts\\setup-core.ps1 after installing CUDA PyTorch."
        )
    if importlib.util.find_spec("funasr") is None:
        raise MissingDependencyError(
            "FunASR is not installed. Run scripts\\setup-core.ps1 after installing CUDA PyTorch."
        )


def funasr_kwargs(
    spec: EngineSpec,
    device: str,
    cache_dir: Path | None = None,
    model_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    aliases = model_aliases or {}
    kwargs: dict[str, Any] = {
        "model": resolve_model_ref(spec.model, cache_dir, aliases),
        "device": device,
        "disable_update": True,
    }
    if spec.vad_model:
        kwargs["vad_model"] = resolve_model_ref(spec.vad_model, cache_dir, aliases)
        kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
    if spec.punc_model:
        kwargs["punc_model"] = resolve_model_ref(spec.punc_model, cache_dir, aliases)
    if spec.spk_model:
        kwargs["spk_model"] = resolve_model_ref(spec.spk_model, cache_dir, aliases)
    # Only pass the small, documented loader options used by custom FunASR
    # checkpoints.  Registry metadata such as ``requires_gpu`` and ``runtime``
    # must not become arbitrary AutoModel kwargs.
    options = spec.options or {}
    for key in _MODEL_OPTION_KEYS:
        if key not in options:
            continue
        value = options[key]
        if key == "trust_remote_code":
            kwargs[key] = bool(value)
        elif value is not None and str(value).strip():
            kwargs[key] = str(value).strip()
    return kwargs


def resolve_model_ref(model_ref: str, cache_dir: Path | None, model_aliases: dict[str, str]) -> str:
    canonical = model_aliases.get(model_ref, model_ref)
    if cache_dir and "/" in canonical:
        local = cache_dir / Path(*canonical.split("/"))
        if local.exists():
            return str(local)
    return canonical

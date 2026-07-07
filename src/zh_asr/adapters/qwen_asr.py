from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from zh_asr.adapters.base import MissingDependencyError
from zh_asr.config import EngineSpec


class QwenASRAdapter:
    name = "qwen-asr"

    def build_model(
        self,
        spec: EngineSpec,
        device: str,
        cache_dir: Path,
        model_aliases: dict[str, str],
    ) -> Any:
        ensure_qwen_asr_available()
        from qwen_asr import Qwen3ASRModel

        kwargs = qwen_from_pretrained_kwargs(spec, device, cache_dir, model_aliases)
        model_ref = kwargs.pop("model")
        model = Qwen3ASRModel.from_pretrained(model_ref, **kwargs)
        return self.wrap_model(model, spec)

    def wrap_model(self, model: Any, spec: EngineSpec) -> "QwenASRModelWrapper":
        return QwenASRModelWrapper(model, language=spec.language)


class QwenASRModelWrapper:
    def __init__(self, model: Any, language: str) -> None:
        self.model = model
        self.language = None if language.lower() == "auto" else language

    def generate(self, input: str, **_: Any) -> list[dict[str, Any]]:
        results = self.model.transcribe(audio=input, language=self.language)
        return [_normalize_qwen_result(item) for item in results]


def ensure_qwen_asr_available() -> None:
    if os.environ.get("ZH_ASR_TEST_FORCE_MISSING_QWEN_ASR") == "1":
        raise MissingDependencyError(
            "Qwen ASR is not installed. Run scripts\\setup-qwen.ps1 before warming qwen3-asr-1.7b."
        )
    if importlib.util.find_spec("qwen_asr") is None:
        raise MissingDependencyError(
            "Qwen ASR is not installed. Run scripts\\setup-qwen.ps1 before warming qwen3-asr-1.7b."
        )


def qwen_from_pretrained_kwargs(
    spec: EngineSpec,
    device: str,
    cache_dir: Path | None,
    model_aliases: dict[str, str],
) -> dict[str, Any]:
    options = spec.options or {}
    model_ref = _resolve_required_local_qwen_model(spec.model, cache_dir, model_aliases)
    kwargs: dict[str, Any] = {
        "model": model_ref,
        "device_map": device,
    }
    if "dtype" in options:
        kwargs["dtype"] = _torch_dtype(str(options["dtype"]))
    for key in ("max_inference_batch_size", "max_new_tokens"):
        if key in options:
            kwargs[key] = options[key]
    return kwargs


def _resolve_required_local_qwen_model(
    model_ref: str,
    cache_dir: Path | None,
    model_aliases: dict[str, str],
) -> str:
    canonical = model_aliases.get(model_ref, model_ref)
    if cache_dir and "/" in canonical:
        local = cache_dir / Path(*canonical.split("/"))
        if local.exists():
            return str(local)
        raise FileNotFoundError(
            f"Qwen ASR model cache not found: {local}. "
            "Run scripts\\download-models.ps1 -Engine qwen3-asr-1.7b to prefetch from ModelScope."
        )
    return canonical


def _normalize_qwen_result(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        text = str(item.get("text", ""))
        language = item.get("language")
    else:
        text = str(getattr(item, "text", ""))
        language = getattr(item, "language", None)
    normalized: dict[str, Any] = {"text": text}
    if language:
        normalized["language"] = str(language)
    return normalized


def _torch_dtype(name: str) -> Any:
    import torch

    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype for qwen-asr: {name}") from exc

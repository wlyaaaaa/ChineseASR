from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

from zh_asr.adapters.base import MissingDependencyError
from zh_asr.config import EngineSpec
from zh_asr.qwen_identity import (
    QWEN_RUNTIME_DISTRIBUTION,
    QWEN_RUNTIME_VERSION,
    qwen_runtime_identity,
    require_runtime_version,
)
from zh_asr.text_normalizer import to_simplified


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
        kwargs, runtime_identity = _prepare_qwen_runtime(
            spec, device, cache_dir, model_aliases
        )
        from qwen_asr import Qwen3ASRModel

        model_ref = kwargs.pop("model")
        model = Qwen3ASRModel.from_pretrained(model_ref, **kwargs)
        return self.wrap_model(model, spec, runtime_identity=runtime_identity)

    def wrap_model(
        self,
        model: Any,
        spec: EngineSpec,
        *,
        runtime_identity: dict[str, str] | None = None,
    ) -> "QwenASRModelWrapper":
        options = spec.options or {}
        context = str(
            options.get(
                "context",
                "请只输出简体中文转写文本，不要输出繁体字、解释、翻译或额外说明。",
            )
        )
        return QwenASRModelWrapper(
            model,
            language=spec.language,
            context=context,
            runtime_identity=runtime_identity,
        )


class QwenASRModelWrapper:
    def __init__(
        self,
        model: Any,
        language: str,
        context: str,
        runtime_identity: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.language = None if language.lower() == "auto" else language
        self.context = context
        self.runtime_identity = dict(runtime_identity or {})

    def generate(self, input: str, **_: Any) -> list[dict[str, Any]]:
        results = self.model.transcribe(audio=input, context=self.context, language=self.language)
        return [_normalize_qwen_result(item) for item in results]


def ensure_qwen_asr_available(
    distribution: str = QWEN_RUNTIME_DISTRIBUTION,
    required_version: str = QWEN_RUNTIME_VERSION,
) -> None:
    if os.environ.get("ZH_ASR_TEST_FORCE_MISSING_QWEN_ASR") == "1":
        raise MissingDependencyError(
            "Qwen ASR is not installed. Run scripts\\setup-qwen.ps1 before warming qwen3-asr-1.7b."
        )
    if importlib.util.find_spec("qwen_asr") is None:
        raise MissingDependencyError(
            "Qwen ASR is not installed. Run scripts\\setup-qwen.ps1 before warming qwen3-asr-1.7b."
        )
    require_runtime_version(distribution, required_version)


def qwen_from_pretrained_kwargs(
    spec: EngineSpec,
    device: str,
    cache_dir: Path | None,
    model_aliases: dict[str, str],
) -> dict[str, Any]:
    kwargs, _ = _prepare_qwen_runtime(spec, device, cache_dir, model_aliases)
    return kwargs


def _prepare_qwen_runtime(
    spec: EngineSpec,
    device: str,
    cache_dir: Path | None,
    model_aliases: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    options = spec.options or {}
    identity = qwen_runtime_identity(spec, cache_dir, model_aliases)
    kwargs: dict[str, Any] = {
        "model": identity["model_dir"],
        "device_map": device,
    }
    if "dtype" in options:
        kwargs["dtype"] = _torch_dtype(str(options["dtype"]))
    for key in ("max_inference_batch_size", "max_new_tokens"):
        if key in options:
            kwargs[key] = options[key]
    return kwargs, identity


def _normalize_qwen_result(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        original_text = str(item.get("text", ""))
        language = item.get("language")
    else:
        original_text = str(getattr(item, "text", ""))
        language = getattr(item, "language", None)
    text = to_simplified(original_text)
    normalized: dict[str, Any] = {"text": text}
    if language:
        normalized["language"] = str(language)
    if original_text != text:
        normalized["original_text"] = original_text
    return normalized


def _torch_dtype(name: str) -> Any:
    import torch

    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype for qwen-asr: {name}") from exc

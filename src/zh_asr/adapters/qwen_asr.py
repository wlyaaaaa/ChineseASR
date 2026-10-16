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
from zh_asr.inference_quality import isolated_batch, generation_diagnostics


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
        configure_qwen_audio_attention(model)
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
        normalized = [_normalize_qwen_result(item) for item in results]
        for item in normalized:
            item["generation"] = generation_diagnostics(self.model, item["text"])
        return normalized

    def generate_many(self, inputs: list[str], **_: Any) -> list[Any]:
        def batch(values):
            results = self.model.transcribe(audio=values, context=self.context, language=self.language)
            normalized = [_normalize_qwen_result(item) for item in results]
            for item in normalized:
                item["generation"] = generation_diagnostics(self.model, item["text"])
            return [[item] for item in normalized]
        return isolated_batch(inputs, batch, self.generate)


def configure_qwen_audio_attention(model: Any) -> None:
    """Preserve packed audio windows in the pinned qwen-asr SDPA path."""
    audio_config = model.model.thinker.audio_tower.config
    if audio_config._attn_implementation != "sdpa":
        return
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    name = "zh_asr_qwen_audio_sdpa"
    ALL_ATTENTION_FUNCTIONS.register(name, _qwen_audio_window_sdpa)
    audio_config._attn_implementation = name


def _qwen_audio_window_sdpa(module, query, key, value, attention_mask=None, **kwargs):
    import torch
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    cu_seqlens = kwargs.get("cu_seq_lens_q")
    if cu_seqlens is None or len(cu_seqlens) <= 2:
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    boundaries = cu_seqlens.tolist()
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        mask = None if attention_mask is None else attention_mask[..., start:end, start:end]
        output, _ = sdpa_attention_forward(
            module, query[:, :, start:end], key[:, :, start:end], value[:, :, start:end],
            mask, **kwargs,
        )
        outputs.append(output)
    return torch.cat(outputs, dim=1), None


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

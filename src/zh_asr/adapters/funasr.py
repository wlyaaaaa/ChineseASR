from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any
import wave

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


def detect_speech_segments(model: Any, audio_path: str | Path) -> dict[str, Any]:
    """Run only the already-loaded FunASR VAD for an empty ASR result.

    ``AutoModel.generate`` internally uses ``inference_with_vad`` when a VAD
    model is configured, but it discards the distinction between zero VAD
    segments and VAD segments whose ASR text is empty.  Calling the same VAD
    submodel directly avoids loading a second model and records that
    distinction for the objective-result sidecar.  A missing/failed VAD is
    explicitly unavailable rather than evidence of silence.
    """

    vad_model = getattr(model, "vad_model", None)
    inference = getattr(model, "inference", None)
    vad_kwargs = getattr(model, "vad_kwargs", None)
    if vad_model is None or not callable(inference):
        return {
            "status": "unavailable",
            "reason": "vad_not_configured",
            "segments": [],
            "coverage_complete": False,
        }
    try:
        kwargs = dict(vad_kwargs or {})
        # AutoModel mutates the kwargs dictionary during inference; passing a
        # copy preserves the model's normal runtime configuration for callers.
        response = inference(
            str(audio_path),
            model=vad_model,
            kwargs=kwargs,
        )
        segments = _normalize_vad_segments(response)
        duration_ms = _wav_duration_ms(Path(audio_path))
        coverage_complete = duration_ms is not None and duration_ms > 0
        return {
            "status": "no_speech_detected" if not segments and coverage_complete else "speech_detected" if segments else "unavailable",
            "reason": "vad_zero_segments" if not segments and coverage_complete else "vad_segments_present" if segments else "vad_coverage_unknown",
            "segments": segments,
            "coverage_complete": coverage_complete,
            "coverage": {
                "start_ms": 0,
                "end_ms": duration_ms,
                "excluded_ranges_ms": [],
                "complete": coverage_complete,
            },
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"vad_failed:{type(exc).__name__}",
            "error": str(exc),
            "segments": [],
            "coverage_complete": False,
        }


def _normalize_vad_segments(value: Any) -> list[list[int | float]]:
    """Normalize FunASR VAD ``value`` arrays to ``[[start_ms, end_ms], ...]``."""

    if isinstance(value, dict):
        values = value.get("value")
        if values is None:
            values = value.get("segments")
        return _normalize_vad_segments(values)
    if isinstance(value, list):
        if value and all(isinstance(item, (int, float)) for item in value[:2]):
            if len(value) >= 2:
                return [[value[0], value[1]]]
            return []
        segments: list[list[int | float]] = []
        for item in value:
            segments.extend(_normalize_vad_segments(item))
        return segments
    return []


def _wav_duration_ms(path: Path) -> int | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            if rate <= 0:
                return None
            return round(handle.getnframes() * 1000 / rate)
    except (OSError, EOFError, wave.Error):
        return None


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

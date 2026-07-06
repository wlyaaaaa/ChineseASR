from __future__ import annotations

from dataclasses import dataclass


DEFAULT_ENGINE = "sensevoice"


@dataclass(frozen=True)
class EngineSpec:
    name: str
    role: str
    model: str
    vad_model: str | None = None
    punc_model: str | None = None
    spk_model: str | None = None
    language: str = "auto"
    is_whisper: bool = False
    note: str = ""


ENGINES: dict[str, EngineSpec] = {
    "sensevoice": EngineSpec(
        name="sensevoice",
        role="primary",
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        spk_model="cam++",
        language="auto",
        note="Primary local Chinese low-hallucination engine.",
    ),
    "paraformer": EngineSpec(
        name="paraformer",
        role="baseline",
        model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        spk_model="cam++",
        language="zh",
        note="Conservative Mandarin/CN baseline for cross-checking.",
    ),
    "whisper-large-v3": EngineSpec(
        name="whisper-large-v3",
        role="fallback",
        model="openai/whisper-large-v3",
        language="zh",
        is_whisper=True,
        note="Fallback/comparison only; not the primary Chinese low-hallucination path.",
    ),
}


def get_engine_spec(name: str) -> EngineSpec:
    key = name.strip().lower()
    try:
        return ENGINES[key]
    except KeyError as exc:
        known = ", ".join(sorted(ENGINES))
        raise ValueError(f"Unknown ASR engine '{name}'. Known engines: {known}") from exc


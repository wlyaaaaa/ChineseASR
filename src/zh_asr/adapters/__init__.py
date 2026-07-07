from __future__ import annotations

from .base import ModelAdapter
from .funasr import FunASRAdapter


ADAPTERS: dict[str, ModelAdapter] = {
    "funasr": FunASRAdapter(),
}


def get_adapter(name: str) -> ModelAdapter:
    key = name.strip().lower()
    try:
        return ADAPTERS[key]
    except KeyError as exc:
        known = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unsupported ASR adapter '{name}'. Known adapters: {known}") from exc

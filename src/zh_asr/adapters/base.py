from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from zh_asr.config import EngineSpec


class MissingDependencyError(RuntimeError):
    pass


class ModelAdapter(Protocol):
    name: str

    def build_model(
        self,
        spec: EngineSpec,
        device: str,
        cache_dir: Path,
        model_aliases: dict[str, str],
    ) -> Any:
        ...

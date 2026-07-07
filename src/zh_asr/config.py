from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


MODEL_CONFIG_ENV = "ZH_ASR_MODEL_CONFIG"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_config_path() -> Path:
    return project_root() / "configs" / "models.yaml"


@dataclass(frozen=True)
class EngineSpec:
    name: str
    adapter: str
    role: str
    model: str
    vad_model: str | None = None
    punc_model: str | None = None
    spk_model: str | None = None
    language: str = "auto"
    is_whisper: bool = False
    note: str = ""
    options: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    default_engine: str
    strict_primary_engine: str
    strict_secondary_engine: str
    model_aliases: dict[str, str]
    engines: dict[str, EngineSpec]


def load_model_config(path: Path | str | None = None) -> ModelConfig:
    config_path = Path(path) if path else _configured_model_config_path()
    data = _read_yaml(config_path)
    engines = _parse_engines(data.get("engines", {}), config_path)
    if not engines:
        raise ValueError(f"No ASR engines configured in {config_path}")

    defaults = _mapping(data.get("defaults", {}), "defaults", config_path)
    strict = _mapping(data.get("strict", {}), "strict", config_path)
    aliases = {str(key): str(value) for key, value in _mapping(data.get("aliases", {}), "aliases", config_path).items()}

    default_engine = str(defaults.get("engine", "sensevoice")).strip()
    strict_primary = str(strict.get("primary_engine", default_engine)).strip()
    strict_secondary = str(strict.get("secondary_engine", strict_primary)).strip()

    for selected in (default_engine, strict_primary, strict_secondary):
        if selected not in engines:
            known = ", ".join(sorted(engines))
            raise ValueError(f"Configured ASR engine '{selected}' is not defined in {config_path}. Known engines: {known}")

    return ModelConfig(
        path=config_path,
        default_engine=default_engine,
        strict_primary_engine=strict_primary,
        strict_secondary_engine=strict_secondary,
        model_aliases=aliases,
        engines=engines,
    )


def get_engine_spec(name: str, config: ModelConfig | None = None) -> EngineSpec:
    model_config = config or MODEL_CONFIG
    key = name.strip().lower()
    try:
        return model_config.engines[key]
    except KeyError as exc:
        known = ", ".join(sorted(model_config.engines))
        raise ValueError(f"Unknown ASR engine '{name}'. Known engines: {known}") from exc


def list_engine_names(config: ModelConfig | None = None) -> tuple[str, ...]:
    model_config = config or MODEL_CONFIG
    return tuple(sorted(model_config.engines))


def list_transcription_engine_names(config: ModelConfig | None = None) -> tuple[str, ...]:
    model_config = config or MODEL_CONFIG
    return tuple(sorted(name for name, spec in model_config.engines.items() if not spec.is_whisper))


def _configured_model_config_path() -> Path:
    configured = os.environ.get(MODEL_CONFIG_ENV)
    return Path(configured) if configured else default_model_config_path()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read configs\\models.yaml. Run scripts\\setup-core.ps1.") from exc

    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _mapping(data, "root", path)


def _parse_engines(raw: Any, path: Path) -> dict[str, EngineSpec]:
    engines: dict[str, EngineSpec] = {}
    for name, values in _mapping(raw, "engines", path).items():
        key = str(name).strip().lower()
        item = _mapping(values, f"engines.{name}", path)
        engines[key] = EngineSpec(
            name=key,
            adapter=str(item.get("adapter", "funasr")).strip().lower(),
            role=str(item.get("role", "candidate")).strip(),
            model=_required_str(item, "model", f"engines.{name}", path),
            vad_model=_optional_str(item.get("vad_model")),
            punc_model=_optional_str(item.get("punc_model")),
            spk_model=_optional_str(item.get("spk_model")),
            language=str(item.get("language", "auto")).strip(),
            is_whisper=bool(item.get("is_whisper", False)),
            note=str(item.get("note", "")).strip(),
            options=dict(_mapping(item.get("options", {}), f"engines.{name}.options", path)),
        )
    return engines


def _mapping(value: Any, label: str, path: Path) -> dict[Any, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for {label} in {path}")
    return value


def _required_str(item: dict[Any, Any], key: str, label: str, path: Path) -> str:
    value = item.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required '{key}' in {label} of {path}")
    return str(value).strip()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


MODEL_CONFIG = load_model_config()
DEFAULT_ENGINE = MODEL_CONFIG.default_engine
ENGINES = MODEL_CONFIG.engines

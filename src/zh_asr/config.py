from __future__ import annotations

from dataclasses import dataclass, field
import math
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
class SpeakerVerificationSpec:
    """Narrow local configuration for the optional ``person:self`` anchor."""

    model_alias: str
    model_revision: str
    model_file: str
    threshold: float


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    default_engine: str
    strict_primary_engine: str
    strict_secondary_engine: str
    model_aliases: dict[str, str]
    engines: dict[str, EngineSpec]
    speaker_verification: SpeakerVerificationSpec | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    alignment: dict[str, Any] = field(default_factory=dict)
    profiles: dict[str, Any] = field(default_factory=dict)


def load_model_config(path: Path | str | None = None) -> ModelConfig:
    config_path = Path(path) if path else _configured_model_config_path()
    data = _read_yaml(config_path)
    engines = _parse_engines(data.get("engines", {}), config_path)
    if not engines:
        raise ValueError(f"No ASR engines configured in {config_path}")

    defaults = _mapping(data.get("defaults", {}), "defaults", config_path)
    strict = _mapping(data.get("strict", {}), "strict", config_path)
    aliases = {str(key): str(value) for key, value in _mapping(data.get("aliases", {}), "aliases", config_path).items()}
    speaker_verification = _parse_speaker_verification(
        data.get("speaker_verification"),
        aliases,
        config_path,
    )

    default_engine = str(defaults.get("engine", "sensevoice")).strip()
    strict_primary = str(strict.get("primary_engine", default_engine)).strip()
    strict_secondary = str(strict.get("secondary_engine", strict_primary)).strip()

    for selected in (default_engine, strict_primary, strict_secondary):
        if selected not in engines:
            known = ", ".join(sorted(engines))
            raise ValueError(f"Configured ASR engine '{selected}' is not defined in {config_path}. Known engines: {known}")

    quality, alignment, profiles = _parse_quality_configuration(data, engines, config_path)

    return ModelConfig(
        path=config_path,
        default_engine=default_engine,
        strict_primary_engine=strict_primary,
        strict_secondary_engine=strict_secondary,
        model_aliases=aliases,
        engines=engines,
        speaker_verification=speaker_verification,
        quality=quality,
        alignment=alignment,
        profiles=profiles,
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
    from .adapters import ADAPTERS
    return tuple(sorted(name for name, spec in model_config.engines.items()
                        if spec.adapter in ADAPTERS and not spec.is_whisper))


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


def _parse_speaker_verification(
    raw: Any,
    aliases: dict[str, str],
    path: Path,
) -> SpeakerVerificationSpec | None:
    """Parse an optional, deliberately self-only speaker-verification model."""

    if raw is None:
        return None
    item = _mapping(raw, "speaker_verification", path)
    model_alias = _required_str(item, "model_alias", "speaker_verification", path)
    if model_alias not in aliases:
        raise ValueError(
            f"speaker_verification.model_alias '{model_alias}' is not defined in aliases of {path}"
        )
    model_revision = _required_str(item, "model_revision", "speaker_verification", path)
    model_file = _required_str(item, "model_file", "speaker_verification", path)
    model_file_path = Path(model_file)
    if model_file_path.is_absolute() or len(model_file_path.parts) != 1:
        raise ValueError("speaker_verification.model_file must name one file in the model directory")
    try:
        threshold = float(item.get("threshold"))
    except (TypeError, ValueError) as exc:
        raise ValueError("speaker_verification.threshold must be a finite number in [-1, 1]") from exc
    if not math.isfinite(threshold) or threshold < -1 or threshold > 1:
        raise ValueError("speaker_verification.threshold must be a finite number in [-1, 1]")
    return SpeakerVerificationSpec(
        model_alias=model_alias,
        model_revision=model_revision,
        model_file=model_file,
        threshold=threshold,
    )


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


def _parse_quality_configuration(data, engines, path):
    quality = dict(_mapping(data.get("quality", {}), "quality", path))
    alignment = dict(_mapping(data.get("alignment", {}), "alignment", path))
    profiles = dict(_mapping(data.get("profiles", {}), "profiles", path))
    if quality.get("cut_strategy", "fixed") not in {"fixed", "vad"}:
        raise ValueError("quality.cut_strategy must be fixed or vad")
    for key in ("min_chunk_sec", "boundary_search_sec", "review_context_sec", "max_chunk_sec"):
        if key in quality:
            value = quality[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"quality.{key} must be a finite nonnegative number")
            if key in {"min_chunk_sec", "max_chunk_sec"} and value < 1:
                raise ValueError(f"quality.{key} must be at least one second")
    if "align" in quality and not isinstance(quality["align"], bool):
        raise ValueError("quality.align must be a boolean")
    terms = quality.get("critical_terms", [])
    if not isinstance(terms, list) or any(not isinstance(x, str) or not x.strip() for x in terms):
        raise ValueError("quality.critical_terms must be a list of nonempty strings")
    reviewer = quality.get("review_engine")
    if reviewer and reviewer not in engines:
        raise ValueError("quality.review_engine must identify a configured engine")
    for name, profile in profiles.items():
        profile = _mapping(profile, f"profiles.{name}", path)
        primary, secondary = profile.get("primary_engine"), profile.get("secondary_engine")
        if primary not in engines or secondary not in engines or primary == secondary:
            raise ValueError(f"profiles.{name} requires two different configured engines")
        if any(engines[x].is_whisper for x in (primary, secondary)):
            raise ValueError(f"profiles.{name} selects an unintegrated engine")
    return quality, alignment, profiles


def resolve_profile(config: ModelConfig, name: str | None = None,
                    primary: str | None = None, secondary: str | None = None) -> tuple[str, str]:
    profile = {}
    if name:
        if name not in config.profiles:
            raise ValueError(f"Unknown ASR profile: {name}")
        profile = config.profiles[name]
    first = primary or profile.get("primary_engine") or config.strict_primary_engine
    second = secondary or profile.get("secondary_engine") or config.strict_secondary_engine
    for value in (first, second):
        if value not in config.engines or config.engines[value].is_whisper:
            raise ValueError(f"Profile selects an unavailable transcription engine: {value}")
    if first == second:
        raise ValueError("Strict ASR requires two different engines")
    return first, second


MODEL_CONFIG = load_model_config()
DEFAULT_ENGINE = MODEL_CONFIG.default_engine
ENGINES = MODEL_CONFIG.engines

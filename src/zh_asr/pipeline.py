from __future__ import annotations

import importlib.util
import gc
import os
import time
from pathlib import Path
from typing import Any, Mapping

from .adapters import get_adapter
from .adapters.base import MissingDependencyError  # noqa: F401 - compatibility re-export
from .adapters.funasr import (  # noqa: F401 - compatibility re-export
    detect_speech_segments,
    ensure_funasr_available,
    funasr_kwargs as _funasr_kwargs,
)
from .audio_frontend import (
    PreparedAudio,
    _locked_prepared_audio_owner,
    prepare_pcm16_mono,
    validate_prepared_audio_owner,
)
from .config import ModelConfig, get_engine_spec, load_model_config
from .proxy_guard import sanitize_current_process_env
from .result_writer import canonical_json_sha256, extract_text, write_transcript_bundle
from .strict_writer import write_strict_bundle


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_cache_dir() -> Path:
    return project_root() / "models" / "modelscope"


def prepare_model_env(cache_dir: Path | None = None) -> Path:
    sanitize_current_process_env()
    cache = cache_dir or default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MODELSCOPE_CACHE"] = str(cache)
    os.environ.setdefault("PYTHONUTF8", "1")
    return cache


def build_model(
    engine: str,
    device: str = "cuda:0",
    cache_dir: Path | None = None,
    config: ModelConfig | None = None,
) -> Any:
    model_config = config or load_model_config()
    spec = get_engine_spec(engine, config=model_config)
    if spec.is_whisper:
        raise ValueError(f"Engine '{engine}' is fallback/comparison only and cannot be used for direct transcription.")
    cache = prepare_model_env(cache_dir)
    if spec.adapter == "funasr":
        ensure_funasr_available()

    adapter = get_adapter(spec.adapter)
    return adapter.build_model(spec, device, cache, model_config.model_aliases)


def transcribe_audio(
    audio_path: Path,
    engine: str | None = None,
    device: str = "cuda:0",
    out_dir: Path | None = None,
    cache_dir: Path | None = None,
    config: ModelConfig | None = None,
    caller_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    model_config = config or load_model_config()
    engine_name = engine or model_config.default_engine
    output_dir = out_dir or project_root() / "outputs"
    engine_audio, provenance = _prepare_engine_input(
        audio_path,
        engine_name,
        output_dir / "_derived",
        model_config,
    )
    result, runtime_identity = _generate_once_with_identity(
        engine_audio,
        engine_name,
        device,
        cache_dir,
        model_config,
    )
    if runtime_identity:
        provenance["runtime_identity"] = runtime_identity
    return write_transcript_bundle(
        audio_path,
        result,
        output_dir,
        engine_name,
        primary_provenance=provenance,
        caller_binding=caller_binding,
    )


def strict_transcribe_audio(
    audio_path: Path,
    primary_engine: str | None = None,
    secondary_engine: str | None = None,
    device: str = "cuda:0",
    out_dir: Path | None = None,
    cache_dir: Path | None = None,
    config: ModelConfig | None = None,
    expect_empty: bool = False,
    caller_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    model_config = config or load_model_config()
    primary_name = primary_engine or model_config.strict_primary_engine
    secondary_name = secondary_engine or model_config.strict_secondary_engine
    output_dir = out_dir or project_root() / "outputs"
    derived_dir = output_dir / "_derived"
    total_started = time.perf_counter()
    if _uses_shared_default_strict_audio(primary_name, secondary_name, model_config):
        prepared = prepare_pcm16_mono(
            audio_path,
            derived_dir,
            materialize_owner=True,
        )
        audio_provenance = prepared.as_dict()
        with _locked_prepared_audio_owner(prepared, derived_dir):
            primary_result, primary_error, primary_sec, primary_provenance = (
                _generate_prepared_for_strict(
                    prepared,
                    audio_provenance,
                    primary_name,
                    device,
                    cache_dir,
                    model_config,
                    derived_dir,
                )
            )
            secondary_result, secondary_error, secondary_sec, secondary_provenance = (
                _generate_prepared_for_strict(
                    prepared,
                    audio_provenance,
                    secondary_name,
                    device,
                    cache_dir,
                    model_config,
                    derived_dir,
                )
            )
    else:
        primary_result, primary_error, primary_sec, primary_provenance = _generate_for_strict(
            audio_path, primary_name, device, cache_dir, model_config, derived_dir
        )
        secondary_result, secondary_error, secondary_sec, secondary_provenance = _generate_for_strict(
            audio_path, secondary_name, device, cache_dir, model_config, derived_dir
        )
    paths = write_strict_bundle(
        audio_path=audio_path,
        primary_engine=primary_name,
        primary_result=primary_result,
        secondary_engine=secondary_name,
        secondary_result=secondary_result,
        out_dir=output_dir,
        expect_empty=expect_empty,
        primary_error=primary_error,
        secondary_error=secondary_error,
        primary_role="lexical_primary",
        secondary_role="lexical_verifier",
        primary_provenance=primary_provenance,
        secondary_provenance=secondary_provenance,
        caller_binding=caller_binding,
    )
    paths["timing"] = {
        "total_sec": time.perf_counter() - total_started,
        "primary_sec": primary_sec,
        "secondary_sec": secondary_sec,
    }
    return paths


def strict_transcribe_many(
    audio_paths: list[Path],
    *,
    out_dirs: list[Path],
    primary_engine: str | None = None,
    secondary_engine: str | None = None,
    device: str = "cuda:0",
    cache_dir: Path | None = None,
    config: ModelConfig | None = None,
    expect_empty: bool = False,
    caller_binding: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if len(audio_paths) != len(out_dirs):
        raise ValueError("audio_paths and out_dirs must have the same length")
    if not audio_paths:
        return []
    missing = [path for path in audio_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Audio file not found: {missing[0]}")

    model_config = config or load_model_config()
    primary_name = primary_engine or model_config.strict_primary_engine
    secondary_name = secondary_engine or model_config.strict_secondary_engine
    total_started = time.perf_counter()
    primary = _generate_many_for_strict(
        audio_paths,
        out_dirs,
        primary_name,
        device,
        cache_dir,
        model_config,
    )
    secondary = _generate_many_for_strict(
        audio_paths,
        out_dirs,
        secondary_name,
        device,
        cache_dir,
        model_config,
    )

    bundles: list[dict[str, Any]] = []
    for index, audio_path in enumerate(audio_paths):
        paths = write_strict_bundle(
            audio_path=audio_path,
            primary_engine=primary_name,
            primary_result=primary["results"][index],
            secondary_engine=secondary_name,
            secondary_result=secondary["results"][index],
            out_dir=out_dirs[index],
            expect_empty=expect_empty,
            primary_error=primary["errors"][index],
            secondary_error=secondary["errors"][index],
            primary_role="lexical_primary",
            secondary_role="lexical_verifier",
            primary_provenance=primary["provenance"][index],
            secondary_provenance=secondary["provenance"][index],
            caller_binding=caller_binding,
        )
        paths["timing"] = {
            "total_sec": time.perf_counter() - total_started,
            "primary_batch_sec": primary["elapsed_sec"],
            "secondary_batch_sec": secondary["elapsed_sec"],
        }
        bundles.append(paths)
    return bundles


def _generate_many_for_strict(
    audio_paths: list[Path],
    out_dirs: list[Path],
    engine: str,
    device: str,
    cache_dir: Path | None,
    config: ModelConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    count = len(audio_paths)
    results: list[Any | None] = [None] * count
    errors: list[str | None] = [None] * count
    provenance: list[dict[str, Any]] = [_engine_provenance(engine, config) for _ in audio_paths]
    prepared: list[Path | None] = [None] * count

    for index, (audio_path, out_dir) in enumerate(zip(audio_paths, out_dirs)):
        try:
            prepared[index], provenance[index] = _prepare_engine_input(
                audio_path,
                engine,
                out_dir / "_derived",
                config,
            )
        except Exception as exc:
            errors[index] = f"{type(exc).__name__}: {exc}"
            results[index] = _engine_failure_result(engine, exc)

    valid_indices = [index for index, path in enumerate(prepared) if path is not None]
    if valid_indices:
        model = None
        try:
            model = build_model(engine, device=device, cache_dir=cache_dir, config=config)
            runtime_identity = _model_runtime_identity(model)
            if runtime_identity:
                for index in valid_indices:
                    provenance[index]["runtime_identity"] = runtime_identity
            generate_many = getattr(model, "generate_many", None)
            if callable(generate_many):
                generated = generate_many([str(prepared[index]) for index in valid_indices])
                if not isinstance(generated, list) or len(generated) != len(valid_indices):
                    raise RuntimeError(
                        f"Engine '{engine}' generate_many returned "
                        f"{len(generated) if isinstance(generated, list) else type(generated).__name__} "
                        f"for {len(valid_indices)} inputs."
                    )
                for index, value in zip(valid_indices, generated):
                    normalized = value if isinstance(value, list) else [value]
                    results[index] = _attach_speech_detection_if_empty(
                        normalized,
                        model,
                        prepared[index],
                        engine,
                        config,
                    )
            else:
                for index in valid_indices:
                    try:
                        generated = model.generate(
                            input=str(prepared[index]),
                            batch_size_s=300,
                        )
                        results[index] = _attach_speech_detection_if_empty(
                            generated,
                            model,
                            prepared[index],
                            engine,
                            config,
                        )
                    except Exception as exc:
                        errors[index] = f"{type(exc).__name__}: {exc}"
                        results[index] = _engine_failure_result(engine, exc)
        except Exception as exc:
            for index in valid_indices:
                if results[index] is None:
                    errors[index] = f"{type(exc).__name__}: {exc}"
                    results[index] = _engine_failure_result(engine, exc)
        finally:
            if model is not None:
                del model
            gc.collect()
            _empty_cuda_cache()

    return {
        "results": [
            value if value is not None else {"engine": engine, "text": ""}
            for value in results
        ],
        "errors": errors,
        "provenance": provenance,
        "elapsed_sec": time.perf_counter() - started,
    }


def _generate_for_strict(
    audio_path: Path,
    engine: str,
    device: str,
    cache_dir: Path | None,
    config: ModelConfig,
    derived_dir: Path,
) -> tuple[Any, str | None, float, dict[str, Any]]:
    started = time.perf_counter()
    provenance = _engine_provenance(engine, config)
    try:
        engine_audio, provenance = _prepare_engine_input(
            audio_path,
            engine,
            derived_dir,
            config,
        )
        result, runtime_identity = _generate_once_with_identity(
            engine_audio,
            engine,
            device,
            cache_dir,
            config,
        )
        if runtime_identity:
            provenance["runtime_identity"] = runtime_identity
        return result, None, time.perf_counter() - started, provenance
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return _engine_failure_result(engine, exc), error, time.perf_counter() - started, provenance


def _generate_prepared_for_strict(
    prepared: PreparedAudio,
    audio_provenance: dict[str, object],
    engine: str,
    device: str,
    cache_dir: Path | None,
    config: ModelConfig,
    derived_dir: Path,
) -> tuple[Any, str | None, float, dict[str, Any]]:
    started = time.perf_counter()
    provenance = _engine_provenance(engine, config)
    provenance["audio"] = dict(audio_provenance)
    validate_prepared_audio_owner(prepared, derived_dir)
    try:
        result, runtime_identity = _generate_once_with_identity(
            prepared.path,
            engine,
            device,
            cache_dir,
            config,
        )
        if runtime_identity:
            provenance["runtime_identity"] = runtime_identity
        error = None
    except Exception as exc:
        result = _engine_failure_result(engine, exc)
        error = f"{type(exc).__name__}: {exc}"
    validate_prepared_audio_owner(prepared, derived_dir)
    return result, error, time.perf_counter() - started, provenance


def _engine_failure_result(engine: str, exc: Exception) -> dict[str, Any]:
    return {
        "engine": engine,
        "text": "",
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def _generate_once(audio_path: Path, engine: str, device: str, cache_dir: Path | None, config: ModelConfig) -> Any:
    result, _ = _generate_once_with_identity(
        audio_path,
        engine,
        device,
        cache_dir,
        config,
    )
    return result


def _generate_once_with_identity(
    audio_path: Path,
    engine: str,
    device: str,
    cache_dir: Path | None,
    config: ModelConfig,
) -> tuple[Any, dict[str, Any]]:
    model = build_model(engine, device=device, cache_dir=cache_dir, config=config)
    try:
        result = model.generate(input=str(audio_path), batch_size_s=300)
        result = _attach_speech_detection_if_empty(
            result,
            model,
            audio_path,
            engine,
            config,
        )
        return result, _model_runtime_identity(model)
    finally:
        del model
        gc.collect()
        _empty_cuda_cache()


def _model_runtime_identity(model: Any) -> dict[str, Any]:
    identity = getattr(model, "runtime_identity", None)
    return dict(identity) if isinstance(identity, dict) else {}


def _attach_speech_detection_if_empty(
    result: Any,
    model: Any,
    audio_path: Path | None,
    engine: str,
    config: ModelConfig,
) -> Any:
    """Attach same-lifecycle FunASR VAD evidence without adding an engine."""

    if extract_text(result) or audio_path is None:
        return result
    spec = get_engine_spec(engine, config=config)
    if spec.adapter != "funasr":
        return result
    detection = detect_speech_segments(model, audio_path)
    resolved_vad_model = (
        config.model_aliases.get(spec.vad_model, spec.vad_model)
        if spec.vad_model
        else None
    )
    # FunASR mutates ``model.vad_kwargs`` during inference and can leave
    # runtime-only objects such as ``WavFrontendOnline`` in that mapping.
    # Objective evidence must bind stable JSON configuration, never live
    # model objects that make the raw/result sidecars unserializable.
    stable_vad_kwargs: dict[str, Any] = {}
    for key, value in dict(getattr(model, "vad_kwargs", {}) or {}).items():
        try:
            canonical_json_sha256(value)
        except (TypeError, ValueError, OverflowError):
            continue
        stable_vad_kwargs[str(key)] = value
    vad_config = {
        "engine": engine,
        "vad_model": resolved_vad_model,
        "vad_kwargs": stable_vad_kwargs,
    }
    detection.update(
        {
            "processor": "funasr-vad",
            "processor_version": "funasr-auto-model",
            "model": resolved_vad_model,
            "config": vad_config,
            "config_sha256": canonical_json_sha256(vad_config),
        }
    )
    if isinstance(result, dict):
        updated = dict(result)
        updated["speech_detection"] = detection
        return updated
    if isinstance(result, list):
        if result and isinstance(result[0], dict):
            updated = list(result)
            first = dict(updated[0])
            first["speech_detection"] = detection
            updated[0] = first
            return updated
        return [{"text": "", "speech_detection": detection}]
    return {
        "text": "",
        "result": result,
        "speech_detection": detection,
    }


def _prepare_engine_input(
    audio_path: Path,
    engine: str,
    derived_dir: Path,
    config: ModelConfig,
) -> tuple[Path, dict[str, Any]]:
    spec = get_engine_spec(engine, config=config)
    provenance = _engine_provenance(engine, config)
    options = spec.options or {}
    if not bool(options.get("requires_pcm16_mono", False)):
        provenance["audio"] = {
            "source_path": str(audio_path.resolve()),
            "path": str(audio_path.resolve()),
            "converted": False,
        }
        return audio_path, provenance

    prepared = prepare_pcm16_mono(audio_path, derived_dir)
    max_audio_sec = float(options.get("max_audio_sec", 0) or 0)
    if max_audio_sec and prepared.duration_sec > max_audio_sec:
        raise ValueError(
            f"Engine '{engine}' supports at most {max_audio_sec:g} seconds per input; "
            "use long-strict mode so the recording is split safely."
        )
    provenance["audio"] = prepared.as_dict()
    return prepared.path, provenance


def _uses_shared_default_strict_audio(
    primary_engine: str,
    secondary_engine: str,
    config: ModelConfig,
) -> bool:
    if (
        primary_engine != config.strict_primary_engine
        or secondary_engine != config.strict_secondary_engine
    ):
        return False
    for engine in (primary_engine, secondary_engine):
        options = get_engine_spec(engine, config=config).options or {}
        if bool(options.get("requires_pcm16_mono", False)):
            return False
    return True


def _engine_provenance(engine: str, config: ModelConfig) -> dict[str, Any]:
    spec = get_engine_spec(engine, config=config)
    return {
        "engine": engine,
        "adapter": spec.adapter,
        "model": spec.model,
        "registry_role": spec.role,
        "options": dict(spec.options or {}),
    }


def _empty_cuda_cache() -> None:
    if importlib.util.find_spec("torch") is None:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return

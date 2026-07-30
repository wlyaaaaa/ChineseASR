from __future__ import annotations

import importlib.util
import gc
import os
import time
from pathlib import Path
from typing import Any

from .adapters import get_adapter
from .adapters.base import MissingDependencyError
from .adapters.funasr import ensure_funasr_available, funasr_kwargs as _funasr_kwargs
from .audio_frontend import prepare_pcm16_mono
from .config import ModelConfig, get_engine_spec, load_model_config
from .proxy_guard import sanitize_current_process_env
from .result_writer import write_transcript_bundle
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
) -> dict[str, Any]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    model_config = config or load_model_config()
    engine_name = engine or model_config.default_engine
    output_dir = out_dir or project_root() / "outputs"
    engine_audio, _ = _prepare_engine_input(
        audio_path,
        engine_name,
        output_dir / "_derived",
        model_config,
    )
    result = _generate_once(engine_audio, engine_name, device, cache_dir, model_config)
    return write_transcript_bundle(audio_path, result, output_dir, engine_name)


def strict_transcribe_audio(
    audio_path: Path,
    primary_engine: str | None = None,
    secondary_engine: str | None = None,
    device: str = "cuda:0",
    out_dir: Path | None = None,
    cache_dir: Path | None = None,
    config: ModelConfig | None = None,
    expect_empty: bool = False,
) -> dict[str, Any]:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    model_config = config or load_model_config()
    primary_name = primary_engine or model_config.strict_primary_engine
    secondary_name = secondary_engine or model_config.strict_secondary_engine
    output_dir = out_dir or project_root() / "outputs"
    derived_dir = output_dir / "_derived"
    total_started = time.perf_counter()
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
                    results[index] = value if isinstance(value, list) else [value]
            else:
                for index in valid_indices:
                    try:
                        results[index] = model.generate(
                            input=str(prepared[index]),
                            batch_size_s=300,
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
        return (
            _generate_once(engine_audio, engine, device, cache_dir, config),
            None,
            time.perf_counter() - started,
            provenance,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return _engine_failure_result(engine, exc), error, time.perf_counter() - started, provenance


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
    model = build_model(engine, device=device, cache_dir=cache_dir, config=config)
    try:
        return model.generate(input=str(audio_path), batch_size_s=300)
    finally:
        del model
        gc.collect()
        _empty_cuda_cache()


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

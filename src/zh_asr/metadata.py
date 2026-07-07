from __future__ import annotations

import hashlib
import os
import platform
import shlex
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import ModelConfig


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path | None) -> dict[str, int | str]:
    if path is None or not path.exists() or not path.is_file():
        return {"sha256": "", "size_bytes": 0}
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def snapshot_model_config(config: ModelConfig, selected_engines: tuple[str, ...] | None = None) -> dict[str, Any]:
    selected = selected_engines or (config.strict_primary_engine, config.strict_secondary_engine)
    selected_specs = {
        engine: asdict(config.engines[engine])
        for engine in selected
        if engine in config.engines
    }
    return {
        "path": str(config.path.resolve()),
        "sha256": sha256_file(config.path) if config.path.exists() else "",
        "default_engine": config.default_engine,
        "strict_primary_engine": config.strict_primary_engine,
        "strict_secondary_engine": config.strict_secondary_engine,
        "aliases": dict(config.model_aliases),
        "selected_engines": selected_specs,
    }


def capture_invocation(argv: list[str] | None = None, wrapper: str | None = None) -> dict[str, str | list[str]]:
    args = list(sys.argv[1:] if argv is None else argv)
    wrapper_name = wrapper if wrapper is not None else os.environ.get("ZH_ASR_WRAPPER", "")
    payload: dict[str, str | list[str]] = {
        "cwd": str(Path.cwd()),
        "argv": args,
        "command_line": " ".join(shlex.quote(part) for part in args),
        "python": sys.executable,
        "project_root": str(project_root()),
        "wrapper": wrapper_name,
    }
    return payload


def runtime_info(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "device": device,
        "torch": "",
        "cuda_available": False,
        "cuda_version": "",
        "gpu_name": "",
    }
    try:
        import torch
    except Exception:
        return info

    info["torch"] = str(getattr(torch, "__version__", ""))
    try:
        cuda_available = bool(torch.cuda.is_available())
        info["cuda_available"] = cuda_available
        info["cuda_version"] = str(getattr(torch.version, "cuda", "") or "")
        if cuda_available:
            info["gpu_name"] = str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return info

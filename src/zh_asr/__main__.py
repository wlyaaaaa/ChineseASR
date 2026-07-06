from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from .config import DEFAULT_ENGINE, ENGINES
from .pipeline import MissingDependencyError, build_model, default_cache_dir, transcribe_audio
from .proxy_guard import PROXY_ENV_NAMES, sanitize_current_process_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zh-asr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Check local runtime state without loading ASR models.")

    warmup = subparsers.add_parser("warmup", help="Load the selected engine and download weights if needed.")
    warmup.add_argument("--engine", choices=sorted(ENGINES), default=DEFAULT_ENGINE)
    warmup.add_argument("--device", default="cuda:0")
    warmup.add_argument("--cache-dir", type=Path, default=default_cache_dir())

    transcribe = subparsers.add_parser("transcribe", help="Transcribe one audio file.")
    transcribe.add_argument("audio", type=Path)
    transcribe.add_argument("--engine", choices=sorted(ENGINES), default=DEFAULT_ENGINE)
    transcribe.add_argument("--device", default="cuda:0")
    transcribe.add_argument("--out-dir", type=Path, default=Path("outputs"))
    transcribe.add_argument("--cache-dir", type=Path, default=default_cache_dir())

    args = parser.parse_args(argv)

    try:
        if args.command == "doctor":
            return _doctor()
        if args.command == "warmup":
            build_model(args.engine, device=args.device, cache_dir=args.cache_dir)
            print(f"Loaded engine: {args.engine}")
            return 0
        if args.command == "transcribe":
            paths = transcribe_audio(
                args.audio,
                engine=args.engine,
                device=args.device,
                out_dir=args.out_dir,
                cache_dir=args.cache_dir,
            )
            print(f"Markdown: {paths['markdown']}")
            print(f"Raw JSON: {paths['json']}")
            return 0
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except MissingDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 1


def _doctor() -> int:
    sanitize_current_process_env()
    dirty = [name for name in PROXY_ENV_NAMES if os.environ.get(name)]
    proxy_status = "dirty: " + ", ".join(dirty) if dirty else "clean"
    print(f"Default engine: {DEFAULT_ENGINE}")
    print(f"Available engines: {', '.join(sorted(ENGINES))}")
    print(f"Proxy variables: {proxy_status}")
    print(f"FunASR installed: {'yes' if importlib.util.find_spec('funasr') else 'no'}")
    print(f"PyTorch installed: {'yes' if importlib.util.find_spec('torch') else 'no'}")
    print(f"Model cache: {default_cache_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


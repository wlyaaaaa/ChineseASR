from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from .batch import run_batch
from .benchmark import run_benchmark
from .config import list_engine_names, list_transcription_engine_names, load_model_config
from .eval_pack import generate_builtin_corpus, run_evaluation
from .arbitration import load_arbitration_config, make_arbiter
from .long_audio import run_long_transcription
from .pipeline import MissingDependencyError, build_model, default_cache_dir, project_root, strict_transcribe_audio, transcribe_audio
from .proxy_guard import PROXY_ENV_NAMES, sanitize_current_process_env
from .service import serve_api


def main(argv: list[str] | None = None) -> int:
    try:
        model_config = load_model_config()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    engine_choices = list_engine_names(model_config)
    transcription_choices = list_transcription_engine_names(model_config)

    parser = argparse.ArgumentParser(prog="zh-asr")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Check local runtime state without loading ASR models.")

    warmup = subparsers.add_parser("warmup", help="Load the selected engine and download weights if needed.")
    warmup.add_argument("--engine", choices=engine_choices, default=model_config.default_engine)
    warmup.add_argument("--device", default="cuda:0")
    warmup.add_argument("--cache-dir", type=Path, default=default_cache_dir())

    transcribe = subparsers.add_parser("transcribe", help="Transcribe one audio file.")
    transcribe.add_argument("audio", type=Path)
    transcribe.add_argument("--engine", choices=engine_choices, default=model_config.default_engine)
    transcribe.add_argument("--device", default="cuda:0")
    transcribe.add_argument("--out-dir", type=Path, default=Path("outputs"))
    transcribe.add_argument("--cache-dir", type=Path, default=default_cache_dir())

    strict = subparsers.add_parser("strict", help="Transcribe with two engines and write final + audit outputs.")
    strict.add_argument("audio", type=Path)
    strict.add_argument("--primary-engine", choices=transcription_choices, default=model_config.strict_primary_engine)
    strict.add_argument("--secondary-engine", choices=transcription_choices, default=model_config.strict_secondary_engine)
    strict.add_argument("--device", default="cuda:0")
    strict.add_argument("--out-dir", type=Path, default=Path("outputs"))
    strict.add_argument("--cache-dir", type=Path, default=default_cache_dir())

    long_cmd = subparsers.add_parser("long", help="Resumable long-audio strict transcription.")
    long_cmd.add_argument("audio", type=Path)
    long_cmd.add_argument("--primary-engine", choices=transcription_choices, default=model_config.strict_primary_engine)
    long_cmd.add_argument("--secondary-engine", choices=transcription_choices, default=model_config.strict_secondary_engine)
    long_cmd.add_argument("--device", default="cuda:0")
    long_cmd.add_argument("--out-dir", type=Path, default=Path("outputs") / "long")
    long_cmd.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    long_cmd.add_argument("--chunk-sec", type=int, default=300)
    long_cmd.add_argument("--overlap-sec", type=int, default=1)
    long_cmd.add_argument("--force", action="store_true")

    batch = subparsers.add_parser("batch", help="Transcribe every supported audio file in a folder.")
    batch.add_argument("input_dir", type=Path)
    batch.add_argument("--mode", choices=("strict", "quick"), default="strict")
    batch.add_argument("--engine", choices=transcription_choices, default=model_config.default_engine)
    batch.add_argument("--primary-engine", choices=transcription_choices, default=model_config.strict_primary_engine)
    batch.add_argument("--secondary-engine", choices=transcription_choices, default=model_config.strict_secondary_engine)
    batch.add_argument("--device", default="cuda:0")
    batch.add_argument("--out-dir", type=Path, default=Path("outputs") / "batch")
    batch.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    batch.add_argument("--force", action="store_true")

    eval_cmd = subparsers.add_parser("eval", help="Generate or run a privacy-free local ASR evaluation pack.")
    eval_cmd.add_argument("--corpus-dir", type=Path, default=Path("eval") / "corpus" / "builtin")
    eval_cmd.add_argument("--out-dir", type=Path, default=Path("outputs") / "eval")
    eval_cmd.add_argument("--generate", action="store_true")
    eval_cmd.add_argument("--generate-only", action="store_true")
    eval_cmd.add_argument("--no-tts", action="store_true")
    eval_cmd.add_argument("--primary-engine", choices=transcription_choices, default=model_config.strict_primary_engine)
    eval_cmd.add_argument("--secondary-engine", choices=transcription_choices, default=model_config.strict_secondary_engine)
    eval_cmd.add_argument("--device", default="cuda:0")
    eval_cmd.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    eval_cmd.add_argument("--force", action="store_true")
    eval_cmd.add_argument("--fail-on-findings", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="Benchmark strict ASR against human truth text files.")
    benchmark.add_argument("--audio-dir", type=Path, required=True)
    benchmark.add_argument("--truth-dir", type=Path, required=True)
    benchmark.add_argument("--out-dir", type=Path, default=Path("outputs") / "benchmark")
    benchmark.add_argument("--primary-engine", choices=transcription_choices, default=model_config.strict_primary_engine)
    benchmark.add_argument("--secondary-engine", choices=transcription_choices, default=model_config.strict_secondary_engine)
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    benchmark.add_argument("--force", action="store_true")
    benchmark.add_argument("--fail-on-findings", action="store_true")

    serve = subparsers.add_parser("serve", help="Run the local ASR job API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--state-dir", type=Path, default=Path("outputs") / "api")
    serve.add_argument("--check", action="store_true", help="Validate serve configuration without blocking.")

    args = parser.parse_args(argv)

    try:
        if args.command == "doctor":
            return _doctor(model_config)
        if args.command == "warmup":
            build_model(args.engine, device=args.device, cache_dir=args.cache_dir, config=model_config)
            print(f"Loaded engine: {args.engine}")
            return 0
        if args.command == "transcribe":
            paths = transcribe_audio(
                args.audio,
                engine=args.engine,
                device=args.device,
                out_dir=args.out_dir,
                cache_dir=args.cache_dir,
                config=model_config,
            )
            print(f"Markdown: {paths['markdown']}")
            print(f"Raw JSON: {paths['json']}")
            return 0
        if args.command == "strict":
            paths = strict_transcribe_audio(
                args.audio,
                primary_engine=args.primary_engine,
                secondary_engine=args.secondary_engine,
                device=args.device,
                out_dir=args.out_dir,
                cache_dir=args.cache_dir,
                config=model_config,
            )
            print(f"Final: {paths['final']}")
            print(f"Audit: {paths['audit']}")
            print(f"Audit JSON: {paths['audit_json']}")
            print(f"Primary raw JSON: {paths['primary_json']}")
            print(f"Secondary raw JSON: {paths['secondary_json']}")
            return 0
        if args.command == "long":
            arbiter = make_arbiter(load_arbitration_config(model_config.path))
            summary = run_long_transcription(
                args.audio,
                args.out_dir,
                chunk_sec=args.chunk_sec,
                overlap_sec=args.overlap_sec,
                primary_engine=args.primary_engine,
                secondary_engine=args.secondary_engine,
                device=args.device,
                cache_dir=args.cache_dir,
                force=args.force,
                arbiter=arbiter,
            )
            print(f"Transcript: {summary.transcript_path}")
            print(f"Audit: {summary.audit_path}")
            print(f"Metrics: {summary.metrics_path}")
            print(f"Manifest: {summary.manifest_path}")
            print(
                f"Total: {summary.total}; "
                f"Processed: {summary.processed}; "
                f"Skipped: {summary.skipped}; "
                f"Failed: {summary.failed}"
            )
            return 1 if summary.failed else 0
        if args.command == "batch":
            summary = run_batch(
                input_dir=args.input_dir,
                out_dir=args.out_dir,
                mode=args.mode,
                engine=args.engine,
                primary_engine=args.primary_engine,
                secondary_engine=args.secondary_engine,
                device=args.device,
                cache_dir=args.cache_dir,
                config=model_config,
                force=args.force,
            )
            print(f"Summary: {summary.out_dir / 'summary.md'}")
            print(
                f"Total: {summary.total}; "
                f"Processed: {summary.processed}; "
                f"Skipped: {summary.skipped}; "
                f"Failed: {summary.failed}"
            )
            if summary.failed:
                print(f"Failures: {summary.out_dir / 'failed.jsonl'}", file=sys.stderr)
                return 1
            return 0
        if args.command == "eval":
            manifest_path = args.corpus_dir / "manifest.json"
            should_generate = args.generate or args.generate_only
            if should_generate:
                manifest_path = generate_builtin_corpus(
                    args.corpus_dir,
                    include_tts=not args.no_tts,
                    force=args.force,
                )
                print(f"Corpus manifest: {manifest_path}")
            if args.generate_only:
                return 0
            if not manifest_path.exists():
                raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")
            eval_summary = run_evaluation(
                corpus_dir=args.corpus_dir,
                out_dir=args.out_dir,
                device=args.device,
                cache_dir=args.cache_dir,
                force=args.force,
                primary_engine=args.primary_engine,
                secondary_engine=args.secondary_engine,
                config=model_config,
            )
            print(f"Metrics: {eval_summary.out_dir / 'metrics.json'}")
            print(f"Benchmark: {eval_summary.out_dir / 'benchmark.md'}")
            print(f"Review: {eval_summary.out_dir / 'review.md'}")
            print(
                f"Total: {eval_summary.total}; "
                f"Evaluated: {eval_summary.evaluated}; "
                f"Skipped: {eval_summary.skipped}; "
                f"Hallucinations: {eval_summary.hallucination_count}; "
                f"False confident: {eval_summary.false_confident_count}"
            )
            return 1 if args.fail_on_findings and eval_summary.false_confident_count else 0
        if args.command == "benchmark":
            benchmark_summary = run_benchmark(
                audio_dir=args.audio_dir,
                truth_dir=args.truth_dir,
                out_dir=args.out_dir,
                device=args.device,
                cache_dir=args.cache_dir,
                force=args.force,
                primary_engine=args.primary_engine,
                secondary_engine=args.secondary_engine,
                config=model_config,
            )
            print(f"Benchmark JSON: {benchmark_summary.out_dir / 'benchmark.json'}")
            print(f"Benchmark Markdown: {benchmark_summary.out_dir / 'benchmark.md'}")
            print(f"Review: {benchmark_summary.out_dir / 'review.md'}")
            print(
                f"Total: {benchmark_summary.total}; "
                f"Evaluated: {benchmark_summary.evaluated}; "
                f"Skipped: {benchmark_summary.skipped}; "
                f"False confident: {benchmark_summary.false_confident_count}"
            )
            return 1 if args.fail_on_findings and benchmark_summary.false_confident_count else 0
        if args.command == "serve":
            state_dir = args.state_dir.resolve()
            if args.check:
                print(f"ASR API ready at http://{args.host}:{args.port}")
                print(f"State directory: {state_dir}")
                return 0
            return serve_api(args.host, args.port, state_dir, root=project_root())
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


def _doctor(model_config) -> int:
    sanitize_current_process_env()
    dirty = [name for name in PROXY_ENV_NAMES if os.environ.get(name)]
    proxy_status = "dirty: " + ", ".join(dirty) if dirty else "clean"
    print(f"Model config: {model_config.path}")
    print(f"Default engine: {model_config.default_engine}")
    print(f"Strict engines: {model_config.strict_primary_engine}, {model_config.strict_secondary_engine}")
    print(f"Available engines: {', '.join(list_engine_names(model_config))}")
    print(f"Proxy variables: {proxy_status}")
    print(f"FunASR installed: {'yes' if importlib.util.find_spec('funasr') else 'no'}")
    print(f"Qwen ASR installed: {'yes' if importlib.util.find_spec('qwen_asr') else 'no'}")
    print(f"PyTorch installed: {'yes' if importlib.util.find_spec('torch') else 'no'}")
    print(f"Model cache: {default_cache_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

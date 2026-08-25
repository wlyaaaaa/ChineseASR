from __future__ import annotations

import argparse
import contextvars
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from .batch import run_batch
from .benchmark import run_benchmark
from .config import list_engine_names, list_transcription_engine_names, load_model_config
from .eval_pack import generate_builtin_corpus, run_evaluation
from .arbitration import load_arbitration_config, make_arbiter
from .gpu_broker import (
    GPU_BROKER_CHILD_TOKEN_ENV,
    GpuBrokerError,
    GpuBrokerLease,
    verify_inherited_gpu_lease,
)
from .long_audio import run_long_transcription
from .pipeline import MissingDependencyError, build_model, default_cache_dir, project_root, strict_transcribe_audio, transcribe_audio
from .process_control import managed_popen_kwargs, terminate_process_tree
from .proxy_guard import PROXY_ENV_NAMES, sanitize_current_process_env
from .result_writer import canonical_json_sha256, file_sha256
from .speaker_attribution import write_speaker_attribution
from .speaker_evidence import (
    SELF_PERSON_ID,
    default_self_speaker_profile_path,
    delete_self_speaker_profile,
    enroll_self_speaker,
    load_self_speaker_profile,
    write_self_speaker_evidence,
)
from .service import CALLER_BINDING_ENV, serve_api


_GPU_LEASE_AUTHENTICATED = contextvars.ContextVar(
    "zh_asr_gpu_lease_authenticated",
    default=False,
)


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
    transcribe.add_argument(
        "--preset-spk-num",
        type=int,
        help=(
            "Only with explicit --engine paraformer: constrain anonymous diarization to a known "
            "positive speaker count. Omit to retain automatic clustering."
        ),
    )
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
    serve.add_argument("--port", type=int, default=18666)
    serve.add_argument("--state-dir", type=Path, default=Path("outputs") / "api")
    serve.add_argument("--check", action="store_true", help="Validate serve configuration without blocking.")

    attribution = subparsers.add_parser(
        "attribute-speakers",
        help="Create a fail-closed speaker-attribution projection from existing ASR JSON.",
    )
    attribution.add_argument("transcript_json", type=Path)
    attribution.add_argument("--context", type=Path, required=True)
    attribution.add_argument("--out", type=Path, required=True)
    attribution.add_argument(
        "--voice-evidence",
        type=Path,
        action="append",
        default=[],
        help="Hash-bound person:self evidence JSON; repeat for more than one segment.",
    )
    attribution.add_argument(
        "--voice-profile",
        type=Path,
        default=default_self_speaker_profile_path(),
        help="Current private person:self profile; old evidence is inactive when this file is absent or has been replaced.",
    )

    speaker_enroll = subparsers.add_parser(
        "speaker-enroll",
        help="Create the private local person:self CAM++ reference profile.",
        description="Create the sole private local person:self CAM++ reference profile.",
    )
    speaker_enroll.add_argument("reference_audio", type=Path)
    speaker_enroll.add_argument("--start-ms", type=float, required=True)
    speaker_enroll.add_argument("--end-ms", type=float, required=True)
    speaker_enroll.add_argument("--channel", choices=("mix", "left", "right"), default="mix")
    speaker_enroll.add_argument(
        "--inference-basis",
        required=True,
        help="One-sentence, reversible basis for this inferred person:self anchor; never a confirmed identity.",
    )
    speaker_enroll.add_argument("--profile", type=Path, default=default_self_speaker_profile_path())
    speaker_enroll.add_argument("--replace", action="store_true")
    speaker_enroll.add_argument("--device", default="cuda:0")
    speaker_enroll.add_argument("--cache-dir", type=Path, default=default_cache_dir())

    speaker_evidence = subparsers.add_parser(
        "speaker-evidence",
        help="Compare one bounded audio interval with the private person:self reference.",
        description="Generate unconfirmed person:self evidence for one bounded audio interval.",
    )
    speaker_evidence.add_argument("target_audio", type=Path)
    speaker_evidence.add_argument("--start-ms", type=float, required=True)
    speaker_evidence.add_argument("--end-ms", type=float, required=True)
    speaker_evidence.add_argument("--channel", choices=("mix", "left", "right"), default="mix")
    speaker_evidence.add_argument("--profile", type=Path, default=default_self_speaker_profile_path())
    speaker_evidence.add_argument("--out", type=Path, required=True)
    speaker_evidence.add_argument("--device", default="cuda:0")
    speaker_evidence.add_argument("--cache-dir", type=Path, default=default_cache_dir())

    speaker_delete = subparsers.add_parser(
        "speaker-profile-delete",
        help="Delete the private person:self profile after an explicit confirmation.",
        description="Delete only the private person:self profile after an explicit confirmation.",
    )
    speaker_delete.add_argument("--profile", type=Path, default=default_self_speaker_profile_path())
    speaker_delete.add_argument("--confirm-delete", required=True)

    args = parser.parse_args(argv)
    try:
        caller_binding = _caller_binding_from_env()
    except ValueError as exc:
        print(f"ValueError: {exc}", file=sys.stderr)
        return 1

    if _command_requires_gpu_supervision(args):
        try:
            inherited_token = os.environ.pop(
                GPU_BROKER_CHILD_TOKEN_ENV,
                "",
            )
            if inherited_token:
                verify_inherited_gpu_lease(inherited_token)
                _GPU_LEASE_AUTHENTICATED.set(True)
            else:
                return _supervise_gpu_cli(
                    list(sys.argv[1:] if argv is None else argv)
                )
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    try:
        if args.command == "doctor":
            return _doctor(model_config)
        if args.command == "warmup":
            _run_with_gpu_lease(
                args.device,
                lambda: build_model(
                    args.engine,
                    device=args.device,
                    cache_dir=args.cache_dir,
                    config=model_config,
                ),
            )
            print(f"Loaded engine: {args.engine}")
            return 0
        if args.command == "transcribe":
            paths = _run_with_gpu_lease(
                args.device,
                lambda: transcribe_audio(
                    args.audio,
                    engine=args.engine,
                    device=args.device,
                    out_dir=args.out_dir,
                    cache_dir=args.cache_dir,
                    config=model_config,
                    preset_spk_num=args.preset_spk_num,
                    caller_binding=caller_binding,
                ),
            )
            print(f"Markdown: {paths['markdown']}")
            print(f"Raw JSON: {paths['json']}")
            print(f"Objective: {paths.get('objective_outcome', 'indeterminate')}")
            if paths.get("objective_result"):
                print(f"Objective result: {paths['objective_result']}")
            return 0
        if args.command == "strict":
            paths = _run_with_gpu_lease(
                args.device,
                lambda: strict_transcribe_audio(
                    args.audio,
                    primary_engine=args.primary_engine,
                    secondary_engine=args.secondary_engine,
                    device=args.device,
                    out_dir=args.out_dir,
                    cache_dir=args.cache_dir,
                    config=model_config,
                    caller_binding=caller_binding,
                ),
            )
            print(f"Final: {paths['final']}")
            print(f"Audit: {paths['audit']}")
            print(f"Audit JSON: {paths['audit_json']}")
            print(f"Primary raw JSON: {paths['primary_json']}")
            print(f"Secondary raw JSON: {paths['secondary_json']}")
            print(f"Objective: {paths.get('objective_outcome', 'indeterminate')}")
            if paths.get("objective_result"):
                print(f"Objective result: {paths['objective_result']}")
            return 0
        if args.command == "long":
            arbiter = make_arbiter(load_arbitration_config(model_config.path))
            summary = _run_with_gpu_lease(
                args.device,
                lambda: run_long_transcription(
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
                    caller_binding=caller_binding,
                ),
            )
            print(f"Transcript: {summary.transcript_path}")
            print(f"Audit: {summary.audit_path}")
            print(f"Metrics: {summary.metrics_path}")
            print(f"Manifest: {summary.manifest_path}")
            print(f"Objective: {summary.objective_outcome}")
            print(
                f"Total: {summary.total}; "
                f"Processed: {summary.processed}; "
                f"Skipped: {summary.skipped}; "
                f"Failed: {summary.failed}"
            )
            return 1 if summary.failed else 0
        if args.command == "batch":
            summary = _run_with_gpu_lease(
                args.device,
                lambda: run_batch(
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
                    caller_binding=caller_binding,
                ),
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
            eval_summary = _run_with_gpu_lease(
                args.device,
                lambda: run_evaluation(
                    corpus_dir=args.corpus_dir,
                    out_dir=args.out_dir,
                    device=args.device,
                    cache_dir=args.cache_dir,
                    force=args.force,
                    primary_engine=args.primary_engine,
                    secondary_engine=args.secondary_engine,
                    config=model_config,
                ),
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
            benchmark_summary = _run_with_gpu_lease(
                args.device,
                lambda: run_benchmark(
                    audio_dir=args.audio_dir,
                    truth_dir=args.truth_dir,
                    out_dir=args.out_dir,
                    device=args.device,
                    cache_dir=args.cache_dir,
                    force=args.force,
                    primary_engine=args.primary_engine,
                    secondary_engine=args.secondary_engine,
                    config=model_config,
                ),
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
        if args.command == "attribute-speakers":
            transcript_result = json.loads(args.transcript_json.read_text(encoding="utf-8"))
            context = json.loads(args.context.read_text(encoding="utf-8"))
            voice_evidence = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in args.voice_evidence
            ]
            active_voice_profile_sha256 = None
            if voice_evidence and args.voice_profile.is_file():
                active_voice_profile_sha256 = canonical_json_sha256(
                    load_self_speaker_profile(args.voice_profile)
                )
            payload = write_speaker_attribution(
                args.out,
                transcript_result,
                context,
                voice_evidence=voice_evidence,
                active_voice_profile_sha256=active_voice_profile_sha256,
                input_hashes={
                    "transcript_json_sha256": file_sha256(args.transcript_json),
                    "context_json_sha256": file_sha256(args.context),
                    "voice_evidence_json_sha256": [
                        file_sha256(path) for path in args.voice_evidence
                    ],
                },
            )
            print(f"Speaker attribution: {args.out}")
            print(f"Segments: {len(payload['segments'])}")
            print(f"Speaker attribution gap: {payload['speaker_attribution_gap']}")
            return 0
        if args.command == "speaker-enroll":
            profile = _run_with_gpu_lease(
                args.device,
                lambda: enroll_self_speaker(
                    args.reference_audio,
                    start_ms=args.start_ms,
                    end_ms=args.end_ms,
                    channel=args.channel,
                    inference_basis=args.inference_basis,
                    profile_path=args.profile,
                    replace=args.replace,
                    device=args.device,
                    cache_dir=args.cache_dir,
                    model_config=model_config,
                ),
            )
            print(f"person:self profile: {args.profile}")
            print(f"Reference SHA-256: {profile['enrollment_reference']['source']['sha256']}")
            print("Identity status: inferred and reversible (profile is only a fusion anchor)")
            return 0
        if args.command == "speaker-evidence":
            evidence = _run_with_gpu_lease(
                args.device,
                lambda: write_self_speaker_evidence(
                    args.out,
                    args.target_audio,
                    start_ms=args.start_ms,
                    end_ms=args.end_ms,
                    channel=args.channel,
                    profile_path=args.profile,
                    device=args.device,
                    cache_dir=args.cache_dir,
                    model_config=model_config,
                ),
            )
            print(f"person:self evidence: {args.out}")
            print(f"Similarity: {evidence['score']['value']:.4f}")
            print("Identity status: unconfirmed (fuse this evidence before attribution)")
            return 0
        if args.command == "speaker-profile-delete":
            delete_self_speaker_profile(
                args.profile,
                confirmation=args.confirm_delete,
            )
            print(f"Deleted {SELF_PERSON_ID} profile: {args.profile}")
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


def _caller_binding_from_env() -> dict | None:
    raw = os.environ.get(CALLER_BINDING_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{CALLER_BINDING_ENV} must contain a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{CALLER_BINDING_ENV} must contain a JSON object")
    return value


def _run_with_gpu_lease(device: str, operation):
    if not str(device).lower().startswith(("cuda", "gpu")):
        return operation()
    if not _GPU_LEASE_AUTHENTICATED.get():
        raise GpuBrokerError(
            "GPU operation is not running under an authenticated "
            "LocalGpuBroker supervisor."
        )
    return operation()


def _command_requires_gpu_supervision(args: argparse.Namespace) -> bool:
    if args.command not in {
        "warmup",
        "transcribe",
        "strict",
        "long",
        "batch",
        "eval",
        "benchmark",
        "speaker-enroll",
        "speaker-evidence",
    }:
        return False
    if args.command == "eval" and args.generate_only:
        return False
    return str(getattr(args, "device", "")).lower().startswith(("cuda", "gpu"))


def _supervise_gpu_cli(argv: list[str]) -> int:
    """Run one public GPU CLI invocation in a killable supervised worker."""
    process_holder: dict[str, subprocess.Popen] = {}

    def terminate_on_loss(_error) -> None:
        process = process_holder.get("process")
        if process is not None:
            terminate_process_tree(process)

    lease = GpuBrokerLease("chineseasr-cli")
    lease.set_on_lost(terminate_on_loss)
    with lease:
        env = os.environ.copy()
        env.pop("ZH_ASR_GPU_BROKER_LEASE_HELD", None)
        env[GPU_BROKER_CHILD_TOKEN_ENV] = lease.token
        process = subprocess.Popen(
            [sys.executable, "-m", "zh_asr", *argv],
            cwd=project_root(),
            env=env,
            **managed_popen_kwargs(),
        )
        process_holder["process"] = process
        try:
            lease.raise_if_lost()
            returncode = process.wait()
            lease.raise_if_lost()
            return int(returncode)
        finally:
            if process.poll() is None:
                terminate_process_tree(process)


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

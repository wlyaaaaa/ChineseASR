from __future__ import annotations

import hashlib
import json
import math
import random
import re
import struct
import subprocess
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import ModelConfig, load_model_config
from .metadata import capture_invocation, file_metadata, runtime_info, snapshot_model_config
from .pipeline import strict_transcribe_audio
from .text_normalizer import to_simplified


SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2

TTS_CASES = (
    {
        "id": "tts-clean-001",
        "category": "synthetic",
        "kind": "tts",
        "truth_text": "开放时间早上九点至下午五点。",
        "rate": 0,
        "notes": "Clean Mandarin TTS sentence with known truth.",
    },
)

ADVERSARIAL_CASES = (
    {
        "id": "silence-001",
        "category": "adversarial",
        "kind": "silence",
        "truth_text": "",
        "duration_sec": 2.0,
        "notes": "Pure silence; correct ASR behavior is empty or unclear.",
    },
    {
        "id": "white-noise-001",
        "category": "adversarial",
        "kind": "white_noise",
        "truth_text": "",
        "duration_sec": 2.0,
        "notes": "Deterministic white noise; correct ASR behavior is empty or unclear.",
    },
    {
        "id": "tone-001",
        "category": "adversarial",
        "kind": "tone",
        "truth_text": "",
        "duration_sec": 2.0,
        "notes": "Pure tone; correct ASR behavior is empty or unclear.",
    },
)


@dataclass(frozen=True)
class EvalCaseResult:
    case_id: str
    category: str
    kind: str
    audio: Path | None
    audio_sha256: str
    truth_sha256: str
    truth_text: str
    final_text: str
    cer: float | None
    disagreement_score: float
    audit_status: str
    primary_engine: str
    primary_text: str
    secondary_engine: str
    secondary_text: str
    primary_secondary_similarity: float
    timing_total_sec: float | None
    timing_primary_sec: float | None
    timing_secondary_sec: float | None
    audit_json: Path | None
    primary_json: Path | None
    secondary_json: Path | None
    empty_audio_text_len: int
    risk_flags: tuple[str, ...]
    false_confident: bool
    simplified_only: bool
    needs_review: bool
    skipped: bool = False
    skip_reason: str = ""


@dataclass(frozen=True)
class EvalSummary:
    corpus_dir: Path
    out_dir: Path
    total: int
    evaluated: int
    skipped: int
    hallucination_count: int
    false_confident_count: int
    cases: tuple[EvalCaseResult, ...] = field(default_factory=tuple)


StrictFn = Callable[..., dict[str, Any]]
TtsWriter = Callable[[str, Path, int], None]


def generate_builtin_corpus(
    corpus_dir: Path,
    include_tts: bool = True,
    force: bool = False,
    tts_writer: TtsWriter | None = None,
) -> Path:
    corpus_root = corpus_dir.resolve()
    corpus_root.mkdir(parents=True, exist_ok=True)
    (corpus_root / "truth").mkdir(parents=True, exist_ok=True)
    (corpus_root / "synthetic").mkdir(parents=True, exist_ok=True)
    (corpus_root / "adversarial").mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    writer = tts_writer or _write_tts_wav_windows

    for spec in TTS_CASES:
        audio = Path("synthetic") / f"{spec['id']}.wav"
        truth = Path("truth") / f"{spec['id']}.txt"
        truth_path = corpus_root / truth
        truth_path.write_text(str(spec["truth_text"]), encoding="utf-8")
        case = _case_manifest_item(spec, audio, truth, expect_empty=False)

        if include_tts:
            audio_path = corpus_root / audio
            if force or not audio_path.exists():
                try:
                    writer(str(spec["truth_text"]), audio_path, int(spec.get("rate", 0)))
                except Exception as exc:
                    case["available"] = False
                    case["error"] = f"{type(exc).__name__}: {exc}"
                else:
                    case["available"] = True
            else:
                case["available"] = True
        else:
            case["available"] = False
            case["error"] = "TTS generation disabled."
        cases.append(case)

    for spec in ADVERSARIAL_CASES:
        audio = Path("adversarial") / f"{spec['id']}.wav"
        truth = Path("truth") / f"{spec['id']}.txt"
        truth_path = corpus_root / truth
        truth_path.write_text("", encoding="utf-8")
        audio_path = corpus_root / audio
        if force or not audio_path.exists():
            _write_adversarial_audio(audio_path, str(spec["kind"]), float(spec["duration_sec"]))
        cases.append(_case_manifest_item(spec, audio, truth, expect_empty=True))

    model_config = load_model_config()
    cases = [_with_case_file_metadata(corpus_root, case) for case in cases]
    manifest = {
        "schema_version": 2,
        "version": 1,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "description": "Built-in privacy-free ChineseASR evaluation corpus.",
        "model_config": snapshot_model_config(model_config),
        "invocation": capture_invocation(),
        "cases": cases,
    }
    manifest_path = corpus_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def run_evaluation(
    corpus_dir: Path,
    out_dir: Path,
    device: str = "cuda:0",
    cache_dir: Path | None = None,
    force: bool = False,
    primary_engine: str | None = None,
    secondary_engine: str | None = None,
    config: ModelConfig | None = None,
    strict_fn: StrictFn = strict_transcribe_audio,
) -> EvalSummary:
    corpus_root = corpus_dir.resolve()
    output_root = out_dir.resolve()
    manifest_path = corpus_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_config = config or load_model_config()
    primary = primary_engine or model_config.strict_primary_engine
    secondary = secondary_engine or model_config.strict_secondary_engine
    output_root.mkdir(parents=True, exist_ok=True)

    run_started_at = datetime.now()
    run_started_perf = time.perf_counter()
    results: list[EvalCaseResult] = []
    for case in manifest.get("cases", []):
        case_id = str(case["id"])
        if not case.get("available", True):
            results.append(_skipped_case(case, corpus_root, str(case.get("error", "case unavailable"))))
            continue

        audio_path = corpus_root / str(case["audio"])
        truth_text = _truth_text(corpus_root, case)
        case_out = output_root / "cases" / case_id
        expected_audit = case_out / f"{audio_path.stem}.strict.audit.json"
        case_started_perf = time.perf_counter()
        if not expected_audit.exists() or force:
            paths = strict_fn(
                audio_path,
                primary_engine=primary,
                secondary_engine=secondary,
                device=device,
                out_dir=case_out,
                cache_dir=cache_dir,
                config=model_config,
            )
            audit_json = paths.get("audit_json", expected_audit)
        else:
            audit_json = expected_audit
            paths = {"audit_json": audit_json}
        case_elapsed_sec = time.perf_counter() - case_started_perf
        audit = json.loads(Path(audit_json).read_text(encoding="utf-8"))
        results.append(_evaluate_case(case, corpus_root, audio_path, truth_text, audit, paths, case_elapsed_sec))

    summary = _summary(corpus_root, output_root, results)
    run_finished_at = datetime.now()
    run_elapsed_sec = time.perf_counter() - run_started_perf
    selected_engines = (primary, secondary)
    _write_metrics(
        output_root / "metrics.json",
        summary,
        model_config=snapshot_model_config(model_config, selected_engines),
        invocation=dict(manifest.get("invocation") or capture_invocation()),
        runtime=runtime_info(device),
        started_at=run_started_at,
        finished_at=run_finished_at,
        elapsed_sec=run_elapsed_sec,
    )
    _write_benchmark(output_root / "benchmark.md", summary)
    _write_review(output_root / "review.md", summary)
    return summary


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = _normalize_metric_text(reference)
    hyp = _normalize_metric_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return float(len(hyp))
    return _levenshtein(ref, hyp) / len(ref)


def _case_manifest_item(spec: dict[str, Any], audio: Path, truth: Path, expect_empty: bool) -> dict[str, Any]:
    return {
        "id": spec["id"],
        "category": spec["category"],
        "kind": spec["kind"],
        "audio": audio.as_posix(),
        "truth": truth.as_posix(),
        "truth_text": spec["truth_text"],
        "expect_empty": expect_empty,
        "available": True,
        "notes": spec["notes"],
    }


def _with_case_file_metadata(corpus_root: Path, case: dict[str, Any]) -> dict[str, Any]:
    audio_path = corpus_root / str(case["audio"]) if case.get("audio") else None
    truth_path = corpus_root / str(case["truth"]) if case.get("truth") else None
    audio_meta = file_metadata(audio_path)
    truth_meta = file_metadata(truth_path)
    return {
        **case,
        "audio_sha256": audio_meta["sha256"],
        "audio_size_bytes": audio_meta["size_bytes"],
        "truth_sha256": truth_meta["sha256"],
        "truth_size_bytes": truth_meta["size_bytes"],
    }


def _write_adversarial_audio(path: Path, kind: str, duration_sec: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(SAMPLE_RATE * duration_sec))
    if kind == "silence":
        samples = [0] * frame_count
    elif kind == "white_noise":
        rng = random.Random(20260707)
        samples = [rng.randint(-1200, 1200) for _ in range(frame_count)]
    elif kind == "tone":
        samples = [int(1200 * math.sin(2 * math.pi * 440 * index / SAMPLE_RATE)) for index in range(frame_count)]
    else:
        raise ValueError(f"Unknown adversarial audio kind: {kind}")
    _write_pcm_wav(path, samples)


def _write_pcm_wav(path: Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH_BYTES)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"".join(struct.pack("<h", max(-32768, min(32767, value))) for value in samples))


def _write_tts_wav_windows(text: str, path: Path, rate: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = "\n".join(
        [
            "Add-Type -AssemblyName System.Speech",
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer",
            "$voice = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq 'zh-CN' } | Select-Object -First 1",
            "if ($null -eq $voice) { throw 'No zh-CN SAPI voice is installed.' }",
            "$s.SelectVoice($voice.VoiceInfo.Name)",
            f"$s.Rate = {rate}",
            f"$s.SetOutputToWaveFile('{_ps_single_quote(path)}')",
            f"$s.Speak('{_ps_single_quote(text)}')",
            "$s.Dispose()",
        ]
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "PowerShell TTS failed").strip()
        raise RuntimeError(detail)


def _ps_single_quote(value: Path | str) -> str:
    return str(value).replace("'", "''")


def _truth_text(corpus_root: Path, case: dict[str, Any]) -> str:
    truth_path = corpus_root / str(case["truth"])
    if truth_path.exists():
        return truth_path.read_text(encoding="utf-8").strip()
    return str(case.get("truth_text", "")).strip()


def _evaluate_case(
    case: dict[str, Any],
    corpus_root: Path,
    audio_path: Path,
    truth_text: str,
    audit: dict[str, Any],
    paths: dict[str, Any],
    elapsed_sec: float,
) -> EvalCaseResult:
    final_text = to_simplified(str(audit.get("final_text", "")))
    expect_empty = bool(case.get("expect_empty", False))
    similarity = float(audit.get("similarity", 0.0))
    disagreement_score = max(0.0, min(1.0, 1.0 - similarity))
    empty_len = _empty_audio_text_len(final_text) if expect_empty else 0
    cer = None if expect_empty else char_error_rate(truth_text, final_text)
    audit_flags = tuple(str(flag) for flag in audit.get("flags", []))
    needs_review = bool(audit.get("needs_review", False))
    risk_flags = list(audit_flags)

    if expect_empty and empty_len > 0:
        risk_flags.append("empty_audio_hallucination")
    if cer is not None and cer > 0.1:
        risk_flags.append("high_cer")
    if final_text != to_simplified(final_text):
        risk_flags.append("non_simplified_output")

    false_confident = bool(risk_flags) and not needs_review and not audit_flags
    timing = paths.get("timing") if isinstance(paths.get("timing"), dict) else {}
    total_sec = _optional_float(timing.get("total_sec")) or elapsed_sec
    return EvalCaseResult(
        case_id=str(case["id"]),
        category=str(case.get("category", "")),
        kind=str(case.get("kind", "")),
        audio=audio_path,
        audio_sha256=str(case.get("audio_sha256", "")),
        truth_sha256=str(case.get("truth_sha256", "")),
        truth_text=truth_text,
        final_text=final_text,
        cer=cer,
        disagreement_score=disagreement_score,
        audit_status=str(audit.get("status", "")),
        primary_engine=str(audit.get("primary_engine", "")),
        primary_text=to_simplified(str(audit.get("primary_text", ""))),
        secondary_engine=str(audit.get("secondary_engine", "")),
        secondary_text=to_simplified(str(audit.get("secondary_text", ""))),
        primary_secondary_similarity=similarity,
        timing_total_sec=total_sec,
        timing_primary_sec=_optional_float(timing.get("primary_sec")),
        timing_secondary_sec=_optional_float(timing.get("secondary_sec")),
        audit_json=_optional_path(paths.get("audit_json")),
        primary_json=_optional_path(paths.get("primary_json")),
        secondary_json=_optional_path(paths.get("secondary_json")),
        empty_audio_text_len=empty_len,
        risk_flags=tuple(sorted(set(risk_flags))),
        false_confident=false_confident,
        simplified_only=final_text == to_simplified(final_text),
        needs_review=needs_review or bool(risk_flags),
    )


def _skipped_case(case: dict[str, Any], corpus_root: Path, reason: str) -> EvalCaseResult:
    audio_value = case.get("audio")
    return EvalCaseResult(
        case_id=str(case["id"]),
        category=str(case.get("category", "")),
        kind=str(case.get("kind", "")),
        audio=(corpus_root / str(audio_value)) if audio_value else None,
        audio_sha256=str(case.get("audio_sha256", "")),
        truth_sha256=str(case.get("truth_sha256", "")),
        truth_text=str(case.get("truth_text", "")),
        final_text="",
        cer=None,
        disagreement_score=0.0,
        audit_status="skipped",
        primary_engine="",
        primary_text="",
        secondary_engine="",
        secondary_text="",
        primary_secondary_similarity=0.0,
        timing_total_sec=None,
        timing_primary_sec=None,
        timing_secondary_sec=None,
        audit_json=None,
        primary_json=None,
        secondary_json=None,
        empty_audio_text_len=0,
        risk_flags=(),
        false_confident=False,
        simplified_only=True,
        needs_review=False,
        skipped=True,
        skip_reason=reason,
    )


def _summary(corpus_dir: Path, out_dir: Path, results: list[EvalCaseResult]) -> EvalSummary:
    evaluated = [item for item in results if not item.skipped]
    return EvalSummary(
        corpus_dir=corpus_dir,
        out_dir=out_dir,
        total=len(results),
        evaluated=len(evaluated),
        skipped=sum(1 for item in results if item.skipped),
        hallucination_count=sum(1 for item in evaluated if item.empty_audio_text_len > 0),
        false_confident_count=sum(1 for item in evaluated if item.false_confident),
        cases=tuple(results),
    )


def _write_metrics(
    path: Path,
    summary: EvalSummary,
    model_config: dict[str, Any],
    invocation: dict[str, Any],
    runtime: dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
    elapsed_sec: float,
) -> None:
    payload = {
        "schema_version": 2,
        "summary": {
            "corpus_dir": str(summary.corpus_dir),
            "out_dir": str(summary.out_dir),
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "elapsed_sec": round(elapsed_sec, 6),
            "total": summary.total,
            "evaluated": summary.evaluated,
            "skipped": summary.skipped,
            "hallucination_count": summary.hallucination_count,
            "false_confident_count": summary.false_confident_count,
            "generated": datetime.now().isoformat(timespec="seconds"),
        },
        "runtime": runtime,
        "model_config": model_config,
        "invocation": invocation,
        "cases": [
            {
                "id": item.case_id,
                "category": item.category,
                "kind": item.kind,
                "audio": str(item.audio) if item.audio else "",
                "audio_sha256": item.audio_sha256 or (_sha256(item.audio) if item.audio and item.audio.exists() else ""),
                "truth_sha256": item.truth_sha256,
                "truth_text": item.truth_text,
                "final_text": item.final_text,
                "cer": item.cer,
                "disagreement_score": item.disagreement_score,
                "audit_status": item.audit_status,
                "models": {
                    "primary": item.primary_engine,
                    "secondary": item.secondary_engine,
                },
                "texts": {
                    "primary": item.primary_text,
                    "secondary": item.secondary_text,
                    "final": item.final_text,
                },
                "text_similarity": {
                    "primary_secondary": item.primary_secondary_similarity,
                    "disagreement_score": item.disagreement_score,
                    "cer": item.cer,
                },
                "timing": {
                    "total_sec": item.timing_total_sec,
                    "primary_sec": item.timing_primary_sec,
                    "secondary_sec": item.timing_secondary_sec,
                },
                "empty_audio_text_len": item.empty_audio_text_len,
                "risk_flags": list(item.risk_flags),
                "false_confident": item.false_confident,
                "simplified_only": item.simplified_only,
                "needs_review": item.needs_review,
                "skipped": item.skipped,
                "skip_reason": item.skip_reason,
                "paths": {
                    "audit_json": str(item.audit_json) if item.audit_json else "",
                    "primary_raw_json": str(item.primary_json) if item.primary_json else "",
                    "secondary_raw_json": str(item.secondary_json) if item.secondary_json else "",
                },
            }
            for item in summary.cases
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _write_benchmark(path: Path, summary: EvalSummary) -> None:
    lines = [
        "# ChineseASR Evaluation Benchmark",
        "",
        f"- Corpus: `{summary.corpus_dir}`",
        f"- Output: `{summary.out_dir}`",
        f"- Total: {summary.total}",
        f"- Evaluated: {summary.evaluated}",
        f"- Skipped: {summary.skipped}",
        f"- Hallucinations: {summary.hallucination_count}",
        f"- False confident: {summary.false_confident_count}",
        "",
        "| Case | Kind | CER | Empty Text Len | Flags |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in summary.cases:
        cer = "" if item.cer is None else f"{item.cer:.3f}"
        flags = ", ".join(item.risk_flags) if item.risk_flags else ""
        if item.skipped:
            flags = f"skipped: {item.skip_reason}"
        lines.append(f"| {item.case_id} | {item.kind} | {cer} | {item.empty_audio_text_len} | {flags} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_review(path: Path, summary: EvalSummary) -> None:
    review_items = [
        item
        for item in summary.cases
        if item.false_confident or item.needs_review or item.empty_audio_text_len > 0 or item.skipped
    ]
    lines = ["# Evaluation Review Queue", ""]
    if not review_items:
        lines.append("- No review items.")
    else:
        for item in review_items:
            flags = ", ".join(item.risk_flags) if item.risk_flags else "none"
            if item.false_confident:
                flags = f"{flags}, false_confident" if flags != "none" else "false_confident"
            lines.extend(
                [
                    f"## {item.case_id}",
                    "",
                    f"- Kind: `{item.kind}`",
                    f"- Flags: `{flags}`",
                    f"- Truth: `{item.truth_text}`",
                    f"- Final: `{item.final_text}`",
                    f"- Skipped: `{str(item.skipped).lower()}`",
                    f"- Skip reason: `{item.skip_reason}`",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_metric_text(text: str) -> str:
    simplified = to_simplified(text)
    return re.sub(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()\[\]【】<>《》|-]+", "", simplified).lower()


def _empty_audio_text_len(text: str) -> int:
    normalized = _normalize_metric_text(text)
    normalized = normalized.replace("疑似", "").replace("听不清", "").replace("nospeechtextreturned", "")
    return len(normalized)


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current[right_index - 1] + 1
            delete_cost = previous[right_index] + 1
            replace_cost = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]

"""Pinned, on-demand forced alignment. Timing is not lexical truth."""
from __future__ import annotations
import gc
import math
import os
from pathlib import Path
import wave
from typing import Any

from .config import ModelConfig, load_model_config
from .model_lifecycle import alignment_contract, verify_aligner, file_hash, write_json_atomic
from .result_writer import text_sha256
from .text_comparison import normalize_comparison, strip_audit_markers


def validate_items(items: list[dict[str, Any]], duration_ms: int) -> None:
    previous = -1
    for item in items:
        start, end = item["start_ms"], item["end_ms"]
        if not all(not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) for x in (start, end)):
            raise ValueError("Forced alignment returned non-finite timestamps")
        if start < 0 or end < start or end > duration_ms + 100 or start < previous:
            raise ValueError("Forced alignment returned invalid or non-monotonic timestamps")
        previous = start


def audio_duration_ms(audio: Path) -> int:
    with wave.open(str(audio), "rb") as handle:
        return round(handle.getnframes() * 1000 / handle.getframerate())


def align_many(
    jobs: list[tuple[Path, str, Path]], *, device: str,
    config: ModelConfig | None = None,
) -> list[dict[str, Any]]:
    """Caller owns the existing GPU lease. Load once and isolate input failures.

    jobs contain PCM audio, supplied text and a sidecar destination. Original
    ASR text/raw files are never rewritten by this function.
    """
    from .gpu_broker import require_worker_gpu_lease
    require_worker_gpu_lease(device)
    config = config or load_model_config()
    options, directory, _ = alignment_contract(config)
    outcomes: list[dict[str, Any]] = []
    valid = []
    for index, (audio, raw_text, target) in enumerate(jobs):
        text = strip_audit_markers(raw_text).strip()
        result = {"schema": "zh_asr.forced_alignment.v1", "status": "pending",
            "audio": str(audio.resolve()), "text": text, "text_sha256": text_sha256(text),
            "items": [], "lexical_truth_verified": False, "exact_text_coverage": False}
        try:
            duration = audio_duration_ms(audio)
            if not text:
                result.update(status="skipped", reason="empty_text")
            elif duration <= 0 or duration > float(options.get("max_audio_sec", 300)) * 1000:
                raise ValueError("Audio exceeds the configured forced-aligner duration contract")
            else:
                result.update(audio_sha256=file_hash(audio), duration_ms=duration)
                valid.append(index)
        except Exception as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
        outcomes.append(result)
    model = None
    try:
        if valid:
            identity = verify_aligner(config)
            import torch
            from qwen_asr import Qwen3ForcedAligner
            dtype = getattr(torch, str(options.get("dtype", "bfloat16")))
            model = Qwen3ForcedAligner.from_pretrained(str(directory), dtype=dtype,
                device_map=device, local_files_only=True)
            for index in valid:
                audio, _, _ = jobs[index]
                result = outcomes[index]
                try:
                    aligned = model.align(audio=str(audio), text=result["text"],
                        language=str(options.get("language", "Chinese")))
                    if len(aligned) != 1:
                        raise RuntimeError("Forced alignment returned an unexpected result count")
                    items = [{"text": token.text, "start_ms": round(token.start_time * 1000),
                              "end_ms": round(token.end_time * 1000)} for token in aligned[0]]
                    validate_items(items, result["duration_ms"])
                    aligned_text = "".join(item["text"] for item in items)
                    result.update(status="succeeded", items=items, model_identity=identity,
                        exact_text_coverage=normalize_comparison(aligned_text) == normalize_comparison(result["text"]))
                except Exception as error:
                    result.update(status="failed", error=f"{type(error).__name__}: {error}")
    except Exception as error:
        for index in valid:
            if outcomes[index]["status"] == "pending":
                outcomes[index].update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if model is not None:
            del model
            gc.collect()
            if str(device).startswith("cuda"):
                import torch
                torch.cuda.empty_cache()
    for (_, _, target), result in zip(jobs, outcomes):
        write_json_atomic(target, result)
    return outcomes


def timestamp_supported_overlap(
    previous_text: str, current_text: str, previous: dict[str, Any], current: dict[str, Any],
    previous_start_ms: int, current_start_ms: int, previous_end_ms: int,
) -> int:
    """Only remove a literal prefix whose tokens map to the same overlap in both clips.

    Missing/uncertain timing returns zero. Genuine repetitions outside the shared
    source interval are preserved, and no fuzzy or semantic deduplication occurs.
    """
    if current_start_ms >= previous_end_ms:
        return 0
    for payload, text in ((previous, previous_text), (current, current_text)):
        if payload.get("status") != "succeeded" or not payload.get("exact_text_coverage"):
            return 0
        if normalize_comparison(payload.get("text", "")) != normalize_comparison(strip_audit_markers(text)):
            return 0
    if any(marker in previous_text or marker in current_text for marker in ("[疑似]", "[听不清]")):
        return 0

    def character_times(payload, offset):
        chars, times = "", []
        for item in payload["items"]:
            part = normalize_comparison(item["text"])
            chars += part
            times.extend([(offset + item["start_ms"], offset + item["end_ms"])] * len(part))
        return chars, times

    left_chars, left_times = character_times(previous, previous_start_ms)
    right_chars, right_times = character_times(current, current_start_ms)
    left, right = previous_text.rstrip(), current_text.lstrip()
    for size in range(min(len(left), len(right)), 1, -1):
        prefix = right[:size]
        key = normalize_comparison(prefix)
        if not (left.endswith(prefix) and key and left_chars.endswith(key) and right_chars.startswith(key)):
            continue
        matched = zip(left_times[-len(key):], right_times[:len(key)])
        supported = True
        for (a, b), (c, d) in matched:
            # Match the same sound, not separate repetitions in a shared clip.
            if (min(a, c) < current_start_ms - 80 or max(b, d) > previous_end_ms + 80
                    or min(b, d) <= max(a, c) or abs((a+b-c-d)/2) > 120):
                supported = False
                break
        if supported:
            return size
    return 0

"""Small public adapter for known-text alignment, using the existing ASR runtime.

No ASR pair, service, model download, transcription correction or audio rewrite.
The CLI owns the established broker/supervisor; results are published only after
coverage and both original input hashes have been rechecked.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path
from .alignment import align_many, validate_items
from .audio_frontend import prepare_pcm16_mono
from .config import load_model_config
from .model_lifecycle import alignment_contract, file_hash, write_json_atomic
from .result_writer import text_sha256

SCHEMA = "zh_asr.alignment-entry.v1"


def describe(config=None):
    config = config or load_model_config()
    options, directory, _ = alignment_contract(config)
    return {"schema": SCHEMA, "operation": "known_text_alignment", "read_only": True,
        "model": options.get("model"), "model_directory": str(directory),
        "weights_present": (directory / "model.safetensors").is_file(),
        "max_audio_sec": float(options.get("max_audio_sec", 300)),
        "word_unit": "seconds", "lexical_truth_verified": False}


def align_file(audio, text_file, output, *, device="cuda:0", config=None, timeout_sec=300):
    if not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise ValueError("alignment timeout must be finite and positive")
    audio, text_file, output = map(lambda p: Path(p).resolve(), (audio, text_file, output))
    if output in (audio, text_file):
        raise ValueError("Alignment output must not replace either original input")
    raw_text = text_file.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError("Supplied alignment text is empty")
    audio_hash, text_file_hash = file_hash(audio), file_hash(text_file)
    config = config or load_model_config()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".align-", dir=output.parent) as temporary:
        root = Path(temporary)
        prepared = prepare_pcm16_mono(audio, root, timeout_sec=min(timeout_sec, 120))
        options, _, _ = alignment_contract(config)
        maximum = float(options.get("max_audio_sec", 300))
        if prepared.duration_sec > maximum:
            raise ValueError(f"Known-text alignment supports at most {maximum:g}s per clip; split at reviewed scene boundaries")
        results = align_many([(prepared.path, raw_text, root / "aligned.json")], device=device, config=config)
        if len(results) != 1:
            raise RuntimeError("Unexpected alignment result count")
        result = results[0]
        if result.get("status") != "succeeded" or not result.get("exact_text_coverage"):
            raise RuntimeError("Alignment failed or text coverage is incomplete: " + str(result.get("error", result.get("reason", "coverage"))))
        items = result.get("items", [])
        validate_items(items, round(prepared.duration_sec * 1000))
        if not items or any(not isinstance(item.get("text"), str) or not item["text"].strip() for item in items):
            raise RuntimeError("Alignment returned no usable text units")
        if file_hash(audio) != audio_hash or file_hash(text_file) != text_file_hash:
            raise RuntimeError("Alignment inputs changed during processing")
        value = {"schema": SCHEMA, "status": "succeeded", "audio_sha256": audio_hash,
            "text_sha256": text_sha256(raw_text), "text_file_sha256": text_file_hash,
            "duration_seconds": prepared.duration_sec, "exact_text_coverage": True,
            "lexical_truth_verified": False, "model_identity": result.get("model_identity"),
            "words": [{"word": x["text"], "start": x["start_ms"] / 1000,
                "end": x["end_ms"] / 1000} for x in items]}
        write_json_atomic(output, value)
    return value

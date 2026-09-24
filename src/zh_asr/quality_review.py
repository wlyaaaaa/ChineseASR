"""Audio-grounded review sidecars. Never overwrite raw ASR or vote text into truth."""
from __future__ import annotations
from difflib import SequenceMatcher
import html
import json
import os
from pathlib import Path
import wave
from typing import Any

from .alignment import align_many, audio_duration_ms, validate_items
from .config import ModelConfig
from .model_lifecycle import file_hash, verify_aligner, write_json_atomic
from .result_writer import extract_text, text_sha256
from .text_comparison import normalize_comparison, critical_differences, strip_audit_markers


def disputed_ranges(primary: str, secondary: str, alignment: dict[str, Any], duration_ms: int,
                    context_ms: int = 1500) -> list[dict[str, Any]]:
    left, right = normalize_comparison(primary), normalize_comparison(secondary)
    if left == right:
        return []
    owners = []
    aligned = ""
    for token in alignment.get("items", []):
        part = normalize_comparison(token["text"])
        aligned += part
        owners.extend([token] * len(part))
    timing_usable = alignment.get("status") == "succeeded" and aligned == left
    spans = []
    for tag, a, b, c, d in SequenceMatcher(a=left, b=right, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if timing_usable and owners:
            first = owners[min(a, len(owners)-1)]
            last = owners[min(max(a, b-1), len(owners)-1)]
            start = max(0, first["start_ms"] - context_ms)
            end = min(duration_ms, last["end_ms"] + context_ms)
        else:
            start, end = 0, duration_ms
        reason = ",".join(critical_differences(left[a:b], right[c:d])) or "lexical_difference"
        entry = {"start_ms": start, "end_ms": end, "reasons": [reason],
                 "primary_fragment": left[max(0, a-12):b+12],
                 "secondary_fragment": right[max(0, c-12):d+12],
                 "timing_precision": "forced_alignment_with_context" if timing_usable else "whole_chunk"}
        if spans and start <= spans[-1]["end_ms"]:
            spans[-1]["end_ms"] = max(end, spans[-1]["end_ms"])
            spans[-1]["reasons"] = sorted(set(spans[-1]["reasons"] + [reason]))
            spans[-1]["primary_fragment"] += " / " + entry["primary_fragment"]
            spans[-1]["secondary_fragment"] += " / " + entry["secondary_fragment"]
        else:
            spans.append(entry)
    return spans


def uncovered_speech_ranges(vad_segments: list[list[int]], alignment: dict[str, Any],
                            source_start_ms: int, duration_ms: int, min_gap_ms: int = 650) -> list[list[int]]:
    if alignment.get("status") != "succeeded" or not alignment.get("exact_text_coverage"):
        return []
    coverage = sorted((max(0, x["start_ms"]-120), min(duration_ms, x["end_ms"]+120))
                      for x in alignment.get("items", []))
    gaps = []
    for a, b in vad_segments:
        cursor, end = max(0, a-source_start_ms), min(duration_ms, b-source_start_ms)
        if cursor >= end:
            continue
        for left, right in coverage:
            if right <= cursor or left >= end:
                continue
            if left - cursor >= min_gap_ms:
                gaps.append([cursor, min(left, end)])
            cursor = max(cursor, right)
        if end - cursor >= min_gap_ms:
            gaps.append([cursor, end])
    return gaps


def write_clip(source: Path, target: Path, start_ms: int, end_ms: int) -> None:
    with wave.open(str(source), "rb") as reader:
        rate = reader.getframerate()
        start = min(reader.getnframes(), max(0, start_ms * rate // 1000))
        end = min(reader.getnframes(), max(start, end_ms * rate // 1000))
        if start == end:
            raise ValueError("Review interval contains no audio samples")
        reader.setpos(start)
        frames = reader.readframes(end-start)
        target.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target), "wb") as writer:
            writer.setparams(reader.getparams())
            writer.writeframes(frames)


def enhance_chunks(states, *, config: ModelConfig, device: str, cache_dir: Path,
                   output_dir: Path, vad: dict[str, Any] | None = None,
                   align_fn=align_many, generate_fn=None) -> dict[str, Any]:
    """Extend a completed long run with timing and bounded supplementary audio reads.

    The caller's GPU lease spans this operation. Original audit statuses remain
    textual-comparison results, while this sidecar adds coverage warnings.
    """
    from .gpu_broker import require_worker_gpu_lease
    require_worker_gpu_lease(device)
    options = config.quality or {}
    root = output_dir / "quality"
    root.mkdir(parents=True, exist_ok=True)
    context_ms = round(float(options.get("review_context_sec", 1.5)) * 1000)
    records = []
    jobs = []
    pending = []
    identity = None
    alignment_error = ""
    if options.get("align"):
        try:
            identity = verify_aligner(config)
        except Exception as error:
            alignment_error = f"{type(error).__name__}: {error}"
    for state in states:
        if state.status != "succeeded":
            continue
        state.outputs.pop("alignment_json", None)
        audit_path = state.outputs.get("audit_json")
        if not audit_path or not Path(audit_path).is_file():
            continue
        audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        audio = Path(state.spec.audio_path)
        if not audio.is_file():
            continue
        chosen = strip_audit_markers(str(audit.get("final_text", "")))
        path = root / (state.spec.chunk_id + ".alignment.json")
        record = {"state": state, "audit": audit, "audio": audio,
                  "alignment": {"status": "unavailable", "items": []}, "alignment_path": path}
        if identity and chosen:
            cached = None
            if path.is_file():
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    validate_items(value.get("items", []), audio_duration_ms(audio))
                    if (value.get("status") == "succeeded" and value.get("audio_sha256") == file_hash(audio)
                            and value.get("text_sha256") == text_sha256(chosen)
                            and value.get("model_identity") == identity):
                        cached = value
                except (ValueError, KeyError, OSError):
                    pass
            if cached:
                record["alignment"] = cached
            else:
                pending.append(len(records))
                jobs.append((audio, chosen, path))
        records.append(record)
    if jobs:
        aligned_results = align_fn(jobs, device=device, config=config)
        if not isinstance(aligned_results, list) or len(aligned_results) != len(pending):
            raise RuntimeError("Forced alignment result count does not match submitted clips")
        for index, aligned in zip(pending, aligned_results, strict=True):
            records[index]["alignment"] = aligned
    entries = []
    review_inputs = []
    review_indices = []
    review_engine = str(options.get("review_engine") or "")
    for record in records:
        state, audit, audio, aligned = (record[k] for k in ("state", "audit", "audio", "alignment"))
        duration = audio_duration_ms(audio)
        if aligned.get("status") == "succeeded" and record["alignment_path"].is_file():
            state.outputs["alignment_json"] = str(record["alignment_path"])
        ranges = disputed_ranges(audit.get("primary_text", ""), audit.get("secondary_text", ""),
                                 aligned, duration, context_ms)
        coverage_gaps = uncovered_speech_ranges((vad or {}).get("segments", []), aligned,
                                                state.spec.start_ms, duration)
        if not audit.get("primary_text") and not audit.get("secondary_text"):
            coverage_gaps.extend((max(0, a-state.spec.start_ms), min(duration, b-state.spec.start_ms))
                for a, b in (vad or {}).get("segments", [])
                if a < state.spec.end_ms and b > state.spec.start_ms)
        for a, b in coverage_gaps:
            ranges.append({"start_ms": max(0, a-context_ms), "end_ms": min(duration, b+context_ms),
                "reasons": ["vad_speech_without_aligned_text"], "primary_fragment": "",
                "secondary_fragment": "", "timing_precision": "vad_and_forced_alignment_hint"})
        if (vad or {}).get("status") == "available" and aligned.get("text"):
            overlaps_speech = any(a < state.spec.end_ms and b > state.spec.start_ms
                                  for a, b in (vad or {}).get("segments", []))
            if not overlaps_speech:
                ranges.append({"start_ms": 0, "end_ms": duration,
                    "reasons": ["text_without_vad_speech_hint_not_proof_of_silence"],
                    "primary_fragment": audit.get("primary_text", ""),
                    "secondary_fragment": audit.get("secondary_text", ""), "timing_precision": "whole_chunk"})
        if not ranges and audit.get("needs_review"):
            ranges = [{"start_ms": 0, "end_ms": duration, "reasons": list(audit.get("flags", [])) or ["audit_review"],
                       "primary_fragment": audit.get("primary_text", ""),
                       "secondary_fragment": audit.get("secondary_text", ""), "timing_precision": "whole_chunk"}]
        for number, span in enumerate(merge_review_ranges(ranges)):
            identifier = f"{state.spec.chunk_id}-review-{number:03d}"
            clip = root / "clips" / (identifier + ".wav")
            write_clip(audio, clip, span["start_ms"], span["end_ms"])
            entry = {"id": identifier, "chunk_id": state.spec.chunk_id,
                "audio_start_ms": state.spec.start_ms + span["start_ms"],
                "audio_end_ms": state.spec.start_ms + span["end_ms"],
                "clip": str(clip.resolve()), "clip_sha256": file_hash(clip),
                "audit": str(Path(state.outputs["audit_json"]).resolve()),
                "audit_sha256": file_hash(Path(state.outputs["audit_json"])),
                "primary_engine": audit.get("primary_engine"), "secondary_engine": audit.get("secondary_engine"),
                "supplemental_status": "not_requested", "needs_human_review": True,
                **span}
            if review_engine and review_engine not in {audit.get("primary_engine"), audit.get("secondary_engine")}:
                review_inputs.append(clip)
                review_indices.append(len(entries))
            elif review_engine:
                entry["supplemental_status"] = "engine_already_present_not_repeated"
            entries.append(entry)
    if review_inputs:
        from .long_audio import _runtime_artifact_identity, _runtime_code_identity
        from .result_writer import canonical_json_sha256
        model_identity = _runtime_artifact_identity(config, (review_engine,), cache_dir)
        code_identity = _runtime_code_identity()["sha256"]
        pending_inputs, pending_indices, cache_keys = [], [], {}
        for index, clip in zip(review_indices, review_inputs, strict=True):
            entry = entries[index]
            key = canonical_json_sha256({"clip_sha256": entry["clip_sha256"],
                "engine": review_engine, "artifacts": model_identity, "code": code_identity,
                "config": file_hash(config.path)})
            raw_path = root / (entry["id"] + ".supplemental.raw.json")
            cache_path = root / (entry["id"] + ".supplemental.cache.json")
            cache_keys[index] = (key, raw_path, cache_path)
            cached = None
            try:
                metadata = json.loads(cache_path.read_text(encoding="utf-8"))
                if metadata.get("key") == key and metadata.get("raw_sha256") == file_hash(raw_path):
                    cached = json.loads(raw_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                pass
            if cached is None:
                pending_inputs.append(clip)
                pending_indices.append(index)
            else:
                entry.update(supplemental_engine=review_engine, supplemental_status="reused",
                    supplemental_text=extract_text(cached), supplemental_raw=str(raw_path.resolve()),
                    supplemental_raw_sha256=file_hash(raw_path),
                    decision="additional_audio_evidence_only_no_majority_vote")
        if pending_inputs:
            if generate_fn is None:
                from .pipeline import _generate_many_for_strict
                generate_fn = _generate_many_for_strict
            generated = generate_fn(audio_paths=pending_inputs,
                out_dirs=[root / "supplemental" / entries[i]["id"] for i in pending_indices],
                engine=review_engine, device=device, cache_dir=cache_dir, config=config)
            if len(generated["results"]) != len(pending_indices) or len(generated["errors"]) != len(pending_indices):
                raise RuntimeError("Supplemental ASR result count does not match review windows")
            for index, raw, error in zip(pending_indices, generated["results"], generated["errors"], strict=True):
                entry = entries[index]
                key, raw_path, cache_path = cache_keys[index]
                write_json_atomic(raw_path, raw)
                if not error:
                    write_json_atomic(cache_path, {"key": key, "raw_sha256": file_hash(raw_path)})
                entry.update(supplemental_engine=review_engine,
                    supplemental_status="failed" if error else "succeeded", supplemental_error=error,
                    supplemental_text=extract_text(raw), supplemental_raw=str(raw_path.resolve()),
                    supplemental_raw_sha256=file_hash(raw_path),
                    decision="additional_audio_evidence_only_no_majority_vote")
    alignment_failures = sum(r["alignment"].get("status") == "failed" for r in records)
    incomplete_alignment = sum(r["alignment"].get("status") == "succeeded" and not r["alignment"].get("exact_text_coverage") for r in records)
    supplemental_failed = sum(x.get("supplemental_status") == "failed" for x in entries)
    payload = {"schema": "zh_asr.quality_review.v1", "status": "degraded" if alignment_error or alignment_failures or incomplete_alignment or supplemental_failed else "completed",
        "supplemental_failed": supplemental_failed,
        "alignment_requested": bool(options.get("align")), "alignment_error": alignment_error,
        "incomplete_alignment": incomplete_alignment,
        "alignment_model": identity, "alignment_succeeded": sum(r["alignment"].get("status") == "succeeded" for r in records),
        "alignment_failed": sum(r["alignment"].get("status") == "failed" for r in records),
        "needs_review": bool(entries) or bool(alignment_error) or bool(alignment_failures) or bool(incomplete_alignment), "review_count": len(entries), "entries": entries,
        "audio_uploaded": False, "lexical_results_rewritten": False,
        "limitations": "Alignment locates supplied text, not truth. Supplemental ASR does not certify a winner."}
    target = output_dir / "quality.review.json"
    write_json_atomic(target, payload)
    write_review_html(output_dir / "quality.review.html", payload)
    return payload


def write_review_html(path: Path, payload: dict[str, Any]) -> None:
    cards = []
    for item in payload.get("entries", []):
        def escape(value):
            return html.escape(str(value), quote=True)
        from urllib.parse import quote
        source = quote(os.path.relpath(item["clip"], path.parent).replace("\\", "/"), safe="/")
        cards.append(f'<section><h2>{escape(item["id"])} · {item["audio_start_ms"]/1000:.2f}–{item["audio_end_ms"]/1000:.2f} 秒</h2>'
            f'<p>{escape(", ".join(item["reasons"]))}</p><audio controls preload="none" src="{escape(source)}"></audio>'
            f'<p><b>{escape(item["primary_engine"])}</b>：{escape(item["primary_fragment"])}</p>'
            f'<p><b>{escape(item["secondary_engine"])}</b>：{escape(item["secondary_fragment"])}</p>'
            f'<p><b>补充识别</b>：{escape(item.get("supplemental_text", item["supplemental_status"]))}</p>'
            f'<textarea data-id="{escape(item["id"])}" data-clip="{escape(item["clip_sha256"])}" data-audit="{escape(item["audit_sha256"])}" placeholder="听完原音后记录修订或备注；不会改写原始转写"></textarea></section>')
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ASR 音频复核</title>'
    page += '<style>body{font:17px/1.7 system-ui;max-width:980px;margin:36px auto;padding:0 20px}section{border:1px solid #ccc;border-radius:12px;padding:20px;margin:22px 0}audio,textarea{width:100%}textarea{min-height:90px}button{padding:10px 18px;font:inherit}</style>'
    page += '<h1>音频复核</h1><p>先听音频，再判断候选。对齐时间和第三路转写都不等于真相；本页不联网、不自动保存或改写原件。</p>'
    page += '<p><b>本轮复核状态：</b>' + html.escape(str(payload.get("status", "unknown"))) + '</p>'
    if payload.get("alignment_error") or payload.get("error"):
        page += '<p>' + html.escape(str(payload.get("alignment_error") or payload.get("error"))) + '</p>'
    page += '<button id="export">导出本次复核笔记</button>' + (''.join(cards) or '<p>本轮没有生成待复核片段；这不代表逐字准确已经得到证明。</p>')
    page += '<script>document.getElementById("export").onclick=()=>{const notes=[...document.querySelectorAll("textarea")].filter(x=>x.value.trim()).map(x=>({id:x.dataset.id,clip_sha256:x.dataset.clip,audit_sha256:x.dataset.audit,note:x.value}));const blob=new Blob([JSON.stringify({schema:"zh_asr.review_notes.v1",notes},null,2)],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="asr-review-notes.json";a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};</script></html>'
    path.write_text(page, encoding="utf-8")


def merge_review_ranges(ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    for original in sorted(ranges, key=lambda x: (x["start_ms"], x["end_ms"])):
        span = dict(original)
        if merged and span["start_ms"] <= merged[-1]["end_ms"]:
            old = merged[-1]
            old["end_ms"] = max(old["end_ms"], span["end_ms"])
            old["reasons"] = sorted(set(old["reasons"] + span["reasons"]))
            for key in ("primary_fragment", "secondary_fragment"):
                if span.get(key) and span[key] not in old[key]:
                    old[key] += " / " + span[key]
            if span["timing_precision"] != old["timing_precision"]:
                old["timing_precision"] = "combined_context_window"
        else:
            merged.append(span)
    return merged


def enhance_single(audio: Path, paths: dict[str, Any], *, config: ModelConfig,
                   device: str, cache_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Give short-file strict transcription the same sidecars as long recordings."""
    from types import SimpleNamespace
    from .audio_frontend import prepare_pcm16_mono
    from .audio_quality import speech_boundaries
    try:
        prepared = prepare_pcm16_mono(audio.resolve(), output_dir / "_derived")
        duration = audio_duration_ms(prepared.path)
        state = SimpleNamespace(status="succeeded", outputs=dict(paths),
            spec=SimpleNamespace(audio_path=prepared.path, chunk_id="single", start_ms=0, end_ms=duration))
        vad = (speech_boundaries(prepared.path, cache_dir, config, output_dir / "_derived")
               if config.quality.get("cut_strategy") == "vad" else None)
        result = enhance_chunks([state], config=config, device=device, cache_dir=cache_dir,
            output_dir=output_dir, vad=vad)
    except Exception as error:
        result = {"schema": "zh_asr.quality_review.v1", "status": "failed", "needs_review": True,
            "error": f"{type(error).__name__}: {error}", "entries": [], "lexical_results_rewritten": False}
        write_json_atomic(output_dir / "quality.review.json", result)
        write_review_html(output_dir / "quality.review.html", result)
    paths["quality_review"] = output_dir / "quality.review.json"
    if (output_dir / "quality.review.html").is_file():
        paths["review_html"] = output_dir / "quality.review.html"
    return result

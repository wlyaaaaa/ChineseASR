from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid
import wave
import yaml

from zh_asr.cloud_review import (auto_cloud_status, finalize_cloud_result,
    prepare_cloud_request, CloudReviewError, _split_prepared_wav)
from zh_asr.transcript_readback import _inspect_cloud_result_candidate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models.yaml"


def _wav(path: Path, *, stereo: bool = False, frames: int = 1600) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2 if stereo else 1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        if stereo:
            handle.writeframes(((0).to_bytes(2, "little", signed=True) +
                                (12000).to_bytes(2, "little", signed=True)) * frames)
        else:
            handle.writeframes(b"\x00\x00" * frames)


class CloudPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "outputs" / "cloud-jobs"
        self.root.mkdir(parents=True)
        self.audio = self.base / "voice.wav"
        _wav(self.audio)
        self.local = self.base / "local"
        self.local.mkdir()
        (self.local / "quality.review.json").write_text('{"needs_review":true}', encoding="utf-8")

    def intent(self, **changes) -> Path:
        job_id = str(uuid.uuid4())
        payload = {"schema": "zh_asr.cloud_review_intent.v1", "job_id": job_id,
            "purpose": "quality_review", "cloud_upload_authorized": True,
            "automatic_review": True, "audio_path": str(self.audio),
            "local_out_dir": str(self.local), "hotwords": {},
            "chunk_sec": 180, "overlap_sec": 1}
        payload.update(changes)
        path = self.root / (job_id + ".intent.json")
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_prepared_chunks_and_protocol_are_in_root_and_configurable(self) -> None:
        intent = self.intent(model_profile="message", hotwords={"田家庵": 3})
        prepared = prepare_cloud_request(intent, self.root, CONFIG)
        self.assertEqual("ready", prepared["status"])
        self.assertEqual("websocket", prepared["protocol"])
        request = json.loads(Path(prepared["request_path"]).read_text(encoding="utf-8"))
        self.assertEqual("qwen-audio-3.1-asr-flash-message", request["model"])
        self.assertEqual({"田家庵": 3}, request["parameters"]["vocabulary"])
        self.assertEqual("pcm", request["parameters"]["format"])
        self.assertEqual("2026-12-21T00:00:00+08:00", request["send_before_utc"])
        self.assertTrue(all(Path(item["path"]).resolve().is_relative_to(self.root.resolve())
                            for item in request["chunks"]))
        self.assertEqual(1, len(request["chunks"]))

    def test_automatic_expiry_skips_and_writes_plain_reason(self) -> None:
        intent = self.intent()
        result = prepare_cloud_request(intent, self.root, CONFIG,
            now=datetime.fromisoformat("2026-12-21T00:00:00+08:00"))
        self.assertEqual("skipped", result["status"])
        self.assertEqual("free_period_expired", result["pause_reason"])
        self.assertFalse(list(self.root.glob("*.pending.json")))
        sidecar = json.loads((self.local / "cloud.review.json").read_text(encoding="utf-8"))
        self.assertEqual("免费期已到期，云端未跑", sidecar["message"])
        self.assertFalse(sidecar["cloud_upload_performed"])

    def test_explicit_expired_upload_is_allowed_and_marked_billable(self) -> None:
        intent = self.intent(automatic_review=False, model_profile="short")
        result = prepare_cloud_request(intent, self.root, CONFIG,
            now=datetime.fromisoformat("2026-12-21T01:00:00+08:00"))
        self.assertEqual("ready", result["status"])
        self.assertIn("可能计费", result["billing_warning"])
        request = json.loads(Path(result["request_path"]).read_text(encoding="utf-8"))
        self.assertIsNone(request["send_before_utc"])
        self.assertIn("可能计费", request["billing_warning"])
        projected = finalize_cloud_result(intent, self.root,
            broker_error="fixture_broker_unavailable", config_path=CONFIG)
        self.assertIn("可能计费", projected["billing_warning"])
        self.assertIn("可能计费", json.loads((self.local / "cloud.review.json").read_text(
            encoding="utf-8"))["billing_warning"])

    def test_selected_channel_is_prepared_before_secret_worker(self) -> None:
        stereo = self.base / "stereo.wav"
        _wav(stereo, stereo=True)
        intent = self.intent(audio_path=str(stereo), channel_index=1)
        prepared = prepare_cloud_request(intent, self.root, CONFIG)
        request = json.loads(Path(prepared["request_path"]).read_text(encoding="utf-8"))
        with wave.open(request["chunks"][0]["path"], "rb") as chunk:
            self.assertEqual(1, chunk.getnchannels())
            value = int.from_bytes(chunk.readframes(1)[:2], "little", signed=True)
        self.assertEqual(12000, value)
        self.assertEqual(1, request["selected_channel"])

    def test_chunks_are_balanced_within_requested_and_provider_limits(self) -> None:
        _wav(self.audio, frames=16000 * 12 + 1600)
        requested = _split_prepared_wav(self.audio, self.base / "requested",
            chunk_sec=12, overlap_sec=1, max_chunk_sec=18)
        self.assertEqual([(0, 6550), (5550, 12100)],
                         [(c["start_ms"], c["end_ms"]) for c in requested])
        bounded = _split_prepared_wav(self.audio, self.base / "bounded",
            chunk_sec=18, overlap_sec=1, max_chunk_sec=12)
        self.assertEqual([(c["start_ms"], c["end_ms"]) for c in requested],
                         [(c["start_ms"], c["end_ms"]) for c in bounded])
        _wav(self.audio, frames=16000 * 120 + 1600)
        overlapped = _split_prepared_wav(self.audio, self.base / "overlapped",
            chunk_sec=120, overlap_sec=119, max_chunk_sec=120)
        self.assertEqual(2, len(overlapped))

    def test_important_recording_has_no_tiny_tail_or_over_limit_chunk(self) -> None:
        total = 16000 * 2143 + 1664
        _wav(self.audio, frames=total)
        chunks = _split_prepared_wav(self.audio, self.base / "balanced",
            chunk_sec=120, overlap_sec=1, max_chunk_sec=120)
        self.assertEqual(19, len(chunks))
        self.assertEqual((0, 2143104), (chunks[0]["start_ms"], chunks[-1]["end_ms"]))
        lengths = []
        for chunk in chunks:
            with wave.open(chunk["path"], "rb") as reader:
                lengths.append(reader.getnframes())
        self.assertTrue(all(5 * 16000 <= length <= 120 * 16000 for length in lengths))
        self.assertLessEqual(max(lengths) - min(lengths), 1)
        self.assertEqual(total, sum(lengths) - 18 * 16000)
        self.assertTrue(all(left["end_ms"] - right["start_ms"] == 1000
                            for left, right in zip(chunks, chunks[1:])))

    def test_resume_keeps_original_request_and_parameter_changes_start_new_job(self) -> None:
        original = self.intent()
        ready = prepare_cloud_request(original, self.root, CONFIG)
        pending = Path(ready["request_path"])
        payload = pending.read_bytes()
        pending.replace(pending.with_name(ready["job_id"] + ".unclaimed.json"))
        retry = self.intent()
        resumed = prepare_cloud_request(retry, self.root, CONFIG)
        self.assertEqual(ready["job_id"], resumed["job_id"])
        self.assertEqual(payload, Path(resumed["request_path"]).read_bytes())
        self.assertFalse(retry.exists())
        changed = self.intent(overlap_sec=0)
        fresh = prepare_cloud_request(changed, self.root, CONFIG)
        self.assertNotEqual(ready["job_id"], fresh["job_id"])
        with self.assertRaisesRegex(CloudReviewError, "resume_job_not_compatible"):
            prepare_cloud_request(changed, self.root, CONFIG, resume_job_id=ready["job_id"])

    def test_vendor_no_words_preserves_other_chunks_and_failed_intermediates(self) -> None:
        _wav(self.audio, frames=16000 * 14)
        intent = self.intent(chunk_sec=8, overlap_sec=1)
        ready = prepare_cloud_request(intent, self.root, CONFIG)
        request = json.loads(Path(ready["request_path"]).read_text(encoding="utf-8"))
        document = json.loads(intent.read_text(encoding="utf-8"))
        provider = self.root / (ready["job_id"] + ".provider.json")
        body = {"schema": "passwordcenter.vendor-result.v1", "vendor": "qwen",
            "request_sha256": document["vendor_request_sha256"], "status": "failed",
            "credential_result": "Scope-Error", "error_code": "http_400",
            "provider_error_message": "ASR_RESPONSE_HAVE_NO_WORDS", "new_requests": 2,
            "results": [{"id": "1", "http_status": 200,
                         "raw_response": {"output": {"text": "已有文字"}}}]}
        pending = Path(ready["request_path"])
        pending.replace(pending.with_name(ready["job_id"] + ".unclaimed.json"))
        body["provider_error_message"] = "network_failure"
        provider.write_text(json.dumps(body), encoding="utf-8")
        failed = finalize_cloud_result(intent, self.root, provider_path=provider, config_path=CONFIG)
        work = self.root / (ready["job_id"] + ".work")
        self.assertEqual("failed", failed["status"])
        self.assertTrue(work.is_dir())
        body["provider_error_message"] = "ASR_RESPONSE_HAVE_NO_WORDS"
        provider.write_text(json.dumps(body), encoding="utf-8")
        final = finalize_cloud_result(intent, self.root, provider_path=provider, config_path=CONFIG)
        self.assertEqual("succeeded", final["status"])
        self.assertEqual("已有文字", final["text"])
        self.assertEqual("no_speech", final["chunks"][-1]["status"])
        self.assertEqual("Success", final["credential_result"])
        self.assertFalse(work.exists())
        self.assertTrue(self.audio.is_file())
        self.assertEqual("已有文字", (self.root / (ready["job_id"] + ".transcript.txt")).read_text(encoding="utf-8").strip())
        self.audio.unlink()
        restored = finalize_cloud_result(intent, self.root, provider_path=provider, config_path=CONFIG)
        self.assertEqual("已有文字", restored["text"])

    def test_explicit_resume_after_expiry_updates_warning_without_changing_request(self) -> None:
        intent = self.intent(automatic_review=False, model_profile="short")
        ready = prepare_cloud_request(intent, self.root, CONFIG,
            now=datetime.fromisoformat("2026-10-20T00:00:00+08:00"))
        payload = Path(ready["request_path"]).read_bytes()
        retry = self.intent(automatic_review=False, model_profile="short")
        resumed = prepare_cloud_request(retry, self.root, CONFIG,
            now=datetime.fromisoformat("2026-12-22T00:00:00+08:00"))
        self.assertEqual(ready["job_id"], resumed["job_id"])
        self.assertEqual(payload, Path(resumed["request_path"]).read_bytes())
        result = finalize_cloud_result(intent, self.root,
            broker_error="fixture_broker_unavailable", config_path=CONFIG)
        self.assertIn("可能计费", result["billing_warning"])

    def test_model_replacement_capabilities_come_from_config(self) -> None:
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        short = document["cloud_review"]["models"]["short"]
        short["id"] = "future-provider-model_1.0"
        short["speaker_dialect_exclusive"] = False
        temporary_config = self.base / "models.yaml"
        temporary_config.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
        intent = self.intent(model_profile="short", speaker_diarization=True, keep_dialect=True)
        ready = prepare_cloud_request(intent, self.root, temporary_config)
        request = json.loads(Path(ready["request_path"]).read_text(encoding="utf-8"))
        self.assertEqual("future-provider-model_1.0", request["model"])
        self.assertTrue(request["parameters"]["speaker_diarization_enabled"])
        self.assertTrue(request["parameters"]["keep_dialect"])

    def test_finalizer_keeps_raw_provider_result_and_parallel_comparison(self) -> None:
        (self.local / "one.strict.audit.json").write_text(json.dumps({
            "final_text": "[疑似]最高院", "primary_text": "最高院",
            "secondary_text": "省高院", "needs_review": True}, ensure_ascii=False), encoding="utf-8")
        intent = self.intent()
        ready = prepare_cloud_request(intent, self.root, CONFIG)
        request = json.loads(Path(ready["request_path"]).read_text(encoding="utf-8"))
        item = request["chunks"][0]
        provider = self.root / (request["job_id"] + ".provider.json")
        provider.write_text(json.dumps({"schema": "zh_asr.dashscope_transport_result.v1",
            "job_id": request["job_id"], "status": "succeeded", "model": request["model"],
            "protocol": "http", "credential_result": "Success",
            "cloud_upload_performed": True, "source_audio_sha256": request["source_audio_sha256"],
            "chunks": [{"index": 1, "start_ms": item["start_ms"],
                "end_ms": item["end_ms"], "audio_sha256": item["audio_sha256"],
                "provider_request_id": "fixture-request",
                "raw_response": {"output": {"text": "省高院"}},
                "usage": {"total_tokens": 7}}]}, ensure_ascii=False), encoding="utf-8")
        result = finalize_cloud_result(intent, self.root, provider_path=provider,
                                       config_path=CONFIG)
        self.assertEqual("succeeded", result["status"])
        self.assertEqual("省高院", result["text"])
        self.assertEqual(7, result["chunks"][0]["usage"]["total_tokens"])
        self.assertTrue(result["disagreement"])
        self.assertTrue((self.local / "cloud.review.json").is_file())
        candidate, matched, rejected = _inspect_cloud_result_candidate(
            self.root / (request["job_id"] + ".result.json"), request["source_audio_sha256"])
        self.assertTrue(matched)
        self.assertEqual("", rejected)
        self.assertEqual(request["model"], candidate["engine"])

    def test_finalizer_fails_when_all_provider_chunks_lack_text(self) -> None:
        _wav(self.audio, frames=16000 * 12)
        for profile in ("short", "message"):
            intent = self.intent(model_profile=profile, chunk_sec=6, overlap_sec=0)
            ready = prepare_cloud_request(intent, self.root, CONFIG)
            document = json.loads(intent.read_text(encoding="utf-8"))
            provider = self.root / (ready["job_id"] + ".provider.json")
            chunks = [{**binding, "id": str(binding["index"]), "http_status": 200,
                "raw_response": {"output": {"text": " \n"} if binding["index"] == 1
                    else {"changed_text": "响应结构已变化"}},
                "raw_events": [{"header": {"event": "task-finished"}}]}
                for binding in document["chunk_bindings"]]
            for generic in (False, True):
                with self.subTest(profile=profile, generic=generic):
                    body = {"status": "succeeded", "credential_result": "Success",
                        "new_requests": 2, "cloud_upload_performed": True,
                        "model": document["model"], "protocol": document["protocol"],
                        "job_id": document["job_id"],
                        "source_audio_sha256": document["source_audio_sha256"]}
                    body.update({"schema": "passwordcenter.vendor-result.v1", "vendor": "qwen",
                        "request_sha256": document["vendor_request_sha256"], "results": chunks}
                        if generic else {"schema": "zh_asr.dashscope_transport_result.v1",
                                         "chunks": chunks})
                    provider.write_text(json.dumps(body), encoding="utf-8")
                    final = finalize_cloud_result(intent, self.root, provider_path=provider,
                                                  config_path=CONFIG)
                    self.assertEqual("failed", final["status"])
                    self.assertEqual("cloud_transcript_empty", final["error_code"])
                    sidecar = json.loads((self.local / "cloud.review.json").read_text(encoding="utf-8"))
                    self.assertEqual("failed", sidecar["status"])
                    self.assertEqual(final["error_code"], sidecar["error_code"])
                    self.assertTrue((self.root / (ready["job_id"] + ".work")).is_dir())
                    self.assertFalse((self.root / (ready["job_id"] + ".transcript.txt")).exists())

    def test_finalizer_pauses_on_provider_quota_error(self) -> None:
        intent = self.intent()
        ready = prepare_cloud_request(intent, self.root, CONFIG)
        request = json.loads(Path(ready["request_path"]).read_text(encoding="utf-8"))
        provider = self.root / (request["job_id"] + ".provider.json")
        provider.write_text(json.dumps({"schema": "zh_asr.dashscope_transport_result.v1",
            "job_id": request["job_id"], "status": "failed", "model": request["model"],
            "protocol": "http", "credential_result": "Permission-Denied",
            "cloud_upload_performed": True, "source_audio_sha256": request["source_audio_sha256"],
            "provider_error_code": "AllocationQuota.FreeTierOnly", "chunks": []}), encoding="utf-8")
        state = self.root / "auto-cloud-state.json"
        result = finalize_cloud_result(intent, self.root, provider_path=provider,
            state_path=state, config_path=CONFIG)
        self.assertEqual("free_quota_exhausted", result["pause_reason"])
        self.assertEqual("paused", auto_cloud_status(state,
            __import__("zh_asr.cloud_review", fromlist=["load_cloud_config"]).load_cloud_config(CONFIG))["status"])

    def test_finalizer_rejects_wrong_model_and_chunk_identity(self) -> None:
        intent = self.intent()
        ready = prepare_cloud_request(intent, self.root, CONFIG)
        request = json.loads(Path(ready["request_path"]).read_text(encoding="utf-8"))
        expected = request["chunks"][0]
        provider = self.root / (request["job_id"] + ".provider.json")
        body = {"schema": "zh_asr.dashscope_transport_result.v1",
            "job_id": request["job_id"], "status": "succeeded",
            "model": "wrong-model", "protocol": "http", "source_audio_sha256":
            request["source_audio_sha256"], "chunks": [{"index": 1,
            "start_ms": expected["start_ms"], "end_ms": expected["end_ms"],
            "audio_sha256": expected["audio_sha256"],
            "provider_request_id": "fake", "raw_response": {"output": {"text": "文本"}}}]}
        provider.write_text(json.dumps(body), encoding="utf-8")
        with self.assertRaises(CloudReviewError):
            finalize_cloud_result(intent, self.root, provider_path=provider, config_path=CONFIG)
        body["model"] = request["model"]
        body["chunks"][0]["audio_sha256"] = "0" * 64
        provider.write_text(json.dumps(body), encoding="utf-8")
        with self.assertRaises(CloudReviewError):
            finalize_cloud_result(intent, self.root, provider_path=provider, config_path=CONFIG)

    def test_websocket_raw_events_become_separate_local_candidate(self) -> None:
        intent = self.intent(model_profile="message")
        ready = prepare_cloud_request(intent, self.root, CONFIG)
        request = json.loads(Path(ready["request_path"]).read_text(encoding="utf-8"))
        chunk = request["chunks"][0]
        provider = self.root / (request["job_id"] + ".provider.json")
        provider.write_text(json.dumps({"schema": "zh_asr.dashscope_transport_result.v1",
            "job_id": request["job_id"], "status": "succeeded", "model": request["model"],
            "protocol": "websocket", "credential_result": "Success",
            "cloud_upload_performed": True, "source_audio_sha256": request["source_audio_sha256"],
            "chunks": [{"index": chunk["index"], "start_ms": chunk["start_ms"],
                "end_ms": chunk["end_ms"], "audio_sha256": chunk["audio_sha256"],
                "provider_request_id": "task-1", "raw_events": [{"header": {
                    "event": "result-generated"}, "payload": {"output": {"sentence": {
                    "sentence_id": 1, "sentence_end": True, "text": "第一句"}}}},
                    {"header": {"event": "task-finished"},
                     "payload": {"usage": {"total_tokens": 4}}}],
                "usage": {"total_tokens": 4}}]}, ensure_ascii=False), encoding="utf-8")
        final = finalize_cloud_result(intent, self.root, provider_path=provider, config_path=CONFIG)
        self.assertEqual("succeeded", final["status"])
        self.assertEqual("第一句", final["text"])
        self.assertEqual(4, final["chunks"][0]["usage"]["total_tokens"])

    def test_websocket_usage_prefers_final_task_and_sums_one_snapshot_per_chunk(self) -> None:
        _wav(self.audio, frames=16000 * 12)
        intent = self.intent(model_profile="message", chunk_sec=6, overlap_sec=0)
        ready = prepare_cloud_request(intent, self.root, CONFIG)
        request = json.loads(Path(ready["request_path"]).read_text(encoding="utf-8"))
        self.assertEqual(2, len(request["chunks"]))
        first_early = {"duration": 1, "input_tokens": 30,
            "output_tokens": 1, "total_tokens": 31}
        first_later = {"duration": 1, "input_tokens": 60,
            "output_tokens": 3, "total_tokens": 63}
        first_final = {"duration": 1, "input_tokens": 80,
            "output_tokens": 4, "total_tokens": 84}
        second_early = {"duration": 1, "input_tokens": 40,
            "output_tokens": 1, "total_tokens": 41}
        second_last = {"duration": 1, "input_tokens": 70,
            "output_tokens": 3, "total_tokens": 73}

        def record(binding, events):
            return {"index": binding["index"], "start_ms": binding["start_ms"],
                "end_ms": binding["end_ms"], "audio_sha256": binding["audio_sha256"],
                "provider_request_id": f"task-{binding['index']}",
                "usage": None, "raw_events": events}

        def generated(sentence, usage):
            return {"header": {"event": "result-generated"}, "payload": {
                "output": {"sentence": {"sentence_id": 1, "sentence_end": True,
                                         "text": sentence}}, "usage": usage}}

        chunks = [record(request["chunks"][0], [
            generated("第一段", first_early), generated("第一段", first_later),
            {"header": {"event": "task-finished"},
             "payload": {"usage": first_final}}]),
            record(request["chunks"][1], [
                generated("第二段", second_early), generated("第二段", second_last),
                {"header": {"event": "task-finished"}, "payload": {"usage": None}}])]
        provider = self.root / (request["job_id"] + ".provider.json")
        body = {"schema": "zh_asr.dashscope_transport_result.v1",
            "job_id": request["job_id"], "status": "succeeded", "model": request["model"],
            "protocol": "websocket", "credential_result": "Success",
            "cloud_upload_performed": True,
            "source_audio_sha256": request["source_audio_sha256"], "chunks": chunks}
        provider.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        final = finalize_cloud_result(intent, self.root, provider_path=provider, config_path=CONFIG)
        self.assertEqual(first_final, final["chunks"][0]["usage"])
        self.assertEqual(second_last, final["chunks"][1]["usage"])
        self.assertEqual({"duration": 2, "input_tokens": 150,
                          "output_tokens": 7, "total_tokens": 157}, final["usage"])
        saved = json.loads((self.root / (request["job_id"] + ".result.json")).read_text(
            encoding="utf-8"))
        self.assertEqual(final["usage"], saved["usage"])

        for event in body["chunks"][1]["raw_events"]:
            event["payload"].pop("usage", None)
        provider.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        incomplete = finalize_cloud_result(intent, self.root, provider_path=provider,
                                           config_path=CONFIG)
        self.assertIsNone(incomplete["chunks"][1]["usage"])
        self.assertIsNone(incomplete["usage"])

    def test_websocket_usage_from_task_finished_without_intermediate_usage(self) -> None:
        intent = self.intent(model_profile="message")
        ready = prepare_cloud_request(intent, self.root, CONFIG)
        request = json.loads(Path(ready["request_path"]).read_text(encoding="utf-8"))
        binding = request["chunks"][0]
        usage = {"duration": 1, "input_tokens": 42,
            "output_tokens": 2, "total_tokens": 44}
        provider = self.root / (request["job_id"] + ".provider.json")
        provider.write_text(json.dumps({"schema": "zh_asr.dashscope_transport_result.v1",
            "job_id": request["job_id"], "status": "succeeded", "model": request["model"],
            "protocol": "websocket", "credential_result": "Success",
            "cloud_upload_performed": True,
            "source_audio_sha256": request["source_audio_sha256"],
            "chunks": [{"index": binding["index"], "start_ms": binding["start_ms"],
                "end_ms": binding["end_ms"], "audio_sha256": binding["audio_sha256"],
                "usage": None, "raw_events": [
                    {"header": {"event": "result-generated"}, "payload": {
                        "output": {"sentence": {"sentence_id": 1,
                            "sentence_end": True, "text": "一句话"}}}},
                    {"header": {"event": "task-finished"},
                     "payload": {"usage": usage}}]}]}, ensure_ascii=False), encoding="utf-8")
        final = finalize_cloud_result(intent, self.root, provider_path=provider, config_path=CONFIG)
        self.assertEqual(usage, final["chunks"][0]["usage"])
        self.assertEqual(usage, final["usage"])

    def test_cli_handoff_marks_broker_failure_without_secret_or_upload(self) -> None:
        intent = self.intent()
        script = ROOT / "scripts" / "cloud-review-pipeline.py"
        prepare = subprocess.run([sys.executable, str(script), "prepare", "--root",
            str(self.root), "--intent", str(intent), "--config", str(CONFIG)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(0, prepare.returncode, prepare.stderr)
        self.assertEqual("ready", json.loads(prepare.stdout)["status"])
        final = subprocess.run([sys.executable, str(script), "finalize", "--root",
            str(self.root), "--intent", str(intent), "--config", str(CONFIG),
            "--broker-error", "secret_broker_target_failed"], cwd=ROOT,
            capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(0, final.returncode, final.stderr)
        value = json.loads(final.stdout)
        self.assertEqual("secret_broker_target_failed", value["error_code"])
        self.assertFalse(value["cloud_upload_performed"])
        self.assertEqual("failed", json.loads((self.local / "cloud.review.json").read_text(encoding="utf-8"))["status"])


if __name__ == "__main__":
    unittest.main()

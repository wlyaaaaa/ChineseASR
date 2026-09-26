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
    prepare_cloud_request, CloudReviewError)
from zh_asr.transcript_readback import _inspect_cloud_result_candidate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models.yaml"


def _wav(path: Path, *, stereo: bool = False) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2 if stereo else 1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        if stereo:
            handle.writeframes(((0).to_bytes(2, "little", signed=True) +
                                (12000).to_bytes(2, "little", signed=True)) * 1600)
        else:
            handle.writeframes(b"\x00\x00" * 1600)


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
        intent = self.intent(automatic_review=False, model_profile="legacy")
        result = prepare_cloud_request(intent, self.root, CONFIG,
            now=datetime.fromisoformat("2026-09-27T03:00:00+08:00"))
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

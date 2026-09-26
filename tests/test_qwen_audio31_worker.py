from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import uuid
import wave
from unittest.mock import patch

from zh_asr.cloud_review import auto_cloud_status, load_cloud_config
from zh_asr.transcript_readback import _inspect_cloud_result_candidate


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "qwen_audio31_broker_worker.py"


def load_worker():
    spec = importlib.util.spec_from_file_location("qwen_audio31_broker_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def wav(path: Path, *, stereo: bool = False) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2 if stereo else 1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        if stereo:
            handle.writeframes((0).to_bytes(2, "little", signed=True) +
                               (12000).to_bytes(2, "little", signed=True))
        else:
            handle.writeframes(b"\x00\x00" * 3200)


class NewWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.audio = self.root / "voice.wav"
        wav(self.audio)
        self.local = self.root / "local"
        self.local.mkdir()
        (self.local / "quality.review.json").write_text('{"needs_review":true}', encoding="utf-8")
        self.state_path = self.root / "auto-cloud-state.json"

    def request(self, **overrides) -> Path:
        data = {"schema": "chineseasr.qwen-audio31-request.v1", "job_id": str(uuid.uuid4()),
                "purpose": "quality_review", "cloud_upload_authorized": True,
                "automatic_review": True, "audio_path": str(self.audio),
                "local_out_dir": str(self.local), "hotwords": {},
                "speaker_diarization": False, "keep_dialect": False}
        data.update(overrides)
        path = self.root / (data["job_id"] + ".running31.json")
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path

    def run_worker(self, request: Path, **kwargs):
        return load_worker().process_request_file(request, api_key="fixture-secret-never-disk",
            state_path=self.state_path, **kwargs)

    def test_flash_payload_record_readback_and_key_not_persisted(self) -> None:
        calls = []
        def fake_http(url, headers, payload, timeout):
            calls.append((url, headers, payload))
            return {"output": {"text": "给省高院写信。", "sentences": [
                {"speaker_id": 0, "text": "给省高院写信。"}]},
                "usage": {"input_tokens": 77, "output_tokens": 9, "total_tokens": 86},
                "request_id": "provider-test-1"}
        result = self.run_worker(self.request(model_profile="short", hotwords={"省高院": 2},
                                                speaker_diarization=True), http_transport=fake_http)
        self.assertEqual("succeeded", result["status"], result)
        self.assertEqual(86, result["chunks"][0]["usage"]["total_tokens"])
        self.assertEqual("qwen-audio-3.1-asr-flash", calls[0][2]["model"])
        self.assertEqual({"省高院": 2}, calls[0][2]["parameters"]["vocabulary"])
        self.assertTrue(calls[0][2]["parameters"]["speaker_diarization_enabled"])
        self.assertNotIn("fixture-secret-never-disk", json.dumps(result, ensure_ascii=False))
        self.assertNotIn("fixture-secret-never-disk", "".join(
            item.read_text(encoding="utf-8", errors="ignore")
            for item in self.root.rglob("*.json")))
        retained = self.root / "retained.result.json"
        retained.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        candidate, matched, rejected = _inspect_cloud_result_candidate(
            retained, hashlib.sha256(self.audio.read_bytes()).hexdigest())
        self.assertTrue(matched)
        self.assertEqual("", rejected)
        self.assertEqual("qwen-audio-3.1-asr-flash", candidate["engine"])

    def test_message_route_records_usage_without_a_quota_gate(self) -> None:
        calls = []
        def fake_ws(model, key, audio, payload):
            calls.append((model, audio, payload))
            return {"text": "田家庵", "provider_request_id": "task-1",
                    "usage": None, "sentences": []}
        result = self.run_worker(self.request(hotwords={"田家庵": 3}),
                                 websocket_transport=fake_ws)
        self.assertEqual("succeeded", result["status"], result)
        self.assertEqual("qwen-audio-3.1-asr-flash-message", result["model"])
        self.assertIsNone(result["chunks"][0]["usage"])
        self.assertEqual(1, len(calls))

    def test_message_websocket_follows_finish_task_event_shape(self) -> None:
        worker = load_worker()
        model = load_cloud_config(ROOT / "configs" / "models.yaml")["models"]["message"]
        class Socket:
            def __init__(self):
                self.messages = []
                self.started = False
                self.finished = False
                self.generated = False
                self.task_id = ""
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def send(self, value):
                if isinstance(value, str):
                    parsed = json.loads(value)
                    self.messages.append(parsed)
                    self.task_id = parsed["header"]["task_id"]
                    if parsed["header"]["action"] == "finish-task":
                        self.finished = True
            def recv(self, timeout):
                if not self.started:
                    self.started = True
                    return json.dumps({"header": {"task_id": self.task_id,
                        "event": "task-started"}, "payload": {}})
                if not self.finished:
                    raise TimeoutError()
                if not self.generated:
                    self.generated = True
                    return json.dumps({"header": {"task_id": self.task_id,
                        "event": "result-generated"}, "payload": {"output": {"sentence": {
                        "sentence_id": 1, "sentence_end": True, "text": "测试"}}}})
                return json.dumps({"header": {"task_id": self.task_id,
                    "event": "task-finished"}, "payload": {"usage": {"total_tokens": 4}}})
        socket = Socket()
        with (patch("websockets.sync.client.connect", return_value=socket),
              patch.object(worker.time, "sleep", return_value=None)):
            answer = worker._websocket_call(model, "fixture-secret", self.audio,
                {"hotwords": {}, "keep_dialect": False})
        self.assertEqual("测试", answer["text"])
        self.assertEqual({"input": {}}, socket.messages[-1]["payload"])

    def test_permanent_provider_error_stops_later_automatic_calls(self) -> None:
        worker = load_worker()
        def no_credit(*args):
            raise worker.legacy.CloudApiError("AllocationQuota.FreeTierOnly",
                                              credential_result="Permission-Denied")
        first = self.run_worker(self.request(), http_transport=no_credit)
        self.assertEqual("failed", first["status"])
        self.assertEqual("free_quota_exhausted", first["auto_pause_reason"])
        self.assertEqual("paused", auto_cloud_status(self.state_path,
                         load_cloud_config(ROOT / "configs" / "models.yaml"))["status"])
        second = self.run_worker(self.request(),
            http_transport=lambda *args: self.fail("paused cloud must not upload"))
        self.assertEqual("blocked", second["status"])
        self.assertEqual("auto_cloud_paused", second["error_code"])
        self.assertFalse(second["cloud_upload_performed"])

    def test_network_error_is_only_a_single_job_failure(self) -> None:
        worker = load_worker()
        def network_down(*args):
            raise worker.legacy.CloudApiError("network_failure",
                                              credential_result="Network-Failure")
        result = self.run_worker(self.request(), http_transport=network_down)
        self.assertEqual("failed", result["status"])
        self.assertFalse(self.state_path.exists())

    def test_explicit_legacy_is_available_without_local_credit_count(self) -> None:
        request = self.request(automatic_review=False, model_profile="legacy")
        result = self.run_worker(request, http_transport=lambda *args: {
            "output": {"text": "测试"}, "request_id": "legacy-1",
            "usage": {"duration": 1}})
        self.assertEqual("succeeded", result["status"], result)
        self.assertEqual("qwen-audio-3.0-asr-flash", result["model"])
        self.assertEqual(1, result["chunks"][0]["usage"]["duration"])

    def test_selected_channel_matches_local_evidence_source(self) -> None:
        audio = self.root / "two-channels.wav"
        wav(audio, stereo=True)
        samples = []
        def fake_http(url, headers, payload, timeout):
            data_uri = payload["input"]["messages"][-1]["content"][0]["input_audio"]["data"]
            with wave.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1])), "rb") as item:
                self.assertEqual(1, item.getnchannels())
                samples.append(int.from_bytes(item.readframes(1)[:2], "little", signed=True))
            return {"output": {"text": "右声道"}, "request_id": "channel-1"}
        result = self.run_worker(self.request(audio_path=str(audio), channel_index=1),
                                 http_transport=fake_http)
        self.assertEqual("succeeded", result["status"], result)
        self.assertEqual(2, result["source_channels"])
        self.assertEqual(1, result["selected_channel"])
        self.assertEqual([12000], samples)

    def test_important_mark_suffices_as_automatic_difficulty(self) -> None:
        (self.local / "quality.review.json").unlink()
        request = self.request(purpose="important_evidence", importance="important")
        result = self.run_worker(request, http_transport=lambda *args: {
            "output": {"text": "重要录音"}, "request_id": "important-1"})
        self.assertEqual("succeeded", result["status"], result)
        self.assertIn("important_recording", result["review_reasons"])


if __name__ == "__main__":
    unittest.main()

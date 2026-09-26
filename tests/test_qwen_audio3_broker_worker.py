from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid
import wave


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "scripts" / "qwen_audio3_broker_worker.py"


def _load_worker():
    spec = importlib.util.spec_from_file_location("qwen_audio3_broker_worker", WORKER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_wav(path: Path, *, duration_sec: float = 0.1) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * max(1, round(duration_sec * 16000)))


def _write_request(root: Path, *, model: str = "qwen-audio-3.1-asr-flash",
                   protocol: str = "http", audio: Path | None = None,
                   chunks: int = 1, deadline: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    chosen = []
    for index in range(1, chunks + 1):
        chunk = audio or root / f"chunk-{index}.wav"
        if audio is None:
            _write_wav(chunk)
        chosen.append({"index": index, "start_ms": (index - 1) * 100,
            "end_ms": index * 100, "path": str(chunk.resolve()),
            "audio_sha256": hashlib.sha256(chunk.read_bytes()).hexdigest()})
    request = {"schema": "zh_asr.dashscope_transport_request.v1",
        "job_id": job_id, "purpose": "quality_review", "important_only": False,
        "cloud_upload_authorized": True, "model": model, "protocol": protocol,
        "parameters": {"format": "wav" if protocol == "http" else "pcm",
                       "sample_rate": "16000" if protocol == "http" else 16000,
                       "arbitrary_json": {"term": "技术词"}},
        "messages_prefix": [], "input": {},
        "ws_task": {"task_group": "audio", "task": "asr", "function": "recognition"},
        "chunks": chosen, "source_audio_path": str(root.parent / "original.wav"),
        "source_audio_sha256": "a" * 64, "source_audio_bytes": 3200,
        "send_before_utc": deadline}
    path = root / (job_id + ".running.json")
    path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    return path


class DashScopeWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "cloud-jobs"
        self.root.mkdir()
        self.worker = _load_worker()

    def test_pinned_worker_has_no_mutable_project_imports_or_model_config(self) -> None:
        source = WORKER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                     for alias in node.names]
        self.assertFalse(any(name and (name.startswith("zh_asr") or
                         name.startswith("qwen_audio")) for name in imported))
        self.assertNotIn("models.yaml", source)

    def test_model_name_has_format_only_no_family_whitelist(self) -> None:
        for model in ("qwen-audio-3.1-asr-flash", "qwen3-omni", "glm_5.3", "deepseek-v3.1"):
            with self.subTest(model=model):
                path = _write_request(self.root, model=model)
                result = self.worker.process_request_file(path, api_key="fixture",
                    request_root=self.root, http_transport=lambda payload, key: {
                        "http_status": 200, "raw_response": {"output": {"text": "测试"}},
                        "usage": {}, "provider_request_id": "fixture"})
                self.assertEqual("succeeded", result["status"], result)
        for model in ("", "bad/model", "https://example.com", "x" * 129, "中文模型"):
            with self.subTest(model=model):
                path = _write_request(self.root, model=model)
                result = self.worker.process_request_file(path, api_key="fixture",
                    request_root=self.root, http_transport=lambda *_: self.fail("invalid model uploaded"))
                self.assertEqual("model_name_invalid", result["error_code"])

    def test_transport_does_not_interpret_asr_business_purpose(self) -> None:
        path = _write_request(self.root)
        request = json.loads(path.read_text(encoding="utf-8"))
        request.pop("purpose")
        request.pop("important_only")
        path.write_text(json.dumps(request), encoding="utf-8")
        result = self.worker.process_request_file(path, api_key="fixture",
            request_root=self.root, http_transport=lambda *_: {
                "http_status": 200, "raw_response": {"output": {"text": "测试"}},
                "usage": {}, "provider_request_id": "fixture"})
        self.assertEqual("succeeded", result["status"])
        self.assertNotIn("purpose", result)

    def test_request_and_audio_must_stay_inside_request_root(self) -> None:
        outside = Path(self.tmp.name) / "outside.wav"
        _write_wav(outside)
        path = _write_request(self.root, audio=outside)
        result = self.worker.process_request_file(path, api_key="fixture",
            request_root=self.root, http_transport=lambda *_: self.fail("outside audio uploaded"))
        self.assertEqual("request_root_escape", result["error_code"])
        self.assertFalse(result["cloud_upload_performed"])
        outside_request = Path(self.tmp.name) / "outside.running.json"
        outside_request.write_bytes(path.read_bytes())
        blocked = self.worker.process_request_file(outside_request, api_key="fixture",
            request_root=self.root, http_transport=lambda *_: self.fail("outside request uploaded"))
        self.assertEqual("request_root_escape", blocked["error_code"])

    def test_text_context_cannot_sneak_unbound_audio_or_url_into_request(self) -> None:
        http = _write_request(self.root)
        payload = json.loads(http.read_text(encoding="utf-8"))
        payload["messages_prefix"] = [{"role": "user", "content": [{
            "type": "input_audio", "input_audio": {"data": "https://example.com/outside.wav"}}]}]
        http.write_text(json.dumps(payload), encoding="utf-8")
        result = self.worker.process_request_file(http, api_key="fixture",
            request_root=self.root, http_transport=lambda *_: self.fail("unbound audio sent"))
        self.assertEqual("input_invalid", result["error_code"])
        websocket = _write_request(self.root, protocol="websocket")
        payload = json.loads(websocket.read_text(encoding="utf-8"))
        payload["input"] = {"file_urls": ["https://example.com/outside.wav"]}
        websocket.write_text(json.dumps(payload), encoding="utf-8")
        blocked = self.worker.process_request_file(websocket, api_key="fixture",
            request_root=self.root,
            websocket_transport=lambda *_: self.fail("unbound audio sent"))
        self.assertEqual("external_audio_input_forbidden", blocked["error_code"])

    def test_http_uses_fixed_provider_address_and_no_redirect_handler(self) -> None:
        worker = self.worker
        opened = []
        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, _limit): return b'{"output":{"text":"ok"},"request_id":"r"}'
        class Opener:
            def open(self, request, timeout):
                opened.append((request.full_url, request.get_header("Authorization")))
                return Response()
        handlers = []
        def factory(*items):
            handlers.extend(items)
            return Opener()
        with patch.object(worker, "build_opener", side_effect=factory):
            worker._post_http({"model": "model", "input": {}}, "fixture-secret")
        self.assertEqual([(worker.HTTP_ENDPOINT, "Bearer fixture-secret")], opened)
        self.assertTrue(any(isinstance(item, worker._NoRedirect) for item in handlers))

    def test_arbitrary_json_parameters_and_secret_never_persist(self) -> None:
        path = _write_request(self.root)
        payloads = []
        def fake(payload, key):
            payloads.append(payload)
            return {"http_status": 200,
                "raw_response": {"output": {"text": "测试"}, "echo": "fixture-secret-never-disk",
                                 "fixture-secret-never-disk": "provider-key"},
                "usage": {"total_tokens": 10}, "provider_request_id": "r1"}
        result = self.worker.process_request_file(path, api_key="fixture-secret-never-disk",
            request_root=self.root, http_transport=fake)
        self.assertEqual("succeeded", result["status"])
        self.assertEqual({"term": "技术词"}, payloads[0]["parameters"]["arbitrary_json"])
        self.assertTrue(payloads[0]["input"]["messages"][-1]["content"][0]["input_audio"]["data"].startswith("data:audio/wav;base64,"))
        self.assertEqual("[secret-removed]", result["chunks"][0]["raw_response"]["echo"])
        self.assertEqual("provider-key", result["chunks"][0]["raw_response"]["[secret-removed]"])
        self.assertNotIn("fixture-secret-never-disk", json.dumps(result, ensure_ascii=False))
        self.assertNotIn("fixture-secret-never-disk", "".join(
            file.read_text(encoding="utf-8", errors="ignore")
            for file in self.root.rglob("*.json")))

    def test_provider_error_cannot_echo_key_into_receipts_or_checkpoints(self) -> None:
        path = _write_request(self.root)
        key = "fixture-secret-never-disk"
        def rejected(_payload, _key):
            raise self.worker.ProviderError("error-" + key, http_status=400,
                message="provider echoed " + key, credential_result="Scope-Error")
        result = self.worker.process_request_file(path, api_key=key,
            request_root=self.root, http_transport=rejected)
        self.assertEqual("failed", result["status"])
        self.assertNotIn(key, json.dumps(result, ensure_ascii=False))
        for file in self.root.rglob("*.json"):
            self.assertNotIn(key, file.read_text(encoding="utf-8", errors="ignore"))

    def test_protocol_is_explicit_not_guessed_from_model_name(self) -> None:
        path = _write_request(self.root, model="qwen-audio-3.1-asr-flash", protocol="websocket")
        calls = []
        result = self.worker.process_request_file(path, api_key="fixture",
            request_root=self.root,
            http_transport=lambda *_: self.fail("http not selected"),
            websocket_transport=lambda model, chunk, request, key, deadline: (
                calls.append((model, request["parameters"]["arbitrary_json"])) or
                {"http_status": 101, "raw_events": [], "usage": {}, "provider_request_id": "task"}))
        self.assertEqual("succeeded", result["status"])
        self.assertEqual(1, len(calls))
        bad = _write_request(self.root, protocol="filetrans")
        blocked = self.worker.process_request_file(bad, api_key="fixture",
            request_root=self.root, http_transport=lambda *_: self.fail("unsupported uploaded"))
        self.assertEqual("protocol_unsupported", blocked["error_code"])

    def test_websocket_uses_fixed_address_and_official_event_shape(self) -> None:
        worker = self.worker
        chunk = self.root / "ws.wav"
        _write_wav(chunk)
        class Socket:
            def __init__(self):
                self.sent = []
                self.started = False
                self.finished = False
                self.generated = False
                self.task_id = ""
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def send(self, value):
                if isinstance(value, str):
                    item = json.loads(value)
                    self.sent.append(item)
                    self.task_id = item["header"]["task_id"]
                    if item["header"]["action"] == "finish-task":
                        self.finished = True
            def recv(self, timeout):
                if not self.started:
                    self.started = True
                    return json.dumps({"header": {"event": "task-started",
                        "task_id": self.task_id}, "payload": {}})
                if not self.finished:
                    raise TimeoutError()
                if not self.generated:
                    self.generated = True
                    return json.dumps({"header": {"event": "result-generated",
                        "task_id": self.task_id}, "payload": {"output": {"sentence": {
                        "sentence_id": 1, "sentence_end": True, "text": "测试"}}}})
                return json.dumps({"header": {"event": "task-finished",
                    "task_id": self.task_id}, "payload": {"usage": {"total_tokens": 2}}})
        socket = Socket()
        addresses = []
        def fake_connect(uri, **kwargs):
            addresses.append((uri, kwargs["proxy"], kwargs["additional_headers"]["Authorization"]))
            return socket
        request = {"ws_task": {"task_group": "audio", "task": "asr", "function": "recognition"},
            "parameters": {"format": "pcm", "sample_rate": 16000, "arbitrary": {"x": 1}},
            "input": {}}
        with (patch("websockets.sync.client.connect", side_effect=fake_connect),
              patch.object(worker.time, "sleep", return_value=None)):
            response = worker._post_websocket("model_without_message_suffix", chunk,
                request, "fixture-key", None)
        self.assertEqual([(worker.WEBSOCKET_ENDPOINT, None, "Bearer fixture-key")], addresses)
        self.assertEqual("asr", socket.sent[0]["payload"]["task"])
        self.assertEqual({"x": 1}, socket.sent[0]["payload"]["parameters"]["arbitrary"])
        self.assertEqual({"input": {}}, socket.sent[-1]["payload"])
        self.assertEqual(2, response["usage"]["total_tokens"])

    def test_send_deadline_blocks_before_transport(self) -> None:
        path = _write_request(self.root, deadline="2020-01-01T00:00:00+08:00")
        result = self.worker.process_request_file(path, api_key="fixture",
            request_root=self.root, http_transport=lambda *_: self.fail("expired uploaded"))
        self.assertEqual("request_send_deadline_passed", result["error_code"])
        self.assertFalse(result["cloud_upload_performed"])

    def test_claim_requires_one_pending(self) -> None:
        first = self.root / "first.pending.json"
        second = self.root / "second.pending.json"
        first.write_text("{}", encoding="utf-8")
        second.write_text("{}", encoding="utf-8")
        with self.assertRaises(self.worker.WorkerPolicyError):
            self.worker.claim_single_pending_request(self.root)
        second.unlink()
        self.assertEqual("first.running.json",
            self.worker.claim_single_pending_request(self.root).name)


if __name__ == "__main__":
    unittest.main()

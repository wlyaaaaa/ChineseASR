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


class PreparedVendorRequestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_compatibility_launcher_has_no_key_or_business_import(self):
        source = WORKER_PATH.read_text(encoding="utf-8")
        imported = [node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)]
        self.assertFalse(any(name and name.startswith("zh_asr") for name in imported))
        self.assertNotIn("DASHSCOPE_API_KEY", source)
        self.assertNotIn("glob(", source)
        worker = _load_worker()
        with patch.object(worker.Path, "is_file", return_value=True), patch.object(worker.subprocess, "run") as run:
            run.return_value.returncode = 0
            self.assertEqual(0, worker.main(["--request-path", str(self.root / "selected.json")]))
        self.assertIn(str(self.root / "selected.json"), run.call_args.args[0])
        self.assertIn("qwen", run.call_args.args[0])

    def test_http_body_is_prepared_before_broker_and_model_not_whitelisted(self):
        from zh_asr.cloud_review import prepare_vendor_operations
        for model in ("new/model:中文", "future+name"):
            request = json.loads(_write_request(self.root, model=model).read_text("utf-8"))
            operation = prepare_vendor_operations(request, self.root)[0]
            self.assertEqual("http", operation["protocol"])
            body = json.loads(Path(operation["body_file"]["path"]).read_text("utf-8"))
            self.assertEqual(model, body["model"])
            self.assertTrue(body["input"]["messages"][-1]["content"][0]["input_audio"]["data"].startswith("data:audio/wav;base64,"))
            self.assertNotIn("api_key", json.dumps(body))

    def test_websocket_frames_are_business_prepared_and_repeatable(self):
        from zh_asr.cloud_review import prepare_vendor_operations
        request = json.loads(_write_request(self.root, protocol="websocket").read_text("utf-8"))
        first = prepare_vendor_operations(request, self.root)
        second = prepare_vendor_operations(request, self.root)
        self.assertEqual(first, second)
        frames = first[0]["frames"]
        self.assertEqual("asr", frames[0]["json"]["payload"]["task"])
        self.assertEqual("task-started", frames[1]["receive_until"]["equals"])
        self.assertEqual("task-finished", frames[-1]["receive_until"]["equals"])
        self.assertFalse(Path(frames[2]["binary_file"]["path"]).read_bytes().startswith(b"RIFF"))

    def test_changed_audio_is_rejected_before_building_request(self):
        from zh_asr.cloud_review import prepare_vendor_operations, CloudReviewError
        request = json.loads(_write_request(self.root).read_text("utf-8"))
        Path(request["chunks"][0]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(CloudReviewError, "chunk_hash_mismatch"):
            prepare_vendor_operations(request, self.root)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import wave


ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "scripts" / "qwen_audio3_broker_worker.py"


def _load_worker():
    spec = importlib.util.spec_from_file_location("qwen_audio3_broker_worker", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_wav(path: Path, *, duration_sec: float = 0.1) -> None:
    frame_count = max(1, int(16_000 * duration_sec))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * frame_count)


def _write_request(
    path: Path,
    audio_path: Path,
    *,
    importance: str = "important",
    cloud_upload_authorized: bool = True,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "chineseasr.qwen-audio3-important-request.v1",
                "job_id": "00000000-0000-4000-8000-000000000001",
                "importance": importance,
                "cloud_upload_authorized": cloud_upload_authorized,
                "audio_path": str(audio_path.resolve()),
                "created_utc": "2026-08-01T00:00:00Z",
                "chunk_sec": 180,
                "overlap_sec": 1,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class QwenAudio3BrokerWorkerTests(unittest.TestCase):
    def test_rejects_nonimportant_audio_before_reading_or_uploading(self) -> None:
        worker = _load_worker()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request_path = root / "job.running.json"
            missing_audio = root / "must-not-be-read.wav"
            _write_request(request_path, missing_audio, importance="ordinary")
            calls: list[object] = []

            def transport(*args, **kwargs):
                calls.append((args, kwargs))
                raise AssertionError("transport must not be called")

            result = worker.process_request_file(
                request_path,
                api_key="test-key",
                transport=transport,
            )

            self.assertEqual("blocked", result["status"])
            self.assertEqual("importance_required", result["error_code"])
            self.assertFalse(result["cloud_upload_performed"])
            self.assertEqual([], calls)

    def test_rejects_missing_explicit_cloud_authorization_before_upload(self) -> None:
        worker = _load_worker()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "short.wav"
            _write_wav(audio)
            request_path = root / "job.running.json"
            _write_request(request_path, audio, cloud_upload_authorized=False)
            calls: list[object] = []

            def transport(*args, **kwargs):
                calls.append((args, kwargs))
                raise AssertionError("transport must not be called")

            result = worker.process_request_file(
                request_path,
                api_key="test-key",
                transport=transport,
            )

            self.assertEqual("blocked", result["status"])
            self.assertEqual(
                "cloud_upload_authorization_required", result["error_code"]
            )
            self.assertFalse(result["cloud_upload_performed"])
            self.assertEqual([], calls)

    def test_posts_official_payload_and_never_persists_key(self) -> None:
        worker = _load_worker()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "short.wav"
            _write_wav(audio)
            request_path = root / "job.running.json"
            _write_request(request_path, audio)
            calls: list[dict[str, object]] = []

            def transport(url, headers, payload, timeout):
                calls.append(
                    {
                        "url": url,
                        "headers": headers,
                        "payload": payload,
                        "timeout": timeout,
                    }
                )
                return {
                    "output": {
                        "output": {"sentence": {"text": "这是专业录音连通测试。"}},
                        "text": "这是专业录音连通测试。",
                    },
                    "request_id": "request-test-1",
                }

            result = worker.process_request_file(
                request_path,
                api_key="test-key-must-not-persist",
                transport=transport,
            )

            self.assertEqual("succeeded", result["status"])
            self.assertEqual("这是专业录音连通测试。", result["text"])
            self.assertEqual("Success", result["credential_result"])
            self.assertTrue(result["cloud_upload_performed"])
            self.assertEqual(1, len(calls))
            call = calls[0]
            self.assertEqual(worker.DEFAULT_ENDPOINT, call["url"])
            self.assertEqual(
                "Bearer test-key-must-not-persist",
                call["headers"]["Authorization"],
            )
            payload = call["payload"]
            self.assertEqual("qwen-audio-3.0-asr-flash", payload["model"])
            content = payload["input"]["messages"][0]["content"][0]
            self.assertEqual("input_audio", content["type"])
            self.assertTrue(
                content["input_audio"]["data"].startswith(
                    "data:audio/wav;base64,"
                )
            )
            self.assertEqual(
                {"format": "wav", "sample_rate": "16000"},
                payload["parameters"],
            )
            self.assertNotIn(
                "test-key-must-not-persist",
                json.dumps(result, ensure_ascii=False),
            )

    def test_long_audio_is_split_locally_and_merged_in_order(self) -> None:
        worker = _load_worker()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "long.wav"
            _write_wav(audio, duration_sec=2.2)
            request_path = root / "job.running.json"
            _write_request(request_path, audio)
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["chunk_sec"] = 1
            request["overlap_sec"] = 0
            request_path.write_text(json.dumps(request), encoding="utf-8")
            responses = iter(("第一段。", "第二段。", "第三段。"))

            def transport(url, headers, payload, timeout):
                text = next(responses)
                return {
                    "output": {"text": text},
                    "request_id": f"request-{text}",
                }

            result = worker.process_request_file(
                request_path,
                api_key="test-key",
                transport=transport,
            )

            self.assertEqual("succeeded", result["status"])
            self.assertEqual("第一段。\n第二段。\n第三段。", result["text"])
            self.assertEqual(3, len(result["chunks"]))
            self.assertEqual([1, 2, 3], [item["index"] for item in result["chunks"]])

    def test_worker_claims_exactly_one_pending_request(self) -> None:
        worker = _load_worker()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "short.wav"
            _write_wav(audio)
            first = root / "first.pending.json"
            second = root / "second.pending.json"
            _write_request(first, audio)
            _write_request(second, audio)

            with self.assertRaises(worker.CloudPolicyError) as captured:
                worker.claim_single_pending_request(root)
            self.assertEqual("pending_request_ambiguous", captured.exception.code)

            second.unlink()
            claimed = worker.claim_single_pending_request(root)
            self.assertEqual("first.running.json", claimed.name)
            self.assertTrue(claimed.exists())
            self.assertFalse(first.exists())


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from test_qwen_audio3_broker_worker import _load_worker, _write_request, _write_wav


class CloudResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "cloud-jobs"
        self.root.mkdir()
        self.worker = _load_worker()
        self.request = _write_request(self.root, chunks=3)

    def _response(self, text):
        return {"http_status": 200, "raw_response": {"output": {"text": text}},
                "provider_request_id": "fixture", "usage": {"total_tokens": 3}}

    def test_successful_chunks_survive_failure_and_unknown_needs_explicit_retry(self):
        transport = Mock(side_effect=[self._response("第一段"),
                                      self.worker.ProviderError("network_failure",
                                          credential_result="Network-Failure")])
        failed = self.worker.process_request_file(self.request, api_key="fixture",
            request_root=self.root, http_transport=transport)
        self.assertEqual("failed", failed["status"])
        self.assertEqual(1, len(failed["chunks"]))
        no_transport = Mock(side_effect=AssertionError("must not resend uncertain call"))
        blocked = self.worker.process_request_file(self.request, api_key="fixture",
            request_root=self.root, http_transport=no_transport)
        self.assertEqual("cloud_chunk_outcome_unknown_explicit_retry_required", blocked["error_code"])
        self.assertEqual(1, blocked["reused_chunks"])
        no_transport.assert_not_called()
        payload = json.loads(self.request.read_text(encoding="utf-8"))
        payload["retry_uncertain_chunks"] = True
        self.request.write_text(json.dumps(payload), encoding="utf-8")
        resumed = self.worker.process_request_file(self.request, api_key="fixture",
            request_root=self.root, http_transport=Mock(return_value=self._response("后续段")))
        self.assertEqual("succeeded", resumed["status"])
        self.assertEqual(1, resumed["reused_chunks"])
        self.assertEqual(2, resumed["new_requests"])
        self.assertEqual(3, len(resumed["chunks"]))

    def test_modified_checkpoint_is_not_reused_or_reuploaded(self):
        self.worker.process_request_file(self.request, api_key="fixture",
            request_root=self.root, http_transport=Mock(return_value=self._response("测试")))
        checkpoint = next(self.root.glob("*.work/provider-chunk-000001.json"))
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        data["result"]["raw_response"]["output"]["text"] = "改写"
        checkpoint.write_text(json.dumps(data), encoding="utf-8")
        no_transport = Mock(side_effect=AssertionError("must not resend tampered checkpoint"))
        result = self.worker.process_request_file(self.request, api_key="fixture",
            request_root=self.root, http_transport=no_transport)
        self.assertEqual("cloud_checkpoint_invalid", result["error_code"])
        no_transport.assert_not_called()

    def test_changed_audio_fails_binding_before_transport(self):
        payload = json.loads(self.request.read_text(encoding="utf-8"))
        chunk = Path(payload["chunks"][0]["path"])
        changed = bytearray(chunk.read_bytes())
        changed[44] = 1
        chunk.write_bytes(changed)
        no_transport = Mock(side_effect=AssertionError("changed audio uploaded"))
        result = self.worker.process_request_file(self.request, api_key="fixture",
            request_root=self.root, http_transport=no_transport)
        self.assertEqual("chunk_hash_mismatch", result["error_code"])
        no_transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()

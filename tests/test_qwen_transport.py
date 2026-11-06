import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from urllib.error import HTTPError, URLError

from zh_asr import cloud_file_transport as chunks, qwen_transport as direct


class Copy:
    def __init__(self, candidate="new-fake-key", available=True):
        self.key, self.candidate, self.available = "old-fake-key", candidate, available
        self.ciphertext, self.syncs = b"old ciphertext", 0

    def load_key(self, vendor): return self.key

    def sync_key(self, vendor, *, validate):
        self.syncs += 1
        if not self.available: return dict(status="failed", reason="native_delivery_failed")
        if not validate(self.candidate): return dict(status="failed", reason="candidate_validation_failed")
        self.key, self.ciphertext = self.candidate, b"new ciphertext"
        return dict(status="ok")


class DirectTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.body = self.root / "body.json"
        self.body.write_text('{"model":"qwen-fake-asr"}', encoding="utf-8")
        self.operation = dict(id="1", protocol="http", path="/api/v1/services/aigc/multimodal-generation/generation",
            body_file=dict(path=str(self.body), sha256=hashlib.sha256(self.body.read_bytes()).hexdigest()))
        self.request = self.root / "part.request.json"
        self.provider = self.root / "part.provider.json"
        self.request.write_text(json.dumps(dict(vendor="qwen", operations=[self.operation],
            result_path=str(self.provider), send_before_utc="2099-01-01T00:00:00Z")), encoding="utf-8")
        self.store = Copy()

    def run_part(self, sender):
        with mock.patch.object(chunks, "key_copy", return_value=self.store), mock.patch.object(direct, "http_transport", sender):
            error = chunks._send(self.request, self.root)
        return error, json.loads(self.provider.read_text("utf-8"))

    def test_direct_success_and_legacy_checkpoint_reuse_without_key_or_network(self):
        sender = mock.Mock(return_value=dict(raw_response={"output": {"text": "合成"}}))
        self.assertEqual(self.run_part(sender)[0], "")
        original = self.provider.read_bytes()
        self.provider.unlink()
        with mock.patch.object(chunks, "key_copy", side_effect=AssertionError("reuse cannot load key")):
            self.assertEqual(chunks._send(self.request, self.root), "")
        self.assertEqual(json.loads(self.provider.read_bytes())["results"], json.loads(original)["results"])
        self.assertEqual(sender.call_count, 1)

    def test_explicit_401_retries_candidate_then_replaces_copy_after_success(self):
        def sender(url, operation, key, root):
            if key == "old-fake-key": raise direct.provider_error("InvalidApiKey", key, 401)
            self.assertEqual(self.store.ciphertext, b"old ciphertext")
            return dict(raw_response={"output": {"text": "合成 " + key}})
        error, result = self.run_part(sender)
        self.assertEqual((error, result["status"], self.store.syncs, self.store.key), ("", "succeeded", 1, "new-fake-key"))
        self.assertNotIn("fake-key", self.provider.read_text())

    def test_rejected_candidate_preserves_original_copy_and_reports_reason(self):
        sender = mock.Mock(side_effect=direct.provider_error("InvalidApiKey", "old-fake-key new-fake-key", 401))
        error, result = self.run_part(sender)
        self.assertEqual((error, self.store.syncs, sender.call_count), ("cloud_key_refresh_http_401", 1, 2))
        self.assertEqual((self.store.key, self.store.ciphertext), ("old-fake-key", b"old ciphertext"))
        self.assertNotIn("fake-key", json.dumps(result))

    def test_sync_unavailable_preserves_copy_and_exposes_structured_reason(self):
        self.store.available = False
        error, _ = self.run_part(mock.Mock(side_effect=direct.provider_error("InvalidApiKey", "", 401)))
        self.assertEqual((error, self.store.syncs, self.store.key), ("cloud_key_sync_native_delivery_failed", 1, "old-fake-key"))

    def test_candidate_success_keeps_transcript_when_copy_commit_fails(self):
        def sync(vendor, *, validate):
            self.assertTrue(validate("new-fake-key"))
            return dict(status="failed", reason="key_copy_commit_failed")
        self.store.sync_key = sync
        sender = mock.Mock(side_effect=[direct.provider_error("InvalidApiKey", "", 401), dict(raw_response={"output": {"text": "合成"}})])
        error, result = self.run_part(sender)
        self.assertEqual((error, result["status"], self.store.key), ("", "succeeded", "old-fake-key"))
        self.assertEqual(result["results"][0]["key_copy_sync"]["reason"], "key_copy_commit_failed")

    def test_candidate_unknown_outcome_preserves_copy_and_is_never_sent_again(self):
        sender = mock.Mock(side_effect=[direct.provider_error("InvalidApiKey", "", 401),
            direct.CloudSendError("network_failure", uncertain=True)])
        self.assertEqual(self.run_part(sender)[0], "cloud_key_refresh_network_failure")
        self.assertEqual(self.run_part(sender)[0], "previous_outcome_unknown")
        self.assertEqual((sender.call_count, self.store.syncs, self.store.ciphertext), (2, 1, b"old ciphertext"))

    def test_unknown_network_outcome_never_refreshes_or_repeats(self):
        sender = mock.Mock(side_effect=direct.CloudSendError("network_failure", uncertain=True))
        self.assertEqual(self.run_part(sender)[0], "network_failure")
        self.assertEqual(self.run_part(sender)[0], "previous_outcome_unknown")
        self.assertEqual((sender.call_count, self.store.syncs), (1, 0))
        marker = json.loads(next(self.root.glob("*.checkpoints/1.json")).read_text())
        self.assertEqual(marker["status"], "outcome_unknown")

    def test_interrupted_process_marker_blocks_future_send(self):
        with self.assertRaises(KeyboardInterrupt): self.run_part(mock.Mock(side_effect=KeyboardInterrupt))
        sender = mock.Mock(side_effect=AssertionError("must not resend"))
        self.assertEqual(self.run_part(sender)[0], "previous_outcome_unknown")

    def test_corrupt_or_changed_legacy_checkpoint_cannot_start_a_request(self):
        self.assertEqual(self.run_part(mock.Mock(return_value=dict(raw_response={})))[0], "")
        marker = next(self.root.glob("*.checkpoints/1.json"))
        sender = mock.Mock(side_effect=AssertionError("must not send"))
        marker.write_text("broken", encoding="utf-8")
        self.assertEqual(self.run_part(sender)[0], "checkpoint_corrupt")
        marker.write_text('{"identity":"other","status":"failed"}', encoding="utf-8")
        self.assertEqual(self.run_part(sender)[0], "checkpoint_request_changed")

    def test_deadline_and_changed_blob_stop_before_network(self):
        request = json.loads(self.request.read_text()); request["send_before_utc"] = "2000-01-01T00:00:00Z"
        self.request.write_text(json.dumps(request), encoding="utf-8")
        sender = mock.Mock(side_effect=AssertionError("expired"))
        self.assertEqual(self.run_part(sender)[0], "request_send_deadline_passed")
        sender.assert_not_called()
        self.body.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(direct.CloudSendError, "request_file_changed"):
            direct.read_blob(self.operation["body_file"], self.root)

    def test_http_bearer_success_and_known_status_errors(self):
        response = mock.MagicMock(status=200)
        response.__enter__.return_value = response
        response.read.return_value = b'{"request_id":"fixture","output":{"text":"hello"}}'
        opener = mock.Mock(); opener.open.return_value = response
        with mock.patch.object(direct, "build_opener", return_value=opener):
            result = direct.http_transport("https://fixture", self.operation, "synthetic-key", self.root)
        self.assertEqual(result["provider_request_id"], "fixture")
        self.assertEqual(opener.open.call_args.args[0].headers["Authorization"], "Bearer synthetic-key")
        for status, category in ((401, "Invalid"), (429, "Rate-Limited"), (500, "Provider-5xx")):
            opener.open.side_effect = HTTPError("https://fixture", status, "failed", {}, io.BytesIO(b'{"code":"fixture"}'))
            with mock.patch.object(direct, "build_opener", return_value=opener), self.assertRaises(direct.CloudSendError) as caught:
                direct.http_transport("https://fixture", self.operation, "synthetic-key", self.root)
            self.assertEqual(caught.exception.fields["credential_result"], category)
            self.assertFalse(caught.exception.uncertain)
        opener.open.side_effect = URLError("private exception must not escape")
        with mock.patch.object(direct, "build_opener", return_value=opener), self.assertRaises(direct.CloudSendError) as caught:
            direct.http_transport("https://fixture", self.operation, "synthetic-key", self.root)
        self.assertTrue(caught.exception.uncertain)
        self.assertNotIn("private", str(caught.exception))

    def test_ws_duplex_events_and_no_speech_projection(self):
        blob = self.root / "frames.pcm"; blob.write_bytes(b"abcd")
        event = lambda kind: json.dumps(dict(header=dict(event=kind, task_id="test")))
        socket = mock.MagicMock()
        socket.__enter__.return_value = socket
        socket.recv.side_effect = [event("task-started"), event("result-generated"), TimeoutError(), TimeoutError(), event("task-finished")]
        operation = dict(protocol="websocket", path="/api-ws/v1/inference", timeout_seconds=1,
            event_bindings=[dict(field=["header", "task_id"], value="test")],
            event_error=dict(field=["header", "event"], values=["task-failed"],
                code_field=["header", "error_code"], message_field=["header", "error_message"]),
            frames=[dict(json={"run": "test"}), dict(receive_until=dict(field=["header", "event"], equals="task-started")),
                dict(binary_file=dict(path=str(blob), sha256=hashlib.sha256(blob.read_bytes()).hexdigest()), chunk_bytes=2, drain_available=True),
                dict(json={"finish": "test"}), dict(receive_until=dict(field=["header", "event"], equals="task-finished"))])
        with mock.patch("websockets.sync.client.connect", return_value=socket):
            result = direct.websocket_transport("wss://fixture", operation, "synthetic-key", self.root)
        self.assertEqual(len(result["raw_events"]), 3)
        self.assertEqual(result["provider_request_id"], "test")
        self.assertEqual(socket.send.call_args_list[1:3], [mock.call(b"ab"), mock.call(b"cd")])
        for error in (direct.provider_error("ASR_RESPONSE_HAVE_NO_WORDS", ""), direct.CloudSendError("provider_task_timeout", uncertain=True)):
            with mock.patch.object(direct, "websocket_transport", side_effect=error):
                if error.uncertain:
                    with self.assertRaises(direct.CloudSendError): direct.transmit(operation, self.root, None, self.store, ["old-fake-key"])
                else: self.assertTrue(direct.transmit(operation, self.root, None, self.store, ["old-fake-key"])["no_speech"])
        self.assertEqual(self.store.syncs, 0)

    def test_ws_handshake_401_is_known_failure_and_wrong_task_binding_is_unknown(self):
        from websockets.exceptions import InvalidStatus
        response = mock.Mock(status_code=401, reason_phrase="Unauthorized")
        with mock.patch("websockets.sync.client.connect", side_effect=InvalidStatus(response)), self.assertRaises(direct.CloudSendError) as caught:
            direct.websocket_transport("wss://fixture", dict(frames=[]), "synthetic-key", self.root)
        self.assertTrue(caught.exception.invalid_key)
        self.assertFalse(caught.exception.uncertain)
        socket = mock.MagicMock(); socket.__enter__.return_value = socket
        socket.recv.return_value = '{"header":{"task_id":"wrong"}}'
        operation = dict(frames=[dict(receive_until=dict(field=["header", "event"], equals="task-finished"))],
            event_bindings=[dict(field=["header", "task_id"], value="test")])
        with mock.patch("websockets.sync.client.connect", return_value=socket), self.assertRaises(direct.CloudSendError) as caught:
            direct.websocket_transport("wss://fixture", operation, "synthetic-key", self.root)
        self.assertTrue(caught.exception.uncertain)


if __name__ == "__main__": unittest.main()

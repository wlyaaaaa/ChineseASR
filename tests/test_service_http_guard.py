from __future__ import annotations

import http.client
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

from zh_asr.service import TranscriptionService, create_handler, serve_api


class ServiceHttpGuardTests(unittest.TestCase):
    def test_programmatic_server_rejects_non_loopback_before_state_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state_dir = root / "api-state"
            with self.assertRaisesRegex(ValueError, "127.0.0.1"):
                serve_api("0.0.0.0", 0, state_dir, root)
            self.assertFalse(state_dir.exists())

    def test_rejects_cross_site_and_non_json_requests_without_creating_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "sample.wav"
            audio.write_bytes(b"RIFF")
            service = TranscriptionService(root=root, autostart=False)
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(service))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_port
            body = json.dumps({"audio": str(audio), "mode": "quick"})

            def request(method: str, path: str, *, headers: dict[str, str], data: str | None = None) -> int:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
                try:
                    connection.request(method, path, body=data, headers=headers)
                    response = connection.getresponse()
                    response.read()
                    return response.status
                finally:
                    connection.close()

            local_host = f"127.0.0.1:{port}"
            try:
                self.assertEqual(400, request("GET", "/jobs", headers={"Host": f"evil.example:{port}"}))
                self.assertEqual(400, request("GET", "/jobs", headers={
                    "Host": local_host, "Origin": "https://evil.example"}))
                self.assertEqual(415, request("POST", "/jobs/transcribe", headers={
                    "Host": local_host, "Content-Type": "text/plain"}, data=body))
                self.assertEqual(415, request("POST", "/jobs/transcribe", headers={
                    "Host": local_host}, data=body))
                self.assertEqual(400, request("POST", "/jobs/transcribe", headers={
                    "Host": local_host, "Origin": "https://evil.example",
                    "Content-Type": "application/json"}, data=body))
                self.assertEqual(400, request("POST", "/jobs/unknown/cancel", headers={
                    "Host": local_host, "Origin": "https://evil.example"}))
                self.assertEqual(200, request("GET", "/health", headers={"Host": local_host}))
                self.assertEqual(202, request("POST", "/jobs/transcribe", headers={
                    "Host": local_host, "Content-Type": "application/json"}, data=body))
                self.assertEqual(1, len(service.list_jobs()))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                service.stop()

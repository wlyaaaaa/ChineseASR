from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]


class _Api(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _reply(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply({"status": "ok"})

    def do_POST(self):
        count = int(self.headers["Content-Length"])
        self.server.submitted_payload = json.loads(self.rfile.read(count))
        self._reply({"deduplicated": False, "job": {
            "status": "succeeded", "evidence_status": "verified",
            "evidence_failures": [], "objective_outcome": "met",
            "audio_result_status": "met", "objective_execution_status": "succeeded",
            "job_id": "job-1", "out_dir": "E:/example/job-1", "outputs": {},
            "message": "Completed.", "cloud_review": {
                "status": "pending_ai_session", "message": "疑难录音，等 AI 会话补跑云端复核",
                "next_command": "& 'asr-professional-cloud.ps1' -AutomaticReview"}}})


class AsrSmartCloudStatusTests(unittest.TestCase):
    def test_json_exposes_pending_cloud_review_from_completed_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "中文 录音[1].wav"
            audio.write_bytes(b"fixture")
            audio.with_name("中文 录音1.wav").write_bytes(b"decoy")
            cache = Path(tmp) / "中文 缓存[1]"
            cache.mkdir()
            cache.with_name("中文 缓存1").mkdir()
            server = ThreadingHTTPServer(("127.0.0.1", 0), _Api)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive",
                    "-File", str(ROOT / "scripts" / "asr-smart.ps1"),
                    "-Audio", str(audio), "-CacheDir", str(cache), "-Port", str(server.server_port),
                    "-WaitSec", "0", "-Json"], cwd=ROOT, capture_output=True,
                    text=True, encoding="utf-8", errors="replace", timeout=30)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual({"audio": str(audio), "cache_dir": str(cache)}, {
                field: server.submitted_payload[field] for field in ("audio", "cache_dir")})
            payload = json.loads(result.stdout)
            self.assertEqual("succeeded", payload["status"])
            self.assertEqual("pending_ai_session", payload["cloud_review"]["status"])
            self.assertIn("AutomaticReview", payload["cloud_review"]["next_command"])


if __name__ == "__main__":
    unittest.main()

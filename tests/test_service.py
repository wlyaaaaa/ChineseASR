import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from pathlib import Path

from zh_asr.service import GpuProcess, JobRequest, ProcessResult, TranscriptionService, create_handler


class ServiceTests(unittest.TestCase):
    def test_submit_deduplicates_identical_queued_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            service = TranscriptionService(root=root, gpu_process_detector=lambda: [], autostart=False)
            request = JobRequest.from_payload({"audio": str(audio), "mode": "strict"}, root=root)

            first, first_deduped = service.submit(request)
            second, second_deduped = service.submit(request)

            self.assertFalse(first_deduped)
            self.assertTrue(second_deduped)
            self.assertEqual(first.job_id, second.job_id)
            self.assertEqual("queued", first.status)

    def test_submit_blocks_when_foreign_gpu_process_is_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            conflict = GpuProcess(pid=9999, process_name="ollama.exe", used_memory_mib=4096)
            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [conflict],
                current_process_ids=lambda: {1111},
                autostart=False,
            )
            request = JobRequest.from_payload({"audio": str(audio), "mode": "strict"}, root=root)

            job, deduped = service.submit(request)

            self.assertFalse(deduped)
            self.assertEqual("blocked", job.status)
            self.assertEqual("gpu_conflict", job.stage)
            self.assertEqual(1, len(job.conflicts))
            self.assertIn("ollama.exe", job.message)

    def test_submit_can_override_gpu_conflict(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            conflict = GpuProcess(pid=9999, process_name="python.exe", used_memory_mib=2048)
            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [conflict],
                current_process_ids=lambda: {1111},
                autostart=False,
            )
            request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "strict", "allow_gpu_conflicts": True},
                root=root,
            )

            job, deduped = service.submit(request)

            self.assertFalse(deduped)
            self.assertEqual("queued", job.status)
            self.assertEqual([], job.conflicts)

    def test_submit_accepts_long_strict_mode_and_builds_long_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            service = TranscriptionService(root=root, gpu_process_detector=lambda: [], autostart=False)
            request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "long-strict", "chunk_sec": 120, "overlap_sec": 2},
                root=root,
            )

            job, deduped = service.submit(request)

            self.assertFalse(deduped)
            self.assertEqual("queued", job.status)
            self.assertIn("long", job.command)
            self.assertIn("--chunk-sec", job.command)
            self.assertIn("120", job.command)
            self.assertIn("--overlap-sec", job.command)
            self.assertIn("2", job.command)

    def test_run_next_job_marks_success_and_collects_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")

            def fake_runner(job):
                job.out_dir.mkdir(parents=True, exist_ok=True)
                (job.out_dir / "sample.strict.md").write_text("正文", encoding="utf-8")
                (job.out_dir / "sample.strict.audit.md").write_text("审计", encoding="utf-8")
                return ProcessResult(returncode=0, stdout="Final: sample.strict.md", stderr="")

            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                process_runner=fake_runner,
                autostart=False,
            )
            request = JobRequest.from_payload({"audio": str(audio), "mode": "strict"}, root=root)
            job, _ = service.submit(request)

            service.run_next_job()
            refreshed = service.get_job(job.job_id)

            self.assertIsNotNone(refreshed)
            self.assertEqual("succeeded", refreshed.status)
            self.assertIn("final", refreshed.outputs)
            self.assertTrue(refreshed.outputs["final"].endswith("sample.strict.md"))

    def test_http_api_serves_health_submit_status_and_cancel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            service = TranscriptionService(root=root, gpu_process_detector=lambda: [], autostart=False)
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(service))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"
            try:
                health = _json_get(f"{base_url}/health")
                submitted = _json_post(f"{base_url}/jobs/transcribe", {"audio": str(audio), "mode": "strict"})
                status = _json_get(f"{base_url}/jobs/{submitted['job']['job_id']}")
                canceled = _json_post(f"{base_url}/jobs/{submitted['job']['job_id']}/cancel", {})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            self.assertEqual("ok", health["status"])
            self.assertEqual("queued", submitted["job"]["status"])
            self.assertFalse(submitted["deduplicated"])
            self.assertEqual(submitted["job"]["job_id"], status["job"]["job_id"])
            self.assertEqual("canceled", canceled["job"]["status"])

    def test_http_api_uses_service_default_output_root_when_payload_omits_out_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            output_root = root / "custom-api-output"
            service = TranscriptionService(
                root=root,
                default_out_root=output_root,
                gpu_process_detector=lambda: [],
                autostart=False,
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(service))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base_url = f"http://127.0.0.1:{server.server_port}"
            try:
                submitted = _json_post(f"{base_url}/jobs/transcribe", {"audio": str(audio), "mode": "strict"})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            self.assertTrue(submitted["job"]["out_dir"].startswith(str(output_root.resolve())))


def _write_audio(path: Path) -> Path:
    path.write_bytes(b"RIFF")
    return path


def _json_get(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _json_post(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()

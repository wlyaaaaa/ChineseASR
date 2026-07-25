from __future__ import annotations

import json
import tempfile
import threading
import unittest
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
from unittest.mock import patch

import zh_asr.service as service_module
from zh_asr.service import JobRequest, TranscriptionService, create_handler


EXPECTED_JOB_KEYS = {
    "job_id",
    "state",
    "stage",
    "mode",
    "model",
    "progress",
    "timing",
    "tokens",
    "throughput",
}


class ObserverProjectionTests(unittest.TestCase):
    def test_list_projection_is_whitelisted_and_hides_sensitive_job_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_wav(root / "private-source-name.wav", duration_seconds=2)
            service = TranscriptionService(root=root, gpu_process_detector=lambda: [], autostart=False)
            request = JobRequest.from_payload(
                {
                    "audio": str(audio),
                    "mode": "strict",
                    "primary_engine": "qwen3-asr-1.7b",
                    "secondary_engine": "sensevoice",
                },
                root=root,
            )
            job, _ = service.submit(request)
            job.command = ["python", "--secret-token", "broker-secret"]
            job.process_id = 424242
            job.stdout_tail = "recognised private transcript"
            job.stderr_tail = "C:\\private\\stderr.log"
            job.message = "Broker token: broker-secret"
            job.outputs = {"final": str(root / "private-transcript.md")}
            job.stage = "C:\\private\\stage"

            with _running_server(service) as base_url:
                payload = _json_get(f"{base_url}/observer/jobs")

            self.assertEqual("local-ai-observer.jobs.v1", payload["schema"])
            self.assertEqual("chinese-asr", payload["service"])
            self.assertTrue(payload["observed_utc"].endswith("Z"))
            self.assertEqual(1, len(payload["jobs"]))
            observed_job = payload["jobs"][0]
            self.assertEqual(EXPECTED_JOB_KEYS, set(observed_job))
            self.assertEqual("unknown", observed_job["stage"])
            self.assertEqual("qwen3-asr-1.7b + sensevoice", observed_job["model"])
            self.assertEqual(
                {"status": "not_applicable", "input": None, "output": None, "tps": None},
                observed_job["tokens"],
            )

            serialized = json.dumps(payload, ensure_ascii=False)
            for forbidden in (
                str(root),
                "private-source-name",
                "private transcript",
                "private-transcript",
                "broker-secret",
                "424242",
                "--secret-token",
                "stderr.log",
            ):
                self.assertNotIn(forbidden, serialized)

    def test_list_limit_is_bounded_and_http_query_is_applied(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_wav(root / "sample.wav", duration_seconds=1)
            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                autostart=False,
            )
            for _ in range(205):
                request = JobRequest.from_payload(
                    {"audio": str(audio), "mode": "quick", "force": True},
                    root=root,
                )
                service.submit(request)

            with (
                patch("zh_asr.service._wav_audio_seconds", return_value=1.0),
                patch(
                    "zh_asr.service._load_observer_model_config",
                    wraps=service_module._load_observer_model_config,
                ) as load_config,
            ):
                self.assertEqual(50, len(service.observer_jobs()["jobs"]))
                self.assertEqual(200, len(service.observer_jobs(limit=999)["jobs"]))
                self.assertEqual(1, len(service.observer_jobs(limit=0)["jobs"]))
                self.assertEqual(3, load_config.call_count)

                with _running_server(service) as base_url:
                    payload = _json_get(f"{base_url}/observer/jobs?limit=7")

            self.assertEqual(7, len(payload["jobs"]))
            self.assertEqual(4, load_config.call_count)

    def test_slow_list_and_detail_projection_never_hold_the_job_lock(self):
        for projection_kind in ("list", "detail"):
            with self.subTest(projection_kind=projection_kind):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    audio = _write_wav(root / "sample.wav", duration_seconds=1)
                    next_audio = _write_wav(root / "next.wav", duration_seconds=1)
                    service = TranscriptionService(
                        root=root,
                        gpu_process_detector=lambda: [],
                        autostart=False,
                    )
                    request = JobRequest.from_payload(
                        {"audio": str(audio), "mode": "quick"},
                        root=root,
                    )
                    job, _ = service.submit(request)
                    next_request = JobRequest.from_payload(
                        {"audio": str(next_audio), "mode": "quick"},
                        root=root,
                    )

                    projection_started = threading.Event()
                    release_projection = threading.Event()
                    cancel_done = threading.Event()
                    submit_done = threading.Event()
                    errors: list[BaseException] = []

                    def slow_audio_seconds(_path: Path) -> float:
                        projection_started.set()
                        release_projection.wait(timeout=3)
                        return 1.0

                    def project() -> None:
                        try:
                            if projection_kind == "list":
                                service.observer_jobs()
                            else:
                                service.observer_job(job.job_id)
                        except BaseException as exc:
                            errors.append(exc)

                    def cancel_job() -> None:
                        try:
                            service.cancel(job.job_id)
                        except BaseException as exc:
                            errors.append(exc)
                        finally:
                            cancel_done.set()

                    def submit_job() -> None:
                        try:
                            service.submit(next_request)
                        except BaseException as exc:
                            errors.append(exc)
                        finally:
                            submit_done.set()

                    with patch("zh_asr.service._wav_audio_seconds", side_effect=slow_audio_seconds):
                        projection_thread = threading.Thread(target=project)
                        projection_thread.start()
                        self.assertTrue(projection_started.wait(timeout=1))

                        cancel_thread = threading.Thread(target=cancel_job)
                        submit_thread = threading.Thread(target=submit_job)
                        cancel_thread.start()
                        submit_thread.start()

                        lock_acquired = service._lock.acquire(timeout=0.5)
                        if lock_acquired:
                            service._lock.release()
                        cancel_responsive = cancel_done.wait(timeout=0.5)
                        submit_responsive = submit_done.wait(timeout=0.5)

                        release_projection.set()
                        projection_thread.join(timeout=2)
                        cancel_thread.join(timeout=2)
                        submit_thread.join(timeout=2)

                    self.assertTrue(lock_acquired)
                    self.assertTrue(cancel_responsive)
                    self.assertTrue(submit_responsive)
                    self.assertFalse(projection_thread.is_alive())
                    self.assertFalse(cancel_thread.is_alive())
                    self.assertFalse(submit_thread.is_alive())
                    self.assertEqual([], errors)

    def test_detail_projects_existing_long_audio_chunk_progress_without_manifest_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_wav(root / "meeting.wav", duration_seconds=3)
            service = TranscriptionService(root=root, gpu_process_detector=lambda: [], autostart=False)
            request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "long-strict"},
                root=root,
            )
            job, _ = service.submit(request)
            job.status = "running"
            job.stage = "running_command"
            job.started_at = job.created_at
            job.updated_at = job.created_at + 0.5
            job.out_dir.mkdir(parents=True)
            (job.out_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "audio": str(audio),
                        "private_transcript": "不能泄漏的识别正文",
                        "broker_token": "broker-secret",
                        "chunks": [
                            {"chunk_id": "chunk-000001", "status": "succeeded", "audio_path": str(audio)},
                            {"chunk_id": "chunk-000002", "status": "running", "text": "隐私正文"},
                            {"chunk_id": "chunk-000003", "status": "failed", "error": "private stderr"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with _running_server(service) as base_url:
                payload = _json_get(f"{base_url}/observer/jobs/{job.job_id}")

            self.assertEqual("local-ai-observer.job.v1", payload["schema"])
            self.assertEqual("chinese-asr", payload["service"])
            observed_job = payload["job"]
            self.assertEqual(EXPECTED_JOB_KEYS, set(observed_job))
            self.assertEqual(
                {"status": "available", "completed": 2, "total": 3, "unit": "chunks"},
                observed_job["progress"],
            )
            self.assertEqual("running", observed_job["timing"]["status"])
            self.assertGreaterEqual(observed_job["timing"]["elapsed_ms"], 0)
            self.assertEqual("unavailable", observed_job["throughput"]["status"])
            self.assertEqual(3.0, observed_job["throughput"]["audio_seconds"])
            self.assertIsNone(observed_job["throughput"]["rtf"])

            serialized = json.dumps(payload, ensure_ascii=False)
            for forbidden in (str(root), "不能泄漏", "隐私正文", "broker-secret", "private stderr"):
                self.assertNotIn(forbidden, serialized)

    def test_oversized_long_audio_manifest_is_not_read_or_projected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_wav(root / "sample.wav", duration_seconds=1)
            service = TranscriptionService(root=root, gpu_process_detector=lambda: [], autostart=False)
            request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "long-strict"},
                root=root,
            )
            job, _ = service.submit(request)
            job.status = "running"
            job.stage = "running_command"
            job.out_dir.mkdir(parents=True)
            (job.out_dir / "manifest.json").write_bytes(
                b"{" + (b"x" * service_module.OBSERVER_MANIFEST_LIMIT_BYTES)
            )

            payload = service.observer_job(job.job_id)

            self.assertEqual(
                {"status": "unavailable", "completed": None, "total": None, "unit": None},
                payload["job"]["progress"],
            )

    def test_terminal_job_exposes_measured_real_time_factor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_wav(root / "sample.wav", duration_seconds=2)
            service = TranscriptionService(root=root, gpu_process_detector=lambda: [], autostart=False)
            request = JobRequest.from_payload({"audio": str(audio), "mode": "quick"}, root=root)
            job, _ = service.submit(request)
            job.status = "succeeded"
            job.stage = "finished"
            job.started_at = 100.0
            job.finished_at = 102.5
            job.updated_at = 102.5

            with _running_server(service) as base_url:
                payload = _json_get(f"{base_url}/observer/jobs/{job.job_id}")

            self.assertEqual(
                {"status": "measured", "rtf": 1.25, "audio_seconds": 2.0},
                payload["job"]["throughput"],
            )
            self.assertEqual(2500, payload["job"]["timing"]["elapsed_ms"])
            self.assertEqual("complete", payload["job"]["timing"]["status"])

    def test_unknown_detail_is_generic_and_does_not_reflect_requested_identifier(self):
        service = TranscriptionService(
            root=Path.cwd(),
            gpu_process_detector=lambda: [],
            autostart=False,
        )
        secret_identifier = "private-path-and-token"

        with _running_server(service) as base_url:
            with self.assertRaises(HTTPError) as raised:
                _json_get(f"{base_url}/observer/jobs/{secret_identifier}")
            body = raised.exception.read().decode("utf-8")
            raised.exception.close()

        self.assertEqual(404, raised.exception.code)
        self.assertNotIn(secret_identifier, body)
        self.assertEqual("job_not_found", json.loads(body)["error"])


class _running_server:
    def __init__(self, service: TranscriptionService):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(service))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, exc_type, exc, tb) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def _write_wav(path: Path, *, duration_seconds: int) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * 16_000 * duration_seconds)
    return path


def _json_get(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()

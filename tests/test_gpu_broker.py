import tempfile
import unittest
from pathlib import Path

from zh_asr.gpu_broker import GpuBrokerConflict, GpuBrokerLease
from zh_asr.service import JobRequest, ProcessResult, TranscriptionService


class RecordingTransport:
    def __init__(self, acquire_ok=True):
        self.acquire_ok = acquire_ok
        self.calls = []

    def __call__(self, action, payload):
        self.calls.append((action, dict(payload)))
        if action == "acquire":
            if not self.acquire_ok:
                return {"ok": False, "reason": "gpu_lease_active", "owner": "localocr"}
            return {"ok": True, "token": "asr-token", "owner": payload["owner"]}
        return {"ok": True}


class LeaseTests(unittest.TestCase):
    def test_context_acquires_and_releases(self):
        transport = RecordingTransport()

        with GpuBrokerLease("chineseasr", transport=transport, renew_interval_seconds=0):
            pass

        self.assertEqual([call[0] for call in transport.calls], ["acquire", "release"])

    def test_conflict_raises(self):
        with self.assertRaises(GpuBrokerConflict):
            with GpuBrokerLease(
                "chineseasr",
                transport=RecordingTransport(acquire_ok=False),
                renew_interval_seconds=0,
            ):
                self.fail("work must not start")


class ServiceLeaseTests(unittest.TestCase):
    def test_job_holds_gpu_lease_around_runner(self):
        events = []

        class Lease:
            def __enter__(self):
                events.append("lease_enter")
                return self

            def __exit__(self, *_args):
                events.append("lease_exit")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "sample.wav"
            audio.write_bytes(b"RIFF")

            def runner(_job):
                events.append("runner")
                return ProcessResult(returncode=0)

            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                process_runner=runner,
                gpu_lease_factory=lambda owner: Lease(),
                autostart=False,
            )
            request = JobRequest.from_payload({"audio": str(audio), "mode": "quick"}, root=root)
            job, _ = service.submit(request)
            service.run_next_job()

        self.assertEqual(service.get_job(job.job_id).status, "succeeded")
        self.assertEqual(events, ["lease_enter", "runner", "lease_exit"])

    def test_broker_conflict_marks_job_blocked(self):
        class ConflictLease:
            def __enter__(self):
                raise GpuBrokerConflict("active=localocr")

            def __exit__(self, *_args):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "sample.wav"
            audio.write_bytes(b"RIFF")
            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                process_runner=lambda _job: ProcessResult(returncode=0),
                gpu_lease_factory=lambda owner: ConflictLease(),
                autostart=False,
            )
            request = JobRequest.from_payload({"audio": str(audio), "mode": "quick"}, root=root)
            job, _ = service.submit(request)
            service.run_next_job()

        refreshed = service.get_job(job.job_id)
        self.assertEqual(refreshed.status, "blocked")
        self.assertEqual(refreshed.stage, "gpu_broker_conflict")

    def test_explicit_allow_gpu_conflicts_bypasses_broker(self):
        events = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "sample.wav"
            audio.write_bytes(b"RIFF")
            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                process_runner=lambda _job: events.append("runner") or ProcessResult(returncode=0),
                gpu_lease_factory=lambda owner: self.fail("broker must be bypassed"),
                autostart=False,
            )
            request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "quick", "allow_gpu_conflicts": True}, root=root
            )
            service.submit(request)
            service.run_next_job()

        self.assertEqual(events, ["runner"])


if __name__ == "__main__":
    unittest.main()

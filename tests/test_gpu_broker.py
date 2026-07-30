import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from zh_asr.gpu_broker import (
    GpuBrokerConflict,
    GpuBrokerError,
    GpuBrokerLease,
    GpuBrokerLeaseLost,
    verify_inherited_gpu_lease,
)
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
        if action == "renew":
            return {
                "ok": True,
                "token": payload["token"],
                "owner": "chineseasr",
            }
        return {"ok": True}


class LeaseTests(unittest.TestCase):
    def test_inherited_child_token_must_resolve_to_live_chineseasr_lease(self):
        transport = RecordingTransport()

        owner = verify_inherited_gpu_lease(
            "asr-token",
            transport=transport,
        )

        self.assertEqual(owner, "chineseasr")
        self.assertEqual(
            transport.calls,
            [
                (
                    "renew",
                    {
                        "token": "asr-token",
                        "ttl_seconds": 21_600,
                    },
                )
            ],
        )

    def test_inherited_child_token_rejects_non_asr_owner(self):
        def transport(action, payload):
            self.assertEqual(action, "renew")
            return {"ok": True, "token": payload["token"], "owner": "localocr"}

        with self.assertRaises(GpuBrokerError):
            verify_inherited_gpu_lease("ocr-token", transport=transport)

    def test_inherited_child_token_rejects_missing_live_lease(self):
        def transport(_action, _payload):
            return {"ok": False, "reason": "lease_not_found"}

        with self.assertRaises(GpuBrokerLeaseLost):
            verify_inherited_gpu_lease("stale-token", transport=transport)

    def test_context_acquires_and_releases(self):
        transport = RecordingTransport()

        with GpuBrokerLease("chineseasr", transport=transport, renew_interval_seconds=0):
            pass

        self.assertEqual([call[0] for call in transport.calls], ["acquire", "release"])

    def test_release_failure_does_not_mask_authoritative_body_lease_loss(self):
        class ReleaseFailureTransport(RecordingTransport):
            def __call__(self, action, payload):
                if action == "release":
                    raise OSError("release channel failed")
                return super().__call__(action, payload)

        with self.assertRaisesRegex(GpuBrokerLeaseLost, "authoritative lease loss"):
            with GpuBrokerLease(
                "chineseasr",
                transport=ReleaseFailureTransport(),
                renew_interval_seconds=0,
            ):
                raise GpuBrokerLeaseLost("authoritative lease loss")

    def test_release_failure_is_reported_when_no_prior_failure_exists(self):
        class ReleaseFailureTransport(RecordingTransport):
            def __call__(self, action, payload):
                if action == "release":
                    raise OSError("release channel failed")
                return super().__call__(action, payload)

        with self.assertRaisesRegex(OSError, "release channel failed"):
            with GpuBrokerLease(
                "chineseasr",
                transport=ReleaseFailureTransport(),
                renew_interval_seconds=0,
            ):
                pass

    def test_conflict_raises(self):
        with self.assertRaises(GpuBrokerConflict):
            with GpuBrokerLease(
                "chineseasr",
                transport=RecordingTransport(acquire_ok=False),
                renew_interval_seconds=0,
            ):
                self.fail("work must not start")

    def test_renew_failure_is_fail_closed_and_not_silently_swallowed(self):
        renewed = threading.Event()

        class RenewFailureTransport(RecordingTransport):
            def __call__(self, action, payload):
                if action == "renew":
                    renewed.set()
                    return {
                        "ok": False,
                        "reason": "lease_not_found",
                        "owner": "localocr",
                    }
                return super().__call__(action, payload)

        lease = GpuBrokerLease(
            "chineseasr",
            transport=RenewFailureTransport(),
            renew_interval_seconds=0.01,
        )
        with self.assertRaises(GpuBrokerLeaseLost):
            with lease:
                self.assertTrue(renewed.wait(timeout=1))
                deadline = time.monotonic() + 1
                while not lease.lost and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(lease.lost)

    def test_renew_failure_notifies_running_job_callback(self):
        callback_called = threading.Event()

        class RenewFailureTransport(RecordingTransport):
            def __call__(self, action, payload):
                if action == "renew":
                    return {"ok": False, "reason": "lease_expired"}
                return super().__call__(action, payload)

        lease = GpuBrokerLease(
            "chineseasr",
            transport=RenewFailureTransport(),
            renew_interval_seconds=0.01,
        )
        lease.set_on_lost(lambda _error: callback_called.set())
        with self.assertRaises(GpuBrokerLeaseLost):
            with lease:
                self.assertTrue(callback_called.wait(timeout=1))


class ServiceLeaseTests(unittest.TestCase):
    def test_default_subprocess_requires_and_passes_live_token_for_all_cuda_jobs(self):
        from zh_asr.gpu_broker import GPU_BROKER_CHILD_TOKEN_ENV

        for allow_gpu_conflicts in (False, True):
            with self.subTest(
                allow_gpu_conflicts=allow_gpu_conflicts
            ), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                audio = root / "sample.wav"
                audio.write_bytes(b"RIFF")
                captured = {}

                class Process:
                    pid = 4321
                    returncode = 0

                    def communicate(self, timeout):
                        captured["timeout"] = timeout
                        return "", ""

                    def poll(self):
                        return self.returncode

                def popen(command, **kwargs):
                    captured["command"] = command
                    captured["env"] = kwargs["env"]
                    return Process()

                service = TranscriptionService(
                    root=root,
                    gpu_process_detector=lambda: [],
                    autostart=False,
                )
                request = JobRequest.from_payload(
                    {
                        "audio": str(audio),
                        "mode": "quick",
                        "allow_gpu_conflicts": allow_gpu_conflicts,
                    },
                    root=root,
                )
                job, _ = service.submit(request)
                job.gpu_broker_token = "live-service-token"
                with patch(
                    "zh_asr.service.subprocess.Popen",
                    side_effect=popen,
                ):
                    result = service._run_subprocess(job)

                self.assertEqual(result.returncode, 0)
                self.assertEqual(
                    captured["env"][GPU_BROKER_CHILD_TOKEN_ENV],
                    "live-service-token",
                )
                self.assertNotIn(
                    "ZH_ASR_GPU_BROKER_LEASE_HELD",
                    captured["env"],
                )

    def test_subprocess_closes_race_when_lease_was_lost_before_process_registration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "sample.wav"
            audio.write_bytes(b"RIFF")

            class Process:
                pid = 4321
                returncode = -1

                def communicate(self, timeout=None):
                    return "", ""

                def poll(self):
                    return self.returncode

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
            job.gpu_broker_token = "live-service-token"
            job.gpu_broker_loss_error = "lease expired before process registration"

            with patch(
                "zh_asr.service.subprocess.Popen",
                return_value=Process(),
            ), self.assertRaises(GpuBrokerLeaseLost):
                service._run_subprocess(job)

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

    def test_lost_broker_lease_marks_evidence_unavailable(self):
        class LostLease:
            def set_on_lost(self, callback):
                self.callback = callback

            def __enter__(self):
                return self

            def raise_if_lost(self):
                error = GpuBrokerLeaseLost("lease expired while job was running")
                self.callback(error)
                raise error

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
                gpu_lease_factory=lambda owner: LostLease(),
                autostart=False,
            )
            request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "strict"},
                root=root,
            )
            job, _ = service.submit(request)
            service.run_next_job()

        refreshed = service.get_job(job.job_id)
        self.assertEqual(refreshed.status, "failed")
        self.assertEqual(refreshed.stage, "gpu_broker_lost")
        self.assertEqual(refreshed.evidence_status, "unavailable")
        self.assertIn("lease expired", refreshed.message)

    def test_legacy_allow_gpu_conflicts_does_not_bypass_machine_broker(self):
        events = []

        class Lease:
            token = "live-asr-token"

            def __enter__(self):
                events.append("lease_enter")
                return self

            def __exit__(self, *_args):
                events.append("lease_exit")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "sample.wav"
            audio.write_bytes(b"RIFF")
            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                process_runner=lambda _job: events.append("runner") or ProcessResult(returncode=0),
                gpu_lease_factory=lambda _owner: Lease(),
                autostart=False,
            )
            request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "quick", "allow_gpu_conflicts": True}, root=root
            )
            service.submit(request)
            service.run_next_job()

        self.assertEqual(events, ["lease_enter", "runner", "lease_exit"])


if __name__ == "__main__":
    unittest.main()

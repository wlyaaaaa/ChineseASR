from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from zh_asr.service import JobRequest, ProcessResult, TranscriptionService


class ServiceCloudReviewTests(unittest.TestCase):
    def _run_job(self, root: Path, *, cloud_state: dict, free_until: str) -> tuple[dict, dict]:
        audio = root / "call.wav"
        audio.write_bytes(b"RIFF fixture")

        def local_runner(job):
            job.out_dir.mkdir(parents=True, exist_ok=True)
            (job.out_dir / "quality.review.json").write_text(
                '{"needs_review":true}', encoding="utf-8")
            return ProcessResult(returncode=0)

        service = TranscriptionService(root=root, process_runner=local_runner,
            gpu_process_detector=lambda: [], autostart=False)
        service._cloud_review_marker_enabled = True
        request = JobRequest.from_payload({"audio": str(audio), "mode": "strict",
            "device": "cpu"}, root=root)
        config = {"routes": {"short": "short", "long": "short"},
            "models": {"short": {"id": "cloud-short", "free_until": free_until}}}
        with (patch("zh_asr.audio_quality.probe_duration_ms", return_value=1000),
              patch("zh_asr.cloud_review.load_cloud_config", return_value=config),
              patch("zh_asr.cloud_review.auto_cloud_status", return_value=cloud_state),
              patch("zh_asr.service.subprocess.run", side_effect=AssertionError("cloud call"))):
            job, _ = service.submit(request)
            self.assertTrue(service.run_next_job())
        self.assertEqual("succeeded", job.status, job.message)
        sidecar = json.loads((job.out_dir / "cloud.review.json").read_text(encoding="utf-8"))
        return job.to_dict(), sidecar

    def test_difficult_job_is_marked_before_terminal_response_without_cloud_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job, sidecar = self._run_job(root, cloud_state={"status": "running"},
                free_until="2099-01-01T00:00:00+08:00")
            self.assertEqual("succeeded", job["status"])
            self.assertEqual("pending_ai_session", sidecar["status"])
            self.assertEqual("pending_ai_session", job["cloud_review"]["status"])
            self.assertIn("quality_needs_review", sidecar["review_reasons"])
            self.assertIn("-AutomaticReview", sidecar["next_command"])
            self.assertIn("-LocalOutDir", sidecar["next_command"])
            self.assertIn("-RuntimePrincipal Codex", sidecar["next_command"])
            self.assertIn("Claude", sidecar["runtime_principal_note"])
            self.assertFalse(sidecar["cloud_upload_performed"])
            snapshot = json.loads((root / "outputs" / "api" / "jobs.json").read_text(
                encoding="utf-8"))
            self.assertEqual(job["outputs"]["cloud_review"],
                snapshot["jobs"][0]["outputs"]["cloud_review"])

    def test_paused_and_expired_jobs_keep_reason_and_do_not_queue_cloud(self):
        cases = (
            ({"status": "paused", "reason": "free_quota_exhausted",
              "message": "云端未跑"}, "2099-01-01T00:00:00+08:00", "auto_cloud_paused"),
            ({"status": "running"}, "2000-01-01T00:00:00+08:00", "free_period_expired"),
        )
        for state, cutoff, reason in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                job, sidecar = self._run_job(Path(tmp), cloud_state=state,
                    free_until=cutoff)
                self.assertEqual("succeeded", job["status"])
                self.assertEqual("skipped", sidecar["status"])
                self.assertEqual(reason, sidecar["error_code"])
                self.assertFalse(sidecar["cloud_upload_performed"])


if __name__ == "__main__":
    unittest.main()

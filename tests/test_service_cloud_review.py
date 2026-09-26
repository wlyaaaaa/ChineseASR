from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import patch

from zh_asr.service import JobRequest, ProcessResult, TranscriptionService
from zh_asr.cloud_review import load_cloud_config


ROOT = Path(__file__).resolve().parents[1]


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
            # The Secret Broker recognises the real caller itself.
            self.assertNotIn("-RuntimePrincipal", sidecar["next_command"])
            self.assertNotIn("runtime_principal_note", sidecar)
            self.assertFalse(sidecar["cloud_upload_performed"])
            snapshot = json.loads((root / "outputs" / "api" / "jobs.json").read_text(
                encoding="utf-8"))
            self.assertEqual(job["outputs"]["cloud_review"],
                snapshot["jobs"][0]["outputs"]["cloud_review"])
            Path(job["outputs"]["cloud_review"]).write_text("{broken", encoding="utf-8")
            restored = TranscriptionService(root=root, process_runner=lambda _: None,
                gpu_process_detector=lambda: [], autostart=False)
            self.assertEqual("unreadable", restored.get_job(job["job_id"]).to_dict()["cloud_review"]["status"])

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

    def _run_real_failure(self, root: Path, fault: str, *, prior_status: str | None = None):
        audio = root / "call.wav"
        with wave.open(str(audio), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\0\0" * 1600)
        config = load_cloud_config(ROOT / "configs" / "models.yaml")
        for model in config["models"].values():
            model["free_until"] = "2099-01-01T00:00:00+08:00"
        (root / "configs").mkdir()
        config_path = root / "configs" / "models.yaml"
        config_path.write_text(json.dumps({"cloud_review": config}), encoding="utf-8")
        if fault == "config":
            config_path.write_text("{}", encoding="utf-8")
        elif fault == "yaml":
            config_path.write_text("cloud_review: [", encoding="utf-8")
        elif fault == "audio":
            audio.write_bytes(b"not an audio file")

        def local_runner(job):
            job.out_dir.mkdir(parents=True, exist_ok=True)
            name = "transcript.md" if job.request.mode == "long-strict" else "call.strict.md"
            (job.out_dir / name).write_text("保留本地稿", encoding="utf-8")
            (job.out_dir / "quality.review.json").write_text(
                '{"needs_review":true}', encoding="utf-8")
            if fault == "write":
                # A real filesystem error, not a mocked writer exception.
                (job.out_dir / "cloud.review.json").mkdir()
            elif fault == "missing_audio":
                audio.rename(root / "moved.wav")
            return ProcessResult(returncode=0)

        service = TranscriptionService(root=root, process_runner=local_runner,
            gpu_process_detector=lambda: [], autostart=False)
        service._cloud_review_marker_enabled = True
        request = JobRequest.from_payload({"audio": str(audio),
            "mode": "long-strict" if prior_status else "strict",
            "device": "cpu"}, root=root)
        job, _ = service.submit(request)
        if prior_status:
            job.out_dir.mkdir(parents=True, exist_ok=True)
            (job.out_dir / "cloud.review.json").write_text(json.dumps({
                "schema": "zh_asr.cloud_review.v1", "status": prior_status,
                "message": "上一次的云复核提示",
            }), encoding="utf-8")
        self.assertTrue(service.run_next_job())
        return job

    def test_real_handoff_failures_survive_restart_and_keep_local_transcript(self):
        cases = {"config": "cloud_config_invalid", "yaml": "ParserError",
            "audio": "RuntimeError", "write": "Error", "missing_audio": "FileNotFoundError"}
        for fault, reason in cases.items():
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                job = self._run_real_failure(root, fault)
                result = job.to_dict()
                self.assertEqual("succeeded", result["status"])
                self.assertEqual("finished", result["stage"])
                self.assertEqual(0, result["returncode"])
                self.assertEqual("保留本地稿", Path(result["outputs"]["final"]).read_text(
                    encoding="utf-8"))
                failure = result["cloud_review"]
                self.assertEqual("failed", failure["status"])
                self.assertEqual("cloud_review_not_scheduled", failure["error_code"])
                self.assertIn("云复核未安排成功", failure["message"])
                self.assertIn(reason, failure["message"])
                self.assertFalse(failure["cloud_upload_performed"])
                self.assertFalse(failure["local_text_rewritten"])
                self.assertIn("quality_needs_review", failure["review_reasons"])
                snapshot = json.loads((root / "outputs" / "api" / "jobs.json").read_text(
                    encoding="utf-8"))
                self.assertEqual(failure, snapshot["jobs"][0]["cloud_review"])
                restored = TranscriptionService(root=root, process_runner=lambda _: None,
                    gpu_process_detector=lambda: [], autostart=False)
                self.assertEqual(failure, restored.get_job(job.job_id).to_dict()["cloud_review"])
                self.assertFalse(restored.run_next_job())

    def test_reused_long_output_cannot_hide_a_new_failure_behind_an_old_sidecar(self):
        for prior_status in ("pending_ai_session", "skipped"):
            with self.subTest(prior_status=prior_status), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                job = self._run_real_failure(root, "config", prior_status=prior_status)
                result = job.to_dict()
                self.assertEqual("succeeded", result["status"])
                self.assertEqual("保留本地稿", Path(result["outputs"]["transcript"]).read_text(
                    encoding="utf-8"))
                self.assertEqual("cloud_review_not_scheduled", result["cloud_review"]["error_code"])
                restored = TranscriptionService(root=root, process_runner=lambda _: None,
                    gpu_process_detector=lambda: [], autostart=False)
                self.assertEqual(result["cloud_review"], restored.get_job(job.job_id).to_dict()["cloud_review"])

    def test_existing_batch_recovery_can_replace_a_failed_handoff_without_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = self._run_real_failure(root, "config", prior_status="pending_ai_session")
            spec = importlib.util.spec_from_file_location(
                "service_cloud_batch_test", ROOT / "scripts" / "cloud-review-batch.py")
            batch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(batch)
            with patch.object(batch, "CLOUD_RESULTS_ROOT", root / "cloud-results"):
                candidates = batch.candidates(root / "outputs" / "api" / "jobs.json")
            self.assertEqual([job.job_id], [item["job_id"] for item in candidates])
            retained = root / "retained.result.json"
            retained.write_text(json.dumps({
                "schema": "chineseasr.qwen-audio3-quality-review-result.v1",
                "job_id": "retained", "status": "succeeded", "purpose": "quality_review",
                "credential_result": "Success", "cloud_upload_performed": True,
                "source_audio_sha256": job.request.audio_sha256,
                "selected_channel": None, "text": "离线构造的复核候选",
            }), encoding="utf-8")
            # Exercise the existing retained-result recovery; this sends no audio.
            self.assertTrue(batch._link_result(candidates[0], retained, reused=True))
            self.assertEqual("succeeded", job.to_dict()["cloud_review"]["status"])
            restored = TranscriptionService(root=root, process_runner=lambda _: None,
                gpu_process_detector=lambda: [], autostart=False)
            self.assertEqual("succeeded", restored.get_job(job.job_id).to_dict()["cloud_review"]["status"])
            self.assertEqual("保留本地稿", (job.out_dir / "transcript.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

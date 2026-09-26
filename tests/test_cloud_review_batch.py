from __future__ import annotations

import importlib.util
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "cloud-review-batch.py"


def load_batch():
    spec = importlib.util.spec_from_file_location("cloud_review_batch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class CloudBatchTests(unittest.TestCase):
    def test_pending_job_remains_candidate_and_missing_source_is_reported(self):
        batch = load_batch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "old" / "call.wav"
            recovered = root / "music" / "calls" / "call.wav"
            recovered.parent.mkdir(parents=True)
            recovered.write_bytes(b"matching recording")
            digest = hashlib.sha256(recovered.read_bytes()).hexdigest()
            out_dir = root / "job-1"
            out_dir.mkdir()
            (out_dir / "quality.review.json").write_text(
                '{"needs_review":true}', encoding="utf-8")
            (out_dir / "cloud.review.json").write_text(json.dumps({
                "status": "pending_ai_session"}), encoding="utf-8")
            jobs = root / "jobs.json"
            jobs.write_text(json.dumps({"schema": "zh_asr.jobs.v1", "jobs": [{
                "job_id": "job-1", "status": "succeeded", "out_dir": str(out_dir),
                "evidence_status": "verified", "request": {"audio": str(original),
                    "audio_sha256": digest, "mode": "strict", "important": False}}]}),
                encoding="utf-8")

            missing = []
            self.assertEqual([], batch.candidates(jobs, missing_sources=missing))
            self.assertEqual("job-1", missing[0]["job_id"])
            preview = io.StringIO()
            with redirect_stdout(preview):
                code = batch.main(["--jobs", str(jobs), "--dry-run",
                    "--recover-root", str(root / "absent")])
            self.assertEqual(0, code)
            payload = json.loads(preview.getvalue())
            self.assertEqual(1, payload["missing_source_count"])
            self.assertEqual("job-1", payload["missing_source_jobs"][0]["job_id"])

            preview = io.StringIO()
            with redirect_stdout(preview):
                code = batch.main(["--jobs", str(jobs), "--dry-run",
                    "--recover-root", str(root / "music")])
            self.assertEqual(0, code)
            payload = json.loads(preview.getvalue())
            self.assertEqual(0, payload["missing_source_count"])
            self.assertEqual(1, payload["recovered_source_count"])
            self.assertEqual(str(recovered.resolve()), payload["jobs"][0]["audio"])
            self.assertTrue(payload["jobs"][0]["source_recovered"])

            recovered.write_bytes(b"different recording")
            preview = io.StringIO()
            with redirect_stdout(preview):
                batch.main(["--jobs", str(jobs), "--dry-run",
                    "--recover-root", str(root / "music")])
            payload = json.loads(preview.getvalue())
            self.assertEqual(1, payload["missing_source_count"])
            self.assertEqual(0, payload["count"])

    def test_dedupes_same_audio_but_preserves_selected_channels_and_important(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "call.wav"
            audio.write_bytes(b"fixture")
            jobs = []
            for index, (channel, important) in enumerate(((None, False), (None, True), (1, False))):
                out_dir = root / f"job-{index}"
                out_dir.mkdir()
                if not important:
                    (out_dir / "quality.review.json").write_text('{"needs_review":true}', encoding="utf-8")
                jobs.append({"job_id": str(index), "status": "succeeded",
                    "finished_at": index, "evidence_status": "verified",
                    "out_dir": str(out_dir), "request": {"audio": str(audio),
                    "audio_sha256": "same-audio", "mode": "strict",
                    "channel_index": channel, "important": important}})
            # A quality-review result cannot satisfy a later important-evidence job.
            prior = root / "prior-quality.result.json"
            prior.write_text(json.dumps({"status": "succeeded", "purpose": "quality_review",
                "source_audio_sha256": "same-audio", "selected_channel": None}), encoding="utf-8")
            (root / "job-1" / "cloud.review.json").write_text(json.dumps({
                "status": "succeeded", "cloud_result_path": str(prior)}),
                encoding="utf-8")
            path = root / "jobs.json"
            path.write_text(json.dumps({"schema": "zh_asr.jobs.v1", "jobs": jobs}), encoding="utf-8")
            selected = load_batch().candidates(path)
            self.assertEqual(2, len(selected))
            self.assertEqual("1", selected[0]["job_id"])
            self.assertTrue(selected[0]["important"])
            self.assertEqual(2, len(selected[0]["local_runs"]))
            self.assertEqual({None, 1}, {item["channel_index"] for item in selected})
            cloud_result = root / "cloud.result.json"
            cloud_result.write_text(json.dumps({"status": "succeeded",
                "source_audio_sha256": "same-audio", "selected_channel": None,
                "purpose": "important_evidence", "text": "云端候选", "model": "model",
                "job_id": "cloud-1"}, ensure_ascii=False), encoding="utf-8")
            self.assertTrue(load_batch()._link_result(selected[0], cloud_result))
            for run in selected[0]["local_runs"]:
                sidecar = json.loads((Path(run["out_dir"]) / "cloud.review.json").read_text(encoding="utf-8"))
                self.assertEqual("云端候选", sidecar["cloud_text"])

    def test_transient_cloud_failure_does_not_stop_remaining_batch(self):
        batch = load_batch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = []
            for number in (1, 2):
                out_dir = root / f"job-{number}"
                out_dir.mkdir()
                groups.append({"job_id": str(number), "audio": str(root / "fixture.wav"),
                    "out_dir": str(out_dir), "evidence_status": "verified",
                    "important": False, "audio_sha256": str(number),
                    "channel_index": None, "local_runs": [{"job_id": str(number),
                    "out_dir": str(out_dir), "evidence_status": "verified",
                    "important": False}], "existing_result_path": ""})
            receipts = [Mock(stdout='{"status":"failed","error_code":"network_failure"}'),
                        Mock(stdout='{"status":"succeeded","result_path":"retained.json"}')]
            output = io.StringIO()
            with (patch.object(batch, "candidates", return_value=groups),
                  patch.object(batch, "auto_cloud_status", return_value={"status": "running"}),
                  patch.object(batch.subprocess, "run", side_effect=receipts) as runner,
                  patch.object(batch, "_link_result", return_value=True),
                  redirect_stdout(output)):
                code = batch.main(["--jobs", str(root / "unused.json")])
            self.assertEqual(3, code)
            self.assertEqual(2, runner.call_count)
            self.assertEqual("completed_with_failures", json.loads(output.getvalue())["status"])
            failed_sidecar = json.loads((root / "job-1" / "cloud.review.json").read_text(encoding="utf-8"))
            self.assertEqual("network_failure", failed_sidecar["error_code"])


if __name__ == "__main__":
    unittest.main()

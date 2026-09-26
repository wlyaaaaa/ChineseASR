from __future__ import annotations

import importlib.util
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

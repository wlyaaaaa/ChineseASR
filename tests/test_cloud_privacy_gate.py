import importlib.util
import io
import json
import runpy
from pathlib import Path
import subprocess
import unittest
from contextlib import redirect_stdout
from unittest import mock

from zh_asr import file_job


ROOT = Path(__file__).resolve().parents[1]


def pipeline():
    spec = importlib.util.spec_from_file_location("privacy_pipeline", ROOT / "scripts/cloud-review-pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CloudPrivacyGateTests(unittest.TestCase):
    def test_background_worker_ignores_locked_or_unknown_state(self):
        for state in ("locked", "unknown"):
            with mock.patch.dict(file_job.os.environ, {"AGENTS_BACKGROUND_AUTOMATION": ""}), mock.patch.object(file_job, "_json_process", return_value={"status": "blocked", "data_state": state}) as check, mock.patch.object(file_job, "file_job", side_effect=lambda *a, **k: self.assertTrue(file_job._access_open())) as worker, mock.patch("sys.argv", ["worker", "synthetic.wav", "--background-automation"]), mock.patch.object(file_job.sys, "path", file_job.sys.path[:]):
                runpy.run_path(str(ROOT / "scripts/asr-file-job-worker.py"), run_name="__main__")
            check.assert_not_called()
            worker.assert_called_once_with(Path("synthetic.wav"), worker=True, root=ROOT)

    def test_unconfirmed_access_never_reaches_audio_preparation(self):
        entry = pipeline()
        for result in ({"status": "blocked"}, {}, None, [], "pass"):
            with self.subTest(result=result), mock.patch.dict(file_job.os.environ, {"AGENTS_BACKGROUND_AUTOMATION": ""}), mock.patch.object(file_job, "_json_process", return_value=result):
                output = io.StringIO()
                with mock.patch.object(entry, "prepare_cloud_request") as prepare, redirect_stdout(output):
                    code = entry.main(["prepare", "--root", "unused", "--intent", "unread.json"])
                self.assertEqual(code, 3)
                self.assertEqual(json.loads(output.getvalue()), {"status": "blocked",
                    "error_code": "personal_access_protection_required", "cloud_upload_performed": False})
                prepare.assert_not_called()

    def test_query_failure_is_closed_without_escaping_to_caller(self):
        for error in (OSError(), ValueError(), RuntimeError(), subprocess.TimeoutExpired("access", 40)):
            with self.subTest(error=type(error).__name__), mock.patch.dict(file_job.os.environ, {"AGENTS_BACKGROUND_AUTOMATION": ""}), mock.patch.object(file_job, "_json_process", side_effect=error):
                self.assertFalse(file_job._access_open())

    def test_transmit_without_old_flag_uses_default_guard(self):
        entry = pipeline()
        with (mock.patch("zh_asr.cloud_file_transport.transmit_cloud_chunks") as transmit,
              redirect_stdout(io.StringIO())):
            transmit.return_value = {"status": "blocked", "error_code": "personal_access_protection_required"}
            entry.main(["transmit", "--root", "unused", "--intent", "unread.json", "--request", "pending.json"])
        # No permissive access callback may replace the transport's live default check.
        self.assertNotIn("access", transmit.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()

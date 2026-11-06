import hashlib
import json
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from zh_asr import file_job as jobs

REAL_POPEN = subprocess.Popen


class FileJobTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.audio = self.root / "selected.wav"
        self.audio.write_bytes(b"synthetic audio fixture")
        self.digest = hashlib.sha256(self.audio.read_bytes()).hexdigest()
        self.access = self.enterContext(mock.patch.object(jobs, "_access_open", return_value=True))
        self.reuse = self.enterContext(mock.patch.object(jobs, "_reuse", return_value={}))
        self.gpu = self.enterContext(mock.patch.object(jobs, "_gpu_wait_reason", return_value=""))
        self.process = self.enterContext(mock.patch.object(jobs, "_json_process"))
        self.http = self.enterContext(mock.patch.object(jobs, "_http"))
        self.spawn = self.enterContext(mock.patch.object(jobs.subprocess, "Popen"))
        self.spawn.return_value.pid = 1234
        self.enterContext(mock.patch.object(jobs, "_alive", return_value=True))

    def _artifact(self):
        raw = self.root / "kept.raw.json"
        raw.write_bytes(b'{"text":"synthetic fixture"}')
        return raw, {"raw_json": {"path": str(raw), "size_bytes": raw.stat().st_size,
                                  "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}}

    def test_locked_entry_and_worker_do_not_read_audio_or_start_work(self):
        self.access.return_value = False
        with mock.patch.object(jobs, "file_sha256") as read_audio:
            for worker in (False, True):
                result = jobs.file_job(self.audio, worker=worker, root=self.root)
                self.assertEqual((result["status"], result["reason"]), ("queued", "personal_access_required"))
            read_audio.assert_not_called()
        self.spawn.assert_not_called()
        self.process.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())

    def test_same_sha_success_is_reused_across_paths_without_new_work(self):
        _, artifact = self._artifact()
        self.reuse.return_value = dict(status="succeeded", route="cloud", timestamps="unavailable", job_id="kept", artifact=artifact)
        first = jobs.file_job(self.audio, root=self.root)
        renamed = self.root / "renamed.wav"
        renamed.write_bytes(self.audio.read_bytes())
        second = jobs.file_job(renamed, root=self.root)
        self.assertEqual((first["job_id"], second["job_id"]), ("kept", "kept"))
        self.reuse.assert_called_once_with(self.digest, self.root)
        self.spawn.assert_not_called()
        self.process.assert_not_called()

    def test_repeated_poll_does_not_spawn_second_queued_worker(self):
        self.assertEqual(jobs.file_job(self.audio, root=self.root)["status"], "queued")
        self.assertEqual(jobs.file_job(self.audio, root=self.root)["status"], "queued")
        self.spawn.assert_called_once()
        self.assertIn("--audio", self.spawn.call_args.args[0])
        self.process.assert_not_called()

    def test_cloud_success_without_sentence_times_does_not_run_local(self):
        self.process.return_value = dict(status="succeeded", source_audio_sha256=self.digest,
                                         text="合成测试", job_id="cloud", result_path="kept.result.json")
        self.reuse.return_value = dict(status="succeeded", timestamps="unavailable")
        result = jobs.file_job(self.audio, worker=True, root=self.root)
        self.assertEqual((result["status"], result["route"], result["timestamps"]),
                         ("succeeded", "cloud", "unavailable"))
        self.process.assert_called_once()
        self.assertIn("-CloudFirst", self.process.call_args.args[0])
        self.gpu.assert_not_called()
        self.http.assert_not_called()

    def test_unknown_cloud_falls_back_once_and_polls_existing_local_job(self):
        self.process.side_effect = [dict(status="failed", error_code="previous_outcome_unknown"),
                                    dict(status="queued", job_id="local")]
        self.http.return_value = {"job": dict(status="running", job_id="local", outputs={})}
        first = jobs.file_job(self.audio, worker=True, root=self.root)
        second = jobs.file_job(self.audio, root=self.root)
        self.assertEqual((first["route"], second["status"]), ("local", "running"))
        self.assertEqual(second["fallback_reason"], "previous_outcome_unknown")
        self.assertEqual(self.process.call_count, 2)
        self.assertEqual(sum("-CloudFirst" in call.args[0] for call in self.process.call_args_list), 1)
        self.assertIn("paraformer", self.process.call_args_list[1].args[0])
        self.spawn.assert_not_called()

    def test_gpu_busy_leaves_waiting_state_without_starting_local(self):
        self.process.return_value = dict(status="skipped", error_code="free_period_expired")
        self.gpu.return_value = "gpu_lease_active:chineseasr"
        result = jobs.file_job(self.audio, worker=True, root=self.root)
        self.assertEqual((result["status"], result["fallback_reason"]), ("waiting_gpu", "free_period_expired"))
        self.assertEqual(result["error"], self.gpu.return_value)
        self.process.assert_called_once()
        self.http.assert_not_called()

    def test_local_success_and_blocked_are_projected_without_retranscription(self):
        path = self.root / "outputs/api/file-jobs" / (self.digest + ".json")
        path.parent.mkdir(parents=True)
        state = dict(status="running", route="local", local_job_id="local", source_audio_sha256=self.digest)
        path.write_text(json.dumps(state), encoding="utf-8")
        self.http.return_value = {"job": dict(status="blocked", job_id="local", message="GPU busy")}
        blocked = jobs.file_job(self.audio, root=self.root)
        self.assertEqual(blocked["status"], "waiting_gpu")
        self.assertNotIn("local_job_id", blocked)
        path.write_text(json.dumps(state), encoding="utf-8")
        self.http.return_value = {"job": dict(status="succeeded", job_id="local", outputs={})}
        self.reuse.return_value = dict(status="unavailable", gap={"code": "temporarily_missing"})
        gap = jobs.file_job(self.audio, root=self.root)
        self.assertEqual((gap["status"], gap["error"], gap["local_job_id"]), ("failed", "transcript_readback_gap", "local"))
        self.reuse.return_value = dict(status="succeeded", timestamps="available", artifact={"raw_json": {"path": "raw.json"}})
        completed = jobs.file_job(self.audio, root=self.root)
        self.assertEqual((completed["status"], completed["timestamps"]), ("succeeded", "available"))
        self.process.assert_not_called()
        self.spawn.assert_not_called()

    def test_cloud_readback_gap_rechecks_without_upload_or_local_transcription(self):
        self.process.return_value = dict(status="succeeded", source_audio_sha256=self.digest, text="合成测试", job_id="cloud")
        self.reuse.return_value = dict(status="unavailable", gap={"code": "temporarily_missing"})
        first = jobs.file_job(self.audio, worker=True, root=self.root)
        self.assertTrue(first["cloud_completed"])
        self.assertEqual((first["status"], first["error"]), ("failed", "transcript_readback_gap"))
        self.assertEqual(jobs.file_job(self.audio, root=self.root)["status"], "failed")
        self.reuse.return_value = dict(status="succeeded", timestamps="unavailable", route="cloud")
        self.assertEqual(jobs.file_job(self.audio, root=self.root)["status"], "succeeded")
        self.process.assert_called_once()
        self.spawn.assert_not_called()
        self.gpu.assert_not_called()

    def test_failed_or_blocked_local_retries_only_locally(self):
        path = self.root / "outputs/api/file-jobs" / (self.digest + ".json")
        path.parent.mkdir(parents=True)
        for status in ("failed", "blocked"):
            with self.subTest(status=status):
                path.write_text(json.dumps(dict(status="running", route="local", local_job_id="local", cloud_attempted=True)), encoding="utf-8")
                self.http.return_value = {"job": dict(status=status, job_id="local", message="retry later")}
                self.assertNotIn("local_job_id", jobs.file_job(self.audio, root=self.root))
                self.assertEqual(jobs.file_job(self.audio, root=self.root)["status"], "queued")
                self.process.return_value = dict(status="queued", job_id="retry")
                self.http.return_value = {"job": dict(status="running", job_id="retry")}
                self.assertEqual(jobs.file_job(self.audio, worker=True, root=self.root)["status"], "running")
        self.assertEqual(self.process.call_count, 2)
        self.assertEqual(self.spawn.call_count, 2)
        self.assertTrue(all("-CloudFirst" not in call.args[0] for call in self.process.call_args_list))

    def test_interrupted_cloud_attempt_is_durable_and_never_uploaded_again(self):
        self.process.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            jobs.file_job(self.audio, worker=True, root=self.root)
        path = self.root / "outputs/api/file-jobs" / (self.digest + ".json")
        self.assertTrue(json.loads(path.read_text(encoding="utf-8"))["cloud_attempted"])
        self.process.side_effect = None
        self.process.return_value = dict(status="queued", job_id="fallback")
        self.http.return_value = {"job": dict(status="running", job_id="fallback")}
        result = jobs.file_job(self.audio, worker=True, root=self.root)
        self.assertEqual((result["status"], result["route"]), ("running", "local"))
        self.assertNotIn("-CloudFirst", self.process.call_args.args[0])
        self.assertEqual(self.process.call_count, 2)

    def test_missing_or_corrupt_completed_artifact_only_rechecks_readback(self):
        for damage in ("missing", "size", "hash"):
            with self.subTest(damage=damage):
                raw, artifact = self._artifact()
                self.reuse.return_value = dict(status="succeeded", route="cloud", timestamps="unavailable", artifact=artifact)
                self.assertEqual(jobs.file_job(self.audio, root=self.root)["status"], "succeeded")
                if damage == "missing":
                    raw.unlink()
                else:
                    raw.write_bytes(b"x" * (raw.stat().st_size + (damage == "size")))
                self.reuse.reset_mock()
                self.reuse.return_value = dict(status="unavailable", gap={"code": "raw_artifact_invalid"})
                result = jobs.file_job(self.audio, root=self.root)
                self.assertEqual((result["status"], result["error"]), ("failed", "transcript_readback_gap"))
                self.assertTrue(result["transcript_completed"])
                self.reuse.assert_called_once_with(self.digest, self.root)
                self.assertEqual(jobs.file_job(self.audio, worker=True, root=self.root)["status"], "failed")
                self.assertEqual(self.reuse.call_count, 2)
        self.spawn.assert_not_called()
        self.process.assert_not_called()

    def test_os_lock_competition_is_nonblocking(self):
        lock = self.root / "worker.lock"
        script = "from pathlib import Path\nimport sys\nfrom zh_asr.file_job import _claim\nwith _claim(Path(sys.argv[1])) as held:\n print('ready' if held else 'busy', flush=True)\n sys.stdin.readline()\n"
        with REAL_POPEN([sys.executable, "-B", "-c", script, str(lock)], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                        **jobs.managed_popen_kwargs()) as child:
            try:
                ready = queue.Queue()
                threading.Thread(target=lambda: ready.put(child.stdout.readline()), daemon=True).start()
                self.assertEqual(ready.get(timeout=10), "ready\n")
                started = time.monotonic()
                with jobs._claim(lock) as held:
                    self.assertFalse(held)
                self.assertLess(time.monotonic() - started, 1)
                child.communicate(input="\n", timeout=5)
                self.assertEqual(child.returncode, 0)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)

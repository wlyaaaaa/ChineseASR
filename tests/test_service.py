import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
import wave
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import Request, urlopen
from pathlib import Path

from zh_asr.service import (
    CALLER_BINDING_ENV,
    GpuProcess,
    JobRequest,
    PersistenceNotReadyError,
    ProcessResult,
    TERMINAL_STATUSES,
    TranscriptionService,
    create_handler,
    detect_gpu_processes,
)
from zh_asr.audio_outcome import build_objective_result, write_objective_result
from zh_asr.service import _objective_from_job_outputs
from zh_asr.strict_writer import write_strict_bundle


class ServiceTests(unittest.TestCase):
    def test_caller_binding_reaches_child_only_via_json_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            request = JobRequest.from_payload(
                {
                    "audio": str(audio),
                    "mode": "quick",
                    "device": "cpu",
                    "caller_binding": {"opaque_ref": "caller-owned"},
                },
                root=root,
            )
            service = TranscriptionService(root=root, autostart=False)
            job = service._new_job(request, conflicts=[])
            process = SimpleNamespace(
                pid=1234,
                returncode=0,
                communicate=lambda timeout: ("", ""),
            )
            with (
                patch("zh_asr.service.subprocess.Popen", return_value=process) as popen,
                patch("zh_asr.service.managed_popen_kwargs", return_value={}),
            ):
                service._run_subprocess(job)

        env = popen.call_args.kwargs["env"]
        self.assertEqual('{"opaque_ref":"caller-owned"}', env[CALLER_BINDING_ENV])
        self.assertNotIn("caller-owned", " ".join(job.command))

    def test_caller_binding_is_opaque_in_request_identity_and_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            binding = {"opaque_ref": "caller-owned"}
            request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "quick", "caller_binding": binding},
                root=root,
            )
            other = JobRequest.from_payload(
                {"audio": str(audio), "mode": "quick", "caller_binding": {"opaque_ref": "other"}},
                root=root,
            )

        self.assertEqual(request.caller_binding, binding)
        self.assertEqual(request.to_dict()["caller_binding"], binding)
        self.assertNotEqual(request.fingerprint(), other.fingerprint())

    def test_persisted_snapshot_preserves_caller_binding_passthrough(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            state_dir = root / "state"
            binding = {"opaque_ref": "caller-owned", "source": "caller"}
            request = JobRequest.from_payload(
                {
                    "audio": str(audio),
                    "mode": "quick",
                    "device": "cpu",
                    "caller_binding": binding,
                },
                root=root,
            )
            service = TranscriptionService(
                root=root,
                default_out_root=state_dir,
                gpu_process_detector=lambda: [],
                autostart=False,
            )
            job, _ = service.submit(request)
            snapshot = json.loads(
                (state_dir / "jobs.json").read_text(encoding="utf-8")
            )
            restarted = TranscriptionService(
                root=root,
                default_out_root=state_dir,
                gpu_process_detector=lambda: [],
                autostart=False,
            )
            restored = restarted.get_job(job.job_id)

        self.assertEqual(
            binding,
            snapshot["jobs"][0]["request"]["caller_binding"],
        )
        self.assertIsNotNone(restored)
        self.assertEqual(binding, restored.request.caller_binding)

    def test_service_rejects_tampered_objective_sidecar_before_reporting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "silence.wav")
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\0\0" * 1600)
            payload = build_objective_result(
                audio_path=audio,
                mode="quick",
                engines=["sensevoice"],
                primary_text="",
            )
            sidecar = root / "silence.objective-result.json"
            write_objective_result(sidecar, payload)
            valid = _objective_from_job_outputs(
                {"objective_result": str(sidecar)},
                "quick",
            )
            payload["quality"]["status"] = "unknown"
            write_objective_result(sidecar, payload)
            tampered = _objective_from_job_outputs(
                {"objective_result": str(sidecar)},
                "quick",
            )

        self.assertEqual(("no_speech_detected", "completed"), valid)
        self.assertEqual(("indeterminate", "failed"), tampered)
    def test_stop_cancels_queued_and_running_jobs_and_terminates_active_process(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_audio = _write_audio(root / "first.wav")
            second_audio = _write_audio(root / "second.wav")
            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                autostart=False,
            )
            running, _ = service.submit(
                JobRequest.from_payload(
                    {"audio": str(first_audio), "mode": "quick"},
                    root=root,
                )
            )
            queued, _ = service.submit(
                JobRequest.from_payload(
                    {"audio": str(second_audio), "mode": "quick"},
                    root=root,
                )
            )
            running.status = "running"
            process = SimpleNamespace()
            service._processes[running.job_id] = process

            terminated = []
            with patch(
                "zh_asr.service._terminate_job_process",
                side_effect=lambda job, value: terminated.append(
                    (job.job_id, value)
                ),
            ):
                service.stop()

        self.assertEqual(running.status, "canceled")
        self.assertEqual(queued.status, "canceled")
        self.assertEqual(terminated, [(running.job_id, process)])

    def test_cancel_still_terminates_when_history_persistence_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                autostart=False,
            )
            request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "quick", "device": "cpu"},
                root=root,
            )
            job, _ = service.submit(request)
            job.status = "running"
            process = SimpleNamespace(pid=1234)
            service._processes[job.job_id] = process

            with (
                patch(
                    "zh_asr.service._write_json_atomic",
                    side_effect=OSError("read-only state directory"),
                ),
                patch("zh_asr.service._terminate_job_process") as terminate,
            ):
                canceled = service.cancel(job.job_id)
                health = service.health()

        self.assertEqual("canceled", canceled.status)
        terminate.assert_called_once_with(job, process)
        self.assertEqual("degraded", health["persistence"]["status"])
        self.assertFalse(health["persistence"]["ready"])
        self.assertIn("memory-only", canceled.message)

    def test_lease_loss_still_terminates_when_history_persistence_fails(self):
        from zh_asr.gpu_broker import GpuBrokerLeaseLost

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                autostart=False,
            )
            request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "quick", "device": "cuda:0"},
                root=root,
            )
            job, _ = service.submit(request)
            job.status = "running"
            process = SimpleNamespace(pid=1234)
            service._processes[job.job_id] = process
            error = GpuBrokerLeaseLost("lease expired")

            with (
                patch(
                    "zh_asr.service._write_json_atomic",
                    side_effect=OSError("read-only state directory"),
                ),
                patch("zh_asr.service._terminate_job_process") as terminate,
            ):
                service._terminate_job_after_lease_loss(job.job_id, error)
                health = service.health()

        terminate.assert_called_once_with(job, process)
        self.assertEqual("lease expired", job.gpu_broker_loss_error)
        self.assertIn("lease expired", job.message)
        self.assertEqual("degraded", health["persistence"]["status"])

    def test_run_next_job_marks_persistence_failure_and_rejects_new_work(self):
        runner_calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_audio = _write_audio(root / "first.wav")
            second_audio = _write_audio(root / "second.wav")
            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                process_runner=lambda _job: runner_calls.append(True)
                or ProcessResult(returncode=0),
                autostart=False,
            )
            first_request = JobRequest.from_payload(
                {"audio": str(first_audio), "mode": "quick", "device": "cpu"},
                root=root,
            )
            second_request = JobRequest.from_payload(
                {"audio": str(second_audio), "mode": "quick", "device": "cpu"},
                root=root,
            )
            first, _ = service.submit(first_request)

            with patch(
                "zh_asr.service._write_json_atomic",
                side_effect=OSError("read-only state directory"),
            ):
                self.assertTrue(service.run_next_job())
                health = service.health()
                with self.assertRaises(PersistenceNotReadyError):
                    service.submit(second_request)

        self.assertEqual([], runner_calls)
        self.assertEqual("failed", first.status)
        self.assertEqual("persistence_degraded", first.stage)
        self.assertEqual("not_applicable", first.evidence_status)
        self.assertEqual("degraded", health["persistence"]["status"])
        self.assertIs(service.get_job(first.job_id), first)

    def test_successful_command_is_not_reported_success_when_terminal_history_fails(self):
        runner_calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                process_runner=lambda _job: runner_calls.append(True)
                or ProcessResult(returncode=0),
                autostart=False,
            )
            request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "quick", "device": "cpu"},
                root=root,
            )
            job, _ = service.submit(request)

            with patch(
                "zh_asr.service._write_json_atomic",
                side_effect=[None, OSError("read-only state directory")],
            ):
                self.assertTrue(service.run_next_job())

        self.assertEqual([True], runner_calls)
        self.assertEqual("failed", job.status)
        self.assertEqual("persistence_degraded", job.stage)
        self.assertIn("terminal job history is not durable", job.message)
        self.assertIn("transcription output files may already exist", job.message)

    def test_service_restart_resets_stop_event_and_runs_new_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            service = TranscriptionService(
                root=root,
                default_out_root=root / "state",
                gpu_process_detector=lambda: [],
                process_runner=lambda _job: ProcessResult(returncode=0),
                autostart=False,
            )
            service.start()
            service.stop()
            service.start()
            try:
                job, _ = service.submit(
                    JobRequest.from_payload(
                        {"audio": str(audio), "mode": "quick", "device": "cpu"},
                        root=root,
                    )
                )
                deadline = time.monotonic() + 2
                while job.status not in TERMINAL_STATUSES and time.monotonic() < deadline:
                    time.sleep(0.01)
            finally:
                service.stop()

        self.assertEqual("succeeded", job.status)

    def test_service_persists_terminal_jobs_and_marks_interrupted_jobs_without_rerun(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state_dir = root / "state"
            completed_audio = _write_audio(root / "completed.wav")
            queued_audio = _write_audio(root / "queued.wav")
            service = TranscriptionService(
                root=root,
                default_out_root=state_dir,
                gpu_process_detector=lambda: [],
                process_runner=lambda _job: ProcessResult(returncode=0),
                autostart=False,
            )
            completed_request = JobRequest.from_payload(
                {"audio": str(completed_audio), "mode": "quick", "device": "cpu"},
                root=root,
            )
            queued_request = JobRequest.from_payload(
                {"audio": str(queued_audio), "mode": "quick", "device": "cpu"},
                root=root,
            )
            completed, _ = service.submit(completed_request)
            service.run_next_job()
            queued, _ = service.submit(queued_request)
            state_path = state_dir / "jobs.json"
            self.assertTrue(state_path.is_file())
            self.assertEqual("succeeded", completed.status)
            self.assertEqual("queued", queued.status)

            restarted = TranscriptionService(
                root=root,
                default_out_root=state_dir,
                gpu_process_detector=lambda: [],
                process_runner=lambda _job: (_ for _ in ()).throw(
                    AssertionError("restarted service must not auto-rerun")
                ),
                autostart=False,
            )
            restored_completed = restarted.get_job(completed.job_id)
            restored_queued = restarted.get_job(queued.job_id)
            self.assertIsNotNone(restored_completed)
            self.assertIsNotNone(restored_queued)
            self.assertEqual("succeeded", restored_completed.status)
            self.assertEqual("failed", restored_queued.status)
            self.assertEqual("service_restarted", restored_queued.stage)
            self.assertFalse(restarted.run_next_job())
            replacement, deduplicated = restarted.submit(completed_request)
            self.assertFalse(deduplicated)
            self.assertNotEqual(completed.job_id, replacement.job_id)

    def test_service_bounds_terminal_snapshot_without_deleting_old_outputs(self):
        from zh_asr import service as service_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = TranscriptionService(
                root=root,
                default_out_root=root / "state",
                gpu_process_detector=lambda: [],
                process_runner=lambda _job: ProcessResult(returncode=0),
                autostart=False,
            )
            first_audio = _write_audio(root / "first.wav")
            second_audio = _write_audio(root / "second.wav")
            first, _ = service.submit(
                JobRequest.from_payload(
                    {"audio": str(first_audio), "mode": "quick", "device": "cpu"},
                    root=root,
                )
            )
            service.run_next_job()
            first.out_dir.mkdir(parents=True, exist_ok=True)
            marker = first.out_dir / "keep.marker"
            marker.write_text("outputs remain", encoding="utf-8")
            second, _ = service.submit(
                JobRequest.from_payload(
                    {"audio": str(second_audio), "mode": "quick", "device": "cpu"},
                    root=root,
                )
            )
            service.run_next_job()

            with patch.object(service_module, "MAX_PERSISTED_TERMINAL_JOBS", 1):
                service._persist_jobs_locked()
            state = json.loads((root / "state" / "jobs.json").read_text(encoding="utf-8"))
            self.assertTrue(marker.is_file())

        self.assertEqual([second.job_id], [item["job_id"] for item in state["jobs"]])

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

    def test_submit_defers_foreign_gpu_process_to_broker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            conflict = GpuProcess(pid=9999, process_name="other-cuda.exe", used_memory_mib=4096)
            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [conflict],
                current_process_ids=lambda: {1111},
                autostart=False,
            )
            request = JobRequest.from_payload({"audio": str(audio), "mode": "strict"}, root=root)

            job, deduped = service.submit(request)

            self.assertFalse(deduped)
            self.assertEqual("queued", job.status)
            self.assertEqual([], job.conflicts)

    def test_submit_defers_managed_ollama_conflict_to_broker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            conflict = GpuProcess(pid=9999, process_name="llama-server.exe", used_memory_mib=26000)
            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [conflict],
                current_process_ids=lambda: {1111},
                autostart=False,
            )
            request = JobRequest.from_payload({"audio": str(audio), "mode": "strict"}, root=root)

            job, _ = service.submit(request)

            self.assertEqual("queued", job.status)
            self.assertEqual([], job.conflicts)

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

    def test_allow_gpu_conflicts_policy_is_part_of_request_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                process_runner=lambda _job: ProcessResult(returncode=0),
                autostart=False,
            )
            strict_request = JobRequest.from_payload(
                {
                    "audio": str(audio),
                    "mode": "strict",
                    "allow_gpu_conflicts": False,
                },
                root=root,
            )
            permissive_request = JobRequest.from_payload(
                {
                    "audio": str(audio),
                    "mode": "strict",
                    "allow_gpu_conflicts": True,
                },
                root=root,
            )

            first, _ = service.submit(permissive_request)
            second, deduped = service.submit(strict_request)

        self.assertNotEqual(
            permissive_request.fingerprint(),
            strict_request.fingerprint(),
        )
        self.assertFalse(deduped)
        self.assertNotEqual(first.job_id, second.job_id)

    def test_detect_gpu_processes_ignores_rows_without_numeric_memory(self):
        nvidia_smi_output = (
            "2800, [Insufficient Permissions], [N/A]\n"
            "1234, C:\\tools\\ollama.exe, 4096\n"
        )
        completed = SimpleNamespace(returncode=0, stdout=nvidia_smi_output)

        with patch("zh_asr.service.subprocess.run", return_value=completed):
            processes = detect_gpu_processes()

        self.assertEqual(1, len(processes))
        self.assertEqual(1234, processes[0].pid)
        self.assertEqual("C:\\tools\\ollama.exe", processes[0].process_name)
        self.assertEqual(4096, processes[0].used_memory_mib)

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

    def test_long_retry_after_failure_or_cancel_reuses_stable_output_directory(self):
        for terminal_status in ("failed", "canceled"):
            with self.subTest(terminal_status=terminal_status), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                audio = _write_audio(root / "sample.wav")
                service = TranscriptionService(
                    root=root,
                    gpu_process_detector=lambda: [],
                    process_runner=lambda _job: ProcessResult(returncode=1, stderr="failed"),
                    autostart=False,
                )
                request = JobRequest.from_payload(
                    {"audio": str(audio), "mode": "long-strict"},
                    root=root,
                )

                first, first_deduped = service.submit(request)
                if terminal_status == "failed":
                    service.run_next_job()
                else:
                    service.cancel(first.job_id)
                retry, retry_deduped = service.submit(request)

                self.assertFalse(first_deduped)
                self.assertFalse(retry_deduped)
                self.assertEqual(terminal_status, first.status)
                self.assertNotEqual(first.job_id, retry.job_id)
                self.assertEqual(first.out_dir, retry.out_dir)
                self.assertIn(str(first.out_dir), first.command)
                self.assertIn(str(retry.out_dir), retry.command)

    def test_long_active_force_duplicate_is_deduplicated_to_avoid_output_collision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            service = TranscriptionService(root=root, gpu_process_detector=lambda: [], autostart=False)
            request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "long-strict", "force": True},
                root=root,
            )

            first, first_deduped = service.submit(request)
            duplicate, duplicate_deduped = service.submit(request)

        self.assertFalse(first_deduped)
        self.assertTrue(duplicate_deduped)
        self.assertEqual(first.job_id, duplicate.job_id)
        self.assertEqual(first.out_dir, duplicate.out_dir)
        self.assertIn("--force", first.command)

    def test_long_request_freezes_resolved_default_engines_and_config_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            config_path = root / "models.yaml"
            config_path.write_text("version one", encoding="utf-8")
            config = _model_config(config_path, primary="primary-a", secondary="anchor")

            with patch("zh_asr.config.load_model_config", return_value=config):
                request = JobRequest.from_payload(
                    {"audio": str(audio), "mode": "long-strict"},
                    root=root,
                )

            service = TranscriptionService(root=root, gpu_process_detector=lambda: [], autostart=False)
            job, _ = service.submit(request)

        self.assertEqual("primary-a", request.resolved_primary_engine)
        self.assertEqual("anchor", request.resolved_secondary_engine)
        self.assertTrue(request.model_config_sha256)
        self.assertIn("--primary-engine", job.command)
        self.assertIn("primary-a", job.command)
        self.assertIn("--secondary-engine", job.command)
        self.assertIn("anchor", job.command)

    def test_submit_does_not_deduplicate_when_resolved_default_engine_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            config_path = root / "models.yaml"
            config_path.write_text("unchanged bytes", encoding="utf-8")
            first_config = _model_config(config_path, primary="primary-a", secondary="anchor")
            second_config = _model_config(config_path, primary="primary-b", secondary="anchor")

            with patch(
                "zh_asr.config.load_model_config",
                side_effect=[first_config, second_config],
            ):
                first_request = JobRequest.from_payload(
                    {"audio": str(audio), "mode": "long-strict"},
                    root=root,
                )
                second_request = JobRequest.from_payload(
                    {"audio": str(audio), "mode": "long-strict"},
                    root=root,
                )

            service = TranscriptionService(root=root, gpu_process_detector=lambda: [], autostart=False)
            first, first_deduped = service.submit(first_request)
            second, second_deduped = service.submit(second_request)
            fingerprints = (first_request.fingerprint(), second_request.fingerprint())

        self.assertFalse(first_deduped)
        self.assertFalse(second_deduped)
        self.assertNotEqual(first.job_id, second.job_id)
        self.assertNotEqual(*fingerprints)

    def test_submit_does_not_deduplicate_when_model_config_bytes_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            config_path = root / "models.yaml"
            config = _model_config(config_path, primary="primary-a", secondary="anchor")

            config_path.write_text("version one", encoding="utf-8")
            with patch("zh_asr.config.load_model_config", return_value=config):
                first_request = JobRequest.from_payload(
                    {"audio": str(audio), "mode": "long-strict"},
                    root=root,
                )

            config_path.write_text("version two", encoding="utf-8")
            with patch("zh_asr.config.load_model_config", return_value=config):
                second_request = JobRequest.from_payload(
                    {"audio": str(audio), "mode": "long-strict"},
                    root=root,
                )

            service = TranscriptionService(root=root, gpu_process_detector=lambda: [], autostart=False)
            first, first_deduped = service.submit(first_request)
            second, second_deduped = service.submit(second_request)

        self.assertFalse(first_deduped)
        self.assertFalse(second_deduped)
        self.assertNotEqual(first.job_id, second.job_id)
        self.assertNotEqual(first_request.model_config_sha256, second_request.model_config_sha256)

    def test_request_fingerprint_uses_audio_content_not_only_size_and_mtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            original_stat = audio.stat()
            first_request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "long-strict"},
                root=root,
            )

            audio.write_bytes(b"RIFX")
            os.utime(audio, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            second_request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "long-strict"},
                root=root,
            )
            fingerprints = (first_request.fingerprint(), second_request.fingerprint())

        self.assertNotEqual(first_request.audio_sha256, second_request.audio_sha256)
        self.assertNotEqual(*fingerprints)

    def test_long_request_fingerprint_is_stable_when_only_audio_mtime_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            first_request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "long-strict"},
                root=root,
            )
            original_stat = audio.stat()
            os.utime(
                audio,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000),
            )
            second_request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "long-strict"},
                root=root,
            )

            first_service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                autostart=False,
            )
            second_service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                autostart=False,
            )
            first_job, _ = first_service.submit(first_request)
            second_job, _ = second_service.submit(second_request)

        self.assertEqual(first_request.audio_sha256, second_request.audio_sha256)
        self.assertEqual(first_request.fingerprint(), second_request.fingerprint())
        self.assertEqual(first_job.out_dir, second_job.out_dir)

    def test_run_next_job_marks_success_and_collects_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")

            def fake_runner(job):
                _write_service_strict_artifacts(job)
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
            self.assertEqual("verified", refreshed.evidence_status)
            self.assertIn("final", refreshed.outputs)
            self.assertIn("audit", refreshed.outputs)
            self.assertIn("audit_json", refreshed.outputs)
            self.assertIn("primary_raw_json", refreshed.outputs)
            self.assertIn("secondary_raw_json", refreshed.outputs)
            self.assertTrue(refreshed.outputs["final"].endswith("sample.strict.md"))
            self.assertTrue(refreshed.outputs["primary_raw_json"].endswith("sample.qwen3-asr-1.7b.raw.json"))
            self.assertTrue(refreshed.outputs["secondary_raw_json"].endswith("sample.sensevoice.raw.json"))

    def test_api_job_exposes_successful_firered_fallback_as_provisional(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")

            def fake_runner(job):
                _write_service_strict_artifacts(job, fail_primary=True)
                return ProcessResult(returncode=0)

            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                process_runner=fake_runner,
                autostart=False,
            )
            request = JobRequest.from_payload(
                {
                    "audio": str(audio),
                    "mode": "strict",
                    "primary_engine": "fireredasr2-llm",
                    "secondary_engine": "qwen3-asr-1.7b",
                },
                root=root,
            )
            job, _ = service.submit(request)
            service.run_next_job()
            payload = job.to_dict()

        self.assertEqual(payload["status"], "succeeded")
        self.assertEqual(payload["evidence_status"], "provisional")
        self.assertEqual(
            payload["evidence_failures"],
            [
                {
                    "engine": "fireredasr2-llm",
                    "role": "lexical_primary",
                    "error": "RuntimeError: worker exited 9",
                }
            ],
        )

    def test_successful_strict_job_is_evidence_unavailable_when_raw_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")

            def fake_runner(job):
                _write_service_strict_artifacts(job)
                primary = (
                    job.request.resolved_primary_engine or job.request.primary_engine
                )
                (job.out_dir / f"sample.{primary}.raw.json").unlink()
                return ProcessResult(returncode=0)

            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                process_runner=fake_runner,
                autostart=False,
            )
            request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "strict"},
                root=root,
            )
            job, _ = service.submit(request)
            service.run_next_job()

        self.assertEqual(job.status, "succeeded")
        self.assertEqual(job.evidence_status, "unavailable")
        self.assertIn("primary raw JSON", job.evidence_failures[0]["error"])

    def test_long_api_job_propagates_manifest_evidence_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")

            def fake_runner(job):
                job.out_dir.mkdir(parents=True, exist_ok=True)
                outputs = _write_service_strict_artifacts(
                    job,
                    fail_primary=True,
                )
                (job.out_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            "resolved_primary_engine": "fireredasr2-llm",
                            "resolved_secondary_engine": "qwen3-asr-1.7b",
                            "evidence_status": "provisional",
                            "evidence_failures": [
                                {
                                    "chunk_id": "chunk-000001",
                                    "engine": "fireredasr2-llm",
                                    "role": "lexical_primary",
                                    "error": "RuntimeError: worker exited 9",
                                }
                            ],
                            "chunks": [
                                {
                                    "chunk_id": "chunk-000001",
                                    "status": "succeeded",
                                    "evidence_status": "provisional",
                                    "outputs": {
                                        key: str(value)
                                        for key, value in outputs.items()
                                        if isinstance(value, Path)
                                    },
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return ProcessResult(returncode=0)

            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                process_runner=fake_runner,
                autostart=False,
            )
            request = JobRequest.from_payload(
                {
                    "audio": str(audio),
                    "mode": "long-strict",
                    "primary_engine": "fireredasr2-llm",
                    "secondary_engine": "qwen3-asr-1.7b",
                },
                root=root,
            )
            job, _ = service.submit(request)
            service.run_next_job()

        self.assertEqual(job.status, "succeeded")
        self.assertEqual(job.evidence_status, "provisional")
        self.assertEqual(job.evidence_failures[0]["engine"], "fireredasr2-llm")

    def test_run_subprocess_uses_finite_deadline_and_tagged_environment(self):
        from zh_asr.service import ProcessExecutionTimeout

        class HangingProcess:
            pid = 4242
            returncode = None

            def communicate(self, timeout=None):
                raise subprocess.TimeoutExpired(cmd=["python"], timeout=timeout)

            def poll(self):
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            request = JobRequest(
                audio=audio,
                mode="strict",
                engine=None,
                primary_engine="qwen3-asr-1.7b",
                secondary_engine="sensevoice",
                device="cpu",
                out_root=root / "outputs",
                timeout_sec=0.1,
            )
            service = TranscriptionService(root=root, gpu_process_detector=lambda: [], autostart=False)
            job = service._new_job(request, conflicts=[])
            process = HangingProcess()

            with (
                patch("zh_asr.service.subprocess.Popen", return_value=process) as popen,
                patch("zh_asr.service.terminate_process_tree") as terminate_tree,
                patch("zh_asr.service.terminate_wsl_processes") as terminate_wsl,
            ):
                with self.assertRaisesRegex(ProcessExecutionTimeout, "0.1"):
                    service._run_subprocess(job)

            env = popen.call_args.kwargs["env"]
            self.assertEqual(env["ZH_ASR_PROCESS_TOKEN"], f"chineseasr-{job.job_id}")
            self.assertIn("ZH_ASR_PROCESS_TOKEN", env["WSLENV"].split(":"))
            terminate_tree.assert_called_once_with(process)
            terminate_wsl.assert_called_once_with((), f"chineseasr-{job.job_id}")

    def test_cancel_running_job_terminates_tree_and_only_matching_wsl_token(self):
        process = SimpleNamespace(pid=4343, poll=lambda: None)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            request = JobRequest.from_payload(
                {
                    "audio": str(audio),
                    "mode": "strict",
                    "primary_engine": "fireredasr2-llm",
                    "secondary_engine": "sensevoice",
                },
                root=root,
            )
            service = TranscriptionService(root=root, gpu_process_detector=lambda: [], autostart=False)
            job = service._new_job(request, conflicts=[])
            job.status = "running"
            service._jobs[job.job_id] = job
            service._processes[job.job_id] = process
            self.assertEqual(("Ubuntu",), request.wsl_distributions)

            with (
                patch("zh_asr.service.terminate_process_tree") as terminate_tree,
                patch(
                    "zh_asr.service._request_wsl_distributions",
                    return_value=("Ubuntu",),
                ),
                patch("zh_asr.service.terminate_wsl_processes") as terminate_wsl,
            ):
                canceled = service.cancel(job.job_id)

        self.assertEqual("canceled", canceled.status)
        terminate_tree.assert_called_once_with(process)
        terminate_wsl.assert_called_once_with(
            ("Ubuntu",),
            f"chineseasr-{job.job_id}",
        )

    def test_gpu_lease_is_released_when_process_runner_times_out(self):
        events = []

        class Lease:
            def __enter__(self):
                events.append("lease_enter")

            def __exit__(self, exc_type, exc, traceback):
                events.append(("lease_exit", exc_type))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            request = JobRequest.from_payload(
                {"audio": str(audio), "mode": "strict"},
                root=root,
            )
            service = TranscriptionService(
                root=root,
                gpu_process_detector=lambda: [],
                process_runner=lambda _job: (_ for _ in ()).throw(TimeoutError("deadline")),
                gpu_lease_factory=lambda _owner: Lease(),
                autostart=False,
            )
            job, _ = service.submit(request)
            service.run_next_job()

        self.assertEqual("failed", job.status)
        self.assertEqual(
            ["lease_enter", ("lease_exit", TimeoutError)],
            events,
        )

    def test_request_timeout_must_be_positive_and_finite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = _write_audio(root / "sample.wav")
            for invalid in (0, -1, float("inf"), "not-a-number"):
                with self.subTest(invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "timeout_sec"):
                        JobRequest.from_payload(
                            {
                                "audio": str(audio),
                                "mode": "strict",
                                "timeout_sec": invalid,
                            },
                            root=root,
                        )

            short = JobRequest.from_payload(
                {
                    "audio": str(audio),
                    "mode": "long-strict",
                    "timeout_sec": 30,
                },
                root=root,
            )
            extended = JobRequest.from_payload(
                {
                    "audio": str(audio),
                    "mode": "long-strict",
                    "timeout_sec": 300,
                },
                root=root,
            )

        self.assertEqual(short.fingerprint(), extended.fingerprint())

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
            self.assertEqual("pending", submitted["job"]["evidence_status"])
            self.assertFalse(submitted["deduplicated"])
            self.assertEqual(submitted["job"]["job_id"], status["job"]["job_id"])
            self.assertEqual("canceled", canceled["job"]["status"])
            self.assertEqual("unavailable", canceled["job"]["evidence_status"])

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


def _write_service_strict_artifacts(
    job,
    *,
    fail_primary: bool = False,
) -> dict[str, object]:
    primary = job.request.resolved_primary_engine or job.request.primary_engine
    secondary = job.request.resolved_secondary_engine or job.request.secondary_engine
    primary_error = "RuntimeError: worker exited 9" if fail_primary else None
    primary_result = {
        "engine": primary,
        "text": "" if fail_primary else "正文",
        "error": (
            {"type": "RuntimeError", "message": "worker exited 9"}
            if fail_primary
            else None
        ),
    }
    return write_strict_bundle(
        audio_path=job.request.audio,
        primary_engine=primary,
        primary_result=primary_result,
        secondary_engine=secondary,
        secondary_result={"engine": secondary, "text": "正文", "error": None},
        out_dir=job.out_dir,
        primary_error=primary_error,
    )


def _model_config(path: Path, *, primary: str, secondary: str):
    return SimpleNamespace(
        path=path,
        default_engine="sensevoice",
        strict_primary_engine=primary,
        strict_secondary_engine=secondary,
        engines={},
    )


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

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHONPATH = str(PROJECT_ROOT / "src")


class CliTests(unittest.TestCase):
    def test_caller_binding_env_round_trips_as_opaque_json_object(self):
        from zh_asr.__main__ import CALLER_BINDING_ENV, _caller_binding_from_env

        binding = {"opaque_ref": "caller-owned"}
        with patch.dict(os.environ, {CALLER_BINDING_ENV: '{"opaque_ref":"caller-owned"}'}, clear=False):
            self.assertEqual(binding, _caller_binding_from_env())
        with patch.dict(os.environ, {CALLER_BINDING_ENV: "[]"}, clear=False):
            with self.assertRaises(ValueError):
                _caller_binding_from_env()

    def test_gpu_operation_refuses_unauthenticated_in_process_execution(self):
        from zh_asr.__main__ import _run_with_gpu_lease
        from zh_asr.gpu_broker import GpuBrokerError

        with self.assertRaises(GpuBrokerError):
            _run_with_gpu_lease("cuda:0", lambda: "must not run")

    def test_legacy_plain_marker_cannot_bypass_gpu_broker(self):
        from zh_asr.__main__ import _run_with_gpu_lease
        from zh_asr.gpu_broker import GpuBrokerError

        with patch.dict(
            os.environ,
            {"ZH_ASR_GPU_BROKER_LEASE_HELD": "1"},
            clear=False,
        ), self.assertRaises(GpuBrokerError):
            _run_with_gpu_lease("cuda:0", lambda: "must not run")

    def test_gpu_cli_supervisor_passes_live_token_to_worker(self):
        from zh_asr.__main__ import _supervise_gpu_cli
        from zh_asr.gpu_broker import GPU_BROKER_CHILD_TOKEN_ENV
        from zh_asr.process_control import PROCESS_TOKEN_ENV

        captured = {}

        class Lease:
            token = "live-asr-token"

            def __init__(self, owner):
                captured["owner"] = owner

            def set_on_lost(self, callback):
                captured["callback"] = callback

            def __enter__(self):
                return self

            def raise_if_lost(self):
                return None

            def __exit__(self, *_args):
                return None

        class Process:
            pid = 1234
            returncode = 0

            def wait(self):
                return self.returncode

            def poll(self):
                return self.returncode

        def popen(command, **kwargs):
            captured["command"] = command
            captured["env"] = kwargs["env"]
            return Process()

        with patch("zh_asr.__main__.GpuBrokerLease", Lease), patch(
            "zh_asr.__main__.subprocess.Popen",
            side_effect=popen,
        ):
            returncode = _supervise_gpu_cli(["transcribe", "note.wav"])

        self.assertEqual(returncode, 0)
        self.assertEqual(captured["owner"], "chineseasr-cli")
        self.assertEqual(
            captured["env"][GPU_BROKER_CHILD_TOKEN_ENV],
            "live-asr-token",
        )
        self.assertNotIn("ZH_ASR_GPU_BROKER_LEASE_HELD", captured["env"])
        self.assertTrue(captured["env"][PROCESS_TOKEN_ENV].startswith("chineseasr-cli-"))
        self.assertIn(PROCESS_TOKEN_ENV, captured["env"]["WSLENV"].split(":"))
        self.assertEqual(
            captured["command"][:3],
            [sys.executable, "-m", "zh_asr"],
        )

    def test_gpu_cli_supervisor_terminates_worker_immediately_on_lease_loss(self):
        from zh_asr.__main__ import _supervise_gpu_cli
        from zh_asr.gpu_broker import GpuBrokerLeaseLost

        state = {}

        class Lease:
            token = "live-asr-token"

            def __init__(self, _owner):
                self.error = None
                state["lease"] = self

            def set_on_lost(self, callback):
                self.callback = callback

            def __enter__(self):
                return self

            def raise_if_lost(self):
                if self.error is not None:
                    raise self.error

            def __exit__(self, *_args):
                return None

        class Process:
            pid = 1234
            returncode = None

            def wait(self):
                lease = state["lease"]
                lease.error = GpuBrokerLeaseLost("lease expired")
                lease.callback(lease.error)
                self.returncode = -1
                return self.returncode

            def poll(self):
                return self.returncode

        terminated = []
        with patch("zh_asr.__main__.GpuBrokerLease", Lease), patch(
            "zh_asr.__main__.subprocess.Popen",
            return_value=Process(),
        ), patch(
            "zh_asr.__main__.terminate_process_tree",
            side_effect=lambda process: terminated.append(process.pid),
        ), patch("zh_asr.__main__.terminate_wsl_processes") as cleanup_wsl:
            with self.assertRaises(GpuBrokerLeaseLost):
                with patch(
                    "zh_asr.__main__._cli_wsl_distributions",
                    return_value=("Ubuntu",),
                ):
                    _supervise_gpu_cli(["transcribe", "note.wav"])

        self.assertEqual(terminated, [1234])
        self.assertEqual(cleanup_wsl.call_count, 2)
        self.assertEqual(cleanup_wsl.call_args.args[0], ("Ubuntu",))
        self.assertTrue(cleanup_wsl.call_args.args[1].startswith("chineseasr-cli-"))

    def test_gpu_cli_supervisor_terminates_worker_before_releasing_on_interrupt(self):
        from zh_asr.__main__ import _supervise_gpu_cli

        events = []

        class Lease:
            token = "live-asr-token"

            def __init__(self, _owner):
                pass

            def set_on_lost(self, _callback):
                pass

            def __enter__(self):
                events.append("lease_enter")
                return self

            def raise_if_lost(self):
                return None

            def __exit__(self, *_args):
                events.append("lease_release")
                return None

        class Process:
            pid = 1234

            def wait(self):
                events.append("wait")
                raise KeyboardInterrupt()

            def poll(self):
                return None

        with patch("zh_asr.__main__.GpuBrokerLease", Lease), patch(
            "zh_asr.__main__.subprocess.Popen",
            return_value=Process(),
        ), patch(
            "zh_asr.__main__.terminate_process_tree",
            side_effect=lambda _process: events.append("terminate"),
        ):
            with self.assertRaises(KeyboardInterrupt):
                _supervise_gpu_cli(["transcribe", "note.wav"])

        self.assertEqual(
            events,
            ["lease_enter", "wait", "terminate", "lease_release"],
        )

    def test_cli_wsl_cleanup_scope_follows_selected_engine_runtime(self):
        from zh_asr.__main__ import _cli_wsl_distributions

        self.assertEqual(
            ("Ubuntu",),
            _cli_wsl_distributions(
                [
                    "strict",
                    "sample.wav",
                    "--primary-engine",
                    "fireredasr2-llm",
                    "--secondary-engine",
                    "qwen3-asr-1.7b",
                ]
            ),
        )
        self.assertEqual(
            (),
            _cli_wsl_distributions(
                [
                    "strict",
                    "sample.wav",
                    "--primary-engine",
                    "qwen3-asr-1.7b",
                    "--secondary-engine",
                    "sensevoice",
                ]
            ),
        )

    def run_cli(self, *args: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": PYTHONPATH,
                "PYTHONUTF8": "1",
            }
        )
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            [sys.executable, "-m", "zh_asr", *args],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_doctor_runs_without_funasr_dependency(self):
        result = self.run_cli("doctor")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Default engine: sensevoice", result.stdout)
        self.assertIn("Strict engines: qwen3-asr-1.7b, sensevoice", result.stdout)
        self.assertIn("Proxy variables: clean", result.stdout)
        self.assertIn("Qwen ASR installed:", result.stdout)

    def test_transcribe_missing_audio_fails_clearly_before_model_load(self):
        result = self.run_cli("transcribe", "missing.wav", "--device", "cpu")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Audio file not found", result.stderr)

    def test_transcribe_cli_passes_explicit_paraformer_preset_speaker_count_without_loading_model(self):
        from zh_asr.__main__ import main

        paths = {
            "markdown": Path("out.md"),
            "json": Path("out.raw.json"),
            "objective_outcome": "speech_transcribed",
        }
        with patch("zh_asr.__main__.transcribe_audio", return_value=paths) as transcribe:
            returncode = main(
                [
                    "transcribe",
                    "not-read.wav",
                    "--engine",
                    "paraformer",
                    "--preset-spk-num",
                    "2",
                    "--device",
                    "cpu",
                ]
            )

        self.assertEqual(returncode, 0)
        self.assertEqual(transcribe.call_args.kwargs["engine"], "paraformer")
        self.assertEqual(transcribe.call_args.kwargs["preset_spk_num"], 2)

    def test_preset_speaker_count_is_not_accepted_by_default_engine(self):
        result = self.run_cli(
            "transcribe",
            "not-read.wav",
            "--preset-spk-num",
            "2",
            "--device",
            "cpu",
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("only by explicit paraformer", result.stderr)

    def test_strict_missing_audio_fails_clearly_before_model_load(self):
        result = self.run_cli("strict", "missing.wav", "--device", "cpu")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Audio file not found", result.stderr)

    def test_long_missing_audio_fails_clearly_before_model_load(self):
        result = self.run_cli("long", "missing.wav", "--device", "cpu")

        self.assertEqual(result.returncode, 2)
        self.assertIn("Audio file not found", result.stderr)

    def test_batch_empty_folder_writes_summary_without_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()

            result = self.run_cli(
                "batch",
                str(input_dir),
                "--mode",
                "quick",
                "--device",
                "cpu",
                "--out-dir",
                str(output_dir),
            )
            summary = (output_dir / "summary.md").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Summary:", result.stdout)
        self.assertIn("Total: 0", result.stdout)
        self.assertIn("Total: 0", summary)

    def test_batch_missing_folder_fails_clearly_before_model_load(self):
        result = self.run_cli("batch", "missing-folder", "--device", "cpu")

        self.assertEqual(result.returncode, 2)
        self.assertIn("Input directory not found", result.stderr)

    def test_eval_generate_only_no_tts_creates_manifest_without_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus_dir = root / "corpus"
            out_dir = root / "runs"

            result = self.run_cli(
                "eval",
                "--generate",
                "--generate-only",
                "--no-tts",
                "--corpus-dir",
                str(corpus_dir),
                "--out-dir",
                str(out_dir),
            )
            manifest = (corpus_dir / "manifest.json").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Corpus manifest:", result.stdout)
        self.assertIn("silence-001", manifest)

    def test_benchmark_missing_audio_dir_fails_clearly_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth_dir = root / "truth"
            truth_dir.mkdir()

            result = self.run_cli(
                "benchmark",
                "--audio-dir",
                str(root / "missing-audio"),
                "--truth-dir",
                str(truth_dir),
                "--device",
                "cpu",
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Audio directory not found", result.stderr)

    def test_warmup_mentions_dependency_when_funasr_is_not_installed(self):
        result = self.run_cli(
            "warmup",
            "--engine",
            "sensevoice",
            "--device",
            "cpu",
            extra_env={"ZH_ASR_TEST_FORCE_MISSING_FUNASR": "1"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FunASR is not installed", result.stderr)

    def test_warmup_mentions_qwen_setup_when_qwen_asr_is_not_installed(self):
        result = self.run_cli(
            "warmup",
            "--engine",
            "qwen3-asr-1.7b",
            "--device",
            "cpu",
            extra_env={"ZH_ASR_TEST_FORCE_MISSING_QWEN_ASR": "1"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Qwen ASR is not installed", result.stderr)
        self.assertIn("scripts\\setup-qwen.ps1", result.stderr)

    def test_cli_reads_engine_choices_from_model_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "models.yaml"
            config.write_text(
                """
defaults:
  engine: custom-primary
strict:
  primary_engine: custom-primary
  secondary_engine: custom-secondary
aliases: {}
engines:
  custom-primary:
    adapter: funasr
    role: primary
    model: iic/CustomPrimary
    language: zh
  custom-secondary:
    adapter: funasr
    role: baseline
    model: iic/CustomSecondary
    language: zh
""",
                encoding="utf-8",
            )

            doctor = self.run_cli("doctor", extra_env={"ZH_ASR_MODEL_CONFIG": str(config)})
            strict = self.run_cli(
                "strict",
                "missing.wav",
                "--primary-engine",
                "custom-primary",
                "--secondary-engine",
                "custom-secondary",
                "--device",
                "cpu",
                extra_env={"ZH_ASR_MODEL_CONFIG": str(config)},
            )

        self.assertEqual(doctor.returncode, 0, doctor.stderr)
        self.assertIn("Default engine: custom-primary", doctor.stdout)
        self.assertIn("Available engines: custom-primary, custom-secondary", doctor.stdout)
        self.assertNotEqual(strict.returncode, 0)
        self.assertIn("Audio file not found", strict.stderr)
        self.assertNotIn("invalid choice", strict.stderr)

    def test_serve_check_registers_local_api_command_without_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli("serve", "--check", "--port", "0", "--state-dir", tmp)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ASR API ready", result.stdout)
        self.assertIn("127.0.0.1", result.stdout)

    def test_serve_check_uses_non_localocr_default_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli("serve", "--check", "--state-dir", tmp)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("http://127.0.0.1:18666", result.stdout)

    def test_attribute_speakers_writes_projection_without_loading_a_model(self):
        fixture_root = PROJECT_ROOT / "tests" / "fixtures" / "speaker_attribution"
        transcript = fixture_root / "mono_unknown_transcript.json"
        with tempfile.TemporaryDirectory() as tmp:
            context = Path(tmp) / "context.json"
            output = Path(tmp) / "attribution.json"
            context.write_text(
                json.dumps(
                    {
                        "schema": "chinese-asr.speaker-attribution-context.v2",
                        "recording_kind": "mono_call",
                        "segment_evidence": [
                            {
                                "index": 0,
                                "dialogue_role": {
                                    "candidate_role": "self",
                                    "reason": "该句回答了本人正在处理的设备故障。",
                                },
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            result = self.run_cli(
                "attribute-speakers",
                str(transcript),
                "--context",
                str(context),
                "--out",
                str(output),
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            transcript_hash = hashlib.sha256(transcript.read_bytes()).hexdigest()
            context_hash = hashlib.sha256(context.read_bytes()).hexdigest()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Speaker attribution gap: False", result.stdout)
        self.assertEqual(payload["segments"][0]["candidate_role"], "self")
        self.assertEqual(payload["input_binding"]["hash_kind"], "file_bytes")
        self.assertEqual(payload["input_binding"]["transcript_json_sha256"], transcript_hash)
        self.assertEqual(payload["input_binding"]["context_json_sha256"], context_hash)
        self.assertEqual(payload["input_binding"]["voice_evidence_json_sha256"], [])
        self.assertEqual(
            payload["segments"][0]["raw_json_pointer"],
            "$[0].sentence_info[0]",
        )

    def test_speaker_commands_are_discoverable_without_loading_a_model(self):
        enroll = self.run_cli("speaker-enroll", "--help")
        evidence = self.run_cli("speaker-evidence", "--help")
        readback = self.run_cli("speaker-evidence-readback", "--help")
        delete = self.run_cli("speaker-profile-delete", "--help")

        self.assertEqual(enroll.returncode, 0, enroll.stderr)
        self.assertIn("person:self", enroll.stdout)
        self.assertIn("--inference-basis", enroll.stdout)
        self.assertIn("--references", enroll.stdout)
        self.assertEqual(evidence.returncode, 0, evidence.stderr)
        self.assertIn("--channel", evidence.stdout)
        self.assertIn("--require-held-out", evidence.stdout)
        self.assertEqual(readback.returncode, 0, readback.stderr)
        self.assertIn("--evidence", readback.stdout)
        self.assertIn("target_media", readback.stdout)
        self.assertEqual(delete.returncode, 0, delete.stderr)
        self.assertIn("--confirm-delete", delete.stdout)

    def test_speaker_evidence_readback_returns_unknown_json_without_writing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "named.aac"
            target.write_bytes(b"named-media")
            missing_profile = root / "missing-profile.json"
            missing_evidence = root / "missing.voice-evidence.json"
            before = sorted(path.name for path in root.iterdir())
            result = self.run_cli(
                "speaker-evidence-readback",
                str(target),
                "--profile",
                str(missing_profile),
                "--evidence",
                str(missing_evidence),
            )
            after = sorted(path.name for path in root.iterdir())

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["identity_status"], "unknown")
        self.assertEqual(payload["current_valid_evidence_count"], 0)
        self.assertEqual(payload["profile_status"], "missing")
        self.assertEqual(before, after)
        self.assertNotIn("score", result.stdout)
        self.assertNotIn("embedding", result.stdout)


if __name__ == "__main__":
    unittest.main()

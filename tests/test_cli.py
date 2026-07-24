import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHONPATH = str(PROJECT_ROOT / "src")


class CliTests(unittest.TestCase):
    def run_cli(self, *args: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        env = {"PYTHONPATH": PYTHONPATH}
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
        result = self.run_cli("transcribe", "missing.wav")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Audio file not found", result.stderr)

    def test_strict_missing_audio_fails_clearly_before_model_load(self):
        result = self.run_cli("strict", "missing.wav")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Audio file not found", result.stderr)

    def test_long_missing_audio_fails_clearly_before_model_load(self):
        result = self.run_cli("long", "missing.wav")

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
                "--out-dir",
                str(output_dir),
            )
            summary = (output_dir / "summary.md").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Summary:", result.stdout)
        self.assertIn("Total: 0", result.stdout)
        self.assertIn("Total: 0", summary)

    def test_batch_missing_folder_fails_clearly_before_model_load(self):
        result = self.run_cli("batch", "missing-folder")

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
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Audio directory not found", result.stderr)

    def test_warmup_mentions_dependency_when_funasr_is_not_installed(self):
        result = self.run_cli(
            "warmup",
            "--engine",
            "sensevoice",
            extra_env={"ZH_ASR_TEST_FORCE_MISSING_FUNASR": "1"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FunASR is not installed", result.stderr)

    def test_warmup_mentions_qwen_setup_when_qwen_asr_is_not_installed(self):
        result = self.run_cli(
            "warmup",
            "--engine",
            "qwen3-asr-1.7b",
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


if __name__ == "__main__":
    unittest.main()

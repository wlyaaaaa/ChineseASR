import subprocess
import sys
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
        self.assertIn("Proxy variables: clean", result.stdout)

    def test_transcribe_missing_audio_fails_clearly_before_model_load(self):
        result = self.run_cli("transcribe", "missing.wav")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Audio file not found", result.stderr)

    def test_warmup_mentions_dependency_when_funasr_is_not_installed(self):
        result = self.run_cli(
            "warmup",
            "--engine",
            "sensevoice",
            extra_env={"ZH_ASR_TEST_FORCE_MISSING_FUNASR": "1"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FunASR is not installed", result.stderr)


if __name__ == "__main__":
    unittest.main()

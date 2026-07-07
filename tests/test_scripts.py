import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScriptTests(unittest.TestCase):
    def test_setup_qwen_uses_no_proxy_and_local_requirements_file(self):
        script = (PROJECT_ROOT / "scripts" / "setup-qwen.ps1").read_text(encoding="utf-8")

        self.assertIn("Clear-ProxyEnv", script)
        self.assertIn("requirements-qwen.txt", script)
        self.assertIn("pypi.tuna.tsinghua.edu.cn", script)

    def test_download_models_prefetches_qwen_from_modelscope_local_dir(self):
        script = (PROJECT_ROOT / "scripts" / "download-models.ps1").read_text(encoding="utf-8")

        self.assertIn("Qwen/Qwen3-ASR-1.7B", script)
        self.assertIn("snapshot_download", script)
        self.assertIn("Qwen\\Qwen3-ASR-1.7B", script)
        self.assertIn("$LASTEXITCODE", script)

    def test_transcribe_folder_uses_no_proxy_batch_cli_and_force_flag(self):
        script = (PROJECT_ROOT / "scripts" / "transcribe-folder.ps1").read_text(encoding="utf-8")

        self.assertIn("Clear-ProxyEnv", script)
        self.assertIn("'batch', $InputDir", script)
        self.assertIn("--cache-dir", script)
        self.assertIn("[switch]$Force", script)
        self.assertIn("'--force'", script)
        self.assertIn("$LASTEXITCODE", script)

    def test_eval_script_uses_no_proxy_generate_flags_and_cache_dir(self):
        script = (PROJECT_ROOT / "scripts" / "eval.ps1").read_text(encoding="utf-8")

        self.assertIn("Clear-ProxyEnv", script)
        self.assertIn("'eval'", script)
        self.assertIn("[switch]$Generate", script)
        self.assertIn("'--generate'", script)
        self.assertIn("'--no-tts'", script)
        self.assertIn("--cache-dir", script)
        self.assertIn("$LASTEXITCODE", script)


if __name__ == "__main__":
    unittest.main()

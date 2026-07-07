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
        self.assertIn("modelscope download", script)
        self.assertIn("Qwen\\Qwen3-ASR-1.7B", script)


if __name__ == "__main__":
    unittest.main()

import hashlib
import os
import tempfile
import unittest
from pathlib import Path


class MetadataTests(unittest.TestCase):
    def test_file_metadata_hashes_size_and_missing_paths(self):
        from zh_asr.metadata import file_metadata

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("abc", encoding="utf-8")

            existing = file_metadata(path)
            missing = file_metadata(path.with_name("missing.txt"))
            empty = file_metadata(None)

        self.assertEqual(existing["sha256"], hashlib.sha256(b"abc").hexdigest())
        self.assertEqual(existing["size_bytes"], 3)
        self.assertEqual(missing["sha256"], "")
        self.assertEqual(missing["size_bytes"], 0)
        self.assertEqual(empty["sha256"], "")
        self.assertEqual(empty["size_bytes"], 0)

    def test_snapshot_model_config_records_selected_engine_specs_and_config_hash(self):
        from zh_asr.config import load_model_config
        from zh_asr.metadata import snapshot_model_config

        config = load_model_config()
        snapshot = snapshot_model_config(config, ("qwen3-asr-1.7b", "sensevoice"))

        self.assertEqual(snapshot["path"], str(config.path.resolve()))
        self.assertEqual(snapshot["sha256"], hashlib.sha256(config.path.read_bytes()).hexdigest())
        self.assertEqual(snapshot["strict_primary_engine"], "qwen3-asr-1.7b")
        self.assertEqual(snapshot["strict_secondary_engine"], "sensevoice")
        self.assertIn("qwen3-asr-1.7b", snapshot["selected_engines"])
        self.assertEqual(snapshot["selected_engines"]["sensevoice"]["model"], "iic/SenseVoiceSmall")
        self.assertEqual(snapshot["selected_engines"]["qwen3-asr-1.7b"]["options"]["dtype"], "bfloat16")

    def test_capture_invocation_and_runtime_info_are_non_fatal(self):
        from zh_asr.metadata import capture_invocation, runtime_info

        invocation = capture_invocation(["-m", "zh_asr", "benchmark"], wrapper="scripts\\benchmark.ps1")
        runtime = runtime_info("cuda:0")

        self.assertEqual(invocation["argv"], ["-m", "zh_asr", "benchmark"])
        self.assertIn("-m zh_asr benchmark", invocation["command_line"])
        self.assertEqual(invocation["wrapper"], "scripts\\benchmark.ps1")
        self.assertIn("python", invocation)
        self.assertEqual(runtime["device"], "cuda:0")
        self.assertIn("platform", runtime)
        self.assertIn("cuda_available", runtime)

    def test_capture_invocation_reads_wrapper_from_environment(self):
        from zh_asr.metadata import capture_invocation

        previous = os.environ.get("ZH_ASR_WRAPPER")
        os.environ["ZH_ASR_WRAPPER"] = "scripts\\eval.ps1"
        try:
            invocation = capture_invocation(["-m", "zh_asr", "eval"])
        finally:
            if previous is None:
                os.environ.pop("ZH_ASR_WRAPPER", None)
            else:
                os.environ["ZH_ASR_WRAPPER"] = previous

        self.assertEqual(invocation["wrapper"], "scripts\\eval.ps1")


if __name__ == "__main__":
    unittest.main()

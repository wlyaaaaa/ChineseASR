from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import wave


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "asr-professional-cloud.ps1"


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 1_600)


class ProfessionalCloudScriptTests(unittest.TestCase):
    def test_cloud_failure_contract_is_bounded_and_recommends_local_smart(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("Get-SafeBrokerErrorCode", source)
        self.assertIn("runtime_rebind_required", source)
        self.assertIn("retry_once_after_runtime_rebind", source)
        self.assertIn("retry_cloud_once_if_still_authorized", source)
        self.assertIn("use_asr_smart_local", source)
        self.assertNotIn("Start-Sleep", source)

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "pwsh",
                "-NoProfile",
                "-File",
                str(SCRIPT),
                *arguments,
                "-Json",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

    def test_nonimportant_call_is_blocked_before_broker_or_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "ordinary.wav"
            queue = Path(tmp) / "queue"
            _write_wav(audio)
            result = self._run("-Audio", str(audio), "-RequestRoot", str(queue))

            self.assertEqual(2, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("blocked", payload["status"])
            self.assertEqual("importance_required", payload["error_code"])
            self.assertFalse(payload["cloud_upload_performed"])
            self.assertEqual("do_not_retry", payload["cloud_retry_policy"])
            self.assertEqual("none", payload["local_fallback_recommendation"])
            self.assertFalse(queue.exists())

    def test_cloud_authorization_is_separate_from_importance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "important.wav"
            queue = Path(tmp) / "queue"
            _write_wav(audio)
            result = self._run(
                "-Audio",
                str(audio),
                "-RequestRoot",
                str(queue),
                "-Important",
            )

            self.assertEqual(2, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("blocked", payload["status"])
            self.assertEqual(
                "cloud_upload_authorization_required", payload["error_code"]
            )
            self.assertFalse(payload["cloud_upload_performed"])
            self.assertEqual("do_not_retry", payload["cloud_retry_policy"])
            self.assertEqual("none", payload["local_fallback_recommendation"])
            self.assertFalse(queue.exists())


if __name__ == "__main__":
    unittest.main()

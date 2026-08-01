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
            self.assertFalse(queue.exists())


if __name__ == "__main__":
    unittest.main()

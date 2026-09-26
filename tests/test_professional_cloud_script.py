from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import uuid
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
    def test_json_stdout_contains_only_final_receipt_and_broker_receipts_are_logged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            audio = root / "recording.wav"
            _write_wav(audio)
            broker = scripts / "fake-broker.ps1"
            request_root = root / "outputs" / "cloud-jobs"
            broker.write_text("""param([string]$Action, [string]$Query, [string]$RuntimePrincipal,
    [string]$ResultCode, [string]$OperationId, [switch]$Json)
$root = '%s'
if ($Action -eq 'AgentSecretRef') {
    [IO.File]::WriteAllText((Join-Path $root ($Query + '.provider.json')), '{}')
    [Console]::Out.WriteLine('{"schema":"pcconfig.secret-broker-result.v1","status":"ok","action":"AgentSecretRef"}')
    exit 0
}
[Console]::Out.WriteLine('{"schema":"pcconfig.secret-broker-result.v1","status":"ok","action":"ReportCredentialResult"}')
exit 0
""" % str(request_root).replace("'", "''"), encoding="utf-8")
            pipeline = scripts / "cloud-review-pipeline.py"
            pipeline.write_text("""import json
from pathlib import Path
import sys

action = sys.argv[1]
root = Path(sys.argv[sys.argv.index('--root') + 1])
intent = json.loads(Path(sys.argv[sys.argv.index('--intent') + 1]).read_text(encoding='utf-8'))
job_id = intent['job_id']
if action == 'prepare':
    (root / (job_id + '.pending.json')).write_text('{}', encoding='utf-8')
    print(json.dumps({'status': 'ready', 'secret_ref_target': job_id,
                      'credential_ref': 'fixture'}))
else:
    result = {'schema': 'chineseasr.qwen-audio3-quality-review-result.v1',
              'job_id': job_id, 'purpose': 'quality_review',
              'important_only': False, 'status': 'succeeded', 'error_code': '',
              'credential_result': 'Success', 'cloud_upload_performed': True,
              'text': 'fixture transcript'}
    (root / (job_id + '.result.json')).write_text(json.dumps(result), encoding='utf-8')
    print(json.dumps({'status': 'succeeded'}))
""", encoding="utf-8")
            source = SCRIPT.read_text(encoding="utf-8")
            original = "$brokerPath = 'C:\\ProgramData\\PCConfig\\AuthorityHost\\tools\\Invoke-SecretBroker.ps1'"
            self.assertIn(original, source)
            relocated = scripts / SCRIPT.name
            relocated.write_text(source.replace(original,
                "$brokerPath = '" + str(broker).replace("'", "''") + "'").replace(
                "Global\\ChineseASRCloudUploadOnce",
                "Global\\ChineseASRCloudUploadTest" + uuid.uuid4().hex), encoding="utf-8")
            result = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-File",
                str(relocated), "-Audio", str(audio), "-QualityReview",
                "-CloudUploadAuthorized", "-Json"], cwd=root, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.assertEqual(0, result.returncode, result.stderr)
            decoder = json.JSONDecoder()
            receipt, end = decoder.raw_decode(result.stdout.strip())
            self.assertEqual(len(result.stdout.strip()), end)
            self.assertEqual("succeeded", receipt["status"])
            log_path = Path(receipt["broker_log_path"])
            self.assertTrue(log_path.is_file())
            log = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual("AgentSecretRef", log["agent_secret_ref_receipt"]["action"])
            self.assertEqual("ReportCredentialResult",
                log["credential_result_report_receipt"]["action"])

    def test_relocated_script_uses_its_own_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            relocated = scripts / SCRIPT.name
            relocated.write_bytes(SCRIPT.read_bytes())
            result = subprocess.run(
                ["pwsh", "-NoProfile", "-File", str(relocated),
                 "-Audio", str(root / "missing.wav"), "-Important",
                 "-CloudUploadAuthorized", "-Json"],
                cwd=root, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=30,
            )
            self.assertEqual(2, result.returncode, result.stderr)
            self.assertEqual("audio_file_missing", json.loads(result.stdout)["error_code"])
            self.assertFalse((root / "outputs" / "cloud-jobs").exists())

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

    def test_unlabelled_call_is_blocked_before_broker_or_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "ordinary.wav"
            queue = Path(tmp) / "queue"
            _write_wav(audio)
            result = self._run("-Audio", str(audio), "-RequestRoot", str(queue))

            self.assertEqual(2, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("blocked", payload["status"])
            self.assertEqual("cloud_use_purpose_required", payload["error_code"])
            self.assertFalse(payload["cloud_upload_performed"])
            self.assertEqual("do_not_retry", payload["cloud_retry_policy"])
            self.assertEqual("none", payload["local_fallback_recommendation"])
            self.assertFalse(queue.exists())

    def test_quality_review_is_explicit_and_not_labeled_important(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "ordinary.wav"
            queue = Path(tmp) / "queue"
            _write_wav(audio)
            result = self._run(
                "-Audio",
                str(audio),
                "-RequestRoot",
                str(queue),
                "-QualityReview",
            )

            self.assertEqual(2, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("blocked", payload["status"])
            self.assertEqual(
                "cloud_upload_authorization_required", payload["error_code"]
            )
            self.assertEqual(
                "chineseasr.qwen-audio3-quality-review-result.v1",
                payload["schema"],
            )
            self.assertEqual("quality_review", payload["purpose"])
            self.assertFalse(payload["important_only"])
            self.assertFalse(payload["cloud_upload_performed"])
            self.assertFalse(queue.exists())

    def test_important_and_quality_review_cannot_be_combined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "ambiguous.wav"
            queue = Path(tmp) / "queue"
            _write_wav(audio)
            result = self._run(
                "-Audio",
                str(audio),
                "-RequestRoot",
                str(queue),
                "-Important",
                "-QualityReview",
                "-CloudUploadAuthorized",
            )

            self.assertEqual(2, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("blocked", payload["status"])
            self.assertEqual("cloud_use_purpose_ambiguous", payload["error_code"])
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
            self.assertEqual(
                "chineseasr.qwen-audio3-important-result.v1", payload["schema"]
            )
            self.assertEqual("important_evidence", payload["purpose"])
            self.assertTrue(payload["important_only"])
            self.assertFalse(payload["cloud_upload_performed"])
            self.assertEqual("do_not_retry", payload["cloud_retry_policy"])
            self.assertEqual("none", payload["local_fallback_recommendation"])
            self.assertFalse(queue.exists())

    def test_automatic_review_requires_existing_local_review_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "uncertain.wav"
            _write_wav(audio)
            result = self._run("-Audio", str(audio), "-QualityReview", "-AutomaticReview")
            self.assertEqual(2, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("local_review_required", payload["error_code"])
            self.assertFalse(payload["cloud_upload_performed"])


if __name__ == "__main__":
    unittest.main()

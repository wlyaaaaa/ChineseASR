from __future__ import annotations

import hashlib
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
        self._assert_cloud_receipt(relative_paths=False, first_failure=True)

    def test_relative_paths_follow_powershell_location(self) -> None:
        self._assert_cloud_receipt(relative_paths=True)

    def test_background_credential_report_denial_keeps_success_and_never_retries_upload(self) -> None:
        self._assert_cloud_receipt(relative_paths=True, report_denied=True)

    def _assert_cloud_receipt(self, *, relative_paths: bool, first_failure: bool = False,
                              report_denied: bool = False) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            caller = root / "调用目录 中文 [输入]"
            caller.mkdir()
            audio = (caller if relative_paths else root) / "recording.wav"
            _write_wav(audio)
            local_out_dir = caller / "本地结果 [输出]"
            local_out_dir.mkdir()
            broker = scripts / "fake-broker.ps1"
            request_root = root / "outputs" / "cloud-jobs"
            broker.write_text("""param([string]$Action, [string]$Query,
    [string]$ResultCode, [string]$OperationId, [string]$SecretRefRequestPath, [switch]$Json)
$root = '%s'
if ($Action -eq 'AgentSecretRef') {
    if (Test-Path (Join-Path $root 'deny.marker')) {
        [Console]::Out.WriteLine('{"schema":"pcconfig.secret-broker-result.v1","status":"error","error":"runtime_rebind_required"}')
        exit 3
    }
    $provider = Join-Path $root ($Query + '.provider.json')
    $body = '{}'
    if (Test-Path (Join-Path $root 'network.marker')) {
        $code = if (Test-Path -LiteralPath $provider) { 'previous_outcome_unknown' } else { 'network_failure' }
        $body = @{ status = 'failed'; error_code = $code } | ConvertTo-Json -Compress
    }
    [IO.File]::WriteAllText($provider, $body)
    [Console]::Out.WriteLine('{"schema":"pcconfig.secret-broker-result.v1","status":"ok","action":"AgentSecretRef"}')
    exit 0
}
if (Test-Path (Join-Path $root 'report-denied.marker')) {
    [Console]::Out.WriteLine('{"schema":"pcconfig.secret-broker-result.v1","status":"error","error":"trusted_desktop_ancestor_required","action":"ReportCredentialResult"}')
    exit 3
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
    if '--resume-job-id' in sys.argv:
        job_id = sys.argv[sys.argv.index('--resume-job-id') + 1]
    (root / (job_id + '.pending.json')).write_text('{}', encoding='utf-8')
    print(json.dumps({'status': 'ready', 'secret_ref_target': job_id,
                      'job_id': job_id, 'intent_path': str(root / (job_id + '.intent.json')),
                      'request_path': str(root / (job_id + '.pending.json')),
                      'credential_ref': 'fixture'}))
else:
    result = {'schema': 'chineseasr.qwen-audio3-quality-review-result.v1',
              'job_id': job_id, 'purpose': 'quality_review',
              'important_only': False, 'status': 'succeeded', 'error_code': '',
              'credential_result': 'Success', 'cloud_upload_performed': True,
              'text': 'fixture transcript'}
    if '--broker-error' in sys.argv:
        result.update(status='failed', error_code=sys.argv[sys.argv.index('--broker-error') + 1])
    elif (root / 'network.marker').exists():
        result.update(status='failed', credential_result='Network-Failure',
                      error_code=json.loads((root / (job_id + '.provider.json')).read_text())['error_code'])
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
            arguments = [str(relocated), "-Audio", str(audio),
                "-LocalOutDir", str(local_out_dir), "-RequestRoot", str(request_root),
                "-QualityReview", "-CloudUploadAuthorized", "-Json"]
            if relative_paths:
                arguments[2] = audio.name
                arguments[4] = local_out_dir.name
                arguments[6] = r"..\outputs\cloud-jobs"
                command = "Set-Location -LiteralPath '" + str(caller).replace("'", "''")
                command += "'; & " + " ".join(
                    value if value.startswith("-") else "'" + value.replace("'", "''") + "'"
                    for value in arguments)
                invocation = ["-Command", command]
            else:
                invocation = ["-File", *arguments]
            if first_failure:
                request_root.mkdir(parents=True)
                (request_root / "network.marker").touch()
            if report_denied:
                request_root.mkdir(parents=True, exist_ok=True)
                (request_root / "report-denied.marker").touch()
            result = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", *invocation],
                cwd=root, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.assertEqual(3 if first_failure else 0, result.returncode, result.stdout + result.stderr)
            decoder = json.JSONDecoder()
            receipt, end = decoder.raw_decode(result.stdout.strip())
            self.assertEqual(len(result.stdout.strip()), end)
            self.assertEqual("failed" if first_failure else "succeeded", receipt["status"])
            log_path = Path(receipt["broker_log_path"])
            self.assertTrue(log_path.is_file())
            self.assertEqual(request_root, log_path.parent)
            intent_path = log_path.with_name(log_path.name.replace(".broker.json", ".intent.json"))
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            self.assertEqual(audio, Path(intent["audio_path"]))
            self.assertEqual(local_out_dir, Path(intent["local_out_dir"]))
            log = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual("AgentSecretRef", log["agent_secret_ref_receipt"]["action"])
            self.assertEqual("ReportCredentialResult",
                log["credential_result_report_receipt"]["action"])
            if report_denied:
                self.assertEqual("report-failed", receipt["credential_report_status"])
                self.assertEqual("trusted_desktop_ancestor_required", log["credential_result_report_receipt"]["error"])
                self.assertEqual("not_needed", receipt["cloud_retry_policy"])
                self.assertEqual("none", receipt["local_fallback_recommendation"])
            if not relative_paths:
                self.assertEqual("network_failure", receipt["error_code"])
                attempts = request_root / (receipt["job_id"] + ".attempts")
                self.assertFalse(attempts.exists())
                previous = {kind: (request_root / (receipt["job_id"] + f".{kind}.json")).read_bytes()
                            for kind in ("provider", "broker", "result")}
                retry = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-File",
                    *arguments, "-ResumeJobId", receipt["job_id"]], cwd=root,
                    capture_output=True, text=True, encoding="utf-8", timeout=60,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.assertEqual(3, retry.returncode, retry.stdout + retry.stderr)
                retry_receipt, end = decoder.raw_decode(retry.stdout.strip())
                self.assertEqual(len(retry.stdout.strip()), end)
                self.assertEqual("previous_outcome_unknown", retry_receipt["error_code"])
                self.assertEqual(1, len(list(attempts.iterdir())))
                index_path = Path(retry_receipt["previous_attempt_path"])
                index = json.loads(index_path.read_text(encoding="utf-8"))
                self.assertEqual(index_path.parent.name, index["attempt_id"])
                self.assertEqual(receipt["job_id"], index["job_id"])
                self.assertEqual(3, len(index["receipts"]))
                for entry in index["receipts"]:
                    kind = Path(entry["source_path"]).name.split(".")[-2]
                    self.assertEqual(previous[kind], Path(entry["path"]).read_bytes())
                    self.assertEqual(hashlib.sha256(previous[kind]).hexdigest(), entry["sha256"])
                archived = json.loads((index_path.parent / (receipt["job_id"] + ".provider.json")).read_text())
                self.assertEqual("network_failure", archived["error_code"])
                (request_root / 'deny.marker').touch()
                denied = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-File",
                    *arguments, "-ResumeJobId", receipt["job_id"]], cwd=root,
                    capture_output=True, text=True, encoding="utf-8", timeout=60,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.assertEqual(3, denied.returncode, denied.stdout + denied.stderr)
                denied_receipt = json.loads(denied.stdout)
                self.assertEqual("runtime_rebind_required", denied_receipt["error_code"])
                self.assertEqual(2, len(list(attempts.iterdir())))
                blocked_archive = Path(denied_receipt["previous_attempt_path"]).parent
                blocked = json.loads((blocked_archive / (receipt["job_id"] + ".provider.json")).read_text())
                self.assertEqual("previous_outcome_unknown", blocked["error_code"])

    def test_success_report_is_skipped_only_while_credential_is_known_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            audio = root / "voice.wav"
            _write_wav(audio)
            request_root = root / "outputs" / "cloud-jobs"
            broker = scripts / "fake-broker.ps1"
            broker.write_text("""param([string]$Action, [string]$Query,
    [string]$ResultCode, [string]$OperationId, [string]$SecretRefRequestPath, [switch]$Json)
$root = '%s'
if ($Action -eq 'AgentSecretRef') {
    [IO.File]::WriteAllText((Join-Path $root ($Query + '.provider.json')), '{}')
    [Console]::Out.WriteLine('{"status":"ok","action":"AgentSecretRef"}')
    exit 0
}
Add-Content -LiteralPath (Join-Path $root 'reports.log') -Value $ResultCode
[Console]::Out.WriteLine('{"status":"pass","credential_update_required":false}')
""" % str(request_root).replace("'", "''"), encoding="utf-8")
            (scripts / "cloud-review-pipeline.py").write_text("""import json
from pathlib import Path
import sys

root = Path(sys.argv[sys.argv.index('--root') + 1])
intent_path = sys.argv[sys.argv.index('--intent') + 1]
job_id = json.loads(Path(intent_path).read_text(encoding='utf-8'))['job_id']
if sys.argv[1] == 'prepare':
    (root / (job_id + '.pending.json')).write_text('{}', encoding='utf-8')
    print(json.dumps({'status': 'ready', 'secret_ref_target': job_id, 'job_id': job_id,
                      'intent_path': intent_path, 'request_path': str(root / (job_id + '.pending.json')),
                      'credential_ref': 'fixture'}))
else:
    failing = (root / 'fail.marker').exists()
    (root / (job_id + '.result.json')).write_text(json.dumps({
        'schema': 'chineseasr.qwen-audio3-transcription-result.v1', 'job_id': job_id,
        'purpose': 'transcription', 'important_only': False,
        'status': 'failed' if failing else 'succeeded', 'error_code': '',
        'credential_result': 'Network-Failure' if failing else 'Success',
        'cloud_upload_performed': True, 'text': 'fixture'}), encoding='utf-8')
    print('{}')
""", encoding="utf-8")
            source = SCRIPT.read_text(encoding="utf-8")
            original = "$brokerPath = 'C:\\ProgramData\\PCConfig\\AuthorityHost\\tools\\Invoke-SecretBroker.ps1'"
            relocated = scripts / SCRIPT.name
            relocated.write_text(source.replace(original,
                "$brokerPath = '" + str(broker).replace("'", "''") + "'"), encoding="utf-8")

            def run() -> str:
                result = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-File", str(relocated),
                    "-Audio", str(audio), "-CloudFirst", "-Json"], cwd=root, capture_output=True,
                    text=True, encoding="utf-8", errors="replace", timeout=60,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                return json.loads(result.stdout)["credential_report_status"]

            reports = request_root / "reports.log"
            self.assertEqual("reported", run())
            self.assertEqual("skipped_no_state_change", run())
            self.assertEqual(["Success"], reports.read_text().split())

            (request_root / "fail.marker").touch()
            self.assertEqual("reported", run())
            (request_root / "fail.marker").unlink()
            self.assertEqual("reported", run())
            self.assertEqual(["Success", "Network-Failure", "Success"], reports.read_text().split())

            health = request_root / "credential-health.json"
            state = json.loads(health.read_text(encoding="utf-8"))
            state["verified_utc"] = "2020-01-01T00:00:00+00:00"
            health.write_text(json.dumps(state), encoding="utf-8")
            self.assertEqual("reported", run())

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

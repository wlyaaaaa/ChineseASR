from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
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
    @unittest.skipUnless(os.name == "nt", "Windows named mutex")
    def test_direct_call_waits_past_five_seconds_but_explicit_short_wait_is_bounded(self) -> None:
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        kernel.CreateMutexW.restype = wintypes.HANDLE
        kernel.ReleaseMutex.argtypes = kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            audio, marker = root / "voice.wav", root / "at-mutex"
            _write_wav(audio)
            prefix = "Global\\ChineseASRCloudWaitTest" + uuid.uuid4().hex
            source = SCRIPT.read_text(encoding="utf-8").replace("Global\\ChineseASRCloudUploadOnce", prefix)
            source = source.replace("$mutexHeld = $false", "$mutexHeld = $false\n[IO.File]::WriteAllText('" +
                                    str(marker).replace("'", "''") + "', '')")
            relocated = scripts / SCRIPT.name
            relocated.write_text(source, encoding="utf-8")
            metadata = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-Command",
                "$t=$null;$e=$null;$a=[Management.Automation.Language.Parser]::ParseFile('" +
                str(relocated).replace("'", "''") + "',[ref]$t,[ref]$e); "
                "($a.ParamBlock.Parameters | Where-Object { $_.Name.VariablePath.UserPath -ceq 'ConcurrentWaitSec' }).DefaultValue.SafeGetValue()"],
                capture_output=True, text=True, encoding="utf-8", timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.assertEqual(metadata.returncode, 0, metadata.stderr)
            self.assertEqual(int(metadata.stdout.strip()), 24 * 3600)
            (scripts / "cloud-review-pipeline.py").write_text(
                'import sys; print(\'{"status":"pass"}\' if sys.argv[1]=="check-access" else '
                '\'{"status":"blocked","error_code":"fixture_prepare_reached"}\')', encoding="utf-8")
            handle = kernel.CreateMutexW(None, True, prefix + "-" + hashlib.sha256(audio.read_bytes()).hexdigest().upper())
            self.assertTrue(handle)
            held, process = True, None
            command = ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(relocated),
                       "-Audio", str(audio), "-CloudFirst", "-Json"]
            try:
                short = subprocess.run([*command, "-ConcurrentWaitSec", "0"], cwd=root,
                    capture_output=True, text=True, encoding="utf-8", timeout=15,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.assertEqual(json.loads(short.stdout)["error_code"], "cloud_upload_busy")
                self.assertFalse((root / "outputs").exists())
                marker.unlink()
                process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                deadline = time.monotonic() + 10
                while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue(marker.exists())
                time.sleep(5.3)
                self.assertIsNone(process.poll(), "direct caller stopped waiting after five seconds")
                self.assertTrue(kernel.ReleaseMutex(handle)); held = False
                stdout, stderr = process.communicate(timeout=15)
                self.assertEqual(json.loads(stdout)["error_code"], "fixture_prepare_reached", stderr)
            finally:
                if held: kernel.ReleaseMutex(handle)
                if process is not None and process.poll() is None:
                    process.kill(); process.communicate(timeout=5)
                kernel.CloseHandle(handle)

    def test_manual_call_uses_transmit_and_stdout_contains_only_final_receipt(self) -> None:
        self._assert_cloud_receipt(relative_paths=False)

    def test_guarded_relative_paths_follow_powershell_location(self) -> None:
        self._assert_cloud_receipt(relative_paths=True, personal_data_guard=True)

    def test_structured_transport_failure_reaches_finalize(self) -> None:
        self._assert_cloud_receipt(relative_paths=False, transport_failure=True)

    def test_resume_archives_previous_provider_and_result_before_transmitting(self) -> None:
        self._assert_cloud_receipt(relative_paths=False, first_failure=True)

    def test_unknown_candidate_result_never_recommends_cloud_retry(self) -> None:
        self._assert_cloud_receipt(relative_paths=False, uncertain_failure=True)

    def _assert_cloud_receipt(self, *, relative_paths: bool, personal_data_guard: bool = False,
                              transport_failure: bool = False, first_failure: bool = False,
                              uncertain_failure: bool = False) -> None:
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
            request_root = root / "outputs" / "cloud-jobs"
            (scripts / "cloud-review-pipeline.py").write_text("""import json
from pathlib import Path
import sys

if sys.argv[1] == 'check-access':
    print(json.dumps({'status': 'pass'}))
    sys.exit(0)
args = sys.argv[2:]
def value(flag):
    return args[args.index(flag) + 1]
action = sys.argv[1]
root = Path(value('--root'))
with (root / 'pipeline-calls.jsonl').open('a', encoding='utf-8') as calls:
    calls.write(json.dumps({'action': action, 'args': args}) + '\\n')
intent_path = Path(value('--intent'))
intent = json.loads(intent_path.read_text(encoding='utf-8'))
job_id = intent['job_id']
if action == 'prepare':
    if '--resume-job-id' in args:
        job_id = value('--resume-job-id')
    request = root / (job_id + '.pending.json')
    request.write_text('{}', encoding='utf-8')
    print(json.dumps({'status': 'ready', 'job_id': job_id,
                      'intent_path': str(root / (job_id + '.intent.json')),
                      'request_path': str(request)}))
elif action == 'transmit':
    if (root / 'transport-failure.marker').exists():
        print(json.dumps({'schema': 'zh_asr.chunk-transmission.v1', 'status': 'blocked',
                          'secret_returned': False, 'error_code': 'personal_data_locked',
                          'new_requests': 0, 'reused_requests': 0}))
        sys.exit(3)
    provider = root / (job_id + '.provider.json')
    code = ''
    if (root / 'network.marker').exists():
        code = 'previous_outcome_unknown' if provider.exists() else 'network_failure'
    if (root / 'unknown.marker').exists():
        code = 'cloud_key_refresh_cloud_transport_failed'
    provider.write_text(json.dumps({'status': 'failed' if code else 'succeeded',
                                    'error_code': code}), encoding='utf-8')
    print(json.dumps({'schema': 'zh_asr.chunk-transmission.v1', 'status': 'failed' if code else 'succeeded',
                      'secret_returned': False, 'error_code': 'ignored_when_provider_updated' if code else '',
                      'new_requests': 1, 'reused_requests': 0}))
    sys.exit(3 if code else 0)
elif action == 'finalize':
    code = value('--broker-error') if '--broker-error' in args else json.loads(
        Path(value('--provider')).read_text(encoding='utf-8'))['error_code']
    result = {'schema': 'chineseasr.qwen-audio3-quality-review-result.v1',
              'job_id': job_id, 'purpose': 'quality_review', 'important_only': False,
              'status': 'failed' if code else 'succeeded', 'error_code': code,
              'credential_result': 'Network-Failure' if code in ('network_failure', 'cloud_key_refresh_cloud_transport_failed') else 'Success',
              'outcome_unknown': code == 'cloud_key_refresh_cloud_transport_failed',
              'cloud_upload_performed': code != 'personal_data_locked', 'text': 'fixture transcript'}
    (root / (job_id + '.result.json')).write_text(json.dumps(result), encoding='utf-8')
    print(json.dumps({'status': 'succeeded'}))
else:
    raise AssertionError(action)
""", encoding="utf-8")
            relocated = scripts / SCRIPT.name
            relocated.write_text(SCRIPT.read_text(encoding="utf-8").replace(
                "Global\\ChineseASRCloudUploadOnce",
                "Global\\ChineseASRCloudUploadTest" + uuid.uuid4().hex), encoding="utf-8")
            arguments = [str(relocated), "-Audio", str(audio),
                "-LocalOutDir", str(local_out_dir), "-RequestRoot", str(request_root),
                "-QualityReview", "-CloudUploadAuthorized", "-Json"]
            if personal_data_guard:
                arguments.append("-PersonalDataGuard")
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
            if first_failure or transport_failure:
                request_root.mkdir(parents=True)
                (request_root / ("network.marker" if first_failure else "transport-failure.marker")).touch()
            if uncertain_failure:
                request_root.mkdir(parents=True)
                (request_root / 'unknown.marker').touch()

            def run(extra: list[str] | None = None) -> dict:
                result = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", *invocation, *(extra or [])],
                    cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.assertEqual(3 if first_failure or transport_failure or uncertain_failure else 0,
                                 result.returncode, result.stdout + result.stderr)
                decoder = json.JSONDecoder()
                receipt, end = decoder.raw_decode(result.stdout.strip())
                self.assertEqual(len(result.stdout.strip()), end, result.stdout)
                self.assertNotIn("broker_log_path", receipt)
                self.assertNotIn("credential_report_status", receipt)
                return receipt

            receipt = run()
            self.assertEqual("failed" if first_failure or transport_failure or uncertain_failure else "succeeded", receipt["status"])
            result_path = Path(receipt["result_path"])
            self.assertTrue(result_path.is_file())
            self.assertEqual(request_root, result_path.parent)
            intent_path = request_root / (receipt["job_id"] + ".intent.json")
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            self.assertEqual(audio, Path(intent["audio_path"]))
            self.assertEqual(local_out_dir, Path(intent["local_out_dir"]))
            calls_path = request_root / "pipeline-calls.jsonl"
            calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(["prepare", "transmit", "finalize"], [call["action"] for call in calls])
            transmit_args = calls[1]["args"]
            self.assertEqual(personal_data_guard, "--personal-data-guard" in transmit_args)
            self.assertNotIn("--broker", transmit_args)
            self.assertNotIn("--target", transmit_args)
            self.assertEqual(str(intent_path), transmit_args[transmit_args.index("--intent") + 1])
            request_path = Path(transmit_args[transmit_args.index("--request") + 1])
            self.assertFalse(request_path.exists())
            self.assertTrue(request_path.with_name(receipt["job_id"] + ".unclaimed.json").is_file())
            self.assertFalse((request_root / "credential-health.json").exists())
            self.assertFalse(list(request_root.glob("*.broker.json")))
            if uncertain_failure:
                self.assertTrue(receipt['outcome_unknown'])
                self.assertEqual(receipt['cloud_retry_policy'], 'do_not_retry_cloud')
            elif transport_failure:
                self.assertEqual("personal_data_locked", receipt["error_code"])
                final_args = calls[-1]["args"]
                self.assertEqual("personal_data_locked", final_args[final_args.index("--broker-error") + 1])
                self.assertNotIn("--provider", final_args)
                self.assertFalse(receipt["cloud_upload_performed"])
                self.assertEqual("do_not_retry_cloud", receipt["cloud_retry_policy"])
                self.assertEqual("use_asr_smart_local", receipt["local_fallback_recommendation"])
            elif first_failure:
                self.assertEqual("network_failure", receipt["error_code"])
                self.assertIn("--provider", calls[-1]["args"])
                self.assertNotIn("--broker-error", calls[-1]["args"])
                self.assertEqual("retry_cloud_once_if_still_authorized", receipt["cloud_retry_policy"])
                self.assertEqual("use_asr_smart_local", receipt["local_fallback_recommendation"])
                attempts = request_root / (receipt["job_id"] + ".attempts")
                self.assertFalse(attempts.exists())
                previous = {kind: (request_root / (receipt["job_id"] + f".{kind}.json")).read_bytes()
                            for kind in ("provider", "result")}
                retry_receipt = run(["-ResumeJobId", receipt["job_id"]])
                self.assertEqual(receipt["job_id"], retry_receipt["job_id"])
                self.assertEqual("previous_outcome_unknown", retry_receipt["error_code"])
                self.assertEqual("do_not_retry_cloud", retry_receipt["cloud_retry_policy"])
                self.assertEqual(1, len(list(attempts.iterdir())))
                index_path = Path(retry_receipt["previous_attempt_path"])
                index = json.loads(index_path.read_text(encoding="utf-8"))
                self.assertEqual(index_path.parent.name, index["attempt_id"])
                self.assertEqual(receipt["job_id"], index["job_id"])
                self.assertEqual(2, len(index["receipts"]))
                for entry in index["receipts"]:
                    kind = Path(entry["source_path"]).name.split(".")[-2]
                    self.assertEqual(previous[kind], Path(entry["path"]).read_bytes())
                    self.assertEqual(hashlib.sha256(previous[kind]).hexdigest(), entry["sha256"])
                calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(["prepare", "transmit", "finalize"] * 2, [call["action"] for call in calls])
                stale = {kind: (request_root / (receipt["job_id"] + f".{kind}.json")).read_bytes()
                         for kind in ("provider", "result")}
                (request_root / "transport-failure.marker").touch()
                blocked_receipt = run(["-ResumeJobId", receipt["job_id"]])
                self.assertEqual("personal_data_locked", blocked_receipt["error_code"])
                self.assertEqual(2, len(list(attempts.iterdir())))
                blocked_archive = Path(blocked_receipt["previous_attempt_path"]).parent
                for kind, previous_bytes in stale.items():
                    self.assertEqual(previous_bytes,
                                     (blocked_archive / (receipt["job_id"] + f".{kind}.json")).read_bytes())
                calls = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(["prepare", "transmit", "finalize"] * 3, [call["action"] for call in calls])
                final_args = calls[-1]["args"]
                self.assertEqual("personal_data_locked", final_args[final_args.index("--broker-error") + 1])
                self.assertNotIn("--provider", final_args)
            else:
                self.assertEqual("not_needed", receipt["cloud_retry_policy"])
                self.assertEqual("none", receipt["local_fallback_recommendation"])

    def test_relocated_script_uses_its_own_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / "scripts"
            scripts.mkdir()
            relocated = scripts / SCRIPT.name
            relocated.write_bytes(SCRIPT.read_bytes())
            (scripts / "cloud-review-pipeline.py").write_text('print(\'{"status":"pass"}\')', encoding="utf-8")
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

    def test_unlabelled_call_is_blocked_before_transport_or_queue(self) -> None:
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

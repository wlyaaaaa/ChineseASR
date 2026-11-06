from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import uuid
import wave

from zh_asr.cloud_review import (BEIJING, CloudReviewError, finalize_cloud_result,
                                prepare_cloud_request)
from zh_asr.cloud_file_transport import request_digest, transmit_cloud_chunks

CONFIG = Path(__file__).resolve().parents[1] / "configs/models.yaml"


class CloudChunkTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.root = self.base / "outputs/cloud-jobs"
        self.root.mkdir(parents=True)
        self.audio = self.base / "synthetic.wav"
        with wave.open(str(self.audio), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\0\0" * 48000)
        self.intent = self.root / (str(uuid.uuid4()) + ".intent.json")
        self.intent.write_text(json.dumps(dict(schema="zh_asr.cloud_review_intent.v1",
            job_id=self.intent.name.removesuffix(".intent.json"), audio_path=str(self.audio),
            purpose="transcription", cloud_first=True, automatic_review=True,
            cloud_upload_authorized=True, model_profile="short", chunk_sec=1, overlap_sec=0)), encoding="utf-8")
        self.prepared = prepare_cloud_request(self.intent, self.root, CONFIG,
                                              now=datetime(2026, 11, 6, tzinfo=BEIJING))
        self.request = Path(self.prepared["request_path"])
        self.sent = []

    def send(self, path, target, broker):
        self.assertEqual(target, "qwen-audio3-asr-important-once")
        part = json.loads(path.read_text("utf-8"))
        self.assertEqual(len(part["operations"]), 1)
        ident = part["operations"][0]["id"]
        self.sent.append(ident)
        Path(part["result_path"]).write_text(json.dumps(dict(schema="passwordcenter.vendor-result.v1",
            vendor="qwen", request_sha256=request_digest(part), status="succeeded", credential_result="Success",
            new_requests=1, results=[dict(id=ident, http_status=200,
                raw_response={"output": {"text": "合成片段" + ident}})])), encoding="utf-8")
        return ""

    def run_chunks(self, access):
        return transmit_cloud_chunks(self.request, self.intent, self.root, Path("fixture-broker.ps1"),
                                     target="qwen-audio3-asr-important-once", access=access, send=self.send)

    def test_lock_between_chunks_and_resume_without_resending_success(self):
        first = self.run_chunks(mock.Mock(side_effect=[True, False]))
        self.assertEqual((first["status"], first["error_code"], self.sent),
                         ("blocked", "personal_access_protection_required", ["1"]))
        partial = finalize_cloud_result(self.intent, self.root,
            provider_path=self.root / (self.prepared["job_id"] + ".provider.json"), config_path=CONFIG)
        self.assertEqual((partial["status"], len(partial["chunks"])), ("blocked", 1))
        again = self.run_chunks(lambda: False)
        self.assertEqual(again["error_code"], "personal_access_protection_required")
        self.assertEqual(self.sent, ["1"])
        completed = self.run_chunks(lambda: True)
        self.assertEqual((completed["status"], completed["reused_requests"], self.sent),
                         ("succeeded", 1, ["1", "2", "3"]))
        final = finalize_cloud_result(self.intent, self.root,
            provider_path=self.root / (self.prepared["job_id"] + ".provider.json"), config_path=CONFIG)
        self.assertEqual((final["status"], len(final["chunks"])), ("succeeded", 3))

    def test_unknown_chunk_checkpoint_never_calls_broker_again(self):
        request = json.loads(self.request.read_text("utf-8"))
        directory = self.root / (self.prepared["job_id"] + ".work")
        marker = directory / "part-1.provider.json.checkpoints/1.json"
        marker.parent.mkdir(parents=True)
        for status in ("in_flight", "outcome_unknown"):
            marker.write_text(json.dumps(dict(status=status)), encoding="utf-8")
            result = self.run_chunks(lambda: True)
            self.assertEqual((result["status"], result["error_code"]), ("blocked", "previous_outcome_unknown"))
        self.assertEqual(self.sent, [])
        self.assertEqual(len(request["operations"]), 3)

    def test_original_request_mutation_is_rejected_before_access_or_transport(self):
        data = json.loads(self.request.read_text("utf-8")); data["model"] = "wrong"
        self.request.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(CloudReviewError, "vendor_request_binding_invalid"):
            self.run_chunks(mock.Mock(side_effect=AssertionError("no privacy or credential call")))
        self.assertEqual(self.sent, [])

    def test_chunk_timeout_is_visible_and_preserves_completed_prefix(self):
        def timeout(path, target, broker):
            if self.sent:
                return "cloud_chunk_timeout"
            return self.send(path, target, broker)
        result = transmit_cloud_chunks(self.request, self.intent, self.root, Path("fixture.ps1"),
                                       target="qwen-audio3-asr-important-once", access=lambda: True, send=timeout)
        self.assertEqual((result["status"], result["error_code"], len(result["results"])),
                         ("failed", "cloud_chunk_timeout", 1))

    def test_real_powershell_and_pipeline_pause_and_resume_same_job(self):
        project = Path(__file__).resolve().parents[1]
        scripts = self.base / "scripts"
        scripts.mkdir()
        (self.base / "configs").mkdir()
        import yaml
        config = yaml.safe_load(CONFIG.read_text("utf-8"))
        for model in config["cloud_review"]["models"].values():
            model["free_until"] = "2099-01-01T00:00:00+08:00"
        (self.base / "configs/models.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        # Relocate the real script/pipeline; only transport and privacy state are fixtures.
        pipeline = (project / "scripts/cloud-review-pipeline.py").read_text("utf-8")
        patch = ("from zh_asr import file_job\n"
                 "file_job._access_open = lambda: not (ROOT / 'locked.marker').exists()\n")
        pipeline = pipeline.replace("def main(argv:", patch + "\ndef main(argv:")
        (scripts / "cloud-review-pipeline.py").write_text(pipeline, encoding="utf-8")
        provider = scripts / "fixture-provider.py"
        provider.write_text("""import json,sys
from pathlib import Path
from zh_asr.cloud_file_transport import request_digest
p=Path(sys.argv[1]); r=json.loads(p.read_text('utf-8')); i=r['operations'][0]['id']
root=Path(sys.argv[2]); counter=root/'sent.log'
with counter.open('a',encoding='utf-8') as f: f.write(i+'\\n')
Path(r['result_path']).write_text(json.dumps(dict(schema='passwordcenter.vendor-result.v1',vendor='qwen',
 request_sha256=request_digest(r),status='succeeded',credential_result='Success',new_requests=1,
 results=[dict(id=i,http_status=200,raw_response={'output':{'text':'fixture '+i}})])),encoding='utf-8')
if i=='1': (root/'locked.marker').touch()
""", encoding="utf-8")
        broker = scripts / "fixture-broker.ps1"
        broker.write_text("""param($Action,$Query,$SecretRefRequestPath,$ResultCode,$OperationId,[switch]$Json)
if ($Action -eq 'AgentSecretRef') {
 & '%s' '%s' $SecretRefRequestPath '%s'
 [Console]::Out.WriteLine('{"schema":"pcconfig.secret-broker-result.v1","status":"ok"}')
 exit 0
}
[Console]::Out.WriteLine('{"schema":"pcconfig.secret-broker-result.v1","status":"error","error":"trusted_desktop_ancestor_required"}')
exit 3
""" % tuple(str(value).replace("'", "''") for value in (sys.executable, provider, self.base)), encoding="utf-8")
        source = (project / "scripts/asr-professional-cloud.ps1").read_text("utf-8")
        source = source.replace("$brokerPath = 'C:\\ProgramData\\PCConfig\\AuthorityHost\\tools\\Invoke-SecretBroker.ps1'",
                                "$brokerPath = '" + str(broker).replace("'", "''") + "'")
        source = source.replace("$python = Join-Path $PSScriptRoot '..\\.venv\\Scripts\\python.exe'",
                                "$python = '" + sys.executable.replace("'", "''") + "'")
        source = source.replace("Global\\ChineseASRCloudUploadOnce", "Global\\ASRCloudFixture" + uuid.uuid4().hex)
        script = scripts / "asr-professional-cloud.ps1"
        script.write_text(source, encoding="utf-8")
        command = ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(script), "-Audio", str(self.audio),
                   "-CloudFirst", "-PersonalDataGuard", "-ChunkSec", "1", "-OverlapSec", "0", "-Json"]
        first = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=60,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(first.returncode, 3, first.stdout + first.stderr)
        receipt = json.loads(first.stdout)
        self.assertEqual((receipt["status"], receipt["error_code"], len(receipt["chunks"])),
                         ("blocked", "personal_access_protection_required", 1))
        (self.base / "locked.marker").unlink()
        resumed = subprocess.run(command + ["-ResumeJobId", receipt["job_id"]], capture_output=True,
                                 text=True, encoding="utf-8", timeout=60,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        final = json.loads(resumed.stdout)
        self.assertEqual((final["status"], final["job_id"], len(final["chunks"])),
                         ("succeeded", receipt["job_id"], 3))
        self.assertEqual((self.base / "sent.log").read_text().split(), ["1", "2", "3"])
        self.assertEqual((final["credential_report_status"], final["cloud_retry_policy"]),
                         ("report-failed", "not_needed"))
        log = json.loads(Path(final["broker_log_path"]).read_text("utf-8"))
        self.assertNotIn("results", log["agent_secret_ref_receipt"])


if __name__ == "__main__":
    unittest.main()

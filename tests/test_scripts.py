import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScriptTests(unittest.TestCase):
    def test_no_proxy_command_restores_environment_without_extra_output(self):
        pwsh = shutil.which("pwsh")
        if not pwsh:
            self.skipTest("PowerShell is required for proxy command testing")
        with tempfile.TemporaryDirectory(prefix="no-proxy-command-") as directory:
            root = Path(directory)
            shutil.copy2(PROJECT_ROOT / "scripts" / "Invoke-NoProxy.ps1", root)
            (root / "child.ps1").write_text("""
param([int]$Code, [string]$NoProxy)
foreach ($name in 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy') {
  if (Test-Path -LiteralPath "Env:$name") { throw 'Proxy inherited' }
}
foreach ($name in 'NO_PROXY', 'no_proxy') {
  if ([Environment]::GetEnvironmentVariable($name, 'Process') -cne $NoProxy) { throw 'Direct bypass changed' }
}
[Console]::WriteLine('fixture-output')
exit $Code
""", encoding="utf-8-sig")
            harness = root / "run.ps1"
            harness.write_text("""
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1')
$pwsh = (Get-Process -Id $PID).Path
$child = Join-Path $PSScriptRoot 'child.ps1'
$keys = 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'NO_PROXY', 'no_proxy'
foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, "fixture-$key", 'Process') }
Remove-Item -LiteralPath Env:ALL_PROXY
[Environment]::SetEnvironmentVariable('HTTPS_PROXY', '', 'Process')
$before = @($keys | ForEach-Object { [pscustomobject]@{ name = $_; value = [Environment]::GetEnvironmentVariable($_, 'Process'); present = Test-Path -LiteralPath "Env:$_" } })
$rows = foreach ($code in '0', '7', 'missing') {
  $output = @()
  $caught = $false
  try {
    if ($code -eq 'missing') { Invoke-NoProxyCommand -FilePath 'fixture-missing-command-7893' }
    else { $output = @(Invoke-NoProxyCommand -FilePath $pwsh -ArgumentList @('-NoProfile', '-File', $child, '-Code', $code, '-NoProxy', $script:NoProxyValue)) }
  } catch { $caught = $true }
  $after = @($keys | ForEach-Object { [pscustomobject]@{ name = $_; value = [Environment]::GetEnvironmentVariable($_, 'Process'); present = Test-Path -LiteralPath "Env:$_" } })
  [pscustomobject]@{ case = $code; output = $output; exit = $LASTEXITCODE; caught = $caught; after = $after }
}
$cliOutput = @(& $pwsh -NoProfile -File (Join-Path $PSScriptRoot 'Invoke-NoProxy.ps1') $pwsh -NoProfile -File $child -Code 7 -NoProxy $script:NoProxyValue)
[pscustomobject]@{ before = $before; rows = @($rows); cliExit = $LASTEXITCODE; cliOutput = $cliOutput } | ConvertTo-Json -Depth 5 -Compress
""", encoding="utf-8-sig")
            result = subprocess.run([pwsh, "-NoProfile", "-File", str(harness)],
                                    capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(len(data["before"]), 8)
            self.assertTrue(any(not entry["present"] for entry in data["before"]))
            self.assertTrue(any(entry["present"] and entry["value"] == "" for entry in data["before"]))
            for row in data["rows"]:
                with self.subTest(case=row["case"]):
                    self.assertEqual(row["after"], data["before"])
                    self.assertEqual(row["caught"], row["case"] == "missing")
                    self.assertEqual(row["output"], [] if row["caught"] else ["fixture-output"])
                    if not row["caught"]:
                        self.assertEqual(row["exit"], int(row["case"]))
            self.assertEqual(data["cliExit"], 7)
            self.assertEqual(data["cliOutput"], ["fixture-output"])

    def test_windows_setup_stops_on_failure_and_restores_proxy_environment(self):
        pwsh = shutil.which("pwsh")
        if not pwsh:
            self.skipTest("PowerShell is required for setup behavior testing")
        cases = (
            ("setup-qwen.ps1", (7, 0, 0), 1, 1),
            ("setup-qwen.ps1", (0, 0, 7), 3, 7),
            ("setup-qwen.ps1", (0, 0, 0), 3, 0),
            ("install-torch-cu128-direct.ps1", (0, 7, 0), 2, 1),
            ("install-torch-cu128-direct.ps1", (0, 0, 7), 3, 7),
            ("install-torch-cu128-direct.ps1", (0, 0, 0), 3, 0),
            ("setup-core.ps1", (7, 0, 0, 0, 0, 0), 6, 0),
        )
        with tempfile.TemporaryDirectory(prefix="windows-setup-") as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            python = root / ".venv" / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.touch()
            for name in {case[0] for case in cases} | {"Invoke-NoProxy.ps1"}:
                shutil.copy2(PROJECT_ROOT / "scripts" / name, root / "scripts" / name)
            runner = root / "runner.ps1"
            runner.write_text("""
param([string]$ScriptName, [string]$Codes, [string]$OutFile)
$ErrorActionPreference = 'Stop'
$keys = 'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy','NO_PROXY','no_proxy'
foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, "fixture-$key", 'Process') }
Remove-Item -LiteralPath Env:ALL_PROXY
[Environment]::SetEnvironmentVariable('HTTPS_PROXY', '', 'Process')
$before = @($keys | ForEach-Object { [pscustomobject]@{ name = $_; value = [Environment]::GetEnvironmentVariable($_,'Process'); present = Test-Path -LiteralPath "Env:$_" } })
$global:FixtureCalls = [System.Collections.Generic.List[object]]::new()
$global:FixturePlan = @($Codes -split ',' | ForEach-Object { [int]$_ })
$python = Join-Path $PSScriptRoot '.venv\\Scripts\\python.exe'
Set-Item -LiteralPath ('Function:global:' + $python) -Value { $global:FixtureCalls.Add(@($args)); $global:LASTEXITCODE = $global:FixturePlan[$global:FixtureCalls.Count - 1] }
$code = 0
try { & (Join-Path $PSScriptRoot "scripts\\$ScriptName"); $code = $LASTEXITCODE } catch { $code = 1 }
$after = @($keys | ForEach-Object { [pscustomobject]@{ name = $_; value = [Environment]::GetEnvironmentVariable($_,'Process'); present = Test-Path -LiteralPath "Env:$_" } })
[pscustomobject]@{ calls = @($global:FixtureCalls.ToArray()); before = $before; after = $after } | ConvertTo-Json -Depth 6 -Compress | Set-Content -LiteralPath $OutFile -Encoding utf8
exit $code
""", encoding="utf-8-sig")
            for index, (name, plan, count, code) in enumerate(cases):
                with self.subTest(script=name, plan=plan):
                    log = root / f"case-{index}.json"
                    result = subprocess.run(
                        [pwsh, "-NoProfile", "-File", str(runner), "-ScriptName", name,
                         "-Codes", ",".join(map(str, plan)), "-OutFile", str(log)],
                        capture_output=True, text=True, encoding="utf-8", timeout=30,
                    )
                    self.assertEqual(result.returncode, code, result.stderr)
                    data = json.loads(log.read_text(encoding="utf-8-sig"))
                    self.assertEqual(len(data["calls"]), count)
                    self.assertEqual(data["after"], data["before"])
                    self.assertEqual(len(data["before"]), 8)
                    self.assertTrue(any(not item["present"] for item in data["before"]))
                    self.assertTrue(any(item["present"] and item["value"] == "" for item in data["before"]))
                    if name == "setup-core.ps1":
                        upgrade = ["-m", "pip", "install", "--upgrade", "pip"]
                        mirror = ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple", "--trusted-host", "pypi.tuna.tsinghua.edu.cn"]
                        self.assertEqual(data["calls"], [
                            upgrade + mirror, upgrade + ["-i", "https://pypi.org/simple"],
                            ["-m", "pip", "install", "-r", str(root / "requirements-core.txt")] + mirror,
                            ["-m", "pip", "install", "-e", str(root), "--no-deps"],
                            ["-m", "pip", "check"], ["-m", "zh_asr", "doctor"],
                        ])
                    else:
                        prefixes = (["-m", "pip", "install", "-r"], ["-m", "pip", "install", "-e"], ["-m", "zh_asr", "doctor"]) if name == "setup-qwen.ps1" else (["-m", "pip", "install", "--upgrade"], ["-m", "pip", "install", "torch"], ["-c"])
                        for call, prefix in zip(data["calls"], prefixes):
                            self.assertEqual(call[:len(prefix)], prefix)

    def test_setup_core_falls_back_to_official_pypi_and_checks_environment(self):
        script = (PROJECT_ROOT / "scripts" / "setup-core.ps1").read_text(encoding="utf-8")

        self.assertIn("pypi.tuna.tsinghua.edu.cn", script)
        self.assertIn("https://pypi.org/simple", script)
        self.assertIn("@('-m', 'pip', 'check')", script)
        self.assertIn("$LASTEXITCODE", script)

    def test_setup_qwen_uses_no_proxy_and_local_requirements_file(self):
        script = (PROJECT_ROOT / "scripts" / "setup-qwen.ps1").read_text(encoding="utf-8")

        self.assertIn("Invoke-NoProxyCommand", script)
        self.assertIn("requirements-qwen.txt", script)
        self.assertIn("pypi.tuna.tsinghua.edu.cn", script)

    def test_download_models_prefetches_qwen_from_modelscope_local_dir(self):
        script = (PROJECT_ROOT / "scripts" / "download-models.ps1").read_text(encoding="utf-8")

        self.assertIn("Qwen/Qwen3-ASR-1.7B", script)
        self.assertIn("snapshot_download", script)
        self.assertIn("Qwen\\Qwen3-ASR-1.7B", script)
        self.assertIn("$LASTEXITCODE", script)

    def test_download_models_prefetches_firered_without_windows_warmup(self):
        script = (PROJECT_ROOT / "scripts" / "download-models.ps1").read_text(encoding="utf-8")

        self.assertIn("FireRedTeam/FireRedASR2-LLM", script)
        self.assertIn("FireRedRevision", script)
        self.assertIn("revision=os.environ['ZH_ASR_FIRERED_REVISION']", script)
        self.assertIn("models\\firered\\FireRedASR2-LLM", script)
        self.assertIn("MODEL_RECEIPT.json", script)
        self.assertIn("Get-Sha256Hex", script)
        self.assertIn("[System.Security.Cryptography.SHA256]::Create()", script)
        self.assertIn("[System.IO.FileOptions]::SequentialScan", script)
        self.assertIn("'Qwen2-7B-Instruct/model-00001-of-00004.safetensors'", script)
        self.assertIn("Select-Object -Unique", script)
        self.assertIn("ConvertTo-Json -Depth 5 -Compress", script)
        self.assertIn("Move-Item -LiteralPath $ReceiptTempPath", script)
        self.assertIn("if ($Engine -eq 'fireredasr2-llm')", script)
        self.assertIn("exit 0", script)

    def test_setup_firered_matches_registry_layout_and_pins_runtime(self):
        script = (PROJECT_ROOT / "scripts" / "setup-firered.ps1").read_text(encoding="utf-8")

        self.assertIn("/opt/chineseasr/firered", script)
        self.assertIn("models\\firered\\FireRedASR2S", script)
        self.assertIn("4e7d9aaf4482a47cec1724807026b9b151926eb5", script)
        self.assertIn("2.10.0+cu128", script)
        self.assertIn('VENV_DIR="`$INSTALL_ROOT/.venv"', script)
        self.assertIn("checkout --detach FETCH_HEAD", script)
        self.assertIn('PYTHONPATH="`$SOURCE_DIR"', script)
        self.assertNotIn("pip install --no-deps -e", script)

    def test_transcribe_folder_uses_no_proxy_batch_cli_and_force_flag(self):
        script = (PROJECT_ROOT / "scripts" / "transcribe-folder.ps1").read_text(encoding="utf-8")

        self.assertIn("Invoke-NoProxyCommand -FilePath $Python -ArgumentList $CliArgs", script)
        self.assertIn("'batch', $InputDir", script)
        self.assertIn("--cache-dir", script)
        self.assertIn("[switch]$Force", script)
        self.assertIn("'--force'", script)
        self.assertIn("$LASTEXITCODE", script)

    def test_eval_script_uses_no_proxy_generate_flags_and_cache_dir(self):
        script = (PROJECT_ROOT / "scripts" / "eval.ps1").read_text(encoding="utf-8")

        self.assertIn("Invoke-NoProxyCommand -FilePath $Python -ArgumentList $CliArgs", script)
        self.assertIn("'eval'", script)
        self.assertIn("[switch]$Generate", script)
        self.assertIn("'--generate'", script)
        self.assertIn("'--no-tts'", script)
        self.assertIn("--cache-dir", script)
        self.assertIn("$env:ZH_ASR_WRAPPER = 'scripts\\eval.ps1'", script)
        self.assertIn("$LASTEXITCODE", script)

    def test_benchmark_script_uses_no_proxy_and_required_dirs(self):
        script = (PROJECT_ROOT / "scripts" / "benchmark.ps1").read_text(encoding="utf-8")

        self.assertIn("Invoke-NoProxyCommand -FilePath $Python -ArgumentList $CliArgs", script)
        self.assertIn("'benchmark'", script)
        self.assertIn("[Parameter(Mandatory = $true)]", script)
        self.assertIn("--audio-dir", script)
        self.assertIn("--truth-dir", script)
        self.assertIn("--cache-dir", script)
        self.assertIn("$env:ZH_ASR_WRAPPER = 'scripts\\benchmark.ps1'", script)
        self.assertIn("$LASTEXITCODE", script)

    def test_offline_runtime_bundle_is_ignored_but_source_files_are_trackable(self):
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

        self.assertIn("offline/wheelhouse/", gitignore)
        self.assertIn("offline/current/", gitignore)
        self.assertIn("offline/staging-*/", gitignore)
        self.assertNotIn("offline/manifests/", gitignore)

    def test_export_lock_script_pins_both_installed_runtimes(self):
        script = (PROJECT_ROOT / "scripts" / "export-lock.ps1").read_text(encoding="utf-8")

        self.assertIn("pip list --format=freeze", script)
        self.assertIn("local-chinese-asr", script)
        self.assertIn("windows-requirements.txt", script)
        self.assertIn("linux-requirements.txt", script)
        self.assertIn("versions.json", script)

    def test_build_wheelhouse_script_prepares_both_platforms_as_wheels(self):
        script = (PROJECT_ROOT / "scripts" / "build-wheelhouse.ps1").read_text(encoding="utf-8")

        self.assertIn("download.pytorch.org/whl/cu128", script)
        self.assertIn("pip download", script)
        self.assertIn("pip wheel", script)
        self.assertIn("windows-requirements.txt", script)
        self.assertIn("linux-requirements.txt", script)
        self.assertIn("SendToRecycleBin", script)

    def test_verify_wheelhouse_script_uses_bundle_hash_verifier(self):
        script = (PROJECT_ROOT / "scripts" / "verify-wheelhouse.ps1").read_text(encoding="utf-8")

        self.assertIn("offline_bundle.py", script)
        self.assertIn("'verify'", script)
        self.assertIn("'seal'", script)
        self.assertIn("throw", script)

    def test_install_offline_script_verifies_then_installs_without_index(self):
        script = (PROJECT_ROOT / "scripts" / "install-offline.ps1").read_text(encoding="utf-8")

        self.assertIn("verify-wheelhouse.ps1", script)
        self.assertIn("--no-index", script)
        self.assertIn("--only-binary=:all:", script)
        self.assertIn("--find-links", script)
        self.assertIn("windows-requirements.txt", script)
        self.assertIn("linux-requirements.txt", script)
        self.assertIn("pip check", script)
        self.assertIn("$env:PYTHONPATH", script)
        self.assertIn("zh_asr doctor", script)

    def test_asr_smart_script_starts_local_api_submits_jobs_and_returns_status(self):
        script = (PROJECT_ROOT / "scripts" / "asr-smart.ps1").read_text(encoding="utf-8")

        self.assertIn("Clear-ProxyEnv", script)
        self.assertIn("[int]$WaitSec = 15", script)
        self.assertIn("[int]$StartupTimeoutSec = 30", script)
        self.assertIn("Start-Process", script)
        self.assertIn("-WindowStyle Hidden", script)
        self.assertIn("Invoke-RestMethod", script)
        self.assertIn("/jobs/transcribe", script)
        self.assertIn("/jobs/$($Submit.job.job_id)", script)
        self.assertIn("allow_gpu_conflicts", script)
        self.assertIn("[switch]$AllowGpuConflicts", script)
        self.assertIn("does not bypass LocalGpuBroker", script)
        self.assertIn("'long-strict'", script)
        self.assertIn("[int]$ChunkSec = 300", script)
        self.assertIn("[int]$OverlapSec = 1", script)
        self.assertIn("chunk_sec", script)
        self.assertIn("overlap_sec", script)
        self.assertIn("evidence_status = $FinalJob.evidence_status", script)
        self.assertIn("evidence_failures = $FinalJob.evidence_failures", script)
        self.assertIn("next_status_command", script)
        self.assertIn("ConvertTo-Json", script)
        self.assertIn("'serve'", script)
        self.assertIn("'--host'", script)
        self.assertIn("'--port'", script)
        self.assertIn("Assert-AsrPortBindable", script)
        self.assertIn("TcpListener", script)
        self.assertIn("excluded port ranges", script)
        self.assertIn("New-Item -ItemType Directory -Force -Path $OutRootPath", script)
        self.assertNotIn("Resolve-Path $OutRoot", script)

    def test_smoke_asr_smart_script_runs_strict_local_chinese_audio(self):
        script = (PROJECT_ROOT / "scripts" / "smoke-asr-smart.ps1").read_text(encoding="utf-8")

        self.assertIn("asr-smart.ps1", script)
        self.assertIn("-Mode strict", script)
        self.assertIn("-Force", script)
        self.assertIn("SenseVoiceSmall", script)
        self.assertIn("zh.mp3", script)
        self.assertIn("final", script)
        self.assertIn("audit", script)
        self.assertIn("secondary_raw_json", script)

    def test_smoke_wrappers_preserve_proxy_environment_and_literal_audio(self):
        pwsh = shutil.which("pwsh")
        if not pwsh:
            self.skipTest("PowerShell is required for wrapper behavior testing")
        with tempfile.TemporaryDirectory(prefix="smoke-wrappers-") as directory:
            root = Path(directory)
            for name in ("smoke-asr-smart.ps1", "smoke-evidence-asr.ps1", "Invoke-NoProxy.ps1"):
                shutil.copy2(PROJECT_ROOT / "scripts" / name, root / name)
            audio = root / "中文 录音[1].wav"
            audio.write_bytes(b"fixture")
            (root / "中文 录音1.wav").write_bytes(b"decoy")
            (root / "asr-smart.ps1").write_text(
                "param([string]$Audio)\n$global:CapturedAudio = $Audio\nthrow 'fixture-stop'\n",
                encoding="utf-8-sig",
            )
            harness = root / "run.ps1"
            harness.write_text("""
$keys = 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'
foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, "fixture-$key", 'Process') }
$before = @($keys | ForEach-Object { [Environment]::GetEnvironmentVariable($_, 'Process') })
$rows = foreach ($name in 'smoke-asr-smart.ps1', 'smoke-evidence-asr.ps1') {
  foreach ($inputName in 'missing.wav', '中文 录音[1].wav') {
    $global:CapturedAudio = $null
    $message = ''
    try { & (Join-Path $PSScriptRoot $name) -Audio (Join-Path $PSScriptRoot $inputName) | Out-Null }
    catch { $message = $_.Exception.Message }
    $after = @($keys | ForEach-Object { [Environment]::GetEnvironmentVariable($_, 'Process') })
    [pscustomobject]@{ wrapper = $name; input = $inputName; audio = $global:CapturedAudio; message = $message; proxies = $after }
  }
}
[pscustomobject]@{ before = $before; rows = @($rows) } | ConvertTo-Json -Depth 5 -Compress
""", encoding="utf-8-sig")
            result = subprocess.run([pwsh, "-NoProfile", "-File", str(harness)],
                                    capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(len(data["rows"]), 4)
            for row in data["rows"]:
                with self.subTest(wrapper=row["wrapper"], audio=row["input"]):
                    self.assertEqual(row["proxies"], data["before"])
                    if row["input"] == "missing.wav":
                        self.assertIsNone(row["audio"])
                        self.assertTrue(row["message"])
                        self.assertNotEqual(row["message"], "fixture-stop")
                    else:
                        self.assertEqual(row["audio"], str(audio))
                        self.assertEqual(row["message"], "fixture-stop")

    def test_evidence_smoke_uses_canonical_pair_and_verifies_every_chunk(self):
        script = (PROJECT_ROOT / "scripts" / "smoke-evidence-asr.ps1").read_text(
            encoding="utf-8"
        )

        self.assertIn("[Parameter(Mandatory = $true)]", script)
        self.assertIn("-Mode long-strict", script)
        self.assertIn("-PrimaryEngine fireredasr2-llm", script)
        self.assertIn("-SecondaryEngine qwen3-asr-1.7b", script)
        self.assertIn("evidence_status -ne 'verified'", script)
        self.assertIn("'receipt'", script)
        self.assertIn("engine_failure", script)
        self.assertIn("llm_initial_load_dtype", script)
        self.assertIn("portable bundle receipt reference", script)
        self.assertIn("IsPathRooted", script)
        self.assertIn("-Force", script)

    def test_strict_wrapper_routes_through_smart_service_and_gpu_broker(self):
        script = (PROJECT_ROOT / "scripts" / "strict.ps1").read_text(encoding="utf-8")

        self.assertIn("asr-smart.ps1", script)
        self.assertIn("-Mode", script)
        self.assertIn("'strict'", script)
        self.assertNotIn("'-m', 'zh_asr'", script)

    def test_public_wrappers_use_project_relative_default_output_paths(self):
        for name in ("strict.ps1", "eval.ps1", "benchmark.ps1", "transcribe-folder.ps1"):
            with self.subTest(script=name):
                script = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")

                self.assertNotIn("E:\\ChineseASR", script)
                self.assertIn("Join-Path $Root 'outputs", script)


if __name__ == "__main__":
    unittest.main()

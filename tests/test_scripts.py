import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScriptTests(unittest.TestCase):
    def test_setup_core_falls_back_to_official_pypi_and_checks_environment(self):
        script = (PROJECT_ROOT / "scripts" / "setup-core.ps1").read_text(encoding="utf-8")

        self.assertIn("pypi.tuna.tsinghua.edu.cn", script)
        self.assertIn("https://pypi.org/simple", script)
        self.assertIn("pip check", script)
        self.assertIn("$LASTEXITCODE", script)

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
        self.assertIn("$env:ZH_ASR_WRAPPER = 'scripts\\eval.ps1'", script)
        self.assertIn("$LASTEXITCODE", script)

    def test_benchmark_script_uses_no_proxy_and_required_dirs(self):
        script = (PROJECT_ROOT / "scripts" / "benchmark.ps1").read_text(encoding="utf-8")

        self.assertIn("Clear-ProxyEnv", script)
        self.assertIn("'benchmark'", script)
        self.assertIn("[Parameter(Mandatory = $true)]", script)
        self.assertIn("--audio-dir", script)
        self.assertIn("--truth-dir", script)
        self.assertIn("--cache-dir", script)
        self.assertIn("$env:ZH_ASR_WRAPPER = 'scripts\\benchmark.ps1'", script)
        self.assertIn("$LASTEXITCODE", script)

    def test_offline_wheelhouse_directory_is_ignored_but_manifests_are_trackable(self):
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

        self.assertIn("offline/wheelhouse/", gitignore)
        self.assertNotIn("offline/manifests/", gitignore)

    def test_export_lock_script_freezes_current_venv_without_editable_project(self):
        script = (PROJECT_ROOT / "scripts" / "export-lock.ps1").read_text(encoding="utf-8")

        self.assertIn("Clear-ProxyEnv", script)
        self.assertIn("requirements-lock.txt", script)
        self.assertIn("pip freeze", script)
        self.assertIn("--exclude-editable", script)
        self.assertIn("pip check", script)

    def test_build_wheelhouse_script_downloads_pinned_wheels_and_writes_checksums(self):
        script = (PROJECT_ROOT / "scripts" / "build-wheelhouse.ps1").read_text(encoding="utf-8")

        self.assertIn("Clear-ProxyEnv", script)
        self.assertIn("download.pytorch.org/whl/cu128", script)
        self.assertIn("pypi.tuna.tsinghua.edu.cn", script)
        self.assertIn("pip download", script)
        self.assertIn("requirements-lock.txt", script)
        self.assertIn("wheelhouse.sha256", script)
        self.assertIn("wheelhouse.json", script)
        self.assertIn("Get-FileHash", script)

    def test_verify_wheelhouse_script_fails_on_missing_or_mismatched_hashes(self):
        script = (PROJECT_ROOT / "scripts" / "verify-wheelhouse.ps1").read_text(encoding="utf-8")

        self.assertIn("wheelhouse.sha256", script)
        self.assertIn("Get-FileHash", script)
        self.assertIn("Missing wheelhouse file", script)
        self.assertIn("Checksum mismatch", script)
        self.assertIn("throw", script)

    def test_install_offline_script_verifies_then_installs_without_index(self):
        script = (PROJECT_ROOT / "scripts" / "install-offline.ps1").read_text(encoding="utf-8")

        self.assertIn("verify-wheelhouse.ps1", script)
        self.assertIn("--no-index", script)
        self.assertIn("--find-links", script)
        self.assertIn("requirements-lock.txt", script)
        self.assertIn("pip check", script)
        self.assertIn("pip install -e", script)
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

        self.assertIn("Clear-ProxyEnv", script)
        self.assertIn("asr-smart.ps1", script)
        self.assertIn("-Mode strict", script)
        self.assertIn("-Force", script)
        self.assertIn("SenseVoiceSmall", script)
        self.assertIn("zh.mp3", script)
        self.assertIn("final", script)
        self.assertIn("audit", script)
        self.assertIn("secondary_raw_json", script)

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

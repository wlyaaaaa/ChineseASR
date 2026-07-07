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
        self.assertIn("snapshot_download", script)
        self.assertIn("Qwen\\Qwen3-ASR-1.7B", script)
        self.assertIn("$LASTEXITCODE", script)

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
        self.assertIn("'long-strict'", script)
        self.assertIn("[int]$ChunkSec = 300", script)
        self.assertIn("[int]$OverlapSec = 1", script)
        self.assertIn("chunk_sec", script)
        self.assertIn("overlap_sec", script)
        self.assertIn("next_status_command", script)
        self.assertIn("ConvertTo-Json", script)
        self.assertIn("'serve'", script)
        self.assertIn("'--host'", script)
        self.assertIn("'--port'", script)
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


if __name__ == "__main__":
    unittest.main()

# Chinese ASR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first Chinese ASR project for rigorous low-hallucination transcription on Ryzen 9950X3D + RTX 5090 D.

**Architecture:** Use FunASR as the primary toolkit, with SenseVoiceSmall as the first default engine and Paraformer-zh as the conservative Chinese baseline. Keep Whisper as a documented comparison/fallback only. Every installer and runtime entrypoint clears proxy environment variables before any dependency or model download.

**Tech Stack:** Python 3.11, FunASR, ModelScope, CUDA PyTorch, PowerShell, `unittest`.

## Global Constraints

- Project root is `<repo-root>`.
- Downloads must not use `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, lowercase variants, or inherited proxy variables.
- Prefer ModelScope model IDs and project-local cache under `<repo-root>\models`.
- Core tests must run without installing FunASR, PyTorch, or model weights.
- Primary engine is `sensevoice`; secondary evaluation engine is `paraformer`; Whisper is fallback/comparison only.

---

### Task 1: Core Configuration and Proxy Guard

**Files:**
- Create: `src/zh_asr/config.py`
- Create: `src/zh_asr/proxy_guard.py`
- Test: `tests/test_config.py`
- Test: `tests/test_proxy_guard.py`

**Interfaces:**
- Produces: `EngineSpec`, `get_engine_spec(name: str) -> EngineSpec`, `DEFAULT_ENGINE`.
- Produces: `sanitized_env(extra: Mapping[str, str] | None = None) -> dict[str, str]`.
- Produces: `sanitize_current_process_env() -> None`.

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run `python -m unittest discover -s tests -v` and verify missing-module failures**
- [ ] **Step 3: Implement `config.py` and `proxy_guard.py`**
- [ ] **Step 4: Run tests and verify pass**

### Task 2: Transcript Bundle Writer

**Files:**
- Create: `src/zh_asr/result_writer.py`
- Test: `tests/test_result_writer.py`

**Interfaces:**
- Consumes: engine names from `config.py`.
- Produces: `write_transcript_bundle(audio_path: Path, result: object, out_dir: Path, engine: str) -> dict[str, Path]`.

- [ ] **Step 1: Write failing tests for Markdown and raw JSON outputs**
- [ ] **Step 2: Run the focused test and verify expected failure**
- [ ] **Step 3: Implement minimal writer**
- [ ] **Step 4: Run tests and verify pass**

### Task 3: Lazy FunASR Pipeline and CLI

**Files:**
- Create: `src/zh_asr/pipeline.py`
- Create: `src/zh_asr/__main__.py`
- Create: `src/zh_asr/__init__.py`
- Create: `pyproject.toml`

**Interfaces:**
- Consumes: `get_engine_spec`, `sanitize_current_process_env`, `write_transcript_bundle`.
- Produces: CLI commands `doctor`, `warmup`, and `transcribe`.

- [ ] **Step 1: Add CLI tests that do not import FunASR**
- [ ] **Step 2: Run tests and verify expected failure**
- [ ] **Step 3: Implement lazy import and helpful dependency errors**
- [ ] **Step 4: Run tests and verify pass**

### Task 4: No-Proxy Setup Scripts and Chinese Docs

**Files:**
- Create: `README.md`
- Create: `requirements-core.txt`
- Create: `scripts\Invoke-NoProxy.ps1`
- Create: `scripts\setup-core.ps1`
- Create: `scripts\install-torch-cu128-direct.ps1`
- Create: `scripts\download-models.ps1`
- Create: `scripts\doctor.ps1`

**Interfaces:**
- Scripts call project CLI with proxy variables cleared.
- Docs describe the Chinese low-hallucination model policy and verification path.

- [ ] **Step 1: Add docs and scripts**
- [ ] **Step 2: Run script syntax checks**
- [ ] **Step 3: Run project tests**

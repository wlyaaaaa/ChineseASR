# Long Audio Resume And Arbitration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add resumable long-audio strict transcription and optional local Ollama arbitration for uncertain chunks.

**Architecture:** Use deterministic duration-based chunk planning and a `manifest.json` to make long runs restartable. Run each chunk through existing strict transcription, skip completed chunks on resume, then merge chunk outputs into final transcript, audit, and metrics files. Add a small Ollama adapter that can arbitrate structured ASR disagreement evidence only when enabled.

**Tech Stack:** Python standard library (`wave`, `json`, `hashlib`, `urllib.request`), existing `strict_transcribe_audio`, existing unittest suite, PowerShell wrapper static checks.

## Global Constraints

- Do not run actual ASR models in tests.
- Do not call a real Ollama model in tests.
- Keep LLM arbitration optional and disabled by default.
- Preserve raw ASR outputs and audit evidence.
- Use `keep_alive=0` for Ollama arbitration by default.
- Resume must skip completed chunks with matching manifest and output files.

---

### Task 1: Long Audio Manifest And Runner

**Files:**
- Create: `src/zh_asr/long_audio.py`
- Create: `tests/test_long_audio.py`

**Interfaces:**
- `plan_chunks(audio_path: Path, chunk_sec: int, overlap_sec: int) -> list[ChunkSpec]`
- `run_long_transcription(audio_path: Path, out_dir: Path, ..., strict_fn=strict_transcribe_audio, arbiter=None) -> LongRunSummary`

- [x] **Step 1: Write failing tests for chunk planning, resume skip, stale reset, merged transcript, and metrics.**
- [x] **Step 2: Run `python -m unittest tests.test_long_audio -v` and verify failure.**
- [x] **Step 3: Implement chunk dataclasses, wave slicing, manifest read/write, resume logic, strict chunk runner, and merge writers.**
- [x] **Step 4: Re-run targeted tests and verify pass.**

### Task 2: Ollama Arbitration Adapter

**Files:**
- Create: `src/zh_asr/arbitration.py`
- Create: `tests/test_arbitration.py`
- Modify: `configs/models.yaml`

**Interfaces:**
- `ArbitrationConfig.from_mapping(mapping: dict) -> ArbitrationConfig`
- `OllamaArbiter.arbitrate(evidence: ArbitrationEvidence) -> ArbitrationDecision`
- `NullArbiter.arbitrate(evidence: ArbitrationEvidence) -> None`

- [x] **Step 1: Write failing tests for default disabled config, Ollama request payload, JSON response parsing, and fallback on invalid JSON.**
- [x] **Step 2: Run `python -m unittest tests.test_arbitration -v` and verify failure.**
- [x] **Step 3: Implement config dataclasses, prompt/evidence schema, Ollama adapter, and safe disabled adapter.**
- [x] **Step 4: Re-run targeted arbitration tests and verify pass.**

### Task 3: CLI, API, And Script Integration

**Files:**
- Modify: `src/zh_asr/__main__.py`
- Modify: `src/zh_asr/service.py`
- Modify: `scripts/asr-smart.ps1`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_service.py`
- Modify: `tests/test_scripts.py`

**Interfaces:**
- CLI: `python -m zh_asr long <audio> --out-dir outputs\long --chunk-sec 300 --overlap-sec 1`
- API payload: `mode=long-strict`
- Script: `scripts\asr-smart.ps1 -Mode long-strict`

- [x] **Step 1: Write failing tests for CLI registration, API command construction, and script static contract.**
- [x] **Step 2: Run targeted tests and verify failure.**
- [x] **Step 3: Add CLI command, service command builder support, and script mode support.**
- [x] **Step 4: Re-run targeted integration tests and verify pass.**

### Task 4: Documentation And Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: this plan file

- [x] **Step 1: Document long-audio resume and Ollama arbitration usage.**
- [x] **Step 2: Run full unittest discovery, compile check, script parser check, and `git diff --check`.**
- [ ] **Step 3: Commit the branch.**

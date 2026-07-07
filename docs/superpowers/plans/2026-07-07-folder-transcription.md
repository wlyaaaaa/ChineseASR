# Folder Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a daily-use folder transcription entry that scans audio files, runs quick or strict transcription, skips completed files, records failures, and writes a summary.

**Architecture:** Add `zh_asr.batch` for file discovery and orchestration, then expose it through `zh-asr batch` and `scripts/transcribe-folder.ps1`. The batch layer calls existing `transcribe_audio` / `strict_transcribe_audio` functions and does not duplicate model logic.

**Tech Stack:** Python 3.11, unittest, PowerShell, existing ASR pipeline.

## Global Constraints

- Do not route downloads or model calls through proxy variables.
- Default mode is `strict`.
- Supported extensions are `.wav`, `.mp3`, `.m4a`, `.flac`.
- Completed outputs are skipped unless `--force` / `-Force` is provided.
- Failures are written to `failed.jsonl`.
- A human-readable `summary.md` is always written.

---

### Task 1: Batch Orchestration

**Files:**
- Create: `src/zh_asr/batch.py`
- Test: `tests/test_batch.py`

**Interfaces:**
- Produces: `find_audio_files(input_dir: Path) -> list[Path]`
- Produces: `run_batch(input_dir, out_dir, mode, device, cache_dir, force=False, config=None, transcribe_fn=None, strict_fn=None) -> BatchSummary`

- [x] **Step 1: Write failing tests for scanning, per-file output dirs, skipping, failure JSONL, and summary markdown.**
- [x] **Step 2: Implement minimal batch orchestration.**
- [x] **Step 3: Run `python -m unittest tests.test_batch -v`.**

### Task 2: CLI And PowerShell Entry

**Files:**
- Modify: `src/zh_asr/__main__.py`
- Create: `scripts/transcribe-folder.ps1`
- Test: `tests/test_cli.py`
- Test: `tests/test_scripts.py`

**Interfaces:**
- Produces: `zh-asr batch <input-dir> --mode strict --out-dir outputs\batch`
- Produces: `scripts\transcribe-folder.ps1 -InputDir <dir> -Mode strict`

- [x] **Step 1: Write failing tests for CLI argument parsing and PowerShell script contents.**
- [x] **Step 2: Add CLI parser and script wrapper.**
- [x] **Step 3: Run targeted tests and full suite.**

### Task 3: Smoke And Docs

**Files:**
- Modify: `README.md`

- [x] **Step 1: Document the folder entry command.**
- [x] **Step 2: Run a smoke test on a tiny local folder using the cached sample audio.**
- [x] **Step 3: Commit verified changes.**

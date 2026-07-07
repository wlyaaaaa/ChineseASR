# Benchmark CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin benchmark entry that compares strict ASR output against human truth files for a user-provided audio folder.

**Architecture:** Add `zh_asr.benchmark` to build an eval-compatible manifest from `audio_dir` and `truth_dir`, then call existing `eval_pack.run_evaluation`. Expose it through `zh_asr benchmark` and `scripts/benchmark.ps1`; do not copy user audio into the repository.

**Tech Stack:** Python 3.11 standard library, existing strict/eval pipeline, PowerShell wrapper, unittest.

## Global Constraints

- Do not require private samples to be committed.
- Do not copy source audio or truth files into project output.
- Match audio to truth by filename stem.
- Missing truth files are reported in `review.md` and skipped from CER scoring.
- Unit tests must not load ASR models.
- Scripts must clear proxy variables.

---

### Task 1: Benchmark Core

**Files:**
- Create: `src/zh_asr/benchmark.py`
- Create: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `build_benchmark_manifest(audio_dir: Path, truth_dir: Path, manifest_dir: Path, force: bool = False) -> Path`
- Produces: `run_benchmark(audio_dir: Path, truth_dir: Path, out_dir: Path, device: str = "cuda:0", cache_dir: Path | None = None, force: bool = False, primary_engine: str | None = None, secondary_engine: str | None = None, config=None, strict_fn=strict_transcribe_audio) -> EvalSummary`

- [x] **Step 1: Write failing tests for manifest matching, missing truth, and benchmark outputs.**
- [x] **Step 2: Implement benchmark core with no audio/truth copying.**
- [x] **Step 3: Run `python -m unittest tests.test_benchmark -v`.**

### Task 2: CLI And Script Entry

**Files:**
- Modify: `src/zh_asr/__main__.py`
- Create: `scripts/benchmark.ps1`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_scripts.py`

**Interfaces:**
- Produces: `zh-asr benchmark --audio-dir <dir> --truth-dir <dir> --out-dir <dir>`
- Produces: `scripts\benchmark.ps1 -AudioDir <dir> -TruthDir <dir>`

- [x] **Step 1: Write failing tests for CLI missing dirs and script wrapper contents.**
- [x] **Step 2: Add CLI parser and PowerShell wrapper.**
- [x] **Step 3: Run targeted CLI/script tests.**

### Task 3: Docs And Verification

**Files:**
- Modify: `README.md`

- [x] **Step 1: Document folder layout and output files.**
- [x] **Step 2: Run full tests and compile check.**
- [x] **Step 3: Run a no-model benchmark smoke with fake/injected tests only; do not run ASR unless using an existing small sample intentionally.**

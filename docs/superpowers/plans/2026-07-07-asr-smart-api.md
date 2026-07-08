# ASR Smart API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local ASR job API and smart wrapper that return quickly, queue GPU work, detect model conflicts, and let Codex poll instead of blocking on long commands.

**Architecture:** Implement a standard-library local HTTP server backed by an in-memory job manager and one worker thread. Jobs run existing `zh_asr strict` / `zh_asr transcribe` commands in child Python processes, so cancellation and memory isolation are possible. `scripts\asr-smart.ps1` starts the server if needed, submits work, polls for a bounded time, and returns machine-readable status.

**Tech Stack:** Python standard library `http.server`, `threading`, `queue`, `subprocess`; PowerShell `Invoke-RestMethod`; existing unittest suite.

## Global Constraints

- Bind API to `127.0.0.1` by default.
- Do not add a new Python web framework dependency.
- Do not run actual ASR models in tests.
- Preserve no-proxy behavior.
- Detect foreign CUDA compute processes with `nvidia-smi` when available.
- Block GPU-conflicting submissions by default; allow explicit override.
- Keep command-line wait bounded through `scripts\asr-smart.ps1 -WaitSec`.

---

### Task 1: Job Manager Core

**Files:**
- Create: `src/zh_asr/service.py`
- Create: `tests/test_service.py`

**Interfaces:**
- `TranscriptionService.submit(request: JobRequest) -> tuple[Job, bool]`
- `TranscriptionService.get_job(job_id: str) -> Job | None`
- `TranscriptionService.cancel(job_id: str) -> Job`
- `TranscriptionService.health() -> dict`
- `JobRequest.from_payload(payload: dict, root: Path) -> JobRequest`

- [x] **Step 1: Write failing tests for dedupe, conflict blocking, override, and fake worker success.**
- [x] **Step 2: Run `python -m unittest tests.test_service -v` and verify failure.**
- [x] **Step 3: Implement dataclasses, conflict detector, queue, fake-runner-friendly worker, and job serialization.**
- [x] **Step 4: Re-run targeted service tests and verify pass.**

### Task 2: HTTP API And CLI Entry

**Files:**
- Modify: `src/zh_asr/service.py`
- Modify: `src/zh_asr/__main__.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- CLI: `python -m zh_asr serve --host 127.0.0.1 --port 8766 --state-dir outputs\api`
- Endpoints: `GET /health`, `GET /jobs`, `GET /jobs/{job_id}`, `POST /jobs/transcribe`, `POST /jobs/{job_id}/cancel`

- [x] **Step 1: Write failing tests for `serve` CLI registration and HTTP handler behavior with an injected service.**
- [x] **Step 2: Run targeted tests and verify failure.**
- [x] **Step 3: Implement HTTP routing, JSON responses, and CLI `serve` command.**
- [x] **Step 4: Re-run targeted tests and verify pass.**

### Task 3: Smart PowerShell Wrapper

**Files:**
- Create: `scripts/asr-smart.ps1`
- Modify: `tests/test_scripts.py`

**Interfaces:**
- `scripts\asr-smart.ps1 -Audio <path> [-Mode strict|quick] [-WaitSec 15] [-Port 8766] [-AllowGpuConflicts] [-Json]`

- [x] **Step 1: Write failing static script tests for no-proxy setup, bounded wait, server start, conflict override, JSON output, and status command.**
- [x] **Step 2: Run `python -m unittest tests.test_scripts -v` and verify failure.**
- [x] **Step 3: Implement `asr-smart.ps1` with startup health check, POST submit, bounded polling, and concise output.**
- [x] **Step 4: Run parser check for the script and targeted script tests.**

### Task 4: Documentation And Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: this plan file

**Interfaces:**
- User-facing command examples and conflict behavior documentation.

- [x] **Step 1: Document API/smart usage and conflict policy.**
- [x] **Step 2: Run full unittest discovery, compile check, PowerShell parser check, and `git diff --check`.**
- [x] **Step 3: Commit the branch.**

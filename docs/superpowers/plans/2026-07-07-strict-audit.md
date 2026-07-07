# Strict Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a personal strict transcription mode that compares SenseVoice and Paraformer outputs, writes a clean final transcript, and preserves uncertainty in an audit file.

**Architecture:** Keep ASR execution in `pipeline.py`, isolate comparison logic in `audit.py`, and isolate strict output formatting in `strict_writer.py`. `transcript.md` stays readable; severe uncertainty gets a small `[疑似]` or `[听不清]` inline marker, while detailed evidence goes to `audit.md` and raw JSON.

**Tech Stack:** Python 3.11, FunASR, ModelScope, CUDA PyTorch, PowerShell, `unittest`.

## Global Constraints

- Primary engine is `sensevoice`; secondary engine is `paraformer`.
- Do not call external LLMs in this phase.
- Preserve both raw ASR JSON outputs.
- Keep normal transcript clean unless uncertainty changes meaning.
- All scripts must clear proxy environment variables before running.

---

### Task 1: Audit Core

**Files:**
- Create: `src/zh_asr/audit.py`
- Test: `tests/test_audit.py`

**Interfaces:**
- Produces: `build_audit_report(primary_engine, primary_text, secondary_engine, secondary_text) -> AuditReport`

- [x] Write failing tests for consistent, conflict, unclear, and suspicious-stock-phrase cases.
- [x] Implement normalized comparison, similarity score, flags, final guess, alternatives, and rationale.
- [x] Verify tests pass.

### Task 2: Strict Bundle Writer

**Files:**
- Create: `src/zh_asr/strict_writer.py`
- Test: `tests/test_strict_writer.py`

**Interfaces:**
- Consumes: `extract_text()` and `build_audit_report()`.
- Produces: `write_strict_bundle(...) -> dict[str, Path]`.

- [x] Write failing test that expects final Markdown, audit Markdown, audit JSON, and both raw JSON files.
- [x] Implement strict bundle writer.
- [x] Verify tests pass.

### Task 3: CLI and No-Proxy Wrapper

**Files:**
- Modify: `src/zh_asr/pipeline.py`
- Modify: `src/zh_asr/__main__.py`
- Create: `scripts/strict.ps1`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `python -m zh_asr strict <audio>`.
- Produces: `scripts\strict.ps1 -Audio <path>`.

- [x] Add CLI test for missing strict audio path.
- [x] Implement sequential two-model strict transcription.
- [x] Add no-proxy PowerShell wrapper.
- [x] Verify tests pass.


# Offline Wheelhouse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight offline dependency recovery workflow with wheelhouse scripts and checksum manifests.

**Architecture:** Keep large wheel files out of Git under `offline/wheelhouse/`, while committing scripts and small manifest placeholders. Reuse `Invoke-NoProxy.ps1` for all network-capable scripts. Keep model weight checksums out of this first version.

**Tech Stack:** PowerShell, Python/pip, SHA256 checksums, existing unittest static script checks.

## Global Constraints

- Do not download wheels during unit tests.
- Do not commit `offline/wheelhouse/` contents.
- Preserve existing no-proxy behavior.
- Support PyTorch CUDA 12.8 wheels separately from normal PyPI/Tsinghua dependencies.
- Offline install must verify SHA256 before pip install.

---

### Task 1: Script Contract Tests

**Files:**
- Modify: `tests/test_scripts.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces static assertions for `export-lock.ps1`, `build-wheelhouse.ps1`, `verify-wheelhouse.ps1`, and `install-offline.ps1`.

- [x] **Step 1: Write failing tests for offline scripts and `.gitignore`.**
- [x] **Step 2: Run `python -m unittest tests.test_scripts -v` and verify failure.**

### Task 2: Offline Scripts

**Files:**
- Create: `scripts/export-lock.ps1`
- Create: `scripts/build-wheelhouse.ps1`
- Create: `scripts/verify-wheelhouse.ps1`
- Create: `scripts/install-offline.ps1`

**Interfaces:**
- `export-lock.ps1 [-Venv <path>] [-OutDir <path>]`
- `build-wheelhouse.ps1 [-Wheelhouse <path>] [-ManifestDir <path>] [-TorchIndexUrl <url>] [-PypiIndexUrl <url>]`
- `verify-wheelhouse.ps1 [-Wheelhouse <path>] [-ChecksumFile <path>]`
- `install-offline.ps1 [-Venv <path>] [-Wheelhouse <path>] [-ManifestDir <path>] [-SkipVerify]`

- [x] **Step 1: Implement scripts with no-proxy setup, fail-fast errors, and clear paths.**
- [x] **Step 2: Re-run targeted tests and verify pass.**

### Task 3: Documentation And Verification

**Files:**
- Modify: `README.md`
- Modify: this plan file

- [x] **Step 1: Document normal online install vs offline recovery workflow.**
- [x] **Step 2: Run full unit tests, compile check, and diff check.**

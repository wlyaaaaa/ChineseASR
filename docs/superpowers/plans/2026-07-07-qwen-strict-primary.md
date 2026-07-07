# Qwen Strict Primary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change strict transcription from `SenseVoice + Paraformer` to `Qwen3-ASR-1.7B + SenseVoiceSmall` while keeping Paraformer as a fallback baseline.

**Architecture:** Add a `qwen-asr` adapter that wraps Qwen's `transcribe()` API into the existing `generate()` result shape. Keep quick mode on SenseVoice for fast local runs, but route strict mode through Qwen as primary and SenseVoice as the low-hallucination anchor. Use ModelScope prefetch scripts so Qwen weights land in the existing local cache before runtime.

**Tech Stack:** Python 3.11, `qwen-asr`, PyTorch bfloat16, ModelScope, unittest, PowerShell.

## Global Constraints

- Do not download through proxy variables.
- Do not force Qwen dependencies into the lightweight core setup.
- Keep Paraformer configured as a baseline engine.
- Preserve current output bundle names and audit behavior.

---

### Task 1: Registry Change

**Files:**
- Modify: `configs/models.yaml`
- Modify: `src/zh_asr/config.py`
- Test: `tests/test_config.py`

- [x] **Step 1: Write failing tests for Qwen strict primary**
- [x] **Step 2: Add `EngineSpec.options` parsing**
- [x] **Step 3: Register `qwen3-asr-1.7b` and set strict pair to Qwen + SenseVoice**

### Task 2: Qwen Adapter

**Files:**
- Create: `src/zh_asr/adapters/qwen_asr.py`
- Modify: `src/zh_asr/adapters/__init__.py`
- Modify: `src/zh_asr/adapters/base.py`
- Test: `tests/test_pipeline.py`

- [x] **Step 1: Write failing tests for result normalization and local cache resolution**
- [x] **Step 2: Add `QwenASRAdapter` and `QwenASRModelWrapper`**
- [x] **Step 3: Surface missing `qwen-asr` as a clear setup error**

### Task 3: Setup And Documentation

**Files:**
- Create: `requirements-qwen.txt`
- Create: `scripts/setup-qwen.ps1`
- Modify: `scripts/download-models.ps1`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Test: `tests/test_scripts.py`

- [x] **Step 1: Write failing script tests for no-proxy Qwen setup and ModelScope prefetch**
- [x] **Step 2: Add optional Qwen setup script**
- [x] **Step 3: Add ModelScope prefetch before Qwen warmup**
- [x] **Step 4: Document the new strict default and install path**

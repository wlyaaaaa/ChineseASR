# Model Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move ASR model selection out of Python hardcoding so same-adapter model swaps are config-only.

**Architecture:** `configs/models.yaml` owns defaults, strict pair selection, aliases, and engine specs. `zh_asr.config` loads and validates the registry, while `zh_asr.adapters` owns runtime-specific model construction. `zh_asr.pipeline` depends on the registry and adapter interface rather than individual model names.

**Tech Stack:** Python 3.11, PyYAML, FunASR, unittest, PowerShell.

## Global Constraints

- Preserve the current default route: `sensevoice` as primary and `paraformer` as strict secondary.
- Do not route downloads through proxy variables.
- Same `funasr` adapter model swaps should require YAML edits only.
- Cross-runtime models require a new adapter.

---

### Task 1: Config-Driven Registry

**Files:**
- Create: `configs/models.yaml`
- Modify: `src/zh_asr/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_model_config(path: Path | str | None = None) -> ModelConfig`
- Produces: `get_engine_spec(name: str, config: ModelConfig | None = None) -> EngineSpec`

- [x] **Step 1: Write failing tests for YAML-backed defaults and custom same-adapter engines**
- [x] **Step 2: Run `python -m unittest tests.test_config` and verify failure**
- [x] **Step 3: Implement `ModelConfig`, YAML loading, validation, and env override**
- [x] **Step 4: Run `python -m unittest tests.test_config` and verify pass**

### Task 2: Adapter Boundary

**Files:**
- Create: `src/zh_asr/adapters/base.py`
- Create: `src/zh_asr/adapters/funasr.py`
- Modify: `src/zh_asr/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `get_adapter(name: str) -> ModelAdapter`
- Produces: `funasr_kwargs(spec, device, cache_dir, model_aliases) -> dict`

- [x] **Step 1: Write failing tests for alias resolution from registry**
- [x] **Step 2: Run `python -m unittest tests.test_pipeline` and verify failure**
- [x] **Step 3: Move FunASR construction into `FunASRAdapter`**
- [x] **Step 4: Run `python -m unittest tests.test_pipeline` and verify pass**

### Task 3: CLI And Script Integration

**Files:**
- Modify: `src/zh_asr/__main__.py`
- Modify: `scripts/download-models.ps1`
- Modify: `scripts/strict.ps1`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `ModelConfig.default_engine`
- Consumes: `ModelConfig.strict_primary_engine`
- Consumes: `ModelConfig.strict_secondary_engine`

- [x] **Step 1: Write failing CLI test using `ZH_ASR_MODEL_CONFIG`**
- [x] **Step 2: Run `python -m unittest tests.test_cli` and verify failure**
- [x] **Step 3: Load CLI choices and defaults from model config**
- [x] **Step 4: Remove hardcoded PowerShell engine validation**
- [x] **Step 5: Run `python -m unittest tests.test_cli` and verify pass**

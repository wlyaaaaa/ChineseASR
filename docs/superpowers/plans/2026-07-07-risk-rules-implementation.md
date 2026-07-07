# Risk Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic hallucination risk rules and write their hits into strict audit, metrics, and review outputs.

**Architecture:** Create `zh_asr.risk_rules` for pure rule detection, then consume it from `audit.py` and `eval_pack.py`. Keep `flags` backward-compatible by deriving it from rule ids plus existing metric-only flags.

**Tech Stack:** Python 3.11 standard library, dataclasses, existing unittest suite.

## Global Constraints

- No model calls in unit tests.
- Do not change transcript正文 policy beyond existing `[疑似]` and `[听不清]`.
- Do not copy long transcript spans into rule evidence.
- Keep existing `flags` fields for compatibility.
- Keep thresholds deterministic and local to `risk_rules.py`.

---

### Task 1: Rule Library

**Files:**
- Create: `src/zh_asr/risk_rules.py`
- Create: `tests/test_risk_rules.py`

**Interfaces:**
- Produces: `RuleHit(id: str, severity: str, message: str, evidence: str)`
- Produces: `evaluate_risk_rules(primary_text: str, secondary_text: str, final_text: str, similarity: float, expect_empty: bool = False) -> tuple[RuleHit, ...]`

- [x] **Step 1: Write failing tests covering six rule ids.**
- [x] **Step 2: Run `python -m unittest tests.test_risk_rules -v` and verify failure.**
- [x] **Step 3: Implement `risk_rules.py`.**
- [x] **Step 4: Re-run the test and verify pass.**

### Task 2: Strict Audit Integration

**Files:**
- Modify: `src/zh_asr/audit.py`
- Modify: `src/zh_asr/strict_writer.py`
- Modify: `src/zh_asr/pipeline.py`
- Modify: `tests/test_audit.py`
- Modify: `tests/test_strict_writer.py`

**Interfaces:**
- Produces: `AuditReport.rule_hits`
- Produces: optional `expect_empty` parameter on `build_audit_report`, `write_strict_bundle`, and `strict_transcribe_audio`.

- [x] **Step 1: Write failing audit and strict writer tests.**
- [x] **Step 2: Run targeted tests and verify failure.**
- [x] **Step 3: Add rule hits to audit report JSON/Markdown.**
- [x] **Step 4: Re-run targeted tests and verify pass.**

### Task 3: Eval And Benchmark Propagation

**Files:**
- Modify: `src/zh_asr/eval_pack.py`
- Modify: `tests/test_eval_pack.py`
- Modify: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `metrics.json.cases[].rule_hits`
- Produces: review lines with rule evidence.

- [x] **Step 1: Write failing metrics/review tests.**
- [x] **Step 2: Run targeted tests and verify failure.**
- [x] **Step 3: Merge rule hits into eval results and review output.**
- [x] **Step 4: Re-run targeted tests and verify pass.**

### Task 4: Documentation And Verification

**Files:**
- Modify: `README.md`
- Modify: this plan file

- [x] **Step 1: Document deterministic risk rules.**
- [x] **Step 2: Run full tests.**
- [x] **Step 3: Run compile check.**
- [x] **Step 4: Run benchmark smoke and inspect rule fields.**

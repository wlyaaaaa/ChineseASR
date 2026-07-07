# Review Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `review.md` from a flat findings dump into a prioritized human review queue.

**Architecture:** Keep the feature inside `zh_asr.eval_pack` because review output is derived from `EvalCaseResult`. Add small pure helpers for priority, reason, action text, and clipped evidence, then keep `_write_review()` as the file writer. Do not add UI, external services, or LLM arbitration.

**Tech Stack:** Python 3.11 standard library, dataclasses already in use, existing `unittest` suite.

## Global Constraints

- `review.md` remains Markdown only.
- No model calls in unit tests.
- Do not copy long transcript spans into review evidence.
- Keep `metrics.json` and `benchmark.json` schema stable.
- Sort review items by value to the human: P0 first, then P1, then P2, then case id.

---

### Task 1: Prioritized Markdown Review Queue

**Files:**
- Modify: `src/zh_asr/eval_pack.py`
- Modify: `tests/test_eval_pack.py`
- Modify: `tests/test_benchmark.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `EvalCaseResult`
- Produces: enhanced `review.md` with priority, reason, action, clipped text evidence, and source paths.

- [x] **Step 1: Write failing tests for P0/P1/P2 ordering and evidence fields.**
- [x] **Step 2: Run targeted tests and verify failure.**
- [x] **Step 3: Implement pure review helpers and wire `_write_review()`.**
- [x] **Step 4: Re-run targeted tests and verify pass.**
- [x] **Step 5: Update README to describe the review queue contract.**
- [x] **Step 6: Run full unit tests, compile check, diff check, and smoke benchmark.**

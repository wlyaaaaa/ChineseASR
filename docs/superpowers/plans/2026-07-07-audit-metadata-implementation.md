# Audit Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance `manifest.json`, `metrics.json`, and `benchmark.json` so every ASR eval run records input hashes, model configuration, invocation, runtime, timing, text similarity, and risk markers.

**Architecture:** Add a focused `zh_asr.metadata` module for hashing, config snapshots, invocation capture, runtime capture, and timers. Keep ASR adapters unchanged; enrich benchmark manifest generation, strict pipeline timing metadata, and eval metrics serialization around the existing strict/audit flow.

**Tech Stack:** Python 3.11 standard library, existing unittest suite, existing strict/eval/benchmark code.

## Global Constraints

- Do not copy private source audio into output directories.
- Do not upload data.
- Keep tests model-free by injecting fake strict functions.
- Preserve existing `eval` and `benchmark` CLI behavior.
- Preserve `benchmark.json` as a compatibility artifact.
- Per-model timing may be `null` when unavailable, but fields must exist.

---

### Task 1: Metadata Helpers

**Files:**
- Create: `src/zh_asr/metadata.py`
- Create: `tests/test_metadata.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`
- Produces: `file_metadata(path: Path | None) -> dict[str, int | str]`
- Produces: `snapshot_model_config(config: ModelConfig, selected_engines: tuple[str, ...] | None = None) -> dict[str, object]`
- Produces: `capture_invocation(argv: list[str] | None = None, wrapper: str | None = None) -> dict[str, str | list[str]]`
- Produces: `runtime_info(device: str) -> dict[str, object]`

- [x] **Step 1: Write failing metadata tests**

```python
def test_file_metadata_hashes_size_and_missing_paths():
    from zh_asr.metadata import file_metadata
    meta = file_metadata(path)
    self.assertEqual(meta["sha256"], expected_hash)
    self.assertEqual(meta["size_bytes"], 3)
```

- [x] **Step 2: Run metadata tests and verify failure**

Run: `$env:PYTHONPATH='<repo-root>\src'; .\.venv\Scripts\python.exe -m unittest tests.test_metadata -v`

- [x] **Step 3: Implement `metadata.py`**

Use dataclass-safe conversion for engine specs, read config file bytes for hash, avoid failing if Torch is unavailable.

- [x] **Step 4: Run metadata tests and verify pass**

Run the same command and expect `OK`.

### Task 2: Manifest Audit Ledger

**Files:**
- Modify: `src/zh_asr/benchmark.py`
- Modify: `src/zh_asr/eval_pack.py`
- Modify: `tests/test_benchmark.py`
- Modify: `tests/test_eval_pack.py`

**Interfaces:**
- Consumes: `snapshot_model_config`, `capture_invocation`, `file_metadata`
- Produces: schema version 2 manifest fields in built-in eval and benchmark manifests.

- [x] **Step 1: Write failing manifest tests**

Assert `_manifest/manifest.json` contains `schema_version`, `model_config.sha256`, `model_config.selected_engines`, `invocation.argv`, `audio_sha256`, `audio_size_bytes`, `truth_sha256`, and `truth_size_bytes`.

- [x] **Step 2: Run targeted tests and verify failure**

Run: `$env:PYTHONPATH='<repo-root>\src'; .\.venv\Scripts\python.exe -m unittest tests.test_benchmark tests.test_eval_pack -v`

- [x] **Step 3: Enrich manifests**

Add metadata to benchmark manifest generation and built-in corpus manifest generation. Keep relative paths for built-in corpus and absolute paths for benchmark manifests as currently used.

- [x] **Step 4: Run targeted tests and verify pass**

Run the same command and expect `OK`.

### Task 3: Metrics Run Ledger And Timing

**Files:**
- Modify: `src/zh_asr/pipeline.py`
- Modify: `src/zh_asr/eval_pack.py`
- Modify: `tests/test_eval_pack.py`
- Modify: `tests/test_benchmark.py`

**Interfaces:**
- Produces optional strict result metadata keys: `timing`, `primary_json`, `secondary_json`, `audit_json`
- Produces metrics case fields: `models`, `texts`, `text_similarity`, `timing`, `audit_status`, `paths`, `truth_sha256`

- [x] **Step 1: Write failing metrics tests**

Assert `metrics.json` includes top-level `schema_version`, `runtime`, `model_config`, `invocation`; case fields include `models.primary`, `texts.primary`, `texts.secondary`, `texts.final`, `text_similarity.cer`, `timing.total_sec`, `timing.primary_sec`, `paths.audit_json`, and `risk_flags`.

- [x] **Step 2: Run targeted tests and verify failure**

Run: `$env:PYTHONPATH='<repo-root>\src'; .\.venv\Scripts\python.exe -m unittest tests.test_eval_pack tests.test_benchmark -v`

- [x] **Step 3: Implement timing and metrics serialization**

Time each eval case around `strict_fn`. Time primary and secondary model calls in `strict_transcribe_audio`. Preserve fake strict functions that only return `audit_json`.

- [x] **Step 4: Run targeted tests and verify pass**

Run the same command and expect `OK`.

### Task 4: Compatibility, Docs, And Verification

**Files:**
- Modify: `src/zh_asr/benchmark.py`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-07-07-audit-metadata-implementation.md`

**Interfaces:**
- Produces enhanced `benchmark.json` copied from metrics payload plus benchmark pointers.

- [x] **Step 1: Write failing benchmark compatibility assertion**

Assert `benchmark.json["schema_version"] == 2`, it contains `benchmark.manifest`, and its first case includes `truth_sha256`.

- [x] **Step 2: Implement benchmark JSON compatibility payload**

Write `benchmark.json` from enhanced `metrics.json`, preserving summary/cases and adding `benchmark`.

- [x] **Step 3: Update README metadata notes**

Document that manifest and metrics now contain hashes, model config, invocation, runtime, timing, and risk fields.

- [x] **Step 4: Run full verification**

Run:

```powershell
$env:PYTHONPATH='<repo-root>\src'; .\.venv\Scripts\python.exe -m unittest discover -s <repo-root>\tests -v
$env:PYTHONPATH='<repo-root>\src'; .\.venv\Scripts\python.exe -m compileall src tests
.\scripts\benchmark.ps1 -AudioDir <repo-root>\eval\corpus\smoke-tts\synthetic -TruthDir <repo-root>\eval\corpus\smoke-tts\truth -OutDir <repo-root>\outputs\benchmark-smoke-tts -Force
```

Expected: tests pass, compile exits 0, smoke writes schema version 2 `manifest.json`, `metrics.json`, and `benchmark.json`.

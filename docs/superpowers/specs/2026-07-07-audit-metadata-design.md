# Audit Metadata Design

## Goal

Make each benchmark and evaluation run auditable and reproducible without copying private audio. The output should answer what was run, which model configuration was used, how long it took, how similar the model texts were, and why any case needs review.

## Scope

This design covers the existing `eval` and `benchmark` flows. It enhances `_manifest/manifest.json`, `metrics.json`, and `benchmark.json`. It does not change ASR model selection, transcript wording, or the strict dual-model decision policy.

## Output Roles

`_manifest/manifest.json` is the input ledger. It is generated before ASR execution and records the planned cases, input hashes, selected engines, model configuration snapshot, and invocation metadata.

`metrics.json` is the run ledger. It records runtime environment, elapsed time, model outputs, similarity metrics, CER, risk flags, skip reasons, and output paths for each case.

`benchmark.json` remains a compatibility artifact for benchmark users. It should contain the same enhanced result payload as `metrics.json`, plus a `benchmark` section pointing back to the user audio directory, truth directory, and generated manifest.

## Manifest Schema

The manifest should use `schema_version: 2` and keep `version: 1` only if needed for backward compatibility. Top-level fields:

- `schema_version`
- `generated`
- `description`
- `audio_dir` and `truth_dir` for benchmark runs when available
- `model_config`
- `invocation`
- `cases`

`model_config` stores:

- `path`: absolute path to the config file
- `sha256`: hash of the config file bytes
- `default_engine`
- `strict_primary_engine`
- `strict_secondary_engine`
- `aliases`: model aliases used by adapters
- `selected_engines`: full snapshots of the selected primary and secondary engine specs, including adapter, role, model id, VAD, punctuation model, speaker model, language, options, and note

`invocation` stores:

- `cwd`
- `argv`
- `command_line`
- `python`
- `project_root`
- `wrapper`, when a PowerShell wrapper passes this via environment or explicit CLI option

Each manifest case stores:

- `id`
- `category`
- `kind`
- `audio`
- `audio_sha256`
- `audio_size_bytes`
- `truth`
- `truth_sha256`
- `truth_size_bytes`
- `truth_text`
- `expect_empty`
- `available`
- `error`, when unavailable
- `notes`

Private audio is not copied into output directories. Hashes are enough to prove identity across reruns.

## Metrics Schema

`metrics.json` should include:

- `schema_version`
- `summary`
- `runtime`
- `model_config`
- `invocation`
- `cases`

`summary` extends the current fields with:

- `started_at`
- `finished_at`
- `elapsed_sec`
- `generated`
- `total`
- `evaluated`
- `skipped`
- `hallucination_count`
- `false_confident_count`

`runtime` stores low-risk local facts:

- `platform`
- `python`
- `device`
- `torch`
- `cuda_available`
- `cuda_version`
- `gpu_name`

Each metrics case stores current fields plus:

- `audio_sha256`
- `truth_sha256`
- `models.primary`
- `models.secondary`
- `texts.primary`
- `texts.secondary`
- `texts.final`
- `text_similarity.primary_secondary`
- `text_similarity.disagreement_score`
- `text_similarity.cer`
- `timing.total_sec`
- `timing.primary_sec`, when available
- `timing.secondary_sec`, when available
- `risk_flags`
- `audit_status`
- `needs_review`
- `false_confident`
- `simplified_only`
- `paths.audit_json`
- `paths.primary_raw_json`
- `paths.secondary_raw_json`

If primary and secondary per-model timing cannot be captured without changing adapter contracts, the first implementation can record `total_sec` and leave per-model timing as `null`. The field must still exist so downstream tools can rely on the schema.

## Data Flow

Add a small metadata module that owns hashing, model config snapshots, invocation capture, runtime capture, and monotonic timers. Keep it independent from adapters so future models can be swapped without changing audit file shape.

Benchmark manifest building computes file hashes and model config snapshots before calling `run_evaluation`. The eval flow loads manifest metadata, times each strict call, reads the strict audit JSON, and writes the enhanced metrics payload. `benchmark.json` is produced from the same metrics payload with benchmark-specific pointers added.

Strict transcription should return optional timing and raw output path metadata when available. Existing callers should keep working if this metadata is absent.

## Error Handling

Missing audio or truth files stay explicit and non-fatal at case level when the manifest can be built. Missing benchmark input directories remain command-level errors. Hashing failures mark the case unavailable with an error message.

If runtime metadata cannot detect Torch or CUDA, record empty or false values instead of failing the run.

## Privacy And Reproducibility

The system may store absolute local paths, hashes, sizes, truth text, ASR text, model configuration, and command metadata. It must not copy private source audio or upload data. Outputs remain under ignored directories unless the user explicitly commits an evaluation artifact.

The model config hash and selected engine snapshot are both required. The hash proves which config file was used, while the snapshot keeps results readable even after `configs/models.yaml` changes later.

## Testing

Unit tests should cover:

- Manifest cases include audio and truth hashes without copying audio.
- Manifest includes selected engine snapshots and config hash.
- Metrics include invocation, runtime, model config, timing, model names, primary/secondary/final text, similarity, CER, risk flags, and paths.
- Missing truth still appears in review and metrics with `skipped: true`.
- Benchmark JSON preserves enhanced metrics and adds benchmark pointers.
- Tests use injected strict functions and do not load ASR models.

## Acceptance Criteria

Running the existing benchmark smoke should produce `_manifest/manifest.json`, `metrics.json`, and `benchmark.json` with schema version 2 metadata. A future user can inspect those files and know the input hashes, model identities, model config, command, device, elapsed time, text similarity, CER, and review risk flags without opening source code.

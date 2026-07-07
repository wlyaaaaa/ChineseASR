# Long Audio Resume And LLM Arbitration Design

## Goal

Support long Chinese recordings without losing work on failure, and optionally use a local Ollama LLM to arbitrate only uncertain ASR disagreements while preserving all evidence.

## Long Audio Strategy

The first version uses deterministic chunk planning by duration, with all boundaries recorded in a manifest. The default chunk length is five minutes with a one-second overlap. This is less clever than VAD boundary detection, but it is reproducible, dependency-light, and safe for a first resumable pipeline. Future VAD-aware splitting can replace the planner without changing the manifest schema.

Each long run writes:

- `manifest.json`: audio hash, model config hash, chunk settings, chunk status, and output paths.
- `chunks/chunk-000001.wav`: generated chunk files.
- `chunks/chunk-000001/`: strict output bundle for that chunk.
- `transcript.md`: merged human-readable transcript.
- `audit.md`: merged audit evidence, chunk status, and risk markers.
- `metrics.json`: machine-readable run summary.

Resume behavior is based on the manifest. A chunk with matching audio hash, config hash, chunk settings, and existing final/audit files is skipped. `running` or `stale` chunks are reset to `pending`. Failed chunks can be retried without reprocessing successful chunks.

## LLM Arbitration Strategy

LLM arbitration is optional and off by default. When enabled, the pipeline sends only uncertain chunks to a local Ollama model. The LLM does not receive audio. It receives structured evidence: chunk id, time range, neighboring context, primary text, secondary text, similarity, flags, and rule hits.

The default provider is Ollama:

```yaml
llm_arbitration:
  enabled: false
  provider: ollama
  base_url: http://127.0.0.1:11434
  model: qwen-main-v1:latest
  fallback_model: qwen3.6-27b-256k:latest
  mode: uncertain_only
  temperature: 0.1
  keep_alive: 0
```

The Ollama request uses `/api/chat`, `stream=false`, `format=json`, low temperature, and `keep_alive=0` so the model unloads after use. Arbitration runs after ASR chunk work, not concurrently with ASR, to avoid GPU contention.

## Evidence Rules

The LLM may choose a final text, but it must not overwrite raw ASR outputs. Its output is saved under `arbitration` in `metrics.json` and summarized in `audit.md`. Low-confidence decisions keep `[疑似]` or `[听不清]` markers in `transcript.md`.

## Out Of Scope

- Full VAD-aware splitting.
- Speaker diarization across chunks.
- Cloud LLM providers.
- GUI review workflow.
- Actual Ollama model smoke during unit tests.

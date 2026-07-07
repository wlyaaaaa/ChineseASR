# ChineseASR

ChineseASR is a local-first Chinese speech-to-text project focused on low hallucination, auditable outputs, and reproducible evaluation. It is designed for Windows + CUDA workstations, with an accuracy-first strict mode and a bounded local API wrapper for automation agents.

## What This Project Optimizes For

- **Chinese accuracy first**: strict mode defaults to `Qwen3-ASR-1.7B` as the primary engine and `SenseVoiceSmall` as the acoustic cross-check.
- **Low hallucination behavior**: deterministic audit rules flag silent-audio hallucinations, stock phrases, abnormal repetition, model conflicts, Traditional Chinese residue, and long unpunctuated text.
- **Evidence preservation**: final transcripts, audit reports, raw model JSON, metrics, manifests, and review queues are written as separate files.
- **Model/runtime decoupling**: model choices live in `configs/models.yaml`; adapters expose a common `generate(...)` shape.
- **Agent-safe execution**: `scripts\asr-smart.ps1` submits a job to a local API and returns quickly instead of blocking the caller indefinitely.
- **No proxy by default**: wrapper scripts clear proxy environment variables before setup, download, evaluation, and transcription commands.

## Current Default Engines

| Mode | Default | Purpose |
| --- | --- | --- |
| `strict` primary | `qwen3-asr-1.7b` | Accuracy-first Chinese transcription |
| `strict` secondary | `sensevoice` | Fast local Chinese acoustic anchor and disagreement detector |
| `quick` | `sensevoice` | Fast single-model transcription |
| baseline | `paraformer` | Conservative Mandarin comparison line |
| fallback/comparison | `whisper-large-v3` | Registered for comparison only; not used as the default Chinese strict path |

Strict mode can still produce a usable audit package if one engine fails. The final transcript is marked `[疑似]`, and `strict.audit.md` records `engine_failure` plus the failing engine and error summary.

## Public Repository Boundary

This repository tracks source code, scripts, tests, configuration, and documentation. It intentionally does **not** track:

- `.venv/`
- `models/`
- `outputs/`
- `eval/corpus/`
- `offline/wheelhouse/`
- Python cache and build artifacts

Source audio, generated transcripts, model weights, and wheel files stay local unless you deliberately publish them elsewhere.

## Quick Start

Open PowerShell in the repository root:

```powershell
cd <repo-root>
.\scripts\install-torch-cu128-direct.ps1
.\scripts\setup-core.ps1
.\scripts\download-models.ps1 -Engine sensevoice
```

For strict mode, install and prefetch the Qwen ASR runtime and weights:

```powershell
.\scripts\setup-qwen.ps1
.\scripts\download-models.ps1 -Engine qwen3-asr-1.7b
```

Then check the environment:

```powershell
.\scripts\doctor.ps1
```

## Basic Usage

Fast single-model transcription:

```powershell
.\.venv\Scripts\python.exe -m zh_asr transcribe C:\path\to\audio.wav --engine sensevoice --device cuda:0 --out-dir outputs
```

Strict dual-model transcription:

```powershell
.\scripts\strict.ps1 -Audio C:\path\to\audio.wav
```

Agent-safe smart entry:

```powershell
.\scripts\asr-smart.ps1 -Audio C:\path\to\audio.wav -Mode strict -WaitSec 15 -Json
```

Long audio with resumable chunking:

```powershell
.\scripts\asr-smart.ps1 -Audio C:\path\to\long.wav -Mode long-strict -ChunkSec 300 -OverlapSec 1 -WaitSec 15 -Json
```

Batch transcription:

```powershell
.\scripts\transcribe-folder.ps1 -InputDir C:\path\to\audio-folder
```

Fixed local smoke test:

```powershell
.\scripts\smoke-asr-smart.ps1 -Json
```

## Output Files

Strict mode writes:

- `*.strict.md`: final human-readable transcript.
- `*.strict.audit.md`: model texts, similarity, flags, alternatives, and rationale.
- `*.strict.audit.json`: machine-readable audit report.
- `*.qwen3-asr-1.7b.raw.json`: primary raw result.
- `*.sensevoice.raw.json`: secondary raw result.

`asr-smart.ps1` returns the output paths in JSON, including `primary_raw_json` and `secondary_raw_json`.

## Local API

The smart wrapper starts a local API when needed:

```powershell
.\.venv\Scripts\python.exe -m zh_asr serve --host 127.0.0.1 --port 8765 --state-dir outputs\api
```

Endpoints:

- `GET /health`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `POST /jobs/transcribe`
- `POST /jobs/{job_id}/cancel`

The service binds to `127.0.0.1` only. Jobs run in child Python processes, so the HTTP request thread is not occupied by model inference. The service also checks `nvidia-smi` for external CUDA compute processes and blocks new jobs unless `allow_gpu_conflicts=true` is explicitly set.

## Evaluation and Benchmarking

Generate a privacy-friendly local evaluation corpus:

```powershell
.\scripts\eval.ps1 -Generate -GenerateOnly
```

Run the built-in evaluation:

```powershell
.\scripts\eval.ps1 -Generate -Force
```

Run benchmark mode against your own audio/truth folders:

```powershell
.\scripts\benchmark.ps1 -AudioDir C:\path\to\audio -TruthDir C:\path\to\truth
```

Benchmark mode matches files by stem, writes `_manifest\manifest.json`, and does not copy your source audio or truth files.

## Offline Wheelhouse

Freeze and download the current Python wheel set:

```powershell
.\scripts\export-lock.ps1
.\scripts\build-wheelhouse.ps1
.\scripts\verify-wheelhouse.ps1
```

Install from the local wheelhouse:

```powershell
.\scripts\install-offline.ps1 -Venv .venv-offline-smoke
```

`offline\wheelhouse\` is ignored by Git. Small lock/checksum manifests can be tracked under `offline\manifests\`.

## Model Registry

Edit `configs\models.yaml` to change engines:

- `defaults.engine`: quick-mode default.
- `strict.primary_engine`: strict primary.
- `strict.secondary_engine`: strict secondary.
- `engines.*.adapter`: runtime adapter, currently `funasr` and `qwen-asr`.
- `llm_arbitration`: optional local Ollama arbitration config, disabled by default.

Temporary override:

```powershell
$env:ZH_ASR_MODEL_CONFIG='C:\path\to\models.yaml'
```

## Testing

Unit tests do not require loading ASR models:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Useful end-to-end checks:

```powershell
.\scripts\smoke-asr-smart.ps1 -Json
.\scripts\asr-smart.ps1 -Audio outputs\smoke\asr-smart-zh-smoke.wav -Mode strict -Force -WaitSec 300 -Json
```

The second command assumes you have generated or provided that local smoke audio.

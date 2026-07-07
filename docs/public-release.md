# Public Release Notes

This repository is intended to be safe to publish as source code.

## Included

- Python source under `src/zh_asr`
- PowerShell wrappers under `scripts`
- model registry config under `configs`
- unit tests under `tests`
- architecture and usage documentation
- small offline manifest placeholders under `offline/manifests`

## Excluded

The following paths are ignored and should not be pushed:

- `.venv/`
- `models/`
- `outputs/`
- `eval/corpus/`
- `offline/wheelhouse/`
- `__pycache__/`
- `*.egg-info/`

These directories may contain model weights, generated transcripts, private audio paths, evaluation artifacts, Python build metadata, or large wheel files.

## Publication Checklist

Before pushing:

1. Run unit tests:

   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -s tests
   ```

2. Run a source-only secret/path scan over tracked files.
3. Confirm `git status --ignored=matching` shows large runtime directories as ignored.
4. Push only tracked source/documentation files.
5. Update `E:\GitHub总索引` with the public repository URL and push record.

## Current Public Boundary

ChineseASR is a local ASR orchestration project. It does not provide model weights, sample user recordings, benchmark truth data, or generated transcript outputs in the public repository.

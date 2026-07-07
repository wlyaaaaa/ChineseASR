# ASR Smart API Design

## Goal

Add a local-only API and smart PowerShell wrapper so Codex can submit ASR work without blocking the terminal for long-running model inference.

## Design

The service runs on `127.0.0.1` and is started by `scripts\asr-smart.ps1` when needed. The wrapper submits a transcription job, polls for a short bounded window, and returns either final output paths or a `job_id` with a follow-up status command.

The API uses a single GPU worker queue. Each job launches the existing CLI in a child Python process instead of running ASR inside the HTTP request thread. This keeps HTTP requests fast, makes cancellation possible, and isolates model memory from the API process.

## API

- `GET /health`: service status, active job, queue length, GPU conflict report.
- `GET /jobs`: recent jobs.
- `GET /jobs/{job_id}`: job status and output paths.
- `POST /jobs/transcribe`: submit `quick` or `strict` transcription.
- `POST /jobs/{job_id}/cancel`: cancel queued or running work.

## Conflict Policy

The service serializes jobs from this project and deduplicates identical queued/running/completed requests. Before accepting a new job, it checks `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`. If foreign CUDA compute processes are using the GPU, submission is blocked by default with status `blocked`. Users can override with `allow_gpu_conflicts=true` or `scripts\asr-smart.ps1 -AllowGpuConflicts`.

This prevents accidental collisions with Ollama, LocalOCR, LM Studio, other ASR runs, or another Python model process. The service never kills unrelated model processes automatically.

## Wrapper Behavior

`scripts\asr-smart.ps1` defaults to a bounded wait:

```powershell
.\scripts\asr-smart.ps1 -Audio E:\audio.wav -Mode strict -WaitSec 15
```

If the job finishes within `WaitSec`, the wrapper prints output paths. If not, it prints JSON containing `job_id`, `status`, `out_dir`, and the next status command, then exits without waiting for the model to finish.

## Out Of Scope

- Public network service.
- GUI.
- Multi-GPU scheduling.
- Long-audio chunking and resume.
- LLM arbitration.

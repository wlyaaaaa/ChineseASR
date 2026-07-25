from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
import wave
from contextlib import nullcontext
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import parse_qs, urlparse

from .gpu_broker import GpuBrokerConflict, GpuBrokerLease


TERMINAL_STATUSES = {"succeeded", "failed", "canceled", "blocked"}
OBSERVER_LIST_SCHEMA = "local-ai-observer.jobs.v1"
OBSERVER_JOB_SCHEMA = "local-ai-observer.job.v1"
OBSERVER_SERVICE = "chinese-asr"
OBSERVER_JOB_ID_PATTERN = re.compile(r"\A\d{8}-\d{6}-[0-9a-f]{8}\Z")
OBSERVER_MODEL_PATTERN = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
OBSERVER_MANIFEST_LIMIT_BYTES = 8 * 1024 * 1024
OBSERVER_DEFAULT_LIMIT = 50
OBSERVER_MAX_LIMIT = 200
OBSERVER_STATES = {"queued", "running", "succeeded", "failed", "canceled", "blocked"}
OBSERVER_STAGES = {
    "queued",
    "gpu_conflict",
    "running_command",
    "finished",
    "failed",
    "gpu_broker_conflict",
    "canceled",
}


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    process_name: str
    used_memory_mib: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "pid": self.pid,
            "process_name": self.process_name,
            "used_memory_mib": self.used_memory_mib,
        }


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class JobRequest:
    audio: Path
    mode: str
    engine: str | None
    primary_engine: str | None
    secondary_engine: str | None
    device: str
    out_root: Path
    cache_dir: Path | None = None
    chunk_sec: int = 300
    overlap_sec: int = 1
    force: bool = False
    allow_gpu_conflicts: bool = False

    @classmethod
    def from_payload(cls, payload: dict, root: Path, default_out_root: Path | None = None) -> "JobRequest":
        if "audio" not in payload:
            raise ValueError("Missing required field: audio")

        audio = Path(str(payload["audio"])).expanduser().resolve()
        if not audio.exists():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        mode = str(payload.get("mode", "strict")).strip().lower()
        if mode not in {"strict", "quick", "long-strict"}:
            raise ValueError("mode must be 'strict', 'quick', or 'long-strict'")

        fallback_out_root = default_out_root or (root / "outputs" / "api")
        out_root = Path(str(payload.get("out_root") or payload.get("out_dir") or fallback_out_root)).expanduser()
        cache_value = payload.get("cache_dir")
        cache_dir = Path(str(cache_value)).expanduser().resolve() if cache_value else None

        return cls(
            audio=audio,
            mode=mode,
            engine=_optional_str(payload.get("engine")),
            primary_engine=_optional_str(payload.get("primary_engine")),
            secondary_engine=_optional_str(payload.get("secondary_engine")),
            device=str(payload.get("device", "cuda:0")),
            out_root=out_root.resolve(),
            cache_dir=cache_dir,
            chunk_sec=int(payload.get("chunk_sec", 300)),
            overlap_sec=int(payload.get("overlap_sec", 1)),
            force=bool(payload.get("force", False)),
            allow_gpu_conflicts=bool(payload.get("allow_gpu_conflicts", False)),
        )

    def fingerprint(self) -> str:
        stat = self.audio.stat()
        payload = {
            "audio": str(self.audio),
            "audio_size": stat.st_size,
            "audio_mtime_ns": stat.st_mtime_ns,
            "mode": self.mode,
            "engine": self.engine,
            "primary_engine": self.primary_engine,
            "secondary_engine": self.secondary_engine,
            "device": self.device,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "out_root": str(self.out_root),
            "chunk_sec": self.chunk_sec,
            "overlap_sec": self.overlap_sec,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "audio": str(self.audio),
            "mode": self.mode,
            "engine": self.engine,
            "primary_engine": self.primary_engine,
            "secondary_engine": self.secondary_engine,
            "device": self.device,
            "out_root": str(self.out_root),
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "chunk_sec": self.chunk_sec,
            "overlap_sec": self.overlap_sec,
            "force": self.force,
            "allow_gpu_conflicts": self.allow_gpu_conflicts,
        }


@dataclass
class Job:
    job_id: str
    request: JobRequest
    out_dir: Path
    command: list[str]
    status: str = "queued"
    stage: str = "queued"
    message: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None
    process_id: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    conflicts: list[GpuProcess] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "request": self.request.to_dict(),
            "out_dir": str(self.out_dir),
            "command": self.command,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "process_id": self.process_id,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "outputs": self.outputs,
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
        }


ProcessRunner = Callable[[Job], ProcessResult]
GpuProcessDetector = Callable[[], list[GpuProcess]]
CurrentProcessIds = Callable[[], set[int]]
GpuLeaseFactory = Callable[[str], object]


class TranscriptionService:
    def __init__(
        self,
        root: Path,
        *,
        default_out_root: Path | None = None,
        gpu_process_detector: GpuProcessDetector = None,
        current_process_ids: CurrentProcessIds = None,
        process_runner: ProcessRunner = None,
        gpu_lease_factory: GpuLeaseFactory = None,
        autostart: bool = True,
    ) -> None:
        self.root = root.resolve()
        self.default_out_root = (default_out_root or (self.root / "outputs" / "api")).resolve()
        self._gpu_process_detector = gpu_process_detector or detect_gpu_processes
        self._current_process_ids = current_process_ids or self._owned_process_ids
        self._process_runner = process_runner or self._run_subprocess
        if gpu_lease_factory is not None:
            self._gpu_lease_factory = gpu_lease_factory
        elif process_runner is None:
            self._gpu_lease_factory = lambda owner: GpuBrokerLease(owner)
        else:
            self._gpu_lease_factory = lambda _owner: nullcontext()
        self._jobs: dict[str, Job] = {}
        self._fingerprints: dict[str, str] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.RLock()
        self._active_job_id: str | None = None
        self._processes: dict[str, subprocess.Popen] = {}
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        if autostart:
            self.start()

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._worker_loop, name="zh-asr-worker", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker:
            self._worker.join(timeout=2)

    def submit(self, request: JobRequest) -> tuple[Job, bool]:
        fingerprint = request.fingerprint()
        with self._lock:
            if not request.force:
                existing_id = self._fingerprints.get(fingerprint)
                existing = self._jobs.get(existing_id or "")
                if existing and existing.status in {"queued", "running", "succeeded"}:
                    return existing, True

            conflicts = [] if request.allow_gpu_conflicts else self._foreign_gpu_processes()
            job = self._new_job(request, conflicts=conflicts)
            self._jobs[job.job_id] = job
            self._fingerprints[fingerprint] = job.job_id
            if conflicts:
                job.status = "blocked"
                job.stage = "gpu_conflict"
                job.message = _format_conflict_message(conflicts)
                job.finished_at = time.time()
                job.updated_at = job.finished_at
            else:
                self._queue.put(job.job_id)
            return job, False

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def observer_jobs(self, limit: int | str | None = None) -> dict:
        observed_at = time.time()
        bounded_limit = _observer_limit(limit)
        with self._lock:
            snapshots = [copy(job) for job in self._jobs.values()]
        jobs = sorted(snapshots, key=lambda job: job.created_at, reverse=True)[:bounded_limit]
        model_config = _load_observer_model_config() if jobs else None
        projected = [
            _observer_job_payload(job, observed_at, model_config=model_config)
            for job in jobs
        ]
        return {
            "schema": OBSERVER_LIST_SCHEMA,
            "service": OBSERVER_SERVICE,
            "observed_utc": _utc_timestamp(observed_at),
            "jobs": projected,
        }

    def observer_job(self, job_id: str) -> dict | None:
        if not OBSERVER_JOB_ID_PATTERN.fullmatch(job_id):
            return None
        observed_at = time.time()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            snapshot = copy(job)
        model_config = _load_observer_model_config()
        projected = _observer_job_payload(
            snapshot,
            observed_at,
            model_config=model_config,
        )
        return {
            "schema": OBSERVER_JOB_SCHEMA,
            "service": OBSERVER_SERVICE,
            "observed_utc": _utc_timestamp(observed_at),
            "job": projected,
        }

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            if job.status in TERMINAL_STATUSES:
                return job
            job.status = "canceled"
            job.stage = "canceled"
            job.message = "Canceled by request."
            job.finished_at = time.time()
            job.updated_at = job.finished_at
            process = self._processes.get(job_id)
            if process and process.poll() is None:
                process.terminate()
            return job

    def health(self) -> dict:
        with self._lock:
            queued = sum(1 for job in self._jobs.values() if job.status == "queued")
            active_job = self._jobs.get(self._active_job_id or "")
        conflicts = self._foreign_gpu_processes()
        return {
            "status": "ok",
            "active_job_id": active_job.job_id if active_job else None,
            "active_job_status": active_job.status if active_job else None,
            "queue_length": queued,
            "gpu_conflicts": [conflict.to_dict() for conflict in conflicts],
        }

    def run_next_job(self) -> bool:
        try:
            job_id = self._queue.get_nowait()
        except queue.Empty:
            return False
        try:
            self._process_job(job_id)
        finally:
            self._queue.task_done()
        return True

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._process_job(job_id)
            finally:
                self._queue.task_done()

    def _process_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job.status == "canceled":
                return
            job.status = "running"
            job.stage = "running_command"
            job.started_at = time.time()
            job.updated_at = job.started_at
            self._active_job_id = job_id

        try:
            use_gpu_broker = (
                job.request.device.lower().startswith(("cuda", "gpu"))
                and not job.request.allow_gpu_conflicts
            )
            lease = self._gpu_lease_factory("chineseasr") if use_gpu_broker else nullcontext()
            with lease:
                result = self._process_runner(job)
            with self._lock:
                if job.status == "canceled":
                    return
                job.returncode = result.returncode
                job.stdout_tail = _tail(result.stdout)
                job.stderr_tail = _tail(result.stderr)
                job.outputs = _collect_outputs(
                    job.out_dir,
                    job.request.mode,
                    job.request.engine,
                    primary_engine=job.request.primary_engine,
                    secondary_engine=job.request.secondary_engine,
                )
                job.status = "succeeded" if result.returncode == 0 else "failed"
                job.stage = "finished" if result.returncode == 0 else "failed"
                job.message = "Completed." if result.returncode == 0 else "Transcription command failed."
                job.finished_at = time.time()
                job.updated_at = job.finished_at
        except GpuBrokerConflict as exc:
            with self._lock:
                if job.status != "canceled":
                    job.status = "blocked"
                    job.stage = "gpu_broker_conflict"
                    job.message = str(exc)
                    job.finished_at = time.time()
                    job.updated_at = job.finished_at
        except Exception as exc:
            with self._lock:
                if job.status != "canceled":
                    job.status = "failed"
                    job.stage = "failed"
                    job.message = f"{type(exc).__name__}: {exc}"
                    job.finished_at = time.time()
                    job.updated_at = job.finished_at
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def _new_job(self, request: JobRequest, conflicts: list[GpuProcess]) -> Job:
        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        out_dir = request.out_root / job_id
        command = self._build_command(request, out_dir)
        return Job(job_id=job_id, request=request, out_dir=out_dir, command=command, conflicts=list(conflicts))

    def _build_command(self, request: JobRequest, out_dir: Path) -> list[str]:
        if request.mode == "long-strict":
            command = [
                sys.executable,
                "-m",
                "zh_asr",
                "long",
                str(request.audio),
                "--device",
                request.device,
                "--out-dir",
                str(out_dir),
                "--chunk-sec",
                str(request.chunk_sec),
                "--overlap-sec",
                str(request.overlap_sec),
            ]
            if request.primary_engine:
                command.extend(["--primary-engine", request.primary_engine])
            if request.secondary_engine:
                command.extend(["--secondary-engine", request.secondary_engine])
        elif request.mode == "strict":
            command = [
                sys.executable,
                "-m",
                "zh_asr",
                "strict",
                str(request.audio),
                "--device",
                request.device,
                "--out-dir",
                str(out_dir),
            ]
            if request.primary_engine:
                command.extend(["--primary-engine", request.primary_engine])
            if request.secondary_engine:
                command.extend(["--secondary-engine", request.secondary_engine])
        else:
            command = [
                sys.executable,
                "-m",
                "zh_asr",
                "transcribe",
                str(request.audio),
                "--device",
                request.device,
                "--out-dir",
                str(out_dir),
            ]
            if request.engine:
                command.extend(["--engine", request.engine])
        if request.cache_dir:
            command.extend(["--cache-dir", str(request.cache_dir)])
        return command

    def _run_subprocess(self, job: Job) -> ProcessResult:
        job.out_dir.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            job.command,
            cwd=self.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        with self._lock:
            job.process_id = process.pid
            self._processes[job.job_id] = process
            job.updated_at = time.time()
        stdout, stderr = process.communicate()
        return ProcessResult(returncode=process.returncode, stdout=stdout, stderr=stderr)

    def _foreign_gpu_processes(self) -> list[GpuProcess]:
        owned = self._current_process_ids()
        managed_names = ("ollama", "llama-server")
        return [
            process
            for process in self._gpu_process_detector()
            if process.pid not in owned
            and not any(name in process.process_name.lower() for name in managed_names)
        ]

    def _owned_process_ids(self) -> set[int]:
        with self._lock:
            pids = {os.getpid()}
            pids.update(process.pid for process in self._processes.values() if process.pid)
            return pids


def detect_gpu_processes() -> list[GpuProcess]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    processes = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        process = _parse_gpu_process_line(line)
        if process:
            processes.append(process)
    return processes


def _parse_gpu_process_line(line: str) -> GpuProcess | None:
    parts = [part.strip() for part in line.split(",")]
    try:
        pid = int(parts[0])
    except (IndexError, ValueError):
        return None
    process_name = parts[1] if len(parts) > 1 else "unknown"
    memory = (parts[2] if len(parts) > 2 else "0").replace("MiB", "").strip()
    try:
        used_memory_mib = int(memory)
    except ValueError:
        return None
    return GpuProcess(pid=pid, process_name=process_name, used_memory_mib=used_memory_mib)


def _collect_outputs(
    out_dir: Path,
    mode: str,
    engine: str | None,
    primary_engine: str | None = None,
    secondary_engine: str | None = None,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if not out_dir.exists():
        return outputs
    if mode == "long-strict":
        for key, name in {
            "transcript": "transcript.md",
            "audit": "audit.md",
            "metrics": "metrics.json",
            "manifest": "manifest.json",
        }.items():
            path = out_dir / name
            if path.exists():
                outputs[key] = str(path)
    elif mode == "strict":
        mapping = {
            "final": "*.strict.md",
            "audit": "*.strict.audit.md",
            "audit_json": "*.strict.audit.json",
        }
        for key, pattern in mapping.items():
            matches = sorted(out_dir.glob(pattern))
            if matches:
                outputs[key] = str(matches[0])
        primary_name, secondary_name = _strict_engine_names(primary_engine, secondary_engine)
        raw_matches = sorted(out_dir.glob("*.raw.json"))
        primary_raw = _find_raw_json(raw_matches, primary_name)
        secondary_raw = _find_raw_json(raw_matches, secondary_name)
        if primary_raw is None and raw_matches:
            primary_raw = raw_matches[0]
        if secondary_raw is None:
            secondary_raw = next((path for path in raw_matches if path != primary_raw), None)
        if primary_raw:
            outputs["primary_raw_json"] = str(primary_raw)
        if secondary_raw:
            outputs["secondary_raw_json"] = str(secondary_raw)
    else:
        pattern = f"*.{engine}.md" if engine else "*.md"
        matches = sorted(out_dir.glob(pattern))
        if matches:
            outputs["markdown"] = str(matches[0])
        json_matches = sorted(out_dir.glob("*.raw.json"))
        if json_matches:
            outputs["raw_json"] = str(json_matches[0])
    return outputs


def _strict_engine_names(primary_engine: str | None, secondary_engine: str | None) -> tuple[str | None, str | None]:
    if primary_engine and secondary_engine:
        return primary_engine, secondary_engine
    try:
        from .config import load_model_config

        config = load_model_config()
        return primary_engine or config.strict_primary_engine, secondary_engine or config.strict_secondary_engine
    except Exception:
        return primary_engine, secondary_engine


def _find_raw_json(matches: list[Path], engine: str | None) -> Path | None:
    if not engine:
        return None
    suffix = f".{engine}.raw.json"
    return next((path for path in matches if path.name.endswith(suffix)), None)


def _format_conflict_message(conflicts: Iterable[GpuProcess]) -> str:
    details = ", ".join(
        f"{process.process_name} pid={process.pid} memory={process.used_memory_mib}MiB"
        for process in conflicts
    )
    return f"GPU conflict detected: {details}"


def _optional_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _tail(text: str, max_chars: int = 4000) -> str:
    return text[-max_chars:] if len(text) > max_chars else text


def _observer_limit(value: int | str | None) -> int:
    try:
        requested = OBSERVER_DEFAULT_LIMIT if value is None else int(value)
    except (TypeError, ValueError):
        requested = OBSERVER_DEFAULT_LIMIT
    return max(1, min(requested, OBSERVER_MAX_LIMIT))


def _load_observer_model_config():
    try:
        from .config import load_model_config

        return load_model_config()
    except Exception:
        return None


def _observer_job_payload(job: Job, observed_at: float, *, model_config) -> dict:
    elapsed_ms, timing_status = _observer_elapsed(job, observed_at)
    progress, manifest_audio_seconds = _observer_progress(job)
    audio_seconds = manifest_audio_seconds or _wav_audio_seconds(job.request.audio)
    throughput = {
        "status": "unavailable",
        "rtf": None,
        "audio_seconds": audio_seconds,
    }
    if (
        job.status in TERMINAL_STATUSES
        and elapsed_ms is not None
        and audio_seconds is not None
        and audio_seconds > 0
    ):
        throughput = {
            "status": "measured",
            "rtf": round((elapsed_ms / 1000) / audio_seconds, 6),
            "audio_seconds": audio_seconds,
        }

    return {
        "job_id": job.job_id,
        "state": job.status if job.status in OBSERVER_STATES else "unknown",
        "stage": job.stage if job.stage in OBSERVER_STAGES else "unknown",
        "mode": job.request.mode,
        "model": _observer_model(job.request, model_config),
        "progress": progress,
        "timing": {
            "status": timing_status,
            "started_utc": _utc_timestamp(job.started_at),
            "updated_utc": _utc_timestamp(job.updated_at),
            "elapsed_ms": elapsed_ms,
        },
        "tokens": {
            "status": "not_applicable",
            "input": None,
            "output": None,
            "tps": None,
        },
        "throughput": throughput,
    }


def _observer_model(request: JobRequest, config) -> str:
    if config is None:
        return "unavailable"

    def configured(value: str | None, fallback: str) -> str | None:
        selected = value or fallback
        if selected not in config.engines or not OBSERVER_MODEL_PATTERN.fullmatch(selected):
            return None
        return selected

    if request.mode == "quick":
        return configured(request.engine, config.default_engine) or "unknown"

    primary = configured(request.primary_engine, config.strict_primary_engine)
    secondary = configured(request.secondary_engine, config.strict_secondary_engine)
    if primary is None or secondary is None:
        return "unknown"
    return f"{primary} + {secondary}"


def _observer_elapsed(job: Job, observed_at: float) -> tuple[int | None, str]:
    if job.started_at is None:
        return None, "not_started"
    end = job.finished_at if job.finished_at is not None else observed_at
    elapsed_ms = max(0, round((end - job.started_at) * 1000))
    if job.finished_at is not None or job.status in TERMINAL_STATUSES:
        return elapsed_ms, "complete"
    return elapsed_ms, "running"


def _observer_progress(job: Job) -> tuple[dict, float | None]:
    unavailable = {
        "status": "unavailable",
        "completed": None,
        "total": None,
        "unit": None,
    }
    if job.request.mode != "long-strict":
        return unavailable, None

    manifest_path = job.out_dir / "manifest.json"
    try:
        with manifest_path.open("rb") as handle:
            raw_manifest = handle.read(OBSERVER_MANIFEST_LIMIT_BYTES + 1)
        if len(raw_manifest) > OBSERVER_MANIFEST_LIMIT_BYTES:
            return unavailable, None
        manifest = json.loads(raw_manifest.decode("utf-8"))
        chunks = manifest.get("chunks")
        if not isinstance(chunks, list):
            return unavailable, None
        completed = sum(
            1
            for chunk in chunks
            if isinstance(chunk, dict) and chunk.get("status") in {"succeeded", "failed"}
        )
        end_points = [
            float(chunk.get("end_ms"))
            for chunk in chunks
            if (
                isinstance(chunk, dict)
                and isinstance(chunk.get("end_ms"), (int, float))
                and not isinstance(chunk.get("end_ms"), bool)
                and math.isfinite(float(chunk.get("end_ms")))
                and float(chunk.get("end_ms")) >= 0
            )
        ]
        audio_seconds = round(max(end_points) / 1000, 3) if end_points else None
        return {
            "status": "available",
            "completed": completed,
            "total": len(chunks),
            "unit": "chunks",
        }, audio_seconds
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return unavailable, None


def _wav_audio_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            if rate <= 0:
                return None
            return round(handle.getnframes() / rate, 3)
    except (OSError, EOFError, wave.Error):
        return None


def _utc_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return (
        datetime.fromtimestamp(value, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def create_handler(service: TranscriptionService) -> type[BaseHTTPRequestHandler]:
    class AsrRequestHandler(BaseHTTPRequestHandler):
        server_version = "ChineseASR/0.1"

        def do_GET(self) -> None:
            path = "/"
            try:
                parsed = urlparse(self.path)
                path = parsed.path.rstrip("/") or "/"
                if path == "/observer/jobs":
                    query = parse_qs(parsed.query)
                    limit = (query.get("limit") or [None])[0]
                    self._send_json(200, service.observer_jobs(limit=limit))
                    return
                if path.startswith("/observer/jobs/"):
                    job_id = path.removeprefix("/observer/jobs/")
                    payload = service.observer_job(job_id)
                    if payload is None:
                        self._send_json(404, {"error": "job_not_found"})
                        return
                    self._send_json(200, payload)
                    return
                if path == "/health":
                    self._send_json(200, service.health())
                    return
                if path == "/jobs":
                    self._send_json(200, {"jobs": [job.to_dict() for job in service.list_jobs()]})
                    return
                if path.startswith("/jobs/"):
                    job_id = path.split("/", 2)[2]
                    job = service.get_job(job_id)
                    if not job:
                        self._send_json(404, {"error": f"Job not found: {job_id}"})
                        return
                    self._send_json(200, {"job": job.to_dict()})
                    return
                self._send_json(404, {"error": f"Unknown endpoint: {path}"})
            except Exception as exc:
                if path.startswith("/observer/"):
                    self._send_json(500, {"error": "observer_unavailable"})
                    return
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self) -> None:
            try:
                path = urlparse(self.path).path.rstrip("/") or "/"
                payload = self._read_json()
                if path == "/jobs/transcribe":
                    request = JobRequest.from_payload(
                        payload,
                        root=service.root,
                        default_out_root=service.default_out_root,
                    )
                    job, deduplicated = service.submit(request)
                    self._send_json(202, {"job": job.to_dict(), "deduplicated": deduplicated})
                    return
                if path.startswith("/jobs/") and path.endswith("/cancel"):
                    job_id = path.split("/")[2]
                    job = service.cancel(job_id)
                    self._send_json(200, {"job": job.to_dict()})
                    return
                self._send_json(404, {"error": f"Unknown endpoint: {path}"})
            except (FileNotFoundError, ValueError) as exc:
                self._send_json(400, {"error": str(exc)})
            except KeyError as exc:
                self._send_json(404, {"error": f"Job not found: {exc.args[0]}"})
            except Exception as exc:
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, format: str, *args) -> None:
            return

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length == 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw)

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return AsrRequestHandler


def serve_api(host: str, port: int, state_dir: Path, root: Path) -> int:
    state_dir.mkdir(parents=True, exist_ok=True)
    service = TranscriptionService(root=root, default_out_root=state_dir, autostart=True)
    server = ThreadingHTTPServer((host, port), create_handler(service))
    try:
        print(f"ASR API ready at http://{host}:{server.server_port}")
        print(f"State directory: {state_dir}")
        server.serve_forever()
    finally:
        server.server_close()
        service.stop()
    return 0

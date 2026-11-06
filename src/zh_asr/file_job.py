"""One-file, cloud-first submission/polling over the existing ASR entrypoints."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from .config import project_root
from .model_lifecycle import write_json_atomic
from .process_control import managed_popen_kwargs
from .result_writer import file_sha256
from .transcript_readback import read_transcript_readback

ACCESS_CHECK = Path(r"C:\ProgramData\PCConfig\AuthorityHost\tools\personal_data_access.py")
API = "http://127.0.0.1:18666"
BROKER = "http://127.0.0.1:32100/_gpu_broker/status"


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _json_process(command: list[str], root: Path, timeout: int = 120) -> dict:
    done = subprocess.run(command, cwd=root, capture_output=True, text=True,
                          encoding="utf-8-sig", timeout=timeout,
                          **managed_popen_kwargs())
    return json.loads(done.stdout)


def _access_open() -> bool:
    try:
        return _json_process([sys.executable, "-B", str(ACCESS_CHECK),
                              "--privacy-level", "factor"], project_root(), 40).get("status") == "pass"
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


@contextmanager
def _claim(path: Path, *, wait: bool = False):
    """The worker holds this OS lock; a poll never waits for cloud inference."""
    with path.open("a+b") as handle:
        if not handle.tell():
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + (10 if wait else 0)
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    yield False
                    return
                time.sleep(0.05)
        try:
            yield True
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        return False
    try:
        return kernel.WaitForSingleObject(handle, 0) == 258
    finally:
        kernel.CloseHandle(handle)


def _http(url: str) -> dict:
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=5) as response:
        return json.load(response)


def _gpu_wait_reason() -> str:
    try:
        state = _http(BROKER)
        if state.get("ok") is not True:
            return "gpu_broker_unavailable"
        if state.get("ollama_session") or state.get("active_ollama_requests") or state.get("waiting_ollama_requests"):
            return "ollama_gpu_active"
        for lease in state.get("leases", []):
            if lease.get("owner") not in {"localocr", "localocr-cli"}:
                return "gpu_lease_active:" + str(lease.get("owner") or "unknown")
        return ""
    except (OSError, ValueError):
        return "gpu_broker_unavailable"


def _reuse(digest: str, root: Path) -> dict:
    result = read_transcript_readback(digest, root / "outputs/api/jobs.json", root / "outputs/cloud-jobs")
    artifact = result.get("artifact", {})
    transcript = artifact.get("transcript", {}).get("path")
    cloud = artifact.get("raw_artifact_role") == "cloud_result"
    readable = cloud or (isinstance(transcript, str) and Path(transcript).is_file())
    if readable and (result.get("status") == "ok" or result.get("gap", {}).get("code") == "timestamps_unavailable"):
        raw = artifact.get("raw_json", {}).get("path")
        precise = any(str(part.get("timestamp_granularity", "")).startswith("sentence")
                      for part in result.get("segments", []))
        return dict(status="succeeded", route="cloud" if cloud else "local",
                    job_id=artifact.get("job_id"), out_dir=str(Path(raw).parent) if raw else "",
                    timestamps="available" if precise else "unavailable", artifact=artifact,
                    evidence_status=result.get("evidence_status", "unknown"),
                    reason="reused_existing_transcript", gap=result.get("gap"))
    pending = result.get("lookup_scope", {}).get("pending_jobs", [])
    if pending:
        return dict(status="running", route="local", local_job_id=pending[0]["job_id"],
                    job_id=pending[0]["job_id"], reason="existing_local_job")
    return {"status": "unavailable", "gap": result.get("gap", {})}


def _saved_success(state: dict) -> bool:
    artifact = state.get("artifact", {})
    refs = [ref for ref in artifact.values() if isinstance(ref, dict) and ref.get("sha256")]
    try:
        transcript = artifact.get("transcript", {}).get("path")
        return bool(refs) and (not transcript or Path(transcript).is_file()) and all(
            Path(ref["path"]).stat().st_size == ref["size_bytes"]
            and file_sha256(Path(ref["path"])) == ref["sha256"] for ref in refs)
    except (OSError, KeyError, TypeError):
        return False


def _local_status(state: dict, root: Path) -> dict:
    try:
        job = _http(API + "/jobs/" + state["local_job_id"])["job"]
    except (OSError, ValueError, KeyError):
        # A stopped API does not authorize submitting the same audio again.
        return {**state, "status": "queued", "error": "local_status_unavailable"}
    state.update(job_id=job["job_id"], out_dir=job.get("out_dir"), outputs=job.get("outputs", {}))
    status = job["status"]
    if status == "blocked":
        state.pop("local_job_id", None)
        return {**state, "status": "waiting_gpu", "error": job.get("message", "gpu_broker_conflict")}
    if status == "succeeded":
        reused = _reuse(state["source_audio_sha256"], root)
        if reused.get("status") == "succeeded":
            state.update(status="succeeded", timestamps=reused["timestamps"],
                         artifact=reused.get("artifact", {}),
                         evidence_status=reused.get("evidence_status", "unknown"), error="")
        else:
            state.update(status="failed", transcript_completed=True,
                         error="transcript_readback_gap", gap=reused.get("gap", {}))
    else:
        state.update(status="failed" if status in {"failed", "canceled"} else status,
                     error=job.get("message", "") if status in {"failed", "canceled"} else "")
        if state["status"] == "failed":
            state.pop("local_job_id", None)
    return state


def _work(audio: Path, state: dict, root: Path, path: Path) -> dict:
    if not state.get("cloud_attempted"):
        # Persist before sending: even an interrupted worker must not upload again.
        state.update(status="running", route="cloud", cloud_attempted=True,
                     reason="cloud_attempt_started")
        write_json_atomic(path, state)
        try:
            cloud = _json_process(["pwsh", "-NoProfile", "-NonInteractive", "-File",
                str(root / "scripts/asr-professional-cloud.ps1"), "-Audio", str(audio),
                "-CloudFirst", "-Json"], root, 86400)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            cloud = {"error_code": "cloud_entry_" + type(exc).__name__}
        if (cloud.get("status") == "succeeded" and cloud.get("text", "").strip()
                and cloud.get("source_audio_sha256") == state["source_audio_sha256"]):
            state.update(cloud_completed=True, job_id=cloud.get("job_id"),
                         out_dir=str(root / "outputs/cloud-jobs"), result_path=cloud.get("result_path"),
                         reason="cloud_transcription_completed")
            write_json_atomic(path, state)
            reused = _reuse(state["source_audio_sha256"], root)
            if reused.get("status") == "succeeded":
                state.update(status="succeeded", timestamps=reused["timestamps"],
                             artifact=reused.get("artifact", {}), error="")
            else:
                state.update(status="failed", error="transcript_readback_gap", gap=reused.get("gap", {}))
            return state
        reason = str(cloud.get("pause_reason") or cloud.get("error_code") or "cloud_failed")
        state.update(route="local", reason=reason, fallback_reason=reason,
                     cloud_result_path=cloud.get("result_path", ""))
        write_json_atomic(path, state)
    if not _access_open():
        return {**state, "status": "queued", "reason": "personal_access_required"}
    state.update(route="local", reason=state.get("fallback_reason", "cloud_attempt_interrupted"))
    wait_reason = _gpu_wait_reason()
    if wait_reason:
        return {**state, "status": "waiting_gpu", "error": wait_reason}
    try:
        local = _json_process(["pwsh", "-NoProfile", "-NonInteractive", "-File",
            str(root / "scripts/asr-smart.ps1"), "-Audio", str(audio), "-Mode", "quick",
            "-Engine", "paraformer", "-WaitSec", "0", "-StartupTimeoutSec", "120", "-Json"], root, 150)
        state.update(local_job_id=local["job_id"], job_id=local["job_id"],
                     status=local["status"], out_dir=local.get("out_dir"), outputs=local.get("outputs", {}))
        return _local_status(state, root)
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        return {**state, "status": "failed", "error": "local_entry_" + type(exc).__name__}


def file_job(audio: Path, *, worker: bool = False, root: Path | None = None) -> dict:
    """Only an original path is public input; repeated calls poll or reuse it."""
    root = root or project_root()
    base = dict(schema="zh_asr.file-job.v1", status="queued", route="cloud",
                timestamps="unavailable", reason="", error="")
    if not _access_open():
        return {**base, "reason": "personal_access_required"}
    audio = audio.expanduser().resolve()
    try:
        digest = file_sha256(audio)
    except OSError as exc:
        return {**base, "status": "failed", "error": "audio_" + type(exc).__name__}
    path = root / "outputs/api/file-jobs" / (digest + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _claim(path.with_suffix(".lock"), wait=worker) as claimed:
        state = {**base, **_read(path), "source_audio_sha256": digest}
        if not claimed:
            return state
        if state["status"] == "succeeded" and _saved_success(state):
            return state
        if state["status"] == "succeeded" or state.get("cloud_completed") or state.get("transcript_completed"):
            # A completed cloud transcript with a readback gap is rechecked,
            # never uploaded again or recognized a second time just for timing.
            reused = _reuse(digest, root)
            if reused.get("status") == "succeeded":
                state.update(reused, error="")
            else:
                state.update(status="failed", transcript_completed=True,
                             error="transcript_readback_gap", gap=reused.get("gap", {}))
        elif worker:
            try:
                state = _work(audio, state, root, path)
            except Exception as exc:
                state.update(status="failed", error="file_worker_" + type(exc).__name__)
        elif state.get("local_job_id"):
            state = _local_status(state, root)
        elif state.get("worker_pid") and state["status"] == "queued" and _alive(state["worker_pid"]):
            return state
        else:
            existing = _reuse(digest, root)
            if existing.get("status") in {"succeeded", "running"}:
                state.update(existing)
            else:
                state.update(status="queued", error="")
                write_json_atomic(path, state)
                try:
                    process = subprocess.Popen([sys.executable, "-B", "-m", "zh_asr", "file-job",
                        "--audio", str(audio), "--worker"], cwd=root, stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **managed_popen_kwargs())
                    state["worker_pid"] = process.pid
                except OSError as exc:
                    state.update(status="failed", error="worker_start_" + type(exc).__name__)
        write_json_atomic(path, state)
        return state

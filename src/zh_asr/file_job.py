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
from .process_control import managed_popen_kwargs, terminate_process_tree
from .result_writer import file_sha256
from .transcript_readback import read_transcript_readback

ACCESS_CHECK = Path(r"C:\ProgramData\PCConfig\AuthorityHost\tools\personal_data_access.py")
API = "http://127.0.0.1:18666"
BROKER = "http://127.0.0.1:32100/_gpu_broker/status"
CLOUD_IDLE_SECONDS = 300
LOCAL_TIMEOUT_SECONDS = 3600


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _json_process(command: list[str], root: Path, timeout: int = 120, *, progress_path=None) -> dict:
    if progress_path is None:
        done = subprocess.run(command, cwd=root, capture_output=True, text=True,
                              encoding="utf-8-sig", timeout=timeout, **managed_popen_kwargs())
        return json.loads(done.stdout)
    started = time.monotonic()
    last_progress = time.time()
    process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               **managed_popen_kwargs())
    try:
        while True:
            try:
                stdout, _ = process.communicate(timeout=2)
                return json.loads(stdout.decode("utf-8-sig"))
            except subprocess.TimeoutExpired:
                last_progress = max(last_progress, float(_read(progress_path).get("updated_at", 0)))
                if time.monotonic() - started > timeout or time.time() - last_progress > CLOUD_IDLE_SECONDS:
                    raise subprocess.TimeoutExpired(command, timeout)
    except BaseException:
        terminate_process_tree(process)
        raise


def _cloud_budget(audio: Path) -> int:
    from .audio_quality import probe_duration_ms
    try:
        duration = probe_duration_ms(audio) / 1000
    except (OSError, ValueError, RuntimeError):
        duration = 300
    return int(min(6 * 3600, max(600, 2 * duration + 300)))


def _spawn_worker(audio: Path, root: Path) -> int:
    arguments = [sys.executable, "-B", str(root / "scripts/asr-file-job-worker.py"), str(audio)]
    if os.name != "nt":
        return subprocess.Popen(arguments, cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, **managed_popen_kwargs()).pid
    # WMI is outside the submitting task's Windows Job, including its 25-minute kill limit.
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.is_file():
        raise OSError("pythonw_runtime_unavailable")
    arguments[0] = str(pythonw)
    script = """[Console]::InputEncoding=[Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$d=[Console]::In.ReadToEnd() | ConvertFrom-Json
$r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$d.command;CurrentDirectory=$d.cwd}
if ($r.ReturnValue -ne 0) { throw 'file_worker_start_failed' }
@{pid=[int]$r.ProcessId} | ConvertTo-Json -Compress
"""
    done = subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        input=json.dumps(dict(command=subprocess.list2cmdline(arguments), cwd=str(root))).encode("utf-8"),
        capture_output=True, timeout=20, **managed_popen_kwargs())
    if done.returncode:
        raise OSError("file_worker_start_failed")
    return int(json.loads(done.stdout)["pid"])


def _schedule_worker(state: dict, audio: Path, root: Path) -> dict:
    try:
        state.update(worker_pid=_spawn_worker(audio, root), worker_started_at=time.time())
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        state.update(status="failed", error="worker_start_" + type(exc).__name__, retryable=True)
    return state


def _access_open() -> bool:
    try:
        checked = _json_process([sys.executable, "-B", str(ACCESS_CHECK),
                                 "--privacy-level", "factor"], project_root(), 40)
        return isinstance(checked, dict) and checked.get("status") == "pass"
    except Exception:
        # An unavailable decision is closed, including an unexpected checker failure.
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


def _http(url: str, *, method="GET") -> dict:
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            urllib.request.Request(url, method=method), timeout=5) as response:
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


def _completed_local_bundle(state: dict, root: Path) -> dict:
    """Recover a fully written quick bundle even if API terminal persistence was interrupted."""
    from .result_writer import extract_text
    from .transcript_readback import _inspect_candidate
    try:
        record = next(row for row in _read(root / "outputs/api/jobs.json").get("jobs", [])
                      if row.get("job_id") == state.get("local_job_id"))
        request = record["request"]
        if request.get("mode") != "quick" or request.get("audio_sha256") != state["source_audio_sha256"]:
            return {}
        engine = request.get("resolved_engine") or request.get("engine")
        if engine != "paraformer":
            return {}
        directory = Path(record["out_dir"]).resolve()
        if not directory.is_relative_to((root / "outputs/api").resolve()):
            return {}
        prefix = Path(request["audio"]).stem + ".paraformer"
        raw, objective, transcript = (directory / (prefix + suffix)
                                     for suffix in (".raw.json", ".objective-result.json", ".md"))
        if not transcript.is_file():
            return {}
        candidate, _ = _inspect_candidate(job=record, request_hash=state["source_audio_sha256"],
            output_key="raw_json", raw_value=str(raw), objective_path=objective)
        if candidate is None:
            return {}
        text = extract_text(json.loads(raw.read_text("utf-8"))) or "_No speech text returned._"
        if text not in transcript.read_text("utf-8"):
            return {}
        artifact = {**candidate["artifact"], "transcript": {"path": str(transcript),
                    "size_bytes": transcript.stat().st_size, "sha256": file_sha256(transcript)}}
        return dict(status="succeeded", route="local", job_id=record["job_id"], out_dir=str(directory),
                    artifact=artifact, evidence_status=candidate["evidence_status"],
                    timestamps="available" if candidate["timed_segments"] else "unavailable",
                    reason="reused_interrupted_local_bundle", error="")
    except (OSError, ValueError, TypeError, KeyError, StopIteration):
        return {}


def _local_status(state: dict, root: Path) -> dict:
    reused = _reuse(state["source_audio_sha256"], root)
    if reused.get("status") == "succeeded":
        return {**state, **reused, "error": ""}
    recovered = _completed_local_bundle(state, root)
    if recovered:
        return {**state, **recovered}
    try:
        job = _http(API + "/jobs/" + state["local_job_id"])["job"]
    except (OSError, ValueError, KeyError):
        state.pop("local_job_id", None)
        return {**state, "status": "queued", "route": "local", "cloud_attempted": True,
                "reason": "local_api_recovering", "error": "local_status_unavailable", "recover_local": True}
    request = job.get("request") or {}
    if (job.get("job_id") != state["local_job_id"] or
            request.get("audio_sha256", state["source_audio_sha256"]) != state["source_audio_sha256"]):
        state.pop("local_job_id", None)
        return {**state, "status": "failed", "error": "local_job_binding_invalid", "retryable": True}
    state.update(job_id=job["job_id"], out_dir=job.get("out_dir"), outputs=job.get("outputs", {}))
    status = job["status"]
    if status not in {"queued", "running", "blocked", "succeeded", "failed", "canceled"}:
        state.pop("local_job_id", None)
        return {**state, "status": "failed", "error": "local_status_invalid", "retryable": True}
    state.setdefault("local_started_at", job.get("created_at", time.time()))
    if status in {"queued", "running"} and time.time() - state["local_started_at"] > LOCAL_TIMEOUT_SECONDS:
        try:
            if request.get("audio_sha256") == state["source_audio_sha256"]:
                _http(API + "/jobs/" + state["local_job_id"] + "/cancel", method="POST")
        except (OSError, ValueError, KeyError):
            pass
        state.pop("local_job_id", None)
        return {**state, "status": "failed", "error": "local_job_timeout", "retryable": True}
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
    if not state.get("cloud_attempted") or state.get("cloud_resume_pending"):
        # Persist before sending: even an interrupted worker must not upload again.
        resuming = state.pop("cloud_resume_pending", False)
        state.update(status="running", route="cloud", cloud_attempted=True,
                     reason="cloud_attempt_started")
        write_json_atomic(path, state)
        progress_path = path.with_suffix(".progress.json")
        args = ["pwsh", "-NoProfile", "-NonInteractive", "-File",
                str(root / "scripts/asr-professional-cloud.ps1"), "-Audio", str(audio),
                "-CloudFirst", "-PersonalDataGuard", "-ConcurrentWaitSec", "5",
                "-ProgressPath", str(progress_path), "-Json"]
        if resuming and state.get("cloud_job_id"):
            args.extend(["-ResumeJobId", state["cloud_job_id"]])
        try:
            cloud = _json_process(args, root, _cloud_budget(audio), progress_path=progress_path)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            cloud = {"error_code": "cloud_entry_" + type(exc).__name__}
        if cloud.get("error_code") == "personal_access_protection_required":
            return {**state, "status": "queued", "route": "cloud", "cloud_resume_pending": True,
                    "cloud_job_id": cloud.get("job_id"), "reason": "personal_access_required", "error": ""}
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
                     status=local["status"], out_dir=local.get("out_dir"), outputs=local.get("outputs", {}),
                     local_started_at=time.time())
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
            progress = _read(path.with_suffix(".progress.json"))
            since = progress.get("updated_at") or state.get("worker_started_at") or time.time()
            if state.get("route") == "cloud" and time.time() - since > CLOUD_IDLE_SECONDS + 60:
                return {**state, "status": "failed", "error": "cloud_worker_stalled", "retryable": True}
            return {**state, "progress_at": progress.get("updated_at"),
                    "completed_chunks": progress.get("completed_chunks", 0)}
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
            if state.pop("recover_local", False):
                state = _schedule_worker(state, audio, root)
        elif (state.get("worker_pid") and state["status"] == "queued"
                and time.time() - state.get("worker_started_at", 0) < 120 and _alive(state["worker_pid"])):
            return state
        else:
            existing = _reuse(digest, root)
            if existing.get("status") in {"succeeded", "running"}:
                state.update(existing, error="")
                if state.get("local_job_id"):
                    state = _local_status(state, root)
                    if state.pop("recover_local", False):
                        state = _schedule_worker(state, audio, root)
            else:
                state.update(status="queued", error="")
                write_json_atomic(path, state)
                state = _schedule_worker(state, audio, root)
        write_json_atomic(path, state)
        return state

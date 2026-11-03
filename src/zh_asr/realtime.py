"""Ready-only selected-WAV recognition over stdin/stdout JSONL; no listener."""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

from .dictation_worker import DictationWorkerNotReady, ProcessDictationEngine, _engine_main


def _worker(connection, settings):
    # Third-party model progress must never contaminate the JSONL protocol.
    with redirect_stdout(sys.stderr):
        _engine_main(connection, settings)


class RealtimeSession:
    def __init__(self, settings, engine=None):
        self.settings = settings
        self.engine = engine or ProcessDictationEngine(settings, worker_target=_worker)
        self._prepared = False

    def status(self):
        return {"event": "status", "state": "ready" if self._prepared and self.engine.is_ready else "not_ready",
                "model": self.settings.engine, "workerPID": self.engine.worker_pid}

    def prepare(self):
        self._prepared = False
        timings = {}
        for name, operation in (("coldLoad", self.engine.load), ("warmup", self.engine.warmup),
                                ("park", self.engine.park)):
            started = time.monotonic()
            operation()
            timings[name] = time.monotonic() - started
        if not self.engine.is_ready:
            raise DictationWorkerNotReady("Worker exited during prepare")
        self._prepared = True
        return dict(self.status(), event="ready", timings=timings)

    def transcribe(self, request):
        started = time.monotonic()
        timings = dict.fromkeys(("prepare", "activate", "infer", "park"), 0.0)
        result = {"event": "result", "request_id": request.get("request_id"),
                  "state": "failed", "text": "", "model": self.settings.engine,
                  "workerPID": self.engine.worker_pid, "timings": timings,
                  "coverage": {"status": "unknown", "segments": []}}
        stage = "prepare"
        stage_started = started
        deadline = started
        try:
            budget = float(request.get("budget_seconds", 10))
            if not math.isfinite(budget) or not 0 < budget <= 10:
                raise ValueError("budget_seconds must be finite and in (0, 10]")
            deadline = started + budget
            cleanup_reserve = min(0.3, budget / 2)

            def remaining():
                seconds = deadline - time.monotonic() - cleanup_reserve
                if seconds <= 0:
                    raise TimeoutError("Shared request deadline exhausted")
                return seconds

            if not self._prepared or not self.engine.is_ready:
                raise DictationWorkerNotReady("not_ready: explicit prepare required")
            path = Path(request["path"])
            if not path.is_absolute() or path.suffix.lower() != ".wav":
                raise ValueError("Select one absolute WAV path")
            body = path.read_bytes()
            import soundfile as sf
            from .dictation import resample_audio
            with sf.SoundFile(io.BytesIO(body)) as source:
                if source.format not in {"WAV", "WAVEX", "RF64"}:
                    raise ValueError("Selected source is not WAV")
                duration = len(source) / source.samplerate
                result.update(source=str(path), source_sha256=hashlib.sha256(body).hexdigest(),
                              audio_seconds=duration)
                if not 0 < duration <= 60:
                    raise ValueError("WAV duration must be in (0, 60] seconds; never truncated")
                audio = resample_audio(source.read(dtype="float32", always_2d=True).mean(axis=1),
                                       source.samplerate, self.settings.sample_rate)
            timings[stage] = time.monotonic() - stage_started
            stage, stage_started = "activate", time.monotonic()
            self.engine.activate_ready(timeout=remaining())
            timings[stage] = time.monotonic() - stage_started
            stage, stage_started = "infer", time.monotonic()
            width = max(1, int(self.settings.max_chunk_sec * self.settings.sample_rate))
            segments = result["coverage"]["segments"]
            for offset in range(0, len(audio), width):
                stop = min(offset + width, len(audio))
                before = time.monotonic()
                text = self.engine.transcribe_ready(audio[offset:stop], timeout=remaining()).strip()
                remaining()
                segments.append({"start_seconds": offset / self.settings.sample_rate,
                                 "end_seconds": min(stop / self.settings.sample_rate, duration),
                                 "text": text, "state": "text" if text else "empty",
                                 "infer_seconds": time.monotonic() - before})
            timings[stage] = time.monotonic() - stage_started
            stage, stage_started = "park", time.monotonic()
            self.engine.park_ready(timeout=remaining())
            remaining()
            timings[stage] = time.monotonic() - stage_started
            result["text"] = "".join(segment["text"] for segment in segments)
            result["coverage"]["status"] = "complete"
            result["state"] = ("empty" if not result["text"] else
                               "partial" if any(not s["text"] for s in segments) else "completed")
        except Exception as error:
            timings[stage] = time.monotonic() - stage_started
            result.update(state="not_ready" if isinstance(error, DictationWorkerNotReady) else "failed",
                          error_type=type(error).__name__, message=str(error))
            result["coverage"]["status"] = "partial" if result["coverage"]["segments"] else "unknown"
            # Input rejection preserves the hot worker; model failures reap it.
            if stage != "prepare" or not self.engine.is_ready:
                self._prepared = False
                cleanup_started = time.monotonic()
                try:
                    self.engine.abort(timeout=min(0.25, max(0, deadline - cleanup_started)))
                except Exception as cleanup_error:
                    result["cleanup_error"] = str(cleanup_error)
                timings["cleanup"] = time.monotonic() - cleanup_started
        timings["total"] = time.monotonic() - started
        result["workerPID_after"] = self.engine.worker_pid
        return result

    def close(self):
        previous = self.engine.worker_pid
        self.engine.dispose()
        return {"event": "closed", "previous_workerPID": previous,
                "workerPID": self.engine.worker_pid}


def serve(session, input_stream, output_stream):
    def emit(value):
        print(json.dumps(value, ensure_ascii=False), file=output_stream, flush=True)

    try:
        emit(session.prepare())
        for line in input_stream:
            try:
                request = json.loads(line)
                action = request.get("action")
                if action == "close":
                    break
                if action == "transcribe":
                    emit(session.transcribe(request))
                elif action == "status":
                    emit(session.status())
                elif action == "prepare":
                    emit(session.prepare())
                else:
                    raise ValueError("action must be transcribe, status, prepare or close")
            except Exception as error:
                emit({"event": "error", "error_type": type(error).__name__, "message": str(error)})
    except Exception as error:
        emit({"event": "error", "state": "not_ready",
              "error_type": type(error).__name__, "message": str(error)})
    finally:
        emit(session.close())


def main():
    from .dictation import DictationSettings
    # This entry accepts no caller-provided answer/context/hotwords.
    serve(RealtimeSession(replace(DictationSettings.load(), hotwords="")), sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()

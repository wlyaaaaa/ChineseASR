"""Supervised in-memory dictation engine, without audio files or a listening port."""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import threading
import time

from .gpu_broker import GpuBrokerConflict

LOG = logging.getLogger("zh_asr.dictation")


class DictationWorkerFailure(RuntimeError):
    """The exact model process stopped responding or exited."""


class DictationWorkerCancelled(RuntimeError):
    """The parent requested shutdown while a model operation was pending."""


def _engine_main(connection, settings) -> None:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    parent = mp.parent_process()

    def watch_parent():
        while parent is not None and parent.is_alive():
            time.sleep(1)
        if parent is not None:
            os._exit(1)  # Ends a stuck native call and the lease renewal thread.

    threading.Thread(target=watch_parent, name="dictation-parent", daemon=True).start()
    from .dictation import QwenDictationEngine
    engine = QwenDictationEngine(settings)
    try:
        while True:
            operation, argument = connection.recv()
            try:
                if operation == "transcribe":
                    value = engine.transcribe(argument)
                elif operation in {"load", "activate", "warmup", "park", "dispose"}:
                    value = getattr(engine, operation)()
                else:
                    raise ValueError("Unknown dictation engine operation")
                connection.send({"ok": True, "value": value})
                if operation == "dispose":
                    break
            except GpuBrokerConflict as error:
                connection.send({"ok": False, "kind": "gpu_conflict",
                                 "owner": error.owner, "reason": error.reason,
                                 "message": str(error)})
            except Exception as error:
                connection.send({"ok": False, "kind": type(error).__name__,
                                 "message": str(error)[:600]})
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        try:
            engine.dispose()
        finally:
            connection.close()


class ProcessDictationEngine:
    """One persistent RAM-resident model with bounded native calls and recovery.

    Only the recognition thread calls this object. A failed process is reaped
    before replacement. Audio travels through an anonymous pipe and never lands
    in the filesystem. Inference may be retried once before returning any text.
    """
    def __init__(self, settings, *, cancel_event=None, load_timeout=180.0,
                 operation_timeout=60.0, warmup_timeout=40.0, worker_target=None):
        self.settings = settings
        self.cancel_event = cancel_event or threading.Event()
        self.load_timeout = load_timeout
        self.operation_timeout = operation_timeout
        self.warmup_timeout = warmup_timeout
        self._target = worker_target or _engine_main
        self._process = None
        self._connection = None
        self._loaded = False
        self._context = mp.get_context("spawn")

    @property
    def worker_pid(self):
        return self._process.pid if self._process is not None else None

    def _start(self):
        if self.cancel_event.is_set():
            raise DictationWorkerCancelled("Dictation is shutting down")
        if self._process is not None and self._process.is_alive():
            if self._connection is None:
                raise DictationWorkerFailure("The previous model process has not exited")
            return
        self._stop()
        parent, child = self._context.Pipe()
        process = self._context.Process(target=self._target, args=(child, self.settings),
                                        name="ChineseASR-dictation-engine", daemon=True)
        try:
            process.start()
        except BaseException:
            parent.close()
            child.close()
            raise
        child.close()
        self._connection, self._process = parent, process
        LOG.info("dictation model process started pid=%s", process.pid)

    def _stop(self):
        process, connection = self._process, self._connection
        self._process = self._connection = None
        self._loaded = False
        if connection is not None:
            connection.close()
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
            if process.is_alive():
                self._process = process
                raise DictationWorkerFailure("The previous model process has not exited")
            process.close()

    def _call(self, operation, argument=None, *, timeout=None):
        self._start()
        process, connection = self._process, self._connection
        started = time.monotonic()
        deadline = started + (self.operation_timeout if timeout is None else timeout)
        try:
            connection.send((operation, argument))
            while not connection.poll(0.1):
                if self.cancel_event.is_set():
                    raise DictationWorkerCancelled("Dictation operation cancelled")
                if not process.is_alive():
                    raise DictationWorkerFailure(f"Model process exited during {operation}")
                if time.monotonic() >= deadline:
                    raise DictationWorkerFailure(f"Model operation timed out: {operation}")
            response = connection.recv()
        except (EOFError, BrokenPipeError, OSError, DictationWorkerFailure, DictationWorkerCancelled) as error:
            self._stop()
            if isinstance(error, DictationWorkerCancelled):
                raise
            raise DictationWorkerFailure(f"{operation}: {error}") from error
        if not response.get("ok"):
            if response.get("kind") == "gpu_conflict":
                raise GpuBrokerConflict(response["message"], owner=response["owner"], reason=response["reason"])
            message = f"{operation}: {response.get('kind')}: {response.get('message')}"
            # A native/runtime failure must not poison every subsequent session.
            self._stop()
            raise DictationWorkerFailure(message)
        LOG.info("dictation model operation=%s seconds=%.3f", operation, time.monotonic() - started)
        return response.get("value")

    def load(self):
        result = self._call("load", timeout=self.load_timeout)
        self._loaded = True
        return result

    def _ensure_loaded(self):
        if not self._loaded or self._process is None or not self._process.is_alive():
            self.load()

    def activate(self):
        self._ensure_loaded()
        return self._call("activate")

    def warmup(self):
        self._ensure_loaded()
        return self._call("warmup", timeout=self.warmup_timeout)

    def transcribe(self, audio):
        self._ensure_loaded()
        for attempt in range(2):
            try:
                return self._call("transcribe", audio)
            except DictationWorkerFailure:
                if attempt or self.cancel_event.is_set():
                    raise
                LOG.warning("model unresponsive; retrying the in-memory phrase once")
                self.load()

    def park(self):
        if self._process is not None and self._process.is_alive() and self._loaded:
            return self._call("park", timeout=20)
        self._stop()

    def dispose(self):
        try:
            if self._process is not None and self._process.is_alive() and not self.cancel_event.is_set():
                self._call("dispose", timeout=5)
                if self._process is not None:
                    self._process.join(timeout=1)
        finally:
            self._stop()

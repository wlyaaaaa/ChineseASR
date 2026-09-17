from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
import unittest

from zh_asr.dictation_worker import (
    ProcessDictationEngine, DictationWorkerFailure, DictationWorkerCancelled,
)
from zh_asr.gpu_broker import GpuBrokerConflict, GpuBrokerLease
from test_gpu_broker import RecordingTransport


def fake_engine(connection, settings):
    try:
        while True:
            operation, argument = connection.recv()
            if operation == settings.get("hang"):
                time.sleep(60)
            if operation == settings.get("conflict"):
                connection.send({"ok": False, "kind": "gpu_conflict", "message": "busy",
                                 "owner": "chineseasr-cli", "reason": "gpu_lease_active"})
                continue
            if operation == "transcribe" and settings.get("crashes") is not None:
                counter = settings["crashes"]
                with counter.get_lock():
                    counter.value += 1
                    value = counter.value
                if value <= settings.get("crash_count", 1):
                    os._exit(2)
            value = argument if operation == "transcribe" else None
            connection.send({"ok": True, "value": value})
            if operation == "dispose":
                return
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


class DictationWorkerTests(unittest.TestCase):
    def engine(self, settings=None, **kwargs):
        engine = ProcessDictationEngine(settings or {}, worker_target=fake_engine,
                                        load_timeout=15, **kwargs)
        self.addCleanup(engine.dispose)
        return engine

    def test_model_process_persists_between_parked_sessions(self):
        engine = self.engine()
        engine.load()
        pid = engine.worker_pid
        self.assertNotEqual(pid, os.getpid())
        self.assertEqual(engine.transcribe("不同意，1.5万元 C++"), "不同意，1.5万元 C++")
        engine.park()
        engine.activate()
        self.assertEqual(engine.worker_pid, pid)

    def test_native_timeout_reaps_only_worker_and_allows_fresh_load(self):
        engine = self.engine({"hang": "warmup"}, warmup_timeout=0.2)
        engine.load()
        old_pid = engine.worker_pid
        started = time.monotonic()
        with self.assertRaises(DictationWorkerFailure):
            engine.warmup()
        self.assertLess(time.monotonic() - started, 10)
        self.assertIsNone(engine.worker_pid)
        engine.load()
        self.assertNotEqual(engine.worker_pid, old_pid)

    def test_shutdown_cancels_a_native_wait(self):
        cancelled = threading.Event()
        engine = self.engine({"hang": "warmup"}, cancel_event=cancelled)
        engine.load()
        timer = threading.Timer(0.1, cancelled.set)
        timer.start()
        try:
            with self.assertRaises(DictationWorkerCancelled):
                engine.warmup()
        finally:
            timer.cancel()
        self.assertIsNone(engine.worker_pid)

    def test_gpu_contention_preserves_model_and_names_blocker(self):
        engine = self.engine({"conflict": "activate"})
        engine.load()
        pid = engine.worker_pid
        with self.assertRaises(GpuBrokerConflict) as result:
            engine.activate()
        self.assertEqual(result.exception.owner, "chineseasr-cli")
        self.assertEqual(engine.worker_pid, pid)

    def test_inference_crash_retries_unreturned_phrase_once(self):
        counter = mp.get_context("spawn").Value("i", 0)
        engine = self.engine({"crashes": counter, "crash_count": 1})
        engine.load()
        self.assertEqual(engine.transcribe("完整的原句"), "完整的原句")
        self.assertEqual(counter.value, 2)

    def test_repeated_inference_failure_is_bounded(self):
        counter = mp.get_context("spawn").Value("i", 0)
        engine = self.engine({"crashes": counter, "crash_count": 10})
        engine.load()
        with self.assertRaises(DictationWorkerFailure):
            engine.transcribe("不得无限重试")
        self.assertEqual(counter.value, 2)
        self.assertIsNone(engine.worker_pid)

    def test_gpu_acquisition_is_bound_to_actual_model_process(self):
        transport = RecordingTransport()
        with GpuBrokerLease("chineseasr", transport=transport, renew_interval_seconds=0):
            payload = transport.calls[0][1]
            self.assertEqual(payload["owner_pid"], os.getpid())
            self.assertEqual(payload["ttl_seconds"], 120)


if __name__ == "__main__":
    unittest.main()

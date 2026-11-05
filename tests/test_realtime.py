from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from zh_asr.dictation_worker import (
    ProcessDictationEngine, DictationWorkerNotReady, DictationWorkerFailure,
)
from zh_asr.realtime import RealtimeSession, serve
from test_dictation_worker import fake_engine


class FakeEngine:
    def __init__(self, texts=("正文",), delay=0):
        self.is_ready = False
        self.worker_pid = None
        self.texts = iter(texts)
        self.delay = delay
        self.loads = 0
        self.sizes = []

    def load(self):
        self.loads += 1
        self.is_ready, self.worker_pid = True, 123

    def warmup(self):
        pass

    def park(self):
        pass

    def activate_ready(self, *, timeout):
        assert timeout > 0 and self.is_ready

    def transcribe_ready(self, audio, *, timeout):
        assert timeout > 0 and self.is_ready
        self.sizes.append(len(audio))
        time.sleep(self.delay)
        return next(self.texts)

    def park_ready(self, *, timeout):
        assert timeout > 0

    def dispose(self):
        self.is_ready, self.worker_pid = False, None

    def abort(self, *, timeout):
        self.dispose()


class RealtimeTests(unittest.TestCase):
    def session(self, engine):
        return RealtimeSession(SimpleNamespace(engine="test", sample_rate=16000,
                                               max_chunk_sec=20), engine)

    def wav(self, seconds):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "selected.wav"
        sf.write(path, np.zeros(int(seconds * 16000), dtype=np.float32), 16000)
        return str(path)

    def test_real_worker_ready_only_never_starts_or_reloads(self):
        engine = ProcessDictationEngine({}, worker_target=fake_engine, load_timeout=15)
        self.addCleanup(engine.dispose)
        with patch.object(engine, "_start", side_effect=AssertionError("unexpected start")):
            with self.assertRaises(DictationWorkerNotReady):
                engine.activate_ready(timeout=1)
        engine.load()
        pid = engine.worker_pid
        self.assertTrue(engine.is_ready)
        self.assertEqual(engine.transcribe_ready("原文", timeout=1), "原文")
        engine.park_ready(timeout=1)
        self.assertEqual(engine.worker_pid, pid)
        engine._process.terminate()
        engine._process.join(5)
        self.assertFalse(engine.is_ready)
        with patch.object(engine, "load", side_effect=AssertionError("unexpected reload")):
            with self.assertRaises(DictationWorkerNotReady):
                engine.transcribe_ready("原文", timeout=1)
        self.assertIsNone(engine.worker_pid)

    def test_all_chunks_hash_duration_and_empty_are_explicit(self):
        engine = FakeEngine(("甲", "", "尾段"))
        session = self.session(engine)
        session.prepare()
        rejected = session.transcribe({"path": "not-an-absolute.wav"})
        self.assertEqual(rejected["state"], "failed")
        self.assertEqual(session.status()["state"], "ready")
        self.assertEqual(engine.worker_pid, 123)
        path = self.wav(45)
        before = Path(path).read_bytes()
        result = session.transcribe({"path": path, "request_id": "opaque"})
        self.assertEqual(result["state"], "partial")
        self.assertEqual(result["text"], "甲尾段")
        self.assertEqual(engine.sizes, [320000, 320000, 80000])
        self.assertEqual(result["coverage"]["status"], "complete")
        self.assertEqual([(s["start_seconds"], s["end_seconds"])
                          for s in result["coverage"]["segments"]], [(0, 20), (20, 40), (40, 45)])
        import hashlib
        self.assertEqual(result["source_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(Path(path).read_bytes(), before)
        self.assertEqual(result["audio_seconds"], 45)

    def test_native_ready_timeout_reaps_without_cold_retry(self):
        engine = ProcessDictationEngine({"hang": "transcribe"}, worker_target=fake_engine,
                                        load_timeout=15)
        self.addCleanup(engine.dispose)
        engine.load()
        started = time.monotonic()
        with patch.object(engine, "load", side_effect=AssertionError("unexpected reload")):
            with self.assertRaises(DictationWorkerFailure):
                engine.transcribe_ready("原文", timeout=0.05)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertFalse(engine.is_ready)

    def test_shared_deadline_cannot_return_partial_success_or_reload(self):
        path = self.wav(40)
        # Binary fractions keep every deadline and elapsed-time assertion exact.
        for stage, preparation, delays, infer_timeouts, covered in (
            ("prepare", 0.375, (), [], 0),
            ("first_chunk", 0.0625, (0.25,), [0.1875], 0),
            ("later_chunk", 0.0625, (0.125, 0.125), [0.1875, 0.0625], 1),
        ):
            with self.subTest(stage=stage):
                clock = SimpleNamespace(now=0.0)
                calls = {name: [] for name in ("activate", "infer", "park", "abort")}

                class TimedEngine(FakeEngine):
                    def activate_ready(self, *, timeout):
                        calls["activate"].append(timeout)
                        super().activate_ready(timeout=timeout)

                    def transcribe_ready(self, audio, *, timeout):
                        calls["infer"].append(timeout)
                        self.sizes.append(len(audio))
                        clock.now += delays[len(calls["infer"]) - 1]
                        return next(self.texts)

                    def park_ready(self, *, timeout):
                        calls["park"].append(timeout)
                        super().park_ready(timeout=timeout)

                    def abort(self, *, timeout):
                        calls["abort"].append(timeout)
                        super().abort(timeout=timeout)

                def prepare_audio(audio, source_rate, target_rate):
                    clock.now += preparation
                    return audio

                engine = TimedEngine(("甲", "乙"))
                session = self.session(engine)
                with patch("zh_asr.realtime.time", SimpleNamespace(monotonic=lambda: clock.now)), \
                     patch("zh_asr.dictation.resample_audio", side_effect=prepare_audio):
                    session.prepare()
                    result = session.transcribe({"path": path, "budget_seconds": 0.5})
                    self.assertEqual(result["state"], "failed")
                    self.assertEqual(result["error_type"], "TimeoutError")
                    self.assertEqual(result["text"], "")
                    self.assertEqual(result["timings"]["prepare"], preparation)
                    self.assertEqual(result["timings"]["infer"], sum(delays))
                    self.assertEqual(result["timings"]["total"], preparation + sum(delays))
                    self.assertEqual(calls["activate"], [] if stage == "prepare" else [0.1875])
                    self.assertEqual(calls["infer"], infer_timeouts)
                    self.assertEqual(engine.sizes, [320000] * len(infer_timeouts))
                    self.assertEqual(calls["park"], [])
                    self.assertEqual(calls["abort"], [0.5 - clock.now])
                    self.assertEqual(result["coverage"]["status"], "partial" if covered else "unknown")
                    self.assertEqual(result["coverage"]["segments"], [
                        {"start_seconds": 0, "end_seconds": 20, "text": "甲",
                         "state": "text", "infer_seconds": 0.125}
                    ] if covered else [])
                    self.assertFalse(engine.is_ready)
                    self.assertIsNone(result["workerPID_after"])
                    self.assertEqual(session.transcribe({"path": "ignored.wav"})["state"], "not_ready")
                    self.assertEqual(engine.loads, 1)

    def _served(self, engine, requests, idle_park_seconds):
        class Events(list):
            def write(self, text):
                if text.strip():
                    self.append(json.loads(text))
                    engine.calls.append(self[-1]["event"])

            def flush(self):
                pass

        def stream():
            for request, pause in requests:
                yield json.dumps(request) + "\n"
                time.sleep(pause)

        events = Events()
        serve(self.session(engine), stream(), events, idle_park_seconds=idle_park_seconds)
        return events

    class RecordingEngine(FakeEngine):
        def __init__(self, texts=("正文",)):
            super().__init__(texts)
            self.calls = []

        def activate_ready(self, *, timeout):
            self.calls.append("activate")

        def park_ready(self, *, timeout):
            self.calls.append("park")

    def test_result_arrives_before_park_and_consecutive_voices_stay_on_gpu(self):
        engine = self.RecordingEngine(("甲", "乙"))
        request = {"action": "transcribe", "path": self.wav(1)}
        events = self._served(engine, [(request, 0), (request, 0.4)], idle_park_seconds=0.1)
        self.assertEqual([event["text"] for event in events if event["event"] == "result"], ["甲", "乙"])
        self.assertEqual(engine.calls, ["ready", "activate", "result", "activate", "result", "park", "closed"])

    def test_idle_park_failure_reaps_worker_and_later_voice_is_not_ready(self):
        class FailingPark(self.RecordingEngine):
            def park_ready(self, *, timeout):
                raise RuntimeError("park failed")

        engine = FailingPark()
        request = {"action": "transcribe", "path": self.wav(1)}
        events = self._served(engine, [(request, 0.4), (request, 0)], idle_park_seconds=0.1)
        states = [event["state"] for event in events if event["event"] == "result"]
        self.assertEqual(states, ["completed", "not_ready"])
        self.assertEqual(engine.loads, 1)

    def test_eof_close_and_empty_result_cleanup(self):
        for request in ("", '{"action":"close"}\n'):
            engine = FakeEngine()
            output = io.StringIO()
            serve(self.session(engine), io.StringIO(request), output)
            events = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual([event["event"] for event in events], ["ready", "closed"])
            self.assertFalse(engine.is_ready)
        session = self.session(FakeEngine(("",)))
        session.prepare()
        result = session.transcribe({"path": self.wav(1)})
        self.assertEqual(result["state"], "empty")
        self.assertEqual(result["coverage"]["status"], "complete")


if __name__ == "__main__":
    unittest.main()

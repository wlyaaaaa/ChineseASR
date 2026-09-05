from __future__ import annotations

import queue
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from zh_asr.dictation import (
    DictationController, DictationSettings, PauseSegmenter, Recording,
)


class FakeHost:
    def __init__(self, allow=True):
        self.allow = allow
        self.insertions = []
        self.messages = []
        self.last_text = ""

    def insert_text(self, text, target):
        self.insertions.append((text, target))
        return self.allow

    def show(self, status, detail="", **kwargs):
        self.messages.append((status, detail))

    def set_last_text(self, text):
        self.last_text = text


class FakeEngine:
    def __init__(self, texts):
        self.texts = iter(texts)
        self.before_return = lambda: None

    def activate(self):
        pass

    def transcribe(self, audio):
        self.before_return()
        return next(self.texts)


def recording_with_chunks(count=2):
    recording = Recording("original-focus", queue.Queue(), threading.Event(), threading.Event())
    for _ in range(count):
        recording.chunks.put(np.ones(1600, dtype=np.float32))
    recording.chunks.put(None)
    recording.stopped.set()
    return recording


class DictationTests(unittest.TestCase):
    def test_simultaneous_ui_and_worker_cleanup_closes_stream_once(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def stop():
            calls.append("stop")
            entered.set()
            release.wait(2)
        controller = DictationController(FakeHost(), DictationSettings(), FakeEngine([]))
        controller.stream = SimpleNamespace(stop=stop, close=lambda: calls.append("close"))
        worker = threading.Thread(target=controller._close_stream)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            controller._close_stream()
        finally:
            release.set()
            worker.join(2)
        self.assertEqual(calls, ["stop", "close"])

    def test_disconnect_while_stopping_cancels_pending_text_and_closes_stream(self):
        host = FakeHost()
        controller = DictationController(host, DictationSettings(), FakeEngine(["不应输入"]))
        controller.recording = recording_with_chunks(1)
        controller.recording.stopped.clear()
        controller.segmenter = PauseSegmenter(DictationSettings())
        closed = []
        def failed_stop():
            raise OSError("device disconnected")
        controller.stream = SimpleNamespace(stop=failed_stop, close=lambda: closed.append(True))
        with self.assertLogs("zh_asr.dictation", level="ERROR"):
            controller.stop()
        self.assertEqual(closed, [True])
        self.assertTrue(controller.recording.cancelled.is_set())
        controller._recognize(controller.recording)
        self.assertEqual(host.insertions, [])

    def test_silence_does_not_trigger_model(self):
        segmenter = PauseSegmenter(DictationSettings())
        for _ in range(150):
            self.assertEqual(segmenter.feed(np.zeros(320)), [])
        self.assertIsNone(segmenter.flush())

    def test_pause_keeps_preroll_and_quiet_word_ending(self):
        segmenter = PauseSegmenter(DictationSettings())
        for _ in range(10):
            segmenter.feed(np.zeros(320))
        for _ in range(20):
            self.assertEqual(segmenter.feed(np.full(320, 0.03)), [])
        emitted = []
        for _ in range(30):
            emitted.extend(segmenter.feed(np.zeros(320)))
        self.assertEqual(len(emitted), 1)
        self.assertEqual(len(emitted[0]), 60 * 320)
        self.assertEqual(np.count_nonzero(emitted[0]), 20 * 320)
        self.assertIsNone(segmenter.flush())

    def test_long_continuous_speech_and_final_tail_have_no_sample_gap(self):
        settings = DictationSettings(max_chunk_sec=3)
        segmenter = PauseSegmenter(settings)
        emitted = []
        for _ in range(174):
            emitted.extend(segmenter.feed(np.full(320, 0.03)))
        emitted.append(segmenter.flush())
        self.assertEqual(sum(len(chunk) for chunk in emitted), 174 * 320)

    def test_focus_change_stops_all_later_insertion_but_keeps_complete_text(self):
        host = FakeHost(allow=False)
        controller = DictationController(host, DictationSettings(), FakeEngine(["不要更新。", "保留 API。 "]))
        recording = recording_with_chunks()
        controller._recognize(recording)
        self.assertEqual(len(host.insertions), 1)
        self.assertEqual(host.last_text, "不要更新。保留 API。 ")
        self.assertTrue(recording.insertion_failed)

    def test_cancel_during_inference_never_inserts_late_text(self):
        host = FakeHost()
        engine = FakeEngine(["不应输入"])
        recording = recording_with_chunks(1)
        engine.before_return = recording.cancelled.set
        controller = DictationController(host, DictationSettings(), engine)
        controller._recognize(recording)
        self.assertEqual(host.insertions, [])

    def test_chinese_english_and_negation_pass_through_without_rewriting(self):
        text = "不要替换 Qwen，仅更新 API。"
        host = FakeHost()
        controller = DictationController(host, DictationSettings(), FakeEngine([text]))
        controller._recognize(recording_with_chunks(1))
        self.assertEqual(host.insertions, [(text, "original-focus")])


if __name__ == "__main__":
    unittest.main()

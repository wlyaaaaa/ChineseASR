from __future__ import annotations

import queue
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
import json

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
        self.visible = False
        self.callbacks = queue.Queue()
        self.busy = False

    def insert_text(self, text, target):
        self.insertions.append((text, target))
        return self.allow

    def show(self, status, detail="", **kwargs):
        self.messages.append((status, detail))

    def set_last_text(self, text):
        self.last_text = text

    def capture_target(self):
        return "original-focus"

    def open_panel(self):
        self.visible = True

    @property
    def panel_visible(self):
        return self.visible

    def hide_panel(self):
        self.visible = False

    def set_busy(self, value):
        self.busy = value

    def post_to_ui(self, callback):
        self.callbacks.put(callback)

    def pump(self):
        while not self.callbacks.empty():
            self.callbacks.get_nowait()()


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
    def test_hide_immediately_pauses_without_waiting_or_discarding_the_tail(self):
        host = FakeHost()
        controller = DictationController(host, DictationSettings(), FakeEngine([]))
        controller.toggle()
        recording = controller.recording
        controller.hide()
        self.assertFalse(host.visible)
        self.assertTrue(recording.stopped.is_set())
        self.assertFalse(recording.cancelled.is_set())
        self.assertFalse(controller.quit_event.is_set())
        self.assertEqual(controller.audio_commands.qsize(), 2)

    def test_hotkey_toggles_visibility_while_button_only_toggles_recording(self):
        host = FakeHost()
        controller = DictationController(host, DictationSettings(), FakeEngine([]))
        controller.toggle_visibility()
        first = controller.recording
        self.assertTrue(host.visible)
        self.assertFalse(first.stopped.is_set())
        controller.toggle()
        self.assertTrue(host.visible)
        self.assertTrue(first.stopped.is_set())
        controller.toggle_visibility()
        self.assertFalse(host.visible)
        self.assertFalse(controller.pending_start)
        controller.toggle_visibility()
        self.assertTrue(host.visible)
        self.assertTrue(controller.pending_start)
        first.capture_finished.set()
        first.recognition_finished.set()
        controller._recording_finished(first)
        self.assertIsNot(controller.recording, first)
        self.assertFalse(controller.recording.stopped.is_set())

    def test_cancel_during_open_closes_late_stream_without_starting_transcription(self):
        entered, release = threading.Event(), threading.Event()
        closed = []
        def open_late(*args):
            entered.set()
            release.wait(2)
            return SimpleNamespace(stop=lambda: closed.append("stop"), close=lambda: closed.append("close"))
        host = FakeHost()
        controller = DictationController(host, DictationSettings(), FakeEngine([]))
        with patch("zh_asr.dictation_audio.open_microphone", side_effect=open_late):
            controller.audio_worker.start()
            controller.toggle()
            self.assertTrue(entered.wait(1))
            controller.hide()
            release.set()
            controller.audio_commands.put(None)
            controller.audio_worker.join(2)
        host.pump()
        self.assertEqual(closed, ["stop", "close"])
        self.assertTrue(controller.commands.empty())
        self.assertIsNone(controller.recording)
        self.assertFalse(host.visible)

    def test_microphone_selection_is_saved_and_applies_to_next_session(self):
        host = FakeHost()
        settings = DictationSettings(input_device="DJI Mic Mini")
        controller = DictationController(host, settings, FakeEngine([]))
        controller.toggle()
        recording = controller.recording
        with TemporaryDirectory() as directory:
            path = Path(directory) / "preferences.json"
            with patch("zh_asr.dictation.preferences_path", return_value=path):
                controller.select_microphone("UGREEN Camera Audio")
                self.assertEqual(json.loads(path.read_text())["input_device"], "UGREEN Camera Audio")
                self.assertEqual(settings.with_preferences().input_device, "UGREEN Camera Audio")
                path.write_text("{bad json")
                self.assertEqual(settings.with_preferences().input_device, "DJI Mic Mini")
        self.assertEqual(recording.settings.input_device, "DJI Mic Mini")
        self.assertTrue(recording.stopped.is_set())
        self.assertEqual(controller.settings.input_device, "UGREEN Camera Audio")

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

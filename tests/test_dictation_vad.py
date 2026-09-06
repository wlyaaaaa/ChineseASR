from __future__ import annotations

from pathlib import Path
import os
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from zh_asr.dictation_vad import VoiceActivityDetector, contains_speech


SAMPLE = Path(os.environ.get("CHINESE_ASR_PUBLIC_TEST_AUDIO", "tests/fixtures/asr_example_zh.wav"))


class DictationVadTests(unittest.TestCase):
    def setUp(self):
        self.detector = VoiceActivityDetector(sample_rate=16000, mode=2)

    def test_empty_and_short_audio_are_not_speech(self):
        self.assertFalse(contains_speech(np.zeros(0, dtype=np.float32)))
        self.assertFalse(contains_speech(np.zeros(160, dtype=np.float32)))

    def test_dc_and_low_level_noise_are_rejected(self):
        frame_count = 250
        dc = np.full(320 * frame_count, 0.01, dtype=np.float32)
        noise = np.random.default_rng(0).normal(
            0.0, 0.0001, 320 * frame_count
        ).astype(np.float32)
        self.assertFalse(contains_speech(dc))
        self.assertFalse(contains_speech(noise))
        normal_noise = np.random.default_rng(0).normal(0, 0.01, 320 * frame_count)
        self.assertFalse(contains_speech(normal_noise))

    def test_isolated_activity_does_not_add_up_to_speech(self):
        audio = np.zeros(320 * 24, dtype=np.float32)
        with patch.object(VoiceActivityDetector, "is_speech", side_effect=[True, False] * 12):
            self.assertFalse(contains_speech(audio))
        flags = [True] * 3 + [False] * 3 + [True, False] * 9
        with patch.object(VoiceActivityDetector, "is_speech", side_effect=flags):
            self.assertTrue(contains_speech(audio))

    def test_voiced_harmonics_are_not_removed_by_a_spectral_rule(self):
        time = np.arange(16000) / 16000
        voiced = sum(0.03 / harmonic * np.sin(2 * np.pi * 120 * harmonic * time)
                     for harmonic in (1, 2, 3)).astype(np.float32)
        self.assertTrue(contains_speech(voiced))

    @unittest.skipUnless(SAMPLE.is_file(), "Optional public Chinese audio fixture is not installed")
    def test_public_sample_is_retained_as_speech(self):
        audio, sample_rate = sf.read(SAMPLE, dtype="float32", always_2d=False)
        self.assertEqual(sample_rate, 16000)
        self.assertTrue(contains_speech(np.asarray(audio, dtype=np.float32)))
        # A real, harmonically voiced 240 ms Chinese fragment must survive.
        self.assertTrue(contains_speech(audio[int(2.540 * sample_rate):int(2.780 * sample_rate)]))

    def test_frame_size_and_sample_rate_are_explicit(self):
        with self.assertRaises(ValueError):
            self.detector.is_speech(np.zeros(319, dtype=np.float32))
        with self.assertRaises(ValueError):
            contains_speech(np.zeros(320, dtype=np.float32), sample_rate=8000)


if __name__ == "__main__":
    unittest.main()

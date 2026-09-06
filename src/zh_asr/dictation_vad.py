"""Small CPU voice-activity gate for desktop dictation."""
from __future__ import annotations

import math

import numpy as np


class VoiceActivityDetector:
    """WebRTC VAD over fixed 20 ms float32 frames.

    The energy floor rejects almost-silent input. Leave speech classification
    to WebRTC; narrow-band guards can remove real voiced Chinese syllables.
    """

    frame_ms = 20

    def __init__(self, sample_rate: int = 16000, mode: int = 2) -> None:
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("WebRTC VAD sample_rate must be 8000, 16000, 32000, or 48000.")
        if mode not in (0, 1, 2, 3):
            raise ValueError("WebRTC VAD mode must be 0, 1, 2, or 3.")
        try:
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError(
                "webrtcvad-wheels is required for desktop dictation VAD."
            ) from exc
        self.sample_rate = sample_rate
        self.mode = mode
        self.frame_samples = sample_rate * self.frame_ms // 1000
        self._vad = webrtcvad.Vad(mode)

    def _frame(self, frame: np.ndarray) -> np.ndarray:
        values = np.asarray(frame, dtype=np.float32).reshape(-1)
        if len(values) != self.frame_samples:
            raise ValueError(
                f"VAD requires exactly {self.frame_samples} samples per 20 ms frame."
            )
        if not np.isfinite(values).all():
            raise ValueError("VAD frame contains non-finite samples.")
        return values

    def is_speech(self, frame: np.ndarray) -> bool:
        values = self._frame(frame)
        pcm16 = np.clip(values, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16).tobytes()
        speech = self._vad.is_speech(pcm16, self.sample_rate)
        # Still feed quiet frames into WebRTC so its internal hangover expires.
        return speech and float(np.sqrt(np.mean(values * values))) >= 0.0008


def contains_speech(
    audio: np.ndarray,
    sample_rate: int = 16000,
    min_speech_ms: int = 240,
    *,
    mode: int = 2,
) -> bool:
    """Return whether enough fixed frames contain likely speech.

    This is an input gate for deciding whether to call ASR. It is not an
    authoritative claim that an audio source contains or lacks a speaker.
    Trailing partial frames are ignored; no resampling is performed here.
    """

    if sample_rate != 16000:
        raise ValueError("Desktop dictation VAD currently requires 16 kHz audio.")
    if min_speech_ms <= 0:
        raise ValueError("min_speech_ms must be positive.")
    detector = VoiceActivityDetector(sample_rate=sample_rate, mode=mode)
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(values) < detector.frame_samples:
        return False
    required = math.ceil(min_speech_ms / detector.frame_ms)
    count = run = longest_run = 0
    for start in range(0, len(values) - detector.frame_samples + 1, detector.frame_samples):
        if detector.is_speech(values[start : start + detector.frame_samples]):
            count += 1
            run += 1
            longest_run = max(longest_run, run)
            # Do not accumulate isolated clicks into a speech decision. Brief
            # gaps are allowed; only 60 ms of the total needs to be consecutive.
            if count >= required and longest_run >= min(3, required):
                return True
        else:
            run = 0
    return False

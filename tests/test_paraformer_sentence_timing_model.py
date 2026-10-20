"""Opt-in checks that Paraformer sentence text sits on its own token times.

The model test repeats the public example shipped with the pinned Paraformer
model with silence between copies, so FunASR sees several Chinese VAD chunks:
the case in which FunASR 1.4.14 shifts sentence text against its timestamps by
one token per chunk boundary.  The library test runs FunASR's own sentence
builder on the same situation, so a FunASR upgrade shows whether the repair is
still needed.  Both import FunASR/torch and the first loads the real model, so
they run only with ``ZH_ASR_RUN_MODEL_TESTS=1``; ``ZH_ASR_MODEL_TEST_DEVICE``
picks the model device (default ``cpu``; a CUDA device goes through the CLI's
LocalGpuBroker lease).
"""

from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = (
    ROOT
    / "models"
    / "modelscope"
    / "iic"
    / "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    / "example"
    / "asr_example.wav"
)
RUN_MODEL_TESTS = os.environ.get("ZH_ASR_RUN_MODEL_TESTS") == "1"
COPIES = 3
GAP_MS = 3000
WINDOW_SLACK_MS = 200
# One token per Chinese character or ASCII word, ignoring the punctuation that
# FunASR appends to sentences.
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9'，。？、,.?!]")


@unittest.skipUnless(RUN_MODEL_TESTS, "set ZH_ASR_RUN_MODEL_TESTS=1 to load the real Paraformer model")
@unittest.skipUnless(EXAMPLE.is_file(), "pinned Paraformer model and its public example audio are not installed")
class ParaformerSentenceTimingModelTest(unittest.TestCase):
    def test_sentence_text_stays_on_its_times_across_vad_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "public-example-repeated.wav"
            offsets_ms, clip_ms = _write_repeated_example(audio)
            out_dir = root / "out"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "zh_asr",
                    "transcribe",
                    str(audio),
                    "--engine",
                    "paraformer",
                    "--device",
                    os.environ.get("ZH_ASR_MODEL_TEST_DEVICE", "cpu"),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
            payload = json.loads(
                (out_dir / f"{audio.stem}.paraformer.raw.json").read_text(encoding="utf-8")
            )

        items = [item for item in payload if isinstance(item, dict) and "sentence_info" in item]
        self.assertEqual(len(items), 1, "expected one Paraformer result with sentence_info")
        sentences = items[0]["sentence_info"]

        timed_tokens: list[tuple[str, int]] = []
        for index, sentence in enumerate(sentences):
            tokens = _TOKEN_RE.findall(sentence["text"])
            spans = sentence["timestamp"]
            self.assertTrue(tokens, f"sentence {index} has no text: {sentence['text']!r}")
            self.assertEqual(
                len(tokens),
                len(spans),
                f"sentence {index} {sentence['text']!r} holds {len(tokens)} tokens for "
                f"{len(spans)} token timestamps",
            )
            timed_tokens.extend(zip(tokens, (span[0] for span in spans)))

        windows = [
            [
                (token, start)
                for token, start in timed_tokens
                if offset - WINDOW_SLACK_MS <= start < offset + clip_ms + WINDOW_SLACK_MS
            ]
            for offset in offsets_ms
        ]
        self.assertEqual(
            sum(len(window) for window in windows),
            len(timed_tokens),
            "every token must start inside one copy of the example, not in the silence",
        )
        reference = "".join(token for token, _ in windows[0])
        self.assertTrue(reference, "the first copy produced no tokens")
        for copy_index, window in enumerate(windows[1:], start=2):
            text = "".join(token for token, _ in window)
            similarity = difflib.SequenceMatcher(None, reference, text).ratio()
            self.assertGreaterEqual(
                similarity,
                0.9,
                f"copy {copy_index} reads {text!r}; copy 1 reads {reference!r}",
            )
        onsets = [window[0][1] - offset for window, offset in zip(windows, offsets_ms)]
        self.assertLess(max(onsets) - min(onsets), 300, f"speech onsets per copy: {onsets} ms")

        from zh_asr.sentence_alignment import sentence_alignment_failures

        self.assertEqual(items[0]["sentence_info_alignment"]["status"], "aligned")
        self.assertEqual(sentence_alignment_failures(payload), [])


@unittest.skipUnless(RUN_MODEL_TESTS, "set ZH_ASR_RUN_MODEL_TESTS=1 to import the FunASR runtime")
class FunASRSentenceBuilderTest(unittest.TestCase):
    def test_funasr_sentence_text_still_needs_realignment_across_vad_chunks(self):
        from funasr.auto.auto_model import _join_vad_texts
        from funasr.utils.timestamp_tools import timestamp_sentence
        from test_sentence_alignment import TIMESTAMPS
        from zh_asr.sentence_alignment import realign_sentence_info, sentence_alignment_failures

        chunks = ["今 天 天 气", "我 们 出 去", "走 走 吧"]
        timestamps = [list(span) for span in TIMESTAMPS]
        punctuation_ids = [1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 3]
        sentences = timestamp_sentence(
            punctuation_ids,
            timestamps,
            _join_vad_texts(chunks),
            return_raw_text=True,
        )
        self.assertEqual(
            sentences[0]["text"],
            "今天天气我，",
            "FunASR no longer shifts sentence text across VAD chunks; "
            "check whether realign_sentence_info is still needed.",
        )

        result = realign_sentence_info(
            {"raw_text": " ".join(chunks), "timestamp": timestamps, "sentence_info": sentences}
        )

        self.assertEqual(
            [sentence["text"] for sentence in result["sentence_info"]],
            ["今天天气，", "我们出去。", "走走吧。"],
        )
        self.assertEqual(sentence_alignment_failures(result), [])


def _write_repeated_example(target: Path) -> tuple[list[int], int]:
    with wave.open(str(EXAMPLE), "rb") as source:
        params = source.getparams()
        frames = source.readframes(source.getnframes())
    rate = params.framerate
    clip_ms = round(params.nframes * 1000 / rate)
    silence = b"\x00" * (round(rate * GAP_MS / 1000) * params.sampwidth * params.nchannels)
    offsets_ms = [index * (clip_ms + GAP_MS) for index in range(COPIES)]
    with wave.open(str(target), "wb") as output:
        output.setnchannels(params.nchannels)
        output.setsampwidth(params.sampwidth)
        output.setframerate(rate)
        output.writeframes(silence.join([frames] * COPIES))
    return offsets_ms, clip_ms


if __name__ == "__main__":
    unittest.main()

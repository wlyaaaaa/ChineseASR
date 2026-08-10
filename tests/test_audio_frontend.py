import hashlib
import os
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


class AudioFrontendTests(unittest.TestCase):
    def _write_wav(
        self,
        path: Path,
        *,
        channels: int = 1,
        sample_width: int = 2,
        sample_rate: int = 16000,
        frames: int = 1600,
    ) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(sample_width)
            handle.setframerate(sample_rate)
            handle.writeframes(b"\0" * frames * channels * sample_width)

    def test_compliant_pcm_wav_is_used_without_conversion(self):
        from zh_asr.audio_frontend import prepare_pcm16_mono

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "source.wav"
            self._write_wav(audio)

            prepared = prepare_pcm16_mono(audio, root / "derived")

        self.assertEqual(prepared.path, audio)
        self.assertFalse(prepared.converted)
        self.assertEqual(prepared.sample_rate, 16000)
        self.assertEqual(prepared.channels, 1)
        self.assertEqual(prepared.sample_width_bytes, 2)
        self.assertEqual(prepared.source_sha256, prepared.derivative_sha256)

    def test_compliant_pcm_wav_can_be_materialized_as_a_verified_owner_file(self):
        from zh_asr.audio_frontend import (
            prepare_pcm16_mono,
            validate_prepared_audio_owner,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "source.wav"
            derived_dir = root / "outputs" / "_derived"
            self._write_wav(audio, frames=3200)

            prepared = prepare_pcm16_mono(
                audio,
                derived_dir,
                materialize_owner=True,
            )
            evidence = prepared.as_dict()

            self.assertNotEqual(audio.resolve(), prepared.path)
            self.assertEqual(derived_dir.resolve(), prepared.path.parent)
            self.assertTrue(prepared.path.exists())
            self.assertFalse(prepared.converted)
            self.assertEqual(prepared.source_sha256, prepared.derivative_sha256)
            self.assertEqual(evidence["format"], "wav")
            self.assertEqual(evidence["sample_rate"], 16000)
            self.assertEqual(evidence["channels"], 1)
            self.assertEqual(evidence["sample_width"], 2)
            self.assertEqual(evidence["sample_width_bytes"], 2)
            self.assertAlmostEqual(evidence["duration_sec"], 0.2)
            validate_prepared_audio_owner(prepared, derived_dir)

    def test_owner_validation_fails_closed_for_tamper_missing_or_multiple_derivatives(self):
        from zh_asr.audio_frontend import (
            PreparedAudioIntegrityError,
            prepare_pcm16_mono,
            validate_prepared_audio_owner,
        )

        def tamper(prepared, _derived_dir):
            prepared.path.write_bytes(b"tampered")

        def remove(prepared, _derived_dir):
            prepared.path.unlink()

        def duplicate(_prepared, derived_dir):
            self._write_wav(derived_dir / "unexpected.wav")

        for label, mutate in (
            ("tamper", tamper),
            ("missing", remove),
            ("multiple", duplicate),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                audio = root / "source.wav"
                derived_dir = root / "outputs" / "_derived"
                self._write_wav(audio)
                prepared = prepare_pcm16_mono(
                    audio,
                    derived_dir,
                    materialize_owner=True,
                )

                mutate(prepared, derived_dir)

                with self.assertRaises(PreparedAudioIntegrityError):
                    validate_prepared_audio_owner(prepared, derived_dir)

    def test_owner_materialization_rejects_reparse_directory_chain(self):
        from zh_asr.audio_frontend import (
            PreparedAudioIntegrityError,
            prepare_pcm16_mono,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            outside = root / "outside"
            linked_out_dir = root / "linked-output"
            self._write_wav(source)
            outside.mkdir()
            try:
                linked_out_dir.symlink_to(outside, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"directory symlink unavailable: {exc}")

            with self.assertRaises(PreparedAudioIntegrityError):
                prepare_pcm16_mono(
                    source,
                    linked_out_dir / "_derived",
                    materialize_owner=True,
                )

            self.assertEqual(list(outside.iterdir()), [])

    def test_owner_materialization_rejects_preexisting_partial_hardlink(self):
        from zh_asr.audio_frontend import (
            PreparedAudioIntegrityError,
            prepare_pcm16_mono,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            sentinel = root / "sentinel.wav"
            derived_dir = root / "outputs" / "_derived"
            derived_dir.mkdir(parents=True)
            self._write_wav(source, frames=1600)
            self._write_wav(sentinel, frames=3200)
            sentinel_bytes = sentinel.read_bytes()
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            partial = derived_dir / (
                f"source.{source_hash[:16]}.16k-mono.partial.wav"
            )
            os.link(sentinel, partial)

            with self.assertRaises(PreparedAudioIntegrityError):
                prepare_pcm16_mono(
                    source,
                    derived_dir,
                    materialize_owner=True,
                )

            self.assertEqual(sentinel.read_bytes(), sentinel_bytes)
            self.assertEqual(partial.stat().st_nlink, 2)

    def test_owner_materialization_rejects_preexisting_partial_reparse(self):
        from zh_asr.audio_frontend import (
            PreparedAudioIntegrityError,
            prepare_pcm16_mono,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            sentinel = root / "sentinel.wav"
            derived_dir = root / "outputs" / "_derived"
            derived_dir.mkdir(parents=True)
            self._write_wav(source, frames=1600)
            self._write_wav(sentinel, frames=3200)
            sentinel_bytes = sentinel.read_bytes()
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            partial = derived_dir / (
                f"source.{source_hash[:16]}.16k-mono.partial.wav"
            )
            try:
                partial.symlink_to(sentinel)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"file symlink unavailable: {exc}")

            with self.assertRaises(PreparedAudioIntegrityError):
                prepare_pcm16_mono(
                    source,
                    derived_dir,
                    materialize_owner=True,
                )

            self.assertEqual(sentinel.read_bytes(), sentinel_bytes)
            self.assertTrue(partial.is_symlink())

    def test_noncompliant_input_uses_deterministic_ffmpeg_derivative(self):
        from zh_asr.audio_frontend import prepare_pcm16_mono

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "call.mp3"
            source.write_bytes(b"fake mp3")
            derived_dir = root / "derived"

            def fake_run(command, **kwargs):
                destination = Path(command[-1])
                self._write_wav(destination)
                return type("Completed", (), {"stdout": "", "stderr": ""})()

            with (
                patch("zh_asr.audio_frontend._ffmpeg_version", return_value="ffmpeg test"),
                patch("zh_asr.audio_frontend.subprocess.run", side_effect=fake_run) as run,
            ):
                first = prepare_pcm16_mono(source, derived_dir)
                second = prepare_pcm16_mono(source, derived_dir)

            self.assertTrue(first.converted)
            self.assertEqual(first.path, second.path)
            self.assertTrue(first.path.exists())
            self.assertEqual(first.conversion_command, second.conversion_command)
            self.assertTrue(second.conversion_command)
            self.assertEqual(first.sample_rate, 16000)
            self.assertEqual(first.channels, 1)
            self.assertEqual(first.sample_width_bytes, 2)
            self.assertNotEqual(first.source_sha256, first.derivative_sha256)
            self.assertNotEqual(source.resolve(), first.path)
            self.assertEqual(first.as_dict()["format"], "wav")
            self.assertEqual(first.as_dict()["sample_width"], 2)
            self.assertEqual(run.call_count, 1)
            command = run.call_args.args[0]
            self.assertIn("-ac", command)
            self.assertIn("-ar", command)
            self.assertIn("16000", command)
            self.assertIn("pcm_s16le", command)

    def test_conversion_failure_does_not_leave_partial_derivative(self):
        from zh_asr.audio_frontend import prepare_pcm16_mono

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "broken.mp3"
            source.write_bytes(b"broken")
            derived_dir = root / "derived"

            with (
                patch("zh_asr.audio_frontend._ffmpeg_version", return_value="ffmpeg test"),
                patch(
                    "zh_asr.audio_frontend.subprocess.run",
                    side_effect=RuntimeError("ffmpeg failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "ffmpeg failed"):
                    prepare_pcm16_mono(source, derived_dir)

            self.assertEqual(list(derived_dir.glob("*.partial.wav")), [])

    def test_conversion_timeout_is_actionable_and_removes_partial_output(self):
        from zh_asr.audio_frontend import AudioConversionTimeout, prepare_pcm16_mono

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "slow.mp3"
            source.write_bytes(b"slow")
            derived_dir = root / "derived"

            def time_out(command, **kwargs):
                Path(command[-1]).write_bytes(b"partial")
                raise subprocess.TimeoutExpired(cmd=command, timeout=kwargs["timeout"])

            with (
                patch("zh_asr.audio_frontend._ffmpeg_version", return_value="ffmpeg test"),
                patch("zh_asr.audio_frontend.subprocess.run", side_effect=time_out) as run,
            ):
                with self.assertRaisesRegex(AudioConversionTimeout, "0.1"):
                    prepare_pcm16_mono(
                        source,
                        derived_dir,
                        ffmpeg="ffmpeg",
                        timeout_sec=0.1,
                    )

            self.assertEqual(run.call_args.kwargs["timeout"], 0.1)
            self.assertEqual(list(derived_dir.glob("*.partial.wav")), [])


if __name__ == "__main__":
    unittest.main()

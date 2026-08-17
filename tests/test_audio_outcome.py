import tempfile
import unittest
import wave
from pathlib import Path


class AudioOutcomeTests(unittest.TestCase):
    def test_zero_pcm_requires_nonempty_negative_evidence_for_no_speech(self):
        from zh_asr.audio_outcome import (
            build_objective_result,
            validate_objective_result,
        )

        with tempfile.TemporaryDirectory() as tmp:
            audio = _write_wav(Path(tmp) / "silence.wav", seconds=1, value=0)
            payload = build_objective_result(
                audio_path=audio,
                mode="strict",
                engines=["qwen", "sensevoice"],
                primary_text="",
                secondary_text="",
                primary_result={"text": ""},
                secondary_result={"text": ""},
            )

        self.assertEqual(payload["objective_outcome"], "no_speech_detected")
        self.assertEqual(payload["empty_semantics"], "confirmed_no_speech")
        self.assertTrue(payload["negative_evidence"]["artifact"]["non_empty"])
        self.assertEqual(payload["media_kind"], "audio")
        self.assertEqual(payload["execution"]["status"], "completed")
        self.assertEqual(payload["coverage"]["status"], "complete")
        self.assertEqual(payload["quality"]["status"], "sufficient")
        self.assertGreater(payload["negative_evidence"]["size_bytes"], 0)
        self.assertTrue(payload["negative_evidence"]["sha256"])
        self.assertGreater(payload["negative_evidence"]["artifact"]["size_bytes"], 0)
        self.assertTrue(payload["negative_evidence"]["artifact"]["sha256"])
        negative = payload["negative_evidence"]["artifact"]
        self.assertEqual(negative["source_audio_sha256"], payload["audio"]["raw_sha256"])
        self.assertEqual(negative["request_sha256"], payload["request"]["sha256"])
        self.assertEqual(negative["processor_config_sha256"], payload["processor_config_sha256"])
        self.assertEqual(validate_objective_result(payload), [])

    def test_model_provenance_retains_options_revision_version_and_config_hash(self):
        from zh_asr.audio_outcome import build_objective_result, canonical_json_sha256

        with tempfile.TemporaryDirectory() as tmp:
            audio = _write_wav(Path(tmp) / "silence.wav", seconds=1, value=0)
            provenance = {
                "adapter": "funasr",
                "model": "iic/SenseVoiceSmall",
                "registry_role": "anchor",
                "options": {"runtime_version": "1.4.2", "model_revision": "rev-1"},
            }
            payload = build_objective_result(
                audio_path=audio,
                mode="quick",
                engines=["sensevoice"],
                primary_text="",
                primary_provenance=provenance,
            )

        model = payload["models"][0]
        self.assertEqual(model["revision"], "rev-1")
        self.assertEqual(model["version"], "1.4.2")
        self.assertEqual(model["config_sha256"], canonical_json_sha256({
            key: model[key]
            for key in ("engine", "adapter", "model", "registry_role", "options", "runtime_identity")
        }))
        self.assertEqual(payload["request"]["model_config_sha256"], payload["model_config_sha256"])

    def test_nonzero_audio_without_vad_stays_indeterminate(self):
        from zh_asr.audio_outcome import build_objective_result

        with tempfile.TemporaryDirectory() as tmp:
            audio = _write_wav(Path(tmp) / "noise.wav", seconds=1, value=1200)
            payload = build_objective_result(
                audio_path=audio,
                mode="strict",
                engines=["qwen", "sensevoice"],
                primary_text="",
                secondary_text="",
            )

        self.assertEqual(payload["objective_outcome"], "indeterminate")
        self.assertIsNone(payload["negative_evidence"])

    def test_text_with_one_engine_failure_remains_transcribed_but_low_confidence(self):
        from zh_asr.audio_outcome import build_objective_result

        with tempfile.TemporaryDirectory() as tmp:
            audio = _write_wav(Path(tmp) / "speech.wav", seconds=1, value=1200)
            payload = build_objective_result(
                audio_path=audio,
                mode="strict",
                engines=["qwen", "sensevoice"],
                primary_text="可观察文本",
                secondary_text="",
                secondary_error="engine timeout",
            )

        self.assertEqual(payload["objective_outcome"], "speech_transcribed")
        self.assertEqual(payload["execution"]["status"], "failed")
        self.assertEqual(payload["quality"]["status"], "low_confidence")

    def test_vad_segments_distinguish_speech_without_text(self):
        from zh_asr.audio_outcome import build_objective_result

        with tempfile.TemporaryDirectory() as tmp:
            audio = _write_wav(Path(tmp) / "speech.wav", seconds=1, value=1200)
            result = {
                "text": "",
                "speech_detection": {
                    "status": "speech_detected",
                    "segments": [[100, 800]],
                    "coverage_complete": True,
                },
            }
            payload = build_objective_result(
                audio_path=audio,
                mode="quick",
                engines=["sensevoice"],
                primary_text="",
                primary_result=result,
            )

        self.assertEqual(
            payload["objective_outcome"],
            "speech_detected_but_not_transcribable",
        )
        self.assertEqual(payload["confidence"], "deferred")

    def test_vad_zero_segments_negative_evidence_keeps_actual_detector_telemetry(self):
        from zh_asr.audio_outcome import build_objective_result, validate_objective_result

        with tempfile.TemporaryDirectory() as tmp:
            audio = _write_wav(Path(tmp) / "quiet.wav", seconds=1, value=1200)
            result = {
                "text": "",
                "speech_detection": {
                    "status": "no_speech_detected",
                    "segments": [],
                    "coverage_complete": True,
                    "processor": "funasr-vad",
                    "processor_version": "funasr-auto-model",
                    "model": "iic/fsmn-vad",
                    "config_sha256": "vad-config",
                },
            }
            payload = build_objective_result(
                audio_path=audio,
                mode="quick",
                engines=["sensevoice"],
                primary_text="",
                primary_result=result,
            )

        self.assertEqual(payload["objective_outcome"], "no_speech_detected")
        self.assertEqual(
            payload["negative_evidence"]["artifact"]["detection"],
            payload["detection"],
        )
        self.assertEqual(validate_objective_result(payload), [])

    def test_incomplete_vad_does_not_upgrade_empty_text(self):
        from zh_asr.audio_outcome import build_objective_result

        with tempfile.TemporaryDirectory() as tmp:
            audio = _write_wav(Path(tmp) / "speech.wav", seconds=1, value=1200)
            payload = build_objective_result(
                audio_path=audio,
                mode="quick",
                engines=["sensevoice"],
                primary_text="",
                primary_result={
                    "text": "",
                    "speech_detection": {
                        "status": "speech_detected",
                        "segments": [[100, 800]],
                        "coverage_complete": False,
                    },
                },
            )

        self.assertEqual(payload["objective_outcome"], "indeterminate")
        self.assertEqual(payload["coverage"]["status"], "partial")

    def test_no_speech_cross_field_gate_rejects_unqualified_payload(self):
        from zh_asr.audio_outcome import build_objective_result, validate_objective_result

        with tempfile.TemporaryDirectory() as tmp:
            audio = _write_wav(Path(tmp) / "silence.wav", seconds=1, value=0)
            payload = build_objective_result(
                audio_path=audio,
                mode="quick",
                engines=["sensevoice"],
                primary_text="",
            )
        payload["quality"]["status"] = "unknown"
        self.assertTrue(any("completed execution" in failure for failure in validate_objective_result(payload)))

    def test_long_aggregate_rejects_gap_before_no_speech(self):
        from zh_asr.audio_outcome import aggregate_objective_result

        with tempfile.TemporaryDirectory() as tmp:
            audio = _write_wav(Path(tmp) / "long.wav", seconds=2, value=0)
            child = {
                "objective_outcome": "no_speech_detected",
                "execution": {"status": "completed"},
                "coverage": {"status": "complete"},
                "quality": {"status": "sufficient"},
                "idempotency_key": "child",
                "raw_artifacts": [],
                "audio": {
                    "coverage": {
                        "start_ms": 0,
                        "end_ms": 900,
                        "excluded_ranges_ms": [],
                        "complete": True,
                    }
                },
            }
            payload = aggregate_objective_result(
                audio_path=audio,
                mode="long-strict",
                engines=["qwen", "sensevoice"],
                children=[child, {**child, "idempotency_key": "child-2", "audio": {"coverage": {
                    "start_ms": 1000,
                    "end_ms": 2000,
                    "excluded_ranges_ms": [],
                    "complete": True,
                }} }],
                request={"duration_ms": 2000},
            )

        self.assertEqual(payload["objective_outcome"], "indeterminate")
        self.assertEqual(payload["coverage"]["status"], "partial")
        self.assertIsNone(payload["negative_evidence"])

    def test_long_aggregate_allows_configured_overlap(self):
        from zh_asr.audio_outcome import aggregate_objective_result, validate_objective_result

        with tempfile.TemporaryDirectory() as tmp:
            audio = _write_wav(Path(tmp) / "long.wav", seconds=2, value=0)
            child = {
                "objective_outcome": "no_speech_detected",
                "execution": {"status": "completed"},
                "coverage": {"status": "complete"},
                "quality": {"status": "sufficient"},
                "idempotency_key": "child",
                "raw_artifacts": [],
                "audio": {
                    "coverage": {
                        "start_ms": 0,
                        "end_ms": 1100,
                        "excluded_ranges_ms": [],
                        "complete": True,
                    }
                },
            }
            payload = aggregate_objective_result(
                audio_path=audio,
                mode="long-strict",
                engines=["qwen", "sensevoice"],
                children=[
                    child,
                    {
                        **child,
                        "idempotency_key": "child-2",
                        "audio": {
                            "coverage": {
                                "start_ms": 1000,
                                "end_ms": 2000,
                                "excluded_ranges_ms": [],
                                "complete": True,
                            }
                        },
                    },
                ],
                request={"duration_ms": 2000},
            )

        self.assertEqual(payload["objective_outcome"], "no_speech_detected")
        self.assertEqual(payload["coverage"]["status"], "complete")
        self.assertGreater(payload["audio"]["coverage"]["overlap_ms"], 0)
        self.assertEqual(validate_objective_result(payload), [])

    def test_caller_binding_is_transparent_and_hashed(self):
        from zh_asr.audio_outcome import build_objective_result

        with tempfile.TemporaryDirectory() as tmp:
            audio = _write_wav(Path(tmp) / "silence.wav", seconds=1, value=0)
            binding = {"opaque_ref": "caller-owned"}
            payload = build_objective_result(
                audio_path=audio,
                mode="quick",
                engines=["sensevoice"],
                primary_text="",
                caller_binding=binding,
            )

        self.assertEqual(payload["caller_binding"], binding)
        self.assertEqual(payload["request"]["caller_binding"], binding)
        self.assertEqual(payload["request"]["sha256"], payload["idempotency_basis"]["request_sha256"])


def _write_wav(path: Path, *, seconds: int, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes((int(value).to_bytes(2, "little", signed=True)) * 16_000 * seconds)
    return path


if __name__ == "__main__":
    unittest.main()

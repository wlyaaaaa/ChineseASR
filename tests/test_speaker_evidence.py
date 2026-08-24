import json
import math
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


HASH_A = "a" * 64


def model_evidence(*, marker="a"):
    from zh_asr.speaker_evidence import SPEAKER_MODEL_EVIDENCE_SCHEMA

    return {
        "schema": SPEAKER_MODEL_EVIDENCE_SCHEMA,
        "model_id": "iic/speech_campplus_sv_zh-cn_16k-common",
        "configured_revision": "v1.0.0",
        "threshold": 0.31,
        "local_model_dir": "C:/private/campp",
        "registry_sha256": marker * 64,
        "runtime": {"package": "funasr", "version": "1.4.2"},
        "files": [
            {
                "path": "campplus_cn_common.bin",
                "bytes": 1,
                "sha256": "b" * 64,
            }
        ],
    }


def write_stereo_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00\x00\x00" * 1600)


class SpeakerEvidenceTests(unittest.TestCase):
    def test_enrollment_persists_only_person_self_vector_and_reference_provenance(self):
        from zh_asr.speaker_evidence import (
            SELF_PERSON_ID,
            SELF_SPEAKER_PROFILE_SCHEMA,
            enroll_self_speaker,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.raw"
            reference.write_bytes(b"reference bytes")
            profile_path = root / "private" / "person-self.json"
            with patch("zh_asr.speaker_evidence.speaker_model_evidence", return_value=model_evidence()), patch(
                "zh_asr.speaker_evidence._prepare_segment_wav",
                return_value="mixed_not_channel_evidence",
            ), patch("zh_asr.speaker_evidence._extract_embedding", return_value=[1.0, 0.0]):
                profile = enroll_self_speaker(
                    reference,
                    start_ms=0,
                    end_ms=1000,
                    inference_basis="该有限参考由本地用户在私有边界内指定为本人候选，可随时替换。",
                    profile_path=profile_path,
                    cache_dir=root / "cache",
                    device="cpu",
                )
            profile_text = profile_path.read_text(encoding="utf-8")
            persisted = json.loads(profile_text)

        self.assertEqual(profile["schema"], SELF_SPEAKER_PROFILE_SCHEMA)
        self.assertEqual(persisted["person_id"], SELF_PERSON_ID)
        self.assertEqual(persisted["embedding"], [1.0, 0.0])
        self.assertEqual(persisted["identity"]["status"], "inferred")
        self.assertTrue(persisted["identity"]["reversible"])
        self.assertEqual(persisted["enrollment_reference"]["segment"]["start_ms"], 0.0)
        self.assertEqual(persisted["enrollment_reference"]["segment"]["channel_binding"], "mixed_not_channel_evidence")
        self.assertIn("sha256", persisted["enrollment_reference"]["source"])
        self.assertNotIn("reference bytes", profile_text)

    def test_profile_replacement_and_explicit_deletion_are_bounded_to_person_self(self):
        from zh_asr.speaker_evidence import (
            SELF_PERSON_ID,
            SpeakerEvidenceError,
            delete_self_speaker_profile,
            enroll_self_speaker,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.raw"
            reference.write_bytes(b"reference")
            profile_path = root / "profile.json"
            common = {
                "start_ms": 0,
                "end_ms": 1000,
                "inference_basis": "这条本地有限参考只作为可替换的本人推定锚。",
                "profile_path": profile_path,
                "cache_dir": root / "cache",
                "device": "cpu",
            }
            with patch("zh_asr.speaker_evidence.speaker_model_evidence", return_value=model_evidence()), patch(
                "zh_asr.speaker_evidence._prepare_segment_wav",
                return_value="mixed_not_channel_evidence",
            ), patch("zh_asr.speaker_evidence._extract_embedding", return_value=[1.0, 0.0]):
                enroll_self_speaker(reference, **common)
                with self.assertRaises(FileExistsError):
                    enroll_self_speaker(reference, **common)
                enroll_self_speaker(reference, replace=True, **common)
            with self.assertRaisesRegex(SpeakerEvidenceError, "confirm-delete"):
                delete_self_speaker_profile(profile_path, confirmation="yes")
            delete_self_speaker_profile(profile_path, confirmation=SELF_PERSON_ID)
            self.assertFalse(profile_path.exists())

    def test_target_evidence_keeps_score_not_target_embedding_and_is_model_bound(self):
        from zh_asr.speaker_evidence import (
            SELF_SPEAKER_EVIDENCE_SCHEMA,
            create_self_speaker_evidence,
            enroll_self_speaker,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.raw"
            target = root / "target.raw"
            reference.write_bytes(b"reference")
            target.write_bytes(b"target")
            profile_path = root / "profile.json"
            common = {
                "profile_path": profile_path,
                "cache_dir": root / "cache",
                "device": "cpu",
            }
            with patch("zh_asr.speaker_evidence.speaker_model_evidence", return_value=model_evidence()), patch(
                "zh_asr.speaker_evidence._prepare_segment_wav",
                return_value="mixed_not_channel_evidence",
            ), patch("zh_asr.speaker_evidence._extract_embedding", side_effect=[[1.0, 0.0], [0.8, 0.6]]):
                enroll_self_speaker(
                    reference,
                    start_ms=0,
                    end_ms=1000,
                    inference_basis="这条本地有限参考只作为可替换的本人推定锚。",
                    **common,
                )
                evidence = create_self_speaker_evidence(target, start_ms=100, end_ms=900, **common)

        self.assertEqual(evidence["schema"], SELF_SPEAKER_EVIDENCE_SCHEMA)
        self.assertAlmostEqual(evidence["score"]["value"], 0.8)
        self.assertEqual(evidence["score"]["comparison"], "above_threshold")
        self.assertNotIn("fusion_band", evidence["score"])
        self.assertNotIn("fusion_ambiguity_margin", evidence["score"])
        self.assertEqual(evidence["identity_status"], "unconfirmed")
        self.assertEqual(evidence["profile"]["identity_status"], "inferred")
        self.assertIn("可替换", evidence["profile"]["enrollment_basis"])
        self.assertEqual(evidence["target"]["segment"]["channel_binding"], "mixed_not_channel_evidence")
        self.assertNotIn("embedding", json.dumps(evidence))
        self.assertIn("sha256", evidence["profile"])
        self.assertIn("files", evidence["model"])

    def test_model_or_runtime_drift_requires_reenrollment(self):
        from zh_asr.speaker_evidence import SpeakerEvidenceError, create_self_speaker_evidence, enroll_self_speaker

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.raw"
            target = root / "target.raw"
            reference.write_bytes(b"reference")
            target.write_bytes(b"target")
            profile_path = root / "profile.json"
            common = {
                "profile_path": profile_path,
                "cache_dir": root / "cache",
                "device": "cpu",
            }
            with patch("zh_asr.speaker_evidence.speaker_model_evidence", return_value=model_evidence()), patch(
                "zh_asr.speaker_evidence._prepare_segment_wav",
                return_value="mixed_not_channel_evidence",
            ), patch("zh_asr.speaker_evidence._extract_embedding", return_value=[1.0, 0.0]):
                enroll_self_speaker(
                    reference,
                    start_ms=0,
                    end_ms=1000,
                    inference_basis="这条本地有限参考只作为可替换的本人推定锚。",
                    **common,
                )
            with patch("zh_asr.speaker_evidence.speaker_model_evidence", return_value=model_evidence(marker="c")):
                with self.assertRaisesRegex(SpeakerEvidenceError, "re-enroll"):
                    create_self_speaker_evidence(target, start_ms=0, end_ms=1000, **common)

    def test_exact_right_channel_extracts_from_original_stereo_not_mixed_audio(self):
        from zh_asr.speaker_evidence import _prepare_segment_wav

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "stereo.wav"
            output = root / "clip.wav"
            write_stereo_wav(source)
            commands = []

            def fake_run(command, **_kwargs):
                commands.append(command)
                with wave.open(str(output), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(16000)
                    handle.writeframes(b"\x00\x00" * 100)
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("zh_asr.speaker_evidence.shutil.which", return_value="ffmpeg"), patch(
                "zh_asr.speaker_evidence.subprocess.run",
                side_effect=fake_run,
            ):
                binding = _prepare_segment_wav(
                    source,
                    output,
                    start_ms=0,
                    end_ms=100,
                    channel="right",
                )

        self.assertEqual(binding, "exact_stereo_channel")
        self.assertIn("-af", commands[0])
        self.assertIn("pan=mono|c0=c1", commands[0])

    def test_right_channel_rejects_mono_source_before_ffmpeg(self):
        from zh_asr.speaker_evidence import SpeakerEvidenceError, _prepare_segment_wav

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "mono.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 100)
            with self.assertRaisesRegex(SpeakerEvidenceError, "stereo"):
                _prepare_segment_wav(
                    source,
                    root / "clip.wav",
                    start_ms=0,
                    end_ms=100,
                    channel="right",
                    ffmpeg="ffmpeg",
                )

    def test_cosine_score_is_normalized(self):
        from zh_asr.speaker_evidence import _cosine_similarity

        self.assertAlmostEqual(_cosine_similarity([3.0, 4.0], [6.0, 8.0]), 1.0)
        self.assertTrue(math.isfinite(_cosine_similarity([1.0, 0.0], [0.0, 1.0])))


if __name__ == "__main__":
    unittest.main()

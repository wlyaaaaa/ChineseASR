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


def write_reference_set(root: Path, names: list[str], *, duplicate_first: bool = False) -> Path:
    references = []
    for index, name in enumerate(names):
        source = root / name
        source.write_bytes(b"same" if duplicate_first and index < 2 else name.encode("utf-8"))
        references.append(
            {
                "source_path": str(source),
                "start_ms": index * 100,
                "end_ms": 1000 + index * 100,
                "channel": "mix",
                "inference_basis": f"第{index + 1}条有限参考有可回查的本人候选依据。",
                "selection_binding": {
                    "kind": "dialogue_role",
                    "evidence_json_sha256": f"{index + 1:x}" * 64,
                    "raw_json_pointer": f"$.sentence_info[{index}]",
                },
            }
        )
    manifest = root / f"references-{len(names)}.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "chinese-asr.person-self-voice-reference-set.v1",
                "inference_basis": "这些有限参考各有可回查依据，只作为可替换的本人推定锚。",
                "references": references,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest


class SpeakerEvidenceTests(unittest.TestCase):
    def test_readback_is_thin_current_and_never_counts_same_source(self):
        from zh_asr.result_writer import canonical_json_sha256
        from zh_asr.speaker_evidence import (
            readback_self_speaker_evidence,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "held-out.raw"
            target.write_bytes(b"audio")
            profile_path = root / "person-self.voice-profile.json"
            profile = {"profile": "current"}
            profile_hash = canonical_json_sha256(profile)
            base = {
                "target": {
                    "source": {"path": str(target), "sha256": "a" * 64, "bytes": 5},
                    "segment": {"start_ms": 100, "end_ms": 900, "channel": "right"},
                },
                "profile": {"sha256": profile_hash},
                "source_relation": "held_out_source",
                "identity_status": "unconfirmed",
                "meaning": "只支持本人候选，不能单独确认身份。",
            }

            def persist(name, document):
                path = root / f"{name}.voice-evidence.json"
                path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
                return path

            current_path = persist("current", base)
            stale = json.loads(json.dumps(base))
            stale["profile"]["sha256"] = "f" * 64
            stale_path = persist("stale", stale)
            drifted = json.loads(json.dumps(base))
            drifted["target"]["source"]["sha256"] = "c" * 64
            drift_path = persist("drift", drifted)
            same_source = json.loads(json.dumps(base))
            same_source["source_relation"] = "enrollment_source"
            same_path = persist("same-source", same_source)

            with patch("zh_asr.speaker_evidence.load_self_speaker_profile", return_value=profile), patch(
                "zh_asr.speaker_evidence.file_sha256", return_value="a" * 64
            ) as hash_file, patch(
                "zh_asr.speaker_attribution._parse_voice_evidence",
                side_effect=lambda documents: tuple(documents),
            ), patch(
                "zh_asr.speaker_attribution._voice_source_relation",
                side_effect=lambda document: document["source_relation"],
            ):
                payload = readback_self_speaker_evidence(
                    target,
                    profile_path=profile_path,
                    evidence_paths=[current_path, stale_path, drift_path, same_path],
                )

        self.assertEqual(hash_file.call_count, 1)
        self.assertEqual(payload["current_valid_evidence_count"], 1)
        self.assertEqual(payload["evidence"][0]["source_relation"], "held_out_source")
        self.assertEqual(
            payload["invalid_evidence_count_by_reason"],
            {
                "target_source_hash_drift": 1,
                "inactive_profile": 1,
                "enrollment_source_not_directional": 1,
            },
        )
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn('"score"', serialized)
        self.assertNotIn('"embedding"', serialized)

    def test_multi_reference_enrollment_persists_one_normalized_centroid_without_paths(self):
        from zh_asr.result_writer import canonical_json_sha256
        from zh_asr.speaker_evidence import (
            CENTROID_AGGREGATION_METHOD,
            SELF_SPEAKER_MULTI_PROFILE_SCHEMA,
            SELF_SPEAKER_REFERENCE_SET_SCHEMA,
            enroll_self_speaker_reference_set,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_reference_set(root, ["z.raw", "a.raw", "m.raw"])
            profile_path = root / "private" / "person-self.json"
            with patch("zh_asr.speaker_evidence.speaker_model_evidence", return_value=model_evidence()), patch(
                "zh_asr.speaker_evidence._prepare_segment_wav",
                return_value="mixed_not_channel_evidence",
            ), patch(
                "zh_asr.speaker_evidence._extract_embedding",
                side_effect=[[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]],
            ):
                profile = enroll_self_speaker_reference_set(
                    manifest,
                    profile_path=profile_path,
                    cache_dir=root / "cache",
                    device="cpu",
                )
            persisted_text = profile_path.read_text(encoding="utf-8")
            persisted = json.loads(persisted_text)

        expected = [
            (1.0 + 0.0 + 2 ** -0.5) / 3,
            (0.0 + 1.0 + 2 ** -0.5) / 3,
        ]
        expected_norm = math.sqrt(sum(item * item for item in expected))
        self.assertEqual(profile["schema"], SELF_SPEAKER_MULTI_PROFILE_SCHEMA)
        self.assertEqual(profile["aggregation"]["method"], CENTROID_AGGREGATION_METHOD)
        self.assertEqual(profile["reference_set"]["reference_count"], 3)
        self.assertAlmostEqual(profile["embedding"][0], expected[0] / expected_norm)
        self.assertAlmostEqual(profile["embedding"][1], expected[1] / expected_norm)
        self.assertAlmostEqual(sum(item * item for item in profile["embedding"]), 1.0)
        self.assertNotIn("source_path", persisted_text)
        self.assertNotIn(str(root), persisted_text)
        self.assertEqual(persisted_text.count('"embedding"'), 1)
        reference_payload = {
            "schema": SELF_SPEAKER_REFERENCE_SET_SCHEMA,
            "references": persisted["reference_set"]["references"],
        }
        self.assertEqual(
            persisted["reference_set"]["sha256"],
            canonical_json_sha256(reference_payload),
        )
        hashes = [item["source"]["sha256"] for item in persisted["reference_set"]["references"]]
        self.assertEqual(hashes, sorted(hashes))

    def test_multi_reference_enrollment_requires_two_or_three_distinct_sources(self):
        from zh_asr.speaker_evidence import SpeakerEvidenceError, enroll_self_speaker_reference_set

        for count in (1, 4):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest = write_reference_set(root, [f"source-{index}.raw" for index in range(count)])
                with self.assertRaisesRegex(SpeakerEvidenceError, "2 to 3"):
                    enroll_self_speaker_reference_set(
                        manifest,
                        profile_path=root / "profile.json",
                        cache_dir=root / "cache",
                        device="cpu",
                    )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_reference_set(root, ["first.raw", "second.raw"], duplicate_first=True)
            with self.assertRaisesRegex(SpeakerEvidenceError, "distinct source hash"):
                enroll_self_speaker_reference_set(
                    manifest,
                    profile_path=root / "profile.json",
                    cache_dir=root / "cache",
                    device="cpu",
                )

    def test_multi_reference_order_and_reference_set_hash_are_deterministic(self):
        from zh_asr.speaker_evidence import enroll_self_speaker_reference_set

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_manifest = write_reference_set(root, ["one.raw", "two.raw", "three.raw"])
            first_payload = json.loads(first_manifest.read_text(encoding="utf-8"))
            second_manifest = root / "reversed.json"
            first_payload["references"].reverse()
            second_manifest.write_text(json.dumps(first_payload, ensure_ascii=False), encoding="utf-8")
            common_patches = (
                patch("zh_asr.speaker_evidence.speaker_model_evidence", return_value=model_evidence()),
                patch("zh_asr.speaker_evidence._prepare_segment_wav", return_value="mixed_not_channel_evidence"),
                patch("zh_asr.speaker_evidence._extract_embedding", return_value=[1.0, 0.0]),
            )
            with common_patches[0], common_patches[1], common_patches[2]:
                first = enroll_self_speaker_reference_set(
                    first_manifest,
                    profile_path=root / "first.json",
                    cache_dir=root / "cache",
                    device="cpu",
                )
            with patch("zh_asr.speaker_evidence.speaker_model_evidence", return_value=model_evidence()), patch(
                "zh_asr.speaker_evidence._prepare_segment_wav",
                return_value="mixed_not_channel_evidence",
            ), patch("zh_asr.speaker_evidence._extract_embedding", return_value=[1.0, 0.0]):
                second = enroll_self_speaker_reference_set(
                    second_manifest,
                    profile_path=root / "second.json",
                    cache_dir=root / "cache",
                    device="cpu",
                )

        self.assertEqual(first["reference_set"], second["reference_set"])
        self.assertEqual(first["embedding"], second["embedding"])

    def test_multi_reference_evidence_can_require_a_held_out_source(self):
        from zh_asr.speaker_evidence import (
            SELF_SPEAKER_MULTI_EVIDENCE_SCHEMA,
            SpeakerEvidenceError,
            create_self_speaker_evidence,
            enroll_self_speaker_reference_set,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_reference_set(root, ["one.raw", "two.raw"])
            target = root / "held-out.raw"
            target.write_bytes(b"held-out")
            profile_path = root / "profile.json"
            with patch("zh_asr.speaker_evidence.speaker_model_evidence", return_value=model_evidence()), patch(
                "zh_asr.speaker_evidence._prepare_segment_wav",
                return_value="mixed_not_channel_evidence",
            ), patch("zh_asr.speaker_evidence._extract_embedding", return_value=[1.0, 0.0]):
                enroll_self_speaker_reference_set(
                    manifest,
                    profile_path=profile_path,
                    cache_dir=root / "cache",
                    device="cpu",
                )
                with self.assertRaisesRegex(SpeakerEvidenceError, "Held-out"):
                    create_self_speaker_evidence(
                        root / "one.raw",
                        start_ms=0,
                        end_ms=500,
                        profile_path=profile_path,
                        cache_dir=root / "cache",
                        device="cpu",
                        require_held_out=True,
                    )
                same_source = create_self_speaker_evidence(
                    root / "one.raw",
                    start_ms=0,
                    end_ms=500,
                    profile_path=profile_path,
                    cache_dir=root / "cache",
                    device="cpu",
                )
                evidence = create_self_speaker_evidence(
                    target,
                    start_ms=0,
                    end_ms=500,
                    profile_path=profile_path,
                    cache_dir=root / "cache",
                    device="cpu",
                    require_held_out=True,
                )

        self.assertEqual(evidence["schema"], SELF_SPEAKER_MULTI_EVIDENCE_SCHEMA)
        self.assertEqual(evidence["source_relation"], "held_out_source")
        self.assertEqual(same_source["source_relation"], "enrollment_source")
        self.assertIn("不作为方向性身份线索", same_source["meaning"])
        self.assertEqual(evidence["profile"]["reference_count"], 2)
        self.assertEqual(len(evidence["profile"]["enrollment_source_sha256s"]), 2)
        self.assertNotIn("embedding", json.dumps(evidence))

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

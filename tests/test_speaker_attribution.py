import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "speaker_attribution"
HASH_A = "a" * 64
HASH_B = "b" * 64


def load_fixture(name: str):
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


def context(*, recording_kind="mono_call", source_hash=None, evidence=None, cohort_id=None):
    from zh_asr.speaker_attribution import SPEAKER_ATTRIBUTION_CONTEXT_SCHEMA

    payload = {
        "schema": SPEAKER_ATTRIBUTION_CONTEXT_SCHEMA,
        "recording_kind": recording_kind,
        "segment_evidence": evidence or [],
    }
    if source_hash is not None:
        payload["recording_audio"] = {"sha256": source_hash}
    if cohort_id is not None:
        payload["stereo_cohort_id"] = cohort_id
    return payload


def voice_document(*, source_hash=HASH_A, start_ms=0, end_ms=980, score=0.75, threshold=0.31, channel="mix", channel_binding="mixed_not_channel_evidence"):
    from zh_asr.speaker_evidence import (
        SELF_PERSON_ID,
        SELF_SPEAKER_EVIDENCE_SCHEMA,
        SPEAKER_MODEL_EVIDENCE_SCHEMA,
    )

    return {
        "schema": SELF_SPEAKER_EVIDENCE_SCHEMA,
        "person_id": SELF_PERSON_ID,
        "target": {
            "source": {"sha256": source_hash},
            "segment": {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "channel": channel,
                "channel_binding": channel_binding,
            },
        },
        "model": {"schema": SPEAKER_MODEL_EVIDENCE_SCHEMA},
        "score": {
            "metric": "cosine_similarity",
            "value": score,
            "threshold": threshold,
        },
    }


class SpeakerAttributionTests(unittest.TestCase):
    def test_unbound_legacy_right_channel_is_not_evidence(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        result = attribute_transcript_result(
            load_fixture("stereo_transcript.json"),
            load_fixture("stereo_context.json"),
        )

        self.assertTrue(result["speaker_attribution_gap"])
        self.assertEqual(result["segments"][0]["attribution_status"], "unknown")
        self.assertEqual(result["segments"][0]["candidate_role"], "unknown")
        self.assertEqual(result["segments"][1]["candidate_role"], "other")

    def test_mono_fixture_stays_unknown_and_never_turns_speaker_number_into_identity(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        result = attribute_transcript_result(
            load_fixture("mono_unknown_transcript.json"),
            load_fixture("mono_unknown_context.json"),
        )

        segment = result["segments"][0]
        self.assertTrue(result["speaker_attribution_gap"])
        self.assertEqual(segment["speaker"], "speaker-1")
        self.assertEqual(segment["attribution_status"], "unknown")
        self.assertEqual(segment["candidate_role"], "unknown")
        self.assertNotIn("42", json.dumps(result, ensure_ascii=False))

    def test_one_unopposed_dialogue_signal_is_reversible_inferred(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        result = attribute_transcript_result(
            load_fixture("mono_unknown_transcript.json"),
            context(
                evidence=[
                    {
                        "index": 0,
                        "dialogue_role": {
                            "candidate_role": "self",
                            "reason": "该句在快递员询问后回答了本人持有物的故障。",
                        },
                    }
                ]
            ),
        )

        segment = result["segments"][0]
        self.assertFalse(result["speaker_attribution_gap"])
        self.assertEqual(segment["attribution_status"], "inferred")
        self.assertEqual(segment["candidate_role"], "self")
        self.assertNotIn("confidence", segment)

    def test_confirmed_requires_authoritative_source_reference(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        base = {
            "index": 0,
            "source_identity": {
                "candidate_role": "other",
                "reason": "来源容器标明该段来自对方。",
            },
        }
        inferred = attribute_transcript_result(
            load_fixture("mono_unknown_transcript.json"),
            context(evidence=[base]),
        )
        self.assertEqual(inferred["segments"][0]["attribution_status"], "inferred")

        confirmed_item = json.loads(json.dumps(base))
        confirmed_item["source_identity"]["authority_ref"] = "wechat-export:conversation/42/message/7"
        confirmed = attribute_transcript_result(
            load_fixture("mono_unknown_transcript.json"),
            context(evidence=[confirmed_item]),
        )
        self.assertEqual(confirmed["segments"][0]["attribution_status"], "confirmed")
        self.assertEqual(confirmed["segments"][0]["candidate_role"], "other")
        self.assertNotIn("confidence", confirmed["segments"][0])

    def test_voice_score_alone_is_inferred_not_confirmed(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        result = attribute_transcript_result(
            load_fixture("stereo_transcript.json"),
            context(recording_kind="other", source_hash=HASH_A),
            voice_evidence=[voice_document()],
        )

        segment = result["segments"][0]
        self.assertEqual(segment["attribution_status"], "inferred")
        self.assertEqual(segment["candidate_role"], "self")
        self.assertNotEqual(segment["attribution_status"], "confirmed")
        self.assertEqual(segment["evidence"][0]["kind"], "voice_similarity")

    def test_voice_and_contact_soft_fuse_with_all_reasons_visible(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        result = attribute_transcript_result(
            load_fixture("stereo_transcript.json"),
            context(
                recording_kind="other",
                source_hash=HASH_A,
                evidence=[
                    {
                        "index": 0,
                        "contact_role": {
                            "candidate_role": "self",
                            "reason": "联系人上下文把这句标为本人回复。",
                        },
                    }
                ],
            ),
            voice_evidence=[voice_document()],
        )

        segment = result["segments"][0]
        self.assertEqual(segment["attribution_status"], "inferred")
        self.assertEqual(segment["candidate_role"], "self")
        self.assertEqual(
            {item["kind"] for item in segment["evidence"]},
            {"contact_role", "voice_similarity"},
        )

    def test_specific_dialogue_reason_can_override_weak_voice_conflict(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        result = attribute_transcript_result(
            load_fixture("stereo_transcript.json"),
            context(
                recording_kind="other",
                source_hash=HASH_A,
                evidence=[
                    {
                        "index": 0,
                        "dialogue_role": {
                            "candidate_role": "self",
                            "reason": "对话顺序像是本人回答。",
                        },
                    }
                ],
            ),
            voice_evidence=[voice_document(score=0.10)],
        )

        segment = result["segments"][0]
        self.assertEqual(segment["attribution_status"], "inferred")
        self.assertEqual(segment["candidate_role"], "self")
        self.assertIn("不能替代这条具体的来源/语义判断", segment["basis"])

    def test_conflicting_contextual_judgements_stay_unknown(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        result = attribute_transcript_result(
            load_fixture("mono_unknown_transcript.json"),
            context(
                evidence=[
                    {
                        "index": 0,
                        "dialogue_role": {
                            "candidate_role": "self",
                            "reason": "这句回答了对方刚提出的本人事项。",
                        },
                        "semantic_role": {
                            "candidate_role": "other",
                            "reason": "句义又明确指向对方在陈述自己的动作。",
                        },
                    }
                ]
            ),
        )

        segment = result["segments"][0]
        self.assertEqual(segment["attribution_status"], "unknown")
        self.assertEqual(segment["candidate_role"], "unknown")
        self.assertIn("判断彼此冲突", segment["basis"])

    def test_near_threshold_voice_is_unknown_without_other_signal(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        result = attribute_transcript_result(
            load_fixture("stereo_transcript.json"),
            context(recording_kind="other", source_hash=HASH_A),
            voice_evidence=[voice_document(score=0.32)],
        )

        segment = result["segments"][0]
        self.assertEqual(segment["attribution_status"], "unknown")
        self.assertEqual(segment["evidence"][0]["candidate_role"], "unknown")
        self.assertNotIn("weight", segment["evidence"][0])

    def test_xiaomi_right_channel_requires_hash_bound_exact_extraction(self):
        from zh_asr.speaker_attribution import (
            XIAOMI_APP_STEREO_COHORT_ID,
            attribute_transcript_result,
        )

        result = attribute_transcript_result(
            load_fixture("stereo_transcript.json"),
            context(
                recording_kind="xiaomi_app_stereo",
                source_hash=HASH_A,
                cohort_id=XIAOMI_APP_STEREO_COHORT_ID,
            ),
            voice_evidence=[
                voice_document(
                    channel="right",
                    channel_binding="exact_stereo_channel",
                )
            ],
        )

        segment = result["segments"][0]
        self.assertEqual(segment["attribution_status"], "inferred")
        self.assertEqual(segment["candidate_role"], "self")
        self.assertIn("xiaomi_exact_stereo_channel", {item["kind"] for item in segment["evidence"]})

    def test_mismatched_voice_source_is_ignored(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        result = attribute_transcript_result(
            load_fixture("stereo_transcript.json"),
            context(recording_kind="other", source_hash=HASH_A),
            voice_evidence=[voice_document(source_hash=HASH_B)],
        )

        self.assertEqual(result["segments"][0]["attribution_status"], "unknown")

    def test_voice_evidence_requires_context_recording_hash(self):
        from zh_asr.speaker_attribution import SpeakerAttributionError, attribute_transcript_result

        with self.assertRaisesRegex(SpeakerAttributionError, "recording_audio.sha256"):
            attribute_transcript_result(
                load_fixture("stereo_transcript.json"),
                context(recording_kind="other"),
                voice_evidence=[voice_document()],
            )

    def test_mono_without_timestamps_stays_unknown_even_with_positive_context(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        result = attribute_transcript_result(
            [{"text": "我这边没法充电。", "spk": 1}],
            context(
                evidence=[
                    {
                        "index": 0,
                        "dialogue_role": {
                            "candidate_role": "self",
                            "reason": "句义像是本人描述故障。",
                        },
                    }
                ]
            ),
        )

        self.assertTrue(result["speaker_attribution_gap"])
        self.assertEqual(result["segments"][0]["attribution_status"], "unknown")

    def test_write_speaker_attribution_writes_evidence_explaining_projection(self):
        from zh_asr.speaker_attribution import SPEAKER_ATTRIBUTION_SCHEMA, write_speaker_attribution

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "attribution.json"
            write_speaker_attribution(
                output,
                load_fixture("mono_unknown_transcript.json"),
                context(
                    evidence=[
                        {
                            "index": 0,
                            "semantic_role": {
                                "candidate_role": "self",
                                "reason": "该句自述本人正在处理故障。",
                            },
                        }
                    ]
                ),
            )
            persisted = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(persisted["schema"], SPEAKER_ATTRIBUTION_SCHEMA)
        self.assertEqual(
            set(persisted["segments"][0]),
            {
                "start_ms",
                "end_ms",
                "text",
                "speaker",
                "attribution_status",
                "candidate_role",
                "basis",
                "evidence",
            },
        )


if __name__ == "__main__":
    unittest.main()

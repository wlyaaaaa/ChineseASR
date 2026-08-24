import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "speaker_attribution"


def load_fixture(name: str):
    return json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))


class SpeakerAttributionTests(unittest.TestCase):
    def test_exact_stereo_cohort_marks_right_channel_as_candidate_and_allows_semantic_veto(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        result = attribute_transcript_result(
            load_fixture("stereo_transcript.json"),
            load_fixture("stereo_context.json"),
        )

        self.assertEqual(result["schema"], "chinese-asr.speaker-attribution.v1")
        self.assertFalse(result["speaker_attribution_gap"])
        self.assertEqual(result["segments"][0]["speaker"], "speaker-1")
        self.assertEqual(result["segments"][0]["attribution_status"], "inferred")
        self.assertEqual(result["segments"][0]["candidate_role"], "self")
        self.assertEqual(result["segments"][1]["attribution_status"], "inferred")
        self.assertEqual(result["segments"][1]["candidate_role"], "other")
        self.assertIn("对方承诺", result["segments"][1]["basis"])

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

    def test_timestamped_mono_dialogue_role_can_be_inferred_without_voiceprint(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        context = load_fixture("mono_unknown_context.json")
        context["segment_evidence"] = [
            {
                "index": 0,
                "dialogue_role": {
                    "candidate_role": "self",
                    "reason": "该句在快递员询问后回答了本人持有物的故障。",
                },
            }
        ]

        result = attribute_transcript_result(load_fixture("mono_unknown_transcript.json"), context)

        self.assertFalse(result["speaker_attribution_gap"])
        self.assertEqual(result["segments"][0]["attribution_status"], "inferred")
        self.assertEqual(result["segments"][0]["candidate_role"], "self")

    def test_explicit_source_identity_is_the_only_confirmed_route(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        context = load_fixture("mono_unknown_context.json")
        context["segment_evidence"] = [
            {
                "index": 0,
                "source_identity": {
                    "candidate_role": "other",
                    "reason": "来源容器明确标明该段来自对方。",
                },
            }
        ]

        result = attribute_transcript_result(load_fixture("mono_unknown_transcript.json"), context)

        self.assertFalse(result["speaker_attribution_gap"])
        self.assertEqual(result["segments"][0]["attribution_status"], "confirmed")
        self.assertEqual(result["segments"][0]["candidate_role"], "other")

    def test_right_channel_without_exact_cohort_fails_closed(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        context = load_fixture("stereo_context.json")
        context["stereo_cohort_id"] = "unverified-stereo-source"
        context["segment_evidence"] = [{"index": 0, "channel": "right"}]
        result = attribute_transcript_result(load_fixture("stereo_transcript.json"), context)

        self.assertTrue(result["speaker_attribution_gap"])
        self.assertEqual(result["segments"][0]["candidate_role"], "unknown")

    def test_mono_without_timestamps_fails_closed_even_when_dialogue_role_is_supplied(self):
        from zh_asr.speaker_attribution import attribute_transcript_result

        result = attribute_transcript_result(
            [{"text": "我这边没法充电。", "spk": 1}],
            {
                "schema": "chinese-asr.speaker-attribution-context.v1",
                "recording_kind": "mono_call",
                "segment_evidence": [
                    {
                        "index": 0,
                        "dialogue_role": {
                            "candidate_role": "self",
                            "reason": "句义像是本人描述故障。",
                        },
                    }
                ],
            },
        )

        self.assertTrue(result["speaker_attribution_gap"])
        self.assertEqual(result["segments"][0]["attribution_status"], "unknown")

    def test_write_speaker_attribution_writes_only_the_consumer_projection(self):
        from zh_asr.speaker_attribution import write_speaker_attribution

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "attribution.json"
            write_speaker_attribution(
                output,
                load_fixture("mono_unknown_transcript.json"),
                load_fixture("mono_unknown_context.json"),
            )
            persisted = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(set(persisted), {"schema", "segments", "speaker_attribution_gap"})
        self.assertEqual(
            set(persisted["segments"][0]),
            {"start_ms", "end_ms", "text", "speaker", "attribution_status", "candidate_role", "basis"},
        )


if __name__ == "__main__":
    unittest.main()

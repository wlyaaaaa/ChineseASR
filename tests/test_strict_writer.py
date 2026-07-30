import json
import tempfile
import unittest
from pathlib import Path


class StrictWriterTests(unittest.TestCase):
    def test_write_strict_bundle_outputs_final_audit_and_raw_results(self):
        from zh_asr.strict_writer import write_strict_bundle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "note.wav"
            audio.write_bytes(b"fake wav")

            paths = write_strict_bundle(
                audio_path=audio,
                primary_engine="sensevoice",
                primary_result=[{"text": "明天上午九点去医院。"}],
                secondary_engine="paraformer",
                secondary_result=[{"text": "明天上午九点去会议室。"}],
                out_dir=root / "outputs",
            )

            final_text = paths["final"].read_text(encoding="utf-8")
            audit_text = paths["audit"].read_text(encoding="utf-8")
            audit_json = json.loads(paths["audit_json"].read_text(encoding="utf-8"))

            self.assertTrue(paths["primary_json"].exists())
            self.assertTrue(paths["secondary_json"].exists())
            self.assertTrue(final_text.startswith("# note Strict Transcript"))
            self.assertIn("[疑似]明天上午九点去医院。", final_text)
            self.assertIn("Status: `conflict`", audit_text)
            self.assertIn("## Rule Hits", audit_text)
            self.assertIn("model_conflict", audit_text)
            self.assertIn("明天上午九点去会议室。", audit_text)
            self.assertIn("rule_hits", audit_json)
            self.assertEqual(audit_json["rule_hits"][0]["id"], "model_conflict")
            self.assertEqual(json.loads(paths["primary_json"].read_text(encoding="utf-8")), [{"text": "明天上午九点去医院。"}])

    def test_write_strict_bundle_accepts_expect_empty_for_silence_audits(self):
        from zh_asr.strict_writer import write_strict_bundle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "silence.wav"
            audio.write_bytes(b"fake wav")

            paths = write_strict_bundle(
                audio_path=audio,
                primary_engine="sensevoice",
                primary_result=[{"text": "谢谢观看。"}],
                secondary_engine="paraformer",
                secondary_result=[{"text": ""}],
                out_dir=root / "outputs",
                expect_empty=True,
            )
            audit_json = json.loads(paths["audit_json"].read_text(encoding="utf-8"))

        rule_ids = {hit["id"] for hit in audit_json["rule_hits"]}
        self.assertIn("empty_audio_hallucination", rule_ids)
        self.assertIn("suspicious_stock_phrase", rule_ids)

    def test_write_strict_bundle_emits_machine_readable_evidence_and_review_json(self):
        from zh_asr.strict_writer import write_strict_bundle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "court-call.wav"
            audio.write_bytes(b"fake wav")
            primary_result = [
                {
                    "sentence_info": [
                        {"start": 1000, "end": 2000, "spk": 0, "text": "可以去一楼换票。"},
                        {"start": 3000, "end": 4200, "spk": 1, "text": "远程方式没法调。"},
                    ]
                }
            ]
            secondary_result = [
                {
                    "sentence_info": [
                        {"start": 1000, "end": 2000, "spk": 0, "text": "可以去一楼换票。"},
                        {"start": 3000, "end": 4200, "spk": 1, "text": "远程方式可以调。"},
                    ]
                }
            ]

            paths = write_strict_bundle(
                audio_path=audio,
                primary_engine="fireredasr2-llm",
                primary_result=primary_result,
                secondary_engine="qwen3-asr-1.7b",
                secondary_result=secondary_result,
                out_dir=root / "outputs",
                primary_role="lexical_primary",
                secondary_role="lexical_verifier",
                primary_provenance={"model_revision": "local-test-revision"},
            )

            audit_json = json.loads(paths["audit_json"].read_text(encoding="utf-8"))
            review_json = json.loads(paths["review_json"].read_text(encoding="utf-8"))

        self.assertEqual(audit_json["schema_version"], "2.0")
        self.assertEqual(audit_json["evidence_status"], "verified")
        self.assertEqual(paths["evidence_status"], "verified")
        self.assertEqual(audit_json["engine_evidence"][0]["role"], "lexical_primary")
        self.assertEqual(
            audit_json["engine_evidence"][0]["provenance"]["model_revision"],
            "local-test-revision",
        )
        self.assertTrue(
            audit_json["engine_evidence"][0]["raw_result_reference"].endswith(
                "court-call.fireredasr2-llm.raw.json"
            )
        )
        self.assertEqual(audit_json["disagreements"][0]["scope"], "segment")
        self.assertTrue(audit_json["review_items"])
        self.assertEqual(
            review_json["selection_policy"],
            "primary_preserving_no_majority_vote_no_semantic_rewrite",
        )
        self.assertEqual(review_json["disagreements"], audit_json["disagreements"])
        self.assertEqual(review_json["review_items"], audit_json["review_items"])
        self.assertEqual(review_json["evidence_status"], "verified")

    def test_write_strict_bundle_exposes_primary_fallback_as_provisional(self):
        from zh_asr.strict_writer import write_strict_bundle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "court-call.wav"
            audio.write_bytes(b"fake wav")
            paths = write_strict_bundle(
                audio_path=audio,
                primary_engine="fireredasr2-llm",
                primary_result={
                    "engine": "fireredasr2-llm",
                    "text": "",
                    "error": {"type": "RuntimeError", "message": "worker exited 9"},
                },
                secondary_engine="qwen3-asr-1.7b",
                secondary_result=[{"text": "他目前还没交。"}],
                out_dir=root / "outputs",
                primary_error="RuntimeError: worker exited 9",
            )
            audit_json = json.loads(paths["audit_json"].read_text(encoding="utf-8"))

        self.assertEqual(paths["evidence_status"], "provisional")
        self.assertEqual(audit_json["evidence_status"], "provisional")
        self.assertEqual(audit_json["status"], "engine_failure")
        self.assertEqual(
            audit_json["engine_evidence"][0]["execution_status"],
            "failed",
        )


if __name__ == "__main__":
    unittest.main()

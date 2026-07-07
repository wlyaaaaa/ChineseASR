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


if __name__ == "__main__":
    unittest.main()

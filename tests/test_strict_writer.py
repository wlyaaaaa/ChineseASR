import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class StrictWriterTests(unittest.TestCase):
    def test_strict_bundle_remains_verified_after_whole_directory_move(self):
        from zh_asr.audit import validate_strict_artifact_bundle
        from zh_asr.strict_writer import write_strict_bundle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "note.wav"
            audio.write_bytes(b"fake wav")
            source_dir = root / "source"
            paths = write_strict_bundle(
                audio_path=audio,
                primary_engine="fireredasr2-llm",
                primary_result={
                    "engine": "fireredasr2-llm",
                    "text": "可以去一楼换票。",
                },
                secondary_engine="qwen3-asr-1.7b",
                secondary_result={
                    "engine": "qwen3-asr-1.7b",
                    "text": "可以去一楼换票。",
                },
                out_dir=source_dir,
            )
            audit = json.loads(paths["audit_json"].read_text(encoding="utf-8"))
            review = json.loads(paths["review_json"].read_text(encoding="utf-8"))

            self.assertEqual(
                audit["engine_evidence"][0]["raw_result_reference"],
                paths["primary_json"].name,
            )
            self.assertEqual(
                audit["engine_evidence"][1]["raw_result_reference"],
                paths["secondary_json"].name,
            )
            self.assertEqual(
                audit["bundle_receipt_reference"],
                paths["receipt"].name,
            )
            self.assertEqual(
                review["bundle_receipt_reference"],
                paths["receipt"].name,
            )

            moved_dir = root / "moved"
            source_dir.rename(moved_dir)
            moved_paths = {
                key: moved_dir / Path(value).name
                for key, value in paths.items()
                if isinstance(value, Path)
            }
            status, failures = validate_strict_artifact_bundle(
                moved_paths,
                expected_primary_engine="fireredasr2-llm",
                expected_secondary_engine="qwen3-asr-1.7b",
            )

        self.assertEqual(status, "verified")
        self.assertEqual(failures, [])

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
        from zh_asr.audit import validate_strict_artifact_bundle
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
            status, failures = validate_strict_artifact_bundle(
                paths,
                expected_primary_engine="sensevoice",
                expected_secondary_engine="paraformer",
            )

        rule_ids = {hit["id"] for hit in audit_json["rule_hits"]}
        self.assertIn("empty_audio_hallucination", rule_ids)
        self.assertIn("suspicious_stock_phrase", rule_ids)
        self.assertEqual(status, "verified")
        self.assertEqual(failures, [])

    def test_successful_empty_dual_engine_bundle_remains_verified_but_unclear(self):
        from zh_asr.audit import validate_strict_artifact_bundle
        from zh_asr.strict_writer import write_strict_bundle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "silence.wav"
            audio.write_bytes(b"fake wav")
            paths = write_strict_bundle(
                audio_path=audio,
                primary_engine="sensevoice",
                primary_result={"engine": "sensevoice", "text": "", "error": None},
                secondary_engine="paraformer",
                secondary_result={"engine": "paraformer", "text": "", "error": None},
                out_dir=root / "outputs",
                expect_empty=True,
            )

            audit_json = json.loads(paths["audit_json"].read_text(encoding="utf-8"))
            status, failures = validate_strict_artifact_bundle(
                paths,
                expected_primary_engine="sensevoice",
                expected_secondary_engine="paraformer",
            )

        self.assertEqual(audit_json["status"], "unclear")
        self.assertEqual(status, "verified")
        self.assertEqual(failures, [])

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
        from zh_asr.audit import validate_strict_artifact_bundle
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
            evidence_status, failures = validate_strict_artifact_bundle(
                paths,
                expected_primary_engine="fireredasr2-llm",
                expected_secondary_engine="qwen3-asr-1.7b",
            )

        self.assertEqual(paths["evidence_status"], "provisional")
        self.assertEqual(audit_json["evidence_status"], "provisional")
        self.assertEqual(audit_json["status"], "engine_failure")
        self.assertEqual(evidence_status, "provisional")
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["role"], "lexical_primary")
        self.assertEqual(
            audit_json["engine_evidence"][0]["execution_status"],
            "failed",
        )

    def test_write_strict_bundle_emits_receipt_binding_every_critical_artifact(self):
        from zh_asr.audit import validate_strict_artifact_bundle
        from zh_asr.strict_writer import (
            STRICT_BUNDLE_ARTIFACT_KEYS,
            STRICT_BUNDLE_RECEIPT_SCHEMA_VERSION,
            write_strict_bundle,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "note.wav"
            audio.write_bytes(b"fake wav")
            paths = write_strict_bundle(
                audio_path=audio,
                primary_engine="fireredasr2-llm",
                primary_result={"engine": "fireredasr2-llm", "text": "可以去一楼换票。"},
                secondary_engine="qwen3-asr-1.7b",
                secondary_result={"engine": "qwen3-asr-1.7b", "text": "可以去一楼换票。"},
                out_dir=root / "outputs",
            )

            receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
            status, failures = validate_strict_artifact_bundle(
                paths,
                expected_primary_engine="fireredasr2-llm",
                expected_secondary_engine="qwen3-asr-1.7b",
            )

            self.assertEqual(
                receipt["schema_version"],
                STRICT_BUNDLE_RECEIPT_SCHEMA_VERSION,
            )
            self.assertEqual(set(receipt["artifacts"]), set(STRICT_BUNDLE_ARTIFACT_KEYS))
            for key in STRICT_BUNDLE_ARTIFACT_KEYS:
                artifact_path = Path(paths[key])
                entry = receipt["artifacts"][key]
                self.assertEqual(entry["path"], artifact_path.name)
                self.assertEqual(entry["size_bytes"], artifact_path.stat().st_size)
                self.assertEqual(
                    entry["sha256"],
                    hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                )
            self.assertEqual(status, "verified")
            self.assertEqual(failures, [])
            self.assertEqual(
                receipt["claims"]["final_text_sha256"],
                hashlib.sha256("可以去一楼换票。".encode("utf-8")).hexdigest(),
            )

    def test_same_engine_cannot_create_verified_two_route_evidence(self):
        from zh_asr.strict_writer import write_strict_bundle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "note.wav"
            audio.write_bytes(b"fake wav")
            paths = write_strict_bundle(
                audio_path=audio,
                primary_engine="same-engine",
                primary_result={"engine": "same-engine", "text": "相同文本。"},
                secondary_engine="same-engine",
                secondary_result={"engine": "same-engine", "text": "相同文本。"},
                out_dir=root / "outputs",
            )

        self.assertEqual(paths["evidence_status"], "unavailable")
        self.assertTrue(
            any(
                "distinct" in failure["error"]
                for failure in paths["evidence_failures"]
            ),
            paths["evidence_failures"],
        )


if __name__ == "__main__":
    unittest.main()

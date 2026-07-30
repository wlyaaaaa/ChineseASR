import json
import tempfile
import unittest
from pathlib import Path


class AuditTests(unittest.TestCase):
    def test_consistent_transcripts_keep_final_text_clean(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="sensevoice",
            primary_text="今天下午三点开会。",
            secondary_engine="paraformer",
            secondary_text="今天下午三点开会。",
        )

        self.assertEqual(report.status, "consistent")
        self.assertEqual(report.final_text, "今天下午三点开会。")
        self.assertNotIn("[疑似]", report.final_text)
        self.assertFalse(report.needs_review)
        self.assertEqual(report.evidence_status, "verified")
        self.assertTrue(
            all(item.execution_status == "succeeded" for item in report.engine_evidence)
        )

    def test_major_conflict_marks_final_text_and_preserves_alternative(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="sensevoice",
            primary_text="明天上午九点去医院。",
            secondary_engine="paraformer",
            secondary_text="明天上午九点去会议室。",
        )

        self.assertEqual(report.status, "conflict")
        self.assertTrue(report.needs_review)
        self.assertTrue(report.final_text.startswith("[疑似]"))
        self.assertIn("明天上午九点去医院。", report.final_text)
        self.assertIn("明天上午九点去会议室。", report.alternatives)

    def test_empty_transcripts_become_unclear(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="sensevoice",
            primary_text="",
            secondary_engine="paraformer",
            secondary_text="",
        )

        self.assertEqual(report.status, "unclear")
        self.assertEqual(report.final_text, "[听不清]")
        self.assertTrue(report.needs_review)

    def test_suspicious_stock_phrase_is_flagged(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="sensevoice",
            primary_text="谢谢观看。",
            secondary_engine="paraformer",
            secondary_text="",
        )

        self.assertIn("suspicious_stock_phrase", report.flags)
        self.assertTrue(report.needs_review)
        self.assertIn("suspicious_stock_phrase", {hit.id for hit in report.rule_hits})

    def test_repetition_rule_marks_final_text_suspicious(self):
        from zh_asr.audit import build_audit_report

        repeated = "今天下午开会今天下午开会今天下午开会"
        report = build_audit_report(
            primary_engine="sensevoice",
            primary_text=repeated,
            secondary_engine="paraformer",
            secondary_text=repeated,
        )

        self.assertEqual(report.status, "suspicious")
        self.assertTrue(report.final_text.startswith("[疑似]"))
        self.assertIn("abnormal_repetition", report.flags)
        self.assertIn("abnormal_repetition", {hit.id for hit in report.rule_hits})

    def test_expect_empty_rule_marks_substantive_text_as_hallucination(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="sensevoice",
            primary_text="开放时间早上九点。",
            secondary_engine="paraformer",
            secondary_text="开放时间早上九点。",
            expect_empty=True,
        )

        self.assertEqual(report.status, "suspicious")
        self.assertIn("empty_audio_hallucination", report.flags)
        self.assertEqual(next(hit for hit in report.rule_hits if hit.id == "empty_audio_hallucination").severity, "high")

    def test_short_semantic_difference_is_conflict(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="sensevoice",
            primary_text="开饭时间早上九点至下午五点。",
            secondary_engine="paraformer",
            secondary_text="开放时间早上九点至下午五点",
        )

        self.assertEqual(report.status, "conflict")
        self.assertTrue(report.needs_review)
        self.assertTrue(report.final_text.startswith("[疑似]"))
        self.assertIn("model_conflict", {hit.id for hit in report.rule_hits})

    def test_final_text_and_comparison_use_simplified_chinese(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="qwen3-asr-1.7b",
            primary_text="開放時間：早上九點至下午五點。",
            secondary_engine="sensevoice",
            secondary_text="开放时间早上九点至下午五点。",
        )

        self.assertEqual(report.status, "consistent")
        self.assertEqual(report.final_text, "开放时间：早上九点至下午五点。")
        self.assertEqual(report.primary_text, "开放时间：早上九点至下午五点。")
        self.assertEqual(report.secondary_text, "开放时间早上九点至下午五点。")
        self.assertEqual(report.similarity, 1.0)
        self.assertEqual(report.rule_hits, ())

    def test_report_records_roles_segment_disagreements_and_review_items_without_voting(self):
        from zh_asr.audit import build_audit_report
        from zh_asr.result_writer import TranscriptSegment

        primary_segments = (
            TranscriptSegment(0, "可以去一楼换票。", 1000, 2000, 0, "$[0]"),
            TranscriptSegment(1, "远程方式没法调。", 3000, 4200, 1, "$[1]"),
        )
        secondary_segments = (
            TranscriptSegment(0, "可以去一楼换票。", 1000, 2000, 0, "$[0]"),
            TranscriptSegment(1, "远程方式可以调。", 3000, 4200, 1, "$[1]"),
        )

        report = build_audit_report(
            primary_engine="fireredasr2-llm",
            primary_text="可以去一楼换票。\n远程方式没法调。",
            secondary_engine="qwen3-asr-1.7b",
            secondary_text="可以去一楼换票。\n远程方式可以调。",
            primary_role="lexical_primary",
            secondary_role="lexical_verifier",
            primary_segments=primary_segments,
            secondary_segments=secondary_segments,
            primary_raw_result_reference="call.firered.raw.json",
            secondary_raw_result_reference="call.qwen.raw.json",
        )

        self.assertEqual(
            report.selection_policy,
            "primary_preserving_no_majority_vote_no_semantic_rewrite",
        )
        self.assertEqual(report.engine_evidence[0].role, "lexical_primary")
        self.assertEqual(
            report.engine_evidence[0].raw_result_reference,
            "call.firered.raw.json",
        )
        self.assertEqual(report.engine_evidence[0].segments[1].text, "远程方式没法调。")
        self.assertEqual(len(report.disagreements), 1)
        disagreement = report.disagreements[0]
        self.assertEqual(disagreement.scope, "segment")
        self.assertEqual(disagreement.primary_segment_index, 1)
        self.assertEqual(disagreement.secondary_segment_index, 1)
        self.assertTrue(disagreement.review_required)
        self.assertEqual(disagreement.audio_start_ms, 3000)
        self.assertEqual(disagreement.audio_end_ms, 4200)
        self.assertEqual(report.review_items[0].disagreement_ids, (disagreement.id,))
        self.assertIn("远程方式没法调。", report.final_text)
        self.assertNotIn("远程方式可以调。", report.final_text)

    def test_engine_failure_marks_evidence_provisional_and_identifies_failed_primary(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="fireredasr2-llm",
            primary_text="",
            secondary_engine="qwen3-asr-1.7b",
            secondary_text="可以去一楼换票。",
            primary_error="RuntimeError: CUDA out of memory",
        )

        self.assertEqual(report.status, "engine_failure")
        self.assertEqual(report.evidence_status, "provisional")
        self.assertEqual(report.engine_evidence[0].engine, "fireredasr2-llm")
        self.assertEqual(report.engine_evidence[0].execution_status, "failed")
        self.assertIn("CUDA out of memory", report.engine_evidence[0].error)
        self.assertEqual(report.engine_evidence[1].execution_status, "succeeded")

    def test_both_engine_failures_make_evidence_unavailable(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="fireredasr2-llm",
            primary_text="",
            secondary_engine="qwen3-asr-1.7b",
            secondary_text="",
            primary_error="RuntimeError: primary failed",
            secondary_error="RuntimeError: secondary failed",
        )

        self.assertEqual(report.status, "engine_failure")
        self.assertEqual(report.evidence_status, "unavailable")

    def test_verified_audit_is_unavailable_when_required_raw_is_missing(self):
        from zh_asr.audit import validate_strict_artifact_bundle

        with tempfile.TemporaryDirectory() as tmp:
            outputs = _write_strict_artifact_fixture(Path(tmp))
            Path(outputs["primary_json"]).unlink()
            status, failures = validate_strict_artifact_bundle(
                outputs,
                expected_primary_engine="fireredasr2-llm",
                expected_secondary_engine="qwen3-asr-1.7b",
            )

        self.assertEqual(status, "unavailable")
        self.assertIn("primary raw JSON", failures[0]["error"])

    def test_verified_audit_is_unavailable_when_raw_json_is_corrupt(self):
        from zh_asr.audit import validate_strict_artifact_bundle

        with tempfile.TemporaryDirectory() as tmp:
            outputs = _write_strict_artifact_fixture(Path(tmp))
            Path(outputs["secondary_json"]).write_text("{", encoding="utf-8")
            status, failures = validate_strict_artifact_bundle(
                outputs,
                expected_primary_engine="fireredasr2-llm",
                expected_secondary_engine="qwen3-asr-1.7b",
            )

        self.assertEqual(status, "unavailable")
        self.assertIn("secondary raw JSON", failures[0]["error"])

    def test_verified_audit_is_unavailable_when_raw_engine_identity_mismatches(self):
        from zh_asr.audit import validate_strict_artifact_bundle

        with tempfile.TemporaryDirectory() as tmp:
            outputs = _write_strict_artifact_fixture(Path(tmp))
            audit_path = Path(outputs["audit_json"])
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["engine_evidence"][0]["engine"] = "wrong-engine"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            status, failures = validate_strict_artifact_bundle(
                outputs,
                expected_primary_engine="fireredasr2-llm",
                expected_secondary_engine="qwen3-asr-1.7b",
            )

        self.assertEqual(status, "unavailable")
        self.assertIn("identity", failures[0]["error"])


def _write_strict_artifact_fixture(root: Path) -> dict[str, str]:
    final = root / "call.strict.md"
    audit_md = root / "call.strict.audit.md"
    audit_json = root / "call.strict.audit.json"
    primary_json = root / "call.fireredasr2-llm.raw.json"
    secondary_json = root / "call.qwen3-asr-1.7b.raw.json"
    final.write_text("# Final", encoding="utf-8")
    audit_md.write_text("# Audit", encoding="utf-8")
    primary_json.write_text(
        json.dumps({"engine": "fireredasr2-llm", "text": "可以去一楼换票。", "error": None}),
        encoding="utf-8",
    )
    secondary_json.write_text(
        json.dumps({"engine": "qwen3-asr-1.7b", "text": "可以去一楼换票。", "error": None}),
        encoding="utf-8",
    )
    audit_json.write_text(
        json.dumps(
            {
                "status": "consistent",
                "evidence_status": "verified",
                "engine_evidence": [
                    {
                        "engine": "fireredasr2-llm",
                        "role": "lexical_primary",
                        "execution_status": "succeeded",
                        "error": None,
                        "raw_result_reference": str(primary_json),
                    },
                    {
                        "engine": "qwen3-asr-1.7b",
                        "role": "lexical_verifier",
                        "execution_status": "succeeded",
                        "error": None,
                        "raw_result_reference": str(secondary_json),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "final": str(final),
        "audit": str(audit_md),
        "audit_json": str(audit_json),
        "primary_json": str(primary_json),
        "secondary_json": str(secondary_json),
    }


if __name__ == "__main__":
    unittest.main()

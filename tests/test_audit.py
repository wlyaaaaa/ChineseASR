import unittest


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


if __name__ == "__main__":
    unittest.main()

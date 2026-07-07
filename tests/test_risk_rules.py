import unittest


class RiskRulesTests(unittest.TestCase):
    def test_empty_audio_hallucination_flags_substantive_text(self):
        from zh_asr.risk_rules import evaluate_risk_rules

        hits = evaluate_risk_rules("", "", "谢谢观看", similarity=1.0, expect_empty=True)

        self.assertIn("empty_audio_hallucination", {hit.id for hit in hits})
        self.assertEqual(_hit(hits, "empty_audio_hallucination").severity, "high")

    def test_suspicious_stock_phrase_flags_template_filler(self):
        from zh_asr.risk_rules import evaluate_risk_rules

        hits = evaluate_risk_rules("谢谢观看。", "", "谢谢观看。", similarity=0.0)

        self.assertIn("suspicious_stock_phrase", {hit.id for hit in hits})
        self.assertIn("谢谢观看", _hit(hits, "suspicious_stock_phrase").evidence)

    def test_abnormal_repetition_flags_repeated_spans(self):
        from zh_asr.risk_rules import evaluate_risk_rules

        text = "今天下午开会今天下午开会今天下午开会"
        hits = evaluate_risk_rules(text, text, text, similarity=1.0)

        self.assertIn("abnormal_repetition", {hit.id for hit in hits})

    def test_model_conflict_has_medium_and_high_severity(self):
        from zh_asr.risk_rules import evaluate_risk_rules

        medium = evaluate_risk_rules("去医院", "去会议室", "去医院", similarity=0.9)
        high = evaluate_risk_rules("去医院", "完全不同的内容", "去医院", similarity=0.5)

        self.assertEqual(_hit(medium, "model_conflict").severity, "medium")
        self.assertEqual(_hit(high, "model_conflict").severity, "high")

    def test_model_conflict_requires_two_substantive_outputs(self):
        from zh_asr.risk_rules import evaluate_risk_rules

        hits = evaluate_risk_rules("谢谢观看。", "", "谢谢观看。", similarity=0.0)

        self.assertNotIn("model_conflict", {hit.id for hit in hits})

    def test_traditional_residue_flags_final_text_that_is_not_simplified(self):
        from zh_asr.risk_rules import evaluate_risk_rules

        hits = evaluate_risk_rules("開放時間", "开放时间", "開放時間", similarity=1.0)

        self.assertIn("traditional_residue", {hit.id for hit in hits})

    def test_long_unpunctuated_text_flags_run_on_output(self):
        from zh_asr.risk_rules import evaluate_risk_rules

        text = "开放时间早上九点至下午五点" * 12
        hits = evaluate_risk_rules(text, text, text, similarity=1.0)

        self.assertIn("long_unpunctuated_text", {hit.id for hit in hits})

    def test_clean_short_consistent_text_has_no_hits(self):
        from zh_asr.risk_rules import evaluate_risk_rules

        hits = evaluate_risk_rules("今天下午三点开会。", "今天下午三点开会。", "今天下午三点开会。", similarity=1.0)

        self.assertEqual(hits, ())


def _hit(hits, rule_id):
    return next(hit for hit in hits if hit.id == rule_id)


if __name__ == "__main__":
    unittest.main()

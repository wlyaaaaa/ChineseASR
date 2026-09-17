import unittest
from zh_asr.audit import build_audit_report
from zh_asr.eval_pack import char_error_rate, standard_char_error_rate
from zh_asr.text_comparison import normalize_comparison, critical_differences


class QualityRegressions(unittest.TestCase):
    def test_critical_differences_are_not_auto_accepted(self):
        prefix = "交付细节已经全部核对过，这里需要明确说明后续安排。" * 8
        pairs = [("合同金额是1.5万元。", "合同金额是15万元。"),
                 ("本次温度为-5度。", "本次温度为5度。"),
                 (prefix + "我同意这个方案。", prefix + "我不同意这个方案。"),
                 ("版本为Python3.11", "版本为Python311")]
        for left, right in pairs:
            with self.subTest(left=left):
                result = build_audit_report("p", left, "s", right)
                self.assertTrue(result.needs_review)
                self.assertTrue(result.review_items)
                self.assertIn("critical_content_difference", result.flags)
                self.assertGreater(char_error_rate(left, right), 0)

    def test_legacy_metric_is_explicit(self):
        self.assertEqual(standard_char_error_rate("1.5万元", "15万元"), 0)
        self.assertGreater(char_error_rate("1.5万元", "15万元"), 0)

    def test_real_subtitle_mention_is_not_hallucination(self):
        result = build_audit_report("p", "请打开字幕，声音太小了。", "s", "请打开字幕，声音太小了。")
        self.assertFalse(result.needs_review)
        self.assertNotIn("suspicious_stock_phrase", result.flags)

    def test_sentence_boundaries_do_not_create_disagreement(self):
        result = build_audit_report("p", "今天开会，明天交付。", "s", "今天开会，明天交付。",
            primary_segments=[{"text": "今天开会，明天交付。"}],
            secondary_segments=[{"text": "今天开会。"}, {"text": "明天交付。"}])
        self.assertFalse(result.needs_review)
        self.assertEqual(result.disagreements, ())
        self.assertEqual(result.review_items, ())

    def test_audit_markers_are_not_recognition_errors(self):
        self.assertEqual(char_error_rate("你好", "[疑似]你好"), 0)

    def test_technical_symbols_and_width(self):
        self.assertEqual(normalize_comparison("１．５万元"), normalize_comparison("1.5万元"))
        self.assertNotEqual(normalize_comparison("C++"), normalize_comparison("C"))
        self.assertTrue(critical_differences("剂量5mg", "剂量5kg"))

    def test_moved_negation_remains_critical(self):
        self.assertIn("negation", critical_differences("我不收钱你收钱", "我收钱你不收钱"))

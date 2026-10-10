import unittest
from unittest.mock import patch


class TextNormalizerTests(unittest.TestCase):
    def test_to_simplified_uses_full_traditional_conversion(self):
        from zh_asr.text_normalizer import to_simplified

        self.assertEqual(to_simplified("軟體與資料庫"), "软体与资料库")

    def test_normalization_preserves_words_and_is_idempotent(self):
        from zh_asr.text_normalizer import to_simplified

        cases = (
            ("文字、文件、核心、软件、信息", "文字、文件、核心、软件、信息"),
            ("文字、文件、核心、軟件、信息", "文字、文件、核心、软件、信息"),
            ("檔案、軟體、資訊、資料庫", "档案、软体、资讯、资料库"),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(to_simplified(text), expected)
                self.assertEqual(to_simplified(to_simplified(text)), expected)

    def test_fallback_preserves_words(self):
        from zh_asr.text_normalizer import to_simplified

        with patch("zh_asr.text_normalizer._opencc_converter", return_value=None):
            self.assertEqual(
                to_simplified("文字、文件、核心、軟件、信息、檔案、軟體、資訊、資料庫"),
                "文字、文件、核心、软件、信息、档案、软体、资讯、资料库",
            )


if __name__ == "__main__":
    unittest.main()

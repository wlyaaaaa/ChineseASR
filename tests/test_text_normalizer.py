import unittest


class TextNormalizerTests(unittest.TestCase):
    def test_to_simplified_uses_full_traditional_conversion(self):
        from zh_asr.text_normalizer import to_simplified

        self.assertEqual(to_simplified("軟體與資料庫"), "软件与数据库")


if __name__ == "__main__":
    unittest.main()

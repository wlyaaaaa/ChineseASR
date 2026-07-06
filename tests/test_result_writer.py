import json
import tempfile
import unittest
from pathlib import Path


class ResultWriterTests(unittest.TestCase):
    def test_write_transcript_bundle_writes_markdown_and_raw_json(self):
        from zh_asr.result_writer import write_transcript_bundle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "sample.wav"
            audio.write_bytes(b"fake wav")
            result = [{"text": "欢迎大家来体验本地中文语音识别。"}]

            paths = write_transcript_bundle(audio, result, root / "outputs", "sensevoice")

            self.assertTrue(paths["markdown"].exists())
            self.assertTrue(paths["json"].exists())
            self.assertIn("欢迎大家", paths["markdown"].read_text(encoding="utf-8"))
            self.assertIn("Engine: `sensevoice`", paths["markdown"].read_text(encoding="utf-8"))
            self.assertEqual(json.loads(paths["json"].read_text(encoding="utf-8")), result)

    def test_write_transcript_bundle_handles_sentence_info_segments(self):
        from zh_asr.result_writer import extract_text

        result = [
            {
                "sentence_info": [
                    {"start": 100, "end": 900, "spk": 0, "text": "第一句"},
                    {"start": 1100, "end": 1700, "spk": 1, "sentence": "第二句"},
                ]
            }
        ]

        self.assertEqual(extract_text(result), "第一句\n第二句")

    def test_extract_text_removes_sensevoice_rich_tags(self):
        from zh_asr.result_writer import extract_text

        result = [{"text": "<|zh|><|NEUTRAL|><|Speech|><|woitn|>开饭时间早上九点。"}]

        self.assertEqual(extract_text(result), "开饭时间早上九点。")


if __name__ == "__main__":
    unittest.main()

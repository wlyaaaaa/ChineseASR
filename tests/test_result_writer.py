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
            self.assertIn(
                "Evidence status: `not_applicable`",
                paths["markdown"].read_text(encoding="utf-8"),
            )
            self.assertEqual(paths["evidence_status"], "not_applicable")
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

    def test_extract_segments_preserves_timing_speaker_and_raw_provenance(self):
        from zh_asr.result_writer import extract_segments

        result = [
            {
                "sentence_info": [
                    {
                        "start": 100,
                        "end": 900,
                        "spk": 0,
                        "text": "<|zh|>可以去一楼换票。",
                    },
                    {
                        "start": 1100,
                        "end": 1700,
                        "spk": 1,
                        "sentence": "远程方式没法调。",
                    },
                ]
            }
        ]

        segments = extract_segments(result)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].index, 0)
        self.assertEqual(segments[0].text, "可以去一楼换票。")
        self.assertEqual(segments[0].start_ms, 100)
        self.assertEqual(segments[0].end_ms, 900)
        self.assertEqual(segments[0].speaker, 0)
        self.assertEqual(segments[0].raw_path, "$[0].sentence_info[0]")
        self.assertEqual(segments[1].raw_path, "$[0].sentence_info[1]")

    def test_empty_sentence_info_falls_back_to_top_level_text(self):
        from zh_asr.result_writer import extract_text

        result = [{"text": "顶层原始转写。", "sentence_info": []}]

        self.assertEqual(extract_text(result), "顶层原始转写。")


if __name__ == "__main__":
    unittest.main()

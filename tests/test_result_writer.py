import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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

    def test_write_transcript_bundle_records_auditable_diarization_request_without_reading_audio(self):
        from zh_asr.result_writer import write_transcript_bundle

        request_options = {
            "speaker_diarization": {
                "mode": "preset",
                "preset_spk_num": 2,
                "identity": "anonymous_only",
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "zh_asr.result_writer.build_objective_result",
                return_value={"objective_outcome": "speech_transcribed"},
            ) as build, patch("zh_asr.result_writer.write_objective_result"):
                write_transcript_bundle(
                    Path("not-read.wav"),
                    [{"text": "测试"}],
                    Path(tmp) / "outputs",
                    "paraformer",
                    request_options=request_options,
                )

        self.assertEqual(
            build.call_args.kwargs["request"],
            {**request_options, "engine": "paraformer"},
        )

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

    def test_file_sha256_hashes_exact_persisted_bytes(self):
        from zh_asr.result_writer import file_sha256

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            payload = '{"text":"原始证据"}\r\n'.encode("utf-8")
            path.write_bytes(payload)

            digest = file_sha256(path)

        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()

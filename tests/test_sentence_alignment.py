import unittest


TIMESTAMPS = [
    [0, 100],
    [100, 200],
    [200, 300],
    [300, 400],
    [1000, 1100],
    [1100, 1200],
    [1200, 1300],
    [1300, 1400],
    [2000, 2100],
    [2100, 2200],
    [2200, 2300],
]


def funasr_shifted_result(*, raw_tokens: bool = True) -> list[dict]:
    """Paraformer output as FunASR 1.4.14 writes it for three Chinese VAD chunks.

    The chunks are 今天天气 / 我们出去 / 走走吧.  ``_join_vad_texts`` fuses 气|我 and
    去|走, so each sentence's text runs ahead of its (correct) timestamps and the
    last sentence keeps only one character.
    """

    sentences = [
        {"text": "今天天气我，", "start": 0, "end": 400, "timestamp": TIMESTAMPS[0:4], "spk": 0},
        {"text": "们出去走走。", "start": 1000, "end": 1400, "timestamp": TIMESTAMPS[4:8], "spk": 1},
        {"text": "吧。", "start": 2000, "end": 2300, "timestamp": TIMESTAMPS[8:11], "spk": 0},
    ]
    item = {
        "key": "synthetic",
        "text": "今天天气，我们出去。走走吧。",
        "timestamp": [list(span) for span in TIMESTAMPS],
        "sentence_info": sentences,
    }
    if raw_tokens:
        item["raw_text"] = "今 天 天 气 我 们 出 去 走 走 吧"
        for sentence, raw in zip(sentences, ("今 天 天 气我", "们 出 去走 走", "吧")):
            sentence["raw_text"] = raw
    return [item]


class SentenceAlignmentTests(unittest.TestCase):
    def test_realign_puts_each_sentence_text_back_on_its_timestamps(self):
        from zh_asr.sentence_alignment import realign_sentence_info, sentence_alignment_failures

        result = realign_sentence_info(funasr_shifted_result())
        sentences = result[0]["sentence_info"]

        self.assertEqual([item["text"] for item in sentences], ["今天天气，", "我们出去。", "走走吧。"])
        self.assertEqual(
            [(item["start"], item["end"]) for item in sentences],
            [(0, 400), (1000, 1400), (2000, 2300)],
        )
        self.assertEqual([item["spk"] for item in sentences], [0, 1, 0])
        self.assertEqual([item["raw_text"] for item in sentences], ["今 天 天 气", "我 们 出 去", "走 走 吧"])
        self.assertEqual(result[0]["sentence_info_alignment"]["status"], "aligned")
        self.assertEqual(result[0]["sentence_info_alignment"]["realigned_sentence_count"], 3)
        self.assertEqual(sentence_alignment_failures(result), [])

    def test_shifted_sentence_text_is_reported_with_or_without_raw_tokens(self):
        from zh_asr.sentence_alignment import sentence_alignment_failures

        for raw_tokens in (True, False):
            with self.subTest(raw_tokens=raw_tokens):
                failures = sentence_alignment_failures(funasr_shifted_result(raw_tokens=raw_tokens))
                self.assertTrue(failures)
                self.assertTrue(failures[0].startswith("$[0]"))

    def test_sentences_without_token_timestamps_are_not_judged(self):
        from zh_asr.sentence_alignment import sentence_alignment_failures

        transcript = [{"sentence_info": [{"start": 100, "end": 800, "spk": 0, "text": "这个充电宝鼓包了。"}]}]

        self.assertEqual(sentence_alignment_failures(transcript), [])

    def test_tokens_after_the_last_closed_sentence_are_kept_without_a_speaker(self):
        from zh_asr.sentence_alignment import realign_sentence_info, sentence_alignment_failures

        spans = [[0, 100], [100, 200], [300, 400], [400, 500]]
        result = realign_sentence_info(
            {
                "raw_text": "好 的 谢 谢",
                "timestamp": spans,
                "sentence_info": [
                    {"text": "好的，", "start": 0, "end": 200, "timestamp": spans[:2], "spk": 2}
                ],
            }
        )

        self.assertEqual([item["text"] for item in result["sentence_info"]], ["好的，", "谢谢"])
        self.assertEqual(
            (result["sentence_info"][1]["start"], result["sentence_info"][1]["end"]),
            (300, 500),
        )
        self.assertNotIn("spk", result["sentence_info"][1])
        self.assertEqual(sentence_alignment_failures(result), [])

    def test_ascii_words_keep_funasr_spacing(self):
        from zh_asr.sentence_alignment import realign_sentence_info

        spans = [[index * 100, index * 100 + 100] for index in range(6)]
        result = realign_sentence_info(
            {
                "raw_text": "我 用 AI 写 代 码",
                "timestamp": spans,
                "sentence_info": [
                    {"text": "我用 AI 写代码。", "start": 0, "end": 600, "timestamp": spans, "spk": 0}
                ],
            }
        )

        self.assertEqual(result["sentence_info"][0]["text"], "我用 AI 写代码。")
        self.assertEqual(result["sentence_info_alignment"]["realigned_sentence_count"], 0)

    def test_unmatched_tokens_set_sentences_aside_without_times(self):
        from zh_asr.result_writer import extract_segments
        from zh_asr.sentence_alignment import realign_sentence_info

        item = funasr_shifted_result()[0]
        item["raw_text"] = "今 天 天 气 我 们 出 去 走 走"
        result = realign_sentence_info(item)

        self.assertNotIn("sentence_info", result)
        self.assertEqual(len(result["sentence_info_unaligned"]), 3)
        self.assertEqual(result["sentence_info_alignment"]["status"], "unaligned")
        self.assertEqual(result["sentence_info_alignment"]["reason"], "token_timestamp_count_mismatch")
        self.assertEqual(
            [(segment.start_ms, segment.end_ms) for segment in extract_segments(result)],
            [(None, None)],
        )


if __name__ == "__main__":
    unittest.main()

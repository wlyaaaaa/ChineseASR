import unittest


class ConfigTests(unittest.TestCase):
    def test_default_engine_is_sensevoice_for_low_hallucination_chinese(self):
        from zh_asr.config import DEFAULT_ENGINE

        self.assertEqual(DEFAULT_ENGINE, "sensevoice")

    def test_sensevoice_uses_modelscope_model_ids(self):
        from zh_asr.config import get_engine_spec

        spec = get_engine_spec("sensevoice")

        self.assertEqual(spec.model, "iic/SenseVoiceSmall")
        self.assertEqual(spec.vad_model, "fsmn-vad")
        self.assertEqual(spec.punc_model, "ct-punc")
        self.assertEqual(spec.spk_model, "cam++")
        self.assertFalse(spec.is_whisper)

    def test_whisper_is_marked_as_fallback_only(self):
        from zh_asr.config import get_engine_spec

        spec = get_engine_spec("whisper-large-v3")

        self.assertTrue(spec.is_whisper)
        self.assertEqual(spec.role, "fallback")

    def test_unknown_engine_raises_clear_error(self):
        from zh_asr.config import get_engine_spec

        with self.assertRaisesRegex(ValueError, "Unknown ASR engine"):
            get_engine_spec("not-a-model")


if __name__ == "__main__":
    unittest.main()


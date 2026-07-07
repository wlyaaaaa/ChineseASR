import unittest
import tempfile
from pathlib import Path


class ConfigTests(unittest.TestCase):
    def test_default_engine_is_sensevoice_for_low_hallucination_chinese(self):
        from zh_asr.config import DEFAULT_ENGINE

        self.assertEqual(DEFAULT_ENGINE, "sensevoice")

    def test_sensevoice_uses_modelscope_model_ids(self):
        from zh_asr.config import get_engine_spec

        spec = get_engine_spec("sensevoice")

        self.assertEqual(spec.adapter, "funasr")
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

    def test_engines_are_loaded_from_yaml_model_registry(self):
        from zh_asr.config import load_model_config

        config = load_model_config()

        self.assertEqual(config.default_engine, "sensevoice")
        self.assertEqual(config.strict_primary_engine, "qwen3-asr-1.7b")
        self.assertEqual(config.strict_secondary_engine, "sensevoice")
        self.assertIn("speech_fsmn_vad_zh-cn-16k-common-pytorch", config.model_aliases["fsmn-vad"])

    def test_qwen3_asr_is_registered_as_strict_primary_candidate(self):
        from zh_asr.config import get_engine_spec

        spec = get_engine_spec("qwen3-asr-1.7b")

        self.assertEqual(spec.adapter, "qwen-asr")
        self.assertEqual(spec.role, "primary")
        self.assertEqual(spec.model, "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(spec.language, "Chinese")
        self.assertEqual(spec.options["dtype"], "bfloat16")
        self.assertEqual(spec.options["max_new_tokens"], 256)

    def test_same_adapter_model_can_be_added_without_code_changes(self):
        from zh_asr.config import get_engine_spec, load_model_config

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.yaml"
            path.write_text(
                """
defaults:
  engine: custom-sensevoice
strict:
  primary_engine: custom-sensevoice
  secondary_engine: conservative-baseline
aliases:
  tiny-vad: iic/custom-vad
engines:
  custom-sensevoice:
    adapter: funasr
    role: primary
    model: iic/CustomSenseVoice
    options:
      max_new_tokens: 128
    vad_model: tiny-vad
    punc_model: null
    spk_model: null
    language: auto
    note: Custom primary model.
  conservative-baseline:
    adapter: funasr
    role: baseline
    model: iic/CustomBaseline
    language: zh
""",
                encoding="utf-8",
            )

            config = load_model_config(path)
            spec = get_engine_spec("custom-sensevoice", config=config)

        self.assertEqual(config.default_engine, "custom-sensevoice")
        self.assertEqual(config.model_aliases["tiny-vad"], "iic/custom-vad")
        self.assertEqual(spec.adapter, "funasr")
        self.assertEqual(spec.model, "iic/CustomSenseVoice")
        self.assertEqual(spec.vad_model, "tiny-vad")
        self.assertEqual(spec.options["max_new_tokens"], 128)


if __name__ == "__main__":
    unittest.main()

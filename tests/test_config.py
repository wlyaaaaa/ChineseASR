import unittest
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        self.assertIsNone(spec.spk_model)
        self.assertFalse(spec.is_whisper)

    def test_fun_asr_nano_is_an_explicit_gpu_flagship_profile(self):
        from zh_asr.config import get_engine_spec

        spec = get_engine_spec("fun-asr-nano")

        self.assertEqual(spec.adapter, "funasr")
        self.assertEqual(spec.role, "gpu_flagship")
        self.assertEqual(spec.model, "FunAudioLLM/Fun-ASR-Nano-2512")
        self.assertEqual(spec.vad_model, "fsmn-vad")
        self.assertEqual(spec.language, "Chinese")
        self.assertTrue(spec.options["requires_gpu"])
        self.assertEqual(spec.options["runtime"], "funasr-automodel")
        self.assertEqual(spec.options["model_revision"], "05201c46f1c38592b1567f857c0d56eab3d0d8ef")
        self.assertTrue(spec.options["trust_remote_code"])

    def test_paraformer_is_an_explicit_timestamped_anonymous_diarization_profile(self):
        from zh_asr.config import get_engine_spec

        spec = get_engine_spec("paraformer")

        self.assertEqual(spec.adapter, "funasr")
        self.assertEqual(spec.role, "baseline")
        self.assertEqual(
            spec.model,
            "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        )
        self.assertEqual(spec.options["model_revision"], "v2.0.4")
        self.assertEqual(spec.vad_model, "fsmn-vad")
        self.assertEqual(spec.punc_model, "ct-punc")
        self.assertEqual(spec.spk_model, "cam++")

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

    def test_speaker_verification_is_explicit_and_not_a_default_engine_setting(self):
        from zh_asr.config import load_model_config

        config = load_model_config()

        self.assertIsNotNone(config.speaker_verification)
        assert config.speaker_verification is not None
        self.assertEqual(config.speaker_verification.model_alias, "cam++")
        self.assertEqual(config.speaker_verification.model_revision, "v1.0.0")
        self.assertEqual(config.speaker_verification.model_file, "campplus_cn_common.bin")
        self.assertEqual(config.speaker_verification.threshold, 0.31)
        self.assertIsNone(config.engines["sensevoice"].spk_model)

    def test_core_requirements_pin_funasr_142(self):
        requirements = (PROJECT_ROOT / "requirements-core.txt").read_text(encoding="utf-8").splitlines()

        self.assertIn("funasr==1.4.2", requirements)
        self.assertTrue(any(line.startswith("more-itertools>=") for line in requirements))
        self.assertTrue(any(line.startswith("rapidfuzz>=") for line in requirements))
        self.assertTrue(any(line.startswith("websockets>=") for line in requirements))

    def test_llm_arbitration_defaults_to_disabled_ollama_with_keep_alive_zero(self):
        import yaml

        data = yaml.safe_load((PROJECT_ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
        arbitration = data["llm_arbitration"]

        self.assertFalse(arbitration["enabled"])
        self.assertEqual(arbitration["provider"], "ollama")
        self.assertEqual(arbitration["model"], "qwen-main-v1:latest")
        self.assertEqual(arbitration["keep_alive"], 0)

    def test_qwen3_asr_is_registered_as_strict_primary_candidate(self):
        from zh_asr.config import get_engine_spec

        spec = get_engine_spec("qwen3-asr-1.7b")

        self.assertEqual(spec.adapter, "qwen-asr")
        self.assertEqual(spec.role, "primary")
        self.assertEqual(spec.model, "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(spec.language, "Chinese")
        self.assertEqual(spec.options["dtype"], "bfloat16")
        self.assertEqual(spec.options["max_new_tokens"], 256)

    def test_firered_is_registered_as_optional_forensic_primary(self):
        from zh_asr.config import get_engine_spec, load_model_config

        config = load_model_config()
        spec = get_engine_spec("fireredasr2-llm", config=config)

        self.assertEqual(config.strict_primary_engine, "qwen3-asr-1.7b")
        self.assertEqual(spec.adapter, "firered-worker")
        self.assertEqual(spec.role, "lexical_primary")
        self.assertEqual(spec.model, "FireRedTeam/FireRedASR2-LLM")
        self.assertEqual(spec.options["max_audio_sec"], 40)
        self.assertEqual(spec.options["recommended_chunk_sec"], 35)
        self.assertEqual(spec.options["batch_size"], 1)

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
        self.assertIsNone(config.speaker_verification)


if __name__ == "__main__":
    unittest.main()

import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class PipelineTests(unittest.TestCase):
    def test_funasr_kwargs_use_local_cache_paths_when_available(self):
        from zh_asr.config import get_engine_spec, load_model_config
        from zh_asr.pipeline import _funasr_kwargs

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            for relative in [
                "iic/SenseVoiceSmall",
                "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
                "iic/speech_campplus_sv_zh-cn_16k-common",
            ]:
                (cache / relative).mkdir(parents=True)

            config = load_model_config()
            kwargs = _funasr_kwargs(get_engine_spec("sensevoice"), "cuda:0", cache, config.model_aliases)

            self.assertEqual(kwargs["model"], str(cache / "iic/SenseVoiceSmall"))
            self.assertEqual(kwargs["vad_model"], str(cache / "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"))
            self.assertEqual(kwargs["punc_model"], str(cache / "iic/punc_ct-transformer_cn-en-common-vocab471067-large"))
            self.assertEqual(kwargs["spk_model"], str(cache / "iic/speech_campplus_sv_zh-cn_16k-common"))

    def test_funasr_kwargs_resolve_aliases_from_model_registry(self):
        from zh_asr.config import EngineSpec
        from zh_asr.pipeline import _funasr_kwargs

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "iic/custom-vad").mkdir(parents=True)
            spec = EngineSpec(
                name="custom",
                adapter="funasr",
                role="primary",
                model="iic/CustomModel",
                vad_model="tiny-vad",
            )

            kwargs = _funasr_kwargs(spec, "cuda:0", cache, {"tiny-vad": "iic/custom-vad"})

        self.assertEqual(kwargs["vad_model"], str(cache / "iic/custom-vad"))

    def test_build_model_passes_resolved_cache_paths_to_automodel(self):
        from zh_asr.pipeline import build_model

        class DummyAutoModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_funasr = types.ModuleType("funasr")
        fake_funasr.AutoModel = DummyAutoModel

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "iic/SenseVoiceSmall").mkdir(parents=True)
            with patch.dict(sys.modules, {"funasr": fake_funasr}):
                with patch("zh_asr.pipeline.ensure_funasr_available"):
                    model = build_model("sensevoice", cache_dir=cache)

            self.assertEqual(model.kwargs["model"], str(cache / "iic/SenseVoiceSmall"))

    def test_qwen_adapter_wraps_transcribe_results_for_common_writer(self):
        from zh_asr.adapters.qwen_asr import QwenASRAdapter
        from zh_asr.config import EngineSpec

        class DummyResult:
            language = "zh"
            text = "开放时间早上九点至下午五点。"

        class DummyQwenModel:
            def transcribe(self, audio, language):
                self.audio = audio
                self.language = language
                return [DummyResult()]

        spec = EngineSpec(
            name="qwen3-asr-1.7b",
            adapter="qwen-asr",
            role="primary",
            model="Qwen/Qwen3-ASR-1.7B",
            language="Chinese",
            options={"dtype": "bfloat16"},
        )
        wrapper = QwenASRAdapter().wrap_model(DummyQwenModel(), spec)

        result = wrapper.generate(input="sample.wav")

        self.assertEqual(result, [{"text": "开放时间早上九点至下午五点。", "language": "zh"}])

    def test_qwen_adapter_uses_local_modelscope_cache_path_when_available(self):
        from zh_asr.adapters.qwen_asr import qwen_from_pretrained_kwargs
        from zh_asr.config import EngineSpec

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "Qwen/Qwen3-ASR-1.7B").mkdir(parents=True)
            spec = EngineSpec(
                name="qwen3-asr-1.7b",
                adapter="qwen-asr",
                role="primary",
                model="Qwen/Qwen3-ASR-1.7B",
                language="Chinese",
                options={"dtype": "bfloat16", "max_new_tokens": 256, "max_inference_batch_size": 8},
            )

            kwargs = qwen_from_pretrained_kwargs(spec, "cuda:0", cache, {})

        self.assertEqual(kwargs["model"], str(cache / "Qwen/Qwen3-ASR-1.7B"))
        self.assertEqual(kwargs["device_map"], "cuda:0")
        self.assertEqual(kwargs["max_new_tokens"], 256)
        self.assertEqual(kwargs["max_inference_batch_size"], 8)


if __name__ == "__main__":
    unittest.main()

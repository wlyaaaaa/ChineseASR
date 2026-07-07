import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class PipelineTests(unittest.TestCase):
    def test_funasr_kwargs_use_local_cache_paths_when_available(self):
        from zh_asr.config import get_engine_spec
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

            kwargs = _funasr_kwargs(get_engine_spec("sensevoice"), "cuda:0", cache)

            self.assertEqual(kwargs["model"], str(cache / "iic/SenseVoiceSmall"))
            self.assertEqual(kwargs["vad_model"], str(cache / "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"))
            self.assertEqual(kwargs["punc_model"], str(cache / "iic/punc_ct-transformer_cn-en-common-vocab471067-large"))
            self.assertEqual(kwargs["spk_model"], str(cache / "iic/speech_campplus_sv_zh-cn_16k-common"))

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


if __name__ == "__main__":
    unittest.main()

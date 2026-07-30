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
            text = "開放時間早上九點至下午五點。"

        class DummyQwenModel:
            def transcribe(self, audio, context, language):
                self.audio = audio
                self.context = context
                self.language = language
                return [DummyResult()]

        spec = EngineSpec(
            name="qwen3-asr-1.7b",
            adapter="qwen-asr",
            role="primary",
            model="Qwen/Qwen3-ASR-1.7B",
            language="Chinese",
            options={"dtype": "bfloat16", "context": "请只输出简体中文转写文本。"},
        )
        model = DummyQwenModel()
        wrapper = QwenASRAdapter().wrap_model(model, spec)

        result = wrapper.generate(input="sample.wav")

        self.assertEqual(
            result,
            [
                {
                    "text": "开放时间早上九点至下午五点。",
                    "language": "zh",
                    "original_text": "開放時間早上九點至下午五點。",
                }
            ],
        )
        self.assertEqual(model.context, "请只输出简体中文转写文本。")

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

    def test_qwen_adapter_requires_prefetched_local_cache(self):
        from zh_asr.adapters.qwen_asr import qwen_from_pretrained_kwargs
        from zh_asr.config import EngineSpec

        with tempfile.TemporaryDirectory() as tmp:
            spec = EngineSpec(
                name="qwen3-asr-1.7b",
                adapter="qwen-asr",
                role="primary",
                model="Qwen/Qwen3-ASR-1.7B",
                language="Chinese",
            )

            with self.assertRaisesRegex(FileNotFoundError, "Qwen ASR model cache not found"):
                qwen_from_pretrained_kwargs(spec, "cuda:0", Path(tmp), {})

    def test_strict_transcribe_writes_audit_when_secondary_engine_fails(self):
        import json
        from zh_asr.pipeline import strict_transcribe_audio

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "sample.wav"
            audio.write_bytes(b"fake wav")

            def fake_generate(audio_path, engine, device, cache_dir, config):
                if engine == "sensevoice":
                    raise TypeError("'>' not supported between instances of 'float' and 'NoneType'")
                return [{"text": "开放时间早上九点至下午五点。"}]

            with patch("zh_asr.pipeline._generate_once", side_effect=fake_generate):
                paths = strict_transcribe_audio(
                    audio,
                    primary_engine="qwen3-asr-1.7b",
                    secondary_engine="sensevoice",
                    out_dir=root / "outputs",
                )

            final_text = paths["final"].read_text(encoding="utf-8")
            audit_text = paths["audit"].read_text(encoding="utf-8")
            audit_json = json.loads(paths["audit_json"].read_text(encoding="utf-8"))
            secondary_raw = json.loads(paths["secondary_json"].read_text(encoding="utf-8"))

            self.assertIn("[疑似]开放时间早上九点至下午五点。", final_text)
            self.assertIn("engine_failure", audit_text)
            self.assertIn("sensevoice", audit_text)
            self.assertEqual("engine_failure", audit_json["status"])
            self.assertEqual("provisional", audit_json["evidence_status"])
            self.assertEqual("provisional", paths["evidence_status"])
            self.assertEqual(
                ["sensevoice"],
                [
                    item["engine"]
                    for item in audit_json["engine_evidence"]
                    if item["execution_status"] == "failed"
                ],
            )
            self.assertIn("engine_failure", audit_json["flags"])
            self.assertEqual("sensevoice", secondary_raw["engine"])
            self.assertEqual("TypeError", secondary_raw["error"]["type"])

    def test_firered_input_is_normalized_and_rejects_audio_over_model_limit(self):
        from zh_asr.audio_frontend import PreparedAudio
        from zh_asr.config import load_model_config
        from zh_asr.pipeline import _prepare_engine_input

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "call.mp3"
            source.write_bytes(b"source")
            derivative = root / "derived.wav"
            derivative.write_bytes(b"derived")
            prepared = PreparedAudio(
                source_path=source,
                path=derivative,
                converted=True,
                source_sha256="source-hash",
                derivative_sha256="derived-hash",
                sample_rate=16000,
                channels=1,
                sample_width_bytes=2,
                duration_sec=41.0,
                ffmpeg_version="ffmpeg test",
            )

            with patch("zh_asr.pipeline.prepare_pcm16_mono", return_value=prepared):
                with self.assertRaisesRegex(ValueError, "40 seconds"):
                    _prepare_engine_input(
                        source,
                        "fireredasr2-llm",
                        root / "_derived",
                        load_model_config(),
                    )

    def test_strict_transcribe_records_prepared_audio_provenance_and_roles(self):
        from zh_asr.audio_frontend import PreparedAudio
        from zh_asr.pipeline import strict_transcribe_audio

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "call.mp3"
            source.write_bytes(b"source")
            derivative = root / "call.wav"
            derivative.write_bytes(b"derived")
            prepared = PreparedAudio(
                source_path=source,
                path=derivative,
                converted=True,
                source_sha256="source-hash",
                derivative_sha256="derived-hash",
                sample_rate=16000,
                channels=1,
                sample_width_bytes=2,
                duration_sec=10.0,
            )

            with (
                patch("zh_asr.pipeline.prepare_pcm16_mono", return_value=prepared),
                patch(
                    "zh_asr.pipeline._generate_once",
                    side_effect=lambda audio, engine, *_: [{"text": f"{engine}:{audio.name}"}],
                ),
                patch("zh_asr.pipeline.write_strict_bundle") as writer,
            ):
                writer.return_value = {}
                strict_transcribe_audio(
                    source,
                    primary_engine="fireredasr2-llm",
                    secondary_engine="sensevoice",
                    out_dir=root / "outputs",
                )

            kwargs = writer.call_args.kwargs
            self.assertEqual(kwargs["primary_role"], "lexical_primary")
            self.assertEqual(kwargs["secondary_role"], "lexical_verifier")
            self.assertEqual(kwargs["primary_provenance"]["audio"]["derivative_sha256"], "derived-hash")
            self.assertEqual(kwargs["primary_provenance"]["registry_role"], "lexical_primary")

    def test_strict_transcribe_many_loads_each_engine_once_and_uses_generate_many(self):
        from zh_asr.pipeline import strict_transcribe_many

        class PrimaryModel:
            def __init__(self):
                self.calls = []

            def generate_many(self, inputs):
                self.calls.append(list(inputs))
                return [
                    {"text": f"primary-{Path(value).stem}"}
                    for value in inputs
                ]

        class SecondaryModel:
            def __init__(self):
                self.calls = []

            def generate(self, input, **_kwargs):
                self.calls.append(input)
                return [{"text": f"secondary-{Path(input).stem}"}]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audios = []
            out_dirs = []
            for name in ("one.wav", "two.wav"):
                audio = root / name
                audio.write_bytes(b"fake wav")
                audios.append(audio)
                out_dirs.append(root / f"out-{audio.stem}")
            primary = PrimaryModel()
            secondary = SecondaryModel()

            def fake_build(engine, **_kwargs):
                return primary if engine == "fireredasr2-llm" else secondary

            with (
                patch("zh_asr.pipeline.build_model", side_effect=fake_build) as build,
                patch(
                    "zh_asr.pipeline._prepare_engine_input",
                    side_effect=lambda audio, _engine, _derived, _config: (
                        audio,
                        {"audio": {"path": str(audio)}},
                    ),
                ),
                patch("zh_asr.pipeline.write_strict_bundle") as writer,
            ):
                writer.side_effect = lambda **kwargs: {
                    "final": kwargs["out_dir"] / "final.md"
                }
                results = strict_transcribe_many(
                    audios,
                    out_dirs=out_dirs,
                    primary_engine="fireredasr2-llm",
                    secondary_engine="sensevoice",
                )

        self.assertEqual(build.call_count, 2)
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(len(primary.calls[0]), 2)
        self.assertEqual(len(secondary.calls), 2)
        self.assertEqual(writer.call_count, 2)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()

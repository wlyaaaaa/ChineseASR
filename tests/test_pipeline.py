from contextlib import contextmanager
import json
import os
import tempfile
import sys
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


class PipelineTests(unittest.TestCase):
    def _write_wav(
        self,
        path: Path,
        *,
        frames: int = 1600,
        sample_rate: int = 16000,
    ) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(b"\0" * frames * 2)

    def test_funasr_vad_zero_segments_is_full_coverage_negative_evidence(self):
        from zh_asr.adapters.funasr import detect_speech_segments

        class DummyModel:
            vad_model = object()
            vad_kwargs = {"max_single_segment_time": 30000}

            def inference(self, audio, *, model, kwargs):
                self.call = (audio, model, kwargs)
                return [{"value": []}]

        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "silence.wav"
            self._write_wav(audio)
            detection = detect_speech_segments(DummyModel(), audio)

        self.assertEqual(detection["status"], "no_speech_detected")
        self.assertTrue(detection["coverage_complete"])
        self.assertEqual(detection["segments"], [])

    def test_pipeline_attaches_vad_processor_model_and_config_telemetry(self):
        from zh_asr.config import load_model_config
        from zh_asr.pipeline import _attach_speech_detection_if_empty

        class DummyModel:
            vad_model = object()
            vad_kwargs = {"max_single_segment_time": 30000}

            def inference(self, audio, *, model, kwargs):
                return [{"value": [[0, 50]]}]

        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "speech.wav"
            self._write_wav(audio)
            result = _attach_speech_detection_if_empty(
                {"text": ""},
                DummyModel(),
                audio,
                "sensevoice",
                load_model_config(),
            )

        detection = result["speech_detection"]
        self.assertEqual(detection["status"], "speech_detected")
        self.assertEqual(detection["processor"], "funasr-vad")
        self.assertEqual(detection["model"], "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
        self.assertTrue(detection["config_sha256"])

    def test_pipeline_excludes_runtime_only_objects_from_vad_config(self):
        from zh_asr.config import load_model_config
        from zh_asr.pipeline import _attach_speech_detection_if_empty

        class WavFrontendOnline:
            pass

        class DummyModel:
            vad_model = object()
            vad_kwargs = {
                "max_single_segment_time": 30000,
                "frontend": WavFrontendOnline(),
            }

            def inference(self, audio, *, model, kwargs):
                return [{"value": []}]

        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "silence.wav"
            self._write_wav(audio)
            result = _attach_speech_detection_if_empty(
                {"text": ""},
                DummyModel(),
                audio,
                "sensevoice",
                load_model_config(),
            )

        json.dumps(result)
        vad_config = result["speech_detection"]["config"]["vad_kwargs"]
        self.assertEqual(vad_config, {"max_single_segment_time": 30000})

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
            self.assertNotIn("spk_model", kwargs)

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

    def test_funasr_kwargs_pass_only_documented_custom_model_options(self):
        from zh_asr.config import get_engine_spec, load_model_config
        from zh_asr.adapters.funasr import funasr_kwargs

        with tempfile.TemporaryDirectory() as tmp:
            kwargs = funasr_kwargs(
                get_engine_spec("fun-asr-nano"),
                "cuda:0",
                Path(tmp),
                load_model_config().model_aliases,
            )

        self.assertEqual(kwargs["hub"], "ms")
        self.assertEqual(kwargs["model_revision"], "05201c46f1c38592b1567f857c0d56eab3d0d8ef")
        self.assertTrue(kwargs["trust_remote_code"])
        self.assertEqual(kwargs["remote_code"], "./model.py")
        self.assertNotIn("requires_gpu", kwargs)
        self.assertNotIn("runtime", kwargs)

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
        from zh_asr.qwen_identity import (
            QWEN_MODEL_REVISION,
            RequiredModelFile,
            write_model_receipt,
        )
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            model_dir = cache / "Qwen/Qwen3-ASR-1.7B"
            model_dir.mkdir(parents=True)
            payload = b"small qwen test artifact"
            (model_dir / "model.bin").write_bytes(payload)
            required_files = (
                RequiredModelFile(
                    path="model.bin",
                    bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                ),
            )
            write_model_receipt(
                model_dir,
                repository="Qwen/Qwen3-ASR-1.7B",
                revision=QWEN_MODEL_REVISION,
                required_files=required_files,
            )
            spec = EngineSpec(
                name="qwen3-asr-1.7b",
                adapter="qwen-asr",
                role="primary",
                model="Qwen/Qwen3-ASR-1.7B",
                language="Chinese",
                options={
                    "model_revision": QWEN_MODEL_REVISION,
                    "runtime_distribution": "qwen-asr",
                    "runtime_version": "0.0.6",
                    "dtype": "bfloat16",
                    "max_new_tokens": 256,
                    "max_inference_batch_size": 8,
                },
            )

            with patch(
                "zh_asr.qwen_identity.QWEN_MODEL_FILES", required_files
            ):
                kwargs = qwen_from_pretrained_kwargs(
                    spec, "cuda:0", cache, {}
                )

        self.assertEqual(kwargs["model"], str(cache / "Qwen/Qwen3-ASR-1.7B"))
        self.assertEqual(kwargs["device_map"], "cuda:0")
        self.assertEqual(kwargs["max_new_tokens"], 256)
        self.assertEqual(kwargs["max_inference_batch_size"], 8)

    def test_qwen_adapter_requires_prefetched_local_cache(self):
        from zh_asr.adapters.qwen_asr import qwen_from_pretrained_kwargs
        from zh_asr.config import EngineSpec
        from zh_asr.qwen_identity import QWEN_MODEL_REVISION

        with tempfile.TemporaryDirectory() as tmp:
            spec = EngineSpec(
                name="qwen3-asr-1.7b",
                adapter="qwen-asr",
                role="primary",
                model="Qwen/Qwen3-ASR-1.7B",
                language="Chinese",
                options={
                    "model_revision": QWEN_MODEL_REVISION,
                    "runtime_distribution": "qwen-asr",
                    "runtime_version": "0.0.6",
                },
            )

            with self.assertRaisesRegex(FileNotFoundError, "Qwen ASR model cache not found"):
                qwen_from_pretrained_kwargs(spec, "cuda:0", Path(tmp), {})

    def test_qwen_adapter_rejects_missing_receipt_before_model_load(self):
        from zh_asr.adapters.qwen_asr import QwenASRAdapter
        from zh_asr.config import get_engine_spec

        class DummyQwenModel:
            called = False

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                cls.called = True
                return cls()

        fake_qwen = types.ModuleType("qwen_asr")
        fake_qwen.Qwen3ASRModel = DummyQwenModel

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "Qwen/Qwen3-ASR-1.7B").mkdir(parents=True)
            with (
                patch.dict(sys.modules, {"qwen_asr": fake_qwen}),
                patch(
                    "zh_asr.adapters.qwen_asr.ensure_qwen_asr_available"
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "receipt is missing"
                ):
                    QwenASRAdapter().build_model(
                        get_engine_spec("qwen3-asr-1.7b"),
                        "cuda:0",
                        cache,
                        {},
                    )

        self.assertFalse(DummyQwenModel.called)

    def test_strict_transcribe_writes_audit_when_secondary_engine_fails(self):
        import json
        from zh_asr.pipeline import strict_transcribe_audio

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "sample.wav"
            self._write_wav(audio)

            def fake_generate(audio_path, engine, device, cache_dir, config):
                if engine == "sensevoice":
                    raise TypeError("'>' not supported between instances of 'float' and 'NoneType'")
                return [{"text": "开放时间早上九点至下午五点。"}], {}

            with patch(
                "zh_asr.pipeline._generate_once_with_identity",
                side_effect=fake_generate,
            ):
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
            evidence_by_engine = {
                item["engine"]: item["provenance"]["audio"]
                for item in audit_json["engine_evidence"]
            }
            self.assertEqual(
                evidence_by_engine["qwen3-asr-1.7b"],
                evidence_by_engine["sensevoice"],
            )
            self.assertTrue(
                {
                    "source_sha256",
                    "derivative_sha256",
                    "duration_sec",
                    "sample_rate",
                    "channels",
                    "sample_width",
                    "format",
                    "converted",
                }.issubset(evidence_by_engine["qwen3-asr-1.7b"])
            )

    def test_default_strict_materializes_one_owner_wav_for_both_engines(self):
        from zh_asr.audio_frontend import prepare_pcm16_mono
        from zh_asr.pipeline import strict_transcribe_audio

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "sample.wav"
            out_dir = root / "outputs"
            self._write_wav(audio, frames=3200)

            with (
                patch(
                    "zh_asr.pipeline.prepare_pcm16_mono",
                    wraps=prepare_pcm16_mono,
                ) as prepare,
                patch(
                    "zh_asr.pipeline._generate_once_with_identity",
                    side_effect=lambda prepared, engine, *_: (
                        [{"text": engine}],
                        {},
                    ),
                ) as generate,
                patch("zh_asr.pipeline.write_strict_bundle") as writer,
            ):
                writer.return_value = {}
                strict_transcribe_audio(
                    audio,
                    primary_engine="qwen3-asr-1.7b",
                    secondary_engine="sensevoice",
                    out_dir=out_dir,
                )

            self.assertEqual(prepare.call_count, 1)
            self.assertTrue(prepare.call_args.kwargs["materialize_owner"])
            prepared_paths = [call.args[0] for call in generate.call_args_list]
            self.assertEqual(prepared_paths[0], prepared_paths[1])
            self.assertNotEqual(audio.resolve(), prepared_paths[0])
            self.assertEqual((out_dir / "_derived").resolve(), prepared_paths[0].parent)

            kwargs = writer.call_args.kwargs
            primary_audio = kwargs["primary_provenance"]["audio"]
            secondary_audio = kwargs["secondary_provenance"]["audio"]
            self.assertEqual(primary_audio, secondary_audio)
            self.assertEqual(primary_audio["source_sha256"], primary_audio["derivative_sha256"])
            self.assertEqual(primary_audio["sample_rate"], 16000)
            self.assertEqual(primary_audio["channels"], 1)
            self.assertEqual(primary_audio["sample_width"], 2)
            self.assertEqual(primary_audio["format"], "wav")
            self.assertFalse(primary_audio["converted"])

    def test_default_strict_converts_one_synthetic_wav_for_both_engines(self):
        from zh_asr.pipeline import strict_transcribe_audio

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "sample-8k.wav"
            out_dir = root / "outputs"
            self._write_wav(audio, frames=800, sample_rate=8000)

            def fake_ffmpeg(command, **_kwargs):
                self._write_wav(Path(command[-1]), frames=1600, sample_rate=16000)
                return type("Completed", (), {"stdout": "", "stderr": ""})()

            with (
                patch(
                    "zh_asr.audio_frontend.subprocess.run",
                    side_effect=fake_ffmpeg,
                ) as ffmpeg,
                patch(
                    "zh_asr.audio_frontend._ffmpeg_version",
                    return_value="ffmpeg test",
                ),
                patch(
                    "zh_asr.pipeline._generate_once_with_identity",
                    side_effect=lambda prepared, engine, *_: (
                        [{"text": engine}],
                        {},
                    ),
                ) as generate,
                patch("zh_asr.pipeline.write_strict_bundle") as writer,
            ):
                writer.return_value = {}
                strict_transcribe_audio(
                    audio,
                    primary_engine="qwen3-asr-1.7b",
                    secondary_engine="sensevoice",
                    out_dir=out_dir,
                )

            self.assertEqual(ffmpeg.call_count, 1)
            prepared_paths = [call.args[0] for call in generate.call_args_list]
            self.assertEqual(prepared_paths[0], prepared_paths[1])
            primary_audio = writer.call_args.kwargs["primary_provenance"]["audio"]
            secondary_audio = writer.call_args.kwargs["secondary_provenance"]["audio"]
            self.assertEqual(primary_audio, secondary_audio)
            self.assertNotEqual(
                primary_audio["source_sha256"],
                primary_audio["derivative_sha256"],
            )
            self.assertTrue(primary_audio["converted"])
            self.assertEqual(primary_audio["sample_rate"], 16000)
            self.assertEqual(primary_audio["channels"], 1)
            self.assertEqual(primary_audio["sample_width"], 2)
            self.assertEqual(primary_audio["format"], "wav")

    def test_default_strict_fails_closed_if_owner_audio_drifts_before_lock(self):
        from zh_asr.audio_frontend import (
            PreparedAudioIntegrityError,
            _locked_prepared_audio_owner,
        )
        from zh_asr.pipeline import strict_transcribe_audio

        def tamper(path: Path) -> None:
            path.write_bytes(b"tampered")

        def remove(path: Path) -> None:
            path.unlink()

        def duplicate(path: Path) -> None:
            self._write_wav(path.parent / "unexpected.wav")

        for label, mutate in (
            ("tamper", tamper),
            ("missing", remove),
            ("multiple", duplicate),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                audio = root / "sample.wav"
                self._write_wav(audio)

                @contextmanager
                def drift_before_lock(prepared, derived_dir):
                    mutate(prepared.path)
                    with _locked_prepared_audio_owner(prepared, derived_dir):
                        yield

                with (
                    patch(
                        "zh_asr.pipeline._locked_prepared_audio_owner",
                        side_effect=drift_before_lock,
                    ),
                    patch(
                        "zh_asr.pipeline._generate_once_with_identity",
                    ) as generate,
                    patch("zh_asr.pipeline.write_strict_bundle") as writer,
                ):
                    with self.assertRaises(PreparedAudioIntegrityError):
                        strict_transcribe_audio(
                            audio,
                            primary_engine="qwen3-asr-1.7b",
                            secondary_engine="sensevoice",
                            out_dir=root / "outputs",
                        )

                generate.assert_not_called()
                writer.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows share-mode lock contract")
    def test_default_strict_holds_one_read_lock_across_both_engine_reads(self):
        from zh_asr.audio_frontend import _open_windows_owner_read_lock
        from zh_asr.pipeline import strict_transcribe_audio

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "sample.wav"
            replacement = root / "replacement.bin"
            backup = root / "owner-backup.bin"
            self._write_wav(audio, frames=3200)
            replacement.write_bytes(b"replacement")
            observed: list[bytes] = []
            mutation_succeeded: list[str] = []

            def must_be_blocked(label, operation):
                try:
                    operation()
                except OSError:
                    return
                mutation_succeeded.append(label)

            def fake_generate(prepared, engine, *_):
                observed.append(prepared.read_bytes())
                if len(observed) == 1:
                    must_be_blocked(
                        "write",
                        lambda: prepared.write_bytes(b"tampered"),
                    )
                    must_be_blocked(
                        "swap",
                        lambda: os.replace(prepared, backup),
                    )
                    must_be_blocked(
                        "replace",
                        lambda: os.replace(replacement, prepared),
                    )
                    must_be_blocked("delete", prepared.unlink)
                return [{"text": engine}], {}

            with (
                patch(
                    "zh_asr.pipeline._generate_once_with_identity",
                    side_effect=fake_generate,
                ),
                patch(
                    "zh_asr.audio_frontend._open_windows_owner_read_lock",
                    wraps=_open_windows_owner_read_lock,
                ) as open_lock,
                patch("zh_asr.pipeline.write_strict_bundle") as writer,
            ):
                writer.return_value = {}
                strict_transcribe_audio(
                    audio,
                    primary_engine="qwen3-asr-1.7b",
                    secondary_engine="sensevoice",
                    out_dir=root / "outputs",
                )

            self.assertEqual(mutation_succeeded, [])
            self.assertEqual(open_lock.call_count, 1)
            self.assertEqual(len(observed), 2)
            self.assertEqual(observed[0], observed[1])

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
                    "zh_asr.pipeline._generate_once_with_identity",
                    side_effect=lambda audio, engine, *_: (
                        [{"text": f"{engine}:{audio.name}"}],
                        {},
                    ),
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
                self.runtime_identity = {
                    "model_revision": "pinned-revision",
                    "model_receipt_status": "verified",
                }

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
        self.assertEqual(
            "verified",
            writer.call_args_list[0].kwargs["primary_provenance"][
                "runtime_identity"
            ]["model_receipt_status"],
        )
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()

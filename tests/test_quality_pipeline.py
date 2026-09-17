import json
import tempfile
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from zh_asr.config import load_model_config, resolve_profile
from zh_asr.file_entry import chunk_budget, needs_long_route, transcribe_file
from zh_asr.alignment import validate_items, timestamp_supported_overlap
from zh_asr.inference_quality import isolated_batch, BatchItemFailure
from zh_asr.model_lifecycle import file_hash
from zh_asr.quality_review import enhance_chunks, enhance_single
from zh_asr.service import JobRequest, TranscriptionService


def wav(path, seconds=1, channels=1):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels); f.setsampwidth(2); f.setframerate(16000)
        f.writeframes(b"\0" * (int(seconds * 16000) * 2 * channels))
    return path


class QualityPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_model_config()
        self.audio = wav(self.root / "audio.wav")

    def test_profile_and_explicit_engine_precedence(self):
        self.assertEqual(resolve_profile(self.config, "high_quality"), ("fireredasr2-llm", "qwen3-asr-1.7b"))
        self.assertEqual(resolve_profile(self.config, "high_quality", secondary="sensevoice"), ("fireredasr2-llm", "sensevoice"))
        with self.assertRaises(ValueError):
            resolve_profile(self.config, "missing")
        with self.assertRaises(ValueError):
            resolve_profile(self.config, primary="sensevoice", secondary="sensevoice")

    def test_profiles_supervise_correct_wsl_for_both_flag_forms(self):
        from zh_asr.__main__ import _cli_wsl_distributions
        for args in (["--profile", "high_quality"], ["--profile=high_quality"]):
            self.assertEqual(_cli_wsl_distributions(["strict", "a.wav", *args]), ("Ubuntu",))

    def test_file_budget_includes_model_and_quality_limits(self):
        self.assertEqual(chunk_budget(self.config, *resolve_profile(self.config, "high_quality")), 35)
        self.assertEqual(chunk_budget(self.config, *resolve_profile(self.config, "baseline")), 60)
        self.assertFalse(needs_long_route(self.audio, self.config, *resolve_profile(self.config)))

    def test_short_entry_preserves_core_result_and_calls_one_primary_pair(self):
        core = Mock(return_value={"final": self.root / "final.md", "objective_outcome": "speech_transcribed"})
        result = transcribe_file(self.audio, out_dir=self.root, device="cpu", config=self.config, strict_fn=core)
        self.assertEqual(result["resolved_mode"], "strict")
        self.assertEqual(core.call_count, 1)
        self.assertEqual(core.call_args.args[0], self.audio)

    def test_long_entry_routes_before_loading_an_oversized_model(self):
        audio = wav(self.root / "long.wav", 41)
        summary = SimpleNamespace(transcript_path=self.root/"transcript.md", audit_path=self.root/"audit.md",
            metrics_path=self.root/"metrics.json", manifest_path=self.root/"manifest.json",
            objective_outcome="speech_transcribed", evidence_status="verified", failed=0)
        core = Mock(side_effect=AssertionError("oversized audio reached short core"))
        with patch("zh_asr.long_audio.run_long_transcription", return_value=summary) as run:
            result = transcribe_file(audio, out_dir=self.root, profile="high_quality", config=self.config, device="cpu", strict_fn=core)
        self.assertEqual(result["resolved_mode"], "long-strict")
        self.assertEqual(run.call_args.kwargs["chunk_sec"], 35)
        self.assertIs(run.call_args.kwargs["config"], self.config)
        core.assert_not_called()

    def test_http_profile_route_and_channel_survive_request_identity(self):
        audio = wav(self.root / "stereo.wav", 41, 2)
        request = JobRequest.from_payload({"audio": str(audio), "profile": "high_quality", "channel_index": 1}, self.root)
        self.assertEqual(request.mode, "long-strict")
        self.assertEqual(request.to_dict()["profile"], "high_quality")
        self.assertEqual(request.to_dict()["channel_index"], 1)
        service = TranscriptionService(root=self.root, gpu_process_detector=lambda: [], autostart=False)
        command = service._build_command(request, self.root / "out")
        self.assertIn("--profile", command); self.assertIn("--channel-index", command)
        self.assertNotEqual(request.fingerprint(), replace(request, channel_index=0).fingerprint())

    def test_invalid_channel_is_rejected_before_scheduling(self):
        for value in (-1, True, "1", 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                JobRequest.from_payload({"audio": str(self.audio), "channel_index": value}, self.root)

    def test_channel_core_writes_original_source_not_derived_source(self):
        from zh_asr.pipeline import strict_transcribe_audio
        selected = wav(self.root / "selected.wav")
        generated = lambda *a, **k: ([{"text": "同意"}], None, 0.1, {})
        with patch("zh_asr.audio_quality.extract_channel", return_value=(selected, {"selected_channel": 1})), \
             patch("zh_asr.pipeline._uses_shared_default_strict_audio", return_value=False), \
             patch("zh_asr.pipeline._generate_for_strict", side_effect=generated) as gen, \
             patch("zh_asr.pipeline.write_strict_bundle", return_value={}) as writer:
            strict_transcribe_audio(self.audio, out_dir=self.root, config=self.config, device="cpu", channel_index=1)
        self.assertEqual(writer.call_args.kwargs["audio_path"], self.audio)
        self.assertTrue(all(call.args[0] == selected for call in gen.call_args_list))
        self.assertEqual(writer.call_args.kwargs["primary_provenance"]["channel_selection"]["selected_channel"], 1)

    def test_timestamp_validation_rejects_nonfinite_boolean_and_out_of_bounds(self):
        for start, end in ((-1, 100), (float("nan"), 100), (False, 100), (300, 100), (0, 1200)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                validate_items([{"start_ms": start, "end_ms": end}], 1000)
        validate_items([{"start_ms": 0, "end_ms": 300}, {"start_ms": 400, "end_ms": 500}], 1000)

    def test_temporal_overlap_removes_only_the_same_sound(self):
        left = {"status": "succeeded", "exact_text_coverage": True, "text": "前面同意", "items": [
            {"text": "前面", "start_ms": 0, "end_ms": 800}, {"text": "同意", "start_ms": 1100, "end_ms": 1500}]}
        right = {"status": "succeeded", "exact_text_coverage": True, "text": "同意然后", "items": [
            {"text": "同意", "start_ms": 100, "end_ms": 500}, {"text": "然后", "start_ms": 1000, "end_ms": 1500}]}
        args = ("前面同意", "同意然后", left, right, 0, 1000, 2000)
        self.assertEqual(timestamp_supported_overlap(*args), 2)
        right["items"][0].update(start_ms=600, end_ms=900)
        self.assertEqual(timestamp_supported_overlap(*args), 0)
        right["exact_text_coverage"] = False
        self.assertEqual(timestamp_supported_overlap(*args), 0)

    def test_failed_batch_isolates_bad_input_without_losing_neighbors(self):
        batch = Mock(side_effect=RuntimeError("batch failure"))
        def single(value):
            if value == "bad": raise ValueError("unreadable")
            return value
        values = isolated_batch(["a", "bad", "c"], batch, single)
        self.assertEqual(values[0], "a"); self.assertIsInstance(values[1], BatchItemFailure)
        self.assertEqual(values[2], "c")

    def state(self, same=False):
        audit = self.root / "audit.json"
        data = {"primary_engine": "fireredasr2-llm", "secondary_engine": "qwen3-asr-1.7b",
            "primary_text": "金额1.5万元", "secondary_text": "金额1.5万元" if same else "金额15万元",
            "final_text": "金额1.5万元", "needs_review": not same, "flags": []}
        audit.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return SimpleNamespace(status="succeeded", outputs={"audit_json": str(audit)},
            spec=SimpleNamespace(audio_path=self.audio, chunk_id="c000", start_ms=0, end_ms=1000))

    def test_supplemental_call_is_wired_and_cached_without_rewriting_audit(self):
        state = self.state(); before = file_hash(Path(state.outputs["audit_json"]))
        config = replace(self.config, quality={"align": False, "review_engine": "sensevoice"})
        generate = Mock(return_value={"results": [[{"text": "金额一点五万元"}]], "errors": [None]})
        with patch("zh_asr.long_audio._runtime_artifact_identity", return_value={}), patch("zh_asr.long_audio._runtime_code_identity", return_value={"sha256": "code"}):
            first = enhance_chunks([state], config=config, device="cpu", cache_dir=self.root, output_dir=self.root, generate_fn=generate)
            second = enhance_chunks([state], config=config, device="cpu", cache_dir=self.root, output_dir=self.root, generate_fn=generate)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(len(generate.call_args.kwargs["out_dirs"]), 1)
        self.assertEqual(first["entries"][0]["supplemental_status"], "succeeded")
        self.assertEqual(second["entries"][0]["supplemental_status"], "reused")
        self.assertEqual(file_hash(Path(state.outputs["audit_json"])), before)
        self.assertFalse(first["lexical_results_rewritten"])

    def test_optional_enhancement_failure_preserves_existing_output(self):
        final = self.root / "final.md"; final.write_text("保留正文", encoding="utf-8")
        with patch("zh_asr.audio_frontend.prepare_pcm16_mono", side_effect=RuntimeError("conversion failed")):
            result = enhance_single(self.audio, {"final": final}, config=self.config, device="cpu", cache_dir=self.root, output_dir=self.root)
        self.assertEqual(result["status"], "failed"); self.assertTrue(result["needs_review"])
        self.assertEqual(final.read_text(encoding="utf-8"), "保留正文")
        self.assertTrue((self.root / "quality.review.html").is_file())


class LeaseContinuityTests(unittest.TestCase):
    def test_default_lease_is_short_and_renewed_during_long_inference(self):
        from zh_asr.gpu_broker import GpuBrokerLease
        lease = GpuBrokerLease("chineseasr-cli", transport=Mock())
        self.assertEqual(lease.ttl_seconds, 120)
        self.assertEqual(lease.renew_interval_seconds, 20)

    def test_inherited_verification_does_not_extend_an_orphan_for_six_hours(self):
        from zh_asr.gpu_broker import verify_inherited_gpu_lease, _VERIFIED_WORKER_TOKEN
        transport = Mock(return_value={"ok": True, "owner": "chineseasr-cli"})
        context = _VERIFIED_WORKER_TOKEN.set("")
        try:
            self.assertEqual(verify_inherited_gpu_lease("fixture-token", transport=transport), "chineseasr-cli")
            self.assertEqual(transport.call_args.args[1]["ttl_seconds"], 120)
        finally:
            _VERIFIED_WORKER_TOKEN.reset(context)


class LongBatchIntegrationTests(unittest.TestCase):
    def test_long_files_are_split_and_completed_batch_is_hash_verified_on_resume(self):
        from zh_asr.batch import run_batch
        from zh_asr.strict_writer import write_strict_bundle
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs, outputs = root / 'in', root / 'out'
            inputs.mkdir()
            wav(inputs / 'long.wav', 41)
            config = replace(load_model_config(), quality={})
            pair = resolve_profile(config, 'high_quality')
            def fake_many(audios, *, out_dirs, primary_engine, secondary_engine, **kwargs):
                return [write_strict_bundle(audio_path=audio, out_dir=out,
                    primary_engine=primary_engine, primary_result={'text': '测试录音'},
                    secondary_engine=secondary_engine, secondary_result={'text': '测试录音'})
                    for audio, out in zip(audios, out_dirs)]
            with patch('zh_asr.long_audio._runtime_artifact_identity', return_value={}), \
                 patch('zh_asr.long_audio.strict_transcribe_many', side_effect=fake_many) as chunks:
                first = run_batch(inputs, outputs, device='cpu', config=config,
                    primary_engine=pair[0], secondary_engine=pair[1], strict_many_fn=fake_many,
                    cache_dir=root / 'cache')
                self.assertEqual(first.failed, 0)
                self.assertEqual(first.processed, 1)
                self.assertEqual(len(chunks.call_args.args[0]), 2)
                second = run_batch(inputs, outputs, device='cpu', config=config,
                    primary_engine=pair[0], secondary_engine=pair[1], strict_many_fn=fake_many,
                    cache_dir=root / 'cache')
                self.assertEqual(second.skipped, 1)
                self.assertEqual(chunks.call_count, 1)
                manifest = json.loads((outputs / 'long' / 'manifest.json').read_text(encoding='utf-8'))
                raw = Path(manifest['chunks'][0]['outputs']['primary_json'])
                raw.write_text('{"text":"changed"}', encoding='utf-8')
                third = run_batch(inputs, outputs, device='cpu', config=config,
                    primary_engine=pair[0], secondary_engine=pair[1], strict_many_fn=fake_many,
                    cache_dir=root / 'cache')
                self.assertEqual(third.skipped, 0)
                self.assertEqual(third.failed, 0)
                self.assertEqual(chunks.call_count, 2)

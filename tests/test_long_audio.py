import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from zh_asr.audio_frontend import PreparedAudio
from zh_asr.long_audio import plan_chunks, run_long_transcription
from zh_asr.strict_writer import write_strict_bundle


class LongAudioTests(unittest.TestCase):
    def test_runtime_code_identity_changes_with_operational_source_bytes(self):
        from zh_asr.long_audio import _runtime_code_identity

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src" / "zh_asr" / "worker.py"
            runtime = root / "runtime" / "bridge.py"
            source.parent.mkdir(parents=True)
            runtime.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            runtime.write_text("RUNTIME = 1\n", encoding="utf-8")

            first = _runtime_code_identity(root)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            second = _runtime_code_identity(root)

        self.assertEqual("zh_asr.runtime_code_identity.v1", first["schema"])
        self.assertEqual(2, first["file_count"])
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_runtime_artifact_identity_uses_qwen_fail_closed_verifier(self):
        from zh_asr.long_audio import _runtime_artifact_identity

        spec = SimpleNamespace(
            name="qwen3-asr-1.7b",
            adapter="qwen-asr",
            model="Qwen/Qwen3-ASR-1.7B",
            options={},
        )
        config = SimpleNamespace(
            path=Path("models.yaml"),
            engines={"qwen3-asr-1.7b": spec},
            model_aliases={},
        )
        expected = {
            "engine": "qwen3-asr-1.7b",
            "model_receipt_status": "verified",
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "zh_asr.qwen_identity.qwen_runtime_identity",
                return_value=expected,
            ) as verifier,
        ):
            cache = Path(tmp)
            identities = _runtime_artifact_identity(
                config,
                ("qwen3-asr-1.7b",),
                cache,
            )

        self.assertEqual([expected], identities)
        verifier.assert_called_once_with(spec, cache, {})

    def test_plan_chunks_uses_duration_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "long.wav"
            _write_wav(audio, seconds=7)

            chunks = plan_chunks(audio, chunk_sec=3, overlap_sec=1)

        self.assertEqual(3, len(chunks))
        self.assertEqual((0, 3000), (chunks[0].start_ms, chunks[0].end_ms))
        self.assertEqual((2000, 5000), (chunks[1].start_ms, chunks[1].end_ms))
        self.assertEqual((4000, 7000), (chunks[2].start_ms, chunks[2].end_ms))

    def test_plan_chunks_keeps_legacy_300_second_default_without_capability_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "long.wav"
            _write_wav(audio, seconds=301)

            chunks = plan_chunks(audio)

        self.assertEqual(2, len(chunks))
        self.assertEqual((0, 300_000), (chunks[0].start_ms, chunks[0].end_ms))
        self.assertEqual((299_000, 301_000), (chunks[1].start_ms, chunks[1].end_ms))

    def test_plan_chunks_applies_recommended_and_absolute_engine_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "long.wav"
            _write_wav(audio, seconds=81)

            chunks = plan_chunks(
                audio,
                chunk_sec=300,
                overlap_sec=1,
                recommended_chunk_sec=35,
                absolute_max_chunk_sec=40,
            )

        self.assertEqual(3, len(chunks))
        self.assertTrue(all(chunk.end_ms - chunk.start_ms <= 35_000 for chunk in chunks))
        self.assertEqual((0, 35_000), (chunks[0].start_ms, chunks[0].end_ms))
        self.assertEqual((34_000, 69_000), (chunks[1].start_ms, chunks[1].end_ms))

    def test_plan_chunks_uses_absolute_limit_when_recommendation_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "long.wav"
            _write_wav(audio, seconds=81)

            chunks = plan_chunks(
                audio,
                chunk_sec=300,
                overlap_sec=1,
                absolute_max_chunk_sec=40,
            )

        self.assertTrue(all(chunk.end_ms - chunk.start_ms <= 40_000 for chunk in chunks))
        self.assertEqual((0, 40_000), (chunks[0].start_ms, chunks[0].end_ms))

    def test_plan_chunks_validates_overlap_against_effective_capability_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "long.wav"
            _write_wav(audio, seconds=81)

            with self.assertRaisesRegex(ValueError, "effective chunk"):
                plan_chunks(
                    audio,
                    chunk_sec=300,
                    overlap_sec=35,
                    recommended_chunk_sec=35,
                    absolute_max_chunk_sec=40,
                )

    def test_run_long_transcription_resumes_completed_chunks_and_merges_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            out_dir = root / "out"
            _write_wav(audio, seconds=7)
            calls: list[str] = []

            def fake_strict(audio_path, **kwargs):
                calls.append(audio_path.stem)
                return _write_fake_outputs(
                    audio_path,
                    kwargs["out_dir"],
                    f"文本 {audio_path.stem}",
                    primary_engine=kwargs["primary_engine"],
                    secondary_engine=kwargs["secondary_engine"],
                )

            first = run_long_transcription(audio, out_dir, chunk_sec=3, overlap_sec=1, strict_fn=fake_strict)
            second = run_long_transcription(audio, out_dir, chunk_sec=3, overlap_sec=1, strict_fn=fake_strict)
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            transcript = (out_dir / "transcript.md").read_text(encoding="utf-8")
            audit = (out_dir / "audit.md").read_text(encoding="utf-8")
            metrics_exists = (out_dir / "metrics.json").exists()

        self.assertEqual(3, first.total)
        self.assertEqual(3, first.processed)
        self.assertEqual(0, first.skipped)
        self.assertEqual(3, second.skipped)
        self.assertEqual(3, len(calls))
        self.assertEqual("succeeded", manifest["chunks"][0]["status"])
        self.assertEqual("verified", manifest["evidence_status"])
        self.assertTrue(
            all(chunk["evidence_status"] == "verified" for chunk in manifest["chunks"])
        )
        self.assertIn("文本 chunk-000001", transcript)
        self.assertIn("chunk-000002 Strict Audit", audit)
        self.assertTrue(metrics_exists)

    def test_run_long_transcription_default_batches_pending_chunks_and_preserves_resume_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            out_dir = root / "out"
            config_path = root / "models.yaml"
            config_path.write_text("fixture", encoding="utf-8")
            _write_wav(audio, seconds=8)
            config = _model_config(config_path, primary="primary", secondary="secondary")
            batch_calls: list[dict] = []

            def fake_strict_many(audio_paths, *, out_dirs, **kwargs):
                batch_calls.append(
                    {
                        "audio_stems": [path.stem for path in audio_paths],
                        "out_dirs": [path.name for path in out_dirs],
                        **kwargs,
                    }
                )
                return [
                    _write_fake_outputs(
                        audio_path,
                        chunk_out_dir,
                        f"文本 {audio_path.stem}",
                        primary_engine=kwargs["primary_engine"],
                        secondary_engine=kwargs["secondary_engine"],
                    )
                    for audio_path, chunk_out_dir in zip(audio_paths, out_dirs)
                ]

            with (
                patch("zh_asr.long_audio.load_model_config", return_value=config),
                patch("zh_asr.long_audio.strict_transcribe_many", side_effect=fake_strict_many),
            ):
                first = run_long_transcription(audio, out_dir, chunk_sec=2, overlap_sec=0)
                resumed = run_long_transcription(audio, out_dir, chunk_sec=2, overlap_sec=0)
                forced = run_long_transcription(
                    audio,
                    out_dir,
                    chunk_sec=2,
                    overlap_sec=0,
                    force=True,
                )

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual((4, 0, 0), (first.processed, first.skipped, first.failed))
        self.assertEqual((0, 4, 0), (resumed.processed, resumed.skipped, resumed.failed))
        self.assertEqual((4, 0, 0), (forced.processed, forced.skipped, forced.failed))
        self.assertEqual(2, len(batch_calls))
        self.assertEqual(
            ["chunk-000001", "chunk-000002", "chunk-000003", "chunk-000004"],
            batch_calls[0]["audio_stems"],
        )
        self.assertEqual(batch_calls[0]["audio_stems"], batch_calls[0]["out_dirs"])
        self.assertEqual("primary", batch_calls[0]["primary_engine"])
        self.assertEqual("secondary", batch_calls[0]["secondary_engine"])
        self.assertIs(config, batch_calls[0]["config"])
        self.assertTrue(all(chunk["status"] == "succeeded" for chunk in manifest["chunks"]))
        self.assertTrue(all("final" in chunk["outputs"] for chunk in manifest["chunks"]))

    def test_run_long_transcription_bounds_batches_and_continues_after_batch_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "hearing.wav"
            out_dir = root / "out"
            manifest_path = out_dir / "manifest.json"
            config_path = root / "models.yaml"
            config_path.write_text("fixture", encoding="utf-8")
            _write_wav(audio, seconds=5)
            config = _model_config(
                config_path,
                primary="primary",
                secondary="secondary",
                engine_options={
                    "primary": {"max_request_inputs": 2},
                    "secondary": {"max_request_inputs": 3},
                },
            )
            batch_calls: list[list[str]] = []
            manifest_snapshots: list[list[str]] = []

            def fake_strict_many(audio_paths, *, out_dirs, **kwargs):
                batch_calls.append([path.stem for path in audio_paths])
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_snapshots.append([chunk["status"] for chunk in manifest["chunks"]])
                if len(batch_calls) == 1:
                    raise RuntimeError("first batch failed")
                return [
                    _write_fake_outputs(
                        audio_path,
                        chunk_out_dir,
                        f"文本 {audio_path.stem}",
                        primary_engine=kwargs["primary_engine"],
                        secondary_engine=kwargs["secondary_engine"],
                    )
                    for audio_path, chunk_out_dir in zip(audio_paths, out_dirs)
                ]

            with (
                patch("zh_asr.long_audio.load_model_config", return_value=config),
                patch("zh_asr.long_audio.strict_transcribe_many", side_effect=fake_strict_many),
            ):
                summary = run_long_transcription(
                    audio,
                    out_dir,
                    chunk_sec=1,
                    overlap_sec=0,
                )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(
            [
                ["chunk-000001", "chunk-000002"],
                ["chunk-000003", "chunk-000004"],
                ["chunk-000005"],
            ],
            batch_calls,
        )
        self.assertEqual(
            [
                ["running", "running", "pending", "pending", "pending"],
                ["failed", "failed", "running", "running", "pending"],
                ["failed", "failed", "succeeded", "succeeded", "running"],
            ],
            manifest_snapshots,
        )
        self.assertEqual((5, 3, 0, 2), (summary.total, summary.processed, summary.skipped, summary.failed))
        self.assertEqual(
            ["failed", "failed", "succeeded", "succeeded", "succeeded"],
            [chunk["status"] for chunk in manifest["chunks"]],
        )
        self.assertTrue(all("first batch failed" in chunk["error"] for chunk in manifest["chunks"][:2]))
        self.assertTrue(all("final" in chunk["outputs"] for chunk in manifest["chunks"][2:]))
        self.assertEqual("unavailable", manifest["evidence_status"])
        self.assertTrue(
            all(
                chunk["evidence_status"] == "unavailable"
                for chunk in manifest["chunks"][:2]
            )
        )

    def test_long_manifest_marks_primary_engine_fallback_provisional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "court-call.wav"
            out_dir = root / "out"
            _write_wav(audio, seconds=2)

            def fake_strict(audio_path, **kwargs):
                outputs = _write_fake_outputs(
                    audio_path,
                    kwargs["out_dir"],
                    "他目前还没交。",
                    evidence_status="provisional",
                    primary_engine=kwargs["primary_engine"],
                    secondary_engine=kwargs["secondary_engine"],
                    engine_evidence=[
                        {
                            "engine": "fireredasr2-llm",
                            "role": "lexical_primary",
                            "execution_status": "failed",
                            "error": "RuntimeError: worker exited 9",
                        },
                        {
                            "engine": "qwen3-asr-1.7b",
                            "role": "lexical_verifier",
                            "execution_status": "succeeded",
                            "error": None,
                        },
                    ],
                )
                return outputs

            run_long_transcription(
                audio,
                out_dir,
                chunk_sec=2,
                overlap_sec=0,
                primary_engine="fireredasr2-llm",
                secondary_engine="qwen3-asr-1.7b",
                strict_fn=fake_strict,
            )
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["evidence_status"], "provisional")
        self.assertEqual(manifest["chunks"][0]["evidence_status"], "provisional")
        self.assertEqual(
            manifest["chunks"][0]["evidence_failures"][0]["engine"],
            "fireredasr2-llm",
        )
        self.assertEqual(
            manifest["evidence_failures"][0]["role"],
            "lexical_primary",
        )

    def test_run_long_transcription_resets_stale_running_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            out_dir = root / "out"
            _write_wav(audio, seconds=4)
            calls: list[str] = []

            def fake_strict(audio_path, **kwargs):
                calls.append(audio_path.stem)
                return _write_fake_outputs(
                    audio_path,
                    kwargs["out_dir"],
                    f"文本 {audio_path.stem}",
                    primary_engine=kwargs["primary_engine"],
                    secondary_engine=kwargs["secondary_engine"],
                )

            run_long_transcription(audio, out_dir, chunk_sec=2, overlap_sec=0, strict_fn=fake_strict)
            manifest_path = out_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["chunks"][1]["status"] = "running"
            manifest["chunks"][1]["outputs"] = {}
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            calls.clear()

            summary = run_long_transcription(audio, out_dir, chunk_sec=2, overlap_sec=0, strict_fn=fake_strict)
            refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(["chunk-000002"], calls)
        self.assertEqual(1, summary.processed)
        self.assertEqual("succeeded", refreshed["chunks"][1]["status"])

    def test_long_resume_reprocesses_chunk_when_required_raw_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            out_dir = root / "out"
            _write_wav(audio, seconds=4)
            calls: list[str] = []

            def fake_strict(audio_path, **kwargs):
                calls.append(audio_path.stem)
                return _write_fake_outputs(
                    audio_path,
                    kwargs["out_dir"],
                    f"文本 {audio_path.stem}",
                    primary_engine=kwargs["primary_engine"],
                    secondary_engine=kwargs["secondary_engine"],
                )

            run_long_transcription(
                audio,
                out_dir,
                chunk_sec=2,
                overlap_sec=0,
                strict_fn=fake_strict,
            )
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            Path(manifest["chunks"][0]["outputs"]["primary_json"]).unlink()
            calls.clear()

            resumed = run_long_transcription(
                audio,
                out_dir,
                chunk_sec=2,
                overlap_sec=0,
                strict_fn=fake_strict,
            )

        self.assertEqual(["chunk-000001"], calls)
        self.assertEqual((1, 1, 0), (resumed.processed, resumed.skipped, resumed.failed))
        self.assertEqual("verified", resumed.evidence_status)

    def test_long_resume_reprocesses_chunk_when_receipt_bound_text_is_tampered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            out_dir = root / "out"
            _write_wav(audio, seconds=4)
            calls: list[str] = []

            def fake_strict(audio_path, **kwargs):
                calls.append(audio_path.stem)
                return _write_fake_outputs(
                    audio_path,
                    kwargs["out_dir"],
                    f"文本 {audio_path.stem}",
                    primary_engine=kwargs["primary_engine"],
                    secondary_engine=kwargs["secondary_engine"],
                )

            run_long_transcription(
                audio,
                out_dir,
                chunk_sec=2,
                overlap_sec=0,
                strict_fn=fake_strict,
            )
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            final_path = Path(manifest["chunks"][0]["outputs"]["final"])
            final_path.write_text(
                final_path.read_text(encoding="utf-8") + "\n篡改文本\n",
                encoding="utf-8",
            )
            calls.clear()

            resumed = run_long_transcription(
                audio,
                out_dir,
                chunk_sec=2,
                overlap_sec=0,
                strict_fn=fake_strict,
            )

        self.assertEqual(["chunk-000001"], calls)
        self.assertEqual((1, 1, 0), (resumed.processed, resumed.skipped, resumed.failed))
        self.assertEqual("verified", resumed.evidence_status)

    def test_run_long_transcription_arbitrates_only_flagged_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            out_dir = root / "out"
            _write_wav(audio, seconds=4)

            def fake_strict(audio_path, **kwargs):
                conflict = audio_path.stem == "chunk-000002"
                return _write_fake_outputs(
                    audio_path,
                    kwargs["out_dir"],
                    f"文本 {audio_path.stem}",
                    primary_engine=kwargs["primary_engine"],
                    secondary_engine=kwargs["secondary_engine"],
                    primary_text=(
                        f"甲方明确表示不同意 {audio_path.stem}"
                        if conflict
                        else None
                    ),
                    secondary_text=(
                        f"乙方完全没有相关表态 {audio_path.stem}"
                        if conflict
                        else None
                    ),
                )

            arbiter = _FakeArbiter()

            run_long_transcription(audio, out_dir, chunk_sec=2, overlap_sec=0, strict_fn=fake_strict, arbiter=arbiter)
            metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(["chunk-000002"], arbiter.calls)
        self.assertEqual("LLM 仲裁文本", metrics["chunks"][1]["arbitration"]["final_text"])

    def test_run_long_transcription_uses_both_engine_capabilities_and_records_effective_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            out_dir = root / "out"
            config_path = root / "models.yaml"
            config_path.write_text("fixture", encoding="utf-8")
            _write_wav(audio, seconds=70)
            calls: list[str] = []

            config = _model_config(
                config_path,
                primary="firered",
                secondary="anchor",
                engine_options={
                    "firered": {"recommended_chunk_sec": 35, "max_audio_sec": 40},
                    "anchor": {"recommended_chunk_sec": 30, "max_audio_sec": 45},
                },
            )

            def fake_strict(audio_path, **kwargs):
                calls.append(audio_path.stem)
                return _write_fake_outputs(
                    audio_path,
                    kwargs["out_dir"],
                    f"文本 {audio_path.stem}",
                    primary_engine=kwargs["primary_engine"],
                    secondary_engine=kwargs["secondary_engine"],
                )

            with patch("zh_asr.long_audio.load_model_config", return_value=config):
                run_long_transcription(audio, out_dir, strict_fn=fake_strict)

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(3, len(calls))
        self.assertEqual(300, manifest["requested_chunk_sec"])
        self.assertEqual(30, manifest["chunk_sec"])
        self.assertEqual(30, manifest["effective_chunk_sec"])
        self.assertEqual("firered", manifest["resolved_primary_engine"])
        self.assertEqual("anchor", manifest["resolved_secondary_engine"])
        self.assertTrue(
            all(chunk["end_ms"] - chunk["start_ms"] <= 30_000 for chunk in manifest["chunks"])
        )

    def test_run_long_transcription_reprocesses_when_resolved_default_engine_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            out_dir = root / "out"
            config_path = root / "models.yaml"
            config_path.write_text("same config bytes", encoding="utf-8")
            _write_wav(audio, seconds=4)
            calls: list[tuple[str, str]] = []

            def fake_strict(audio_path, **kwargs):
                calls.append((audio_path.stem, kwargs["primary_engine"]))
                return _write_fake_outputs(
                    audio_path,
                    kwargs["out_dir"],
                    f"文本 {audio_path.stem}",
                    primary_engine=kwargs["primary_engine"],
                    secondary_engine=kwargs["secondary_engine"],
                )

            first_config = _model_config(config_path, primary="primary-a", secondary="anchor")
            second_config = _model_config(config_path, primary="primary-b", secondary="anchor")
            with patch(
                "zh_asr.long_audio.load_model_config",
                side_effect=[first_config, second_config],
            ):
                run_long_transcription(
                    audio,
                    out_dir,
                    chunk_sec=2,
                    overlap_sec=0,
                    strict_fn=fake_strict,
                )
                run_long_transcription(
                    audio,
                    out_dir,
                    chunk_sec=2,
                    overlap_sec=0,
                    strict_fn=fake_strict,
                )

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(4, len(calls))
        self.assertEqual(["primary-a", "primary-a", "primary-b", "primary-b"], [item[1] for item in calls])
        self.assertEqual("primary-b", manifest["resolved_primary_engine"])

    def test_run_long_transcription_reprocesses_when_explicit_engine_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            out_dir = root / "out"
            config_path = root / "models.yaml"
            config_path.write_text("same config bytes", encoding="utf-8")
            _write_wav(audio, seconds=4)
            calls: list[str] = []
            config = _model_config(config_path, primary="default-primary", secondary="anchor")
            config.engines.update(
                {
                    "primary-a": SimpleNamespace(name="primary-a", adapter="fixture", options={}),
                    "primary-b": SimpleNamespace(name="primary-b", adapter="fixture", options={}),
                }
            )

            def fake_strict(audio_path, **kwargs):
                calls.append(kwargs["primary_engine"])
                return _write_fake_outputs(
                    audio_path,
                    kwargs["out_dir"],
                    f"文本 {audio_path.stem}",
                    primary_engine=kwargs["primary_engine"],
                    secondary_engine=kwargs["secondary_engine"],
                )

            with patch("zh_asr.long_audio.load_model_config", return_value=config):
                run_long_transcription(
                    audio,
                    out_dir,
                    chunk_sec=2,
                    overlap_sec=0,
                    primary_engine="primary-a",
                    strict_fn=fake_strict,
                )
                run_long_transcription(
                    audio,
                    out_dir,
                    chunk_sec=2,
                    overlap_sec=0,
                    primary_engine="primary-b",
                    strict_fn=fake_strict,
                )

        self.assertEqual(["primary-a", "primary-a", "primary-b", "primary-b"], calls)

    def test_run_long_transcription_fingerprint_includes_device_cache_and_runtime_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            out_dir = root / "out"
            config_path = root / "models.yaml"
            config_path.write_text("stable config bytes", encoding="utf-8")
            model_dir = root / "models" / "firered"
            model_dir.mkdir(parents=True)
            receipt_path = model_dir / "MODEL_RECEIPT.json"
            receipt_path.write_text('{"revision":"revision-a","files":[]}', encoding="utf-8")
            _write_wav(audio, seconds=2)
            config = _model_config(
                config_path,
                primary="firered",
                secondary="anchor",
                engine_options={
                    "firered": {
                        "model_dir": str(model_dir),
                        "model_revision": "revision-a",
                        "source_revision": "source-a",
                    },
                },
            )
            calls: list[tuple[str, str]] = []

            def fake_strict(audio_path, **kwargs):
                calls.append((kwargs["device"], str(kwargs["cache_dir"])))
                return _write_fake_outputs(
                    audio_path,
                    kwargs["out_dir"],
                    f"文本 {audio_path.stem}",
                    primary_engine=kwargs["primary_engine"],
                    secondary_engine=kwargs["secondary_engine"],
                )

            cache_a = root / "cache-a"
            cache_b = root / "cache-b"
            with patch("zh_asr.long_audio.load_model_config", return_value=config):
                first = run_long_transcription(
                    audio,
                    out_dir,
                    chunk_sec=1,
                    overlap_sec=0,
                    device="cuda:0",
                    cache_dir=cache_a,
                    strict_fn=fake_strict,
                )
                resumed = run_long_transcription(
                    audio,
                    out_dir,
                    chunk_sec=1,
                    overlap_sec=0,
                    device="cuda:0",
                    cache_dir=cache_a,
                    strict_fn=fake_strict,
                )
                changed_device = run_long_transcription(
                    audio,
                    out_dir,
                    chunk_sec=1,
                    overlap_sec=0,
                    device="cpu",
                    cache_dir=cache_a,
                    strict_fn=fake_strict,
                )
                changed_cache = run_long_transcription(
                    audio,
                    out_dir,
                    chunk_sec=1,
                    overlap_sec=0,
                    device="cpu",
                    cache_dir=cache_b,
                    strict_fn=fake_strict,
                )
                receipt_path.write_text(
                    '{"revision":"revision-a","files":[{"path":"model.bin","sha256":"new"}]}',
                    encoding="utf-8",
                )
                changed_receipt = run_long_transcription(
                    audio,
                    out_dir,
                    chunk_sec=1,
                    overlap_sec=0,
                    device="cpu",
                    cache_dir=cache_b,
                    strict_fn=fake_strict,
                )

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual((2, 0), (first.processed, first.skipped))
        self.assertEqual((0, 2), (resumed.processed, resumed.skipped))
        self.assertEqual(2, changed_device.processed)
        self.assertEqual(2, changed_cache.processed)
        self.assertEqual(2, changed_receipt.processed)
        self.assertEqual(8, len(calls))
        self.assertEqual("cpu", manifest["device"])
        self.assertEqual(str(cache_b.resolve()), manifest["cache_dir"])
        self.assertEqual(
            "present",
            manifest["runtime_artifact_identity"][0]["model_receipt_status"],
        )
        self.assertEqual(
            "source-a",
            manifest["runtime_artifact_identity"][0]["source_revision"],
        )

    def test_run_long_transcription_schema_one_manifest_is_explicit_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            out_dir = root / "out"
            _write_wav(audio, seconds=2)
            calls: list[str] = []

            def fake_strict(audio_path, **kwargs):
                calls.append(audio_path.stem)
                return _write_fake_outputs(
                    audio_path,
                    kwargs["out_dir"],
                    f"文本 {audio_path.stem}",
                    primary_engine=kwargs["primary_engine"],
                    secondary_engine=kwargs["secondary_engine"],
                )

            first = run_long_transcription(
                audio,
                out_dir,
                chunk_sec=1,
                overlap_sec=0,
                strict_fn=fake_strict,
            )
            manifest_path = out_dir / "manifest.json"
            legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
            legacy["schema_version"] = 1
            manifest_path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2), encoding="utf-8")
            calls.clear()

            rerun = run_long_transcription(
                audio,
                out_dir,
                chunk_sec=1,
                overlap_sec=0,
                strict_fn=fake_strict,
            )
            refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(2, first.processed)
        self.assertEqual(["chunk-000001", "chunk-000002"], calls)
        self.assertEqual((2, 0), (rerun.processed, rerun.skipped))
        self.assertEqual(2, refreshed["schema_version"])

    def test_merged_transcript_removes_only_exact_adjacent_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            out_dir = root / "out"
            _write_wav(audio, seconds=4)
            texts = {
                "chunk-000001": "法院说可以换票",
                "chunk-000002": "可以换票并携带缴费凭证",
            }

            def fake_strict(audio_path, **kwargs):
                return _write_fake_outputs(
                    audio_path,
                    kwargs["out_dir"],
                    texts[audio_path.stem],
                    primary_engine=kwargs["primary_engine"],
                    secondary_engine=kwargs["secondary_engine"],
                )

            run_long_transcription(
                audio,
                out_dir,
                chunk_sec=3,
                overlap_sec=1,
                strict_fn=fake_strict,
            )
            transcript = (out_dir / "transcript.md").read_text(encoding="utf-8")

        self.assertEqual(1, transcript.count("可以换票"))
        self.assertIn("并携带缴费凭证", transcript)
        self.assertIn("exact-boundary-overlap-removed: 4 chars", transcript)

    def test_run_long_transcription_prepares_mp3_and_records_derivative_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "court-call.mp3"
            audio.write_bytes(b"source mp3 bytes")
            out_dir = root / "out"
            derived = out_dir / "_derived" / "court-call.16k-mono.wav"
            derived.parent.mkdir(parents=True)
            _write_wav(derived, seconds=4, sample_rate=16000)
            prepared = PreparedAudio(
                source_path=audio.resolve(),
                path=derived.resolve(),
                converted=True,
                source_sha256="source-sha",
                derivative_sha256="derivative-sha",
                sample_rate=16000,
                channels=1,
                sample_width_bytes=2,
                duration_sec=4.0,
                ffmpeg_version="ffmpeg fixture",
                conversion_command=("ffmpeg", "-i", str(audio), str(derived)),
            )
            calls: list[Path] = []

            def fake_strict(audio_path, **kwargs):
                calls.append(audio_path)
                return _write_fake_outputs(
                    audio_path,
                    kwargs["out_dir"],
                    f"文本 {audio_path.stem}",
                    primary_engine=kwargs["primary_engine"],
                    secondary_engine=kwargs["secondary_engine"],
                )

            with patch(
                "zh_asr.long_audio.prepare_pcm16_mono",
                return_value=prepared,
            ) as prepare:
                run_long_transcription(
                    audio,
                    out_dir,
                    chunk_sec=2,
                    overlap_sec=0,
                    strict_fn=fake_strict,
                )

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

        prepare.assert_called_once_with(audio.resolve(), out_dir.resolve() / "_derived")
        self.assertEqual(2, len(calls))
        self.assertTrue(all(path.suffix == ".wav" for path in calls))
        self.assertEqual(str(audio.resolve()), manifest["audio"])
        self.assertEqual(str(derived.resolve()), manifest["prepared_audio"]["path"])
        self.assertEqual("source-sha", manifest["prepared_audio"]["source_sha256"])
        self.assertEqual("derivative-sha", manifest["prepared_audio"]["derivative_sha256"])


def _write_wav(path: Path, seconds: int, sample_rate: int = 8000) -> None:
    frames = b"\x00\x00" * sample_rate * seconds
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)


def _write_fake_outputs(
    audio_path: Path,
    chunk_dir: Path,
    text: str,
    *,
    evidence_status: str = "verified",
    primary_engine: str = "primary",
    secondary_engine: str = "secondary",
    engine_evidence: list[dict] | None = None,
    primary_text: str | None = None,
    secondary_text: str | None = None,
) -> dict[str, Path]:
    evidence = engine_evidence or [
        {
            "engine": primary_engine,
            "role": "lexical_primary",
            "execution_status": "succeeded",
            "error": None,
        },
        {
            "engine": secondary_engine,
            "role": "lexical_verifier",
            "execution_status": "succeeded",
            "error": None,
        },
    ]
    by_role = {item["role"]: item for item in evidence}
    primary_item = by_role["lexical_primary"]
    secondary_item = by_role["lexical_verifier"]

    def result_for(item: dict, value: str) -> dict:
        if item["execution_status"] != "failed":
            return {"engine": item["engine"], "text": value, "error": None}
        error_type, _, message = str(item["error"]).partition(":")
        return {
            "engine": item["engine"],
            "text": "",
            "error": {
                "type": error_type.strip(),
                "message": message.strip(),
            },
        }

    outputs = write_strict_bundle(
        audio_path=audio_path,
        primary_engine=primary_engine,
        primary_result=result_for(primary_item, primary_text or text),
        secondary_engine=secondary_engine,
        secondary_result=result_for(secondary_item, secondary_text or text),
        out_dir=chunk_dir,
        primary_error=primary_item.get("error"),
        secondary_error=secondary_item.get("error"),
    )
    if outputs["evidence_status"] != evidence_status:
        raise AssertionError(
            f"fixture expected {evidence_status}, got {outputs['evidence_status']}"
        )
    return outputs


def _model_config(
    path: Path,
    *,
    primary: str,
    secondary: str,
    engine_options: dict[str, dict] | None = None,
):
    options_by_engine = engine_options or {}
    names = {primary, secondary}
    engines = {
        name: SimpleNamespace(name=name, adapter="fixture", options=options_by_engine.get(name, {}))
        for name in names
    }
    return SimpleNamespace(
        path=path,
        strict_primary_engine=primary,
        strict_secondary_engine=secondary,
        engines=engines,
    )


class _FakeDecision:
    def to_dict(self):
        return {"final_text": "LLM 仲裁文本", "confidence": 0.9}


class _FakeArbiter:
    def __init__(self):
        self.calls = []

    def arbitrate(self, evidence):
        self.calls.append(evidence.chunk_id)
        return _FakeDecision()


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from zh_asr.strict_writer import write_strict_bundle


class BatchTests(unittest.TestCase):
    def test_find_audio_files_recurses_and_filters_supported_extensions(self):
        from zh_asr.batch import find_audio_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.wav").write_text("", encoding="utf-8")
            (root / "b.MP3").write_text("", encoding="utf-8")
            (root / "notes.txt").write_text("", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "c.flac").write_text("", encoding="utf-8")
            (nested / "d.m4a").write_text("", encoding="utf-8")

            found = [path.name for path in find_audio_files(root)]

        self.assertEqual(found, ["a.wav", "b.MP3", "c.flac", "d.m4a"])

    def test_run_batch_skips_completed_files_records_failures_and_writes_summary(self):
        from zh_asr.batch import run_batch

        calls: list[Path] = []

        def fake_strict(audio_path, *, primary_engine, secondary_engine, device, out_dir, cache_dir, config):
            calls.append(audio_path)
            if audio_path.name == "bad.flac":
                raise RuntimeError("decode failed")
            return write_strict_bundle(
                audio_path=audio_path,
                primary_engine=primary_engine,
                primary_result={"engine": primary_engine, "text": "ok", "error": None},
                secondary_engine=secondary_engine,
                secondary_result={"engine": secondary_engine, "text": "ok", "error": None},
                out_dir=out_dir,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            (input_dir / "done.mp3").write_text("", encoding="utf-8")

            # Seed a complete, hash-bound batch receipt for the file that the
            # second invocation should skip.  A path-only Markdown file is no
            # longer treated as a valid cache hit.
            run_batch(
                input_dir=input_dir,
                out_dir=output_dir,
                mode="strict",
                device="cuda:0",
                cache_dir=root / "cache",
                strict_fn=fake_strict,
            )
            calls.clear()
            (input_dir / "ok.wav").write_text("", encoding="utf-8")
            (input_dir / "bad.flac").write_text("", encoding="utf-8")

            summary = run_batch(
                input_dir=input_dir,
                out_dir=output_dir,
                mode="strict",
                device="cuda:0",
                cache_dir=root / "cache",
                strict_fn=fake_strict,
            )

            failed_lines = (output_dir / "failed.jsonl").read_text(encoding="utf-8").splitlines()
            failed = json.loads(failed_lines[0])
            summary_text = (output_dir / "summary.md").read_text(encoding="utf-8")
            ok_output_exists = (output_dir / "ok" / "ok.strict.md").exists()

        self.assertEqual(summary.total, 3)
        self.assertEqual(summary.processed, 1)
        self.assertEqual(summary.skipped, 1)
        self.assertEqual(summary.failed, 1)
        self.assertEqual([path.name for path in calls], ["bad.flac", "ok.wav"])
        self.assertEqual(failed["audio"].endswith("bad.flac"), True)
        self.assertEqual(failed["error_type"], "RuntimeError")
        self.assertIn("decode failed", failed["error"])
        self.assertIn("Processed: 1", summary_text)
        self.assertTrue(ok_output_exists)

    def test_run_batch_reprocesses_when_source_bytes_change(self):
        from zh_asr.batch import run_batch

        calls: list[bytes] = []

        def fake_strict(audio_path, *, primary_engine, secondary_engine, out_dir, **_kwargs):
            calls.append(audio_path.read_bytes())
            return write_strict_bundle(
                audio_path=audio_path,
                primary_engine=primary_engine,
                primary_result={"engine": primary_engine, "text": "ok", "error": None},
                secondary_engine=secondary_engine,
                secondary_result={"engine": secondary_engine, "text": "ok", "error": None},
                out_dir=out_dir,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            audio = input_dir / "same.wav"
            audio.write_bytes(b"first")

            first = run_batch(input_dir, output_dir, mode="strict", strict_fn=fake_strict)
            audio.write_bytes(b"second")
            second = run_batch(input_dir, output_dir, mode="strict", strict_fn=fake_strict)

        self.assertEqual((1, 0, 0), (first.processed, first.skipped, first.failed))
        self.assertEqual((1, 0, 0), (second.processed, second.skipped, second.failed))
        self.assertEqual([b"first", b"second"], calls)

    def test_run_batch_keeps_colliding_sources_separate_and_resumable(self):
        from zh_asr.batch import run_batch

        calls = []

        def fake_strict(audio_path, *, primary_engine, secondary_engine, out_dir, **_kwargs):
            calls.append(audio_path)
            text = audio_path.read_text(encoding="utf-8")
            return write_strict_bundle(
                audio_path=audio_path,
                primary_engine=primary_engine,
                primary_result={"engine": primary_engine, "text": text, "error": None},
                secondary_engine=secondary_engine,
                secondary_result={"engine": secondary_engine, "text": text, "error": None},
                out_dir=out_dir,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir, output_dir = root / "input", root / "output"
            names = ("recording.wav", "recording.mp3", "recording[1].wav",
                     "recording(1).wav", "group/recording.wav", "group__recording.wav",
                     "Case.wav", "case.mp3", "...wav", "....wav", "_.wav", "_..wav")
            for index, name in enumerate(names):
                audio = input_dir / name
                audio.parent.mkdir(parents=True, exist_ok=True)
                audio.write_text(f"source transcript {index}", encoding="utf-8")
            first = run_batch(input_dir, output_dir, mode="strict", strict_fn=fake_strict)
            snapshots = []
            for item in first.items:
                self.assertTrue(item.out_dir.resolve().is_relative_to(output_dir.resolve()), str(item))
                state = json.loads((item.out_dir / "batch-item.json").read_text(encoding="utf-8"))
                final = Path(state["outputs"]["final"]["path"]).read_text(encoding="utf-8")
                snapshots.append((str(item.audio), str(item.out_dir), state["identity"]["audio_path"], final))
            self.assertEqual(len(names), len({item.out_dir for item in first.items}), repr(snapshots))
            self.assertEqual((len(names), 0, 0), (first.processed, first.skipped, first.failed))
            for audio, _out_dir, saved_audio, final in snapshots:
                self.assertEqual(str(Path(audio).resolve()), saved_audio)
                self.assertIn(Path(audio).read_text(encoding="utf-8"), final)
            calls.clear()
            second = run_batch(input_dir, output_dir, mode="strict", strict_fn=fake_strict)
            self.assertEqual((0, len(names), 0), (second.processed, second.skipped, second.failed))
            self.assertEqual([], calls)
            for name in names[1::2]:
                (input_dir / name).rename(root / (name.replace("/", "__") + ".removed"))
            third = run_batch(input_dir, output_dir, mode="strict", strict_fn=fake_strict)
            self.assertEqual((0, len(names) // 2, 0), (third.processed, third.skipped, third.failed))
            self.assertEqual([], calls)
            self.assertEqual(
                {item.audio: item.out_dir for item in first.items if item.audio.exists()},
                {item.audio: item.out_dir for item in third.items},
            )
            for _audio, out_dir, saved_audio, final in snapshots:
                state = json.loads((Path(out_dir) / "batch-item.json").read_text(encoding="utf-8"))
                self.assertEqual(saved_audio, state["identity"]["audio_path"])
                self.assertEqual(final, Path(state["outputs"]["final"]["path"]).read_text(encoding="utf-8"))

    def test_run_batch_excludes_nested_outputs_and_rejects_overlapping_roots(self):
        from zh_asr.batch import run_batch

        calls = []

        def fake_strict(audio_path, *, primary_engine, secondary_engine, out_dir, **_kwargs):
            calls.append(audio_path)
            derived = out_dir / "_derived"
            derived.mkdir(exist_ok=True)
            (derived / "fixture.wav").write_bytes(b"synthetic derived audio")
            return write_strict_bundle(
                audio_path=audio_path,
                primary_engine=primary_engine,
                primary_result={"engine": primary_engine, "text": "ok", "error": None},
                secondary_engine=secondary_engine,
                secondary_result={"engine": secondary_engine, "text": "ok", "error": None},
                out_dir=out_dir,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = input_dir / "results"
            input_dir.mkdir()
            audio = input_dir / "original.wav"
            audio.write_bytes(b"synthetic source audio")
            first = run_batch(input_dir, output_dir, mode="strict", strict_fn=fake_strict)
            self.assertEqual((1, 1, 0, 0), (first.total, first.processed, first.skipped, first.failed))
            self.assertTrue((first.items[0].out_dir / "_derived" / "fixture.wav").is_file())
            second = run_batch(input_dir, output_dir, mode="strict", strict_fn=fake_strict)
            self.assertEqual(
                (1, 0, 1, 0), (second.total, second.processed, second.skipped, second.failed),
                repr(second.items),
            )
            self.assertEqual([audio.resolve()], calls)
            for invalid_output in (input_dir, root):
                with self.subTest(output=invalid_output):
                    with self.assertRaises(ValueError):
                        run_batch(input_dir, invalid_output, mode="strict", strict_fn=fake_strict)
                    self.assertFalse((invalid_output / "failed.jsonl").exists())
            self.assertEqual([audio.resolve()], calls)

    def test_run_batch_reprocesses_when_model_config_changes(self):
        from zh_asr.batch import run_batch
        from zh_asr.config import load_model_config

        calls: list[int] = []

        def fake_strict(audio_path, *, primary_engine, secondary_engine, out_dir, **_kwargs):
            calls.append(1)
            return write_strict_bundle(
                audio_path=audio_path,
                primary_engine=primary_engine,
                primary_result={"engine": primary_engine, "text": "ok", "error": None},
                secondary_engine=secondary_engine,
                secondary_result={"engine": secondary_engine, "text": "ok", "error": None},
                out_dir=out_dir,
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "sample.wav").write_bytes(b"sample")
            config = load_model_config()
            changed_config = replace(config, default_engine="fun-asr-nano")

            first = run_batch(
                input_dir,
                output_dir,
                mode="strict",
                strict_fn=fake_strict,
                config=config,
            )
            second = run_batch(
                input_dir,
                output_dir,
                mode="strict",
                strict_fn=fake_strict,
                config=changed_config,
            )

        self.assertEqual((1, 0), (first.processed, first.skipped))
        self.assertEqual((1, 0), (second.processed, second.skipped))
        self.assertEqual(2, len(calls))

    def test_run_batch_does_not_skip_provisional_strict_bundle(self):
        from zh_asr.batch import run_batch

        calls: list[Path] = []

        def fake_strict(audio_path, *, primary_engine, secondary_engine, out_dir, **_kwargs):
            calls.append(audio_path)
            return write_strict_bundle(
                audio_path=audio_path,
                primary_engine=primary_engine,
                primary_result={
                    "engine": primary_engine,
                    "text": "",
                    "error": {"type": "RuntimeError", "message": "primary failed"},
                },
                secondary_engine=secondary_engine,
                secondary_result={"engine": secondary_engine, "text": "ok", "error": None},
                out_dir=out_dir,
                primary_error="RuntimeError: primary failed",
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "sample.wav").write_bytes(b"sample")

            first = run_batch(input_dir, output_dir, mode="strict", strict_fn=fake_strict)
            second = run_batch(input_dir, output_dir, mode="strict", strict_fn=fake_strict)
            self.assertIn("ok", (output_dir / "sample" / "sample.strict.md").read_text(encoding="utf-8"))

        self.assertEqual((0, 0, 1), (first.processed, first.skipped, first.failed))
        self.assertEqual((0, 0, 1), (second.processed, second.skipped, second.failed))
        self.assertEqual(2, len(calls))

    def test_run_batch_retries_failed_quick_artifacts_without_losing_adjacent_results(self):
        import wave
        from zh_asr.audio_outcome import load_objective_result, validate_objective_result
        from zh_asr.batch import _batch_item_identity, _write_batch_item_state, run_batch
        from zh_asr.config import load_model_config
        from zh_asr.result_writer import write_transcript_bundle

        calls, bundles = [], {}

        def fake_many(audio_paths, *, out_dirs, engine, **_kwargs):
            calls.append([audio.name for audio in audio_paths])
            results = []
            for audio, out_dir in zip(audio_paths, out_dirs):
                raw = {"engine": engine, "text": "good text" if audio.name == "a_good.wav" else ""}
                if audio.name == "b_bad.wav":
                    raw["error"] = {"type": "RuntimeError", "message": "synthetic decode failure"}
                bundle = write_transcript_bundle(audio, raw, out_dir, engine)
                bundles[audio.name] = bundle
                results.append(bundle)
            return results

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir, output_dir = root / "input", root / "output"
            input_dir.mkdir()
            names = ["a_good.wav", "b_bad.wav", "c_uncertain.wav"]
            for name in (names[0], names[2]):
                with wave.open(str(input_dir / name), "wb") as audio:
                    audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    audio.writeframes(b"\x01\x00" * 160)
            bad_audio = input_dir / names[1]
            bad_audio.write_bytes(b"synthetic corrupt audio")
            config = load_model_config()
            first = run_batch(input_dir, output_dir, mode="quick", engine="sensevoice",
                              config=config, transcribe_many_fn=fake_many)
            self.assertEqual((2, 0, 1), (first.processed, first.skipped, first.failed))
            self.assertEqual(["processed", "failed", "processed"], [item.status for item in first.items])
            failed_objective = load_objective_result(bundles[names[1]]["objective_result"])
            self.assertEqual([], validate_objective_result(failed_objective))
            self.assertEqual("corrupt", failed_objective["execution"]["status"])
            uncertain = load_objective_result(bundles[names[2]]["objective_result"])
            self.assertEqual(("completed", "indeterminate"),
                             (uncertain["execution"]["status"], uncertain["objective_outcome"]))
            preserved = {path: path.read_bytes() for name in (names[0], names[2])
                         for path in bundles[name].values() if isinstance(path, Path)}
            # Recreate the old version's valid, hash-bound failed-item cache.
            identity = _batch_item_identity(
                bad_audio.resolve(), mode="quick", quick_engine="sensevoice",
                strict_primary=config.strict_primary_engine,
                strict_secondary=config.strict_secondary_engine, config=config,
                device="cuda:0", cache_dir=None, caller_binding=None,
            )
            _write_batch_item_state(output_dir / "b_bad", identity, bundles[names[1]],
                                    failed_objective["objective_outcome"])
            second = run_batch(input_dir, output_dir, mode="quick", engine="sensevoice",
                               config=config, transcribe_many_fn=fake_many)
            self.assertEqual((0, 2, 1), (second.processed, second.skipped, second.failed))
            self.assertEqual([names, [names[1]]], calls)
            self.assertEqual(preserved, {path: path.read_bytes() for path in preserved})
            self.assertTrue(all(path.is_file() for path in bundles[names[1]].values() if isinstance(path, Path)))
            failures = [json.loads(line) for line in (output_dir / "failed.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([str(bad_audio.resolve())] * 2, [item["audio"] for item in failures])

    def test_run_batch_keeps_previous_failure_receipts(self):
        from zh_asr.batch import run_batch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            audio = input_dir / "sample.wav"
            audio.write_bytes(b"sample")

            def fail(_audio_path, **_kwargs):
                raise RuntimeError("first failure")

            run_batch(input_dir, output_dir, mode="strict", strict_fn=fail)
            before = (output_dir / "failed.jsonl").read_text(encoding="utf-8")

            def succeed(audio_path, *, primary_engine, secondary_engine, out_dir, **_kwargs):
                return write_strict_bundle(
                    audio_path=audio_path,
                    primary_engine=primary_engine,
                    primary_result={"engine": primary_engine, "text": "ok", "error": None},
                    secondary_engine=secondary_engine,
                    secondary_result={"engine": secondary_engine, "text": "ok", "error": None},
                    out_dir=out_dir,
                )

            run_batch(input_dir, output_dir, mode="strict", strict_fn=succeed, force=True)
            after = (output_dir / "failed.jsonl").read_text(encoding="utf-8")

        self.assertTrue(before.strip())
        self.assertEqual(before, after)

    def test_run_batch_force_reprocesses_completed_files(self):
        from zh_asr.batch import run_batch

        calls: list[Path] = []

        def fake_strict(audio_path, *, primary_engine, secondary_engine, device, out_dir, cache_dir, config):
            calls.append(audio_path)
            final = out_dir / f"{audio_path.stem}.strict.md"
            final.write_text("new", encoding="utf-8")
            return {"final": final}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            (input_dir / "done.wav").write_text("", encoding="utf-8")
            done_dir = output_dir / "done"
            done_dir.mkdir(parents=True)
            (done_dir / "done.strict.md").write_text("old", encoding="utf-8")

            summary = run_batch(
                input_dir=input_dir,
                out_dir=output_dir,
                mode="strict",
                device="cuda:0",
                cache_dir=root / "cache",
                force=True,
                strict_fn=fake_strict,
            )
            final_text = (output_dir / "done" / "done.strict.md").read_text(encoding="utf-8")

        self.assertEqual(summary.processed, 1)
        self.assertEqual(summary.skipped, 0)
        self.assertEqual([path.name for path in calls], ["done.wav"])
        self.assertEqual(final_text, "new")

    def test_run_batch_quick_uses_one_many_call_for_pending_files(self):
        from zh_asr.batch import run_batch

        calls: list[list[Path]] = []

        def fake_many(audio_paths, *, out_dirs, engine, device, cache_dir, config):
            calls.append(list(audio_paths))
            results = []
            for audio_path, out_dir in zip(audio_paths, out_dirs):
                out_dir.mkdir(parents=True, exist_ok=True)
                final = out_dir / f"{audio_path.stem}.{engine}.md"
                final.write_text("ok", encoding="utf-8")
                results.append({"markdown": final, "objective_outcome": "speech_transcribed"})
            return results

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            (input_dir / "one.wav").write_text("", encoding="utf-8")
            (input_dir / "two.mp3").write_text("", encoding="utf-8")

            summary = run_batch(
                input_dir=input_dir,
                out_dir=output_dir,
                mode="quick",
                engine="sensevoice",
                transcribe_many_fn=fake_many,
            )

        self.assertEqual([[path.name for path in call] for call in calls], [["one.wav", "two.mp3"]])
        self.assertEqual(summary.processed, 2)
        self.assertEqual(summary.failed, 0)


if __name__ == "__main__":
    unittest.main()

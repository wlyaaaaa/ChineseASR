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

        self.assertEqual((1, 0), (first.processed, first.skipped))
        self.assertEqual((1, 0), (second.processed, second.skipped))
        self.assertEqual(2, len(calls))

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

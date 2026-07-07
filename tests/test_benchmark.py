import json
import tempfile
import unittest
import wave
from pathlib import Path


def write_tiny_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)


class BenchmarkTests(unittest.TestCase):
    def test_build_benchmark_manifest_matches_audio_and_truth_by_stem_without_copying_audio(self):
        from zh_asr.benchmark import build_benchmark_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            truth_dir = root / "truth"
            manifest_dir = root / "out" / "_manifest"
            audio_dir.mkdir()
            truth_dir.mkdir()
            write_tiny_wav(audio_dir / "001.wav")
            (audio_dir / "002.mp3").write_bytes(b"fake")
            (truth_dir / "001.txt").write_text("开放时间早上九点。", encoding="utf-8")

            manifest_path = build_benchmark_manifest(audio_dir, truth_dir, manifest_dir)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            case_001 = next(case for case in manifest["cases"] if case["id"] == "001")
            case_002 = next(case for case in manifest["cases"] if case["id"] == "002")

            copied_audio = list(manifest_dir.rglob("*.wav")) + list(manifest_dir.rglob("*.mp3"))

        self.assertEqual(manifest["schema_version"], 2)
        self.assertIn("sha256", manifest["model_config"])
        self.assertIn("qwen3-asr-1.7b", manifest["model_config"]["selected_engines"])
        self.assertIn("sensevoice", manifest["model_config"]["selected_engines"])
        self.assertIn("argv", manifest["invocation"])
        self.assertEqual(len(manifest["cases"]), 2)
        self.assertTrue(case_001["available"])
        self.assertEqual(case_001["truth_text"], "开放时间早上九点。")
        self.assertNotEqual(case_001["audio_sha256"], "")
        self.assertGreater(case_001["audio_size_bytes"], 0)
        self.assertNotEqual(case_001["truth_sha256"], "")
        self.assertGreater(case_001["truth_size_bytes"], 0)
        self.assertFalse(case_002["available"])
        self.assertIn("Missing truth file", case_002["error"])
        self.assertNotEqual(case_002["audio_sha256"], "")
        self.assertEqual(case_002["truth_sha256"], "")
        self.assertEqual(case_002["truth_size_bytes"], 0)
        self.assertEqual(copied_audio, [])

    def test_run_benchmark_writes_benchmark_json_markdown_and_review(self):
        from zh_asr.benchmark import run_benchmark

        def fake_strict(audio_path, *, primary_engine, secondary_engine, device, out_dir, cache_dir, config):
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = audio_path.stem
            final_text = "开放时间早上九点。" if stem == "001" else ""
            audit = {
                "status": "consistent",
                "final_text": final_text,
                "primary_engine": primary_engine,
                "primary_text": final_text,
                "secondary_engine": secondary_engine,
                "secondary_text": final_text,
                "similarity": 1.0,
                "needs_review": False,
                "flags": [],
                "alternatives": [],
                "rationale": "fake",
            }
            audit_json = out_dir / f"{stem}.strict.audit.json"
            primary_json = out_dir / f"{stem}.{primary_engine}.raw.json"
            secondary_json = out_dir / f"{stem}.{secondary_engine}.raw.json"
            audit_json.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
            primary_json.write_text(json.dumps([{"text": final_text}], ensure_ascii=False), encoding="utf-8")
            secondary_json.write_text(json.dumps([{"text": final_text}], ensure_ascii=False), encoding="utf-8")
            return {
                "audit_json": audit_json,
                "primary_json": primary_json,
                "secondary_json": secondary_json,
                "timing": {"total_sec": 0.6, "primary_sec": 0.4, "secondary_sec": 0.2},
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "audio"
            truth_dir = root / "truth"
            out_dir = root / "benchmark"
            audio_dir.mkdir()
            truth_dir.mkdir()
            write_tiny_wav(audio_dir / "001.wav")
            write_tiny_wav(audio_dir / "missing.wav")
            (truth_dir / "001.txt").write_text("开放时间早上九点。", encoding="utf-8")

            summary = run_benchmark(audio_dir, truth_dir, out_dir, strict_fn=fake_strict)
            benchmark_json = json.loads((out_dir / "benchmark.json").read_text(encoding="utf-8"))
            benchmark_md = (out_dir / "benchmark.md").read_text(encoding="utf-8")
            review_md = (out_dir / "review.md").read_text(encoding="utf-8")

        self.assertEqual(summary.total, 2)
        self.assertEqual(summary.evaluated, 1)
        self.assertEqual(summary.skipped, 1)
        self.assertEqual(benchmark_json["schema_version"], 2)
        self.assertEqual(benchmark_json["summary"]["evaluated"], 1)
        self.assertIn("manifest", benchmark_json["benchmark"])
        self.assertEqual(benchmark_json["benchmark"]["audio_dir"], str(audio_dir.resolve()))
        self.assertEqual(benchmark_json["benchmark"]["truth_dir"], str(truth_dir.resolve()))
        scored_case = next(case for case in benchmark_json["cases"] if case["id"] == "001")
        self.assertNotEqual(scored_case["truth_sha256"], "")
        self.assertEqual(scored_case["models"]["primary"], "qwen3-asr-1.7b")
        self.assertEqual(scored_case["timing"]["primary_sec"], 0.4)
        self.assertTrue(scored_case["paths"]["primary_raw_json"].endswith(".raw.json"))
        self.assertIn("rule_hits", scored_case)
        self.assertIn("001", benchmark_md)
        self.assertIn("missing", review_md)
        self.assertIn("Missing truth file", review_md)


if __name__ == "__main__":
    unittest.main()

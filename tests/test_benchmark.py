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

        self.assertEqual(len(manifest["cases"]), 2)
        self.assertTrue(case_001["available"])
        self.assertEqual(case_001["truth_text"], "开放时间早上九点。")
        self.assertFalse(case_002["available"])
        self.assertIn("Missing truth file", case_002["error"])
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
            audit_json.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
            return {"audit_json": audit_json}

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
        self.assertEqual(benchmark_json["summary"]["evaluated"], 1)
        self.assertIn("001", benchmark_md)
        self.assertIn("missing", review_md)
        self.assertIn("Missing truth file", review_md)


if __name__ == "__main__":
    unittest.main()

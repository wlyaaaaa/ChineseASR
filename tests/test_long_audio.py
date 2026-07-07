import json
import tempfile
import unittest
import wave
from pathlib import Path

from zh_asr.long_audio import plan_chunks, run_long_transcription


class LongAudioTests(unittest.TestCase):
    def test_plan_chunks_uses_duration_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "long.wav"
            _write_wav(audio, seconds=7)

            chunks = plan_chunks(audio, chunk_sec=3, overlap_sec=1)

        self.assertEqual(3, len(chunks))
        self.assertEqual((0, 3000), (chunks[0].start_ms, chunks[0].end_ms))
        self.assertEqual((2000, 5000), (chunks[1].start_ms, chunks[1].end_ms))
        self.assertEqual((4000, 7000), (chunks[2].start_ms, chunks[2].end_ms))

    def test_run_long_transcription_resumes_completed_chunks_and_merges_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            out_dir = root / "out"
            _write_wav(audio, seconds=7)
            calls: list[str] = []

            def fake_strict(audio_path, **kwargs):
                calls.append(audio_path.stem)
                chunk_dir = kwargs["out_dir"]
                chunk_dir.mkdir(parents=True, exist_ok=True)
                final_path = chunk_dir / f"{audio_path.stem}.strict.md"
                audit_path = chunk_dir / f"{audio_path.stem}.strict.audit.md"
                audit_json_path = chunk_dir / f"{audio_path.stem}.strict.audit.json"
                final_path.write_text(f"# Chunk\n\n## Transcript\n\n文本 {audio_path.stem}\n", encoding="utf-8")
                audit_path.write_text(f"# Audit\n\n证据 {audio_path.stem}\n", encoding="utf-8")
                audit_json_path.write_text(
                    json.dumps(
                        {
                            "status": "ok",
                            "final_text": f"文本 {audio_path.stem}",
                            "flags": [],
                            "rule_hits": [],
                            "similarity": 1.0,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return {"final": final_path, "audit": audit_path, "audit_json": audit_json_path}

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
        self.assertIn("文本 chunk-000001", transcript)
        self.assertIn("证据 chunk-000002", audit)
        self.assertTrue(metrics_exists)

    def test_run_long_transcription_resets_stale_running_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            out_dir = root / "out"
            _write_wav(audio, seconds=4)
            calls: list[str] = []

            def fake_strict(audio_path, **kwargs):
                calls.append(audio_path.stem)
                chunk_dir = kwargs["out_dir"]
                chunk_dir.mkdir(parents=True, exist_ok=True)
                final_path = chunk_dir / f"{audio_path.stem}.strict.md"
                audit_path = chunk_dir / f"{audio_path.stem}.strict.audit.md"
                audit_json_path = chunk_dir / f"{audio_path.stem}.strict.audit.json"
                final_path.write_text(f"## Transcript\n\n文本 {audio_path.stem}\n", encoding="utf-8")
                audit_path.write_text(f"证据 {audio_path.stem}\n", encoding="utf-8")
                audit_json_path.write_text('{"final_text": "文本", "flags": [], "rule_hits": []}', encoding="utf-8")
                return {"final": final_path, "audit": audit_path, "audit_json": audit_json_path}

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

    def test_run_long_transcription_arbitrates_only_flagged_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "meeting.wav"
            out_dir = root / "out"
            _write_wav(audio, seconds=4)

            def fake_strict(audio_path, **kwargs):
                chunk_dir = kwargs["out_dir"]
                chunk_dir.mkdir(parents=True, exist_ok=True)
                final_path = chunk_dir / f"{audio_path.stem}.strict.md"
                audit_path = chunk_dir / f"{audio_path.stem}.strict.audit.md"
                audit_json_path = chunk_dir / f"{audio_path.stem}.strict.audit.json"
                flags = ["model_conflict"] if audio_path.stem == "chunk-000002" else []
                similarity = 0.5 if flags else 1.0
                final_path.write_text(f"## Transcript\n\n文本 {audio_path.stem}\n", encoding="utf-8")
                audit_path.write_text(f"证据 {audio_path.stem}\n", encoding="utf-8")
                audit_json_path.write_text(
                    json.dumps(
                        {
                            "primary_text": f"甲 {audio_path.stem}",
                            "secondary_text": f"乙 {audio_path.stem}",
                            "final_text": f"文本 {audio_path.stem}",
                            "similarity": similarity,
                            "flags": flags,
                            "rule_hits": [],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return {"final": final_path, "audit": audit_path, "audit_json": audit_json_path}

            arbiter = _FakeArbiter()

            run_long_transcription(audio, out_dir, chunk_sec=2, overlap_sec=0, strict_fn=fake_strict, arbiter=arbiter)
            metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(["chunk-000002"], arbiter.calls)
        self.assertEqual("LLM 仲裁文本", metrics["chunks"][1]["arbitration"]["final_text"])


def _write_wav(path: Path, seconds: int, sample_rate: int = 8000) -> None:
    frames = b"\x00\x00" * sample_rate * seconds
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)


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

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


class EvalPackTests(unittest.TestCase):
    def test_char_error_rate_ignores_punctuation_and_simplifies_text(self):
        from zh_asr.eval_pack import char_error_rate

        self.assertEqual(char_error_rate("開放時間早上九點", "开放时间：早上九点。"), 0.0)
        self.assertAlmostEqual(char_error_rate("开放时间", "开放世界"), 0.5)

    def test_generate_builtin_corpus_creates_manifest_truth_and_adversarial_audio(self):
        from zh_asr.eval_pack import generate_builtin_corpus

        def fake_tts_writer(text: str, path: Path, rate: int = 0) -> None:
            write_tiny_wav(path)

        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus"
            manifest_path = generate_builtin_corpus(corpus, include_tts=True, tts_writer=fake_tts_writer)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            case_ids = {case["id"] for case in manifest["cases"]}
            normal_case = next(case for case in manifest["cases"] if case["id"] == "tts-clean-001")
            silence_case = next(case for case in manifest["cases"] if case["id"] == "silence-001")
            normal_audio_exists = (corpus / normal_case["audio"]).exists()
            silence_audio_exists = (corpus / silence_case["audio"]).exists()
            normal_truth_exists = (corpus / normal_case["truth"]).exists()

        self.assertEqual(manifest["schema_version"], 2)
        self.assertIn("sha256", manifest["model_config"])
        self.assertIn("qwen3-asr-1.7b", manifest["model_config"]["selected_engines"])
        self.assertIn("sensevoice", manifest["model_config"]["selected_engines"])
        self.assertIn("argv", manifest["invocation"])
        self.assertIn("tts-clean-001", case_ids)
        self.assertIn("silence-001", case_ids)
        self.assertFalse(normal_case["expect_empty"])
        self.assertTrue(silence_case["expect_empty"])
        self.assertEqual(normal_case["truth_text"], "开放时间早上九点至下午五点。")
        self.assertEqual(silence_case["truth_text"], "")
        self.assertNotEqual(normal_case["audio_sha256"], "")
        self.assertGreater(normal_case["audio_size_bytes"], 0)
        self.assertNotEqual(normal_case["truth_sha256"], "")
        self.assertGreater(normal_case["truth_size_bytes"], 0)
        self.assertNotEqual(silence_case["audio_sha256"], "")
        self.assertNotEqual(silence_case["truth_sha256"], "")
        self.assertTrue(normal_audio_exists)
        self.assertTrue(silence_audio_exists)
        self.assertTrue(normal_truth_exists)

    def test_generate_builtin_corpus_without_tts_keeps_adversarial_cases_available(self):
        from zh_asr.eval_pack import generate_builtin_corpus

        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus"
            manifest_path = generate_builtin_corpus(corpus, include_tts=False)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertTrue(any(case["expect_empty"] for case in manifest["cases"]))
        self.assertFalse(any(case["kind"] == "tts" and case["available"] for case in manifest["cases"]))

    def test_run_evaluation_writes_metrics_benchmark_and_review(self):
        from zh_asr.eval_pack import generate_builtin_corpus, run_evaluation

        def fake_tts_writer(text: str, path: Path, rate: int = 0) -> None:
            write_tiny_wav(path)

        def fake_strict(audio_path, *, primary_engine, secondary_engine, device, out_dir, cache_dir, config):
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = audio_path.stem
            if stem == "silence-001":
                final_text = "谢谢观看"
                similarity = 0.2
                needs_review = False
                flags = []
            elif stem in {"white-noise-001", "tone-001"}:
                final_text = "[听不清]"
                similarity = 1.0
                needs_review = True
                flags = ["empty_transcript"]
            else:
                final_text = "开放时间早上九点至下午五点。"
                similarity = 1.0
                needs_review = False
                flags = []
            audit = {
                "status": "consistent",
                "final_text": final_text,
                "primary_engine": primary_engine,
                "primary_text": final_text,
                "secondary_engine": secondary_engine,
                "secondary_text": final_text,
                "similarity": similarity,
                "needs_review": needs_review,
                "flags": flags,
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
                "timing": {"total_sec": 1.25, "primary_sec": 0.75, "secondary_sec": 0.5},
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            out_dir = root / "runs"
            generate_builtin_corpus(corpus, include_tts=True, tts_writer=fake_tts_writer)

            summary = run_evaluation(
                corpus_dir=corpus,
                out_dir=out_dir,
                strict_fn=fake_strict,
            )
            metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
            benchmark = (out_dir / "benchmark.md").read_text(encoding="utf-8")
            review = (out_dir / "review.md").read_text(encoding="utf-8")

        self.assertGreaterEqual(summary.total, 2)
        self.assertEqual(metrics["schema_version"], 2)
        self.assertIn("runtime", metrics)
        self.assertEqual(metrics["runtime"]["device"], "cuda:0")
        self.assertIn("model_config", metrics)
        self.assertIn("qwen3-asr-1.7b", metrics["model_config"]["selected_engines"])
        self.assertIn("invocation", metrics)
        self.assertIn("elapsed_sec", metrics["summary"])
        self.assertIn("started_at", metrics["summary"])
        self.assertIn("finished_at", metrics["summary"])
        self.assertEqual(metrics["summary"]["hallucination_count"], 1)
        self.assertEqual(metrics["summary"]["false_confident_count"], 1)
        clean_case = next(case for case in metrics["cases"] if case["id"] == "tts-clean-001")
        self.assertEqual(clean_case["models"]["primary"], "qwen3-asr-1.7b")
        self.assertEqual(clean_case["models"]["secondary"], "sensevoice")
        self.assertEqual(clean_case["texts"]["primary"], "开放时间早上九点至下午五点。")
        self.assertEqual(clean_case["texts"]["secondary"], "开放时间早上九点至下午五点。")
        self.assertEqual(clean_case["texts"]["final"], "开放时间早上九点至下午五点。")
        self.assertEqual(clean_case["text_similarity"]["primary_secondary"], 1.0)
        self.assertEqual(clean_case["text_similarity"]["disagreement_score"], 0.0)
        self.assertEqual(clean_case["text_similarity"]["cer"], 0.0)
        self.assertEqual(clean_case["timing"]["total_sec"], 1.25)
        self.assertEqual(clean_case["timing"]["primary_sec"], 0.75)
        self.assertTrue(clean_case["paths"]["audit_json"].endswith(".strict.audit.json"))
        self.assertTrue(clean_case["paths"]["primary_raw_json"].endswith(".raw.json"))
        self.assertTrue(clean_case["paths"]["secondary_raw_json"].endswith(".raw.json"))
        self.assertEqual(clean_case["audit_status"], "consistent")
        self.assertEqual(clean_case["rule_hits"], [])
        silence_case = next(case for case in metrics["cases"] if case["id"] == "silence-001")
        silence_rule_ids = {hit["id"] for hit in silence_case["rule_hits"]}
        self.assertIn("empty_audio_hallucination", silence_rule_ids)
        self.assertIn("suspicious_stock_phrase", silence_rule_ids)
        self.assertIn("silence-001", review)
        self.assertIn("false_confident", review)
        self.assertIn("- Review items: `3`", review)
        self.assertIn("## P0 silence-001", review)
        self.assertIn("## P1 tone-001", review)
        self.assertLess(review.index("## P0 silence-001"), review.index("## P1 tone-001"))
        self.assertIn("- Action: `回听原音频，确认空音频/负样本是否被模型编出了文字。`", review)
        self.assertIn("- Audio: `", review)
        self.assertIn("- Audit JSON: `", review)
        self.assertIn("- Primary: `谢谢观看`", review)
        self.assertIn("- Secondary: `谢谢观看`", review)
        self.assertIn("- Metrics: `CER=n/a; similarity=0.200; empty_text_len=4`", review)
        self.assertIn("- Rule hits: `", review)
        self.assertIn("empty_audio_hallucination", review)
        self.assertIn("tts-clean-001", benchmark)

    def test_strict_fn_adapter_preserves_positional_audio_argument(self):
        from zh_asr.eval_pack import _call_strict_fn

        def fake_strict(path, *, primary_engine, secondary_engine, device, out_dir, cache_dir, config):
            return {"audio_name": path.name}

        result = _call_strict_fn(
            fake_strict,
            audio_path=Path("sample.wav"),
            primary_engine="primary",
            secondary_engine="secondary",
            device="cpu",
            out_dir=Path("out"),
            cache_dir=None,
            config=None,
            expect_empty=True,
        )

        self.assertEqual(result["audio_name"], "sample.wav")

    def test_review_queue_promotes_stock_phrase_hits_to_p0(self):
        from zh_asr.eval_pack import EvalCaseResult, EvalSummary, _write_review
        from zh_asr.risk_rules import RuleHit

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = EvalCaseResult(
                case_id="stock-phrase",
                category="benchmark",
                kind="wav",
                audio=root / "stock-phrase.wav",
                audio_sha256="audiohash",
                truth_sha256="truthhash",
                truth_text="今天下午开会。",
                final_text="谢谢观看。",
                cer=1.0,
                disagreement_score=0.0,
                audit_status="suspicious",
                primary_engine="sensevoice",
                primary_text="谢谢观看。",
                secondary_engine="qwen3-asr-1.7b",
                secondary_text="谢谢观看。",
                primary_secondary_similarity=1.0,
                timing_total_sec=1.0,
                timing_primary_sec=0.5,
                timing_secondary_sec=0.5,
                audit_json=root / "stock-phrase.strict.audit.json",
                primary_json=root / "stock-phrase.sensevoice.raw.json",
                secondary_json=root / "stock-phrase.qwen3-asr-1.7b.raw.json",
                empty_audio_text_len=0,
                risk_flags=("suspicious_stock_phrase",),
                rule_hits=(
                    RuleHit(
                        id="suspicious_stock_phrase",
                        severity="high",
                        message="stock phrase",
                        evidence="phrase=谢谢观看",
                    ),
                ),
                false_confident=False,
                simplified_only=True,
                needs_review=True,
            )
            review_path = root / "review.md"
            _write_review(
                review_path,
                EvalSummary(
                    corpus_dir=root,
                    out_dir=root,
                    total=1,
                    evaluated=1,
                    skipped=0,
                    hallucination_count=0,
                    false_confident_count=0,
                    cases=(item,),
                ),
            )
            review = review_path.read_text(encoding="utf-8")

        self.assertIn("## P0 stock-phrase", review)


if __name__ == "__main__":
    unittest.main()

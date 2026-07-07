import json
import tempfile
import unittest
from pathlib import Path

from zh_asr.arbitration import ArbitrationConfig, ArbitrationEvidence, NullArbiter, OllamaArbiter, load_arbitration_config, make_arbiter


class ArbitrationTests(unittest.TestCase):
    def test_default_config_is_disabled_and_uses_ollama_defaults(self):
        config = ArbitrationConfig.from_mapping({})

        self.assertFalse(config.enabled)
        self.assertEqual("ollama", config.provider)
        self.assertEqual("qwen-main-v1:latest", config.model)
        self.assertEqual("uncertain_only", config.mode)
        self.assertEqual(0, config.keep_alive)

    def test_null_arbiter_returns_none(self):
        evidence = ArbitrationEvidence(
            chunk_id="chunk-000001",
            time_range="00:00:00-00:01:00",
            primary_text="你好世界",
            secondary_text="你好，世界",
        )

        self.assertIsNone(NullArbiter().arbitrate(evidence))

    def test_ollama_arbiter_sends_structured_json_request_and_parses_decision(self):
        calls = []

        def fake_post(url, payload, timeout):
            calls.append((url, payload, timeout))
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "final_text": "你好世界",
                            "confidence": 0.91,
                            "decisions": [{"span": "你好世界", "chosen": "你好世界", "reason": "两路一致"}],
                            "unresolved": [],
                        },
                        ensure_ascii=False,
                    )
                }
            }

        config = ArbitrationConfig(enabled=True, model="qwen-main-v1:latest", keep_alive=0, temperature=0.1)
        arbiter = OllamaArbiter(config=config, post_json=fake_post)
        evidence = ArbitrationEvidence(
            chunk_id="chunk-000001",
            time_range="00:00:00-00:01:00",
            primary_text="你好世界",
            secondary_text="你好，世界",
            similarity=0.8,
            flags=["model_conflict"],
        )

        decision = arbiter.arbitrate(evidence)

        self.assertEqual("你好世界", decision.final_text)
        self.assertEqual(0.91, decision.confidence)
        self.assertEqual("http://127.0.0.1:11434/api/chat", calls[0][0])
        payload = calls[0][1]
        self.assertEqual("qwen-main-v1:latest", payload["model"])
        self.assertFalse(payload["stream"])
        self.assertEqual("json", payload["format"])
        self.assertEqual(0, payload["keep_alive"])
        self.assertEqual(0.1, payload["options"]["temperature"])

    def test_ollama_arbiter_returns_low_confidence_decision_on_invalid_json(self):
        def fake_post(url, payload, timeout):
            return {"message": {"content": "not json"}}

        config = ArbitrationConfig(enabled=True)
        arbiter = OllamaArbiter(config=config, post_json=fake_post)
        evidence = ArbitrationEvidence(chunk_id="chunk-000001", time_range="00:00:00-00:01:00")

        decision = arbiter.arbitrate(evidence)

        self.assertEqual("", decision.final_text)
        self.assertEqual(0.0, decision.confidence)
        self.assertTrue(decision.unresolved)

    def test_load_arbitration_config_from_model_yaml_and_make_arbiter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.yaml"
            path.write_text(
                """
llm_arbitration:
  enabled: true
  provider: ollama
  base_url: http://127.0.0.1:11434
  model: qwen-main-v1:latest
  keep_alive: 0
""",
                encoding="utf-8",
            )

            config = load_arbitration_config(path)
            arbiter = make_arbiter(config)

        self.assertTrue(config.enabled)
        self.assertIsInstance(arbiter, OllamaArbiter)


if __name__ == "__main__":
    unittest.main()

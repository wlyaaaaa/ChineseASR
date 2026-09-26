from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from zh_asr.cloud_review import (
    auto_cloud_status, compare_text, load_cloud_config, local_review_signals,
    pause_cloud, resume_cloud, select_model, stopping_error,
)


ROOT = Path(__file__).resolve().parents[1]


class CloudReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = load_cloud_config(ROOT / "configs" / "models.yaml")

    def test_config_routes_without_local_credit_gate(self) -> None:
        self.assertEqual("short", select_model(self.config, duration_sec=12))
        self.assertEqual("message", select_model(self.config, duration_sec=300))
        self.assertEqual("message", select_model(self.config, duration_sec=12,
                                                 hotwords={"田家庵": 4}))
        self.assertEqual("short", select_model(self.config, duration_sec=300,
                                               speaker=True))
        self.assertEqual("short", select_model(self.config, duration_sec=300,
                                               dialect=True))
        for item in self.config["models"].values():
            self.assertFalse(any(key in item for key in (
                "initial_remaining", "free_total", "free_expires_before",
                "reserve_per_chunk", "free_only_confirmed")))

    def test_local_evidence_and_parallel_difference(self) -> None:
        (self.root / "quality.review.json").write_text('{"needs_review":true}', encoding="utf-8")
        (self.root / "one.strict.audit.json").write_text(json.dumps({
            "final_text": "[疑似]给省高院写信", "primary_text": "给最高院写信",
            "secondary_text": "给省高院写信", "flags": ["engine_failure"]},
            ensure_ascii=False), encoding="utf-8")
        reasons, text = local_review_signals(self.root, evidence_status="provisional")
        self.assertTrue({"quality_needs_review", "engine_disagreement", "suspected_text",
                         "engine_failure", "provisional_evidence"} <= set(reasons))
        self.assertIn("省高院", text)
        self.assertTrue(compare_text("最高院", "省高院"))

    def test_long_chunk_audits_supply_local_text_and_disagreement(self) -> None:
        chunk = self.root / "chunks" / "chunk-000001"
        chunk.mkdir(parents=True)
        (chunk / "chunk-000001.strict.audit.json").write_text(json.dumps({
            "final_text": "[疑似]给最高院发了一个", "primary_text": "给最高院发了一个",
            "secondary_text": "给省高院发了一个", "needs_review": True},
            ensure_ascii=False), encoding="utf-8")
        reasons, local_text = local_review_signals(self.root)
        self.assertIn("engine_disagreement", reasons)
        self.assertIn("最高院", local_text)
        self.assertTrue(compare_text(local_text, "给省高院发了一个"))

    def test_provider_stop_persists_until_owner_or_new_configuration(self) -> None:
        path = self.root / "auto-cloud-state.json"
        self.assertEqual("running", auto_cloud_status(path, self.config)["status"])
        pause_cloud(path, self.config, reason="account_arrears",
                    provider_error="Arrearage", model="qwen-audio-3.1-asr-flash")
        self.assertEqual("account_arrears", auto_cloud_status(path, self.config)["reason"])
        changed = dict(self.config)
        changed["credit_cycle"] = "new-grant"
        self.assertEqual("running", auto_cloud_status(path, changed)["status"])
        rerouted = dict(self.config)
        rerouted["routes"] = dict(self.config["routes"], short="message")
        self.assertEqual("paused", auto_cloud_status(path, rerouted)["status"])
        new_model = dict(self.config)
        new_model["models"] = dict(self.config["models"], next={
            "id": "qwen-audio-next", "api": "http_base64", "endpoint":
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            "max_chunk_sec": 120})
        new_model["routes"] = dict(self.config["routes"], short="next")
        self.assertEqual("running", auto_cloud_status(path, new_model)["status"])
        resume_cloud(path, self.config)
        self.assertEqual("running", auto_cloud_status(path, self.config)["status"])
        path.write_text("{broken", encoding="utf-8")
        self.assertEqual("cloud_state_unreadable", auto_cloud_status(path, self.config)["reason"])

    def test_provider_error_categories_pause_only_as_requested(self) -> None:
        self.assertEqual("free_quota_exhausted", stopping_error("AllocationQuota.FreeTierOnly"))
        self.assertEqual("account_arrears", stopping_error("Arrearage"))
        self.assertEqual("balance_insufficient", stopping_error("InsufficientBalance"))
        self.assertEqual("access_denied", stopping_error("http_403"))
        self.assertEqual("model_unavailable", stopping_error("ModelNotFound"))
        self.assertIsNone(stopping_error("network_failure", "Network-Failure"))
        self.assertIsNone(stopping_error("http_429", "Rate-Limited"))
        self.assertIsNone(stopping_error("Throttling.AllocationQuota", "Rate-Limited"))
        self.assertEqual("free_quota_exhausted", stopping_error(
            "Throttling.AllocationQuota", "Rate-Limited", "Free allocated quota exceeded."))
        self.assertEqual("account_arrears", stopping_error("PostpaidBillOverdue", "Rate-Limited"))
        self.assertEqual("access_denied", stopping_error("CommodityNotPurchased", "Rate-Limited"))
        self.assertIsNone(stopping_error("http_500", "Provider-5xx"))


if __name__ == "__main__":
    unittest.main()

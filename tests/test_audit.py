import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class AuditTests(unittest.TestCase):
    def test_consistent_transcripts_keep_final_text_clean(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="sensevoice",
            primary_text="今天下午三点开会。",
            secondary_engine="paraformer",
            secondary_text="今天下午三点开会。",
        )

        self.assertEqual(report.status, "consistent")
        self.assertEqual(report.final_text, "今天下午三点开会。")
        self.assertNotIn("[疑似]", report.final_text)
        self.assertFalse(report.needs_review)
        self.assertEqual(report.evidence_status, "verified")
        self.assertTrue(
            all(item.execution_status == "succeeded" for item in report.engine_evidence)
        )

    def test_major_conflict_marks_final_text_and_preserves_alternative(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="sensevoice",
            primary_text="明天上午九点去医院。",
            secondary_engine="paraformer",
            secondary_text="明天上午九点去会议室。",
        )

        self.assertEqual(report.status, "conflict")
        self.assertTrue(report.needs_review)
        self.assertTrue(report.final_text.startswith("[疑似]"))
        self.assertIn("明天上午九点去医院。", report.final_text)
        self.assertIn("明天上午九点去会议室。", report.alternatives)

    def test_empty_transcripts_become_unclear(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="sensevoice",
            primary_text="",
            secondary_engine="paraformer",
            secondary_text="",
        )

        self.assertEqual(report.status, "unclear")
        self.assertEqual(report.final_text, "[听不清]")
        self.assertTrue(report.needs_review)

    def test_suspicious_stock_phrase_is_flagged(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="sensevoice",
            primary_text="谢谢观看。",
            secondary_engine="paraformer",
            secondary_text="",
        )

        self.assertIn("suspicious_stock_phrase", report.flags)
        self.assertTrue(report.needs_review)
        self.assertIn("suspicious_stock_phrase", {hit.id for hit in report.rule_hits})

    def test_repetition_rule_marks_final_text_suspicious(self):
        from zh_asr.audit import build_audit_report

        repeated = "今天下午开会今天下午开会今天下午开会"
        report = build_audit_report(
            primary_engine="sensevoice",
            primary_text=repeated,
            secondary_engine="paraformer",
            secondary_text=repeated,
        )

        self.assertEqual(report.status, "suspicious")
        self.assertTrue(report.final_text.startswith("[疑似]"))
        self.assertIn("abnormal_repetition", report.flags)
        self.assertIn("abnormal_repetition", {hit.id for hit in report.rule_hits})

    def test_expect_empty_rule_marks_substantive_text_as_hallucination(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="sensevoice",
            primary_text="开放时间早上九点。",
            secondary_engine="paraformer",
            secondary_text="开放时间早上九点。",
            expect_empty=True,
        )

        self.assertEqual(report.status, "suspicious")
        self.assertIn("empty_audio_hallucination", report.flags)
        self.assertEqual(next(hit for hit in report.rule_hits if hit.id == "empty_audio_hallucination").severity, "high")

    def test_short_semantic_difference_is_conflict(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="sensevoice",
            primary_text="开饭时间早上九点至下午五点。",
            secondary_engine="paraformer",
            secondary_text="开放时间早上九点至下午五点",
        )

        self.assertEqual(report.status, "conflict")
        self.assertTrue(report.needs_review)
        self.assertTrue(report.final_text.startswith("[疑似]"))
        self.assertIn("model_conflict", {hit.id for hit in report.rule_hits})

    def test_final_text_and_comparison_use_simplified_chinese(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="qwen3-asr-1.7b",
            primary_text="開放時間：早上九點至下午五點。",
            secondary_engine="sensevoice",
            secondary_text="开放时间早上九点至下午五点。",
        )

        self.assertEqual(report.status, "consistent")
        self.assertEqual(report.final_text, "开放时间：早上九点至下午五点。")
        self.assertEqual(report.primary_text, "开放时间：早上九点至下午五点。")
        self.assertEqual(report.secondary_text, "开放时间早上九点至下午五点。")
        self.assertEqual(report.similarity, 1.0)
        self.assertEqual(report.rule_hits, ())

    def test_report_records_roles_segment_disagreements_and_review_items_without_voting(self):
        from zh_asr.audit import build_audit_report
        from zh_asr.result_writer import TranscriptSegment

        primary_segments = (
            TranscriptSegment(0, "可以去一楼换票。", 1000, 2000, 0, "$[0]"),
            TranscriptSegment(1, "远程方式没法调。", 3000, 4200, 1, "$[1]"),
        )
        secondary_segments = (
            TranscriptSegment(0, "可以去一楼换票。", 1000, 2000, 0, "$[0]"),
            TranscriptSegment(1, "远程方式可以调。", 3000, 4200, 1, "$[1]"),
        )

        report = build_audit_report(
            primary_engine="fireredasr2-llm",
            primary_text="可以去一楼换票。\n远程方式没法调。",
            secondary_engine="qwen3-asr-1.7b",
            secondary_text="可以去一楼换票。\n远程方式可以调。",
            primary_role="lexical_primary",
            secondary_role="lexical_verifier",
            primary_segments=primary_segments,
            secondary_segments=secondary_segments,
            primary_raw_result_reference="call.firered.raw.json",
            secondary_raw_result_reference="call.qwen.raw.json",
        )

        self.assertEqual(
            report.selection_policy,
            "primary_preserving_no_majority_vote_no_semantic_rewrite",
        )
        self.assertEqual(report.engine_evidence[0].role, "lexical_primary")
        self.assertEqual(
            report.engine_evidence[0].raw_result_reference,
            "call.firered.raw.json",
        )
        self.assertEqual(report.engine_evidence[0].segments[1].text, "远程方式没法调。")
        self.assertEqual(len(report.disagreements), 1)
        disagreement = report.disagreements[0]
        self.assertEqual(disagreement.scope, "segment")
        self.assertEqual(disagreement.primary_segment_index, 1)
        self.assertEqual(disagreement.secondary_segment_index, 1)
        self.assertTrue(disagreement.review_required)
        self.assertEqual(disagreement.audio_start_ms, 3000)
        self.assertEqual(disagreement.audio_end_ms, 4200)
        self.assertEqual(report.review_items[0].disagreement_ids, (disagreement.id,))
        self.assertIn("远程方式没法调。", report.final_text)
        self.assertNotIn("远程方式可以调。", report.final_text)

    def test_engine_failure_marks_evidence_provisional_and_identifies_failed_primary(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="fireredasr2-llm",
            primary_text="",
            secondary_engine="qwen3-asr-1.7b",
            secondary_text="可以去一楼换票。",
            primary_error="RuntimeError: CUDA out of memory",
        )

        self.assertEqual(report.status, "engine_failure")
        self.assertEqual(report.evidence_status, "provisional")
        self.assertEqual(report.engine_evidence[0].engine, "fireredasr2-llm")
        self.assertEqual(report.engine_evidence[0].execution_status, "failed")
        self.assertIn("CUDA out of memory", report.engine_evidence[0].error)
        self.assertEqual(report.engine_evidence[1].execution_status, "succeeded")

    def test_both_engine_failures_make_evidence_unavailable(self):
        from zh_asr.audit import build_audit_report

        report = build_audit_report(
            primary_engine="fireredasr2-llm",
            primary_text="",
            secondary_engine="qwen3-asr-1.7b",
            secondary_text="",
            primary_error="RuntimeError: primary failed",
            secondary_error="RuntimeError: secondary failed",
        )

        self.assertEqual(report.status, "engine_failure")
        self.assertEqual(report.evidence_status, "unavailable")

    def test_verified_audit_is_unavailable_when_required_raw_is_missing(self):
        from zh_asr.audit import validate_strict_artifact_bundle

        with tempfile.TemporaryDirectory() as tmp:
            outputs = _write_strict_artifact_fixture(Path(tmp))
            Path(outputs["primary_json"]).unlink()
            status, failures = validate_strict_artifact_bundle(
                outputs,
                expected_primary_engine="fireredasr2-llm",
                expected_secondary_engine="qwen3-asr-1.7b",
            )

        self.assertEqual(status, "unavailable")
        self.assertIn("primary raw JSON", failures[0]["error"])

    def test_verified_audit_is_unavailable_when_raw_json_is_corrupt(self):
        from zh_asr.audit import validate_strict_artifact_bundle

        with tempfile.TemporaryDirectory() as tmp:
            outputs = _write_strict_artifact_fixture(Path(tmp))
            Path(outputs["secondary_json"]).write_text("{", encoding="utf-8")
            status, failures = validate_strict_artifact_bundle(
                outputs,
                expected_primary_engine="fireredasr2-llm",
                expected_secondary_engine="qwen3-asr-1.7b",
            )

        self.assertEqual(status, "unavailable")
        self.assertIn("secondary raw JSON", failures[0]["error"])

    def test_verified_audit_is_unavailable_when_raw_engine_identity_mismatches(self):
        from zh_asr.audit import validate_strict_artifact_bundle

        with tempfile.TemporaryDirectory() as tmp:
            outputs = _write_strict_artifact_fixture(Path(tmp))
            audit_path = Path(outputs["audit_json"])
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["engine_evidence"][0]["engine"] = "wrong-engine"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            _refresh_receipt(outputs, "audit_json")
            status, failures = validate_strict_artifact_bundle(
                outputs,
                expected_primary_engine="fireredasr2-llm",
                expected_secondary_engine="qwen3-asr-1.7b",
            )

        self.assertEqual(status, "unavailable")
        self.assertIn("identity", failures[0]["error"])

    def test_verifier_detects_replaced_critical_artifact_by_sha256(self):
        from zh_asr.audit import validate_strict_artifact_bundle
        from zh_asr.strict_writer import STRICT_BUNDLE_ARTIFACT_KEYS

        for artifact_key in STRICT_BUNDLE_ARTIFACT_KEYS:
            with self.subTest(artifact_key=artifact_key):
                with tempfile.TemporaryDirectory() as tmp:
                    outputs = _write_strict_artifact_fixture(Path(tmp))
                    artifact = Path(outputs[artifact_key])
                    artifact.write_bytes(artifact.read_bytes() + b"\n")

                    status, failures = validate_strict_artifact_bundle(
                        outputs,
                        expected_primary_engine="fireredasr2-llm",
                        expected_secondary_engine="qwen3-asr-1.7b",
                    )

                self.assertEqual(status, "unavailable")
                self.assertTrue(
                    any(
                        artifact_key in failure["error"]
                        and "SHA-256" in failure["error"]
                        for failure in failures
                    ),
                    failures,
                )

    def test_verifier_rejects_windows_rooted_or_drive_relative_receipt_paths(self):
        from zh_asr.audit import validate_strict_artifact_bundle
        from zh_asr.result_writer import canonical_json_sha256

        for hostile_path in (
            r"C:\outside\primary.raw.json",
            r"\Users\outside\primary.raw.json",
            r"C:outside\primary.raw.json",
        ):
            with self.subTest(hostile_path=hostile_path), tempfile.TemporaryDirectory() as tmp:
                outputs = _write_strict_artifact_fixture(Path(tmp))
                receipt_path = Path(outputs["receipt"])
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                receipt["artifacts"]["primary_json"]["path"] = hostile_path
                receipt["bundle_sha256"] = canonical_json_sha256(
                    {
                        "schema_version": receipt["schema_version"],
                        "artifacts": receipt["artifacts"],
                        "claims": receipt["claims"],
                    }
                )
                receipt_path.write_text(
                    json.dumps(receipt, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                status, failures = validate_strict_artifact_bundle(
                    outputs,
                    expected_primary_engine="fireredasr2-llm",
                    expected_secondary_engine="qwen3-asr-1.7b",
                )

            self.assertEqual(status, "unavailable")
            self.assertTrue(
                any(
                    "primary_json receipt path is not bundle-relative"
                    in failure["error"]
                    for failure in failures
                ),
                failures,
            )

    def test_verifier_rejects_bundle_symlink_escape(self):
        from zh_asr.audit import validate_strict_artifact_bundle
        from zh_asr.result_writer import canonical_json_sha256

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bundle"
            root.mkdir()
            outputs = _write_strict_artifact_fixture(root)
            outside = Path(tmp) / "outside"
            outside.mkdir()
            link = root / "escape"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (NotImplementedError, OSError) as exc:
                self.skipTest(f"directory symlink unavailable: {exc}")

            primary_path = Path(outputs["primary_json"])
            outside_primary = outside / primary_path.name
            primary_path.replace(outside_primary)
            outputs["primary_json"] = str(outside_primary)

            receipt_path = Path(outputs["receipt"])
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["artifacts"]["primary_json"]["path"] = (
                f"escape/{outside_primary.name}"
            )
            receipt["bundle_sha256"] = canonical_json_sha256(
                {
                    "schema_version": receipt["schema_version"],
                    "artifacts": receipt["artifacts"],
                    "claims": receipt["claims"],
                }
            )
            receipt_path.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            status, failures = validate_strict_artifact_bundle(
                outputs,
                expected_primary_engine="fireredasr2-llm",
                expected_secondary_engine="qwen3-asr-1.7b",
            )

        self.assertEqual(status, "unavailable")
        self.assertTrue(
            any(
                "primary_json receipt path is not bundle-relative"
                in failure["error"]
                for failure in failures
            ),
            failures,
        )

    def test_verifier_accepts_service_aliases_and_discovers_sidecar_paths(self):
        from zh_asr.audit import validate_strict_artifact_bundle

        with tempfile.TemporaryDirectory() as tmp:
            outputs = _write_strict_artifact_fixture(Path(tmp))
            service_outputs = {
                "final": outputs["final"],
                "audit": outputs["audit"],
                "audit_json": outputs["audit_json"],
                "primary_raw_json": outputs["primary_json"],
                "secondary_raw_json": outputs["secondary_json"],
            }

            status, failures = validate_strict_artifact_bundle(
                service_outputs,
                expected_primary_engine="fireredasr2-llm",
                expected_secondary_engine="qwen3-asr-1.7b",
            )

        self.assertEqual(status, "verified")
        self.assertEqual(failures, [])

    def test_verifier_keeps_legacy_absolute_references_valid_at_original_location(self):
        from zh_asr.audit import validate_strict_artifact_bundle

        with tempfile.TemporaryDirectory() as tmp:
            outputs = _write_strict_artifact_fixture(Path(tmp))
            receipt_path = Path(outputs["receipt"]).resolve()
            primary_path = Path(outputs["primary_json"]).resolve()
            secondary_path = Path(outputs["secondary_json"]).resolve()

            audit_json_path = Path(outputs["audit_json"])
            audit = json.loads(audit_json_path.read_text(encoding="utf-8"))
            audit["bundle_receipt_reference"] = str(receipt_path)
            audit["engine_evidence"][0]["raw_result_reference"] = str(primary_path)
            audit["engine_evidence"][1]["raw_result_reference"] = str(secondary_path)
            audit_json_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            review_json_path = Path(outputs["review_json"])
            review = json.loads(review_json_path.read_text(encoding="utf-8"))
            review["bundle_receipt_reference"] = str(receipt_path)
            review["engine_evidence"][0]["raw_result_reference"] = str(primary_path)
            review["engine_evidence"][1]["raw_result_reference"] = str(secondary_path)
            review_json_path.write_text(
                json.dumps(review, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            final_path = Path(outputs["final"])
            final_path.write_text(
                final_path.read_text(encoding="utf-8").replace(
                    Path(outputs["receipt"]).name,
                    str(receipt_path),
                ),
                encoding="utf-8",
            )
            audit_path = Path(outputs["audit"])
            audit_markdown = audit_path.read_text(encoding="utf-8")
            for old, new in (
                (Path(outputs["receipt"]).name, str(receipt_path)),
                (Path(outputs["primary_json"]).name, str(primary_path)),
                (Path(outputs["secondary_json"]).name, str(secondary_path)),
            ):
                audit_markdown = audit_markdown.replace(old, new)
            audit_path.write_text(audit_markdown, encoding="utf-8")

            for artifact_key in (
                "final",
                "audit",
                "audit_json",
                "review_json",
            ):
                _refresh_receipt(outputs, artifact_key)

            status, failures = validate_strict_artifact_bundle(
                outputs,
                expected_primary_engine="fireredasr2-llm",
                expected_secondary_engine="qwen3-asr-1.7b",
            )

        self.assertEqual(status, "verified")
        self.assertEqual(failures, [])

    def test_verifier_detects_raw_text_disagreeing_with_audit_engine_evidence(self):
        from zh_asr.audit import validate_strict_artifact_bundle

        with tempfile.TemporaryDirectory() as tmp:
            outputs = _write_strict_artifact_fixture(Path(tmp))
            primary_path = Path(outputs["primary_json"])
            primary = json.loads(primary_path.read_text(encoding="utf-8"))
            primary["text"] = "已经替换的原始文本。"
            primary_path.write_text(
                json.dumps(primary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _refresh_receipt(outputs, "primary_json", refresh_raw_claim=True)

            status, failures = validate_strict_artifact_bundle(
                outputs,
                expected_primary_engine="fireredasr2-llm",
                expected_secondary_engine="qwen3-asr-1.7b",
            )

        self.assertEqual(status, "unavailable")
        self.assertTrue(
            any(
                "lexical_primary raw text does not match strict audit engine_evidence text"
                in failure["error"]
                for failure in failures
            ),
            failures,
        )

    def test_verifier_detects_final_text_and_status_disagreeing_with_audit(self):
        from zh_asr.audit import validate_strict_artifact_bundle

        for field, old, new, expected_error in (
            (
                "text",
                "可以去一楼换票。",
                "已经替换的最终文本。",
                "final transcript text does not match strict audit final_text",
            ),
            (
                "status",
                "Status: `consistent`",
                "Status: `conflict`",
                "final transcript status does not match strict audit status",
            ),
        ):
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as tmp:
                    outputs = _write_strict_artifact_fixture(Path(tmp))
                    final_path = Path(outputs["final"])
                    final_path.write_text(
                        final_path.read_text(encoding="utf-8").replace(old, new, 1),
                        encoding="utf-8",
                    )
                    _refresh_receipt(
                        outputs,
                        "final",
                        final_text=new if field == "text" else None,
                    )

                    status, failures = validate_strict_artifact_bundle(
                        outputs,
                        expected_primary_engine="fireredasr2-llm",
                        expected_secondary_engine="qwen3-asr-1.7b",
                    )

                self.assertEqual(status, "unavailable")
                self.assertTrue(
                    any(expected_error in failure["error"] for failure in failures),
                    failures,
                )

    def test_verifier_detects_audit_markdown_and_review_disagreeing_with_audit_json(self):
        from zh_asr.audit import validate_strict_artifact_bundle

        for artifact_key, old, new, expected_error in (
            (
                "audit",
                "Status: `consistent`",
                "Status: `conflict`",
                "audit Markdown status does not match strict audit JSON status",
            ),
            (
                "review_json",
                '"status": "consistent"',
                '"status": "conflict"',
                "review JSON status does not match strict audit JSON status",
            ),
        ):
            with self.subTest(artifact_key=artifact_key):
                with tempfile.TemporaryDirectory() as tmp:
                    outputs = _write_strict_artifact_fixture(Path(tmp))
                    path = Path(outputs[artifact_key])
                    path.write_text(
                        path.read_text(encoding="utf-8").replace(old, new, 1),
                        encoding="utf-8",
                    )
                    _refresh_receipt(outputs, artifact_key)

                    status, failures = validate_strict_artifact_bundle(
                        outputs,
                        expected_primary_engine="fireredasr2-llm",
                        expected_secondary_engine="qwen3-asr-1.7b",
                    )

                self.assertEqual(status, "unavailable")
                self.assertTrue(
                    any(expected_error in failure["error"] for failure in failures),
                    failures,
                )


def _write_strict_artifact_fixture(root: Path) -> dict[str, str]:
    from zh_asr.strict_writer import write_strict_bundle

    audio = root / "call.wav"
    audio.write_bytes(b"fake wav")
    paths = write_strict_bundle(
        audio_path=audio,
        primary_engine="fireredasr2-llm",
        primary_result={
            "engine": "fireredasr2-llm",
            "text": "可以去一楼换票。",
            "error": None,
        },
        secondary_engine="qwen3-asr-1.7b",
        secondary_result={
            "engine": "qwen3-asr-1.7b",
            "text": "可以去一楼换票。",
            "error": None,
        },
        out_dir=root,
    )
    return {key: str(value) for key, value in paths.items()}


def _refresh_receipt(
    outputs: dict[str, str],
    artifact_key: str,
    *,
    refresh_raw_claim: bool = False,
    final_text: str | None = None,
) -> None:
    receipt_value = outputs.get("receipt")
    if not receipt_value:
        return
    receipt_path = Path(receipt_value)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifact_path = Path(outputs[artifact_key])
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    receipt["artifacts"][artifact_key]["sha256"] = digest
    receipt["artifacts"][artifact_key]["size_bytes"] = artifact_path.stat().st_size
    if refresh_raw_claim:
        for claim in receipt["claims"]["engine_evidence"]:
            if claim["raw_artifact"] == artifact_key:
                claim["raw_sha256"] = digest
    if final_text is not None:
        receipt["claims"]["final_text_sha256"] = hashlib.sha256(
            final_text.encode("utf-8")
        ).hexdigest()
    bundle_payload = {
        "schema_version": receipt["schema_version"],
        "artifacts": receipt["artifacts"],
        "claims": receipt["claims"],
    }
    receipt["bundle_sha256"] = hashlib.sha256(
        json.dumps(
            bundle_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()

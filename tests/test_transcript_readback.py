import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TranscriptReadbackTests(unittest.TestCase):
    def _write_bundle(
        self,
        root: Path,
        *,
        timestamped: bool = True,
        invalid_timing: bool = False,
        source_hash: str | None = None,
        quality_status: str | None = None,
        coverage_status: str | None = None,
        raw_payload=None,
    ) -> tuple[Path, str, Path, Path]:
        from zh_asr.audio_outcome import build_objective_result

        source = root / "source.mp3"
        source.write_bytes(b"synthetic source bytes")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        requested_hash = source_hash or digest
        raw = root / "sample.paraformer.raw.json"
        segment = {
            "start": -1 if invalid_timing else 100,
            "end": 0 if invalid_timing else 250,
            "spk": 0,
            "text": "第一句",
        }
        if timestamped:
            segment["timestamp"] = [[-1, 0]] if invalid_timing else [[100, 180], [180, 250]]
        if raw_payload is None:
            raw_payload = [{"sentence_info": [segment]}] if timestamped else [{"text": "全文转写"}]
        raw.write_text(json.dumps(raw_payload, ensure_ascii=False), encoding="utf-8")
        raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
        objective = build_objective_result(
            audio_path=source,
            mode="quick",
            engines=["paraformer"],
            primary_text="第一句" if timestamped else "全文转写",
            primary_result=raw_payload,
            raw_artifacts=[
                {
                    "schema": "media.raw-artifact-ref.v1",
                    "role": "lexical_primary",
                    "engine": "paraformer",
                    "path": raw.name,
                    "size_bytes": raw.stat().st_size,
                    "sha256": raw_hash,
                }
            ],
        )
        if quality_status is not None:
            objective["quality"]["status"] = quality_status
        if coverage_status is not None:
            objective["coverage"] = {
                "status": coverage_status,
                "start_ms": 0,
                "end_ms": None,
                "intervals_ms": [],
                "excluded_ranges_ms": [],
                "overlap_ms": 0,
                "gap_ms": 250,
                "complete": False,
            }
        objective_path = root / "sample.paraformer.objective-result.json"
        objective_path.write_text(
            json.dumps(objective, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        snapshot = root / "jobs.json"
        snapshot.write_text(
            json.dumps(
                {
                    "schema": "agents.test-jobs.v1",
                    "jobs": [
                        {
                            "job_id": "job-1",
                            "status": "succeeded",
                            "request": {
                                "audio_sha256": requested_hash,
                                "mode": "quick",
                                "engine": "paraformer",
                                "audio": str(root / "missing-source-that-must-not-be-read.mp3"),
                            },
                            "outputs": {
                                "raw_json": str(raw),
                                "objective_result": str(objective_path),
                            },
                            "evidence_status": "not_applicable",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return snapshot, digest, raw, objective_path

    def test_readback_returns_verified_hash_and_timestamped_segments(self):
        from zh_asr.transcript_readback import read_transcript_readback

        with tempfile.TemporaryDirectory() as tmp:
            snapshot, source_hash, raw, _objective = self._write_bundle(Path(tmp))
            payload = read_transcript_readback(source_hash, snapshot)
            expected_raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["source_audio_sha256"], source_hash)
        self.assertEqual(payload["segments"][0]["start_ms"], 100)
        self.assertEqual(payload["segments"][0]["end_ms"], 250)
        self.assertEqual(payload["segments"][0]["text"], "第一句")
        self.assertEqual(payload["segments"][0]["speaker"], 0)
        self.assertEqual(payload["segments"][0]["timestamp_granularity"], "sentence_with_subspans")
        self.assertEqual(
            payload["artifact"]["raw_json"]["sha256"],
            expected_raw_hash,
        )
        self.assertFalse(payload["lookup_scope"]["original_audio_read"])
        self.assertFalse(payload["lookup_scope"]["model_run"])

    def test_wrong_source_hash_returns_explicit_gap(self):
        from zh_asr.transcript_readback import read_transcript_readback

        with tempfile.TemporaryDirectory() as tmp:
            snapshot, _source_hash, _raw, _objective = self._write_bundle(Path(tmp))
            payload = read_transcript_readback("0" * 64, snapshot)

        self.assertEqual(payload["status"], "not_found")
        self.assertEqual(payload["gap"]["code"], "source_hash_not_in_jobs_snapshot")
        self.assertEqual(payload["segments"], [])

    def test_changed_raw_artifact_fails_hash_validation(self):
        from zh_asr.transcript_readback import read_transcript_readback

        with tempfile.TemporaryDirectory() as tmp:
            snapshot, source_hash, raw, _objective = self._write_bundle(Path(tmp))
            raw.write_text('{"sentence_info": [{"start": 100, "end": 250, "text": "被改写"}]}', encoding="utf-8")
            payload = read_transcript_readback(source_hash, snapshot)

        self.assertEqual(payload["status"], "gap")
        self.assertEqual(payload["gap"]["code"], "no_valid_transcript_artifact")
        self.assertEqual(payload["segments"], [])
        self.assertIn(
            "raw_artifact_sha256_mismatch",
            {item["reason"] for item in payload["lookup_scope"]["rejections"]},
        )

    def test_missing_timestamps_returns_gap_without_guessing(self):
        from zh_asr.transcript_readback import read_transcript_readback

        with tempfile.TemporaryDirectory() as tmp:
            snapshot, source_hash, _raw, _objective = self._write_bundle(Path(tmp), timestamped=False)
            payload = read_transcript_readback(source_hash, snapshot)

        self.assertEqual(payload["status"], "gap")
        self.assertEqual(payload["gap"]["code"], "timestamps_unavailable")
        self.assertEqual(payload["segments"], [])
        self.assertEqual(payload["artifact"]["engine"], "paraformer")

    def test_invalid_timestamps_return_gap_without_normalizing_a_guess(self):
        from zh_asr.transcript_readback import read_transcript_readback

        with tempfile.TemporaryDirectory() as tmp:
            snapshot, source_hash, _raw, _objective = self._write_bundle(
                Path(tmp), invalid_timing=True
            )
            payload = read_transcript_readback(source_hash, snapshot)

        self.assertEqual(payload["status"], "gap")
        self.assertEqual(payload["gap"]["code"], "timestamps_unavailable")
        self.assertEqual(payload["segments"], [])

    def test_partial_coverage_and_low_confidence_are_preserved(self):
        from zh_asr.transcript_readback import read_transcript_readback

        with tempfile.TemporaryDirectory() as tmp:
            snapshot, source_hash, _raw, _objective = self._write_bundle(
                Path(tmp),
                quality_status="low_confidence",
                coverage_status="partial",
            )
            payload = read_transcript_readback(source_hash, snapshot)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["quality"]["status"], "low_confidence")
        self.assertEqual(payload["coverage"]["status"], "partial")
        self.assertFalse(payload["coverage"]["complete"])
        self.assertEqual(payload["coverage"]["gap_ms"], 250)

    def test_duplicate_hash_prefers_quality_and_coverage_over_job_freshness(self):
        from zh_asr.audio_outcome import build_objective_result
        from zh_asr.transcript_readback import read_transcript_readback

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot, source_hash, _raw, _objective = self._write_bundle(
                root,
                quality_status="low_confidence",
                coverage_status="partial",
            )
            second = root / "second"
            second.mkdir()
            raw = second / "second.paraformer.raw.json"
            raw_payload = [
                {
                    "sentence_info": [
                        {"start": 300, "end": 420, "spk": 1, "text": "第二句"}
                    ]
                }
            ]
            raw.write_text(json.dumps(raw_payload, ensure_ascii=False), encoding="utf-8")
            raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
            source = root / "source.mp3"
            objective = build_objective_result(
                audio_path=source,
                mode="quick",
                engines=["paraformer"],
                primary_text="第二句",
                primary_result=raw_payload,
                raw_artifacts=[
                    {
                        "schema": "media.raw-artifact-ref.v1",
                        "role": "lexical_primary",
                        "engine": "paraformer",
                        "path": raw.name,
                        "size_bytes": raw.stat().st_size,
                        "sha256": raw_hash,
                    }
                ],
            )
            objective["coverage"] = {
                "status": "complete",
                "start_ms": 0,
                "end_ms": 420,
                "intervals_ms": [],
                "excluded_ranges_ms": [],
                "overlap_ms": 0,
                "gap_ms": 0,
                "complete": True,
            }
            objective["quality"]["status"] = "sufficient"
            objective_path = second / "second.paraformer.objective-result.json"
            objective_path.write_text(
                json.dumps(objective, ensure_ascii=False), encoding="utf-8"
            )
            jobs = json.loads(snapshot.read_text(encoding="utf-8"))
            jobs["jobs"][0]["job_id"] = "job-new"
            jobs["jobs"].append(
                {
                    "job_id": "job-old",
                    "status": "succeeded",
                    "request": {
                        "audio_sha256": source_hash,
                        "mode": "quick",
                        "engine": "paraformer",
                        "audio": str(root / "another-missing-source.mp3"),
                    },
                    "outputs": {
                        "raw_json": str(raw),
                        "objective_result": str(objective_path),
                    },
                    "evidence_status": "verified",
                }
            )
            snapshot.write_text(json.dumps(jobs, ensure_ascii=False), encoding="utf-8")
            payload = read_transcript_readback(source_hash, snapshot)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["artifact"]["job_id"], "job-old")
        self.assertEqual(payload["quality"]["status"], "sufficient")
        self.assertTrue(payload["coverage"]["complete"])
        self.assertEqual(payload["segments"][0]["text"], "第二句")

    def test_objective_source_hash_mismatch_is_rejected(self):
        from zh_asr.transcript_readback import read_transcript_readback

        with tempfile.TemporaryDirectory() as tmp:
            snapshot, source_hash, _raw, objective_path = self._write_bundle(Path(tmp))
            objective = json.loads(objective_path.read_text(encoding="utf-8"))
            objective["audio"]["raw_sha256"] = "1" * 64
            objective_path.write_text(json.dumps(objective, ensure_ascii=False), encoding="utf-8")
            payload = read_transcript_readback(source_hash, snapshot)

        self.assertEqual(payload["status"], "gap")
        self.assertIn(
            "objective_source_hash_mismatch",
            {item["reason"] for item in payload["lookup_scope"]["rejections"]},
        )

    def test_cli_exposes_json_readback_without_model_execution(self):
        from zh_asr.__main__ import main

        with tempfile.TemporaryDirectory() as tmp:
            snapshot, source_hash, _raw, _objective = self._write_bundle(Path(tmp))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                returncode = main(
                    [
                        "transcript-readback",
                        "--audio-sha256",
                        source_hash,
                        "--jobs-snapshot",
                        str(snapshot),
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(returncode, 0)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["segments"][0]["start_ms"], 100)

    def test_artifact_identity_describes_the_bytes_actually_parsed(self):
        from zh_asr.result_writer import extract_segments
        from zh_asr.transcript_readback import read_transcript_readback

        with tempfile.TemporaryDirectory() as tmp:
            snapshot, source_hash, raw, objective = self._write_bundle(Path(tmp))
            before = {path: path.read_bytes() for path in (raw, objective)}

            def replace_artifacts_after_parse(value):
                for path, contents in before.items():
                    path.write_bytes(contents + b"\n ")
                return extract_segments(value)

            with mock.patch("zh_asr.transcript_readback.extract_segments", side_effect=replace_artifacts_after_parse):
                payload = read_transcript_readback(source_hash, snapshot)

        self.assertEqual(payload["status"], "ok")
        for key, path in (("raw_json", raw), ("objective_result", objective)):
            self.assertEqual(payload["artifact"][key]["sha256"], hashlib.sha256(before[path]).hexdigest())
            self.assertEqual(payload["artifact"][key]["size_bytes"], len(before[path]))

    def test_failed_execution_cannot_supply_successful_locations(self):
        from zh_asr.transcript_readback import read_transcript_readback

        with tempfile.TemporaryDirectory() as tmp:
            snapshot, source_hash, _raw, objective = self._write_bundle(Path(tmp))
            content = json.loads(objective.read_bytes())
            content["execution"]["status"] = "failed"
            objective.write_text(json.dumps(content), encoding="utf-8")
            payload = read_transcript_readback(source_hash, snapshot)

        self.assertEqual(payload["status"], "gap")
        self.assertEqual(payload["segments"], [])
        self.assertEqual(payload["lookup_scope"]["rejections"][0]["reason"], "objective_execution_not_completed")

    def test_pending_jobs_are_returned_without_truncating_rejection_count(self):
        from zh_asr.transcript_readback import read_transcript_readback

        with tempfile.TemporaryDirectory() as tmp:
            snapshot, source_hash, _raw, _objective = self._write_bundle(Path(tmp))
            job = json.loads(snapshot.read_bytes())["jobs"][0]
            snapshot.write_text(json.dumps({"jobs": [
                {**job, "job_id": f"pending-{i}", "status": "running"} for i in range(40)
            ]}), encoding="utf-8")
            payload = read_transcript_readback(source_hash, snapshot)

        self.assertEqual(payload["gap"]["code"], "job_execution_pending")
        self.assertEqual(payload["lookup_scope"]["pending_job_count"], 40)
        self.assertEqual(payload["lookup_scope"]["rejected_candidates"], 40)
        self.assertEqual(len(payload["lookup_scope"]["rejections"]), 32)
        self.assertEqual(payload["lookup_scope"]["pending_jobs"][0]["job_id"], "pending-0")

    def test_generic_segments_do_not_claim_sentence_precision(self):
        from zh_asr.transcript_readback import read_transcript_readback

        with tempfile.TemporaryDirectory() as tmp:
            snapshot, source_hash, _raw, _objective = self._write_bundle(Path(tmp), raw_payload={
                "segments": [{"start": 100, "end": 250, "text": "第一句"}]
            })
            payload = read_transcript_readback(source_hash, snapshot)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["segments"][0]["timestamp_granularity"], "segment")

    def test_contradictory_complete_flag_is_not_preferred_over_partial(self):
        from zh_asr.transcript_readback import read_transcript_readback

        with tempfile.TemporaryDirectory() as tmp:
            jobs = []
            for index, status in enumerate(("complete", "partial")):
                root = Path(tmp) / str(index)
                root.mkdir()
                snapshot, source_hash, _raw, _objective = self._write_bundle(root, coverage_status=status)
                job = json.loads(snapshot.read_bytes())["jobs"][0]
                jobs.append({**job, "job_id": status})
            snapshot.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
            payload = read_transcript_readback(source_hash, snapshot)

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["artifact"]["job_id"], "partial")


if __name__ == "__main__":
    unittest.main()

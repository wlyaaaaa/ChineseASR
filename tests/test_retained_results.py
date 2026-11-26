import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from zh_asr.live_snapshot import covered_ms, dictation_history, write_snapshot
from zh_asr.retained_results import read_latest_complete_results


class RetainedResultsTests(unittest.TestCase):
    def test_dictation_history_keeps_retained_samples_after_log_rotation(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "runtime.log"
            rotated = log.with_suffix(".log.1")
            self.assertIsNone(dictation_history(log))
            rotated.write_text("2026-11-25 10:05:06,000 INFO dictation model operation=transcribe seconds=1.3\n")
            log.write_text("2026-11-25 11:05:07,000 INFO dictation model operation=prepare seconds=2.4\n")
            old_only = dictation_history(log)
            self.assertEqual((old_only["samples"], old_only["median_seconds"]), (1, 1.3))
            with log.open("a") as stream:
                stream.write("2026-11-25 12:05:08,000 INFO dictation model operation=transcribe seconds=2.7\n")
            both = dictation_history(log)
            self.assertEqual((both["samples"], both["median_seconds"]), (2, 2.0))
            self.assertEqual(both["through"], "2026-11-26T03:05:08+08:00")

    def test_batch_uses_latest_complete_once_per_original_with_real_timezones(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, seen = [], []
            rows = [("audio", "2026-11-16T00:00:00Z", True),
                    ("audio", "2026-11-16T09:00:00+08:00", True),
                    ("audio", "2026-11-16T02:00:00Z", False),
                    ("image", "2026-11-15T23:00:00Z", True), ("undated", None, True)]
            for index, (identity, at, complete) in enumerate(rows):
                path = Path(folder) / f"{index}.json"
                path.write_text(json.dumps(dict(source_id=identity, at=at, complete=complete)))
                paths.append(path)
            def normalize(path, value):
                seen.append(path)
                return value
            selected = read_latest_complete_results(paths + paths, normalize)
            self.assertEqual(len(seen), len(paths))
            self.assertEqual(selected["audio"]["at"], rows[1][1])
            self.assertEqual(set(selected), {"audio", "image"})
            self.assertEqual(read_latest_complete_results(paths, normalize, source_ids={"missing"}), {})
            paths[0].write_text("invalid json")
            with self.assertRaises(ValueError):
                read_latest_complete_results(paths, normalize)

    def test_retained_jobs_timestamp_changes_when_the_count_stays_the_same(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = root / "outputs/api/jobs.json"
            jobs.parent.mkdir(parents=True)
            public, owner, sources = (root / f"{name}.json" for name in ("public", "owner", "sources"))
            with patch("zh_asr.live_snapshot.subprocess.check_output", return_value=b""):
                for day in (24, 25):
                    at = datetime(2026, 11, day, tzinfo=timezone.utc)
                    jobs.write_text(json.dumps(dict(updated_at=at.timestamp(), jobs=[{}] * 200)))
                    result = write_snapshot(root, root, public, owner, sources)
                    self.assertEqual(result["metrics"]["api_retained_jobs"], 200)
                    self.assertEqual(result["times"]["api_retained_jobs"], at.isoformat())

    def test_duration_excludes_overlap_and_gaps_and_unchanged_snapshot_keeps_time(self):
        self.assertEqual(covered_ms([dict(start_ms=a, end_ms=b) for a, b in [(0, 10), (5, 20), (30, 40)]]), 30)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, owner, sources = [root / (name + ".json") for name in ("public", "owner", "sources")]
            metrics = dict.fromkeys(("complete_audio", "local_complete", "cloud_complete", "overlap", "local_audio",
                                     "local_receipts", "local_complete_receipts", "local_latest", "receipt_states"), 1)
            sample = (metrics, [], {"code_lines": 2}, {"metrics": {}, "artifacts": []}, {})
            times = [datetime(2026, 11, 16, hour, tzinfo=timezone.utc) for hour in (1, 2)]
            with patch("zh_asr.live_snapshot.collect", side_effect=lambda *args: json.loads(json.dumps(sample))), patch("zh_asr.live_snapshot.datetime") as clock:
                clock.now.side_effect = times
                first = write_snapshot(root, root, output, owner, sources)
                self.assertNotIn(b"\r\n", output.read_bytes())
                self.assertNotIn(b"\r\n", owner.read_bytes())
                second = write_snapshot(root, root, output, owner, sources)
                self.assertEqual([first.pop("checked_at"), second.pop("checked_at")], [at.isoformat() for at in times])
                self.assertEqual(first, second)

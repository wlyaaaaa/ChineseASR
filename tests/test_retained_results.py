import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from zh_asr.live_snapshot import covered_ms, write_snapshot
from zh_asr.retained_results import read_latest_complete_results


class RetainedResultsTests(unittest.TestCase):
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

    def test_duration_excludes_overlap_and_gaps_and_unchanged_snapshot_keeps_time(self):
        self.assertEqual(covered_ms([dict(start_ms=a, end_ms=b) for a, b in [(0, 10), (5, 20), (30, 40)]]), 30)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output, owner, sources = [root / (name + ".json") for name in ("public", "owner", "sources")]
            metrics = dict.fromkeys(("complete_audio", "local_complete", "cloud_complete", "overlap", "local_audio",
                                     "local_receipts", "local_complete_receipts", "local_latest", "receipt_states"), 1)
            sample = (metrics, [], {"code_lines": 2}, {"metrics": {}, "artifacts": []}, {})
            with patch("zh_asr.live_snapshot.collect", side_effect=lambda *args: json.loads(json.dumps(sample))):
                first = write_snapshot(root, root, output, owner, sources)
                self.assertNotIn(b"\r\n", output.read_bytes())
                self.assertNotIn(b"\r\n", owner.read_bytes())
                self.assertEqual(first, write_snapshot(root, root, output, owner, sources))

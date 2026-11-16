"""One-pass metadata lookup shared by retained ASR and OCR inventories."""
from datetime import datetime
import json
from pathlib import Path


def read_latest_complete_results(paths, normalize, *, source_ids=None):
    """Read each JSON once; normalize(path, payload) supplies source_id/at/complete.

    Return {source_id: latest complete row}; an incomplete newer attempt cannot
    hide an older complete result. Missing sources are absent, never empty text.
    Parse/read errors propagate so a partial scan cannot masquerade as a total.
    This checks retained metadata, not transcription/OCR accuracy.
    """
    wanted = None if source_ids is None else set(source_ids)
    latest, order = {}, {}
    for path in sorted(set(map(Path, paths))):
        row = normalize(path, json.loads(path.read_text(encoding="utf-8-sig")))
        if not row or not row.get("complete") or not row.get("source_id") or not row.get("at"):
            continue
        identity = row["source_id"]
        if wanted is not None and identity not in wanted:
            continue
        instant = datetime.fromisoformat(row["at"].replace("Z", "+00:00"))
        if instant.tzinfo is None:
            raise ValueError("result timestamp requires timezone")
        rank = instant.timestamp(), str(path)
        if identity not in order or rank > order[identity]:
            latest[identity], order[identity] = row, rank
    return latest

"""Inspect, pause or resume automatic cloud review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zh_asr.cloud_review import (CloudReviewError, auto_cloud_status,
                                 load_cloud_config, pause_cloud_manually,
                                 public_cloud_status, resume_cloud)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("status", "pause", "resume", "public-status"))
    parser.add_argument("--reason", default="本人手动暂停",
                        help="Reason shown when automatic review is manually paused")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "models.yaml")
    parser.add_argument("--state", type=Path,
                        default=ROOT / "outputs" / "cloud-jobs" / "auto-cloud-state.json")
    args = parser.parse_args(argv)
    try:
        config = load_cloud_config(args.config)
        if args.action == "pause":
            pause_cloud_manually(args.state, config, reason=args.reason)
        elif args.action == "resume":
            resume_cloud(args.state, config)
        status = (public_cloud_status(args.state, config) if args.action == "public-status"
                  else auto_cloud_status(args.state, config))
        print(json.dumps(status, ensure_ascii=False))
        return 0
    except CloudReviewError as exc:
        print(json.dumps({"status": "blocked", "error_code": exc.code}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

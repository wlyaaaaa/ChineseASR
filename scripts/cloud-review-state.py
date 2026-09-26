"""Inspect or resume the provider-error pause for automatic cloud review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zh_asr.cloud_review import (CloudReviewError, auto_cloud_status,
                                 load_cloud_config, resume_cloud)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("status", "resume"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "models.yaml")
    parser.add_argument("--state", type=Path,
                        default=ROOT / "outputs" / "cloud-jobs" / "auto-cloud-state.json")
    args = parser.parse_args(argv)
    try:
        config = load_cloud_config(args.config)
        if args.action == "resume":
            resume_cloud(args.state, config)
        status = auto_cloud_status(args.state, config)
        print(json.dumps(status, ensure_ascii=False))
        return 0
    except CloudReviewError as exc:
        print(json.dumps({"status": "blocked", "error_code": exc.code}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

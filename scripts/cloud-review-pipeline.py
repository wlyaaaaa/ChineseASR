"""Unprivileged cloud preparation and result projection; never reads an API key."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zh_asr.cloud_review import (CloudReviewError, finalize_cloud_result,
                                 prepare_cloud_request)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--provider", type=Path)
    parser.add_argument("--broker-error", default="")
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            value = prepare_cloud_request(args.intent, args.root,
                args.config or ROOT / "configs" / "models.yaml")
            code = 0 if value["status"] == "ready" else 2
        else:
            value = finalize_cloud_result(args.intent, args.root,
                provider_path=args.provider, broker_error=args.broker_error,
                config_path=args.config or ROOT / "configs" / "models.yaml")
            code = 0
        print(json.dumps(value, ensure_ascii=False, default=str))
        return code
    except CloudReviewError as exc:
        print(json.dumps({"status": "blocked", "error_code": exc.code}))
        return 3
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_code": "cloud_pipeline_error",
                          "error_type": type(exc).__name__}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

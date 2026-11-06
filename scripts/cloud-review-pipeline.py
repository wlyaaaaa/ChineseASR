"""Cloud preparation, direct transmission and result projection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zh_asr.cloud_review import (CloudReviewError, finalize_cloud_result,
                                 prepare_cloud_request)
from zh_asr import file_job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check-access", "prepare", "finalize", "transmit"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--intent", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--provider", type=Path)
    parser.add_argument("--broker-error", default="")
    parser.add_argument("--resume-job-id", default="")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--personal-data-guard", action="store_true", help="compatibility flag; access is always checked")
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args(argv)
    if args.action != "check-access" and (args.root is None or args.intent is None):
        parser.error("--root and --intent are required")
    try:
        if args.action in {"check-access", "prepare"} and not file_job._access_open():
            raise CloudReviewError("personal_access_protection_required")
        if args.action == "check-access":
            value, code = {"status": "pass", "cloud_upload_performed": False}, 0
        elif args.action == "prepare":
            value = prepare_cloud_request(args.intent, args.root,
                args.config or ROOT / "configs" / "models.yaml", resume_job_id=args.resume_job_id)
            code = 0 if value["status"] == "ready" else 2
        elif args.action == "transmit":
            from zh_asr.cloud_file_transport import transmit_cloud_chunks
            provider = transmit_cloud_chunks(args.request, args.intent, args.root,
                progress_path=args.progress)
            value = {"schema": "zh_asr.chunk-transmission.v1", "secret_returned": False,
                     **{key: provider.get(key) for key in ("status", "error_code", "new_requests", "reused_requests")}}
            code = 0
        else:
            value = finalize_cloud_result(args.intent, args.root,
                provider_path=args.provider, broker_error=args.broker_error,
                config_path=args.config or ROOT / "configs" / "models.yaml")
            code = 0
        print(json.dumps(value, ensure_ascii=False, default=str))
        return code
    except CloudReviewError as exc:
        print(json.dumps({"status": "blocked", "error_code": exc.code,
                          "cloud_upload_performed": False}))
        return 3
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_code": "cloud_pipeline_error",
                          "error_type": type(exc).__name__}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())

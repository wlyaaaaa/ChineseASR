"""Compatibility launcher, without credentials or business imports.

The registered target now runs Password Center's vendor_api_worker.py directly.
This entry forwards one prepared request; it never scans a shared job directory.
"""
import argparse
from pathlib import Path
import subprocess


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-path", required=True)
    args = parser.parse_args(argv)
    entry = Path("C:/ProgramData/PCConfig/AuthorityHost/tools/Invoke-PasswordCenterVendor.ps1")
    if not entry.is_file():
        return 20
    return subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-File", str(entry),
        "-Vendor", "qwen", "-RequestPath", str(Path(args.request_path).resolve()), "-Json"],
        shell=False, timeout=14460, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode


if __name__ == "__main__":
    raise SystemExit(main())

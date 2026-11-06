"""Fixed one-file worker; its service may outlive the submitting scheduled task."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zh_asr.file_job import file_job

if __name__ == "__main__":
    file_job(Path(sys.argv[1]), worker=True, root=ROOT)

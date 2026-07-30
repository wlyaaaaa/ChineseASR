import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QWEN_REVISION = "a04930dbe5419bfee073f7cade734f572689a3a8"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class QwenIdentityTests(unittest.TestCase):
    def test_qwen_dependency_and_model_runtime_are_exactly_pinned(self):
        import yaml

        requirement = (PROJECT_ROOT / "requirements-qwen.txt").read_text(
            encoding="utf-8"
        )
        config = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "models.yaml").read_text(encoding="utf-8")
        )
        options = config["engines"]["qwen3-asr-1.7b"]["options"]

        self.assertEqual("qwen-asr==0.0.6\n", requirement)
        self.assertEqual(QWEN_REVISION, options["model_revision"])
        self.assertEqual("qwen-asr", options["runtime_distribution"])
        self.assertEqual("0.0.6", options["runtime_version"])

    def test_model_receipt_rejects_revision_drift(self):
        from zh_asr.qwen_identity import (
            RequiredModelFile,
            verify_model_receipt,
            write_model_receipt,
        )

        payload = b"pinned model artifact"
        required = (
            RequiredModelFile(
                path="model.bin",
                bytes=len(payload),
                sha256=_sha256(payload),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "model.bin").write_bytes(payload)
            write_model_receipt(
                model_dir,
                repository="Qwen/Test",
                revision="revision-a",
                required_files=required,
            )

            with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
                verify_model_receipt(
                    model_dir,
                    repository="Qwen/Test",
                    revision="revision-b",
                    required_files=required,
                )

    def test_model_receipt_rejects_missing_size_and_sha256_drift(self):
        from zh_asr.qwen_identity import (
            RequiredModelFile,
            verify_model_receipt,
            write_model_receipt,
        )

        payload = b"pinned model artifact"
        required = (
            RequiredModelFile(
                path="model.bin",
                bytes=len(payload),
                sha256=_sha256(payload),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            artifact = model_dir / "model.bin"
            artifact.write_bytes(payload)

            with self.assertRaisesRegex(RuntimeError, "receipt is missing"):
                verify_model_receipt(
                    model_dir,
                    repository="Qwen/Test",
                    revision="revision-a",
                    required_files=required,
                )

            write_model_receipt(
                model_dir,
                repository="Qwen/Test",
                revision="revision-a",
                required_files=required,
            )
            artifact.write_bytes(payload + b"-tampered")
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                verify_model_receipt(
                    model_dir,
                    repository="Qwen/Test",
                    revision="revision-a",
                    required_files=required,
                )

            artifact.write_bytes(b"x" * len(payload))
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                verify_model_receipt(
                    model_dir,
                    repository="Qwen/Test",
                    revision="revision-a",
                    required_files=required,
                )

    def test_model_receipt_rejects_noncanonical_or_duplicate_file_list(self):
        from zh_asr.qwen_identity import (
            RequiredModelFile,
            verify_model_receipt,
            write_model_receipt,
        )

        payload = b"pinned model artifact"
        required = (
            RequiredModelFile(
                path="model.bin",
                bytes=len(payload),
                sha256=_sha256(payload),
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "model.bin").write_bytes(payload)
            receipt_path = write_model_receipt(
                model_dir,
                repository="Qwen/Test",
                revision="revision-a",
                required_files=required,
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["files"].append(dict(receipt["files"][0]))
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "duplicate path"):
                verify_model_receipt(
                    model_dir,
                    repository="Qwen/Test",
                    revision="revision-a",
                    required_files=required,
                )

    def test_qwen_runtime_identity_reports_verified_artifact_and_runtime(self):
        from zh_asr.config import EngineSpec
        from zh_asr.qwen_identity import (
            RequiredModelFile,
            qwen_runtime_identity,
            write_model_receipt,
        )

        payload = b"small test stand-in"
        required = (
            RequiredModelFile(
                path="model.bin",
                bytes=len(payload),
                sha256=_sha256(payload),
            ),
        )
        spec = EngineSpec(
            name="qwen3-asr-1.7b",
            adapter="qwen-asr",
            role="primary",
            model="Qwen/Qwen3-ASR-1.7B",
            language="Chinese",
            options={
                "model_revision": QWEN_REVISION,
                "runtime_distribution": "qwen-asr",
                "runtime_version": "0.0.6",
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            model_dir = cache_dir / "Qwen" / "Qwen3-ASR-1.7B"
            model_dir.mkdir(parents=True)
            (model_dir / "model.bin").write_bytes(payload)
            write_model_receipt(
                model_dir,
                repository=spec.model,
                revision=QWEN_REVISION,
                required_files=required,
            )

            with (
                patch("zh_asr.qwen_identity.QWEN_MODEL_FILES", required),
                patch(
                    "zh_asr.qwen_identity.importlib_metadata.version",
                    return_value="0.0.6",
                ),
            ):
                identity = qwen_runtime_identity(spec, cache_dir, {})

        self.assertEqual("verified", identity["model_receipt_status"])
        self.assertEqual(QWEN_REVISION, identity["model_revision"])
        self.assertEqual("qwen-asr", identity["runtime_distribution"])
        self.assertEqual("0.0.6", identity["runtime_version"])
        self.assertRegex(identity["model_receipt_sha256"], r"^[0-9a-f]{64}$")

    def test_qwen_runtime_identity_rejects_runtime_version_drift(self):
        from zh_asr.config import EngineSpec
        from zh_asr.qwen_identity import qwen_runtime_identity

        spec = EngineSpec(
            name="qwen3-asr-1.7b",
            adapter="qwen-asr",
            role="primary",
            model="Qwen/Qwen3-ASR-1.7B",
            language="Chinese",
            options={
                "model_revision": QWEN_REVISION,
                "runtime_distribution": "qwen-asr",
                "runtime_version": "0.0.6",
            },
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "zh_asr.qwen_identity.importlib_metadata.version",
                return_value="0.0.7",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "runtime version mismatch"):
                qwen_runtime_identity(spec, Path(tmp), {})

    def test_download_script_pins_revision_and_supports_receipt_only_migration(self):
        script = (PROJECT_ROOT / "scripts" / "download-models.ps1").read_text(
            encoding="utf-8"
        )

        self.assertIn(QWEN_REVISION, script)
        self.assertIn("revision=os.environ['ZH_ASR_QWEN_REVISION']", script)
        self.assertIn("[switch]$ReceiptOnly", script)
        self.assertIn("-m', 'zh_asr.qwen_identity'", script)
        self.assertIn("'write-receipt'", script)
        self.assertIn("MODEL_RECEIPT.json", script)


if __name__ == "__main__":
    unittest.main()

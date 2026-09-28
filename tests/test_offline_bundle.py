from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "offline_bundle.py"
spec = importlib.util.spec_from_file_location("offline_bundle", SCRIPT)
offline = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(offline)


class OfflineBundleTests(unittest.TestCase):
    def test_configured_models_are_copied_without_cache_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runtime"
            model_root = root / "models"
            for relative in ("modelscope/Qwen/ASR", "modelscope/iic/Sense",
                             "modelscope/iic/VAD", "aligner/Aligner"):
                path = model_root / relative
                path.mkdir(parents=True)
                (path / "weights.bin").write_bytes(relative.encode())
                (path / ".cache").mkdir()
                (path / ".cache" / "download.tmp").write_bytes(b"rebuildable")
            config = {"defaults": {"engine": "sense"},
                      "strict": {"primary_engine": "qwen", "secondary_engine": "sense"},
                      "profiles": {"high_quality": {"primary_engine": "qwen",
                                                     "secondary_engine": "sense"}},
                      "aliases": {"vad": "iic/VAD"},
                      "engines": {"qwen": {"adapter": "qwen-asr", "model": "Qwen/ASR"},
                                  "sense": {"adapter": "funasr", "model": "iic/Sense",
                                            "vad_model": "vad"},
                                  "optional": {"adapter": "funasr", "model": "iic/Missing"},
                                  "whisper": {"adapter": "whisper", "model": "openai/whisper"}},
                      "alignment": {"model_dir": "models/aligner/Aligner"}}
            config_path = root / "models.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            bundle = Path(tmp) / "bundle"
            result = offline.copy_models(root, config_path, bundle)
            self.assertEqual(4, len(result["models"]))
            self.assertFalse(list((bundle / "models").rglob("download.tmp")))
            self.assertEqual(offline.compare_models(root, config_path, bundle)[
                "model_files_verified"], 4)
            for platform in ("windows", "linux"):
                path = bundle / "wheelhouse" / platform
                path.mkdir(parents=True)
                (path / f"package-1.0-{platform}.whl").write_bytes(b"fixture")
                lock = bundle / "manifests" / f"{platform}-requirements.txt"
                lock.parent.mkdir(parents=True, exist_ok=True)
                lock.write_text("package==1.0\n", encoding="utf-8")
            offline.seal(bundle)
            self.assertEqual(offline.verify(bundle)["files"], 10)
            restored_root = Path(tmp) / "restored"
            self.assertEqual(offline.restore_models(restored_root, bundle)[
                "model_files_restored"], 4)
            self.assertEqual(offline.restore_models(restored_root, bundle)[
                "model_files_restored"], 0)
            (bundle / "models" / "modelscope" / "Qwen" / "ASR" / "weights.bin").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                offline.verify(bundle)

    def test_restore_pinned_firered_source_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            (bundle / "manifests").mkdir(parents=True)
            (bundle / "model-selection.json").write_text(json.dumps({
                "firered_source_commit": "fixed-commit", "models": []}), encoding="utf-8")
            (bundle / "manifests" / "model-source-sha256.json").write_text(json.dumps({
                "schema": "zh_asr.offline_model_source_hashes.v1", "files": []}), encoding="utf-8")
            for platform in ("windows", "linux"):
                (bundle / "manifests" / f"{platform}-requirements.txt").write_text(
                    "package==1.0\n", encoding="utf-8")
                wheelhouse = bundle / "wheelhouse" / platform
                wheelhouse.mkdir(parents=True)
                (wheelhouse / f"package-1.0-{platform}.whl").write_bytes(b"fixture")
            source = bundle / "source" / "FireRedASR2S.tar"
            source.parent.mkdir()
            with tarfile.open(source, "w") as archive:
                payload = b"source = 'pinned'\n"
                info = tarfile.TarInfo("fireredasr2s/__init__.py")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            offline.seal(bundle)
            restored = Path(tmp) / "restored"
            self.assertTrue(offline.restore_models(restored, bundle)["firered_source_restored"])
            self.assertEqual((restored / "models" / "firered" / "FireRedASR2S" /
                              "fireredasr2s" / "__init__.py").read_bytes(), payload)

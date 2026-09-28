"""Build and verify a single offline copy of runnable model assets.

The bundle is runtime data, never a Git artifact.  All model selection comes
from models.yaml; download caches, examples and source-control metadata are
excluded.  This module does not fetch packages or alter the installed models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Any
import uuid

SKIP_DIRS = {".cache", ".git", "__pycache__", "example", "examples", "fig"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
SCHEMA = "zh_asr.offline_bundle.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def within(root: Path, path: Path) -> Path:
    root = root.resolve(strict=True)
    path = path.resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes model root: {path}")
    return path


def selected_models(runtime_root: Path, config_path: Path) -> tuple[list[dict[str, str]], Path | None]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_root = runtime_root / "models"
    required_engines = {config["defaults"]["engine"],
                        config["strict"]["primary_engine"],
                        config["strict"]["secondary_engine"]}
    for profile in config.get("profiles", {}).values():
        required_engines.update((profile["primary_engine"], profile["secondary_engine"]))
    chosen: dict[str, str] = {}
    firered_source: Path | None = None

    def add(relative: str, purpose: str, *, required: bool = True) -> None:
        source = model_root / relative
        if not source.is_dir():
            if required:
                raise FileNotFoundError(f"configured model absent: {source} ({purpose})")
            return
        within(model_root, source)
        chosen[relative.replace("\\", "/")] = purpose

    engines = config["engines"]
    for name, spec in engines.items():
        adapter = spec.get("adapter")
        if adapter == "firered-worker":
            relative = spec["options"]["model_dir"].replace("\\", "/")
            if not relative.startswith("models/"):
                raise ValueError("FireRed model directory must stay under models")
            add(relative[len("models/"):], f"engine:{name}")
            source = spec["options"]["source_dir"].replace("\\", "/")
            if not source.startswith("models/"):
                raise ValueError("FireRed source directory must stay under models")
            firered_source = within(model_root, runtime_root / source)
        elif adapter in {"funasr", "qwen-asr"}:
            # Preserve every installed, runnable configured engine, including
            # explicit Paraformer/Fun-ASR routes.  An absent optional engine is
            # not presented as recoverable.
            repo = spec["model"]
            add(f"modelscope/{repo}", f"engine:{name}",
                required=name in required_engines)
            for field in ("vad_model", "punc_model", "spk_model"):
                alias = spec.get(field)
                if alias:
                    add(f"modelscope/{config['aliases'].get(alias, alias)}",
                        f"support:{name}:{field}", required=name in required_engines)
    speaker_alias = config.get("speaker_verification", {}).get("model_alias")
    if speaker_alias:
        add(f"modelscope/{config['aliases'].get(speaker_alias, speaker_alias)}",
            "speaker_verification")
    aligner = config["alignment"]["model_dir"].replace("\\", "/")
    if not aligner.startswith("models/"):
        raise ValueError("aligner directory must stay under models")
    add(aligner[len("models/"):], "alignment")
    selected = [{"path": key, "purpose": value} for key, value in sorted(chosen.items())]
    return selected, firered_source


def payload_files(source: Path):
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"model payload contains a symlink: {path}")
        if path.is_file() and path.suffix.lower() not in SKIP_SUFFIXES:
            yield path


def inventory(runtime_root: Path, config_path: Path) -> dict[str, Any]:
    chosen, firered_source = selected_models(runtime_root, config_path)
    rows = []
    for entry in chosen:
        source = runtime_root / "models" / entry["path"]
        files = list(payload_files(source))
        rows.append({**entry, "files": len(files), "bytes": sum(p.stat().st_size for p in files)})
    return {"models": rows, "model_bytes": sum(row["bytes"] for row in rows),
            "firered_source": str(firered_source) if firered_source else ""}


def copy_models(runtime_root: Path, config_path: Path, bundle: Path) -> dict[str, Any]:
    import yaml

    chosen, firered_source = selected_models(runtime_root, config_path)
    target_root = bundle / "models"
    if target_root.exists():
        raise FileExistsError(f"model staging already exists: {target_root}")
    target_root.mkdir(parents=True)
    copied = 0
    total = 0
    for entry in chosen:
        source = runtime_root / "models" / entry["path"]
        for path in payload_files(source):
            destination = target_root / entry["path"] / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied += 1
            total += path.stat().st_size
    source_commit = ""
    if firered_source:
        source_commit = subprocess.check_output(
            ["git", "-C", str(firered_source), "rev-parse", "HEAD"], text=True).strip()
        configured = yaml.safe_load(config_path.read_text(encoding="utf-8"))[
            "engines"]["fireredasr2-llm"]["options"]["source_revision"]
        if source_commit != configured:
            raise ValueError(f"FireRed source revision mismatch: {source_commit} != {configured}")
        archive = bundle / "source" / "FireRedASR2S.tar"
        archive.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(firered_source), "archive", "--format=tar",
                        f"--output={archive}", "HEAD"], check=True)
    summary = {"models": chosen, "model_files": copied, "model_bytes": total,
               "firered_source_commit": source_commit}
    (bundle / "model-selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def compare_models(runtime_root: Path, config_path: Path, bundle: Path) -> dict[str, Any]:
    chosen, _ = selected_models(runtime_root, config_path)
    records = []
    for entry in chosen:
        source = runtime_root / "models" / entry["path"]
        destination = bundle / "models" / entry["path"]
        expected = {p.relative_to(source).as_posix(): p for p in payload_files(source)}
        actual = {p.relative_to(destination).as_posix(): p for p in payload_files(destination)}
        if expected.keys() != actual.keys():
            raise ValueError(f"model file set differs: {entry['path']}")
        for relative, original in sorted(expected.items()):
            saved = actual[relative]
            original_hash = sha256(original)
            if original.stat().st_size != saved.stat().st_size or original_hash != sha256(saved):
                raise ValueError(f"model copy differs: {entry['path']}/{relative}")
            records.append({"path": f"{entry['path']}/{relative}",
                            "bytes": original.stat().st_size, "sha256": original_hash})
    receipt = {"schema": "zh_asr.offline_model_source_hashes.v1", "files": records}
    path = bundle / "manifests" / "model-source-sha256.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"model_files_verified": len(records),
            "model_bytes_verified": sum(row["bytes"] for row in records)}


def seal(bundle: Path) -> dict[str, Any]:
    selection = bundle / "model-selection.json"
    if not selection.is_file():
        raise FileNotFoundError("model selection missing")
    selected = json.loads(selection.read_text(encoding="utf-8"))
    if selected.get("firered_source_commit") and not (
        bundle / "source" / "FireRedASR2S.tar").is_file():
        raise FileNotFoundError("pinned FireRed source archive missing")
    for relative in ("manifests/windows-requirements.txt",
                     "manifests/linux-requirements.txt",
                     "manifests/model-source-sha256.json"):
        if not (bundle / relative).is_file():
            raise FileNotFoundError(relative)
    for platform in ("windows", "linux"):
        if not list((bundle / "wheelhouse" / platform).glob("*.whl")):
            raise FileNotFoundError(f"{platform} wheelhouse empty")
    files = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file() and path.name != "bundle.json":
            files.append({"path": path.relative_to(bundle).as_posix(),
                          "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {"schema": SCHEMA, "selection": selected,
                "files": files}
    (bundle / "bundle.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"files": len(files), "bytes": sum(row["bytes"] for row in files)}


def verify(bundle: Path) -> dict[str, Any]:
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or not manifest.get("files"):
        raise ValueError("invalid bundle manifest")
    seen = set()
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or row["path"] in seen:
            raise ValueError(f"invalid manifest path: {row['path']}")
        seen.add(row["path"])
        path = bundle / relative
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"bundle file mismatch: {relative}")
    actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*")
              if path.is_file() and path.name != "bundle.json"}
    if actual != seen:
        raise ValueError("bundle file set differs from manifest")
    return {"files": len(seen), "bytes": sum(row["bytes"] for row in manifest["files"])}


def restore_models(runtime_root: Path, bundle: Path) -> dict[str, Any]:
    verify(bundle)
    selection = json.loads((bundle / "model-selection.json").read_text(encoding="utf-8"))
    receipt = json.loads((bundle / "manifests" / "model-source-sha256.json").read_text(
        encoding="utf-8"))
    if receipt.get("schema") != "zh_asr.offline_model_source_hashes.v1":
        raise ValueError("model source receipt invalid")
    restored = 0
    for row in receipt["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid model path: {relative}")
        source = bundle / "models" / relative
        destination = runtime_root / "models" / relative
        if destination.exists():
            if destination.stat().st_size != row["bytes"] or sha256(destination) != row["sha256"]:
                raise ValueError(f"installed model differs; refusing overwrite: {relative}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if destination.stat().st_size != row["bytes"] or sha256(destination) != row["sha256"]:
            raise ValueError(f"restored model hash differs: {relative}")
        restored += 1
    source_restored = False
    if selection.get("firered_source_commit"):
        source_dir = runtime_root / "models" / "firered" / "FireRedASR2S"
        if (source_dir / ".git").exists():
            installed_commit = subprocess.check_output(
                ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
            if installed_commit != selection["firered_source_commit"]:
                raise ValueError("installed FireRed source commit differs; refusing overwrite")
        if not source_dir.exists():
            staging = source_dir.with_name(source_dir.name + ".partial-" + uuid.uuid4().hex)
            staging.mkdir(parents=True)
            with tarfile.open(bundle / "source" / "FireRedASR2S.tar", "r") as archive:
                for member in archive:
                    relative = Path(member.name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(f"invalid FireRed source archive path: {member.name}")
                    if not member.isfile():
                        continue  # Source imports do not need example symlinks.
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
            staging.rename(source_dir)
            source_restored = True
    return {"model_files_restored": restored, "model_files_checked": len(receipt["files"]),
            "firered_source_restored": source_restored}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("inventory", "copy-models", "compare-models",
                                           "seal", "verify", "restore-models"))
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    try:
        if args.action in {"inventory", "copy-models", "compare-models"}:
            if not args.runtime_root or not args.config:
                parser.error("model actions need --runtime-root and --config")
            if args.action == "inventory":
                result = inventory(args.runtime_root, args.config)
            elif args.action == "copy-models":
                result = copy_models(args.runtime_root, args.config, args.bundle)
            else:
                result = compare_models(args.runtime_root, args.config, args.bundle)
        else:
            if not args.bundle:
                parser.error("seal/verify need --bundle")
            if args.action == "restore-models" and not args.runtime_root:
                parser.error("restore-models needs --runtime-root")
            if args.action == "seal":
                result = seal(args.bundle)
            elif args.action == "restore-models":
                result = restore_models(args.runtime_root, args.bundle)
            else:
                result = verify(args.bundle)
        print(json.dumps({"status": "ok", **result}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

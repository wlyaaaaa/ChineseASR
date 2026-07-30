from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable

from zh_asr.config import EngineSpec


MODEL_RECEIPT_SCHEMA = "zh_asr.model_receipt.v1"
MODEL_RECEIPT_FIELDS = frozenset(
    {"schema", "repository", "revision", "created_utc", "files"}
)
MODEL_FILE_RECORD_FIELDS = frozenset({"path", "bytes", "sha256"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

QWEN_MODEL_REPOSITORY = "Qwen/Qwen3-ASR-1.7B"
QWEN_MODEL_REVISION = "a04930dbe5419bfee073f7cade734f572689a3a8"
QWEN_RUNTIME_DISTRIBUTION = "qwen-asr"
QWEN_RUNTIME_VERSION = "0.0.6"


@dataclass(frozen=True)
class RequiredModelFile:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class VerifiedModelReceipt:
    path: Path
    repository: str
    revision: str
    sha256: str
    files: tuple[RequiredModelFile, ...]


QWEN_MODEL_FILES = (
    RequiredModelFile(
        ".gitattributes",
        2260,
        "aa1a1955fa6e6ff0a89a3b2a975cc51bcb808bd0ed59c7a93306defa0f187276",
    ),
    RequiredModelFile(
        "chat_template.json",
        1161,
        "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff",
    ),
    RequiredModelFile(
        "config.json",
        6194,
        "2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f",
    ),
    RequiredModelFile(
        "configuration.json",
        56,
        "c57f6a580d63f7465c6a22ba95847aee05a1ae1181f5abddffb943d9febda061",
    ),
    RequiredModelFile(
        "generation_config.json",
        142,
        "1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c",
    ),
    RequiredModelFile(
        "merges.txt",
        1671853,
        "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    ),
    RequiredModelFile(
        "model-00001-of-00002.safetensors",
        4220320824,
        "a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6",
    ),
    RequiredModelFile(
        "model-00002-of-00002.safetensors",
        478200688,
        "6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc",
    ),
    RequiredModelFile(
        "model.safetensors.index.json",
        64821,
        "f994739fe38e5210b9e3e8ce6c6307315e2ceac3cb630e7b7414d69dce520f60",
    ),
    RequiredModelFile(
        "preprocessor_config.json",
        330,
        "45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d",
    ),
    RequiredModelFile(
        "README.md",
        57456,
        "5058416891bc47a2051557765997e8c42f8eb78a0e33c3e775bd17d4b0ba4d50",
    ),
    RequiredModelFile(
        "tokenizer_config.json",
        12487,
        "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c",
    ),
    RequiredModelFile(
        "vocab.json",
        2776833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
)


def qwen_runtime_identity(
    spec: EngineSpec,
    cache_dir: Path | None,
    model_aliases: dict[str, str],
) -> dict[str, str]:
    repository, revision, distribution, required_version = _qwen_contract(
        spec, model_aliases
    )
    actual_version = require_runtime_version(distribution, required_version)
    model_dir = resolve_qwen_model_dir(repository, cache_dir)
    receipt = verify_model_receipt(
        model_dir,
        repository=repository,
        revision=revision,
        required_files=QWEN_MODEL_FILES,
    )
    return {
        "engine": spec.name,
        "adapter": spec.adapter,
        "model": repository,
        "model_dir": str(model_dir.resolve()),
        "model_revision": revision,
        "model_receipt_path": str(receipt.path),
        "model_receipt_status": "verified",
        "model_receipt_sha256": receipt.sha256,
        "runtime_distribution": distribution,
        "runtime_version": actual_version,
        "runtime_version_required": required_version,
    }


def resolve_qwen_model_dir(repository: str, cache_dir: Path | None) -> Path:
    if cache_dir is None:
        raise RuntimeError(
            "Qwen ASR requires an explicit local ModelScope cache directory; "
            "remote runtime resolution is disabled."
        )
    if repository != QWEN_MODEL_REPOSITORY:
        raise RuntimeError(
            "Qwen ASR model repository mismatch: "
            f"expected {QWEN_MODEL_REPOSITORY}, got {repository or '<missing>'}."
        )
    model_dir = Path(cache_dir).joinpath(*repository.split("/"))
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Qwen ASR model cache not found: {model_dir}. "
            "Run scripts\\download-models.ps1 -Engine qwen3-asr-1.7b "
            "to prefetch the pinned ModelScope revision."
        )
    return model_dir


def write_qwen_model_receipt(
    model_dir: Path,
    *,
    repository: str = QWEN_MODEL_REPOSITORY,
    revision: str = QWEN_MODEL_REVISION,
) -> Path:
    _require_qwen_artifact_contract(repository, revision)
    receipt_path = Path(model_dir) / "MODEL_RECEIPT.json"
    if receipt_path.is_file():
        try:
            return verify_qwen_model_receipt(
                model_dir,
                repository=repository,
                revision=revision,
            ).path
        except RuntimeError:
            pass
    return write_model_receipt(
        model_dir,
        repository=repository,
        revision=revision,
        required_files=QWEN_MODEL_FILES,
    )


def verify_qwen_model_receipt(
    model_dir: Path,
    *,
    repository: str = QWEN_MODEL_REPOSITORY,
    revision: str = QWEN_MODEL_REVISION,
) -> VerifiedModelReceipt:
    _require_qwen_artifact_contract(repository, revision)
    return verify_model_receipt(
        model_dir,
        repository=repository,
        revision=revision,
        required_files=QWEN_MODEL_FILES,
    )


def write_model_receipt(
    model_dir: Path,
    *,
    repository: str,
    revision: str,
    required_files: Iterable[RequiredModelFile],
) -> Path:
    expected_repository = _required_text(repository, "repository")
    expected_revision = _required_text(revision, "revision")
    root, canonical_files = _verify_required_artifacts(
        model_dir, tuple(required_files)
    )
    receipt = {
        "schema": MODEL_RECEIPT_SCHEMA,
        "repository": expected_repository,
        "revision": expected_revision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": [
            {
                "path": record.path,
                "bytes": record.bytes,
                "sha256": record.sha256,
            }
            for record in canonical_files
        ],
    }
    receipt_path = root / "MODEL_RECEIPT.json"
    temporary_path = root / f"MODEL_RECEIPT.json.partial-{os.getpid()}"
    try:
        temporary_path.write_text(
            json.dumps(
                receipt,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary_path.replace(receipt_path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
    return receipt_path


def verify_model_receipt(
    model_dir: Path,
    *,
    repository: str,
    revision: str,
    required_files: Iterable[RequiredModelFile],
) -> VerifiedModelReceipt:
    expected_repository = _required_text(repository, "repository")
    expected_revision = _required_text(revision, "revision")
    root = Path(model_dir).resolve(strict=True)
    receipt_path = root / "MODEL_RECEIPT.json"
    if not receipt_path.is_file():
        raise RuntimeError(
            f"Qwen pinned model receipt is missing: {receipt_path}. "
            "Run scripts\\download-models.ps1 -Engine qwen3-asr-1.7b "
            "-ReceiptOnly for an existing pinned cache."
        )
    receipt = _read_receipt(receipt_path)
    if frozenset(receipt) != MODEL_RECEIPT_FIELDS:
        missing = sorted(MODEL_RECEIPT_FIELDS - frozenset(receipt))
        extra = sorted(frozenset(receipt) - MODEL_RECEIPT_FIELDS)
        raise RuntimeError(
            "Qwen model receipt top-level fields mismatch: "
            f"missing={missing or '<none>'}, extra={extra or '<none>'}."
        )
    if receipt.get("schema") != MODEL_RECEIPT_SCHEMA:
        raise RuntimeError(
            "Qwen model receipt schema mismatch: "
            f"expected {MODEL_RECEIPT_SCHEMA}, got {receipt.get('schema')!r}."
        )
    if receipt.get("repository") != expected_repository:
        raise RuntimeError(
            "Qwen model receipt repository mismatch: "
            f"expected {expected_repository}, got {receipt.get('repository')!r}."
        )
    actual_revision = str(receipt.get("revision") or "").strip()
    if actual_revision != expected_revision:
        raise RuntimeError(
            "Qwen model revision mismatch: "
            f"expected {expected_revision}, receipt has "
            f"{actual_revision or '<missing>'}."
        )
    _verify_created_utc(receipt.get("created_utc"))

    expected_files = _validate_required_file_contract(tuple(required_files))
    records = receipt.get("files")
    if not isinstance(records, list):
        raise RuntimeError("Qwen model receipt files must be an array.")
    parsed_files: list[RequiredModelFile] = []
    seen_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise RuntimeError(
                f"Qwen model receipt files[{index}] must be an object."
            )
        if frozenset(record) != MODEL_FILE_RECORD_FIELDS:
            raise RuntimeError(
                f"Qwen model receipt files[{index}] fields mismatch."
            )
        parsed = _required_file_from_mapping(record, f"files[{index}]")
        if parsed.path in seen_paths:
            raise RuntimeError(
                f"Qwen model receipt contains duplicate path: {parsed.path}."
            )
        seen_paths.add(parsed.path)
        parsed_files.append(parsed)

    actual_paths = [record.path for record in parsed_files]
    expected_paths = [record.path for record in expected_files]
    if actual_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(expected_paths))
        raise RuntimeError(
            "Qwen model receipt required file list mismatch "
            "(including canonical order): "
            f"missing={missing or '<none>'}, extra={extra or '<none>'}."
        )
    for parsed, expected in zip(parsed_files, expected_files, strict=True):
        if parsed.bytes != expected.bytes:
            raise RuntimeError(
                f"Qwen model receipt byte size mismatch for {parsed.path}: "
                f"expected {expected.bytes}, got {parsed.bytes}."
            )
        if parsed.sha256 != expected.sha256:
            raise RuntimeError(
                f"Qwen model receipt SHA-256 mismatch for {parsed.path}: "
                f"expected {expected.sha256}, got {parsed.sha256}."
            )

    _verify_required_artifacts(root, expected_files)
    return VerifiedModelReceipt(
        path=receipt_path,
        repository=expected_repository,
        revision=expected_revision,
        sha256=_sha256_file(receipt_path),
        files=expected_files,
    )


def _qwen_contract(
    spec: EngineSpec,
    model_aliases: dict[str, str],
) -> tuple[str, str, str, str]:
    if spec.adapter != "qwen-asr":
        raise RuntimeError(
            f"Qwen runtime identity requires adapter 'qwen-asr', got {spec.adapter!r}."
        )
    repository = model_aliases.get(spec.model, spec.model)
    options = spec.options or {}
    revision = str(options.get("model_revision") or "").strip()
    distribution = str(options.get("runtime_distribution") or "").strip()
    runtime_version = str(options.get("runtime_version") or "").strip()
    _require_qwen_artifact_contract(repository, revision)
    if distribution != QWEN_RUNTIME_DISTRIBUTION:
        raise RuntimeError(
            "Qwen ASR runtime distribution mismatch: "
            f"expected {QWEN_RUNTIME_DISTRIBUTION}, "
            f"got {distribution or '<missing>'}."
        )
    if runtime_version != QWEN_RUNTIME_VERSION:
        raise RuntimeError(
            "Qwen ASR configured runtime version mismatch: "
            f"expected {QWEN_RUNTIME_VERSION}, "
            f"got {runtime_version or '<missing>'}."
        )
    return repository, revision, distribution, runtime_version


def _require_qwen_artifact_contract(repository: str, revision: str) -> None:
    if repository != QWEN_MODEL_REPOSITORY:
        raise RuntimeError(
            "Qwen ASR model repository mismatch: "
            f"expected {QWEN_MODEL_REPOSITORY}, "
            f"got {repository or '<missing>'}."
        )
    if revision != QWEN_MODEL_REVISION:
        raise RuntimeError(
            "Qwen ASR configured model revision mismatch: "
            f"expected {QWEN_MODEL_REVISION}, got {revision or '<missing>'}."
        )


def require_runtime_version(distribution: str, required_version: str) -> str:
    try:
        actual_version = importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Qwen ASR runtime distribution is missing: {distribution}=="
            f"{required_version}. Run scripts\\setup-qwen.ps1."
        ) from exc
    if actual_version != required_version:
        raise RuntimeError(
            "Qwen ASR runtime version mismatch: "
            f"expected {distribution}=={required_version}, "
            f"got {distribution}=={actual_version}."
        )
    return actual_version


def _read_receipt(receipt_path: Path) -> dict[str, Any]:
    try:
        receipt = json.loads(
            receipt_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Qwen model receipt is unreadable: {receipt_path}: {exc}"
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            f"Qwen model receipt is invalid: {receipt_path}: {exc}"
        ) from exc
    if not isinstance(receipt, dict):
        raise RuntimeError("Qwen model receipt must be a JSON object.")
    return receipt


def _verify_created_utc(value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            "Qwen model receipt created_utc must be a non-empty UTC timestamp."
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(
            "Qwen model receipt created_utc must be an ISO-8601 UTC timestamp."
        ) from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RuntimeError("Qwen model receipt created_utc must use UTC.")


def _verify_required_artifacts(
    model_dir: Path,
    required_files: tuple[RequiredModelFile, ...],
) -> tuple[Path, tuple[RequiredModelFile, ...]]:
    root = Path(model_dir).resolve(strict=True)
    canonical_files = _validate_required_file_contract(required_files)
    for record in canonical_files:
        candidate = root.joinpath(*PurePosixPath(record.path).parts)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise RuntimeError(
                "Qwen required model artifact is missing or escapes model_dir: "
                f"{record.path}: {exc}"
            ) from exc
        if not resolved.is_file():
            raise RuntimeError(
                f"Qwen required model artifact is not a file: {record.path}."
            )
        actual_bytes = resolved.stat().st_size
        if actual_bytes != record.bytes:
            raise RuntimeError(
                f"Qwen model artifact size mismatch for {record.path}: "
                f"expected {record.bytes}, got {actual_bytes}."
            )
        actual_sha256 = _sha256_file(resolved)
        if actual_sha256 != record.sha256:
            raise RuntimeError(
                f"Qwen model artifact SHA-256 mismatch for {record.path}: "
                f"expected {record.sha256}, got {actual_sha256}."
            )
    return root, canonical_files


def _validate_required_file_contract(
    required_files: tuple[RequiredModelFile, ...],
) -> tuple[RequiredModelFile, ...]:
    if not required_files:
        raise RuntimeError("Qwen canonical required-file list must not be empty.")
    seen_paths: set[str] = set()
    for index, record in enumerate(required_files):
        if not isinstance(record, RequiredModelFile):
            raise RuntimeError(
                f"Qwen required_files[{index}] must be RequiredModelFile."
            )
        _validate_relative_path(record.path, f"required_files[{index}]")
        if record.path in seen_paths:
            raise RuntimeError(
                f"Qwen canonical required-file list contains duplicate path: "
                f"{record.path}."
            )
        seen_paths.add(record.path)
        if (
            isinstance(record.bytes, bool)
            or not isinstance(record.bytes, int)
            or record.bytes <= 0
        ):
            raise RuntimeError(
                f"Qwen canonical byte size is invalid for {record.path}: "
                f"{record.bytes!r}."
            )
        if not SHA256_PATTERN.fullmatch(record.sha256):
            raise RuntimeError(
                f"Qwen canonical SHA-256 is invalid for {record.path}."
            )
    return required_files


def _required_file_from_mapping(
    record: dict[str, Any],
    label: str,
) -> RequiredModelFile:
    path = record.get("path")
    _validate_relative_path(path, label)
    size = record.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise RuntimeError(
            f"Qwen model receipt has invalid byte size at {label}: {size!r}."
        )
    sha256 = record.get("sha256")
    if not isinstance(sha256, str) or not SHA256_PATTERN.fullmatch(sha256):
        raise RuntimeError(
            f"Qwen model receipt has invalid SHA-256 at {label}."
        )
    return RequiredModelFile(path=path, bytes=size, sha256=sha256)


def _validate_relative_path(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or PurePosixPath(value).is_absolute()
        or str(PurePosixPath(value)) != value
        or any(part in {"", ".", ".."} for part in PurePosixPath(value).parts)
    ):
        raise RuntimeError(
            f"Qwen model receipt contains unsafe or noncanonical path at "
            f"{label}: {value!r}."
        )


def _required_text(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"Qwen model receipt {label} must not be empty.")
    return text


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or verify the pinned Qwen3-ASR model receipt."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("write-receipt", "verify-receipt"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--model-dir", type=Path, required=True)
        subparser.add_argument(
            "--repository", default=QWEN_MODEL_REPOSITORY
        )
        subparser.add_argument("--revision", default=QWEN_MODEL_REVISION)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "write-receipt":
        receipt_path = write_qwen_model_receipt(
            args.model_dir,
            repository=args.repository,
            revision=args.revision,
        )
        verified = VerifiedModelReceipt(
            path=receipt_path,
            repository=args.repository,
            revision=args.revision,
            sha256=_sha256_file(receipt_path),
            files=QWEN_MODEL_FILES,
        )
    else:
        verified = verify_qwen_model_receipt(
            args.model_dir,
            repository=args.repository,
            revision=args.revision,
        )
        receipt_path = verified.path
    print(
        json.dumps(
            {
                "schema": MODEL_RECEIPT_SCHEMA,
                "repository": verified.repository,
                "revision": verified.revision,
                "receipt_path": str(receipt_path),
                "receipt_sha256": verified.sha256,
                "file_count": len(verified.files),
                "status": "verified",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate a custom-agent ZIP package without executing its contents.

Run locally:
    python examples/manifest_validator.py

Google Colab setup cell:
    # This lesson uses only the Python standard library.

The validator demonstrates the first control in a secure upload workflow.
Production also needs quarantine storage, malware/dependency scanning, signed
artifacts, approval, and a sandboxed runtime.
"""

# %% Imports
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile


MAX_ENTRIES = 100
MAX_UNCOMPRESSED_BYTES = 2_000_000
SUPPORTED_PROTOCOL_VERSION = "1.0"


class PackageValidationError(ValueError):
    """A package was rejected before it could reach an execution environment."""


@dataclass(frozen=True)
class AgentManifest:
    name: str
    version: str
    protocol_version: str
    entrypoint: str
    runtime: str


@dataclass(frozen=True)
class PackageValidation:
    manifest: AgentManifest
    sha256_digest: str
    file_count: int
    uncompressed_bytes: int


# %% Step 1 — archive safety checks happen before parsing files
def _safe_archive_path(filename: str) -> bool:
    path = PurePosixPath(filename)
    return (
        bool(filename)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in filename
        and not filename.startswith("/")
    )


def _read_manifest(raw_manifest: bytes, archive_names: set[str]) -> AgentManifest:
    try:
        data: Any = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageValidationError("agent.json must be valid UTF-8 JSON") from exc
    if not isinstance(data, dict):
        raise PackageValidationError("agent.json must be an object")

    required = {"name", "version", "protocol_version", "entrypoint", "runtime"}
    missing = sorted(required.difference(data))
    if missing:
        raise PackageValidationError(f"manifest is missing: {', '.join(missing)}")

    values = {key: data[key] for key in required}
    if not all(isinstance(value, str) and value.strip() for value in values.values()):
        raise PackageValidationError("manifest fields must be non-empty strings")
    if values["protocol_version"] != SUPPORTED_PROTOCOL_VERSION:
        raise PackageValidationError("unsupported protocol_version")
    if values["runtime"] != "python3.12":
        raise PackageValidationError("unsupported runtime")
    if not _safe_archive_path(values["entrypoint"]):
        raise PackageValidationError("entrypoint path is unsafe")
    if values["entrypoint"] not in archive_names:
        raise PackageValidationError("entrypoint is not present in the package")

    return AgentManifest(**values)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# %% Step 2 — validate metadata and return immutable artifact identity
def validate_package(path: Path) -> PackageValidation:
    """Reject unsafe archives and return validated manifest metadata plus digest."""
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ENTRIES:
                raise PackageValidationError("archive contains too many entries")
            if any(not _safe_archive_path(info.filename) for info in infos):
                raise PackageValidationError("archive contains an unsafe path")
            uncompressed_bytes = sum(info.file_size for info in infos)
            if uncompressed_bytes > MAX_UNCOMPRESSED_BYTES:
                raise PackageValidationError("archive uncompressed size exceeds policy")

            archive_names = {info.filename for info in infos}
            if "agent.json" not in archive_names:
                raise PackageValidationError("archive is missing agent.json")
            manifest = _read_manifest(archive.read("agent.json"), archive_names)
    except PackageValidationError:
        raise
    except Exception as exc:  # zipfile.BadZipFile and I/O errors become safe domain errors.
        raise PackageValidationError("package is not a readable ZIP archive") from exc

    return PackageValidation(manifest, _sha256_file(path), len(infos), uncompressed_bytes)


# %% Step 3 — build explicit sample inputs; the validator never imports them
def build_demo_package(path: Path, include_traversal_path: bool = False) -> None:
    manifest = {
        "name": "deployment-coach",
        "version": "1.2.0",
        "protocol_version": "1.0",
        "entrypoint": "src/agent.py",
        "runtime": "python3.12",
    }
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("agent.json", json.dumps(manifest, indent=2))
        archive.writestr("src/agent.py", "# Package source is never executed during validation.\n")
        if include_traversal_path:
            archive.writestr("../escape.py", "This archive entry must be rejected.\n")


def demo() -> None:
    with TemporaryDirectory() as directory:
        directory_path = Path(directory)
        safe_package = directory_path / "safe-agent.zip"
        malicious_package = directory_path / "malicious-agent.zip"
        build_demo_package(safe_package)
        build_demo_package(malicious_package, include_traversal_path=True)

        validation = validate_package(safe_package)
        print("Validated package:", validation)

        try:
            validate_package(malicious_package)
        except PackageValidationError as exc:
            print("Rejected malicious package:", exc)
            rejected = True
        else:
            rejected = False

        assert validation.manifest.name == "deployment-coach"
        assert len(validation.sha256_digest) == 64
        assert rejected
        print("PASS: package manifest and archive-safety checks completed.")


if __name__ == "__main__":
    demo()

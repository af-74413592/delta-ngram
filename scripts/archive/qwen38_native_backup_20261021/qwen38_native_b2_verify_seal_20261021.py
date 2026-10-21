#!/usr/bin/env python3
"""Verify a restored native snapshot against its preregistered staging seal.

This verifier never reads the original source paths from the member manifest.
The CLI always uses the fixed three metadata hashes and 2,466-payload count.
Write the receipt outside the restored tree so the restored file set stays exact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time


EXPECTED_METADATA_SHA256 = {
    "metadata/STAGED.json": "b60df53b3c4301ede6fd8a577868e1e2630ba51fbcda6cc9d5477da57331d68d",
    "metadata/member_manifest.jsonl": "17cd2f3a10594aadf34814a7d61308183076200a88136d0a4b4cc80bb8104158",
    "metadata/member_sha256.txt": "c5949680daf7a13eacdc83112d699b0aee6ae899641ad53b88b16c803c84b888",
}
EXPECTED_PAYLOAD_FILES = 2466
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_relative(value: object) -> str:
    require(isinstance(value, str) and bool(value), "Empty or non-string member path")
    require(not any(c in value for c in ("\\", ":", "\x00", "\r", "\n", "\t")),
            "Unsafe character in member path")
    require(not value.startswith("/") and all(p not in ("", ".", "..") for p in value.split("/")),
            "Absolute or noncanonical member path")
    require(PurePosixPath(value).as_posix() == value, "Noncanonical member path")
    return value


def pure_source_root(value: object) -> PurePosixPath:
    require(isinstance(value, str) and value.startswith("/"), "Source root must be absolute POSIX")
    require(not any(c in value for c in ("\\", ":", "\x00", "\r", "\n", "\t")),
            "Unsafe character in source root")
    path = PurePosixPath(value)
    require(value == path.as_posix() and value != "/", "Source root must be canonical")
    require(all(p not in ("", ".", "..") for p in value[1:].split("/")),
            "Noncanonical source root")
    return path


def safe_file(root: Path, relative: str) -> Path:
    relative = safe_relative(relative)
    path = root
    for part in relative.split("/"):
        path = path / part
        require(not path.is_symlink(), "Symlink forbidden: " + relative)
    require(path.resolve(strict=True).is_relative_to(root), "Member escaped restored root")
    require(stat.S_ISREG(path.lstat().st_mode), "Nonregular member: " + relative)
    return path


def file_hash(path: Path) -> str:
    before = path.stat()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "File changed while hashing: " + path.name)
    return digest


def restored_file_set(root: Path) -> set[str]:
    result = set()
    require(not (root / "verification").exists(), "Unexpected verification directory in original readback")
    for directory, folders, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in folders:
            path = base / name
            require(not path.is_symlink(), "Symlink directory forbidden")
            require(stat.S_ISDIR(path.lstat().st_mode), "Non-directory in directory listing")
        for name in files:
            path = base / name
            relative = path.relative_to(root).as_posix()
            safe_file(root, relative)
            require(relative not in result, "Duplicate restored member")
            result.add(relative)
    return result


def verify_restore(restore: Path, *,
                   expected_metadata: dict[str, str] = EXPECTED_METADATA_SHA256,
                   expected_payload_files: int = EXPECTED_PAYLOAD_FILES) -> dict:
    """Return evidence after full verification; overrides exist only for unit fixtures."""
    started = time.monotonic()
    require(not restore.is_symlink(), "Restored root cannot be a symlink")
    root = restore.resolve(strict=True)
    require(root.is_dir(), "Restored root must be a directory")
    metadata = {}
    for relative, expected in expected_metadata.items():
        path = safe_file(root, relative)
        actual = file_hash(path)
        require(actual == expected, "Preregistered metadata hash mismatch: " + relative)
        metadata[relative] = {"size_bytes": path.stat().st_size, "sha256": actual}

    staged = json.loads((root / "metadata/STAGED.json").read_text(encoding="utf-8"))
    require(staged["status"] == "staged_sha256_verified", "Unexpected staging status")
    require(staged["payload_files"] == expected_payload_files, "Sealed payload count differs")
    require(staged["metadata_self_excluded_from_member_manifest"] is True,
            "Unexpected staging metadata membership policy")
    source_root = pure_source_root(staged["source_root"])
    manifest_bytes = (root / "metadata/member_manifest.jsonl").read_bytes()
    manifest_text = manifest_bytes.decode("utf-8")
    lines = manifest_text.splitlines()
    require(lines and all(line.strip() for line in lines), "Blank manifest member line")
    members = [json.loads(line) for line in lines]
    require(len(members) == expected_payload_files, "Manifest payload count differs")
    names = set()
    total_bytes = 0
    expected_checksum_lines = []
    for member in members:
        relative = safe_relative(member["relative_path"])
        require(relative not in names and relative not in expected_metadata, "Duplicate payload or metadata member")
        require(not relative.startswith("verification/"), "Unexpected verification payload member")
        names.add(relative)
        size = member["size_bytes"]
        require(type(size) is int and size >= 0, "Invalid member size")
        digest = member["sha256"]
        require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, "Invalid member SHA256")
        # Pure string/path arithmetic only: never stat/open/resolve this original path.
        require(member["source_path"] == (source_root / relative).as_posix(),
                "Source path does not match sealed root and safe member path")
        expected_checksum_lines.append(f"{digest}  {relative}\n")
        total_bytes += size
    require(total_bytes == staged["payload_bytes"], "Sealed payload byte total differs")
    require((root / "metadata/member_sha256.txt").read_bytes() ==
            "".join(expected_checksum_lines).encode("utf-8"),
            "Checksum text does not exactly match manifest order and contents")
    require(set(staged["seals"]) == {"metadata/member_manifest.jsonl", "metadata/member_sha256.txt"},
            "Unexpected staging seal members")
    for relative, seal in staged["seals"].items():
        require(seal == metadata[relative], "STAGED seal differs: " + relative)

    expected_files = names | set(expected_metadata)
    actual_files = restored_file_set(root)
    require(actual_files == expected_files,
            "Restored file set differs: missing=" + str(len(expected_files - actual_files)) +
            ", extra=" + str(len(actual_files - expected_files)))
    verified_bytes = 0
    for member in members:
        relative = member["relative_path"]
        path = safe_file(root, relative)
        require(path.stat().st_size == member["size_bytes"], "Payload size mismatch: " + relative)
        require(file_hash(path) == member["sha256"], "Payload SHA256 mismatch: " + relative)
        verified_bytes += member["size_bytes"]

    # Recheck metadata and the file set after the complete hashing pass.
    for relative, expected in expected_metadata.items():
        require(file_hash(safe_file(root, relative)) == expected, "Metadata changed during verification")
    require(restored_file_set(root) == expected_files, "Restored file set changed during verification")
    return {
        "protocol": "qwen38_native_b2_original_seal_verification_v1",
        "status": "complete",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "restore": str(root),
        "preregistered_metadata": metadata,
        "payload_files": len(members),
        "payload_bytes": verified_bytes,
        "total_restored_files": len(actual_files),
        "total_restored_bytes": verified_bytes + sum(x["size_bytes"] for x in metadata.values()),
        "all_payload_sha256_and_sizes_verified": True,
        "staged_seals_verified": True,
        "checksum_text_matches_manifest_exactly": True,
        "exact_file_set_payload_plus_three_metadata": True,
        "safe_relative_paths_and_no_symlinks": True,
        "source_paths_validated_without_reading_original_sources": True,
        "verification_subdirectory_absent": True,
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    root = args.restore.resolve()
    require(not output.is_relative_to(root), "Receipt must be outside restored tree")
    require(not output.exists(), "Refusing to replace an existing verification receipt")
    try:
        result = verify_restore(args.restore)
        result["verifier_sha256"] = file_hash(Path(__file__).resolve())
        exit_code = 0
    except Exception as error:
        result = {"protocol": "qwen38_native_b2_original_seal_verification_v1", "status": "failed",
                  "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                  "restore": str(root), "error_type": type(error).__name__, "error": str(error)}
        exit_code = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(result), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

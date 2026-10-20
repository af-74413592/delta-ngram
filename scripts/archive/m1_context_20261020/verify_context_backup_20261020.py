"""Verify the sealed DEV64 context increment without loading models or changing it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat

MANIFEST = "metadata/member_manifest.jsonl"
CHECKSUMS = "metadata/sha256.txt"
COMPLETE = "metadata/COMPLETE.json"
SEALS = {MANIFEST, CHECKSUMS, COMPLETE}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def inventory(root):
    require(root.is_dir() and not root.is_symlink(), "Root is missing or linked")
    paths = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        require(not path.is_symlink(), "Symlink: " + relative)
        require(".part" not in path.name and not path.name.endswith(".tmp"),
                "Unfinished file: " + relative)
        mode = path.lstat().st_mode
        require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), "Special file: " + relative)
        require(path.resolve().is_relative_to(root), "Escaping path: " + relative)
        if stat.S_ISREG(mode):
            paths[relative] = path
    return paths


def checked_hash(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    stamp = (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev)
    require((before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev) == stamp,
            "File changed while hashing: " + str(path))
    return digest.hexdigest(), after.st_size, stamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-complete-sha256", required=True)
    parser.add_argument("--source-stage", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.root.is_symlink(), "Linked root")
    root = args.root.resolve()
    source = args.source_stage.resolve() if args.source_stage else None
    receipt = args.receipt.resolve()
    require(not receipt.is_relative_to(root), "Receipt must be outside the verified tree")
    require(source is None or (source != root and not receipt.is_relative_to(source)),
            "Restore must be independent and receipt outside source")
    require(not receipt.exists(), "Refusing to replace an existing receipt")
    files = inventory(root)
    require(SEALS <= set(files), "Missing seal")
    complete_sha, _, _ = checked_hash(files[COMPLETE])
    require(complete_sha == args.expected_complete_sha256,
            "COMPLETE differs from the external pre-upload SHA pin")
    complete = json.loads(files[COMPLETE].read_text(encoding="utf-8"))
    require(complete["complete"] is True and complete.get("protocol") ==
            "m1_dev64_context_backup_20261020_v1", "Incomplete or wrong context backup")
    for relative in (MANIFEST, CHECKSUMS):
        actual_sha, actual_size, _ = checked_hash(files[relative])
        expected = complete["seals"][relative]
        require((actual_sha, actual_size) == (expected["sha256"], expected["size_bytes"]),
                "Seal differs: " + relative)
    rows = [json.loads(line) for line in
            files[MANIFEST].read_text(encoding="utf-8").splitlines()]
    expected = {}
    for row in rows:
        relative = row["relative_path"]
        pure = PurePosixPath(relative)
        require(not pure.is_absolute() and ".." not in pure.parts and
                pure.as_posix() == relative and relative not in SEALS,
                "Unsafe or reserved manifest path")
        require(relative not in expected, "Duplicate manifest path")
        expected[relative] = row
    require(set(files) == set(expected) | SEALS, "Object set differs from the sealed manifest")
    require(len(rows) == complete["payload_files"] and
            sum(row["size_bytes"] for row in rows) == complete["payload_bytes"],
            "Payload totals differ")
    checksums = "".join(
        row["sha256"] + "  " + row["relative_path"] + "\n"
        for row in sorted(rows, key=lambda row: row["relative_path"]))
    checksums += complete["seals"][MANIFEST]["sha256"] + "  " + MANIFEST + "\n"
    require(files[CHECKSUMS].read_bytes() == checksums.encode("utf-8"),
            "Checksum listing differs from manifest")
    stamps = {}
    total_bytes = 0
    for relative, path in sorted(files.items()):
        actual_sha, size, stamp = checked_hash(path)
        if relative in expected:
            require((actual_sha, size) ==
                    (expected[relative]["sha256"], expected[relative]["size_bytes"]),
                    "Member differs: " + relative)
        elif relative == COMPLETE:
            require(actual_sha == args.expected_complete_sha256, "COMPLETE changed during verification")
        else:
            require((actual_sha, size) ==
                    (complete["seals"][relative]["sha256"], complete["seals"][relative]["size_bytes"]),
                    "Seal changed during verification: " + relative)
        if source is not None:
            original = source.joinpath(*PurePosixPath(relative).parts)
            require(original.is_file() and not original.is_symlink(),
                    "Source member unavailable: " + relative)
            require(not os.path.samefile(path, original),
                    "Restored member shares source inode: " + relative)
        stamps[relative] = stamp
        total_bytes += size
    require(len(files) == complete["total_objects"] and total_bytes == complete["total_bytes"],
            "Final object/byte totals differ")
    require(set(inventory(root)) == set(files), "Object set changed during verification")
    for relative, stamp in stamps.items():
        current = files[relative].stat()
        require((current.st_size, current.st_mtime_ns, current.st_ino, current.st_dev) == stamp,
                "Member changed after hashing: " + relative)
    result = {
        "complete": True, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "expected_complete_sha256": args.expected_complete_sha256,
        "verified_files": len(files), "verified_bytes": total_bytes,
        "all_sha256_matched": True, "object_set_exact": True,
        "no_symlinks_or_partial_files": True, "files_stable_during_verification": True,
        "source_stage": str(source) if source else None,
        "independent_restore_inodes_verified": source is not None,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

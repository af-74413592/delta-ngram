#!/usr/bin/env python3
"""Restore the sealed M1 core from B2 using its already registered SHA256 manifest."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import urllib.error
import urllib.parse
import urllib.request

PREFIX = "cl-research/delta-ngram/20261020-rtx4090/"
SEALS = {
    "metadata/COMPLETE.json": "ae0efc8a826d4173664cff3ee8e046a8a55ee38b9aa6d6c53123c6fce9507e95",
    "metadata/member_manifest.jsonl": "be4a8b39dd2f5c3e50fe9aea472a1f2d06c9a28e3717db674d1762b821502611",
    "metadata/member_sha256.txt": "72dd24744e0c0551851e0edbb917189b860e88c63c88cc4a8443a5f9d4f34b43",
}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or "\\" in relative:
        raise ValueError("Unsafe manifest path")
    path = root / parts
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Path escaped restore root")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auth-file", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    auth = json.loads(args.auth_file.read_text())
    origin = auth["apiInfo"]["storageApi"]["downloadUrl"]
    parsed = urllib.parse.urlsplit(origin)
    if parsed.scheme != "https" or not parsed.hostname.endswith(".backblazeb2.com"):
        raise ValueError("Unexpected B2 download origin")
    args.target.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def download(relative: str, expected: str, size: int | None = None) -> dict:
        path = safe_path(args.target, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if (size is not None and path.stat().st_size != size) or digest(path) != expected:
                raise ValueError(f"Existing restore file differs: {relative}")
            return {"path": relative, "size": path.stat().st_size, "sha256": expected}
        url = origin + "/file/clclcl/" + urllib.parse.quote(PREFIX + relative, safe="/")
        partial = path.with_name(path.name + ".restore-part")
        request = urllib.request.Request(url, headers={"Authorization": auth["authorizationToken"]})
        for attempt in range(5):
            try:
                sha = hashlib.sha256()
                count = 0
                with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                    while block := response.read(8 * 1024 * 1024):
                        handle.write(block)
                        sha.update(block)
                        count += len(block)
                if (size is not None and count != size) or sha.hexdigest() != expected:
                    raise ValueError(f"Downloaded content mismatch: {relative}")
                partial.replace(path)
                return {"path": relative, "size": count, "sha256": expected}
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 4:
                    raise RuntimeError(f"B2 transfer failed after retries: {relative}") from None
                time.sleep(min(2 ** attempt, 8))
        raise AssertionError("unreachable")

    verified = [download(relative, sha) for relative, sha in SEALS.items()]
    members = [json.loads(line) for line in
               (args.target / "metadata/member_manifest.jsonl").read_text().splitlines()]
    names = [item["relative_path"] for item in members]
    if len(names) != 1730 or len(set(names)) != len(names) or set(names) & SEALS.keys():
        raise ValueError("Unexpected core member set")
    if sum(item["size_bytes"] for item in members) != 23884626562:
        raise ValueError("Unexpected core total size")
    print(json.dumps({"event": "seals_verified", "members": len(members),
                      "bytes": sum(item["size_bytes"] for item in members)}), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, item["relative_path"], item["sha256"], item["size_bytes"])
                   for item in members]
        for future in as_completed(futures):
            verified.append(future.result())
            if len(verified) % 50 == 0 or len(verified) == len(members) + 3:
                print(json.dumps({"event": "download_and_hash_verified", "files": len(verified),
                                  "bytes": sum(item["size"] for item in verified),
                                  "elapsed_seconds": time.monotonic() - started}), flush=True)
    expected_paths = set(names) | SEALS.keys()
    actual_paths = {p.relative_to(args.target).as_posix() for p in args.target.rglob("*") if p.is_file()}
    if actual_paths != expected_paths:
        raise ValueError("Restored file set differs from sealed core")
    receipt = {
        "protocol": "qwen38_native_b2_core_restore_v1",
        "source": "b2://clclcl/" + PREFIX,
        "target": str(args.target.resolve()),
        "registered_seals": SEALS,
        "verified_files": len(verified),
        "verified_bytes": sum(item["size"] for item in verified),
        "all_sha256_verified": True, "exact_core_file_set": True,
        "final_content_inspected": False, "old_token_ids_reused": False,
        "elapsed_seconds": time.monotonic() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

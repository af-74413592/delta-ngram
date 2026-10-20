#!/usr/bin/env python3
"""Download the registered native model and verify every file against HF metadata."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import time

REPO = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_file(root: Path, item: dict) -> dict:
    path = root / item["path"]
    before = path.stat()
    if before.st_size != item["size"]:
        raise ValueError(f"Size mismatch: {item['path']}")
    sha256 = hashlib.sha256()
    git = hashlib.sha1()
    git.update(f"blob {before.st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            sha256.update(block)
            if not item["lfs_sha256"]:
                git.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed during verification: {item['path']}")
    actual = sha256.hexdigest() if item["lfs_sha256"] else git.hexdigest()
    expected = item["lfs_sha256"] or item["git_blob_sha1"]
    if actual != expected:
        raise ValueError(f"Content hash mismatch: {item['path']}")
    return {**item, "sha256": sha256.hexdigest(), "verified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=False)
    args.target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    from huggingface_hub import HfApi, snapshot_download

    start = time.monotonic()
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError(f"Unexpected revision: {info.sha}")
    files = []
    for sibling in info.siblings:
        relative = PurePosixPath(sibling.rfilename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe repository path")
        lfs = sibling.lfs
        if sibling.size is None or not sibling.blob_id:
            raise ValueError(f"Missing integrity metadata: {relative}")
        files.append({
            "path": sibling.rfilename, "size": sibling.size,
            "git_blob_sha1": sibling.blob_id,
            "lfs_sha256": lfs.sha256 if lfs else None,
        })
    shards = [f for f in files if f["path"].endswith(".safetensors")]
    if len(shards) != 131 or any(not f["lfs_sha256"] for f in shards):
        raise ValueError("Expected 131 weight shards with SHA256 metadata")
    manifest = {
        "repo": REPO, "revision": info.sha,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "packages": {n: importlib.metadata.version(n) for n in ("huggingface-hub", "hf-xet")},
        "files": files, "bytes": sum(f["size"] for f in files),
    }
    save(args.report_dir / "hf_expected_manifest.json", manifest)
    print(json.dumps({"event": "manifest_verified", "revision": info.sha,
                      "files": len(files), "shards": len(shards), "bytes": manifest["bytes"]}), flush=True)
    snapshot_download(REPO, revision=REVISION, local_dir=args.target, max_workers=args.workers)
    print(json.dumps({"event": "download_finished", "elapsed": time.monotonic() - start}), flush=True)
    verified = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item in pool.map(lambda f: verify_file(args.target, f), files):
            verified.append(item)
            print(json.dumps({"event": "hash_verified", "path": item["path"],
                              "completed": len(verified), "total": len(files)}), flush=True)
    save(args.report_dir / "COMPLETE.json", {
        **manifest, "files": verified, "all_file_hashes_verified": True,
        "elapsed_seconds": time.monotonic() - start,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    print("COMPLETE: pinned revision and all file hashes verified", flush=True)


if __name__ == "__main__":
    main()

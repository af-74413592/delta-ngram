"""Upload an immutable experiment snapshot and verify every file by full readback."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--ops", type=Path, required=True)
    p.add_argument("--restore", type=Path, required=True)
    p.add_argument("--auth-file", type=Path, required=True)
    p.add_argument("--destination", required=True)
    a = p.parse_args()
    if a.destination != "clclcl/cl-research/delta-ngram/20261021-qwen38-native/":
        raise ValueError("Unexpected backup destination")
    a.ops.mkdir(parents=True, exist_ok=True)
    status_path = a.ops / "status.json"
    started = time.monotonic()
    status = {"protocol": "qwen38_native_b2_backup_v1", "started_utc": now(),
              "destination": "b2://" + a.destination, "stage": str(a.stage),
              "restore": str(a.restore), "status": "preparing"}

    def save(**updates):
        status.update(updates, updated_utc=now(), elapsed_seconds=time.monotonic() - started)
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        temp.replace(status_path)
        print(json.dumps({k: status[k] for k in ("status", "updated_utc", "elapsed_seconds")}), flush=True)

    try:
        seal = a.stage / "metadata/STAGED.json"
        if not seal.is_file():
            raise ValueError("Snapshot has not been sealed")
        if a.stage.resolve() == a.restore.resolve() or a.restore.exists():
            raise ValueError("Readback must use a new independent directory")
        members = []
        for f in sorted(a.stage.rglob("*")):
            if not f.is_file():
                continue
            if f.is_symlink():
                raise ValueError("Symlink in snapshot")
            st = f.stat()
            digest = sha256(f)
            after = f.stat()
            if (st.st_size, st.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Snapshot changed during hash")
            members.append({"relative_path": f.relative_to(a.stage).as_posix(),
                            "size_bytes": st.st_size, "sha256": digest})
        inventory = a.ops / "uploaded_file_manifest.jsonl"
        inventory.write_text("".join(json.dumps(m) + "\n" for m in members), encoding="utf-8")
        total = sum(m["size_bytes"] for m in members)
        save(files=len(members), bytes=total, uploaded_manifest_sha256=sha256(inventory))
        auth = json.loads(a.auth_file.read_text(encoding="utf-8"))
        env = os.environ.copy()
        env.update(RCLONE_CONFIG_CLEXPERIMENT_TYPE="b2",
                   RCLONE_CONFIG_CLEXPERIMENT_ACCOUNT=auth["account"],
                   RCLONE_CONFIG_CLEXPERIMENT_KEY=auth["key"])
        del auth
        destination = "clexperiment:" + a.destination

        def transfer(src, dst, label, extra=()):
            cmd = ["rclone", "copy", str(src), str(dst), "--config", "/dev/null",
                   "--immutable", "--checksum", "--transfers", "4", "--checkers", "8",
                   "--stats", "30s", "--log-level", "INFO", "--log-file", str(a.ops / (label + ".log"))]
            cmd.extend(extra)
            result = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode:
                raise RuntimeError(f"rclone {label} exited {result.returncode}; inspect its log")

        priority = [m["relative_path"] for m in members if
                    ("native_main_training_08/" in m["relative_path"] and
                     (m["relative_path"].endswith("joint-policy-stage-2-alpha100_rmsnorm-model_milestone-100000000.pt") or
                      m["relative_path"].endswith("joint-policy-stage-3-final.pt"))) or
                    ("native_main_training_07/" in m["relative_path"] and
                     m["relative_path"].endswith("joint-stage-3-alpha099_rmsnorm-nominal-40000000.pt")) or
                    ("/evaluations/" in m["relative_path"]) or
                    m["relative_path"].startswith("data/native_dev64/")]
        required_weights = [name for name in priority if name.endswith(".pt")]
        if len(required_weights) != 3:
            raise ValueError("Priority checkpoint set differs")
        priority_file = a.ops / "priority-files.txt"
        priority_file.write_text("\n".join(priority) + "\n", encoding="utf-8")
        save(status="uploading_priority", priority_files=len(priority), priority_checkpoints=required_weights)
        transfer(a.stage, destination, "priority_upload", ["--files-from", str(priority_file)])
        save(status="uploading_snapshot", priority_upload_complete=True)
        transfer(a.stage, destination, "snapshot_upload")
        save(status="reading_back", upload_complete=True)
        transfer(destination, a.restore, "full_readback")
        save(status="verifying_sha256")
        expected = {m["relative_path"] for m in members}
        actual = {f.relative_to(a.restore).as_posix() for f in a.restore.rglob("*") if f.is_file()}
        if expected != actual:
            raise ValueError("Readback file set differs from uploaded snapshot")
        verified_bytes = 0
        for i, m in enumerate(members, 1):
            f = a.restore / m["relative_path"]
            if f.stat().st_size != m["size_bytes"] or sha256(f) != m["sha256"]:
                raise ValueError("Readback checksum mismatch: " + m["relative_path"])
            # Ensure source hardlinks remained immutable throughout transfer.
            source = a.stage / m["relative_path"]
            if source.stat().st_size != m["size_bytes"] or sha256(source) != m["sha256"]:
                raise ValueError("Staged source changed: " + m["relative_path"])
            verified_bytes += m["size_bytes"]
            if i % 100 == 0:
                save(verified_files=i, verified_bytes=verified_bytes)
        save(status="complete", completed_utc=now(), all_sha256_verified=True,
             exact_file_set=True, verified_files=len(members), verified_bytes=verified_bytes)
        receipt = a.ops / "COMPLETE.json"
        receipt.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        # Upload verification metadata separately, leaving the sealed source untouched.
        verification_files = [receipt, inventory, Path(__file__),
                              a.ops / "checkpoint_identity.json",
                              a.ops / "qwen38_code_backup_receipt_20261021.json",
                              a.ops / "qwen38_native_backup_prepare_20261021.py"]
        for f in verification_files:
            if not f.is_file():
                raise ValueError("Missing verification artifact: " + f.name)
            result = subprocess.run(["rclone", "copyto", str(f), destination + "verification/" + f.name,
                                     "--config", "/dev/null", "--immutable", "--checksum"],
                                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode:
                raise RuntimeError("Could not upload verification metadata")
        verify_dir = a.ops / "verification_metadata_readback"
        transfer(destination + "verification/", verify_dir, "verification_metadata_readback")
        for f in verification_files:
            if sha256(f) != sha256(verify_dir / f.name):
                raise ValueError("Verification metadata readback differs")
        save(verification_metadata_uploaded_and_readback_verified=True)
    except Exception as error:
        save(status="failed", error_type=type(error).__name__, error=str(error))
        raise SystemExit(1)
    finally:
        if a.auth_file.exists():
            a.auth_file.unlink()


if __name__ == "__main__":
    main()

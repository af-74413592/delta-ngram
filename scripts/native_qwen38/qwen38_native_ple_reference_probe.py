"""Thin, CPU-only CLI for a previously captured native PLE reference snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--delta-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--visibility-only", action="store_true",
                        help="Forward precision observations only; requires no cotangents.")
    parser.add_argument("--zero-snapshot", type=Path,
                        help="Matching actual zero-delta native forward, for baseline/effect comparisons.")
    options = parser.parse_args()
    if not 1 <= options.threads <= 64:
        raise ValueError("CPU threads must be in [1,64]")
    if options.zero_snapshot is not None and not options.visibility_only:
        raise ValueError("--zero-snapshot is for --visibility-only")
    options.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(options.delta_repo.resolve() / "src"))
    import torch
    import delta_ngram.qwen38_ple_reference as reference

    torch.set_num_threads(options.threads)
    started = time.monotonic()
    identity = {
        "snapshot_path": str(options.snapshot.resolve()),
        "snapshot_sha256": digest(options.snapshot),
        "reference_source_sha256": digest(Path(reference.__file__)),
        "cli_source_sha256": digest(Path(__file__)),
        "torch_version": torch.__version__, "cpu_threads": options.threads,
        "visibility_only": options.visibility_only,
        "zero_snapshot_sha256": digest(options.zero_snapshot) if options.zero_snapshot else None,
    }
    (options.output / "STARTED.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8")

    def progress(row):
        row["elapsed_seconds"] = time.monotonic() - started
        with (options.output / "progress.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    snapshot = torch.load(options.snapshot, map_location="cpu", weights_only=True)
    zero_snapshot = (torch.load(options.zero_snapshot, map_location="cpu", weights_only=True)
                     if options.zero_snapshot else None)
    result = (reference.run_visibility_audit(snapshot, zero_snapshot=zero_snapshot)
              if options.visibility_only else reference.run_reference_audit(snapshot, progress=progress))
    result.update(identity)
    result["elapsed_seconds"] = time.monotonic() - started
    temporary = options.output / "COMPLETE.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(options.output / "COMPLETE.json")
    print(json.dumps({
        "status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
        "native_vjp_relative_l2_error": result.get("native_embedding_vjp_vs_fp64_reference", {}).get("relative_l2_error"),
        "quantization": result["bf16_overlay_quantization"]["overall"],
    }), flush=True)
    return 0 if result["status"] in ("reference_fd_passed", "visibility_observations_complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())

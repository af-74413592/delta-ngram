"""Run raw training with recorded deterministic CUDA settings and Git identity."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    # This must be established before importing torch or creating a CUDA context.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    import torch
    from delta_ngram.config import ExperimentConfig
    from delta_ngram.raw_lm import train_raw

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    config = ExperimentConfig.load(args.config)
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True).strip()
    if status:
        raise RuntimeError("Commit or restore repository changes before a recorded training run")
    output = Path(config.values["run"]["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    launch_path = output.with_name(output.name + ".launch.json")
    launch = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "config_path": str(config.source),
        "config_sha256": hashlib.sha256(Path(config.source).read_bytes()).hexdigest(),
        "output_dir": str(output),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "threads": torch.get_num_threads(),
        "source_sha256": {
            str(path.relative_to(repository / "src")): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((repository / "src/delta_ngram").rglob("*.py"))
        },
    }
    launch_path.parent.mkdir(parents=True, exist_ok=True)
    with launch_path.open("x", encoding="utf-8") as stream:
        json.dump(launch, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"event": "launch", **launch}, allow_nan=False), flush=True)
    result = train_raw(config)
    print(json.dumps({"event": "finished", "output_dir": str(output),
                      "stop_reason": result["stop_reason"], "complete": result["complete"],
                      "loss_tokens": result["counters"]["loss_tokens"],
                      "unspent_loss_tokens": result["unspent_loss_tokens"]}, allow_nan=False), flush=True)
    # A nonzero unspendable tail remains explicit in summary.json and exit status.
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Build an audited M1 prefix, or train a separate alpha variant on it."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-prefix", help="audit actual first-5M M1 optimizer prefix")
    build.add_argument("--config", required=True)
    build.add_argument("--m1-checkpoint", required=True)
    build.add_argument("--m1-metrics", required=True)
    build.add_argument("--writer-init", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--nominal-loss-tokens", type=int, default=5_000_000)
    build.add_argument("--threads", type=int, default=8)
    train = sub.add_parser("train", help="independent AdamW + alpha, exact M1 batches")
    train.add_argument("--config", required=True)
    train.add_argument("--variant", required=True,
                       choices=("fixed_099", "fixed_095", "learned_global_099", "evidence_block_099"))
    train.add_argument("--writer-init", required=True)
    train.add_argument("--prefix-manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    train.add_argument("--stop-after-steps", type=int)
    train.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    # Match M1's deterministic entry point before importing torch or CUDA.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch
    from delta_ngram.alpha_training import build_m1_prefix, train_alpha
    from delta_ngram.config import ExperimentConfig
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    values = vars(args).copy()
    command = values.pop("command")
    values.pop("threads")
    config = ExperimentConfig.load(values.pop("config"))
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).strip()
    if status:
        raise RuntimeError("Commit or restore repository changes before a recorded alpha run")
    output = Path(args.output)
    if output.exists() and (output.is_file() or any(output.iterdir())):
        raise FileExistsError(f"use a fresh output path: {output}")
    launch_path = output.with_name(output.name + ".launch.json")
    launch = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "command": command, "arguments": vars(args), "python": platform.python_version(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"], "threads": torch.get_num_threads(),
        "config_sha256": hashlib.sha256(config.source.read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(repository)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((repository / "src/delta_ngram").rglob("*.py"))}}
    launch_path.parent.mkdir(parents=True, exist_ok=True)
    with launch_path.open("x", encoding="utf-8") as handle:
        json.dump(launch, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"event": "launch", **launch}, allow_nan=False), flush=True)
    result = build_m1_prefix(config, **values) if command == "build-prefix" else train_alpha(config, **values)
    if command == "build-prefix":
        result = {k: result[k] for k in ("format", "stop_step", "actual_loss_tokens", "manifest_sha256")}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

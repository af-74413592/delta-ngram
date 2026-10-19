"""Validate a fixed news manifest, or score it with one frozen writer checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--config", help="Existing checkpoint-compatible experiment YAML")
    parser.add_argument("--checkpoint", help="Fixed writer checkpoint; never trained by this command")
    parser.add_argument("--split", choices=("dev", "final_holdout"), default="dev")
    parser.add_argument("--output", help="New result JSON path; existing files are not overwritten")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from delta_ngram.news_eval import evaluate_news, validate_manifest

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    audit = validate_manifest(manifest)
    if args.validate_only:
        print(json.dumps({"status": "valid", **audit}, allow_nan=False))
        return 0
    if not all((args.config, args.checkpoint, args.output)):
        parser.error("evaluation requires --config, --checkpoint, and --output")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Keep previous evaluation results immutable: {output}")
    from delta_ngram.checkpoints import load_checkpoint, restore_writer
    from delta_ngram.config import ExperimentConfig
    from delta_ngram.train import create_runtime

    config = ExperimentConfig.load(args.config)
    if config.values["method"]["name"] != "forward":
        raise ValueError("only a learned forward writer is supported")
    model = manifest["model"]
    if any(config.values["model"][key] != model[key] for key in ("name_or_path", "revision")):
        raise ValueError("manifest model/revision differs from the configured frozen reader")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = load_checkpoint(checkpoint)
    method, _ = create_runtime(config)
    restore_writer(method, payload, config)
    resolved = getattr(method.reader.model.config, "_commit_hash", None)
    if resolved and resolved != model["revision"]:
        raise ValueError("manifest revision must be the resolved immutable model commit")
    repo = Path(__file__).resolve().parents[1]
    metadata = {"path": str(checkpoint), "sha256": checkpoint_hash, "step": payload["step"],
                "protocol": payload["protocol"], "model": model,
                "training_loss_tokens": payload["config"].get("training_loss_tokens"),
                "raw_identity": payload["config"].get("raw_identity"),
                "resolved_model_commit": resolved, "smoke": config.values["model"]["smoke"]}
    result = evaluate_news(method, manifest, split=args.split, checkpoint_metadata=metadata,
                           progress_callback=lambda row: print(json.dumps({"progress": row}), flush=True))
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != checkpoint_hash:
        raise RuntimeError("writer checkpoint changed during evaluation")
    result["run"] = {"created_utc": datetime.now(timezone.utc).isoformat(),
                     "config": config.values, "manifest_path": str(Path(args.manifest).resolve()),
                     "evaluation_git_commit": subprocess.check_output(
                         ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                     "evaluation_git_dirty": bool(subprocess.check_output(
                         ["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()),
                     "torch_version": str(torch.__version__), "device": str(method.device)}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create prevents an accidental replacement of a completed result.
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "manifest_sha256": audit["manifest_sha256"],
                      "split": args.split, "summary_by_kind": result["summary_by_kind"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

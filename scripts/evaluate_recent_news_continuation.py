"""Evaluate a frozen checkpoint on DEV original-news continuation pairs.

All article distances and self/empty/fixed-donor arms are always evaluated.
This command never trains, constructs an optimizer, or writes query targets to
memory. A registry SHA256 is mandatory in effect: the default pins the frozen
16-pair engineering dataset; an alternate archive must provide its own hash.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
FROZEN_DEV_SHA256 = "708e2ff104ff4e0541ac6c0092026b5921a2632e04c049422f54866a890cd667"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_identity():
    return {"commit": subprocess.check_output(
        ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(
            ["git", "-C", str(REPOSITORY), "status", "--porcelain"], text=True).strip())}


def evaluate_checkpoint(config, checkpoint_path, registry_path, *, registry_sha256=FROZEN_DEV_SHA256,
                        bootstrap_seed=20261019, bootstrap_samples=10000,
                        require_clean_git=True, progress_callback=None,
                        annotations_path=None, annotations_sha256=None):
    import torch
    import delta_ngram.recent_news_evaluation as evaluation
    from delta_ngram.checkpoints import load_checkpoint, restore_writer
    from delta_ngram.raw_lm import _json_hash, tokenizer_identity
    from delta_ngram.train import create_runtime

    values = config.values
    if values["method"]["name"] != "forward" or values["reader"]["backend"] != "external_hash":
        raise ValueError("continuation requires a learned forward external_hash writer")
    if not re.fullmatch(r"[a-f0-9]{64}", registry_sha256 or ""):
        raise ValueError("registry_sha256 must be a 64-hex file hash")
    smoke = values["model"]["smoke"]
    revision = values["model"]["revision"]
    if not smoke and not re.fullmatch(r"[a-f0-9]{40}", revision or ""):
        raise ValueError("non-smoke continuation requires an immutable model commit")
    if not smoke and (annotations_path is None or annotations_sha256 is None):
        raise ValueError("recorded continuation requires frozen prefix annotations and their file SHA256")
    if (annotations_path is None) != (annotations_sha256 is None):
        raise ValueError("prefix annotations path and SHA256 must be supplied together")
    if values["raw"]["block_size"] != 512 or values["raw"]["prefix_tokens"] != 128:
        raise ValueError("continuation config must use raw.block_size=512 and raw.prefix_tokens=128")
    recorded = require_clean_git and not smoke
    git_before = git_identity()
    if recorded and git_before["dirty"]:
        raise RuntimeError("Commit repository changes before a recorded continuation evaluation")
    checkpoint_path, registry_path = Path(checkpoint_path).resolve(), Path(registry_path).resolve()
    config_hash = file_sha256(config.source)
    if file_sha256(registry_path) != registry_sha256:
        raise ValueError("registry file SHA256 mismatch")
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    payload = load_checkpoint(io.BytesIO(checkpoint_bytes))
    if payload["config"].get("requires_alpha_payload"):
        raise ValueError("alpha writer requires its paired alpha.pt; use evaluate_alpha_checkpoint.py")
    del checkpoint_bytes
    started = time.monotonic()
    method, _ = create_runtime(config)
    restore_writer(method, payload, config)
    resolved = getattr(method.reader.model.config, "_commit_hash", None)
    if not smoke and resolved != revision:
        raise ValueError("runtime model commit differs from pinned configured revision")
    tokenizer = tokenizer_identity(method)
    fingerprint = "sha256:" + _json_hash(tokenizer)
    old = payload["config"]
    if old.get("tokenizer_fingerprint") is not None and old["tokenizer_fingerprint"] != fingerprint:
        raise ValueError("checkpoint tokenizer fingerprint differs from runtime")
    old_tokenizer = old.get("raw_identity", {}).get("tokenizer")
    if old_tokenizer is not None and old_tokenizer != tokenizer:
        raise ValueError("checkpoint raw tokenizer identity differs from runtime")
    tokenizer_source = values["model"].get("tokenizer_name_or_path") or values["model"]["name_or_path"]
    tokenizer_revision = values["model"].get("tokenizer_revision") or revision
    prepared = evaluation.prepare_continuation(registry_path, method, tokenizer_source=tokenizer_source,
        tokenizer_revision=tokenizer_revision, expected_sha256=registry_sha256)
    prefix_annotations = (evaluation.prepare_prefix_annotations(annotations_path, prepared, method,
                          expected_sha256=annotations_sha256) if annotations_path is not None else None)
    if progress_callback is not None:
        progress_callback({"stage": "prepared", "families": prepared.manifest["independent_families"],
            "target_tokens_per_distance": prepared.manifest["target_tokens_per_distance"],
            "registry_file_sha256": registry_sha256, "tokenizer_fingerprint": fingerprint})
    with torch.no_grad():
        result = evaluation.evaluate_continuation(method, prepared, bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples, progress_callback=progress_callback,
            prefix_annotations=prefix_annotations)
    for path, expected in ((config.source, config_hash), (checkpoint_path, checkpoint_hash),
                           (registry_path, registry_sha256)):
        if file_sha256(path) != expected:
            raise RuntimeError(f"input file changed during continuation evaluation: {path}")
    git_after = git_identity()
    if recorded and (git_after["dirty"] or git_after["commit"] != git_before["commit"]):
        raise RuntimeError("repository changed during recorded continuation evaluation")
    model = {"name_or_path": values["model"]["name_or_path"], "revision": revision,
        "resolved_commit": resolved, "smoke": smoke,
        "dtype": str(next(method.reader.model.parameters()).dtype), "tokenizer_fingerprint": fingerprint,
        "matches_registry_reader": all(values["model"][key] == prepared.registry["model"][key]
                                       for key in ("name_or_path", "revision"))}
    result.update(checkpoint={"path": str(checkpoint_path), "sha256": checkpoint_hash,
        "step": payload["step"], "protocol": payload["protocol"],
        "training_loss_tokens": old.get("training_loss_tokens"),
        "writer_tensor_sha256": result["writer_tensor_sha256"],
        "checkpoint_tokenizer_fingerprint_verified": old.get("tokenizer_fingerprint") is not None},
        model=model, model_identity_sha256=_json_hash(model),
        run={"created_utc": datetime.now(timezone.utc).isoformat(), "config": values,
             "config_path": str(config.source), "config_sha256": config_hash,
             "resolved_config_sha256": _json_hash(values), "git_before": git_before, "git_after": git_after,
             "clean_git_enforced": recorded, "script_sha256": file_sha256(__file__),
             "evaluator_sha256": file_sha256(evaluation.__file__), "torch_version": str(torch.__version__),
             "device": str(method.device), "threads": torch.get_num_threads(),
             "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
             "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
             "elapsed_seconds": time.monotonic() - started})
    json.dumps(result, allow_nan=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--registry-sha256", default=FROZEN_DEV_SHA256)
    parser.add_argument("--annotations", help="Frozen per-pair 128-token prefix restatement annotations; required outside smoke")
    parser.add_argument("--annotations-sha256", help="Expected annotation file SHA256")
    parser.add_argument("--output", required=True, help="New JSON result; existing results are never overwritten")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bootstrap-seed", type=int, default=20261019)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.bootstrap_samples < 1:
        parser.error("threads and bootstrap-samples must be positive")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Keep previous evaluation results immutable: {output}")
    import torch
    from delta_ngram.config import ExperimentConfig

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    result = evaluate_checkpoint(ExperimentConfig.load(args.config), args.checkpoint, args.registry,
        registry_sha256=args.registry_sha256, bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        annotations_path=args.annotations, annotations_sha256=args.annotations_sha256,
        progress_callback=lambda row: print(json.dumps(row), flush=True))
    result["run"]["requested_threads"] = args.threads
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "complete": True,
        "checkpoint_sha256": result["checkpoint"]["sha256"], "manifest_sha256": result["manifest_sha256"],
        "families": result["independent_families"], "retention_curve": result["retention_curve"]}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

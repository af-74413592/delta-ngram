"""Evaluate the immutable M1 first-5M D1 writer on frozen DEV revision queries.

Uses the same candidate scorer, old/old-plus-new stages, fixed donor controls,
and k=0/1/4/8/16 distances as alpha revision evaluation. No optimizer is created
and no training corpus is opened. The caller pins checkpoint, registry and
annotation file hashes; the M1 prefix pins the configuration bytes and budget.
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
RUNNER_PROTOCOL = "delta_ngram_frozen_d1_revision_checkpoint_v1"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_identity():
    return {"commit": subprocess.check_output(
        ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(
            ["git", "-C", str(REPOSITORY), "status", "--porcelain"], text=True).strip())}


def load_frozen_inputs(config, checkpoint_path, *, checkpoint_sha256, prefix_manifest):
    """Metadata-only preflight, including exact configuration file bytes."""
    from delta_ngram.alpha_evaluation import _canonical
    from delta_ngram.checkpoints import load_checkpoint
    from delta_ngram.raw_lm import _json_hash

    if not re.fullmatch(r"[a-f0-9]{64}", checkpoint_sha256 or ""):
        raise ValueError("checkpoint_sha256 must be an explicit 64-hex file hash")
    checkpoint_path, prefix_path = Path(checkpoint_path).resolve(), Path(prefix_manifest).resolve()
    checkpoint_bytes = checkpoint_path.read_bytes()
    if hashlib.sha256(checkpoint_bytes).hexdigest() != checkpoint_sha256:
        raise ValueError("D1 checkpoint file SHA256 mismatch")
    payload = load_checkpoint(io.BytesIO(checkpoint_bytes))
    old = payload["config"]
    if old.get("requires_alpha_payload") or payload["protocol"] != "delta_ngram_raw_lm_v1":
        raise ValueError("D1 baseline requires an original raw-LM writer, not an alpha writer")
    prefix_hash = file_sha256(prefix_path)
    prefix = json.loads(prefix_path.read_bytes())
    declared = prefix.pop("manifest_sha256", None)
    if (prefix.get("format") != "delta_ngram_m1_optimizer_prefix_v1"
            or declared != _json_hash(_canonical(prefix))):
        raise ValueError("invalid frozen M1 optimizer prefix format/hash")
    config_hash = file_sha256(config.source)
    if config_hash != prefix["config_sha256"]:
        raise ValueError("configuration file bytes differ from the frozen M1 prefix")
    identity = dict(old.get("raw_identity", {}))
    identity.pop("validation", None)
    if (_canonical(identity) != _canonical(prefix["m1_identity"])
            or prefix["full_loss_token_budget"] != config.values["raw"]["loss_tokens"]):
        raise ValueError("D1 checkpoint/configuration differs from the frozen M1 prefix identity")
    if (payload["step"] != prefix["stop_step"]
            or old.get("training_loss_tokens") != prefix["actual_loss_tokens"]
            or prefix["stop_step"] != len(prefix["groups"])):
        raise ValueError("D1 checkpoint is not the complete registered M1 prefix endpoint")
    if not config.values["model"]["smoke"] and prefix["nominal_loss_tokens"] != 5_000_000:
        raise ValueError("D1 baseline requires the nominal first-5M M1 prefix")
    manifest = {"path": str(prefix_path), "file_sha256": prefix_hash, "manifest_sha256": declared,
        "config_sha256": config_hash, "m1_training_checkpoint_sha256": prefix["m1_checkpoint_sha256"],
        "writer_init_sha256": prefix["writer_init_sha256"], "stop_step": prefix["stop_step"],
        "nominal_loss_tokens": prefix["nominal_loss_tokens"], "actual_loss_tokens": prefix["actual_loss_tokens"],
        "full_loss_token_budget": prefix["full_loss_token_budget"],
        "config_bytes_and_complete_prefix_verified": True,
        "export_binding": "Explicit caller-pinned writer file SHA256 plus prefix step, target count and raw identity; training snapshot not opened."}
    return payload, manifest, ((checkpoint_path, checkpoint_sha256), (prefix_path, prefix_hash),
                               (config.source, config_hash))


def evaluate_checkpoint(config, checkpoint_path, *, checkpoint_sha256, prefix_manifest,
        registry_path, registry_sha256, annotations_path, annotations_sha256,
        bootstrap_seed=20261019, bootstrap_samples=10000, require_clean_git=True,
        progress_callback=None):
    import torch
    import delta_ngram.alpha_evaluation as revision_evaluation
    import delta_ngram.recent_news_evaluation as recent
    from delta_ngram.checkpoints import restore_writer
    from delta_ngram.p2_initialization import writer_tensor_sha256
    from delta_ngram.raw_lm import _json_hash, tokenizer_identity
    from delta_ngram.train import create_runtime

    values, raw = config.values, config.values["raw"]
    smoke, revision = values["model"]["smoke"], values["model"]["revision"]
    if (values["method"]["name"] != "forward" or values["method"]["operator"] != "D1"
            or values["reader"]["backend"] != "external_hash"):
        raise ValueError("D1 revision requires a learned forward D1 external_hash writer")
    if raw["block_size"] != 512 or raw["prefix_tokens"] != 128:
        raise ValueError("D1 revision uses the registered 512-token block and 128-token prefix configuration")
    if not smoke and not re.fullmatch(r"[a-f0-9]{40}", revision or ""):
        raise ValueError("non-smoke evaluation requires an immutable model commit")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    for name, value in (("registry_sha256", registry_sha256), ("annotations_sha256", annotations_sha256)):
        if not re.fullmatch(r"[a-f0-9]{64}", value or ""):
            raise ValueError(f"{name} must be an explicit 64-hex file hash")
    recorded, git_before = require_clean_git and not smoke, git_identity()
    if recorded and git_before["dirty"]:
        raise RuntimeError("Commit repository changes before a recorded D1 revision evaluation")
    started = time.monotonic()
    payload, prefix, files = load_frozen_inputs(config, checkpoint_path,
        checkpoint_sha256=checkpoint_sha256, prefix_manifest=prefix_manifest)
    for path, expected, label in ((registry_path, registry_sha256, "registry"),
                                  (annotations_path, annotations_sha256, "annotations")):
        if file_sha256(path) != expected:
            raise ValueError(f"{label} file SHA256 mismatch")
    method, _ = create_runtime(config)
    restore_writer(method, payload, config)
    resolved = getattr(method.reader.model.config, "_commit_hash", None)
    if not smoke and resolved != revision:
        raise ValueError("runtime model commit differs from the pinned configured revision")
    tokenizer = tokenizer_identity(method)
    fingerprint = "sha256:" + _json_hash(tokenizer)
    old = payload["config"]
    if (old.get("tokenizer_fingerprint") != fingerprint
            or old.get("raw_identity", {}).get("tokenizer") != tokenizer):
        raise ValueError("D1 checkpoint tokenizer identity differs from the runtime")
    prepared = recent.prepare_continuation(registry_path, method,
        tokenizer_source=values["model"].get("tokenizer_name_or_path") or values["model"]["name_or_path"],
        tokenizer_revision=values["model"].get("tokenizer_revision") or revision,
        expected_sha256=registry_sha256)
    labels = recent.prepare_prefix_annotations(annotations_path, prepared, method,
                                               expected_sha256=annotations_sha256)
    if progress_callback is not None:
        progress_callback({"stage": "revision_prepared", "registry_families": len(prepared.registry["pairs"]),
            "revision_families": sum(row.get("revision", {}).get("eligible", False) for row in labels.by_pair.values())})
    result = revision_evaluation.evaluate_revision(method, prepared, labels, backend=None,
        bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples, progress_callback=progress_callback)
    for path, expected in files:
        if file_sha256(path) != expected:
            raise RuntimeError(f"D1 evaluation input changed: {path}")
    prepared.verify_files()
    labels.verify_file()
    git_after = git_identity()
    if recorded and (git_after["dirty"] or git_after["commit"] != git_before["commit"]):
        raise RuntimeError("repository changed during recorded D1 revision evaluation")
    model = {"name_or_path": values["model"]["name_or_path"], "revision": revision,
        "resolved_commit": resolved, "smoke": smoke, "tokenizer_fingerprint": fingerprint,
        "dtype": str(next(method.reader.model.parameters()).dtype),
        "matches_registry_reader": all(values["model"][key] == prepared.registry["model"][key]
                                       for key in ("name_or_path", "revision"))}
    result.update(runner_protocol=RUNNER_PROTOCOL, m1_prefix=prefix, model=model,
        model_identity_sha256=_json_hash(model),
        checkpoint={"path": str(Path(checkpoint_path).resolve()), "sha256": checkpoint_sha256,
            "step": payload["step"], "training_loss_tokens": old["training_loss_tokens"],
            "protocol": payload["protocol"], "writer_tensor_sha256": writer_tensor_sha256(payload["writer_state_dict"]),
            "checkpoint_tokenizer_fingerprint_verified": True},
        manifest={"continuation": prepared.manifest, "prefix_annotations": labels.manifest},
        run={"created_utc": datetime.now(timezone.utc).isoformat(), "config": values,
            "config_path": str(config.source), "config_sha256": prefix["config_sha256"],
            "resolved_config_sha256": _json_hash(revision_evaluation._canonical(values)),
            "git_before": git_before, "git_after": git_after, "clean_git_enforced": recorded,
            "script_sha256": file_sha256(__file__), "evaluator_sha256": file_sha256(revision_evaluation.__file__),
            "torch_version": str(torch.__version__), "device": str(method.device), "threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "elapsed_seconds": time.monotonic() - started})
    result["manifest_sha256"] = _json_hash(result["manifest"])
    result["integrity_checks"].update(checkpoint_config_prefix_files_unchanged=True,
                                     reader_weight_content_hash_verified=False)
    json.dumps(result, allow_nan=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "checkpoint-sha256", "prefix-manifest", "registry", "registry-sha256",
                 "annotations", "annotations-sha256", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20261019)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.bootstrap_samples < 1:
        parser.error("threads and bootstrap-samples must be positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Keep prior evaluation results immutable: {output}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch
    from delta_ngram.config import ExperimentConfig
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    result = evaluate_checkpoint(ExperimentConfig.load(args.config), args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256, prefix_manifest=args.prefix_manifest,
        registry_path=args.registry, registry_sha256=args.registry_sha256,
        annotations_path=args.annotations, annotations_sha256=args.annotations_sha256,
        bootstrap_seed=args.bootstrap_seed, bootstrap_samples=args.bootstrap_samples,
        progress_callback=lambda row: print(json.dumps(row), flush=True))
    result["run"]["requested_threads"] = args.threads
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "complete": True,
        "revision_families": result["independent_families"], "checkpoint_sha256": args.checkpoint_sha256}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

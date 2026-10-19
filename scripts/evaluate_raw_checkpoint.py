"""Evaluate a frozen writer on the raw trainer's exact fixed development sample.

Self/empty and configured fixed donor controls are scored. This command never constructs an
optimizer, resumes training, or saves changed model parameters. The checkpoint
argument overrides run.checkpoint in the supplied training configuration.
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
PROTOCOL = "delta_ngram_frozen_raw_checkpoint_v1"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_identity():
    return {"commit": subprocess.check_output(
                ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip(),
            "dirty": bool(subprocess.check_output(
                ["git", "-C", str(REPOSITORY), "status", "--porcelain"], text=True).strip())}


def prepare_validation(config, method, *, include_donors=False):
    """Reproduce train_raw selection without materializing training windows.

    Training cells are checked by complete-document length only. Their windows
    and shuffled orders are unnecessary for this optimizer-free command.
    """
    from delta_ngram.data.raw_text import _weights, build_windows, load_documents
    from delta_ngram.raw_corpora import load_corpora, trajectory_pools, event_pool
    from delta_ngram.raw_lm import (MINIMUM_EVALUATION_GROUPS, VALIDATION_PROTOCOL,
                                    _json_hash, _validation_sample,
                                    select_validation_windows, tokenizer_identity)

    raw = config.values["raw"]
    if raw["dataset"] is None:
        raise ValueError("frozen raw evaluation requires raw.dataset")
    identity = tokenizer_identity(method)
    fingerprint = "sha256:" + _json_hash(identity)
    vocabulary_size = method.reader.model.get_input_embeddings().weight.shape[0]
    documents, events = load_corpora(raw, method.tokenizer, fingerprint, vocabulary_size=vocabulary_size)
    counts = tuple(raw["block_counts"])
    lengths = _weights(raw["length_weights"], counts, "length_weights")
    train = [document for document in documents if document.split == "train"]
    event_training = event_pool(raw, events, "train")
    available = sorted({document.source for document in train
                        if any(len(document.token_ids) >= count * raw["block_size"] for count in lengths)}
                       | {window.source for items in event_training.values() for window in items})
    sources = _weights(raw["source_weights"], available, "source_weights")
    conditional = raw.get("source_length_weights")
    active_lengths = {source: lengths if conditional is None else
                      _weights(conditional[source], counts, f"source_length_weights[{source}]")
                      for source in sources}
    for source, source_counts in active_lengths.items():
        for count in source_counts:
            if source == "news_event" and event_training.get(count):
                continue
            if not any(document.source == source and len(document.token_ids) >= count * raw["block_size"]
                       for document in train):
                raise ValueError(f"nonzero quota has no windows: blocks={count}, source={source}")
    window_options = {} if conditional is None else {
        "source_block_counts": {source: list(source_counts) for source, source_counts in active_lengths.items()}}
    pool = trajectory_pools(raw, documents, events, split="validation")
    windows = [window for count in counts for window in pool[count]]
    minimum = raw.get("eval_min_groups_per_source", 0)
    if conditional is not None or minimum:
        windows = [window for window in windows if window.source in sources
                   and (conditional is None or window.blocks in active_lengths[window.source])]
    if not windows:
        raise ValueError("raw.dataset needs a disjoint validation split with complete windows")
    fixed_donors, donor_excluded = None, []
    if raw.get("eval_donor"):
        from delta_ngram.raw_evaluation import donor_supported_pool
        windows, fixed_donors, donor_excluded = donor_supported_pool(windows, raw["eval_seed"])
    available_groups = len({window.group_id for window in windows})
    available_per_source = {source: len({window.group_id for window in windows if window.source == source})
                            for source in sources}
    windows = select_validation_windows(windows, limit=raw["eval_windows"], seed=raw["eval_seed"],
                                        min_groups_per_source=minimum, required_sources=sorted(sources))
    sample = _validation_sample(windows)
    protocol = {"name": VALIDATION_PROTOCOL, "selection": "group_first_round_robin",
                "group_key": "group_id", "seed": raw["eval_seed"], "window_limit": raw["eval_windows"],
                "bootstrap_samples": raw["eval_bootstrap_samples"], "minimum_groups": MINIMUM_EVALUATION_GROUPS,
                "evaluation_batching": "within_group", "batch_size": raw["batch_size"],
                "prefix_tokens": raw["prefix_tokens"]}
    if minimum:
        protocol.update(selection="source_quota_then_group_first_round_robin",
                        minimum_groups_per_source=minimum, required_sources=sorted(sources))
    if conditional is not None:
        protocol["source_length_weights"] = conditional
    if raw.get("eval_donor"):
        from delta_ngram.raw_evaluation import donor_manifest
        donor_sample = donor_manifest(windows, fixed_donors)
        protocol.update(conditions=["self", "empty", "donor"],
            donor_matching="same_source_same_length_different_family_fixed_development_v1",
            donor_pool="all_validation_windows; unsupported targets excluded before fixed group selection",
            donor_unsupported_windows=donor_excluded,
            donor_sample_sha256=_json_hash(donor_sample),
            macro_source_groups=raw.get("macro_source_groups"))
    if raw.get('event_dataset'):
        from delta_ngram.raw_evaluation import first_article_block_protocol
        protocol['later_article_first_block'] = first_article_block_protocol(raw['prefix_tokens'])
    metadata = {"validation_protocol": protocol,
                     "validation_protocol_sha256": _json_hash(protocol),
                     "validation_sample": sample, "validation_sample_sha256": _json_hash(sample),
                     "available_validation_groups": available_groups,
                     "selected_validation_groups": len({window.group_id for window in windows}),
                     "available_validation_groups_per_source": available_per_source,
                     "selected_validation_groups_per_source": {
                         source: len({window.group_id for window in windows if window.source == source})
                         for source in sources},
                     "tokenizer": identity, "tokenizer_fingerprint": fingerprint}
    return (windows, metadata, fixed_donors) if include_donors else (windows, metadata)


def _parameter_snapshot(module):
    return {name: (parameter, parameter._version) for name, parameter in module.named_parameters()}


def _check_parameters(module, before, label):
    after = dict(module.named_parameters())
    if (after.keys() != before.keys()
            or any(after[name] is not parameter or after[name]._version != version
                   for name, (parameter, version) in before.items())
            or any(parameter.requires_grad or parameter.grad is not None for parameter in after.values())):
        raise RuntimeError(f"{label} parameters changed, were replaced, or received gradients")


def evaluate_checkpoint(config, checkpoint_path, *, require_clean_git=True, progress_callback=None):
    """Return one immutable, evaluation-only result; tests may use a smoke model."""
    import torch
    from delta_ngram.checkpoints import load_checkpoint, restore_writer
    from delta_ngram.p2_initialization import writer_tensor_sha256
    from delta_ngram.raw_lm import _json_hash, evaluate_raw, tokenizer_identity
    from delta_ngram.train import create_runtime

    if config.values["method"]["name"] != "forward" or config.values["reader"]["backend"] != "external_hash":
        raise ValueError("frozen raw evaluation requires a learned forward external_hash writer")
    recorded = require_clean_git and not config.values["model"]["smoke"]
    git_before = git_identity()
    if recorded and git_before["dirty"]:
        raise RuntimeError("Commit repository changes before a recorded frozen evaluation")
    requested_revision = config.values["model"]["revision"]
    if not config.values["model"]["smoke"] and not re.fullmatch(r"[a-f0-9]{40}", requested_revision or ""):
        raise ValueError("non-smoke evaluation requires an immutable model commit")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    payload = load_checkpoint(io.BytesIO(checkpoint_bytes))
    del checkpoint_bytes
    config_hash = file_sha256(config.source)
    dataset_path = Path(config.values["raw"]["dataset"]).resolve() if config.values["raw"]["dataset"] else None
    if dataset_path is None:
        raise ValueError("frozen raw evaluation requires raw.dataset")
    dataset_hash = file_sha256(dataset_path)
    event_path = config.values["raw"].get("event_dataset")
    event_hash = file_sha256(event_path) if event_path else None
    started = time.monotonic()
    method, _ = create_runtime(config)
    restore_writer(method, payload, config)
    resolved = getattr(method.reader.model.config, "_commit_hash", None)
    if not config.values["model"]["smoke"] and resolved != requested_revision:
        raise ValueError("runtime model commit differs from the pinned configured revision")
    tokenizer = tokenizer_identity(method)
    fingerprint = "sha256:" + _json_hash(tokenizer)
    old = payload["config"]
    if old.get("tokenizer_fingerprint") is not None and old["tokenizer_fingerprint"] != fingerprint:
        raise ValueError("checkpoint tokenizer fingerprint differs from the runtime tokenizer")
    old_tokenizer = old.get("raw_identity", {}).get("tokenizer")
    if old_tokenizer is not None and old_tokenizer != tokenizer:
        raise ValueError("checkpoint raw tokenizer identity differs from the runtime tokenizer")
    windows, validation, fixed_donors = prepare_validation(config, method, include_donors=True)
    if progress_callback is not None:
        progress_callback({"stage": "prepared", "windows": len(windows),
                           "groups_per_source": validation["selected_validation_groups_per_source"],
                           "validation_sample_sha256": validation["validation_sample_sha256"]})
    method.eval()
    for module in (method.writer, method.reader.model):
        module.eval()
        module.requires_grad_(False)
        module.zero_grad(set_to_none=True)
    writer_before = {name: value.detach().clone() for name, value in method.writer.state_dict().items()}
    writer_versions, reader_versions = _parameter_snapshot(method.writer), _parameter_snapshot(method.reader.model)
    tensor_hash = writer_tensor_sha256(writer_before)
    raw = config.values["raw"]
    with torch.no_grad():
        evaluation = evaluate_raw(method, windows, prefix_tokens=raw["prefix_tokens"], batch_size=raw["batch_size"],
                                  bootstrap_seed=raw["eval_seed"], bootstrap_samples=raw["eval_bootstrap_samples"],
                                  report_per_source=True, report_donor=raw.get("eval_donor", False),
                                  macro_source_groups=raw.get("macro_source_groups"), fixed_donors=fixed_donors)
    writer_after = method.writer.state_dict()
    if writer_before.keys() != writer_after.keys() or any(
            before.shape != writer_after[name].shape or before.dtype != writer_after[name].dtype
            or before.device != writer_after[name].device or not torch.equal(before, writer_after[name])
            for name, before in writer_before.items()):
        raise RuntimeError("writer state_dict changed during frozen evaluation")
    _check_parameters(method.writer, writer_versions, "writer")
    _check_parameters(method.reader.model, reader_versions, "reader")
    for path, expected, label in ((checkpoint_path, checkpoint_hash, "checkpoint"),
                                  (dataset_path, dataset_hash, "dataset"), (config.source, config_hash, "config")):
        if file_sha256(path) != expected:
            raise RuntimeError(f"{label} file changed during frozen evaluation")
    if event_path and file_sha256(event_path) != event_hash:
        raise RuntimeError("event dataset file changed during frozen evaluation")
    git_after = git_identity()
    if recorded and (git_after["dirty"] or git_after["commit"] != git_before["commit"]):
        raise RuntimeError("repository changed during recorded frozen evaluation")
    model = {"name_or_path": config.values["model"]["name_or_path"], "revision": requested_revision,
             "resolved_commit": resolved, "smoke": config.values["model"]["smoke"],
             "dtype": str(next(method.reader.model.parameters()).dtype), "tokenizer_fingerprint": fingerprint}
    result = {**evaluation, **validation, "protocol": PROTOCOL, "complete": True,
              "conditions": ["self", "empty", "donor"] if raw.get("eval_donor") else ["self", "empty"],
              "evaluation_only": True, "optimizer_steps": 0, "backward_calls": 0,
              "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_hash,
                  "step": payload["step"], "protocol": payload["protocol"],
                  "training_loss_tokens": old.get("training_loss_tokens"),
                  "raw_identity": old.get("raw_identity"), "writer_tensor_sha256": tensor_hash},
              "dataset": {"path": str(dataset_path), "sha256": dataset_hash},
              **({"event_dataset": {"path": str(event_path), "sha256": event_hash}} if event_path else {}),
              "model": model, "model_identity_sha256": _json_hash(model),
              "integrity_checks": {"writer_state_dict_exact_equal": True,
                  "writer_parameter_objects_and_versions_unchanged": True,
                  "reader_parameter_objects_and_versions_unchanged": True,
                  "parameters_frozen_and_no_gradients": True,
                  "reader_content_hash_verified": False,
                  "checkpoint_dataset_config_files_unchanged": True,
                  "checkpoint_tokenizer_fingerprint_verified": old.get("tokenizer_fingerprint") is not None},
              "run": {"created_utc": datetime.now(timezone.utc).isoformat(), "config": config.values,
                  "config_path": str(config.source), "config_sha256": config_hash,
                  "resolved_config_sha256": _json_hash(config.values),
                  "git_before": git_before, "git_after": git_after, "clean_git_enforced": recorded,
                  "script_sha256": file_sha256(__file__), "torch_version": str(torch.__version__),
                  "device": str(method.device), "threads": torch.get_num_threads(),
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                  "elapsed_seconds": time.monotonic() - started},
              "limitations": ["Development-set evaluation; no final-test or pretraining-novelty claim.",
                  "Full-context comparison is a separate diagnostic; fixed donor is run only when configured.",
                  "Model identity SHA256 hashes identity metadata, not reader weight contents.",
                  "Reader checks do not detect every .data or buffer mutation; writer tensors are compared exactly."]}
    json.dumps(result, allow_nan=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New result JSON; never overwrites existing results")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    # These variables must be set before importing torch or initializing CUDA.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch
    from delta_ngram.config import ExperimentConfig

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Keep previous evaluation results immutable: {output}")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    result = evaluate_checkpoint(ExperimentConfig.load(args.config), args.checkpoint,
                                 progress_callback=lambda row: print(json.dumps(row), flush=True))
    result["run"]["requested_threads"] = args.threads
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
    print(json.dumps({"output": str(output), "complete": True,
                      "checkpoint_sha256": result["checkpoint"]["sha256"],
                      "validation_sample_sha256": result["validation_sample_sha256"],
                      "source_macro_delta_empty": result["source_macro_delta_empty"]}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

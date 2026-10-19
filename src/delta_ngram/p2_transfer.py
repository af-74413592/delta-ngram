"""Explicit research-only writer transfer onto one frozen destination reader.

The destination model performs BOTH evidence encoding and memory readout. No
source model, optimizer, or memory state is installed. Normal checkpoint loading
retains its original model-identity checks.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import torch

from .checkpoints import load_checkpoint, restore_writer
from .data.raw_text import TrajectoryWindow
from .p2_initialization import validate_writer_tensors, writer_tensor_sha256
from .operators import WriteOperator
from .raw_lm import (PROTOCOL as RAW_PROTOCOL, VALIDATION_PROTOCOL, _identity,
                     _json_hash, _validation_sample, evaluate_raw, summarize_group_losses, tokenizer_identity)
from .reader import ExternalHashAdapter


PROTOCOL = "delta_ngram_p2_research_transfer_v1"


def _json_value(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _same(left, right, label):
    if _json_value(left) != _json_value(right):
        raise ValueError(f"P2 {label} differs")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_raw_payload(payload, expected_identity):
    metadata = payload["config"]
    identity = metadata.get("raw_identity")
    if payload.get("protocol") != RAW_PROTOCOL or not isinstance(identity, dict):
        raise ValueError("P2 requires an identified raw-LM writer checkpoint")
    if expected_identity is None:
        raise ValueError("P2 requires the selected training run's raw identity")
    _same(identity, expected_identity, "selected training run identity")
    package = metadata["package_config"]
    required = {"protocol", "seed", "dataset_sha256", "tokenizer", "model",
                "resolved_model_commit", "device_type", "dtype", "reader", "writer",
                "method", "optimizer", "raw", "validation"}
    if not required.issubset(identity) or identity["protocol"] != RAW_PROTOCOL:
        raise ValueError("P2 checkpoint has incomplete raw identity")
    for section in ("reader", "writer", "method", "seed"):
        _same(identity[section], package[section], f"checkpoint package.{section}")
    _same(identity["model"], {key: package["model"][key]
                             for key in ("name_or_path", "revision", "smoke")}, "checkpoint model")
    _same(identity["raw"], {key: package["raw"][key] for key in identity["raw"]}, "checkpoint raw recipe")
    _same(identity["optimizer"], {"lr": package["train"]["lr"],
                                  "gradient_clip": package["train"]["gradient_clip"],
                                  "name": "AdamW", "weight_decay": 0.0}, "checkpoint optimizer")
    for key, value in (("model", identity["model"]["name_or_path"]),
                       ("revision", identity["model"]["revision"]),
                       ("smoke", identity["model"]["smoke"]), ("seed", identity["seed"]),
                       ("buckets", identity["reader"]["buckets"]),
                       ("layer_index", identity["reader"]["layer_index"]),
                       ("orders", identity["reader"]["orders"]),
                       ("hidden_dim", identity["writer"]["hidden_dim"]),
                       ("resolved_model_commit", identity["resolved_model_commit"])):
        _same(metadata.get(key), value, f"checkpoint metadata.{key}")
    fingerprint = "sha256:" + _json_hash(identity["tokenizer"])
    _same(metadata.get("tokenizer_fingerprint"), fingerprint, "checkpoint tokenizer fingerprint")
    tokens = metadata.get("training_loss_tokens", 0 if payload["step"] == 0 else None)
    if (type(tokens) is not int or tokens < 0 or
            (payload["step"] == 0) != (tokens == 0)):
        raise ValueError("P2 checkpoint needs valid training_loss_tokens (only step zero may omit it)")
    return identity, tokens


def load_research_transfer_checkpoint(method, config, source_checkpoint, destination_checkpoint, *,
                                      expected_source_identity, expected_destination_identity,
                                      research_transfer=False):
    """Validate both identities, then load only source writer tensors on I.

    Source/destination must be from the same step and actual target-token budget.
    The opt-in never changes ``restore_writer`` or the original saved identities.
    """
    if research_transfer is not True:
        raise ValueError("P2 cross-model loading requires explicit research_transfer=True")
    source_path, destination_path = Path(source_checkpoint).resolve(), Path(destination_checkpoint).resolve()
    source_bytes, destination_bytes = source_path.read_bytes(), destination_path.read_bytes()
    source, destination = load_checkpoint(io.BytesIO(source_bytes)), load_checkpoint(io.BytesIO(destination_bytes))
    source_identity, source_tokens = _validate_raw_payload(source, expected_source_identity)
    destination_identity, destination_tokens = _validate_raw_payload(destination, expected_destination_identity)
    _same(source["step"], destination["step"], "training step")
    _same(source_tokens, destination_tokens, "actual training loss tokens")
    current_tokenizer = tokenizer_identity(method)
    current = _identity(config, method, destination_identity["dataset_sha256"], current_tokenizer)
    current["validation"] = destination_identity["validation"]
    _same(current, destination_identity, "destination runtime/training recipe")
    raw = config.values["raw"]
    validation = {"name": VALIDATION_PROTOCOL, "selection": "group_first_round_robin",
                  "group_key": "group_id", "seed": raw["eval_seed"], "window_limit": raw["eval_windows"],
                  "bootstrap_samples": raw["eval_bootstrap_samples"], "minimum_groups": 64,
                  "evaluation_batching": "within_group", "batch_size": raw["batch_size"],
                  "prefix_tokens": raw["prefix_tokens"]}
    _same(destination_identity["validation"].get("validation_protocol_sha256"),
          _json_hash(validation), "fixed validation protocol")
    # Model identity is the one intentional difference. All of these fields
    # affect comparability or the numerical/address interface and must match.
    compatible_fields = ("protocol", "seed", "dataset_sha256", "tokenizer", "device_type", "dtype",
                         "reader", "writer", "method", "optimizer", "raw", "validation")
    for field in compatible_fields:
        _same(source_identity[field], destination_identity[field], f"transfer {field}")
    _same(source["config"]["width"], method.reader.width, "transfer width")
    if method.reader.backend != "external_hash" or config.values["method"]["name"] != "forward":
        raise ValueError("P2 requires the learned external-hash forward interface")
    adapter = ExternalHashAdapter(method.reader.buckets, method.reader.orders).config()
    _same(method.reader.adapter.config(), adapter, "address/readout adapter")
    operator = WriteOperator(config.values["method"]["operator"]).name
    _same(method.writer.operator.name, operator, "runtime writer operator")
    validate_writer_tensors(method.writer, destination["writer_state_dict"])
    validate_writer_tensors(method.writer, source["writer_state_dict"])
    source_fingerprint = writer_tensor_sha256(source["writer_state_dict"])
    if source["step"] == 0:
        _same(source_fingerprint, writer_tensor_sha256(destination["writer_state_dict"]), "step-zero initialization")
    # Keep the ordinary loader's full identity/revision validation for I.
    restore_writer(method, destination, config)
    destination_fingerprint = writer_tensor_sha256(method.writer.state_dict())
    method.writer.load_state_dict(source["writer_state_dict"], strict=True)
    _same(writer_tensor_sha256(method.writer.state_dict()), source_fingerprint, "loaded writer fingerprint")
    method.eval()
    method.writer.requires_grad_(False)
    method.reader.model.requires_grad_(False)
    method.writer.zero_grad(set_to_none=True)
    method.reader.model.zero_grad(set_to_none=True)

    def checkpoint_metadata(path, payload, identity, tokens, fingerprint, content):
        return {"checkpoint_path": str(path), "checkpoint_sha256": hashlib.sha256(content).hexdigest(),
                "step": payload["step"], "training_loss_tokens": tokens,
                "writer_tensor_sha256": fingerprint, "raw_identity": _json_value(identity)}

    return {"protocol": PROTOCOL, "research_transfer": True,
            "condition": "B_to_I", "evidence_encoder": "destination", "readout": "destination",
            "source": checkpoint_metadata(source_path, source, source_identity, source_tokens, source_fingerprint, source_bytes),
            "destination": checkpoint_metadata(destination_path, destination, destination_identity,
                                                destination_tokens, destination_fingerprint, destination_bytes),
            "compatibility": {"checked_fields": list(compatible_fields), "tensor_schema": "strict keys/shape/dtype/finite",
                              "width": method.reader.width, "adapter": adapter,
                              "validation_protocol": validation},
            "fingerprint_scope": "writer tensors and exported checkpoint files; model revisions remain identities, not model-weight hashes"}


def fixed_validation_windows(documents, manifest, *, expected_identity, block_size):
    """Reconstruct exactly the archived I target cohort; never resample it."""
    for key in ("dataset_sha256", "tokenizer"):
        _same(manifest.get(key), expected_identity[key], f"validation manifest {key}")
    fingerprint = "sha256:" + _json_hash(expected_identity["tokenizer"])
    _same(manifest.get("tokenizer_fingerprint"), fingerprint, "validation tokenizer fingerprint")
    for key in ("validation_protocol_sha256", "validation_sample_sha256"):
        _same(manifest.get(key), expected_identity["validation"][key], f"validation {key}")
    _same(_json_hash(manifest["validation_protocol"]), manifest["validation_protocol_sha256"], "validation protocol digest")
    _same(_json_hash(manifest["validation_sample"]), manifest["validation_sample_sha256"], "validation sample digest")
    indexed = {document.id: document for document in documents}
    if len(indexed) != len(documents):
        raise ValueError("P2 validation documents must have unique IDs")
    selected, seen = [], set()
    for item in manifest["validation_sample"]:
        document = indexed[item["document_id"]]
        if document.split != "validation":
            raise ValueError("P2 targets must belong to the fixed validation split")
        _same(document.tokenizer_identity, fingerprint, "document tokenizer")
        _same(document.group_id, item["group_id"], "validation group")
        _same(document.source, item["source"], "validation source")
        key = (document.id, item["start"], item["blocks"])
        if key in seen:
            raise ValueError("P2 validation sample contains duplicate windows")
        seen.add(key)
        tokens = document.token_ids[item["start"]:item["start"] + item["blocks"] * block_size]
        if len(tokens) != item["blocks"] * block_size:
            raise ValueError("P2 validation window is incomplete")
        _same(_json_hash(tokens), item["token_sha256"], "validation target token hash")
        window = TrajectoryWindow(document.id, document.group_id, document.source,
                                  item["start"], tokens, item["blocks"])
        _same(window.loss_tokens, item["loss_tokens"], "validation target count")
        selected.append(window)
    if not selected:
        raise ValueError("P2 validation sample must not be empty")
    return selected


def compare_transfer_evaluations(baseline, transfer, *, bootstrap_seed=104729, bootstrap_samples=2000):
    """G = L(I->I) - L(B->I), using aligned document-group loss sums."""
    for key in ("validation_protocol_sha256", "validation_sample_sha256"):
        if not baseline.get(key) or not transfer.get(key):
            raise ValueError(f"P2 paired comparison requires {key}")
        _same(baseline[key], transfer[key], f"paired {key}")
    for key in ("loss_tokens", "windows", "group_count"):
        _same(baseline[key], transfer[key], f"paired {key}")
    left, right = ({item["group_id"]: item for item in result["groups"]} for result in (baseline, transfer))
    if len(left) != len(baseline["groups"]) or len(right) != len(transfer["groups"]) or set(left) != set(right):
        raise ValueError("P2 paired groups differ or contain duplicates")
    records, maximum_empty_difference = [], 0.0
    for group_id in sorted(left):
        old, new = left[group_id], right[group_id]
        for key in ("loss_tokens", "document_ids", "windows"):
            _same(old[key], new[key], f"paired group {group_id} {key}")
        if type(old["loss_tokens"]) is not int or old["loss_tokens"] < 1:
            raise ValueError("P2 groups need positive target counts")
        if not all(type(item[key]) in (int, float) and math.isfinite(item[key])
                   for item in (old, new) for key in ("self_loss_sum", "empty_loss_sum")):
            raise ValueError("P2 group losses must be finite")
        difference = abs(old["empty_loss_sum"] - new["empty_loss_sum"]) / old["loss_tokens"]
        maximum_empty_difference = max(maximum_empty_difference, difference)
        if difference > 1e-6:
            raise ValueError("P2 shared destination empty-memory NLL differs beyond 1e-6 per target")
        records.append({"group_id": group_id, "loss_tokens": old["loss_tokens"],
                        "empty_loss_sum": old["self_loss_sum"], "self_loss_sum": new["self_loss_sum"]})
    result = summarize_group_losses(records, bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples)
    _same(result["loss_tokens"], baseline["loss_tokens"], "paired group/total target counts")
    _same(result["group_count"], baseline["group_count"], "paired group/total group counts")
    ci = result["paired_group_bootstrap"]
    decision = ("insufficient_independent_groups" if result["group_count"] < 64 else
                "supports_B_to_I_on_this_seed" if ci["ci_lower"] > 0 else
                "supports_I_to_I_on_this_seed" if ci["ci_upper"] < 0 else "undetermined_keep_I")
    return {"definition": "G = L_self(I_to_I) - L_self(B_to_I); positive favors B_to_I",
            "G": result["delta_empty"], "I_to_I_nll": result["empty_nll"], "B_to_I_nll": result["self_nll"],
            "loss_tokens": result["loss_tokens"], "group_count": result["group_count"],
            "paired_group_bootstrap": ci, "decision": decision,
            "empty_memory_check": {"status": "passed", "max_group_nll_difference": maximum_empty_difference,
                                   "per_target_tolerance": 1e-6},
            "groups": [{"group_id": item["group_id"], "loss_tokens": item["loss_tokens"],
                        "I_to_I_loss_sum": item["empty_loss_sum"], "B_to_I_loss_sum": item["self_loss_sum"]}
                       for item in records]}


def evaluate_transfer(method, windows, *, metadata, baseline_evaluation,
                      bootstrap_seed=104729, bootstrap_samples=2000):
    """Score the existing raw trajectory with the transferred writer, once."""
    if metadata.get("protocol") != PROTOCOL or metadata.get("research_transfer") is not True:
        raise ValueError("P2 evaluation requires validated research-transfer metadata")
    destination = metadata["destination"]
    _same(baseline_evaluation.get("step"), destination["step"], "baseline checkpoint step")
    _same(baseline_evaluation.get("training_loss_tokens"), destination["training_loss_tokens"], "baseline checkpoint budget")
    identity = destination["raw_identity"]
    _same(_json_hash(_validation_sample(windows)), identity["validation"]["validation_sample_sha256"],
          "evaluation target sample")
    for key, value in identity["validation"].items():
        _same(baseline_evaluation.get(key), value, f"baseline {key}")
    _same(writer_tensor_sha256(method.writer.state_dict()), metadata["source"]["writer_tensor_sha256"], "evaluation writer")
    result = evaluate_raw(method, windows, prefix_tokens=identity["raw"]["prefix_tokens"],
                          batch_size=identity["raw"]["batch_size"],
                          bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples)
    result.update(identity["validation"])
    paired = compare_transfer_evaluations(baseline_evaluation, result,
                                         bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples)
    _same(writer_tensor_sha256(method.writer.state_dict()), metadata["source"]["writer_tensor_sha256"], "post-evaluation writer")
    return {"protocol": PROTOCOL, "complete": True, "checkpoint": metadata,
            "transfer_evaluation": result, "comparison": paired,
            "claims": {"single_seed_only": True, "holdout_used": False, "p6_gate_evaluated": False,
                       "main_conclusion_requires_registered_5M_endpoint": True}}

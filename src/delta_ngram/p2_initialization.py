"""Explicit P2 initial-weight transfer; ordinary checkpoint loading stays strict.

Only the archived, untrained post-trained-reader writer may initialize the Base
arm. This is neither optimizer resume nor a claim of learned transfer quality.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Mapping

import torch

from .checkpoints import load_checkpoint


RAW_PROTOCOL = "delta_ngram_raw_lm_v1"
SOURCE_MODEL = "Qwen/Qwen3-0.6B"
DESTINATION_MODEL = "Qwen/Qwen3-0.6B-Base"
MATCHED_IDENTITY_FIELDS = ("protocol", "seed", "dataset_sha256", "tokenizer",
                           "device_type", "dtype", "reader", "writer", "method",
                           "optimizer", "raw", "validation")


def writer_tensor_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash names, dtypes, shapes and exact tensor bytes, including BF16."""
    if not isinstance(state, Mapping) or any(not isinstance(name, str) for name in state):
        raise ValueError("writer state must map string names to tensors")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("writer state must map string names to tensors")
        metadata = {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_writer_tensors(writer, state: Mapping[str, torch.Tensor]) -> None:
    """Validate every tensor before any load can partially mutate the writer."""
    expected = writer.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("P2 writer tensor keys differ")
    for name, target in expected.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError(f"P2 writer tensor {name} must be a dense tensor")
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"P2 writer tensor {name} shape or dtype differs")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"P2 writer tensor {name} is non-finite")


def _same(left, right, label):
    if left != right:
        raise ValueError(f"P2 initialization {label} differs")


def _tokenizer_binding(package):
    model = package["model"]
    name, revision = model.get("tokenizer_name_or_path"), model.get("tokenizer_revision")
    if (name is None) != (revision is None):
        raise ValueError("P2 tokenizer name and revision override must be set together")
    return {"name_or_path": name if name is not None else model["name_or_path"],
            "revision": revision if revision is not None else model["revision"]}


def _validate_declared_identity(package, identity, label):
    """Reject contradictory provenance instead of trusting one chosen field."""
    if not isinstance(package, dict) or not isinstance(identity, dict):
        raise ValueError(f"P2 initialization requires {label} package/raw identity")
    required = set(MATCHED_IDENTITY_FIELDS) | {"model", "resolved_model_commit"}
    if not required.issubset(identity):
        raise ValueError(f"P2 initialization {label} raw identity is incomplete")
    tokenizer = identity["tokenizer"]
    if (not isinstance(tokenizer, dict)
            or not {"class", "vocab_sha256", "pipeline_sha256", "special_tokens"}.issubset(tokenizer)
            or not isinstance(tokenizer["special_tokens"], dict)
            or not {"pad_token_id", "eos_token_id", "bos_token_id"}.issubset(tokenizer["special_tokens"])):
        raise ValueError(f"P2 initialization {label} tokenizer identity is incomplete")
    validation = identity["validation"]
    if (not isinstance(validation, dict)
            or not {"validation_protocol_sha256", "validation_sample_sha256"}.issubset(validation)):
        raise ValueError(f"P2 initialization {label} validation identity is incomplete")
    hashes = [identity["dataset_sha256"], tokenizer["vocab_sha256"], *validation.values()]
    if tokenizer["pipeline_sha256"] is not None:
        hashes.append(tokenizer["pipeline_sha256"])
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes):
        raise ValueError(f"P2 initialization {label} data/tokenizer hashes are malformed")
    _same(identity["protocol"], RAW_PROTOCOL, f"{label} protocol")
    for key in ("seed", "reader", "writer", "method"):
        _same(identity[key], package.get(key), f"{label} declared {key}")
    try:
        model = {key: package["model"][key] for key in ("name_or_path", "revision", "smoke")}
        optimizer = {"lr": package["train"]["lr"],
                     "gradient_clip": package["train"]["gradient_clip"],
                     "name": "AdamW", "weight_decay": 0.0}
        raw = {key: package["raw"][key] for key in
               ("block_size", "prefix_tokens", "block_counts", "length_weights", "source_weights",
                "batch_size", "gradient_accumulation_steps", "checkpoint_reader")}
    except (KeyError, TypeError) as error:
        raise ValueError(f"P2 initialization {label} package identity is incomplete") from error
    _same(identity["model"], model, f"{label} declared model")
    _same(identity["optimizer"], optimizer, f"{label} declared optimizer")
    _same(identity["raw"], raw, f"{label} declared raw trajectory")
    revision = model["revision"]
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(f"P2 initialization {label} model revision must be pinned")
    _same(identity["resolved_model_commit"], revision, f"{label} resolved revision")
    if model["smoke"] is not False:
        raise ValueError("P2 initialization requires real, non-smoke model identities")
    if package["method"].get("name") != "forward" or package["method"].get("operator") != "D1":
        raise ValueError("P2 initialization requires the P1 forward D1 writer")
    if package["reader"].get("backend") != "external_hash":
        raise ValueError("P2 initialization requires external_hash addressing")
    if package.get("run", {}).get("checkpoint") or package["raw"].get("resume"):
        raise ValueError("P2 initialization requires fresh source and destination runs")


def initialize_p2_writer(method, config, path, identity) -> dict:
    """Load an exact P1 initialization into the explicitly compatible Base arm.

    The caller creates a fresh optimizer and sampler. Their states, counters and
    RNG are never read from or restored by this helper. All compatibility checks
    precede the only mutation, strict loading of the writer tensors.
    """
    path = Path(path).resolve()
    source_bytes = path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    payload = load_checkpoint(io.BytesIO(source_bytes))
    if payload.get("protocol") != RAW_PROTOCOL or payload["step"] != 0:
        raise ValueError("P2 initialization requires a raw-LM step-zero writer checkpoint")
    if "optimizer_state_dict" in payload:
        raise ValueError("P2 initialization must not contain optimizer state")
    old = payload["config"]
    if old.get("training_loss_tokens", 0) != 0:
        raise ValueError("P2 initialization has consumed training tokens")
    source = old.get("raw_identity")
    source_package, destination_package = old["package_config"], config.values
    _validate_declared_identity(source_package, source, "source")
    _validate_declared_identity(destination_package, identity, "destination")
    _same(source["model"]["name_or_path"], SOURCE_MODEL, "source model name")
    _same(identity["model"]["name_or_path"], DESTINATION_MODEL, "destination model name")
    source_tokenizer = _tokenizer_binding(source_package)
    destination_tokenizer = _tokenizer_binding(destination_package)
    source_model_binding = {key: source["model"][key] for key in ("name_or_path", "revision")}
    _same(source_tokenizer, source_model_binding, "source tokenizer binding")
    tokenizer_override = destination_package["model"].get("tokenizer_name_or_path") is not None
    if tokenizer_override:
        _same(destination_tokenizer, source_model_binding, "explicit tokenizer override")
    for field in MATCHED_IDENTITY_FIELDS:
        _same(source[field], identity[field], field)
    _same(source_package["raw"]["loss_tokens"], destination_package["raw"]["loss_tokens"],
          "loss-token budget")
    for field, expected in (("seed", source["seed"]),
                            ("model", source["model"]["name_or_path"]),
                            ("revision", source["model"]["revision"]),
                            ("smoke", False),
                            ("resolved_model_commit", source["resolved_model_commit"])):
        _same(old.get(field), expected, f"source metadata {field}")
    reader = destination_package["reader"]
    for field in ("buckets", "layer_index"):
        _same(old[field], reader[field], f"source {field}")
        _same(getattr(method.reader, field), reader[field], f"runtime {field}")
    _same(old["orders"], reader["orders"], "source orders")
    _same(list(method.reader.orders), reader["orders"], "runtime orders")
    _same(old["width"], method.reader.width, "reader hidden width")
    _same(old["hidden_dim"], destination_package["writer"]["hidden_dim"], "writer hidden dimension")
    _same(method.reader.backend, reader["backend"], "runtime address backend")
    _same(method.device.type, identity["device_type"], "runtime device type")
    _same(str(next(method.reader.model.parameters()).dtype), identity["dtype"], "runtime reader dtype")
    _same(getattr(method.reader.model.config, "_commit_hash", None),
          identity["resolved_model_commit"], "runtime resolved model")
    _same(method.writer.width, method.reader.width, "runtime writer width")
    _same(list(method.writer.orders), reader["orders"], "runtime writer orders")
    _same(method.writer.right_window, destination_package["writer"]["right_window"],
          "runtime writer right window")
    _same(method.writer.operator.name, "D1", "runtime writer operator")
    state = payload["writer_state_dict"]
    validate_writer_tensors(method.writer, state)
    tensor_sha256 = writer_tensor_sha256(state)
    method.writer.load_state_dict(state, strict=True)
    for name, value in method.writer.state_dict().items():
        if not torch.equal(value.detach().cpu(), state[name].detach().cpu()):
            raise RuntimeError(f"P2 initialization tensor copy was not exact: {name}")
    return {"protocol": "p2_exact_writer_initialization_v1", "source_path": str(path),
            "source_file_sha256": source_sha256, "source_step": payload["step"],
            "writer_tensor_sha256": tensor_sha256, "writer_tensor_count": len(state),
            "writer_parameter_count": sum(value.numel() for value in state.values()),
            "source_identity": deepcopy(source), "destination_identity": deepcopy(identity),
            "tokenizer_binding": {"source": source_tokenizer, "destination": destination_tokenizer,
                                  "destination_override": tokenizer_override},
            "restored": "writer tensors only; exact keys, shapes, dtypes and values",
            "optimizer": "fresh AdamW; no state restored", "sampler": "fresh; same independent seed",
            "rng": "not restored; sampler owns an independent random.Random stream",
            "scope": "P2 I-initialization to Base only; ordinary checkpoint identity is unchanged"}

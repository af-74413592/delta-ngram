"""Versioned writer checkpoints, with an explicit historical-pilot migration path."""
from __future__ import annotations

from pathlib import Path
import torch

FORMAT = "delta_ngram_writer"
VERSION = 1
LEGACY_PROTOCOL = "forward_memory_two_write_fact_pilot_v1"


def save_checkpoint(path, writer, config, step, optimizer=None):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    payload = {"format": FORMAT, "format_version": VERSION, "protocol": "delta_ngram_pilot_v1",
               "step": step, "config": config,
               "writer_state_dict": {name: tensor.detach().cpu() for name, tensor in writer.state_dict().items()}}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, *, allow_legacy=False):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("malformed writer checkpoint")
    current = payload.get("format") == FORMAT and type(payload.get("format_version")) is int and payload.get("format_version") == VERSION
    legacy = allow_legacy and payload.get("protocol") == LEGACY_PROTOCOL and "format" not in payload
    if not (current or legacy):
        raise ValueError("unsupported writer checkpoint format/version")
    if type(payload.get("step")) is not int or payload["step"] < 0 or not isinstance(payload.get("writer_state_dict"), dict):
        raise ValueError("malformed writer checkpoint")
    _validate_identity(payload)
    return payload


def _validate_identity(payload):
    old = payload.get("config")
    required = {"buckets", "layer_index", "width", "hidden_dim", "orders", "model", "smoke", "seed", "revision"}
    if not isinstance(old, dict) or not required.issubset(old):
        raise ValueError("checkpoint is missing required reader/model identity")
    if payload.get("format") == FORMAT:
        if type(payload.get("format_version")) is not int or payload["format_version"] != VERSION:
            raise ValueError("unsupported writer checkpoint format/version")
        package = old.get("package_config")
        if not isinstance(package, dict) or not {"model", "reader", "writer", "method", "seed"}.issubset(package):
            raise ValueError("current checkpoint requires package_config identity")
        if not isinstance(package["model"], dict) or not {"name_or_path", "revision", "smoke"}.issubset(package["model"]):
            raise ValueError("current checkpoint is missing model identity")
    elif "format" in payload or payload.get("protocol") != LEGACY_PROTOCOL:
        raise ValueError("unsupported writer checkpoint identity")
    return old


def restore_writer(method, payload, config):
    if not method.supports_outer_training:
        raise ValueError("writer checkpoints are only valid for learned forward methods")
    old = _validate_identity(payload)
    values = config.values
    for key, current in (("buckets", method.reader.buckets), ("layer_index", method.reader.layer_index),
                         ("width", method.reader.width), ("hidden_dim", values["writer"]["hidden_dim"]),
                         ("orders", list(method.reader.orders))):
        if old[key] != current:
            raise ValueError(f"checkpoint {key} differs from the configured reader/writer")
    old_package = old.get("package_config")
    if old_package is not None:
        if values["model"]["smoke"] and old_package["seed"] != values["seed"]:
            raise ValueError("checkpoint smoke seed differs (random base-model identity)")
        for key in ("name_or_path", "revision", "smoke"):
            if old_package["model"][key] != values["model"][key]:
                raise ValueError(f"checkpoint model.{key} differs")
        for section in ("reader", "writer", "method"):
            if old_package[section] != values[section]:
                raise ValueError(f"checkpoint {section} configuration differs")
    else:
        if method.writer.operator.name != "D1":
            raise ValueError("historical checkpoints are only compatible with the D1 operator")
        if values["writer"]["right_window"] != 4:
            raise ValueError("historical checkpoints require right_window=4")
        if old["model"] != values["model"]["name_or_path"] or old["smoke"] != values["model"]["smoke"]:
            raise ValueError("historical checkpoint model identity differs")
        if old["smoke"] and old["seed"] != values["seed"]:
            raise ValueError("historical checkpoint smoke seed differs")
        if old["revision"] != values["model"]["revision"]:
            raise ValueError("historical checkpoint requested revision differs")
    expected_revision = old.get("resolved_model_commit") or old.get("revision")
    actual_revision = getattr(method.reader.model.config, "_commit_hash", None) or values["model"]["revision"]
    if expected_revision and actual_revision and expected_revision != actual_revision:
        raise ValueError("checkpoint base-model revision differs")
    method.writer.load_state_dict(payload["writer_state_dict"], strict=True)


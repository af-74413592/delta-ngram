"""Capture one existing native PLE forward for offline precision diagnostics.

This helper adds no forward/backward and preserves the original return objects.
Call it independently on either/both first-stage TP ranks around an already
planned query, with rank-specific output filenames. It uses no collective and
captures full PLE compute before sequence scattering. It stores the
effective lazy-decayed delta, rather than the memory's stale stored values.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path


def _cpu(value):
    return value.detach().to("cpu").clone()


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def capture_native_ple_snapshot(adapter, *, state, valid_mask,
                                runtime_identity_sha256: str, metadata: dict,
                                owner_ids=None, ple_source_sha256: str | None = None):
    """Yield a dict populated by exactly one forward, with no cotangents.

    Example (each selected first-stage TP rank; other ranks use ``nullcontext``):

        with capture_native_ple_snapshot(
            session.adapter, state=memory, valid_mask=valid, owner_ids=owners,
            runtime_identity_sha256=runtime_sha, metadata={
                "sample_id": sample_id, "source": "books",
                "delta_origin": "calibrated_initial_writer",
                "query_mode": "grad", "alpha": .99,
            }) as snapshot:
            result = schedule_query(...)
        save_native_ple_snapshot(snapshot, output / "initial_writer_ple.pt")

    ``snapshot["delta"]`` contains effective values for the query's unique
    physical rows, including zero values for rows absent from the memory.
    """
    import torch

    _check(len(runtime_identity_sha256) == 64
           and all(c in "0123456789abcdef" for c in runtime_identity_sha256),
           "runtime identity must be a lowercase SHA256")
    _check(valid_mask.dtype == torch.bool and valid_mask.ndim == 2 and valid_mask.shape[0] == 1,
           "snapshot requires a one-owner [1,T] boolean mask")
    _check(state.batch_size == 1 and state.width > 0,
           "snapshot requires a one-owner memory")
    if owner_ids is not None:
        _check(owner_ids.shape == (1,) and int(owner_ids.item()) == 0,
               "snapshot supports owner zero only")
    _check(isinstance(metadata, dict) and metadata.get("delta_origin") == "calibrated_initial_writer",
           "capture must explicitly identify calibrated initial-writer delta")
    json.dumps(metadata, allow_nan=False)
    ple = adapter.ple_layer
    source = inspect.getsourcefile(type(ple))
    if ple_source_sha256 is None:
        _check(source is not None, "native PLE source file is unavailable")
        ple_source_sha256 = _hash(source)
    _check(len(ple_source_sha256) == 64
           and all(c in "0123456789abcdef" for c in ple_source_sha256),
           "PLE source identity must be a lowercase SHA256")
    snapshot = {
        "schema": "qwen38_native_ple_visibility_snapshot_v1",
        "runtime_identity_sha256": runtime_identity_sha256,
        "ple_source_sha256": ple_source_sha256,
        "metadata": dict(metadata),
        "cotangents_captured": False, "extra_forward_passes": 0,
        "extra_backward_passes": 0, "new_ce_target_tokens": 0,
        "valid_mask": _cpu(valid_mask),
    }
    counts = {"overlay": 0, "compute": 0}
    original_compute, original_overlay = ple.compute, adapter._overlay
    had_compute = "compute" in ple.__dict__
    previous_compute = ple.__dict__.get("compute")
    had_overlay = "_overlay" in adapter.__dict__
    previous_overlay = adapter.__dict__.get("_overlay")

    def overlay(native_ids, base):
        counts["overlay"] += 1
        _check(counts["overlay"] == 1, "snapshot observed multiple native overlays")
        result = original_overlay(native_ids, base)
        with torch.no_grad():
            physical = adapter.layout.physical_rows(native_ids, valid_mask)
            rows = torch.unique(physical[valid_mask].reshape(-1), sorted=True)
            rows = rows[rows > 0]
            _check(rows.numel() > 0, "snapshot query contains no valid native rows")
            effective, _ = state.lookup(rows[None])
            _check(effective.shape == (1, rows.numel(), state.width),
                   "effective lookup shape differs from sparse row layout")
            _check(effective.dtype == torch.float32,
                   "effective writer delta must remain FP32")
            snapshot.update(
                base_lookup=_cpu(base), embedding_output=_cpu(result),
                physical_rows=_cpu(physical), rows=_cpu(rows), delta=_cpu(effective[0]),
                delta_semantics="effective state.lookup at query time; lazy decay already applied",
                native_ids=_cpu(native_ids),
            )
            if hasattr(state, "block_clock"):
                snapshot["owner_block_clock"] = _cpu(state.block_clock)
        return result

    def compute(*args, **kwargs):
        counts["compute"] += 1
        _check(counts["compute"] == 1, "snapshot observed multiple native PLE computes")
        hidden = args[0] if args else kwargs["hidden_states"]
        snapshot["hidden_input"] = _cpu(hidden)
        snapshot["grad_enabled_during_forward"] = torch.is_grad_enabled()
        result = original_compute(*args, **kwargs)
        _check(torch.is_tensor(result), "native PLE compute must return one tensor")
        snapshot["native_output"] = _cpu(result)
        return result

    try:
        ple.compute = compute
        adapter._overlay = overlay
        yield snapshot
        _check(counts == {"overlay": 1, "compute": 1},
               "snapshot requires exactly one native PLE forward")
        snapshot.update(
            weights={name: _cpu(value) for name, value in ple.named_parameters()
                     if not name.startswith("ple_embedding.")},
            layout={"hidden_size": int(ple.hidden_size), "hc_count": int(ple.hc_count),
                    "short_conv_state_len": int(ple.short_conv_state_len)},
            norms={name: {"eps": float(getattr(ple, name).eps),
                          "group_size": int(getattr(ple, name).group_size)}
                   for name in ("norm_key", "norm_query", "norm_conv")},
            conv={name: getattr(ple.conv1d, name)
                  for name in ("stride", "padding", "dilation", "groups")},
        )
        _check(torch.equal(
            (snapshot["base_lookup"].float().reshape_as(snapshot["embedding_output"])
             + _dense_delta(snapshot)).to(snapshot["embedding_output"].dtype),
            snapshot["embedding_output"]),
            "captured effective delta does not reconstruct the native overlay")
        snapshot["capture_complete"] = True
    finally:
        if had_compute:
            ple.compute = previous_compute
        else:
            delattr(ple, "compute")
        if had_overlay:
            adapter._overlay = previous_overlay
        else:
            delattr(adapter, "_overlay")


def _dense_delta(snapshot):
    import torch
    rows, physical = snapshot["rows"], snapshot["physical_rows"]
    indices = torch.searchsorted(rows, physical)
    safe = indices.clamp_max(rows.numel() - 1)
    found = indices.lt(rows.numel()) & rows[safe].eq(physical) & physical.gt(0)
    found &= snapshot["valid_mask"].unsqueeze(-1)
    return (snapshot["delta"][safe] * found.unsqueeze(-1)).flatten(-2)


def save_native_ple_snapshot(snapshot: dict, path: Path) -> dict:
    """Write a completed capture once and return an independently hashable seal."""
    import torch
    path = Path(path)
    _check(snapshot.get("capture_complete") is True, "cannot save an incomplete capture")
    _check(not path.exists(), "refusing to overwrite a native snapshot")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        torch.save(snapshot, output)
    receipt = {
        "protocol": "qwen38_native_ple_visibility_snapshot_v1",
        "snapshot_file": path.name, "bytes": path.stat().st_size, "sha256": _hash(path),
        "runtime_identity_sha256": snapshot["runtime_identity_sha256"],
        "ple_source_sha256": snapshot["ple_source_sha256"], "metadata": snapshot["metadata"],
        "cotangents_captured": False, "extra_forward_passes": 0,
        "extra_backward_passes": 0, "new_ce_target_tokens": 0,
    }
    with path.with_suffix(path.suffix + ".json").open("x", encoding="utf-8") as output:
        output.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt

"""Fixed-delta PLE inference for the isolated, audited vLLM 0.30.0 fork.

The native hash, base-table lookup, BF16 base projection, request routing, and
short-convolution kernels remain in vLLM. Only the delta projection and PLE
body use FP32. Request caches are private to one immutable delta binding.
No live snapshot replacement or training/write path is exposed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from .qwen38_native import Qwen38PLELayout
from .qwen38_precision import PROTOCOL as NATIVE_PRECISION_PROTOCOL
from .qwen38_precision import _ieee_matmul, grouped_rms_fp32


PROTOCOL = "qwen38_vllm_fixed_delta_split_ple_v1"
BINDING_ENV = "DELTA_NGRAM_VLLM_BINDING"
BINDING_SHA_ENV = "DELTA_NGRAM_VLLM_BINDING_SHA256"
_UNSET = object()
_environment: object | tuple[str | None, str | None] = _UNSET
_binding: dict[str, Any] | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: str, label: str) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{label} must be a lowercase SHA256")
    return value


def _layout_from_mapping(value: dict[str, Any]) -> Qwen38PLELayout:
    _require(isinstance(value, dict), "binding layout must be an object")
    fields = dict(value)
    for key in ("head_offsets", "head_vocab_sizes", "orders"):
        if key in fields:
            _require(isinstance(fields[key], (list, tuple)), f"layout {key} must be a sequence")
            fields[key] = tuple(fields[key])
    return Qwen38PLELayout(**fields)


def _read_binding(path: str, digest: str) -> dict[str, Any]:
    _sha256(digest, BINDING_SHA_ENV)
    source = Path(path)
    _require(source.is_absolute() and source.is_file(), "binding path must be an existing absolute file")
    raw = source.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == digest, "binding SHA256 mismatch")
    data = json.loads(raw)
    _require(isinstance(data, dict) and type(data.get("schema_version")) is int
             and data["schema_version"] == 1, "unsupported fixed-delta binding schema")
    _require(type(data.get("owner_id")) is int and data["owner_id"] == 0,
             "fixed-delta vLLM binding requires owner_id=0")
    _require(isinstance(data.get("identity"), dict) and bool(data["identity"]),
             "binding must contain the caller's expected snapshot identity")
    snapshot_path = data.get("snapshot_path")
    _require(isinstance(snapshot_path, str) and Path(snapshot_path).is_absolute(),
             "snapshot_path must be absolute")
    _sha256(data.get("snapshot_sha256"), "snapshot_sha256")
    data["_layout"] = _layout_from_mapping(data.get("layout"))
    data["_binding_sha256"] = digest
    data["_binding_path"] = str(source.resolve())
    return data


def split_enabled() -> bool:
    """Lock the environment on first inspection, including the disabled case."""
    global _environment, _binding
    current = (os.environ.get(BINDING_ENV), os.environ.get(BINDING_SHA_ENV))
    if _environment is _UNSET:
        if current == (None, None):
            _environment = current
            return False
        _require(all(current), f"{BINDING_ENV} and {BINDING_SHA_ENV} must be set together")
        # Commit only after validating the complete binding.
        parsed = _read_binding(current[0], current[1])
        _binding, _environment = parsed, current
    _require(current == _environment, "fixed-delta binding environment changed; start a new process")
    return _binding is not None


@dataclass
class _Runtime:
    snapshot: Any
    binding_sha256: str
    fp32_weight: Tensor
    fp32_weight_version: int
    parameters: tuple[tuple[Tensor, int], ...]
    layout: Qwen38PLELayout

    def check(self, ple) -> None:
        _require(split_enabled(), "fixed-delta binding is not enabled")
        _require(self.binding_sha256 == _binding["_binding_sha256"], "binding changed")
        observed = tuple(ple.parameters())
        _require(len(observed) == len(self.parameters)
                 and all(actual is parameter and parameter._version == version
                         for actual, (parameter, version) in zip(observed, self.parameters)),
                 "frozen PLE weights changed after the cached split reader was bound")
        _require(all(not parameter.requires_grad and parameter.grad is None for parameter in observed),
                 "fixed-delta PLE parameters must remain frozen and without gradients")
        _require(self.fp32_weight._version == self.fp32_weight_version,
                 "immutable FP32 projection copy changed")
        _require(ple.kv_proj.weight.device == self.fp32_weight.device,
                 "PLE device changed after binding; start a new process")


def _validate_ple_layout(ple, layout: Qwen38PLELayout) -> None:
    embedding = ple.ple_embedding
    _require(int(ple.layer_idx) + 1 == layout.ple_layer_id,
             "binding PLE layer does not match the loaded model")
    _require(int(ple.ple_dense_layer_id) == 0, "the audited model has one dense PLE layer")
    _require(int(ple.hidden_size) == layout.input_dim and int(ple.hc_count) == layout.hc_count,
             "binding PLE width/hyperconnection count differs from the loaded model")
    _require(int(embedding.embedding_dim) == layout.input_dim
             and int(embedding.head_dim) == layout.row_width
             and int(embedding.heads_per_ngram) == layout.heads_per_order
             and tuple(range(2, int(embedding.ngram_size) + 1)) == layout.orders,
             "binding n-gram layout differs from the loaded model")
    for name, expected in (("ngram_heads_offsets", layout.head_offsets),
                           ("ngram_heads_vocab_sizes", layout.head_vocab_sizes)):
        observed = getattr(embedding, name)
        _require(observed.dtype == torch.long and observed.ndim == 1
                 and tuple(observed.detach().cpu().tolist()) == expected,
                 f"binding {name} differs from the loaded native buffers")


def _runtime(ple) -> _Runtime:
    _require(split_enabled(), "split PLE requires a fixed-delta binding")
    current = getattr(ple, "_delta_ngram_fixed_runtime", None)
    if current is None:
        from .qwen38_vllm_snapshot import load_vllm_snapshot

        layout = _binding["_layout"]
        _validate_ple_layout(ple, layout)
        weight = ple.kv_proj.weight
        _require(weight.dtype == torch.bfloat16 and weight.ndim == 2
                 and tuple(weight.shape) == (layout.input_dim * (layout.hc_count + 1), layout.input_dim),
                 "fixed-delta reader requires the native unquantized BF16 merged PLE projection")
        _require(tuple(ple.kv_proj.output_sizes) == (layout.input_dim * layout.hc_count, layout.input_dim),
                 "native PLE key/value projection order changed")
        _require(not any(parameter.requires_grad for parameter in ple.parameters()),
                 "fixed-delta vLLM reader requires frozen PLE parameters")
        snapshot = load_vllm_snapshot(
            _binding["snapshot_path"], expected_sha256=_binding["snapshot_sha256"],
            expected_identity=_binding["identity"], expected_layout=layout,
            expected_owner_id=0, device=weight.device)
        # vLLM calls forward under inference_mode; keep a version counter for
        # the derived immutable projection as well as the source parameters.
        with torch.inference_mode(False), torch.no_grad():
            fp32_weight = weight.detach().float().clone()
        current = _Runtime(snapshot, _binding["_binding_sha256"], fp32_weight, fp32_weight._version,
                           tuple((parameter, parameter._version) for parameter in ple.parameters()), layout)
        ple._delta_ngram_fixed_runtime = current
    current.check(ple)
    return current


@torch.no_grad()
def split_cached_body(ple, hidden_states: Tensor, base_embeddings: Tensor,
                      delta_embeddings: Tensor, fp32_weight: Tensor) -> Tensor:
    """Small-tensor-testable split body using the native cached conv dispatch.

    The caller supplies the immutable FP32 copy of the merged projection.
    Native Triton conv may fuse multiply-adds: this is not a bitwise claim
    against the training reader's explicit FP32 cross-correlation.
    """
    _require(hidden_states.dtype == base_embeddings.dtype == torch.bfloat16,
             "split PLE requires BF16 native hidden states and base embeddings")
    _require(delta_embeddings.dtype == fp32_weight.dtype == torch.float32,
             "delta projection must retain FP32 inputs and weights")
    _require(delta_embeddings.shape == base_embeddings.shape
             and hidden_states.ndim == base_embeddings.ndim == 2
             and hidden_states.shape[0] == base_embeddings.shape[0],
             "split PLE tensor shapes differ")
    _require(delta_embeddings.device == base_embeddings.device == hidden_states.device == fp32_weight.device,
             "split PLE tensors must be on one device")
    cache = ple.kv_cache[0]
    _require(cache.numel() == 0 or cache.dtype == torch.float32,
             "PLE convolution cache must be allocated in FP32 before engine initialization")
    with torch.autocast(device_type=hidden_states.device.type, enabled=False):
        projected = ple.kv_proj(base_embeddings)
        if isinstance(projected, tuple):
            _require(len(projected) == 2 and projected[1] is None,
                     "native PLE projection must be bias-free")
            projected = projected[0]
        _require(projected.dtype == torch.bfloat16, "base PLE projection must remain native BF16")
        with _ieee_matmul(hidden_states.device):
            delta_projected = F.linear(delta_embeddings, fp32_weight)
        key, value = (projected.float() + delta_projected).split(ple.kv_proj.output_sizes, dim=-1)
        shape = (int(ple.hc_count), int(ple.hidden_size))
        key = grouped_rms_fp32(key, ple.norm_key).unflatten(-1, shape)
        query = grouped_rms_fp32(hidden_states, ple.norm_query).unflatten(-1, shape)
        gate = (key * query).sum(-1, keepdim=True) / math.sqrt(ple.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
        conv_input = grouped_rms_fp32(gated, ple.norm_conv)
        # vLLM adds outer_residual inside its kernel. Delay that addition until
        # after the PLE result is cast, matching the training reader boundary.
        ple._short_conv(conv_input, gated, torch.zeros_like(gated))
        return hidden_states + gated.to(hidden_states.dtype)


@torch.no_grad()
def split_ple_forward(ple, hidden_states: Tensor, input_ids: Tensor,
                      query_start_loc: Tensor, ngram_context: Tensor) -> Tensor:
    """Entry called only by the patched vLLM PLE.forward branch."""
    runtime = _runtime(ple)
    input_ids = input_ids.reshape(-1)
    _require(input_ids.shape[0] == hidden_states.shape[0], "PLE input and hidden token counts differ")
    base = ple.ple_embedding(hidden_states, input_ids, query_start_loc, ngram_context)
    base = ple._dequantize_embeddings(base, hidden_states.dtype)
    # Reuse the native hash implementation and its request-boundary/EOS rules.
    # A second hash call keeps pinned-host prefetch unchanged in this first fork.
    native_ids = ple.ple_embedding.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
    delta = runtime.snapshot.lookup(native_ids, owner_id=0)
    return split_cached_body(ple, hidden_states, base, delta, runtime.fp32_weight)


def runtime_identity(ple) -> dict[str, Any]:
    """Return small, JSON-safe worker evidence for llm.apply_model checks."""
    if not split_enabled():
        return {"enabled": False}
    runtime = _runtime(ple)
    from . import qwen38_vllm_snapshot as snapshot_module

    def source_identity(filename):
        path = Path(filename).resolve()
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    cache = ple.kv_cache[0]
    return {
        "enabled": True,
        "binding_sha256": runtime.binding_sha256,
        "snapshot_sha256": runtime.snapshot.sha256,
        "snapshot_identity": runtime.snapshot.identity,
        "snapshot_rows": runtime.snapshot.row_count,
        "owner_id": runtime.snapshot.owner_id,
        "ple_layer_id": runtime.layout.ple_layer_id,
        "precision_protocol": PROTOCOL,
        "native_precision_protocol": NATIVE_PRECISION_PROTOCOL,
        "base_projection_dtype": str(ple.kv_proj.weight.dtype),
        "delta_projection_dtype": str(runtime.fp32_weight.dtype),
        "cache_dtype": str(cache.dtype) if cache.numel() else str(ple.get_state_dtype()[0]),
        "conv_kernel": "native vLLM ple_conv; FP32 state; FMA rounding may differ",
        "output_cast": "PLE FP32 to BF16, then add BF16 outer residual",
        "snapshot_replacement": "forbidden; use a new process",
        "helper_source": source_identity(__file__),
        "snapshot_helper_source": source_identity(snapshot_module.__file__),
    }

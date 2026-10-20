"""Scoped split PLE read: native BF16 base projection plus an FP32 delta branch.

This is an explicitly different reader numerical path. Both empty/self queries
and evidence use it uniformly while the context is installed. Empty output is
not claimed to equal the historical BF16 PLE; regenerate that baseline and H.
Native hashing, table lookup, TP lookup reduction, and SP scatter are unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor
import torch.nn.functional as F

from .qwen38_adapter import _OwnerView


PROTOCOL = "qwen38_split_ple_native_base_fp32_delta_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _linear_output(value):
    if isinstance(value, tuple):
        _require(len(value) == 2 and value[1] is None,
                 "native PLE projection must be bias-free")
        return value[0]
    return value


def grouped_rms_fp32(x: Tensor, module) -> Tensor:
    x = x.float()
    grouped = x.unflatten(-1, (-1, int(module.group_size)))
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + float(module.eps))
    return normalized.flatten(-2) * (1.0 + module.weight.detach().float())


def depthwise_causal_fp32(x: Tensor, ple) -> Tensor:
    """FP32 direct cross-correlation avoids TF32/cudnn precision dispatch."""
    conv = ple.conv1d
    weight = conv.weight.detach().float()
    dilation = int(conv.dilation[0])
    length = x.shape[1]
    padded = F.pad(x.transpose(1, 2), (int(ple.short_conv_state_len), 0))
    output = padded[..., :length] * weight[:, 0, 0][None, :, None]
    for kernel in range(1, weight.shape[-1]):
        offset = kernel * dilation
        output = output + padded[..., offset:offset + length] * weight[:, 0, kernel][None, :, None]
    return output.transpose(1, 2)


@contextmanager
def _ieee_matmul(device):
    """The setting affects only dispatch of the scoped FP32 delta GEMMs."""
    if device.type != "cuda":
        yield
        return
    backend = torch.backends.cuda.matmul
    # Newer PyTorch forbids mixing this setting with the legacy allow_tf32 API.
    setting = "fp32_precision" if hasattr(backend, "fp32_precision") else "allow_tf32"
    prior = getattr(backend, setting)
    try:
        setattr(backend, setting, "ieee" if setting == "fp32_precision" else False)
        yield
    finally:
        setattr(backend, setting, prior)


class _FrozenFP32Linear(torch.autograd.Function):
    """Frozen linear with the same IEEE dispatch contract in its backward."""

    @staticmethod
    def forward(ctx, x, weight):
        _require(x.dtype == weight.dtype == torch.float32 and not weight.requires_grad,
                 "delta projection requires frozen FP32 weights and FP32 input")
        ctx.save_for_backward(weight)
        with torch.autocast(device_type=x.device.type, enabled=False), _ieee_matmul(x.device):
            return F.linear(x, weight)

    @staticmethod
    def backward(ctx, cotangent):
        weight, = ctx.saved_tensors
        with torch.autocast(device_type=cotangent.device.type, enabled=False), _ieee_matmul(cotangent.device):
            gradient = F.linear(cotangent.float(), weight.transpose(0, 1))
        return gradient, None


def split_projections(base_key: Tensor, base_value: Tensor, delta: Tensor, ple):
    """FP32 delta never passes through the BF16 embedding sum or a BF16 GEMM."""
    with torch.autocast(device_type=delta.device.type, enabled=False), _ieee_matmul(delta.device):
        key_delta = _FrozenFP32Linear.apply(delta.float(), ple.key_proj.weight.detach().float())
        value_delta = _FrozenFP32Linear.apply(delta.float(), ple.value_proj.weight.detach().float())
        key = base_key.float() + key_delta
        value = base_value.float() + value_delta
    return key, value, key_delta, value_delta


def ple_body_fp32(hidden: Tensor, key: Tensor, value: Tensor, ple) -> Tensor:
    """FP32 norm, signed sqrt, gate, convolution and PLE residual."""
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        shape = (int(ple.hc_count), int(ple.hidden_size))
        key = grouped_rms_fp32(key, ple.norm_key).unflatten(-1, shape)
        query = grouped_rms_fp32(hidden, ple.norm_query).unflatten(-1, shape)
        gate = (key * query).sum(-1, keepdim=True) / math.sqrt(ple.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = (torch.sigmoid(gate) * value.float().unsqueeze(-2)).flatten(-2)
        normalized = grouped_rms_fp32(gated, ple.norm_conv)
        return gated + F.silu(depthwise_causal_fp32(normalized, ple))


def _native_body_from_projections(hidden, key, value, ple):
    """Diagnostic replay with the already computed native base projections."""
    shape = (int(ple.hc_count), int(ple.hidden_size))
    key = ple.norm_key(key).unflatten(-1, shape)
    query = ple.norm_query(hidden).unflatten(-1, shape)
    gate = (key * query).sum(-1, keepdim=True) / math.sqrt(ple.hidden_size)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
    return gated + ple._short_conv(ple.norm_conv(gated))


@torch.no_grad()
def signal_statistics(requested: Tensor, effective: Tensor) -> dict:
    requested, effective = requested.detach().double(), effective.detach().double()
    _require(requested.shape == effective.shape, "signal statistics shapes differ")
    energy = float(requested.square().sum())
    effective_energy = float(effective.square().sum())
    error_energy = float((effective - requested).square().sum())
    dot = float((requested * effective).sum())
    active = requested.ne(0)
    count = int(active.sum())
    return {
        "components": requested.numel(), "requested_nonzero_components": count,
        "changed_fraction_of_nonzero_requested": (
            float((active & effective.ne(0)).sum()) / count if count else None),
        "requested_energy": energy, "effective_energy": effective_energy,
        "dot_effective_requested": dot,
        "effective_over_requested_energy": effective_energy / energy if energy else None,
        "aligned_amplitude_ratio": dot / energy if energy else None,
        "cosine": dot / math.sqrt(energy * effective_energy) if energy and effective_energy else None,
        "relative_l2_error": math.sqrt(error_energy / energy) if energy else None,
        "requested_rms": math.sqrt(energy / requested.numel()),
        "effective_rms": math.sqrt(effective_energy / requested.numel()),
    }


class Qwen38SplitPLEPrecision:
    """Reversible, single-forward split read attached to an existing adapter.

    ``observer`` receives detached aggregate metrics only. When
    ``compare_native_zero=True``, diagnostics additionally replay the small PLE
    body at zero delta from the same base projections; no table lookup, backbone
    forward, CE or backward is added. Leave observer=None during ordinary work.
    """

    _marker = "_delta_ngram_split_ple_precision"

    def __init__(self, adapter, *, observer: Callable[[dict], None] | None = None,
                 compare_native_zero: bool = False,
                 delta_observer: Callable[[dict], None] | None = None):
        self.adapter, self.ple = adapter, adapter.ple_layer
        self.observer, self.compare_native_zero = observer, compare_native_zero
        self.delta_observer = delta_observer
        self._active = False
        self._inside_compute = False
        self._packet = None
        _require(not compare_native_zero or observer is not None,
                 "native-zero comparison requires an observer")
        _require(not any(p.requires_grad for p in self.ple.parameters()),
                 "split PLE requires a frozen reader")
        conv = self.ple.conv1d
        width = int(self.ple.hidden_size) * int(self.ple.hc_count)
        _require(conv.groups == width and conv.weight.shape[:2] == (width, 1)
                 and tuple(conv.stride) == (1,) and tuple(conv.padding) == (0,)
                 and conv.bias is None,
                 "split PLE requires the native bias-free depthwise causal convolution")
        _require(int(self.ple.short_conv_state_len)
                 == (conv.weight.shape[-1] - 1) * int(conv.dilation[0]),
                 "native causal convolution alignment differs")
        _require(self.ple.key_proj.weight.dtype == self.ple.value_proj.weight.dtype == torch.bfloat16,
                 "this numerical contract requires native BF16 key/value weights")
        self._parameters = [(p, p._version) for p in self.ple.parameters()]

    def identity(self) -> dict:
        return {
            "protocol": PROTOCOL,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "base_projection": "unchanged native BF16 TELinear on T0",
            "delta_projection": "FP32 frozen F.linear with TF32 disabled in both forward and input-gradient GEMMs",
            "norm_gate_conv_residual": "FP32; explicit depthwise cross-correlation",
            "output_cast": "explicit cast to native hidden-input dtype, normally BF16",
            "empty_baseline": "new numerical path; compare and recalibrate, no old-native identity claim",
            "evidence_and_query_use_same_path": True,
            "reader_weights_updated": False,
        }

    def __enter__(self):
        _require(not self._active and not hasattr(self.ple, self._marker),
                 "split PLE precision context is already installed")
        self._had_compute = "compute" in self.ple.__dict__
        self._compute_instance = self.ple.__dict__.get("compute")
        self._had_overlay = "_overlay" in self.adapter.__dict__
        self._overlay_instance = self.adapter.__dict__.get("_overlay")
        self._original_compute = self.ple.compute
        self._original_overlay = self.adapter._overlay
        self.ple.compute = self._compute
        self.adapter._overlay = self._overlay
        setattr(self.ple, self._marker, self)
        self._active = True
        return self

    def _overlay(self, native_ids, base):
        _require(self._inside_compute and self._packet is None,
                 "split overlay must execute once within bound PLE.compute")
        # Existing adapter validates layout, owner, request binding, and records
        # native IDs/T0. Its historical rounded return is deliberately unused.
        self._original_overlay(native_ids, base)
        read = self.adapter._current()
        flat = base.detach().reshape(*native_ids.shape[:2], -1)
        if read.mode == "evidence":
            delta = torch.zeros_like(flat, dtype=torch.float32)
            physical = self.adapter.layout.physical_rows(native_ids, read.valid_mask)
        else:
            physical = self.adapter.layout.physical_rows(native_ids, read.valid_mask)
            delta, _ = _OwnerView(read.state, read.owner_ids).lookup(physical)
            delta = delta.flatten(-2)
        _require(delta.dtype == torch.float32 and delta.shape == flat.shape,
                 "split branch requires FP32 delta with native concatenated-head width")
        if self.delta_observer is not None:
            self.delta_observer({
                "dense_delta": delta, "native_ids": native_ids,
                "physical_rows": physical, "valid_mask": read.valid_mask,
                "owner_ids": read.owner_ids, "mode": read.mode,
            })
        self._packet = (delta, read.valid_mask, read.mode)
        return base.detach()

    def _compute(self, hidden_states, input_ids):
        _require(self._active and not self._inside_compute, "nested or unbound split PLE compute")
        self._inside_compute, self._packet = True, None
        try:
            base = self.ple.ple_embedding(input_ids)
            _require(self._packet is not None, "native embedding did not execute the split overlay")
            delta, valid, mode = self._packet
            base = base.reshape(*input_ids.shape, -1)
            base_key = _linear_output(self.ple.key_proj(base))
            base_value = _linear_output(self.ple.value_proj(base))
            _require(base_key.dtype == base_value.dtype == torch.bfloat16,
                     "native base projection no longer returns BF16")
            key, value, key_delta, value_delta = split_projections(base_key, base_value, delta, self.ple)
            output_fp32 = ple_body_fp32(hidden_states, key, value, self.ple)
            output = output_fp32.to(hidden_states.dtype)
            if self.observer is not None:
                self._observe(hidden_states, base_key, base_value, key, value,
                              key_delta, value_delta, output_fp32, output, valid, mode)
            return output
        finally:
            self._inside_compute, self._packet = False, None

    @torch.no_grad()
    def _observe(self, hidden, base_key, base_value, key, value, key_delta, value_delta,
                 output_fp32, output, valid, mode):
        _require(bool(valid.any()), "precision visibility requires a valid token")
        zero_fp32 = ple_body_fp32(hidden, base_key.float(), base_value.float(), self.ple)
        zero = zero_fp32.to(hidden.dtype)
        effect_fp32 = (output_fp32 - zero_fp32)[valid]
        effect_cast = (output.float() - zero.float())[valid]
        report = {
            "protocol": PROTOCOL, "mode": mode, "valid_tokens": int(valid.sum()),
            "key_projection_add": signal_statistics(key_delta[valid], (key - base_key.float())[valid]),
            "value_projection_add": signal_statistics(value_delta[valid], (value - base_value.float())[valid]),
            "final_output_cast": signal_statistics(effect_fp32, effect_cast),
            "output_dtype": str(output.dtype), "fp32_body_dtype": str(output_fp32.dtype),
            "extra_backbone_forwards": 0, "extra_ce_target_tokens": 0,
        }
        if self.compare_native_zero:
            native_zero = _native_body_from_projections(hidden, base_key, base_value, self.ple)
            reference, observed = native_zero[valid].float(), zero[valid].float()
            report["empty_baseline_comparison"] = {
                "scope": "native BF16 body replay vs new body, from identical current native base projections",
                "bitwise_equal": bool(torch.equal(native_zero, zero)),
                "native_zero_to_new_zero": signal_statistics(reference, observed),
                "difference_rms": float((observed - reference).square().mean().sqrt()),
                "difference_max_abs": float((observed - reference).abs().max()),
            }
        self.observer(report)

    def __exit__(self, exc_type, exc, traceback):
        if self._active:
            if self._had_compute:
                self.ple.compute = self._compute_instance
            else:
                delattr(self.ple, "compute")
            if self._had_overlay:
                self.adapter._overlay = self._overlay_instance
            else:
                delattr(self.adapter, "_overlay")
            delattr(self.ple, self._marker)
            self._active = False
        _require(all(not p.requires_grad and p._version == version for p, version in self._parameters),
                 "reader parameters changed while split precision was installed")
        return False

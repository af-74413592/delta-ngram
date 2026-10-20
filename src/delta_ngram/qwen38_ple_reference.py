"""Offline PLE arithmetic checks using a captured native input and cotangent.

The FP64 chain is a mathematical reference, not an emulation of BF16 kernels.
Its finite differences validate the reference derivative. Comparing that VJP
with the captured native VJP separately measures their numerical disagreement.
No model, tokenizer, distributed group, or CUDA allocation is needed.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor
import torch.nn.functional as F


REFERENCE_PROTOCOL = "qwen38_native_ple_fp64_reference_v1"
FD_RELATIVE_STEPS = (1e-4, 3e-5, 1e-5)
FD_RTOL = 1e-5
FD_ATOL = 1e-8


def grouped_rms_norm(x: Tensor, weight: Tensor, group_size: int,
                     eps: float) -> Tensor:
    """Native mathematical formula, including its zero-centered scale."""
    if group_size <= 0 or x.shape[-1] % group_size or eps <= 0:
        raise ValueError("invalid grouped RMSNorm configuration")
    grouped = x.unflatten(-1, (-1, group_size))
    normalized = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + eps)
    return normalized.flatten(-2) * (1 + weight)


def signed_sqrt(x: Tensor) -> Tensor:
    """Exactly the native clamp/sign convention, including x == 0."""
    return x.abs().clamp_min(1e-6).sqrt() * x.sign()


def causal_depthwise_conv(x: Tensor, weight: Tensor, *, left_pad: int,
                          dilation: int, stride: int = 1) -> Tensor:
    """Explicit depthwise cross-correlation, independent of conv1d kernels."""
    if weight.shape[:2] != (x.shape[-1], 1) or min(dilation, stride) < 1:
        raise ValueError("expected a depthwise convolution without bias")
    padded = F.pad(x.transpose(1, 2), (left_pad, 0))
    length = (padded.shape[-1] - dilation * (weight.shape[-1] - 1) - 1) // stride + 1
    if length <= 0:
        raise ValueError("empty causal convolution")
    terms = [
        padded[..., k * dilation:k * dilation + length * stride:stride]
        * weight[:, 0, k][None, :, None]
        for k in range(weight.shape[-1])
    ]
    return torch.stack(terms).sum(0).transpose(1, 2)


def _one(value) -> int:
    return int(value[0] if isinstance(value, (tuple, list)) else value)


def validate_snapshot(snapshot: dict, *, require_cotangents: bool = True) -> None:
    required = {
        "runtime_identity_sha256", "ple_source_sha256", "hidden_input",
        "embedding_output", "base_lookup", "native_output", "rows",
        "delta", "physical_rows", "valid_mask", "weights", "layout", "norms", "conv",
    }
    if require_cotangents:
        required |= {"native_output_cotangent", "native_embedding_cotangent"}
    missing = required - snapshot.keys()
    if missing:
        raise ValueError(f"snapshot fields missing: {sorted(missing)}")
    hidden, embedding = snapshot["hidden_input"], snapshot["embedding_output"]
    layout, weights = snapshot["layout"], snapshot["weights"]
    width, streams = int(layout["hidden_size"]), int(layout["hc_count"])
    if hidden.ndim != 3 or embedding.ndim != 3 or hidden.shape[:2] != embedding.shape[:2]:
        raise ValueError("hidden and embedding must have matching [B,T] axes")
    if hidden.shape[-1] != width * streams:
        raise ValueError("hyper-connection width differs from layout")
    expected = {
        "key_proj.weight": (width * streams, embedding.shape[-1]),
        "value_proj.weight": (width, embedding.shape[-1]),
        "norm_key.weight": (width * streams,),
        "norm_query.weight": (width * streams,),
        "norm_conv.weight": (width * streams,),
    }
    for name, shape in expected.items():
        if name not in weights or tuple(weights[name].shape) != shape:
            raise ValueError(f"invalid native weight {name}")
    if set(weights) != set(expected) | {"conv1d.weight"}:
        raise ValueError("snapshot contains unexpected PLE parameters")
    conv = snapshot["conv"]
    if (int(conv["groups"]) != width * streams or _one(conv["padding"]) != 0
            or _one(conv["stride"]) != 1):
        raise ValueError("reference requires the registered depthwise causal convolution")
    kernel = weights["conv1d.weight"]
    if kernel.ndim != 3 or kernel.shape[:2] != (width * streams, 1):
        raise ValueError("invalid depthwise convolution weight")
    if int(layout["short_conv_state_len"]) != (kernel.shape[-1] - 1) * _one(conv["dilation"]):
        raise ValueError("causal left padding differs from native kernel")
    for name in ("norm_key", "norm_query", "norm_conv"):
        norm = snapshot["norms"][name]
        if int(norm["group_size"]) != width or not 0 < float(norm["eps"]) < 1:
            raise ValueError("unexpected grouped RMSNorm configuration")
    for name in ("native_output", "native_output_cotangent") if require_cotangents else ("native_output",):
        if snapshot[name].shape != hidden.shape:
            raise ValueError(f"{name} shape differs from hidden")
    if require_cotangents and snapshot["native_embedding_cotangent"].shape != embedding.shape:
        raise ValueError("native embedding cotangent shape mismatch")
    rows, delta = snapshot["rows"], snapshot["delta"]
    physical, valid = snapshot["physical_rows"], snapshot["valid_mask"]
    if (rows.ndim != 1 or rows.dtype != torch.long or rows.numel() == 0
            or not bool(rows.gt(0).all())
            or (rows.numel() > 1 and not bool((rows[1:] > rows[:-1]).all()))):
        raise ValueError("state rows must be unique, sorted and positive")
    if delta.ndim != 2 or delta.shape[0] != rows.numel() or delta.dtype != torch.float32:
        raise ValueError("delta must be an FP32 sparse row matrix")
    if (physical.ndim != 3 or physical.dtype != torch.long
            or physical.shape[:2] != hidden.shape[:2]
            or physical.shape[-1] * delta.shape[-1] != embedding.shape[-1]
            or valid.shape != hidden.shape[:2] or valid.dtype != torch.bool):
        raise ValueError("native address or mask layout mismatch")
    if snapshot["base_lookup"].numel() != embedding.numel():
        raise ValueError("base lookup width mismatch")
    tensor_fields = ["hidden_input", "embedding_output", "base_lookup", "native_output", "delta"]
    if require_cotangents:
        tensor_fields += ["native_output_cotangent", "native_embedding_cotangent"]
    tensors = list(weights.values()) + [snapshot[name] for name in tensor_fields]
    if any(t.device.type != "cpu" or not bool(torch.isfinite(t).all()) for t in tensors):
        raise ValueError("reference snapshot must contain finite CPU tensors")
    if embedding.dtype != torch.bfloat16 or snapshot["base_lookup"].dtype != torch.bfloat16:
        raise ValueError("this receipt measures the registered BF16 lookup cast")


def _converted(snapshot: dict) -> dict:
    return {
        **snapshot,
        "weights": {name: value.double() for name, value in snapshot["weights"].items()},
    }


def ple_reference(hidden: Tensor, embedding: Tensor, snapshot: dict,
                  *, trace: dict | None = None,
                  projected: tuple[Tensor, Tensor] | None = None) -> Tensor:
    """Execute the unfused native mathematical PLE chain in supplied dtype."""
    width = int(snapshot["layout"]["hidden_size"])
    streams = int(snapshot["layout"]["hc_count"])
    weights, norms = snapshot["weights"], snapshot["norms"]

    def operation(name: str, function: Callable, *inputs: Tensor) -> Tensor:
        result = function(*inputs)
        if trace is not None:
            result.retain_grad()
            trace[name] = (function, inputs, result)
        return result

    def norm(name):
        return lambda x: grouped_rms_norm(
            x, weights[name + ".weight"], int(norms[name]["group_size"]),
            float(norms[name]["eps"]))

    if projected is not None and trace is not None:
        raise ValueError("external projections are only supported for visibility diagnostics")
    key = (projected[0] if projected is not None else
           operation("key_projection", lambda x: F.linear(x, weights["key_proj.weight"]), embedding))
    key = operation("key_rmsnorm", norm("norm_key"), key)
    value = (projected[1] if projected is not None else
             operation("value_projection", lambda x: F.linear(x, weights["value_proj.weight"]), embedding))
    query = operation("query_rmsnorm", norm("norm_query"), hidden)
    gate = operation(
        "gate_dot",
        lambda k, q: (k.unflatten(-1, (streams, width))
                      * q.unflatten(-1, (streams, width))).sum(-1, keepdim=True) / math.sqrt(width),
        key, query,
    )
    gate = operation("signed_sqrt", signed_sqrt, gate)
    gate = operation("sigmoid", torch.sigmoid, gate)
    gated = operation("gate_value_product", lambda g, v: (g * v.unsqueeze(-2)).flatten(-2), gate, value)
    normalized = operation("conv_rmsnorm", norm("norm_conv"), gated)
    convolved = operation(
        "causal_depthwise_conv",
        lambda x: causal_depthwise_conv(
            x, weights["conv1d.weight"],
            left_pad=int(snapshot["layout"]["short_conv_state_len"]),
            dilation=_one(snapshot["conv"]["dilation"]),
            stride=_one(snapshot["conv"]["stride"])),
        normalized,
    )
    activated = operation("conv_silu", F.silu, convolved)
    return operation("residual_add", lambda x, y: x + y, gated, activated)


def _rms(x: Tensor) -> float:
    return float(x.double().square().mean().sqrt())


def _comparison(actual: Tensor, reference: Tensor) -> dict:
    actual, reference = actual.detach().double(), reference.detach().double()
    error = actual - reference
    actual_norm, reference_norm = actual.norm(), reference.norm()
    return {
        "rms_actual": _rms(actual), "rms_reference": _rms(reference),
        "rms_error": _rms(error), "max_abs_error": float(error.abs().max()),
        "relative_l2_error": float(error.norm() / reference_norm) if reference_norm > 0 else None,
        "cosine": float((actual.flatten() @ reference.flatten())
                        / (actual_norm * reference_norm)) if min(actual_norm, reference_norm) > 0 else None,
        "same_sign_fraction_nonzero_reference": (
            float((actual[reference.ne(0)].sign() == reference[reference.ne(0)].sign()).double().mean())
            if bool(reference.ne(0).any()) else None),
    }


def directional_fd(function: Callable, inputs: tuple[Tensor, ...], cotangent: Tensor,
                   *, seed: int, direction: tuple[Tensor, ...] | None = None) -> dict:
    """Fixed directions and steps; acceptance requires both smallest steps."""
    leaves = tuple(x.detach().double().requires_grad_(True) for x in inputs)
    cotangent = cotangent.detach().double()
    output = function(*leaves)
    objective = (output * cotangent).sum()
    gradients = torch.autograd.grad(objective, leaves)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if direction is None:
        direction = tuple(torch.randn(x.shape, dtype=torch.float64, generator=generator) for x in leaves)
    if len(direction) != len(leaves) or any(d.shape != x.shape for d, x in zip(direction, leaves)):
        raise ValueError("direction shape mismatch")
    direction = tuple(d.detach().double() for d in direction)
    count = sum(x.numel() for x in leaves)
    direction_rms = math.sqrt(sum(float(d.square().sum()) for d in direction) / count)
    if not direction_rms > 0:
        raise ValueError("zero finite-difference direction")
    direction = tuple(d / direction_rms for d in direction)
    input_scale = max(math.sqrt(sum(float(x.detach().square().sum()) for x in leaves) / count), 1e-4)
    derivative = sum(float((g * d).sum()) for g, d in zip(gradients, direction))
    probes = []
    with torch.no_grad():
        for relative in FD_RELATIVE_STEPS:
            step = relative * input_scale
            plus = float((function(*(x + step * d for x, d in zip(leaves, direction))) * cotangent).sum())
            minus = float((function(*(x - step * d for x, d in zip(leaves, direction))) * cotangent).sum())
            measured = (plus - minus) / (2 * step)
            error = abs(measured - derivative)
            tolerance = FD_ATOL + FD_RTOL * max(abs(measured), abs(derivative))
            probes.append({
                "relative_step": relative, "step": step, "plus": plus, "minus": minus,
                "finite_difference": measured, "absolute_error": error,
                "tolerance": tolerance, "passed": bool(error <= tolerance),
            })
    return {
        "seed": seed, "direction_rms": 1.0, "input_scale": input_scale,
        "autograd_directional_derivative": derivative, "objective": float(objective.detach()),
        "probes": probes, "passed": all(p["passed"] for p in probes[-2:]),
        "acceptance": "both two smallest preregistered steps must pass; no step selection",
    }


def dense_delta(snapshot: dict) -> Tensor:
    rows, physical = snapshot["rows"], snapshot["physical_rows"]
    indices = torch.searchsorted(rows, physical)
    safe = indices.clamp_max(rows.numel() - 1)
    found = indices.lt(rows.numel()) & rows[safe].eq(physical) & physical.gt(0)
    found &= snapshot["valid_mask"].unsqueeze(-1)
    return (snapshot["delta"][safe] * found.unsqueeze(-1)).flatten(-2)


def _quantization_stats(delta: Tensor, effective: Tensor) -> dict:
    delta, effective = delta.double().flatten(), effective.double().flatten()
    active = delta.ne(0)
    swallowed = active & effective.eq(0)
    energy = float(delta.square().sum())
    effective_energy = float(effective.square().sum())
    error_energy = float((effective - delta).square().sum())
    dot = float((effective * delta).sum())
    count = int(active.sum())
    return {
        "components": delta.numel(), "nonzero_delta_components": count,
        "swallowed_components": int(swallowed.sum()),
        "swallowed_fraction_of_nonzero_delta": float(swallowed.sum()) / count if count else None,
        "changed_fraction_of_nonzero_delta": float((active & effective.ne(0)).sum()) / count if count else None,
        "delta_energy": energy,
        "effective_delta_energy": effective_energy,
        "dot_effective_requested": dot,
        "cosine": dot / math.sqrt(energy * effective_energy) if energy and effective_energy else None,
        "aligned_amplitude_ratio": dot / energy if energy else None,
        "relative_l2_error": math.sqrt(error_energy / energy) if energy else None,
        "swallowed_delta_energy_fraction": float(delta[swallowed].square().sum()) / energy if energy else None,
        "quantization_error_energy_fraction": error_energy / energy if energy else None,
        "effective_delta_energy_fraction": effective_energy / energy if energy else None,
        "effective_delta_rms": _rms(effective), "delta_rms": _rms(delta),
    }


def overlay_quantization(snapshot: dict) -> dict:
    """Measure the actual pre-projection FP32 sum -> BF16 cast, including zeros."""
    base = snapshot["base_lookup"].reshape_as(snapshot["embedding_output"])
    delta = dense_delta(snapshot)
    quantized = (base.float() + delta).to(torch.bfloat16)
    if not torch.equal(quantized, snapshot["embedding_output"]):
        raise ValueError("captured embedding is not the registered BF16 overlay formula")
    effective = quantized.float() - base.float()
    valid = snapshot["valid_mask"]
    heads = snapshot["physical_rows"].shape[-1]
    row_width = snapshot["delta"].shape[-1]
    delta_head, effective_head = [
        x.reshape(*valid.shape, heads, row_width)[valid] for x in (delta, effective)
    ]
    if delta_head.numel() == 0:
        raise ValueError("quantization measurement needs at least one valid token")
    projections = {}
    for name in ("key_proj.weight", "value_proj.weight"):
        weight = snapshot["weights"][name].double()
        ideal_projected = F.linear(delta.double(), weight)[valid]
        effective_projected = F.linear(effective.double(), weight)[valid]
        projections[name] = _comparison(effective_projected, ideal_projected)
    return {
        "formula": "BF16(FP32(T0) + FP32(delta)) - FP32(T0)",
        "captured_embedding_exactly_reproduced": True,
        "valid_tokens": int(valid.sum()), "overall": _quantization_stats(delta_head, effective_head),
        "per_head": [_quantization_stats(delta_head[:, h], effective_head[:, h]) for h in range(heads)],
        "projection_of_effective_delta_vs_ideal_delta_fp64": projections,
        "projection_limit": "FP64 linear diagnostic isolates input-cast loss; excludes projection/output casts.",
    }


@torch.no_grad()
def split_projection_diagnostics(snapshot: dict) -> dict:
    """Measure a candidate's local signal; this is not a proposed runtime edit."""
    base = snapshot["base_lookup"].reshape_as(snapshot["embedding_output"]).float()
    delta = dense_delta(snapshot)
    valid = snapshot["valid_mask"]
    projections = {}
    for name in ("key_proj.weight", "value_proj.weight"):
        weight = snapshot["weights"][name].float()
        base_projected = F.linear(base, weight)
        delta_projected = F.linear(delta, weight)
        combined = base_projected + delta_projected
        ideal = F.linear(delta.double(), weight.double())[valid]
        effective_fp32 = (combined - base_projected)[valid]
        effective_final_bf16 = (
            combined.to(torch.bfloat16).float()
            - base_projected.to(torch.bfloat16).float())[valid]
        projections[name] = {
            "fp32_until_norm_gate": _quantization_stats(ideal, effective_fp32),
            "if_immediately_cast_back_to_bf16": _quantization_stats(ideal, effective_final_bf16),
            "fp32_delta_projection_vs_fp64": _comparison(delta_projected[valid], ideal),
        }
    return {
        "formula": "linear(FP32(T0), FP32(W)) + linear(FP32(delta), FP32(W))",
        "projections": projections,
        "empty_state_native_identity": "unverified; FP32 native-base projection may alter the empty baseline",
        "scope": "CPU arithmetic diagnostic only; no reader or adapter implementation was changed.",
        "limits": [
            "Signal at projection output is measured relative to this candidate's own zero-delta output.",
            "Candidate-vs-native empty-state equivalence requires a separate actual-runtime check.",
            "Final BF16 cast is measured separately and is not assumed lossless.",
            "This diagnostic does not measure subsequent gate/norm/conv output changes.",
        ],
    }


@torch.no_grad()
def candidate_ple_output_visibility(snapshot: dict, zero_snapshot: dict | None = None) -> dict:
    """Compare the FP32 candidate's effect and baseline as separate quantities."""
    base = snapshot["base_lookup"].reshape_as(snapshot["embedding_output"]).float()
    delta = dense_delta(snapshot)
    hidden = snapshot["hidden_input"]
    valid = snapshot["valid_mask"]
    fp32 = {**snapshot, "weights": {k: v.float() for k, v in snapshot["weights"].items()}}
    fp64 = _converted(snapshot)
    base_projected, changed_projected = [], []
    for name in ("key_proj.weight", "value_proj.weight"):
        base_branch = F.linear(base, fp32["weights"][name])
        base_projected.append(base_branch)
        changed_projected.append(base_branch + F.linear(delta, fp32["weights"][name]))
    candidate_base = ple_reference(hidden.float(), base, fp32, projected=tuple(base_projected))
    candidate_changed = ple_reference(hidden.float(), base, fp32, projected=tuple(changed_projected))
    ideal_base = ple_reference(hidden.double(), base.double(), fp64)
    ideal_changed = ple_reference(hidden.double(), base.double() + delta.double(), fp64)
    ideal_effect = (ideal_changed - ideal_base)[valid]
    candidate_effect = (candidate_changed - candidate_base)[valid]
    final_cast_effect = (
        candidate_changed.bfloat16().float() - candidate_base.bfloat16().float())[valid]
    result = {
        "candidate": "separate FP32 base/delta projections; FP32 norm/gate/conv/residual",
        "fp32_candidate_effect_vs_fp64_mathematical_effect": _quantization_stats(
            ideal_effect, candidate_effect),
        "candidate_effect_after_final_bf16_cast": _quantization_stats(ideal_effect, final_cast_effect),
        "native_empty_baseline_comparison_available": zero_snapshot is not None,
        "scope": "Offline candidate only; full FP32 arithmetic may move the empty baseline.",
    }
    if zero_snapshot is not None:
        validate_snapshot(zero_snapshot, require_cotangents=False)
        if (snapshot["runtime_identity_sha256"] != zero_snapshot["runtime_identity_sha256"]
                or snapshot["ple_source_sha256"] != zero_snapshot["ple_source_sha256"]
                or snapshot["layout"] != zero_snapshot["layout"]
                or snapshot["norms"] != zero_snapshot["norms"]
                or snapshot["conv"] != zero_snapshot["conv"]):
            raise ValueError("zero-state snapshot belongs to a different runtime")
        for name in ("hidden_input", "base_lookup", "physical_rows", "valid_mask"):
            if not torch.equal(snapshot[name], zero_snapshot[name]):
                raise ValueError(f"zero-state snapshot changed {name}")
        for name in snapshot["weights"]:
            if not torch.equal(snapshot["weights"][name], zero_snapshot["weights"][name]):
                raise ValueError(f"zero-state snapshot changed weight {name}")
        if bool(dense_delta(zero_snapshot).ne(0).any()):
            raise ValueError("zero-state snapshot contains a nonzero effective delta")
        overlay_quantization(zero_snapshot)
        native_base = zero_snapshot["native_output"].float()
        native_effect = (snapshot["native_output"].float() - native_base)[valid]
        result.update(
            native_effect_vs_fp64_mathematical_effect=_quantization_stats(ideal_effect, native_effect),
            candidate_empty_vs_native_empty=_comparison(candidate_base[valid], native_base[valid]),
            candidate_empty_after_bf16_vs_native_empty=_comparison(
                candidate_base.bfloat16()[valid], native_base[valid]),
            candidate_empty_after_bf16_bitwise_equal_to_native=bool(
                torch.equal(candidate_base.bfloat16(), zero_snapshot["native_output"])),
            candidate_effect_vs_native_effect=_comparison(candidate_effect, native_effect),
        )
    return result


def run_visibility_audit(snapshot: dict, *, zero_snapshot: dict | None = None) -> dict:
    """Audit forward precision without fabricating cotangents or derivative claims."""
    validate_snapshot(snapshot, require_cotangents=False)
    return {
        "protocol": "qwen38_native_ple_actual_writer_visibility_v1",
        "complete": True, "status": "visibility_observations_complete",
        "runtime_identity_sha256": snapshot["runtime_identity_sha256"],
        "ple_source_sha256": snapshot["ple_source_sha256"],
        "snapshot_metadata": snapshot.get("metadata"),
        "delta_scope": "Only the supplied captured delta; no inference about other writer states.",
        "bf16_overlay_quantization": overlay_quantization(snapshot),
        "split_fp32_projection_candidate": split_projection_diagnostics(snapshot),
        "candidate_ple_output_visibility": candidate_ple_output_visibility(snapshot, zero_snapshot),
        "derivative_validation": "not_performed; no native cotangents are required or fabricated",
        "native_runtime_changed": False, "new_reader_forwards": 0,
        "new_reader_backwards": 0, "new_ce_target_tokens": 0, "optimizer_steps": 0,
    }


def run_reference_audit(snapshot: dict, *, progress: Callable | None = None) -> dict:
    """Return reference FD evidence and numerical comparisons as distinct claims."""
    validate_snapshot(snapshot)
    converted = _converted(snapshot)
    hidden = snapshot["hidden_input"].double().requires_grad_(True)
    embedding = snapshot["embedding_output"].double().requires_grad_(True)
    cotangent = snapshot["native_output_cotangent"].double()
    if not bool(cotangent.ne(0).any()):
        raise ValueError("native output cotangent is zero")
    trace = {}
    output = ple_reference(hidden, embedding, converted, trace=trace)
    (output * cotangent).sum().backward()
    stage_probes = {}
    for index, (name, (function, inputs, result)) in enumerate(trace.items()):
        if result.grad is None:
            raise ValueError(f"reference cotangent missing at {name}")
        stage_probes[name] = directional_fd(
            function, inputs, result.grad, seed=20261020 + index)
        if progress:
            progress({"stage": name, "passed": stage_probes[name]["passed"]})
    full_probes = {}
    function = lambda e: ple_reference(hidden.detach(), e, converted)
    for seed in (1701, 2903):
        full_probes[f"embedding_random_seed_{seed}"] = directional_fd(
            function, (embedding,), cotangent, seed=seed)
        if progress:
            progress({"stage": f"full_embedding_{seed}", "passed": full_probes[f"embedding_random_seed_{seed}"]["passed"]})
    delta = dense_delta(snapshot).double()
    if bool(delta.ne(0).any()):
        full_probes["observed_delta_direction"] = directional_fd(
            function, (embedding,), cotangent, seed=0, direction=(delta,))
    gate = trace["signed_sqrt"][1][0].detach()
    reference_passed = all(p["passed"] for p in (*stage_probes.values(), *full_probes.values()))
    return {
        "protocol": REFERENCE_PROTOCOL, "complete": True,
        "runtime_identity_sha256": snapshot["runtime_identity_sha256"],
        "ple_source_sha256": snapshot["ple_source_sha256"],
        "status": "reference_fd_passed" if reference_passed else "reference_fd_inconclusive",
        "native_derivative_acceptance": "not_established_by_reference_fd",
        "shapes": {"hidden": list(hidden.shape), "embedding": list(embedding.shape)},
        "reference_dtype": "torch.float64", "fd_relative_steps": list(FD_RELATIVE_STEPS),
        "fd_rtol": FD_RTOL, "fd_atol": FD_ATOL,
        "stage_finite_differences": stage_probes, "full_chain_finite_differences": full_probes,
        "native_output_vs_fp64_reference": _comparison(snapshot["native_output"], output),
        "native_embedding_vjp_vs_fp64_reference": _comparison(
            snapshot["native_embedding_cotangent"], embedding.grad),
        "signed_sqrt_boundary": {
            "elements": gate.numel(), "exact_zeros": int(gate.eq(0).sum()),
            "clamped_nonzero": int((gate.abs().lt(1e-6) & gate.ne(0)).sum()),
            "min_abs_input": float(gate.abs().min()),
            "minimum_distance_to_clamp_boundary": float((gate.abs() - 1e-6).abs().min()),
        },
        "bf16_overlay_quantization": overlay_quantization(snapshot),
        "split_fp32_projection_candidate": split_projection_diagnostics(snapshot),
        "new_reader_forwards": 0, "new_reader_backwards": 0,
        "new_ce_target_tokens": 0, "optimizer_steps": 0,
        "limits": [
            "Per-stage FD uses real weights and FP64 intermediate inputs from the captured native input.",
            "Its upstream objective cotangent is the captured native PLE output cotangent.",
            "Native BF16-vs-FP64 VJP disagreement includes rounding and changed intermediate points.",
            "Reference FD success alone does not validate every native GPU kernel derivative.",
            "Earlier full-reader CE finite differences remain inconclusive.",
            "No split projection or reader arithmetic was changed by this audit.",
        ],
    }

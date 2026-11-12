"""CPU accounting for the existing D1 ForwardWriter/NativeForwardWriter only.

This module does not load Torch, a backbone, a checkpoint, or a GPU. Linear MACs
are arithmetic counts, not a throughput or total training-cost prediction.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


def _integer(name: str, value: int, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(name: str, value: float, *, positive: bool = False) -> None:
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


@dataclass(frozen=True)
class WriterDimensions:
    input_dim: int
    row_dim: int
    hidden_dim: int
    orders: tuple[int, ...] = (2, 3)
    heads_per_order: int = 1

    def __post_init__(self) -> None:
        for name in ("input_dim", "row_dim", "hidden_dim", "heads_per_order"):
            _integer(name, getattr(self, name))
        if not isinstance(self.orders, tuple) or not self.orders:
            raise ValueError("orders must be a nonempty tuple of positive integers")
        for order in self.orders:
            _integer("order", order)

    @property
    def head_dim(self) -> int:
        return min(32, self.hidden_dim)

    @property
    def num_heads(self) -> int:
        return len(self.orders) * self.heads_per_order


def parameter_breakdown(dim: WriterDimensions) -> dict[str, int]:
    """One entry per actual D1 named parameter, including every bias."""
    d, r, h, e = dim.input_dim, dim.row_dim, dim.hidden_dim, dim.head_dim
    features = 3 * d + e
    return {
        "pool_score.weight": d,
        "head_embedding.weight": dim.num_heads * e,
        "target_head.0.weight": features * h,
        "target_head.0.bias": h,
        "target_head.2.weight": h * r,
        "target_head.2.bias": r,
        "gate_head.0.weight": (features + 2 * r + 1) * h,
        "gate_head.0.bias": h,
        "gate_head.2.weight": h,
        "gate_head.2.bias": 1,
    }


def fp32_training_state_bytes(dim: WriterDimensions) -> dict[str, int]:
    """Per replica, with all gradients and both Adam moments materialized.

    Excludes buffers, Adam step scalars, activations, sparse state, allocator
    overhead, and the backbone. Gradients may be absent after set_to_none=True.
    This is not total or peak GPU memory.
    """
    size = 4 * sum(parameter_breakdown(dim).values())
    return {"parameters": size, "gradients": size, "adam_first_moment": size,
            "adam_second_moment": size, "total": 4 * size}


def _linear_terms(dim: WriterDimensions) -> dict[str, tuple[int, int]]:
    """Coefficients (per padded position, per valid event), when E > 0."""
    d, r, h, e = dim.input_dim, dim.row_dim, dim.hidden_dim, dim.head_dim
    terms = {"pool_score": (d, 0)}
    if dim.heads_per_order == 1:
        terms.update(target_first_dense=(0, (3 * d + e) * h),
                     gate_first_dense=(0, (3 * d + e + 2 * r + 1) * h))
    else:
        terms.update(target_evidence=(3 * d * h, 0),
                     target_head_features=(0, e * h),
                     gate_evidence=(3 * d * h, 0),
                     gate_head_features=(0, e * h),
                     gate_state_features=(0, (2 * r + 1) * h))
    terms.update(target_output=(0, h * r), gate_output=(0, h))
    return terms


def linear_forward_macs(dim: WriterDimensions, *, padded_positions: int,
                        valid_events: int) -> dict[str, int]:
    """Count only the linear operations in one actual writer call.

    Padded positions = B*T; valid events = count(mask & (rows != 0)), before
    row aggregation. Shared evidence projections include padding. Head-feature
    projections still run per event, not once per head. With zero events the
    writer returns before even pool_score, so every linear count is zero.
    Bias addition, nonlinearities, softmax/pooling, right-context reductions,
    embedding lookup, sparse operations, backbone and backward are excluded.
    """
    _integer("padded_positions", padded_positions, minimum=0)
    _integer("valid_events", valid_events, minimum=0)
    if valid_events > padded_positions * dim.num_heads:
        raise ValueError("valid_events exceeds padded_positions * num_heads")
    counts = {name: (p * padded_positions + e * valid_events if valid_events else 0)
              for name, (p, e) in _linear_terms(dim).items()}
    return {**counts, "total": sum(counts.values())}


def training_window_cost(*, target_tokens: int, elapsed_seconds: float,
                         gpu_count: int, price_per_gpu_hour: float,
                         checkpoint_seconds: float = 0, eval_seconds: float = 0,
                         extra_eval_gpu_billed_seconds: float = 0) -> dict:
    """Account for a measured window in the price's user-supplied currency.

    elapsed_seconds is the complete billed wall window for the main GPU pool,
    already including checkpoint/evaluation pauses. The two category durations
    are diagnostics only: they may overlap and are never added or subtracted.
    Extra evaluation is separately billed GPU-seconds, already summed over its
    GPUs and excluding GPU time counted in the main pool. It adds cost, not a
    guessed combined wall time. Supply a common price for all counted GPUs.
    """
    _integer("target_tokens", target_tokens)
    _integer("gpu_count", gpu_count)
    _finite("elapsed_seconds", elapsed_seconds, positive=True)
    for name, value in (("price_per_gpu_hour", price_per_gpu_hour),
                        ("checkpoint_seconds", checkpoint_seconds),
                        ("eval_seconds", eval_seconds),
                        ("extra_eval_gpu_billed_seconds", extra_eval_gpu_billed_seconds)):
        _finite(name, value)
    if checkpoint_seconds > elapsed_seconds or eval_seconds > elapsed_seconds:
        raise ValueError("checkpoint/eval durations must each fit inside the measured window")
    main_hours = gpu_count * elapsed_seconds / 3600
    extra_hours = extra_eval_gpu_billed_seconds / 3600
    total_hours = main_hours + extra_hours
    result = {
        "target_tokens": target_tokens,
        "elapsed_seconds": elapsed_seconds,
        "gpu_count": gpu_count,
        "price_per_gpu_hour": price_per_gpu_hour,
        "checkpoint_seconds_in_window": checkpoint_seconds,
        "eval_seconds_in_window": eval_seconds,
        "extra_eval_gpu_billed_seconds": extra_eval_gpu_billed_seconds,
        "target_tokens_per_second": target_tokens / elapsed_seconds,
        "main_pool_gpu_hours": main_hours,
        "extra_eval_gpu_hours": extra_hours,
        "total_gpu_hours": total_hours,
        "gpu_hours_per_million_target_tokens": total_hours * 1_000_000 / target_tokens,
        "main_pool_cost": main_hours * price_per_gpu_hour,
        "extra_eval_cost": extra_hours * price_per_gpu_hour,
        "total_cost": total_hours * price_per_gpu_hour,
        "cost_per_million_target_tokens": total_hours * price_per_gpu_hour * 1_000_000 / target_tokens,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("measurement arithmetic overflowed")
    result["timing_scope"] = (
        "Main-pool wall time already includes checkpoint/eval; extra evaluation "
        "adds billed GPU-hours only. Combined parallel/serial wall time is unknown.")
    result["currency"] = "same currency as the supplied price_per_gpu_hour"
    return result


def preset_dimensions(preset: str, hidden_dim: int) -> WriterDimensions:
    if preset == "m1":
        return WriterDimensions(1024, 1024, hidden_dim)
    if preset == "native":
        return WriterDimensions(2560, 160, hidden_dim, heads_per_order=8)
    raise ValueError("preset must be m1 or native")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("m1", "native"), required=True)
    parser.add_argument("--hidden-dims", nargs="+", type=int)
    parser.add_argument("--padded-positions", type=int, help="B*T in one actual writer call")
    parser.add_argument("--valid-events", type=int, help="valid token-head events before aggregation")
    parser.add_argument("--measurement", type=Path, help="JSON object for training_window_cost")
    args = parser.parse_args(argv)
    try:
        if (args.padded_positions is None) != (args.valid_events is None):
            raise ValueError("provide both --padded-positions and --valid-events")
        comparisons = []
        for hidden in args.hidden_dims or ([128] if args.preset == "m1" else [256]):
            dim = preset_dimensions(args.preset, hidden)
            params = parameter_breakdown(dim)
            terms = _linear_terms(dim)
            comparisons.append({
                "dimensions": asdict(dim), "head_dim": dim.head_dim,
                "num_heads": dim.num_heads, "parameters": params,
                "total_parameters": sum(params.values()),
                "fp32_training_state_bytes": fp32_training_state_bytes(dim),
                "linear_path": "single_head_dense" if dim.heads_per_order == 1 else "shared_token_evidence",
                "linear_mac_coefficients_when_events_positive": {
                    "per_padded_position": sum(p for p, _ in terms.values()),
                    "per_valid_event": sum(e for _, e in terms.values())},
                "linear_forward_macs": None if args.padded_positions is None else linear_forward_macs(
                    dim, padded_positions=args.padded_positions, valid_events=args.valid_events),
            })
        measurement = None
        if args.measurement:
            inputs = json.loads(args.measurement.read_text(encoding="utf-8"))
            if not isinstance(inputs, dict):
                raise ValueError("measurement JSON must be an object")
            measurement = training_window_cost(**inputs)
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "schema": "d1_writer_cost_v1", "preset": args.preset,
        "scope": "Existing D1 writer only; no V7.5 architecture or runtime changes.",
        "memory_scope": "FP32 parameters, all gradients and two Adam moments per replica; not total GPU memory. "
                        "Excludes buffers, optimizer step scalars, activations, sparse state and allocator overhead.",
        "mac_scope": "Linear forward MACs only; excludes bias adds, nonlinearities, softmax/pooling, right-context "
                     "reductions, embedding lookup, sparse operations, backbone and backward. Zero events means zero MACs.",
        "mac_inputs": {"padded_positions": args.padded_positions, "valid_events": args.valid_events},
        "comparisons": comparisons, "measurement": measurement,
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

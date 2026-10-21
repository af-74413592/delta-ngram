"""Finite-sample sensitivity diagnostics; none certifies global contraction.

Encoding replacement holds the direct writer state fixed. Full-update probes
instead rerun the supplied complete encoding/write map at perturbed states.
Both are label-free and must be charged as encoding by their native caller.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Callable

import torch

from .batching import BatchMemoryState
from .qwen38_staged_batching import materialize_staged_snapshot
from .qwen38_training import writer_evidence_features
from .state import SparseMemoryState


def _positive_finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _validate_state(state):
    if (not bool(state.values.isfinite().all()) or not bool(state.counts.isfinite().all())
            or bool((state.counts < 0).any()) or bool((state.counts != state.counts.floor()).any())):
        raise ValueError("diagnostic state values/counts must be finite with nonnegative integer counts")


def _quantiles(values):
    if not values:
        return {"median": None, "p99": None, "maximum": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    if not bool(tensor.isfinite().all()):
        raise ValueError("nonfinite diagnostic samples")
    return {"median": float(tensor.quantile(.5)), "p99": float(tensor.quantile(.99)),
            "maximum": float(tensor.max())}


def _aggregated(events):
    if (events.rows.dtype != torch.long or events.rows.ndim != 1
            or events.target.ndim != 2 or events.target.shape[0] != events.rows.numel()
            or events.beta.shape != events.rows.shape
            or not (events.rows.device == events.target.device == events.beta.device)
            or bool((events.rows <= 0).any())):
        raise ValueError("invalid diagnostic event shapes, devices, or rows")
    if (not bool(events.target.isfinite().all()) or not bool(events.beta.isfinite().all())
            or bool(((events.beta < 0) | (events.beta > 1)).any())):
        raise ValueError("diagnostic targets/gates must be finite and gates must be in [0,1]")
    rows, inverse, counts = torch.unique(events.rows, sorted=True, return_inverse=True,
                                         return_counts=True)
    beta = events.beta.double()
    total_beta = beta.new_zeros(rows.numel()).index_add(0, inverse, beta)
    weighted = events.target.double().new_zeros((rows.numel(), events.target.shape[-1]))
    weighted.index_add_(0, inverse, beta[:, None] * events.target.double())
    if bool((total_beta <= 0).any()):
        raise ValueError("diagnostic weighted target has a nonpositive gate denominator")
    return rows, weighted / total_beta[:, None], total_beta / counts.double()


@torch.no_grad()
def measure_encoding_sensitivity(writer, state, block_mem, block_base, *,
                                 encoding_mode, denominator_floor=1e-8,
                                 include_relationship=False):
    """Compare H_mem and same-frame/same-lane H_base at fixed direct state.

    ``physical_rows_read`` includes PREFIX and new-token addresses. It must not
    be approximated by the write rows. The floor is an absolute FP32 state L2
    threshold, explicitly reported rather than silently replacing denominators.
    """
    _positive_finite(denominator_floor, "denominator_floor")
    if type(include_relationship) is not bool:
        raise ValueError("include_relationship must be boolean")
    if encoding_mode not in ("full", "residual"):
        raise ValueError("invalid encoding sensitivity policy")
    if "physical_rows_read" not in block_mem or "physical_rows_read" not in block_base:
        raise ValueError("diagnostic requires all prefix and new-token read addresses")
    materialized = materialize_staged_snapshot(state)
    _validate_state(materialized)
    memory, device = materialized.memory, materialized.values.device
    rows = block_mem["physical_rows_new"].to(device)
    if rows.ndim != 3 or rows.shape[0] != 1:
        raise ValueError("write addresses must have shape [1,T,K]")
    if not torch.equal(rows, block_base["physical_rows_new"].to(device)):
        raise ValueError("paired encodings must have identical write addresses")
    # materialize_staged_snapshot enforces one owner. Packing still validates
    # the physical namespace before any sparse lookup can turn bad IDs into 0.
    rows = materialized._pack_rows(rows)
    read_addresses = block_mem["physical_rows_read"].to(device)
    if read_addresses.ndim == 2:
        read_addresses = read_addresses.unsqueeze(0)
    if (read_addresses.ndim != 3 or read_addresses.shape[0] != 1
            or read_addresses.shape[-1] != rows.shape[-1]):
        raise ValueError("read addresses must have shape [T,K] or [1,T,K]")
    read_addresses = materialized._pack_rows(read_addresses)
    base_addresses = block_base["physical_rows_read"].to(device)
    if base_addresses.ndim == 2:
        base_addresses = base_addresses.unsqueeze(0)
    if not torch.equal(read_addresses, materialized._pack_rows(base_addresses)):
        raise ValueError("paired encodings must have identical prefix and new-token read addresses")
    read_rows = torch.unique(read_addresses.flatten(), sorted=True)
    read_rows = read_rows[read_rows > 0]
    written_addresses = torch.unique(rows[rows > 0], sorted=True)
    if not bool(torch.isin(written_addresses, read_rows).all()):
        raise ValueError("read addresses must include every written address")
    hidden_mem = writer_evidence_features(block_mem["H_new"].to(device), mode="rmsnorm", eps=1e-6)
    hidden_base = writer_evidence_features(block_base["H_new"].to(device), mode="rmsnorm", eps=1e-6)
    if hidden_mem.shape != hidden_base.shape:
        raise ValueError("paired encodings must have identical hidden shapes")
    mask = torch.ones(rows.shape[:2], dtype=torch.bool, device=device)
    if encoding_mode == "full":
        changed = writer.events(hidden_mem, rows, mask, memory)
        reference = writer.events(hidden_base, rows, mask, memory)
    else:
        # Keep the actual H12 base branch fixed; only replace the memory branch.
        if "H_base_new" not in block_mem:
            raise ValueError("residual diagnostic requires the actual base writer features")
        fixed_base = writer_evidence_features(block_mem["H_base_new"].to(device), mode="rmsnorm", eps=1e-6)
        matched = ({"hidden_base_matched": hidden_base}
                   if getattr(writer, "difference_baseline", "actual_base") == "matched_base" else {})
        changed = writer.events(fixed_base, rows, mask, memory, hidden_mem=hidden_mem, **matched)
        reference = writer.events(fixed_base, rows, mask, memory, hidden_mem=hidden_base, **matched)
    event_rows, changed_target, changed_gate = _aggregated(changed)
    base_rows, base_target, base_gate = _aggregated(reference)
    if not torch.equal(event_rows, base_rows):
        raise ValueError("encoding replacement changed event address coverage")
    target_difference = changed_target - base_target
    read_values = memory.lookup(read_rows)[0].double()
    denominator = float(torch.linalg.vector_norm(read_values))
    numerator = float(torch.linalg.vector_norm(target_difference))
    old_written = memory.lookup(event_rows)[0].double()
    row_denominators = torch.linalg.vector_norm(old_written, dim=-1)
    eligible = row_denominators >= denominator_floor
    row_gains = (torch.linalg.vector_norm(target_difference, dim=-1)[eligible]
                 / row_denominators[eligible]).cpu().tolist()
    feature_denominator = float(torch.linalg.vector_norm(hidden_base.double()))
    has_writes = bool(event_rows.numel())
    assessable = has_writes and denominator >= denominator_floor
    record = {
        "metric": "encoding_replacement_sensitivity", "mode": encoding_mode,
        "block_g": numerator / denominator if assessable else None,
        "assessable": assessable,
        "filtered_reason": (None if assessable else "no_write_events" if not has_writes
                            else "read_state_l2_below_floor"),
        "target_difference_l2": numerator, "read_state_l2": denominator,
        "denominator_floor": denominator_floor, "unique_read_rows": read_rows.numel(),
        "written_rows": event_rows.numel(), "eligible_written_rows": int(eligible.sum()),
        "filtered_written_rows": int((~eligible).sum()),
        "row_output_gain_values": row_gains,
        "gate_mean_absolute_change": (float((changed_gate - base_gate).abs().mean())
                                      if has_writes else None),
        "gate_max_absolute_change": (float((changed_gate - base_gate).abs().max())
                                     if has_writes else None),
        "matched_read_addresses_verified": True,
        "target_definition": "per-physical-row gate-weighted target mean; direct writer state held fixed",
        "block_g_definition": "written target-difference L2 / unique prefix-plus-new read-state L2",
        "row_output_gain_definition": "row target-difference L2 / that written row state L2; not a self-loop gain",
        "normalized_H_relative_l2": (float(torch.linalg.vector_norm(
            hidden_mem.double() - hidden_base.double())) / feature_denominator
            if feature_denominator >= denominator_floor else None),
    }
    if include_relationship:
        target_energy = target_difference.square().sum(-1)
        old_energy = row_denominators.square()
        read_energy = read_values.square().sum()
        eligible_old_energy = old_energy[eligible].sum()
        eligible_target_energy = target_energy[eligible].sum()
        filtered_target_energy = target_energy[~eligible].sum()
        # Row ratios omit near-zero denominators. Their contribution to block
        # G is an energy-weighted sum, plus the omitted rows' numerator energy.
        # Prefix-only read rows enlarge the block denominator without adding a
        # target-difference numerator; an unweighted row quantile is unrelated.
        record["denominator_relationship"] = {
            "read_state_energy": float(read_energy),
            "written_state_energy": float(old_energy.sum()),
            "eligible_written_state_energy": float(eligible_old_energy),
            "filtered_written_state_energy": float(old_energy[~eligible].sum()),
            "read_only_state_energy": float(read_energy - old_energy.sum()),
            "eligible_target_difference_energy": float(eligible_target_energy),
            "filtered_target_difference_energy": float(filtered_target_energy),
            "eligible_weighted_row_gain_squared": (float(eligible_target_energy / eligible_old_energy)
                                                    if bool(eligible_old_energy > 0) else None),
            "eligible_contribution_to_block_g_squared": (float(eligible_target_energy / read_energy)
                                                         if assessable else None),
            "filtered_contribution_to_block_g_squared": (float(filtered_target_energy / read_energy)
                                                         if assessable else None),
            "identity": "G^2 = eligible_target_energy/read_energy + filtered_target_energy/read_energy",
            "interpretation": "energy weighting over unique rows; row quantiles are not block G or self-loop gains",
        }
    for key, value in record.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"nonfinite encoding diagnostic: {key}")
    return record


def summarize_encoding_sensitivity(records, *, stop_p99=1., monitor_only=False):
    """A registered conservative stop rule, explicitly not a stability theorem."""
    _positive_finite(stop_p99, "stop_p99")
    if type(monitor_only) is not bool:
        raise ValueError("monitor_only must be boolean")
    records = list(records)
    modes = {row["mode"] for row in records}
    floors = {row["denominator_floor"] for row in records}
    if len(modes) > 1 or len(floors) > 1:
        raise ValueError("summarize one encoding mode and denominator floor at a time")
    for row in records:
        _positive_finite(row["denominator_floor"], "denominator_floor")
        if row["mode"] not in ("full", "residual"):
            raise ValueError("invalid encoding sensitivity mode")
        counts = [row[key] for key in ("written_rows", "eligible_written_rows", "filtered_written_rows")]
        if (any(type(value) is not int or value < 0 for value in counts)
                or counts[0] != counts[1] + counts[2]
                or len(row["row_output_gain_values"]) != counts[1]):
            raise ValueError("diagnostic written-row counts do not match retained samples")
        samples = row["row_output_gain_values"] + ([] if row["block_g"] is None else [row["block_g"]])
        if any(not isinstance(value, Real) or not math.isfinite(value) or value < 0 for value in samples):
            raise ValueError("nonfinite or negative sensitivity samples")
    gains = [row["block_g"] for row in records if row["block_g"] is not None]
    row_gains = [value for row in records for value in row["row_output_gain_values"]]
    quantiles = _quantiles(gains)
    written = sum(row["written_rows"] for row in records)
    filtered = sum(row["filtered_written_rows"] for row in records)
    return {
        "metric": "encoding_replacement_sensitivity", "blocks": len(records),
        "mode": next(iter(modes)) if modes else None,
        "denominator_floor": next(iter(floors)) if floors else None,
        "eligible_blocks": len(gains), "filtered_blocks": len(records) - len(gains),
        "block_g": quantiles, "row_output_gain": _quantiles(row_gains),
        "written_rows": written, "filtered_written_rows": filtered,
        "filtered_written_fraction": filtered / written if written else None,
        "block_fraction_above_one": sum(value > 1 for value in gains) / len(gains) if gains else None,
        "block_fraction_above_threshold": (sum(value > stop_p99 for value in gains) / len(gains)
                                           if gains else None),
        "rule": ("monitor_sampled_block_g_p99; no_automatic_stop; not a contraction test" if monitor_only
                 else "stop_if_sampled_block_g_p99_exceeds_threshold; not a contraction test"),
        "stop_p99_threshold": stop_p99,
        "stop_requested": bool(not monitor_only and gains and quantiles["p99"] > stop_p99),
        "assessable": bool(gains),
        "interpretation": "sampled encoding replacement only; does not include complete state-update feedback",
    }


def _copy_with_values(state, values):
    return BatchMemoryState(SparseMemoryState(
        state.rows.detach().clone(), values.detach().clone(), state.counts.detach().clone()),
        state.batch_size, state.row_capacity)


def _aligned_difference(a, b):
    if (a.batch_size, a.row_capacity, a.width) != (b.batch_size, b.row_capacity, b.width):
        raise ValueError("probe outputs must share owner namespace and row width")
    rows = torch.unique(torch.cat((a.rows, b.rows)), sorted=True)
    return a.memory.lookup(rows)[0].double() - b.memory.lookup(rows)[0].double()


def _require_same_counts(a, b):
    rows = torch.unique(torch.cat((a.rows, b.rows)), sorted=True)
    if not torch.equal(a.memory.lookup(rows)[1], b.memory.lookup(rows)[1]):
        raise ValueError("probe outputs changed observation counts across the fixed update")


@torch.no_grad()
def probe_full_update(update: Callable, state, *, relative_epsilons=(1e-3, 1e-2),
                      random_directions=2, seed=17, absolute_floor=1e-8):
    """Rerun the COMPLETE update, including H/targets/gates, at +/- eps*u.

    The caller must hold evidence, counters, masks, BOS, padding, lane and RNG
    fixed, and durably charge every native encoding. Output support may expand.
    Measurements are directional finite differences, not a Jacobian norm bound.
    """
    _positive_finite(absolute_floor, "absolute_floor")
    if not callable(update):
        raise TypeError("update must be callable")
    if type(random_directions) is not int or random_directions < 1 or type(seed) is not int:
        raise ValueError("random_directions must be a positive integer and seed an integer")
    relative_epsilons = tuple(relative_epsilons)
    if not relative_epsilons:
        raise ValueError("finite difference scales must be positive and finite")
    for eps in relative_epsilons:
        _positive_finite(eps, "relative_epsilon")
    if len(set(relative_epsilons)) != len(relative_epsilons):
        raise ValueError("finite difference scales must be distinct")
    base = materialize_staged_snapshot(state)
    _validate_state(base)
    scale = float(torch.linalg.vector_norm(base.values.double()))
    if scale < absolute_floor:
        return {"assessable": False, "reason": "state_l2_below_floor", "calls": 0,
                "absolute_floor": absolute_floor, "input_state_l2": scale}

    calls = 0
    def run(values):
        nonlocal calls
        supplied = _copy_with_values(base, values)
        saved = tuple(x.clone() for x in (supplied.rows, supplied.values, supplied.counts))
        result = update(supplied)
        calls += 1
        if any(not torch.equal(before, after) for before, after in
               zip(saved, (supplied.rows, supplied.values, supplied.counts))):
            raise RuntimeError("full-update probe mutated its input snapshot")
        result = materialize_staged_snapshot(result)
        if (result.batch_size, result.row_capacity, result.width) != (
                base.batch_size, base.row_capacity, base.width):
            raise ValueError("full-update probe changed owner namespace or row width")
        _validate_state(result)
        return result

    reference, repeat = run(base.values), run(base.values)
    _require_same_counts(reference, repeat)
    repeat_error = float(torch.linalg.vector_norm(_aligned_difference(reference, repeat)))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = []
    for index in range(random_directions):
        direction = torch.randn(base.values.shape, generator=generator, dtype=torch.float64)
        directions.append((f"random_{index}", direction.to(base.values.device)))
    strongest = int(base.values.double().square().sum(-1).argmax())
    structured = torch.zeros_like(base.values, dtype=torch.float64)
    structured[strongest] = base.values[strongest].double()
    directions.append(("largest_row_radial", structured))
    rows = []
    for name, direction in directions:
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(absolute_floor)
        for relative in relative_epsilons:
            step = max(relative * scale, absolute_floor)
            plus_input = (base.values.double() + step * direction).to(base.values.dtype)
            minus_input = (base.values.double() - step * direction).to(base.values.dtype)
            denominator = float(torch.linalg.vector_norm(plus_input.double() - minus_input.double()))
            if denominator < absolute_floor:
                rows.append({"direction": name, "relative_epsilon": relative,
                             "filtered": True, "reason": "rounded_perturbation_below_floor"})
                continue
            plus, minus = run(plus_input), run(minus_input)
            _require_same_counts(reference, plus)
            _require_same_counts(reference, minus)
            differences = _aligned_difference(plus, minus)
            row_norms = torch.linalg.vector_norm(differences, dim=-1)
            numerator = float(torch.linalg.vector_norm(differences))
            rows.append({"direction": name, "relative_epsilon": relative, "filtered": False,
                         "actual_input_separation_l2": denominator,
                         "output_separation_l2": numerator, "directional_amplification": numerator / denominator,
                         "output_union_rows": differences.shape[0],
                         "output_changed_rows": int((row_norms > 0).sum()),
                         "output_row_difference_l2": _quantiles(row_norms.cpu().tolist()),
                         "repeat_error_to_signal": repeat_error / max(numerator, absolute_floor)})
    assessable = any(not row["filtered"] for row in rows)
    return {"assessable": assessable, "calls": calls, "input_state_l2": scale,
            "reason": None if assessable else "all_rounded_perturbations_below_floor",
            "relative_epsilons": list(relative_epsilons), "random_directions": random_directions,
            "direction_seed": seed, "structured_row": int(base.rows[strongest]),
            "rng_control": "caller holds update RNG fixed; direction generator is independent",
            "repeat_error_l2": repeat_error, "absolute_floor": absolute_floor, "probes": rows,
            "interpretation": "sampled complete-update directional amplification; no global stability certification"}

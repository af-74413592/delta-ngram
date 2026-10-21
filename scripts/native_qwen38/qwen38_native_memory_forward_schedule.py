"""Bounded, label-free 80M sensitivity and perturbed-trajectory probe.

This module never creates an optimizer, scores CE, or performs backward. The
base track supplies paired H_base/H_mem replacement measurements. Four other
tracks independently encode/write identical evidence: reference, repeat and
two fixed perturbation scales. These sampled observations do not certify
contraction or global stability, including when BF16 hides a perturbation.
"""
from __future__ import annotations

import math
import time

import torch

from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.qwen38_memory_batching import collate_memory_encoding_frames
from delta_ngram.qwen38_memory_diagnostics import (
    measure_encoding_sensitivity, summarize_encoding_sensitivity,
)
from delta_ngram.qwen38_memory_writer import write_memory_aware_native_block
from delta_ngram.qwen38_staged_batching import materialize_staged_snapshot
from delta_ngram.qwen38_train_batching import crop_evidence_lanes
from delta_ngram.qwen38_training import writer_evidence_features
from delta_ngram.state import SparseMemoryState
from qwen38_native_memory_schedule import collect_memory_encoding_microbatches


PROTOCOL = "qwen38_native_memory_forward_probe_v1"
DEFAULT_PROTOCOL = {
    "protocol": PROTOCOL, "relative_epsilons": [1e-3, 1e-2],
    "perturbation_seed": 17, "absolute_floor": 1e-8, "batch_limit": 3,
    "expected_global_trajectories": 12, "stop_g_p99": 1.,
    "retain_row_gain_samples": False,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def forward_probe_protocol(overrides=None):
    result = {**DEFAULT_PROTOCOL, "relative_epsilons": list(DEFAULT_PROTOCOL["relative_epsilons"])}
    if overrides is not None:
        _require(isinstance(overrides, dict) and not set(overrides).difference(result),
                 "unknown forward-probe protocol fields")
        result.update(overrides)
    epsilons = result["relative_epsilons"]
    _require(result["protocol"] == PROTOCOL and isinstance(epsilons, (list, tuple))
             and len(epsilons) == 2 and len(set(epsilons)) == 2
             and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                     and math.isfinite(value) and value > 0 for value in epsilons),
             "forward probe requires its registered two positive distinct perturbation scales")
    _require(type(result["perturbation_seed"]) is int and result["perturbation_seed"] >= 0
             and type(result["batch_limit"]) is int and result["batch_limit"] > 0
             and type(result["expected_global_trajectories"]) is int
             and result["expected_global_trajectories"] > 0
             and type(result["retain_row_gain_samples"]) is bool,
             "invalid forward-probe seed, batch size or trajectory count")
    _require(math.isfinite(result["absolute_floor"]) and result["absolute_floor"] > 0
             and math.isfinite(result["stop_g_p99"]) and result["stop_g_p99"] > 0,
             "forward-probe floors and report thresholds must be positive finite")
    result["relative_epsilons"] = list(epsilons)
    return result


def _clone_state(state):
    _require(isinstance(state, (BatchMemoryState, LazyDecayState)) and state.batch_size == 1
             and state.precision is None, "probe tracks require one-owner native D1 states")
    source = state.memory if isinstance(state, LazyDecayState) else state
    copied = BatchMemoryState(SparseMemoryState(
        source.rows.detach().clone(), source.values.detach().clone(), source.counts.detach().clone()),
        1, source.row_capacity)
    if isinstance(state, BatchMemoryState):
        return LazyDecayState.from_batch(copied)
    return LazyDecayState(copied, state.block_clock.detach().clone(),
                          state.cumulative_log_alpha.detach().clone(),
                          state.row_last_written_block.detach().clone(),
                          state.row_log_alpha_at_write.detach().clone())


def _tensors(state):
    values = [state.rows, state.values, state.counts]
    if isinstance(state, LazyDecayState):
        values += [state.block_clock, state.cumulative_log_alpha,
                   state.row_last_written_block, state.row_log_alpha_at_write]
    return values


def _state_norm(state):
    current = materialize_staged_snapshot(state)
    return float(torch.linalg.vector_norm(current.values.double()))


def _separation(first, second):
    first, second = materialize_staged_snapshot(first), materialize_staged_snapshot(second)
    rows = torch.unique(torch.cat((first.rows, second.rows)), sorted=True)
    a, ac = first.memory.lookup(rows)
    b, bc = second.memory.lookup(rows)
    _require(torch.equal(ac, bc), "paired probe tracks changed counts or write support")
    difference = a.double() - b.double()
    row_norms = torch.linalg.vector_norm(difference, dim=-1)
    return {"state_error_l2": float(torch.linalg.vector_norm(difference)),
            "maximum_row_error_l2": float(row_norms.max()) if row_norms.numel() else 0.,
            "rows": rows.numel()}


def _value_statistics(state):
    values = materialize_staged_snapshot(state).values.double()
    row_norms = torch.linalg.vector_norm(values, dim=-1)
    return {"state_l2": float(torch.linalg.vector_norm(values)),
            "state_rms": float(values.square().mean().sqrt()) if values.numel() else 0.,
            "maximum_row_l2": float(row_norms.max()) if row_norms.numel() else 0.,
            "rows": values.shape[0]}


def _seed_perturbations(reference, *, relative_epsilons, seed, absolute_floor):
    """Change only values at the current clock, measuring actual FP32 error."""
    effective = materialize_staged_snapshot(reference)
    norm = _state_norm(reference)
    _require(norm >= absolute_floor, "cannot seed a perturbation from an effectively zero state")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(effective.values.shape, generator=generator, dtype=torch.float64)
    direction /= torch.linalg.vector_norm(direction).clamp_min(absolute_floor)
    direction = direction.to(effective.values.device)
    owners = reference.memory._owner_ids(reference.rows)
    factor = (reference.cumulative_log_alpha[owners] - reference.row_log_alpha_at_write).exp()
    outputs, records = [], []
    for relative in relative_epsilons:
        step = relative * norm
        # Conversion through the current lazy decay factors preserves all
        # counters/clocks. No row is added or removed to inject the perturbation.
        changed = (reference.values.double() + step * direction / factor[:, None]).float()
        cloned = _clone_state(reference)
        perturbed = cloned.with_values(changed)
        actual = _separation(reference, perturbed)["state_error_l2"]
        _require(math.isfinite(actual) and actual > 0,
                 "registered perturbation rounded to zero in FP32; diagnostic is unassessable")
        _require(all(torch.equal(first, second) for first, second in zip(
            [reference.rows, reference.counts, reference.block_clock, reference.cumulative_log_alpha,
             reference.row_last_written_block, reference.row_log_alpha_at_write],
            [perturbed.rows, perturbed.counts, perturbed.block_clock, perturbed.cumulative_log_alpha,
             perturbed.row_last_written_block, perturbed.row_log_alpha_at_write])),
            "perturbation changed non-value state")
        outputs.append(perturbed)
        records.append({"relative_epsilon": relative, "requested_perturbation_l2": step,
                        "actual_fp32_perturbation_l2": actual,
                        "actual_relative_perturbation": actual / norm,
                        "reference_state_l2": norm, "seed": seed,
                        "direction": "CPU_float64_gaussian_normalized_full_state",
                        "changed": "values_only; rows/counts/clocks_unchanged"})
    return outputs, records


def _quantiles(values):
    if not values:
        return {"median": None, "p99": None, "maximum": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {"median": float(tensor.quantile(.5)), "p99": float(tensor.quantile(.99)),
            "maximum": float(tensor.max())}


@torch.no_grad()
def run_memory_forward_probe(session, writer, alpha, local, *, gather,
                             on_encoding=None, protocol=None):
    """Run the bounded six-call-per-wave probe and return rank-global metrics.

    local items contain ``index``, the complete ``plan``, and an EMPTY ``state``.
    Caller state objects and writer parameters remain untouched. on_encoding
    receives rank-local input-token counts before every native dispatch; the
    driver is responsible for durable global charges. Query frames are ignored.
    """
    policy = forward_probe_protocol(protocol)
    _require(callable(gather) and (on_encoding is None or callable(on_encoding)),
             "forward probe requires a collective gather and optional charge callback")
    _require(alpha.kind == "fixed" and alpha.initial_alpha == 1.,
             "the registered 80M forward probe uses the alpha=1 checkpoint")
    _require(all(value == policy for value in gather(policy)),
             "EP ranks disagree on the forward-probe protocol")
    identities = [index for values in gather([item["index"] for item in local]) for index in values]
    _require(len(identities) == policy["expected_global_trajectories"]
             and len(set(identities)) == len(identities)
             and all(type(index) is int and index >= 0 for index in identities),
             "forward probe must use the registered unique complete trajectories")
    _require(len({id(item) for item in local}) == len(local), "duplicate local trajectory owner")
    source_guards = []
    tracks = []
    for item in local:
        state = item["state"]
        _require(state.values.numel() == 0 and bool(item["plan"]),
                 "forward probe trajectories start empty and retain their complete frame plans")
        source_guards.extend((value, value._version) for value in _tensors(state))
        tracks.append({"index": item["index"], "plan": item["plan"],
                       "base": _clone_state(state), "reference": _clone_state(state),
                       "repeat": _clone_state(state), "perturbed_0": _clone_state(state),
                       "perturbed_1": _clone_state(state), "seed_records": None,
                       "seed_ordinal": None, "seed_write_count": None, "write_count": 0})
    parameter_guards = [(value, value._version, value.requires_grad)
                        for value in list(writer.parameters()) + list(writer.buffers())
                        + list(alpha.parameters()) + list(alpha.buffers())]
    kinds = ("base", "memory_base", "reference", "repeat", "perturbed_0", "perturbed_1")
    inputs, padded, calls = (dict.fromkeys(kinds, 0) for _ in range(3))
    gains, paths, seeds = [], [], []
    started = time.monotonic()
    waves = max(gather(max((len(item["plan"]) for item in tracks), default=0)))
    for ordinal in range(waves):
        ready = [(item, item["plan"][ordinal]["evidence"])
                 for item in tracks if ordinal < len(item["plan"])
                 and item["plan"][ordinal]["evidence"] is not None]
        maximum = max(gather(len(ready)))
        for start in range(0, maximum, policy["batch_limit"]):
            size = min(policy["batch_limit"], maximum - start)
            active = ready[start:start + size]
            frames = [frame for _, frame in active] + [None] * (size - len(active))
            encodings = {}
            for kind in kinds:
                state_key = "base" if kind == "memory_base" else kind
                states = None if kind == "base" else (
                    [item[state_key] for item, _ in active] + [None] * (size - len(active)))
                mb = collate_memory_encoding_frames(
                    frames, states, row_capacity=session.layout.row_capacity,
                    width=session.layout.row_width, device=session.device)
                cost = {"phase": "forward_diagnostic", "kind": "forward_" + kind,
                        "real_input_tokens": mb["real_input_tokens"],
                        "padded_input_tokens": mb["padded_input_tokens"], "ce_targets": 0,
                        "wave": ordinal, "batch_size": size}
                if on_encoding is not None:
                    on_encoding(cost)
                result = collect_memory_encoding_microbatches(
                    session, [mb], encoding="base" if kind == "base" else "memory")["microbatches"][0]
                blocks = crop_evidence_lanes(mb, result["hidden"], result["physical_rows"])
                for lane, (item, _) in enumerate(active):
                    _require(blocks[lane] is not None, "forward probe lost a real evidence lane")
                    blocks[lane]["physical_rows_read"] = result["physical_rows"][lane][
                        mb["valid_mask"][lane].cpu()].detach().clone()
                encodings[kind] = blocks
                inputs[kind] += cost["real_input_tokens"]
                padded[kind] += cost["padded_input_tokens"]
                calls[kind] += 1
            for lane, (item, frame) in enumerate(active):
                source, paired = encodings["base"][lane], encodings["memory_base"][lane]
                gain = measure_encoding_sensitivity(
                    writer, item["base"], paired, source, encoding_mode="full",
                    denominator_floor=policy["absolute_floor"])
                gain.update(trajectory_index=item["index"], frame_index=ordinal,
                            article_id=frame.get("article_id"), source=frame.get("source"))
                gains.append(gain)
                write_stats = {}
                for kind in ("base", "reference", "repeat", "perturbed_0", "perturbed_1"):
                    block = encodings[kind][lane]
                    _require(torch.equal(block["physical_rows_new"], source["physical_rows_new"]),
                             "paired probe tracks disagree on write addresses")
                    hidden = writer_evidence_features(
                        block["H_new"].to(session.device), mode="rmsnorm", eps=1e-6)
                    event_statistics = {}

                    def observe(events):
                        beta = events.beta.double()
                        event_statistics.update(
                            gate_mean=float(beta.mean()), gate_max=float(beta.max()),
                            target_rms=float(events.target.double().square().mean().sqrt()))

                    item[kind] = write_memory_aware_native_block(
                        writer, item[kind], hidden, block["physical_rows_new"].to(session.device),
                        torch.ones(hidden.shape[:2], dtype=torch.bool, device=session.device),
                        alpha, event_observer=observe)
                    _require(bool(item[kind].values.isfinite().all()), "nonfinite forward-probe writer state")
                    write_stats[kind] = {**_value_statistics(item[kind]), **event_statistics}
                item["write_count"] += 1
                ref_norm = write_stats["reference"]["state_l2"]
                repeat_error = _separation(item["reference"], item["repeat"])
                if item["seed_records"] is None and ref_norm >= policy["absolute_floor"]:
                    perturbed, seed_records = _seed_perturbations(
                        item["reference"], relative_epsilons=policy["relative_epsilons"],
                        seed=policy["perturbation_seed"] + item["index"],
                        absolute_floor=policy["absolute_floor"])
                    for index, (value, record) in enumerate(zip(perturbed, seed_records)):
                        item[f"perturbed_{index}"] = value
                        record.update(trajectory_index=item["index"], frame_index=ordinal)
                        seeds.append(record)
                    item["seed_records"], item["seed_ordinal"] = seed_records, ordinal
                    item["seed_write_count"] = item["write_count"]
                for index, relative in enumerate(policy["relative_epsilons"]):
                    separation = _separation(item["reference"], item[f"perturbed_{index}"])
                    error = separation["state_error_l2"]
                    initial = (item["seed_records"][index]["actual_fp32_perturbation_l2"]
                               if item["seed_records"] is not None else None)
                    path = {"trajectory_index": item["index"], "frame_index": ordinal,
                            "write_index": item["write_count"] - 1,
                            "relative_epsilon": relative, "seeded": initial is not None,
                            "blocks_after_seed": item["write_count"] - item["seed_write_count"] if initial is not None else None,
                            "initial_error_l2": initial, **separation,
                            "error_over_initial": error / initial if initial is not None else None,
                            "error_over_reference_state": error / ref_norm if ref_norm >= policy["absolute_floor"] else None,
                            "repeat_error_l2": repeat_error["state_error_l2"],
                            "repeat_error_to_signal": repeat_error["state_error_l2"] / max(error, policy["absolute_floor"]),
                            "reference_state_l2": ref_norm,
                            "seed_injected_after_this_write": item["seed_ordinal"] == ordinal,
                            "reference_write": write_stats["reference"],
                            "perturbed_write_before_optional_seed": write_stats[f"perturbed_{index}"],
                            "base_write": write_stats["base"]}
                    _require(all(not isinstance(value, float) or math.isfinite(value) for value in path.values()),
                             "nonfinite forward-probe trajectory metric")
                    paths.append(path)
            del encodings
    for value, version in source_guards:
        _require(value._version == version, "forward probe mutated an input trajectory state")
    for value, version, requires_grad in parameter_guards:
        _require(value._version == version and value.requires_grad == requires_grad,
                 "forward probe changed writer/alpha parameters or gradient flags")
    local_report = {"gains": gains, "paths": paths, "seeds": seeds,
                    "encoding_input_tokens": inputs, "encoding_padded_input_tokens": padded,
                    "encoding_native_calls": calls, "seconds": time.monotonic() - started,
                    "trajectories": len(tracks)}
    gathered = gather(local_report)
    gains = [row for report in gathered for row in report["gains"]]
    paths = [row for report in gathered for row in report["paths"]]
    seeds = [row for report in gathered for row in report["seeds"]]
    summary = summarize_encoding_sensitivity(gains, stop_p99=policy["stop_g_p99"])
    for record in gains:
        record["row_output_gain"] = _quantiles(record["row_output_gain_values"])
        if not policy["retain_row_gain_samples"]:
            record.pop("row_output_gain_values")
    twin_summaries = []
    for relative in policy["relative_epsilons"]:
        selected = [row for row in paths if row["relative_epsilon"] == relative and row["seeded"]]
        evolved = [row for row in selected if row["blocks_after_seed"] > 0]
        seeded_final = {}
        for row in selected:
            key = row["trajectory_index"]
            if key not in seeded_final or row["frame_index"] > seeded_final[key]["frame_index"]:
                seeded_final[key] = row
        # A one-write trajectory only measures the injected separation (= 1
        # after normalization). It has no subsequent dynamical observation and
        # must not enter the final amplification/contraction distribution.
        final = {key: row for key, row in seeded_final.items() if row["blocks_after_seed"] > 0}
        twin_summaries.append({
            "relative_epsilon": relative, "seeded_trajectories": len(seeded_final),
            "evolved_trajectories": len(final),
            "seed_only_trajectories": len(seeded_final) - len(final),
            "post_seed_write_observations": len(evolved),
            "error_over_initial_all_post_seed": _quantiles([row["error_over_initial"] for row in evolved]),
            "error_over_initial_final": _quantiles([row["error_over_initial"] for row in final.values()]),
            "error_over_reference_final": _quantiles([row["error_over_reference_state"] for row in final.values()
                                                       if row["error_over_reference_state"] is not None]),
            "final_distribution_scope": "only trajectories with at least one write after perturbation injection",
            "repeat_error_l2": _quantiles([row["repeat_error_l2"] for row in selected]),
            "zero_observed_error_blocks": sum(row["state_error_l2"] == 0 for row in evolved),
            "noise_dominates_signal_blocks": sum(row["repeat_error_to_signal"] >= 1 for row in evolved),
        })
    costs = {key: {kind: sum(report[key][kind] for report in gathered) for kind in kinds}
             for key in ("encoding_input_tokens", "encoding_padded_input_tokens")}
    rank_calls = [report["encoding_native_calls"] for report in gathered]
    _require(all(value == rank_calls[0] for value in rank_calls),
             "ranks completed different numbers of collective-bearing encoding calls")
    costs.update(collective_calls=rank_calls[0], native_calls_per_rank=rank_calls,
                 rank_call_sum={kind: sum(row[kind] for row in rank_calls) for kind in kinds},
                 call_definition="one collective call runs once on each rank; rank_call_sum is not an extra dispatch charge")
    return {"protocol": policy, "trajectories": sum(report["trajectories"] for report in gathered),
            "ready_waves": waves, "encoding_replacement": summary, "encoding_replacement_records": gains,
            "twin_trajectory_summary": twin_summaries, "twin_trajectory_records": paths,
            "perturbation_seeds": seeds, "costs": costs,
            "rank_seconds": [report["seconds"] for report in gathered],
            "critical_path_seconds": max(report["seconds"] for report in gathered),
            "training_targets": 0, "ce_targets": 0, "backward_calls": 0, "optimizer_steps": 0,
            "training_launch_authorized_by_this_probe": False,
            "interpretation": "finite sampled forward diagnostics only; zero BF16 error or small G is not a stability certificate",
            "risk_rule": "G p99 threshold is a conservative flag; finite probe completes without automatically launching training"}

"""Supplemental base-only twins and bounded final-query callbacks.

The original v1 entrypoint is unchanged. Six encoding calls per ready wave
produce both memory-aware twins and four base-only tracks. The latter share
the legacy H12 base cache and add no reader forwards. The driver separately charges
optional H12 feature capture and final-query CE before each native dispatch.
"""
from __future__ import annotations

import time

import torch

from delta_ngram.qwen38_memory_batching import collate_memory_encoding_frames
from delta_ngram.qwen38_memory_diagnostics import measure_encoding_sensitivity, summarize_encoding_sensitivity
from delta_ngram.qwen38_memory_writer import write_memory_aware_native_block
from delta_ngram.qwen38_train_batching import crop_evidence_lanes
from delta_ngram.qwen38_training import writer_evidence_features
from qwen38_native_memory_forward_schedule import (
    _clone_state, _quantiles, _require, _seed_perturbations, _separation,
    _tensors, _value_statistics, forward_probe_protocol,
)
from qwen38_native_memory_schedule import collect_memory_encoding_microbatches


PROTOCOL = "qwen38_native_memory_supplemental_probe_v2"
FINAL_CONDITIONS = ("memory_ref", "memory_eps01", "base_ref", "base_eps01")


def supplemental_probe_protocol(overrides=None):
    result = {**forward_probe_protocol(), "protocol": PROTOCOL, "expected_evolved_trajectories": 6}
    if overrides is not None:
        _require(isinstance(overrides, dict) and not set(overrides).difference(result),
                 "unknown supplemental-probe protocol fields")
        result.update(overrides)
    inherited = {key: value for key, value in result.items()
                 if key not in ("protocol", "expected_evolved_trajectories")}
    forward_probe_protocol(inherited)
    _require(result["protocol"] == PROTOCOL and result["relative_epsilons"] == [1e-3, 1e-2]
             and type(result["expected_evolved_trajectories"]) is int
             and 0 <= result["expected_evolved_trajectories"] <= result["expected_global_trajectories"],
             "supplemental probe requires registered scales and evolved-trajectory count")
    return result


def _path_records(item, family, ordinal, policy, write_stats):
    prefix = "base_" if family == "base" else ""
    reference_key = "base" if family == "base" else "reference"
    reference = item[reference_key]
    ref_norm = write_stats[reference_key]["state_l2"]
    repeat = _separation(reference, item[prefix + "repeat"])
    seed_key = prefix + "seed_records"
    if item[seed_key] is None and ref_norm >= policy["absolute_floor"]:
        states, seeds = _seed_perturbations(
            reference, relative_epsilons=policy["relative_epsilons"],
            seed=policy["perturbation_seed"] + item["index"], absolute_floor=policy["absolute_floor"])
        for index, (state, record) in enumerate(zip(states, seeds)):
            item[prefix + f"perturbed_{index}"] = state
            record.update(trajectory_index=item["index"], frame_index=ordinal, track_family=family)
        item[seed_key] = seeds
        item[prefix + "seed_ordinal"] = ordinal
        item[prefix + "seed_write_count"] = item["write_count"]
        new_seeds = seeds
    else:
        new_seeds = []
    paths = []
    for index, relative in enumerate(policy["relative_epsilons"]):
        separation = _separation(reference, item[prefix + f"perturbed_{index}"])
        initial = (item[seed_key][index]["actual_fp32_perturbation_l2"] if item[seed_key] else None)
        error = separation["state_error_l2"]
        paths.append({
            "track_family": family, "trajectory_index": item["index"], "frame_index": ordinal,
            "write_index": item["write_count"] - 1, "relative_epsilon": relative,
            "seeded": initial is not None,
            "blocks_after_seed": item["write_count"] - item[prefix + "seed_write_count"] if initial else None,
            "initial_error_l2": initial, **separation,
            "error_over_initial": error / initial if initial else None,
            "error_over_reference_state": error / ref_norm if ref_norm >= policy["absolute_floor"] else None,
            "repeat_error_l2": repeat["state_error_l2"],
            "repeat_error_to_signal": repeat["state_error_l2"] / max(error, policy["absolute_floor"]),
            "repeat_error_over_reference_state": (repeat["state_error_l2"] / ref_norm
                                                   if ref_norm >= policy["absolute_floor"] else None),
            "repeat_definition": "independent from empty; autonomous trajectory separation, not a local noise floor",
            "reference_state_l2": ref_norm,
            "seed_injected_after_this_write": item[prefix + "seed_ordinal"] == ordinal,
            "reference_write": write_stats[reference_key],
            "perturbed_write_before_optional_seed": write_stats[prefix + f"perturbed_{index}"],
        })
    return paths, new_seeds


def _summaries(paths, policy):
    summaries = []
    for family in ("memory", "base"):
        for relative in policy["relative_epsilons"]:
            selected = [row for row in paths if row["track_family"] == family
                        and row["relative_epsilon"] == relative and row["seeded"]]
            evolved = [row for row in selected if row["blocks_after_seed"] > 0]
            final_seeded = {}
            for row in selected:
                previous = final_seeded.get(row["trajectory_index"])
                if previous is None or row["frame_index"] > previous["frame_index"]:
                    final_seeded[row["trajectory_index"]] = row
            final = [row for row in final_seeded.values() if row["blocks_after_seed"] > 0]
            summaries.append({
                "track_family": family, "relative_epsilon": relative,
                "seeded_trajectories": len(final_seeded), "evolved_trajectories": len(final),
                "seed_only_trajectories": len(final_seeded) - len(final),
                "post_seed_write_observations": len(evolved),
                "error_over_initial_all_post_seed": _quantiles([row["error_over_initial"] for row in evolved]),
                "error_over_initial_final": _quantiles([row["error_over_initial"] for row in final]),
                "error_over_reference_final": _quantiles([row["error_over_reference_state"] for row in final
                                                           if row["error_over_reference_state"] is not None]),
                "repeat_error_l2_final": _quantiles([row["repeat_error_l2"] for row in final]),
                "repeat_error_over_reference_final": _quantiles([row["repeat_error_over_reference_state"]
                    for row in final if row["repeat_error_over_reference_state"] is not None]),
                "zero_observed_error_blocks": sum(row["state_error_l2"] == 0 for row in evolved),
                "repeat_at_least_perturbed_separation_blocks": sum(row["repeat_error_to_signal"] >= 1 for row in evolved),
                "repeat_definition": "autonomous repeated trajectory; never subtracted from perturbed separation",
                "final_distribution_scope": "only trajectories with writes after perturbation injection",
            })
    return summaries


@torch.no_grad()
def run_memory_supplemental_probe(session, writer, alpha, local, *, gather,
                                  on_encoding=None, on_final_states=None, on_capture=None,
                                  h12_base_blocks=None, protocol=None):
    """Run six encodings per wave; callback scores only evolved final states.

    ``on_capture`` receives one dictionary per real lane before writing, with
    state_before, base_block, mem_block, H12_block, reference_block and identity.
    It may return JSON metrics and must not execute collective forwards.
    ``on_final_states`` is called once on every rank with local task dictionaries
    containing plan/frame/index and four independent states. It owns reservation
    and scoring. Its global result must report an integer ``ce_targets``.
    """
    policy = supplemental_probe_protocol(protocol)
    _require(callable(gather) and all(value is None or callable(value)
             for value in (on_encoding, on_final_states, on_capture)), "invalid supplemental callbacks")
    _require(alpha.kind == "fixed" and alpha.initial_alpha == 1., "supplemental probe uses alpha=1")
    _require(all(value == policy for value in gather(policy)), "ranks disagree on supplemental policy")
    identities = [index for values in gather([item["index"] for item in local]) for index in values]
    _require(len(identities) == policy["expected_global_trajectories"]
             and len(set(identities)) == len(identities)
             and all(type(index) is int and index >= 0 for index in identities), "invalid trajectory identities")
    _require(len({id(item) for item in local}) == len(local), "duplicate local trajectory owner")
    required_h12 = {(item["index"], ordinal) for item in local
                    for ordinal, frame in enumerate(item["plan"]) if frame["evidence"] is not None}
    _require(isinstance(h12_base_blocks, dict) and required_h12.issubset(h12_base_blocks),
             "every local write requires its actual legacy H12 base encoding")
    tracks, source_guards = [], []
    state_keys = ("base", "reference", "repeat", "perturbed_0", "perturbed_1",
                  "base_repeat", "base_perturbed_0", "base_perturbed_1")
    for item in local:
        _require(item["state"].values.numel() == 0 and bool(item["plan"]), "supplemental tracks start empty")
        source_guards.extend((value, value._version) for value in _tensors(item["state"]))
        tracks.append({"index": item["index"], "plan": item["plan"], "write_count": 0,
                       **{key: _clone_state(item["state"]) for key in state_keys},
                       **{prefix + name: None for prefix in ("", "base_")
                          for name in ("seed_records", "seed_ordinal", "seed_write_count")}})
    parameter_guards = [(value, value._version, value.requires_grad)
                        for value in list(writer.parameters()) + list(writer.buffers())
                        + list(alpha.parameters()) + list(alpha.buffers())]
    kinds = ("base", "memory_base", "reference", "repeat", "perturbed_0", "perturbed_1")
    inputs, padded, calls = (dict.fromkeys(kinds, 0) for _ in range(3))
    gains, paths, seeds, captures = [], [], [], []
    started = time.monotonic()
    waves = max(gather(max((len(item["plan"]) for item in tracks), default=0)))
    for ordinal in range(waves):
        ready = [(item, item["plan"][ordinal]["evidence"]) for item in tracks
                 if ordinal < len(item["plan"]) and item["plan"][ordinal]["evidence"] is not None]
        maximum = max(gather(len(ready)))
        for start in range(0, maximum, policy["batch_limit"]):
            size = min(policy["batch_limit"], maximum - start)
            active = ready[start:start + size]
            frames = [frame for _, frame in active] + [None] * (size - len(active))
            encodings = {}
            for kind in kinds:
                state_key = "base" if kind == "memory_base" else kind
                states = None if kind == "base" else [item[state_key] for item, _ in active] + [None] * (size - len(active))
                mb = collate_memory_encoding_frames(frames, states, row_capacity=session.layout.row_capacity,
                                                   width=session.layout.row_width, device=session.device)
                cost = {"phase": "supplemental_diagnostic", "kind": "supplemental_" + kind,
                        "real_input_tokens": mb["real_input_tokens"], "padded_input_tokens": mb["padded_input_tokens"],
                        "ce_targets": 0, "wave": ordinal, "batch_size": size}
                if on_encoding is not None:
                    on_encoding(cost)
                result = collect_memory_encoding_microbatches(session, [mb], encoding="base" if kind == "base" else "memory")["microbatches"][0]
                blocks = crop_evidence_lanes(mb, result["hidden"], result["physical_rows"])
                for lane, (item, _) in enumerate(active):
                    _require(blocks[lane] is not None, "supplement lost real lane")
                    blocks[lane]["physical_rows_read"] = result["physical_rows"][lane][mb["valid_mask"][lane].cpu()].detach().clone()
                encodings[kind] = blocks
                inputs[kind] += cost["real_input_tokens"]
                padded[kind] += cost["padded_input_tokens"]
                calls[kind] += 1
            for lane, (item, frame) in enumerate(active):
                source, paired = encodings["base"][lane], encodings["memory_base"][lane]
                h12 = h12_base_blocks[(item["index"], ordinal)]
                _require(h12 is not None and torch.equal(h12["physical_rows_new"], source["physical_rows_new"])
                         and h12["H_new"].shape == source["H_new"].shape,
                         "legacy H12 and matched base must share write addresses and hidden shapes")
                gain = measure_encoding_sensitivity(writer, item["base"], paired, source,
                    encoding_mode="full", denominator_floor=policy["absolute_floor"], include_relationship=True)
                gain.update(trajectory_index=item["index"], frame_index=ordinal,
                            article_id=frame.get("article_id"), source=frame.get("source"))
                gains.append(gain)
                if on_capture is not None:
                    captured = on_capture({"trajectory_index": item["index"], "frame_index": ordinal,
                        "frame": frame, "state_before": item["base"], "base_block": source,
                        "mem_block": paired, "reference_block": encodings["reference"][lane],
                        "H12_block": h12,
                        "batch_size": size, "lane": lane})
                    if captured is not None:
                        captures.append(captured)
                write_stats = {}
                base_hidden = writer_evidence_features(h12["H_new"].to(session.device), mode="rmsnorm", eps=1e-6)
                for kind in state_keys:
                    block = h12 if kind == "base" or kind.startswith("base_") else encodings[kind][lane]
                    _require(torch.equal(block["physical_rows_new"], source["physical_rows_new"]), "write addresses differ")
                    hidden = (base_hidden if kind == "base" or kind.startswith("base_") else
                              writer_evidence_features(block["H_new"].to(session.device), mode="rmsnorm", eps=1e-6))
                    statistics = {}
                    def observe(events):
                        beta = events.beta.double()
                        statistics.update(gate_mean=float(beta.mean()) if beta.numel() else None,
                            gate_max=float(beta.max()) if beta.numel() else None,
                            target_rms=float(events.target.double().square().mean().sqrt()) if beta.numel() else None)
                    item[kind] = write_memory_aware_native_block(writer, item[kind], hidden,
                        block["physical_rows_new"].to(session.device),
                        torch.ones(hidden.shape[:2], dtype=torch.bool, device=session.device), alpha, event_observer=observe)
                    _require(bool(item[kind].values.isfinite().all()), "nonfinite supplemental state")
                    write_stats[kind] = {**_value_statistics(item[kind]), **statistics}
                item["write_count"] += 1
                for family in ("memory", "base"):
                    new_paths, new_seeds = _path_records(item, family, ordinal, policy, write_stats)
                    paths.extend(new_paths)
                    seeds.extend(new_seeds)
            del encodings
    final_local = []
    for item in tracks:
        evolved = (item["seed_write_count"] is not None and item["base_seed_write_count"] is not None
                   and item["write_count"] > max(item["seed_write_count"], item["base_seed_write_count"]))
        if evolved:
            _require(item["plan"][-1]["evidence"] is None and item["plan"][-1]["query"] is not None,
                     "final scoring requires a query-only last frame")
            final_local.append({"index": item["index"], "trajectory_index": item["index"],
                "plan": item["plan"], "frame": item["plan"][-1]["query"], "write_count": item["write_count"],
                "blocks_after_seed": item["write_count"] - item["seed_write_count"],
                "states": dict(zip(FINAL_CONDITIONS, (item["reference"], item["perturbed_1"], item["base"], item["base_perturbed_1"])))})
    evolved_ids = [index for ids in gather([item["index"] for item in final_local]) for index in ids]
    _require(len(evolved_ids) == policy["expected_evolved_trajectories"] and len(set(evolved_ids)) == len(evolved_ids),
             "evolved final-scoring trajectory set differs from registration")
    final_guard = [(tensor, tensor._version) for item in final_local for state in item["states"].values() for tensor in _tensors(state)]
    final_result = on_final_states(final_local) if on_final_states is not None else {"ce_targets": 0, "status": "not_requested"}
    _require(isinstance(final_result, dict) and type(final_result.get("ce_targets")) is int
             and final_result["ce_targets"] >= 0, "final scorer must report global CE targets")
    _require(all(value == final_result["ce_targets"] for value in gather(final_result["ce_targets"])),
             "ranks disagree on final CE charge")
    for value, version in source_guards + final_guard:
        _require(value._version == version, "supplemental callback mutated an input or final state")
    for value, version, requires_grad in parameter_guards:
        _require(value._version == version and value.requires_grad == requires_grad,
                 "supplemental probe changed writer/alpha parameters or gradient flags")
    local_report = {"gains": gains, "paths": paths, "seeds": seeds, "captures": captures,
        "encoding_input_tokens": inputs, "encoding_padded_input_tokens": padded, "encoding_native_calls": calls,
        "seconds": time.monotonic() - started, "trajectories": len(tracks)}
    gathered = gather(local_report)
    gains = [row for report in gathered for row in report["gains"]]
    paths = [row for report in gathered for row in report["paths"]]
    summary = summarize_encoding_sensitivity(gains, stop_p99=policy["stop_g_p99"], monitor_only=True)
    for record in gains:
        record["row_output_gain"] = _quantiles(record["row_output_gain_values"])
        if not policy["retain_row_gain_samples"]:
            record.pop("row_output_gain_values")
    costs = {key: {kind: sum(report[key][kind] for report in gathered) for kind in kinds}
             for key in ("encoding_input_tokens", "encoding_padded_input_tokens")}
    rank_calls = [report["encoding_native_calls"] for report in gathered]
    _require(all(value == rank_calls[0] for value in rank_calls), "ranks differ in encoding dispatch counts")
    costs.update(collective_calls=rank_calls[0], native_calls_per_rank=rank_calls,
        rank_call_sum={kind: sum(row[kind] for row in rank_calls) for kind in kinds},
        base_twin_extra_encoding_calls=0,
        call_definition="one collective call per rank; rank sum is not an extra dispatch charge")
    return {"protocol": policy, "trajectories": sum(report["trajectories"] for report in gathered),
        "ready_waves": waves, "encoding_replacement": summary, "encoding_replacement_records": gains,
        "twin_trajectory_summary": _summaries(paths, policy), "twin_trajectory_records": paths,
        "perturbation_seeds": [row for report in gathered for row in report["seeds"]],
        "capture_records": [row for report in gathered for row in report["captures"]],
        "final_query_result": final_result, "final_query_trajectory_indices": sorted(evolved_ids),
        "costs": costs, "rank_seconds": [report["seconds"] for report in gathered],
        "critical_path_seconds": max(report["seconds"] for report in gathered),
        "training_targets": 0, "ce_targets": final_result["ce_targets"], "backward_calls": 0, "optimizer_steps": 0,
        "training_launch_authorized_by_this_probe": False,
        "base_control_encoding": "actual legacy H12 cache shared by all four base tracks",
        "g_direct_state": "H12 base-reference trajectory; differs from v1 matched-small-base trajectory",
        "interpretation": "observed platform sensitivity; neither G nor twin separations certify contraction or MoE causality",
        "risk_rule": "G threshold monitoring only; no automatic stop or training authorization"}

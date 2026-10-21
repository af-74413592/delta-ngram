"""Independent repeat controls; no injected state perturbation or extra G probe."""
from __future__ import annotations

import time

import torch

from delta_ngram.qwen38_memory_batching import collate_memory_encoding_frames
from delta_ngram.qwen38_memory_writer import write_memory_aware_native_block
from delta_ngram.qwen38_train_batching import crop_evidence_lanes
from delta_ngram.qwen38_training import writer_evidence_features
from qwen38_native_memory_forward_schedule import _clone_state, _quantiles, _require, _separation, _tensors, _value_statistics
from qwen38_native_memory_schedule import collect_memory_encoding_microbatches


PROTOCOL = "qwen38_native_memory_repeat_probe_v3"
FINAL_CONDITIONS = ("memory_ref", "memory_repeat", "base_ref", "base_repeat")


def repeat_probe_protocol(overrides=None):
    result = {"protocol": PROTOCOL, "batch_limit": 3, "expected_global_trajectories": 12,
              "expected_evolved_trajectories": 6, "absolute_floor": 1e-8}
    if overrides is not None:
        _require(isinstance(overrides, dict) and not set(overrides).difference(result),
                 "unknown repeat-only protocol fields")
        result.update(overrides)
    _require(result["protocol"] == PROTOCOL and type(result["batch_limit"]) is int
             and result["batch_limit"] == 3 and type(result["expected_global_trajectories"]) is int
             and result["expected_global_trajectories"] > 0
             and type(result["expected_evolved_trajectories"]) is int
             and 0 <= result["expected_evolved_trajectories"] <= result["expected_global_trajectories"]
             and type(result["absolute_floor"]) is float and result["absolute_floor"] == 1e-8,
             "repeat-only protocol requires its registered trajectory counts, B3 and absolute floor")
    return result


def score_repeat_query_states(session, tasks, *, gather, on_score, batch_limit=3):
    """Reuse V2 native CE arithmetic with explicit repeat-state label mapping.

    The reused scorer treats its state labels as opaque; no perturbation is
    created there. Both input states and all public costs/results are mapped
    to the actual reference/repeat names. Training/runtime modules stay intact.
    """
    from qwen38_native_memory_probe_runtime import CONDITIONS, score_final_query_states
    mapping = dict(zip(CONDITIONS, FINAL_CONDITIONS))
    adapted = []
    for task in tasks:
        _require(set(task["states"]) == set(FINAL_CONDITIONS), "repeat query conditions differ")
        adapted.append({**task, "states": {old: task["states"][new] for old, new in mapping.items()}})
    def charge(row):
        return on_score({**row, "phase": "repeat_final_query", "kind": mapping[row["kind"]]})
    result = score_final_query_states(session, adapted, gather=gather, on_score=charge,
                                      conditions=CONDITIONS, batch_limit=batch_limit)
    rename = {"memory_perturbed_minus_reference_nll": "memory_repeat_minus_reference_nll",
              "base_perturbed_minus_reference_nll": "base_repeat_minus_reference_nll",
              **{old + "_nll": new + "_nll" for old, new in mapping.items()}}
    return {**result, "protocol": "qwen38_repeat_final_query_scoring_v3", "conditions": list(FINAL_CONDITIONS),
        "records": [{**row, "condition": mapping[row["condition"]]} for row in result["records"]],
        "paired_records": [{rename.get(key, key): value for key, value in row.items()} for row in result["paired_records"]],
        "summary": {rename.get(key, key): value for key, value in result["summary"].items()},
        "state_perturbations_injected": 0,
        "interpretation": "independent repeated final states on identical held-out queries; signed repeat-minus-reference NLL, not a quality comparison"}


@torch.no_grad()
def run_memory_repeat_probe(session, writer, alpha, local, *, gather, h12_base_blocks,
                            on_encoding=None, on_final_states=None, protocol=None):
    """Two memory reader calls per ready wave plus two writes reusing H12.

    All four states start empty and evolve independently. No state is copied
    from its partner after initialization. H12 is encoded once by the driver.
    Query callbacks run on every rank, only for final frames after >1 writes.
    """
    policy = repeat_probe_protocol(protocol)
    _require(callable(gather) and all(callback is None or callable(callback)
             for callback in (on_encoding, on_final_states)), "invalid repeat callbacks")
    _require(alpha.kind == "fixed" and alpha.initial_alpha == 1., "repeat probe uses frozen alpha=1 source")
    _require(all(value == policy for value in gather(policy)), "ranks disagree on repeat protocol")
    identities = [index for values in gather([item["index"] for item in local]) for index in values]
    _require(len(identities) == policy["expected_global_trajectories"] and len(set(identities)) == len(identities)
             and all(type(index) is int and index >= 0 for index in identities), "repeat trajectory identity mismatch")
    required_h12 = {(item["index"], ordinal) for item in local for ordinal, step in enumerate(item["plan"])
                    if step["evidence"] is not None}
    _require(isinstance(h12_base_blocks, dict) and required_h12.issubset(h12_base_blocks),
             "repeat controls require every actual legacy H12 block")
    tracks, source_guards = [], []
    for item in local:
        _require(item["state"].values.numel() == 0 and bool(item["plan"]), "repeat controls start empty")
        source_guards.extend((value, value._version) for value in _tensors(item["state"]))
        tracks.append({"index": item["index"], "plan": item["plan"], "writes": 0,
                       **{key: _clone_state(item["state"]) for key in FINAL_CONDITIONS}})
    parameter_guards = [(value, value._version, value.requires_grad)
                        for value in list(writer.parameters()) + list(writer.buffers())
                        + list(alpha.parameters()) + list(alpha.buffers())]
    kinds = FINAL_CONDITIONS[:2]
    inputs, padded, calls = (dict.fromkeys(kinds, 0) for _ in range(3))
    records = []
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
                mb = collate_memory_encoding_frames(frames,
                    [item[kind] for item, _ in active] + [None] * (size - len(active)),
                    row_capacity=session.layout.row_capacity, width=session.layout.row_width, device=session.device)
                cost = {"phase": "repeat_diagnostic", "kind": "repeat_" + kind,
                        "real_input_tokens": mb["real_input_tokens"], "padded_input_tokens": mb["padded_input_tokens"],
                        "ce_targets": 0, "wave": ordinal, "batch_size": size}
                if on_encoding is not None:
                    on_encoding(cost)
                result = collect_memory_encoding_microbatches(session, [mb], encoding="memory")["microbatches"][0]
                encodings[kind] = crop_evidence_lanes(mb, result["hidden"], result["physical_rows"])
                inputs[kind] += cost["real_input_tokens"]
                padded[kind] += cost["padded_input_tokens"]
                calls[kind] += 1
            for lane, (item, frame) in enumerate(active):
                h12 = h12_base_blocks[(item["index"], ordinal)]
                base_hidden = writer_evidence_features(h12["H_new"].to(session.device), mode="rmsnorm", eps=1e-6)
                stats = {}
                for kind in FINAL_CONDITIONS:
                    block = h12 if kind.startswith("base_") else encodings[kind][lane]
                    _require(block is not None and block["H_new"].shape == h12["H_new"].shape
                             and torch.equal(block["physical_rows_new"], h12["physical_rows_new"]),
                             "repeat and H12 block framing differs")
                    hidden = base_hidden if kind.startswith("base_") else writer_evidence_features(
                        block["H_new"].to(session.device), mode="rmsnorm", eps=1e-6)
                    item[kind] = write_memory_aware_native_block(writer, item[kind], hidden,
                        block["physical_rows_new"].to(session.device),
                        torch.ones(hidden.shape[:2], dtype=torch.bool, device=session.device), alpha)
                    _require(bool(item[kind].values.isfinite().all()), "nonfinite repeat state")
                    stats[kind] = _value_statistics(item[kind])
                item["writes"] += 1
                for family in ("memory", "base"):
                    separation = _separation(item[family + "_ref"], item[family + "_repeat"])
                    norm = stats[family + "_ref"]["state_l2"]
                    records.append({"trajectory_index": item["index"], "frame_index": ordinal,
                        "write_index": item["writes"] - 1, "track_family": family,
                        **separation, "reference_state_l2": norm,
                        "repeat_over_reference_state": separation["state_error_l2"] / norm if norm >= policy["absolute_floor"] else None,
                        "first_write_from_empty": item["writes"] == 1,
                        "definition": "independently evolved reference/repeat from empty; no injected perturbation"})
            del encodings
    final_tasks = []
    for item in tracks:
        if item["writes"] > 1:
            _require(item["plan"][-1]["query"] is not None and item["plan"][-1]["evidence"] is None,
                     "repeat final query must be a final query-only frame")
            final_tasks.append({"index": item["index"], "trajectory_index": item["index"],
                "plan": item["plan"], "frame": item["plan"][-1]["query"], "write_count": item["writes"],
                "states": {key: item[key] for key in FINAL_CONDITIONS}})
    evolved = [index for values in gather([task["index"] for task in final_tasks]) for index in values]
    _require(len(evolved) == policy["expected_evolved_trajectories"] and len(set(evolved)) == len(evolved),
             "repeat final-query population differs from registration")
    final_guards = [(value, value._version) for task in final_tasks for state in task["states"].values() for value in _tensors(state)]
    query = on_final_states(final_tasks) if on_final_states is not None else {"ce_targets": 0, "status": "not_requested"}
    _require(isinstance(query, dict) and type(query.get("ce_targets")) is int and query["ce_targets"] >= 0,
             "repeat scorer must return global CE charge")
    _require(all(value == query["ce_targets"] for value in gather(query["ce_targets"])), "repeat CE charge differs by rank")
    for value, version in source_guards + final_guards:
        _require(value._version == version, "repeat probe mutated input/final state")
    for value, version, requires_grad in parameter_guards:
        _require(value._version == version and value.requires_grad == requires_grad, "repeat probe changed writer or alpha")
    reports = gather({"records": records, "encoding_input_tokens": inputs, "encoding_padded_input_tokens": padded,
        "encoding_native_calls": calls, "trajectories": len(tracks), "seconds": time.monotonic() - started})
    records = [row for report in reports for row in report["records"]]
    summaries = []
    for family in ("memory", "base"):
        selected = [row for row in records if row["track_family"] == family]
        final = {}
        for row in selected:
            if row["trajectory_index"] not in final or row["frame_index"] > final[row["trajectory_index"]]["frame_index"]:
                final[row["trajectory_index"]] = row
        endpoint = [row for row in final.values() if row["trajectory_index"] in evolved]
        summaries.append({"track_family": family, "final_evolved_trajectories": len(endpoint),
            "final_state_error_l2": _quantiles([row["state_error_l2"] for row in endpoint]),
            "final_repeat_over_reference": _quantiles([row["repeat_over_reference_state"] for row in endpoint
                                                        if row["repeat_over_reference_state"] is not None]),
            "first_write_from_empty_error_l2": _quantiles([row["state_error_l2"] for row in selected if row["first_write_from_empty"]])})
    rank_calls = [report["encoding_native_calls"] for report in reports]
    _require(all(value == rank_calls[0] for value in rank_calls), "repeat native call count differs by rank")
    costs = {key: {kind: sum(report[key][kind] for report in reports) for kind in kinds}
             for key in ("encoding_input_tokens", "encoding_padded_input_tokens")}
    costs.update(collective_calls=rank_calls[0], native_calls_per_rank=rank_calls, base_control_extra_encoding_calls=0)
    return {"protocol": policy, "trajectories": sum(report["trajectories"] for report in reports),
        "ready_waves": waves, "repeat_trajectory_records": records, "repeat_trajectory_summary": summaries,
        "final_query_result": query, "final_query_trajectory_indices": sorted(evolved), "costs": costs,
        "rank_seconds": [report["seconds"] for report in reports], "critical_path_seconds": max(report["seconds"] for report in reports),
        "training_targets": 0, "ce_targets": query["ce_targets"], "backward_calls": 0, "optimizer_steps": 0,
        "state_perturbations_injected": 0, "base_control_encoding": "shared actual legacy H12 features, independently updated states",
        "interpretation": "independent repeat state and final-query NLL sensitivity; no new stability threshold or training gate",
        "training_launch_authorized_by_this_probe": False}

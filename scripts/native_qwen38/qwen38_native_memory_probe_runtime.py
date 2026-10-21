"""Supplemental probe I/O: H12 base features and final-query-only CE scoring.

The original forward-only probe and training/evaluation algorithms are not
modified. Every query dispatch is charged before the native model is called.
"""
from __future__ import annotations

import math
import torch

from delta_ngram.batching import BatchMemoryState
from delta_ngram.qwen38_memory_batching import collate_memory_encoding_frames, detached_encoding_snapshot
from delta_ngram.qwen38_staged_batching import collate_staged_frames
from delta_ngram.qwen38_train_batching import crop_evidence_lanes
from delta_ngram.qwen38_training import writer_evidence_features
from delta_ngram.state import SparseMemoryState
from qwen38_native_ep4_schedule import _runtime
from qwen38_native_memory_schedule import collect_memory_encoding_microbatches, preserve_encoding_rng


CONDITIONS = ("memory_ref", "memory_eps01", "base_ref", "base_eps01")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@torch.no_grad()
def collect_probe_h12_base(session, local, *, gather, on_encoding=None, batch_limit=12):
    """Retain the original trajectory-major H12 base batch and lane ordering."""
    _require(type(batch_limit) is int and batch_limit == 12, "registered base cache uses H12")
    tasks = [(item["index"], ordinal, step["evidence"])
             for item in local for ordinal, step in enumerate(item["plan"])
             if step["evidence"] is not None]
    cache, calls, real, padded = {}, 0, 0, 0
    maximum = max(gather(len(tasks)))
    for start in range(0, maximum, batch_limit):
        size = min(batch_limit, maximum - start)
        active = tasks[start:start + size]
        mb = collate_memory_encoding_frames(
            [task[2] for task in active] + [None] * (size - len(active)),
            row_capacity=session.layout.row_capacity, width=session.layout.row_width, device=session.device)
        cost = {"phase": "supplemental_diagnostic", "kind": "supplemental_h12_base",
                "real_input_tokens": mb["real_input_tokens"], "padded_input_tokens": mb["padded_input_tokens"],
                "ce_targets": 0, "batch_size": size, "wave": None}
        if on_encoding is not None:
            on_encoding(cost)
        captured = collect_memory_encoding_microbatches(session, [mb], encoding="base")["microbatches"][0]
        blocks = crop_evidence_lanes(mb, captured["hidden"], captured["physical_rows"])
        for lane, (task, block) in enumerate(zip(active, blocks)):
            _require(block is not None and task[:2] not in cache, "duplicate or missing H12 block")
            block["physical_rows_read"] = captured["physical_rows"][lane][mb["valid_mask"][lane].cpu()].clone()
            block["encoding_batch_size"], block["encoding_lane"] = size, lane
            cache[task[:2]] = block
        calls += 1
        real += cost["real_input_tokens"]
        padded += cost["padded_input_tokens"]
    return {"blocks": cache, "costs": {"native_calls_rank_local": calls,
            "real_input_tokens_rank_local": real, "padded_input_tokens_rank_local": padded, "ce_targets": 0}}


def _query_state(states, size, *, row_capacity, width, device):
    memories = []
    for state in states:
        _require(state.row_capacity == row_capacity and state.width == width
                 and state.values.device == torch.device(device) and state.values.dtype == torch.float32,
                 "final-query state dimensions, device or dtype differ from native layout")
        memories.append(detached_encoding_snapshot(state).memory)
    memories += [SparseMemoryState.empty(width, device) for _ in range(size - len(states))]
    packed = BatchMemoryState.from_states(memories, row_capacity)
    if not packed.values.numel():
        # Preserve the existing DEV64 empty-memory sentinel arithmetic.
        sentinel = SparseMemoryState(torch.tensor([1], dtype=torch.long, device=device),
                                     torch.zeros(1, width, device=device), torch.zeros(1, device=device))
        packed = BatchMemoryState(sentinel, size, row_capacity)
    _require(not packed.values.requires_grad, "supplemental query acquired a training graph")
    return packed


@torch.no_grad()
def score_final_query_states(session, tasks, *, gather, on_score,
                             conditions=CONDITIONS, batch_limit=3):
    """Score only held-out final frames from the same evolved trajectories.

    tasks contain trajectory_index, frame, plan and states keyed by CONDITIONS.
    on_score receives rank-local CE/input costs, and must durably gather/charge
    them before returning. It is mandatory even on an all-inactive local rank.
    No query state is used for a later write or modified by this helper.
    """
    _require(callable(on_score), "final-query scoring requires durable precharge")
    _require(tuple(conditions) == CONDITIONS and batch_limit == 3,
             "supplemental final-query protocol requires its fixed four conditions and B3 cap")
    _runtime(session)
    tasks = tuple(tasks)
    for task in tasks:
        plan, frame = task["plan"], task["frame"]
        _require(bool(plan) and plan[-1]["query"] == frame and plan[-1]["evidence"] is None
                 and sum(step["evidence"] is not None for step in plan) > 1,
                 "score only an evolved trajectory's final held-out query, never an earlier written target")
        _require(set(task["states"]) == set(conditions), "final-query conditions do not match registration")
        _require(type(task["trajectory_index"]) is int, "trajectory index must be an integer")
    identities = [index for indices in gather([task["trajectory_index"] for task in tasks]) for index in indices]
    _require(len(set(identities)) == len(identities), "duplicate final-query trajectory")
    records, calls, charged, real, padded = [], 0, 0, 0, 0
    maximum = max(gather(len(tasks)))
    for start in range(0, maximum, batch_limit):
        size = min(batch_limit, maximum - start)
        active = tasks[start:start + size]
        mb = collate_staged_frames([task["frame"] for task in active] + [None] * (size - len(active)),
                                   kind="query", device=session.device)
        for condition in conditions:
            state = _query_state([task["states"][condition] for task in active], size,
                                 row_capacity=session.layout.row_capacity, width=session.layout.row_width,
                                 device=session.device)
            versions = tuple(value._version for value in (state.rows, state.values, state.counts))
            description = {"condition": condition, "shape": list(mb["batch"]["input_ids"].shape),
                           "native_no_grad": True, "final_query_only": True}
            _require(all(value == description for value in gather(description)),
                     "EP ranks disagree on supplemental final-query dispatch")
            cost = {"phase": "supplemental_final_query", "kind": condition,
                    "ce_targets": mb["loss_target_tokens"], "real_input_tokens": mb["real_input_tokens"],
                    "padded_input_tokens": mb["padded_input_tokens"], "batch_size": size}
            # A callback rejection occurs before all collective-bearing CE work.
            on_score(cost)
            batch = mb["batch"]
            with preserve_encoding_rng(session.device):
                losses, capture = session.adapter.run_native(
                    lambda: session.models[0](**batch), input_ids=batch["input_ids"],
                    valid_mask=mb["valid_mask"], owner_ids=mb["owner_ids"], mode="query", state=state)
            labels, valid = batch["labels"], batch["labels"].ne(-100)
            _require(isinstance(losses, torch.Tensor) and losses.shape == labels.shape
                     and not losses.requires_grad and bool(losses[valid].isfinite().all()),
                     "final query requires finite native CE without a backward graph")
            # Accumulate the native per-token CE in FP64; small paired deltas
            # must not be rounded away by a second FP32 reduction.
            lane_sums = torch.where(valid, losses.double(), torch.zeros_like(losses, dtype=torch.float64)).sum(1)
            lane_counts = valid.sum(1).cpu().tolist()
            _require(tuple(lane_counts) == mb["loss_target_tokens_by_lane"], "final-query target count changed")
            _require(versions == tuple(value._version for value in (state.rows, state.values, state.counts)),
                     "native final-query scoring mutated memory")
            for lane, task in enumerate(active):
                total, count = float(lane_sums[lane]), lane_counts[lane]
                records.append({"trajectory_index": task["trajectory_index"], "condition": condition,
                                "loss_sum": total, "target_tokens": count, "nll": total / count,
                                "frame_index": len(task["plan"]) - 1,
                                "frame_sha256": task["frame"].get("frame_sha256")})
            calls += 1
            charged += cost["ce_targets"]
            real += cost["real_input_tokens"]
            padded += cost["padded_input_tokens"]
            del losses, capture, lane_sums, state
    gathered = gather({"records": records, "calls": calls, "ce_targets": charged,
                       "real_input_tokens": real, "padded_input_tokens": padded})
    _require(all(value["calls"] == calls for value in gathered), "EP query call counts differ")
    records = [row for value in gathered for row in value["records"]]
    grouped = {}
    for row in records:
        grouped.setdefault(row["trajectory_index"], {})[row["condition"]] = row
    pairs = []
    for index, rows in sorted(grouped.items()):
        _require(set(rows) == set(conditions) and len({row["target_tokens"] for row in rows.values()}) == 1,
                 "paired final-query coverage differs")
        pairs.append({"trajectory_index": index, "target_tokens": rows["memory_ref"]["target_tokens"],
                      **{condition + "_nll": rows[condition]["nll"] for condition in conditions},
                      "memory_perturbed_minus_reference_nll": rows["memory_eps01"]["nll"] - rows["memory_ref"]["nll"],
                      "base_perturbed_minus_reference_nll": rows["base_eps01"]["nll"] - rows["base_ref"]["nll"]})
    summary = {}
    for kind in ("memory", "base"):
        key = kind + "_perturbed_minus_reference_nll"
        denominator = sum(row["target_tokens"] for row in pairs)
        summary[key] = {"trajectory_mean": math.fsum(row[key] for row in pairs) / len(pairs) if pairs else None,
                        "token_weighted_mean": math.fsum(row[key] * row["target_tokens"] for row in pairs) / denominator if denominator else None,
                        "maximum_absolute": max((abs(row[key]) for row in pairs), default=None)}
    return {"protocol": "qwen38_supplemental_final_query_scoring_v1", "conditions": list(conditions),
            "records": records, "paired_records": pairs, "summary": summary,
            "ce_targets": sum(value["ce_targets"] for value in gathered),
            "real_input_tokens": sum(value["real_input_tokens"] for value in gathered),
            "padded_input_tokens": sum(value["padded_input_tokens"] for value in gathered),
            "collective_calls": calls, "backward_calls": 0, "optimizer_steps": 0,
            "interpretation": "same final held-out query and state-specific paired NLL; positive delta is worse"}


def _feature_contrast(chunks, *, absolute_floor):
    """Measure actual feature vectors without expanding all sixteen heads."""
    parts = {key: [] for key in ("h12", "small", "mem", "noise", "signal", "residual",
                               "noise_signal_dot", "noise_residual_dot")}
    count = 0
    for h12, small, mem in chunks:
        h12, small, mem = (value.detach().double() for value in (h12, small, mem))
        _require(h12.shape == small.shape == mem.shape
                 and all(bool(value.isfinite().all()) for value in (h12, small, mem)),
                 "feature contrast requires aligned finite tensors")
        noise, signal, residual = small - h12, mem - small, mem - h12
        for key, value in zip(("h12", "small", "mem", "noise", "signal", "residual"),
                              (h12, small, mem, noise, signal, residual)):
            parts[key].append(float(value.square().sum()))
        parts["noise_signal_dot"].append(float((noise * signal).sum()))
        parts["noise_residual_dot"].append(float((noise * residual).sum()))
        count += h12.numel()
    totals = {key: math.fsum(values) for key, values in parts.items()}
    norms = {key: math.sqrt(totals[key]) for key in ("h12", "small", "mem", "noise", "signal", "residual")}

    def ratio(numerator, denominator):
        return norms[numerator] / norms[denominator] if norms[denominator] > absolute_floor else None

    def cosine(dot, first, second):
        if min(norms[first], norms[second]) <= absolute_floor:
            return None
        return max(-1., min(1., totals[dot] / (norms[first] * norms[second])))

    return {
        "elements": count, "absolute_l2_floor": absolute_floor,
        "norm_l2": {"h12_base": norms["h12"], "matched_small_base": norms["small"], "memory": norms["mem"],
                    "batch_noise": norms["noise"], "matched_memory_delta": norms["signal"],
                    "actual_residual_delta": norms["residual"]},
        "relative_l2": {"h12_vs_matched_small_base": ratio("noise", "small"),
                        "memory_vs_matched_small_base": ratio("signal", "small"),
                        "memory_vs_h12_base": ratio("residual", "h12")},
        "noise_to_matched_memory_delta_l2": ratio("noise", "signal"),
        "noise_to_actual_residual_delta_l2": ratio("noise", "residual"),
        "matched_memory_delta_to_actual_residual_delta_l2": ratio("signal", "residual"),
        "noise_matched_memory_delta_cosine": cosine("noise_signal_dot", "noise", "signal"),
        "noise_actual_residual_delta_cosine": cosine("noise_residual_dot", "noise", "residual"),
        "denominator_filtered": {key: value <= absolute_floor for key, value in norms.items()},
    }


@torch.no_grad()
def capture_batch_feature_metrics(writer, entry, *, absolute_floor=1e-12, event_chunk_size=128):
    """Compare H12 base, matched-small base and memory at one pre-write state.

    The writer-feature comparison uses the actual pooling, right context, head
    embedding and event multiplicities. This is the vector supplied to the
    residual head before its first affine map, including its unchanged head
    coordinates. No native collective, state write or parameter update occurs.
    """
    _require(math.isfinite(absolute_floor) and absolute_floor > 0
             and type(event_chunk_size) is int and event_chunk_size > 0, "invalid feature diagnostic limits")
    base_writer = getattr(writer, "base_writer", writer)
    device = base_writer.pool_score.weight.device
    blocks = [entry[key] for key in ("H12_block", "base_block", "mem_block")]
    rows = blocks[0]["physical_rows_new"].to(device)
    hidden = [block["H_new"].to(device) for block in blocks]
    _require(all(value.shape == hidden[0].shape for value in hidden)
             and all(torch.equal(block["physical_rows_new"].to(device), rows) for block in blocks)
             and hidden[0].ndim == 3 and hidden[0].shape[0] == 1,
             "H12, matched-small and memory features must share token framing and row addresses")
    mask = torch.ones(hidden[0].shape[:2], dtype=torch.bool, device=device)
    normalized = [writer_evidence_features(value, mode="rmsnorm", eps=1e-6) for value in hidden]
    features = [base_writer.event_features(value, rows, mask) for value in normalized]
    _require(all(torch.equal(value.indices, features[0].indices)
                 and torch.equal(value.rows, features[0].rows) for value in features),
             "writer feature event identities differ")

    def feature_chunks():
        for start in range(0, features[0].rows.numel(), event_chunk_size):
            section = slice(start, start + event_chunk_size)
            batch, token, _ = features[0].indices[section].unbind(dim=1)
            yield tuple(value.dense[section] if value.dense is not None else torch.cat(
                (value.evidence[batch, token], value.head_features[section]), dim=-1) for value in features)

    state = detached_encoding_snapshot(entry["state_before"])
    state_norm = float(state.values.double().norm())
    result = {
        "protocol": "qwen38_supplemental_batch_feature_contrast_v1",
        "trajectory_index": entry["trajectory_index"], "frame_index": entry["frame_index"],
        "state_empty": state.rows.numel() == 0, "state_rows": state.rows.numel(),
        "state_effective_value_l2": state_norm, "state_effective_values_all_zero": state_norm == 0.,
        "new_tokens": hidden[0].shape[1], "writer_events": features[0].rows.numel(),
        "h12_batch_size": blocks[0].get("encoding_batch_size"), "h12_lane": blocks[0].get("encoding_lane"),
        "matched_batch_size": entry.get("batch_size"), "matched_lane": entry.get("lane"),
        "raw_hidden": _feature_contrast([hidden], absolute_floor=absolute_floor),
        "rmsnorm_hidden": _feature_contrast([normalized], absolute_floor=absolute_floor),
        "writer_features": _feature_contrast(feature_chunks(), absolute_floor=absolute_floor),
        "contrast_definitions": {"batch_noise": "matched_small_base - h12_base",
            "matched_memory_delta": "memory - matched_small_base", "actual_residual_delta": "memory - h12_base"},
        "interpretation": "observed feature differences; matched memory delta also includes any between-call execution noise",
    }
    if "frame" in entry:
        result.update(article_id=entry["frame"].get("article_id"), source=entry["frame"].get("source"))
    return result


def compare_probe_encodings(entry, writer):
    """Compatibility adapter for the originally proposed callback ordering."""
    return capture_batch_feature_metrics(writer, entry)

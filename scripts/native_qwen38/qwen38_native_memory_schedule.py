"""Causal memory-aware encoding; legacy staged execution remains unchanged.

Only a ready block from each trajectory enters an encoding wave. Native reads
use independent detached snapshots, then the caller writes with its original
state graph. Query backward and joint writer VJP keep the existing stages 3/4.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import random
import time

from qwen38_native_ep4_schedule import _runtime
from qwen38_native_runtime_preflight import language_model
from qwen38_native_staged_schedule import _EvidenceHeadReached, _evidence_head_identity


PROTOCOL = "qwen38_native_causal_memory_encoding_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def preserve_encoding_rng(device):
    """Extra feature forwards cannot advance Python, NumPy, Torch or TP RNG."""
    import torch
    import numpy as np
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    tracker = None
    tracker_state = None
    if device.type == "cuda":
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        tracker = get_cuda_rng_tracker()
        tracker_state = copy.deepcopy(tracker.get_states())
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if tracker is not None:
            tracker.set_states(tracker_state)


def _validate_encoding(session, microbatches, *, encoding):
    from delta_ngram.batching import BatchMemoryState
    torch, dist = _runtime(session)
    _require(encoding in ("base", "memory"), "encoding must be base or memory")
    _require(isinstance(microbatches, (list, tuple)) and bool(microbatches),
             "memory encoding needs at least one microbatch")
    shapes = []
    for item in microbatches:
        batch = item["batch"]
        ids, valid, writes, owners = (batch["input_ids"], item["valid_mask"],
                                      item["write_mask"], item["owner_ids"])
        _require(item.get("encoding_protocol") == "qwen38_memory_encoding_frames_v1"
                 and item.get("encoding_kind") == encoding and item["kind"] == "evidence",
                 "use the explicit memory-encoding frame boundary")
        _require(ids.dtype == torch.long and ids.ndim == 2 and ids.shape[0] > 0
                 and ids.shape[1] == 640 and ids.device == session.device,
                 "encoding requires int64 [B,640] native frames on the model device")
        size = ids.shape[0]
        _require(valid.dtype == writes.dtype == torch.bool
                 and valid.shape == writes.shape == ids.shape
                 and valid.device == writes.device == ids.device
                 and torch.equal(valid, item["encoder_valid_mask"])
                 and not bool((writes & ~valid).any()),
                 "encoding reads all real prefix tokens and only writes new tokens")
        _require(owners.dtype == torch.long and owners.device == ids.device
                 and torch.equal(owners, torch.arange(size, device=ids.device)),
                 "each encoding lane must have its own local owner namespace")
        _require(batch.get("labels") is None and item["loss_target_tokens"] == 0
                 and batch.get("attention_mask") is None
                 and batch["position_ids"].dtype == torch.long
                 and batch["position_ids"].shape == ids.shape
                 and batch["position_ids"].device == ids.device,
                 "encoding has no CE labels and preserves native causal positions")
        state = item.get("reader_state")
        if encoding == "memory":
            _require(isinstance(state, BatchMemoryState) and state.batch_size == size
                     and state.row_capacity == session.layout.row_capacity
                     and state.width == session.layout.row_width and state.precision is None
                     and state.values.device == ids.device and state.values.dtype == torch.float32
                     and not state.values.requires_grad and state.values.grad_fn is None
                     and not state.counts.requires_grad and state.counts.grad_fn is None,
                     "memory encoding requires an independent detached FP32 read snapshot")
        else:
            _require(state is None, "base encoding cannot receive memory state")
        shapes.append([size, 640])
    description = {"protocol": PROTOCOL, "encoding": encoding, "shapes": shapes, "CE_targets": 0}
    copies = [None] * dist.get_world_size()
    dist.all_gather_object(copies, description)
    _require(all(value == description for value in copies),
             "EP ranks disagree on collective-bearing memory encoding actions")
    return torch, shapes


def _collect_one_encoding(session, item, *, encoding, torch, model):
    captured = []
    batch_size = item["batch"]["input_ids"].shape[0]

    def capture_hidden(module, args, kwargs):
        value = args[0] if args else kwargs.get("hidden_states", kwargs.get("input_"))
        _require(isinstance(value, torch.Tensor)
                 and tuple(value.shape) == (640, batch_size, 2560)
                 and value.dtype == torch.bfloat16 and not captured,
                 "memory encoding expects native pre-LM-head [640,B,2560] BF16 H")
        captured.append(value.detach().clone())
        raise _EvidenceHeadReached()

    def native_until_head():
        try:
            session.models[0](**item["batch"])
        except _EvidenceHeadReached:
            return None
        raise RuntimeError("memory encoding failed to reach the pre-LM-head boundary")

    hook = model.output_layer.register_forward_pre_hook(capture_hidden, with_kwargs=True)
    try:
        with preserve_encoding_rng(session.device), torch.no_grad():
            _, capture = session.adapter.run_native(
                native_until_head, input_ids=item["batch"]["input_ids"],
                valid_mask=item["valid_mask"], owner_ids=item["owner_ids"],
                mode="query" if encoding == "memory" else "evidence",
                state=item.get("reader_state"))
        _require(len(captured) == 1, "memory encoding did not capture native H exactly once")
        hidden = captured.pop().transpose(0, 1).contiguous().cpu()
        rows = capture.physical_rows.detach().clone().cpu()
        _require(tuple(rows.shape) == (batch_size, 640, 16), "encoding physical row layout differs")
        _require(bool(hidden[item["encoder_valid_mask"].cpu()].isfinite().all()),
                 "memory encoding produced nonfinite real-token H")
        return {"hidden": hidden, "physical_rows": rows, "capture": capture,
                "encoding_kind": encoding, "query_target_tokens": 0}
    finally:
        hook.remove()


def collect_memory_encoding_microbatches(session, microbatches, *, encoding="memory"):
    """Explicit no-CE, no-LM-head native encoding with detached read snapshots."""
    torch, shapes = _validate_encoding(session, microbatches, encoding=encoding)
    model = language_model(session.raw)
    identity = {**_evidence_head_identity(model), "protocol": PROTOCOL,
                "encoding_kind": encoding, "read_mask": "all_real_tokens_including_prefix",
                "write_mask": "new_tokens_only", "read_state": "independent_detached_copy",
                "rng": "restore_python_numpy_torch_cuda_tensor_parallel",
                "CE_targets": 0, "LM_head_executed": False}
    results = [_collect_one_encoding(session, item, encoding=encoding, torch=torch, model=model)
               for item in microbatches]
    return {"microbatches": results, "num_microbatches": len(results), "shapes": shapes,
            "sequence_length": 640, "encoding_kind": encoding, "base_only": encoding == "base",
            "query_target_tokens": 0, "backward": False, "encoding_execution": identity,
            "real_input_tokens": sum(item["real_input_tokens"] for item in microbatches),
            "padded_input_tokens": sum(item["padded_input_tokens"] for item in microbatches)}


def _empty_repeat_state(state):
    from delta_ngram.alpha_memory import LazyDecayState
    from delta_ngram.batching import BatchMemoryState
    _require(isinstance(state, LazyDecayState) and state.batch_size == 1
             and state.precision is None and state.rows.numel() == state.values.numel() == 0
             and not bool(state.block_clock.any()) and not bool(state.cumulative_log_alpha.any()),
             "training repeat diagnostics require an empty one-owner native state")
    return LazyDecayState.from_batch(BatchMemoryState.empty(
        1, state.width, state.row_capacity, device=state.values.device))


def _repeat_difference(reference, repeated):
    import torch
    from delta_ngram.qwen38_staged_batching import materialize_staged_snapshot
    first, second = materialize_staged_snapshot(reference), materialize_staged_snapshot(repeated)
    _require((first.batch_size, first.width, first.row_capacity)
             == (second.batch_size, second.width, second.row_capacity), "repeat state namespace differs")
    rows = torch.unique(torch.cat((first.rows, second.rows)), sorted=True)
    a, ac = first.memory.lookup(rows)
    b, bc = second.memory.lookup(rows)
    _require(torch.equal(ac, bc), "repeat write support or observation counts differ")
    _require(bool(a.isfinite().all()) and bool(b.isfinite().all()), "nonfinite reference/repeat state")
    a, b = a.double(), b.double()
    difference = a - b
    norm = float(torch.linalg.vector_norm(a))
    error = float(torch.linalg.vector_norm(difference))
    row_norm = torch.linalg.vector_norm(difference, dim=-1)
    return {"state_error_l2": error, "reference_state_l2": norm,
            "repeat_state_l2": float(torch.linalg.vector_norm(b)),
            "relative_state_error": error / norm if norm >= 1e-8 else None,
            "maximum_row_error_l2": float(row_norm.max()) if row_norm.numel() else 0.,
            "state_rows": rows.numel(), "absolute_denominator_floor": 1e-8,
            "counts_match": True}


def _repeat_history_summary(records):
    import torch
    def quantiles(values):
        if not values:
            return {"median": None, "p99": None, "maximum": None}
        tensor = torch.tensor(values, dtype=torch.float64)
        _require(bool(tensor.isfinite().all()), "nonfinite repeat trajectory metric")
        return {"median": float(tensor.quantile(.5)), "p99": float(tensor.quantile(.99)),
                "maximum": float(tensor.max())}
    grouped = {}
    for row in records:
        grouped.setdefault(row["trajectory_index"], []).append(row)
    final = []
    for index, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: row["write_index"])
        _require([row["write_index"] for row in rows] == list(range(len(rows))),
                 "repeat per-trajectory write records are incomplete or duplicated")
        tail = [row["relative_state_error"] for row in rows[-3:] if row["relative_state_error"] is not None]
        final.append({**rows[-1], "write_blocks": len(rows), "tail_window_writes": min(3, len(rows)),
            "tail_eligible_writes": len(tail), "tail_relative_error_mean": sum(tail) / len(tail) if tail else None,
            "tail_relative_error_min": min(tail) if tail else None, "tail_relative_error_max": max(tail) if tail else None,
            "tail_relative_error_range": max(tail) - min(tail) if tail else None,
            "last_minus_previous_relative_error": (rows[-1]["relative_state_error"] - rows[-2]["relative_state_error"]
                if len(rows) > 1 and rows[-1]["relative_state_error"] is not None
                and rows[-2]["relative_state_error"] is not None else None)})
    evolved = [row for row in final if row["write_blocks"] > 1]
    return {"final_records": final, "summary": {
        "trajectories": len(final), "multiwrite_trajectories": len(evolved),
        "single_write_trajectories": len(final) - len(evolved), "write_observations": len(records),
        "final_relative_state_error_multiwrite": quantiles([row["relative_state_error"] for row in evolved
                                                            if row["relative_state_error"] is not None]),
        "final_state_error_l2_multiwrite": quantiles([row["state_error_l2"] for row in evolved]),
        "first_write_state_error_l2": quantiles([row["state_error_l2"] for row in records if row["write_index"] == 0]),
        "tail_interpretation": "last up to three observed writes; no plateau, stationarity or contraction claim"}}


def _replay_training_repeat(session, local, reference_by_owner, initial_states, *,
                            gather, write_repeat, waves, batch_limit, on_encoding):
    """Replay after the training writer graph is built, without reading targets."""
    import torch
    from delta_ngram.qwen38_memory_batching import collate_memory_encoding_frames
    from delta_ngram.qwen38_train_batching import crop_evidence_lanes
    kind = "diagnostic_repeat_memory"
    counts, padded, calls, records = 0, 0, 0, []
    states = dict(initial_states)
    writes = dict.fromkeys(states, 0)
    guards = [(value, value._version) for snapshots in reference_by_owner.values()
              for state in snapshots.values() for value in (state.rows, state.values, state.counts,
                  state.block_clock, state.cumulative_log_alpha, state.row_last_written_block, state.row_log_alpha_at_write)]
    started = time.monotonic()
    # This whole additional phase preserves all training RNG streams, even if
    # a supplied diagnostic writer callback happens to consume random numbers.
    with preserve_encoding_rng(session.device), torch.no_grad():
        for ordinal in range(waves):
            ready = [(item, item["plan"][ordinal]["evidence"]) for item in local
                     if ordinal < len(item["plan"]) and item["plan"][ordinal]["evidence"] is not None]
            maximum = max(gather(len(ready)))
            for start in range(0, maximum, batch_limit):
                size = min(batch_limit, maximum - start)
                active = ready[start:start + size]
                frames = [frame for _, frame in active] + [None] * (size - len(active))
                mb = collate_memory_encoding_frames(frames,
                    [states[id(item)] for item, _ in active] + [None] * (size - len(active)),
                    row_capacity=session.layout.row_capacity, width=session.layout.row_width, device=session.device)
                cost = {"kind": kind, "real_input_tokens": mb["real_input_tokens"],
                        "padded_input_tokens": mb["padded_input_tokens"], "ce_targets": 0,
                        "wave": ordinal, "batch_size": size, "diagnostic_only": True}
                if on_encoding is not None:
                    on_encoding(cost)
                result = collect_memory_encoding_microbatches(session, [mb], encoding="memory")["microbatches"][0]
                blocks = crop_evidence_lanes(mb, result["hidden"], result["physical_rows"])
                counts += cost["real_input_tokens"]
                padded += cost["padded_input_tokens"]
                calls += 1
                for lane, (item, frame) in enumerate(active):
                    owner, block = id(item), blocks[lane]
                    _require(block is not None, "repeat diagnostic lost an active lane")
                    before = states[owner]
                    versions = tuple(value._version for value in (before.rows, before.values, before.counts))
                    repeated = write_repeat(before, block)
                    _require(versions == tuple(value._version for value in (before.rows, before.values, before.counts)),
                             "repeat writer callback mutated its input state")
                    _require(not repeated.values.requires_grad and repeated.values.grad_fn is None,
                             "repeat diagnostic writer acquired a backward graph")
                    states[owner] = repeated
                    record = _repeat_difference(reference_by_owner[owner][ordinal], repeated)
                    record.update(trajectory_index=item["index"], frame_index=ordinal, write_index=writes[owner],
                                  article_id=frame.get("article_id"), source=frame.get("source"))
                    records.append(record)
                    writes[owner] += 1
        for value, version in guards:
            _require(value._version == version, "repeat diagnostic mutated the training reference graph")
    all_records = [row for rank_records in gather(records) for row in rank_records]
    result = {"protocol": "qwen38_training_independent_repeat_monitor_v1",
        "reference": "actual training writer trajectory; original graph retained unchanged",
        "repeat": "independent no-grad replay from empty after the training histories are built",
        "records": all_records, **_repeat_history_summary(all_records),
        "state_perturbations_injected": 0, "CE_targets": 0, "backward_calls": 0, "optimizer_steps": 0,
        "threshold_stop": False, "state_error_definition": "entire union-row memory L2 / reference memory L2",
        "RNG_policy": "restore Python, NumPy, Torch CPU/CUDA and tensor-parallel RNG after repeat phase"}
    return result, {"input_tokens": counts, "padded_input_tokens": padded, "calls": calls}, time.monotonic() - started


def build_causal_writer_histories(session, local, *, gather, write, encoding_mode,
                                  batch_limit=12, on_encoding=None, diagnose_wave=None,
                                  difference_baseline="actual_base", diagnostic_repeat_write=None):
    """Build original state graphs in ready waves and return stage-3 query tasks.

    ``write(item, block)`` updates item['state'] outside no_grad. In full mode
    H_new is H_mem; residual mode also supplies aligned H_base_new. Counters
    are rank local; on_encoding is called before each collective-bearing pass.
    A matched_base residual also receives H_base_matched_new from the same
    batch/lane geometry as H_mem; the original H12 base remains H_base_new.
    Optional diagnose_wave(state_before, block_mem, block_base_matched, frame)
    runs per active lane after a matched-batch base forward, before writing.
    It must not invoke native collectives; its records are gathered at the end.
    Optional diagnostic_repeat_write(state, block) -> new_state enables a full
    writer's independent repeat monitor. It runs only after all training write
    graphs are built, under no_grad and restored RNG, and must have no training
    counters or optimizer side effects. It receives no query labels.
    """
    import torch
    from delta_ngram.qwen38_memory_batching import collate_memory_encoding_frames
    from delta_ngram.qwen38_train_batching import crop_evidence_lanes

    _require(encoding_mode in ("full", "residual"), "writer encoding mode must be full or residual")
    _require(difference_baseline in ("actual_base", "matched_base")
             and (encoding_mode == "residual" or difference_baseline == "actual_base"),
             "matched difference baseline requires residual encoding")
    matched_difference = difference_baseline == "matched_base"
    _require(type(batch_limit) is int and batch_limit > 0, "encoding batch limit must be positive")
    _require(len({id(item) for item in local}) == len(local), "one trajectory cannot occupy two ready lanes")
    _require(torch.is_grad_enabled(), "causal writer histories require grad-enabled original states")
    _require(diagnostic_repeat_write is None or (callable(diagnostic_repeat_write) and encoding_mode == "full"),
             "independent training repeat diagnostics support only a full writer callback")
    extra_base_kind = "base_matched" if matched_difference else "diagnostic_base"
    kinds = ("memory", "base", extra_base_kind) if matched_difference or diagnose_wave is not None else ("memory", "base")
    if diagnostic_repeat_write is not None:
        kinds += ("diagnostic_repeat_memory",)
    counts = dict.fromkeys(kinds, 0)
    padded = dict.fromkeys(kinds, 0)
    calls = dict.fromkeys(kinds, 0)
    diagnostics = []
    diagnostic_seconds = 0.
    queries_by_owner, base_cache = {id(item): [] for item in local}, {}
    total_writes = 0
    repeat_initial = ({id(item): _empty_repeat_state(item["state"]) for item in local}
                      if diagnostic_repeat_write is not None else {})
    repeat_references = {id(item): {} for item in local} if diagnostic_repeat_write is not None else {}
    if diagnostic_repeat_write is not None:
        indices = [item.get("index") for item in local]
        _require(all(type(index) is int for index in indices) and len(indices) == len(set(indices)),
                 "repeat diagnostics require unique integer trajectory indices")
    if encoding_mode == "residual":
        # Preserve the legacy full-group H12 base batching and cache order.
        # Base encodings are state independent; only H_mem must wait for S_t.
        base_tasks = [(id(item), ordinal, frame["evidence"])
                      for item in local for ordinal, frame in enumerate(item["plan"])
                      if frame["evidence"] is not None]
        maximum = max(gather(len(base_tasks)))
        for start in range(0, maximum, batch_limit):
            size = min(batch_limit, maximum - start)
            active = base_tasks[start:start + size]
            mb = collate_memory_encoding_frames(
                [task[2] for task in active] + [None] * (size - len(active)),
                row_capacity=session.layout.row_capacity,
                width=session.layout.row_width, device=session.device)
            cost = {"kind": "base", "real_input_tokens": mb["real_input_tokens"],
                    "padded_input_tokens": mb["padded_input_tokens"], "ce_targets": 0,
                    "wave": None, "batch_size": size}
            if on_encoding is not None:
                on_encoding(cost)
            result = collect_memory_encoding_microbatches(
                session, [mb], encoding="base")["microbatches"][0]
            blocks = crop_evidence_lanes(mb, result["hidden"], result["physical_rows"])
            for task, block in zip(active, blocks):
                _require(block is not None, "base encoding lane disappeared")
                base_cache[task[:2]] = block
            counts["base"] += cost["real_input_tokens"]
            padded["base"] += cost["padded_input_tokens"]
            calls["base"] += 1
    waves = max(gather(max((len(item["plan"]) for item in local), default=0)))
    for ordinal in range(waves):
        ready = []
        for item in local:
            if ordinal >= len(item["plan"]):
                continue
            frame = item["plan"][ordinal]
            if frame["query"] is not None:
                queries_by_owner[id(item)].append(
                    {"owner": item, "frame": frame["query"], "snapshot": item["state"]})
            if frame["evidence"] is not None:
                ready.append((item, frame["evidence"]))
        maximum = max(gather(len(ready)))
        for start in range(0, maximum, batch_limit):
            size = min(batch_limit, maximum - start)
            active = ready[start:start + size]
            frames = [frame for _, frame in active] + [None] * (size - len(active))
            states = [item["state"] for item, _ in active] + [None] * (size - len(active))
            mb = collate_memory_encoding_frames(
                frames, states, row_capacity=session.layout.row_capacity,
                width=session.layout.row_width, device=session.device)
            cost = {"kind": "memory", "real_input_tokens": mb["real_input_tokens"],
                    "padded_input_tokens": mb["padded_input_tokens"], "ce_targets": 0,
                    "wave": ordinal, "batch_size": size}
            if on_encoding is not None:
                on_encoding(cost)
            result = collect_memory_encoding_microbatches(
                session, [mb], encoding="memory")["microbatches"][0]
            blocks = crop_evidence_lanes(mb, result["hidden"], result["physical_rows"])
            read_rows = [result["physical_rows"][lane][mb["valid_mask"][lane].cpu()].clone()
                         for lane in range(len(active))]
            counts["memory"] += cost["real_input_tokens"]
            padded["memory"] += cost["padded_input_tokens"]
            calls["memory"] += 1
            matched_base = None
            if matched_difference or diagnose_wave is not None:
                diagnostic_started = time.monotonic()
                base_mb = collate_memory_encoding_frames(
                    frames, row_capacity=session.layout.row_capacity,
                    width=session.layout.row_width, device=session.device)
                cost = {"kind": extra_base_kind, "real_input_tokens": base_mb["real_input_tokens"],
                        "padded_input_tokens": base_mb["padded_input_tokens"], "ce_targets": 0,
                        "wave": ordinal, "batch_size": size}
                if on_encoding is not None:
                    on_encoding(cost)
                base_result = collect_memory_encoding_microbatches(
                    session, [base_mb], encoding="base")["microbatches"][0]
                matched_base = crop_evidence_lanes(
                    base_mb, base_result["hidden"], base_result["physical_rows"])
                for lane in range(len(active)):
                    matched_base[lane]["physical_rows_read"] = base_result["physical_rows"][lane][
                        base_mb["valid_mask"][lane].cpu()].detach().clone()
                counts[extra_base_kind] += cost["real_input_tokens"]
                padded[extra_base_kind] += cost["padded_input_tokens"]
                calls[extra_base_kind] += 1
                # The matched pass is a required training input. Diagnostics
                # reuse it, so its time must not be deducted from training.
                if not matched_difference:
                    diagnostic_seconds += time.monotonic() - diagnostic_started
            for lane, (item, _) in enumerate(active):
                block = blocks[lane]
                _require(block is not None, "ready memory-encoding lane disappeared")
                block["physical_rows_read"] = read_rows[lane]
                if encoding_mode == "residual":
                    base = base_cache.pop((id(item), ordinal))
                    _require(base is not None and torch.equal(
                        base["physical_rows_new"], block["physical_rows_new"]),
                        "base and memory encodings have different write addresses")
                    block["H_base_new"] = base["H_new"]
                if matched_difference:
                    baseline = matched_base[lane]
                    _require(baseline is not None and torch.equal(
                        baseline["physical_rows_new"], block["physical_rows_new"]),
                        "matched residual base and memory write addresses differ")
                    block["H_base_matched_new"] = baseline["H_new"]
                if diagnose_wave is not None:
                    diagnostic_started = time.monotonic()
                    baseline = matched_base[lane]
                    _require(baseline is not None and torch.equal(
                        baseline["physical_rows_new"], block["physical_rows_new"]),
                        "diagnostic base and memory write addresses differ")
                    record = diagnose_wave(item["state"], block, baseline, frames[lane])
                    if record is not None:
                        diagnostics.append(record)
                    diagnostic_seconds += time.monotonic() - diagnostic_started
                write(item, block)
                if diagnostic_repeat_write is not None:
                    repeat_references[id(item)][ordinal] = item["state"]
                total_writes += 1
        # No trajectory can advance until every write in the current wave has
        # completed. This includes ranks whose current local ready set is empty.
    repeat_diagnostics = None
    if diagnostic_repeat_write is not None:
        repeat_diagnostics, repeat_cost, repeat_seconds = _replay_training_repeat(
            session, local, repeat_references, repeat_initial, gather=gather,
            write_repeat=diagnostic_repeat_write, waves=waves, batch_limit=batch_limit, on_encoding=on_encoding)
        counts["diagnostic_repeat_memory"] = repeat_cost["input_tokens"]
        padded["diagnostic_repeat_memory"] = repeat_cost["padded_input_tokens"]
        calls["diagnostic_repeat_memory"] = repeat_cost["calls"]
        diagnostic_seconds += repeat_seconds
    for item in local:
        item["state"] = None
    # Keep the legacy trajectory-major stage-3 query batching, although writes
    # were interleaved by causal ready wave rather than by trajectory.
    queries = [task for item in local for task in queries_by_owner[id(item)]]
    _require(not base_cache, "precomputed base evidence was not consumed")
    if diagnose_wave is not None:
        diagnostics = [record for rank_records in gather(diagnostics) for record in rank_records]
    return {"queries": queries, "ready_waves": waves, "write_blocks": total_writes,
            "encoding_input_tokens": counts, "encoding_padded_input_tokens": padded,
            "encoding_calls": calls, "encoding_ce_targets": 0, "protocol": PROTOCOL,
            "diagnostics": diagnostics, "diagnostic_seconds": diagnostic_seconds,
            "difference_baseline": difference_baseline, "repeat_diagnostics": repeat_diagnostics}

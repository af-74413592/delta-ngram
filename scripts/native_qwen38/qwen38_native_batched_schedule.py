"""Native TP2/PP2 schedules for several B2/B4 microbatches in one wave.

The caller packs independent trajectory owners and immutable reader leaves.
This module neither writes memory nor changes any optimizer or trajectory
denominator. It calls the real non-interleaved Megatron pipeline schedule once
per wave, so num_microbatches > 1 can use PP warmup/1F1B/cooldown.

Each input is the collator dictionary: ``batch``, ``valid_mask``, ``owner_ids``,
and (for queries on PP0) ``reader_state``. All microbatches in a wave have the
same padded [B,S] shape. Entirely unscored query microbatches must be omitted;
individual inactive lanes are allowed and contribute no targets or gradient.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from qwen38_native_runtime_preflight import language_model


# Actual installed MCore 0.18.0 source preserved by native_runtime_preflight_01.
# Its legacy two-item loss callback multiplies by CP and divides by M.
MCORE_SCHEDULE_SHA256 = "946a71d2fd69dfebaceab461bd8e01a37cc4797221651d64943c2c5c9c4e6f0a"
SCHEDULE_NAME = "forward_backward_pipelining_without_interleaving"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _native_runtime(session):
    """Load the already initialized runtime; never initialize CUDA or models."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.pipeline_parallel import get_forward_backward_func

    _require(dist.is_initialized() and len(session.models) == 1,
             "batched native schedules need one initialized model chunk")
    topology = (
        ps.get_tensor_model_parallel_world_size(),
        ps.get_pipeline_model_parallel_world_size(),
        ps.get_context_parallel_world_size(),
        ps.get_data_parallel_world_size(),
    )
    _require(topology == (2, 2, 1, 1), "batched native schedules require TP2 PP2 CP1 DP1")
    schedule = get_forward_backward_func()
    _require(schedule.__name__ == SCHEDULE_NAME, "unexpected native pipeline schedule")
    source = Path(inspect.getfile(schedule))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    _require(source_sha == MCORE_SCHEDULE_SHA256,
             "native MCore schedule differs from the source with audited microbatch scaling")
    config = session.models[0].config
    _require(config.calculate_per_token_loss is False
             and config.grad_scale_func is None
             and config.finalize_model_grads_func is None
             and config.recompute_granularity is None
             and config.deallocate_pipeline_outputs is False,
             "unsupported hidden loss scaling, gradient finalization or activation recomputation")
    _require(not session.models[0].training, "native reader must remain in evaluation mode")
    return torch, dist, ps, schedule


def _validate_microbatches(microbatches, *, query, group_total_target_tokens=None):
    """Validate frame layout on CPU or CUDA, without executing a model."""
    import torch

    _require(isinstance(microbatches, (list, tuple)) and bool(microbatches),
             "a native wave needs at least one microbatch")
    if query:
        _require(type(group_total_target_tokens) is int and group_total_target_tokens > 0,
                 "provide the whole optimizer group's actual target-token denominator")
    shape = device = None
    descriptions = []
    total_targets = 0
    for index, item in enumerate(microbatches):
        batch = item["batch"]
        ids, valid, owners = batch["input_ids"], item["valid_mask"], item["owner_ids"]
        _require(isinstance(ids, torch.Tensor) and ids.dtype == torch.long and ids.ndim == 2,
                 "native input_ids must be int64 [B,S]")
        current_shape = tuple(ids.shape)
        _require(current_shape[0] in (1, 2, 4) and 2 <= current_shape[1] <= 640
                 and current_shape[1] % 2 == 0,
                 "native waves support B1/B2/B4 and an SP-even sequence length <=640")
        if shape is None:
            shape, device = current_shape, ids.device
        _require(current_shape == shape and ids.device == device,
                 "all microbatches in a wave must share padded shape and device")
        _require(valid.dtype == torch.bool and tuple(valid.shape) == shape and valid.device == device,
                 "valid_mask must be boolean [B,S] on the token device")
        _require(owners.dtype == torch.long and tuple(owners.shape) == (shape[0],)
                 and owners.device == device and bool(owners.ge(0).all()),
                 "owner_ids must be nonnegative int64 [B]")
        positions = batch["position_ids"]
        _require(positions.dtype == torch.long and tuple(positions.shape) == shape
                 and positions.device == device and batch.get("attention_mask") is None,
                 "native waves require explicit per-lane position_ids and causal attention_mask=None")
        if query:
            labels = batch["labels"]
            _require(labels.dtype == torch.long and tuple(labels.shape) == shape
                     and labels.device == device,
                     "native labels must be already shifted int64 [B,S]")
            selected = labels.ne(-100)
            _require(not bool((selected & ~valid).any()),
                     "a query target cannot use an invalid or padded predictor position")
            counts = selected.sum(1).tolist()
            count = sum(counts)
            _require(count > 0, "omit query microbatches with no supervised target tokens")
            if "loss_target_tokens" in item:
                _require(item["loss_target_tokens"] == count,
                         "collated target count differs from the native labels")
            total_targets += count
        else:
            _require(batch.get("labels") is None, "base-only evidence must not contain loss labels")
            counts = [0] * shape[0]
            _require(bool(valid.any()), "omit completely inactive evidence microbatches")
        frames = item.get("frames")
        descriptions.append({
            "index": index, "shape": list(shape), "owners": owners.tolist(),
            "target_tokens_by_lane": counts,
            "frame_sha256_by_lane": ([None if frame is None else frame["frame_sha256"] for frame in frames]
                                    if frames is not None else None),
        })
    if query:
        _require(total_targets <= group_total_target_tokens,
                 "wave targets exceed the whole optimizer-group denominator")
    return shape, device, descriptions


def _agree_wave(dist, description):
    copies = [None] * dist.get_world_size()
    dist.all_gather_object(copies, description)
    _require(all(value == copies[0] for value in copies),
             "model ranks disagree on microbatch wave shape, order, owners or targets")


def _query_loss_callback(labels, *, index, num_microbatches, group_total_target_tokens):
    """Bind one microbatch; compensate the pinned schedule's division by M."""
    import torch

    mask = labels.ne(-100)
    expected_counts = mask.sum(1)

    def loss_func(losses):
        _require(isinstance(losses, torch.Tensor) and losses.shape == labels.shape,
                 "native per-token CE must have the same [B,S] layout as labels")
        losses = losses.float()
        selected = losses[mask]
        _require(selected.numel() == int(expected_counts.sum()) and bool(selected.isfinite().all()),
                 "nonfinite loss or incorrect native target count")
        total = selected.sum()
        lane_sums = torch.where(mask, losses, torch.zeros_like(losses)).sum(1)
        # MCore's two-item callback branch applies CP/M. CP is fixed to one.
        # Its returned gradient is therefore d(sum_selected / group_targets).
        objective = (total / group_total_target_tokens) * num_microbatches
        return objective, {
            "microbatch_index": index, "loss_sum": total.detach(),
            "loss_sum_by_lane": lane_sums.detach(),
            "loss_target_tokens_by_lane": expected_counts.detach(),
        }
    return loss_func


def schedule_query_microbatches(session, microbatches, *, group_total_target_tokens, backward=True):
    """Run one real pipelined query wave, leaving each input reader leaf's grad.

    The bridge must later record every trajectory lane with reader_loss_divisor
    equal to group_total_target_tokens. The return exposes each lane's actual
    CE sum/count; no batch mean, TP sum or microbatch mean is left in the loss.
    All query forwards retain grad mode, including backward=False observations.
    """
    import torch

    _require(type(backward) is bool, "backward must be an explicit boolean")
    shape, device, descriptions = _validate_microbatches(
        microbatches, query=True, group_total_target_tokens=group_total_target_tokens)
    torch, dist, ps, schedule = _native_runtime(session)
    _agree_wave(dist, {"kind": "query", "microbatches": descriptions,
                       "group_total_target_tokens": group_total_target_tokens,
                       "backward": backward})
    count = len(microbatches)
    captures = [None] * count
    seen = []
    if session.adapter is not None:
        for item in microbatches:
            state = item.get("reader_state")
            _require(state is not None and state.values.numel() > 0
                     and (not backward or state.values.requires_grad),
                     "each query microbatch needs its own nonempty reader leaf on PP0")
        leaves = [item["reader_state"].values for item in microbatches]
        _require(len({id(value) for value in leaves}) == count,
                 "query microbatches must not reuse the same reader leaf")

    def forward_step(data_iterator, model):
        index, item = next(data_iterator)
        _require(index == len(seen), "native schedule reordered or repeated a query forward")
        seen.append(index)
        batch = item["batch"]
        current = dict(batch)
        if not ps.is_pipeline_last_stage():
            current["labels"] = None
        if session.adapter is None:
            output = model(**current)
        else:
            output, captures[index] = session.adapter.run_native(
                lambda: model(**current), input_ids=batch["input_ids"],
                valid_mask=item["valid_mask"], owner_ids=item["owner_ids"],
                mode="query", state=item["reader_state"])
        return output, _query_loss_callback(
            batch["labels"], index=index, num_microbatches=count,
            group_total_target_tokens=group_total_target_tokens)

    with torch.enable_grad():
        metrics = schedule(
            forward_step_func=forward_step, data_iterator=iter(enumerate(microbatches)),
            model=session.models, num_microbatches=count, seq_length=shape[1],
            micro_batch_size=shape[0], forward_only=not backward)
    _require(seen == list(range(count)), "native schedule did not execute every query microbatch")
    # Preserve FP32 loss values exactly while transporting integer counts in a
    # small FP64 matrix. No activations or logits enter these result collectives.
    statistics = torch.zeros((count, 1 + 2 * shape[0]), dtype=torch.float64, device=device)
    if ps.is_pipeline_last_stage():
        _require(len(metrics) == count, "last pipeline stage returned the wrong metric count")
        for index, item in enumerate(metrics):
            _require(item["microbatch_index"] == index, "native query metrics are out of order")
            statistics[index, 0] = item["loss_sum"]
            statistics[index, 1:1 + shape[0]] = item["loss_sum_by_lane"]
            statistics[index, 1 + shape[0]:] = item["loss_target_tokens_by_lane"]
    dist.broadcast(statistics, src=ps.get_pipeline_model_parallel_last_rank(),
                   group=ps.get_pipeline_model_parallel_group())
    transported = statistics.cpu().tolist()
    results = []
    for index, values in enumerate(transported):
        lane_counts = [int(value) for value in values[1 + shape[0]:]]
        _require(lane_counts == descriptions[index]["target_tokens_by_lane"],
                 "pipeline transport changed lane target counts")
        results.append({
            "microbatch_index": index, "loss_sum": values[0],
            "loss_sum_by_lane": values[1:1 + shape[0]],
            "loss_target_tokens_by_lane": lane_counts,
            "query_target_tokens": sum(lane_counts), "capture": captures[index],
            "reader_loss_divisor": group_total_target_tokens,
            "normalized_loss": values[0] / group_total_target_tokens,
        })
    return {
        "microbatches": results, "num_microbatches": count, "micro_batch_size": shape[0],
        "sequence_length": shape[1], "schedule": SCHEDULE_NAME,
        "schedule_source_sha256": MCORE_SCHEDULE_SHA256,
        "loss_sum": sum(item["loss_sum"] for item in results),
        "query_target_tokens": sum(item["query_target_tokens"] for item in results),
        "group_total_target_tokens": group_total_target_tokens,
        "reader_loss_divisor": group_total_target_tokens,
        "schedule_microbatch_divisor_compensation": count,
        "query_grad_enabled": True, "backward": backward,
    }


def _head_input(args, kwargs, *, shape):
    import torch

    value = args[0] if args else kwargs.get("hidden_states", kwargs.get("input_"))
    _require(isinstance(value, torch.Tensor) and tuple(value.shape) == (shape[1] // 2, shape[0], 2560)
             and value.dtype == torch.bfloat16,
             "unexpected SP-sharded native final-H shape or dtype")
    return value.detach().contiguous().clone()


def collect_evidence_microbatches(session, microbatches):
    """Run one M-microbatch base-only forward wave and return full H on PP0.

    Each PP0 result has CPU ``hidden[B,S,2560]`` and
    ``physical_rows[B,S,16]``. Crop real new-token positions with the original
    per-lane frame metadata before writing. PP1 results are None. The native
    LM head still executes; its logits are immediately discarded by the
    zero-target callback rather than retained in the schedule's data store.
    No loss labels, CE targets, writer memory or backward are used.
    """
    import torch

    shape, device, descriptions = _validate_microbatches(microbatches, query=False)
    torch, dist, ps, schedule = _native_runtime(session)
    _agree_wave(dist, {"kind": "evidence", "microbatches": descriptions})
    count = len(microbatches)
    captures, local_hidden = [None] * count, [None] * count
    current_index = None
    seen = []
    handle = None
    if ps.is_pipeline_last_stage():
        def capture(module, args, kwargs):
            _require(current_index is not None and local_hidden[current_index] is None,
                     "native evidence head was unbound or repeated")
            local_hidden[current_index] = _head_input(args, kwargs, shape=shape)
        handle = language_model(session.raw).output_layer.register_forward_pre_hook(capture, with_kwargs=True)

    def forward_step(data_iterator, model):
        nonlocal current_index
        index, item = next(data_iterator)
        _require(index == len(seen), "native schedule reordered or repeated an evidence forward")
        seen.append(index)
        current_index = index
        current = dict(item["batch"])
        current["labels"] = None
        try:
            if session.adapter is None:
                output = model(**current)
            else:
                output, captures[index] = session.adapter.run_native(
                    lambda: model(**current), input_ids=current["input_ids"],
                    valid_mask=item["valid_mask"], owner_ids=item["owner_ids"],
                    mode="evidence", state=None)
        finally:
            current_index = None

        def no_loss(output_tensor):
            tensor = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            _require(isinstance(tensor, torch.Tensor), "native evidence output must contain a tensor")
            return tensor.new_zeros((), dtype=torch.float32), {"microbatch_index": index, "target_tokens": 0}
        return output, no_loss

    try:
        with torch.no_grad():
            metrics = schedule(
                forward_step_func=forward_step, data_iterator=iter(enumerate(microbatches)),
                model=session.models, num_microbatches=count, seq_length=shape[1],
                micro_batch_size=shape[0], forward_only=True)
    finally:
        if handle is not None:
            handle.remove()
    _require(seen == list(range(count)), "native schedule did not execute every evidence microbatch")
    if ps.is_pipeline_last_stage():
        _require(len(metrics) == count and all(
            item == {"microbatch_index": index, "target_tokens": 0} for index, item in enumerate(metrics)),
            "native evidence unexpectedly produced targets or reordered results")
    results = []
    with torch.no_grad():
        for index in range(count):
            full = torch.empty((shape[1], shape[0], 2560), dtype=torch.bfloat16, device=device)
            if ps.is_pipeline_last_stage():
                hidden = local_hidden[index]
                _require(hidden is not None, "native last-H capture is missing")
                parts = [torch.empty_like(hidden) for _ in range(2)]
                dist.all_gather(parts, hidden, group=ps.get_tensor_model_parallel_group())
                full.copy_(torch.cat(parts, dim=0))
                local_hidden[index] = None
            dist.broadcast(full, src=ps.get_pipeline_model_parallel_last_rank(),
                           group=ps.get_pipeline_model_parallel_group())
            if session.adapter is not None:
                captured = captures[index]
                _require(captured is not None, "native evidence address capture is missing")
                results.append({
                    "microbatch_index": index,
                    "hidden": full.transpose(0, 1).contiguous().cpu(),
                    "physical_rows": captured.physical_rows.cpu(),
                    "capture": captured, "loss_target_tokens": 0,
                })
            else:
                results.append(None)
    return {
        "microbatches": results, "num_microbatches": count, "micro_batch_size": shape[0],
        "sequence_length": shape[1], "schedule": SCHEDULE_NAME,
        "schedule_source_sha256": MCORE_SCHEDULE_SHA256,
        "loss_target_tokens": 0, "backward": False, "base_only": True,
        "evidence_input_tokens": count * shape[0] * shape[1],
    }

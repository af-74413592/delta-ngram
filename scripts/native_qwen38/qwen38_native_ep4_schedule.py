"""Direct native TP1/PP1/EP4 forwards and backwards for the bounded trial.

Every rank calls each globally active wave, even with three inactive local
lanes. Expert all-to-all still occurs inside the unmodified native model.
No pipeline schedule, TP gather, TP gradient scaling, or DP mean is applied.
"""
from __future__ import annotations

from qwen38_native_runtime_preflight import language_model


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _runtime(session):
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    _require(dist.is_initialized() and len(session.models) == 1
             and session.adapter is not None and session.layout is not None,
             "EP4 direct calls require one complete frozen model and local adapter")
    _require((ps.get_tensor_model_parallel_world_size(),
              ps.get_pipeline_model_parallel_world_size(),
              ps.get_expert_model_parallel_world_size(),
              ps.get_expert_tensor_parallel_world_size(),
              ps.get_data_parallel_world_size(),
              ps.get_context_parallel_world_size()) == (1, 1, 4, 1, 4, 1),
             "EP4 trial requires TP1 PP1 EP4 ETP1 DP4 CP1")
    config = session.models[0].config
    _require(not session.models[0].training and not config.sequence_parallel
             and config.recompute_granularity is None
             and config.grad_scale_func is None
             and config.finalize_model_grads_func is None,
             "EP4 trial requires a frozen reader without SP, recompute or gradient scaling")
    return torch, dist


def _validate(session, microbatches, *, query, denominator=None, backward=False):
    torch, dist = _runtime(session)
    _require(isinstance(microbatches, (list, tuple)) and len(microbatches) == 1,
             "minimal EP4 trial uses exactly one B3 microbatch per local wave")
    if query:
        _require(type(denominator) is int and denominator > 0,
                 "supply the actual global optimizer-group target denominator")
    item = microbatches[0]
    batch = item["batch"]
    ids, valid, owners = batch["input_ids"], item["valid_mask"], item["owner_ids"]
    _require(ids.dtype == torch.long and tuple(ids.shape) == (3, 640)
             and ids.device == session.device,
             "EP4 local input must be int64 [3,640] on the model device")
    _require(valid.dtype == torch.bool and valid.shape == ids.shape
             and valid.device == ids.device
             and owners.dtype == torch.long and tuple(owners.shape) == (3,)
             and owners.device == ids.device
             and owners.tolist() == [0, 1, 2],
             "EP4 lanes require independent local owners and aligned valid masks")
    _require(batch["position_ids"].dtype == torch.long
             and batch["position_ids"].shape == ids.shape
             and batch["position_ids"].device == ids.device
             and batch.get("attention_mask") is None,
             "EP4 direct calls require explicit causal per-lane positions")
    if query:
        labels = batch["labels"]
        _require(labels.dtype == torch.long and labels.shape == ids.shape
                 and labels.device == ids.device
                 and not bool((labels.ne(-100) & ~valid).any()),
                 "EP4 labels must already align with valid predictor positions")
        counts = labels.ne(-100).sum(1).tolist()
        _require(sum(counts) <= denominator
                 and item.get("loss_target_tokens", sum(counts)) == sum(counts),
                 "local EP4 target count exceeds the global group")
        state = item["reader_state"]
        _require(state.batch_size == 3 and state.values.numel() > 0
                 and state.values.dtype == torch.float32
                 and state.values.requires_grad and state.values.is_leaf,
                 "every EP4 rank needs an independent nonempty FP32 reader leaf")
    else:
        _require(batch.get("labels") is None, "evidence must not score CE targets")
    # Token IDs, masks and local target counts deliberately differ across DP.
    description = {"query": query, "backward": backward, "shape": [3, 640],
                   "microbatches": 1, "global_group_targets": denominator}
    copies = [None] * dist.get_world_size()
    dist.all_gather_object(copies, description)
    _require(all(value == description for value in copies),
             "EP4 ranks disagree on the next collective-bearing model action")
    return torch, item


def schedule_query_microbatches(session, microbatches, *,
                               group_total_target_tokens, backward=True):
    """Return LOCAL loss sums; every rank backpropagates its global-normalized loss.

    An empty local target mask still sums an empty graph-connected selection.
    Its zero cotangent traverses expert collectives alongside nonempty ranks.
    The caller later SUMs the four local writer gradients exactly once.
    """
    _require(type(backward) is bool, "backward must be explicit")
    torch, item = _validate(
        session, microbatches, query=True,
        denominator=group_total_target_tokens, backward=backward)
    batch, state = item["batch"], item["reader_state"]
    labels = batch["labels"]
    mask = labels.ne(-100)
    counts = mask.sum(1)
    with torch.enable_grad():
        losses, capture = session.adapter.run_native(
            lambda: session.models[0](**batch),
            input_ids=batch["input_ids"], valid_mask=item["valid_mask"],
            owner_ids=item["owner_ids"], mode="query", state=state)
        _require(isinstance(losses, torch.Tensor) and losses.shape == labels.shape
                 and losses.requires_grad,
                 "EP4 query must return graph-connected per-token native CE")
        selected = losses.float()[mask]
        _require(bool(selected.isfinite().all()), "nonfinite selected native query loss")
        total = selected.sum()
        lane_sums = torch.where(mask, losses.float(), torch.zeros_like(losses.float())).sum(1)
        objective = total / group_total_target_tokens
        values = lane_sums.detach().cpu().tolist()
        count_values = counts.cpu().tolist()
        total_value = float(total.detach())
        if backward:
            objective.backward()
            _require(state.values.grad is not None
                     and bool(state.values.grad.isfinite().all()),
                     "EP4 native query did not return a finite local reader cotangent")
    result = {
        "microbatch_index": 0, "loss_sum": total_value,
        "loss_sum_by_lane": values,
        "loss_target_tokens_by_lane": count_values,
        "query_target_tokens": sum(count_values),
        "capture": capture, "reader_loss_divisor": group_total_target_tokens,
        "normalized_loss": total_value / group_total_target_tokens,
    }
    return {
        "microbatches": [result], "num_microbatches": 1, "micro_batch_size": 3,
        "sequence_length": 640, "schedule": "direct_native_no_pipeline",
        "loss_sum": total_value, "query_target_tokens": sum(count_values),
        "group_total_target_tokens": group_total_target_tokens,
        "reader_loss_divisor": group_total_target_tokens,
        "query_grad_enabled": True, "backward": backward,
        "statistics_are_rank_local": True, "TP_reduction": "none_TP1",
        "DP_loss_average": False,
    }


def collect_evidence_microbatches(session, microbatches):
    """Fresh base-only H on every rank, including zero-write inactive ranks."""
    torch, item = _validate(session, microbatches, query=False)
    captured = []

    def capture_hidden(module, args, kwargs):
        value = args[0] if args else kwargs.get("hidden_states", kwargs.get("input_"))
        _require(isinstance(value, torch.Tensor)
                 and tuple(value.shape) == (640, 3, 2560)
                 and value.dtype == torch.bfloat16 and not captured,
                 "EP4 LM-head input must be one unsharded [640,3,2560] BF16 tensor")
        captured.append(value.detach().clone())

    hook = language_model(session.raw).output_layer.register_forward_pre_hook(
        capture_hidden, with_kwargs=True)
    try:
        with torch.no_grad():
            output, capture = session.adapter.run_native(
                lambda: session.models[0](**item["batch"]),
                input_ids=item["batch"]["input_ids"], valid_mask=item["valid_mask"],
                owner_ids=item["owner_ids"], mode="evidence", state=None)
            del output
        _require(len(captured) == 1, "EP4 evidence did not capture native last H")
        hidden = captured.pop().transpose(0, 1).contiguous().cpu()
        rows = capture.physical_rows.detach().cpu()
        _require(tuple(rows.shape) == (3, 640, 16),
                 "EP4 evidence physical row layout differs")
        result = {"hidden": hidden, "physical_rows": rows, "capture": capture}
    finally:
        hook.remove()
    return {
        "microbatches": [result], "num_microbatches": 1, "micro_batch_size": 3,
        "sequence_length": 640, "query_target_tokens": 0,
        "base_only": True, "backward": False, "TP_gather": False, "PP_gather": False,
    }

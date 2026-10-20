"""Variable-B native EP4 calls for staged evidence and query execution.

Each lane remains one independent original [640]-token frame. The caller
preserves chronological writes and maps every query leaf back to its immutable
time snapshot. Native query arithmetic, PLE, HC and routing are unchanged.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from qwen38_native_ep4_schedule import _runtime
from qwen38_native_runtime_preflight import language_model


GPT_SOURCE_SHA256 = "5b7a0185822c553ab65db3e344ec06b0895472ea9f7647689b2cde639986d4f1"
GDN_SOURCE_SHA256 = "028261836045a3f5ce83f993f5ab92b5833a685099d0706d26e7ee939dc72b3b"
BRIDGE_SOURCE_COMMIT = "3a5ab71264ca0c98db9eab7cc4de58aa57e065e3"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate(session, microbatches, *, query, denominator=None, backward=False):
    torch, dist = _runtime(session)
    _require(isinstance(microbatches, (list, tuple)) and len(microbatches) > 0,
             "staged EP4 calls require at least one microbatch")
    if query:
        _require(type(denominator) is int and denominator > 0,
                 "supply the actual global optimizer-group target denominator")
    shapes, leaves, total_targets = [], [], 0
    for item in microbatches:
        batch = item["batch"]
        ids, valid, owners = batch["input_ids"], item["valid_mask"], item["owner_ids"]
        _require(isinstance(ids, torch.Tensor) and ids.dtype == torch.long
                 and ids.ndim == 2 and ids.shape[0] > 0 and ids.shape[1] == 640
                 and ids.device == session.device,
                 "staged EP4 input must be int64 [B,640] on the model device")
        size = ids.shape[0]
        _require(valid.dtype == torch.bool and valid.shape == ids.shape
                 and valid.device == ids.device and owners.dtype == torch.long
                 and tuple(owners.shape) == (size,) and owners.device == ids.device
                 and torch.equal(owners, torch.arange(size, device=ids.device)),
                 "each staged snapshot must have its own local lane namespace")
        _require(batch["position_ids"].dtype == torch.long
                 and batch["position_ids"].shape == ids.shape
                 and batch["position_ids"].device == ids.device
                 and batch.get("attention_mask") is None,
                 "preserve independent causal frames and explicit per-lane positions")
        if query:
            labels = batch["labels"]
            _require(labels.dtype == torch.long and labels.shape == ids.shape
                     and labels.device == ids.device
                     and not bool((labels.ne(-100) & ~valid).any()),
                     "native labels must already align with valid predictor positions")
            count = int(labels.ne(-100).sum())
            total_targets += count
            _require(item.get("loss_target_tokens", count) == count,
                     "staged native target count differs from its collated frames")
            state = item["reader_state"]
            _require(state.batch_size == size and state.values.numel() > 0
                     and state.values.dtype == torch.float32 and state.values.is_leaf
                     and state.values.requires_grad and state.values.grad is None,
                     "every query microbatch requires a fresh nonempty FP32 reader leaf")
            leaves.append(state.values)
        else:
            _require(batch.get("labels") is None, "base-only evidence cannot score targets")
        shapes.append([size, 640])
    if query:
        _require(total_targets <= denominator
                 and len({id(leaf) for leaf in leaves}) == len(leaves),
                 "query target budget or independent microbatch leaves differ")
    description = {
        "query": query, "backward": backward, "shapes": shapes,
        "global_group_targets": denominator,
    }
    copies = [None] * dist.get_world_size()
    dist.all_gather_object(copies, description)
    _require(all(value == description for value in copies),
             "EP4 ranks disagree on the next collective-bearing staged actions")
    return torch, shapes


def schedule_query_microbatches(session, microbatches, *,
                               group_total_target_tokens, backward=True):
    """Native CE and local leaf cotangents, with no batch or DP averaging."""
    _require(type(backward) is bool, "backward must be explicit")
    torch, shapes = _validate(
        session, microbatches, query=True,
        denominator=group_total_target_tokens, backward=backward)
    results = []
    for index, item in enumerate(microbatches):
        batch, state = item["batch"], item["reader_state"]
        labels = batch["labels"]
        mask = labels.ne(-100)
        with torch.enable_grad():
            losses, capture = session.adapter.run_native(
                lambda: session.models[0](**batch),
                input_ids=batch["input_ids"], valid_mask=item["valid_mask"],
                owner_ids=item["owner_ids"], mode="query", state=state)
            _require(isinstance(losses, torch.Tensor) and losses.shape == labels.shape
                     and losses.requires_grad,
                     "staged native query must return graph-connected per-token CE")
            losses = losses.float()
            selected = losses[mask]
            _require(bool(selected.isfinite().all()), "nonfinite selected native query loss")
            # For an all-inactive rank the empty selection still retains the
            # actual model graph, so zero backward participates in EP all-to-all.
            total = selected.sum()
            lane_sums = torch.where(mask, losses, torch.zeros_like(losses)).sum(1)
            objective = total / group_total_target_tokens
            lane_values = lane_sums.detach().cpu().tolist()
            lane_counts = mask.sum(1).cpu().tolist()
            total_value = float(total.detach())
            if backward:
                objective.backward()
                _require(state.values.grad is not None
                         and bool(state.values.grad.isfinite().all()),
                         "staged native query did not return a finite local cotangent")
        results.append({
            "microbatch_index": index, "loss_sum": total_value,
            "loss_sum_by_lane": lane_values,
            "loss_target_tokens_by_lane": lane_counts,
            "query_target_tokens": sum(lane_counts), "capture": capture,
            "reader_loss_divisor": group_total_target_tokens,
            "normalized_loss": total_value / group_total_target_tokens,
        })
        # No reader graph survives a query microbatch. The independent writer
        # graphs and cotangents remain owned by the caller's trajectory bridges.
        del losses, selected, total, lane_sums, objective, capture
    return {
        "microbatches": results, "num_microbatches": len(results),
        "micro_batch_size": shapes[0][0], "shapes": shapes, "sequence_length": 640,
        "schedule": "staged_direct_native_no_pipeline",
        "loss_sum": sum(item["loss_sum"] for item in results),
        "query_target_tokens": sum(item["query_target_tokens"] for item in results),
        "group_total_target_tokens": group_total_target_tokens,
        "reader_loss_divisor": group_total_target_tokens,
        "query_grad_enabled": True, "backward": backward,
        "statistics_are_rank_local": True,
        "TP_reduction": "none_TP1", "DP_loss_average": False,
    }


class _EvidenceHeadReached(Exception):
    """Private control flow, caught inside the adapter's native-call closure."""


def _evidence_head_identity(model):
    """Pin the exact native H boundary before omitting the unused output head."""
    from mcore_bridge.model.gpt_model import GPTModel
    from mcore_bridge.model.modules.gated_delta_net import GatedDeltaNet
    _require(type(model) is GPTModel and model.post_process
             and model.config.task_type == "causal_lm"
             and not getattr(model, "mtp_process", False),
             "head omission requires the pinned causal GPT without MTP")
    gpt_source = hashlib.sha256(Path(inspect.getfile(GPTModel)).read_bytes()).hexdigest()
    gdn_source = hashlib.sha256(Path(inspect.getfile(GatedDeltaNet)).read_bytes()).hexdigest()
    _require(gpt_source == GPT_SOURCE_SHA256 and gdn_source == GDN_SOURCE_SHA256,
             "native GPT/GDN source differs from the inspected variable-B evidence boundary")
    return {
        "protocol": "qwen38_native_pre_lm_head_evidence_v1",
        "bridge_commit": BRIDGE_SOURCE_COMMIT,
        "gpt_source_sha256": gpt_source, "gdn_source_sha256": gdn_source,
        "boundary": "unchanged_native_output_layer_forward_pre_hook_input",
        "evidence_lm_head_executed": False, "evidence_logits_allocated": False,
        "query_lm_head_unchanged": True,
        "omitted": "unused vocabulary projection and output transpose after final H",
        "loader_cli_micro_batch_size": 3,
        "actual_forward_batch": "derived_from_input_shape; no_pipeline_schedule",
    }


def _collect_one_evidence(session, item, *, torch, model, batch_size):
    """Preserve the complete native decoder; stop before its unused LM head."""
    captured = []

    def capture_hidden(module, args, kwargs):
        value = args[0] if args else kwargs.get("hidden_states", kwargs.get("input_"))
        _require(isinstance(value, torch.Tensor)
                 and tuple(value.shape) == (640, batch_size, 2560)
                 and value.dtype == torch.bfloat16 and not captured,
                 "staged LM-head input must be one unsharded [640,B,2560] BF16 tensor")
        captured.append(value.detach().clone())
        # Every expert collective is upstream of the final LM head in the
        # pinned causal-lm/no-MTP/TP1/PP1 configuration. Catch this *inside*
        # run_native so its normal PLE completion and capture checks still run.
        raise _EvidenceHeadReached()

    def native_until_head():
        try:
            session.models[0](**item["batch"])
        except _EvidenceHeadReached:
            return None
        raise RuntimeError("native evidence failed to reach its registered LM-head boundary")

    hook = model.output_layer.register_forward_pre_hook(capture_hidden, with_kwargs=True)
    try:
        with torch.no_grad():
            _, capture = session.adapter.run_native(
                native_until_head, input_ids=item["batch"]["input_ids"],
                valid_mask=item["valid_mask"], owner_ids=item["owner_ids"],
                mode="evidence", state=None)
        _require(len(captured) == 1, "staged evidence did not capture native last H")
        hidden = captured.pop().transpose(0, 1).contiguous().cpu()
        rows = capture.physical_rows.detach().cpu()
        _require(tuple(rows.shape) == (batch_size, 640, 16),
                 "staged evidence physical row layout differs")
        return {"hidden": hidden, "physical_rows": rows, "capture": capture}
    finally:
        hook.remove()


def collect_evidence_microbatches(session, microbatches):
    """Fresh no-grad base H with exact pre-head capture and no unused logits."""
    torch, shapes = _validate(session, microbatches, query=False)
    model = language_model(session.raw)
    identity = _evidence_head_identity(model)
    results = [
        _collect_one_evidence(session, item, torch=torch, model=model, batch_size=shape[0])
        for item, shape in zip(microbatches, shapes)
    ]
    return {
        "microbatches": results, "num_microbatches": len(results),
        "micro_batch_size": shapes[0][0], "shapes": shapes, "sequence_length": 640,
        "query_target_tokens": 0, "base_only": True, "backward": False,
        "TP_gather": False, "PP_gather": False, "evidence_execution": identity,
    }

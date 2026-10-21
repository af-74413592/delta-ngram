"""Read-only TP2/PP4/EP2 native forward helpers for the A800 experiment.

Evidence returns actual native last-H, never a vLLM surrogate. Queries score
already shifted native labels. This module has no backward or optimizer path.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import importlib
import inspect
from pathlib import Path
import torch

from delta_ngram.qwen38_seven_forward import require, digest, immutable_guard, assert_unchanged
from qwen38_native_runtime_preflight import language_model


def runtime(session):
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.pipeline_parallel import get_forward_backward_func
    require(dist.is_initialized() and len(session.models) == 1, "native session is not initialized")
    require((ps.get_tensor_model_parallel_world_size(), ps.get_pipeline_model_parallel_world_size(),
             ps.get_expert_model_parallel_world_size(), ps.get_context_parallel_world_size(),
             ps.get_data_parallel_world_size()) == (2, 4, 2, 1, 1), "requires TP2/PP4/EP2/CP1/DP1")
    schedule = get_forward_backward_func()
    require(schedule.__name__ == "forward_backward_pipelining_without_interleaving", "unexpected pipeline schedule")
    require(digest(inspect.getfile(schedule)) ==
            "946a71d2fd69dfebaceab461bd8e01a37cc4797221651d64943c2c5c9c4e6f0a", "schedule source changed")
    require(not session.models[0].training and session.models[0].config.recompute_granularity is None,
            "reader must remain eval with no activation recomputation")
    return dist, ps, schedule


@contextmanager
def native_dispatch(session):
    """Same sorted-router policy; separately pinned A800 HC eager source."""
    from qwen38_native_a800_dispatch import a800_hc_dispatch, HC_MODULE
    from qwen38_native_router import native_router_dispatch
    from delta_ngram.qwen38_precision import Qwen38SplitPLEPrecision
    from contextlib import ExitStack
    from qwen38_native_a800_session import runtime_phase
    with ExitStack() as stack:
        runtime_phase("dispatch_hc_begin")
        hc = stack.enter_context(a800_hc_dispatch(importlib.import_module(HC_MODULE), torch=torch))
        runtime_phase("dispatch_router_begin")
        router = stack.enter_context(native_router_dispatch(session.raw))
        runtime_phase("dispatch_precision_begin")
        precision = stack.enter_context(Qwen38SplitPLEPrecision(session.adapter)) if session.adapter is not None else None
        runtime_phase("dispatch_ready")
        yield {"HC": hc, "router": router["identity"],
               "precision": precision.identity() if precision is not None else None}


def detached_read_snapshot(state):
    """Independent FP32 pre-write state; preserve every lazy-clock field."""
    from delta_ngram.alpha_memory import LazyDecayState
    from delta_ngram.batching import BatchMemoryState
    from delta_ngram.state import SparseMemoryState
    require(isinstance(state, LazyDecayState) and state.batch_size == 1
            and state.precision is None and state.values.dtype == torch.float32,
            "memory encoding requires one-owner FP32 lazy state")
    sparse = SparseMemoryState(*(getattr(state, name).detach().clone()
                                 for name in ("rows", "values", "counts")))
    memory = BatchMemoryState(sparse, 1, state.row_capacity)
    return LazyDecayState(memory, *(getattr(state, name).detach().clone() for name in
        ("block_clock", "cumulative_log_alpha", "row_last_written_block", "row_log_alpha_at_write")))


class _EvidenceHeadReached(Exception):
    pass


@torch.no_grad()
def native_forward(session, batch, valid_mask, *, state=None, evidence=False, return_logits=False,
                   encoding_kind=None):
    """One evidence, TF, or read-only last-valid-position logits call on every rank."""
    dist, ps, schedule = runtime(session)
    ids = batch["input_ids"]
    b, s = ids.shape
    require(ids.dtype == torch.long and 2 <= s <= 8192 and s % 2 == 0
            and valid_mask.shape == ids.shape and valid_mask.dtype == torch.bool,
            "invalid SP-even native input layout")
    require(evidence or b == 1, "fixed-state TF captures currently use one query per lane")
    require(not (evidence and return_logits), "evidence and logits modes are mutually exclusive")
    require(encoding_kind is None or evidence, "encoding_kind is only valid for evidence")
    encoding_kind = (encoding_kind or "base") if evidence else None
    require(encoding_kind in (None, "base", "memory"), "unknown evidence encoding")
    require(not (evidence and encoding_kind == "base" and state is not None),
            "base evidence must not receive state")
    require(not (evidence and encoding_kind == "memory" and b != 1),
            "memory evidence is causal B1 per trajectory")
    require((batch.get("labels") is None) == (evidence or return_logits), "evidence/TF/logits label contract mismatch")
    require(batch.get("attention_mask") is None, "this runner uses independent causal lanes")
    guard = immutable_guard(state) if state is not None else None
    read_state = state
    if evidence and encoding_kind == "memory" and session.adapter is not None:
        read_state = detached_read_snapshot(state)
    read_guard = immutable_guard(read_state) if read_state is not None else None
    capture = None
    hidden = []
    next_logits = []
    if return_logits:
        lengths = valid_mask.sum(dim=1)
        require(bool(lengths.gt(0).all()) and torch.equal(valid_mask,
                torch.arange(s, device=ids.device)[None] < lengths[:, None]),
                "logits mode requires a nonempty contiguous valid prefix")
    hook = None
    if evidence and ps.is_pipeline_last_stage():
        def before_head(module, args, kwargs):
            value = args[0] if args else kwargs.get("hidden_states", kwargs.get("input_"))
            require(isinstance(value, torch.Tensor) and value.shape == (s // 2, b, 2560)
                    and value.dtype == torch.bfloat16 and not hidden, "actual last-H layout changed")
            hidden.append(value.detach().clone())
            raise _EvidenceHeadReached()
        hook = language_model(session.raw).output_layer.register_forward_pre_hook(before_head, with_kwargs=True)

    def forward_step(iterator, model):
        nonlocal capture
        current = dict(batch)
        if not ps.is_pipeline_last_stage():
            current["labels"] = None
        def call_model():
            try:
                return model(**current)
            except _EvidenceHeadReached:
                require(evidence and ps.is_pipeline_last_stage() and len(hidden) == 1,
                        "unexpected evidence head boundary")
                return hidden[0].new_zeros((b, s), dtype=torch.float32)

        if session.adapter is None:
            output = call_model()
        else:
            require(evidence or state is not None, "TF read requires even an explicit empty memory")
            output, capture = session.adapter.run_native(call_model, input_ids=ids,
                valid_mask=valid_mask, owner_ids=torch.arange(b, device=ids.device),
                mode="evidence" if evidence and encoding_kind == "base" else "query",
                state=read_state)

        def score(output):
            if return_logits:
                require(isinstance(output, torch.Tensor) and output.ndim == 3
                        and output.shape[:2] == (b, s), "native logits must be [B,S,Vlocal]")
                selected = output[torch.arange(b, device=ids.device), lengths - 1]
                next_logits.append(selected.detach().float().contiguous().clone())
                return output.new_zeros((), dtype=torch.float32), {"diagnostic_logits": True}
            if evidence:
                tensor = output[0] if isinstance(output, tuple) else output
                return tensor.new_zeros((), dtype=torch.float32), {"loss_sum": 0., "target_tokens": 0}
            labels = batch["labels"]
            require(isinstance(output, torch.Tensor) and output.shape == labels.shape,
                    "native TF output must be per-token CE")
            selected = output.float()[labels.ne(-100)]
            require(selected.numel() > 0 and bool(selected.isfinite().all()), "invalid native answer NLL")
            total = selected.sum()
            return total, {"loss_sum": float(total), "target_tokens": selected.numel()}
        return output, score

    try:
        if evidence:
            from qwen38_native_memory_schedule import preserve_encoding_rng
        with preserve_encoding_rng(session.device) if evidence else nullcontext():
            metrics = schedule(forward_step_func=forward_step, data_iterator=None,
                model=session.models, num_microbatches=1, seq_length=s,
                micro_batch_size=b, forward_only=True)
    finally:
        if hook is not None:
            hook.remove()
    if guard is not None:
        assert_unchanged(guard)
    if read_guard is not None:
        assert_unchanged(read_guard)
    if return_logits:
        full = torch.empty((b, 248320), dtype=torch.float32, device=ids.device)
        if ps.is_pipeline_last_stage():
            require(len(next_logits) == 1, "native logits were not captured exactly once")
            local = next_logits[0]
            tp = ps.get_tensor_model_parallel_world_size()
            if local.shape[1] == full.shape[1]:
                full.copy_(local)
            else:
                require(local.shape[1] * tp == full.shape[1], "native vocabulary shard width changed")
                parts = [torch.empty_like(local) for _ in range(tp)]
                dist.all_gather(parts, local, group=ps.get_tensor_model_parallel_group())
                full.copy_(torch.cat(parts, dim=-1))
        dist.broadcast(full, src=ps.get_pipeline_model_parallel_last_rank(),
                       group=ps.get_pipeline_model_parallel_group())
        require(bool(full.isfinite().all()), "native next-token logits contain nonfinite values")
        return {"next_logits": full}
    if evidence:
        full = torch.empty((s, b, 2560), dtype=torch.bfloat16, device=ids.device)
        if ps.is_pipeline_last_stage():
            require(len(hidden) == 1, "native evidence head did not run exactly once")
            parts = [torch.empty_like(hidden[0]) for _ in range(2)]
            dist.all_gather(parts, hidden[0], group=ps.get_tensor_model_parallel_group())
            full.copy_(torch.cat(parts, dim=0))
        dist.broadcast(full, src=ps.get_pipeline_model_parallel_last_rank(),
                       group=ps.get_pipeline_model_parallel_group())
        return {"hidden": full.transpose(0, 1).contiguous(),
                "physical_rows": capture.physical_rows if capture is not None else None}
    result = torch.zeros(2, dtype=torch.float64, device=ids.device)
    if ps.is_pipeline_last_stage():
        require(len(metrics) == 1, "native TF schedule returned unexpected metrics")
        result[:] = torch.tensor([metrics[0]["loss_sum"], metrics[0]["target_tokens"]],
                                 device=ids.device, dtype=torch.float64)
    dist.broadcast(result, src=ps.get_pipeline_model_parallel_last_rank(),
                   group=ps.get_pipeline_model_parallel_group())
    return {"loss_sum": float(result[0]), "token_count": int(result[1]),
            "answer_nll": float(result[0] / result[1])}


@torch.no_grad()
def sync_written_rows(state, touched, *, device):
    """Broadcast only updated rows from TP0; maintain one canonical trajectory.

Only the two first-stage ranks call this. TP1 does not execute D1 independently.
This avoids turning a single-chain experiment into two floating-point replicas.
"""
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from delta_ngram.alpha_memory import LazyDecayState
    from delta_ngram.batching import BatchMemoryState
    from delta_ngram.state import SparseMemoryState
    group, source = ps.get_tensor_model_parallel_group(), ps.get_tensor_model_parallel_src_rank()
    root = ps.get_tensor_model_parallel_rank() == 0
    count = torch.tensor([touched.numel() if root else 0], dtype=torch.long, device=device)
    dist.broadcast(count, src=source, group=group)
    size = int(count)
    rows = touched if root else torch.empty(size, dtype=torch.long, device=device)
    dist.broadcast(rows, src=source, group=group)
    if root:
        indices = torch.searchsorted(state.rows, rows)
        values, counts = state.values[indices].contiguous(), state.counts[indices].contiguous()
        last = state.row_last_written_block[indices].contiguous()
    else:
        values = torch.empty((size, state.width), dtype=torch.float32, device=device)
        counts = torch.empty(size, dtype=torch.float32, device=device)
        last = torch.empty(size, dtype=torch.long, device=device)
    clock = state.block_clock.clone()
    for value in (values, counts, last, clock):
        dist.broadcast(value, src=source, group=group)
    if root:
        return state
    union = torch.unique(torch.cat((state.rows, rows)), sorted=True)
    old_values, old_counts = state.memory.memory.lookup(union)
    slots, prior_slots = torch.searchsorted(union, rows), torch.searchsorted(union, state.rows)
    old_values = old_values.index_copy(0, slots, values)
    old_counts = old_counts.index_copy(0, slots, counts)
    old_last = torch.zeros_like(union).index_copy(0, prior_slots, state.row_last_written_block)
    old_last = old_last.index_copy(0, slots, last)
    memory = BatchMemoryState(SparseMemoryState(union, old_values, old_counts), 1, state.row_capacity)
    return LazyDecayState(memory, clock, torch.zeros_like(state.cumulative_log_alpha), old_last,
                          torch.zeros(union.numel(), dtype=torch.float64, device=device))

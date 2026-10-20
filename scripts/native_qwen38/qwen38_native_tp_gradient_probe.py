"""Bounded TP gradient-scale finite differences inside an already loaded reader.

Call run_tp_gradient_probe on ALL FOUR ranks, after removing observation hooks.
This module does not load a model, construct a writer/optimizer, or access data.
The caller's pinned loader/weight audit remains part of the enclosing receipt.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path

from qwen38_native_runtime_preflight import (
    MODEL_REPO, MODEL_REVISION, checked_tokens, digest, schedule_query, write_json,
)


STEPS = tuple(2. ** exponent for exponent in range(-10, -2))
RTOL = .20
ATOL = 1e-3
MIN_LOOKUP_CHANGED_FRACTION = 1e-3
MAX_LOSS_TARGET_TOKENS = 1000


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tensor_hash(tensor):
    import torch
    value = tensor.detach().contiguous().cpu()
    return canonical_hash({"shape": list(value.shape), "dtype": str(value.dtype),
                           "bytes_sha256": hashlib.sha256(
                               value.view(torch.uint8).numpy().tobytes()).hexdigest()})


def evaluate_fd_series(pairs, rank_gradients, *, runtime_identity_sha256,
                       direction_id, baseline_losses, rtol=RTOL, atol=ATOL):
    """Accept the smallest adjacent, informative, numerically stable FD pair.

    Pair eligibility uses only forward losses, replay noise, and actual lookup
    changes. It never selects a step by its agreement with autograd. Every pair,
    including rejected/nonlinear/noisy steps, remains in the saved evidence.
    """
    from delta_ngram.qwen38_training import GradientScaleProbe, TPGradientPolicy

    if (len(baseline_losses) < 2 or not all(math.isfinite(x) for x in baseline_losses)
            or len(rank_gradients) != 2 or not all(math.isfinite(x) for x in rank_gradients)):
        raise ValueError("finite baseline replays and exactly two TP derivatives are required")
    noise = max(baseline_losses) - min(baseline_losses)
    # Float32 CE summation floor; baseline replays also measure kernel variation.
    noise_floor = max(noise, max(abs(x) for x in baseline_losses) * 2e-7, 1e-7)
    evaluated = []
    for pair in pairs:
        step, plus, minus = pair["step"], pair["plus_loss"], pair["minus_loss"]
        if not all(math.isfinite(x) for x in (step, plus, minus)) or step <= 0:
            raise ValueError("invalid finite difference observation")
        derivative = (plus - minus) / (2 * step)
        changed = min(pair["lookup_changed_fractions"])
        eligible = (changed >= MIN_LOOKUP_CHANGED_FRACTION
                    and abs(plus - minus) > 20 * noise_floor
                    and abs(derivative) > 10 * atol)
        evaluated.append({**pair, "finite_difference": derivative,
                          "eligible_without_autograd": eligible})
    selected = None
    for left, right in zip(evaluated, evaluated[1:]):
        if (left["eligible_without_autograd"] and right["eligible_without_autograd"]
                and right["step"] > left["step"]
                and abs(right["finite_difference"] - left["finite_difference"])
                <= atol + rtol * abs(left["finite_difference"])):
            selected = (left, right)
            break
    receipts, rejected = {}, {}
    if selected is not None:
        probes = tuple(GradientScaleProbe(
            step=row["step"], finite_difference=row["finite_difference"],
            rank_directional_gradients=tuple(rank_gradients)) for row in selected)
        for reduction in ("replica", "mean", "sum"):
            try:
                policy = TPGradientPolicy(
                    reduction=reduction, world_size=2,
                    runtime_identity_sha256=runtime_identity_sha256,
                    direction_id=direction_id, probes=probes, rtol=rtol, atol=atol)
                receipts[reduction] = policy.receipt()
            except ValueError as error:
                rejected[reduction] = str(error)
    else:
        rejected["all"] = "no adjacent informative FD steps agreed without consulting autograd"
    # This explicit preference is applicable only after numerical acceptance.
    accepted = "replica" if "replica" in receipts else (
        "sum" if "sum" in receipts else ("mean" if "mean" in receipts else None))
    return {
        "status": "accepted" if accepted else "inconclusive",
        "selected_reduction": accepted,
        "selected_policy_receipt": receipts.get(accepted),
        "accepted_policy_receipts": receipts, "rejected_policies": rejected,
        "selected_steps": [row["step"] for row in selected] if selected else [],
        "rank_directional_gradients": list(rank_gradients),
        "tp_mean_directional_gradient": sum(rank_gradients) / 2,
        "tp_sum_directional_gradient": sum(rank_gradients),
        "baseline_loss_span": noise, "loss_noise_floor": noise_floor,
        "rtol": rtol, "atol": atol, "pairs": evaluated,
    }


def live_runtime_identity(models):
    """Hash exact source used for this numeric policy; paths are not identities."""
    modules = (
        "megatron.core.tensor_parallel.mappings",
        "megatron.core.tensor_parallel.layers",
        "megatron.core.tensor_parallel.cross_entropy",
        "megatron.core.extensions.transformer_engine",
        "megatron.core.pipeline_parallel.schedules",
        "transformer_engine.pytorch.module.linear",
        "transformer_engine.pytorch.module.base",
        "mcore_bridge.model.modules.ple",
        "delta_ngram.qwen38_adapter",
        "delta_ngram.qwen38_native",
        "delta_ngram.qwen38_training",
    )
    sources = {name: digest(inspect.getfile(importlib.import_module(name))) for name in modules}
    sources["schedule_query"] = digest(inspect.getfile(schedule_query))
    sources["tp_gradient_probe"] = digest(__file__)
    config = models[0].config
    return {
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "megatron-core", "transformer-engine", "ms-swift", "mcore-bridge")},
        "sources": sources,
        "topology": {"tp": 2, "pp": 2, "ep": 2, "etp": 1, "cp": 1, "dp": 1},
        "reader": {"frozen": True, "eval": True, "sequence_parallel": config.sequence_parallel,
                   "recompute_granularity": config.recompute_granularity,
                   "deallocate_pipeline_outputs": config.deallocate_pipeline_outputs,
                   "variable_seq_lengths": config.variable_seq_lengths},
        "loss_contract": "one microbatch; shifted CE sum / explicit group target tokens",
        "checkpoint_proof": "enclosing native observation receipt supplies loader/hash continuity",
    }


def run_tp_gradient_probe(*, models, adapter, layout, device, tokenizer, output):
    """Reuse native TP2/PP2 model; 20 forwards x 11 targets = 220 ledger tokens.

    Exactly one backward obtains raw per-rank gradients. The unit-L2 direction
    is selected from TP0's gradient for signal strength, then broadcast and fixed
    before any perturbed forward. Numerical derivative uses fresh loss-only
    forwards, not a second autograd path. No writer or optimizer is constructed.
    """
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    from delta_ngram.batching import BatchMemoryState
    from delta_ngram.state import SparseMemoryState

    rank = dist.get_rank()
    first = ps.is_pipeline_first_stage()
    tp_rank = ps.get_tensor_model_parallel_rank()
    tp_group = ps.get_tensor_model_parallel_group()
    output = Path(output)

    def gather(value):
        values = [None] * dist.get_world_size()
        dist.all_gather_object(values, value)
        return values

    def check(local_condition, message):
        failures = gather(None if local_condition else f"rank {rank}: {message}")
        if any(value is not None for value in failures):
            raise ValueError("; ".join(value for value in failures if value is not None))

    check((dist.get_world_size(), ps.get_tensor_model_parallel_world_size(),
           ps.get_pipeline_model_parallel_world_size(), ps.get_data_parallel_world_size(),
           ps.get_context_parallel_world_size(), ps.get_expert_model_parallel_world_size())
          == (4, 2, 2, 1, 1, 2), "requires the accepted TP2 PP2 EP2 DP1 CP1 topology")
    check(len(models) == 1 and (adapter is not None) == first
          and (layout is not None) == first and (rank == 0) == (first and tp_rank == 0),
          "requires a single chunk and one adapter per first-stage TP rank")
    parameters = list(models[0].parameters())
    check(not models[0].training and all(not p.requires_grad and p.grad is None for p in parameters),
          "reader must remain frozen and in evaluation mode")
    versions = [(p, p._version) for p in parameters]
    config = models[0].config
    check(config.sequence_parallel and config.recompute_granularity is None
          and not config.deallocate_pipeline_outputs and config.variable_seq_lengths
          and all(getattr(config, name, None) is None for name in (
              "grad_scale_func", "finalize_model_grads_func", "grad_sync_func")),
          "requires the accepted eager no-recompute schedule/scaling configuration")
    case = checked_tokens(tokenizer, 16)
    planned_forwards = 4 + 2 * len(STEPS)
    check(planned_forwards * case["loss_tokens"] <= MAX_LOSS_TARGET_TOKENS,
          "finite-difference target-token cap exceeded")
    identity = live_runtime_identity(models)
    identity_sha = canonical_hash(identity)
    check(len(set(gather(identity_sha))) == 1, "ranks disagree on runtime identity")
    registration = {
        "protocol": "qwen38_native_tp_fd_probe_v2", "runtime_identity": identity,
        "runtime_identity_sha256": identity_sha, "case": case,
        "steps": list(STEPS), "rtol": RTOL, "atol": ATOL,
        "direction_rule": "unit L2 raw TP0 gradient at fixed sine delta; broadcast to both PLE ranks",
        "baseline_delta_rule": "sin(flat_index * .017) * .01 in FP32",
        "step_selection": "smallest adjacent informative stable forward FD pair; no gradient filtering",
        "minimum_actual_lookup_changed_fraction": MIN_LOOKUP_CHANGED_FRACTION,
        "planned_forward_passes": planned_forwards,
        "planned_loss_target_tokens": planned_forwards * case["loss_tokens"],
        "backward_passes": 1, "optimizer_steps": 0,
        "logical_loss": "CE sum / 11; TP loss copies are never summed",
        "mode": "fresh uncached forward; all nonempty delta requires_grad; grad-enabled FD; identical RNG replay",
        "baseline_gate": "analytic and grad-enabled forward-only baseline must match exactly before FD sweep",
    }
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "registration.json", registration)
    dist.barrier()
    batch = {
        "input_ids": torch.tensor([case["input_ids"]], dtype=torch.long, device=device),
        "labels": torch.tensor([case["labels"]], dtype=torch.long, device=device),
        "position_ids": torch.arange(16, device=device).unsqueeze(0), "attention_mask": None,
    }
    valid = torch.tensor([case["valid_mask"]], dtype=torch.bool, device=device)
    owner = torch.zeros(1, dtype=torch.long, device=device)
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    tracker = get_cuda_rng_tracker()
    rng_tp = {name: value.clone() for name, value in tracker.get_states().items()}
    lookup = []
    hook = None
    records = []

    def restore_rng():
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda, device)
        tracker.set_states({name: value.clone() for name, value in rng_tp.items()})

    def capture_lookup(module, args, result):
        if not torch.is_tensor(result) or tuple(result.shape) != (1, 16, 2560):
            raise ValueError("native embedding output shape changed")
        lookup.append(result.detach().cpu().clone())

    if first:
        hook = adapter.embedding.register_forward_hook(capture_lookup)

    def run(name, state, *, backward=False, grad_enabled=True):
        restore_rng()
        lookup.clear()
        result = schedule_query(
            models, batch, valid_mask=valid, owner_ids=owner, adapter=adapter,
            reader_state=state, backward=backward, group_total_target_tokens=case["loss_tokens"],
            query_grad_enabled=grad_enabled)
        torch.cuda.synchronize(device)
        check(not first or (len(lookup) == 1 and lookup[0].dtype == torch.bfloat16),
              "one actual BF16 embedding output per forward is required")
        losses = gather(result["normalized_loss"])
        check(max(losses) - min(losses) <= 1e-7, "TP/PP replicas disagree on the logical loss")
        entry = {key: result[key] for key in (
            "loss_sum", "query_target_tokens", "group_total_target_tokens", "normalized_loss",
            "query_grad_enabled")}
        entry.update({"phase": name, "backward": backward,
                      "lookup_sha256": tensor_hash(lookup[0]) if first else None})
        phase_ranks = gather(entry)
        observed_hashes = [row["lookup_sha256"] for row in phase_ranks if row["lookup_sha256"]]
        check(len(observed_hashes) == 2 and len(set(observed_hashes)) == 1,
              "PLE replicas must see byte-identical actual lookup values")
        records.append(phase_ranks)
        if rank == 0:
            write_json(output / "progress.json", {
                "complete": False, "completed_forward_passes": len(records),
                "loss_target_tokens": len(records) * case["loss_tokens"], "phases": records})
        return result, lookup[0] if first else None

    try:
        empty = layout.empty_state(1, device=device) if first else None
        addressed, _ = run("address_empty", empty, grad_enabled=False)
        rows = counts = base = direction = gradient = None
        if first:
            physical = addressed["capture"].physical_rows
            rows = torch.unique(physical[valid].reshape(-1), sorted=True)
            rows = rows[rows > 0]
            counts = torch.ones(rows.numel(), device=device)
            grid = torch.arange(rows.numel() * 160, device=device, dtype=torch.float32)
            base = grid.mul(.017).sin().reshape(-1, 160) * .01
        local_state_hash = canonical_hash({"rows": tensor_hash(rows), "values": tensor_hash(base)}) \
            if first else None
        state_hashes = [value for value in gather(local_state_hash) if value is not None]
        check(len(state_hashes) == 2 and len(set(state_hashes)) == 1,
              "both PLE replicas must use identical rows and initial delta")

        def state_for(values):
            if not first:
                return None
            if not values.requires_grad:
                values = values.detach().clone().requires_grad_(True)
            # Direct constructor keeps values itself as the leaf (no cat grad_fn).
            return BatchMemoryState(SparseMemoryState(rows, values, counts), 1, layout.row_capacity)

        leaf = base.clone().requires_grad_() if first else None
        baseline, base_lookup = run("delta_backward", state_for(leaf), backward=True)
        check(not first or (leaf.grad is not None and bool(leaf.grad.isfinite().all())
                            and float(leaf.grad.norm()) > 0), "missing/nonfinite/zero native delta gradient")
        if first:
            gradient = leaf.grad.detach().clone()
            direction = gradient / gradient.norm()
            dist.broadcast(direction, src=ps.get_tensor_model_parallel_src_rank(), group=tp_group)
        directions = [value for value in gather(tensor_hash(direction) if first else None)
                      if value is not None]
        check(len(directions) == 2 and len(set(directions)) == 1,
              "the same direction must perturb both PLE replicas")
        direction_id = directions[0]
        derivative = float((gradient.double() * direction.double()).sum()) if first else None
        per_rank_gradients = [value for value in gather(derivative) if value is not None]
        if first:
            torch.save({"rows": rows.cpu(), "base_delta": base.cpu(),
                        "direction": direction.cpu(), "gradient": gradient.cpu()},
                       output / f"direction-rank-{rank}.pt")
        del leaf
        repeat_before, replay_lookup = run("baseline_replay_before", state_for(base))
        check(not first or torch.equal(base_lookup, replay_lookup), "lookup changed at repeated delta")
        if baseline["normalized_loss"] != repeat_before["normalized_loss"]:
            result = {
                "protocol": registration["protocol"], "complete": True, "status": "inconclusive",
                "selected_reduction": None, "selected_policy_receipt": None,
                "reason": "analytic and grad-enabled forward-only baseline still differ; FD sweep not run",
                "runtime_identity_sha256": identity_sha, "direction_id": direction_id,
                "forward_passes": len(records), "backward_passes": 1,
                "loss_target_tokens": len(records) * case["loss_tokens"],
                "optimizer_steps": 0, "phases": records,
            }
            if rank == 0:
                write_json(output / "COMPLETE.json", result)
            dist.barrier()
            return result
        pairs = []
        for index, step in enumerate(STEPS):
            plus, plus_lookup = run(f"step_{index}_plus", state_for(base + step * direction) if first else None)
            minus, minus_lookup = run(f"step_{index}_minus", state_for(base - step * direction) if first else None)
            local_lookup = None
            if first:
                change = plus_lookup[valid.cpu()].float() - minus_lookup[valid.cpu()].float()
                local_lookup = {
                    "changed_fraction": float(change.ne(0).float().mean()),
                    "difference_l2": float(change.norm()), "difference_max_abs": float(change.abs().max()),
                    "plus_vs_base_fraction": float(plus_lookup.ne(base_lookup).float().mean()),
                    "minus_vs_base_fraction": float(minus_lookup.ne(base_lookup).float().mean()),
                }
            lookup_stats = [value for value in gather(local_lookup) if value is not None]
            check(len(lookup_stats) == 2 and lookup_stats[0] == lookup_stats[1],
                  "TP replicas disagree on the actual perturbed embedding")
            pairs.append({
                "step": step, "plus_loss": plus["normalized_loss"], "minus_loss": minus["normalized_loss"],
                "lookup_changed_fractions": [value["changed_fraction"] for value in lookup_stats],
                "lookup_by_tp_rank": lookup_stats,
            })
        repeat_after, replay_lookup = run("baseline_replay_after", state_for(base))
        check(not first or torch.equal(base_lookup, replay_lookup), "lookup changed after FD sweep")
        check(all(p._version == version and p.grad is None and not p.requires_grad for p, version in versions),
              "reader parameters changed or received gradients during finite differences")
        result = evaluate_fd_series(
            pairs, per_rank_gradients, runtime_identity_sha256=identity_sha, direction_id=direction_id,
            baseline_losses=[baseline["normalized_loss"], repeat_before["normalized_loss"],
                             repeat_after["normalized_loss"]])
        result.update({
            "protocol": registration["protocol"], "complete": True,
            "runtime_identity_sha256": identity_sha, "registration_sha256": canonical_hash(registration),
            "direction_id": direction_id, "forward_passes": len(records), "backward_passes": 1,
            "loss_target_tokens": len(records) * case["loss_tokens"],
            "backward_loss_target_tokens": case["loss_tokens"],
            "logical_forward_input_tokens": len(records) * 16,
            "optimizer_steps": 0, "reader_unchanged": True, "phases": records,
            "limits": [
                "A directional FP32-overlay/BF16-reader scaling check, not an optimizer or writer update.",
                "Finite differences approximate the smooth autograd surrogate across BF16 rounding.",
                "No stable informative pair means no TP scaling authorization.",
            ],
        })
        check(len(records) == planned_forwards, "unexpected number of probe passes")
        if rank == 0:
            write_json(output / "COMPLETE.json", result)
            print(json.dumps({"event": "tp_gradient_probe_complete", "status": result["status"],
                              "reduction": result["selected_reduction"],
                              "loss_target_tokens": result["loss_target_tokens"]}), flush=True)
        dist.barrier()
        return result
    finally:
        if hook is not None:
            hook.remove()
        restore_rng()

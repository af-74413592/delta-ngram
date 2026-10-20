"""FP32 full-history writer finite differences with a fixed real reader cotangent.

The native anchor query is supplied by the caller and is never repeated here.
This proves the writer VJP for phi(theta)=<M(theta), fixed dL/dM>, not smoothness
of the BF16 reader CE. Every scalar evaluation rebuilds the actual cached-H
history from empty memory. At least two writes are required.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from qwen38_native_runtime_preflight import write_json


RELATIVE_STEPS = (1e-4, 2e-4, 4e-4, 8e-4)
RTOL, ATOL = .03, 1e-6


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def named_tensor_hash(tensors):
    import torch
    result = hashlib.sha256()
    for name, value in tensors:
        value = value.detach().cpu().contiguous()
        result.update(canonical_hash({"name": name, "shape": list(value.shape),
                                      "dtype": str(value.dtype)}).encode())
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def evaluate_adjoint_series(pairs, derivatives, baseline_scalars):
    """Choose the first stable informative pair using scalar forwards only."""
    if (len(derivatives) != 2 or len(baseline_scalars) != 3
            or not all(math.isfinite(x) for x in (*derivatives, *baseline_scalars))):
        raise ValueError("two finite TP derivatives and three baseline scalars are required")
    noise = max(baseline_scalars) - min(baseline_scalars)
    floor = max(noise, max(abs(x) for x in baseline_scalars) * 1e-7, 1e-10)
    evaluated = []
    for pair in pairs:
        step, plus, minus = pair["step"], pair["plus_scalar"], pair["minus_scalar"]
        if step <= 0 or not all(math.isfinite(x) for x in (step, plus, minus)):
            raise ValueError("finite positive writer finite-difference steps are required")
        derivative = (plus - minus) / (2 * step)
        evaluated.append({**pair, "finite_difference": derivative,
                          "eligible_without_autograd": abs(plus - minus) > 20 * floor
                          and abs(derivative) > 10 * ATOL})
    selected = None
    for left, right in zip(evaluated, evaluated[1:]):
        if (left["eligible_without_autograd"] and right["eligible_without_autograd"]
                and right["step"] > left["step"]
                and abs(right["finite_difference"] - left["finite_difference"])
                <= ATOL + RTOL * abs(left["finite_difference"])):
            selected = (left, right)
            break
    accepted = selected is not None and all(
        abs(value - row["finite_difference"]) <= ATOL + RTOL * abs(row["finite_difference"])
        for value in derivatives for row in selected)
    return {
        "accepted": accepted, "status": "accepted" if accepted else "inconclusive",
        "reason": "fixed-cotangent writer VJP agrees with independent FP32 history finite differences"
        if accepted else "no first informative stable pair agreed with both writer derivatives",
        "rtol": RTOL, "atol": ATOL, "baseline_scalars": list(baseline_scalars),
        "baseline_span": noise, "scalar_noise_floor": floor, "pairs": evaluated,
        "selected_steps": [x["step"] for x in selected] if selected else [],
        "rank_writer_directional_derivatives": list(derivatives)}


def state_scalar(state, rows, cotangent, write_blocks):
    """Validate the original stored-value boundary, including lazy alpha clocks."""
    import torch
    if (not torch.equal(state.rows, rows) or state.values.shape != cotangent.shape
            or state.values.dtype != torch.float32 or cotangent.dtype != torch.float32
            or cotangent.requires_grad or not state.values.requires_grad
            or not bool(torch.isfinite(state.values).all())
            or not bool(torch.isfinite(cotangent).all())):
        raise ValueError("fresh FP32 history must reconstruct the anchor's stored sparse rows")
    clock = getattr(state, "block_clock", None)
    if clock is None or clock.numel() != 1 or int(clock.item()) != write_blocks:
        raise ValueError("fresh history must execute every registered write from an empty owner")
    # Summation in FP64 avoids scalar cancellation; the writer graph and state
    # remain FP32. The derivative casts the fixed cotangent back to FP32.
    return (state.values.double() * cotangent.double()).sum()


def run_fixed_cotangent_writer_probe(*, writer, build_state, cotangent, rows,
                                     output, sample_id, write_blocks, anchor_receipt):
    """Call on all ranks after a single, separately ledgered native anchor.

    On the two PLE ranks, writer.grad must contain ONLY that anchor's writer
    VJP; cotangent is its already validated TP leaf gradient, for the original
    stored values. build_state() rebuilds the same actual cached H/rows history.
    Last-stage ranks pass writer/rows/cotangent=None. Original values, buffers,
    and anchor gradients are restored exactly; no optimizer or native CE runs.
    """
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    rank, first = dist.get_rank(), ps.is_pipeline_first_stage()
    output = Path(output)

    def gather(value):
        result = [None] * dist.get_world_size()
        dist.all_gather_object(result, value)
        return result

    def check(condition, message):
        errors = gather(None if condition else f"rank {rank}: {message}")
        if any(errors):
            raise ValueError("; ".join(x for x in errors if x))

    check(dist.get_world_size() == 4 and ps.get_tensor_model_parallel_world_size() == 2
          and ps.get_pipeline_model_parallel_world_size() == 2 and (writer is not None) == first,
          "writer adjoint proof requires the accepted TP2/PP2 native session")
    check(type(write_blocks) is int and write_blocks >= 2 and isinstance(sample_id, str)
          and bool(sample_id) and isinstance(anchor_receipt, dict)
          and (not first or callable(build_state)),
          "a fixed real native anchor and at least two actual writes are required")
    params = [(name, p) for name, p in writer.named_parameters() if p.requires_grad] if first else []
    buffers = list(writer.named_buffers()) if first else []
    check(not first or (bool(params) and all(p.dtype == torch.float32 for _, p in params)
                       and cotangent is not None and rows is not None
                       and cotangent.dtype == torch.float32 and not cotangent.requires_grad),
          "writer parameters and fixed native cotangent must be FP32")
    baseline = {name: p.detach().clone() for name, p in params}
    baseline_buffers = {name: value.detach().clone() for name, value in buffers}
    anchor_grad = {name: None if p.grad is None else p.grad.detach().clone() for name, p in params}
    anchor_snapshot_hash = named_tensor_hash(
        [(name, value) for name, value in anchor_grad.items() if value is not None]) if first else None
    baseline_hash = named_tensor_hash(params + buffers) if first else None
    anchor_data_hash = named_tensor_hash([("rows", rows), ("cotangent", cotangent)]) if first else None
    agreements = gather((baseline_hash, anchor_snapshot_hash, anchor_data_hash) if first else None)
    agreements = [x for x in agreements if x is not None]
    check(len(agreements) == 2 and agreements[0] == agreements[1],
          "writer/anchor/cotangent must be identical across the two PLE ranks")
    norm = math.sqrt(sum(float(p.double().square().sum()) for p in baseline.values())) if first else None
    norms = [x for x in gather(norm) if x is not None]
    check(len(norms) == 2 and norms[0] == norms[1] and math.isfinite(norms[0]) and norms[0] > 0,
          "writer parameter norm must be positive finite and replicated")
    steps = [x * norms[0] for x in RELATIVE_STEPS]
    registration = {
        "protocol": "native_writer_fixed_cotangent_adjoint_v1", "sample_id": sample_id,
        "history_write_blocks": write_blocks, "anchor_receipt": anchor_receipt,
        "writer_initial_state_sha256": agreements[0][0],
        "anchor_writer_gradient_sha256": agreements[0][1],
        "anchor_rows_cotangent_sha256": agreements[0][2], "initial_parameter_l2": norms[0],
        "relative_steps": list(RELATIVE_STEPS), "steps": steps, "rtol": RTOL, "atol": ATOL,
        "direction": "fixed complete anchor writer gradient, unit global L2 across all parameters",
        "scalar": "FP64 dot of FP32 stored writer state with fixed real native FP32 cotangent",
        "selection": "first adjacent stable informative pair; scalar-only selection",
        "native_ce_forward_passes": 0, "loss_target_tokens": 0, "optimizer_steps": 0,
        "scope": "FP32 complete writer history VJP only; not BF16 CE finite differences"}
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "registration.json", registration)
    dist.barrier()
    device = torch.device("cuda", torch.cuda.current_device())
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    tracker = get_cuda_rng_tracker()
    rng_tp = {name: value.clone() for name, value in tracker.get_states().items()}
    records, direction = [], {}

    def reset_rng():
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda, device)
        tracker.set_states({name: value.clone() for name, value in rng_tp.items()})

    def restore():
        if first:
            with torch.no_grad():
                for name, p in params:
                    p.copy_(baseline[name])
                    p.grad = None if anchor_grad[name] is None else anchor_grad[name].clone()
                for name, value in buffers:
                    value.copy_(baseline_buffers[name])
        reset_rng()

    def scalar(tag, analytic=False):
        reset_rng()
        value, gradients, history_hash = None, None, None
        if first:
            with torch.enable_grad():
                state = build_state()
                phi = state_scalar(state, rows, cotangent, write_blocks)
                value = float(phi.detach())
                history_hash = named_tensor_hash([("rows", state.rows), ("counts", state.counts),
                                                   ("clock", state.block_clock)])
                if analytic:
                    raw = torch.autograd.grad(phi, [p for _, p in params], allow_unused=True)
                    gradients = dict(zip((name for name, _ in params), raw))
        rank_values = [x for x in gather(value) if x is not None]
        check(len(rank_values) == 2 and all(math.isfinite(x) for x in rank_values)
              and rank_values[0] == rank_values[1], "TP writer scalar copies disagree")
        hashes = [x for x in gather(history_hash) if x is not None]
        check(len(hashes) == 2 and len(set(hashes)) == 1,
              "history row/count/clock metadata differ across TP")
        records.append({"tag": tag, "scalar": rank_values[0], "history_metadata_sha256": hashes[0]})
        return rank_values[0], gradients

    def perturb(step):
        if first:
            with torch.no_grad():
                for name, p in params:
                    p.copy_(baseline[name] + step * direction[name])
                for name, value in buffers:
                    if not torch.equal(value, baseline_buffers[name]):
                        raise ValueError("history builder modified a writer buffer")

    def finish(result):
        restore()
        check(not first or named_tensor_hash(params + buffers) == baseline_hash,
              "writer parameter/buffer restoration failed")
        check(not first or named_tensor_hash(
            [(name, p.grad) for name, p in params if p.grad is not None]) == anchor_snapshot_hash,
            "native anchor gradients were not restored exactly")
        result.update({
            "protocol": registration["protocol"], "complete": True,
            "registration_sha256": canonical_hash(registration), "sample_id": sample_id,
            "history_write_blocks": write_blocks, "scalar_evaluations": len(records),
            "records": records, "loss_target_tokens": 0, "native_ce_forward_passes": 0,
            "optimizer_steps": 0, "writer_and_anchor_gradients_restored_exactly": True,
            "scope": registration["scope"]})
        if rank == 0:
            write_json(output / "COMPLETE.json", result)
        dist.barrier()
        return result

    try:
        analytic_scalar, rebuilt = scalar("analytic_phi", analytic=True)
        local_check = None
        if first:
            mismatch = []
            numerator, denominator, gradient_norm_sq = 0., 0., 0.
            for name, p in params:
                expected = anchor_grad[name]
                actual = rebuilt[name]
                if (expected is None) != (actual is None):
                    mismatch.append(name)
                    continue
                if expected is None:
                    continue
                if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
                    mismatch.append(name)
                    continue
                if not torch.allclose(actual, expected, rtol=2e-5, atol=2e-7):
                    mismatch.append(name)
                numerator += float((actual.double() - expected.double()).square().sum())
                denominator += float(expected.double().square().sum())
                gradient_norm_sq += float(expected.double().square().sum())
            local_check = {"mismatched_parameters": mismatch,
                           "relative_l2_error": math.sqrt(numerator / max(denominator, 1e-30)),
                           "anchor_gradient_l2": math.sqrt(gradient_norm_sq)}
        gradient_checks = [x for x in gather(local_check) if x is not None]
        if any(x["mismatched_parameters"] or not math.isfinite(x["anchor_gradient_l2"])
               or x["anchor_gradient_l2"] <= 0 for x in gradient_checks):
            return finish({"accepted": False, "status": "inconclusive",
                           "reason": "fresh fixed-cotangent VJP differs from the native anchor writer gradient",
                           "gradient_checks": gradient_checks})
        norm_grad = gradient_checks[0]["anchor_gradient_l2"]
        if first:
            direction = {name: (torch.zeros_like(p) if anchor_grad[name] is None
                                else anchor_grad[name] / norm_grad) for name, p in params}
        direction_hash = named_tensor_hash(list(direction.items())) if first else None
        direction_hashes = [x for x in gather(direction_hash) if x is not None]
        check(len(set(direction_hashes)) == 1, "fixed writer direction differs across replicas")
        derivative = sum(float((anchor_grad[name].double() * direction[name].double()).sum())
                         for name, _ in params if anchor_grad[name] is not None) if first else None
        derivatives = [x for x in gather(derivative) if x is not None]
        if first and ps.get_tensor_model_parallel_rank() == 0:
            torch.save({"parameters": {k: v.cpu() for k, v in baseline.items()},
                        "buffers": {k: v.cpu() for k, v in baseline_buffers.items()},
                        "rows": rows.cpu(), "cotangent": cotangent.cpu(),
                        "direction": {k: v.cpu() for k, v in direction.items()},
                        "anchor_gradients": {k: None if v is None else v.cpu()
                                             for k, v in anchor_grad.items()}},
                       output / "writer-adjoint-inputs.pt")
        before, _ = scalar("baseline_before")
        pairs = []
        for index, step in enumerate(steps):
            perturb(step)
            plus, _ = scalar(f"step_{index}_plus")
            perturb(-step)
            minus, _ = scalar(f"step_{index}_minus")
            pairs.append({"step": step, "plus_scalar": plus, "minus_scalar": minus})
        perturb(0.)
        after, _ = scalar("baseline_after")
        result = evaluate_adjoint_series(pairs, derivatives, [analytic_scalar, before, after])
        result.update({"gradient_checks": gradient_checks, "direction_sha256": direction_hashes[0]})
        return finish(result)
    finally:
        restore()

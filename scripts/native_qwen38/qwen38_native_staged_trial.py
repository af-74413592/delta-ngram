"""Three paired EP4 groups: original B3 waves versus four-stage H12/Q4.

All trial state stays in memory. Each pair starts from identical writer,
AdamW and rank RNG state; the reference update advances the next common
starting point. Every reference and staged CE dispatch is charged.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from qwen38_native_norm_probe import canonical_hash, file_info, require, write_json
from qwen38_native_ep4_trial import assign_trajectories, optimizer_hash, prepare as prepare_source
from qwen38_native_train_batched import restore_checkpoint_state, restore_rng_state, validate_frame_plan


def global_frame_batches(counts, limit):
    """Match ranks at each call while shrinking the last global frame batch."""
    require(len(counts) == 4 and all(type(count) is int and count >= 0 for count in counts)
            and type(limit) is int and limit > 0, "invalid global frame batch counts")
    maximum = max(counts)
    for start in range(0, maximum, limit):
        size = min(limit, maximum - start)
        yield start, size, [max(0, min(size, count - start)) for count in counts]


def tensor_comparison(reference, candidate):
    """Global-vector comparison without unstable coordinatewise relative error."""
    import torch
    reference, candidate = reference.detach().double().cpu(), candidate.detach().double().cpu()
    require(reference.shape == candidate.shape and bool(reference.isfinite().all())
            and bool(candidate.isfinite().all()), "comparison requires finite matching tensors")
    delta = candidate - reference
    ref_norm, cand_norm = float(reference.norm()), float(candidate.norm())
    error_norm = float(delta.norm())
    cosine = (float(torch.dot(reference.reshape(-1), candidate.reshape(-1))) / (ref_norm * cand_norm)
              if ref_norm and cand_norm else (1. if ref_norm == cand_norm == 0 else 0.))
    return {"reference_l2": ref_norm, "candidate_l2": cand_norm,
            "difference_l2": error_norm,
            "relative_l2": error_norm / ref_norm if ref_norm else (0. if error_norm == 0 else None),
            "cosine": max(-1., min(1., cosine)),
            "max_abs": float(delta.abs().max()) if delta.numel() else 0.}


def clone_cpu(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_cpu(item) for item in value)
    return copy.deepcopy(value)


def compare_results(reference, candidate, initial_writer):
    import torch
    keys = sorted(reference["gradients"])
    require(keys == sorted(candidate["gradients"]), "paired writer gradient presence differs")
    gradient = tensor_comparison(torch.cat([reference["gradients"][key].reshape(-1) for key in keys]),
                                 torch.cat([candidate["gradients"][key].reshape(-1) for key in keys]))
    modules = {}
    for module in sorted({key.split(".", 1)[0] for key in keys}):
        selected = [key for key in keys if key.split(".", 1)[0] == module]
        modules[module] = tensor_comparison(
            torch.cat([reference["gradients"][key].reshape(-1) for key in selected]),
            torch.cat([candidate["gradients"][key].reshape(-1) for key in selected]))
    parameter_keys = sorted(reference["parameters"])
    reference_update = torch.cat([
        (reference["parameters"][key] - initial_writer[key]).reshape(-1) for key in parameter_keys])
    candidate_update = torch.cat([
        (candidate["parameters"][key] - initial_writer[key]).reshape(-1) for key in parameter_keys])
    update = tensor_comparison(reference_update, candidate_update)
    nll_difference = abs(candidate["mean_nll"] - reference["mean_nll"])
    nll_relative = nll_difference / max(abs(reference["mean_nll"]), 1e-30)
    loss_pass = nll_difference <= .001 or nll_relative <= .001
    gradient_pass = (gradient["relative_l2"] is not None
                     and gradient["relative_l2"] <= .02 and gradient["cosine"] >= .999)
    return {
        "reference_mean_nll": reference["mean_nll"], "staged_mean_nll": candidate["mean_nll"],
        "mean_nll_absolute_difference": nll_difference, "mean_nll_relative_difference": nll_relative,
        "preclip_writer_gradient": gradient, "preclip_gradient_by_module": modules,
        "AdamW_parameter_update": update,
        "criteria": {"loss_absolute_or_relative_max": .001, "gradient_relative_l2_max": .02,
                     "gradient_cosine_min": .999},
        "loss_within_tolerance": loss_pass, "gradient_within_tolerance": gradient_pass,
        "within_registered_tolerances": loss_pass and gradient_pass,
        "scope": "paired finite-run numerical closeness; not bitwise equality or a general proof",
    }


def prepare(options):
    source, prior, manifest = prepare_source(options)
    require(options.query_batch in (3, 4, 6) and options.evidence_batch in (3, 4, 6, 12),
            "unsupported bounded trial batch")
    for path in (Path(__file__).resolve(),
                 Path(__file__).with_name("qwen38_native_staged_schedule.py"),
                 options.delta_repo / "src" / "delta_ngram" / "qwen38_staged_batching.py"):
        manifest["code"][path.relative_to(options.delta_repo).as_posix()] = file_info(path)
    manifest.update({
        "protocol": "qwen38_native_four_stage_paired_trial_v1", "mode": options.mode,
        "reference_local_batch": 3, "staged_query_local_batch": options.query_batch,
        "staged_evidence_local_batch": options.evidence_batch,
        "stages": ["fresh_native_evidence_H_to_CPU", "writer_histories_and_immutable_query_snapshots",
                   "batched_query_forward_backward", "joint_trajectory_VJP_then_DP_SUM_clip_AdamW"],
        "staged_evidence_output": "stop_at_native_LM_head_input; omit_unused_evidence_logits",
        "performance_difference_includes": ["four_stage_batching", "omitted_unused_evidence_LM_head"],
        "pair_initialization": "same_writer_AdamW_all_rank_RNG_for_each_group",
        "next_group_state": "reference_update" if options.mode == "compare" else "staged_update",
        "loss_comparison": "absolute_mean_NLL<=0.001 OR relative_mean_NLL<=0.001",
        "gradient_comparison": "global_preclip_relative_L2<=0.02 AND cosine>=0.999",
        "comparison_failure": "report_all_fixed_groups; no automatic extra trials",
        "tail_batches": "min(configured_limit, global_max_tasks - start); same_actual_B_on_all_ranks",
        "warmup": "log_actual_batch_shapes_and_first_use_in_each_group; later_groups_may_use_new_shapes",
        "timing": "CUDA_synchronize_at_phase_boundaries; max_rank_elapsed",
        "trial_training_state_saved": False, "trial_state_discarded_after_run": True,
    })
    return source, prior, manifest


def execute(options, source, prior, manifest):
    from qwen38_native_ep4_session import run_ep4_session
    rank = int(os.environ.get("RANK", "-1"))
    require(0 <= rank < 4 and int(os.environ.get("WORLD_SIZE", "0")) == 4, "use four torchrun ranks")
    initial_budget = {
        "protocol": "qwen38_native_performance_optimizer_ledger_v1",
        "prior_optimizer_target_accesses": prior["total_optimizer_target_accesses"],
        "prior_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"],
        "trial_optimizer_targets_dispatched": 0, "trial_optimizer_targets_completed": 0,
        "total_optimizer_target_accesses": prior["total_optimizer_target_accesses"],
        "total_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"],
        "trial_diagnostic_target_accesses": 0,
        "failed_or_partial_dispatches_are_not_refunded": True, "formal_model_progress_unchanged": True,
    }
    marker = options.output / "input_manifest.json"
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        write_json(marker, manifest)
        write_json(options.output / "budget_ledger.json", initial_budget)
    else:
        deadline = time.monotonic() + 300
        while not marker.is_file():
            require(time.monotonic() < deadline, "rank0 manifest creation timed out")
            time.sleep(.1)
    options.tokens = 640

    def run(session):
        import numpy as np
        import torch
        import torch.distributed as dist
        from megatron.core import parallel_state as ps
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        from delta_ngram.alpha_memory import LazyDecayState
        from delta_ngram.qwen38_native import make_qwen38_writer, write_native_block
        from delta_ngram.qwen38_precision import Qwen38SplitPLEPrecision
        from delta_ngram.qwen38_probe import tensor_state_hash
        from delta_ngram.qwen38_training import NativeTrajectoryGradients, writer_evidence_features
        from delta_ngram.qwen38_train_data import NativeFullTrainingData
        from delta_ngram.qwen38_train_batching import crop_evidence_lanes
        from delta_ngram.qwen38_ep4_batching import (
            collate_ep4_frames, pack_ep4_query_tickets, SingleRankGradientPolicy, sum_ep4_writer_gradients)
        from delta_ngram.qwen38_staged_batching import (
            collate_staged_frames, pack_staged_query_tickets, materialize_staged_snapshot,
            release_staged_reader_leaf)
        from qwen38_native_bounded_checks import BoundedStability
        from qwen38_native_dispatch import native_hc_dispatch
        from qwen38_native_router import native_router_dispatch
        from qwen38_native_ep4_session import runtime_identity
        import qwen38_native_ep4_schedule as reference_schedule
        import qwen38_native_staged_schedule as staged_schedule

        device, started = session.device, time.monotonic()
        dispatched = completed = 0
        branch_dispatches = {"reference": 0, "staged": 0}
        input_tokens = {"reference": {"query": 0, "evidence": 0},
                        "staged": {"query": 0, "evidence": 0}}
        rows = []
        seen_shapes, group_shapes = set(), []

        def shape_event(branch, kind, batch):
            key = (branch, kind, batch, 640)
            first_use = key not in seen_shapes
            seen_shapes.add(key)
            value = {"branch": branch, "kind": kind, "local_batch": batch,
                     "sequence_length": 640, "first_use_in_this_session": first_use}
            if first_use:
                group_shapes.append(value)
            return value

        def gather(value):
            result = [None] * 4
            dist.all_gather_object(result, value)
            return result

        def agree(value, message):
            copies = gather(value)
            require(all(item == copies[0] for item in copies), message)
            return copies[0]

        def budget():
            return {**initial_budget, "trial_optimizer_targets_dispatched": dispatched,
                    "trial_optimizer_targets_completed": completed,
                    "total_optimizer_target_accesses": prior["total_optimizer_target_accesses"] + dispatched,
                    "optimizer_dispatches_by_branch": dict(branch_dispatches),
                    "global_real_input_tokens_by_branch": copy.deepcopy(input_tokens)}

        def save_budget():
            if rank == 0:
                write_json(options.output / "budget_ledger.json", budget())

        def reserve(branch, targets, real_inputs):
            nonlocal dispatched
            require(prior["total_optimizer_target_accesses"] + dispatched + targets <= 100000000,
                    "trial exceeds the shared optimizer target budget")
            dispatched += targets
            branch_dispatches[branch] += targets
            input_tokens[branch]["query"] += real_inputs
            save_budget()

        def rng_state():
            return {"rank": rank, "python": random.getstate(), "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state(device).cpu(),
                    "tensor_parallel": clone_cpu(get_cuda_rng_tracker().get_states())}

        def timed(label, function):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            begin = time.monotonic()
            result = function()
            torch.cuda.synchronize(device)
            measured = gather({
                "rank": rank, "seconds": time.monotonic() - begin,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "end_allocated_bytes": torch.cuda.memory_allocated(device)})
            return result, {"phase": label, "seconds_max_rank": max(item["seconds"] for item in measured),
                            "peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in measured),
                            "ranks": measured}

        with ExitStack() as scope:
            try:
                hc = scope.enter_context(native_hc_dispatch(unwrap=True))
                router = scope.enter_context(native_router_dispatch(session.raw))
                precision = scope.enter_context(Qwen38SplitPLEPrecision(session.adapter))
                runtime = {"native": runtime_identity(session), "HC": hc.identity, "router": router["identity"],
                           "precision": precision.identity(), "code_sha256": canonical_hash(manifest["code"]),
                           "actual_local_frame_batches": {"reference_query_and_evidence": 3,
                                                          "staged_query": options.query_batch,
                                                          "staged_evidence": options.evidence_batch},
                           "staged_evidence_omits_LM_head_and_logits": True}
                runtime_hash = agree(canonical_hash(runtime), "EP4 runtime identity differs across ranks")
                policy = SingleRankGradientPolicy(
                    runtime_hash, tensor_parallel_world_size=ps.get_tensor_model_parallel_world_size())
                if rank == 0:
                    write_json(options.output / "runtime_identity.json", runtime)
                data = NativeFullTrainingData(
                    options.cache_root,
                    expected_complete_sha256=manifest["cache_complete"]["sha256"],
                    expected_manifest_sha256=manifest["cache_manifest"]["sha256"])
                require(data.identity == source["data_identity"], "source corpus identity changed")
                sampler = data.sampler(seed=17)
                writer, alpha = make_qwen38_writer(alpha=1.)
                writer, alpha = writer.to(device).float().train(), alpha.to(device)
                optimizer = torch.optim.AdamW(writer.parameters(), lr=.0003, betas=(.9, .999),
                                              eps=1e-8, weight_decay=0., foreach=False, fused=False)
                stability = BoundedStability(hard_stop_gradient_spikes=False, hard_stop_beta_saturation=False)
                source_step, source_targets = restore_checkpoint_state(
                    source, writer=writer, optimizer=optimizer, sampler=sampler, stability=stability)
                restore_rng_state(next(item for item in source["rng_by_rank"] if item["rank"] == rank),
                                  rank=rank, device=device, tensor_parallel_tracker=get_cuda_rng_tracker())
                agree(tensor_state_hash(writer.state_dict()), "restored writer replicas differ")
                agree(optimizer_hash(optimizer), "restored AdamW replicas differ")

                def new_local(assigned, frame_plans, trajectories, group_targets):
                    return [{"index": index, "plan": frame_plans[index],
                             "state": LazyDecayState.from_batch(session.layout.empty_state(1, device)),
                             "bridge": NativeTrajectoryGradients(
                                 total_loss_tokens=group_targets, expected_loss_tokens=trajectories[index].loss_tokens,
                                 policy=policy, runtime_identity_sha256=runtime_hash),
                             "loss_sum": 0., "loss_targets": 0} for index in assigned]

                def write(item, block):
                    hidden = writer_evidence_features(block["H_new"].to(device), mode="rmsnorm", eps=1e-6)
                    item["state"] = write_native_block(
                        writer, item["state"], hidden, block["physical_rows_new"].to(device),
                        torch.ones(hidden.shape[:2], dtype=torch.bool, device=device), alpha)
                    require(bool(item["state"].values.isfinite().all()), "nonfinite writer memory")

                def record(item, ticket, loss, targets, group_targets):
                    require(math.isfinite(loss), "nonfinite native query loss")
                    item["bridge"].record_query(ticket, loss_sum=loss, loss_tokens=targets,
                                                reader_loss_divisor=group_targets)
                    item["loss_sum"] += loss
                    item["loss_targets"] += targets

                def reference_forward(local, frame_plans, group_targets):
                    for wave in range(max(map(len, frame_plans))):
                        for kind in ("query", "evidence"):
                            global_frames = [plan[wave][kind] for plan in frame_plans if wave < len(plan)]
                            if not any(frame is not None for frame in global_frames):
                                continue
                            shape_event("reference", kind, 3)
                            lane_frames = [item["plan"][wave][kind] if wave < len(item["plan"]) else None
                                           for item in local]
                            mb = collate_ep4_frames(lane_frames, kind=kind, device=device)
                            real_inputs = sum(sum(frame["encoder_valid_mask"]) for frame in global_frames
                                              if frame is not None)
                            if kind == "query":
                                reserve("reference", sum(frame["loss_target_tokens"] for frame in global_frames
                                                         if frame is not None), real_inputs)
                                tickets = [item["bridge"].prepare_query(materialize_staged_snapshot(item["state"]))
                                           if frame is not None else None for item, frame in zip(local, lane_frames)]
                                packed = pack_ep4_query_tickets(
                                    tickets, row_capacity=session.layout.row_capacity, device=device)
                                mb["reader_state"] = packed.reader_state
                                result = reference_schedule.schedule_query_microbatches(
                                    session, [mb], group_total_target_tokens=group_targets)["microbatches"][0]
                                require(tuple(result["loss_target_tokens_by_lane"]) == mb["loss_target_tokens_by_lane"],
                                        "reference target counts differ")
                                packed.scatter_cotangent()
                                for lane, (item, frame, ticket) in enumerate(zip(local, lane_frames, tickets)):
                                    if frame is not None:
                                        record(item, ticket, result["loss_sum_by_lane"][lane],
                                               frame["loss_target_tokens"], group_targets)
                                del tickets, packed, result
                            else:
                                input_tokens["reference"]["evidence"] += real_inputs
                                result = reference_schedule.collect_evidence_microbatches(
                                    session, [mb])["microbatches"][0]
                                blocks = crop_evidence_lanes(mb, result["hidden"], result["physical_rows"])
                                for item, block in zip(local, blocks):
                                    if block is not None:
                                        write(item, block)
                                del result, blocks
                            del mb
                        for item in local:
                            if wave + 1 == len(item["plan"]):
                                item["bridge"].backward_writer()
                                item["bridge"] = item["state"] = None

                def staged_forward(local, frame_plans, group_targets):
                    phases = []
                    cache, queries = {}, []
                    evidence_shapes, query_shapes = [], []
                    def evidence_phase():
                        tasks = [(item["index"], ordinal, frame["evidence"])
                                 for item in local for ordinal, frame in enumerate(item["plan"])
                                 if frame["evidence"] is not None]
                        counts = gather(len(tasks))
                        for start, batch_size, active_counts in global_frame_batches(counts, options.evidence_batch):
                            active = tasks[start:start + batch_size]
                            evidence_shapes.append({**shape_event("staged", "evidence", batch_size),
                                                    "real_frames_by_rank": active_counts})
                            frames = [task[2] for task in active] + [None] * (batch_size - len(active))
                            mb = collate_staged_frames(frames, kind="evidence", device=device)
                            input_tokens["staged"]["evidence"] += sum(gather(mb["real_input_tokens"]))
                            result = staged_schedule.collect_evidence_microbatches(
                                session, [mb])["microbatches"][0]
                            blocks = crop_evidence_lanes(mb, result["hidden"], result["physical_rows"])
                            for task, block in zip(active, blocks):
                                cache[task[:2]] = block
                    _, phase = timed("1_native_evidence_H_to_CPU", evidence_phase)
                    phase["cached_real_evidence_frames_by_rank"] = gather(len(cache))
                    phase["local_batch"] = options.evidence_batch
                    phase["local_batch_is_upper_limit"] = True
                    phase["actual_batch_shapes"] = evidence_shapes
                    phase["native_H_source"] = "LM_head_input_after_final_HC"
                    phase["LM_head_and_unused_logits_executed"] = False
                    phases.append(phase)

                    def writer_phase():
                        for item in local:
                            for ordinal, frame in enumerate(item["plan"]):
                                if frame["query"] is not None:
                                    queries.append({"owner": item, "frame": frame["query"],
                                                    "snapshot": item["state"]})
                                if frame["evidence"] is not None:
                                    write(item, cache.pop((item["index"], ordinal)))
                            item["state"] = None
                        require(not cache, "unconsumed cached evidence")
                    _, phase = timed("2_writer_histories_and_query_snapshots", writer_phase)
                    phase["queries_by_rank"] = gather(len(queries))
                    phases.append(phase)

                    def query_phase():
                        counts = gather(len(queries))
                        for start, batch_size, active_counts in global_frame_batches(counts, options.query_batch):
                            active = queries[start:start + batch_size]
                            query_shapes.append({**shape_event("staged", "query", batch_size),
                                                 "real_frames_by_rank": active_counts})
                            frames = [task["frame"] for task in active] + [None] * (batch_size - len(active))
                            mb = collate_staged_frames(frames, kind="query", device=device)
                            costs = gather((mb["loss_target_tokens"], mb["real_input_tokens"]))
                            reserve("staged", sum(item[0] for item in costs), sum(item[1] for item in costs))
                            tickets = [task["owner"]["bridge"].prepare_query(
                                materialize_staged_snapshot(task["snapshot"])) for task in active]
                            tickets += [None] * (batch_size - len(tickets))
                            packed = pack_staged_query_tickets(
                                tickets, row_capacity=session.layout.row_capacity, device=device)
                            mb["reader_state"] = packed.reader_state
                            result = staged_schedule.schedule_query_microbatches(
                                session, [mb], group_total_target_tokens=group_targets)["microbatches"][0]
                            require(tuple(result["loss_target_tokens_by_lane"]) == mb["loss_target_tokens_by_lane"],
                                    "staged target counts differ")
                            packed.scatter_cotangent()
                            for lane, task in enumerate(active):
                                record(task["owner"], tickets[lane], result["loss_sum_by_lane"][lane],
                                       task["frame"]["loss_target_tokens"], group_targets)
                                release_staged_reader_leaf(tickets[lane])
                                task["snapshot"] = None
                            del packed, tickets, result, mb
                    _, phase = timed("3_batched_query_forward_backward", query_phase)
                    phase["local_batch"] = options.query_batch
                    phase["local_batch_is_upper_limit"] = True
                    phase["actual_batch_shapes"] = query_shapes
                    phase["LM_head_and_native_CE_executed"] = True
                    phases.append(phase)
                    queries.clear()
                    return phases

                def finish(local, group_targets, *, writer_vjp):
                    nonlocal completed
                    if writer_vjp:
                        for item in local:
                            item["bridge"].backward_writer()
                            item["bridge"] = item["state"] = None
                    require(all(item["loss_targets"] == item["bridge_expected_targets"]
                                for item in local), "trajectory target accounting differs")
                    reduction = sum_ep4_writer_gradients(writer, group=dist.group.WORLD)
                    norm = reduction["global_gradient_l2"]
                    require(math.isfinite(norm) and norm > 0, "zero or nonfinite summed writer gradient")
                    gradients = {key: value.grad.detach().cpu().clone() for key, value in writer.named_parameters()
                                 if value.grad is not None} if rank == 0 else {}
                    agree(tensor_state_hash({key: value.grad for key, value in writer.named_parameters()
                                             if value.grad is not None}), "DP-SUM writer gradients differ")
                    torch.nn.utils.clip_grad_norm_(writer.parameters(), 1., error_if_nonfinite=True, foreach=False)
                    optimizer.step()
                    require(all(bool(value.isfinite().all()) for value in writer.parameters()),
                            "nonfinite writer after update")
                    signature = agree(tensor_state_hash(writer.state_dict()), "writer replicas differ after AdamW")
                    adam_signature = agree(optimizer_hash(optimizer), "AdamW replicas differ after update")
                    session.assert_frozen()
                    loss_sum = math.fsum(gather(math.fsum(item["loss_sum"] for item in local)))
                    completed += group_targets
                    require(completed == dispatched, "completed branch did not cover every query dispatch")
                    save_budget()
                    return {"mean_nll": loss_sum / group_targets, "loss_sum": loss_sum,
                            "raw_gradient_l2": norm, "writer_sha256": signature,
                            "optimizer_sha256": adam_signature, "gradients": gradients,
                            "parameters": {key: value.detach().cpu().clone() for key, value in writer.named_parameters()}
                                          if rank == 0 else {}}

                for group_index in range(1, options.max_groups + 1):
                    group_shapes = []
                    group = sampler.next_batch(12)
                    require(len(group) == 12, "source sampler could not supply 12 complete trajectories")
                    frame_plans = [data.frames(item) for item in group]
                    for item, plan in zip(group, frame_plans):
                        validate_frame_plan(plan, item.loss_tokens)
                    group_targets = sum(item.loss_tokens for item in group)
                    branches = ("reference", "staged") if options.mode == "compare" else ("staged",)
                    require(prior["total_optimizer_target_accesses"] + dispatched
                            + len(branches) * group_targets <= 100000000, "paired group exceeds shared budget")
                    identity = [{"id": item.document_id, "group_id": item.group_id, "source": item.source,
                                 "blocks": item.blocks, "start": item.start, "loss_tokens": item.loss_tokens}
                                for item in group]
                    assignments = assign_trajectories(frame_plans)
                    agree(canonical_hash({"group": identity, "assignments": assignments}),
                          "ranks selected different paired groups")
                    common = {"writer": clone_cpu(writer.state_dict()), "optimizer": clone_cpu(optimizer.state_dict()),
                              "rng": rng_state()}
                    results, measured, reference_next = {}, {}, None
                    for branch in branches:
                        writer.load_state_dict(common["writer"], strict=True)
                        # AdamW's CPU step tensors can be retained by
                        # load_state_dict. Clone on every load so the reference
                        # update cannot mutate the staged branch's initial state.
                        optimizer.load_state_dict(clone_cpu(common["optimizer"]))
                        restore_rng_state(common["rng"], rank=rank, device=device,
                                          tensor_parallel_tracker=get_cuda_rng_tracker())
                        optimizer.zero_grad(set_to_none=True)
                        local = new_local(assignments[rank], frame_plans, group, group_targets)
                        for item in local:
                            item["bridge_expected_targets"] = group[item["index"]].loss_tokens
                        torch.cuda.synchronize(device)
                        branch_started = time.monotonic()
                        if branch == "reference":
                            _, computation = timed("reference_interleaved_H_write_query_VJP",
                                                   lambda: reference_forward(local, frame_plans, group_targets))
                            computation["query_and_evidence_local_batch"] = 3
                            computation["evidence_LM_head_and_unused_logits_executed"] = True
                            result, update = timed("reference_DP_SUM_clip_AdamW",
                                                   lambda: finish(local, group_targets, writer_vjp=False))
                            phases = [computation, update]
                        else:
                            phases = staged_forward(local, frame_plans, group_targets)
                            result, phase = timed("4_joint_writer_VJP_DP_SUM_clip_AdamW",
                                                  lambda: finish(local, group_targets, writer_vjp=True))
                            phases.append(phase)
                        torch.cuda.synchronize(device)
                        measured[branch] = {
                            "seconds_max_rank": max(gather(time.monotonic() - branch_started)),
                            "peak_allocated_bytes": max(phase["peak_allocated_bytes"] for phase in phases),
                            "phases": phases}
                        if branch == "reference":
                            # Pair-management copies are outside either branch's
                            # measured compute interval.
                            reference_next = {"writer": clone_cpu(writer.state_dict()),
                                              "optimizer": clone_cpu(optimizer.state_dict()), "rng": rng_state()}
                        results[branch] = result
                        del local
                    comparison = None
                    if options.mode == "compare":
                        if rank == 0:
                            comparison = compare_results(results["reference"], results["staged"], common["writer"])
                        comparison = gather(comparison)[0]
                        writer.load_state_dict(reference_next["writer"], strict=True)
                        optimizer.load_state_dict(clone_cpu(reference_next["optimizer"]))
                        restore_rng_state(reference_next["rng"], rank=rank, device=device,
                                          tensor_parallel_tracker=get_cuda_rng_tracker())
                    row = {
                        "event": "staged_trial_group_complete", "trial_group": group_index,
                        "source_optimizer_step": source_step, "common_optimizer_step_after_group": source_step + group_index,
                        "group_loss_targets": group_targets, "source_model_targets": source_targets,
                        "first_group": group_index == 1,
                        "first_use_batch_shapes": list(group_shapes),
                        "first_use_may_include_compilation": bool(group_shapes),
                        "group_identity_sha256": canonical_hash(identity), "trajectories": identity,
                        "assignment": assignments, "runtime": measured, "comparison": comparison,
                        "branches": {branch: {key: value for key, value in result.items()
                                             if key not in ("gradients", "parameters")}
                                     for branch, result in results.items()},
                        "speedup_reference_over_staged": (measured["reference"]["seconds_max_rank"]
                                                         / measured["staged"]["seconds_max_rank"]
                                                         if options.mode == "compare" else None),
                        "budget": budget(), "training_state_saved": False,
                    }
                    rows.append(row)
                    if rank == 0:
                        with (options.output / "training.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                        write_json(options.output / "progress.json", {"status": "trial_running", "last_group": row})
                        print(json.dumps({"event": row["event"], "group": group_index,
                                          "speedup": row["speedup_reference_over_staged"],
                                          "comparison": comparison}), flush=True)
                    dist.barrier()
                    del common, reference_next, results, group, frame_plans
                if rank == 0:
                    write_json(options.output / "COMPLETE.json", {
                        "protocol": manifest["protocol"], "status": "three_fixed_groups_completed",
                        "groups": rows, "budget": budget(), "input_manifest": file_info(marker),
                        "all_pairs_within_registered_tolerances": (
                            all(row["comparison"]["within_registered_tolerances"] for row in rows)
                            if options.mode == "compare" else None),
                        "reader_unchanged": True, "formal_checkpoint_unchanged": True,
                        "training_state_saved": False, "trial_state_discarded": True,
                        "seconds": time.monotonic() - started})
                dist.barrier()
            except BaseException as error:
                save_budget()
                if rank == 0:
                    write_json(options.output / "FAILURE.json", {
                        "status": "trial_stopped", "error_type": type(error).__name__, "message": str(error),
                        "budget": budget(), "completed_paired_groups": len(rows), "automatic_retry": False,
                        "training_state_saved": False})
                raise
    try:
        run_ep4_session(options, run)
    except BaseException as error:
        failure = options.output / "FAILURE.json"
        if rank == 0 and not failure.exists():
            write_json(failure, {"status": "trial_stopped", "error_type": type(error).__name__,
                                "message": str(error), "automatic_retry": False,
                                "budget": json.loads((options.output / "budget_ledger.json").read_bytes()),
                                "training_state_saved": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("resume-checkpoint", "source-budget-ledger", "source-runtime-identity", "cache-root",
                 "model-path", "verification", "delta-repo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=3)
    parser.add_argument("--mode", choices=("compare", "staged"), default="compare")
    parser.add_argument("--query-batch", type=int, default=4)
    parser.add_argument("--evidence-batch", type=int, default=12)
    parser.add_argument("--validate", action="store_true")
    options = parser.parse_args()
    for key, value in vars(options).items():
        if isinstance(value, Path):
            setattr(options, key, value.resolve())
    sys.path.insert(0, str(options.delta_repo / "src"))
    source, prior, manifest = prepare(options)
    if options.validate:
        print(json.dumps({"status": "staged_trial_inputs_valid", "input_manifest": manifest}, indent=2))
        return
    execute(options, source, prior, manifest)


if __name__ == "__main__":
    main()

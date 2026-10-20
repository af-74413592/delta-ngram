"""Three-group EP4 stability trial cloned from a completed native checkpoint.

The formal checkpoint is read-only. Each selected group still has 12 whole
trajectories: four EP/DP ranks own three each, normalize by the global target
count, sum writer gradients once, clip, and update identical writer replicas.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from qwen38_native_norm_probe import canonical_hash, digest, file_info, require, write_json
from qwen38_native_train_batched import restore_checkpoint_state, restore_rng_state, validate_frame_plan


def assign_trajectories(frame_plans):
    """Deterministic longest-first assignment, with exactly three owners/rank."""
    require(len(frame_plans) == 12, "the EP4 trial requires exactly 12 trajectories")
    ranked = sorted(range(12), key=lambda index: (-len(frame_plans[index]), index))
    assigned, loads = [[] for _ in range(4)], [0] * 4
    for index in ranked:
        target = min((rank for rank in range(4) if len(assigned[rank]) < 3),
                     key=lambda rank: (loads[rank], rank))
        assigned[target].append(index)
        loads[target] += len(frame_plans[index])
    require(sorted(index for items in assigned for index in items) == list(range(12)),
            "trajectory assignment duplicated or omitted an owner")
    return assigned


def optimizer_hash(optimizer):
    """Hash all AdamW tensors and scalar/group settings in stable order."""
    import torch
    hasher = hashlib.sha256()
    def add(value):
        if torch.is_tensor(value):
            value = value.detach().contiguous().cpu()
            hasher.update(str((str(value.dtype), tuple(value.shape))).encode())
            hasher.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
                hasher.update(repr(key).encode())
                add(value[key])
        elif isinstance(value, (tuple, list)):
            hasher.update(str(len(value)).encode())
            for item in value:
                add(item)
        else:
            hasher.update(repr(value).encode())
    add(optimizer.state_dict())
    return hasher.hexdigest()


def code_manifest(repo):
    scripts = Path(__file__).resolve().parent
    paths = [scripts / name for name in (
        "qwen38_native_ep4_trial.py", "qwen38_native_ep4_session.py",
        "qwen38_native_ep4_schedule.py", "qwen38_native_train_batched.py",
        "qwen38_native_batched_schedule.py", "qwen38_native_frozen_session.py",
        "qwen38_native_runtime_preflight.py", "qwen38_native_norm_probe.py",
        "qwen38_native_dispatch.py", "qwen38_native_router.py", "qwen38_native_bounded_checks.py")]
    paths += [repo / "src" / "delta_ngram" / name for name in (
        "qwen38_ep4_batching.py", "qwen38_train_batching.py", "qwen38_train_data.py",
        "qwen38_training.py", "qwen38_native.py", "qwen38_adapter.py", "qwen38_precision.py",
        "qwen38_writer.py", "qwen38_probe.py", "alpha_memory.py", "batching.py", "state.py",
        "operators.py", "raw_corpora.py", "data/raw_text.py", "data/event_streams.py")]
    return {path.relative_to(repo).as_posix(): file_info(path) for path in paths}


def prepare(options):
    import torch
    require(options.max_groups == 3, "this quick trial is limited to exactly three groups")
    source = torch.load(options.resume_checkpoint, map_location="cpu", weights_only=False)
    prior = json.loads(options.source_budget_ledger.read_bytes())
    runtime = json.loads(options.source_runtime_identity.read_bytes())
    require(source.get("trajectory_boundary") is True
            and source.get("memory_reset_at_next_trajectory") is True
            and source["arm"] == "alpha100_rmsnorm" and source["alpha"] == 1.,
            "trial needs the completed alpha=1 checkpoint")
    require(source["sampler"]["total_loss_tokens"] == source["completed_loss_targets"]
            and source["stability"]["completed_steps"] == source["optimizer_step"],
            "checkpoint model, sampler and optimizer counters differ")
    require(canonical_hash(runtime) == source["runtime_identity_sha256"],
            "source runtime identity differs from the checkpoint")
    require(prior["total_optimizer_target_accesses"] >= source["budget"]["total_optimizer_target_accesses"]
            and prior["total_diagnostic_target_accesses"]
            == source["budget"]["total_diagnostic_target_accesses"],
            "trial cannot refund any previous target access")
    require(len(source["rng_by_rank"]) == 4
            and sorted(item["rank"] for item in source["rng_by_rank"]) == list(range(4)),
            "checkpoint needs four rank RNG snapshots")
    manifest = {
        "protocol": "qwen38_native_ep4_three_group_trial_v1",
        "source_checkpoint": file_info(options.resume_checkpoint),
        "source_budget_ledger": file_info(options.source_budget_ledger),
        "source_runtime_identity": file_info(options.source_runtime_identity),
        "source_optimizer_step": source["optimizer_step"],
        "source_completed_model_targets": source["completed_loss_targets"],
        "source_writer_sha256": source["writer_sha256"],
        "cache_complete": file_info(options.cache_root / "COMPLETE.json"),
        "cache_manifest": file_info(options.cache_root / "data_manifest.json"),
        "code": code_manifest(options.delta_repo),
        "topology": {"TP": 1, "PP": 1, "EP": 4, "ETP": 1, "DP": 4, "CP": 1},
        "groups": 3, "global_trajectories_per_group": 12, "local_reader_batch": 3,
        "loss": "each_local_loss_sum/global_group_targets; one_writer_gradient_SUM_then_clip1",
        "sampling": "next_three_whole_groups_from_source_sampler; no_reset_or_replacement",
        "assignment": "stable_longest_frame_count_first; lowest_load_rank; three_owners_each",
        "writer": "unchanged_original", "activation_recomputation": False,
        "runtime": {"router": "always_sorted_topk", "HC": "eager", "PLE": "split"},
        "optimizer": "source_AdamW_state; lr0.0003; clip1_after_DP_SUM",
        "allocator": os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")),
        "formal_checkpoint_overwritten": False, "trial_training_state_saved": False,
        "diagnostic_CE_targets": 0,
        "prior_optimizer_target_accesses": prior["total_optimizer_target_accesses"],
        "prior_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"],
        "shared_optimizer_target_cap": 100000000,
        "automatic_retry": False,
    }
    return source, prior, manifest


def execute(options, source, prior, manifest):
    from qwen38_native_ep4_session import run_ep4_session
    rank = int(os.environ.get("RANK", "-1"))
    require(0 <= rank < 4 and int(os.environ.get("WORLD_SIZE", "0")) == 4, "use four torchrun ranks")
    marker = options.output / "input_manifest.json"
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        write_json(marker, manifest)
        write_json(options.output / "budget_ledger.json", {
            "protocol": "qwen38_native_performance_optimizer_ledger_v1",
            "prior_optimizer_target_accesses": prior["total_optimizer_target_accesses"],
            "prior_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"],
            "trial_optimizer_targets_dispatched": 0, "trial_optimizer_targets_completed": 0,
            "total_optimizer_target_accesses": prior["total_optimizer_target_accesses"],
            "total_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"],
            "trial_diagnostic_target_accesses": 0,
            "failed_or_partial_dispatches_are_not_refunded": True,
            "formal_model_progress_unchanged": True,
        })
    else:
        deadline = time.monotonic() + 300
        while not marker.is_file():
            require(time.monotonic() < deadline, "rank0 manifest creation timed out")
            time.sleep(.1)
    options.tokens = 640

    def run(session):
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
        from delta_ngram.qwen38_ep4_batching import (
            collate_ep4_frames, pack_ep4_query_tickets, SingleRankGradientPolicy,
            sum_ep4_writer_gradients)
        from delta_ngram.qwen38_train_batching import crop_evidence_lanes
        from qwen38_native_bounded_checks import BoundedStability
        from qwen38_native_dispatch import native_hc_dispatch
        from qwen38_native_router import native_router_dispatch
        from qwen38_native_ep4_schedule import schedule_query_microbatches, collect_evidence_microbatches

        device, started = session.device, time.monotonic()
        dispatched = completed = query_inputs = evidence_inputs = 0
        rows = []

        def gather(value):
            result = [None] * 4
            dist.all_gather_object(result, value)
            return result

        def agree(value, message):
            copies = gather(value)
            require(all(item == copies[0] for item in copies), message)
            return copies[0]

        def budget():
            return {
                "protocol": "qwen38_native_performance_optimizer_ledger_v1",
                "prior_optimizer_target_accesses": prior["total_optimizer_target_accesses"],
                "prior_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"],
                "trial_optimizer_targets_dispatched": dispatched,
                "trial_optimizer_targets_completed": completed,
                "total_optimizer_target_accesses": prior["total_optimizer_target_accesses"] + dispatched,
                "total_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"],
                "trial_diagnostic_target_accesses": 0,
                "query_real_input_tokens": query_inputs, "evidence_real_input_tokens": evidence_inputs,
                "failed_or_partial_dispatches_are_not_refunded": True,
                "formal_model_progress_unchanged": True,
            }

        def save_budget():
            if rank == 0:
                write_json(options.output / "budget_ledger.json", budget())

        with ExitStack() as scope:
            try:
                hc = scope.enter_context(native_hc_dispatch(unwrap=True))
                router = scope.enter_context(native_router_dispatch(session.raw))
                precision = scope.enter_context(Qwen38SplitPLEPrecision(session.adapter))
                runtime = {"session": session.identity, "versions": session.versions,
                           "topology": manifest["topology"], "HC": hc.identity,
                           "router": router["identity"], "precision": precision.identity(),
                           "code_manifest_sha256": canonical_hash(manifest["code"])}
                runtime_hash = agree(canonical_hash(runtime), "EP4 runtime identity differs across ranks")
                if rank == 0:
                    write_json(options.output / "runtime_identity.json", runtime)
                policy = SingleRankGradientPolicy(
                    runtime_hash, tensor_parallel_world_size=ps.get_tensor_model_parallel_world_size())
                data = NativeFullTrainingData(
                    options.cache_root,
                    expected_complete_sha256=manifest["cache_complete"]["sha256"],
                    expected_manifest_sha256=manifest["cache_manifest"]["sha256"])
                require(data.identity == source["data_identity"], "source training corpus identity changed")
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
                save_budget()

                for group_index in range(1, options.max_groups + 1):
                    group_started = time.monotonic()
                    group = sampler.next_batch(12)
                    require(len(group) == 12, "source sampler could not supply a full 12-trajectory group")
                    frames = [data.frames(item) for item in group]
                    for item, plan in zip(group, frames):
                        validate_frame_plan(plan, item.loss_tokens)
                    group_targets = sum(item.loss_tokens for item in group)
                    require(prior["total_optimizer_target_accesses"] + dispatched + group_targets <= 100000000,
                            "trial would exceed the shared optimizer target budget")
                    identity = [{"id": item.document_id, "group_id": item.group_id, "source": item.source,
                                 "blocks": item.blocks, "start": item.start, "loss_tokens": item.loss_tokens}
                                for item in group]
                    assignments = assign_trajectories(frames)
                    agree(canonical_hash({"trajectories": identity, "assignment": assignments}),
                          "EP4 ranks selected different training groups")
                    local = [{"index": index, "plan": frames[index],
                              "state": LazyDecayState.from_batch(session.layout.empty_state(1, device)),
                              "bridge": NativeTrajectoryGradients(
                                  total_loss_tokens=group_targets, expected_loss_tokens=group[index].loss_tokens,
                                  policy=policy, runtime_identity_sha256=runtime_hash),
                              "loss_sum": 0., "loss_targets": 0} for index in assignments[rank]]
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.reset_peak_memory_stats(device)
                    beta_counts = [0, 0]
                    def gates(module, args, output):
                        beta = output.detach().float().sigmoid()
                        beta_counts[0] += int(((beta < .001) | (beta > .999)).sum())
                        beta_counts[1] += beta.numel()
                    hook = writer.gate_head[2].register_forward_hook(gates)
                    local_reports = []
                    try:
                        for wave in range(max(map(len, frames))):
                            # Every EP rank enters identical collective-bearing
                            # forward/backward calls, even with no local active lane.
                            for kind in ("query", "evidence"):
                                global_frames = [plan[wave][kind] for plan in frames if wave < len(plan)]
                                if not any(frame is not None for frame in global_frames):
                                    continue
                                lane_frames = [item["plan"][wave][kind] if wave < len(item["plan"]) else None
                                               for item in local]
                                mb = collate_ep4_frames(lane_frames, kind=kind, device=device)
                                if kind == "query":
                                    global_tokens = sum(frame["loss_target_tokens"] for frame in global_frames
                                                        if frame is not None)
                                    dispatched += global_tokens
                                    query_inputs += sum(sum(frame["encoder_valid_mask"]) for frame in global_frames
                                                        if frame is not None)
                                    save_budget()
                                    tickets = [item["bridge"].prepare_query(item["state"].memory.with_values(
                                        item["state"]._lookup_packed(item["state"].rows)[0]))
                                        if frame is not None else None for item, frame in zip(local, lane_frames)]
                                    packed = pack_ep4_query_tickets(
                                        tickets, row_capacity=session.layout.row_capacity, device=device)
                                    mb["reader_state"] = packed.reader_state
                                    result = schedule_query_microbatches(
                                        session, [mb], group_total_target_tokens=group_targets,
                                        backward=True)["microbatches"][0]
                                    require(tuple(result["loss_target_tokens_by_lane"])
                                            == mb["loss_target_tokens_by_lane"], "EP4 query target counts differ")
                                    packed.scatter_cotangent()
                                    for lane, (item, frame, ticket) in enumerate(zip(local, lane_frames, tickets)):
                                        if frame is None:
                                            continue
                                        loss = result["loss_sum_by_lane"][lane]
                                        require(math.isfinite(loss), "nonfinite EP4 query loss")
                                        item["bridge"].record_query(
                                            ticket, loss_sum=loss, loss_tokens=frame["loss_target_tokens"],
                                            reader_loss_divisor=group_targets)
                                        item["loss_sum"] += loss
                                        item["loss_targets"] += frame["loss_target_tokens"]
                                    del packed, tickets, result
                                else:
                                    evidence_inputs += sum(sum(frame["encoder_valid_mask"]) for frame in global_frames
                                                           if frame is not None)
                                    result = collect_evidence_microbatches(session, [mb])["microbatches"][0]
                                    blocks = crop_evidence_lanes(mb, result["hidden"], result["physical_rows"])
                                    for item, block in zip(local, blocks):
                                        if block is None:
                                            continue
                                        hidden = writer_evidence_features(block["H_new"].to(device),
                                                                          mode="rmsnorm", eps=1e-6)
                                        physical_rows = block["physical_rows_new"].to(device)
                                        item["state"] = write_native_block(
                                            writer, item["state"], hidden, physical_rows,
                                            torch.ones(hidden.shape[:2], dtype=torch.bool, device=device), alpha)
                                        require(bool(item["state"].values.isfinite().all()),
                                                "nonfinite EP4 writer memory")
                                    del result, blocks
                                del mb
                            for item in local:
                                if wave + 1 == len(item["plan"]):
                                    item["bridge"].backward_writer()
                                    require(item["loss_targets"] == group[item["index"]].loss_tokens,
                                            "EP4 trajectory loss accounting differs")
                                    local_reports.append({"index": item["index"], **identity[item["index"]],
                                                          "loss_sum": item["loss_sum"]})
                                    item["bridge"] = item["state"] = None
                    finally:
                        hook.remove()
                    require(len(local_reports) == 3, "EP4 rank did not complete its three trajectories")
                    gradient = sum_ep4_writer_gradients(writer, group=dist.group.WORLD)
                    params = list(writer.parameters())
                    norm = math.sqrt(sum(float(value.grad.detach().double().square().sum())
                                         for value in params if value.grad is not None))
                    require(math.isfinite(norm) and norm > 0, "nonfinite or zero globally summed writer gradient")
                    agree(tensor_state_hash({key: value.grad for key, value in writer.named_parameters()
                                             if value.grad is not None}), "summed EP4 writer gradients differ")
                    counts = torch.tensor(beta_counts, dtype=torch.long, device=device)
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                    beta_saturated, beta_events = map(int, counts.tolist())
                    require(not stability.observe_step(norm, beta_saturated, beta_events),
                            "nonfinite EP4 stability observation")
                    torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True, foreach=False)
                    optimizer.step()
                    stability.complete_step(norm)
                    require(all(bool(value.isfinite().all()) for value in writer.parameters()),
                            "nonfinite writer after EP4 update")
                    signature = agree(tensor_state_hash(writer.state_dict()), "updated EP4 writer replicas differ")
                    adam_signature = agree(optimizer_hash(optimizer), "updated EP4 AdamW replicas differ")
                    session.assert_frozen()
                    completed += group_targets
                    require(completed == dispatched, "completed EP4 groups do not account for all dispatches")
                    all_reports = sorted((item for items in gather(local_reports) for item in items),
                                         key=lambda item: item["index"])
                    loss_sum = math.fsum(item["loss_sum"] for item in all_reports)
                    memory = gather({"rank": rank, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                     "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                                     "current_allocated_bytes": torch.cuda.memory_allocated(device)})
                    row = {"event": "ep4_trial_group_complete", "trial_group": group_index,
                           "source_optimizer_step": source_step, "trial_optimizer_step": source_step + group_index,
                           "group_loss_targets": group_targets, "trial_completed_targets": completed,
                           "cloned_model_completed_targets": source_targets + completed,
                           "mean_nll": loss_sum / group_targets, "loss_sum": loss_sum,
                           "raw_gradient_l2_after_SUM": norm, "gradient_reduction": gradient,
                           "writer_sha256": signature, "optimizer_sha256": adam_signature,
                           "assignment": assignments, "trajectories": all_reports,
                           "stability": stability.last_observation, "CUDA_memory_by_rank": memory,
                           "seconds": time.monotonic() - group_started}
                    save_budget()
                    if rank == 0:
                        rows.append(row)
                        with (options.output / "training.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                        write_json(options.output / "progress.json", {"status": "trial_running",
                                   "last_group": row, "budget": budget(), "training_state_saved": False})
                        print(json.dumps({key: row[key] for key in (
                            "event", "trial_group", "mean_nll", "group_loss_targets", "seconds")}), flush=True)
                    dist.barrier()
                    del local, frames, group, params
                if rank == 0:
                    write_json(options.output / "COMPLETE.json", {
                        "protocol": manifest["protocol"], "status": "three_groups_completed",
                        "groups": rows, "budget": budget(), "input_manifest": file_info(marker),
                        "runtime_identity_sha256": runtime_hash, "reader_unchanged": True,
                        "formal_checkpoint_unchanged": True, "training_state_saved": False,
                        "seconds": time.monotonic() - started,
                        "scope": "short EP4 stability trial; not a quality or long-run stability claim"})
                dist.barrier()
            except BaseException as error:
                save_budget()
                if rank == 0:
                    write_json(options.output / "FAILURE.json", {
                        "status": "trial_stopped", "error_type": type(error).__name__, "message": str(error),
                        "budget": budget(), "completed_groups": len(rows), "automatic_retry": False})
                raise
    try:
        run_ep4_session(options, run)
    except BaseException as error:
        failure = options.output / "FAILURE.json"
        if rank == 0 and not failure.exists():
            write_json(failure, {
                "status": "trial_stopped", "error_type": type(error).__name__, "message": str(error),
                "budget": json.loads((options.output / "budget_ledger.json").read_bytes()),
                "training_state_saved": False, "automatic_retry": False,
            })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("resume-checkpoint", "source-budget-ledger", "source-runtime-identity", "cache-root",
                 "model-path", "verification", "delta-repo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=3)
    parser.add_argument("--validate", action="store_true")
    options = parser.parse_args()
    for key, value in vars(options).items():
        if isinstance(value, Path):
            setattr(options, key, value.resolve())
    sys.path.insert(0, str(options.delta_repo / "src"))
    source, prior, manifest = prepare(options)
    if options.validate:
        print(json.dumps({"status": "ep4_trial_inputs_valid", "input_manifest": manifest}, indent=2))
        return
    execute(options, source, prior, manifest)


if __name__ == "__main__":
    main()

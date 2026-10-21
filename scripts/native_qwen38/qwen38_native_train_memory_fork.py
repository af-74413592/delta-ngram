"""Independent, resumable 80M->100M memory-aware writer forks.

--validate / --dry-run are CPU-only and replay every sampler group without CE.
--execute requires four torchrun ranks; --max-groups bounds a preflight at a
complete group boundary. Use a NEW output directory for each explicit resume.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

from qwen38_native_norm_probe import canonical_hash, file_info, require
from qwen38_native_train_staged_alpha100 import (
    assign_trajectories, clone_cpu, global_frame_batches, optimizer_hash, write_json)
from qwen38_native_train_batched import restore_rng_state, validate_frame_plan
from qwen38_native_alpha100_policy import require_event_limit, state_digest
import qwen38_native_memory_fork_policy as fork


def code_manifest(repo):
    from qwen38_native_train_staged_alpha100 import code_manifest as base_manifest
    result = base_manifest(repo)
    for path in sorted((repo / "scripts" / "native_qwen38").glob("*memory*.py")):
        result[path.relative_to(repo).as_posix()] = file_info(path)
    return result


def read_checkpoint(path):
    import torch
    info = file_info(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    require(file_info(path) == info, "checkpoint changed while reading")
    return value, {"path": str(path), **info}


def prepare(options):
    """Validate immutable provenance, migration and every replay group on CPU."""
    from delta_ngram.qwen38_train_data import NativeFullTrainingData
    source, source_info = read_checkpoint(options.source_checkpoint)
    require(source_info["sha256"] == fork.SOURCE_CHECKPOINT_SHA256,
            "source80 archive differs from the registered actual checkpoint")
    saved = fork.validate_source80(source)
    original_manifest = json.loads(options.source_input_manifest.read_bytes())
    original_runtime = json.loads(options.source_runtime_identity.read_bytes())
    historical_budget = json.loads(options.source_budget_ledger.read_bytes())
    require(file_info(options.source_input_manifest)["sha256"] == source["registration_sha256"]
            and canonical_hash(original_runtime) == source["runtime_identity_sha256"]
            and canonical_hash(original_manifest["code"]) == original_manifest["code_manifest_sha256"]
            == source["code_manifest_sha256"] == original_runtime["code_sha256"],
            "mainline registration/runtime/code identity chain differs")
    for key in ("total_optimizer_target_accesses", "total_diagnostic_target_accesses"):
        require(historical_budget[key] >= source["budget"][key], "historical mainline budget was refunded")
    receipt = fork.evaluation_receipt(source, options.source_evaluation_report,
                                     options.source_evaluation_complete)
    require(file_info(options.dev64_plan) == source["dev64_plan"]
            == original_manifest["dev64_plan"], "DEV64 plan changed")
    reference, reference_info = read_checkpoint(options.reference_endpoint_checkpoint)
    require(reference_info["sha256"] == fork.REFERENCE_CHECKPOINT_SHA256,
            "reference100M checkpoint differs from the completed mainline artifact")
    require(reference["protocol"] == fork.SOURCE_PROTOCOL
            and reference["arms"][fork.ACTIVE]["optimizer_step"] == fork.END_STEP
            and reference["arms"][fork.ACTIVE]["completed_loss_targets"] == fork.END_TARGETS
            and reference["data_identity"] == source["data_identity"],
            "reference endpoint is not the completed mainline100M state")
    require(state_digest(reference["arms"][fork.CONTROL]) == state_digest(source["arms"][fork.CONTROL]),
            "mainline fixed alpha099 control changed")
    require(any(row.get("actual_model_targets") == fork.SOURCE_TARGETS
                and row.get("optimizer_step") == fork.SOURCE_STEP
                and row.get("evaluation_report_sha256") == receipt["canonical_report_sha256"]
                for row in reference["schedule_cursor"]["completed_stages"]),
            "80M DEV64 completion receipt is absent from the authenticated mainline endpoint")
    records = fork.reference_groups(options.reference_training_log)
    reference_losses = fork.reference_group_losses(options.reference_training_log)
    reference_first_loss = fork.reference_first_group_loss(options.reference_training_log)
    reference_evaluation = fork.reference_endpoint_evaluation(
        reference, options.reference_evaluation_report, options.reference_evaluation_complete)
    data = NativeFullTrainingData(
        options.cache_root, expected_complete_sha256=original_manifest["cache_complete"]["sha256"],
        expected_manifest_sha256=original_manifest["cache_manifest"]["sha256"])
    require(data.identity == source["data_identity"], "fork training corpus identity changed")
    sampler = data.sampler(seed=17)
    replay = fork.validate_replay_sampler(sampler, saved["sampler"], records,
                                        reference["arms"][fork.ACTIVE]["sampler"])
    endpoint_sampler = copy.deepcopy(reference["arms"][fork.ACTIVE]["sampler"])
    del reference
    codes = code_manifest(options.delta_repo)
    code_hash = canonical_hash(codes)
    paired_config = fork.paired_loss_policy(threshold=options.paired_loss_threshold,
        warmup_groups=options.paired_loss_warmup_groups, window_groups=options.paired_loss_window_groups,
        consecutive_windows=options.paired_loss_consecutive_windows, action=options.paired_loss_soft_stop_action)
    policy = fork.fork_policy(options.branch, options.residual_difference_baseline, paired_loss=paired_config)
    resumed, resumed_info, saved_branch = None, None, None
    ledger = fork.new_ledger()
    if options.resume_checkpoint is not None:
        require(options.resume_budget_ledger is not None, "resume requires its latest budget ledger")
        resumed, resumed_info = read_checkpoint(options.resume_checkpoint)
        ledger = fork.new_ledger(json.loads(options.resume_budget_ledger.read_bytes()))
        saved_branch = fork.validate_resume(resumed, ledger, branch=options.branch,
                                           code_hash=code_hash, replay_hash=canonical_hash(records),
                                           residual_difference_baseline=options.residual_difference_baseline,
                                           paired_loss=paired_config, reference_loss_hash=canonical_hash(reference_losses))
        require(resumed["source80_checkpoint"] == source_info,
                "resume source80 checkpoint identity changed")
        require(resumed["completed_targets"] == sum(
            row["loss_target_tokens"] for row in records[:resumed["completed_groups"]]),
            "resume targets do not match its replay prefix")
        sampler.load_state_dict(saved["sampler"])
        for row in records[:resumed["completed_groups"]]:
            sampler.next_batch(12)
        require(sampler.state_dict() == saved_branch["sampler"],
                "resume sampler differs from the mainline replay prefix")
    else:
        require(options.resume_budget_ledger is None, "resume ledger without checkpoint")
    writer, _, optimizer, monitor = fork.construct_branch(saved, options.branch, saved_branch=saved_branch,
        residual_difference_baseline=options.residual_difference_baseline)
    from delta_ngram.qwen38_probe import tensor_state_hash
    migration = {
        "branch": options.branch, "source_writer_sha256": saved["writer_sha256"],
        "residual_difference_baseline": options.residual_difference_baseline,
        "migrated_writer_sha256": tensor_state_hash(writer.state_dict()),
        "migrated_optimizer_sha256": optimizer_hash(optimizer),
        "trainable_parameters": sum(p.numel() for p in writer.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in writer.parameters() if not p.requires_grad),
        "AdamW_state": policy["optimizer"], "stability_steps": monitor.completed_steps,
        "source_sampler_sha256": canonical_hash(saved["sampler"]),
        "source_rank_RNG_sha256": state_digest(saved["rng_by_rank"]),
        "fixed_alpha099_full_state_sha256": state_digest(source["arms"][fork.CONTROL]),
        "source_pending_DEV64_receipt": receipt,
        "prior_successful_targets_after_resume_checkpoint": (
            ledger["optimizer_targets_completed"] - resumed["budget"]["optimizer_targets_completed"]
            if resumed else 0),
    }
    del writer, optimizer, monitor
    changed = {name: {"source": info, "fork": codes.get(name)}
               for name, info in original_manifest["code"].items() if codes.get(name) != info}
    manifest = {
        "protocol": fork.PROTOCOL, "policy": policy, "code": codes,
        "code_manifest_sha256": code_hash, "source80_checkpoint": source_info,
        "source_input_manifest": file_info(options.source_input_manifest),
        "source_runtime_identity": original_runtime,
        "source_budget_ledger": {"path": str(options.source_budget_ledger),
                                 **file_info(options.source_budget_ledger)},
        "historical_mainline_budget": historical_budget,
        "historical_mainline_charges_added_to_branch_ledger": False,
        "code_migration": changed, "migration": migration,
        "reference_endpoint_checkpoint": reference_info,
        "reference_training_log": file_info(options.reference_training_log),
        "reference_first_group_loss": reference_first_loss,
        "reference_group_losses": reference_losses,
        "reference_endpoint_evaluation": reference_evaluation,
        "node_checkpoint_boundary": fork.node_checkpoint_boundary(records),
        "stop_after_group_file": str(options.stop_after_group_file),
        "replay": replay, "replay_groups": records,
        "cache_complete": original_manifest["cache_complete"],
        "cache_manifest": original_manifest["cache_manifest"],
        "dev64_plan": source["dev64_plan"], "data_identity": data.identity,
        "resume_checkpoint": resumed_info, "max_groups_this_invocation": options.max_groups,
        "diagnostic_cadence": options.diagnostic_cadence,
        "run_id": canonical_hash({"output": str(options.output), "source": source_info,
                                  "resume": resumed_info, "policy": policy}),
    }
    if resumed is not None:
        require(resumed["diagnostic_cadence"] == options.diagnostic_cadence,
                "diagnostic cadence changed on resume")
    require(options.output != options.source_checkpoint.parent
            and (options.resume_checkpoint is None or options.output != options.resume_checkpoint.parent),
            "use a new independent output directory")
    if os.environ.get("RANK") in (None, "0"):
        require(not options.output.exists(), "output directory already exists")
    return {"source": source, "source_arm": saved, "resume": resumed,
            "saved_branch": saved_branch, "ledger": ledger, "manifest": manifest,
            "endpoint_sampler": endpoint_sampler, "data": data}


def comparable_reader_runtime(runtime):
    """Keep every reader source hash, topology and numeric dispatch unchanged."""
    return {key: copy.deepcopy(runtime[key]) for key in ("native", "HC", "router", "precision")}


def execute(options, prepared):
    from qwen38_native_ep4_session import run_ep4_session
    rank = int(os.environ.get("RANK", "-1"))
    require(0 <= rank < 4 and int(os.environ.get("WORLD_SIZE", "0")) == 4,
            "execute requires exactly four torchrun ranks")
    manifest, ledger = prepared["manifest"], prepared["ledger"]
    started = time.monotonic()
    activities = {key: 0. for key in ("training", "evaluation", "diagnostic", "checkpoint")}
    status, latest = "running", None
    marker = options.output / "input_manifest.json"
    def save_ledger():
        if rank == 0:
            write_json(options.output / "budget_ledger.json", ledger)
    def activity(kind, action):
        begin = time.monotonic()
        try:
            return action()
        finally:
            activities[kind] += time.monotonic() - begin
    def local_timing():
        elapsed = time.monotonic() - started
        return {"rank": rank, "run_id": manifest["run_id"], "status": status,
                "wall_seconds": elapsed, "categories_seconds": {
                    **activities, "other": max(0., elapsed - sum(activities.values()))}}
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        write_json(marker, manifest)
        write_json(options.output / "migration.json", manifest["migration"])
        save_ledger()
    else:
        deadline = time.monotonic() + 300
        while not marker.is_file():
            require(time.monotonic() < deadline, "rank0 did not create fork manifest")
            time.sleep(.1)
    options.tokens = 640

    def run(session):
        nonlocal status, latest
        import numpy as np
        import torch
        import torch.distributed as dist
        from megatron.core import parallel_state as ps
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        from delta_ngram.alpha_memory import LazyDecayState
        from delta_ngram.qwen38_memory_writer import write_memory_aware_native_block
        from delta_ngram.qwen38_memory_diagnostics import (
            measure_encoding_sensitivity, summarize_encoding_sensitivity)
        from delta_ngram.qwen38_precision import Qwen38SplitPLEPrecision
        from delta_ngram.qwen38_probe import tensor_state_hash
        from delta_ngram.qwen38_training import NativeTrajectoryGradients, writer_evidence_features
        from delta_ngram.qwen38_ep4_batching import SingleRankGradientPolicy, sum_ep4_writer_gradients
        from delta_ngram.qwen38_staged_batching import (
            collate_staged_frames, pack_staged_query_tickets, materialize_staged_snapshot,
            release_staged_reader_leaf)
        from qwen38_native_dispatch import native_hc_dispatch
        from qwen38_native_router import native_router_dispatch
        from qwen38_native_ep4_session import runtime_identity
        from qwen38_native_memory_schedule import build_causal_writer_histories
        from qwen38_native_memory_dev64_eval import evaluate_memory_dev64
        import qwen38_native_staged_schedule as schedule

        device = session.device
        resumed = prepared["resume"]
        groups = resumed["completed_groups"] if resumed else 0
        targets = resumed["completed_targets"] if resumed else 0
        phase = resumed["phase"] if resumed else "training"
        evaluation_attempt = resumed.get("evaluation_attempt", 0) if resumed else 0
        evaluation_receipt = resumed.get("evaluation_receipt") if resumed else None
        evaluation_receipts = copy.deepcopy(resumed.get("evaluation_receipts", [])) if resumed else []
        terminal_checkpoint = copy.deepcopy(resumed.get("terminal_checkpoint")) if resumed else None
        first_group_loss_comparison = copy.deepcopy(resumed.get("first_group_loss_comparison")) if resumed else None
        paired_loss_state = fork.new_paired_loss_state(resumed.get("paired_loss_state") if resumed else None)
        paired_loss_observation = copy.deepcopy(resumed.get("paired_loss_observation")) if resumed else None
        node_checkpoint = copy.deepcopy(resumed.get("node_checkpoint")) if resumed else None
        data, source_arm = prepared["data"], prepared["source_arm"]
        def gather(value):
            values = [None] * 4
            dist.all_gather_object(values, value)
            return values
        def agree(value, message):
            values = gather(value)
            require(all(item == values[0] for item in values), message)
            return values[0]
        def capture_rng():
            return {"rank": rank, "python": random.getstate(), "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state(device).cpu(),
                    "tensor_parallel": clone_cpu(get_cuda_rng_tracker().get_states())}
        def restore_rng(value):
            restore_rng_state(value, rank=rank, device=device,
                              tensor_parallel_tracker=get_cuda_rng_tracker())
        def timing():
            ranks = gather(local_timing())
            record = {"run_id": manifest["run_id"], "ranks": ranks,
                      "wall_seconds_max_rank": max(row["wall_seconds"] for row in ranks),
                      "gpu_occupied_hours": max(row["wall_seconds"] for row in ranks) * 4 / 3600,
                      "category_gpu_occupied_hours": {key: max(row["categories_seconds"][key]
                          for row in ranks) * 4 / 3600 for key in (*activities, "other")},
                      "definition": "rank_max_wall_times_four; not kernel utilization"}
            if rank == 0:
                write_json(options.output / "timing_ledger.json", {
                    "current_run": record, "source_runs": resumed.get("timing_runs", []) if resumed else []})
            return record
        def encoding_charge(row, *, category="training"):
            # Training callbacks are rank-local before collective dispatch.
            if row.get("kind", "").startswith("diagnostic"):
                category = "diagnostic"
            real = row.get("real_input_tokens", row.get("global_real_input_tokens", 0))
            padded = row.get("padded_input_tokens", row.get("global_padded_input_tokens", real))
            calls = row.get("calls", row.get("native_calls", 1))
            if "global_real_input_tokens" not in row and row.get("phase") != "evaluation":
                real, padded = map(sum, zip(*gather((real, padded))))
            require(row.get("ce_targets", 0) == 0, "no-CE encoding callback charged CE")
            for suffix, value in (("real_tokens", real), ("padded_tokens", padded), ("calls", calls)):
                fork.charge(ledger, f"{category}_encoding_{suffix}", value)
            save_ledger()
        def charge_ce(category, count):
            agree(count, "ranks disagree on CE charge")
            fork.charge(ledger, category, count)
            save_ledger()

        with ExitStack() as scope:
            hc = scope.enter_context(native_hc_dispatch(unwrap=True))
            router = scope.enter_context(native_router_dispatch(session.raw))
            precision = scope.enter_context(Qwen38SplitPLEPrecision(session.adapter))
            runtime = {"protocol": fork.PROTOCOL, "native": runtime_identity(session),
                       "HC": hc.identity, "router": router["identity"], "precision": precision.identity(),
                       "code_sha256": manifest["code_manifest_sha256"], "policy": manifest["policy"],
                       "DEV64_plan": manifest["dev64_plan"]}
            require(comparable_reader_runtime(runtime) == comparable_reader_runtime(manifest["source_runtime_identity"]),
                    "native reader/topology/HC/router/precision changed")
            runtime_hash = agree(canonical_hash(runtime), "ranks disagree on runtime identity")
            if resumed:
                require(runtime_hash == resumed["runtime_identity_sha256"], "fork resume runtime changed")
            if rank == 0:
                write_json(options.output / "runtime_identity.json", runtime)
            gradient_policy = SingleRankGradientPolicy(runtime_hash,
                tensor_parallel_world_size=ps.get_tensor_model_parallel_world_size())
            writer, alpha, optimizer, monitor = fork.construct_branch(
                source_arm, options.branch, saved_branch=prepared["saved_branch"], device=device,
                residual_difference_baseline=options.residual_difference_baseline)
            sampler = data.sampler(seed=17)
            state_source = prepared["saved_branch"] or source_arm
            sampler.load_state_dict(state_source["sampler"])
            restore_rng(next(row for row in state_source["rng_by_rank"] if row["rank"] == rank))
            parameters = [p for p in writer.parameters() if p.requires_grad]
            beta_counts = [0, 0]
            checkpoints_written = 0
            def observe_events(events):
                beta = events.beta.detach()
                beta_counts[0] += int(((beta < .001) | (beta > .999)).sum())
                beta_counts[1] += beta.numel()
            def save_checkpoint():
                nonlocal latest, checkpoints_written
                require(sampler.state_dict()["total_loss_tokens"] == fork.SOURCE_TARGETS + targets,
                        "checkpoint sampler/model target mismatch")
                writer_hash = agree(tensor_state_hash(writer.state_dict()), "writer replicas differ")
                optimizer_digest = agree(optimizer_hash(optimizer), "optimizer replicas differ")
                if options.branch == "residual":
                    require(tensor_state_hash(writer.base_writer.state_dict()) == source_arm["writer_sha256"],
                            "frozen residual base changed")
                ranks_rng = gather(capture_rng())
                if checkpoints_written == 0:
                    require(writer_hash == manifest["migration"]["migrated_writer_sha256"]
                            and optimizer_digest == manifest["migration"]["migrated_optimizer_sha256"],
                            "GPU-restored branch differs from the CPU-validated migration")
                    require(state_digest(ranks_rng) == state_digest(state_source["rng_by_rank"]),
                            "initial checkpoint did not preserve all four source rank RNG states")
                time_record = timing()
                if rank == 0:
                    branch_state = {"writer": clone_cpu(writer.state_dict()), "writer_sha256": writer_hash,
                        "residual_difference_baseline": options.residual_difference_baseline,
                        "optimizer": clone_cpu(optimizer.state_dict()), "optimizer_sha256": optimizer_digest,
                        "sampler": sampler.state_dict(), "rng_by_rank": ranks_rng,
                        "stability": fork.stability_state(monitor),
                        "model_optimizer_step": fork.SOURCE_STEP + groups,
                        "optimizer_updates_since_fork": groups,
                        "completed_loss_targets": fork.SOURCE_TARGETS + targets}
                    payload = {"protocol": fork.CHECKPOINT_PROTOCOL, "policy": manifest["policy"],
                        "branch_state": branch_state, "completed_groups": groups, "completed_targets": targets,
                        "phase": phase, "evaluation_attempt": evaluation_attempt,
                        "diagnostic_G_action": "monitor_only_never_automatic_stop",
                        "evaluation_receipt": evaluation_receipt,
                        "evaluation_receipts": evaluation_receipts, "terminal_checkpoint": terminal_checkpoint,
                        "first_group_loss_comparison": first_group_loss_comparison,
                        "paired_loss_state": paired_loss_state, "paired_loss_observation": paired_loss_observation,
                        "reference_loss_sha256": canonical_hash(manifest["reference_group_losses"]),
                        "node_checkpoint": node_checkpoint,
                        "source80_checkpoint": manifest["source80_checkpoint"],
                        "migration": manifest["migration"], "budget": copy.deepcopy(ledger),
                        "code_manifest_sha256": manifest["code_manifest_sha256"],
                        "runtime_identity_sha256": runtime_hash, "replay_sha256": canonical_hash(manifest["replay_groups"]),
                        "registration_sha256": file_info(marker)["sha256"],
                        "diagnostic_cadence": options.diagnostic_cadence,
                        "trajectory_boundary": True, "memory_reset_at_next_trajectory": True,
                        "reader_saved": False, "automatic_retry": False,
                        "timing_runs": (resumed.get("timing_runs", []) if resumed else []) + [time_record]}
                    path = options.output / "branch-latest.pt"
                    temporary = path.with_suffix(".pt.partial")
                    torch.save(payload, temporary)
                    os.replace(temporary, path)
                    latest = {"path": str(path), **file_info(path), "completed_groups": groups,
                              "completed_targets": targets, "phase": phase}
                    write_json(options.output / "checkpoint_receipt.json", latest)
                latest = gather(latest)[0]
                dist.barrier()
                checkpoints_written += 1
            def checkpoint():
                activity("checkpoint", save_checkpoint)
            def save_node_if_due():
                nonlocal node_checkpoint
                boundary = manifest["node_checkpoint_boundary"]
                if options.branch != "full" or node_checkpoint is not None or groups != boundary["completed_groups"]:
                    return
                require(fork.SOURCE_TARGETS + targets == boundary["actual_model_targets"],
                        "90M node does not match its registered complete-group boundary")
                if rank == 0:
                    path = options.output / "branch-node-090000000.pt"
                    require(not path.exists(), "immutable 90M node checkpoint already exists")
                    shutil.copy2(latest["path"], path)
                    node_checkpoint = {**boundary, "path": str(path), **file_info(path),
                        "writer_sha256": tensor_state_hash(writer.state_dict()),
                        "optimizer_sha256": optimizer_hash(optimizer), "sampler_sha256": canonical_hash(sampler.state_dict())}
                    require(node_checkpoint["sha256"] == latest["sha256"], "90M node checkpoint copy differs")
                    write_json(options.output / "node-090000000-receipt.json", node_checkpoint)
                node_checkpoint = gather(node_checkpoint)[0]
                checkpoint()
            def write(item, block):
                hidden = writer_evidence_features(block["H_new"].to(device), mode="rmsnorm", eps=1e-6)
                base = (writer_evidence_features(block["H_base_new"].to(device), mode="rmsnorm", eps=1e-6)
                        if options.branch == "residual" else None)
                matched = (writer_evidence_features(block["H_base_matched_new"].to(device), mode="rmsnorm", eps=1e-6)
                           if options.residual_difference_baseline == "matched_base" else None)
                item["state"] = write_memory_aware_native_block(
                    writer, item["state"], hidden, block["physical_rows_new"].to(device),
                    torch.ones(hidden.shape[:2], dtype=torch.bool, device=device), alpha,
                    hidden_base=base, hidden_base_matched=matched, event_observer=observe_events)
                require(bool(item["state"].values.isfinite().all()), "nonfinite memory state")
            def diagnostic_repeat_write(state, block):
                hidden = writer_evidence_features(block["H_new"].to(device), mode="rmsnorm", eps=1e-6)
                result = write_memory_aware_native_block(
                    writer, state, hidden, block["physical_rows_new"].to(device),
                    torch.ones(hidden.shape[:2], dtype=torch.bool, device=device), alpha)
                require(bool(result.values.isfinite().all()), "nonfinite repeat diagnostic memory state")
                return result
            def query_phase(queries, group_targets):
                for start, size in global_frame_batches(gather(len(queries)), 4):
                    active = queries[start:start + size]
                    mb = collate_staged_frames([task["frame"] for task in active] + [None] * (size-len(active)),
                                               kind="query", device=device)
                    costs = gather((mb["loss_target_tokens"], mb["real_input_tokens"]))
                    cost = sum(row[0] for row in costs)
                    # Charges survive exceptions and explicit resumes.
                    fork.charge(ledger, "optimizer_targets_dispatched", cost)
                    fork.charge(ledger, "query_input_tokens", sum(row[1] for row in costs))
                    save_ledger()
                    tickets = [task["owner"]["bridge"].prepare_query(
                        materialize_staged_snapshot(task["snapshot"])) for task in active]
                    tickets += [None] * (size-len(tickets))
                    packed = pack_staged_query_tickets(tickets, row_capacity=session.layout.row_capacity, device=device)
                    mb["reader_state"] = packed.reader_state
                    result = schedule.schedule_query_microbatches(
                        session, [mb], group_total_target_tokens=group_targets)["microbatches"][0]
                    require(tuple(result["loss_target_tokens_by_lane"]) == mb["loss_target_tokens_by_lane"],
                            "query target counts differ")
                    packed.scatter_cotangent()
                    for lane, task in enumerate(active):
                        loss, count = result["loss_sum_by_lane"][lane], task["frame"]["loss_target_tokens"]
                        require(math.isfinite(loss), "nonfinite query loss")
                        item = task["owner"]
                        item["bridge"].record_query(tickets[lane], loss_sum=loss, loss_tokens=count,
                                                    reader_loss_divisor=group_targets)
                        item["loss_sum"] += loss
                        item["loss_targets"] += count
                        release_staged_reader_leaf(tickets[lane])
                        task["snapshot"] = None

            checkpoint()
            save_node_if_due()
            while phase == "training" and groups < min(options.max_groups, fork.GROUPS):
                if options.stop_after_group_file and gather(options.stop_after_group_file.exists() if rank == 0 else None)[0]:
                    status = "stop_after_group_file"
                    break
                group = sampler.next_batch(12)
                expected = manifest["replay_groups"][groups]
                cost = fork.validate_replay_group(group, expected, groups)
                require_event_limit(sampler.state_dict())
                agree(expected, "rank replay groups differ")
                plans = [data.frames(item) for item in group]
                for trajectory, plan in zip(group, plans):
                    validate_frame_plan(plan, trajectory.loss_tokens)
                assignments = assign_trajectories(plans)
                optimizer.zero_grad(set_to_none=True)
                residual_head_zero = (all(int(torch.count_nonzero(p)) == 0 for p in writer.residual_head[-1].parameters())
                                      if groups == 0 and options.branch == "residual" else None)
                if groups == 0 and options.branch == "residual":
                    require(residual_head_zero, "first residual group requires an exactly zero output head")
                beta_counts[:] = [0, 0]
                local = [{"index": index, "plan": plans[index],
                    "state": LazyDecayState.from_batch(session.layout.empty_state(1, device)),
                    "bridge": NativeTrajectoryGradients(total_loss_tokens=cost,
                        expected_loss_tokens=group[index].loss_tokens, policy=gradient_policy,
                        runtime_identity_sha256=runtime_hash),
                    "expected_targets": group[index].loss_tokens, "loss_sum": 0., "loss_targets": 0}
                         for index in assignments[rank]]
                begin = time.monotonic()
                torch.cuda.reset_peak_memory_stats(device)
                diagnose = (groups == 0 or groups + 1 == fork.GROUPS
                            or (groups + 1) % options.diagnostic_cadence == 0)
                def diagnostic_callback(state, memory_block, base_block, frame):
                    return measure_encoding_sensitivity(
                        writer, state, memory_block, base_block, encoding_mode=options.branch)
                result = activity("training", lambda: build_causal_writer_histories(
                    session, local, gather=gather, write=write, encoding_mode=options.branch,
                    batch_limit=12, on_encoding=encoding_charge,
                    diagnose_wave=diagnostic_callback if diagnose else None,
                    difference_baseline=options.residual_difference_baseline,
                    diagnostic_repeat_write=diagnostic_repeat_write if diagnose and options.branch == "full" else None))
                diagnostic_seconds = result.get("diagnostic_seconds", 0.)
                activities["training"] -= diagnostic_seconds
                activities["diagnostic"] += diagnostic_seconds
                diagnostic = (fork.diagnostic_monitor(summarize_encoding_sensitivity(result["diagnostics"], stop_p99=1.))
                              if diagnose else None)
                if diagnostic is not None:
                    diagnostic["repeat_trajectories"] = result.get("repeat_diagnostics")
                if diagnostic is not None and rank == 0:
                    write_json(options.output / f"diagnostic-group-{groups + 1:04d}.json", diagnostic)
                queries = result["queries"]
                prior_dispatched = ledger["optimizer_targets_dispatched"]
                activity("training", lambda: query_phase(queries, cost))
                require(ledger["optimizer_targets_dispatched"] - prior_dispatched == cost,
                        "complete group's dispatched CE does not match its registered target cost")
                loss_sum = math.fsum(gather(math.fsum(item["loss_sum"] for item in local)))
                if groups == 0:
                    first_group_loss_comparison = fork.compare_first_group_loss(
                        manifest["reference_first_group_loss"], loss_sum=loss_sum, targets=cost,
                        group_sha256=expected["group_identity_sha256"], residual_head_zero=residual_head_zero,
                        branch=options.branch)
                    agree(first_group_loss_comparison, "ranks disagree on first-group loss comparison")
                    if rank == 0:
                        write_json(options.output / "first-group-loss-comparison.json", first_group_loss_comparison)
                queries.clear()
                def update():
                    for item in local:
                        item["bridge"].backward_writer()
                        item["bridge"] = item["state"] = None
                    require(all(item["loss_targets"] == item["expected_targets"] for item in local),
                            "trajectory CE accounting differs")
                    reduced = sum_ep4_writer_gradients(writer, group=dist.group.WORLD)
                    norm = reduced["global_gradient_l2"]
                    require(math.isfinite(norm) and norm > 0, "zero or nonfinite writer gradient")
                    betas = gather(tuple(beta_counts))
                    stops = monitor.observe_step(norm, sum(row[0] for row in betas), sum(row[1] for row in betas))
                    require(not stops, "optimizer stability stop")
                    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True, foreach=False)
                    first_update = groups == 0
                    update_before = fork.capture_optimizer_update(writer, optimizer) if first_update else None
                    optimizer.step()
                    require(all(bool(p.isfinite().all()) for p in writer.parameters()), "nonfinite writer after AdamW")
                    if options.branch == "residual":
                        require(all(p.grad is None and not p.requires_grad for p in writer.base_writer.parameters()),
                                "residual base acquired gradients or became trainable")
                    first_update_receipt = None
                    if first_update:
                        first_update_receipt = {
                            **fork.validate_first_optimizer_update(writer, optimizer, update_before),
                            "group_target_cost": cost, "CE_dispatch_exact": True,
                            "residual_base_frozen": options.branch == "residual"}
                        agree(first_update_receipt, "ranks disagree on first optimizer update")
                    monitor.complete_step(norm)
                    session.assert_frozen()
                    return norm, first_update_receipt
                norm, first_update_receipt = activity("training", update)
                groups += 1
                targets += cost
                fork.charge(ledger, "optimizer_targets_completed", cost)
                paired_loss_observation = fork.observe_paired_loss(paired_loss_state, manifest["policy"]["paired_loss"],
                    manifest["reference_group_losses"][groups - 1], completed_groups=groups,
                    loss_sum=loss_sum, targets=cost, group_sha256=expected["group_identity_sha256"])
                agree(paired_loss_observation, "ranks disagree on paired-loss monitoring")
                del local
                if groups == fork.GROUPS:
                    require(targets == fork.TARGETS and sampler.state_dict() == prepared["endpoint_sampler"],
                            "fork replay final budget or sampler differs")
                    phase = "evaluation_pending"
                save_ledger()
                checkpoint()
                save_node_if_due()
                row = {"event": "memory_fork_group_complete", "branch": options.branch,
                       "fork_group": groups, "step": fork.SOURCE_STEP + groups,
                       "loss_target_tokens": cost, "cumulative_model_loss_targets": fork.SOURCE_TARGETS + targets,
                       "group_identity_sha256": expected["group_identity_sha256"],
                       "reference_group_matched": True, "mean_nll": loss_sum / cost,
                       "first_update_checks": first_update_receipt,
                       "first_group_loss_comparison": first_group_loss_comparison if groups == 1 else None,
                       "paired_loss": paired_loss_observation,
                       "encoding_sensitivity": diagnostic,
                       "loss_sum": loss_sum, "raw_gradient_l2": norm, "stability": monitor.last_observation,
                       "seconds_including_checkpoint": max(gather(time.monotonic() - begin)),
                       "peak_allocated_bytes": max(gather(torch.cuda.max_memory_allocated(device))),
                       "latest_checkpoint": latest}
                if rank == 0:
                    with (options.output / "training.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    write_json(options.output / "progress.json", row)
                    write_json(options.output / "paired-loss-monitor.json", paired_loss_observation)
                    print(json.dumps({key: row[key] for key in ("event", "branch", "fork_group", "mean_nll")}), flush=True)
            if phase == "evaluation_pending" and terminal_checkpoint is None:
                if rank == 0:
                    path = options.output / "branch-terminal.pt"
                    require(not path.exists(), "immutable terminal checkpoint already exists")
                    shutil.copy2(latest["path"], path)
                    terminal_checkpoint = {"path": str(path), **file_info(path),
                        "completed_groups": groups, "completed_targets": targets,
                        "writer_sha256": tensor_state_hash(writer.state_dict()),
                        "optimizer_sha256": optimizer_hash(optimizer),
                        "sampler_sha256": canonical_hash(sampler.state_dict())}
                    require(terminal_checkpoint["sha256"] == latest["sha256"], "terminal checkpoint copy differs")
                terminal_checkpoint = gather(terminal_checkpoint)[0]
            while phase == "evaluation_pending" and len(evaluation_receipts) < fork.ENDPOINT_EVALUATIONS:
                require(groups == fork.GROUPS and targets == fork.TARGETS, "DEV64 requires the terminal model")
                agree(terminal_checkpoint, "ranks disagree on immutable terminal checkpoint")
                require(file_info(terminal_checkpoint["path"])["sha256"] == terminal_checkpoint["sha256"],
                        "immutable terminal checkpoint changed")
                evaluation_attempt += 1
                checkpoint()
                before_rng = capture_rng()
                unchanged = (tensor_state_hash(writer.state_dict()), optimizer_hash(optimizer), canonical_hash(sampler.state_dict()))
                require(unchanged == tuple(terminal_checkpoint[key] for key in
                    ("writer_sha256", "optimizer_sha256", "sampler_sha256")), "DEV64 model differs from terminal checkpoint")
                budget_before = copy.deepcopy(ledger)
                evaluation_index = len(evaluation_receipts) + 1
                evaluation_output = options.output / f"DEV64-run-{evaluation_index:02d}-attempt-{evaluation_attempt:03d}"
                report = activity("evaluation", lambda: evaluate_memory_dev64(
                    session, writer, alpha, encoding_mode=options.branch,
                    plan_path=options.dev64_plan, arm="memory_" + options.branch,
                    milestone=fork.END_TARGETS,
                    output=evaluation_output,
                    charge_targets=lambda count: charge_ce("evaluation_ce_targets", count),
                    charge_encoding=lambda row: encoding_charge(row, category="evaluation"),
                    checkpoint_metadata={"branch_checkpoint": terminal_checkpoint,
                        "evaluation_run_index": evaluation_index, "evaluation_attempt": evaluation_attempt,
                        "optimizer_step": fork.SOURCE_STEP + groups,
                        "completed_loss_targets": fork.SOURCE_TARGETS + targets,
                        "optimizer_updates_since_fork": groups, "runtime_identity_sha256": runtime_hash}))
                restore_rng(before_rng)
                require(unchanged == (tensor_state_hash(writer.state_dict()), optimizer_hash(optimizer),
                                     canonical_hash(sampler.state_dict())), "DEV64 mutated training state")
                receipt = fork.completed_evaluation_receipt(report, before=budget_before, after=ledger,
                    attempt=evaluation_attempt, run_index=evaluation_index, terminal_checkpoint=terminal_checkpoint,
                    reference=manifest["reference_endpoint_evaluation"]["metrics"])
                receipt["output"] = str(evaluation_output)
                evaluation_receipts.append(receipt)
                evaluation_receipt = fork.summarize_endpoint_evaluations(
                    evaluation_receipts, manifest["reference_endpoint_evaluation"]["metrics"])
                evaluation_receipts = evaluation_receipt["runs"]
                if rank == 0:
                    for completed in evaluation_receipts:
                        write_json(options.output / f"DEV64-run-{completed['run_index']:02d}-receipt.json", completed)
                    write_json(options.output / "DEV64-repeat-comparison.json", evaluation_receipt)
                if len(evaluation_receipts) == fork.ENDPOINT_EVALUATIONS:
                    phase = "complete"
                checkpoint()
            status = "complete" if phase == "complete" else (
                status if status != "running" else "bounded_preflight_complete")
            checkpoint()
            time_record = timing()
            if rank == 0:
                write_json(options.output / ("COMPLETE.json" if phase == "complete" else "STOPPED.json"), {
                    "protocol": fork.PROTOCOL, "status": status, "phase": phase,
                    "completed_groups": groups, "completed_targets": targets,
                    "budget": ledger, "latest_checkpoint": latest,
                    "diagnostic_G_action": "monitor_only_never_automatic_stop",
                    "evaluation_receipt": evaluation_receipt, "automatic_retry": False,
                    "terminal_checkpoint": terminal_checkpoint,
                    "first_group_loss_comparison": first_group_loss_comparison,
                    "paired_loss": paired_loss_observation, "node_checkpoint": node_checkpoint,
                    "resume_supported": True, "timing": time_record})
            dist.barrier()
    try:
        run_ep4_session(options, run)
    except BaseException as error:
        status = "failed"
        save_ledger()
        if rank == 0:
            write_json(options.output / "FAILURE.json", {
                "protocol": fork.PROTOCOL, "error_type": type(error).__name__, "message": str(error),
                "budget": ledger, "latest_checkpoint": latest, "automatic_retry": False,
                "resume": "explicit_new_output_with_latest_complete_checkpoint_and_latest_ledger"})
        raise
    finally:
        write_json(options.output / f"timing-rank-{rank}.json", local_timing())


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source-checkpoint", "source-input-manifest", "source-runtime-identity", "source-budget-ledger",
                 "source-evaluation-report", "source-evaluation-complete", "reference-training-log",
                 "reference-endpoint-checkpoint", "reference-evaluation-report", "reference-evaluation-complete",
                 "cache-root", "dev64-plan", "delta-repo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("resume-checkpoint", "resume-budget-ledger", "model-path", "verification", "stop-after-group-file"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--branch", choices=("full", "residual"), required=True)
    parser.add_argument("--residual-difference-baseline", choices=("actual_base", "matched_base"),
                        default="actual_base")
    parser.add_argument("--max-groups", type=int, default=fork.GROUPS,
                        help="total completed groups since fork; use e.g. 2 for bounded preflight")
    parser.add_argument("--diagnostic-cadence", type=int, default=50)
    parser.add_argument("--paired-loss-threshold", type=float, default=.01)
    parser.add_argument("--paired-loss-warmup-groups", type=int, default=50)
    parser.add_argument("--paired-loss-window-groups", type=int, default=50)
    parser.add_argument("--paired-loss-consecutive-windows", type=int, default=3)
    parser.add_argument("--paired-loss-soft-stop-action", choices=("monitor",), default="monitor",
                        help="monitor only; stopping requires the user stop-after-group file")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate", "--dry-run", dest="validate", action="store_true")
    action.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    fork.fork_policy(options.branch, options.residual_difference_baseline)
    require(0 <= options.max_groups <= fork.GROUPS, "max groups must be in [0,835]")
    require(options.diagnostic_cadence > 0, "diagnostic cadence must be positive")
    require(not options.execute or (options.model_path and options.verification),
            "execute requires model-path and verification")
    for key, value in vars(options).items():
        if isinstance(value, Path):
            setattr(options, key, value.resolve())
    if options.stop_after_group_file is None:
        options.stop_after_group_file = options.output / "stop-after-group"
    sys.path.insert(0, str(options.delta_repo / "src"))
    prepared = prepare(options)
    if options.validate:
        print(json.dumps({"status": "memory_fork_CPU_valid", "new_CE_targets": 0,
                          "optimizer_updates": 0, "manifest": prepared["manifest"]}, indent=2))
        return
    execute(options, prepared)


if __name__ == "__main__":
    main()

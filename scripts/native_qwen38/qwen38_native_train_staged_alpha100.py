"""Continue only alpha=1 after the old shared-cap run finishes naturally.

Both independent arm states and all prior charges are preserved. An explicit
charged/model policy governs the new per-arm cap; native training is unchanged.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

from qwen38_native_norm_probe import canonical_hash, file_info, require
from qwen38_native_train_batched import (
    restore_checkpoint_state, restore_rng_state, validate_frame_plan)
from qwen38_native_alpha100_policy import (
    ACTIVE, CONTROL, ARMS, ARM_CAP, CONTROL_TARGETS, CONTROL_STEPS, MILESTONES,
    training_policy, state_digest, event_repeat_counts, require_event_limit,
    derive_arm_accounting, cap_allows, dispatch_query, initial_policy_cursor, validate_policy_cursor)
import qwen38_native_train_staged_alternating as legacy_runner

PROTOCOL = "qwen38_native_staged_alpha100_continuation_v1"
CHECKPOINT_PROTOCOL = "qwen38_native_staged_alpha100_checkpoint_v1"
TIMING_PROTOCOL = "qwen38_native_gpu_occupied_wall_v1"
STAGE = 20000000
INITIAL_SHA = "aa54ccfcdc800d46a145e7dab9787918fbbc96c0a4f662328e7d88dba6d32f11"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


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


def optimizer_hash(optimizer):
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


def assign_trajectories(plans):
    require(len(plans) == 12, "formal EP4 groups require exactly 12 whole trajectories")
    ranked = sorted(range(12), key=lambda index: (-len(plans[index]), index))
    assigned, loads = [[] for _ in range(4)], [0] * 4
    for index in ranked:
        target = min((rank for rank in range(4) if len(assigned[rank]) < 3),
                     key=lambda rank: (loads[rank], rank))
        assigned[target].append(index)
        loads[target] += len(plans[index])
    return assigned


def global_frame_batches(counts, limit):
    require(len(counts) == 4 and all(type(count) is int and count >= 0 for count in counts),
            "invalid distributed frame counts")
    maximum = max(counts)
    for start in range(0, maximum, limit):
        size = min(limit, maximum - start)
        yield start, size


def code_manifest(repo):
    result = legacy_runner.code_manifest(repo)
    for name in ("qwen38_native_train_staged_alpha100.py", "qwen38_native_alpha100_policy.py"):
        path = Path(__file__).resolve().parent / name
        result[path.relative_to(repo).as_posix()] = file_info(path)
    return result


def compatible_runtime(runtime):
    """Only the runner protocol/code/policy identity is allowed to change."""
    return {key: value for key, value in runtime.items()
            if key not in ("protocol", "code_sha256", "continuation_policy")}


def control_evaluation_receipt(source):
    from delta_ngram.p2_initialization import writer_tensor_sha256
    from delta_ngram.qwen38_probe import tensor_state_hash
    row = source["schedule_cursor"]["completed_stages"][-1]
    directory = Path(row["evaluation_output"])
    report_path, complete_path = directory / "report.json", directory / "COMPLETE.json"
    report, complete = json.loads(report_path.read_bytes()), json.loads(complete_path.read_bytes())
    require(report.get("complete") is True and complete.get("complete") is True
            and report["arm"] == complete["arm"] == CONTROL
            and report["milestone"] == complete["milestone"] == 40000000,
            "alpha099 40M DEV64 did not complete")
    eval_writer_hash = writer_tensor_sha256(source["arms"][CONTROL]["writer"])
    checkpoint_writer_hash = tensor_state_hash(source["arms"][CONTROL]["writer"])
    require(canonical_hash(report) == row["evaluation_report_sha256"]
            and file_info(report_path)["sha256"] == complete["report_sha256"]
            and report["writer_sha256"] == complete["writer_sha256"] == eval_writer_hash
            and checkpoint_writer_hash == source["arms"][CONTROL]["writer_sha256"],
            "alpha099 DEV64 report identity differs")
    require(report["checkpoint"]["optimizer_step"] == CONTROL_STEPS
            and report["checkpoint"]["completed_loss_targets"] == CONTROL_TARGETS,
            "alpha099 DEV64 evaluated a different control state")
    return {"report": {"path": str(report_path), **file_info(report_path)},
            "complete": {"path": str(complete_path), **file_info(complete_path)},
            "canonical_report_sha256": canonical_hash(report),
            "writer_tensor_sha256": eval_writer_hash, "checkpoint_tensor_state_hash": checkpoint_writer_hash}


def summarize_timing_ranks(ranks, *, status):
    require(len(ranks) == 4 and sorted(row["rank"] for row in ranks) == list(range(4))
            and len({row["run_id"] for row in ranks}) == 1, "invalid rank timing records")
    categories = ("training", "evaluation", "checkpoint", "other")
    for row in ranks:
        require(set(row["categories_seconds"]) == set(categories)
                and all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
                        for value in (row["wall_seconds"], *row["categories_seconds"].values())),
                "invalid timing duration")
    maxima = {name: max(row["categories_seconds"][name] for row in ranks) for name in categories}
    wall = max(row["wall_seconds"] for row in ranks)
    return {"run_id": ranks[0]["run_id"], "status": status, "gpu_count": 4,
            "wall_seconds_max_rank": wall, "gpu_occupied_hours": wall * 4 / 3600,
            "categories_seconds_max_rank": maxima,
            "categories_gpu_occupied_hours": {key: value * 4 / 3600 for key, value in maxima.items()},
            "ranks": copy.deepcopy(ranks)}


def recover_timing_history(options, source):
    if options.source_timing_ledger:
        timing = json.loads(options.source_timing_ledger.read_bytes())
        require(timing["protocol"] == TIMING_PROTOCOL and isinstance(timing["runs"], dict),
                "source timing ledger differs")
        runs = copy.deepcopy(timing["runs"])
    elif options.resume_shared_cap_final:
        runs = {}
    else:
        runs = copy.deepcopy(source["timing"]["runs"])
    exits = []
    if not options.resume_shared_cap_final:
        saved_runs = source["timing"]["runs"]
        require(set(saved_runs) <= set(runs) and source["run_id"] in runs,
                "source timing ledger omits checkpoint history")
        for key, old in saved_runs.items():
            require(runs[key]["wall_seconds_max_rank"] >= old["wall_seconds_max_rank"],
                    "source timing ledger would lose recorded GPU duration")
            if key != source["run_id"]:
                require(runs[key] == old, "completed prior timing history changed")
        current = runs[source["run_id"]]
        ranks = {row["rank"]: copy.deepcopy(row) for row in current["ranks"]}
        for rank in range(4):
            path = options.resume_checkpoint.parent / f"timing-rank-{rank}.json"
            if not path.is_file():
                continue
            info = file_info(path)
            row = json.loads(path.read_bytes())
            require(row["rank"] == rank and row["run_id"] == source["run_id"]
                    and row["wall_seconds"] >= ranks[rank]["wall_seconds"]
                    and file_info(path) == info, "source rank-exit timing differs")
            ranks[rank] = row
            exits.append({"path": str(path), **info})
        recovered = summarize_timing_ranks([ranks[rank] for rank in range(4)], status="source_exit_recovered")
        recovered["rank_exit_records"] = len(exits)
        recovered["complete_exit_coverage"] = len(exits) == 4
        recovered["incomplete_exit_duration_is_lower_bound"] = len(exits) < 4
        runs[source["run_id"]] = recovered
    for key, row in runs.items():
        require(key == row["run_id"], "timing run identifier differs")
        computed = summarize_timing_ranks(row["ranks"], status=row["status"])
        require(computed["gpu_occupied_hours"] == row["gpu_occupied_hours"]
                and computed["wall_seconds_max_rank"] == row["wall_seconds_max_rank"],
                "timing summary disagrees with its rank records")
    return runs, exits


def prepare(options):
    import torch
    checkpoint_info = file_info(options.resume_checkpoint)
    source = torch.load(options.resume_checkpoint, map_location="cpu", weights_only=False)
    prior = json.loads(options.source_budget_ledger.read_bytes())
    source_runtime = json.loads(options.source_runtime_identity.read_bytes())
    source_manifest = json.loads(options.source_input_manifest.read_bytes())
    require(file_info(options.resume_checkpoint) == checkpoint_info, "source checkpoint changed while reading")
    require(source.get("trajectory_boundary") is True
            and source.get("memory_reset_at_next_trajectory") is True,
            "requires a completed trajectory-boundary checkpoint")
    require(file_info(options.source_input_manifest)["sha256"] == source["registration_sha256"],
            "source registration file differs from its checkpoint")
    require(canonical_hash(source_runtime) == source["runtime_identity_sha256"],
            "source runtime identity does not match its checkpoint")
    require(canonical_hash(source_manifest["code"]) == source_manifest["code_manifest_sha256"]
            == source["code_manifest_sha256"] == source_runtime["code_sha256"],
            "source code identity chain differs")
    for key in ("total_optimizer_target_accesses", "total_diagnostic_target_accesses"):
        require(type(prior[key]) is int and prior[key] >= source["budget"][key],
                "continuation cannot refund previous target accesses")
    require(prior["total_optimizer_target_accesses"] >= 0
            and prior["total_diagnostic_target_accesses"] == 59620, "invalid historical budget")
    prior_evaluation = prior.get("total_evaluation_target_accesses",
                                prior.get("evaluation_target_accesses", 0))
    require(type(prior_evaluation) is int and prior_evaluation >= 0, "invalid evaluation ledger")
    require(prior.get("evaluation_target_accesses", prior_evaluation) == prior_evaluation,
            "evaluation ledger aliases disagree")
    require(prior_evaluation >= source["budget"].get("total_evaluation_target_accesses", 0),
            "continuation cannot refund evaluation accesses")
    initial_info, plan_info = file_info(options.initial_writer), file_info(options.dev64_plan)
    require(initial_info["sha256"] == INITIAL_SHA, "immutable seed17 initial writer differs")
    require(source["dev64_plan"] == source_manifest["dev64_plan"] == source_runtime["DEV64_plan"] == plan_info
            and source["initial_writer"] == source_manifest["initial_writer"] == initial_info,
            "saved DEV64 plan or initial writer identity differs")
    cache_complete = file_info(options.cache_root / "COMPLETE.json")
    cache_manifest = file_info(options.cache_root / "data_manifest.json")
    require(cache_complete == source_manifest["cache_complete"]
            and cache_manifest == source_manifest["cache_manifest"], "training corpus files differ")
    codes = code_manifest(options.delta_repo)
    code_hash = canonical_hash(codes)
    continuation_policy = training_policy(options.budget_basis)
    arm_sources = source["arms"]
    require(set(arm_sources) == {name for name, _ in ARMS}, "joint checkpoint needs both arms")
    complete_info = None
    if options.resume_shared_cap_final:
        require(options.source_complete is not None, "old-final migration requires --source-complete")
        require(source.get("protocol") == legacy_runner.CHECKPOINT_PROTOCOL
                and source_manifest["protocol"] == legacy_runner.PROTOCOL
                and source_runtime["protocol"] == legacy_runner.PROTOCOL,
                "migration requires the original alternating protocol")
        require(source_manifest["code"] == legacy_runner.code_manifest(options.delta_repo),
                "an existing source file changed; migration only adds the new runner and policy")
        complete = json.loads(options.source_complete.read_bytes())
        complete_info = {"path": str(options.source_complete), **file_info(options.source_complete)}
        require(complete["protocol"] == legacy_runner.PROTOCOL
                and complete["status"] == "shared_budget_complete_group_boundary"
                and complete["latest_checkpoint"]["sha256"] == checkpoint_info["sha256"]
                and complete["latest_checkpoint"]["bytes"] == checkpoint_info["bytes"]
                and complete["runtime_identity_sha256"] == source["runtime_identity_sha256"]
                and complete["schedule_cursor"] == source["schedule_cursor"]
                and complete["arms"] == {name: {"optimizer_step": value["optimizer_step"],
                                                 "completed_loss_targets": value["completed_loss_targets"]}
                                         for name, value in arm_sources.items()}
                and complete["budget"] == source["budget"],
                "old run did not finish naturally at this exact checkpoint")
        require(prior["total_optimizer_target_accesses"] == source["budget"]["total_optimizer_target_accesses"]
                and source["shared_optimizer_target_cap"] == legacy_runner.CAP
                and complete["unused_shared_optimizer_budget"]
                == legacy_runner.CAP - prior["total_optimizer_target_accesses"] >= 0
                and type(complete["next_complete_group_target_cost"]) is int
                and complete["next_complete_group_target_cost"] > complete["unused_shared_optimizer_budget"],
                "old final checkpoint is not the complete-group shared-cap endpoint")
        cursor = initial_policy_cursor(source)
        control_receipt = control_evaluation_receipt(source)
        legacy_cursor = copy.deepcopy(source["schedule_cursor"])
        legacy_metadata = {key: copy.deepcopy(value) for key, value in source.items() if key != "arms"}
        inactive_digest = state_digest(arm_sources[CONTROL])
    else:
        require(source.get("protocol") == CHECKPOINT_PROTOCOL
                and source.get("code_manifest_sha256") == code_hash
                and source["continuation_policy"] == source_manifest["continuation_policy"] == continuation_policy
                and source_runtime["continuation_policy"] == continuation_policy,
                "continuation requires the exact saved new code and budget policy")
        cursor = copy.deepcopy(validate_policy_cursor(source["schedule_cursor"]))
        legacy_cursor = copy.deepcopy(source["legacy_schedule_cursor"])
        legacy_metadata = copy.deepcopy(source["legacy_checkpoint_metadata"])
        control_receipt = copy.deepcopy(source["control_evaluation_receipt"])
        inactive_digest = source["inactive_arm_state_sha256"]
        require(state_digest(arm_sources[CONTROL]) == inactive_digest, "inactive alpha099 state changed")
    for name, saved in arm_sources.items():
        require(saved["sampler"]["total_loss_tokens"] == saved["completed_loss_targets"]
                and saved["stability"]["completed_steps"] == saved["optimizer_step"],
                "checkpoint sampler/model/optimizer counters differ")
        require(saved["arm"] == name
                and saved["alpha"] == dict(ARMS)[name], "checkpoint arm/alpha differs")
        require(len(saved["rng_by_rank"]) == 4
                and sorted(item["rank"] for item in saved["rng_by_rank"]) == list(range(4)),
                "each arm checkpoint needs all four rank RNG states")
        require_event_limit(saved["sampler"])
    accounting = derive_arm_accounting(source, prior, options.resume_shared_cap_final)
    require(cap_allows(continuation_policy, accounting["charged"][ACTIVE], accounting["model"][ACTIVE], 0),
            "source already exceeds the chosen per-arm budget basis")
    timing_history, timing_exit_files = recover_timing_history(options, source)
    require(options.output != options.resume_checkpoint.parent,
            "use a new output directory; the source run remains immutable")
    if os.environ.get("RANK") in (None, "0"):
        require(not options.output.exists(), "output must be a new directory")
    manifest = {
        "protocol": PROTOCOL, "code": codes, "code_manifest_sha256": code_hash,
        "source_checkpoint": checkpoint_info,
        "source_budget_ledger": file_info(options.source_budget_ledger),
        "source_input_manifest": file_info(options.source_input_manifest),
        "source_runtime_identity": file_info(options.source_runtime_identity),
        "source_runtime_identity_sha256": source["runtime_identity_sha256"],
        "source_complete": complete_info,
        "source_runtime_compatibility": compatible_runtime(source_runtime),
        "source_unchanged_code": legacy_runner.code_manifest(options.delta_repo),
        "continuation_policy": continuation_policy,
        "continuation_policy_sha256": canonical_hash(continuation_policy),
        "initial_policy_cursor": cursor,
        "legacy_schedule_cursor": legacy_cursor,
        "legacy_checkpoint_metadata": legacy_metadata,
        "control_evaluation_receipt": control_receipt,
        "inactive_arm_state_sha256": inactive_digest,
        "initial_arm_state_sha256": {name: state_digest(value) for name, value in arm_sources.items()},
        "source_arm_accounting": accounting,
        "source_arms": {name: {"optimizer_step": value["optimizer_step"],
                               "completed_loss_targets": value["completed_loss_targets"],
                               "writer_sha256": value["writer_sha256"]}
                       for name, value in arm_sources.items()},
        "initial_writer": initial_info, "dev64_plan": plan_info,
        "cache_complete": cache_complete, "cache_manifest": cache_manifest,
        "migration": ("old_shared_cap_final_to_alpha100_only" if options.resume_shared_cap_final
                      else "exact_alpha100_continuation_resume"),
        "source_checkpoint_role": "both_independent_arm_states_restored_alpha100_only_activated",
        "source_model_state_restored": True,
        "new_run_arm_initialization": "restore_both_writers_AdamW_samplers_stability_all_rank_RNG_exactly",
        "topology": {"TP": 1, "PP": 1, "EP": 4, "ETP": 1, "DP": 4, "CP": 1},
        "global_trajectories_per_group": 12, "evidence_batch_limit": 12, "query_batch_limit": 4,
        "optimizer_target_cap_per_arm": ARM_CAP,
        "prior_optimizer_target_accesses": prior["total_optimizer_target_accesses"],
        "prior_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"],
        "budget_endpoint": "last_complete_12_trajectory_group_at_or_below_selected_per_arm_basis_then_DEV64",
        "arms": [{"name": name, "alpha": value} for name, value in ARMS],
        "stage_model_target_stride": STAGE,
        "stage_order": "alpha100_only_60M_80M_100M_model_milestones_or_earlier_actual_budget_endpoint",
        "stage_boundary": "last_whole_12_trajectory_group_at_or_below_each_arm_milestone",
        "checkpoint": "atomic_joint_latest_every_group; archive_model_milestones_and_actual_budget_endpoint",
        "evaluation": "fixed_DEV64_at_milestones_and_terminal_budget_endpoint; independent_charges",
        "writer_math": "unchanged_D1_FP32", "deterministic_algorithm_policy": "unchanged_native_default",
        "stages": ["native_evidence_H12_to_CPU", "causal_writer_histories_and_snapshots",
                   "native_query_Q4_backward", "joint_writer_VJP_DP_SUM_clip1_AdamW"],
        "evidence_LM_head": "omit_unused_logits_after_capturing_final_native_H",
        "optimizer": {"name": "AdamW", "lr": .0003, "betas": [.9, .999],
                      "eps": 1e-8, "weight_decay": 0., "clip": 1.},
        "additional_diagnostic_CE_targets": 0, "paired_runs": False, "evaluations": True,
        "automatic_retry": False, "runtime_equivalence_claimed": False,
        "run_id": canonical_hash({"output": str(options.output), "source": checkpoint_info,
                                  "policy": continuation_policy}),
        "prior_policy_timing_runs": timing_history,
        "legacy_gpu_time": ({"status": "separate_not_reconstructed_from_tokens",
                             "source_complete": complete_info}
                            if options.resume_shared_cap_final
                            else copy.deepcopy(source_manifest["legacy_gpu_time"])),
        "source_timing_ledger": file_info(options.source_timing_ledger) if options.source_timing_ledger else None,
        "source_rank_exit_timing_files": timing_exit_files,
    }
    return source, prior, manifest


def execute(options, source, prior, manifest):
    from qwen38_native_ep4_session import run_ep4_session
    rank = int(os.environ.get("RANK", "-1"))
    require(0 <= rank < 4 and int(os.environ.get("WORLD_SIZE", "0")) == 4,
            "use four torchrun ranks")
    marker = options.output / "input_manifest.json"
    execution_started = time.monotonic()
    activities = {name: 0.0 for name in ("training", "evaluation", "checkpoint")}
    continuation_policy = manifest["continuation_policy"]
    accounting = manifest["source_arm_accounting"]
    last_timing = {"protocol": TIMING_PROTOCOL, "runs": copy.deepcopy(manifest["prior_policy_timing_runs"])}
    def activity(kind, action):
        begin = time.monotonic()
        try:
            return action()
        finally:
            activities[kind] += time.monotonic() - begin
    def local_timing(status="running"):
        elapsed = time.monotonic() - execution_started
        return {"run_id": manifest["run_id"], "rank": rank, "status": status,
                "wall_seconds": elapsed,
                "categories_seconds": {**activities, "other": max(0., elapsed - sum(activities.values()))},
                "scope": "execute_entry_including_model_load; excludes_prior_CPU_validation"}
    dispatched = completed = query_inputs = evidence_inputs = 0
    evaluation_dispatched = 0
    prior_evaluation = prior.get("total_evaluation_target_accesses",
                                prior.get("evaluation_target_accesses", 0))
    dispatched_by_arm = {name: 0 for name, _ in ARMS}
    completed_by_arm = {name: 0 for name, _ in ARMS}
    def budget():
        charged = {name: accounting["charged"][name] + dispatched_by_arm[name] for name, _ in ARMS}
        model = {name: accounting["model"][name] + completed_by_arm[name] for name, _ in ARMS}
        return {
            "protocol": "qwen38_native_staged_alpha100_ledger_v1",
            "prior_optimizer_target_accesses": prior["total_optimizer_target_accesses"],
            "prior_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"],
            "optimizer_targets_dispatched": dispatched,
            "optimizer_targets_in_completed_groups": completed,
            "total_optimizer_target_accesses": prior["total_optimizer_target_accesses"] + dispatched,
            "total_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"],
            "prior_evaluation_target_accesses": prior_evaluation,
            "evaluation_target_accesses": prior_evaluation + evaluation_dispatched,
            "total_evaluation_target_accesses": prior_evaluation + evaluation_dispatched,
            "new_evaluation_target_accesses": evaluation_dispatched,
            "optimizer_targets_dispatched_by_arm": dict(dispatched_by_arm),
            "optimizer_targets_completed_by_arm": dict(completed_by_arm),
            "cumulative_optimizer_target_accesses_by_arm": charged,
            "effective_model_training_targets_by_arm": model,
            "historical_optimizer_target_accesses_by_arm": dict(accounting["historical"]),
            "uncompleted_or_failed_optimizer_target_accesses_by_arm": {
                name: charged[name] - model[name] - accounting["historical"][name] for name, _ in ARMS},
            "per_arm_cap_progress": dict(model if continuation_policy["budget_basis"] == "model" else charged),
            "budget_basis": continuation_policy["budget_basis"],
            "optimizer_target_cap_per_arm": ARM_CAP, "new_diagnostic_target_accesses": 0,
            "authorized_continuation_by_arm": {ACTIVE: True, CONTROL: False},
            "lifetime_charged_total_is_accounting_only": continuation_policy["budget_basis"] == "model",
            "query_input_tokens": query_inputs, "evidence_input_tokens": evidence_inputs,
            "failed_or_partial_queries_are_not_refunded": True,
        }
    def save_budget():
        if rank == 0:
            write_json(options.output / "budget_ledger.json", budget())
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        write_json(marker, manifest)
        save_budget()
    else:
        deadline = time.monotonic() + 300
        while not marker.is_file():
            require(time.monotonic() < deadline, "rank0 did not create the input manifest")
            time.sleep(.1)
    options.tokens = 640

    def run(session):
        nonlocal dispatched, completed, query_inputs, evidence_inputs, evaluation_dispatched, exit_status
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
        from delta_ngram.qwen38_ep4_batching import SingleRankGradientPolicy, sum_ep4_writer_gradients
        from delta_ngram.qwen38_staged_batching import (
            collate_staged_frames, pack_staged_query_tickets, materialize_staged_snapshot,
            release_staged_reader_leaf)
        from qwen38_native_bounded_checks import BoundedStability
        from qwen38_native_dispatch import native_hc_dispatch
        from qwen38_native_router import native_router_dispatch
        from qwen38_native_ep4_session import runtime_identity
        import qwen38_native_staged_schedule as schedule
        from qwen38_native_dev64_eval import evaluate_dev64

        device = session.device
        latest = None
        archives = []
        cursor = copy.deepcopy(manifest["initial_policy_cursor"])
        def gather(value):
            values = [None] * 4
            dist.all_gather_object(values, value)
            return values
        def agree(value, message):
            values = gather(value)
            require(all(item == values[0] for item in values), message)
            return values[0]
        def timing_snapshot(*, write=True, status="running"):
            nonlocal last_timing
            ranks = gather(local_timing(status))
            current = summarize_timing_ranks(ranks, status=status)
            runs = copy.deepcopy(manifest["prior_policy_timing_runs"])
            runs[manifest["run_id"]] = current
            last_timing = {"protocol": TIMING_PROTOCOL, "runs": runs,
                           "cumulative_policy_gpu_occupied_hours": sum(row["gpu_occupied_hours"] for row in runs.values()),
                           "cumulative_categories_gpu_occupied_hours": {
                               name: sum(row["categories_gpu_occupied_hours"][name] for row in runs.values())
                               for name in ("training", "evaluation", "checkpoint", "other")},
                           "definition": "rank_max_wall_seconds_times_4_GPUs; includes_waiting_not_kernel_time",
                           "category_definition": "independent_rank_maxima; categories_need_not_sum_to_total_rank_max",
                           "legacy_history": manifest["legacy_gpu_time"]}
            if write and rank == 0:
                write_json(options.output / "timing_ledger.json", last_timing)
            return last_timing
        def rng_state():
            return {"rank": rank, "python": random.getstate(), "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state(device).cpu(),
                    "tensor_parallel": clone_cpu(get_cuda_rng_tracker().get_states())}
        def timed(label, action):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            begin = time.monotonic()
            def synchronized_action():
                result = action()
                torch.cuda.synchronize(device)
                return result
            result = activity("training", synchronized_action)
            stats = gather({"rank": rank, "seconds": time.monotonic() - begin,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
            return result, {"phase": label, "seconds_max_rank": max(item["seconds"] for item in stats),
                            "peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in stats)}

        with ExitStack() as scope:
            hc = scope.enter_context(native_hc_dispatch(unwrap=True))
            router = scope.enter_context(native_router_dispatch(session.raw))
            precision = scope.enter_context(Qwen38SplitPLEPrecision(session.adapter))
            runtime = {"protocol": PROTOCOL, "native": runtime_identity(session),
                       "HC": hc.identity, "router": router["identity"], "precision": precision.identity(),
                       "code_sha256": manifest["code_manifest_sha256"],
                       "evidence_batch_limit": 12, "query_batch_limit": 4,
                       "evidence_omits_LM_head_and_logits": True,
                       "arms": manifest["arms"], "stage_model_target_stride": STAGE,
                       "DEV64_plan": manifest["dev64_plan"],
                       "continuation_policy": continuation_policy}
            runtime_hash = agree(canonical_hash(runtime), "runtime identity differs across ranks")
            require(compatible_runtime(runtime) == manifest["source_runtime_compatibility"],
                    "native reader, HC, router, precision, topology or DEV64 identity changed")
            if not options.resume_shared_cap_final:
                require(runtime_hash == source["runtime_identity_sha256"],
                        "staged continuation runtime differs from the saved checkpoint")
            if rank == 0:
                write_json(options.output / "runtime_identity.json", runtime)
            policy = SingleRankGradientPolicy(
                runtime_hash, tensor_parallel_world_size=ps.get_tensor_model_parallel_world_size())
            data = NativeFullTrainingData(
                options.cache_root, expected_complete_sha256=manifest["cache_complete"]["sha256"],
                expected_manifest_sha256=manifest["cache_manifest"]["sha256"])
            require(data.identity == source["data_identity"], "training corpus identity changed")
            arms = {}
            for name, alpha_value in ARMS:
                saved = source["arms"][name]
                sampler = data.sampler(seed=17)
                writer, alpha = make_qwen38_writer(alpha=alpha_value)
                writer, alpha = writer.to(device).float().train(), alpha.to(device)
                optimizer = torch.optim.AdamW(writer.parameters(), lr=.0003, betas=(.9, .999),
                                              eps=1e-8, weight_decay=0., foreach=False, fused=False)
                stability = BoundedStability(hard_stop_gradient_spikes=False, hard_stop_beta_saturation=False)
                require(saved["arm"] == name and saved["alpha"] == alpha_value, "arm checkpoint identity differs")
                steps, targets = restore_checkpoint_state(
                    saved, writer=writer, optimizer=optimizer, sampler=sampler, stability=stability)
                require(optimizer_hash(optimizer) == saved["optimizer_sha256"], "restored AdamW state differs")
                restore_rng_state(next(item for item in saved["rng_by_rank"] if item["rank"] == rank),
                                  rank=rank, device=device, tensor_parallel_tracker=get_cuda_rng_tracker())
                agree(tensor_state_hash(writer.state_dict()), "arm writer replicas differ")
                agree(optimizer_hash(optimizer), "arm AdamW replicas differ")
                arms[name] = {"writer": writer, "alpha": alpha, "alpha_value": alpha_value,
                              "optimizer": optimizer, "sampler": sampler, "stability": stability,
                              "steps": steps, "targets": targets, "rng": rng_state()}
            beta_counts = [0, 0]
            def gates(module, args, output):
                beta = output.detach().float().sigmoid()
                beta_counts[0] += int(((beta < .001) | (beta > .999)).sum())
                beta_counts[1] += beta.numel()
            for value in arms.values():
                scope.callback(value["writer"].gate_head[2].register_forward_hook(gates).remove)
            active_name = cursor["arm"]

            def activate():
                nonlocal active_name, writer, alpha, optimizer, sampler, stability, steps, targets
                active_name = cursor["arm"]
                require(active_name == ACTIVE, "alpha099 continuation is not authorized")
                item = arms[active_name]
                writer, alpha, optimizer = item["writer"], item["alpha"], item["optimizer"]
                sampler, stability = item["sampler"], item["stability"]
                steps, targets = item["steps"], item["targets"]
                restore_rng_state(item["rng"], rank=rank, device=device,
                                  tensor_parallel_tracker=get_cuda_rng_tracker())

            def capture_active():
                arms[active_name].update({"steps": steps, "targets": targets, "rng": rng_state()})

            checkpoint_count = 0
            def _checkpoint(*, labels=()):
                nonlocal latest
                payload_arms = {}
                for name, value in arms.items():
                    signature = agree(tensor_state_hash(value["writer"].state_dict()), "checkpoint writer replicas differ")
                    adam_signature = agree(optimizer_hash(value["optimizer"]), "checkpoint AdamW replicas differ")
                    require(value["sampler"].state_dict()["total_loss_tokens"] == value["targets"]
                            and value["stability"].completed_steps == value["steps"], "checkpoint counters differ")
                    rng = gather(value["rng"])
                    if rank == 0:
                        monitor = value["stability"]
                        payload_arms[name] = {
                            "arm": name, "alpha": value["alpha_value"], "optimizer_step": value["steps"],
                            "completed_loss_targets": value["targets"], "writer_sha256": signature,
                            "optimizer_sha256": adam_signature, "writer": clone_cpu(value["writer"].state_dict()),
                            "optimizer": clone_cpu(value["optimizer"].state_dict()),
                            "sampler": value["sampler"].state_dict(), "rng_by_rank": rng,
                            "stability": {
                                "completed_gradients": list(monitor.completed_gradients),
                                "completed_steps": monitor.completed_steps,
                                "gradient_spike_streak": monitor.gradient_spike_streak,
                                "beta_saturation_streak": monitor.beta_saturation_streak,
                                "hard_stop_gradient_spikes": False, "hard_stop_beta_saturation": False}}
                snapshot = timing_snapshot(write=False)
                if rank == 0:
                    require(state_digest(payload_arms[CONTROL]) == manifest["inactive_arm_state_sha256"],
                            "inactive alpha099 full checkpoint state changed")
                    if checkpoint_count == 0:
                        require(all(state_digest(payload_arms[name]) == manifest["initial_arm_state_sha256"][name]
                                    for name, _ in ARMS), "initial restored arm state differs")
                    path = options.output / "joint-latest.pt"
                    temporary = path.with_suffix(".pt.partial")
                    torch.save({
                        "protocol": CHECKPOINT_PROTOCOL, "arms": payload_arms,
                        "schedule_cursor": copy.deepcopy(cursor),
                        "legacy_schedule_cursor": copy.deepcopy(manifest["legacy_schedule_cursor"]),
                        "legacy_completed_stages": copy.deepcopy(manifest["legacy_schedule_cursor"]["completed_stages"]),
                        "legacy_checkpoint_metadata": copy.deepcopy(manifest["legacy_checkpoint_metadata"]),
                        "continuation_policy": continuation_policy,
                        "inactive_arm_state_sha256": manifest["inactive_arm_state_sha256"],
                        "control_evaluation_receipt": manifest["control_evaluation_receipt"],
                        "run_id": manifest["run_id"], "timing": snapshot,
                        "budget": budget(), "runtime_identity_sha256": runtime_hash,
                        "registration_sha256": file_info(marker)["sha256"],
                        "code_manifest_sha256": manifest["code_manifest_sha256"],
                        "data_identity": data.identity, "trajectory_boundary": True,
                        "memory_reset_at_next_trajectory": True, "reader_saved": False,
                        "resume_cli_supported": True, "source_resume": manifest["source_checkpoint"],
                        "source_migration": manifest["migration"],
                        "source_checkpoint_role": manifest["source_checkpoint_role"],
                        "source_model_state_restored": manifest["source_model_state_restored"],
                        "initial_writer": manifest["initial_writer"], "dev64_plan": manifest["dev64_plan"],
                        "optimizer_target_cap_per_arm": ARM_CAP,
                        "cumulative_budget_loss_targets": budget()["total_optimizer_target_accesses"],
                        "evidence_batch_limit": 12, "query_batch_limit": 4,
                    }, temporary)
                    os.replace(temporary, path)
                    latest = {"path": path.name, "schedule_cursor": copy.deepcopy(cursor),
                              "arms": {name: {"step": value["steps"], "actual_loss_targets": value["targets"]}
                                       for name, value in arms.items()},
                              "shared_optimizer_targets": budget()["total_optimizer_target_accesses"],
                              **file_info(path)}
                    if labels:
                        archived = options.output / (
                            "joint-policy-stage-" + str(cursor["policy_stage_index"]) + "-"
                            + "-".join(str(label) for label in labels) + ".pt")
                        require(not archived.exists(), "checkpoint archive already exists")
                        temp_archive = archived.with_suffix(".pt.partial")
                        shutil.copyfile(path, temp_archive)
                        os.replace(temp_archive, archived)
                        archives.append({**latest, "path": archived.name, "labels": list(labels)})
                    write_json(options.output / "checkpoints.json", {"latest": latest, "archives": archives})
                latest = gather(latest)[0]
                dist.barrier()

            def checkpoint(*, labels=()):
                nonlocal checkpoint_count
                activity("checkpoint", lambda: _checkpoint(labels=labels))
                checkpoint_count += 1
                timing_snapshot()

            def write(item, block):
                hidden = writer_evidence_features(block["H_new"].to(device), mode="rmsnorm", eps=1e-6)
                item["state"] = write_native_block(
                    writer, item["state"], hidden, block["physical_rows_new"].to(device),
                    torch.ones(hidden.shape[:2], dtype=torch.bool, device=device), alpha)
                require(bool(item["state"].values.isfinite().all()), "nonfinite writer memory")

            def staged_forward(local, group_targets):
                nonlocal dispatched, query_inputs, evidence_inputs
                cache, queries, phases = {}, [], []
                def evidence_phase():
                    nonlocal evidence_inputs
                    tasks = [(item["index"], ordinal, frame["evidence"])
                             for item in local for ordinal, frame in enumerate(item["plan"])
                             if frame["evidence"] is not None]
                    for start, size in global_frame_batches(gather(len(tasks)), 12):
                        active = tasks[start:start + size]
                        mb = collate_staged_frames(
                            [task[2] for task in active] + [None] * (size - len(active)),
                            kind="evidence", device=device)
                        evidence_inputs += sum(gather(mb["real_input_tokens"]))
                        result = schedule.collect_evidence_microbatches(session, [mb])["microbatches"][0]
                        blocks = crop_evidence_lanes(mb, result["hidden"], result["physical_rows"])
                        for task, block in zip(active, blocks):
                            require(block is not None, "real evidence frame was lost")
                            cache[task[:2]] = block
                _, phase = timed("1_native_evidence_H12_to_CPU", evidence_phase)
                phases.append(phase)
                def writer_phase():
                    for item in local:
                        for ordinal, frame in enumerate(item["plan"]):
                            if frame["query"] is not None:
                                queries.append({"owner": item, "frame": frame["query"], "snapshot": item["state"]})
                            if frame["evidence"] is not None:
                                write(item, cache.pop((item["index"], ordinal)))
                        item["state"] = None
                    require(not cache, "cached evidence was not consumed")
                _, phase = timed("2_causal_writer_histories", writer_phase)
                phases.append(phase)
                def query_phase():
                    nonlocal dispatched, query_inputs
                    for start, size in global_frame_batches(gather(len(queries)), 4):
                        active = queries[start:start + size]
                        mb = collate_staged_frames(
                            [task["frame"] for task in active] + [None] * (size - len(active)),
                            kind="query", device=device)
                        costs = gather((mb["loss_target_tokens"], mb["real_input_tokens"]))
                        cost = sum(item[0] for item in costs)
                        def reserve_and_persist(count):
                            nonlocal dispatched, query_inputs
                            dispatched += count
                            dispatched_by_arm[active_name] += count
                            query_inputs += sum(item[1] for item in costs)
                            save_budget()
                        def native_query():
                            tickets = [task["owner"]["bridge"].prepare_query(
                                materialize_staged_snapshot(task["snapshot"])) for task in active]
                            tickets += [None] * (size - len(tickets))
                            packed = pack_staged_query_tickets(
                                tickets, row_capacity=session.layout.row_capacity, device=device)
                            mb["reader_state"] = packed.reader_state
                            result = schedule.schedule_query_microbatches(
                                session, [mb], group_total_target_tokens=group_targets)["microbatches"][0]
                            require(tuple(result["loss_target_tokens_by_lane"]) == mb["loss_target_tokens_by_lane"],
                                    "native query target counts differ")
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
                        dispatch_query(
                            continuation_policy, accounting["charged"][ACTIVE] + dispatched,
                            accounting["model"][ACTIVE] + dispatched, cost, reserve_and_persist, native_query)
                        del mb
                _, phase = timed("3_native_query_Q4_backward", query_phase)
                phases.append(phase)
                queries.clear()
                return phases

            activate()
            checkpoint(labels=("migration" if options.resume_shared_cap_final else "resume",))
            stop_reason, next_group_cost, blocked_events = (
                None, cursor.get("next_complete_group_target_cost"), None)
            while True:
                if options.stop_after_group_file:
                    stop = gather(options.stop_after_group_file.exists() if rank == 0 else None)[0]
                    if stop:
                        stop_reason = "stop_after_group_file"
                        break
                validate_policy_cursor(cursor)
                if cursor["phase"] == "complete":
                    stop_reason = cursor["terminal_reason"]
                    break
                if cursor["phase"] == "evaluation_pending":
                    cursor["evaluation_attempt"] += 1
                    capture_active()
                    checkpoint()
                    attempt = cursor["evaluation_attempt"]
                    evaluation_milestone = (targets if cursor["evaluation_kind"] == "actual_budget_endpoint"
                                            else cursor["milestone"])
                    eval_output = options.output / "evaluations" / (
                        f"policy-stage-{cursor['policy_stage_index']:02d}-{active_name}-"
                        f"{cursor['evaluation_kind']}-{evaluation_milestone:09d}-attempt-{attempt:03d}")
                    before_writer = tensor_state_hash(writer.state_dict())
                    before_optimizer = optimizer_hash(optimizer)
                    before_sampler = canonical_hash(sampler.state_dict())
                    before_rng = rng_state()
                    def charge_evaluation(count):
                        nonlocal evaluation_dispatched
                        require(type(count) is int and count > 0, "invalid DEV64 CE charge")
                        agree(count, "ranks disagree on DEV64 target charge")
                        evaluation_dispatched += count
                        save_budget()
                    report = activity("evaluation", lambda: evaluate_dev64(
                        session, writer, alpha, plan_path=options.dev64_plan, arm=active_name,
                        milestone=evaluation_milestone, output=eval_output,
                        charge_targets=charge_evaluation,
                        checkpoint_metadata={"joint_checkpoint": latest, "optimizer_step": steps,
                                             "completed_loss_targets": targets,
                                             "nominal_model_milestone": cursor["milestone"],
                                             "evaluation_kind": cursor["evaluation_kind"],
                                             "budget_basis": continuation_policy["budget_basis"],
                                             "cumulative_arm_optimizer_targets": budget()["cumulative_optimizer_target_accesses_by_arm"][ACTIVE],
                                             "runtime_identity_sha256": runtime_hash}))
                    require(tensor_state_hash(writer.state_dict()) == before_writer
                            and optimizer_hash(optimizer) == before_optimizer
                            and canonical_hash(sampler.state_dict()) == before_sampler,
                            "DEV64 changed training state")
                    # The evaluator restores RNG itself; explicitly restore the
                    # registered arm state again before checkpoint/switch.
                    restore_rng_state(before_rng, rank=rank, device=device,
                                      tensor_parallel_tracker=get_cuda_rng_tracker())
                    capture_active()
                    cursor["completed_stages"].append({
                        "policy_stage_index": cursor["policy_stage_index"], "arm": active_name,
                        "nominal_model_targets": cursor["milestone"], "actual_model_targets": targets,
                        "evaluation_kind": cursor["evaluation_kind"], "evaluation_milestone": evaluation_milestone,
                        "optimizer_step": steps, "evaluation_output": str(eval_output),
                        "evaluation_report_sha256": canonical_hash(report)})
                    cursor["policy_stage_index"] += 1
                    cursor["evaluation_attempt"] = 0
                    if cursor["terminal_after_evaluation"]:
                        cursor["phase"] = "complete"
                    else:
                        require(cursor["policy_stage_index"] < len(MILESTONES), "milestone sequence exhausted without terminal evaluation")
                        cursor["milestone"] = MILESTONES[cursor["policy_stage_index"]]
                        cursor["phase"] = "training"
                        cursor["evaluation_kind"] = None
                        cursor["terminal_reason"] = None
                    if not cursor["terminal_after_evaluation"]:
                        cursor.pop("next_complete_group_target_cost", None)
                    checkpoint()
                    if rank == 0:
                        write_json(options.output / "progress.json", {
                            "status": "alpha100_DEV64_complete", "schedule_cursor": cursor,
                            "budget": budget(), "latest_checkpoint": latest,
                            "timing": last_timing,
                            "updated_at_utc": datetime.now(timezone.utc).isoformat()})
                    continue
                require(cursor["phase"] == "training", "invalid stage phase")
                pre_sampler = copy.deepcopy(sampler.state_dict())
                group = sampler.next_batch(12)
                require(len(group) == 12, "sampler exhausted before supplying 12 whole trajectories")
                group_targets = sum(item.loss_tokens for item in group)
                before_budget = budget()["total_optimizer_target_accesses"]
                arm_charged = accounting["charged"][ACTIVE] + dispatched
                fits_cap = cap_allows(continuation_policy, arm_charged, targets, group_targets)
                crosses_milestone = targets + group_targets > cursor["milestone"]
                if crosses_milestone or not fits_cap:
                    sampler.load_state_dict(pre_sampler)
                    cursor["phase"] = "evaluation_pending"
                    cursor["next_complete_group_target_cost"] = group_targets
                    cursor["evaluation_kind"] = ("model_milestone" if crosses_milestone else "actual_budget_endpoint")
                    cursor["terminal_after_evaluation"] = not fits_cap or cursor["milestone"] == ARM_CAP
                    cursor["terminal_reason"] = (
                        f"{continuation_policy['budget_basis']}_target_cap_complete_group_boundary"
                        if cursor["terminal_after_evaluation"] else None)
                    next_group_cost = group_targets
                    capture_active()
                    checkpoint(labels=(active_name, cursor["evaluation_kind"],
                                       cursor["milestone"] if crosses_milestone else targets))
                    if rank == 0:
                        write_json(options.output / "progress.json", {
                            "status": "DEV64_pending", "schedule_cursor": cursor,
                            "budget": budget(), "latest_checkpoint": latest,
                            "actual_model_targets": targets, "actual_arm_charged_targets": arm_charged,
                            "updated_at_utc": datetime.now(timezone.utc).isoformat()})
                    continue
                event_counts = event_repeat_counts(sampler.state_dict())
                agree(canonical_hash(event_counts), "ranks disagree on event exposure")
                if not all(row["within_limit"] for row in event_counts):
                    sampler.load_state_dict(pre_sampler)
                    next_group_cost = group_targets
                    blocked_events = [row for row in event_counts if not row["within_limit"]]
                    stop_reason = "event_three_pass_limit_group_boundary"
                    break
                plans = [data.frames(item) for item in group]
                for trajectory, plan in zip(group, plans):
                    validate_frame_plan(plan, trajectory.loss_tokens)
                assignments = assign_trajectories(plans)
                identity = [{"id": item.document_id, "group_id": item.group_id, "source": item.source,
                             "blocks": item.blocks, "start": item.start, "loss_tokens": item.loss_tokens}
                            for item in group]
                agree(canonical_hash({"group": identity, "assignments": assignments}),
                      "ranks selected different optimizer groups")
                optimizer.zero_grad(set_to_none=True)
                beta_counts[:] = [0, 0]
                local = [{"index": index, "plan": plans[index],
                          "state": LazyDecayState.from_batch(session.layout.empty_state(1, device)),
                          "bridge": NativeTrajectoryGradients(
                              total_loss_tokens=group_targets, expected_loss_tokens=group[index].loss_tokens,
                              policy=policy, runtime_identity_sha256=runtime_hash),
                          "expected_targets": group[index].loss_tokens, "loss_sum": 0., "loss_targets": 0}
                         for index in assignments[rank]]
                group_started = time.monotonic()
                phases = staged_forward(local, group_targets)
                def update():
                    for item in local:
                        item["bridge"].backward_writer()
                        item["bridge"] = item["state"] = None
                    require(all(item["loss_targets"] == item["expected_targets"] for item in local),
                            "trajectory target accounting differs")
                    reduction = sum_ep4_writer_gradients(writer, group=dist.group.WORLD)
                    norm = reduction["global_gradient_l2"]
                    require(math.isfinite(norm) and norm > 0, "zero or nonfinite global writer gradient")
                    betas = gather(tuple(beta_counts))
                    stops = stability.observe_step(
                        norm, sum(item[0] for item in betas), sum(item[1] for item in betas))
                    require(not stops, "nonfinite optimizer observation")
                    torch.nn.utils.clip_grad_norm_(writer.parameters(), 1., error_if_nonfinite=True, foreach=False)
                    optimizer.step()
                    require(all(bool(value.isfinite().all()) for value in writer.parameters()),
                            "nonfinite writer after AdamW")
                    stability.complete_step(norm)
                    session.assert_frozen()
                    return norm
                norm, phase = timed("4_joint_writer_VJP_DP_SUM_clip_AdamW", update)
                phases.append(phase)
                completed += group_targets
                completed_by_arm[active_name] += group_targets
                require(completed == dispatched, "completed group did not account for every query dispatch")
                steps += 1
                targets += group_targets
                capture_active()
                loss_sum = math.fsum(gather(math.fsum(item["loss_sum"] for item in local)))
                del local
                save_budget()
                after_budget = budget()["total_optimizer_target_accesses"]
                checkpoint()
                row = {"event": "staged_alpha100_group_complete", "arm": active_name, "step": steps,
                       "loss_target_tokens": group_targets, "cumulative_model_loss_targets": targets,
                       "policy_stage_index": cursor["policy_stage_index"], "stage_milestone": cursor["milestone"],
                       "lifetime_optimizer_targets": after_budget,
                       "active_arm_charged_targets": budget()["cumulative_optimizer_target_accesses_by_arm"][ACTIVE],
                       "budget_basis": continuation_policy["budget_basis"], "mean_nll": loss_sum / group_targets,
                       "loss_sum": loss_sum, "raw_gradient_l2": norm, "stability": stability.last_observation,
                       "group_identity_sha256": canonical_hash(identity), "phases": phases,
                       "event_exposure": event_counts,
                       "seconds_including_checkpoint": max(gather(time.monotonic() - group_started)),
                       "timing": last_timing, "latest_checkpoint": latest}
                if rank == 0:
                    with (options.output / "training.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    write_json(options.output / "progress.json", {
                        "status": "training", "last_group": row, "budget": budget(),
                        "schedule_cursor": cursor,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat()})
                    print(json.dumps({key: row[key] for key in (
                        "event", "arm", "step", "cumulative_model_loss_targets",
                        "lifetime_optimizer_targets", "mean_nll", "seconds_including_checkpoint")}), flush=True)
            capture_active()
            finished = cursor["phase"] == "complete"
            exit_status = "complete" if finished else "stopped"
            checkpoint(labels=("final" if finished else "stopped",))
            save_budget()
            timing_snapshot(status="complete" if finished else "stopped")
            if rank == 0:
                write_json(options.output / ("COMPLETE.json" if finished else "STOPPED.json"), {
                    "protocol": PROTOCOL, "status": stop_reason, "schedule_cursor": cursor,
                    "arms": {name: {"optimizer_step": value["steps"], "completed_loss_targets": value["targets"]}
                             for name, value in arms.items()},
                    "budget": budget(), "latest_checkpoint": latest,
                    "budget_basis": continuation_policy["budget_basis"],
                    "unused_active_arm_budget": ARM_CAP - budget()["per_arm_cap_progress"][ACTIVE],
                    "actual_active_model_targets": targets,
                    "actual_active_charged_targets": budget()["cumulative_optimizer_target_accesses_by_arm"][ACTIVE],
                    "nominal_model_target_cap": ARM_CAP,
                    "model_target_tail": ARM_CAP - targets,
                    "next_complete_group_target_cost": next_group_cost,
                    "blocked_event_exposure": blocked_events,
                    "event_exposure_by_arm": {name: require_event_limit(value["sampler"].state_dict())
                                              for name, value in arms.items()},
                    "runtime_identity_sha256": runtime_hash, "additional_diagnostic_CE_targets": 0,
                    "timing": last_timing, "resume_cli_supported": True})
            dist.barrier()
    exit_status = "complete"
    try:
        run_ep4_session(options, run)
    except BaseException as error:
        exit_status = "failed"
        save_budget()
        if rank == 0:
            write_json(options.output / "FAILURE.json", {
                "protocol": PROTOCOL, "status": "stopped", "error_type": type(error).__name__,
                "message": str(error), "budget": budget(), "automatic_retry": False,
                "timing": last_timing, "rank_exit_timing_files": "timing-rank-0.json through timing-rank-3.json",
                "resume": "use_last_complete_checkpoint_and_this_latest_budget_ledger_in_a_new_output_directory"})
        raise
    finally:
        if options.output.is_dir():
            write_json(options.output / f"timing-rank-{rank}.json", local_timing(exit_status))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("resume-checkpoint", "source-budget-ledger", "source-runtime-identity",
                 "cache-root", "model-path", "verification", "delta-repo", "output",
                 "initial-writer", "dev64-plan", "source-input-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume-shared-cap-final", action="store_true")
    parser.add_argument("--source-complete", type=Path)
    parser.add_argument("--source-timing-ledger", type=Path)
    parser.add_argument("--budget-basis", required=True, choices=("charged", "model"))
    parser.add_argument("--stop-after-group-file", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate", action="store_true")
    action.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    for key, value in vars(options).items():
        if isinstance(value, Path):
            setattr(options, key, value.resolve())
    sys.path.insert(0, str(options.delta_repo / "src"))
    source, prior, manifest = prepare(options)
    if options.validate:
        print(json.dumps({"status": "staged_alpha100_continuation_inputs_valid", "manifest": manifest,
                          "new_CE_targets": 0, "optimizer_updates": 0}, indent=2))
        return
    execute(options, source, prior, manifest)


if __name__ == "__main__":
    main()

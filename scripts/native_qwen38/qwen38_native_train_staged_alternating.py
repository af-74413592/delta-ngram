"""Alternate independent alpha1/alpha.99 arms at 20M model-token stages.

Each completed group saves both arms and the stage cursor. At each milestone,
archive, evaluate the frozen DEV64 plan, then switch arms. Training,
evaluation, and historical diagnostic CE have independent persistent ledgers.
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

PROTOCOL = "qwen38_native_staged_alternating_training_v1"
CHECKPOINT_PROTOCOL = "qwen38_native_staged_alternating_checkpoint_v1"
CAP = 100000000
STAGE = 20000000
ARMS = (("alpha100_rmsnorm", 1.), ("alpha099_rmsnorm", .99))
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
    scripts = Path(__file__).resolve().parent
    paths = [scripts / name for name in (
        "qwen38_native_train_staged_alternating.py", "qwen38_native_staged_schedule.py",
        "qwen38_native_dev64_eval.py", "qwen38_native_dev64_prepare.py",
        "qwen38_native_ep4_session.py", "qwen38_native_ep4_schedule.py",
        "qwen38_native_train_batched.py", "qwen38_native_batched_schedule.py",
        "qwen38_native_frozen_session.py", "qwen38_native_runtime_preflight.py",
        "qwen38_native_norm_probe.py", "qwen38_native_dispatch.py",
        "qwen38_native_router.py", "qwen38_native_bounded_checks.py")]
    paths.append(repo / "scripts" / "prepare_qwen38_native_data.py")
    paths += sorted((repo / "src" / "delta_ngram").rglob("*.py"))
    return {path.relative_to(repo).as_posix(): file_info(path) for path in paths}


def prepare(options):
    import torch
    source = torch.load(options.resume_checkpoint, map_location="cpu", weights_only=False)
    prior = json.loads(options.source_budget_ledger.read_bytes())
    source_runtime = json.loads(options.source_runtime_identity.read_bytes())
    require(source.get("trajectory_boundary") is True
            and source.get("memory_reset_at_next_trajectory") is True,
            "requires a completed trajectory-boundary checkpoint")
    require(canonical_hash(source_runtime) == source["runtime_identity_sha256"],
            "source runtime identity does not match its checkpoint")
    for key in ("total_optimizer_target_accesses", "total_diagnostic_target_accesses"):
        require(type(prior[key]) is int and prior[key] >= source["budget"][key],
                "continuation cannot refund previous target accesses")
    require(0 <= prior["total_optimizer_target_accesses"] <= CAP
            and prior["total_diagnostic_target_accesses"] >= 59620, "invalid shared budget")
    prior_evaluation = prior.get("total_evaluation_target_accesses",
                                prior.get("evaluation_target_accesses", 0))
    require(type(prior_evaluation) is int and prior_evaluation >= 0, "invalid evaluation ledger")
    require(prior_evaluation >= source["budget"].get("total_evaluation_target_accesses", 0),
            "continuation cannot refund evaluation accesses")
    initial_info, plan_info = file_info(options.initial_writer), file_info(options.dev64_plan)
    require(initial_info["sha256"] == INITIAL_SHA, "immutable seed17 initial writer differs")
    codes = code_manifest(options.delta_repo)
    code_hash = canonical_hash(codes)
    fresh_run = options.performance_source47 or options.restart_from_initial
    if fresh_run:
        require(source.get("protocol") == "qwen38_native_batched_main_checkpoint_v1"
                and source["arm"] == ARMS[0][0] and source["alpha"] == 1.
                and source["optimizer_step"] == 47 and source["completed_loss_targets"] == 1107675,
                "new runs require the original step47 calibration/corpus source")
        require(prior["total_optimizer_target_accesses"] == 1468915
                and prior["total_diagnostic_target_accesses"] == 59620,
                "new runs must carry the final authorized historical ledger")
        arm_sources = {ARMS[0][0]: source}
    else:
        require(source.get("protocol") == CHECKPOINT_PROTOCOL
                and source.get("code_manifest_sha256") == code_hash
                and source.get("dev64_plan") == plan_info
                and source.get("initial_writer") == initial_info,
                "alternating continuation requires the exact saved code, DEV64 and initial writer")
        arm_sources = source["arms"]
        require(set(arm_sources) == {name for name, _ in ARMS}, "joint checkpoint needs both arms")
        cursor = source["schedule_cursor"]
        index = cursor["stage_index"]
        require(type(index) is int and index >= 0
                and cursor["arm"] == ARMS[index % 2][0]
                and cursor["milestone"] == (index // 2 + 1) * STAGE
                and cursor["phase"] in ("training", "evaluation_pending")
                and type(cursor["evaluation_attempt"]) is int and cursor["evaluation_attempt"] >= 0
                and len(cursor["completed_stages"]) == index,
                "joint checkpoint stage cursor is inconsistent")
    for name, saved in arm_sources.items():
        require(saved["sampler"]["total_loss_tokens"] == saved["completed_loss_targets"]
                and saved["stability"]["completed_steps"] == saved["optimizer_step"],
                "checkpoint sampler/model/optimizer counters differ")
        require(saved["arm"] == name
                and saved["alpha"] == dict(ARMS)[name], "checkpoint arm/alpha differs")
        require(len(saved["rng_by_rank"]) == 4
                and sorted(item["rank"] for item in saved["rng_by_rank"]) == list(range(4)),
                "each arm checkpoint needs all four rank RNG states")
    require(options.output != options.resume_checkpoint.parent,
            "use a new output directory; the source run remains immutable")
    if os.environ.get("RANK") in (None, "0"):
        require(not options.output.exists(), "output must be a new directory")
    manifest = {
        "protocol": PROTOCOL, "code": codes, "code_manifest_sha256": code_hash,
        "source_checkpoint": file_info(options.resume_checkpoint),
        "source_budget_ledger": file_info(options.source_budget_ledger),
        "source_runtime_identity": file_info(options.source_runtime_identity),
        "source_runtime_identity_sha256": source["runtime_identity_sha256"],
        "source_arms": {name: {"optimizer_step": value["optimizer_step"],
                               "completed_loss_targets": value["completed_loss_targets"],
                               "writer_sha256": value["writer_sha256"]}
                       for name, value in arm_sources.items()},
        "initial_writer": initial_info, "dev64_plan": plan_info,
        "cache_complete": file_info(options.cache_root / "COMPLETE.json"),
        "cache_manifest": file_info(options.cache_root / "data_manifest.json"),
        "migration": ("restart_both_arms_from_seed17_initial" if options.restart_from_initial
                      else "performance_source47_plus_seed17_alpha099" if options.performance_source47
                      else "exact_alternating_resume"),
        "source_checkpoint_role": ("calibration_and_corpus_identity_only" if options.restart_from_initial
                                   else "alpha1_training_continuation" if options.performance_source47
                                   else "joint_training_continuation"),
        "source_model_state_restored": not options.restart_from_initial,
        "new_run_arm_initialization": (
            "both_seed17_initial_weights; copy_only_calibrated_target_scale; fresh_AdamW_sampler_RNG_stability; steps_targets_zero"
            if options.restart_from_initial else "as_recorded_in_migration"),
        "topology": {"TP": 1, "PP": 1, "EP": 4, "ETP": 1, "DP": 4, "CP": 1},
        "global_trajectories_per_group": 12, "evidence_batch_limit": 12, "query_batch_limit": 4,
        "shared_optimizer_target_cap": CAP,
        "prior_optimizer_target_accesses": prior["total_optimizer_target_accesses"],
        "prior_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"],
        "budget_endpoint": "last_complete_12_trajectory_group_at_or_below_shared_cap; no_shorter_reselection",
        "arms": [{"name": name, "alpha": value} for name, value in ARMS],
        "stage_model_target_stride": STAGE,
        "stage_order": "alpha1_to_20M,alpha099_to_20M,alpha1_to_40M,alpha099_to_40M,...",
        "stage_boundary": "last_whole_12_trajectory_group_at_or_below_each_arm_milestone",
        "checkpoint": "atomic_joint_latest_every_group; archive_each_20M_arm_stage_then_DEV64",
        "evaluation": "fixed_DEV64_between_stages; independent_evaluation_target_accesses; failure_stops_pending_node",
        "writer_math": "unchanged_D1_FP32", "deterministic_algorithm_policy": "unchanged_native_default",
        "stages": ["native_evidence_H12_to_CPU", "causal_writer_histories_and_snapshots",
                   "native_query_Q4_backward", "joint_writer_VJP_DP_SUM_clip1_AdamW"],
        "evidence_LM_head": "omit_unused_logits_after_capturing_final_native_H",
        "optimizer": {"name": "AdamW", "lr": .0003, "betas": [.9, .999],
                      "eps": 1e-8, "weight_decay": 0., "clip": 1.},
        "additional_diagnostic_CE_targets": 0, "paired_runs": False, "evaluations": True,
        "automatic_retry": False, "runtime_equivalence_claimed": False,
    }
    return source, prior, manifest


def execute(options, source, prior, manifest):
    from qwen38_native_ep4_session import run_ep4_session
    rank = int(os.environ.get("RANK", "-1"))
    require(0 <= rank < 4 and int(os.environ.get("WORLD_SIZE", "0")) == 4,
            "use four torchrun ranks")
    marker = options.output / "input_manifest.json"
    dispatched = completed = query_inputs = evidence_inputs = 0
    evaluation_dispatched = 0
    prior_evaluation = prior.get("total_evaluation_target_accesses",
                                prior.get("evaluation_target_accesses", 0))
    dispatched_by_arm = {name: 0 for name, _ in ARMS}
    completed_by_arm = {name: 0 for name, _ in ARMS}
    def budget():
        return {
            "protocol": "qwen38_native_staged_alternating_ledger_v1",
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
            "shared_optimizer_target_cap": CAP, "new_diagnostic_target_accesses": 0,
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
        nonlocal dispatched, completed, query_inputs, evidence_inputs, evaluation_dispatched
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

        device, started = session.device, time.monotonic()
        latest = None
        archives = []
        cursor = ({"stage_index": 0, "arm": ARMS[0][0], "milestone": STAGE,
                   "phase": "training", "evaluation_attempt": 0, "completed_stages": []}
                  if options.performance_source47 or options.restart_from_initial
                  else copy.deepcopy(source["schedule_cursor"]))
        def gather(value):
            values = [None] * 4
            dist.all_gather_object(values, value)
            return values
        def agree(value, message):
            values = gather(value)
            require(all(item == values[0] for item in values), message)
            return values[0]
        def rng_state():
            return {"rank": rank, "python": random.getstate(), "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state(device).cpu(),
                    "tensor_parallel": clone_cpu(get_cuda_rng_tracker().get_states())}
        def timed(label, action):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            begin = time.monotonic()
            result = action()
            torch.cuda.synchronize(device)
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
                       "DEV64_plan": manifest["dev64_plan"]}
            runtime_hash = agree(canonical_hash(runtime), "runtime identity differs across ranks")
            if not (options.performance_source47 or options.restart_from_initial):
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
            initial = torch.load(options.initial_writer, map_location="cpu", weights_only=True)
            require(torch.equal(initial["target_scale"], torch.ones(16)),
                    "immutable seed17 initial writer must have unit target scales")
            initial_tp_rng = clone_cpu(get_cuda_rng_tracker().get_states())
            arms = {}
            for name, alpha_value in ARMS:
                saved = (None if options.restart_from_initial
                         else (source if name == ARMS[0][0] else None) if options.performance_source47
                         else source["arms"][name])
                if saved is None:
                    random.seed(17)
                    np.random.seed(17)
                    torch.manual_seed(17)
                    torch.cuda.manual_seed_all(17)
                    get_cuda_rng_tracker().set_states(copy.deepcopy(initial_tp_rng))
                sampler = data.sampler(seed=17)
                writer, alpha = make_qwen38_writer(alpha=alpha_value)
                writer, alpha = writer.to(device).float().train(), alpha.to(device)
                optimizer = torch.optim.AdamW(writer.parameters(), lr=.0003, betas=(.9, .999),
                                              eps=1e-8, weight_decay=0., foreach=False, fused=False)
                stability = BoundedStability(hard_stop_gradient_spikes=False, hard_stop_beta_saturation=False)
                if saved is None:
                    writer.load_state_dict(initial, strict=True)
                    with torch.no_grad():
                        writer.target_scale.copy_(source["writer"]["target_scale"].to(device))
                    steps = targets = 0
                else:
                    require(saved["arm"] == name and saved["alpha"] == alpha_value, "arm checkpoint identity differs")
                    steps, targets = restore_checkpoint_state(
                        saved, writer=writer, optimizer=optimizer, sampler=sampler, stability=stability)
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
                require(active_name in arms, "invalid active arm")
                item = arms[active_name]
                writer, alpha, optimizer = item["writer"], item["alpha"], item["optimizer"]
                sampler, stability = item["sampler"], item["stability"]
                steps, targets = item["steps"], item["targets"]
                restore_rng_state(item["rng"], rank=rank, device=device,
                                  tensor_parallel_tracker=get_cuda_rng_tracker())

            def capture_active():
                arms[active_name].update({"steps": steps, "targets": targets, "rng": rng_state()})

            def checkpoint(*, labels=()):
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
                if rank == 0:
                    path = options.output / "joint-latest.pt"
                    temporary = path.with_suffix(".pt.partial")
                    torch.save({
                        "protocol": CHECKPOINT_PROTOCOL, "arms": payload_arms,
                        "schedule_cursor": copy.deepcopy(cursor),
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
                        "shared_optimizer_target_cap": CAP,
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
                            "joint-stage-" + str(cursor["stage_index"]) + "-"
                            + "-".join(str(label) for label in labels) + ".pt")
                        require(not archived.exists(), "checkpoint archive already exists")
                        temp_archive = archived.with_suffix(".pt.partial")
                        shutil.copyfile(path, temp_archive)
                        os.replace(temp_archive, archived)
                        archives.append({**latest, "path": archived.name, "labels": list(labels)})
                    write_json(options.output / "checkpoints.json", {"latest": latest, "archives": archives})
                latest = gather(latest)[0]
                dist.barrier()

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
                        require(cost > 0 and prior["total_optimizer_target_accesses"] + dispatched + cost <= CAP,
                                "query dispatch exceeds shared optimizer budget")
                        dispatched += cost
                        dispatched_by_arm[active_name] += cost
                        query_inputs += sum(item[1] for item in costs)
                        save_budget()
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
                        del packed, tickets, result, mb
                _, phase = timed("3_native_query_Q4_backward", query_phase)
                phases.append(phase)
                queries.clear()
                return phases

            activate()
            checkpoint(labels=("restart" if options.restart_from_initial
                               else "migration" if options.performance_source47 else "resume",))
            stop_reason, next_group_cost = "shared_budget_complete_group_boundary", None
            while True:
                if options.stop_after_group_file:
                    stop = gather(options.stop_after_group_file.exists() if rank == 0 else None)[0]
                    if stop:
                        stop_reason = "stop_after_group_file"
                        break
                if cursor["phase"] == "evaluation_pending":
                    cursor["evaluation_attempt"] += 1
                    capture_active()
                    checkpoint()
                    attempt = cursor["evaluation_attempt"]
                    eval_output = options.output / "evaluations" / (
                        f"stage-{cursor['stage_index']:02d}-{active_name}-"
                        f"nominal-{cursor['milestone']:09d}-attempt-{attempt:03d}")
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
                    report = evaluate_dev64(
                        session, writer, alpha, plan_path=options.dev64_plan, arm=active_name,
                        milestone=cursor["milestone"], output=eval_output,
                        charge_targets=charge_evaluation,
                        checkpoint_metadata={"joint_checkpoint": latest, "optimizer_step": steps,
                                             "completed_loss_targets": targets,
                                             "runtime_identity_sha256": runtime_hash})
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
                        "stage_index": cursor["stage_index"], "arm": active_name,
                        "nominal_model_targets": cursor["milestone"], "actual_model_targets": targets,
                        "optimizer_step": steps, "evaluation_output": str(eval_output),
                        "evaluation_report_sha256": canonical_hash(report)})
                    cursor["stage_index"] += 1
                    cursor["arm"] = ARMS[cursor["stage_index"] % 2][0]
                    cursor["milestone"] = (cursor["stage_index"] // 2 + 1) * STAGE
                    cursor["phase"] = "training"
                    cursor["evaluation_attempt"] = 0
                    checkpoint()
                    activate()
                    if rank == 0:
                        write_json(options.output / "progress.json", {
                            "status": "switched_arm_after_DEV64", "schedule_cursor": cursor,
                            "budget": budget(), "latest_checkpoint": latest,
                            "updated_at_utc": datetime.now(timezone.utc).isoformat()})
                    continue
                require(cursor["phase"] == "training", "invalid stage phase")
                pre_sampler = copy.deepcopy(sampler.state_dict())
                group = sampler.next_batch(12)
                require(len(group) == 12, "sampler exhausted before supplying 12 whole trajectories")
                group_targets = sum(item.loss_tokens for item in group)
                before_budget = budget()["total_optimizer_target_accesses"]
                if targets + group_targets > cursor["milestone"]:
                    sampler.load_state_dict(pre_sampler)
                    cursor["phase"] = "evaluation_pending"
                    cursor["next_complete_group_target_cost"] = group_targets
                    capture_active()
                    checkpoint(labels=(active_name, "nominal", cursor["milestone"]))
                    if rank == 0:
                        write_json(options.output / "progress.json", {
                            "status": "DEV64_pending", "schedule_cursor": cursor,
                            "budget": budget(), "latest_checkpoint": latest,
                            "updated_at_utc": datetime.now(timezone.utc).isoformat()})
                    continue
                if before_budget + group_targets > CAP:
                    sampler.load_state_dict(pre_sampler)
                    next_group_cost = group_targets
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
                row = {"event": "staged_alternating_group_complete", "arm": active_name, "step": steps,
                       "loss_target_tokens": group_targets, "cumulative_model_loss_targets": targets,
                       "stage_index": cursor["stage_index"], "stage_milestone": cursor["milestone"],
                       "shared_optimizer_targets": after_budget, "mean_nll": loss_sum / group_targets,
                       "loss_sum": loss_sum, "raw_gradient_l2": norm, "stability": stability.last_observation,
                       "group_identity_sha256": canonical_hash(identity), "phases": phases,
                       "seconds": max(gather(time.monotonic() - group_started)), "latest_checkpoint": latest}
                if rank == 0:
                    with (options.output / "training.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    write_json(options.output / "progress.json", {
                        "status": "training", "last_group": row, "budget": budget(),
                        "schedule_cursor": cursor,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat()})
                    print(json.dumps({key: row[key] for key in (
                        "event", "arm", "step", "cumulative_model_loss_targets",
                        "shared_optimizer_targets", "mean_nll", "seconds")}), flush=True)
            capture_active()
            checkpoint(labels=("final" if stop_reason != "stop_after_group_file" else "stopped",))
            save_budget()
            if rank == 0:
                write_json(options.output / ("STOPPED.json" if stop_reason == "stop_after_group_file" else "COMPLETE.json"), {
                    "protocol": PROTOCOL, "status": stop_reason, "schedule_cursor": cursor,
                    "arms": {name: {"optimizer_step": value["steps"], "completed_loss_targets": value["targets"]}
                             for name, value in arms.items()},
                    "budget": budget(), "latest_checkpoint": latest,
                    "unused_shared_optimizer_budget": CAP - budget()["total_optimizer_target_accesses"],
                    "next_complete_group_target_cost": next_group_cost,
                    "runtime_identity_sha256": runtime_hash, "additional_diagnostic_CE_targets": 0,
                    "seconds": time.monotonic() - started, "resume_cli_supported": True})
            dist.barrier()
    try:
        run_ep4_session(options, run)
    except BaseException as error:
        save_budget()
        if rank == 0:
            write_json(options.output / "FAILURE.json", {
                "protocol": PROTOCOL, "status": "stopped", "error_type": type(error).__name__,
                "message": str(error), "budget": budget(), "automatic_retry": False,
                "resume": "use_last_complete_checkpoint_and_this_latest_budget_ledger_in_a_new_output_directory"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("resume-checkpoint", "source-budget-ledger", "source-runtime-identity",
                 "cache-root", "model-path", "verification", "delta-repo", "output",
                 "initial-writer", "dev64-plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--performance-source47", action="store_true")
    start.add_argument("--restart-from-initial", action="store_true")
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
        print(json.dumps({"status": "staged_alternating_inputs_valid", "manifest": manifest,
                          "new_CE_targets": 0, "optimizer_updates": 0}, indent=2))
        return
    execute(options, source, prior, manifest)


if __name__ == "__main__":
    main()

"""Full-pool, causal native writer training after the registered preparation.

One frozen TP2/PP2 reader serves two sequential fixed-alpha arms. Each logical
optimizer group contains up to 12 whole trajectories, executed as B1 schedules.
Reader query graphs are disposable; writer histories retain full trajectory
BPTT. No calibration, diagnostic CE, validation selection, or retry occurs here.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

from qwen38_native_norm_probe import canonical_hash, digest, file_info, require, write_json
from qwen38_native_frozen_session import run_frozen_session
from qwen38_native_paired_norm_probe import collect_native_evidence
from qwen38_native_runtime_preflight import schedule_query


ARMS = (("alpha100_rmsnorm", 1.), ("alpha099_rmsnorm", .99))


def training_contract():
    return {
        "protocol": "qwen38_native_full_pool_causal_training_v1",
        "arms": [{"name": name, "alpha": alpha} for name, alpha in ARMS],
        "arm_order": "sequential_alpha100_then_alpha099",
        "seed": 17, "loss_target_cap_per_arm": 2000000,
        "shared_optimizer_target_cap": 100000000,
        "trajectories_per_optimizer_group": 12, "physical_reader_batch": 1,
        "optimizer_group_denominator": "sum_actual_loss_targets_of_whole_trajectories",
        "sampler": "existing_RawWindowSampler_token_fair_variable_cost_v3",
        "block_size": 512, "prefix_tokens": 128,
        "article_start_query_bos_token_id": 248044,
        "evidence_article_start": "natural_tokens_without_synthetic_BOS",
        "warmup": "ordinary_first_block_or_entire_first_event_article",
        "causality": "query_before_current_block_write; omit_final_trajectory_write",
        "memory": "fresh_per_trajectory; continuous_across_articles_of_same_event",
        "writer_gradient": "one_joint_full_trajectory_VJP_after_all_queries",
        "reader_boundary": "materialized_effective_FP32_delta_leaf",
        "TP_replica_policy": {"comparison_rtol": 1e-6, "comparison_atol": 1e-7,
                              "canonical_replica": "TP_group_rank0",
                              "at": ["each_reader_cotangent", "optimizer_group_writer_gradient"],
                              "reduction": "checked_selection_not_sum_or_mean"},
        "norm_mode": "rmsnorm", "norm_epsilon": 1e-6, "rho": .01,
        "target_scales": "immutable_shared_vector_from_accepted_preparation",
        "runtime": {"router": "always_sorted_topk", "HC": "eager", "PLE": "split"},
        "optimizer": {"name": "AdamW", "lr": .0003, "betas": [.9, .999],
                      "eps": 1e-8, "weight_decay": 0., "gradient_clip": 1.},
        "checkpoint_nominal_targets": [0, 500000, 1000000, 2000000],
        "checkpoint_boundary": "first_completed_group_crossing_threshold; final_at_or_below_cap",
        "latest_checkpoint": "atomic_overwrite_after_every_completed_optimizer_group",
        "restart_budget": "per_arm_2M_cap_includes_all_previous_dispatched_targets",
        "restart_model": "same_initial_writer_seed17; previous_spend_is_not_model_training_progress",
        "stops": {"gradient": "monitor_only_two_consecutive_gt10x_previous5_completed_median",
                  "beta": "monitor_only_two_consecutive_saturated_fraction_gt0.01",
                  "beta_saturated": "beta_lt0.001_or_gt0.999",
                  "nonfinite": "immediate", "zero_gradient": "stop_before_optimizer",
                  "wall_time": "config_run_max_wall_seconds"},
        "additional_diagnostic_or_probe_CE_targets": 0,
        "automatic_extension": False, "automatic_retry": False,
    }


def code_manifest(delta_repo):
    scripts = Path(__file__).resolve().parent
    names = ("qwen38_native_train.py",
             "qwen38_native_norm_probe.py", "qwen38_native_paired_norm_probe.py",
             "qwen38_native_runtime_preflight.py", "qwen38_native_frozen_session.py",
             "qwen38_native_tp_gradient_probe.py", "qwen38_native_router.py",
             "qwen38_native_dispatch.py", "qwen38_native_bounded_checks.py")
    paths = [scripts / name for name in names]
    paths += [Path(delta_repo) / "src" / "delta_ngram" / name for name in (
        "qwen38_training.py", "qwen38_train_data.py", "qwen38_native.py", "qwen38_adapter.py",
        "qwen38_writer.py", "qwen38_precision.py", "qwen38_probe.py",
        "alpha_memory.py", "batching.py", "state.py", "operators.py",
        "raw_corpora.py", "data/event_streams.py", "data/raw_text.py")]
    return {"protocol": "qwen38_main_training_code_manifest_v1",
            "files": {path.relative_to(Path(delta_repo).resolve()).as_posix(): file_info(path)
                      for path in paths}}


class OptimizationLedger:
    """Charge actual CE dispatches before execution, including failed groups."""

    def __init__(self, *, prior_optimizer, prior_diagnostic, prior_by_arm=None):
        require(type(prior_optimizer) is int and 0 <= prior_optimizer <= 100000000,
                "invalid prior optimizer target count")
        require(type(prior_diagnostic) is int and prior_diagnostic >= 0,
                "invalid prior diagnostic target count")
        self.prior_optimizer = prior_optimizer
        self.prior_diagnostic = prior_diagnostic
        self.prior_by_arm = ({name: 0 for name, _ in ARMS}
                             if prior_by_arm is None else dict(prior_by_arm))
        require(set(self.prior_by_arm) == {name for name, _ in ARMS}
                and all(type(value) is int and 0 <= value <= 2000000
                        for value in self.prior_by_arm.values())
                and sum(self.prior_by_arm.values()) <= prior_optimizer,
                "invalid prior per-arm target accounting")
        self.dispatched = {name: 0 for name, _ in ARMS}
        self.completed = {name: 0 for name, _ in ARMS}
        self.evidence_inputs = 0
        self.query_inputs = 0

    def reserve_query(self, arm, targets, inputs):
        require(arm in self.dispatched and type(targets) is int and targets > 0,
                "invalid optimizer query reservation")
        require(self.prior_by_arm[arm] + self.dispatched[arm] + targets <= 2000000,
                "per-arm optimizer target cap exceeded")
        require(self.prior_optimizer + sum(self.dispatched.values()) + targets <= 100000000,
                "shared optimizer target cap exceeded")
        self.dispatched[arm] += targets
        self.query_inputs += inputs

    def complete_group(self, arm, targets):
        require(self.completed[arm] + targets == self.dispatched[arm],
                "group completion must account for every dispatched target exactly once")
        self.completed[arm] += targets

    def report(self):
        return {
            "protocol": "qwen38_main_optimization_ledger_v1",
            "prior_optimizer_target_accesses": self.prior_optimizer,
            "prior_diagnostic_target_accesses": self.prior_diagnostic,
            "prior_optimizer_targets_by_arm": dict(self.prior_by_arm),
            "optimizer_targets_dispatched_by_arm": dict(self.dispatched),
            "optimizer_targets_in_completed_groups_by_arm": dict(self.completed),
            "cumulative_optimizer_targets_by_arm": {
                name: self.prior_by_arm[name] + self.dispatched[name] for name, _ in ARMS},
            "total_optimizer_target_accesses": self.prior_optimizer + sum(self.dispatched.values()),
            "total_diagnostic_target_accesses": self.prior_diagnostic,
            "new_diagnostic_target_accesses": 0, "query_input_tokens": self.query_inputs,
            "evidence_input_tokens": self.evidence_inputs,
            "failed_or_partial_queries_are_not_refunded": True,
        }


def checkpoint_thresholds(previous, current, *, final=False):
    require(0 <= previous <= current <= 2000000, "invalid checkpoint token counters")
    crossed = [value for value in (500000, 1000000) if previous < value <= current]
    if final:
        crossed.append(2000000)
    return crossed


def validate_frame_plan(frames, expected_targets):
    require(bool(frames) and frames[0]["query"] is None and frames[-1]["evidence"] is None,
            "trajectory must start with warmup and omit its final write")
    observed = 0
    for item in frames:
        for key in ("query", "evidence"):
            frame = item[key]
            if frame is None:
                continue
            length = len(frame["input_ids"])
            require(2 <= length <= 640 and length % 2 == 0,
                    "native frame must be SP-even and at most 640 tokens")
        query = item["query"]
        if query is not None:
            count = sum(value != -100 for value in query["labels"])
            require(count == query["loss_target_tokens"] and count > 0,
                    "query label count differs from registered frame targets")
            require(not any(query["write_mask"]), "queries may not write memory")
            observed += count
    require(observed == expected_targets, "trajectory target count differs from sampler cost")
    return observed


def execute_frame_plan(frames, state, *, query, write):
    """Keep predict-before-write ordering explicit and independently testable."""
    reports = []
    for item in frames:
        if item["query"] is not None:
            reports.append(query(state, item["query"]))
        if item["evidence"] is not None:
            state = write(state, item["evidence"])
    return state, reports


def validate_inputs(options):
    import yaml
    registration = json.loads(options.registration.read_bytes())
    require(registration.get("protocol") == "qwen38_native_main_training_v1"
            and registration.get("execution_allowed") is True
            and registration.get("contract") == training_contract(),
            "main training registration or explicit contract differs")
    required = registration["required_evidence"]
    authorization = registration.get("preparation_authorization", {})
    direct = authorization.get("mode") == "user_authorized_direct_training"
    paths = {
        "initial_writer_sha256": options.initial_writer,
        "tp_gradient_receipt_sha256": options.tp_gradient_receipt,
        "cache_complete_sha256": options.cache_root / "COMPLETE.json",
        "cache_manifest_sha256": options.cache_root / "data_manifest.json",
        "config_sha256": options.config,
    }
    preparation_key = "preparation_authorization_sha256" if direct else "preparation_complete_sha256"
    paths[preparation_key] = options.preparation_root / (
        "PREPARATION_AUTHORIZATION.json" if direct else "PREPARATION_COMPLETE.json")
    for key, path in paths.items():
        require(digest(path) == required[key], f"registered evidence changed: {key}")
    codes = code_manifest(options.delta_repo)
    require(canonical_hash(codes) == required["code_manifest_sha256"],
            "main runner or dependency source changed after registration")
    preparation = json.loads(paths[preparation_key].read_bytes())
    if direct:
        require(preparation.get("status") == "user_authorized_direct_training"
                and preparation.get("protocol") == "qwen38_main_training_authorization_v1"
                and preparation.get("authorization") == authorization
                and authorization.get("source_preparation_status") == "stopped_at_bitwise_replica_gate"
                and authorization.get("uncompleted_checks") == [
                    "writer_fixed_cotangent_fd", "end_to_end_directional"]
                and isinstance(authorization.get("user_instruction"), str)
                and bool(authorization["user_instruction"]),
                "direct main training requires the explicit user authorization receipt")
        for name in ("fresh_scale_calibration.json", "fresh_observation.json",
                     "runtime_identity.json", "budget_ledger.json"):
            require(file_info(options.preparation_root / name) == preparation["artifacts"][name],
                    f"authorized preparation artifact changed: {name}")
        require(preparation["initial_writer_sha256"] == digest(options.initial_writer)
                and preparation["budget"] == json.loads(
                    (options.preparation_root / "budget_ledger.json").read_bytes()),
                "direct authorization initial writer or budget differs")
    else:
        require(preparation.get("status") == "preparation_complete"
                and preparation.get("protocol") == "qwen38_main_training_preparation_v1"
                and preparation.get("optimizer_steps") == 0
                and preparation.get("separate_eight_step_stage_executed") is False
                and preparation.get("reader_unchanged") is True,
                "accepted zero-optimizer main preparation is required")
        for key in ("writer_fd", "end_to_end_directional", "actual_training_forward_comparison"):
            require(preparation[key].get("status") == "accepted",
                    f"preparation requirement was not accepted: {key}")
        final_path = options.preparation_root / "final_registration.json"
        require(file_info(final_path) == preparation["final_registration"],
                "preparation final registration changed")
        final = json.loads(final_path.read_bytes())
        frozen = final["required_evidence"]
        for key, name in (
                ("scale_calibration_receipt_sha256", "fresh_scale_calibration.json"),
                ("H_T0_raw_target_observation_sha256", "fresh_observation.json"),
                ("writer_fixed_cotangent_gradient_receipt_sha256", "writer_fd/COMPLETE.json"),
                ("end_to_end_directional_receipt_sha256", "end_to_end_directional.json"),
                ("actual_training_forward_comparison_receipt_sha256", "actual_training_forward_comparison.json")):
            require(digest(options.preparation_root / name) == frozen[key],
                    f"accepted preparation artifact changed: {name}")
    scale = json.loads((options.preparation_root / "fresh_scale_calibration.json").read_bytes())
    runtime = json.loads((options.preparation_root / "runtime_identity.json").read_bytes())
    require((direct or scale == preparation["fresh_scale_calibration"])
            and scale["rho"] == .01 and scale["historical_H_or_scales_reused"] is False
            and canonical_hash(runtime) == preparation["runtime_identity_sha256"]
            == scale["runtime_identity_sha256"],
            "preparation scale and runtime identities disagree")
    observation = json.loads((options.preparation_root / "fresh_observation.json").read_bytes())
    require(observation["writer_initial_parameters_sha256"] == digest(options.initial_writer),
            "initial writer differs from the writer used for fresh calibration")
    vector = scale["target_scale_float32_by_mode"]["rmsnorm"]
    require(len(vector) == 16 and all(math.isfinite(x) and x > 0 for x in vector),
            "invalid prepared per-head scale vector")
    prior = registration["budgets"]
    budget = preparation["budget"]
    prior_by_arm = prior.get("prior_optimizer_targets_by_arm", {name: 0 for name, _ in ARMS})
    OptimizationLedger(prior_optimizer=prior["prior_optimizer_target_accesses"],
                       prior_diagnostic=prior["prior_diagnostic_target_accesses"],
                       prior_by_arm=prior_by_arm)
    require(prior["prior_optimizer_target_accesses"]
            == budget["global_writer_optimizer_target_accesses"] + sum(prior_by_arm.values())
            and prior["prior_diagnostic_target_accesses"] == budget["global_diagnostic_target_accesses"],
            "main budget does not continue preparation and prior per-arm consumption")
    require(prior["prior_optimizer_target_accesses"]
            + sum(2000000 - value for value in prior_by_arm.values()) <= 100000000,
            "both registered arms do not fit the shared optimizer budget")
    config = yaml.safe_load(options.config.read_text(encoding="utf-8"))
    require(config["seed"] == 17 and config["writer"]["hidden_dim"] == 256
            and config["writer"]["right_window"] == 4
            and config["train"]["lr"] == .0003 and config["train"]["gradient_clip"] == 1.
            and config["raw"]["block_size"] == 512 and config["raw"]["prefix_tokens"] == 128
            and config["raw"]["batch_size"] * config["raw"]["gradient_accumulation_steps"] == 12
            and config["raw"]["loss_tokens"] == 2000000,
            "main config differs from the fixed optimizer/frame contract")
    from qwen38_native_runtime_preflight import MODEL_REPO, MODEL_REVISION
    require(config["model"]["name_or_path"] == MODEL_REPO
            and config["model"]["revision"] == MODEL_REVISION
            and config["writer"]["input_norm"] == "rmsnorm"
            and config["writer"]["input_norm_epsilon"] == 1e-6,
            "model or writer preprocessing differs from accepted preparation")
    require(isinstance(config["run"]["max_wall_seconds"], (int, float))
            and math.isfinite(config["run"]["max_wall_seconds"])
            and config["run"]["max_wall_seconds"] > 0, "invalid main wall-time limit")
    return registration, preparation, runtime, vector, config, codes


def execute(options, validated):
    registration, preparation, accepted_runtime, scales, config, codes = validated
    rank = int(os.environ.get("RANK", "-1"))
    require(rank >= 0 and int(os.environ.get("WORLD_SIZE", "0")) == 4,
            "main execution requires torchrun with exactly four ranks")
    marker = options.output / "registration.json"
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        write_json(marker, registration)
        write_json(options.output / "code_manifest.json", codes)
    else:
        deadline = time.monotonic() + 300
        while not marker.is_file():
            require(time.monotonic() < deadline, "rank0 output preparation timed out")
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
        from delta_ngram.qwen38_probe import policy_from_receipt, tensor_state_hash
        from delta_ngram.qwen38_training import (
            NativeTrajectoryGradients, checked_tp_replica_gradient, writer_evidence_features)
        from qwen38_native_bounded_checks import BoundedStability
        from qwen38_native_dispatch import native_hc_dispatch
        from qwen38_native_router import native_router_dispatch
        from qwen38_native_tp_gradient_probe import live_runtime_identity
        from delta_ngram.qwen38_train_data import NativeFullTrainingData

        first, device = session.adapter is not None, session.device
        started = time.monotonic()
        prior = registration["budgets"]
        ledger = OptimizationLedger(prior_optimizer=prior["prior_optimizer_target_accesses"],
                                    prior_diagnostic=prior["prior_diagnostic_target_accesses"],
                                    prior_by_arm=prior.get("prior_optimizer_targets_by_arm"))
        checkpoints, summaries, latest_checkpoints = [], [], {}

        def gathered(value):
            values = [None] * dist.get_world_size()
            dist.all_gather_object(values, value)
            return values

        def agree(value, message):
            values = gathered(value)
            require(all(item == values[0] for item in values), message)
            return values[0]

        def first_agree(value, message):
            values = [item for item in gathered(value if first else None) if item is not None]
            require(len(values) == 2 and values[0] == values[1], message)
            return values[0]

        def save_ledger():
            if rank == 0:
                write_json(options.output / "budget_ledger.json", ledger.report())

        def check_wall():
            require(time.monotonic() - started < config["run"]["max_wall_seconds"],
                    "registered main wall-time limit reached")

        scope = ExitStack()
        try:
            hc = scope.enter_context(native_hc_dispatch(unwrap=True))
            router = scope.enter_context(native_router_dispatch(session.raw))
            precision = scope.enter_context(Qwen38SplitPLEPrecision(session.adapter)) if first else None
            precision_identity = first_agree(precision.identity() if first else None,
                                             "split precision differs across TP")
            live = {
                "native": live_runtime_identity(session.models),
                "precision": precision_identity, "router": router["identity"],
                "HC_dispatch": hc.identity,
                "TP_structure_provenance": accepted_runtime["TP_structure_provenance"],
                "memory_boundary": accepted_runtime["memory_boundary"],
            }
            runtime_hash = agree(canonical_hash(live), "runtime identities differ across ranks")
            expected_live = copy.deepcopy(accepted_runtime)
            transition = registration.get("runtime_source_transition")
            if transition is not None:
                module = "delta_ngram.qwen38_training"
                require(registration.get("preparation_authorization", {}).get("mode")
                        == "user_authorized_direct_training"
                        and transition.get("module") == module
                        and transition.get("from_sha256") == accepted_runtime["native"]["sources"][module]
                        and transition.get("to_sha256") == digest(
                            options.delta_repo / "src" / "delta_ngram" / "qwen38_training.py"),
                        "runtime source transition differs from explicit main registration")
                expected_live["native"]["sources"][module] = transition["to_sha256"]
            require(live == expected_live,
                    "live reader runtime differs beyond the registered checked TP replica helper")
            tp = json.loads(options.tp_gradient_receipt.read_bytes())
            require(tp["complete"] is True and tp["status"] == "accepted",
                    "TP structure receipt is not accepted")
            policy = replace(policy_from_receipt(tp["selected_policy_receipt"]),
                             runtime_identity_sha256=runtime_hash)
            require(policy.reduction == "replica" and policy.world_size == 2,
                    "main native training requires the accepted TP2 replica policy")
            if rank == 0:
                write_json(options.output / "runtime_identity.json", live)
                write_json(options.output / "runtime_preparation_provenance.json", {
                    "prepared_runtime_identity_sha256": canonical_hash(accepted_runtime),
                    "main_runtime_identity_sha256": runtime_hash,
                    "source_update": {
                        "module": "delta_ngram.qwen38_training",
                        "prepared_sha256": accepted_runtime["native"]["sources"]["delta_ngram.qwen38_training"],
                        "main_sha256": live["native"]["sources"]["delta_ngram.qwen38_training"],
                        "reason": "user-authorized checked TP0 replica selection; no reader arithmetic change"},
                    "preparation_authorization": registration.get("preparation_authorization"),
                    "uncompleted_prechecks_are_not_claimed_accepted": True})
            # All ranks reconstruct the same sealed sampler; no rank samples
            # independently from runtime-dependent data or reader loss values.
            data = NativeFullTrainingData(
                options.cache_root,
                expected_complete_sha256=registration["required_evidence"]["cache_complete_sha256"],
                expected_manifest_sha256=registration["required_evidence"]["cache_manifest_sha256"])
            agree(canonical_hash(data.identity), "sealed data identity differs across ranks")
            for key in ("block_size", "prefix_tokens", "source_weights", "source_length_weights"):
                require(config["raw"][key] == data.raw[key],
                        f"config and sealed full data disagree: {key}")
            initial = torch.load(options.initial_writer, map_location="cpu", weights_only=True)
            require(torch.equal(initial["target_scale"], torch.ones(16)),
                    "immutable initial writer must contain unit target scales")
            save_ledger()

            def rng_state():
                import numpy as np
                value = {"rank": rank, "python": random.getstate(),
                         "numpy": np.random.get_state(),
                         "torch_cpu": torch.get_rng_state(),
                         "torch_cuda": torch.cuda.get_rng_state(device).cpu(),
                         "tensor_parallel": copy.deepcopy(get_cuda_rng_tracker().get_states())}
                value["tensor_parallel"] = {key: tensor.cpu() for key, tensor in value["tensor_parallel"].items()}
                return value

            def writer_hash(writer):
                return first_agree(tensor_state_hash(writer.state_dict()) if first else None,
                                   "writer parameters/buffers differ across TP replicas")

            def archive_latest(name, step, targets, labels):
                if rank != 0:
                    return
                source = options.output / f"{name}-latest.pt"
                suffix = "-".join(str(value) for value in labels)
                path = options.output / f"{name}-step-{step:05d}-targets-{targets:010d}-nominal-{suffix}.pt"
                require(not path.exists(), "archived checkpoint already exists")
                temporary = path.with_suffix(".pt.partial")
                shutil.copyfile(source, temporary)
                os.replace(temporary, path)
                checkpoints.append({**latest_checkpoints[name], "path": path.name,
                                    "kind": "milestone_archive", "nominal_targets": labels})
                write_json(options.output / "checkpoints.json", checkpoints)

            def checkpoint(writer, optimizer, sampler, stability, name, alpha, step, targets, labels):
                signature = writer_hash(writer)
                rng = gathered(rng_state())
                if rank == 0:
                    path = options.output / f"{name}-latest.pt"
                    temporary = path.with_suffix(".pt.partial")
                    torch.save({
                        "protocol": "qwen38_native_main_checkpoint_v1", "arm": name, "alpha": alpha,
                        "optimizer_step": step, "completed_loss_targets": targets,
                        "prior_dispatched_loss_targets": ledger.prior_by_arm[name],
                        "cumulative_budget_loss_targets": ledger.prior_by_arm[name] + targets,
                        "current_run_loss_target_cap": 2000000 - ledger.prior_by_arm[name],
                        "nominal_milestones_are_in_checkpoint_manifest": True,
                        "writer_sha256": signature,
                        "writer": {key: value.detach().cpu() for key, value in writer.state_dict().items()},
                        "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(),
                        "rng_by_rank": rng, "stability": {
                            "completed_gradients": list(stability.completed_gradients),
                            "completed_steps": stability.completed_steps,
                            "gradient_spike_streak": stability.gradient_spike_streak,
                            "beta_saturation_streak": stability.beta_saturation_streak,
                            "hard_stop_gradient_spikes": stability.hard_stop_gradient_spikes,
                            "hard_stop_beta_saturation": stability.hard_stop_beta_saturation},
                        "budget": ledger.report(), "runtime_identity_sha256": runtime_hash,
                        "registration_sha256": digest(marker), "code_manifest_sha256": canonical_hash(codes),
                        "data_identity": data.identity, "trajectory_boundary": True,
                        "reader_saved": False, "memory_reset_at_next_trajectory": True,
                        "resume_cli_supported": False,
                    }, temporary)
                    os.replace(temporary, path)
                    record = {"path": path.name, "kind": "latest", "arm": name, "step": step,
                              "actual_loss_targets": targets,
                              "prior_dispatched_loss_targets": ledger.prior_by_arm[name],
                              "cumulative_budget_loss_targets": ledger.prior_by_arm[name] + targets,
                              "nominal_targets": [],
                              "writer_sha256": signature, **file_info(path)}
                    latest_checkpoints[name] = record
                    checkpoints[:] = [item for item in checkpoints if item["path"] != path.name]
                    checkpoints.append(record)
                    if labels:
                        archive_latest(name, step, targets, labels)
                    write_json(options.output / "checkpoints.json", checkpoints)
                dist.barrier()

            initial_tp_rng = copy.deepcopy(get_cuda_rng_tracker().get_states())
            for name, alpha_value in ARMS:
                import numpy as np
                random.seed(17)
                np.random.seed(17)
                torch.manual_seed(17)
                torch.cuda.manual_seed_all(17)
                get_cuda_rng_tracker().set_states(copy.deepcopy(initial_tp_rng))
                sampler = data.sampler(seed=17)
                stability = BoundedStability(
                    hard_stop_gradient_spikes=False, hard_stop_beta_saturation=False)
                writer = alpha = optimizer = None
                if first:
                    writer, alpha = make_qwen38_writer(alpha=alpha_value)
                    writer.load_state_dict(initial, strict=True)
                    writer = writer.to(device).float().train()
                    alpha = alpha.to(device)
                    with torch.no_grad():
                        writer.target_scale.copy_(torch.tensor(scales, device=device, dtype=torch.float32))
                    optimizer = torch.optim.AdamW(
                        writer.parameters(), lr=.0003, betas=(.9, .999), eps=1e-8,
                        weight_decay=0., amsgrad=False, maximize=False, foreach=False,
                        capturable=False, differentiable=False, fused=False)
                steps = targets = 0
                arm_target_cap = 2000000 - ledger.prior_by_arm[name]
                checkpoint(writer, optimizer, sampler, stability, name, alpha_value, 0, 0, [0])
                while targets < arm_target_cap:
                    check_wall()
                    group = sampler.next_batch(12, remaining_loss_tokens=arm_target_cap - targets)
                    if not group:
                        break
                    group_targets = sum(item.loss_tokens for item in group)
                    group_identity = [{"id": item.document_id, "source": item.source,
                                       "group_id": item.group_id, "blocks": item.blocks,
                                       "loss_tokens": item.loss_tokens, "start": item.start}
                                      for item in group]
                    agree(canonical_hash(group_identity), "sampler groups differ across ranks")
                    require(targets + group_targets <= arm_target_cap, "whole group exceeds remaining arm cap")
                    if first:
                        optimizer.zero_grad(set_to_none=True)
                    group_started = time.monotonic()
                    torch.cuda.reset_peak_memory_stats(device)
                    beta_counts = [0, 0]
                    hook = None
                    if first:
                        def gates(module, args, output):
                            beta = output.detach().float().sigmoid()
                            beta_counts[0] += int(((beta < .001) | (beta > .999)).sum())
                            beta_counts[1] += beta.numel()
                        hook = writer.gate_head[2].register_forward_hook(gates)
                    trajectory_reports = []
                    reader_tp_checks = []
                    try:
                        for trajectory in group:
                            frames = data.frames(trajectory)
                            validate_frame_plan(frames, trajectory.loss_tokens)
                            bridge = NativeTrajectoryGradients(
                                total_loss_tokens=group_targets,
                                expected_loss_tokens=trajectory.loss_tokens,
                                policy=policy, runtime_identity_sha256=runtime_hash) if first else None
                            state = LazyDecayState.from_batch(session.layout.empty_state(1, device)) if first else None

                            def query(current, frame):
                                check_wall()
                                ticket = bridge.prepare_query(current.memory.with_values(
                                    current._lookup_packed(current.rows)[0])) if first else None
                                batch = {key: torch.tensor([frame[key]], dtype=torch.long, device=device)
                                         for key in ("input_ids", "labels", "position_ids")}
                                batch["attention_mask"] = None
                                valid = torch.tensor([frame["encoder_valid_mask"]], dtype=torch.bool, device=device)
                                ledger.reserve_query(name, frame["loss_target_tokens"], len(frame["input_ids"]))
                                save_ledger()
                                result = schedule_query(
                                    session.models, batch, valid_mask=valid,
                                    owner_ids=torch.zeros(1, dtype=torch.long, device=device),
                                    adapter=session.adapter,
                                    reader_state=ticket.reader_state if first else None,
                                    backward=True, mode="query",
                                    group_total_target_tokens=group_targets, query_grad_enabled=True)
                                require(math.isfinite(result["loss_sum"]), "nonfinite native query loss")
                                if first:
                                    reader_tp_checks.append({})
                                    bridge.record_query(
                                        ticket, loss_sum=result["loss_sum"],
                                        loss_tokens=frame["loss_target_tokens"],
                                        reader_loss_divisor=group_targets,
                                        reducer=lambda gradient, selected_policy: checked_tp_replica_gradient(
                                            gradient, selected_policy,
                                            group=ps.get_tensor_model_parallel_group(),
                                            diagnostics=reader_tp_checks[-1]))
                                return {"loss_sum": result["loss_sum"],
                                        "loss_targets": frame["loss_target_tokens"]}

                            def write(current, frame):
                                check_wall()
                                ledger.evidence_inputs += len(frame["input_ids"])
                                block = collect_native_evidence(session, frame)
                                if not first:
                                    return None
                                hidden = writer_evidence_features(
                                    block["H_new"].to(device), mode="rmsnorm", eps=1e-6)
                                rows = block["physical_rows_new"].to(device)
                                updated = write_native_block(
                                    writer, current, hidden, rows,
                                    torch.ones(hidden.shape[:2], dtype=torch.bool, device=device), alpha)
                                require(bool(updated.values.isfinite().all()), "nonfinite writer memory")
                                return updated

                            state, losses = execute_frame_plan(frames, state, query=query, write=write)
                            if first:
                                bridge.backward_writer()
                            trajectory_reports.append({
                                "id": trajectory.document_id, "source": trajectory.source,
                                "group_id": trajectory.group_id, "loss_targets": trajectory.loss_tokens,
                                "queries": len(losses), "writes": sum(item["evidence"] is not None for item in frames),
                                "loss_sum": sum(item["loss_sum"] for item in losses)})
                            del bridge, state, frames
                            if rank == 0:
                                print(json.dumps({"event": "main_trajectory_complete", "arm": name,
                                                  "group": steps + 1, "trajectory": len(trajectory_reports),
                                                  "group_trajectories": len(group),
                                                  "dispatched_arm_targets": ledger.dispatched[name]}), flush=True)
                    finally:
                        if hook is not None:
                            hook.remove()
                    stats = None
                    pattern = [(key, value.grad is not None) for key, value in writer.named_parameters()] if first else None
                    first_agree(pattern, "writer gradient presence differs across TP replicas")
                    if first:
                        params = list(writer.parameters())
                        require(all(value.grad is None or bool(value.grad.isfinite().all()) for value in params),
                                "nonfinite writer gradient")
                        flat = torch.cat([value.grad.reshape(-1) for value in params if value.grad is not None])
                        writer_tp_check = {}
                        canonical = checked_tp_replica_gradient(
                            flat.reshape(-1, 1), policy, group=ps.get_tensor_model_parallel_group(),
                            diagnostics=writer_tp_check).reshape(-1)
                        offset = 0
                        for value in params:
                            if value.grad is not None:
                                value.grad.copy_(canonical[offset:offset + value.numel()].view_as(value))
                                offset += value.numel()
                        del flat, canonical
                        norm = math.sqrt(sum(float(value.grad.detach().double().square().sum())
                                             for value in params if value.grad is not None))
                        stats = {"raw_gradient_l2": norm, "beta_saturated_events": beta_counts[0],
                                 "beta_events": beta_counts[1],
                                 "gradient_sha256": tensor_state_hash({
                                     key: value.grad for key, value in writer.named_parameters()
                                     if value.grad is not None})}
                    tp_diagnostics = [value for value in gathered({
                        "rank": rank, "reader_queries": reader_tp_checks,
                        "writer_group": writer_tp_check} if first else None) if value is not None]
                    stats = first_agree(stats, "writer gradients or beta statistics differ across TP")
                    require(math.isfinite(stats["raw_gradient_l2"]) and stats["raw_gradient_l2"] > 0,
                            "zero or nonfinite writer gradient before optimizer")
                    stops = stability.observe_step(stats["raw_gradient_l2"],
                                                   stats["beta_saturated_events"], stats["beta_events"])
                    if rank == 0:
                        write_json(options.output / f"stability-{name}-{steps + 1:05d}.json",
                                   stability.last_observation)
                    require(not stops, "registered main stability stop: " + "; ".join(stops))
                    check_wall()
                    if first:
                        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True, foreach=False)
                        optimizer.step()
                    finite_writer = first_agree(
                        all(bool(value.isfinite().all()) for value in writer.parameters()) if first else None,
                        "writer state validity differs across TP")
                    require(finite_writer,
                            "nonfinite writer state after optimizer update")
                    stability.complete_step(stats["raw_gradient_l2"])
                    previous, targets, steps = targets, targets + group_targets, steps + 1
                    ledger.complete_group(name, group_targets)
                    signature = writer_hash(writer)
                    session.assert_frozen()
                    loss_sum = sum(item["loss_sum"] for item in trajectory_reports)
                    memory = gathered({
                        "rank": rank,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                        "current_allocated_bytes": torch.cuda.memory_allocated(device)})
                    row = {"event": "main_writer_group_complete", "arm": name, "step": steps,
                           "loss_target_tokens": group_targets, "cumulative_loss_targets": targets,
                           "prior_dispatched_loss_targets": ledger.prior_by_arm[name],
                           "cumulative_budget_loss_targets": ledger.prior_by_arm[name] + targets,
                           "current_run_loss_target_cap": arm_target_cap,
                           "mean_nll": loss_sum / group_targets, "loss_sum": loss_sum,
                           "trajectories": trajectory_reports, "writer_sha256": signature,
                           "gradient": stats, "stability": stability.last_observation,
                           "TP_replica_checks": tp_diagnostics,
                           "CUDA_memory_by_rank": memory,
                           "seconds": time.monotonic() - group_started}
                    save_ledger()
                    labels = checkpoint_thresholds(previous, targets)
                    checkpoint(writer, optimizer, sampler, stability, name, alpha_value, steps, targets, labels)
                    if rank == 0:
                        with (options.output / "training.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
                        write_json(options.output / "progress.json", {
                            "status": "training", "arm": name, "step": steps,
                            "actual_loss_targets": targets, "budget": ledger.report(),
                            "last_group": row, "updated_at_utc": datetime.now(timezone.utc).isoformat()})
                        print(json.dumps({key: row[key] for key in (
                            "event", "arm", "step", "cumulative_loss_targets", "mean_nll", "seconds")}), flush=True)
                # This is the <=2M endpoint. Whole trajectories are never cut
                # to make the displayed counter exactly equal the nominal cap.
                archive_latest(name, steps, targets, [2000000])
                dist.barrier()
                summaries.append({"arm": name, "alpha": alpha_value, "optimizer_steps": steps,
                                  "actual_loss_targets": targets,
                                  "prior_dispatched_loss_targets": ledger.prior_by_arm[name],
                                  "cumulative_budget_loss_targets": ledger.prior_by_arm[name] + targets,
                                  "unused_target_budget": arm_target_cap - targets,
                                  "sampler_status": getattr(sampler, "last_status", None)})
                del writer, alpha, optimizer, sampler
            session.assert_frozen()
            if rank == 0:
                write_json(options.output / "COMPLETE.json", {
                    "status": "complete", "protocol": "qwen38_native_main_training_v1",
                    "runtime_identity_sha256": runtime_hash, "registration": file_info(marker),
                    "arms": summaries, "budget": ledger.report(), "checkpoints": checkpoints,
                    "reader_unchanged": True, "additional_diagnostic_CE_targets": 0,
                    "seconds": time.monotonic() - started})
            dist.barrier()
        except BaseException as error:
            if rank == 0:
                write_json(options.output / "FAILURE.json", {
                    "status": "stopped", "error_type": type(error).__name__, "message": str(error),
                    "budget": ledger.report(), "completed_arms": summaries,
                    "automatic_retry": False})
            raise
        finally:
            scope.close()
    run_frozen_session(options, run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registration", "config", "cache-root", "preparation-root", "initial-writer",
                 "tp-gradient-receipt", "model-path", "verification", "delta-repo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate", action="store_true")
    action.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    for name, value in vars(options).items():
        if isinstance(value, Path):
            setattr(options, name, value.resolve())
    sys.path.insert(0, str(options.delta_repo / "src"))
    validated = validate_inputs(options)
    if options.validate:
        print(json.dumps({"status": "main_training_registration_validated",
                          "registration": file_info(options.registration),
                          "contract": training_contract(), "optimizer_steps": 0,
                          "loss_target_accesses": 0}, indent=2))
        return
    execute(options, validated)


if __name__ == "__main__":
    main()

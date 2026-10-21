"""CPU-only migration, replay and accounting for the registered 80M forks.

No CUDA/model initialization occurs here. A fork is a different experiment,
with a separately charged ledger, never an in-place mainline continuation.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from qwen38_native_norm_probe import canonical_hash, file_info, require
from qwen38_native_train_staged_alpha100 import clone_cpu, optimizer_hash
from qwen38_native_alpha100_policy import state_digest, ACTIVE, CONTROL

PROTOCOL = "qwen38_native_memory_fork_v1"
CHECKPOINT_PROTOCOL = "qwen38_native_memory_fork_checkpoint_v1"
SOURCE_PROTOCOL = "qwen38_native_staged_alpha100_checkpoint_v1"
SOURCE_STEP = 3337
SOURCE_TARGETS = 79_981_959
END_STEP = 4172
END_TARGETS = 99_997_072
GROUPS = END_STEP - SOURCE_STEP
TARGETS = END_TARGETS - SOURCE_TARGETS
RESIDUAL_SEED = 170080
ENDPOINT_EVALUATIONS = 2
NODE_NOMINAL_TARGETS = 90_000_000
SOURCE_CHECKPOINT_SHA256 = "c24b4342c51a072227aa90c8aa370f666626734bd8a5f6ea393598be3f9d59cb"
REFERENCE_CHECKPOINT_SHA256 = "523b951487cee0ec726077db0104a8fc929f4629bdb4ca46604295a4b78fbfe3"
COUNTERS = (
    "optimizer_targets_dispatched", "optimizer_targets_completed",
    "evaluation_ce_targets", "diagnostic_ce_targets", "query_input_tokens",
    "training_encoding_real_tokens", "training_encoding_padded_tokens",
    "training_encoding_calls", "evaluation_encoding_real_tokens",
    "evaluation_encoding_padded_tokens", "evaluation_encoding_calls",
    "diagnostic_encoding_real_tokens", "diagnostic_encoding_padded_tokens",
    "diagnostic_encoding_calls",
)


def paired_loss_policy(*, threshold=.01, warmup_groups=50, window_groups=50, consecutive_windows=3, action="monitor"):
    require(isinstance(threshold, (int, float)) and math.isfinite(threshold) and threshold >= 0,
            "paired-loss threshold must be finite and nonnegative")
    require(type(warmup_groups) is int and warmup_groups >= 0
            and type(window_groups) is int and window_groups > 0
            and type(consecutive_windows) is int and consecutive_windows > 0,
            "invalid paired-loss window configuration")
    require(action == "monitor", "paired loss is monitor-only; stopping requires the user's stop file")
    return {"threshold_nats_per_target": float(threshold), "warmup_groups": warmup_groups,
            "window_groups": window_groups, "consecutive_windows": consecutive_windows,
            "action": "monitor", "automatic_stop": False, "user_decision_required": True,
            "difference": "branch_minus_mainline_mean_NLL_for_identical_group",
            "window": "nonoverlapping_target_weighted_groups_after_warmup"}


def fork_policy(branch, residual_difference_baseline="actual_base", *, paired_loss=None):
    require(branch in ("full", "residual"), "unknown memory fork branch")
    require(residual_difference_baseline in ("actual_base", "matched_base")
            and (branch == "residual" or residual_difference_baseline == "actual_base"),
            "matched_base is only valid for the residual branch")
    paired_loss = paired_loss or paired_loss_policy()
    require(paired_loss == paired_loss_policy(threshold=paired_loss["threshold_nats_per_target"],
        warmup_groups=paired_loss["warmup_groups"], window_groups=paired_loss["window_groups"],
        consecutive_windows=paired_loss["consecutive_windows"], action=paired_loss["action"]),
        "paired-loss policy contains unregistered settings")
    return {
        "protocol": PROTOCOL, "branch": branch, "alpha": 1.,
        "source_step": SOURCE_STEP, "source_model_targets": SOURCE_TARGETS,
        "end_step": END_STEP, "end_model_targets": END_TARGETS,
        "complete_groups": GROUPS, "additional_training_targets": TARGETS,
        "encoding": "detached_prewrite_memory_causal_wave_no_LM_head",
        "encoding_frame": "unchanged_evidence_without_BOS_prefix_read_new_tokens_write",
        "optimizer": "inherited_AdamW" if branch == "full" else "fresh_AdamW_residual_only",
        "optimizer_options": {"lr": .0003, "betas": [.9, .999], "eps": 1e-8,
                              "weight_decay": 0., "clip": 1.},
        "base_writer_trainable": branch == "full", "residual_hidden": 128,
        "residual_difference_baseline": residual_difference_baseline,
        "residual_initialization_seed": RESIDUAL_SEED,
        "residual_output_initialization": "weight_and_bias_exactly_zero",
        "training_RNG": "source80M_rank_RNG_restored_after_branch_construction",
        "control_alpha099": "fixed_original_state_no_forward_no_optimizer",
        "source_evaluation_pending": "skip_only_after_matching_completed80M_receipt",
        "replay": "all835_mainline_group_hashes_costs_and_final_sampler_must_match",
        "checkpoint": "atomic_full_branch_every_complete_group",
        "partial_charges": "persist_before_dispatch_never_refund",
        "automatic_retry": False, "exploratory_after_DEV64": True,
        "diagnostic_G_p99_monitor_threshold": 1.,
        "diagnostic_G_action": "monitor_only_never_automatic_stop",
        "diagnostic_unassessable": "record_and_continue_without_claiming_stability",
        "diagnostic_G_interpretation": "encoding_sensitivity_not_a_contraction_certificate",
        "endpoint_DEV64_runs": ENDPOINT_EVALUATIONS,
        "endpoint_DEV64_checkpoint": "same_immutable_terminal_checkpoint_independent_full_evaluations",
        "endpoint_DEV64_interpretation": "no_conclusion_when_mainline_difference_within_observed_two_run_range",
        "first_group_loss_comparison": "logged_binary64_exact_compare_and_numeric_delta_monitor_only",
        "paired_loss": copy.deepcopy(paired_loss),
        "node_checkpoint": {"nominal_model_targets": NODE_NOMINAL_TARGETS,
                            "boundary": "last_complete_group_at_or_below_nominal", "immutable": True},
        "stop_after_group_file": "explicit_path_or_output_stop-after-group_default; user_requested_only",
        "FINAL_used": False,
    }


def diagnostic_monitor(summary):
    """G is an observation, never a training gate or a contraction proof."""
    result = copy.deepcopy(summary)
    result["threshold_exceeded"] = bool(result.get("stop_requested", False))
    result["stop_requested"] = False
    result["action"] = "monitor_only_continue"
    result["assessable"] = bool(result.get("assessable", False))
    result["rule"] = "record_G_p99_threshold_exceedance; never automatic stop"
    return result


def validate_arm(saved):
    from delta_ngram.qwen38_probe import tensor_state_hash
    require(saved["arm"] == ACTIVE and saved["alpha"] == 1., "requires alpha100")
    require(saved["writer_sha256"] == tensor_state_hash(saved["writer"]),
            "source writer tensor digest differs")
    require(saved["sampler"]["total_loss_tokens"] == saved["completed_loss_targets"],
            "source sampler and model target counts differ")
    require(saved["stability"]["completed_steps"] == saved["optimizer_step"],
            "source stability and optimizer step differ")
    require(sorted(row["rank"] for row in saved["rng_by_rank"]) == list(range(4)),
            "source requires all four rank RNG states")
    for row in saved["rng_by_rank"]:
        require({"rank", "python", "numpy", "torch_cpu", "torch_cuda", "tensor_parallel"}
                <= set(row), "incomplete saved rank RNG")


def validate_source80(source):
    require(source.get("protocol") == SOURCE_PROTOCOL,
            "fork source must use the completed mainline checkpoint protocol")
    require(source.get("trajectory_boundary") is True
            and source.get("memory_reset_at_next_trajectory") is True,
            "fork requires a complete trajectory-group boundary")
    require(set(source["arms"]) == {ACTIVE, CONTROL}, "source must retain both mainline arms")
    saved = source["arms"][ACTIVE]
    validate_arm(saved)
    require(saved["optimizer_step"] == SOURCE_STEP
            and saved["completed_loss_targets"] == SOURCE_TARGETS,
            "requires the actual 79,981,959-target / 3337-step 80M archive")
    cursor = source["schedule_cursor"]
    require(cursor["phase"] == "evaluation_pending" and cursor["arm"] == ACTIVE
            and cursor["milestone"] == 80_000_000,
            "expected the archived pre-DEV64 80M cursor")
    return saved


def evaluation_receipt(source, report_path, complete_path):
    from delta_ngram.p2_initialization import writer_tensor_sha256
    report = json.loads(Path(report_path).read_bytes())
    complete = json.loads(Path(complete_path).read_bytes())
    require(report.get("complete") is True and complete.get("complete") is True
            and report["arm"] == complete["arm"] == ACTIVE
            and report["milestone"] == complete["milestone"] == 80_000_000,
            "80M DEV64 receipt must be complete and match alpha100")
    saved = source["arms"][ACTIVE]
    require(report["checkpoint"]["optimizer_step"] == SOURCE_STEP
            and report["checkpoint"]["completed_loss_targets"] == SOURCE_TARGETS,
            "80M DEV64 evaluated a different state")
    digest = writer_tensor_sha256(saved["writer"])
    require(report["writer_sha256"] == complete["writer_sha256"] == digest
            and complete["report_sha256"] == file_info(report_path)["sha256"],
            "80M DEV64 receipt writer/report identity differs")
    return {"report": {"path": str(report_path), **file_info(report_path)},
            "complete": {"path": str(complete_path), **file_info(complete_path)},
            "canonical_report_sha256": canonical_hash(report),
            "writer_sha256": digest, "skipped_reexecution": True}


def reference_groups(path):
    selected = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("arm") != ACTIVE or not SOURCE_STEP < row.get("step", -1) <= END_STEP:
            continue
        require(row.get("event") == "staged_alpha100_group_complete", "unexpected replay log event")
        selected.append({key: row[key] for key in (
            "step", "loss_target_tokens", "cumulative_model_loss_targets", "group_identity_sha256")})
    require(len(selected) == GROUPS, "mainline replay log must contain exactly 835 groups")
    cumulative = SOURCE_TARGETS
    for index, row in enumerate(selected, SOURCE_STEP + 1):
        require(row["step"] == index and type(row["loss_target_tokens"]) is int
                and row["loss_target_tokens"] > 0, "replay groups are not contiguous")
        cumulative += row["loss_target_tokens"]
        require(cumulative == row["cumulative_model_loss_targets"], "replay cumulative targets differ")
    require(cumulative == END_TARGETS, "mainline replay endpoint differs")
    return selected


def reference_first_group_loss(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row.get("arm") == ACTIVE and row.get("step") == SOURCE_STEP + 1]
    require(len(rows) == 1 and rows[0].get("event") == "staged_alpha100_group_complete",
            "mainline step3338 loss record is missing or duplicated")
    result = {key: rows[0][key] for key in ("step", "group_identity_sha256", "loss_target_tokens", "loss_sum", "mean_nll")}
    require(all(math.isfinite(result[key]) for key in ("loss_sum", "mean_nll"))
            and result["loss_target_tokens"] > 0, "invalid mainline first-group loss")
    return result


def reference_group_losses(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    selected = [row for row in rows if row.get("arm") == ACTIVE and SOURCE_STEP < row.get("step", -1) <= END_STEP]
    require(len(selected) == GROUPS, "paired-loss reference requires all835 mainline groups")
    result = []
    for step, row in enumerate(selected, SOURCE_STEP + 1):
        require(row.get("event") == "staged_alpha100_group_complete" and row["step"] == step
                and type(row["loss_target_tokens"]) is int and row["loss_target_tokens"] > 0
                and all(math.isfinite(row[key]) for key in ("loss_sum", "mean_nll"))
                and row["loss_sum"] / row["loss_target_tokens"] == row["mean_nll"],
                "paired-loss mainline scalar or group identity differs")
        result.append({key: row[key] for key in ("step", "group_identity_sha256", "loss_target_tokens", "loss_sum", "mean_nll")})
    return result


def new_paired_loss_state(prior=None):
    result = copy.deepcopy(prior) if prior is not None else {
        "completed_groups": 0, "cumulative_targets": 0, "cumulative_branch_loss_sum": 0.,
        "cumulative_reference_loss_sum": 0., "window_records": [], "completed_windows": 0,
        "consecutive_alert_windows": 0, "last_complete_window": None, "alert_active": False,
        "stop_requested": False}
    require(result["stop_requested"] is False, "paired loss must never automatically stop training")
    return result


def observe_paired_loss(state, config, reference, *, completed_groups, loss_sum, targets, group_sha256):
    require(completed_groups == state["completed_groups"] + 1
            and reference["step"] == SOURCE_STEP + completed_groups
            and targets == reference["loss_target_tokens"]
            and group_sha256 == reference["group_identity_sha256"],
            "paired-loss observation does not match the next mainline group")
    require(math.isfinite(loss_sum), "nonfinite paired group loss")
    require(config["action"] == "monitor" and config["automatic_stop"] is False,
            "paired-loss action must remain monitor-only")
    branch_nll = loss_sum / targets
    group_delta = branch_nll - reference["mean_nll"]
    state["completed_groups"] = completed_groups
    state["cumulative_targets"] += targets
    state["cumulative_branch_loss_sum"] += loss_sum
    state["cumulative_reference_loss_sum"] += reference["loss_sum"]
    window_completed = None
    if completed_groups > config["warmup_groups"]:
        state["window_records"].append({"fork_group": completed_groups, "targets": targets,
            "branch_loss_sum": float(loss_sum), "reference_loss_sum": reference["loss_sum"]})
        if len(state["window_records"]) == config["window_groups"]:
            records = state["window_records"]
            count = sum(row["targets"] for row in records)
            actual = math.fsum(row["branch_loss_sum"] for row in records) / count
            baseline = math.fsum(row["reference_loss_sum"] for row in records) / count
            difference = actual - baseline
            above = difference > config["threshold_nats_per_target"]
            state["consecutive_alert_windows"] = state["consecutive_alert_windows"] + 1 if above else 0
            state["completed_windows"] += 1
            state["alert_active"] = state["consecutive_alert_windows"] >= config["consecutive_windows"]
            window_completed = {"window_index": state["completed_windows"],
                "first_fork_group": records[0]["fork_group"], "last_fork_group": records[-1]["fork_group"],
                "targets": count, "branch_nll": actual, "mainline_nll": baseline,
                "branch_minus_mainline": difference, "above_threshold": above}
            state["last_complete_window"] = copy.deepcopy(window_completed)
            state["window_records"] = []
    cumulative_branch = state["cumulative_branch_loss_sum"] / state["cumulative_targets"]
    cumulative_reference = state["cumulative_reference_loss_sum"] / state["cumulative_targets"]
    state["stop_requested"] = False
    return {"fork_group": completed_groups, "step": SOURCE_STEP + completed_groups,
        "branch_nll": branch_nll, "mainline_nll": reference["mean_nll"],
        "group_branch_minus_mainline": group_delta,
        "cumulative_branch_nll": cumulative_branch, "cumulative_mainline_nll": cumulative_reference,
        "cumulative_branch_minus_mainline": cumulative_branch - cumulative_reference,
        "cumulative_targets": state["cumulative_targets"],
        "warmup": completed_groups <= config["warmup_groups"],
        "pending_window_groups": len(state["window_records"]), "window_completed": window_completed,
        "last_complete_window": copy.deepcopy(state["last_complete_window"]),
        "consecutive_alert_windows": state["consecutive_alert_windows"], "alert_active": state["alert_active"],
        "threshold_nats_per_target": config["threshold_nats_per_target"],
        "action": "monitor_only_user_decision_required", "stop_requested": False}


def node_checkpoint_boundary(records, nominal_targets=NODE_NOMINAL_TARGETS):
    eligible = [index for index, row in enumerate(records, 1) if row["cumulative_model_loss_targets"] <= nominal_targets]
    require(bool(eligible) and eligible[-1] < len(records), "nominal node must lie inside the replay interval")
    count = eligible[-1]
    return {"nominal_model_targets": nominal_targets, "completed_groups": count,
            "optimizer_step": records[count - 1]["step"],
            "actual_model_targets": records[count - 1]["cumulative_model_loss_targets"],
            "next_group_model_targets": records[count]["cumulative_model_loss_targets"]}


def compare_first_group_loss(reference, *, loss_sum, targets, group_sha256, residual_head_zero, branch="residual"):
    """Exact scalar comparison; archived logs do not contain per-token tensors."""
    require(targets == reference["loss_target_tokens"] and group_sha256 == reference["group_identity_sha256"],
            "first-group loss comparison uses a different replay group")
    actual = {"loss_sum": float(loss_sum), "mean_nll": float(loss_sum / targets)}
    require(all(math.isfinite(value) for value in actual.values()), "nonfinite first-group loss")
    metrics = {key: {"mainline": float(reference[key]), "branch": value,
                    "mainline_binary64_hex": float(reference[key]).hex(), "branch_binary64_hex": value.hex(),
                    "bitwise_equal": float(reference[key]).hex() == value.hex(),
                    "branch_minus_mainline": value - float(reference[key])} for key, value in actual.items()}
    require(branch in ("full", "residual"), "unknown first-group comparison branch")
    return {"step": SOURCE_STEP + 1, "branch": branch, "group_identity_sha256": group_sha256,
            "loss_target_tokens": targets, "residual_head_exactly_zero_before_group": residual_head_zero,
            "scope": "archived group loss_sum and mean_nll binary64; per-token loss tensors unavailable",
            "metrics": metrics, "bitwise_equal": all(row["bitwise_equal"] for row in metrics.values()),
            "bitwise_identity_guaranteed": False, "action": "record_actual_values_and_continue",
            "interpretation": ("direct first-group effect of full memory-aware encoding; equality is not required"
                               if branch == "full" else "zero residual output check against the archived first-group loss"),
            "matched_execution": "Q4 query order/batching, FP32 lane sums and Python group aggregation match mainline",
            "writer_input": ("full intentionally replaces legacy base evidence with causal memory-aware evidence"
                             if branch == "full" else "frozen base retains the mainline H12 frame order and B12 batching"),
            "unproven_numerical_difference_candidates": [
                "memory H12 restores encoding RNG; mainline H12 advances RNG if the native forward consumes it",
                "ordinal writer waves and additional native forwards change CUDA execution/allocation order",
                "shared D1 uses CUDA index_add on repeated addresses without a deterministic-algorithms guarantee"],
            "cause_status": "candidate mechanisms only; a scalar mismatch does not isolate its cause"}


def evaluation_metrics(report):
    require(report.get("complete") is True, "DEV64 report is incomplete")
    metrics = {}
    for row in report["summary"]["retention_curve"]:
        for key, value in row.items():
            if key.endswith("_nll") or key in ("delta_empty", "delta_donor", "direction_delta_self", "direction_delta_donor", "direction_difference"):
                require(isinstance(value, (int, float)) and math.isfinite(value), "nonfinite DEV64 metric")
                metrics[f"d{row['unrelated_article_count']}.{key}"] = float(value)
    require(bool(metrics), "DEV64 report has no NLL metrics")
    return metrics


def reference_endpoint_evaluation(checkpoint, report_path, complete_path):
    from delta_ngram.p2_initialization import writer_tensor_sha256
    report = json.loads(Path(report_path).read_bytes())
    complete = json.loads(Path(complete_path).read_bytes())
    digest = writer_tensor_sha256(checkpoint["arms"][ACTIVE]["writer"])
    require(report.get("complete") is True and complete.get("complete") is True
            and report["arm"] == complete["arm"] == ACTIVE
            and report["checkpoint"]["optimizer_step"] == END_STEP
            and report["checkpoint"]["completed_loss_targets"] == END_TARGETS
            and report["writer_sha256"] == complete["writer_sha256"] == digest
            and complete["report_sha256"] == file_info(report_path)["sha256"],
            "reference endpoint DEV64 identity differs")
    require(any(row.get("optimizer_step") == END_STEP and row.get("actual_model_targets") == END_TARGETS
                and row.get("evaluation_report_sha256") == canonical_hash(report)
                for row in checkpoint["schedule_cursor"]["completed_stages"]),
            "reference endpoint DEV64 is absent from authenticated mainline completion")
    return {"report": {"path": str(report_path), **file_info(report_path)},
            "complete": {"path": str(complete_path), **file_info(complete_path)},
            "canonical_report_sha256": canonical_hash(report), "writer_sha256": digest,
            "metrics": evaluation_metrics(report)}


def completed_evaluation_receipt(report, *, before, after, attempt, run_index, terminal_checkpoint, reference):
    delta = {key: after[key] - before[key] for key in COUNTERS}
    require(all(value >= 0 for value in delta.values()), "DEV64 budget was refunded")
    require(delta["evaluation_ce_targets"] == report["evaluation_target_accesses"]
            == report["expected_evaluation_target_accesses"] > 0,
            "DEV64 completed report differs from independently charged CE budget")
    require(all(value == 0 for key, value in delta.items() if not key.startswith("evaluation_")),
            "DEV64 unexpectedly charged training or diagnostic work")
    metrics = evaluation_metrics(report)
    require(metrics.keys() == reference.keys(), "DEV64 metric coverage differs from mainline")
    return {"run_index": run_index, "attempt": attempt, "report_sha256": canonical_hash(report),
            "terminal_checkpoint": copy.deepcopy(terminal_checkpoint), "budget_delta": delta,
            "metrics": metrics, "branch_minus_mainline": {key: value-reference[key] for key, value in metrics.items()},
            "second_minus_first": None}


def summarize_endpoint_evaluations(receipts, reference):
    require(len(receipts) <= ENDPOINT_EVALUATIONS, "too many completed endpoint evaluations")
    runs = copy.deepcopy(receipts)
    result = {"required_runs": ENDPOINT_EVALUATIONS, "completed_runs": len(runs), "runs": runs,
              "comparison": None, "interpretation": "two repeated executions measure observed numerical variation, not statistical significance"}
    if len(runs) != ENDPOINT_EVALUATIONS:
        return result
    require([row["run_index"] for row in runs] == [1, 2]
            and runs[0]["terminal_checkpoint"] == runs[1]["terminal_checkpoint"]
            and runs[0]["budget_delta"]["evaluation_ce_targets"] == runs[1]["budget_delta"]["evaluation_ce_targets"],
            "repeated DEV64 did not use the same terminal checkpoint and target coverage")
    require(all(row["metrics"].keys() == reference.keys() for row in runs), "repeated DEV64 metric coverage differs")
    differences = {key: runs[1]["metrics"][key] - runs[0]["metrics"][key] for key in reference}
    comparison = {}
    for key, baseline in reference.items():
        values = [row["metrics"][key] for row in runs]
        mean = math.fsum(values) / 2
        within = abs(mean - baseline) <= abs(differences[key])
        comparison[key] = {"mainline": baseline, "run1": values[0], "run2": values[1], "mean": mean,
            "second_minus_first": differences[key], "observed_repeat_range": abs(differences[key]),
            "mean_minus_mainline": mean - baseline, "within_observed_repeat_range": within,
            "conclusion": "inconclusive_within_observed_repeat_range" if within else "exploratory_difference_exceeds_two_run_range"}
    for row in runs:
        row["second_minus_first"] = copy.deepcopy(differences)
    result["comparison"] = comparison
    return result


def group_identity(group):
    return [{"id": item.document_id, "group_id": item.group_id, "source": item.source,
             "blocks": item.blocks, "start": item.start, "loss_tokens": item.loss_tokens}
            for item in group]


def validate_replay_group(group, expected, completed_groups):
    require(len(group) == 12, "replay group must contain 12 whole trajectories")
    cost = sum(item.loss_tokens for item in group)
    require(expected["step"] == SOURCE_STEP + completed_groups + 1
            and cost == expected["loss_target_tokens"]
            and canonical_hash(group_identity(group)) == expected["group_identity_sha256"],
            "sampler group hash, step or target cost differs from mainline")
    return cost


def validate_replay_sampler(sampler, start_state, records, final_state):
    """Dry-run every remaining group, then restore the supplied sampler exactly."""
    old = copy.deepcopy(sampler.state_dict())
    try:
        sampler.load_state_dict(start_state)
        for index, row in enumerate(records):
            validate_replay_group(sampler.next_batch(12), row, index)
        require(sampler.state_dict() == final_state, "mainline final sampler state differs")
        return {"complete_groups": len(records), "additional_targets": TARGETS,
                "records_sha256": canonical_hash(records),
                "final_sampler_sha256": canonical_hash(final_state),
                "optimizer_updates": 0, "CE_targets": 0}
    finally:
        sampler.load_state_dict(old)
        require(sampler.state_dict() == old, "dry run did not restore the sampler")


def stability_state(monitor):
    return {"completed_gradients": list(monitor.completed_gradients),
            **{key: getattr(monitor, key) for key in (
                "completed_steps", "gradient_spike_streak", "beta_saturation_streak",
                "hard_stop_gradient_spikes", "hard_stop_beta_saturation")}}


def restore_stability(monitor, saved):
    monitor.completed_gradients.clear()
    monitor.completed_gradients.extend(saved["completed_gradients"])
    for key, value in saved.items():
        if key != "completed_gradients":
            setattr(monitor, key, value)


def construct_branch(source_arm, branch, *, saved_branch=None, device="cpu",
                     residual_difference_baseline="actual_base"):
    """Construction is isolated from training RNG; only the residual is new."""
    import torch
    from delta_ngram.qwen38_native import make_qwen38_writer
    from delta_ngram.qwen38_probe import tensor_state_hash
    from qwen38_native_bounded_checks import BoundedStability
    fork_policy(branch, residual_difference_baseline)
    if saved_branch is not None:
        require(saved_branch.get("residual_difference_baseline", "actual_base") == residual_difference_baseline,
                "saved residual difference baseline changed")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(RESIDUAL_SEED)
        writer, alpha = make_qwen38_writer(alpha=1.)
        writer.load_state_dict(source_arm["writer"], strict=True)
        if branch == "residual":
            from delta_ngram.qwen38_memory_writer import NativeMemoryResidualWriter
            writer = NativeMemoryResidualWriter(writer, hidden_dim=128,
                                                difference_baseline=residual_difference_baseline)
    writer = writer.to(device).float().train()
    alpha = alpha.to(device)
    optimizer = torch.optim.AdamW([p for p in writer.parameters() if p.requires_grad],
                                 lr=.0003, betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., foreach=False, fused=False)
    monitor = BoundedStability(hard_stop_gradient_spikes=False, hard_stop_beta_saturation=False)
    if saved_branch is not None:
        writer.load_state_dict(saved_branch["writer"], strict=True)
        optimizer.load_state_dict(saved_branch["optimizer"])
        restore_stability(monitor, saved_branch["stability"])
        require(tensor_state_hash(writer.state_dict()) == saved_branch["writer_sha256"]
                and optimizer_hash(optimizer) == saved_branch["optimizer_sha256"],
                "restored fork writer or AdamW differs")
    elif branch == "full":
        optimizer.load_state_dict(source_arm["optimizer"])
        restore_stability(monitor, source_arm["stability"])
        require(optimizer_hash(optimizer) == source_arm["optimizer_sha256"],
                "inherited AdamW differs")
    if branch == "residual":
        require(all(not p.requires_grad for p in writer.base_writer.parameters())
                and tensor_state_hash(writer.base_writer.state_dict()) == source_arm["writer_sha256"],
                "residual branch base writer changed")
        if saved_branch is None:
            require(not optimizer.state, "residual AdamW must begin empty")
    return writer, alpha, optimizer, monitor


def new_ledger(prior=None):
    prior = prior or {}
    ledger = {"protocol": "qwen38_native_memory_fork_ledger_v1",
              **{key: prior.get(key, 0) for key in COUNTERS},
              "failed_or_partial_accesses_are_not_refunded": True}
    require(all(type(ledger[key]) is int and ledger[key] >= 0 for key in COUNTERS),
            "fork ledger counters must be nonnegative integers")
    require(ledger["optimizer_targets_completed"] <= ledger["optimizer_targets_dispatched"],
            "completed targets exceed dispatched targets")
    return ledger


def charge(ledger, kind, count):
    require(kind in COUNTERS and type(count) is int and count >= 0, "invalid fork charge")
    ledger[kind] += count


def capture_optimizer_update(writer, optimizer):
    from delta_ngram.qwen38_probe import tensor_state_hash
    parameters = [p for p in writer.parameters() if p.requires_grad]
    return {"writer_sha256": tensor_state_hash(writer.state_dict()),
            "parameter_steps": [float(optimizer.state.get(p, {}).get("step", 0)) for p in parameters]}


def validate_first_optimizer_update(writer, optimizer, before):
    after = capture_optimizer_update(writer, optimizer)
    parameters = [p for p in writer.parameters() if p.requires_grad]
    require(len(parameters) == len(before["parameter_steps"]) == len(after["parameter_steps"])
            and all(new == old + (p.grad is not None)
                    for p, old, new in zip(parameters, before["parameter_steps"], after["parameter_steps"])),
            "first AdamW step counters did not advance exactly once")
    require(after["writer_sha256"] != before["writer_sha256"],
            "first AdamW step did not change the trainable writer")
    return {"AdamW_parameter_steps_before": before["parameter_steps"],
            "AdamW_parameter_steps_after": after["parameter_steps"],
            "writer_before_sha256": before["writer_sha256"],
            "writer_after_sha256": after["writer_sha256"], "writer_changed": True}


def validate_resume(checkpoint, prior, *, branch, code_hash, replay_hash,
                    residual_difference_baseline="actual_base", paired_loss=None, reference_loss_hash=None):
    require(checkpoint.get("protocol") == CHECKPOINT_PROTOCOL
            and checkpoint["policy"] == fork_policy(branch, residual_difference_baseline, paired_loss=paired_loss)
            and checkpoint["code_manifest_sha256"] == code_hash
            and checkpoint["replay_sha256"] == replay_hash,
            "fork resume code, branch, policy or replay identity differs")
    require(checkpoint.get("trajectory_boundary") is True
            and checkpoint.get("memory_reset_at_next_trajectory") is True,
            "fork resume requires a completed group boundary")
    saved = checkpoint["branch_state"]
    require(saved.get("residual_difference_baseline", "actual_base") == residual_difference_baseline,
            "checkpoint residual difference baseline changed")
    count = checkpoint["completed_groups"]
    if reference_loss_hash is not None:
        require(checkpoint.get("reference_loss_sha256") == reference_loss_hash,
                "paired-loss reference log changed on resume")
        paired = new_paired_loss_state(checkpoint.get("paired_loss_state"))
        require(paired["completed_groups"] == count and paired["cumulative_targets"] == checkpoint["completed_targets"],
                "paired-loss checkpoint progress differs from the model")
    require(type(count) is int and 0 <= count <= GROUPS
            and saved["sampler"]["total_loss_tokens"] == SOURCE_TARGETS + checkpoint["completed_targets"]
            and saved["model_optimizer_step"] == SOURCE_STEP + count
            and saved["optimizer_updates_since_fork"] == count
            and saved["stability"]["completed_steps"] == count + (SOURCE_STEP if branch == "full" else 0),
            "fork checkpoint counters differ")
    new_ledger(prior)
    for key in COUNTERS:
        require(prior[key] >= checkpoint["budget"][key], "resume would refund previous charges")
    require(prior["optimizer_targets_completed"] >= checkpoint["completed_targets"],
            "ledger omits effective completed model targets")
    require(sorted(row["rank"] for row in saved["rng_by_rank"]) == list(range(4)),
            "fork resume needs all rank RNG states")
    receipts = checkpoint.get("evaluation_receipts", [])
    require(len(receipts) <= ENDPOINT_EVALUATIONS
            and [row["run_index"] for row in receipts] == list(range(1, len(receipts) + 1)),
            "completed DEV64 run indices differ")
    if receipts:
        require(count == GROUPS and checkpoint.get("terminal_checkpoint") is not None
                and all(row["terminal_checkpoint"] == checkpoint["terminal_checkpoint"] for row in receipts)
                and sum(row["budget_delta"]["evaluation_ce_targets"] for row in receipts) <= prior["evaluation_ce_targets"],
                "completed DEV64 receipts differ from terminal checkpoint or charged budget")
    if checkpoint.get("phase") == "complete":
        require(len(receipts) == ENDPOINT_EVALUATIONS, "complete fork requires two endpoint DEV64 evaluations")
    return saved

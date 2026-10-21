"""CPU migration/replay tests: real AdamW state, no native model or GPU."""
import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from delta_ngram.alpha_memory import BlockAlpha
from delta_ngram.qwen38_probe import tensor_state_hash
from delta_ngram.qwen38_writer import NativeForwardWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
import qwen38_native_memory_fork_policy as policy


def tiny_factory(*, alpha):
    return (NativeForwardWriter(width=4, input_dim=6, hidden_dim=10, heads_per_order=2,
                                target_scale=torch.ones(4)),
            BlockAlpha(6, kind="fixed", initial_alpha=alpha))


@pytest.fixture
def source_arm(monkeypatch):
    import delta_ngram.qwen38_native as native
    monkeypatch.setattr(native, "make_qwen38_writer", tiny_factory)
    writer, _ = tiny_factory(alpha=1.)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=.0003, betas=(.9, .999),
                                 eps=1e-8, weight_decay=0., foreach=False, fused=False)
    for parameter in writer.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    return {"arm": policy.ACTIVE, "alpha": 1., "writer": copy.deepcopy(writer.state_dict()),
            "writer_sha256": tensor_state_hash(writer.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()), "optimizer_sha256": policy.optimizer_hash(optimizer),
            "optimizer_step": policy.SOURCE_STEP, "completed_loss_targets": policy.SOURCE_TARGETS,
            "sampler": {"total_loss_tokens": policy.SOURCE_TARGETS},
            "stability": {"completed_steps": policy.SOURCE_STEP, "completed_gradients": [1., 2., 3.],
                          "gradient_spike_streak": 0, "beta_saturation_streak": 0,
                          "hard_stop_gradient_spikes": False, "hard_stop_beta_saturation": False},
            "rng_by_rank": [{"rank": rank, "python": (3, (), None), "numpy": np.random.get_state(),
                             "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.zeros(4, dtype=torch.uint8),
                             "tensor_parallel": {}} for rank in range(4)]}


def test_full_migration_preserves_writer_adamw_and_training_rng(source_arm):
    before = torch.get_rng_state().clone()
    writer, _, optimizer, monitor = policy.construct_branch(source_arm, "full")
    assert torch.equal(before, torch.get_rng_state())
    assert tensor_state_hash(writer.state_dict()) == source_arm["writer_sha256"]
    assert policy.optimizer_hash(optimizer) == source_arm["optimizer_sha256"]
    assert all(parameter.requires_grad for parameter in writer.parameters())
    assert monitor.completed_steps == policy.SOURCE_STEP


def test_residual_has_frozen_exact_base_new_adamw_zero_output_and_same_rng(source_arm):
    before = torch.get_rng_state().clone()
    writer, _, optimizer, monitor = policy.construct_branch(source_arm, "residual")
    assert torch.equal(before, torch.get_rng_state())
    assert not optimizer.state and monitor.completed_steps == 0
    assert tensor_state_hash(writer.base_writer.state_dict()) == source_arm["writer_sha256"]
    assert all(not parameter.requires_grad for parameter in writer.base_writer.parameters())
    assert all(parameter.requires_grad for parameter in writer.residual_head.parameters())
    assert torch.count_nonzero(writer.residual_head[-1].weight) == 0
    assert torch.count_nonzero(writer.residual_head[-1].bias) == 0
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in writer.residual_head.parameters()}


@pytest.mark.parametrize("branch", ["full", "residual"])
def test_branch_complete_roundtrip_restores_changed_parameters_and_optimizer(source_arm, branch):
    writer, _, optimizer, monitor = policy.construct_branch(source_arm, branch)
    for parameter in writer.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, .25)
    optimizer.step()
    saved = {"writer": policy.clone_cpu(writer.state_dict()),
             "writer_sha256": tensor_state_hash(writer.state_dict()),
             "optimizer": policy.clone_cpu(optimizer.state_dict()),
             "optimizer_sha256": policy.optimizer_hash(optimizer),
             "stability": policy.stability_state(monitor)}
    restored, _, adamw, restored_monitor = policy.construct_branch(source_arm, branch, saved_branch=saved)
    assert tensor_state_hash(restored.state_dict()) == saved["writer_sha256"]
    assert policy.optimizer_hash(adamw) == saved["optimizer_sha256"]
    assert policy.stability_state(restored_monitor) == saved["stability"]


def test_source_rejects_missing_rng_wrong_endpoint_or_nonboundary(source_arm):
    source = {"protocol": policy.SOURCE_PROTOCOL,
              "arms": {policy.ACTIVE: source_arm, policy.CONTROL: {}},
              "trajectory_boundary": True, "memory_reset_at_next_trajectory": True,
              "schedule_cursor": {"phase": "evaluation_pending", "arm": policy.ACTIVE,
                                  "milestone": 80_000_000}}
    assert policy.validate_source80(source) is source_arm
    for change in ("rng", "targets", "boundary"):
        bad = copy.deepcopy(source)
        if change == "rng":
            bad["arms"][policy.ACTIVE]["rng_by_rank"].pop()
        elif change == "targets":
            bad["arms"][policy.ACTIVE]["completed_loss_targets"] += 1
        else:
            bad["trajectory_boundary"] = False
        with pytest.raises(ValueError):
            policy.validate_source80(bad)


def replay_fixture():
    groups, records, total = [], [], policy.SOURCE_TARGETS
    base, remainder = divmod(policy.TARGETS, policy.GROUPS)
    for index in range(policy.GROUPS):
        cost = base + (index < remainder)
        group = [SimpleNamespace(document_id=f"doc-{index}-{lane}", group_id=f"group-{index}",
                                 source="books", blocks=8, start=0, loss_tokens=cost-11 if lane == 0 else 1)
                 for lane in range(12)]
        total += cost
        groups.append(group)
        records.append({"step": policy.SOURCE_STEP + index + 1,
                        "loss_target_tokens": cost, "cumulative_model_loss_targets": total,
                        "group_identity_sha256": policy.canonical_hash(policy.group_identity(group))})
    class Sampler:
        def __init__(self):
            self.cursor = 0
        def state_dict(self):
            return {"cursor": self.cursor}
        def load_state_dict(self, state):
            self.cursor = state["cursor"]
        def next_batch(self, size):
            assert size == 12
            result = groups[self.cursor]
            self.cursor += 1
            return result
    return Sampler(), records


def test_all835_replay_groups_match_targets_hashes_endpoint_and_restore_sampler(tmp_path):
    sampler, records = replay_fixture()
    sampler.cursor = 17
    result = policy.validate_replay_sampler(sampler, {"cursor": 0}, records, {"cursor": 835})
    assert sampler.cursor == 17
    assert result["additional_targets"] == 20_015_113
    path = tmp_path / "training.jsonl"
    path.write_text("\n".join(json.dumps({**row, "event": "staged_alpha100_group_complete", "arm": policy.ACTIVE})
                              for row in records), encoding="utf-8")
    assert policy.reference_groups(path) == records
    records[432]["group_identity_sha256"] = "wrong"
    with pytest.raises(ValueError, match="sampler group hash"):
        policy.validate_replay_sampler(sampler, {"cursor": 0}, records, {"cursor": 835})
    assert sampler.cursor == 17


def test_resume_carries_failed_charges_but_never_advances_effective_model(source_arm):
    ledger = policy.new_ledger()
    policy.charge(ledger, "optimizer_targets_dispatched", 11)
    checkpoint = {"protocol": policy.CHECKPOINT_PROTOCOL, "policy": policy.fork_policy("full"),
        "code_manifest_sha256": "code", "replay_sha256": "replay", "trajectory_boundary": True,
        "memory_reset_at_next_trajectory": True, "completed_groups": 0, "completed_targets": 0,
        "budget": copy.deepcopy(ledger),
        "branch_state": {"sampler": source_arm["sampler"], "stability": source_arm["stability"],
            "model_optimizer_step": policy.SOURCE_STEP, "optimizer_updates_since_fork": 0,
            "rng_by_rank": source_arm["rng_by_rank"]}}
    policy.charge(ledger, "optimizer_targets_dispatched", 7)
    saved = policy.validate_resume(checkpoint, ledger, branch="full", code_hash="code", replay_hash="replay")
    assert saved["model_optimizer_step"] == policy.SOURCE_STEP
    assert ledger["optimizer_targets_dispatched"] == 18
    ledger["optimizer_targets_dispatched"] = 10
    with pytest.raises(ValueError, match="refund"):
        policy.validate_resume(checkpoint, ledger, branch="full", code_hash="code", replay_hash="replay")


@pytest.mark.parametrize("summary", [
    {"stop_requested": True, "assessable": True, "block_g": {"p99": 2.}},
    {"stop_requested": False, "assessable": False, "block_g": {"p99": None}},
    {"stop_requested": False, "assessable": True, "block_g": {"p99": .5}},
])
def test_encoding_sensitivity_is_monitor_only_even_above_one_or_unassessable(summary):
    original = copy.deepcopy(summary)
    actual = policy.diagnostic_monitor(summary)
    assert actual["stop_requested"] is False
    assert actual["threshold_exceeded"] == summary["stop_requested"]
    assert actual["action"] == "monitor_only_continue" and summary == original
    assert policy.fork_policy("full")["diagnostic_G_action"] == "monitor_only_never_automatic_stop"


@pytest.mark.parametrize("branch", ["full", "residual"])
def test_first_update_receipt_requires_exactly_one_real_adamw_update(source_arm, branch):
    writer, _, optimizer, _ = policy.construct_branch(source_arm, branch)
    for p in writer.parameters():
        if p.requires_grad:
            p.grad = torch.full_like(p, .25)
    before = policy.capture_optimizer_update(writer, optimizer)
    with pytest.raises(ValueError, match="exactly once"):
        policy.validate_first_optimizer_update(writer, optimizer, before)
    optimizer.step()
    receipt = policy.validate_first_optimizer_update(writer, optimizer, before)
    assert receipt["writer_changed"] and receipt["writer_before_sha256"] != receipt["writer_after_sha256"]
    assert all(new == old + 1 for old, new in zip(
        receipt["AdamW_parameter_steps_before"], receipt["AdamW_parameter_steps_after"]))
    optimizer.step()
    with pytest.raises(ValueError, match="exactly once"):
        policy.validate_first_optimizer_update(writer, optimizer, before)


def test_matched_residual_migration_and_restore_preserve_explicit_policy(source_arm):
    writer, _, optimizer, monitor = policy.construct_branch(
        source_arm, "residual", residual_difference_baseline="matched_base")
    assert writer.difference_baseline == "matched_base"
    assert policy.fork_policy("residual", "matched_base")["residual_difference_baseline"] == "matched_base"
    saved = {"writer": policy.clone_cpu(writer.state_dict()), "writer_sha256": tensor_state_hash(writer.state_dict()),
             "optimizer": policy.clone_cpu(optimizer.state_dict()), "optimizer_sha256": policy.optimizer_hash(optimizer),
             "stability": policy.stability_state(monitor), "residual_difference_baseline": "matched_base"}
    restored, _, _, _ = policy.construct_branch(source_arm, "residual", saved_branch=saved,
                                               residual_difference_baseline="matched_base")
    assert restored.difference_baseline == "matched_base"
    with pytest.raises(ValueError, match="baseline changed"):
        policy.construct_branch(source_arm, "residual", saved_branch=saved)
    with pytest.raises(ValueError, match="only valid"):
        policy.construct_branch(source_arm, "full", residual_difference_baseline="matched_base")


def test_first_group_compares_actual_binary64_without_tolerance_or_extra_ce(tmp_path):
    reference = {"arm": policy.ACTIVE, "event": "staged_alpha100_group_complete", "step": 3338,
                 "group_identity_sha256": "same", "loss_target_tokens": 2, "loss_sum": 6., "mean_nll": 3.}
    path = tmp_path / "training.jsonl"
    path.write_text(json.dumps(reference), encoding="utf-8")
    reference = policy.reference_first_group_loss(path)
    equal = policy.compare_first_group_loss(reference, loss_sum=6., targets=2, group_sha256="same", residual_head_zero=True)
    assert equal["bitwise_equal"] and not equal["bitwise_identity_guaranteed"]
    changed = math.nextafter(6., math.inf)
    unequal = policy.compare_first_group_loss(reference, loss_sum=changed, targets=2, group_sha256="same", residual_head_zero=True)
    assert not unequal["bitwise_equal"]
    assert unequal["metrics"]["loss_sum"]["branch_minus_mainline"] == changed - 6.
    assert unequal["metrics"]["loss_sum"]["branch_binary64_hex"] == changed.hex()
    full = policy.compare_first_group_loss(reference, loss_sum=5., targets=2, group_sha256="same",
                                          residual_head_zero=None, branch="full")
    assert full["branch"] == "full" and "direct first-group effect" in full["interpretation"]
    assert full["metrics"]["mean_nll"]["branch_minus_mainline"] == -.5
    with pytest.raises(ValueError, match="different replay group"):
        policy.compare_first_group_loss(reference, loss_sum=6., targets=2, group_sha256="other", residual_head_zero=True)


def fake_dev64(nll):
    return {"complete": True, "evaluation_target_accesses": 100,
            "expected_evaluation_target_accesses": 100,
            "summary": {"retention_curve": [{"unrelated_article_count": 0, "self_nll": nll,
                                              "self_pooled_token_nll": nll + .1}]}}


def test_two_dev64_receipts_charge_independently_and_report_small_difference_inconclusive():
    ledger = policy.new_ledger()
    reference = policy.evaluation_metrics(fake_dev64(3.))
    terminal = {"path": "/same/branch-terminal.pt", "sha256": "same"}
    receipts = []
    for index, nll in enumerate((3.001, 2.999), 1):
        before = copy.deepcopy(ledger)
        policy.charge(ledger, "evaluation_ce_targets", 100)
        policy.charge(ledger, "evaluation_encoding_real_tokens", 250)
        receipts.append(policy.completed_evaluation_receipt(fake_dev64(nll), before=before, after=ledger,
            attempt=index, run_index=index, terminal_checkpoint=terminal, reference=reference))
        if index == 1:
            assert policy.summarize_endpoint_evaluations(receipts, reference)["comparison"] is None
    result = policy.summarize_endpoint_evaluations(receipts, reference)
    assert ledger["evaluation_ce_targets"] == 200
    assert result["completed_runs"] == result["required_runs"] == 2
    comparison = result["comparison"]["d0.self_nll"]
    assert comparison["conclusion"] == "inconclusive_within_observed_repeat_range"
    assert comparison["observed_repeat_range"] == abs(2.999 - 3.001)
    assert comparison["mean_minus_mainline"] == 0.
    assert all(row["second_minus_first"]["d0.self_nll"] == 2.999 - 3.001 for row in result["runs"])
    assert result["runs"][0]["branch_minus_mainline"]["d0.self_nll"] == 3.001 - 3.
    # Failed extra dispatch remains charged; a fresh attempt still must charge all 100 itself.
    policy.charge(ledger, "evaluation_ce_targets", 7)
    before = copy.deepcopy(ledger)
    policy.charge(ledger, "evaluation_ce_targets", 93)
    with pytest.raises(ValueError, match="independently charged"):
        policy.completed_evaluation_receipt(fake_dev64(3.), before=before, after=ledger,
            attempt=3, run_index=2, terminal_checkpoint=terminal, reference=reference)
    assert ledger["evaluation_ce_targets"] == 300
    bad = copy.deepcopy(receipts)
    bad[1]["terminal_checkpoint"]["sha256"] = "different"
    with pytest.raises(ValueError, match="same terminal checkpoint"):
        policy.summarize_endpoint_evaluations(bad, reference)


def test_dev64_rejects_training_charges_and_keeps_larger_difference_exploratory():
    before = policy.new_ledger()
    after = copy.deepcopy(before)
    policy.charge(after, "evaluation_ce_targets", 100)
    reference = policy.evaluation_metrics(fake_dev64(3.))
    receipt = policy.completed_evaluation_receipt(fake_dev64(2.9), before=before, after=after,
        attempt=1, run_index=1, terminal_checkpoint={"sha256": "same"}, reference=reference)
    second = copy.deepcopy(receipt)
    second.update(attempt=2, run_index=2)
    result = policy.summarize_endpoint_evaluations([receipt, second], reference)
    assert result["comparison"]["d0.self_nll"]["conclusion"] == "exploratory_difference_exceeds_two_run_range"
    policy.charge(after, "optimizer_targets_dispatched", 1)
    with pytest.raises(ValueError, match="training or diagnostic"):
        policy.completed_evaluation_receipt(fake_dev64(2.9), before=before, after=after,
            attempt=1, run_index=1, terminal_checkpoint={"sha256": "same"}, reference=reference)


def test_reference_endpoint_dev64_is_authenticated_to_checkpoint_writer_and_stage(source_arm, tmp_path):
    from delta_ngram.p2_initialization import writer_tensor_sha256
    report = fake_dev64(3.)
    report.update(arm=policy.ACTIVE, writer_sha256=writer_tensor_sha256(source_arm["writer"]),
                  checkpoint={"optimizer_step": policy.END_STEP, "completed_loss_targets": policy.END_TARGETS})
    path, complete_path = tmp_path / "report.json", tmp_path / "COMPLETE.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    complete_path.write_text(json.dumps({"complete": True, "arm": policy.ACTIVE,
        "writer_sha256": report["writer_sha256"], "report_sha256": policy.file_info(path)["sha256"]}), encoding="utf-8")
    checkpoint = {"arms": {policy.ACTIVE: source_arm}, "schedule_cursor": {"completed_stages": [{
        "optimizer_step": policy.END_STEP, "actual_model_targets": policy.END_TARGETS,
        "evaluation_report_sha256": policy.canonical_hash(report)}]}}
    receipt = policy.reference_endpoint_evaluation(checkpoint, path, complete_path)
    assert receipt["metrics"]["d0.self_nll"] == 3.
    checkpoint["schedule_cursor"]["completed_stages"][0]["evaluation_report_sha256"] = "wrong"
    with pytest.raises(ValueError, match="authenticated mainline completion"):
        policy.reference_endpoint_evaluation(checkpoint, path, complete_path)


def test_resume_preserves_first_completed_evaluation_and_requires_second_for_completion(source_arm):
    ledger = policy.new_ledger()
    ledger.update(optimizer_targets_dispatched=policy.TARGETS, optimizer_targets_completed=policy.TARGETS,
                  evaluation_ce_targets=100)
    terminal = {"sha256": "same"}
    receipt = {"run_index": 1, "terminal_checkpoint": terminal, "budget_delta": {"evaluation_ce_targets": 100}}
    checkpoint = {"protocol": policy.CHECKPOINT_PROTOCOL, "policy": policy.fork_policy("full"),
        "code_manifest_sha256": "code", "replay_sha256": "replay", "trajectory_boundary": True,
        "memory_reset_at_next_trajectory": True, "completed_groups": policy.GROUPS,
        "completed_targets": policy.TARGETS, "budget": copy.deepcopy(ledger),
        "phase": "evaluation_pending", "terminal_checkpoint": terminal, "evaluation_receipts": [receipt],
        "branch_state": {"sampler": {"total_loss_tokens": policy.END_TARGETS},
            "stability": {"completed_steps": policy.END_STEP}, "model_optimizer_step": policy.END_STEP,
            "optimizer_updates_since_fork": policy.GROUPS, "rng_by_rank": source_arm["rng_by_rank"]}}
    assert policy.validate_resume(checkpoint, ledger, branch="full", code_hash="code", replay_hash="replay")
    checkpoint["phase"] = "complete"
    with pytest.raises(ValueError, match="two endpoint DEV64"):
        policy.validate_resume(checkpoint, ledger, branch="full", code_hash="code", replay_hash="replay")
    checkpoint["evaluation_receipts"].append({**receipt, "run_index": 2})
    with pytest.raises(ValueError, match="charged budget"):
        policy.validate_resume(checkpoint, ledger, branch="full", code_hash="code", replay_hash="replay")
    policy.charge(ledger, "evaluation_ce_targets", 100)
    assert policy.validate_resume(checkpoint, ledger, branch="full", code_hash="code", replay_hash="replay")


def paired_observation(state, config, index, difference, targets=10):
    reference = {"step": policy.SOURCE_STEP + index, "group_identity_sha256": f"group{index}",
                 "loss_target_tokens": targets, "loss_sum": 3. * targets, "mean_nll": 3.}
    return policy.observe_paired_loss(state, config, reference, completed_groups=index,
        loss_sum=(3. + difference) * targets, targets=targets, group_sha256=f"group{index}")


def test_paired_loss_default_alert_requires_three_postwarmup_windows_and_never_stops():
    config, state = policy.paired_loss_policy(), policy.new_paired_loss_state()
    assert config["threshold_nats_per_target"] == .01
    assert config["user_decision_required"] and not config["automatic_stop"]
    for index in range(1, 251):
        row = paired_observation(state, config, index, .02 if index <= 200 else -.02,
                                 targets=10 if index % 2 else 30)
        assert row["stop_requested"] is False
        assert row["alert_active"] == (200 <= index < 250)
        if index <= 50:
            assert row["warmup"] and row["last_complete_window"] is None
        if index == 200:
            assert row["consecutive_alert_windows"] == 3
            assert row["window_completed"]["first_fork_group"] == 151
            assert row["window_completed"]["last_fork_group"] == 200
            assert row["group_branch_minus_mainline"] == pytest.approx(.02)
            assert row["cumulative_branch_minus_mainline"] == pytest.approx(.02)
    assert row["consecutive_alert_windows"] == 0
    assert state["completed_windows"] == 4 and state["window_records"] == []
    assert row["cumulative_branch_minus_mainline"] == pytest.approx(.012)
    with pytest.raises(ValueError, match="monitor-only"):
        policy.paired_loss_policy(action="stop")


def test_paired_loss_window_is_target_weighted_and_resume_preserves_pending_window():
    config = policy.paired_loss_policy(threshold=.5, warmup_groups=0, window_groups=2, consecutive_windows=1)
    state = policy.new_paired_loss_state()
    first = paired_observation(state, config, 1, 1., targets=1)
    assert first["pending_window_groups"] == 1 and first["last_complete_window"] is None
    restored = policy.new_paired_loss_state(json.loads(json.dumps(state)))
    second = paired_observation(state, config, 2, 0., targets=9)
    restored_second = paired_observation(restored, config, 2, 0., targets=9)
    assert second == restored_second and state == restored
    assert second["window_completed"]["branch_minus_mainline"] == pytest.approx(.1)
    assert second["cumulative_branch_minus_mainline"] == pytest.approx(.1)
    assert not second["alert_active"]
    with pytest.raises(ValueError, match="next mainline group"):
        paired_observation(state, config, 4, 0.)


def test_90m_node_uses_last_complete_group_and_reports_actual_targets():
    _, records = replay_fixture()
    node = policy.node_checkpoint_boundary(records)
    assert node["actual_model_targets"] <= 90_000_000 < node["next_group_model_targets"]
    assert node["optimizer_step"] == policy.SOURCE_STEP + node["completed_groups"]
    assert records[node["completed_groups"] - 1]["cumulative_model_loss_targets"] == node["actual_model_targets"]
    exact = records[400]["cumulative_model_loss_targets"]
    assert policy.node_checkpoint_boundary(records, exact)["completed_groups"] == 401


def test_reference_all_group_losses_retains_exact_scalars_and_rejects_bad_mean(tmp_path):
    _, records = replay_fixture()
    rows = [{**row, "event": "staged_alpha100_group_complete", "arm": policy.ACTIVE,
             "loss_sum": 3. * row["loss_target_tokens"], "mean_nll": 3.} for row in records]
    path = tmp_path / "training.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    losses = policy.reference_group_losses(path)
    assert len(losses) == 835 and losses[0]["step"] == 3338 and losses[-1]["step"] == 4172
    assert losses[0]["loss_sum"] == 3. * records[0]["loss_target_tokens"]
    rows[34]["mean_nll"] = 4.
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="scalar or group identity"):
        policy.reference_group_losses(path)

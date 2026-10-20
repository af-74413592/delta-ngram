"""CPU budget, sampler and control-arm contracts for the alpha=1 extension.

The authorized run uses model-target accounting. The alternative charged cap
remains covered so changing that choice cannot silently refund prior work.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from delta_ngram.data.event_streams import VariableCostSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
import qwen38_native_alpha100_policy as policy


ACTIVE = "alpha100_rmsnorm"
CONTROL = "alpha099_rmsnorm"


def source_checkpoint(*, active_targets=58_480_000, failed=0):
    charged = {ACTIVE: active_targets + 1_468_915 + failed, CONTROL: 39_998_107}
    return {
        "arms": {
            ACTIVE: {"completed_loss_targets": active_targets, "optimizer_step": 2450},
            CONTROL: {"completed_loss_targets": 39_998_107, "optimizer_step": 1669},
        },
        "budget": {
            "total_optimizer_target_accesses": sum(charged.values()),
            "cumulative_optimizer_target_accesses_by_arm": charged,
            "historical_optimizer_target_accesses_by_arm": {ACTIVE: 1_468_915, CONTROL: 0},
        },
        "schedule_cursor": {
            "arm": ACTIVE, "phase": "training", "stage_index": 4,
            "milestone": 60_000_000,
            "completed_stages": [
                {"arm": ACTIVE, "nominal_model_targets": 20_000_000},
                {"arm": CONTROL, "nominal_model_targets": 20_000_000},
                {"arm": ACTIVE, "nominal_model_targets": 40_000_000},
                {"arm": CONTROL, "nominal_model_targets": 40_000_000,
                 "actual_model_targets": 39_998_107, "optimizer_step": 1669},
            ],
        },
    }


def event_state():
    return {
        "version": 3,
        "cells": [
            {"source": "news_event", "blocks": blocks, "order": list(range(count)),
             "epoch": 0, "cursor": 0, "loss_tokens": 0}
            for blocks, count in ((4, 123), (8, 510), (16, 74), (32, 6))
        ],
    }


def tiny_sampler():
    # Actual VariableCostSampler with deliberately unequal event-like costs.
    # No tokenizer, native model or GPU is needed to exercise saved cursor/RNG.
    windows = {
        blocks: [SimpleNamespace(
            document_id=f"document:{blocks}:{index}", group_id=f"group:{blocks}:{index}",
            source="news_event", start=0, blocks=blocks, block_size=512,
            loss_tokens=cost, token_ids=(blocks, index), variable_cost=False)
            for index, cost in enumerate(costs)]
        for blocks, costs in ((4, (1, 3, 5)), (8, (2, 7)))
    }
    return VariableCostSampler(
        windows, seed=17, source_weights={"news_event": 1.0},
        source_length_weights={"news_event": {4: .5, 8: .5}})


def test_authorized_model_policy_preserves_history_and_freezes_control():
    selected = policy.training_policy("model")
    assert selected["budget_basis"] == "model"
    assert selected["target_cap_per_arm"] == 100_000_000
    assert selected["active_arm"] == ACTIVE
    assert selected["arms"][ACTIVE]["authorized_continuation"] is True
    assert selected["arms"][CONTROL]["authorized_continuation"] is False
    assert selected["historical_optimizer_target_allocation"] == {ACTIVE: 1_468_915, CONTROL: 0}
    assert selected["model_target_milestones"] == [60_000_000, 80_000_000, 100_000_000]
    assert selected["control_fixed_model_targets"] == 39_998_107
    assert selected["control_fixed_optimizer_step"] == 1669


def test_model_100m_keeps_historical_charge_above_100m():
    charged, model = 101_458_915, 99_990_000
    assert policy.cap_allows(policy.training_policy("model"), charged, model, 10_000)
    assert not policy.cap_allows(policy.training_policy("model"), charged, model, 10_001)
    assert not policy.cap_allows(policy.training_policy("charged"), charged, model, 0)
    assert charged + 10_000 == 101_468_915


@pytest.mark.parametrize("basis", ["model", "charged"])
def test_last_whole_group_must_fit_without_shortening(basis):
    selected = policy.training_policy(basis)
    before = 100_000_000 - 12_345
    charged, model = ((before + 1_468_915, before) if basis == "model"
                      else (before, before - 1_468_915))
    assert policy.cap_allows(selected, charged, model, 12_345)
    # The actual next whole group is larger than the remaining budget.
    assert not policy.cap_allows(selected, charged, model, 12_346)


@pytest.mark.parametrize("index,bad", [(0, -1), (1, -1), (2, -1), (2, True), (2, 1.0)])
def test_cap_rejects_invalid_accounting(index, bad):
    values = [2_000_000, 500_000, 12_000]
    values[index] = bad
    with pytest.raises(ValueError, match="nonnegative integers"):
        policy.cap_allows(policy.training_policy("model"), *values)


@pytest.mark.parametrize("basis", ["model", "charged"])
def test_query_charge_is_persisted_before_native_call_at_exact_boundary(basis):
    selected = policy.training_policy(basis)
    charged, model = ((101_468_905, 99_999_990) if basis == "model"
                      else (99_999_990, 98_531_075))
    ledger, events = {"charged": charged}, []

    def persist(cost):
        ledger["charged"] += cost
        events.append(("persist", ledger["charged"]))

    def execute():
        assert ledger["charged"] == charged + 10
        assert events == [("persist", charged + 10)]
        events.append(("native", 10))
        return "native-result"

    assert policy.dispatch_query(selected, charged, model, 10, persist, execute) == "native-result"
    assert events[-1] == ("native", 10)


@pytest.mark.parametrize("basis", ["model", "charged"])
def test_query_over_cap_neither_reserves_nor_dispatches(basis):
    selected = policy.training_policy(basis)
    events = []
    with pytest.raises(ValueError, match="exceeds per-arm cap"):
        policy.dispatch_query(selected, 100_000_000, 100_000_000, 1,
                              lambda cost: events.append("persist"),
                              lambda: events.append("native"))
    assert events == []


@pytest.mark.parametrize("cost", [0, -1, True, 1.0])
def test_query_requires_positive_integer_cost_before_callbacks(cost):
    events = []
    with pytest.raises(ValueError, match="positive integer"):
        policy.dispatch_query(policy.training_policy("model"), 1_468_915, 0, cost,
                              lambda value: events.append("persist"),
                              lambda: events.append("native"))
    assert events == []


def test_failed_ledger_write_prevents_native_dispatch():
    events = []

    def fail_persist(cost):
        events.append("persist_attempt")
        raise OSError("ledger write failed")

    with pytest.raises(OSError, match="ledger write failed"):
        policy.dispatch_query(policy.training_policy("model"), 1_468_915, 0, 512,
                              fail_persist, lambda: events.append("native"))
    assert events == ["persist_attempt"]


def test_failed_native_query_keeps_charge_and_retry_is_charged_again():
    ledger, events = {"charged": 1_468_915}, []

    def persist(cost):
        ledger["charged"] += cost
        events.append("persist")

    def fail_native():
        events.append("native_failed")
        raise RuntimeError("native forward failed")

    selected = policy.training_policy("model")
    with pytest.raises(RuntimeError, match="native forward failed"):
        policy.dispatch_query(selected, ledger["charged"], 0, 512, persist, fail_native)
    assert ledger["charged"] == 1_469_427
    assert policy.dispatch_query(selected, ledger["charged"], 0, 512,
                                 persist, lambda: "retry-result") == "retry-result"
    assert ledger["charged"] == 1_469_939
    assert events == ["persist", "native_failed", "persist"]


def test_migration_assigns_history_and_failed_queries_to_active_arm_without_mutation():
    source = source_checkpoint(failed=8192)
    ledger = copy.deepcopy(source["budget"])
    before = policy.state_digest((source, ledger))
    result = policy.derive_arm_accounting(source, ledger, migrating=True)
    assert result["historical"] == {ACTIVE: 1_468_915, CONTROL: 0}
    assert result["charged"] == {ACTIVE: 59_957_107, CONTROL: 39_998_107}
    assert result["uncompleted_or_failed"] == {ACTIVE: 8192, CONTROL: 0}
    assert policy.state_digest((source, ledger)) == before


@pytest.mark.parametrize("migrating", [True, False])
def test_source_ledger_cannot_refund_saved_failed_queries(migrating):
    source = source_checkpoint(failed=8192)
    ledger = copy.deepcopy(source["budget"])
    ledger["total_optimizer_target_accesses"] -= 1
    ledger["cumulative_optimizer_target_accesses_by_arm"][ACTIVE] -= 1
    with pytest.raises(ValueError, match="refund|below|source"):
        policy.derive_arm_accounting(source, ledger, migrating=migrating)


def test_resume_preserves_new_failed_charge_in_addition_to_completed_model():
    source = source_checkpoint(failed=8192)
    ledger = copy.deepcopy(source["budget"])
    ledger["total_optimizer_target_accesses"] += 4096
    ledger["cumulative_optimizer_target_accesses_by_arm"][ACTIVE] += 4096
    result = policy.derive_arm_accounting(source, ledger, migrating=False)
    assert result["model"][ACTIVE] == 58_480_000
    assert result["uncompleted_or_failed"] == {ACTIVE: 12_288, CONTROL: 0}
    assert sum(result["charged"].values()) == ledger["total_optimizer_target_accesses"]


@pytest.mark.parametrize("change", ["control_charge", "history_reassignment", "control_step", "control_targets"])
def test_accounting_rejects_control_progress_or_reassigned_history(change):
    source = source_checkpoint()
    ledger = copy.deepcopy(source["budget"])
    if change == "control_charge":
        ledger["cumulative_optimizer_target_accesses_by_arm"][CONTROL] += 1
        ledger["total_optimizer_target_accesses"] += 1
    elif change == "history_reassignment":
        ledger["historical_optimizer_target_accesses_by_arm"] = {ACTIVE: 0, CONTROL: 1_468_915}
    elif change == "control_step":
        source["arms"][CONTROL]["optimizer_step"] += 1
    else:
        source["arms"][CONTROL]["completed_loss_targets"] += 1
    with pytest.raises(ValueError):
        policy.derive_arm_accounting(source, ledger, migrating=False)


def test_exactly_three_complete_event_passes_are_allowed():
    state = event_state()
    for cell in state["cells"]:
        cell.update(epoch=3, cursor=0, loss_tokens=3 * policy.EVENT_CAPACITIES[cell["blocks"]])
    rows = policy.require_event_limit(state)
    assert [row["blocks"] for row in rows] == [4, 8, 16, 32]
    assert all(row["cost_repeats"] == row["draw_passes"] == 3 for row in rows)


@pytest.mark.parametrize("overflow", ["tokens_only", "draws_only", "both"])
def test_event_limit_checks_both_token_and_draw_consumption(overflow):
    state = event_state()
    cell = state["cells"][0]
    cell.update(epoch=3, cursor=0, loss_tokens=3 * 89_431)
    if overflow in ("tokens_only", "both"):
        cell["loss_tokens"] += 1
    if overflow in ("draws_only", "both"):
        cell["cursor"] = 1
    rows = policy.event_repeat_counts(state)
    assert rows[0]["within_limit"] is False
    with pytest.raises(ValueError, match="three passes"):
        policy.require_event_limit(state)


def test_variable_cost_repeat_fraction_is_not_draw_fraction():
    state = {"version": 3, "cells": [{
        "source": "news_event", "blocks": 4, "order": [0, 1, 2],
        "epoch": 1, "cursor": 1, "loss_tokens": 12}]}
    # Per-window costs 2, 3, 5: one complete pass plus the first 2-token window.
    row, = policy.event_repeat_counts(state, capacities={4: 10}, expected_windows={4: 3})
    assert row["cost_repeats"] == 1.2
    assert row["draw_passes"] == pytest.approx(4 / 3)
    assert row["draws"] == 4


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "wrong_windows", "negative_cursor", "bool_epoch"])
def test_event_guard_rejects_invalid_bucket_identity_or_counters(corruption):
    state = event_state()
    if corruption == "missing":
        state["cells"].pop()
    elif corruption == "duplicate":
        state["cells"].append(copy.deepcopy(state["cells"][0]))
    elif corruption == "wrong_windows":
        state["cells"][0]["order"].pop()
    elif corruption == "negative_cursor":
        state["cells"][0]["cursor"] = -1
    else:
        state["cells"][0]["epoch"] = True
    with pytest.raises(ValueError):
        policy.event_repeat_counts(state)


def test_independent_arm_sampler_resume_preserves_other_arm_and_future_order():
    active, control = tiny_sampler(), tiny_sampler()
    initial = active.state_dict()
    control_before = policy.state_digest(control.state_dict())
    active.next_batch(12)
    checkpoint = copy.deepcopy(active.state_dict())
    assert checkpoint["total_loss_tokens"] > initial["total_loss_tokens"]
    assert policy.state_digest(control.state_dict()) == control_before
    resumed = tiny_sampler()
    resumed.load_state_dict(checkpoint)
    expected = [window.document_id for window in active.next_batch(12)]
    assert [window.document_id for window in resumed.next_batch(12)] == expected
    assert policy.state_digest(resumed.state_dict()) == policy.state_digest(active.state_dict())
    assert policy.state_digest(control.state_dict()) == control_before
    # Rejecting an oversized candidate group must restore sampler RNG as well.
    resumed.load_state_dict(checkpoint)
    resumed.next_batch(12)
    resumed.load_state_dict(checkpoint)
    assert policy.state_digest(resumed.state_dict()) == policy.state_digest(checkpoint)


def test_extension_cursor_cannot_return_to_alpha099():
    source = source_checkpoint()
    before = policy.state_digest(source)
    cursor = policy.initial_policy_cursor(source)
    assert policy.validate_policy_cursor(cursor) is cursor
    assert cursor["arm"] == ACTIVE and cursor["milestone"] == 60_000_000
    assert cursor["completed_stages"] == []
    assert policy.state_digest(source) == before
    cursor["arm"] = CONTROL
    with pytest.raises(ValueError, match="alpha100-only"):
        policy.validate_policy_cursor(cursor)


def test_initial_cursor_requires_completed_control_40m_evaluation():
    source = source_checkpoint()
    source["schedule_cursor"]["completed_stages"][-1]["optimizer_step"] -= 1
    with pytest.raises(ValueError, match="40M evaluation"):
        policy.initial_policy_cursor(source)


def test_state_digest_covers_scalar_tensors_numpy_rng_and_container_types():
    state = {"optimizer": {"step": torch.tensor(3.), "moment": torch.arange(6).reshape(2, 3)},
             "rng": ("MT19937", np.arange(5, dtype=np.uint32), 4)}
    copied = copy.deepcopy(state)
    assert policy.state_digest(state) == policy.state_digest(copied)
    copied["optimizer"]["step"].add_(1)
    assert policy.state_digest(state) != policy.state_digest(copied)
    copied = copy.deepcopy(state)
    copied["rng"][1][0] += 1
    assert policy.state_digest(state) != policy.state_digest(copied)
    assert policy.state_digest([1, 2]) != policy.state_digest((1, 2))

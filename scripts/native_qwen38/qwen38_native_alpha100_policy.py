"""Pure CPU policy for the explicitly authorized alpha=1 continuation."""
from __future__ import annotations

import copy
import hashlib

from qwen38_native_norm_probe import require


ACTIVE = "alpha100_rmsnorm"
CONTROL = "alpha099_rmsnorm"
ARMS = ((ACTIVE, 1.0), (CONTROL, 0.99))
ARM_CAP = 100000000
HISTORICAL_ALPHA100 = 1468915
CONTROL_TARGETS = 39998107
CONTROL_STEPS = 1669
MILESTONES = (60000000, 80000000, 100000000)
EVENT_CAPACITIES = {4: 89431, 8: 669232, 16: 209278, 32: 39684}
EVENT_WINDOWS = {4: 123, 8: 510, 16: 74, 32: 6}
POLICY_PROTOCOL = "qwen38_native_alpha100_per_arm_budget_v1"


def state_digest(value):
    """Typed CPU value digest, including exact tensor and NumPy RNG bytes."""
    import numpy as np
    import torch
    hasher = hashlib.sha256()
    def feed(value):
        hasher.update(len(value).to_bytes(8, "big"))
        hasher.update(value)
    def visit(item):
        feed(f"{type(item).__module__}.{type(item).__qualname__}".encode())
        if torch.is_tensor(item):
            require(item.device.type == "cpu", "state comparison requires CPU tensors")
            feed(repr((str(item.dtype), tuple(item.shape), tuple(item.stride()))).encode())
            feed(item.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            feed(repr((item.dtype.str, item.shape, item.strides)).encode())
            feed(item.tobytes())
        elif isinstance(item, dict):
            feed(str(len(item)).encode())
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            feed(str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, int, float, bool, bytes, np.generic)):
            feed(repr(item).encode())
        else:
            raise TypeError(f"unsupported state value: {type(item)!r}")
    visit(value)
    return hasher.hexdigest()


def training_policy(basis):
    require(basis in ("charged", "model"), "budget basis must be charged or model")
    return {
        "protocol": POLICY_PROTOCOL, "budget_basis": basis,
        "target_cap_per_arm": ARM_CAP,
        "cap_measure": ("effective_model_training_targets" if basis == "model"
                        else "all_charged_optimizer_target_accesses"),
        "lifetime_charged_total_is_accounting_only": basis == "model",
        "model_basis_charges": "preserve_all_history_and_failed_queries_without_a_100M_charged_cap",
        "arms": {name: {"alpha": alpha, "authorized_continuation": name == ACTIVE,
                         "target_cap": ARM_CAP} for name, alpha in ARMS},
        "active_arm": ACTIVE, "model_target_milestones": list(MILESTONES),
        "historical_optimizer_target_allocation": {ACTIVE: HISTORICAL_ALPHA100, CONTROL: 0},
        "control_fixed_optimizer_step": CONTROL_STEPS,
        "control_fixed_model_targets": CONTROL_TARGETS,
        "endpoint": "last_whole_12_trajectory_group_then_DEV64_actual_budget_endpoint",
        "event_max_complete_passes": 3,
        "event_capacity_loss_tokens": {str(key): value for key, value in EVENT_CAPACITIES.items()},
        "event_capacity_windows": {str(key): value for key, value in EVENT_WINDOWS.items()},
        "event_exposure_scope": "each_arm_persisted_sampler_lineage; discarded_history_not_inferred",
        "event_limit_action": "rollback_whole_group_and_stop_without_reweighting_or_skipping",
        "failed_queries": "charged_before_dispatch_and_never_refunded",
        "evaluation_targets": "separate_lifetime_counter_not_in_optimizer_cap",
    }


def event_repeat_counts(sampler, capacities=None, expected_windows=None):
    require(sampler.get("version") == 3, "event guard requires variable-cost sampler version 3")
    capacities = EVENT_CAPACITIES if capacities is None else capacities
    expected_windows = EVENT_WINDOWS if expected_windows is None else expected_windows
    rows, seen = [], set()
    for cell in sampler["cells"]:
        if cell["source"] != "news_event":
            continue
        blocks = cell["blocks"]
        require(blocks in capacities and blocks not in seen, "unknown or duplicate event bucket")
        seen.add(blocks)
        count = len(cell["order"])
        epoch, cursor, loss_tokens = cell["epoch"], cell["cursor"], cell["loss_tokens"]
        require(count > 0 and (expected_windows is None or count == expected_windows[blocks]),
                "event bucket window capacity differs")
        require(type(epoch) is int and epoch >= 0 and type(cursor) is int and 0 <= cursor < count,
                "event bucket cursor or epoch is invalid")
        require(type(loss_tokens) is int and loss_tokens >= 0,
                "event bucket loss-token counter is invalid")
        draws, capacity = epoch * count + cursor, capacities[blocks]
        require(type(capacity) is int and capacity > 0, "invalid event token capacity")
        rows.append({
            "blocks": blocks, "source": "news_event", "windows": count,
            "epoch": epoch, "cursor": cursor, "draws": draws,
            "maximum_draws": 3 * count, "draw_passes": epoch + cursor / count,
            "loss_tokens": loss_tokens, "capacity_loss_tokens": capacity,
            "maximum_loss_tokens": 3 * capacity, "cost_repeats": loss_tokens / capacity,
            "within_limit": draws <= 3 * count and loss_tokens <= 3 * capacity,
        })
    require(seen == set(capacities), "missing event bucket")
    return sorted(rows, key=lambda row: row["blocks"])


def require_event_limit(sampler):
    rows = event_repeat_counts(sampler)
    require(all(row["within_limit"] for row in rows), "event bucket exceeds three passes")
    return rows


def cap_allows(policy, charged_before, model_before, additional):
    require(all(type(value) is int and value >= 0
                for value in (charged_before, model_before, additional)),
            "budget counters must be nonnegative integers")
    basis = policy["budget_basis"]
    require(basis in ("charged", "model"), "unknown budget basis")
    return (charged_before if basis == "charged" else model_before) + additional <= policy["target_cap_per_arm"]


def dispatch_query(policy, charged_before, model_before, cost, reserve_and_persist, execute):
    """Persist every charge before native dispatch; failures never refund it."""
    require(type(cost) is int and cost > 0, "query cost must be a positive integer")
    require(cap_allows(policy, charged_before, model_before, cost), "query dispatch exceeds per-arm cap")
    reserve_and_persist(cost)
    return execute()


def derive_arm_accounting(source, ledger, migrating):
    models = {name: source["arms"][name]["completed_loss_targets"] for name, _ in ARMS}
    total = ledger["total_optimizer_target_accesses"]
    require(type(total) is int and total >= 0, "invalid cumulative optimizer accesses")
    require(total >= source["budget"]["total_optimizer_target_accesses"],
            "latest ledger would refund lifetime optimizer accesses")
    require(models[CONTROL] == CONTROL_TARGETS
            and source["arms"][CONTROL]["optimizer_step"] == CONTROL_STEPS,
            "alpha099 must remain at its completed 40M control checkpoint")
    historical = {ACTIVE: HISTORICAL_ALPHA100, CONTROL: 0}
    if migrating:
        # All pre-lineage experiments were explicitly assigned to alpha=1.
        # The unchanged alpha=.99 lineage has no uncompleted optimizer charges.
        charged = {CONTROL: models[CONTROL], ACTIVE: total - models[CONTROL]}
    else:
        charged = copy.deepcopy(ledger["cumulative_optimizer_target_accesses_by_arm"])
        require(set(charged) == {ACTIVE, CONTROL}
                and ledger["historical_optimizer_target_accesses_by_arm"] == historical,
                "saved per-arm accounting policy differs")
        saved_charged = source["budget"]["cumulative_optimizer_target_accesses_by_arm"]
        require(all(charged[name] >= saved_charged[name] for name, _ in ARMS),
                "per-arm ledger would refund optimizer accesses")
    require(all(type(value) is int and value >= 0 for value in charged.values())
            and sum(charged.values()) == total, "per-arm charges do not sum to lifetime total")
    require(charged[CONTROL] == models[CONTROL], "inactive alpha099 acquired optimizer charges")
    remainder = {name: charged[name] - models[name] - historical[name] for name, _ in ARMS}
    require(all(value >= 0 for value in remainder.values()),
            "lifetime optimizer charges are below model plus registered history")
    return {"charged": charged, "model": models, "historical": historical,
            "uncompleted_or_failed": remainder,
            "attribution": "registered_history_and_all_uncompleted_remainder_to_alpha100"}


def initial_policy_cursor(source):
    old = source["schedule_cursor"]
    require(old["arm"] == ACTIVE and old["phase"] == "training"
            and old["stage_index"] == 4 and old["milestone"] == MILESTONES[0]
            and len(old["completed_stages"]) == 4,
            "old final checkpoint is not the alpha100-to-60M stage")
    expected = [(ACTIVE, 20000000), (CONTROL, 20000000),
                (ACTIVE, 40000000), (CONTROL, 40000000)]
    require([(row["arm"], row["nominal_model_targets"]) for row in old["completed_stages"]] == expected,
            "old completed DEV64 stage sequence differs")
    require(old["completed_stages"][-1]["actual_model_targets"] == CONTROL_TARGETS
            and old["completed_stages"][-1]["optimizer_step"] == CONTROL_STEPS,
            "alpha099 40M evaluation does not match the saved control")
    return {"policy_stage_index": 0, "arm": ACTIVE, "milestone": MILESTONES[0],
            "phase": "training", "evaluation_attempt": 0, "completed_stages": [],
            "evaluation_kind": None, "terminal_after_evaluation": False,
            "terminal_reason": None}


def validate_policy_cursor(cursor):
    index = cursor["policy_stage_index"]
    require(type(index) is int and 0 <= index <= len(MILESTONES), "invalid policy stage index")
    require(cursor["arm"] == ACTIVE and cursor["phase"] in ("training", "evaluation_pending", "complete"),
            "invalid alpha100-only cursor")
    require(type(cursor["evaluation_attempt"]) is int and cursor["evaluation_attempt"] >= 0,
            "invalid evaluation attempt")
    if cursor["phase"] != "complete":
        require(index < len(MILESTONES) and cursor["milestone"] == MILESTONES[index],
                "policy milestone differs")
    require(len(cursor["completed_stages"]) == index, "policy completed-stage history differs")
    require(cursor["evaluation_kind"] in (None, "model_milestone", "actual_budget_endpoint", "event_limit_endpoint"),
            "invalid evaluation endpoint kind")
    require(type(cursor["terminal_after_evaluation"]) is bool, "invalid terminal evaluation flag")
    if cursor["phase"] == "evaluation_pending":
        require(cursor["evaluation_kind"] is not None, "pending evaluation has no endpoint kind")
    return cursor

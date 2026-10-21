import copy

import pytest
import torch

from delta_ngram.alpha_memory import BlockAlpha, LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.qwen38_memory_writer import write_memory_aware_native_block
from delta_ngram.qwen38_seven_forward import STATE_FIELDS, restore_state, state_fingerprint, state_payload
from delta_ngram.qwen38_shortchain_protocol import (
    SHORTCHAIN_PROTOCOL, assert_state_payload_equal, shortchain_plan,
    validate_shortchain_cursor, validate_state_payload,
)
from delta_ngram.qwen38_writer import NativeForwardWriter
from delta_ngram.state import SparseMemoryState


def make_state():
    generator = torch.Generator().manual_seed(7)
    rows = torch.tensor([2, 7])
    memory = BatchMemoryState(SparseMemoryState(rows, torch.randn(2, 160, generator=generator),
                                               torch.tensor([4., 1.])), 1, 100)
    return LazyDecayState(memory, torch.tensor([16]), torch.zeros(1, dtype=torch.float64),
                          torch.tensor([15, 16]), torch.zeros(2, dtype=torch.float64))


def test_fixed_plan_pairs_actual_retention_and_bounds_gpu_work():
    plan = shortchain_plan()
    assert plan["protocol"] == SHORTCHAIN_PROTOCOL
    assert plan["total_writes"] == 64 and plan["restore_branch_extra_writes"] == 1
    assert sum(len(node["domains"]) for node in plan["evaluation_nodes"]) == 11
    assert plan["total_queries"] == 13 and plan["total_capture_files"] == 104
    assert plan["extra_evaluations"] == [{"name": "long_context_base", "domains": ["math", "code"],
        "selection": "longest_input_per_domain", "writes_completed": 0}]
    assert [pair["metric"] for pair in plan["capture_pairs"]] == ["K", "K", "ki"]
    retention = plan["capture_pairs"][-1]
    assert retention == {"metric": "ki", "before": "after_task_01_sme",
        "after": "after_task_02_nav", "domains": ["sme"]}


@pytest.mark.parametrize("kwargs", [{"tasks": ("nav", "sme")}, {"tasks": ("sme",)},
    {"frames_per_task": 31}, {"query_limit": 2}, {"query_limit": True}, {"restore_after": 17}])
def test_unregistered_budget_is_rejected(kwargs):
    with pytest.raises(ValueError):
        shortchain_plan(**kwargs)


@pytest.mark.parametrize("writes", [0, 16, 32, 64])
def test_cursor_at_restore_and_task_boundaries(writes):
    task, example = divmod(writes, 32)
    cursor = {"task_index": task, "example_index": example, "writes_completed": writes,
              "real_evidence_tokens": writes * 7, "phase": "write"}
    assert validate_shortchain_cursor(cursor, shortchain_plan()) is cursor


@pytest.mark.parametrize("change", [{"task_index": 1}, {"writes_completed": 65},
    {"example_index": True}, {"real_evidence_tokens": -1}])
def test_inconsistent_cursor_rejected(change):
    cursor = {"task_index": 0, "example_index": 16, "writes_completed": 16, **change}
    with pytest.raises(ValueError):
        validate_shortchain_cursor(cursor, shortchain_plan())


def test_restore_and_actual_next_d1_are_bitwise_equal(tmp_path):
    torch.manual_seed(21)
    state = make_state()
    payload = state_payload(state)
    stats = validate_state_payload(payload, expected_writes=16)
    assert stats["fingerprint"] == state_fingerprint(state)
    path = tmp_path / "resume.pt"
    torch.save(payload, path)
    serialized = torch.load(path, map_location="cpu", weights_only=True)
    restored = restore_state(serialized)
    assert_state_payload_equal(payload, state_payload(restored))
    assert all(getattr(state, field).data_ptr() != getattr(restored, field).data_ptr() for field in STATE_FIELDS)
    writer = NativeForwardWriter(width=160, input_dim=6, hidden_dim=8).eval().requires_grad_(False)
    policy = BlockAlpha(6, "fixed", 1.)
    hidden = torch.randn(1, 2, 6)
    addresses = torch.tensor([[[2, 3], [3, 9]]])
    mask = torch.ones((1, 2), dtype=torch.bool)
    with torch.no_grad():
        next_state = write_memory_aware_native_block(writer, state, hidden, addresses, mask, policy)
        next_restored = write_memory_aware_native_block(writer, restored, hidden, addresses, mask, policy)
    assert_state_payload_equal(state_payload(next_state), state_payload(next_restored))
    validate_state_payload(state_payload(next_state), expected_writes=17)
    assert next_state.rows.tolist() == [2, 3, 7, 9]
    assert state_fingerprint(state) == stats["fingerprint"]
    assert next_state.row_last_written_block.tolist() == [17, 17, 16, 17]


@pytest.mark.parametrize("field", STATE_FIELDS)
def test_every_missing_field_is_rejected(field):
    payload = state_payload(make_state())
    del payload[field]
    with pytest.raises(ValueError):
        validate_state_payload(payload)


@pytest.mark.parametrize("corruption", ["bf16", "nonfinite", "duplicate", "padding", "capacity", "alpha",
    "clock", "row_clock", "negative_counts", "width"])
def test_corrupt_state_rejected(corruption):
    payload = state_payload(make_state())
    if corruption == "bf16":
        payload["values"] = payload["values"].bfloat16()
    elif corruption == "nonfinite":
        payload["values"][0, 0] = float("nan")
    elif corruption == "duplicate":
        payload["rows"][1] = 2
    elif corruption == "padding":
        payload["rows"][0] = 0
    elif corruption == "capacity":
        payload["rows"][1] = payload["row_capacity"] + 1
    elif corruption == "alpha":
        payload["row_log_alpha_at_write"][0] = -.1
    elif corruption == "clock":
        payload["block_clock"][0] = -1
    elif corruption == "row_clock":
        payload["row_last_written_block"][0] = 17
    elif corruption == "negative_counts":
        payload["counts"][0] = -1
    elif corruption == "width":
        payload["width"] = 80
    with pytest.raises(ValueError):
        validate_state_payload(payload)


def test_equality_detects_signed_zero_and_changed_valid_metadata():
    left = state_payload(make_state())
    right = copy.deepcopy(left)
    right["cumulative_log_alpha"][0] = -0.
    with pytest.raises(ValueError, match="bitwise"):
        assert_state_payload_equal(left, right)
    right = copy.deepcopy(left)
    right["row_capacity"] += 1
    with pytest.raises(ValueError, match="metadata"):
        assert_state_payload_equal(left, right)


def test_empty_full_state_is_valid():
    payload = state_payload(LazyDecayState.empty(1, 160, 100))
    stats = validate_state_payload(payload, expected_writes=0)
    assert stats["rows"] == 0 and stats["values_l2"] == 0.


def test_last_native_row_is_a_valid_physical_address():
    payload = state_payload(make_state())
    payload["rows"][-1] = payload["row_capacity"]
    assert validate_state_payload(payload)["rows"] == 2

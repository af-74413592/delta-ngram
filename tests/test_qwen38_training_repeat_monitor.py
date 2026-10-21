"""Training repeat monitoring preserves the original graph, RNG and CE work."""
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.qwen38_train_data import _frame
from delta_ngram.state import SparseMemoryState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
import qwen38_native_memory_schedule as schedule


@pytest.fixture(autouse=True)
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def plan(writes, token):
    return [{"evidence": _frame({}, (token + ordinal,) * 4, 0, 4, query=False),
             "query": {"loss_target_tokens": 1} if ordinal else None} for ordinal in range(writes)] + [
             {"evidence": None, "query": {"loss_target_tokens": 1}}]


def update(state, value):
    old = state.values if state.values.numel() else torch.zeros(1, 2)
    counts = state.counts + 1 if state.counts.numel() else torch.ones(1)
    clock = state.block_clock + 1
    memory = BatchMemoryState(SparseMemoryState(torch.tensor([1]), .7 * old + value, counts), 1, 100)
    return LazyDecayState(memory, clock, state.cumulative_log_alpha, clock.clone(), torch.zeros(1, dtype=torch.float64))


def run_case(monkeypatch, *, diagnose, drift=0., mutate=False):
    random.seed(11)
    np.random.seed(13)
    torch.manual_seed(17)
    parameter = torch.tensor(.2, requires_grad=True)
    session = SimpleNamespace(device=torch.device("cpu"), layout=SimpleNamespace(row_capacity=100, row_width=2))
    local = [{"index": index, "plan": plan(writes, token), "state": LazyDecayState.empty(1, 2, 100)}
             for index, writes, token in ((0, 3, 1), (1, 1, 10))]
    costs, calls, original_states, repeated_states = [], [], [], []
    training_writes = 0
    current_kind = None
    def charge(row):
        nonlocal current_kind
        current_kind = row["kind"]
        costs.append(row)
    def encode(session, batches, *, encoding):
        mb = batches[0]
        assert encoding == "memory" and mb["batch"]["labels"] is None
        if current_kind == "diagnostic_repeat_memory":
            assert training_writes == 4, "repeat pass cannot interleave original writer graphs"
        with schedule.preserve_encoding_rng(session.device), torch.no_grad():
            random.random(); np.random.random(); torch.rand(2)
            hidden = torch.zeros(len(mb["frames"]), 640, 2560)
            old = mb["reader_state"].lookup(torch.ones(len(mb["frames"]), 1, 16, dtype=torch.long))[0]
            for lane, frame in enumerate(mb["frames"]):
                if frame is not None:
                    hidden[lane].fill_(frame["input_ids"][0] + float(old[lane].mean()))
                    if current_kind == "diagnostic_repeat_memory":
                        hidden[lane].add_(drift)
        calls.append((current_kind, len(mb["frames"])))
        return {"microbatches": [{"hidden": hidden,
            "physical_rows": mb["valid_mask"].long()[..., None].expand(-1, -1, 16)}]}
    def write(item, block):
        nonlocal training_writes
        assert torch.is_grad_enabled()
        item["state"] = update(item["state"], parameter * block["H_new"].mean())
        original_states.append(item["state"])
        training_writes += 1
    def repeat_write(state, block):
        assert not torch.is_grad_enabled() and parameter.grad is None
        assert all(state is not reference for reference in original_states)
        random.random(); np.random.random(); torch.rand(2)
        if mutate and state.rows.numel():
            state.values.add_(1)
        repeated = update(state, parameter * block["H_new"].mean())
        assert not repeated.values.requires_grad and repeated.values.grad_fn is None
        repeated_states.append(repeated)
        return repeated
    monkeypatch.setattr(schedule, "collect_memory_encoding_microbatches", encode)
    result = schedule.build_causal_writer_histories(session, local, gather=lambda value: [value],
        write=write, encoding_mode="full", batch_limit=12, on_encoding=charge,
        diagnostic_repeat_write=repeat_write if diagnose else None)
    assert parameter.grad is None
    objective = sum(task["snapshot"].values.square().sum() for task in result["queries"])
    gradient = torch.autograd.grad(objective, parameter)[0]
    rng = random.random(), np.random.random(), torch.rand(3)
    return result, costs, calls, objective.detach(), gradient, rng, original_states, repeated_states


def test_diagnostic_replay_leaves_training_snapshots_gradients_rng_and_costs_intact(monkeypatch):
    baseline = run_case(monkeypatch, diagnose=False)
    checked = run_case(monkeypatch, diagnose=True)
    for index in (3, 4):
        assert torch.equal(baseline[index], checked[index])
    assert baseline[5][:2] == checked[5][:2] and torch.equal(baseline[5][2], checked[5][2])
    assert all(torch.equal(a.values, b.values) for a, b in zip(baseline[6], checked[6]))
    assert [task["owner"]["index"] for task in checked[0]["queries"]] == [0, 0, 0, 1]
    assert all(task["snapshot"].values.requires_grad for task in checked[0]["queries"])
    assert len(checked[7]) == 4 and all(not state.values.requires_grad for state in checked[7])
    assert checked[2] == [("memory", 2), ("memory", 1), ("memory", 1),
                          ("diagnostic_repeat_memory", 2), ("diagnostic_repeat_memory", 1),
                          ("diagnostic_repeat_memory", 1)]
    result, costs = checked[:2]
    assert result["encoding_input_tokens"]["memory"] == result["encoding_input_tokens"]["diagnostic_repeat_memory"] == 16
    assert result["encoding_calls"]["diagnostic_repeat_memory"] == 3
    assert all(row["ce_targets"] == 0 for row in costs)
    repeat = result["repeat_diagnostics"]
    assert len(repeat["records"]) == 4 and len(repeat["final_records"]) == 2
    assert repeat["summary"]["multiwrite_trajectories"] == repeat["summary"]["single_write_trajectories"] == 1
    assert all(row["relative_state_error"] == 0. for row in repeat["records"])
    assert repeat["final_records"][0]["tail_window_writes"] == 3
    assert repeat["final_records"][0]["tail_relative_error_range"] == 0.
    assert not repeat["threshold_stop"]
    assert repeat["state_perturbations_injected"] == repeat["CE_targets"] == repeat["backward_calls"] == 0
    assert result["diagnostic_seconds"] > 0 and baseline[0]["diagnostic_seconds"] == 0
    assert baseline[0]["repeat_diagnostics"] is None


def test_observed_repeat_differences_are_retained_without_reset_or_stop(monkeypatch):
    result = run_case(monkeypatch, diagnose=True, drift=.1)[0]
    repeat = result["repeat_diagnostics"]
    assert all(row["relative_state_error"] > 0 for row in repeat["records"])
    assert repeat["final_records"][0]["tail_relative_error_range"] > 0
    assert not repeat["threshold_stop"]
    assert repeat["summary"]["final_relative_state_error_multiwrite"]["maximum"] > 0


def test_mutating_repeat_writer_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="mutated its input"):
        run_case(monkeypatch, diagnose=True, mutate=True)


def test_repeat_monitor_rejects_nonempty_initial_memory_and_residual_mode():
    state = update(LazyDecayState.empty(1, 2, 100), torch.ones(1, 2))
    with pytest.raises(ValueError, match="empty one-owner"):
        schedule._empty_repeat_state(state)
    with pytest.raises(ValueError, match="only a full writer"):
        schedule.build_causal_writer_histories(SimpleNamespace(), [], gather=lambda value: [value],
            write=lambda *args: None, encoding_mode="residual", diagnostic_repeat_write=lambda *args: None)


def test_empty_rank_repeats_padded_collectives_without_any_local_write(monkeypatch):
    session = SimpleNamespace(device=torch.device("cpu"), layout=SimpleNamespace(row_capacity=100, row_width=2))
    integers = iter((2, 1, 0, 1, 0))
    remote = {"trajectory_index": 7, "frame_index": 0, "write_index": 0, "state_error_l2": 0.,
              "reference_state_l2": 1., "relative_state_error": 0.}
    def gather(value):
        if type(value) is int:
            return [value, next(integers)]
        assert value == []
        return [value, [remote]]
    calls = []
    def encode(session, batches, *, encoding):
        mb = batches[0]
        assert mb["frames"] == (None,) and mb["reader_state"].rows.numel() == 0
        calls.append(encoding)
        return {"microbatches": [{"hidden": torch.zeros(1, 640, 2560),
                                  "physical_rows": torch.zeros(1, 640, 16, dtype=torch.long)}]}
    monkeypatch.setattr(schedule, "collect_memory_encoding_microbatches", encode)
    result = schedule.build_causal_writer_histories(session, [], gather=gather,
        write=lambda *args: pytest.fail("inactive primary writer"), encoding_mode="full",
        diagnostic_repeat_write=lambda *args: pytest.fail("inactive repeat writer"))
    assert calls == ["memory", "memory"]
    assert result["encoding_calls"]["diagnostic_repeat_memory"] == 1
    assert result["encoding_input_tokens"]["diagnostic_repeat_memory"] == 0
    assert result["encoding_padded_input_tokens"]["diagnostic_repeat_memory"] == 640
    assert result["repeat_diagnostics"]["records"] == [remote]

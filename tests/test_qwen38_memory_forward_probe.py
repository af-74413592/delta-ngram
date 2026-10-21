"""Label-free CPU probe: causal G baseline, independent twins and fixed costs."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.alpha_memory import BlockAlpha
from delta_ngram.batching import BatchMemoryState
from delta_ngram.qwen38_staged_batching import materialize_staged_snapshot
from delta_ngram.qwen38_train_data import _frame
from delta_ngram.state import SparseMemoryState
from delta_ngram.qwen38_writer import NativeForwardWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
import qwen38_native_memory_forward_schedule as probe


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def plan(length, first_token):
    return [{"query": {"loss_target_tokens": 7, "must_never_score": True},
             "evidence": _frame({}, (first_token + index,) * 3, 0, 3, query=False)
             if index < length - 1 else None} for index in range(length)]


class Alpha(torch.nn.Module):
    kind = "fixed"
    initial_alpha = 1.


def test_six_forward_tracks_keep_causal_baseline_and_perturb_only_values(monkeypatch):
    session = SimpleNamespace(device=torch.device("cpu"),
                              layout=SimpleNamespace(row_capacity=100, row_width=2))
    writer, alpha = torch.nn.Linear(1, 1), Alpha()
    writer_before = {key: value.clone() for key, value in writer.state_dict().items()}
    original = LazyDecayState.from_batch(BatchMemoryState.empty(1, 2, 100))
    local = [{"index": 4, "state": original, "plan": plan(4, 1)}]
    observed, costs, gains = [], [], []
    passed_kinds = ("base", "memory_base", "reference", "repeat", "perturbed_0", "perturbed_1")

    def encode(session, batches, *, encoding):
        mb = batches[0]
        assert not torch.is_grad_enabled() and mb["batch"]["labels"] is None
        kind = passed_kinds[len(observed) % 6]
        old = 0.
        if encoding == "memory":
            values = mb["reader_state"].values
            assert not values.requires_grad
            old = float(values.mean()) if values.numel() else 0.
        else:
            assert kind == "base"
        token = mb["frames"][0]["input_ids"][0]
        observed.append((kind, token, old))
        hidden = torch.full((1, 640, 2560), token + old)
        return {"microbatches": [{"hidden": hidden,
                                  "physical_rows": mb["valid_mask"].long()[..., None].expand(-1, -1, 16)}]}

    def measure(writer, state, block_mem, block_base, **kwargs):
        token = float(block_base["H_new"].mean())
        old = float(state.values.mean()) if state.values.numel() else 0.
        # G tracks the baseline history, not either memory-aware twin.
        assert float(block_mem["H_new"].mean()) == pytest.approx(token + old)
        assert block_mem["physical_rows_read"].shape == (3, 16)
        assert torch.equal(block_mem["physical_rows_read"], block_base["physical_rows_read"])
        gains.append(old)
        return {"mode": "full", "denominator_floor": 1e-8,
                "block_g": .2 if old else None, "row_output_gain_values": [.1, .2] if old else [],
                "written_rows": 2, "eligible_written_rows": 2 if old else 0,
                "filtered_written_rows": 0 if old else 2}

    def write(writer, previous, hidden, rows, mask, alpha, *, event_observer):
        assert not torch.is_grad_enabled() and not hidden.requires_grad
        old = previous.values if previous.values.numel() else torch.zeros(1, 2)
        feature = hidden.mean()
        # Nonlinear dependence exposes accidental sharing and premature H.
        values = .5 * old + feature * torch.tensor([[1., 1.5]])
        count = previous.counts + 1 if previous.counts.numel() else torch.ones(1)
        clock = previous.block_clock + 1
        memory = BatchMemoryState(SparseMemoryState(torch.tensor([1]), values, count), 1, 100)
        event_observer(SimpleNamespace(beta=torch.tensor([.1, .2]), target=values))
        return LazyDecayState(memory, clock, previous.cumulative_log_alpha,
                              clock.clone(), torch.zeros(1, dtype=torch.float64))

    monkeypatch.setattr(probe, "collect_memory_encoding_microbatches", encode)
    monkeypatch.setattr(probe, "measure_encoding_sensitivity", measure)
    monkeypatch.setattr(probe, "writer_evidence_features", lambda hidden, **kw: hidden.detach())
    monkeypatch.setattr(probe, "write_memory_aware_native_block", write)
    result = probe.run_memory_forward_probe(
        session, writer, alpha, local, gather=lambda value: [value], on_encoding=costs.append,
        protocol={"expected_global_trajectories": 1})
    assert len(observed) == len(costs) == 18
    assert [kind for kind, *_ in observed] == list(passed_kinds) * 3
    assert [value for _, _, value in observed[:6]] == [0.] * 6
    # Baseline S1 mean=1.25, S2 mean=3.125. Twin S2 includes H_mem feedback.
    assert gains == pytest.approx([0., 1.25, 3.125])
    assert observed[8][2] == observed[9][2] == pytest.approx(1.25)
    assert observed[14][2] == observed[15][2] == pytest.approx(4.6875)
    assert observed[10][2] != observed[8][2] and observed[11][2] != observed[8][2]
    assert result["ce_targets"] == result["backward_calls"] == result["optimizer_steps"] == 0
    assert result["training_launch_authorized_by_this_probe"] is False
    assert all(row["ce_targets"] == 0 for row in costs)
    assert sum(result["costs"]["encoding_input_tokens"].values()) == 54
    assert sum(result["costs"]["collective_calls"].values()) == 18
    assert result["costs"]["rank_call_sum"] == result["costs"]["collective_calls"]
    assert len(result["perturbation_seeds"]) == 2
    assert {row["seed"] for row in result["perturbation_seeds"]} == {21}
    assert len(result["twin_trajectory_records"]) == 6
    for row in result["twin_trajectory_records"]:
        assert row["repeat_error_l2"] == 0
        if row["blocks_after_seed"] == 0:
            assert row["error_over_initial"] == pytest.approx(1.)
    assert result["encoding_replacement"]["eligible_blocks"] == 2
    assert all("row_output_gain_values" not in row for row in result["encoding_replacement_records"])
    assert original.values.numel() == original.rows.numel() == 0 and original.block_clock.item() == 0
    assert local[0]["state"] is original  # Caller plan and empty state remain usable.
    for key, value in writer.state_dict().items():
        assert torch.equal(value, writer_before[key])
    assert all(parameter.grad is None for parameter in writer.parameters())
    json.dumps(result, allow_nan=False)


def test_lazy_perturbation_preserves_clock_counts_and_actual_relative_scale():
    values = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
    batch = BatchMemoryState(SparseMemoryState(torch.tensor([1, 3]), values, torch.tensor([2., 5.])), 1, 100)
    original = LazyDecayState(batch, torch.tensor([4]), torch.tensor([-.5], dtype=torch.float64),
                              torch.tensor([2, 3]), torch.tensor([-.2, -.3], dtype=torch.float64))
    states, records = probe._seed_perturbations(
        original, relative_epsilons=[1e-3, 1e-2], seed=23, absolute_floor=1e-8)
    again, _ = probe._seed_perturbations(
        original, relative_epsilons=[1e-3, 1e-2], seed=23, absolute_floor=1e-8)
    for state, repeated, record in zip(states, again, records):
        assert torch.equal(state.values, repeated.values)
        assert not state.values.requires_grad
        assert record["actual_relative_perturbation"] == pytest.approx(record["relative_epsilon"], rel=1e-4)
        assert state.counts.tolist() == [2., 5.] and state.block_clock.item() == 4
        assert torch.equal(state.row_last_written_block, original.row_last_written_block)
        assert state.values.data_ptr() != original.values.data_ptr()
    assert torch.equal(original.values, values)
    effective = materialize_staged_snapshot(original)
    first = materialize_staged_snapshot(states[0]).values - effective.values
    second = materialize_staged_snapshot(states[1]).values - effective.values
    torch.testing.assert_close(second, first * 10, rtol=2e-3, atol=1e-6)


def test_unexpected_protocol_and_nonempty_initial_state_fail_before_encoding():
    with pytest.raises(ValueError, match="unknown"):
        probe.forward_probe_protocol({"train": True})
    with pytest.raises(ValueError, match="two positive"):
        probe.forward_probe_protocol({"relative_epsilons": [0., .1]})
    session = SimpleNamespace(device=torch.device("cpu"))
    initial = BatchMemoryState(SparseMemoryState(torch.tensor([1]), torch.ones(1, 2), torch.ones(1)), 1, 100)
    with pytest.raises(ValueError, match="start empty"):
        probe.run_memory_forward_probe(
            session, torch.nn.Linear(1, 1), Alpha(), [{"index": 0, "state": initial, "plan": plan(2, 1)}],
            gather=lambda value: [value], protocol={"expected_global_trajectories": 1})


def test_empty_rank_executes_all_six_padded_collectives_without_writes(monkeypatch):
    session = SimpleNamespace(device=torch.device("cpu"),
                              layout=SimpleNamespace(row_capacity=100, row_width=2))
    calls = []
    integer_gathers = iter((2, 1, 0))  # Two remote frames; only first writes.

    def gather(value):
        if type(value) is int:
            return [value, next(integer_gathers)]
        if isinstance(value, list):
            assert value == []
            return [value, [3]]
        if isinstance(value, dict) and "gains" in value:
            # The remote rank did the same native calls and had one trajectory.
            remote = {**value, "trajectories": 1}
            return [value, remote]
        return [value, value]

    def encode(session, batches, *, encoding):
        mb = batches[0]
        assert mb["frames"] == (None,)
        assert not mb["valid_mask"].any() and mb["real_input_tokens"] == 0
        if encoding == "memory":
            assert not mb["reader_state"].rows.numel()
        calls.append(encoding)
        return {"microbatches": [{"hidden": torch.zeros(1, 640, 2560),
                                  "physical_rows": torch.zeros(1, 640, 16, dtype=torch.long)}]}

    monkeypatch.setattr(probe, "collect_memory_encoding_microbatches", encode)
    monkeypatch.setattr(probe, "write_memory_aware_native_block",
                        lambda *a, **kw: pytest.fail("inactive rank cannot write"))
    result = probe.run_memory_forward_probe(
        session, torch.nn.Linear(1, 1), Alpha(), [], gather=gather,
        protocol={"expected_global_trajectories": 1})
    assert calls == ["base"] + ["memory"] * 5
    assert result["trajectories"] == 1 and result["ready_waves"] == 2
    assert sum(result["costs"]["collective_calls"].values()) == 6
    assert sum(result["costs"]["rank_call_sum"].values()) == 12
    assert sum(result["costs"]["encoding_padded_input_tokens"].values()) == 2 * 6 * 640


def test_forward_probe_integrates_actual_native_writer_and_strict_sensitivity(monkeypatch):
    session = SimpleNamespace(device=torch.device("cpu"),
                              layout=SimpleNamespace(row_capacity=100, row_width=160))
    writer = NativeForwardWriter(width=160, hidden_dim=4, input_dim=2560, heads_per_order=8)
    alpha = BlockAlpha(width=2560, kind="fixed", initial_alpha=1.)
    initial = LazyDecayState.empty(1, 160, 100)

    def encode(session, batches, *, encoding):
        mb = batches[0]
        token = mb["frames"][0]["input_ids"][0]
        base = torch.linspace(-1., 2., 2560) + token * .1
        hidden = base.expand(1, 640, 2560).clone()
        if encoding == "memory" and mb["reader_state"].values.numel():
            hidden[..., 0] += mb["reader_state"].values.sum()
        return {"microbatches": [{"hidden": hidden,
                                  "physical_rows": mb["valid_mask"].long()[..., None].expand(-1, -1, 16)}]}

    monkeypatch.setattr(probe, "collect_memory_encoding_microbatches", encode)
    result = probe.run_memory_forward_probe(
        session, writer, alpha, [{"index": 0, "state": initial, "plan": plan(3, 1)}],
        gather=lambda value: [value], protocol={"expected_global_trajectories": 1})
    assert result["encoding_replacement"]["eligible_blocks"] == 1
    assert result["encoding_replacement_records"][1]["matched_read_addresses_verified"]
    assert len(result["perturbation_seeds"]) == 2
    assert len(result["twin_trajectory_records"]) == 4
    assert all(row["evolved_trajectories"] == 1 and row["seed_only_trajectories"] == 0
               for row in result["twin_trajectory_summary"])
    assert all(parameter.grad is None for parameter in writer.parameters())
    assert not initial.values.numel()
    json.dumps(result, allow_nan=False)


def test_single_write_seed_only_trajectory_has_no_final_dynamics_distribution(monkeypatch):
    session = SimpleNamespace(device=torch.device("cpu"),
                              layout=SimpleNamespace(row_capacity=100, row_width=160))
    writer = NativeForwardWriter(width=160, hidden_dim=4, input_dim=2560, heads_per_order=8)
    alpha = BlockAlpha(width=2560, kind="fixed", initial_alpha=1.)
    initial = LazyDecayState.empty(1, 160, 100)

    def encode(session, batches, *, encoding):
        mb = batches[0]
        hidden = torch.linspace(-1., 2., 2560).expand(1, 640, 2560).clone()
        return {"microbatches": [{"hidden": hidden,
                                  "physical_rows": mb["valid_mask"].long()[..., None].expand(-1, -1, 16)}]}

    monkeypatch.setattr(probe, "collect_memory_encoding_microbatches", encode)
    result = probe.run_memory_forward_probe(
        session, writer, alpha, [{"index": 0, "state": initial, "plan": plan(2, 1)}],
        gather=lambda value: [value], protocol={"expected_global_trajectories": 1})
    assert len(result["twin_trajectory_records"]) == 2
    assert all(row["error_over_initial"] == 1. and row["blocks_after_seed"] == 0
               for row in result["twin_trajectory_records"])
    for summary in result["twin_trajectory_summary"]:
        assert summary["seeded_trajectories"] == summary["seed_only_trajectories"] == 1
        assert summary["evolved_trajectories"] == summary["post_seed_write_observations"] == 0
        for metric in ("error_over_initial_all_post_seed", "error_over_initial_final",
                       "error_over_reference_final"):
            assert summary[metric] == {"median": None, "p99": None, "maximum": None}
    json.dumps(result, allow_nan=False)

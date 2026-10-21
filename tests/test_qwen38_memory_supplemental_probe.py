"""CPU checks of the H12 control, final-state scoring boundary and fixed costs."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.qwen38_train_data import _frame
from delta_ngram.state import SparseMemoryState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
import qwen38_native_memory_supplemental_schedule as probe


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class Alpha(torch.nn.Module):
    kind = "fixed"
    initial_alpha = 1.


def build_case(monkeypatch, *, writes=3, g=2.):
    session = SimpleNamespace(device=torch.device("cpu"), layout=SimpleNamespace(row_capacity=100, row_width=2))
    writer, alpha = torch.nn.Linear(1, 1), Alpha()
    original = LazyDecayState.empty(1, 2, 100)
    plan = [{"query": None, "evidence": _frame({}, (index + 1,) * 3, 0, 3, query=False)} for index in range(writes)]
    plan.append({"query": {"loss_target_tokens": 7}, "evidence": None})
    local = [{"index": 4, "state": original, "plan": plan}]
    h12 = {(4, ordinal): {"physical_rows_new": torch.ones(1, 3, 16, dtype=torch.long),
                           "H_new": torch.full((1, 3, 2560), float(ordinal + 11))} for ordinal in range(writes)}
    observed, costs, write_inputs, gain_states = [], [], [], []
    kinds = ("base", "memory_base", "reference", "repeat", "perturbed_0", "perturbed_1")

    def encode(session, batches, *, encoding):
        mb = batches[0]
        assert not torch.is_grad_enabled() and mb["batch"]["labels"] is None
        kind = kinds[len(observed) % 6]
        old = 0.
        if encoding == "memory":
            values = mb["reader_state"].values
            assert not values.requires_grad
            old = float(values.mean()) if values.numel() else 0.
        token = mb["frames"][0]["input_ids"][0]
        observed.append((kind, token, old))
        return {"microbatches": [{"hidden": torch.full((1, 640, 2560), token + old),
            "physical_rows": mb["valid_mask"].long()[..., None].expand(-1, -1, 16)}]}

    def measure(writer, state, mem, base, **kwargs):
        old = float(state.values.mean()) if state.values.numel() else 0.
        assert float(mem["H_new"].mean()) == pytest.approx(float(base["H_new"].mean()) + old)
        assert kwargs["include_relationship"] is True
        gain_states.append(old)
        return {"mode": "full", "denominator_floor": 1e-8, "block_g": g if old else None,
            "row_output_gain_values": [.1] if old else [], "written_rows": 1,
            "eligible_written_rows": 1 if old else 0, "filtered_written_rows": 0 if old else 1}

    def write(writer, state, hidden, rows, mask, alpha, *, event_observer):
        assert not torch.is_grad_enabled() and not hidden.requires_grad
        write_inputs.append((hidden.data_ptr(), float(hidden.mean())))
        old = state.values if state.values.numel() else torch.zeros(1, 2)
        values = .5 * old + hidden.mean() * torch.tensor([[1., 1.5]])
        count = state.counts + 1 if state.counts.numel() else torch.ones(1)
        clock = state.block_clock + 1
        memory = BatchMemoryState(SparseMemoryState(torch.tensor([1]), values, count), 1, 100)
        event_observer(SimpleNamespace(beta=torch.tensor([.1, .2]), target=values))
        return LazyDecayState(memory, clock, state.cumulative_log_alpha, clock.clone(), torch.zeros(1, dtype=torch.float64))

    monkeypatch.setattr(probe, "collect_memory_encoding_microbatches", encode)
    monkeypatch.setattr(probe, "measure_encoding_sensitivity", measure)
    monkeypatch.setattr(probe, "write_memory_aware_native_block", write)
    monkeypatch.setattr(probe, "writer_evidence_features", lambda hidden, **kwargs: hidden.detach())
    kwargs = dict(gather=lambda value: [value], h12_base_blocks=h12, on_encoding=costs.append,
        protocol={"expected_global_trajectories": 1, "expected_evolved_trajectories": int(writes > 1)})
    return (session, writer, alpha, local), kwargs, SimpleNamespace(
        observed=observed, costs=costs, write_inputs=write_inputs, gains=gain_states, h12=h12, original=original)


def test_h12_base_twins_share_hidden_without_extra_forward_and_score_four_evolved_states(monkeypatch):
    args, kwargs, seen = build_case(monkeypatch)
    scored, captured = [], []
    def score(tasks):
        assert not torch.is_grad_enabled() and len(tasks) == 1
        task = tasks[0]
        assert task["index"] == task["trajectory_index"] == 4
        assert task["frame"] is task["plan"][-1]["query"]
        assert task["plan"][-1]["evidence"] is None
        assert task["write_count"] == 3 and task["blocks_after_seed"] == 2
        assert tuple(task["states"]) == probe.FINAL_CONDITIONS
        assert len({state.values.data_ptr() for state in task["states"].values()}) == 4
        assert all(state.counts.tolist() == [3.] for state in task["states"].values())
        assert float(task["states"]["base_ref"].values.mean()) == pytest.approx(27.1875)
        assert float(task["states"]["memory_ref"].values.mean()) == pytest.approx(11.953125)
        scored.extend(tasks)
        return {"ce_targets": 28, "nll_pairs": []}
    def capture(entry):
        assert entry["H12_block"] is seen.h12[(4, entry["frame_index"])]
        assert float(entry["H12_block"]["H_new"].mean()) - float(entry["base_block"]["H_new"].mean()) == 10.
        captured.append(entry["frame_index"])
        return {"index": 4, "frame": entry["frame_index"]}
    result = probe.run_memory_supplemental_probe(*args, **kwargs, on_final_states=score, on_capture=capture)
    assert len(seen.observed) == len(seen.costs) == 18 and len(seen.write_inputs) == 24
    for ordinal in range(3):
        block = seen.write_inputs[8 * ordinal:8 * (ordinal + 1)]
        assert len({block[index][0] for index in (0, 5, 6, 7)}) == 1
        assert all(block[index][1] == ordinal + 11 for index in (0, 5, 6, 7))
    assert seen.gains == pytest.approx([0., 13.75, 21.875])
    assert result["costs"]["base_twin_extra_encoding_calls"] == 0
    assert sum(result["costs"]["collective_calls"].values()) == 18
    assert sum(result["costs"]["encoding_input_tokens"].values()) == 54
    assert result["ce_targets"] == 28 and result["backward_calls"] == result["optimizer_steps"] == 0
    assert result["encoding_replacement"]["block_g"]["p99"] == 2.
    assert not result["encoding_replacement"]["stop_requested"]
    assert result["final_query_trajectory_indices"] == [4] and len(result["capture_records"]) == 3
    assert captured == [0, 1, 2] and len(scored) == 1
    assert len(result["perturbation_seeds"]) == 4 and len(result["twin_trajectory_records"]) == 12
    for row in result["twin_trajectory_records"]:
        assert row["repeat_error_l2"] == 0.
        if row["blocks_after_seed"] == 0:
            assert row["error_over_initial"] == 1.
    assert seen.original.values.numel() == 0 and seen.original.block_clock.item() == 0
    assert all(parameter.grad is None for parameter in args[1].parameters())
    json.dumps(result, allow_nan=False)


def test_seed_only_trajectories_are_excluded_from_final_query_callback(monkeypatch):
    args, kwargs, seen = build_case(monkeypatch, writes=1)
    called = []
    def score(tasks):
        called.append(tasks)
        return {"ce_targets": 0}
    result = probe.run_memory_supplemental_probe(*args, **kwargs, on_final_states=score)
    assert called == [[]] and result["final_query_trajectory_indices"] == []
    assert all(row["seed_only_trajectories"] == 1 and row["evolved_trajectories"] == 0
               for row in result["twin_trajectory_summary"])


@pytest.mark.parametrize("case", ["missing", "wrong_rows"])
def test_actual_h12_required_and_address_checked(monkeypatch, case):
    args, kwargs, seen = build_case(monkeypatch)
    if case == "missing":
        kwargs["h12_base_blocks"] = {}
    else:
        seen.h12[(4, 0)]["physical_rows_new"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="H12"):
        probe.run_memory_supplemental_probe(*args, **kwargs)
    if case == "missing":
        assert not seen.observed
    assert not seen.write_inputs


def test_final_state_mutation_and_unregistered_scoring_population_fail(monkeypatch):
    args, kwargs, seen = build_case(monkeypatch)
    def mutate(tasks):
        tasks[0]["states"]["base_ref"].values.add_(1)
        return {"ce_targets": 28}
    with pytest.raises(ValueError, match="mutated"):
        probe.run_memory_supplemental_probe(*args, **kwargs, on_final_states=mutate)
    args, kwargs, seen = build_case(monkeypatch)
    kwargs["protocol"]["expected_evolved_trajectories"] = 0
    with pytest.raises(ValueError, match="trajectory set"):
        probe.run_memory_supplemental_probe(*args, **kwargs,
            on_final_states=lambda tasks: pytest.fail("must reject population before scoring"))


@pytest.mark.parametrize("overrides", [{"train": True}, {"relative_epsilons": [.001, .02]},
    {"expected_evolved_trajectories": True}, {"expected_evolved_trajectories": 13}])
def test_protocol_rejects_unregistered_settings(overrides):
    with pytest.raises(ValueError):
        probe.supplemental_probe_protocol(overrides)

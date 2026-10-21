"""Repeat-only controls keep independent states and inject no perturbation."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from delta_ngram.alpha_memory import BlockAlpha, LazyDecayState
from delta_ngram.qwen38_train_data import _frame
from delta_ngram.qwen38_writer import NativeForwardWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
import qwen38_native_memory_repeat_schedule as probe
import qwen38_native_memory_probe_runtime as runtime


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def inputs(writes=3):
    session = SimpleNamespace(device=torch.device("cpu"), layout=SimpleNamespace(row_capacity=100, row_width=2))
    writer = NativeForwardWriter(width=2, input_dim=2560, hidden_dim=4, heads_per_order=8)
    alpha = BlockAlpha(2560, kind="fixed", initial_alpha=1.)
    state = LazyDecayState.empty(1, 2, 100)
    plan = [{"evidence": _frame({}, (index + 1,) * 3, 0, 3, query=False), "query": None}
            for index in range(writes)] + [{"evidence": None, "query": {"loss_target_tokens": 7}}]
    h12 = {(4, ordinal): {"physical_rows_new": torch.ones(1, 3, 16, dtype=torch.long),
        "H_new": (torch.linspace(-1., 2., 2560) + ordinal).expand(1, 3, 2560).clone()} for ordinal in range(writes)}
    return (session, writer, alpha, [{"index": 4, "state": state, "plan": plan}]), h12


@pytest.mark.parametrize("repeat_difference", [0., .1])
def test_only_two_memory_encodings_no_injection_and_four_independent_final_states(monkeypatch, repeat_difference):
    torch.manual_seed(19)
    args, h12 = inputs()
    observed, costs, scored = [], [], []
    def encode(session, batches, *, encoding):
        mb = batches[0]
        assert encoding == "memory" and not torch.is_grad_enabled()
        values = mb["reader_state"].values
        old = float(values.sum()) if values.numel() else 0.
        repeated = len(observed) % 2
        observed.append(old)
        hidden = (torch.linspace(-2., 1., 2560) + old).expand(1, 640, 2560).clone()
        hidden[..., 0] += repeated * repeat_difference
        return {"microbatches": [{"hidden": hidden,
            "physical_rows": mb["valid_mask"].long()[..., None].expand(-1, -1, 16)}]}
    def score(tasks):
        assert not torch.is_grad_enabled() and len(tasks) == 1
        states = tasks[0]["states"]
        assert tuple(states) == probe.FINAL_CONDITIONS
        assert len({state.values.data_ptr() for state in states.values()}) == 4
        assert torch.equal(states["base_ref"].values, states["base_repeat"].values)
        assert torch.equal(states["memory_ref"].values, states["memory_repeat"].values) == (repeat_difference == 0.)
        assert all(state.counts.tolist() == [3.] for state in states.values())
        scored.extend(tasks)
        return {"ce_targets": 28}
    monkeypatch.setattr(probe, "collect_memory_encoding_microbatches", encode)
    result = probe.run_memory_repeat_probe(*args, gather=lambda value: [value], h12_base_blocks=h12,
        on_encoding=costs.append, on_final_states=score,
        protocol={"expected_global_trajectories": 1, "expected_evolved_trajectories": 1})
    assert len(observed) == len(costs) == 6
    assert observed[:2] == [0., 0.]
    assert {row["kind"] for row in costs} == {"repeat_memory_ref", "repeat_memory_repeat"}
    assert sum(result["costs"]["collective_calls"].values()) == 6
    assert result["costs"]["base_control_extra_encoding_calls"] == 0
    assert result["state_perturbations_injected"] == result["backward_calls"] == result["optimizer_steps"] == 0
    assert result["ce_targets"] == 28 and result["final_query_trajectory_indices"] == [4]
    assert len(result["repeat_trajectory_records"]) == 6
    assert sum(row["first_write_from_empty"] for row in result["repeat_trajectory_records"]) == 2
    assert not args[-1][0]["state"].values.numel()
    assert all(parameter.grad is None for parameter in args[1].parameters())
    json.dumps(result, allow_nan=False)


def test_single_write_still_recorded_but_never_scored(monkeypatch):
    args, h12 = inputs(writes=1)
    def encode(session, batches, *, encoding):
        mb = batches[0]
        return {"microbatches": [{"hidden": torch.ones(1, 640, 2560),
            "physical_rows": mb["valid_mask"].long()[..., None].expand(-1, -1, 16)}]}
    monkeypatch.setattr(probe, "collect_memory_encoding_microbatches", encode)
    called = []
    def score(tasks):
        called.append(tasks)
        return {"ce_targets": 0}
    result = probe.run_memory_repeat_probe(*args, gather=lambda value: [value], h12_base_blocks=h12,
        on_final_states=score, protocol={"expected_global_trajectories": 1, "expected_evolved_trajectories": 0})
    assert called == [[]] and result["final_query_trajectory_indices"] == []
    assert len(result["repeat_trajectory_records"]) == 2


def test_repeat_score_adapter_preserves_state_objects_and_public_repeat_labels(monkeypatch):
    states = {key: object() for key in probe.FINAL_CONDITIONS}
    costs = []
    def scorer(session, tasks, *, gather, on_score, conditions, batch_limit):
        assert batch_limit == 3 and conditions == runtime.CONDITIONS
        assert list(tasks[0]["states"].values()) == list(states.values())
        for condition in conditions:
            on_score({"kind": condition, "ce_targets": 7})
        return {"protocol": "old", "conditions": list(conditions), "ce_targets": 28,
            "records": [{"condition": condition, "nll": 2.} for condition in conditions],
            "paired_records": [{"trajectory_index": 4, "memory_ref_nll": 2., "memory_eps01_nll": 3.,
                "base_ref_nll": 4., "base_eps01_nll": 5., "memory_perturbed_minus_reference_nll": 1.,
                "base_perturbed_minus_reference_nll": 1.}],
            "summary": {"memory_perturbed_minus_reference_nll": {"trajectory_mean": 1.},
                        "base_perturbed_minus_reference_nll": {"trajectory_mean": 1.}}}
    monkeypatch.setattr(runtime, "score_final_query_states", scorer)
    result = probe.score_repeat_query_states(None, [{"states": states}], gather=lambda value: [value], on_score=costs.append)
    assert result["conditions"] == list(probe.FINAL_CONDITIONS)
    assert [row["kind"] for row in costs] == list(probe.FINAL_CONDITIONS)
    assert {row["condition"] for row in result["records"]} == set(probe.FINAL_CONDITIONS)
    assert result["paired_records"][0]["memory_repeat_nll"] == 3.
    assert result["summary"]["memory_repeat_minus_reference_nll"]["trajectory_mean"] == 1.
    assert "eps01" not in json.dumps(result) and "perturbed_minus" not in json.dumps(result)


@pytest.mark.parametrize("override", [{"relative_epsilons": [.01]}, {"perturbation_seed": 17},
    {"batch_limit": 4}, {"absolute_floor": True}, {"expected_evolved_trajectories": 13}])
def test_repeat_protocol_cannot_enable_perturbations_or_change_batch(override):
    with pytest.raises(ValueError):
        probe.repeat_probe_protocol(override)

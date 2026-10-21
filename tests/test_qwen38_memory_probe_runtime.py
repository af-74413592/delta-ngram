"""CPU probe I/O contracts; these do not claim native EP/GPU acceptance."""
import json
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
from delta_ngram.qwen38_training import writer_evidence_features
from delta_ngram.qwen38_writer import NativeForwardWriter
from delta_ngram.state import SparseMemoryState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
import qwen38_native_memory_probe_runtime as runtime


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def frame(start=0, end=4, *, query=False, token=1):
    return _frame({}, tuple(range(token, token + 40)), start, end, query=query)


def memory(value, width=160):
    if value is None:
        return BatchMemoryState.empty(1, width, 100)
    values = torch.full((1, width), float(value), requires_grad=True)
    return BatchMemoryState(SparseMemoryState(torch.tensor([1]), values.square(), torch.ones(1)), 1, 100)


def task(index=7):
    final = frame(8, 11, query=True)
    return {"trajectory_index": index, "frame": final,
            "plan": [{"query": None, "evidence": frame()},
                     {"query": frame(4, 8, query=True), "evidence": frame(4, 8)},
                     {"query": final, "evidence": None}],
            "states": {key: memory(value) for key, value in zip(runtime.CONDITIONS, (1., 2., 3., 4.))}}


class QueryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.bound = None
        self.wrong_shape = False
        self.nonfinite = False
        self.mutate = False
        self.precision_case = False

    def forward(self, input_ids, labels, **kwargs):
        assert not torch.is_grad_enabled()
        assert labels.shape == input_ids.shape
        random.random()
        np.random.random()
        torch.rand(2)
        state = self.bound["state"]
        assert not state.values.requires_grad
        self.calls.append({"labels": labels.clone(), "state": state})
        rows = torch.ones((input_ids.shape[0], 1), dtype=torch.long)
        per_lane = state.lookup(rows)[0].mean((-1, -2))
        losses = per_lane[:, None].expand_as(labels).double().clone()
        losses[labels.eq(-100)] = float("nan")
        if self.precision_case:
            losses[labels.ne(-100)] = torch.tensor([2 ** 24, 1., 1.], dtype=torch.float64)
        if self.wrong_shape:
            losses = losses.transpose(0, 1)
        if self.nonfinite:
            losses.fill_(float("inf"))
        if self.mutate:
            state.values.add_(1.)
        return losses


class Adapter:
    def __init__(self, model):
        self.model = model

    def run_native(self, call, **kwargs):
        assert kwargs["mode"] == "query"
        self.model.bound = kwargs
        return call(), SimpleNamespace()


def session(monkeypatch):
    model = QueryModel()
    monkeypatch.setattr(runtime, "_runtime", lambda session: None)
    return SimpleNamespace(models=[model], adapter=Adapter(model), device=torch.device("cpu"),
                           layout=SimpleNamespace(row_capacity=100, row_width=160))


def test_final_only_four_conditions_independent_states_precharged_and_rng_preserved(monkeypatch):
    native = session(monkeypatch)
    tasks = [task(7), task(9)]
    first = tasks[0]["states"]["memory_ref"]
    tasks[0]["states"]["memory_ref"] = LazyDecayState(
        first, torch.tensor([2]), torch.tensor([-.2], dtype=torch.float64),
        torch.tensor([1]), torch.tensor([-.1], dtype=torch.float64))
    originals = [(state.values, state.values.clone()) for item in tasks for state in item["states"].values()]
    costs = []

    def charge(cost):
        assert len(native.models[0].calls) == len(costs)
        costs.append(cost)

    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    expected = random.random(), np.random.random(), torch.rand(3)
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    result = runtime.score_final_query_states(native, tasks, gather=lambda value: [value], on_score=charge)
    actual = random.random(), np.random.random(), torch.rand(3)
    assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])
    assert len(costs) == len(native.models[0].calls) == 4
    assert result["ce_targets"] == 4 * sum(item["frame"]["loss_target_tokens"] for item in tasks)
    assert [cost["kind"] for cost in costs] == list(runtime.CONDITIONS)
    assert result["backward_calls"] == result["optimizer_steps"] == 0
    assert len(result["paired_records"]) == 2
    pair = result["paired_records"][1]
    assert pair["memory_perturbed_minus_reference_nll"] == 3.
    assert pair["base_perturbed_minus_reference_nll"] == 7.
    assert result["paired_records"][0]["memory_ref_nll"] == pytest.approx(np.exp(-.1))
    for original, saved in originals:
        assert torch.equal(original, saved)
    for call in native.models[0].calls:
        assert call["labels"].ne(-100).sum().item() == 6
        assert call["state"].values.data_ptr() not in {original.data_ptr() for original, _ in originals}
    json.dumps(result, allow_nan=False)


def test_float64_accumulation_retains_small_losses(monkeypatch):
    native = session(monkeypatch)
    native.models[0].precision_case = True
    result = runtime.score_final_query_states(native, [task()], gather=lambda value: [value], on_score=lambda cost: None)
    assert all(row["loss_sum"] == 2 ** 24 + 2 for row in result["records"])


def test_empty_local_rank_dispatches_zero_ce_and_gathers_remote_results(monkeypatch):
    native = session(monkeypatch)
    costs, gather_kinds = [], []

    def gather(value):
        if isinstance(value, list):
            gather_kinds.append("identities")
            assert value == []
            return [value, [7]]
        if isinstance(value, int):
            gather_kinds.append("maximum")
            assert value == 0
            return [0, 1]
        if "condition" in value:
            gather_kinds.append("description")
            return [value, value]
        gather_kinds.append("results")
        assert value["records"] == [] and value["ce_targets"] == 0 and value["calls"] == 4
        remote = {**value, "records": [{"trajectory_index": 7, "condition": condition,
                  "loss_sum": 4., "target_tokens": 2, "nll": 2.} for condition in runtime.CONDITIONS],
                  "ce_targets": 8, "real_input_tokens": 16}
        return [value, remote]

    result = runtime.score_final_query_states(native, [], gather=gather, on_score=costs.append)
    assert gather_kinds == ["identities", "maximum"] + ["description"] * 4 + ["results"]
    assert result["ce_targets"] == 8 and result["collective_calls"] == 4
    assert len(result["paired_records"]) == 1
    assert all(cost["ce_targets"] == cost["real_input_tokens"] == 0 for cost in costs)
    for call in native.models[0].calls:
        assert not call["labels"].ne(-100).any()
        assert call["state"].rows.tolist() == [1] and call["state"].values.eq(0).all()
    json.dumps(result, allow_nan=False)


def test_globally_empty_score_does_not_dispatch(monkeypatch):
    native = session(monkeypatch)
    result = runtime.score_final_query_states(native, [], gather=lambda value: [value],
        on_score=lambda cost: pytest.fail("globally empty scorer dispatched"))
    assert not native.models[0].calls and result["ce_targets"] == result["collective_calls"] == 0


def test_precharge_rejection_occurs_before_native_ce(monkeypatch):
    native = session(monkeypatch)

    def reject(cost):
        raise RuntimeError("CE budget exhausted")

    with pytest.raises(RuntimeError, match="budget"):
        runtime.score_final_query_states(native, [task()], gather=lambda value: [value], on_score=reject)
    assert native.models[0].calls == []


@pytest.mark.parametrize("problem", ["earlier_frame", "seed_only", "duplicate", "width"])
def test_invalid_final_scoring_task_rejected_before_dispatch(monkeypatch, problem):
    native = session(monkeypatch)
    item, gather = task(), lambda value: [value]
    if problem == "earlier_frame":
        item["frame"] = item["plan"][1]["query"]
    elif problem == "seed_only":
        item["plan"].pop(1)
    elif problem == "duplicate":
        gather = lambda value: [value, value]
    else:
        item["states"]["memory_ref"] = memory(1., width=2)
    with pytest.raises(ValueError):
        runtime.score_final_query_states(native, [item], gather=gather, on_score=lambda cost: None)
    assert native.models[0].calls == []


@pytest.mark.parametrize("problem", ["wrong_shape", "nonfinite", "mutate"])
def test_bad_native_loss_or_memory_mutation_rejected(monkeypatch, problem):
    native = session(monkeypatch)
    setattr(native.models[0], problem, True)
    item = task()
    before = item["states"]["memory_ref"].values.clone()
    with pytest.raises(ValueError):
        runtime.score_final_query_states(native, [item], gather=lambda value: [value], on_score=lambda cost: None)
    assert torch.equal(item["states"]["memory_ref"].values, before)


def test_h12_cache_keeps_trajectory_major_order_prefix_and_inactive_lanes(monkeypatch):
    native = session(monkeypatch)
    local = [{"index": index, "plan": [{"evidence": frame(4, 7, token=10 * index + ordinal)}
                                       for ordinal in range(count)]}
             for index, count in ((1, 7), (2, 6))]
    calls, costs = [], []

    def encode(session, batches, *, encoding):
        assert encoding == "base" and len(costs) == len(calls) + 1
        mb = batches[0]
        calls.append(mb)
        ids = mb["batch"]["input_ids"]
        assert mb["batch"]["labels"] is None and mb.get("reader_state") is None
        return {"microbatches": [{"hidden": ids[..., None].expand(-1, -1, 2560).float(),
                                  "physical_rows": mb["valid_mask"].long()[..., None].expand(-1, -1, 16)}]}

    monkeypatch.setattr(runtime, "collect_memory_encoding_microbatches", encode)
    result = runtime.collect_probe_h12_base(native, local, gather=lambda value: [value, 16], on_encoding=costs.append)
    assert [len(call["frames"]) for call in calls] == [12, 4]
    assert calls[1]["frames"][1:] == (None, None, None)
    assert len(result["blocks"]) == 13 and all(cost["ce_targets"] == 0 for cost in costs)
    for position, (key, block) in enumerate(result["blocks"].items()):
        index, ordinal = key
        assert block["encoding_lane"] == position % 12
        assert block["H_new"].shape == (1, 3, 2560)
        assert block["H_new"][0, 0, 0].item() == 10 * index + ordinal + 4
        assert block["physical_rows_read"].shape == (7, 16)
    assert result["costs"]["native_calls_rank_local"] == 2


def metric_entry(hidden, *, populated=False):
    rows = torch.tensor([[[1, 2, 3, 4], [1, 5, 6, 7], [1, 5, 6, 7]]])
    return {"trajectory_index": 7, "frame_index": 0, "state_before": memory(0. if populated else None, width=2),
            **{key: {"H_new": value.clone(), "physical_rows_new": rows.clone()}
               for key, value in zip(("H12_block", "base_block", "mem_block"), hidden)},
            "batch_size": 3, "lane": 1}


def test_feature_metrics_measure_actual_writer_feature_difference_with_head_multiplicity():
    torch.manual_seed(17)
    writer = NativeForwardWriter(2, input_dim=4, hidden_dim=5, heads_per_order=2, right_window=1)
    with torch.no_grad():
        writer.pool_score.weight.copy_(torch.tensor([[.2, -.4, .1, .3]]))
    h12 = torch.tensor([[[1., 2., 3., 4.], [4., 2., 1., 3.], [1., 3., 4., 2.]]])
    small = h12 + torch.tensor([.2, -.3, .1, .4])
    mem = small + torch.tensor([-.1, .5, .4, -.2])
    entry = metric_entry((h12, small, mem))
    originals = [parameter.detach().clone() for parameter in writer.parameters()]
    result = runtime.capture_batch_feature_metrics(writer, entry, event_chunk_size=3)
    normalized = [writer_evidence_features(value, mode="rmsnorm", eps=1e-6) for value in (h12, small, mem)]
    rows = entry["H12_block"]["physical_rows_new"]
    features = [writer.event_features(value, rows, torch.ones(1, 3, dtype=torch.bool)).as_tensor().detach().double()
                for value in normalized]
    noise, signal, actual = features[1] - features[0], features[2] - features[1], features[2] - features[0]
    measured = result["writer_features"]
    assert measured["elements"] == features[0].numel()
    assert measured["norm_l2"]["batch_noise"] == pytest.approx(float(noise.norm()))
    assert measured["norm_l2"]["matched_memory_delta"] == pytest.approx(float(signal.norm()))
    assert measured["norm_l2"]["actual_residual_delta"] == pytest.approx(float(actual.norm()))
    assert measured["noise_to_matched_memory_delta_l2"] == pytest.approx(float(noise.norm() / signal.norm()))
    assert measured["noise_matched_memory_delta_cosine"] == pytest.approx(float((noise * signal).sum() / noise.norm() / signal.norm()))
    assert result["state_empty"] and result["writer_events"] == 12
    assert result["rmsnorm_hidden"]["norm_l2"]["batch_noise"] != result["raw_hidden"]["norm_l2"]["batch_noise"]
    assert all(torch.equal(before, after) for before, after in zip(originals, writer.parameters()))
    assert all(parameter.grad is None for parameter in writer.parameters())
    json.dumps(result, allow_nan=False)


def test_feature_metrics_flag_empty_and_zero_denominators_and_reject_wrong_addresses():
    writer = NativeForwardWriter(2, input_dim=4, hidden_dim=5, heads_per_order=2)
    hidden = torch.ones(1, 3, 4)
    entry = metric_entry((hidden, hidden, hidden), populated=True)
    result = runtime.capture_batch_feature_metrics(writer, entry)
    assert not result["state_empty"] and result["state_effective_values_all_zero"]
    for space in ("raw_hidden", "rmsnorm_hidden", "writer_features"):
        assert result[space]["noise_to_matched_memory_delta_l2"] is None
        assert result[space]["denominator_filtered"]["signal"]
        assert result[space]["norm_l2"]["actual_residual_delta"] == 0.
    json.dumps(result, allow_nan=False)
    entry["base_block"]["physical_rows_new"][0, 0, 0] = 9
    with pytest.raises(ValueError, match="row addresses"):
        runtime.capture_batch_feature_metrics(writer, entry)

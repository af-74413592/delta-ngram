"""CPU contracts for causal detached encoding, not native GPU acceptance."""
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.qwen38_ep4_batching import SingleRankGradientPolicy
from delta_ngram.qwen38_memory_batching import (
    collate_memory_encoding_frames, detached_encoding_snapshot,
)
from delta_ngram.qwen38_staged_batching import pack_staged_query_tickets
from delta_ngram.qwen38_train_data import _frame, BOS
from delta_ngram.qwen38_training import NativeTrajectoryGradients
from delta_ngram.state import SparseMemoryState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
import qwen38_native_memory_schedule as schedule


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def frame(start=0, end=4, tokens=tuple(range(1, 20)), *, query=False):
    return _frame({}, tokens, start, end, query=query)


def memory(value, width=160):
    values = value if isinstance(value, torch.Tensor) else torch.full((1, width), float(value))
    return BatchMemoryState(SparseMemoryState(torch.tensor([1]), values, torch.ones(1)), 1, 100)


class FakeDist:
    @staticmethod
    def get_world_size():
        return 4

    @staticmethod
    def all_gather_object(copies, description):
        copies[:] = [description] * 4


def test_frame_protocol_preserves_bos_prefix_padding_and_independent_detached_owners():
    parameter = torch.full((1, 160), 2., requires_grad=True)
    original = memory(parameter.square())
    lazy = LazyDecayState(original, torch.tensor([3]), torch.tensor([-.4], dtype=torch.float64),
                          torch.tensor([1]), torch.tensor([-.1], dtype=torch.float64))
    first, later = frame(), frame(4, 7)
    assert first["input_ids"][0] != BOS and frame(query=True)["input_ids"][0] == BOS
    item = collate_memory_encoding_frames(
        [first, later, None], [lazy, memory(7), None], row_capacity=100)
    assert item["batch"]["labels"] is None and item["loss_target_tokens"] == 0
    assert item["batch"]["input_ids"][1, :8].tolist() == later["input_ids"]
    assert item["batch"]["position_ids"][1, :8].tolist() == later["position_ids"]
    assert item["valid_mask"][1, :7].all() and not item["valid_mask"][1, 7:].any()
    assert not item["write_mask"][1, :4].any() and item["write_mask"][1, 4:7].all()
    assert not item["valid_mask"][2].any() and not item["write_mask"][2].any()
    state = item["reader_state"]
    assert state.rows.tolist() == [1, 102]
    looked_up = state.lookup(torch.ones(3, 1, 1, dtype=torch.long))[0].flatten(1)
    torch.testing.assert_close(looked_up[0], torch.full((160,), 4 * np.exp(-.3)))
    assert looked_up[1].eq(7).all() and looked_up[2].eq(0).all()
    assert not state.values.requires_grad and state.values.grad_fn is None
    assert not state.counts.requires_grad
    state.values.add_(9)
    state.counts.add_(2)
    state.rows.add_(1)
    torch.testing.assert_close(original.values, parameter.square())
    assert original.counts.eq(1).all() and original.rows.tolist() == [1]
    assert lazy.block_clock.item() == 3
    snapshot = detached_encoding_snapshot(lazy)
    assert snapshot.rows.data_ptr() != lazy.rows.data_ptr()
    assert snapshot.counts.data_ptr() != lazy.counts.data_ptr()
    with pytest.raises(ValueError, match="inactive"):
        collate_memory_encoding_frames([None], [original], row_capacity=100)


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        raise AssertionError("no-CE encoding executed the LM head")


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_layer = Head()
        self.bound = None
        self.fail = False

    def forward(self, input_ids, labels, **kwargs):
        assert labels is None
        assert not torch.is_grad_enabled()
        random.random()
        np.random.random()
        torch.rand(3)
        if self.fail:
            raise RuntimeError("fake decoder failed")
        bound = self.bound
        real = bound["valid_mask"]
        rows = real.long()[..., None].expand(-1, -1, 16)
        addition = torch.zeros_like(input_ids, dtype=torch.float32)
        if bound["mode"] == "query":
            addition = bound["state"].lookup(rows)[0].mean((-1, -2))
        # Prefix memory reads influence later hidden states via causal context.
        hidden = (input_ids.float() * real + addition).cumsum(dim=1)
        self.expected_hidden = hidden[..., None].expand(-1, -1, 2560).to(torch.bfloat16).clone()
        return self.output_layer(self.expected_hidden.transpose(0, 1).contiguous())


class FakeAdapter:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def run_native(self, call, **kwargs):
        assert not torch.is_grad_enabled()
        state = kwargs["state"]
        if kwargs["mode"] == "evidence":
            assert state is None
        else:
            assert not state.values.requires_grad and state.values.grad_fn is None
        self.calls.append(kwargs)
        self.model.bound = kwargs
        output = call()
        assert output is None
        rows = kwargs["valid_mask"].long()[..., None].expand(-1, -1, 16).clone()
        return output, SimpleNamespace(physical_rows=rows)


def fake_session(monkeypatch):
    model = FakeModel()
    session = SimpleNamespace(raw=model, models=[model], adapter=FakeAdapter(model),
                              device=torch.device("cpu"),
                              layout=SimpleNamespace(row_capacity=100, row_width=160))
    monkeypatch.setattr(schedule, "_runtime", lambda session: (torch, FakeDist))
    monkeypatch.setattr(schedule, "language_model", lambda raw: raw)
    monkeypatch.setattr(schedule, "_evidence_head_identity", lambda raw: {"pinned_test": True})
    return session


def rng_samples():
    return random.random(), np.random.random(), torch.rand(3)


def assert_rng_unchanged(before, after):
    assert before[0:2] == after[0:2]
    assert torch.equal(before[2], after[2])


@pytest.mark.parametrize("kind", ["memory", "base"])
def test_no_ce_no_head_reads_prefix_per_lane_restores_rng_and_returns_owned_hidden(monkeypatch, kind):
    session = fake_session(monkeypatch)
    states = [memory(.5), memory(2), None] if kind == "memory" else None
    item = collate_memory_encoding_frames([frame(4, 7), frame(4, 7), None], states, row_capacity=100)
    random.seed(12)
    np.random.seed(13)
    torch.manual_seed(14)
    expected_rng = rng_samples()
    random.seed(12)
    np.random.seed(13)
    torch.manual_seed(14)
    result = schedule.collect_memory_encoding_microbatches(session, [item], encoding=kind)
    actual = result["microbatches"][0]
    assert_rng_unchanged(expected_rng, rng_samples())
    torch.testing.assert_close(actual["hidden"], session.raw.expected_hidden)
    # At the first new token, all four prefix memory reads already contributed.
    expected = 15 + (2.5 if kind == "memory" else 0)
    assert actual["hidden"][0, 4, 0].item() == expected
    assert actual["hidden"][1, 4, 0].item() == 15 + (10 if kind == "memory" else 0)
    assert actual["physical_rows"][0, :7].eq(1).all()
    assert actual["physical_rows"][0, 7:].eq(0).all()
    assert not actual["hidden"].requires_grad and actual["hidden"].grad_fn is None
    assert result["query_target_tokens"] == 0 and session.raw.output_layer.calls == 0
    assert not session.raw.output_layer._forward_pre_hooks
    session.raw.expected_hidden.zero_()
    assert actual["hidden"][0, 4, 0].item() == expected


def test_runtime_rejects_live_graph_and_failed_encoder_removes_hook_restores_rng(monkeypatch):
    session = fake_session(monkeypatch)
    item = collate_memory_encoding_frames([frame()], [memory(1)], row_capacity=100)
    item["reader_state"] = memory(torch.ones(1, 160, requires_grad=True))
    with pytest.raises(ValueError, match="detached"):
        schedule.collect_memory_encoding_microbatches(session, [item])
    item["reader_state"] = memory(1)
    session.raw.fail = True
    torch.manual_seed(51)
    expected = torch.rand(4)
    torch.manual_seed(51)
    with pytest.raises(RuntimeError, match="fake decoder failed"):
        schedule.collect_memory_encoding_microbatches(session, [item])
    assert torch.equal(torch.rand(4), expected)
    assert not session.raw.output_layer._forward_pre_hooks


def simple_plan(length, token):
    result = []
    for ordinal in range(length):
        result.append({"query": {"loss_target_tokens": 1, "ordinal": ordinal} if ordinal else None,
                       "evidence": frame(tokens=(token + ordinal,) * 4) if ordinal < length - 1 else None})
    return result


@pytest.mark.parametrize("mode", ["full", "residual"])
def test_causal_ready_waves_and_retained_stage34_writer_vjp_match_serial_reference(monkeypatch, mode):
    parameter = torch.tensor(.2, requires_grad=True)
    session = SimpleNamespace(device=torch.device("cpu"),
                              layout=SimpleNamespace(row_capacity=100, row_width=2))
    calls, costs = [], []
    written = {1: 0, 10: 0}

    def encode(session, batches, *, encoding):
        item = batches[0]
        ids = item["batch"]["input_ids"]
        hidden = torch.zeros((*ids.shape, 2560))
        state = item.get("reader_state")
        observations = []
        for lane, value in enumerate(item["frames"]):
            if value is None:
                continue
            token = value["input_ids"][0]
            origin = 1 if token < 10 else 10
            ordinal = token - origin
            memory_value = 0.
            if encoding == "memory":
                assert ordinal == written[origin], "future block encoded before preceding write"
                assert not state.values.requires_grad
                rows = torch.ones((ids.shape[0], 1), dtype=torch.long)
                memory_value = float(state.lookup(rows)[0][lane].mean())
            observations.append((token, memory_value))
            hidden[lane].fill_(token + memory_value)
        calls.append((encoding, ids.shape[0], observations))
        return {"microbatches": [{"hidden": hidden,
                                  "physical_rows": item["valid_mask"].long()[..., None].expand(-1, -1, 16)}]}

    def write(item, block):
        assert torch.is_grad_enabled() and not block["H_new"].requires_grad
        old = item["state"].values
        feature = block["H_new"].mean()
        if mode == "residual":
            assert not block["H_base_new"].requires_grad
            feature = feature + block["H_base_new"].mean()
        item["state"] = memory(old * .7 + parameter * feature)
        written[item["token"]] += 1

    monkeypatch.setattr(schedule, "collect_memory_encoding_microbatches", encode)
    local = [{"plan": simple_plan(length, token), "state": memory(parameter.expand(1, 2)), "token": token}
             for length, token in ((4, 1), (2, 10))]
    result = schedule.build_causal_writer_histories(
        session, local, gather=lambda count: [count] * 4, write=write,
        encoding_mode=mode, batch_limit=2, on_encoding=costs.append)
    assert result["write_blocks"] == 4 and result["ready_waves"] == 4
    assert len(result["queries"]) == 4
    assert [task["owner"]["token"] for task in result["queries"]] == [1, 1, 1, 10]
    assert all(item["state"] is None for item in local)
    assert all(cost["ce_targets"] == 0 for cost in costs)
    memory_calls = [call for call in calls if call[0] == "memory"]
    assert [size for _, size, _ in memory_calls] == [2, 1, 1]
    assert memory_calls[0][2][0][0] == 1 and memory_calls[0][2][1][0] == 10
    if mode == "residual":
        assert [call[0] for call in calls] == ["base", "base", "memory", "memory", "memory"]
        assert result["encoding_input_tokens"] == {"base": 16, "memory": 16}
    else:
        assert result["encoding_input_tokens"] == {"base": 0, "memory": 16}
    owners = []
    for expected in (3, 1):
        owners.append(NativeTrajectoryGradients(
            total_loss_tokens=4, expected_loss_tokens=expected,
            policy=SingleRankGradientPolicy("a" * 64, tensor_parallel_world_size=1),
            runtime_identity_sha256="a" * 64))
    tickets = [owners[0 if task["owner"]["token"] == 1 else 1].prepare_query(task["snapshot"])
               for task in result["queries"]]
    packed = pack_staged_query_tickets(tickets, row_capacity=100, width=2)
    (packed.reader_state.values.square().sum() / 4).backward()
    assert parameter.grad is None  # Reader leaf does not consume the writer graph.
    packed.scatter_cotangent()
    for ticket in tickets:
        ticket._owner.record_query(ticket, loss_sum=1., loss_tokens=1, reader_loss_divisor=4)
    for owner in owners:
        owner.backward_writer()
    reference = parameter.detach().clone().requires_grad_()
    objective = 0.
    for length, token in ((4, 1), (2, 10)):
        state = reference.expand(1, 2)
        for ordinal in range(length):
            if ordinal:
                objective = objective + state.square().sum() / 4
            if ordinal < length - 1:
                feature = token + ordinal + state.detach().mean()
                if mode == "residual":
                    feature = feature + token + ordinal
                state = .7 * state + reference * feature
    objective.backward()
    torch.testing.assert_close(parameter.grad, reference.grad)


def test_empty_rank_participates_in_same_ready_waves_without_writes(monkeypatch):
    session = SimpleNamespace(device=torch.device("cpu"),
                              layout=SimpleNamespace(row_capacity=100, row_width=160))
    calls = []
    # Global maximum has 3 frames, 2 writes, then no write at the final query.
    responses = iter(([0, 3, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]))

    def encode(session, batches, *, encoding):
        item = batches[0]
        assert item["frames"] == (None,) and not item["valid_mask"].any()
        assert item["reader_state"].values.numel() == 0
        calls.append(encoding)
        return {"microbatches": [{"hidden": torch.zeros(1, 640, 2560),
                                  "physical_rows": torch.zeros(1, 640, 16, dtype=torch.long)}]}

    monkeypatch.setattr(schedule, "collect_memory_encoding_microbatches", encode)
    result = schedule.build_causal_writer_histories(
        session, [], gather=lambda count: next(responses),
        write=lambda *args: pytest.fail("inactive rank wrote memory"), encoding_mode="full")
    assert calls == ["memory", "memory"]
    assert not result["queries"] and result["write_blocks"] == 0
    assert result["encoding_padded_input_tokens"]["memory"] == 1280


def test_matched_wave_diagnostics_include_prefix_reads_and_actual_residual_base(monkeypatch):
    session = SimpleNamespace(device=torch.device("cpu"),
                              layout=SimpleNamespace(row_capacity=100, row_width=160))
    calls, costs = [], []
    state = memory(torch.ones(1, 160, requires_grad=True))
    local = [{"state": state,
              "plan": [{"query": None, "evidence": frame(4, 7)},
                       {"query": {"loss_target_tokens": 1}, "evidence": None}]}]

    def encode(session, batches, *, encoding):
        mb = batches[0]
        calls.append((encoding, mb["batch"]["input_ids"].clone()))
        hidden = torch.full((1, 640, 2560), len(calls), dtype=torch.float32)
        return {"microbatches": [{"hidden": hidden,
                                  "physical_rows": mb["valid_mask"].long()[..., None].expand(-1, -1, 16)}]}

    def diagnose(old, block, baseline, raw_frame):
        assert old is state and old.values.requires_grad
        assert len(calls) == 3  # H12 base, memory wave, matched-base diagnostic.
        assert block["H_base_new"].eq(1).all()  # Actual residual base stays H12.
        assert block["H_new"].eq(2).all() and baseline["H_new"].eq(3).all()
        assert block["physical_rows_read"].shape == (7, 16)  # 4 prefix + 3 new.
        assert block["physical_rows_new"].shape == (1, 3, 16)
        assert raw_frame["prefix_tokens"] == 4
        return {"checked": True}

    def write(item, block):
        assert len(calls) == 3
        item["state"] = memory(item["state"].values * 2)

    monkeypatch.setattr(schedule, "collect_memory_encoding_microbatches", encode)
    result = schedule.build_causal_writer_histories(
        session, local, gather=lambda value: [value], write=write,
        encoding_mode="residual", diagnose_wave=diagnose, on_encoding=costs.append)
    assert result["diagnostics"] == [{"checked": True}]
    assert all(torch.equal(ids, calls[0][1]) for _, ids in calls)
    assert [cost["kind"] for cost in costs] == ["base", "memory", "diagnostic_base"]
    assert result["encoding_input_tokens"] == {"memory": 7, "base": 7, "diagnostic_base": 7}


@pytest.mark.parametrize("diagnose", [False, True])
def test_matched_residual_retains_h12_and_reuses_required_pass_for_diagnostics(monkeypatch, diagnose):
    session = SimpleNamespace(device=torch.device("cpu"),
                              layout=SimpleNamespace(row_capacity=100, row_width=160))
    calls, costs, writes, seen_diagnostics = [], [], [], []
    local = [{"state": memory(1), "plan": simple_plan(length, token)}
             for length, token in ((3, 1), (2, 10))]

    def encode(session, batches, *, encoding):
        mb = batches[0]
        assert len(costs) == len(calls) + 1
        calls.append((encoding, mb["batch"]["input_ids"].clone()))
        hidden = torch.full((*mb["batch"]["input_ids"].shape, 2560), float(len(calls)))
        return {"microbatches": [{"hidden": hidden,
                                  "physical_rows": mb["valid_mask"].long()[..., None].expand(-1, -1, 16)}]}

    def write(item, block):
        expected_memory, expected_matched = ((2, 3) if len(writes) < 2 else (4, 5))
        assert block["H_new"].eq(expected_memory).all()
        assert block["H_base_new"].eq(1).all()  # Shared original H12 cache.
        assert block["H_base_matched_new"].eq(expected_matched).all()
        assert not block["H_base_matched_new"].requires_grad
        writes.append(True)
        item["state"] = memory(item["state"].values * 2)

    def diagnostic(old, block, baseline, raw_frame):
        assert torch.equal(block["H_base_matched_new"], baseline["H_new"])
        seen_diagnostics.append(True)
        return {"checked": True}

    monkeypatch.setattr(schedule, "collect_memory_encoding_microbatches", encode)
    result = schedule.build_causal_writer_histories(
        session, local, gather=lambda value: [value], write=write,
        encoding_mode="residual", difference_baseline="matched_base", on_encoding=costs.append,
        diagnose_wave=diagnostic if diagnose else None)
    assert [kind for kind, ids in calls] == ["base", "memory", "base", "memory", "base"]
    assert torch.equal(calls[1][1], calls[2][1]) and torch.equal(calls[3][1], calls[4][1])
    assert [cost["kind"] for cost in costs] == ["base", "memory", "base_matched", "memory", "base_matched"]
    assert result["encoding_input_tokens"] == {"memory": 12, "base": 12, "base_matched": 12}
    assert result["difference_baseline"] == "matched_base" and len(writes) == 3
    assert len(seen_diagnostics) == (3 if diagnose else 0)
    assert result["diagnostic_seconds"] >= 0 if diagnose else result["diagnostic_seconds"] == 0


def test_inactive_rank_still_encodes_h12_and_matched_residual_waves(monkeypatch):
    session = SimpleNamespace(device=torch.device("cpu"),
                              layout=SimpleNamespace(row_capacity=100, row_width=160))
    responses = iter(([0, 2], [0, 3], [0, 1], [0, 1], [0, 0]))
    calls, costs = [], []

    def encode(session, batches, *, encoding):
        mb = batches[0]
        assert all(frame is None for frame in mb["frames"])
        assert not mb["valid_mask"].any()
        calls.append(encoding)
        return {"microbatches": [{"hidden": torch.zeros(len(mb["frames"]), 640, 2560),
                                  "physical_rows": torch.zeros(len(mb["frames"]), 640, 16, dtype=torch.long)}]}

    monkeypatch.setattr(schedule, "collect_memory_encoding_microbatches", encode)
    result = schedule.build_causal_writer_histories(
        session, [], gather=lambda value: next(responses),
        write=lambda *args: pytest.fail("inactive rank wrote"), encoding_mode="residual",
        difference_baseline="matched_base", on_encoding=costs.append)
    assert calls == ["base", "memory", "base", "memory", "base"]
    assert result["write_blocks"] == 0 and result["encoding_calls"] == {"memory": 2, "base": 1, "base_matched": 2}
    assert all(cost["ce_targets"] == cost["real_input_tokens"] == 0 for cost in costs)

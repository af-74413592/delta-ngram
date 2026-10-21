"""CPU state/collective contracts for the separate memory-aware DEV64 writer."""
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from delta_ngram.batching import BatchMemoryState
from delta_ngram.qwen38_train_data import _frame
from delta_ngram.state import SparseMemoryState
import delta_ngram.qwen38_memory_writer as writers
import delta_ngram.qwen38_training as training

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
import qwen38_native_memory_dev64_eval as dev64
import qwen38_native_memory_schedule as memory_schedule
import qwen38_native_staged_schedule as base_schedule


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def state(value):
    return BatchMemoryState(SparseMemoryState(
        torch.tensor([1]), torch.full((1, 160), float(value)), torch.ones(1)), 1, 100)


def frames(token):
    tokens = (token,) * 7
    return [_frame({}, tokens, 0, 4, query=False), _frame({}, tokens, 4, 7, query=False)]


@pytest.mark.parametrize("mode", ["full", "residual"])
def test_each_article_encodes_its_own_prewrite_states_with_inactive_ep_padding(monkeypatch, mode):
    session = SimpleNamespace(device=torch.device("cpu"),
                              layout=SimpleNamespace(row_capacity=100, row_width=160))
    observed, writes, charges, completed = [], [], [], []
    provider = dev64.MemoryEvidenceWriter(mode, charges.append)
    execution = {"native_evidence_calls": 0, "evidence_input_tokens": 0}
    started = 0

    def begin():
        nonlocal started
        started += 1
        return started

    def encode(session, batches, *, encoding):
        item = batches[0]
        assert encoding == "memory"
        assert not item["reader_state"].values.requires_grad
        assert item["batch"]["labels"] is None
        active = item["frames"][0] is not None
        if active:
            frame = item["frames"][0]
            previous = float(item["reader_state"].values.mean())
            token = frame["input_ids"][0]
            observed.append((token, frame["article_token_start"], previous))
            assert item["valid_mask"].sum() == frame["new_tokens"] + frame["prefix_tokens"]
            assert item["write_mask"].sum() == frame["new_tokens"]
            hidden = torch.full((1, 640, 2560), token + previous)
        else:
            assert item["reader_state"].values.numel() == 0 and not item["valid_mask"].any()
            observed.append(None)
            hidden = torch.zeros(1, 640, 2560)
        return {"microbatches": [{"hidden": hidden,
                                  "physical_rows": item["valid_mask"].long()[..., None].expand(-1, -1, 16)}]}

    def write(writer, previous, hidden, rows, mask, alpha, *, hidden_base):
        assert not torch.is_grad_enabled() and not hidden.requires_grad
        assert rows.shape == (1, hidden.shape[1], 16) and mask.all()
        feature = float(hidden.mean())
        if mode == "residual":
            assert hidden_base is not None
            assert hidden_base.eq(5).all()
            feature += float(hidden_base.mean())
        else:
            assert hidden_base is None
        writes.append((float(previous.values.mean()), hidden.shape[1]))
        return state(float(previous.values.mean()) + feature)

    monkeypatch.setattr(memory_schedule, "collect_memory_encoding_microbatches", encode)
    monkeypatch.setattr(training, "writer_evidence_features", lambda hidden, **kw: hidden.detach())
    monkeypatch.setattr(writers, "write_memory_aware_native_block", write)

    def write_article(old, token, condition, distance):
        arrived = frames(token)
        bases = ([{"H_new": torch.full((1, item["new_tokens"], 2560), 5.),
                   "physical_rows_new": torch.ones(1, item["new_tokens"], 16, dtype=torch.long)}
                  for item in arrived] if mode == "residual" else [None] * 2)

        def uneven_batches(local, limit):
            assert (local, limit) == (2, 1)
            # Another EP rank has 3 blocks; this local rank must call a third
            # all-padding encoding, without changing or writing its state.
            return [(0, 1), (1, 1), (2, 1)]

        with torch.no_grad():
            return provider.write_article(
                session, None, None, old, arrived, base_blocks=bases,
                gather=lambda counts: [counts] * 4, batches=uneven_batches,
                execution=execution, begin_batch=begin,
                finish_batch=lambda *args, **kw: completed.append((args, kw)),
                condition=condition, distance=distance)

    initial = state(0)
    own = write_article(initial, 1, "self", 0)
    donor = write_article(state(0), 10, "donor", 0)
    # Appending an interference article continues from the unpermuted self
    # state; donor never sees self's preceding encoding or memory values.
    extended = write_article(own, 3, "self", 1)
    bonus = 5 if mode == "residual" else 0
    own_first, donor_first = 1 + bonus, 10 + bonus
    assert observed[:3] == [(1, 0, 0.), (1, 4, float(own_first)), None]
    assert observed[3:6] == [(10, 0, 0.), (10, 4, float(donor_first)), None]
    assert observed[6][2] == float(own.values.mean())
    assert initial.values.eq(0).all() and not own.values.requires_grad
    assert extended.values.mean() > own.values.mean() and donor.values.mean() > own.values.mean()
    assert len(writes) == 6 and len(charges) == len(completed) == started == 9
    assert execution == {"native_evidence_calls": 9, "evidence_input_tokens": 3 * 11 * 4}
    assert all(cost["ce_targets"] == 0 for cost in charges)
    assert [row[1]["local_active_frames"] for row in completed] == [1, 1, 0] * 3


def test_residual_base_cache_uses_original_collector_and_restores_all_cpu_rng(monkeypatch):
    session = SimpleNamespace(device=torch.device("cpu"))
    mb = {"legacy": "original base frame batch"}
    output = {"H": "same collected result"}
    seen = []

    def collect(actual_session, batches):
        assert actual_session is session and batches == [mb]
        seen.append(True)
        random.random()
        np.random.random()
        torch.rand(2)
        return {"microbatches": [output]}

    monkeypatch.setattr(base_schedule, "collect_evidence_microbatches", collect)
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    expected = (random.random(), np.random.random(), torch.rand(2))
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    result = dev64.MemoryEvidenceWriter("residual").collect_base_encoding(session, mb)
    assert result is output and seen == [True]
    assert random.random() == expected[0] and np.random.random() == expected[1]
    assert torch.equal(torch.rand(2), expected[2])


def test_matched_residual_dev64_uses_separate_h12_and_same_lane_base_with_precharge(monkeypatch):
    session = SimpleNamespace(device=torch.device("cpu"),
                              layout=SimpleNamespace(row_capacity=100, row_width=160))
    writer = SimpleNamespace(difference_baseline="matched_base")
    costs, calls, writes = [], [], []
    provider = dev64.MemoryEvidenceWriter("residual", costs.append, difference_baseline="matched_base")
    execution = {"native_evidence_calls": 0, "evidence_input_tokens": 0}
    arrived = frames(2)
    bases = [{"H_new": torch.full((1, item["new_tokens"], 2560), 12.),
              "physical_rows_new": torch.ones(1, item["new_tokens"], 16, dtype=torch.long)} for item in arrived]

    def encode(session, batches, *, encoding):
        assert len(costs) == len(calls) + 1
        mb = batches[0]
        active = mb["frames"][0] is not None
        prior = float(mb["reader_state"].values.mean()) if active and encoding == "memory" else 0.
        calls.append((encoding, mb["batch"]["input_ids"].clone(), prior, active))
        hidden = torch.full((1, 640, 2560), (2. + prior if encoding == "memory" else 2.))
        return {"microbatches": [{"hidden": hidden,
                                  "physical_rows": mb["valid_mask"].long()[..., None].expand(-1, -1, 16)}]}

    def write(writer, previous, hidden, rows, mask, alpha, *, hidden_base, hidden_base_matched):
        assert hidden_base.eq(12).all() and hidden_base_matched.eq(2).all()
        assert hidden.eq(2. + previous.values.mean()).all()
        writes.append(float(previous.values.mean()))
        return state(float(previous.values.mean()) + 1.)

    monkeypatch.setattr(memory_schedule, "collect_memory_encoding_microbatches", encode)
    monkeypatch.setattr(training, "writer_evidence_features", lambda hidden, **kw: hidden.detach())
    monkeypatch.setattr(writers, "write_memory_aware_native_block", write)
    with torch.no_grad():
        result = provider.write_article(session, writer, None, state(0.), arrived, base_blocks=bases,
            gather=lambda value: [value] * 4, batches=lambda count, cap: [(0, 1), (1, 1), (2, 1)],
            execution=execution, begin_batch=lambda: 0., finish_batch=lambda *args, **kwargs: None,
            condition="self", distance=0)
    assert writes == [0., 1.] and result.values.eq(2.).all()
    assert [row[0] for row in calls] == ["memory", "base"] * 3
    assert all(torch.equal(calls[index][1], calls[index + 1][1]) for index in (0, 2, 4))
    assert not calls[-1][3] and not calls[-2][3]
    assert [cost["kind"] for cost in costs] == ["memory", "base_matched"] * 3
    assert execution == {"native_evidence_calls": 6, "evidence_input_tokens": 2 * 11 * 4}
    assert provider.metadata()["residual_difference_baseline"] == "matched_base"
    assert all(cost["ce_targets"] == 0 for cost in costs)


def test_evaluation_provider_inherits_writer_difference_policy(monkeypatch):
    seen = []

    def evaluate(session, writer, alpha, *, evidence_writer, **kwargs):
        seen.append(evidence_writer.difference_baseline)
        return "same scoring protocol"

    monkeypatch.setattr(dev64, "evaluate_dev64", evaluate)
    result = dev64.evaluate_memory_dev64(None, SimpleNamespace(difference_baseline="matched_base"), None,
                                         encoding_mode="residual")
    assert result == "same scoring protocol" and seen == ["matched_base"]

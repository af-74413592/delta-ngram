"""CPU boundary tests; distributed kernels are covered by the bounded GPU run."""
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/native_qwen38"))
import qwen38_native_a800_forward as forward
from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.state import SparseMemoryState
from delta_ngram.qwen38_seven_forward import STATE_FIELDS, state_fingerprint


def state_fixture():
    memory = BatchMemoryState(SparseMemoryState(torch.tensor([1]), torch.ones(1, 160), torch.tensor([2.])), 1, 200)
    return LazyDecayState(memory, torch.tensor([2]), torch.zeros(1, dtype=torch.float64),
                          torch.tensor([2]), torch.zeros(1, dtype=torch.float64))


def test_detached_snapshot_is_independent_for_all_seven_fields():
    source = state_fixture()
    copied = forward.detached_read_snapshot(source)
    assert state_fingerprint(source) == state_fingerprint(copied)
    for field in STATE_FIELDS:
        a, b = getattr(source, field), getattr(copied, field)
        assert a.data_ptr() != b.data_ptr()
        assert b.grad_fn is None and not b.requires_grad
    copied.values.zero_()
    assert source.values.sum() == 160


@pytest.fixture
def fake_runtime(monkeypatch):
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.executions = 0

        def forward(self, value):
            self.executions += 1
            raise AssertionError("evidence must never execute the vocabulary head")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = Head()
            self.read_state = None

        def forward(self, input_ids, **kwargs):
            random.random()
            np.random.random()
            torch.rand(1)
            b, s = input_ids.shape
            amount = 1. if self.read_state is None else float(self.read_state.values.sum())
            h = torch.full((s // 2, b, 2560), amount, dtype=torch.bfloat16)
            return self.output_layer(h)

    model = Model()
    calls = []

    class Adapter:
        mutate_snapshot = False

        def run_native(self, fn, *, input_ids, valid_mask, owner_ids, mode, state):
            calls.append((mode, state))
            model.read_state = state
            output = fn()
            if self.mutate_snapshot:
                state.counts.add_(1)
            return output, SimpleNamespace(physical_rows=torch.ones((*input_ids.shape, 16), dtype=torch.long))

    def gather(parts, tensor, group=None):
        for part in parts:
            part.copy_(tensor)

    dist = SimpleNamespace(all_gather=gather, broadcast=lambda *args, **kwargs: None)
    ps = SimpleNamespace(is_pipeline_last_stage=lambda: True,
                         get_tensor_model_parallel_group=lambda: None,
                         get_pipeline_model_parallel_last_rank=lambda: 0,
                         get_pipeline_model_parallel_group=lambda: None)

    def schedule(**kwargs):
        output, score = kwargs["forward_step_func"](None, model)
        _, metrics = score(output)
        return [metrics]

    monkeypatch.setattr(forward, "runtime", lambda session: (dist, ps, schedule))
    monkeypatch.setattr(forward, "language_model", lambda raw: model)
    session = SimpleNamespace(adapter=Adapter(), raw=model, models=[model], device=torch.device("cpu"))
    batch = {"input_ids": torch.tensor([[4, 5, 6, 7]]), "labels": None, "attention_mask": None}
    return session, batch, torch.ones((1, 4), dtype=torch.bool), calls, model


def test_memory_encoding_reads_current_copy_skips_head_and_restores_rng(fake_runtime):
    session, batch, valid, calls, model = fake_runtime
    state = state_fixture()
    fingerprint = state_fingerprint(state)
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
    result = forward.native_forward(session, batch, valid, state=state, evidence=True, encoding_kind="memory")
    assert calls[0][0] == "query" and calls[0][1] is not state
    assert state_fingerprint(calls[0][1]) == fingerprint == state_fingerprint(state)
    assert result["hidden"].shape == (1, 4, 2560) and bool(result["hidden"].eq(160).all())
    assert model.output_layer.executions == 0
    assert random.getstate() == py
    after = np.random.get_state()
    assert after[0] == np_state[0] and np.array_equal(after[1], np_state[1]) and after[2:] == np_state[2:]
    assert torch.equal(cpu, torch.random.get_rng_state())


def test_base_encoding_remains_state_free(fake_runtime):
    session, batch, valid, calls, _ = fake_runtime
    result = forward.native_forward(session, batch, valid, evidence=True)
    assert calls == [("evidence", None)] and bool(result["hidden"].eq(1).all())
    with pytest.raises(ValueError, match="base evidence"):
        forward.native_forward(session, batch, valid, evidence=True, state=state_fixture())


def test_memory_encoding_requires_state_and_rejects_snapshot_mutation(fake_runtime):
    session, batch, valid, _, _ = fake_runtime
    with pytest.raises(ValueError, match="FP32"):
        forward.native_forward(session, batch, valid, evidence=True, encoding_kind="memory")
    state = state_fixture()
    before = state_fingerprint(state)
    session.adapter.mutate_snapshot = True
    with pytest.raises(ValueError, match="read-only"):
        forward.native_forward(session, batch, valid, evidence=True, state=state, encoding_kind="memory")
    assert state_fingerprint(state) == before


def test_causal_memory_encoding_rejects_multi_example_batch(fake_runtime):
    session, batch, valid, _, _ = fake_runtime
    batch = {**batch, "input_ids": batch["input_ids"].expand(2, -1)}
    with pytest.raises(ValueError, match="causal B1"):
        forward.native_forward(session, batch, valid.expand(2, -1), evidence=True,
                               state=state_fixture(), encoding_kind="memory")

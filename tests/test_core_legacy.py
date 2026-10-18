"""Offline state/operator tests and optional tiny HF frozen-reader checks."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delta_ngram.writer import ForwardWriter
from delta_ngram.reader import FrozenMemoryReader
from delta_ngram.state import SparseMemoryState
from delta_ngram.operators import functional_d1


def test_lookup_missing_padding_and_empty_do_not_allocate():
    empty = SparseMemoryState.empty(3, "cpu")
    ids = torch.tensor([[0, 4], [7, 4]])
    values, counts = empty.lookup(ids)
    assert values.shape == (2, 2, 3) and counts.shape == (2, 2)
    assert not values.any() and not counts.any() and empty.rows.numel() == 0
    state = SparseMemoryState(torch.tensor([4, 8]),
                              torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
                              torch.tensor([2., 5.]))
    before = tuple(t.clone() for t in (state.rows, state.values, state.counts))
    values, counts = state.lookup(ids)
    torch.testing.assert_close(values[0, 1], state.values[0])
    assert not values[0, 0].any() and not values[1, 0].any()
    assert counts.tolist() == [[0., 2.], [0., 2.]]
    for actual, saved in zip((state.rows, state.values, state.counts), before):
        assert torch.equal(actual, saved)


def test_state_rejects_unsorted_duplicate_or_padding_rows():
    for ids in ([2, 1], [1, 1], [0, 1], [-1, 1]):
        with pytest.raises(ValueError, match="sorted, unique, and positive"):
            SparseMemoryState(torch.tensor(ids), torch.zeros(2, 3), torch.zeros(2))


def test_d1_uses_one_old_snapshot_mean_and_block_counts():
    state = SparseMemoryState(torch.tensor([2, 7]),
                              torch.tensor([[2., 4.], [8., 9.]]), torch.tensor([3., 2.]))
    old = tuple(t.clone() for t in (state.rows, state.values, state.counts))
    rows = torch.tensor([2, 2, 5, 0])
    target = torch.tensor([[6., 8.], [10., 12.], [3., 5.], [999., 999.]], requires_grad=True)
    beta = torch.tensor([0.5, 1.0, 0.0, 1.0], requires_grad=True)
    updated = functional_d1(state, rows, target, beta)
    assert updated.rows.tolist() == [2, 5, 7]
    torch.testing.assert_close(updated.values, torch.tensor([[7., 9.], [0., 0.], [8., 9.]]))
    assert updated.counts.tolist() == [4., 1., 2.]
    for actual, saved in zip((state.rows, state.values, state.counts), old):
        assert torch.equal(actual, saved)
    updated.values.sum().backward()
    assert torch.equal(target.grad[3], torch.zeros(2)) and beta.grad[3] == 0
    assert target.grad[0].abs().sum() > 0 and beta.grad[2].abs() > 0


def test_d1_event_permutation_and_empty_block():
    state = SparseMemoryState.empty(2)
    rows = torch.tensor([3, 1, 3, 0, 1])
    target = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    beta = torch.tensor([0.2, 0.4, 0.7, 1., 0.8])
    permutation = torch.tensor([4, 2, 0, 3, 1])
    one = functional_d1(state, rows, target, beta)
    other = functional_d1(state, rows[permutation], target[permutation], beta[permutation])
    assert torch.equal(one.rows, other.rows) and torch.equal(one.counts, other.counts)
    torch.testing.assert_close(one.values, other.values)
    unchanged = functional_d1(one, torch.zeros(2, dtype=torch.long), torch.randn(2, 2), torch.ones(2))
    assert unchanged is one


def test_two_writes_and_queries_backpropagate_to_both_targets():
    first = torch.tensor([[1., 2.]], requires_grad=True)
    second = torch.tensor([[3., 5.], [2., 4.]], requires_grad=True)
    initial = SparseMemoryState.empty(2)
    state1 = functional_d1(initial, torch.tensor([3]), first, torch.tensor([0.4]))
    earlier_query = state1.lookup(torch.tensor([3]))[0].square().sum()
    snapshot = state1.values.detach().clone()
    state2 = functional_d1(state1, torch.tensor([3, 8]), second, torch.tensor([0.3, 0.6]))
    later_query = state2.lookup(torch.tensor([3, 8, 90]))[0].square().sum()
    (earlier_query + later_query).backward()
    assert first.grad.abs().sum() > 0 and second.grad.abs().sum() > 0
    assert torch.equal(state1.values.detach(), snapshot)
    assert state2.counts.tolist() == [2., 1.] and initial.rows.numel() == 0
    assert not state2.detach().values.requires_grad


def test_writer_masks_padding_and_has_no_future_query_inputs():
    torch.manual_seed(31)
    writer = ForwardWriter(8, hidden_dim=12)
    hidden = torch.randn(1, 5, 8, requires_grad=True)
    rows = torch.tensor([[[0, 0], [1, 0], [2, 9], [3, 10], [99, 100]]])
    mask = torch.tensor([[1, 1, 1, 1, 0]])
    state = SparseMemoryState.empty(8)
    first = writer(hidden, rows, mask, state)
    altered = hidden.detach().clone()
    altered[:, -1] = 1e6
    other = writer(altered, rows, mask, state)
    assert first.rows.tolist() == [1, 2, 3, 9, 10]
    torch.testing.assert_close(first.values, other.values, rtol=0, atol=0)
    assert first.values.abs().sum() > 0 and first.values.abs().max() < 0.1
    first.values.square().sum().backward()
    assert hidden.grad is None
    for module in (writer.pool_score, writer.target_head, writer.gate_head, writer.head_embedding):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert list(inspect.signature(writer.forward).parameters) == ["hidden", "rows", "mask", "state"]
    with pytest.raises(TypeError):
        writer(hidden, rows, mask, state, future_answer="not evidence")


def test_writer_all_padding_is_a_noop():
    writer = ForwardWriter(4, hidden_dim=8)
    state = SparseMemoryState.empty(4)
    result = writer(torch.randn(2, 3, 4), torch.ones(2, 3, 2, dtype=torch.long),
                    torch.zeros(2, 3, dtype=torch.long), state)
    assert result is state


@pytest.fixture(params=["llama", "qwen3"])
def tiny_reader(request):
    transformers = pytest.importorskip("transformers")
    config_name, model_name = (("LlamaConfig", "LlamaForCausalLM") if request.param == "llama"
                              else ("Qwen3Config", "Qwen3ForCausalLM"))
    if not hasattr(transformers, config_name) or not hasattr(transformers, model_name):
        pytest.skip(f"installed transformers does not provide {model_name}")
    torch.manual_seed(7)
    config = getattr(transformers, config_name)(
        vocab_size=41, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    model = getattr(transformers, model_name)(config).float().eval()
    return FrozenMemoryReader(model, buckets=17)


def test_reader_zero_state_equals_base_and_captures_actual_head_input(tiny_reader):
    reader = tiny_reader
    ids = torch.tensor([[1, 4, 5, 6, 7]])
    mask = torch.ones_like(ids)
    expected_hidden = []
    handle = reader.model.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: expected_hidden.append(args[0].detach().clone()))
    with torch.no_grad():
        base = reader.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    handle.remove()
    hidden, rows = reader.encode_evidence(ids, mask)
    _, rows_int32 = reader._inputs(ids.to(dtype=torch.int32), mask)
    assert torch.equal(rows_int32, rows)
    torch.testing.assert_close(hidden, expected_hidden[0], rtol=0, atol=0)
    assert not hidden.requires_grad and rows.shape == (1, 5, 2)
    state = SparseMemoryState.empty(reader.width)
    with torch.no_grad():
        zero = reader(ids, mask, state).logits
    torch.testing.assert_close(zero, base, rtol=0, atol=0)
    assert state.rows.numel() == 0
    reader.train()
    assert not reader.model.training and not reader.model.is_gradient_checkpointing
    assert all(not p.requires_grad for p in reader.model.parameters())


def test_two_writes_multiple_query_graphs_keep_their_own_state_and_rows(tiny_reader):
    reader = tiny_reader
    torch.manual_seed(11)
    writer = ForwardWriter(reader.width, hidden_dim=12)
    ids1, ids2 = torch.tensor([[1, 4, 5, 6, 7]]), torch.tensor([[1, 4, 5, 8]])
    mask1, mask2 = torch.ones_like(ids1), torch.ones_like(ids2)
    hidden1, rows1 = reader.encode_evidence(ids1, mask1)
    hidden2, rows2 = reader.encode_evidence(ids2, mask2)
    state0 = SparseMemoryState.empty(reader.width)
    state1 = writer(hidden1, rows1, mask1, state0)
    state1.values.retain_grad()
    saved1 = state1.values.detach().clone()
    hooks_before = len(reader.model.model.layers[1]._forward_pre_hooks)
    out1 = reader(ids1, mask1, state1).logits
    state2 = writer(hidden2, rows2, mask2, state1)
    before_query = tuple(t.detach().clone() for t in (state2.rows, state2.values, state2.counts))
    out2 = reader(ids2, mask2, state2).logits
    # Interleave a different-length query before backward through both graphs.
    out3 = reader(ids1[:, :3], mask1[:, :3], state2).logits
    assert len(reader.model.model.layers[1]._forward_pre_hooks) == hooks_before
    losses = [-torch.log_softmax(out[:, -1].float(), dim=-1)[0, 9]
              for out in (out1, out2, out3)]
    sum(losses).backward()
    assert state1.values.grad is not None and state1.values.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in writer.parameters())
    assert all(p.grad is None for p in reader.model.parameters())
    assert torch.equal(state1.values.detach(), saved1)
    for actual, saved in zip((state2.rows, state2.values, state2.counts), before_query):
        assert torch.equal(actual.detach(), saved)
    assert state0.rows.numel() == 0
    hidden_again, rows_again = reader.encode_evidence(ids1, mask1)
    torch.testing.assert_close(hidden_again, hidden1, rtol=0, atol=0)
    assert torch.equal(rows_again, rows1)
    assert not hasattr(reader, "_rows") and not hasattr(reader, "_state")


def test_reader_rejects_checkpointing_reenabled_after_construction(tiny_reader):
    reader = tiny_reader
    reader.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ids = torch.tensor([[1, 4, 5]])
    with pytest.raises(RuntimeError, match="checkpointing disabled"):
        reader(ids, torch.ones_like(ids), SparseMemoryState.empty(reader.width))


def test_reader_external_extra_residual_keeps_gradient_and_bfloat16_cast_order(tiny_reader):
    import math
    reader = tiny_reader
    reader.model.to(dtype=torch.bfloat16)
    ids, mask = torch.tensor([[1, 4, 5, 6]]), torch.ones(1, 4, dtype=torch.long)
    _, rows = reader._inputs(ids, mask)
    unique = torch.unique(rows[rows.ne(0)], sorted=True)
    state = SparseMemoryState(unique, torch.randn(unique.numel(), reader.width) * .03,
                              torch.ones(unique.numel()))
    extra = (torch.randn(1, 4, reader.width) * .03).requires_grad_()
    actual_inputs = []

    def capture(module, args, kwargs, output):
        actual_inputs.append((args[0] if args else kwargs["hidden_states"]).detach().clone())

    handle = reader.model.model.layers[reader.layer_index].register_forward_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad():
            reader.model(input_ids=ids, attention_mask=mask, use_cache=False)
        output = reader.forward_with_residual(ids, mask, state, extra)
        values, _ = state.lookup(rows)
        expected = actual_inputs[0] + (values.sum(dim=-2) / math.sqrt(len(reader.orders))).to(torch.bfloat16)
        expected = expected + extra.to(torch.bfloat16)
        assert torch.equal(actual_inputs[1], expected)
        output.logits.float().square().sum().backward()
        assert extra.grad is not None and extra.grad.abs().sum() > 0
        assert all(parameter.grad is None for parameter in reader.model.parameters())
    finally:
        handle.remove()
    with torch.no_grad():
        ordinary = reader(ids, mask, state).logits
        explicit_none = reader.forward_with_residual(ids, mask, state, None).logits
    assert torch.equal(ordinary, explicit_none)
    with pytest.raises(ValueError, match="extra_residual must be float32"):
        reader.forward_with_residual(ids, mask, state, extra.to(torch.bfloat16))


def test_native_reader_backend_fails_explicitly():
    with pytest.raises(NotImplementedError, match="native table addressing/readout is not implemented"):
        FrozenMemoryReader(torch.nn.Identity(), backend="native")

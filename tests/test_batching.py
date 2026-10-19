"""Packed-stream addressing, reference parity, and trajectory gradients."""
from copy import deepcopy

import pytest
import torch

from delta_ngram.batching import BatchMemoryState, write_batch
from delta_ngram.operators import WriteOperator
from delta_ngram.reader import ExternalHashAdapter
from delta_ngram.state import SparseMemoryState
from delta_ngram.writer import ForwardWriter


OPERATORS = ("C", "D0", "D1", "D2", "D3", "D4")


def make_writer(name="D1", width=4):
    torch.manual_seed(17)
    return ForwardWriter(width, hidden_dim=8, operator=WriteOperator(
        name, kappa=.7, precision_init=2.5,
        frozen_rows=(3,) if name == "D4" else ()))


def assert_state(actual, expected, *, exact=False):
    assert torch.equal(actual.rows, expected.rows)
    assert torch.equal(actual.counts, expected.counts)
    tolerances = {"rtol": 0, "atol": 0} if exact else {"rtol": 3e-5, "atol": 2e-7}
    torch.testing.assert_close(actual.values, expected.values, **tolerances)
    assert (actual.precision is None) == (expected.precision is None)
    if actual.precision is not None:
        torch.testing.assert_close(actual.precision, expected.precision, **tolerances)


def score(state):
    loss = state.values.square().sum() + .01 * state.counts.sum()
    if state.precision is not None:
        loss = loss + .02 * state.precision.square().sum()
    return loss


def initial_state(owner=0, *, precision=True):
    values = torch.tensor([[.3, -.8, 1.2, .4], [.2, .7, -.5, .9]]) + owner
    return SparseMemoryState(
        torch.tensor([2, 7 + owner]), values.requires_grad_(),
        torch.tensor([2., 4.]).requires_grad_(),
        torch.tensor([1.7, 2.1]).requires_grad_() if precision else None)


def copy_state(state):
    def leaf(value):
        return value.detach().clone().requires_grad_(value.requires_grad) if value is not None else None
    return SparseMemoryState(state.rows.clone(), leaf(state.values), leaf(state.counts), leaf(state.precision))


def assert_writer_grads(actual, expected, *, exact=False):
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for (name, one), (other_name, two) in zip(actual.named_parameters(), expected.named_parameters()):
        assert name == other_name
        assert (one.grad is None) == (two.grad is None), name
        if one.grad is not None:
            torch.testing.assert_close(one.grad, two.grad,
                                       rtol=0 if exact else 6e-5,
                                       atol=0 if exact else 3e-7, msg=name)


@pytest.mark.parametrize("name", OPERATORS)
def test_single_stream_matches_reference_exactly(name):
    writer = make_writer(name)
    reference = deepcopy(writer)
    old = initial_state()
    expected_old = copy_state(old)
    packed = BatchMemoryState.from_states([old], row_capacity=8)
    hidden = torch.randn(1, 4, 4)
    rows = torch.tensor([[[0, 0], [2, 0], [2, 3], [4, 3]]])
    mask = torch.ones(1, 4, dtype=torch.long)
    actual = write_batch(writer, packed, hidden, rows, mask).unpack()[0]
    expected = reference(hidden, rows, mask, expected_old)
    assert_state(actual, expected, exact=True)
    score(actual).backward()
    score(expected).backward()
    assert_writer_grads(writer, reference, exact=True)
    torch.testing.assert_close(old.values.grad, expected_old.values.grad, rtol=0, atol=0)


@pytest.mark.parametrize("name", OPERATORS)
def test_two_steps_match_independent_states_and_all_gradients(name):
    writer = make_writer(name)
    reference = deepcopy(writer)
    old = [initial_state(0), initial_state(1)]
    reference_old = [copy_state(state) for state in old]
    batch = BatchMemoryState.from_states(old, row_capacity=8)
    individual = tuple(reference_old)
    rows = (
        torch.tensor([[[0, 0], [2, 0], [2, 3], [4, 3]],
                      [[0, 0], [2, 0], [5, 3], [2, 6]]]),
        torch.tensor([[[0, 0], [4, 0], [2, 6], [6, 3]],
                      [[0, 0], [5, 0], [2, 4], [5, 3]]]),
    )
    masks = (torch.ones(2, 4, dtype=torch.long), torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]))
    actual_loss = expected_loss = 0
    first = None
    saved = None
    for step in range(2):
        hidden = torch.randn(2, 4, 4)
        batch = write_batch(writer, batch, hidden, rows[step], masks[step])
        individual = tuple(reference(hidden[b:b + 1], rows[step][b:b + 1],
                                     masks[step][b:b + 1], individual[b]) for b in range(2))
        for owner, (actual, expected) in enumerate(zip(batch.unpack(), individual)):
            assert_state(actual, expected)
            weight = (step + 1) * (owner + 1)
            actual_loss = actual_loss + weight * score(actual)
            expected_loss = expected_loss + weight * score(expected)
        if step == 0:
            first = batch
            saved = tuple(value.clone() for value in (batch.values, batch.counts, batch.precision))
    for actual, snapshot in zip((first.values, first.counts, first.precision), saved):
        assert torch.equal(actual, snapshot)
    actual_loss.backward()
    expected_loss.backward()
    assert_writer_grads(writer, reference)
    for actual, expected in zip(old, reference_old):
        for one, two in zip((actual.values, actual.counts, actual.precision),
                            (expected.values, expected.counts, expected.precision)):
            torch.testing.assert_close(one.grad, two.grad, rtol=6e-5, atol=3e-7)


def test_shared_physical_rows_lookup_and_gradients_are_isolated():
    first = SparseMemoryState(torch.tensor([2]), torch.tensor([[1., 2.]], requires_grad=True), torch.tensor([3.]))
    second = SparseMemoryState(torch.tensor([2]), torch.tensor([[8., 9.]], requires_grad=True), torch.tensor([7.]))
    batch = BatchMemoryState.from_states([first, second], row_capacity=8)
    assert batch.rows.tolist() == [2, 11]
    physical = torch.tensor([[[0, 2], [7, 2]], [[0, 2], [7, 2]]], dtype=torch.int32)
    values, counts = batch.lookup(physical)
    assert counts.tolist() == [[[0., 3.], [0., 3.]], [[0., 7.], [0., 7.]]]
    assert values[:, 0, 0].eq(0).all()
    assert values[0, 0, 1].tolist() == [1., 2.]
    assert values[1, 0, 1].tolist() == [8., 9.]
    assert batch.rows.tolist() == [2, 11]  # Query misses never allocate.
    adapter = ExternalHashAdapter(buckets=4, orders=(2, 3))
    actual = adapter.read_residual(batch, physical)
    expected = torch.stack([adapter.read_residual(state, physical[b])
                            for b, state in enumerate((first, second))])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    values[0].sum().backward()
    assert first.values.grad.abs().sum() > 0
    assert second.values.grad.eq(0).all()


@pytest.mark.parametrize("name", OPERATORS)
def test_padding_empty_stream_and_empty_block_preserve_states(name):
    writer = make_writer(name)
    old = (SparseMemoryState.empty(4), initial_state(1, precision=False))
    batch = BatchMemoryState.from_states(old, row_capacity=8)
    hidden = torch.randn(2, 3, 4)
    rows = torch.tensor([[[0, 0], [2, 0], [2, 4]], [[0, 0], [2, 0], [5, 3]]])
    mask = torch.tensor([[0, 0, 0], [1, 1, 1]])
    updated = write_batch(writer, batch, hidden, rows, mask)
    for owner, actual in enumerate(updated.unpack()):
        expected = writer(hidden[owner:owner + 1], rows[owner:owner + 1], mask[owner:owner + 1], old[owner])
        assert_state(actual, expected)
    assert updated.unpack()[0].precision is None
    assert write_batch(writer, updated, hidden, torch.zeros_like(rows), mask) is updated
    assert write_batch(writer, updated, hidden[:, :0], rows[:, :0], mask[:, :0]) is updated


def test_d4_freezes_physical_rows_for_every_owner_without_allocating_them():
    writer = make_writer("D4")
    writer.operator = WriteOperator("D4", frozen_rows=(2, 6))
    states = [initial_state(0), initial_state(1)]
    batch = BatchMemoryState.from_states(states, row_capacity=8)
    hidden = torch.randn(2, 3, 4)
    rows = torch.tensor([[[2, 6], [2, 3], [6, 3]], [[2, 6], [2, 3], [6, 3]]])
    updated = write_batch(writer, batch, hidden, rows, torch.ones(2, 3))
    for owner, actual in enumerate(updated.unpack()):
        expected = writer(hidden[owner:owner + 1], rows[owner:owner + 1], torch.ones(1, 3), states[owner])
        assert_state(actual, expected)
        assert 6 not in actual.rows.tolist()
        assert torch.equal(actual.values[0], states[owner].values[0])
        assert actual.counts[0].item() == states[owner].counts[0].item()
        assert actual.precision[0].item() == states[owner].precision[0].item()
    assert writer.operator.frozen_rows == (2, 6)


def test_mixed_precision_roundtrip_and_d3_use_each_streams_actual_prior():
    writer = make_writer("D3")
    states = (initial_state(0, precision=False), initial_state(1), SparseMemoryState.empty(4))
    batch = BatchMemoryState.from_states(states, row_capacity=8)
    for actual, expected in zip(batch.unpack(), states):
        assert_state(actual, expected, exact=True)
    q = batch.lookup_precision(torch.tensor([[2, 0], [2, 0], [2, 0]]), initial=2.5)
    torch.testing.assert_close(q, torch.tensor([[2.5, 2.5], [1.7, 2.5], [2.5, 2.5]]))
    hidden = torch.randn(3, 3, 4)
    rows = torch.tensor([[[0, 0], [2, 0], [3, 2]], [[0, 0], [2, 0], [4, 2]], [[0, 0], [2, 0], [4, 2]]])
    individual = states
    for mask in (torch.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 0]]),
                 torch.tensor([[1, 1, 1], [1, 1, 1], [0, 0, 0]])):
        batch = write_batch(writer, batch, hidden, rows, mask)
        individual = tuple(writer(hidden[owner:owner + 1], rows[owner:owner + 1],
                                  mask[owner:owner + 1], individual[owner]) for owner in range(3))
        for actual, expected in zip(batch.unpack(), individual):
            assert_state(actual, expected)


def test_with_values_preserves_structure_and_autograd():
    states = [initial_state(0), initial_state(1)]
    original = BatchMemoryState.from_states(states, row_capacity=8)
    replacement = original.values * 2
    updated = original.with_values(replacement)
    assert updated.rows is original.rows and updated.counts is original.counts
    assert updated.precision is original.precision
    assert updated.values is replacement and updated.batch_size == 2 and updated.row_capacity == 8
    updated.lookup(torch.tensor([[2], [2]]))[0].sum().backward()
    for state in states:
        assert torch.equal(state.values.grad[0], torch.full((4,), 2.))
        assert state.values.grad[1].eq(0).all()
    with pytest.raises(ValueError, match="shape"):
        original.with_values(torch.zeros(4, 3))


@pytest.mark.parametrize("batch_size,capacity", [(0, 8), (True, 8), (2, 0), (2, 1.5), (2, torch.iinfo(torch.int64).max)])
def test_namespace_rejects_invalid_dimensions_and_int64_overflow(batch_size, capacity):
    with pytest.raises(ValueError):
        BatchMemoryState.empty(batch_size, 4, capacity)


def test_maximum_int64_row_is_supported_for_single_stream():
    maximum = torch.iinfo(torch.int64).max
    state = SparseMemoryState(torch.tensor([maximum]), torch.tensor([[3.]]), torch.tensor([1.]))
    batch = BatchMemoryState.from_states([state], row_capacity=maximum)
    values, counts = batch.lookup(torch.tensor([[0, maximum]]))
    assert values.tolist() == [[[0.], [3.]]] and counts.tolist() == [[0., 1.]]
    assert_state(batch.unpack()[0], state, exact=True)


def test_int32_lookup_is_promoted_before_large_capacity_comparison():
    batch = BatchMemoryState.empty(2, 1, 2 ** 32)
    values, counts = batch.lookup(torch.tensor([[1], [2]], dtype=torch.int32))
    assert values.shape == (2, 1, 1) and not values.any() and not counts.any()


def test_invalid_rows_shapes_and_owner_padding_are_rejected():
    batch = BatchMemoryState.empty(2, 4, 8)
    for rows in (torch.tensor([[1]]), torch.tensor([[9], [1]]), torch.tensor([[-1], [1]]),
                 torch.tensor([[1.], [2.]])):
        with pytest.raises(ValueError):
            batch.lookup(rows)
    with pytest.raises(ValueError, match="padding"):
        BatchMemoryState(SparseMemoryState(torch.tensor([9]), torch.zeros(1, 4), torch.zeros(1)), 2, 8)
    with pytest.raises(ValueError, match="namespace"):
        BatchMemoryState(SparseMemoryState(torch.tensor([18]), torch.zeros(1, 4), torch.zeros(1)), 2, 8)
    with pytest.raises(ValueError, match="row_capacity"):
        BatchMemoryState.from_states([SparseMemoryState(torch.tensor([9]), torch.zeros(1, 4), torch.zeros(1))], 8)
    with pytest.raises(ValueError, match="width"):
        BatchMemoryState.from_states([SparseMemoryState.empty(4), SparseMemoryState.empty(3)], 8)
    writer = make_writer()
    with pytest.raises(ValueError, match="hidden"):
        write_batch(writer, batch, torch.zeros(1, 3, 4), torch.zeros(1, 3, 2, dtype=torch.long), torch.ones(1, 3))
    with pytest.raises(ValueError, match="physical_rows"):
        write_batch(writer, batch, torch.zeros(2, 3, 4), torch.zeros(2, 3, 1, dtype=torch.long), torch.ones(2, 3))
    writer.operator = WriteOperator("D4", frozen_rows=(9,))
    with pytest.raises(ValueError, match="frozen"):
        write_batch(writer, batch, torch.zeros(2, 3, 4), torch.zeros(2, 3, 2, dtype=torch.long), torch.ones(2, 3))

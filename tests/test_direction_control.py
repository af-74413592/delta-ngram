import hashlib
import json
import random

import pytest
import torch

from delta_ngram.batching import BatchMemoryState
from delta_ngram.direction_control import (
    DIRECTION_CONTROL_SEEDS,
    make_direction_permutation,
    permute_state_for_read,
)
from delta_ngram.reader import ExternalHashAdapter
from delta_ngram.state import SparseMemoryState


@pytest.mark.parametrize("seed", DIRECTION_CONTROL_SEEDS)
def test_shared_permutation_preserves_bits_metadata_and_original(seed):
    values = torch.tensor([[0.0, -0.0, 1.0, -2.0, 3.5, -4.25, 1e-20, 1e20],
                           [-7.0, 8.0, 9.0, -10.0, 11.0, 12.0, 13.0, -14.0]])
    state = SparseMemoryState(torch.tensor([1, 9]), values, torch.tensor([3.0, 5.0]),
                              torch.tensor([2.0, 4.0]))
    saved_bits = state.values.view(torch.int32).clone()
    global_rng = random.getstate()
    plan = make_direction_permutation(state.width, seed)
    assert random.getstate() == global_rng
    assert make_direction_permutation(state.width, seed) == plan
    view = permute_state_for_read(state, plan)
    inverse = torch.argsort(torch.tensor(plan.indices))
    recovered = view.values.index_select(1, inverse)
    assert torch.equal(recovered.view(torch.int32), saved_bits)
    assert torch.equal(state.values.view(torch.int32), saved_bits)
    assert view.values.data_ptr() != state.values.data_ptr()
    assert view.rows is state.rows
    assert view.counts is state.counts
    assert view.precision is state.precision
    torch.testing.assert_close(torch.linalg.vector_norm(view.values.double(), dim=1),
                               torch.linalg.vector_norm(state.values.double(), dim=1))
    metadata = plan.metadata()
    assert metadata["seed"] == seed
    assert metadata["permutation"] == list(plan.indices)
    assert metadata["permutation_sha256"] == hashlib.sha256(
        json.dumps(metadata["permutation"], separators=(",", ":")).encode("ascii")).hexdigest()
    metadata["permutation"].reverse()
    assert plan.indices == make_direction_permutation(state.width, seed).indices


@pytest.mark.parametrize("seed", DIRECTION_CONTROL_SEEDS)
def test_shared_coordinates_preserve_reader_sum_and_row_geometry(seed):
    values = torch.tensor([[1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, 8.0],
                           [2.0, 3.0, 4.0, -5.0, 6.0, 7.0, -8.0, 9.0],
                           [-4.0, 5.0, 6.0, 7.0, -8.0, 9.0, 10.0, 11.0],
                           [6.0, -7.0, 8.0, 9.0, 10.0, -11.0, 12.0, 13.0]])
    state = SparseMemoryState(torch.tensor([1, 2, 9, 10]), values, torch.ones(4))
    plan = make_direction_permutation(state.width, seed)
    view = permute_state_for_read(state, plan)
    adapter = ExternalHashAdapter(buckets=8, orders=(2, 3))
    addresses = torch.tensor([[[1, 9], [2, 10], [0, 9]]])
    original_read = adapter.read_residual(state, addresses)
    permuted_read = adapter.read_residual(view, addresses)
    torch.testing.assert_close(permuted_read, original_read[..., list(plan.indices)], rtol=0, atol=0)
    torch.testing.assert_close(torch.linalg.vector_norm(permuted_read, dim=-1),
                               torch.linalg.vector_norm(original_read, dim=-1))
    torch.testing.assert_close(view.values.double() @ view.values.double().T,
                               state.values.double() @ state.values.double().T, rtol=0, atol=0)


def test_batch_namespace_and_per_owner_precision_survive_read_view():
    first = SparseMemoryState(torch.tensor([1, 5]), torch.arange(8).float().reshape(2, 4),
                              torch.tensor([2.0, 4.0]), torch.tensor([3.0, 6.0]))
    second = SparseMemoryState(torch.tensor([2, 8]), torch.arange(8, 16).float().reshape(2, 4),
                               torch.tensor([5.0, 7.0]))
    state = BatchMemoryState.from_states((first, second), row_capacity=8)
    plan = make_direction_permutation(state.width, 17)
    view = permute_state_for_read(state, plan)
    assert view.rows is state.rows
    assert view.counts is state.counts
    assert view.precision is state.precision
    assert view.batch_size == state.batch_size
    assert view.row_capacity == state.row_capacity
    for original, permuted in zip(state.unpack(), view.unpack()):
        torch.testing.assert_close(permuted.rows, original.rows)
        torch.testing.assert_close(permuted.counts, original.counts)
        torch.testing.assert_close(permuted.values, original.values[:, list(plan.indices)])
        if original.precision is None:
            assert permuted.precision is None
        else:
            torch.testing.assert_close(permuted.precision, original.precision)


def test_empty_and_single_channel_states_are_reported_without_resampling():
    plan = make_direction_permutation(1, 29)
    state = SparseMemoryState.empty(1)
    view = permute_state_for_read(state, plan)
    assert view.values.shape == (0, 1)
    assert plan.indices == (0,)
    assert plan.metadata()["is_identity"] is True
    assert plan.metadata()["fixed_points"] == 1


@pytest.mark.parametrize("width,seed", [(0, 17), (True, 17), (4, True), (4, 1), (4, 17.0)])
def test_invalid_or_unregistered_plans_are_rejected(width, seed):
    with pytest.raises(ValueError):
        make_direction_permutation(width, seed)


def test_read_view_rejects_width_or_type_mismatch():
    with pytest.raises(ValueError, match="width"):
        permute_state_for_read(SparseMemoryState.empty(3), make_direction_permutation(4, 17))
    with pytest.raises(TypeError, match="state"):
        permute_state_for_read(torch.zeros(1, 4), make_direction_permutation(4, 17))
    with pytest.raises(TypeError, match="plan"):
        permute_state_for_read(SparseMemoryState.empty(4), (0, 1, 2, 3))

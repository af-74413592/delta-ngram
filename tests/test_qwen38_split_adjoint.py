import pytest
import torch

from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.qwen38_adapter import _OwnerView
from delta_ngram.qwen38_training import manual_split_delta_adjoint, split_delta_adjoint_report
from delta_ngram.state import SparseMemoryState


def make_state(lazy=False):
    # Owner 0 stores physical 1,3; owner 1 stores physical 1,4.
    values = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], requires_grad=True)
    memory = SparseMemoryState(torch.tensor([1, 3, 12, 15]), values, torch.ones(4))
    batch = BatchMemoryState(memory, 2, 10)
    if not lazy:
        return batch, values
    return LazyDecayState(
        batch, torch.tensor([2, 3]), torch.tensor([-.2, -.6], dtype=torch.float64),
        torch.tensor([0, 1, 1, 2]), torch.tensor([0., -.1, -.2, -.5], dtype=torch.float64)), values


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("flat", [False, True])
def test_manual_adjoint_matches_real_lookup_graph_with_missing_rows_repeated_owners_and_padding(lazy, flat):
    state, values = make_state(lazy)
    owners = torch.tensor([1, 0, 1])
    physical = torch.tensor([
        [[1, 4], [1, 9], [4, 1]],
        [[3, 1], [3, 2], [1, 3]],
        [[4, 4], [1, 8], [4, 1]],
    ])
    valid = torch.tensor([[True, True, False], [True, True, False], [True, False, False]])
    addressed = torch.where(valid[..., None], physical, 0)
    dense, _ = _OwnerView(state, owners).lookup(addressed)
    cotangent = torch.arange(dense.numel(), dtype=torch.float32).reshape_as(dense) / 10 - .7
    (dense * cotangent).sum().backward()
    reference = manual_split_delta_adjoint(
        state, physical_rows=physical,
        dense_delta_cotangent=cotangent.flatten(-2) if flat else cotangent,
        valid_mask=valid, owner_ids=owners)
    torch.testing.assert_close(reference, values.grad, atol=2e-6, rtol=1e-6)
    assert not reference.requires_grad
    receipt = split_delta_adjoint_report(reference, values.grad,
                                         replica_gradients=[values.grad.clone(), values.grad.clone()])
    assert receipt["status"] == "accepted" and receipt["tp_reduction"] == "replica"


def test_lazy_factor_is_necessary_and_materialized_batch_uses_factor_one():
    lazy, values = make_state(True)
    physical = torch.tensor([[[1, 3]], [[1, 4]]])
    owners, valid = torch.tensor([0, 1]), torch.ones((2, 1), dtype=torch.bool)
    gradient = torch.ones((2, 1, 4))
    decayed = manual_split_delta_adjoint(lazy, physical_rows=physical, dense_delta_cotangent=gradient,
                                        valid_mask=valid, owner_ids=owners)
    ordinary = manual_split_delta_adjoint(lazy.memory, physical_rows=physical, dense_delta_cotangent=gradient,
                                         valid_mask=valid, owner_ids=owners)
    expected = torch.tensor([-.2, -.1, -.4, -.1]).exp()[:, None].expand_as(decayed)
    torch.testing.assert_close(decayed, expected)
    torch.testing.assert_close(ordinary, torch.ones_like(ordinary))
    assert split_delta_adjoint_report(decayed, ordinary, replica_gradients=[ordinary, ordinary])["status"] == "rejected"


def test_replica_equality_is_exact_and_missing_replica_cannot_accept():
    reference = torch.tensor([[1., 2.], [-3., 4.]])
    assert split_delta_adjoint_report(reference, reference)["status"] == "local_only"
    changed = reference.clone()
    changed[0, 0] = torch.nextafter(changed[0, 0], torch.tensor(float("inf")))
    receipt = split_delta_adjoint_report(reference, reference, replica_gradients=[reference, changed])
    assert receipt["local_adjoint_passed"] and receipt["status"] == "rejected"
    assert receipt["tp_replicas_bitwise_equal"] is False
    assert split_delta_adjoint_report(reference, reference * 2,
                                      replica_gradients=[reference * 2, reference * 2])["status"] == "rejected"


def test_empty_memory_returns_zero_and_invalid_inputs_fail_closed():
    state = BatchMemoryState.empty(1, 2, 10)
    physical, valid, owners = torch.tensor([[[1, 0]]]), torch.tensor([[True]]), torch.tensor([0])
    result = manual_split_delta_adjoint(state, physical_rows=physical,
                                       dense_delta_cotangent=torch.ones((1, 1, 4)),
                                       valid_mask=valid, owner_ids=owners)
    assert result.shape == (0, 2)
    assert split_delta_adjoint_report(result, result, replica_gradients=[result, result])["status"] == "inconclusive"
    with pytest.raises(ValueError, match="nonfinite"):
        manual_split_delta_adjoint(state, physical_rows=physical,
                                   dense_delta_cotangent=torch.full((1, 1, 4), float("nan")),
                                   valid_mask=valid, owner_ids=owners)
    with pytest.raises(ValueError, match="outside"):
        manual_split_delta_adjoint(state, physical_rows=physical,
                                   dense_delta_cotangent=torch.ones((1, 1, 4)),
                                   valid_mask=valid, owner_ids=torch.tensor([1]))

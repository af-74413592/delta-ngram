import copy

import pytest
import torch

from delta_ngram.operators import functional_d3
from delta_ngram.state import SparseMemoryState


def identity():
    return {"base_model": "pinned-model-commit", "tokenizer": "pinned-tokenizer-commit",
            "addressing": {"backend": "external_hash", "hash_version": 1, "buckets": 17, "orders": [2, 3]},
            "writer": "sha256:checkpoint", "operator": {"name": "D3", "precision_init": .7},
            "trajectory": {"user": "user-A", "version": 1}}


def test_state_round_trip_detaches_only_persistence_and_preserves_live_graph(tmp_path):
    target = torch.tensor([[2., 4.]], requires_grad=True)
    beta = torch.tensor([.4], requires_grad=True)
    state = functional_d3(SparseMemoryState.empty(2), torch.tensor([3]), target, beta, precision_init=.7)
    path = tmp_path / "state.pt"
    state.save(path, identity=identity())
    restored = SparseMemoryState.load(path, expected_identity=identity())
    for name in ("rows", "values", "counts", "precision"):
        assert torch.equal(getattr(state, name), getattr(restored, name))
        assert not getattr(restored, name).requires_grad
    assert state.values.requires_grad and state.precision.requires_grad
    state.values.sum().backward()
    assert target.grad.abs().sum() > 0 and beta.grad.abs().sum() > 0
    unseen = restored.lookup_precision(torch.tensor([0, 3, 19]), initial=.7)
    torch.testing.assert_close(unseen, torch.tensor([.7, 1.1, .7]))
    assert restored.rows.tolist() == [3]


@pytest.mark.parametrize("field", ["base_model", "tokenizer", "addressing", "writer", "operator", "trajectory"])
def test_each_identity_dimension_is_validated(tmp_path, field):
    path = tmp_path / "state.pt"
    SparseMemoryState.empty(2).save(path, identity=identity())
    changed = copy.deepcopy(identity())
    changed[field] = "different"
    with pytest.raises(ValueError, match="identity is incompatible"):
        SparseMemoryState.load(path, expected_identity=changed)


def test_missing_identity_and_unknown_schema_are_rejected(tmp_path):
    state, path = SparseMemoryState.empty(2), tmp_path / "state.pt"
    with pytest.raises(ValueError, match="identity requires"):
        state.save(path, identity={"base_model": "x"})
    state.save(path, identity=identity())
    payload = torch.load(path, weights_only=True)
    payload["schema_version"] = 999
    torch.save(payload, path)
    with pytest.raises(ValueError, match="unsupported state persistence schema"):
        SparseMemoryState.load(path, expected_identity=identity())


def test_user_state_isolation_and_invalid_tensor_restore(tmp_path):
    user_a, user_b = SparseMemoryState.empty(2), SparseMemoryState.empty(2)
    written = functional_d3(user_a, torch.tensor([3]), torch.ones(1, 2), torch.tensor([.5]))
    assert user_b.rows.numel() == 0 and user_a.rows.numel() == 0
    path = tmp_path / "state.pt"
    written.save(path, identity=identity())
    payload = torch.load(path, weights_only=True)
    payload["state"]["counts"] = torch.tensor([-1.])
    torch.save(payload, path)
    with pytest.raises(ValueError, match="invalid values or observation counts"):
        SparseMemoryState.load(path, expected_identity=identity())


def test_empty_precision_state_round_trip_and_explicit_detach(tmp_path):
    state = SparseMemoryState.empty(3, with_precision=True)
    path = tmp_path / "state.pt"
    state.save(path, identity=identity())
    restored = SparseMemoryState.load(path, expected_identity=identity())
    assert restored.precision is not None and restored.precision.numel() == 0
    assert restored.width == 3

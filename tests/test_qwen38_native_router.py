from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
from qwen38_native_router import SortedTopK, _install_sorted_topk


def test_sorted_callback_matches_training_order_and_keeps_gradient():
    scores = torch.tensor([[.1, .8, .2, .6, .4]], requires_grad=True)
    callback = SortedTopK()
    values, indices = callback.get_replay_topk(scores, 3, None, None, None)
    with torch.no_grad():
        other, other_indices = callback.get_replay_topk(scores, 3, None, None, None)
    assert torch.equal(values, other) and torch.equal(indices, other_indices)
    assert indices.tolist() == [[1, 3, 4]]
    values.sum().backward()
    assert scores.grad.tolist() == [[0., 1., 0., 1., 1.]]


def test_scope_restores_after_error_and_rejects_unsupported_routing():
    router = SimpleNamespace(training=False, router_replay=None,
                             config=SimpleNamespace(moe_router_fusion=False,
                                                    moe_router_group_topk=None))
    report = None
    with pytest.raises(RuntimeError, match="query failed"):
        with _install_sorted_topk([("router", router)], {}) as report:
            assert isinstance(router.router_replay, SortedTopK)
            raise RuntimeError("query failed")
    assert router.router_replay is None and report["restored"]
    router.config.moe_router_fusion = True
    with pytest.raises(ValueError, match="unfused"):
        with _install_sorted_topk([("router", router)], {}):
            pass

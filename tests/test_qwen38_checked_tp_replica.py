"""Canonical TP copies preserve one logical gradient without an extra factor."""
from dataclasses import replace

import pytest
import torch

from delta_ngram.qwen38_training import (
    GradientScaleProbe, TPGradientPolicy, checked_tp_replica_gradient,
)


def policy():
    return TPGradientPolicy(
        "replica", 2, "a" * 64, "fixed",
        (GradientScaleProbe(.01, 1., (1., 1.)),
         GradientScaleProbe(.001, 1., (1., 1.))), 1e-6, 1e-7)


def distributed_fixture(monkeypatch, reference, *, remote_flags=None):
    import torch.distributed as dist
    calls, group = [], object()
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda supplied: 2 if supplied is group else 99)
    monkeypatch.setattr(dist, "get_global_rank", lambda supplied, rank: 5 if supplied is group and rank == 0 else -1)

    def broadcast(tensor, *, src, group):
        assert src == 5
        calls.append(("broadcast", tuple(tensor.shape)))
        tensor.copy_(reference)

    def reduce(tensor, *, op, group):
        assert op == dist.ReduceOp.MAX
        calls.append(("all_reduce", tuple(tensor.shape)))
        if tensor.dtype == torch.int64 and remote_flags is not None:
            tensor.copy_(torch.maximum(tensor, torch.tensor(remote_flags)))

    monkeypatch.setattr(dist, "broadcast", broadcast)
    monkeypatch.setattr(dist, "all_reduce", reduce)
    return group, calls


def test_one_ulp_difference_is_checked_reported_and_canonicalized(monkeypatch):
    reference = torch.tensor([[1., -2.], [3., -4.]])
    local = reference.clone().requires_grad_(True)
    with torch.no_grad():
        local[0, 0] = torch.nextafter(local[0, 0], torch.tensor(float("inf")))
    before = local.detach().clone()
    group, calls = distributed_fixture(monkeypatch, reference)
    report = {}
    output = checked_tp_replica_gradient(local, policy(), group=group, diagnostics=report)
    assert torch.equal(output, reference) and torch.equal(local, before)
    assert output.data_ptr() != local.data_ptr() and not output.requires_grad
    assert report["status"] == "accepted" and report["raw_replicas_bitwise_equal"] is False
    assert report["scale_factor"] == 1.0 and report["canonical_global_rank"] == 5
    assert report["raw_replica_max_abs_error"] == float((local - reference).abs().max())
    assert calls == [("broadcast", (2, 2)), ("all_reduce", (2,)), ("all_reduce", (3,))]


def test_exact_copy_has_no_tensor_gather_or_sum_and_optional_diagnostics_cost(monkeypatch):
    reference = torch.tensor([[1., 2.]])
    group, calls = distributed_fixture(monkeypatch, reference)
    output = checked_tp_replica_gradient(reference, policy(), group=group)
    assert torch.equal(output, reference)
    assert calls == [("broadcast", (1, 2)), ("all_reduce", (2,))]


@pytest.mark.parametrize("corrupt", [
    torch.tensor([[2., 4.]]),
    torch.tensor([[1., 2.001]]),
    torch.tensor([[float("nan"), 2.]]),
    torch.tensor([[float("inf"), 2.]]),
])
def test_scale_errors_large_disagreements_and_nonfinite_copies_reject(monkeypatch, corrupt):
    group, _ = distributed_fixture(monkeypatch, torch.tensor([[1., 2.]]))
    report = {}
    with pytest.raises(RuntimeError, match="exceed the fixed"):
        checked_tp_replica_gradient(corrupt, policy(), group=group, diagnostics=report)
    assert report["status"] == "rejected"


def test_remote_failure_rejects_even_when_this_rank_is_canonical(monkeypatch):
    reference = torch.tensor([[1., 2.]])
    group, _ = distributed_fixture(monkeypatch, reference, remote_flags=[1, 1])
    with pytest.raises(RuntimeError, match="exceed the fixed"):
        checked_tp_replica_gradient(reference, policy(), group=group)


def test_unsupported_policy_and_dtype_do_not_dispatch(monkeypatch):
    reference = torch.tensor([[1., 2.]])
    group, calls = distributed_fixture(monkeypatch, reference)
    # This mean policy is accepted by its own existing FD evidence, but the
    # canonical-replica helper still cannot use it.
    mean = replace(policy(), reduction="mean")
    with pytest.raises(ValueError, match="TP2 replica"):
        checked_tp_replica_gradient(reference, mean, group=group)
    with pytest.raises(ValueError, match="FP32"):
        checked_tp_replica_gradient(reference.double(), policy(), group=group)
    assert calls == []

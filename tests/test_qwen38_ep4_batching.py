"""CPU checks of inactive EP ranks, full-history routing, and global SUM math."""
import copy

import pytest
import torch

from delta_ngram.batching import BatchMemoryState
from delta_ngram.qwen38_ep4_batching import (
    SingleRankGradientPolicy, collate_ep4_frames, pack_ep4_query_tickets,
    sum_ep4_writer_gradients,
)
from delta_ngram.qwen38_training import NativeTrajectoryGradients
from delta_ngram.state import SparseMemoryState


def test_all_inactive_rank_keeps_zero_reader_graph_without_targets_or_writes():
    for kind in ("query", "evidence"):
        batch = collate_ep4_frames([None] * 3, kind=kind)
        assert batch["batch"]["input_ids"].shape == (3, 640)
        assert batch["owner_ids"].tolist() == [0, 1, 2]
        assert batch["real_input_tokens"] == batch["loss_target_tokens"] == 0
        assert batch["padded_input_tokens"] == 1920
        assert not batch["valid_mask"].any() and not batch["write_mask"].any()
        assert batch["frames"] == (None,) * 3
    packed = pack_ep4_query_tickets([None] * 3, row_capacity=100, width=2)
    assert packed.reader_state.batch_size == 3
    assert packed.reader_state.values.requires_grad
    values, _ = packed.reader_state.lookup(torch.zeros(3, 640, 1, dtype=torch.long))
    assert values.requires_grad and not values.any()
    # An empty selected CE still traverses the genuine graph in backward.
    values[torch.zeros(values.shape[:-1], dtype=torch.bool)].sum().backward()
    assert torch.equal(packed.reader_state.values.grad, torch.zeros(1, 2))
    packed.scatter_cotangent()
    assert packed._scattered and packed.reader_state.values.grad is None


def test_tp1_identity_and_three_lanes_preserve_full_writer_history_gradient():
    policy = SingleRankGradientPolicy("a" * 64, tensor_parallel_world_size=1)
    receipt = policy.receipt()
    assert receipt["world_size"] == 1 and receipt["numerical_probes"] == []
    assert not receipt["uses_TP2_proof"] and not receipt["validates_reader_derivative"]
    with pytest.raises(ValueError, match="world size"):
        SingleRankGradientPolicy("a" * 64, tensor_parallel_world_size=2)
    parameters = [torch.tensor([[.2, .3]], requires_grad=True),
                  torch.tensor([[.5, .7]], requires_grad=True)]
    bridges = [NativeTrajectoryGradients(
        total_loss_tokens=7, expected_loss_tokens=2, policy=policy,
        runtime_identity_sha256="a" * 64) for _ in parameters]
    sources = []
    for wave in range(2):
        sources = [value.square() if wave == 0 else source + value.pow(3)
                   for value, source in zip(parameters, sources or parameters)]
        tickets = []
        for source, bridge in zip(sources, bridges):
            memory = SparseMemoryState(torch.tensor([1]), source, torch.ones(1))
            tickets.append(bridge.prepare_query(BatchMemoryState(memory, 1, 100)))
        packed = pack_ep4_query_tickets([tickets[0], None, tickets[1]], row_capacity=100, width=2)
        selected, _ = packed.reader_state.lookup(torch.tensor([[[1]], [[1]], [[1]]]))
        assert not selected[1].any()
        (selected.square().sum() / 7).backward()
        packed.scatter_cotangent()
        for bridge, ticket in zip(bridges, tickets):
            bridge.record_query(ticket, loss_sum=1., loss_tokens=1,
                                reader_loss_divisor=7, reducer=None)
    for bridge in bridges:
        bridge.backward_writer()
    for value in parameters:
        reference = value.detach().clone().requires_grad_()
        ((reference.square().square().sum()
          + (reference.square() + reference.pow(3)).square().sum()) / 7).backward()
        torch.testing.assert_close(value.grad, reference.grad)


def test_four_rank_sum_of_global_denominator_gradients_matches_direct_batch(monkeypatch):
    import torch.distributed as dist

    torch.manual_seed(17)
    reference = torch.nn.Linear(2, 1)
    replicas = [copy.deepcopy(reference) for _ in range(4)]
    x, y = torch.randn(12, 2), torch.randn(12, 1)
    ((reference(x) - y).square().sum() / 12).backward()
    cursor = 0
    for replica, count in zip(replicas, (1, 2, 4, 5)):
        ((replica(x[cursor:cursor + count]) - y[cursor:cursor + count]).square().sum() / 12).backward()
        cursor += count
    expected_sum = torch.stack([
        torch.cat([value.grad.reshape(-1) for value in replica.parameters()])
        for replica in replicas]).sum(0)
    explicit_group = object()
    calls = []

    def all_reduce(tensor, *, op, group):
        assert group is explicit_group
        calls.append(op)
        if op == dist.ReduceOp.MAX:
            tensor.copy_(torch.tensor([1, 1, 0]))
        else:
            assert op == dist.ReduceOp.SUM
            tensor.copy_(expected_sum)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    for replica in replicas:
        report = sum_ep4_writer_gradients(replica, group=explicit_group)
        assert report["reduction"] == "SUM" and report["additional_divisor"] == 1
        for actual, expected in zip(replica.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-7, rtol=1e-6)
        torch.nn.utils.clip_grad_norm_(replica.parameters(), 1., foreach=False)
        torch.optim.AdamW(replica.parameters(), lr=.003, weight_decay=0., foreach=False).step()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., foreach=False)
    torch.optim.AdamW(reference.parameters(), lr=.003, weight_decay=0., foreach=False).step()
    for replica in replicas:
        for actual, expected in zip(replica.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
    assert calls == [dist.ReduceOp.MAX, dist.ReduceOp.SUM] * 4

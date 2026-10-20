"""Cross-time query packing preserves distinct snapshots and full writer VJP."""
import pytest
import torch

from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.data.raw_text import TrajectoryWindow
from delta_ngram.qwen38_ep4_batching import SingleRankGradientPolicy
from delta_ngram.qwen38_staged_batching import (
    collate_staged_frames, materialize_staged_snapshot,
    pack_staged_query_tickets, release_staged_reader_leaf,
)
from delta_ngram.qwen38_train_data import trajectory_frames
from delta_ngram.qwen38_training import NativeTrajectoryGradients
from delta_ngram.state import SparseMemoryState


def bridge(expected):
    return NativeTrajectoryGradients(
        total_loss_tokens=7, expected_loss_tokens=expected,
        policy=SingleRankGradientPolicy("a" * 64, tensor_parallel_world_size=1),
        runtime_identity_sha256="a" * 64)


def memory(values):
    return BatchMemoryState(
        SparseMemoryState(torch.tensor([1]), values, torch.ones(1)), 1, 100)


def test_general_batch_keeps_frame_boundaries_and_all_empty_masks():
    plans = trajectory_frames(TrajectoryWindow("d", "g", "code", 0, tuple(range(4096)), 8))
    original = [plans[1]["query"], None, plans[4]["query"], None, plans[-1]["query"]]
    collated = collate_staged_frames(original, kind="query")
    assert collated["batch"]["input_ids"].shape == (5, 640)
    assert collated["owner_ids"].tolist() == list(range(5))
    assert collated["loss_target_tokens_by_lane"] == (512, 0, 512, 0, 512)
    for index, frame in enumerate(original):
        if frame is not None:
            length = len(frame["input_ids"])
            assert collated["batch"]["input_ids"][index, :length].tolist() == frame["input_ids"]
            assert collated["batch"]["labels"][index, :length].tolist() == frame["labels"]
            assert collated["batch"]["position_ids"][index, :length].tolist() == frame["position_ids"]
    for count in (1, 3, 8):
        empty = collate_staged_frames([None] * count, kind="evidence")
        assert not empty["valid_mask"].any() and not empty["write_mask"].any()
        assert empty["loss_target_tokens"] == empty["real_input_tokens"] == 0
        assert empty["padded_input_tokens"] == count * 640
    with pytest.raises(ValueError, match="at least one"):
        collate_staged_frames([], kind="query")


def test_same_trajectory_different_times_isolate_rows_and_joint_vjp_matches_direct():
    parameter = torch.tensor([[.3, -.2]], requires_grad=True)
    other = torch.tensor([[.7, .4]], requires_grad=True)
    long, short = bridge(3), bridge(1)
    states = [memory(parameter.square())]
    states.append(memory(states[-1].values + parameter.pow(3)))
    states.append(memory(states[-1].values * .99 + parameter.sin()))
    # Unequal trajectory lengths and shuffled query-time order. The same
    # physical row=1 means four different values, not one mutable row.
    tickets = [long.prepare_query(state) for state in states]
    short_ticket = short.prepare_query(memory(other.square()))
    lanes = [tickets[2], short_ticket, None, tickets[0], tickets[1]]
    packed = pack_staged_query_tickets(lanes, row_capacity=100, width=2)
    assert packed.reader_state.rows.tolist() == [1, 102, 304, 405]
    looked_up, _ = packed.reader_state.lookup(torch.ones(5, 1, 1, dtype=torch.long))
    expected = [states[2].values, other.square(), torch.zeros(1, 2),
                states[0].values, states[1].values]
    for index, value in enumerate(expected):
        torch.testing.assert_close(looked_up[index].reshape(1, 2), value)
    coefficients = torch.tensor([3., 4., 0., 1., 2.]).view(5, 1, 1, 1)
    (looked_up.square() * coefficients / 7).sum().backward()
    assert parameter.grad is None and other.grad is None
    packed.scatter_cotangent()
    for ticket in lanes:
        if ticket is None:
            continue
        owner = short if ticket is short_ticket else long
        owner.record_query(ticket, loss_sum=1., loss_tokens=1, reader_loss_divisor=7)
        source, cotangent = ticket._source_values, ticket._cotangent
        assert release_staged_reader_leaf(ticket) == 8
        assert ticket.leaf_values.numel() == 0
        assert ticket._source_values is source and ticket._cotangent is cotangent
    long.backward_writer()
    short.backward_writer()
    reference = parameter.detach().clone().requires_grad_()
    reference_other = other.detach().clone().requires_grad_()
    first = reference.square()
    second = first + reference.pow(3)
    third = second * .99 + reference.sin()
    ((first.square() + 2 * second.square() + 3 * third.square()).sum() / 7
     + 4 * reference_other.square().square().sum() / 7).backward()
    torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(other.grad, reference_other.grad, rtol=1e-6, atol=1e-7)


def test_lazy_effective_snapshot_keeps_decay_jacobian_and_immutable_guard():
    parameter = torch.tensor([[.2, .5]], requires_grad=True)
    base = memory(parameter.square())
    lazy = LazyDecayState(
        base, torch.tensor([3]), torch.tensor([-.4], dtype=torch.float64),
        torch.tensor([1]), torch.tensor([-.1], dtype=torch.float64))
    snapshot = materialize_staged_snapshot(lazy)
    owner = bridge(1)
    ticket = owner.prepare_query(snapshot)
    packed = pack_staged_query_tickets([ticket], row_capacity=100, width=2)
    (packed.reader_state.values.square().sum() / 7).backward()
    packed.scatter_cotangent()
    owner.record_query(ticket, loss_sum=1., loss_tokens=1, reader_loss_divisor=7)
    release_staged_reader_leaf(ticket)
    owner.backward_writer()
    reference = parameter.detach().clone().requires_grad_()
    (reference.square().mul(torch.tensor(-.3).exp()).square().sum() / 7).backward()
    torch.testing.assert_close(parameter.grad, reference.grad)

    fresh_owner = bridge(1)
    fresh = fresh_owner.prepare_query(memory(parameter.square()))
    with pytest.raises(ValueError, match="twice"):
        pack_staged_query_tickets([fresh, fresh], row_capacity=100, width=2)
    with pytest.raises(ValueError, match="after recording"):
        release_staged_reader_leaf(fresh)
    with torch.no_grad():
        fresh._source_values.add_(1)
    with pytest.raises(RuntimeError, match="modified in place"):
        pack_staged_query_tickets([fresh], row_capacity=100, width=2)


def test_empty_lanes_and_empty_original_ticket_keep_zero_backward_without_writer_edge():
    packed = pack_staged_query_tickets([None] * 6, row_capacity=100, width=2)
    selected, _ = packed.reader_state.lookup(torch.zeros(6, 1, 1, dtype=torch.long))
    selected.sum().backward()
    assert packed.reader_state.values.grad.eq(0).all()
    packed.scatter_cotangent()
    owner = bridge(1)
    ticket = owner.prepare_query(BatchMemoryState.empty(1, 2, 100))
    packed = pack_staged_query_tickets([None, ticket], row_capacity=100, width=2)
    selected, _ = packed.reader_state.lookup(torch.ones(2, 1, 1, dtype=torch.long))
    selected.sum().backward()
    packed.scatter_cotangent()
    assert ticket.leaf_values.grad is None
    owner.record_query(ticket, loss_sum=0., loss_tokens=1, reader_loss_divisor=7)
    assert release_staged_reader_leaf(ticket) == 0
    assert owner.backward_writer()["writer_vjp_queries"] == 0

"""Owner isolation, causal padding and complete trajectory cotangent routing."""
import pytest
import torch

from delta_ngram.batching import BatchMemoryState
from delta_ngram.data.event_streams import EventArticle, EventTrajectory
from delta_ngram.data.raw_text import TrajectoryWindow
from delta_ngram.qwen38_train_batching import (
    collate_native_frames, crop_evidence_lanes, pack_query_tickets,
)
from delta_ngram.qwen38_train_data import PAD, trajectory_frames
from delta_ngram.qwen38_training import GradientScaleProbe, NativeTrajectoryGradients, TPGradientPolicy
from delta_ngram.state import SparseMemoryState


def policy():
    return TPGradientPolicy("replica", 2, "a" * 64, "fixed",
        (GradientScaleProbe(.01, 1., (1., 1.)), GradientScaleProbe(.001, 1., (1., 1.))),
        1e-6, 1e-7)


def bridge(total, expected):
    return NativeTrajectoryGradients(total_loss_tokens=total, expected_loss_tokens=expected,
                                    policy=policy(), runtime_identity_sha256="a" * 64)


def state(values, rows=(1, 3)):
    return BatchMemoryState.from_states([
        SparseMemoryState(torch.tensor(rows), values, torch.ones(len(rows)))], row_capacity=100)


def frames():
    ordinary = TrajectoryWindow("doc", "doc-group", "code", 0, tuple(range(4096)), 8)
    event = EventTrajectory("event", "event-group", "news_event", (
        EventArticle("article0", tuple(range(512)), {}),
        EventArticle("article1", tuple(range(1000, 1129)), {})), 4)
    return trajectory_frames(ordinary), trajectory_frames(event)


def test_query_padding_preserves_all_targets_and_absent_lane_is_inactive():
    ordinary, event = frames()
    original = [ordinary[1]["query"], event[1]["query"], None, ordinary[2]["query"]]
    packed = collate_native_frames(original, kind="query")
    batch = packed["batch"]
    assert batch["input_ids"].shape == batch["labels"].shape == (4, 640)
    assert packed["loss_target_tokens_by_lane"] == (512, 129, 0, 512)
    assert packed["loss_target_tokens"] == 1153
    assert packed["owner_ids"].tolist() == [0, 1, 2, 3]
    assert not packed["active_mask"][2] and not packed["valid_mask"][2].any()
    assert batch["input_ids"][2].eq(PAD).all() and batch["labels"][2].eq(-100).all()
    assert not packed["write_mask"].any()
    for lane in (0, 1, 3):
        expected = [label for label in original[lane]["labels"] if label != -100]
        assert batch["labels"][lane][batch["labels"][lane].ne(-100)].tolist() == expected
    assert not ((batch["labels"] != -100) & ~packed["encoder_valid_mask"]).any()


def test_evidence_crops_only_new_real_tokens_and_never_materializes_absent_write():
    ordinary, event = frames()
    selected = [ordinary[1]["evidence"], event[0]["evidence"], None, None]
    collated = collate_native_frames(selected, kind="evidence")
    assert collated["batch"]["labels"] is None and collated["loss_target_tokens"] == 0
    assert torch.equal(collated["valid_mask"], collated["write_mask"])
    hidden = torch.arange(4 * 640, dtype=torch.float32).reshape(4, 640, 1).expand(-1, -1, 2560).clone()
    rows = torch.arange(1, 641).reshape(1, 640, 1).expand(4, -1, 16).clone()
    cropped = crop_evidence_lanes(collated, hidden, rows)
    assert cropped[2:] == [None, None]
    assert cropped[0]["H_new"].shape == cropped[1]["H_new"].shape == (1, 512, 2560)
    assert cropped[0]["H_new"][0, 0, 0] == 128
    assert cropped[1]["H_new"][0, 0, 0] == 640
    assert cropped[0]["physical_rows_new"][0, 0, 0] == 129
    hidden.zero_()
    rows.zero_()
    assert cropped[0]["H_new"][0, 0, 0] == 128
    assert cropped[0]["physical_rows_new"][0, 0, 0] == 129


def test_shared_physical_rows_do_not_mix_owners_or_leak_into_absent_lanes():
    left = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    right = torch.tensor([[11., 12.], [13., 14.]], requires_grad=True)
    bridges = [bridge(5, 2), bridge(5, 3)]
    tickets = [bridges[0].prepare_query(state(left)), bridges[1].prepare_query(state(right))]
    packed = pack_query_tickets([tickets[0], None, tickets[1], None], row_capacity=100, width=2)
    assert packed.reader_state.rows.tolist() == [1, 3, 203, 205]
    ids = torch.tensor([[[1, 3]], [[1, 3]], [[1, 3]], [[1, 3]]])
    values, _ = packed.reader_state.lookup(ids)
    assert torch.equal(values[0, 0], left)
    assert torch.equal(values[2, 0], right)
    assert values[1].eq(0).all() and values[3].eq(0).all()
    loss = (values[0].sum() * 2 + values[2].sum() * 3) / 5
    loss.backward()
    assert left.grad is None and right.grad is None
    assert all(ticket.leaf_values.grad is None for ticket in tickets)
    packed.scatter_cotangent()
    torch.testing.assert_close(tickets[0].leaf_values.grad, torch.full_like(left, 2 / 5))
    torch.testing.assert_close(tickets[1].leaf_values.grad, torch.full_like(right, 3 / 5))
    for index, count in enumerate((2, 3)):
        bridges[index].record_query(tickets[index], loss_sum=1., loss_tokens=count, reader_loss_divisor=5)
        bridges[index].backward_writer()
    torch.testing.assert_close(left.grad, torch.full_like(left, 2 / 5))
    torch.testing.assert_close(right.grad, torch.full_like(right, 3 / 5))
    with pytest.raises(ValueError, match="already scattered"):
        packed.scatter_cotangent(torch.ones_like(packed.reader_state.values))


def test_multiple_pipeline_waves_preserve_full_writer_bptt_and_one_group_denominator():
    parameter = torch.tensor(.7, requires_grad=True)
    direct_parameter = parameter.detach().clone().requires_grad_(True)
    bridges = [bridge(10, 4), bridge(10, 6)]
    values = [parameter * torch.tensor([[1., 2.], [3., 4.]]),
              parameter * torch.tensor([[2., 1.], [4., 3.]])]
    direct_values = [direct_parameter * torch.tensor([[1., 2.], [3., 4.]]),
                     direct_parameter * torch.tensor([[2., 1.], [4., 3.]])]
    direct_loss = 0.
    ids = torch.tensor([[[1, 3]], [[1, 3]]])
    for wave in range(2):
        tickets = [item.prepare_query(state(value)) for item, value in zip(bridges, values)]
        packed = pack_query_tickets(tickets, row_capacity=100, width=2)
        observed, _ = packed.reader_state.lookup(ids)
        lane_losses = [observed[index].square().sum() * (index + 1) for index in range(2)]
        (sum(lane_losses) / 10).backward()
        packed.scatter_cotangent()
        for index in range(2):
            bridges[index].record_query(tickets[index], loss_sum=float(lane_losses[index].detach()),
                                       loss_tokens=2 + index, reader_loss_divisor=10)
            direct_loss = direct_loss + direct_values[index].square().sum() * (index + 1) / 10
        if wave == 0:
            # A second nonlinear writer update retains the first write's graph.
            values = [value * parameter + parameter.square() for value in values]
            direct_values = [value * direct_parameter + direct_parameter.square() for value in direct_values]
    assert parameter.grad is None
    for item in bridges:
        item.backward_writer()
    direct_loss.backward()
    torch.testing.assert_close(parameter.grad, direct_parameter.grad, rtol=2e-6, atol=1e-6)


def test_packing_rejects_two_times_of_same_trajectory_and_duplicate_cotangent():
    owner = bridge(2, 2)
    value = torch.ones(2, 2, requires_grad=True)
    first = owner.prepare_query(state(value))
    second = owner.prepare_query(state(value * 2))
    with pytest.raises(ValueError, match="independent trajectories"):
        pack_query_tickets([first, second], row_capacity=100, width=2)
    other = bridge(2, 1).prepare_query(state(value * 3))
    packed = pack_query_tickets([first, other], row_capacity=100, width=2)
    first.leaf_values.grad = torch.zeros_like(first.leaf_values)
    with pytest.raises(ValueError, match="already received"):
        packed.scatter_cotangent(torch.ones_like(packed.reader_state.values))


def test_padding_cannot_create_targets_writes_or_clock_actions():
    ordinary, _ = frames()
    query = dict(ordinary[1]["query"])
    query["encoder_valid_mask"] = [True] * 639 + [False]
    query["labels"] = list(query["labels"])
    query["labels"][-1] = 10
    query["loss_target_tokens"] += 1
    with pytest.raises(ValueError, match="padding"):
        collate_native_frames([query, None], kind="query")
    evidence = dict(ordinary[0]["evidence"])
    evidence["write_mask"] = [False] + [True] * 511
    with pytest.raises(ValueError, match="exactly once"):
        collate_native_frames([evidence, None], kind="evidence")
    empty = collate_native_frames([None, None], kind="evidence")
    assert empty["loss_target_tokens"] == 0 and not empty["active_mask"].any()
    assert crop_evidence_lanes(empty, torch.zeros(2, 640, 2560),
                               torch.zeros(2, 640, 16, dtype=torch.long)) == [None, None]

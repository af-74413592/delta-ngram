import io
import json
import math

import pytest
import torch

from delta_ngram.alpha_memory import (
    ALPHA_VARIANTS,
    PRUNING_RMS_THRESHOLDS,
    BlockAlpha,
    LazyDecayState,
    make_alpha_policy,
    prune_alpha_for_read,
    write_alpha_block,
    write_alpha_block_online,
)
from delta_ngram.batching import BatchMemoryState, write_batch
from delta_ngram.state import SparseMemoryState
from delta_ngram.writer import ForwardWriter


def constant_writer(width=1, target=0.0, gate=0.5):
    writer = ForwardWriter(width, hidden_dim=2, orders=(1,), right_window=0)
    with torch.no_grad():
        for parameter in writer.parameters():
            parameter.zero_()
        writer.target_head[-1].bias.fill_(target)
        writer.gate_head[-1].bias.fill_(math.log(gate / (1 - gate)) if gate else -10000)
    return writer


def one_owner(values, rows=None, counts=None, row_capacity=8):
    values = torch.as_tensor(values, dtype=torch.float32)
    rows = torch.arange(1, values.shape[0] + 1) if rows is None else torch.tensor(rows)
    counts = torch.zeros(len(rows)) if counts is None else torch.tensor(counts, dtype=torch.float32)
    return LazyDecayState.from_batch(BatchMemoryState.from_states(
        [SparseMemoryState(rows, values, counts)], row_capacity=row_capacity))


def no_row_block(state, policy, writer=None, hidden=None, mask=None):
    writer = constant_writer(state.width) if writer is None else writer
    hidden = torch.zeros(state.batch_size, 1, state.width) if hidden is None else hidden
    mask = torch.ones(hidden.shape[:2]) if mask is None else mask
    rows = torch.zeros(*hidden.shape[:2], 1, dtype=torch.long)
    return write_alpha_block(writer, state, hidden, rows, mask, policy)


def test_fixed_decay_is_lazy_and_queries_do_not_advance_clock():
    initial = one_owner([[2.0]], counts=[4])
    policy = make_alpha_policy("fixed_099", 1)
    after_one = no_row_block(initial, policy)
    after_two = no_row_block(after_one, policy)
    assert after_two.memory is initial.memory
    assert after_two.row_last_written_block is initial.row_last_written_block
    torch.testing.assert_close(after_two.values, torch.tensor([[2.0]]), rtol=0, atol=0)
    torch.testing.assert_close(after_two.block_clock, torch.tensor([2]))
    torch.testing.assert_close(after_two.cumulative_log_alpha, torch.tensor([2 * math.log(0.99)], dtype=torch.float64))
    expected = torch.tensor([[[2 * 0.99**2]]])
    for _ in range(3):
        values, counts = after_two.lookup(torch.tensor([[1]]))
        torch.testing.assert_close(values, expected)
        torch.testing.assert_close(counts, torch.tensor([[4.0]]))
        torch.testing.assert_close(after_two.block_clock, torch.tensor([2]))
    torch.testing.assert_close(initial.block_clock, torch.tensor([0]))


def test_varying_alpha_uses_period_product_not_current_alpha_power():
    state = one_owner([[8.0]])
    policy = BlockAlpha(1, "evidence", 0.99)
    with torch.no_grad():
        policy.head.weight.fill_(1)
        policy.head.bias.zero_()
    for alpha in (0.5, 0.25):
        hidden = torch.tensor([[[math.log(alpha / (1 - alpha))]]])
        state = no_row_block(state, policy, hidden=hidden)
    effective = state.lookup(torch.tensor([[1]]))[0]
    torch.testing.assert_close(effective, torch.tensor([[[1.0]]]))
    assert not torch.allclose(effective, torch.tensor([[[8 * 0.25**2]]]))
    torch.testing.assert_close(state.values, torch.tensor([[8.0]]), rtol=0, atol=0)
    torch.testing.assert_close(state.row_last_written_block, torch.tensor([0]))


def test_decay_precedes_common_snapshot_d1_and_counts_advance_once():
    initial = one_owner([[8.0], [6.0]], counts=[3, 7])
    writer, policy = constant_writer(target=3.0), BlockAlpha(1, "fixed", 0.5)
    result = write_alpha_block(writer, initial, torch.zeros(1, 3, 1),
                               torch.tensor([[[1], [1], [0]]]), torch.ones(1, 3), policy)
    # Decay 8 -> 4, then average two increments .5 * (3 - 4): 4 -> 3.5.
    torch.testing.assert_close(result.values, torch.tensor([[3.5], [6.0]]))
    torch.testing.assert_close(result.lookup(torch.tensor([[1, 2]]))[0], torch.tensor([[[3.5], [3.0]]]))
    torch.testing.assert_close(result.counts, torch.tensor([4.0, 7.0]))
    torch.testing.assert_close(result.row_last_written_block, torch.tensor([1, 0]))
    torch.testing.assert_close(result.row_log_alpha_at_write,
                               torch.tensor([math.log(0.5), 0.0], dtype=torch.float64))
    later = no_row_block(result, policy, writer)
    torch.testing.assert_close(later.lookup(torch.tensor([[1, 2]]))[0], torch.tensor([[[1.75], [1.5]]]))
    torch.testing.assert_close(later.counts, result.counts)
    torch.testing.assert_close(initial.values, torch.tensor([[8.0], [6.0]]))


def test_state_dependent_gate_reads_decayed_value_and_zero_gate_still_counts():
    writer = constant_writer()
    # Two SiLU units implement gate logit x: SiLU(x)-SiLU(-x)=x.
    with torch.no_grad():
        old_index = writer.target_head[0].in_features
        writer.gate_head[0].weight[0, old_index] = 1
        writer.gate_head[0].weight[1, old_index] = -1
        writer.gate_head[-1].weight[0] = torch.tensor([1.0, -1.0])
    state = one_owner([[4.0]], counts=[2])
    policy = BlockAlpha(1, "fixed", 0.5)
    result = write_alpha_block(writer, state, torch.zeros(1, 2, 1),
                               torch.ones(1, 2, 1, dtype=torch.long), torch.ones(1, 2), policy)
    expected = 2 * (1 - torch.sigmoid(torch.tensor(2.0)))
    torch.testing.assert_close(result.values[0, 0], expected)
    zero_writer = constant_writer(gate=0)
    zero = write_alpha_block(zero_writer, state, torch.zeros(1, 2, 1),
                             torch.ones(1, 2, 1, dtype=torch.long), torch.ones(1, 2), policy)
    torch.testing.assert_close(zero.values, torch.tensor([[2.0]]))
    torch.testing.assert_close(zero.counts, torch.tensor([3.0]))
    torch.testing.assert_close(zero.row_last_written_block, torch.tensor([1]))


def test_identity_alpha_matches_existing_batched_d1_and_namespace():
    torch.manual_seed(2718)
    first = SparseMemoryState(torch.tensor([1, 3]), torch.randn(2, 2), torch.tensor([2.0, 3.0]),
                              torch.tensor([5.0, 7.0]))
    second = SparseMemoryState(torch.tensor([1]), torch.randn(1, 2), torch.tensor([4.0]))
    baseline = BatchMemoryState.from_states((first, second), row_capacity=8)
    lazy = LazyDecayState.from_batch(baseline)
    writer = ForwardWriter(2, hidden_dim=4, orders=(1,), right_window=0)
    policy = BlockAlpha(2, "fixed", 1.0)
    for mask in (torch.ones(2, 3), torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])):
        hidden = torch.randn(2, 3, 2)
        rows = torch.tensor([[[1], [1], [2]], [[1], [2], [0]]])
        baseline = write_batch(writer, baseline, hidden, rows, mask)
        lazy = write_alpha_block(writer, lazy, hidden, rows, mask, policy)
        for actual, expected in ((lazy.rows, baseline.rows), (lazy.values, baseline.values),
                                 (lazy.counts, baseline.counts), (lazy.precision, baseline.precision)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(lazy.memory.unpack(), baseline.unpack()):
            assert (actual.precision is None) == (expected.precision is None)
        torch.testing.assert_close(lazy.lookup(rows)[0], baseline.lookup(rows)[0], rtol=0, atol=0)
    torch.testing.assert_close(lazy.block_clock, torch.tensor([2, 1]))
    torch.testing.assert_close(lazy.cumulative_log_alpha, torch.zeros(2, dtype=torch.float64))


def test_owner_clocks_ignore_padding_and_are_independent():
    memories = [SparseMemoryState(torch.tensor([1]), torch.tensor([[4.0]]), torch.tensor([2.0]))
                for _ in range(2)]
    state = LazyDecayState.from_batch(BatchMemoryState.from_states(memories, row_capacity=8))
    policy = BlockAlpha(1, "fixed", 0.5)
    state = no_row_block(state, policy, mask=torch.tensor([[1], [0]]))
    torch.testing.assert_close(state.block_clock, torch.tensor([1, 0]))
    torch.testing.assert_close(state.lookup(torch.tensor([[1], [1]]))[0], torch.tensor([[[2.0]], [[4.0]]]))
    torch.testing.assert_close(state.rows, torch.tensor([1, 10]))
    padding_result = no_row_block(state, policy, mask=torch.zeros(2, 1))
    assert padding_result is state
    state = no_row_block(state, policy, mask=torch.tensor([[0], [1]]))
    torch.testing.assert_close(state.block_clock, torch.tensor([1, 1]))
    torch.testing.assert_close(state.lookup(torch.tensor([[1], [1]]))[0], torch.tensor([[[2.0]], [[2.0]]]))


def test_global_alpha_bptt_cancels_prewrite_log_prefix_and_preserves_base_table():
    policy, writer = make_alpha_policy("learned_global_099", 1), constant_writer(target=2.0)
    state = LazyDecayState.empty(1, 1, 8)
    for _ in range(2):
        state = write_alpha_block(writer, state, torch.zeros(1, 1, 1),
                                  torch.ones(1, 1, 1, dtype=torch.long), torch.ones(1, 1), policy)
    state = no_row_block(state, policy, writer)
    frozen_base = torch.nn.Parameter(torch.tensor([[[7.0]]]), requires_grad=False)
    saved_base = frozen_base.detach().clone()
    output = frozen_base + state.lookup(torch.tensor([[1]]))[0]
    output.sum().backward()
    alpha = torch.sigmoid(policy.logit.detach()).item()
    expected_value = alpha * (0.5 * alpha + 1)
    expected_gradient = (alpha + 1) * alpha * (1 - alpha)
    torch.testing.assert_close(output.detach() - saved_base, torch.tensor([[[expected_value]]]))
    assert math.isclose(policy.logit.grad.item(), expected_gradient, rel_tol=1e-5, abs_tol=1e-7)
    assert frozen_base.grad is None
    torch.testing.assert_close(frozen_base, saved_base, rtol=0, atol=0)
    detached = state.detach()
    assert not detached.values.requires_grad
    assert not detached.cumulative_log_alpha.requires_grad
    assert not detached.row_log_alpha_at_write.requires_grad


def test_evidence_alpha_starts_constant_and_only_head_receives_feature_gradient():
    policy = make_alpha_policy("evidence_block_099", 2)
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0]])
    first = policy(hidden, mask)
    other = policy(torch.zeros_like(hidden), mask)
    torch.testing.assert_close(first, other, rtol=0, atol=0)
    assert math.isclose(first.exp().item(), 0.99, abs_tol=1e-7)
    assert policy.metadata()["learned_parameter_count"] == 3
    state = one_owner([[1.0, 1.0]])
    state = no_row_block(state, policy, hidden=hidden, mask=mask)
    state.lookup(torch.tensor([[1]]))[0].sum().backward()
    assert hidden.grad is None
    assert bool(torch.isfinite(policy.head.weight.grad).all())
    assert bool((policy.head.weight.grad != 0).all())
    assert policy.head.bias.grad.abs().item() > 0
    # Invalid padding is excluded from the alpha pool, including NaNs.
    padded = hidden.detach().clone()
    padded[:, 2] = float("nan")
    torch.testing.assert_close(policy(padded, mask), first.detach())


@pytest.mark.parametrize("variant", ALPHA_VARIANTS)
def test_online_forward_does_not_update_any_writer_or_alpha_parameters(variant):
    policy, writer = make_alpha_policy(variant, 2), constant_writer(2, target=1.0)
    parameters = tuple(writer.parameters()) + tuple(policy.parameters())
    before = [p.detach().clone() for p in parameters]
    result = write_alpha_block_online(writer, LazyDecayState.empty(1, 2, 8), torch.ones(1, 1, 2),
                                      torch.ones(1, 1, 1, dtype=torch.long), torch.ones(1, 1), policy)
    assert not result.values.requires_grad
    for parameter, saved in zip(parameters, before):
        torch.testing.assert_close(parameter, saved, rtol=0, atol=0)
        assert parameter.grad is None


@pytest.mark.parametrize("threshold,expected_rows", list(zip(PRUNING_RMS_THRESHOLDS, (5, 4, 3, 2, 1))))
def test_read_only_pruning_uses_effective_delta_and_reports_complete_costs(threshold, expected_rows):
    # Stored values are twice the effective values after one .5 decay block.
    effective = torch.tensor([[0.0], [1e-6], [5e-4], [5e-3], [5e-2], [0.5]])
    initial = one_owner(2 * effective, counts=[1, 2, 3, 4, 5, 6])
    state = no_row_block(initial, BlockAlpha(1, "fixed", 0.5))
    saved = [t.clone() for t in (state.values, state.counts, state.block_clock,
                                state.row_last_written_block, state.row_log_alpha_at_write)]
    query_rows = torch.tensor([[[1, 5], [2, 6], [3, 0]]])
    snapshot = prune_alpha_for_read(state, threshold, query_rows)
    report = snapshot.report
    assert report["retained_rows"] == expected_rows
    assert report["delta_payload_bytes"] == expected_rows * 4
    assert report["metadata_bytes"]["row_index_bytes"] == expected_rows * 8
    assert report["metadata_bytes"]["row_count_bytes"] == expected_rows * 4
    assert report["metadata_bytes"]["row_last_written_block_bytes"] == expected_rows * 8
    assert report["metadata_bytes"]["row_log_alpha_at_write_bytes"] == expected_rows * 8
    assert report["all_row_metadata_for_future_write_semantics"]["row_count_bytes"] == 6 * 4
    assert report["all_row_metadata_bytes"] > sum(report["metadata_bytes"].values())
    assert report["serialized_bytes"] == len(snapshot.archive_bytes)
    assert report["serialization_overhead_bytes"] >= 0
    assert report["resumable"] is False
    assert report["readout"]["measured"] is True
    assert report["readout"]["nll_measured"] is False
    payload = torch.load(io.BytesIO(snapshot.archive_bytes), weights_only=True)
    assert payload["resumable"] is False
    torch.testing.assert_close(payload["state"]["values"], snapshot.state.values)
    assert snapshot.state.values.requires_grad is False
    for current, old in zip((state.values, state.counts, state.block_clock,
                              state.row_last_written_block, state.row_log_alpha_at_write), saved):
        torch.testing.assert_close(current, old, rtol=0, atol=0)
    if threshold == 0:
        assert report["dropped_nonzero_rows"] == 0
        assert report["readout"]["residual_difference_l2_max"] == 0
    else:
        assert report["dropped_nonzero_rows"] > 0


def test_empty_pruning_and_fixed_one_support_do_not_add_experiment_variant():
    empty = LazyDecayState.empty(2, 3, 8)
    result = prune_alpha_for_read(empty, 0.0)
    assert result.report["retained_rows"] == 0
    assert result.report["effective_delta_rms_mean"] == 0
    assert result.report["readout"]["measured"] is False
    assert len(ALPHA_VARIANTS) == 4
    with pytest.raises(ValueError, match="variant"):
        make_alpha_policy("fixed_100", 3)
    assert BlockAlpha(3, "fixed", 1).metadata()["learned_parameter_count"] == 0


@pytest.mark.parametrize("kind,alpha", [("fixed", 0), ("fixed", 1.1), ("global", 1), ("unknown", 0.99)])
def test_invalid_alpha_policy_rejected(kind, alpha):
    with pytest.raises(ValueError):
        BlockAlpha(2, kind, alpha)


def test_invalid_pruning_grid_and_non_d1_operator_rejected():
    state = LazyDecayState.empty(1, 1, 8)
    with pytest.raises(ValueError, match="threshold"):
        prune_alpha_for_read(state, 0.02)
    writer = ForwardWriter(1, hidden_dim=2, orders=(1,), operator="D0")
    with pytest.raises(ValueError, match="D1"):
        no_row_block(state, BlockAlpha(1), writer)


@pytest.mark.parametrize("variant", ALPHA_VARIANTS)
def test_alpha_observer_records_applied_values_without_padding_or_extra_forward(variant):
    policy, writer = make_alpha_policy(variant, 1), constant_writer()
    state = LazyDecayState.empty(2, 1, 8)
    records = []
    result = write_alpha_block(writer, state, torch.tensor([[[1.0]], [[999.0]]]),
                               torch.zeros(2, 1, 1, dtype=torch.long), torch.tensor([[1], [0]]),
                               policy, alpha_observer=records.append)
    assert len(records) == 1
    record = records[0]
    assert record["owner_ids"] == [0]
    assert record["owner_block_clock"] == [1]
    assert math.isclose(record["alpha"][0], policy.initial_alpha, abs_tol=1e-7)
    assert math.isclose(record["equivalent_constant_alpha_half_life_blocks"][0],
                        math.log(0.5) / math.log(policy.initial_alpha), rel_tol=1e-5)
    if policy.kind == "fixed":
        assert record["raw_logit"] is None
    else:
        assert record["raw_logit"] == [policy.initial_raw_logit]
        assert record["delta_logit_from_initial"] == [0.0]
    json.dumps(record, allow_nan=False)
    repeated = write_alpha_block(writer, result, torch.zeros(2, 1, 1),
                                 torch.zeros(2, 1, 1, dtype=torch.long), torch.zeros(2, 1),
                                 policy, alpha_observer=records.append)
    assert repeated is result
    assert len(records) == 1


def test_observer_handles_unit_alpha_and_extreme_learned_logits_without_nonfinite_json():
    writer = constant_writer()
    for policy in (BlockAlpha(1, "fixed", 1.0), BlockAlpha(1, "global", 0.99)):
        if policy.kind == "global":
            with torch.no_grad():
                policy.logit.fill_(1000)
        records = []
        result = write_alpha_block_online(writer, one_owner([[1.0]]), torch.zeros(1, 1, 1),
                                          torch.zeros(1, 1, 1, dtype=torch.long), torch.ones(1, 1),
                                          policy, alpha_observer=records.append)
        assert records[0]["equivalent_constant_alpha_half_life_blocks"] == [None]
        assert records[0]["unit_alpha_blocks"] == 1
        json.dumps(records[0], allow_nan=False)
        torch.testing.assert_close(result.lookup(torch.tensor([[1]]))[0], torch.ones(1, 1, 1))

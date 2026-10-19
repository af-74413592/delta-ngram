"""The block monitor observes actual writes without changing training."""
import json
import random

import pytest
import torch

from delta_ngram.batching import BatchMemoryState, write_batch
from delta_ngram.block_monitor import (BETA_HISTOGRAM_EDGES, MONITOR_PROTOCOL,
                                      distribution_summary, query_hit_summary)
from delta_ngram.operators import WriteOperator
from delta_ngram.raw_lm import train_raw, trajectory_loss
from delta_ngram.state import SparseMemoryState
from delta_ngram.train import create_runtime
from delta_ngram.writer import ForwardWriter
from test_raw_lm import config_for, window


def state(rows, values, counts):
    return SparseMemoryState(torch.tensor(rows, dtype=torch.long),
                             torch.tensor(values, dtype=torch.float32),
                             torch.tensor(counts, dtype=torch.float32))


def assert_json(value):
    # Tensor, NaN, and infinities are all rejected, including nested members.
    assert json.loads(json.dumps(value, allow_nan=False)) == value


def test_query_uses_prediction_position_positive_counts_and_owner_namespace():
    memories = [state([2, 3], [[0., 0.], [5., 5.]], [1., 0.]),
                state([2, 9], [[9., 9.], [0., 0.]], [0., 2.])]
    packed = BatchMemoryState.from_states(memories, row_capacity=10)
    addresses = torch.tensor([[[9, 9], [2, 3], [4, 2], [0, 0], [2, 2]],
                              [[9, 9], [2, 9], [9, 9], [9, 9], [9, 9]]])
    labels = torch.tensor([[-100, -100, 11, 12, 13], [-100, -100, 11, -100, -100]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    first, second = query_hit_summary(packed, addresses, labels, mask)
    assert first["all"] == {"hit_count": 2, "denominator": 4, "hit_rate": .5}
    assert [(item["hit_count"], item["denominator"]) for item in first["per_head"]] == [(1, 2), (1, 2)]
    assert first["token_any_hit_count"] == first["token_denominator"] == 2
    assert first["scored_tokens"] == 3  # one scored position has no valid head
    assert first["written_rows"] == 1
    assert second["all"] == {"hit_count": 1, "denominator": 2, "hit_rate": .5}
    assert second["scored_tokens"] == second["token_any_hit_count"] == 1
    # A B=1 sparse state follows the identical read contract.
    assert query_hit_summary(memories[0], addresses[:1], labels[:1], mask[:1])[0] == first
    assert_json([first, second])


def test_first_target_hits_last_prefix_address_not_target_address():
    memory = state([7], [[0.]], [1.])
    addresses = torch.tensor([[[0], [7], [9]]])
    labels = torch.tensor([[-100, -100, 12]])
    result, = query_hit_summary(memory, addresses, labels)
    assert result["all"] == {"hit_count": 1, "denominator": 1, "hit_rate": 1.}
    # Changing the target token's own address cannot change its prediction hit.
    addresses[:, -1] = 7
    assert query_hit_summary(memory, addresses, labels)[0] == result
    empty, = query_hit_summary(memory, torch.zeros_like(addresses), labels)
    assert empty["all"]["denominator"] == empty["token_denominator"] == 0
    assert empty["all"]["hit_rate"] is empty["token_any_hit_rate"] is None
    assert_json(empty)


def test_distribution_fixed_histogram_is_mergeable_and_nonfinite_safe():
    values = torch.tensor([0., .01, .05, .5, 1., -1., 2., float("nan"), float("inf")])
    full = distribution_summary(values)
    left, right = distribution_summary(values[:4]), distribution_summary(values[4:])
    assert full["count"] == 9 and full["finite_count"] == 7 and full["nonfinite_count"] == 2
    assert full["sum"] == left["sum"] + right["sum"]
    assert full["sumsq"] == pytest.approx(left["sumsq"] + right["sumsq"])
    assert full["histogram"]["edges"] == list(BETA_HISTOGRAM_EDGES)
    assert full["histogram"]["counts"] == [a + b for a, b in zip(
        left["histogram"]["counts"], right["histogram"]["counts"])]
    assert full["histogram"]["underflow"] == full["histogram"]["overflow"] == 1
    assert sum(full["histogram"]["counts"]) == 5  # includes beta=0 and beta=1
    assert_json(full)
    assert_json(distribution_summary(torch.empty(0)))


@pytest.mark.parametrize("operator", ["D1", "D4"])
def test_write_observer_uses_actual_beta_once_and_excludes_padding(monkeypatch, operator):
    torch.manual_seed(17)
    writer = ForwardWriter(2, hidden_dim=8, operator=WriteOperator(
        operator, frozen_rows=(3,) if operator == "D4" else ()))
    packed = BatchMemoryState.from_states([
        state([2], [[1., 2.]], [1.]), state([2], [[3., 4.]], [2.])], row_capacity=10)
    rows = torch.tensor([[[0, 2], [2, 3], [7, 8]], [[2, 4], [5, 6], [7, 8]]])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    calls, records = [], []
    actual_events = writer.events

    def capture(*args):
        events = actual_events(*args)
        calls.append((events.rows.detach().clone(), events.beta.detach().clone()))
        return events

    monkeypatch.setattr(writer, "events", capture)
    before_rng = torch.get_rng_state().clone()
    updated = write_batch(writer, packed, hidden, rows, mask, observer=records.append)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert len(calls) == len(records) == 1
    actual_rows, actual_beta = calls[0]
    for owner, metrics in enumerate(records[0]):
        selected = packed._owner_ids(actual_rows).eq(owner)
        if operator == "D4":
            selected &= actual_rows.remainder(11).ne(3)
        assert metrics["beta"] == distribution_summary(actual_beta[selected])
        assert metrics["beta"]["count"] == (2 if operator == "D4" or owner else 3)
        unique = torch.unique(actual_rows[selected])
        old, _ = packed.memory.lookup(unique)
        new, _ = updated.memory.lookup(unique)
        assert metrics["touched_rows"] == unique.numel()
        assert metrics["new_rows"] == (0 if operator == "D4" and owner == 0 else 1)
        assert metrics["total_written_rows"] == (1 if operator == "D4" and owner == 0 else 2)
        assert metrics["patch_delta_frobenius"] == pytest.approx(float((new - old).detach().norm()))
        assert metrics["patch_delta_row_l2"]["count"] == unique.numel()
        assert metrics["patch_total_norm"] == pytest.approx(float(updated.unpack()[owner].values.detach().norm()))
    assert_json(records[0])


def test_no_write_events_still_produces_one_empty_summary():
    writer = ForwardWriter(2, hidden_dim=4)
    old = BatchMemoryState.empty(2, 2, 10)
    records = []
    updated = write_batch(writer, old, torch.zeros(2, 3, 2), torch.zeros(2, 3, 2, dtype=torch.long),
                          torch.tensor([[1, 0, 0], [0, 0, 0]]), observer=records.append)
    assert updated is old and len(records) == 1
    for item in records[0]:
        assert item["beta"]["count"] == item["touched_rows"] == item["new_rows"] == 0
        assert item["total_written_rows"] == item["patch_delta_frobenius"] == item["patch_total_norm"] == 0
    assert_json(records[0])


@pytest.mark.parametrize("checkpoint_reader", [False, True])
def test_monitor_has_bitwise_loss_gradient_state_rng_parity_and_no_replay(tmp_path, monkeypatch,
                                                                         checkpoint_reader):
    method, _ = create_runtime(config_for(tmp_path))
    items = [window(method, "abc xyz\n" * 4, "long"), window(method, "xyz abc\n" * 2, "short")]
    records, event_calls = [], []
    original_events = method.writer.events

    def events(*args):
        event_calls.append(1)
        return original_events(*args)

    monkeypatch.setattr(method.writer, "events", events)
    rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    expected = trajectory_loss(method, items, prefix_tokens=4, checkpoint_reader=checkpoint_reader)
    expected.mean.backward()
    expected_gradients = [p.grad.clone() if p.grad is not None else None for p in method.parameters()]
    expected_rng = torch.get_rng_state().clone()
    method.writer.zero_grad(set_to_none=True)
    torch.set_rng_state(rng)
    event_calls.clear()
    actual = trajectory_loss(method, items, prefix_tokens=4, checkpoint_reader=checkpoint_reader,
                             block_observer=records.append, block_context={"step": 2, "microbatch": 1})
    record_count, writer_calls = len(records), len(event_calls)
    actual.mean.backward()
    assert len(records) == record_count == 8  # 4 real reads + 4 real owner writes
    assert len(event_calls) == writer_calls == 3  # once per block, not per owner/recompute
    assert torch.equal(torch.get_rng_state(), expected_rng) and random.getstate() == python_rng
    assert torch.equal(actual.loss_sum, expected.loss_sum)
    assert actual.counters == expected.counters
    for name in ("rows", "values", "counts"):
        assert torch.equal(getattr(actual.state, name), getattr(expected.state, name))
    for parameter, gradient in zip(method.parameters(), expected_gradients):
        assert (parameter.grad is None) == (gradient is None)
        if gradient is not None:
            assert torch.equal(parameter.grad, gradient)
    assert all(p.grad is None for p in method.reader.model.parameters())
    assert all(record["step"] == 2 and record["microbatch"] == 1 for record in records)
    assert [(record["kind"], record["block"]) for record in records if record["owner"] == 1] == [
        ("write", 0), ("query", 1)]
    assert all(record["metrics"]["beta"]["count"] > 0 for record in records if record["kind"] == "write")
    assert_json(records)


def test_trainer_logs_default_monitor_and_toggle_preserves_checkpoint(tmp_path):
    settings = {"loss_tokens": 32, "max_steps": 1, "eval_windows": 1}
    enabled = config_for(tmp_path, "enabled", raw=settings)
    disabled = config_for(tmp_path, "disabled", raw=settings)
    train_raw(enabled)
    train_raw(disabled, monitor_blocks=False)
    records = [json.loads(line) for line in (tmp_path / "enabled/block_metrics.jsonl").read_text().splitlines()]
    assert records and all(item["protocol"] == MONITOR_PROTOCOL for item in records)
    assert all(item["phase"] == "train" and item["step"] == 1 for item in records)
    assert not (tmp_path / "disabled/block_metrics.jsonl").exists()
    on = torch.load(tmp_path / "enabled/training.pt", weights_only=True)
    off = torch.load(tmp_path / "disabled/training.pt", weights_only=True)
    assert on["identity"] == off["identity"]
    for name, value in on["writer_state_dict"].items():
        assert torch.equal(value, off["writer_state_dict"][name])
    assert torch.equal(on["torch_rng"], off["torch_rng"]) and on["python_rng"] == off["python_rng"]
    metadata = json.loads((tmp_path / "enabled/config.json").read_text())
    assert metadata["block_monitor"]["enabled"]
    assert "block_monitor.py" in metadata["block_monitor"]["source_sha256"]
    assert_json(records)

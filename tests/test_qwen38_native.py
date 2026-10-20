"""Native interface contracts, without loading a model or touching a GPU."""
import pytest
import torch

from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.operators import functional_d1
from delta_ngram.qwen38_native import (
    Qwen38PLELayout, add_native_ple_delta, calibrate_qwen38_target_scale,
    make_qwen38_writer, write_native_block,
)
from delta_ngram.state import SparseMemoryState
from delta_ngram.writer import ForwardWriter
from delta_ngram.qwen38_writer import NativeForwardWriter


def evidence(batch=2, tokens=3):
    # Tiny per-head regions exercise the same native-ID boundary, not native hashing.
    layout = Qwen38PLELayout(tuple(5 * i for i in range(16)), (5,) * 16)
    ids = torch.tensor(layout.head_offsets).expand(batch, tokens, 16).clone()
    hidden = torch.randn(batch, tokens, 2560, requires_grad=True)
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    return layout, ids, hidden, mask


def test_default_writer_rng_and_checkpoint_compatibility():
    torch.manual_seed(17)
    old = ForwardWriter(4, hidden_dim=8)
    old_rng = torch.get_rng_state()
    torch.manual_seed(17)
    explicit = NativeForwardWriter(4, hidden_dim=8, input_dim=4, heads_per_order=1)
    assert torch.equal(old_rng, torch.get_rng_state())
    assert old.state_dict().keys() == explicit.state_dict().keys()
    assert "target_scale" not in old.state_dict()
    for name, value in old.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name])
    explicit.load_state_dict(old.state_dict(), strict=True)


def test_native_zero_is_valid_head_identity_and_mask_boundaries():
    layout, ids, hidden, mask = evidence()
    mask[1, -1] = False
    ids[1, -1] = -100  # Masked native values are never interpreted as writes.
    rows = layout.physical_rows(ids, mask)
    assert rows[0, 0, 0] == 1
    assert torch.equal(rows[0, 0], torch.arange(1, 81, 5))
    assert rows[1, -1].count_nonzero() == 0
    writer, _ = make_qwen38_writer()
    events = writer.events(hidden, rows, mask, SparseMemoryState.empty(160))
    assert events.target.shape == (5 * 16, 160)
    assert writer.head_embedding.num_embeddings == 16
    assert writer.target_head[0].in_features == 7712
    assert torch.linalg.matrix_rank(writer.target_head[-1].weight).item() == 160
    invalid = ids.clone()
    invalid[0, 0, 1] = 0
    with pytest.raises(ValueError, match="head region"):
        layout.physical_rows(invalid, mask)


@pytest.mark.parametrize("alpha", [1.0, 0.99])
def test_native_bptt_read_boundary_and_lazy_decay(alpha):
    torch.manual_seed(17)
    layout, ids, hidden, mask = evidence()
    writer, policy = make_qwen38_writer(alpha=alpha)
    rows = layout.physical_rows(ids, mask)
    lazy = LazyDecayState.empty(2, 160, layout.row_capacity)
    first = write_native_block(writer, lazy, hidden, rows, mask, policy)
    empty = layout.empty_state(2)
    events = writer.events(hidden, empty._pack_rows(rows), mask, empty.memory)
    ordinary = BatchMemoryState(
        functional_d1(empty.memory, events.rows, events.target, events.beta),
        2, layout.row_capacity)
    # Both fixed alpha arms start from the same zero delta.
    torch.testing.assert_close(first.lookup(rows)[0], ordinary.lookup(rows)[0], rtol=0, atol=0)
    # Next block touches different native rows; old rows decay exactly once.
    second = write_native_block(writer, first, hidden, rows + 1, mask, policy)
    torch.testing.assert_close(second.lookup(rows)[0], first.lookup(rows)[0] * alpha)
    base = torch.randn(2, 3, 2560, requires_grad=True)
    clocks = second.block_clock.clone()
    value = add_native_ple_delta(base, ids, mask, second, layout)
    torch.testing.assert_close(
        value - base.detach(), second.lookup(rows)[0].flatten(-2), atol=3e-7, rtol=2e-3)
    assert torch.equal(clocks, second.block_clock)
    # Stand-in downstream differentiable loss: native PLE itself is not run here.
    value.square().mean().backward()
    assert writer.target_head[-1].weight.grad.abs().sum() > 0
    assert writer.gate_head[-1].weight.grad.abs().sum() > 0
    assert hidden.grad is None and base.grad is None
    assert not torch.equal(first.lookup(rows)[0][0], first.lookup(rows)[0][1])


def test_per_head_calibration_uses_measured_base_rms_and_persists():
    layout, ids, hidden, mask = evidence()
    rows = layout.physical_rows(ids, mask)
    writer, _ = make_qwen38_writer()
    rms = torch.linspace(0.1, 1.6, 16)
    report = calibrate_qwen38_target_scale(
        writer, hidden, rows, mask, rms, target_to_base_rms=0.01)
    events = writer.events(hidden, rows, mask, SparseMemoryState.empty(160))
    observed = events.target.reshape(2, 3, 16, 160).square().mean((0, 1, 3)).sqrt()
    torch.testing.assert_close(observed, rms * 0.01)
    assert report["ple_response_calibrated"] is False
    restored, _ = make_qwen38_writer()
    restored.load_state_dict(writer.state_dict(), strict=True)
    assert torch.equal(restored.target_scale, writer.target_scale)


def test_native_empty_delta_keeps_frozen_base_and_restricts_alpha():
    layout, ids, _, mask = evidence()
    base = torch.randn(2, 3, 2560).bfloat16()
    assert torch.equal(
        add_native_ple_delta(base, ids, mask, layout.empty_state(2), layout), base)
    for alpha in [0.95, 0.5, True, float("nan")]:
        with pytest.raises(ValueError, match="restricted"):
            make_qwen38_writer(alpha=alpha)


def test_native_clock_padding_and_evidence_without_write_events():
    layout, ids, hidden, mask = evidence()
    writer, policy = make_qwen38_writer(alpha=.99)
    empty = LazyDecayState.empty(2, 160, layout.row_capacity)
    rows = layout.physical_rows(ids, mask)
    assert write_native_block(writer, empty, hidden, rows, torch.zeros_like(mask), policy) is empty
    # An active new-evidence block advances its owner even with no write events.
    mask[1] = False
    advanced = write_native_block(writer, empty, hidden, torch.zeros_like(rows), mask, policy)
    assert advanced.block_clock.tolist() == [1, 0]
    assert advanced.rows.numel() == 0
    assert advanced.cumulative_log_alpha[0] < 0
    assert advanced.cumulative_log_alpha[1] == 0


@pytest.mark.parametrize("operator", ["D0", "D1"])
def test_shared_evidence_projection_matches_dense_multhead_values_and_gradients(operator):
    # The single-head execution with 16 channels is an independent dense
    # reference for the same parameterization; order labels never enter math.
    torch.manual_seed(29)
    shared = NativeForwardWriter(5, input_dim=7, hidden_dim=11, heads_per_order=8,
                           operator=operator, target_scale=torch.linspace(.5, 2, 16))
    dense = NativeForwardWriter(5, input_dim=7, hidden_dim=11, orders=tuple(range(1, 17)),
                          operator=operator, target_scale=torch.ones(16))
    dense.load_state_dict(shared.state_dict(), strict=True)
    hidden = torch.randn(2, 4, 7, requires_grad=True)
    rows = torch.randint(1, 13, (2, 4, 16))
    rows[0, 0, 0] = 0
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    initial_values = torch.randn(2, 5)
    states = []
    for writer in (shared, dense):
        values = initial_values.clone().requires_grad_()
        state = SparseMemoryState(torch.tensor([1, 4]), values, torch.tensor([2., 3.]))
        for _ in range(2):
            state = writer(hidden, rows, mask, state)
        state.values.square().sum().backward()
        states.append((state, values.grad))
    torch.testing.assert_close(states[0][0].values, states[1][0].values, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(states[0][1], states[1][1], rtol=2e-6, atol=2e-7)
    for (name, actual), (_, expected) in zip(shared.named_parameters(), dense.named_parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-7, msg=name)
    assert hidden.grad is None

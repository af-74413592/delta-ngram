"""CPU contracts for the memory-aware residual and unchanged native writer."""
import copy
import io

import pytest
import torch

from delta_ngram.alpha_memory import BlockAlpha, LazyDecayState
from delta_ngram.qwen38_memory_writer import (
    NativeMemoryResidualWriter, write_memory_aware_native_block,
)
from delta_ngram.qwen38_native import make_qwen38_writer, write_native_block
from delta_ngram.qwen38_writer import NativeForwardWriter
from delta_ngram.state import SparseMemoryState


def make_base(heads=2, scale=True, operator="D1"):
    return NativeForwardWriter(
        width=4, input_dim=6, hidden_dim=10, heads_per_order=heads,
        target_scale=torch.linspace(.5, 2., 2 * heads) if scale else None,
        operator=operator,
    )


def evidence(writer):
    hidden = torch.randn(2, 4, 6, requires_grad=True)
    memory = torch.randn(2, 4, 6, requires_grad=True)
    rows = torch.randint(1, 12, (2, 4, writer.num_heads))
    rows[0, 0, 0] = 0
    mask = torch.tensor([[1, 1, 0, 1], [1, 1, 0, 0]], dtype=torch.bool)
    return hidden, memory, rows, mask


def sparse(values):
    return SparseMemoryState(torch.tensor([1, 4, 8]), values, torch.tensor([2., 3., 5.]))


def assert_lazy_equal(actual, expected):
    for name in ("rows", "values", "counts", "block_clock", "cumulative_log_alpha",
                 "row_last_written_block", "row_log_alpha_at_write"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


@pytest.mark.parametrize("heads", [1, 2, 8])
@pytest.mark.parametrize("scale", [False, True])
def test_zero_residual_matches_base_events_state_and_state_gradient(heads, scale):
    torch.manual_seed(17)
    base = make_base(heads, scale)
    reference = copy.deepcopy(base).requires_grad_(False)
    writer = NativeMemoryResidualWriter(base)
    hidden, memory, rows, mask = evidence(writer)
    initial = torch.randn(3, 4)
    outputs = []
    for module in (reference, writer):
        leaf = initial.clone().requires_grad_()
        state = sparse(leaf)
        kwargs = {"hidden_mem": memory} if module is writer else {}
        components = module.event_components(hidden, rows, mask, state, **kwargs)
        events = module.events(hidden, rows, mask, state, **kwargs)
        # Freezing base weights must not remove its direct old-state gate path.
        gate_grad = torch.autograd.grad(events.beta.sum(), leaf, retain_graph=True)[0]
        assert gate_grad.abs().sum() > 0
        for _ in range(2):
            state = module(hidden, rows, mask, state, **kwargs)
        state.values.square().sum().backward()
        outputs.append((components, events, state, leaf.grad, gate_grad))
    actual, expected = outputs[1], outputs[0]
    for a, b in [(actual[0].gate_logits, expected[0].gate_logits),
                 (actual[1].target, expected[1].target),
                 (actual[1].beta, expected[1].beta),
                 (actual[2].values, expected[2].values),
                 (actual[2].counts, expected[2].counts),
                 (actual[3], expected[3]), (actual[4], expected[4])]:
        assert torch.equal(a, b)
    assert hidden.grad is None and memory.grad is None
    assert all(not p.requires_grad and p.grad is None for p in base.parameters())
    assert writer.residual_head[-1].weight.grad.abs().sum() > 0
    assert writer.residual_head[-1].bias.grad.abs().sum() > 0
    # Zero output weights deliberately delay hidden-layer data gradients.
    assert writer.residual_head[0].weight.grad.count_nonzero() == 0


@pytest.mark.parametrize("heads", [1, 2, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_joint_residual_matches_dense_feature_formula_and_gradients(heads, dtype):
    torch.manual_seed(29)
    writer = NativeMemoryResidualWriter(make_base(heads).to(dtype=dtype))
    with torch.no_grad():
        writer.residual_head[-1].weight.normal_(std=.1)
        writer.residual_head[-1].bias.normal_(std=.1)
    hidden, memory, rows, mask = evidence(writer)
    state = sparse(torch.randn(3, 4, requires_grad=True))
    base = writer.base_writer.event_components(hidden, rows, mask, state)
    base_features = base.features.as_tensor()
    memory_features = writer.base_writer.event_features(memory, rows, mask).as_tensor()
    difference = memory_features - base_features
    head_dim = writer.base_writer.head_embedding.embedding_dim
    assert difference[:, -head_dim:].count_nonzero() == 0
    assert not base_features.requires_grad and not memory_features.requires_grad
    dense_residual = writer.residual_head(torch.cat((base_features, difference), dim=-1))
    head = base.features.indices[:, 2]
    expected_target = base.target + dense_residual[:, :4].float() * writer.target_scale[head, None].float()
    expected_beta = torch.sigmoid(base.gate_logits + dense_residual[:, 4]).float()
    actual = writer.events(hidden, rows, mask, state, hidden_mem=memory)
    torch.testing.assert_close(actual.target, expected_target, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(actual.beta, expected_beta, rtol=2e-6, atol=2e-7)
    parameters = tuple(writer.residual_head.parameters())
    expected_grads = torch.autograd.grad(
        expected_target.square().sum() + expected_beta.square().sum(), parameters,
        retain_graph=True)
    actual_grads = torch.autograd.grad(
        actual.target.square().sum() + actual.beta.square().sum(), parameters)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        # Shared-token versus event-expanded FP32 reductions have different
        # rounding; FP64 separately checks the affine factorization tightly.
        tolerance = dict(rtol=2e-5, atol=3e-6) if dtype == torch.float32 else dict(rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(actual_grad, expected_grad, **tolerance)


def test_original_logit_is_used_even_when_sigmoid_has_saturated():
    base = make_base()
    with torch.no_grad():
        base.gate_head[-1].weight.zero_()
        base.gate_head[-1].bias.fill_(100.)
    writer = NativeMemoryResidualWriter(base)
    with torch.no_grad():
        writer.residual_head[-1].bias[-1] = -100.
        writer.residual_head[-1].bias[:4] = .25
    hidden, memory, rows, mask = evidence(writer)
    state = SparseMemoryState.empty(4)
    baseline = base.events(hidden, rows, mask, state)
    assert torch.equal(baseline.beta, torch.ones_like(baseline.beta))
    actual = writer.events(hidden, rows, mask, state, hidden_mem=memory)
    assert torch.equal(actual.beta, torch.full_like(actual.beta, .5))
    heads = base.event_features(hidden, rows, mask).indices[:, 2]
    assert torch.equal(actual.target, baseline.target + .25 * writer.target_scale[heads, None])


def test_native_parameter_budget_and_checkpoint_partition():
    base, _ = make_qwen38_writer()
    base_keys = set(base.state_dict())
    writer = NativeMemoryResidualWriter(base)
    assert writer.base_writer is base
    assert writer.residual_head[0].in_features == 15424
    assert writer.residual_head[0].out_features == 128
    assert writer.residual_head[-1].out_features == 161
    assert sum(p.numel() for p in writer.parameters() if p.requires_grad) == 1995169
    assert all(name.startswith("residual_head.") for name, p in writer.named_parameters()
               if p.requires_grad)
    assert {k.removeprefix("base_writer.") for k in writer.state_dict()
            if k.startswith("base_writer.")} == base_keys
    assert writer.target_scale is base.target_scale
    assert writer.width == 160 and writer.input_dim == 2560 and writer.num_heads == 16
    assert writer.operator is base.operator
    assert writer.residual_head[-1].weight.count_nonzero() == 0
    assert writer.residual_head[-1].bias.count_nonzero() == 0


def test_optimizer_changes_only_residual_and_strict_checkpoint_roundtrip():
    torch.manual_seed(43)
    writer = NativeMemoryResidualWriter(make_base())
    frozen_before = {name: value.clone() for name, value in writer.base_writer.state_dict().items()}
    hidden, memory, rows, mask = evidence(writer)
    optimizer = torch.optim.AdamW((p for p in writer.parameters() if p.requires_grad),
                                 lr=3e-4, weight_decay=0.)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        state = writer(hidden, rows, mask, SparseMemoryState.empty(4), hidden_mem=memory)
        state.values.square().sum().backward()
        optimizer.step()
    assert all(torch.equal(value, writer.base_writer.state_dict()[name])
               for name, value in frozen_before.items())
    assert writer.residual_head[0].weight.grad.abs().sum() > 0
    assert len(optimizer.state) == 4
    assert hidden.grad is None and memory.grad is None
    stream = io.BytesIO()
    torch.save(writer.state_dict(), stream)
    stream.seek(0)
    restored = NativeMemoryResidualWriter(make_base())
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    expected = writer.events(hidden, rows, mask, SparseMemoryState.empty(4), hidden_mem=memory)
    actual = restored.events(hidden, rows, mask, SparseMemoryState.empty(4), hidden_mem=memory)
    assert torch.equal(actual.target, expected.target) and torch.equal(actual.beta, expected.beta)
    assert all(not p.requires_grad for p in restored.base_writer.parameters())


@pytest.mark.parametrize("alpha", [1., .99])
def test_residual_lazy_state_matches_base_and_observes_combined_events(alpha):
    torch.manual_seed(17)
    base = make_base()
    reference = copy.deepcopy(base).requires_grad_(False)
    writer = NativeMemoryResidualWriter(base)
    hidden, memory, rows, mask = evidence(writer)
    policy = BlockAlpha(6, kind="fixed", initial_alpha=alpha)
    original = LazyDecayState.empty(2, 4, 32)
    residual = LazyDecayState.empty(2, 4, 32)
    observed = []
    for offset in [0, 12]:
        actual_rows = torch.where(rows > 0, rows + offset, 0)
        original = write_native_block(reference, original, hidden, actual_rows, mask, policy)
        residual = write_memory_aware_native_block(
            writer, residual, memory, actual_rows, mask, policy, hidden_base=hidden,
            event_observer=lambda events: observed.append(events.beta.detach().clone()))
        assert_lazy_equal(residual, original)
    assert len(observed) == 2
    residual.values.square().sum().backward()
    assert writer.residual_head[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in base.parameters())
    assert hidden.grad is None and memory.grad is None
    with torch.no_grad():
        writer.residual_head[-1].bias[-1] = 1.
    observed.clear()
    after = write_memory_aware_native_block(
        writer, LazyDecayState.empty(2, 4, 32), memory, rows, mask, policy,
        hidden_base=hidden, event_observer=lambda events: observed.append(events.beta.detach()))
    packed = after.memory._pack_rows(rows)
    expected = writer.events(hidden, packed, mask, SparseMemoryState.empty(4), hidden_mem=memory)
    assert torch.equal(observed[0], expected.beta)
    assert not torch.equal(observed[0], base.events(hidden, packed, mask, SparseMemoryState.empty(4)).beta)


@pytest.mark.parametrize("alpha", [1., .99])
@pytest.mark.parametrize("observe", [False, True])
def test_full_finetune_helper_preserves_native_math(alpha, observe):
    torch.manual_seed(29)
    writer = make_base()
    hidden, memory, rows, mask = evidence(writer)
    policy = BlockAlpha(6, kind="fixed", initial_alpha=alpha)
    original = LazyDecayState.empty(2, 4, 32)
    actual = LazyDecayState.empty(2, 4, 32)
    for offset in [0, 12]:
        actual_rows = torch.where(rows > 0, rows + offset, 0)
        original = write_native_block(writer, original, memory, actual_rows, mask, policy)
        actual = write_memory_aware_native_block(
            writer, actual, memory, actual_rows, mask, policy,
            event_observer=(lambda events: None) if observe else None)
        assert_lazy_equal(actual, original)
    expected_grad = torch.autograd.grad(original.values.square().sum(), tuple(writer.parameters()))
    actual_grad = torch.autograd.grad(actual.values.square().sum(), tuple(writer.parameters()))
    for a, b in zip(actual_grad, expected_grad):
        assert torch.equal(a, b)
    assert hidden.grad is None and memory.grad is None


def test_empty_and_masked_evidence_preserve_native_clock_contract():
    writer = NativeMemoryResidualWriter(make_base())
    hidden, memory, rows, mask = evidence(writer)
    policy = BlockAlpha(6, kind="fixed", initial_alpha=1.)
    state = LazyDecayState.empty(2, 4, 32)
    events = writer.events(hidden, rows, torch.zeros_like(mask), SparseMemoryState.empty(4),
                           hidden_mem=memory)
    assert events.rows.shape == (0,) and events.target.shape == (0, 4) and events.beta.shape == (0,)
    assert write_memory_aware_native_block(
        writer, state, memory, rows, torch.zeros_like(mask), policy, hidden_base=hidden) is state
    mask[1] = False
    observed = []
    advanced = write_memory_aware_native_block(
        writer, state, memory, torch.zeros_like(rows), mask, policy, hidden_base=hidden,
        event_observer=lambda events: observed.append(events))
    assert advanced.block_clock.tolist() == [1, 0]
    assert advanced.rows.numel() == 0 and not observed


def test_rejects_unsupported_writer_and_misaligned_hidden():
    with pytest.raises(TypeError, match="NativeForwardWriter"):
        NativeMemoryResidualWriter(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match="D1"):
        NativeMemoryResidualWriter(make_base(operator="D0"))
    with pytest.raises(ValueError, match="positive integer"):
        NativeMemoryResidualWriter(make_base(), hidden_dim=True)
    writer = NativeMemoryResidualWriter(make_base())
    hidden, memory, rows, mask = evidence(writer)
    with pytest.raises(ValueError, match="shape and device"):
        writer.events(hidden, rows, mask, SparseMemoryState.empty(4), hidden_mem=memory[:, :-1])
    with pytest.raises(ValueError, match="hidden_base"):
        write_memory_aware_native_block(
            writer, LazyDecayState.empty(2, 4, 32), memory, rows, mask,
            BlockAlpha(6, kind="fixed", initial_alpha=1.))


@pytest.mark.parametrize("heads", [1, 8])
def test_matched_zero_residual_keeps_h12_events_state_and_direct_gate_gradient(heads):
    torch.manual_seed(71)
    base = make_base(heads)
    reference = copy.deepcopy(base).requires_grad_(False)
    writer = NativeMemoryResidualWriter(base, difference_baseline="matched_base")
    hidden, memory, rows, mask = evidence(writer)
    matched = torch.randn_like(hidden, requires_grad=True)
    initial = torch.randn(3, 4)
    outputs = []
    for module in (reference, writer):
        leaf = initial.clone().requires_grad_()
        state = sparse(leaf)
        kwargs = {"hidden_mem": memory, "hidden_base_matched": matched} if module is writer else {}
        components = module.event_components(hidden, rows, mask, state, **kwargs)
        events = module.events(hidden, rows, mask, state, **kwargs)
        gate_grad = torch.autograd.grad(events.beta.sum(), leaf, retain_graph=True)[0]
        assert gate_grad.abs().sum() > 0
        for _ in range(2):
            state = module(hidden, rows, mask, state, **kwargs)
        state.values.square().sum().backward()
        outputs.append((components, events, state, leaf.grad, gate_grad))
    expected, actual = outputs
    for left, right in [(expected[0].gate_logits, actual[0].gate_logits),
                        (expected[1].target, actual[1].target), (expected[1].beta, actual[1].beta),
                        (expected[2].values, actual[2].values), (expected[2].counts, actual[2].counts),
                        (expected[3], actual[3]), (expected[4], actual[4])]:
        assert torch.equal(left, right)
    assert hidden.grad is memory.grad is matched.grad is None
    assert all(parameter.grad is None for parameter in base.parameters())
    assert writer.residual_head[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("single_head", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matched_residual_dense_formula_uses_h12_first_half_and_matched_difference(single_head, dtype):
    torch.manual_seed(73)
    base = (NativeForwardWriter(width=4, input_dim=6, hidden_dim=10, orders=(2,), heads_per_order=1)
            if single_head else make_base(8))
    writer = NativeMemoryResidualWriter(base.to(dtype=dtype), difference_baseline="matched_base")
    with torch.no_grad():
        writer.residual_head[-1].weight.normal_(std=.1)
        writer.residual_head[-1].bias.normal_(std=.1)
    hidden, memory, rows, mask = evidence(writer)
    matched = torch.randn_like(hidden, requires_grad=True)
    state = sparse(torch.randn(3, 4, requires_grad=True))
    components = base.event_components(hidden, rows, mask, state)
    actual_base = components.features.as_tensor()
    mem_features = base.event_features(memory, rows, mask).as_tensor()
    matched_features = base.event_features(matched, rows, mask).as_tensor()
    expected_residual = writer.residual_head(torch.cat((actual_base, mem_features - matched_features), dim=-1))
    expected_target_delta = expected_residual[:, :4].float()
    if writer.target_scale is not None:
        expected_target_delta = expected_target_delta * writer.target_scale[components.features.indices[:, 2], None].float()
    expected_target = components.target + expected_target_delta
    expected_beta = torch.sigmoid(components.gate_logits + expected_residual[:, 4]).float()
    actual = writer.events(hidden, rows, mask, state, hidden_mem=memory, hidden_base_matched=matched)
    torch.testing.assert_close(actual.target, expected_target, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(actual.beta, expected_beta, rtol=2e-6, atol=2e-7)
    # Replacing the frozen base by matched H would be a different algorithm.
    wrong = base.events(matched, rows, mask, state)
    assert not torch.equal(wrong.target, components.target)
    params = tuple(writer.residual_head.parameters())
    expected_grads = torch.autograd.grad(expected_target.square().sum() + expected_beta.square().sum(),
                                         params, retain_graph=True)
    actual_grads = torch.autograd.grad(actual.target.square().sum() + actual.beta.square().sum(),
                                       params, retain_graph=True)
    for first, second in zip(actual_grads, expected_grads):
        torch.testing.assert_close(first, second, rtol=2e-5 if dtype == torch.float32 else 1e-10,
                                   atol=3e-6 if dtype == torch.float32 else 1e-10)
    actual.target.square().sum().backward()
    assert hidden.grad is memory.grad is matched.grad is None


@pytest.mark.parametrize("alpha", [1., .99])
def test_matched_lazy_helper_forwards_distinct_matched_feature_and_keeps_h12_zero_output(alpha):
    torch.manual_seed(79)
    base = make_base()
    original = copy.deepcopy(base).requires_grad_(False)
    writer = NativeMemoryResidualWriter(base, difference_baseline="matched_base")
    hidden, memory, rows, mask = evidence(writer)
    matched = torch.randn_like(hidden)
    policy = BlockAlpha(6, kind="fixed", initial_alpha=alpha)
    expected = LazyDecayState.empty(2, 4, 32)
    actual = LazyDecayState.empty(2, 4, 32)
    for _ in range(2):
        expected = write_native_block(original, expected, hidden, rows, mask, policy)
        actual = write_memory_aware_native_block(writer, actual, memory, rows, mask, policy,
            hidden_base=hidden, hidden_base_matched=matched)
        assert_lazy_equal(actual, expected)
    with torch.no_grad():
        writer.residual_head[-1].weight.normal_(std=.1)
    collected = []
    empty = LazyDecayState.empty(2, 4, 32)
    write_memory_aware_native_block(writer, empty, memory, rows, mask, policy, hidden_base=hidden,
        hidden_base_matched=matched, event_observer=lambda event: collected.append(event))
    direct = writer.events(hidden, empty.memory._pack_rows(rows), mask, SparseMemoryState.empty(4),
                            hidden_mem=memory, hidden_base_matched=matched)
    assert torch.equal(collected[0].target, direct.target)
    assert torch.equal(collected[0].beta, direct.beta)


def test_matched_policy_is_explicit_required_and_tensor_checkpoint_shapes_unchanged():
    with pytest.raises(ValueError, match="difference_baseline"):
        NativeMemoryResidualWriter(make_base(), difference_baseline="unknown")
    writer = NativeMemoryResidualWriter(make_base(), difference_baseline="matched_base")
    legacy = NativeMemoryResidualWriter(make_base())
    legacy.load_state_dict(writer.state_dict(), strict=True)
    assert set(writer.state_dict()) == set(legacy.state_dict())
    assert writer.difference_baseline == "matched_base" and legacy.difference_baseline == "actual_base"
    hidden, memory, rows, mask = evidence(writer)
    for selected_mask in (mask, torch.zeros_like(mask)):
        with pytest.raises(ValueError, match="requires hidden_base_matched"):
            writer.events(hidden, rows, selected_mask, SparseMemoryState.empty(4), hidden_mem=memory)
        with pytest.raises(ValueError, match="requires hidden_base_matched"):
            write_memory_aware_native_block(writer, LazyDecayState.empty(2, 4, 32), memory, rows,
                selected_mask, BlockAlpha(6, kind="fixed", initial_alpha=1.), hidden_base=hidden)
    with pytest.raises(ValueError, match="shape and device"):
        writer.events(hidden, rows, mask, SparseMemoryState.empty(4), hidden_mem=memory,
                       hidden_base_matched=hidden[:, :-1])
    with pytest.raises(ValueError, match="difference_baseline=matched_base"):
        legacy.events(hidden, rows, mask, SparseMemoryState.empty(4), hidden_mem=memory,
                       hidden_base_matched=hidden)

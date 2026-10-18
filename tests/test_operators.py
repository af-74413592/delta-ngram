"""Formula, trajectory-gradient, and physical-row masking contracts."""
import pytest
import torch

from delta_ngram.operators import (
    WriteOperator, functional_d1, functional_d2, functional_d3, functional_d4,
)
from delta_ngram.state import SparseMemoryState
from delta_ngram.writer import ForwardWriter


@pytest.mark.parametrize("name", ["C", "D0", "D1", "D2", "D3", "D4"])
def test_formula_common_old_snapshot_zero_weight_counts_and_padding(name):
    state = SparseMemoryState(torch.tensor([2, 7]), torch.tensor([[2., 4.], [8., 9.]]),
                              torch.tensor([3., 2.]))
    rows = torch.tensor([2, 2, 5, 0])
    target = torch.tensor([[6., 8.], [10., 12.], [3., 5.], [999., 999.]])
    beta = torch.tensor([.5, 1., 0., 1.])
    operator = WriteOperator(name, kappa=.5, precision_init=2., frozen_rows=(2,) if name == "D4" else ())
    result = operator(state, rows, target, beta)
    assert result.rows.tolist() == [2, 5, 7]
    expected = {"C": [8., 10.], "D0": [7., 9.], "D1": [7., 9.],
                "D2": [4., 6.], "D3": [2. + 5. / 2.75, 4. + 5. / 2.75], "D4": [2., 4.]}
    torch.testing.assert_close(result.values[0], torch.tensor(expected[name]))
    torch.testing.assert_close(result.values[1], torch.tensor([3., 5.] if name == "C" else [0., 0.]))
    assert result.counts.tolist() == ([3., 1., 2.] if name == "D4" else [4., 1., 2.])
    assert torch.equal(result.values[-1], state.values[-1])
    if name == "D3":
        assert result.precision.tolist() == [2.75, 2., 2.]
    else:
        assert result.precision is None
    assert state.values.tolist() == [[2., 4.], [8., 9.]] and state.counts.tolist() == [3., 2.]


@pytest.mark.parametrize("name", ["C", "D0", "D1", "D2", "D3", "D4"])
def test_permuting_events_preserves_group_update_and_empty_is_noop(name):
    state = SparseMemoryState.empty(3)
    rows = torch.tensor([3, 5, 3, 0, 5, 6])
    target = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    beta = torch.tensor([.1, .2, .3, 1., .0, .5])
    order = torch.tensor([4, 2, 5, 1, 3, 0])
    operator = WriteOperator(name, frozen_rows=(5,) if name == "D4" else ())
    first, second = operator(state, rows, target, beta), operator(state, rows[order], target[order], beta[order])
    assert torch.equal(first.rows, second.rows) and torch.equal(first.counts, second.counts)
    torch.testing.assert_close(first.values, second.values)
    if name == "D3":
        torch.testing.assert_close(first.precision, second.precision)
    assert operator(first, torch.zeros(2, dtype=torch.long), target[:2], beta[:2]) is first


def test_d2_zero_kappa_is_bit_identical_to_executed_d1():
    torch.manual_seed(43)
    state = SparseMemoryState(torch.tensor([2, 7]), torch.randn(2, 4), torch.tensor([3., 9.]))
    rows, target, beta = torch.tensor([2, 7, 2, 9, 9]), torch.randn(5, 4), torch.rand(5)
    first = functional_d1(state, rows, target, beta)
    second = functional_d2(state, rows, target, beta, kappa=0.)
    assert torch.equal(first.values, second.values)


def test_d3_precision_graph_and_cumulative_fixed_targets_solution():
    beta1 = torch.tensor([.4, .8], requires_grad=True)
    beta2 = torch.tensor([.3], requires_grad=True)
    first_target = torch.tensor([[1., 3.], [5., 7.]], requires_grad=True)
    second_target = torch.tensor([[9., 11.]], requires_grad=True)
    first = functional_d3(SparseMemoryState.empty(2), torch.tensor([3, 3]), first_target, beta1,
                          precision_init=2.)
    first.precision.retain_grad()
    saved_values, saved_q = first.values.clone(), first.precision.clone()
    second = functional_d3(first, torch.tensor([3]), second_target, beta2, precision_init=2.)
    expected_q = 2. + beta1.mean() + beta2[0]
    expected = ((beta1[:, None] * first_target).mean(0) + beta2[0] * second_target[0]) / expected_q
    torch.testing.assert_close(second.values[0], expected)
    torch.testing.assert_close(second.precision[0], expected_q)
    second.values.square().sum().backward()
    for value in (beta1, beta2, first_target, second_target):
        assert value.grad is not None and value.grad.abs().sum() > 0
    assert first.precision.grad is not None and first.precision.grad.abs().sum() > 0
    assert torch.equal(first.values, saved_values) and torch.equal(first.precision, saved_q)
    assert second.counts.tolist() == [2.]


def test_d3_gate_gradient_matches_finite_difference_through_history():
    def trajectory(gates):
        first = functional_d3(SparseMemoryState.empty(1), torch.tensor([4]),
                              torch.tensor([[2.]]), gates[:1], precision_init=.7)
        second = functional_d3(first, torch.tensor([4]), torch.tensor([[5.]]), gates[1:], precision_init=.7)
        return second.values.square().sum()
    gates = torch.tensor([.3, .6], requires_grad=True)
    trajectory(gates).backward()
    differences = []
    for i in range(2):
        direction = torch.zeros(2)
        direction[i] = 1e-3
        differences.append((trajectory(gates.detach() + direction) - trajectory(gates.detach() - direction)) / 2e-3)
    torch.testing.assert_close(gates.grad, torch.stack(differences), rtol=2e-3, atol=2e-3)


def test_d3_zero_gate_retains_values_precision_but_advances_observation_count():
    state = SparseMemoryState(torch.tensor([3]), torch.tensor([[2., 4.]]), torch.tensor([7.]), torch.tensor([2.]))
    result = functional_d3(state, torch.tensor([3, 3]), torch.randn(2, 2), torch.zeros(2))
    assert torch.equal(result.values, state.values) and torch.equal(result.precision, state.precision)
    assert result.counts.item() == 8


def test_d4_freezes_entire_physical_row_including_content_collisions_counts_and_q():
    state = SparseMemoryState(torch.tensor([3, 6]), torch.tensor([[2., 4.], [1., 2.]]),
                              torch.tensor([7., 4.]), torch.tensor([2., 5.]))
    # Both events addressed to row 3 are excluded regardless of token provenance.
    result = functional_d4(state, torch.tensor([3, 3, 6, 9]), torch.full((4, 2), 50.),
                           torch.ones(4), frozen_rows=(3, 9))
    assert result.rows.tolist() == [3, 6]  # frozen unseen row 9 is never allocated
    assert torch.equal(result.values[0], state.values[0])
    assert result.counts.tolist() == [7., 5.] and result.precision.tolist() == [2., 5.]
    assert result.values[1].tolist() == [50., 50.]


def test_d0_gate_reads_evidence_only_while_targets_match_within_each_writer():
    torch.manual_seed(17)
    writer = ForwardWriter(4, hidden_dim=8, operator="D0")
    hidden, rows, mask = torch.randn(1, 3, 4), torch.tensor([[[0, 0], [3, 0], [4, 7]]]), torch.ones(1, 3)
    empty = SparseMemoryState.empty(4)
    old = SparseMemoryState(torch.tensor([3, 4, 7]), torch.full((3, 4), 100.), torch.full((3,), 200.))
    first, second = writer.events(hidden, rows, mask, empty), writer.events(hidden, rows, mask, old)
    assert torch.equal(first.target, second.target) and torch.equal(first.beta, second.beta)
    assert writer.gate_head[0].in_features == writer.target_head[0].in_features


def test_writer_event_right_window_pool_and_batch_boundaries():
    writer = ForwardWriter(1, hidden_dim=4, orders=(1,), right_window=2)
    hidden = torch.tensor([[[1.], [2.], [3.], [99.]], [[10.], [20.], [30.], [40.]]])
    rows = torch.tensor([[[1], [2], [3], [4]], [[5], [6], [7], [8]]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    captured = []
    handle = writer.target_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        writer.events(hidden, rows, mask, SparseMemoryState.empty(1))
    finally:
        handle.remove()
    features = captured[0]
    assert features[:, 0].tolist() == [1., 2., 3., 10., 20., 30., 40.]
    assert features[:, 1].tolist() == [2.5, 3., 0., 25., 35., 40., 0.]
    assert features[:, 2].tolist() == [2., 2., 2., 25., 25., 25., 25.]


@pytest.mark.parametrize("name", ["C", "D0", "D1", "D2", "D3", "D4"])
def test_writer_train_and_no_grad_have_identical_inner_values(name):
    writer = ForwardWriter(4, hidden_dim=8, operator=WriteOperator(name, frozen_rows=(2,) if name == "D4" else ()))
    hidden, rows, mask = torch.randn(1, 3, 4), torch.tensor([[[0, 0], [2, 0], [3, 9]]]), torch.ones(1, 3)
    old = SparseMemoryState.empty(4)
    graph = writer(hidden, rows, mask, old)
    with torch.no_grad():
        inference = writer(hidden, rows, mask, old)
    assert torch.equal(graph.values, inference.values)
    if name == "D3":
        assert torch.equal(graph.precision, inference.precision)


def test_operator_rejects_invalid_hyperparameters_and_fractional_frozen_addresses():
    for kwargs in ({"name": "unknown"}, {"kappa": -1.}, {"precision_init": 0.},
                   {"name": "D1", "frozen_rows": (3,)}, {"name": "D4", "frozen_rows": (3.5,)}):
        with pytest.raises(ValueError):
            WriteOperator(**kwargs)
    with pytest.raises(ValueError, match="positive physical row IDs"):
        functional_d4(SparseMemoryState.empty(2), torch.tensor([3]), torch.ones(1, 2),
                      torch.ones(1), frozen_rows=(3.5,))

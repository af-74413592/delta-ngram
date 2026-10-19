"""Small CPU algebra checks for docs/formal_d1_properties_20261019.md.

This checks finite examples and correspondence to production code. It is not a
formal proof assistant, a model benchmark, or a certification of learned weights.
It reads no corpus or evaluation split and downloads no model.
"""
from __future__ import annotations

from fractions import Fraction as Q
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delta_ngram.operators import functional_d1  # noqa: E402
from delta_ngram.reader import ExternalHashAdapter  # noqa: E402
from delta_ngram.state import SparseMemoryState  # noqa: E402
from delta_ngram.writer import ForwardWriter  # noqa: E402


def rational_update(x, targets, gates):
    """Return the exact result, a and b for a fixed row."""
    assert len(targets) == len(gates)
    if not targets:
        return x, Q(0), tuple(Q(0) for _ in x)
    m = len(targets)
    a = sum(gates, Q(0)) / m
    b = tuple(sum((gate * target[j] for gate, target in zip(gates, targets)), Q(0)) / m
              for j in range(len(x)))
    return tuple((1 - a) * value + offset for value, offset in zip(x, b)), a, b


def exact_convex_and_path_checks():
    count = 0
    values = tuple(map(Q, (-2, -1, 0, 1, 2)))
    gates = (Q(0), Q(1, 4), Q(1, 2), Q(1))
    for x, t1, t2, g1, g2 in itertools.product(values, values, values, gates, gates):
        result, a, _ = rational_update((x,), ((t1,), (t2,)), (g1, g2))
        assert min(x, t1, t2) <= result[0] <= max(x, t1, t2)
        assert 0 <= a <= 1
        assert (1 - a) + g1 / 2 + g2 / 2 == 1
        count += 1
    assert count == 2000

    initial = (Q(1), Q(-1))
    schedule = (
        ((Q(2), Q(0)), (Q(-2), Q(2))),
        (),
        ((Q(1), Q(2)), (Q(0), Q(-2)), (Q(-1), Q(0))),
        ((Q(-2), Q(1)),),
    )
    x = initial
    path = []
    for targets in schedule:
        adaptive = tuple((x[0] + 2) / 4 if i % 2 == 0 else Q(0)
                         for i in range(len(targets)))
        assert all(0 <= gate <= 1 for gate in adaptive)
        x, a, b = rational_update(x, targets, adaptive)
        assert max(map(abs, x)) <= 2
        path.append((targets, adaptive, a, b))
    retention = math.prod(1 - item[2] for item in path)
    expanded = [retention * v for v in initial]
    weights = [retention]
    for k, (targets, adaptive, _, _) in enumerate(path):
        later = math.prod(1 - item[2] for item in path[k + 1:])
        for target, gate in zip(targets, adaptive):
            coefficient = later * gate / len(targets)
            weights.append(coefficient)
            expanded = [v + coefficient * t for v, t in zip(expanded, target)]
    assert tuple(expanded) == x
    assert all(weight >= 0 for weight in weights)
    assert sum(weights) == 1

    # Reuse the realized gates as an evidence-only schedule for two initial values.
    left, right = (Q(1), Q(-1)), (Q(-2), Q(2))
    initial_difference = tuple(u - v for u, v in zip(left, right))
    for targets, fixed_gates, _, _ in path:
        left, _, _ = rational_update(left, targets, fixed_gates)
        right, _, _ = rational_update(right, targets, fixed_gates)
    assert tuple(u - v for u, v in zip(left, right)) == tuple(
        retention * value for value in initial_difference)
    return {"scalar_convex_cases": count, "adaptive_path_blocks": len(path),
            "adaptive_path_value": list(map(str, x)),
            "path_retention": str(retention), "coefficient_sum": "1",
            "d0_exact_sensitivity": True}


def operator_checks():
    state = SparseMemoryState(torch.tensor([1, 3]),
                              torch.tensor([[2.0, -1.0], [7.0, 8.0]]),
                              torch.tensor([4.0, 6.0]))
    before = (state.rows.clone(), state.values.clone(), state.counts.clone())
    rows = torch.tensor([1, 1, 0, 2])
    targets = torch.tensor([[6.0, 3.0], [100.0, 100.0], [900.0, 900.0], [9.0, 9.0]])
    beta = torch.tensor([0.5, 0.0, 1.0, 0.0])
    updated = functional_d1(state, rows, targets, beta)
    torch.testing.assert_close(updated.rows, torch.tensor([1, 2, 3]))
    torch.testing.assert_close(updated.values,
                               torch.tensor([[3.0, 0.0], [0.0, 0.0], [7.0, 8.0]]))
    torch.testing.assert_close(updated.counts, torch.tensor([5.0, 1.0, 6.0]))
    for old, current in zip(before, (state.rows, state.values, state.counts)):
        torch.testing.assert_close(old, current, rtol=0, atol=0)
    for permutation in itertools.permutations(range(len(rows))):
        index = torch.tensor(permutation)
        result = functional_d1(state, rows[index], targets[index], beta[index])
        torch.testing.assert_close(result.values, updated.values)
        torch.testing.assert_close(result.counts, updated.counts)
    without_zero = functional_d1(state, rows[:1], targets[:1], beta[:1])
    torch.testing.assert_close(without_zero.values[0], torch.tensor([4.0, 1.0]))
    padding_only = functional_d1(state, torch.tensor([0]), targets[:1], beta[:1])
    assert padding_only is state
    return {"shared_snapshot": True, "zero_gate_denominator": True,
            "zero_gate_count_advances": True, "padding_excluded": True,
            "input_state_preserved": True, "binary_exact_permutations": 24}


def counterexample_check():
    writer = ForwardWriter(width=1, hidden_dim=2, orders=(1,), right_window=0)
    with torch.no_grad():
        for parameter in writer.parameters():
            parameter.zero_()
        # Gate input = [five evidence features, x, t-x, log1p(c)].
        old_value_index = writer.target_head[0].in_features
        writer.gate_head[0].weight[0, old_value_index] = 1
        writer.gate_head[0].weight[1, old_value_index] = -1
        writer.gate_head[-1].weight[0] = torch.tensor([-1.0, 1.0])
    hidden, rows, mask = torch.zeros(1, 1, 1), torch.ones(1, 1, 1, dtype=torch.long), torch.ones(1, 1)

    def run(value):
        state = SparseMemoryState(torch.tensor([1]), value.reshape(1, 1), torch.zeros(1))
        return writer(hidden, rows, mask, state).values.squeeze()

    old = torch.tensor(2.0, requires_grad=True)
    value = run(old)
    derivative = torch.autograd.grad(value, old)[0]
    beta = 1 / (1 + math.exp(2))
    analytical = 1 - beta + 2 * beta * (1 - beta)
    assert math.isclose(derivative.item(), analytical, abs_tol=2e-7)
    assert derivative.item() > 1.09
    finite_difference = ((run(torch.tensor(2.001)) - run(torch.tensor(1.999))) / 0.002).item()
    assert math.isclose(finite_difference, analytical, abs_tol=2e-4)
    for sample in torch.linspace(0, 3, 31):
        result = run(sample).item()
        assert 0 <= result <= sample.item() + 1e-6
    return {"realized_in_ForwardWriter": True, "target": 0,
            "gate_at_2": beta, "value_at_2": value.item(),
            "derivative_analytic": analytical, "derivative_autograd": derivative.item(),
            "derivative_finite_difference": finite_difference,
            "invariant_interval": [0, 3]}


def jacobian_checks():
    torch.manual_seed(1729)
    hidden = torch.tensor([[[1.0, -1.0], [0.5, 0.25], [-0.25, 0.75]]])
    rows, mask = torch.ones(1, 3, 1, dtype=torch.long), torch.ones(1, 3)
    x = torch.tensor([0.25, -0.5], requires_grad=True)
    counts = torch.tensor([3.0])
    report = {}
    for name in ("D1", "D0"):
        writer = ForwardWriter(width=2, hidden_dim=4, orders=(1,), operator=name)
        with torch.no_grad():
            writer.gate_head[-1].weight.fill_(0.3)
            writer.gate_head[-1].bias.fill_(-0.5)
        state = SparseMemoryState(torch.tensor([1]), x.reshape(1, 2), counts)
        events = writer.events(hidden, rows, mask, state)
        a = events.beta.mean()
        expected = (1 - a) * torch.eye(2)
        if name == "D1":
            gradients = torch.stack([torch.autograd.grad(gate, x, retain_graph=True)[0]
                                     for gate in events.beta])
            expected = expected + (events.target - x).T @ gradients / len(events.beta)

        def value_update(value):
            old = SparseMemoryState(torch.tensor([1]), value.reshape(1, 2), counts)
            return writer(hidden, rows, mask, old).values[0]

        observed = torch.autograd.functional.jacobian(value_update, x)
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=2e-7)
        report[name] = {"max_jacobian_error": (observed - expected).abs().max().item(),
                        "mean_gate": a.item()}

        if name == "D1":
            def augmented_update(value_and_count):
                old = SparseMemoryState(torch.tensor([1]), value_and_count[:2].reshape(1, 2),
                                        value_and_count[2:])
                result = writer(hidden, rows, mask, old)
                return torch.cat((result.values[0], result.counts))

            augmented = torch.autograd.functional.jacobian(
                augmented_update, torch.tensor([0.25, -0.5, 3.0]))
            torch.testing.assert_close(augmented[2], torch.tensor([0.0, 0.0, 1.0]))
            torch.testing.assert_close(augmented[:2, :2], observed)
            report["count_coordinate_derivative"] = augmented[2].tolist()
    return report


def collision_checks():
    x = (Q(1), Q(-1))
    intended = ((Q(2), Q(0)), (Q(0), Q(2)))
    colliding = ((Q(-2), Q(0)), (Q(1), Q(-2)), (Q(2), Q(2)))
    gates_i, gates_c = (Q(1), Q(1, 2)), (Q(1, 3), Q(0), Q(1))
    f_i, _, _ = rational_update(x, intended, gates_i)
    f_c, _, _ = rational_update(x, colliding, gates_c)
    combined, _, _ = rational_update(x, intended + colliding, gates_i + gates_c)
    share = Q(len(colliding), len(intended) + len(colliding))
    assert combined == tuple((1 - share) * u + share * v for u, v in zip(f_i, f_c))
    error = max(abs(u - v) for u, v in zip(combined, f_i))
    assert error <= 2 * 2 * share  # Infinity-norm radius R=2.

    diluted, _, _ = rational_update(x, intended + colliding, gates_i + (Q(0),) * 3)
    assert tuple(u - v for u, v in zip(diluted, f_i)) == tuple(
        -share * (u - v) for u, v in zip(f_i, x))
    tight, _, _ = rational_update((Q(0),), ((Q(2),), (Q(2),), (Q(-2),)), (Q(1),) * 3)
    assert abs(tight[0] - 2) == Q(4, 3)  # 2 R q / (p+q), p=2, q=1.
    return {"partition_identity": True, "zero_gate_dilution_identity": True,
            "infinity_norm_example_error": str(error),
            "infinity_norm_example_bound": str(4 * share),
            "tight_scalar_error_and_bound": "4/3"}


def hypothetical_random_hash_checks():
    buckets, unrelated, intended_events = 4, 3, 2
    outcomes = tuple(itertools.product(range(buckets), repeat=unrelated + 1))
    collisions = [sum(value == assignment[0] for value in assignment[1:])
                  for assignment in outcomes]
    mean_count = Q(sum(collisions), len(outcomes))
    any_collision = Q(sum(q > 0 for q in collisions), len(outcomes))
    mean_share = sum((Q(q, intended_events + q) for q in collisions), Q(0)) / len(outcomes)
    jensen = mean_count / (intended_events + mean_count)
    assert mean_count == Q(unrelated, buckets)
    assert any_collision == 1 - (1 - Q(1, buckets)) ** unrelated
    assert any_collision <= min(Q(1), Q(unrelated, buckets))
    assert mean_share <= jensen
    return {"model": "hypothetical independent uniform key hashes, not repository hash",
            "enumerated_assignments": len(outcomes), "buckets": buckets,
            "distinct_unrelated_keys": unrelated, "expected_colliders": str(mean_count),
            "probability_any_collision": str(any_collision),
            "expected_collision_share": str(mean_share), "jensen_upper_bound": str(jensen)}


def reader_checks():
    adapter = ExternalHashAdapter(buckets=4, orders=(2, 3))
    rows = torch.tensor([[[1, 5], [0, 5]]])
    first = SparseMemoryState(torch.tensor([1, 5]), torch.tensor([[1.0, -1.0], [2.0, 0.0]]),
                              torch.zeros(2))
    second = SparseMemoryState(torch.tensor([1, 5]), torch.tensor([[0.0, 1.0], [-1.0, 1.0]]),
                               torch.zeros(2))
    difference = adapter.read_residual(first, rows) - adapter.read_residual(second, rows)
    row_differences = first.lookup(rows)[0] - second.lookup(rows)[0]
    expected = row_differences.sum(dim=-2) / math.sqrt(2)
    torch.testing.assert_close(difference, expected)
    upper_bound = torch.linalg.vector_norm(row_differences, dim=-1).sum(dim=-1) / math.sqrt(2)
    assert bool((torch.linalg.vector_norm(difference, dim=-1) <= upper_bound + 1e-6).all())
    return {"implemented_readout_matches_sum": True, "triangle_bound": True,
            "padding_contribution_zero": bool((row_differences[0, 1, 0] == 0).all())}


def main():
    torch.set_num_threads(1)
    sources = ("operators.py", "writer.py", "state.py", "reader.py", "ngram_lookup.py", "batching.py")
    report = {
        "status": "all_checks_passed",
        "scope": "CPU algebra and implementation correspondence; not model-performance evidence",
        "torch_version": torch.__version__,
        "device": "cpu",
        "source_sha256": {name: hashlib.sha256((ROOT / "src" / "delta_ngram" / name).read_bytes()).hexdigest()
                          for name in sources},
        "exact_arithmetic": exact_convex_and_path_checks(),
        "operator": operator_checks(),
        "d1_noncontraction_counterexample": counterexample_check(),
        "jacobians": jacobian_checks(),
        "collision_bounds": collision_checks(),
        "hypothetical_random_hash": hypothetical_random_hash_checks(),
        "reader": reader_checks(),
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

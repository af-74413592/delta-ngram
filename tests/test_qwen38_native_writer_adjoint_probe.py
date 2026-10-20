import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "delta-ngram" / "src"))

from qwen38_native_writer_adjoint_probe import evaluate_adjoint_series, state_scalar


def test_full_two_write_history_fd_rejects_detached_first_write():
    theta = torch.tensor([.4, -.2], dtype=torch.float32, requires_grad=True)
    rows, cotangent = torch.tensor([3]), torch.tensor([[.7]])

    def phi(value, detach=False):
        # Two actual parameter-dependent writes; the second retains old state.
        first = value[0].sigmoid() * value[1]
        if detach:
            first = first.detach()
        second = .8 * first + value[0].square()
        state = SimpleNamespace(rows=rows, values=second.reshape(1, 1),
                                block_clock=torch.tensor([2]))
        return state_scalar(state, rows, cotangent, 2)

    baseline = float(phi(theta).detach())
    grad = torch.autograd.grad(phi(theta), theta)[0]
    direction = grad / grad.norm()
    pairs = []
    for step in (.001, .002):
        plus = float(phi(theta + step * direction).detach())
        minus = float(phi(theta - step * direction).detach())
        pairs.append({"step": step, "plus_scalar": plus, "minus_scalar": minus})
    derivative = float((grad * direction).sum())
    assert evaluate_adjoint_series(pairs, [derivative] * 2, [baseline] * 3)["accepted"]
    detached_grad = torch.autograd.grad(phi(theta, detach=True), theta)[0]
    detached_derivative = float((detached_grad * direction).sum())
    assert not evaluate_adjoint_series(
        pairs, [detached_derivative] * 2, [baseline] * 3)["accepted"]
    assert not evaluate_adjoint_series(pairs, [2 * derivative] * 2, [baseline] * 3)["accepted"]


def test_state_contract_requires_actual_two_write_clock():
    rows, cotangent = torch.tensor([1]), torch.ones(1, 2)
    state = SimpleNamespace(rows=rows, values=torch.ones(1, 2, requires_grad=True),
                            block_clock=torch.tensor([1]))
    with pytest.raises(ValueError, match="every registered write"):
        state_scalar(state, rows, cotangent, 2)
    state.block_clock[0] = 2
    with pytest.raises(ValueError, match="stored sparse rows"):
        state_scalar(state, torch.tensor([2]), cotangent, 2)


def test_uninformative_writer_scalar_does_not_authorize():
    pairs = [{"step": step, "plus_scalar": 0., "minus_scalar": 0.}
             for step in (.001, .002)]
    assert not evaluate_adjoint_series(pairs, [0., 0.], [0., 0., 0.])["accepted"]


def test_first_stable_pair_cannot_be_replaced_to_match_gradient():
    pairs = []
    for step, derivative in ((.001, 1.), (.002, 1.), (.004, 2.), (.008, 2.)):
        pairs.append({"step": step, "plus_scalar": step * derivative,
                      "minus_scalar": -step * derivative})
    result = evaluate_adjoint_series(pairs, [2., 2.], [0., 0., 0.])
    assert result["selected_steps"] == [.001, .002]
    assert not result["accepted"]

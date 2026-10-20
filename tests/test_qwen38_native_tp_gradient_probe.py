"""Numerical receipt decisions; no CUDA, model, network, or distributed setup."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "native_qwen38"))
sys.path.insert(0, str(ROOT / "src"))
from qwen38_native_tp_gradient_probe import evaluate_fd_series


def pair(step, slope, *, changed=1.):
    # Independently evaluated quadratic: its central difference is slope.
    return {"step": step, "plus_loss": 5. + slope * step + 3 * step ** 2,
            "minus_loss": 5. - slope * step + 3 * step ** 2,
            "lookup_changed_fractions": [changed, changed]}


def evaluate(pairs, gradients=(2., 2.), baseline=(5., 5., 5.)):
    return evaluate_fd_series(
        pairs, gradients, runtime_identity_sha256="a" * 64,
        direction_id="independent-quadratic-direction",
        baseline_losses=list(baseline))


def test_full_tp_replicas_accept_replica_mean_and_reject_factor_two():
    result = evaluate([pair(.001, 2), pair(.002, 2), pair(.004, 7)])
    assert result["status"] == "accepted"
    assert result["selected_reduction"] == "replica"
    assert set(result["accepted_policy_receipts"]) == {"replica", "mean"}
    assert "sum" in result["rejected_policies"]
    assert result["selected_steps"] == [.001, .002]
    assert result["pairs"][2]["finite_difference"] == pytest.approx(7)


def test_partitioned_tp_contributions_accept_only_sum():
    result = evaluate([pair(.001, 2), pair(.002, 2)], gradients=(.75, 1.25))
    assert result["selected_reduction"] == "sum"
    assert set(result["accepted_policy_receipts"]) == {"sum"}


@pytest.mark.parametrize("pairs,baseline", [
    ([pair(.001, 2, changed=0), pair(.002, 2, changed=0)], (5., 5., 5.)),
    ([pair(.001, 2), pair(.002, 2)], (5., 5.1, 5.)),
    ([pair(.001, 0), pair(.002, 0)], (5., 5., 5.)),
    ([pair(.001, 2), pair(.002, 4), pair(.004, 8)], (5., 5., 5.)),
])
def test_quantization_noise_zero_response_and_unstable_steps_are_inconclusive(pairs, baseline):
    result = evaluate(pairs, baseline=baseline)
    assert result["status"] == "inconclusive"
    assert result["selected_policy_receipt"] is None


def test_do_not_cherry_pick_later_steps_that_match_autograd():
    result = evaluate([pair(.001, 7), pair(.002, 7), pair(.004, 2), pair(.008, 2)])
    assert result["selected_steps"] == [.001, .002]
    assert result["status"] == "inconclusive"
    assert result["accepted_policy_receipts"] == {}


def test_skip_unresolvable_small_steps_using_forward_evidence_only():
    result = evaluate([pair(.001, 0, changed=0), pair(.002, 2), pair(.004, 2)])
    assert result["selected_steps"] == [.002, .004]
    assert result["selected_reduction"] == "replica"

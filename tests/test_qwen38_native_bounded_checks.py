from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
from qwen38_native_bounded_checks import BoundedStability, E2E_STEPS, evaluate_end_to_end


def pairs(slopes):
    return [{"step": step, "plus": 2 + step * slope, "minus": 2 - step * slope}
            for step, slope in zip(E2E_STEPS, slopes)]


def complete(tracker, gradient, saturated=0, events=1000):
    reasons = tracker.observe_step(gradient, saturated, events)
    if not reasons:
        tracker.complete_step(gradient)
    return reasons


def test_both_registered_slopes_and_exact_baseline_are_required():
    assert evaluate_end_to_end(.4, pairs([.4, .4]), True)["status"] == "accepted"
    assert evaluate_end_to_end(.4, pairs([.4, .7]), True)["status"] == "inconclusive"
    assert evaluate_end_to_end(.4, pairs([.4, .4]), False)["status"] == "inconclusive"
    assert evaluate_end_to_end(.005, pairs([.005, .005]), True)["status"] == "inconclusive"


def test_e2e_rejects_nonfinite_and_protocol_drift():
    assert evaluate_end_to_end(float("nan"), pairs([1, 1]), True)["status"] == "rejected"
    with pytest.raises(ValueError, match="fixed step"):
        evaluate_end_to_end(1, pairs([1, 1])[::-1], True)
    with pytest.raises(ValueError, match="boolean"):
        evaluate_end_to_end(1, pairs([1, 1]), 1)


def test_spike_uses_only_previous_five_completed_gradients_and_two_steps():
    tracker = BoundedStability()
    for _ in range(5):
        assert not complete(tracker, 1)
    assert not complete(tracker, 11)
    reasons = tracker.observe_step(11, 0, 1000)
    assert reasons == ["raw_gradient_exceeds_ten_times_previous_five_median_twice"]
    assert tracker.last_observation["previous_five_completed_gradient_median"] == 1
    assert tracker.completed_steps == 6
    with pytest.raises(ValueError, match="stopped"):
        tracker.complete_step(11)


def test_spike_streak_resets_and_threshold_is_strict():
    tracker = BoundedStability()
    for _ in range(5):
        complete(tracker, 1)
    assert not complete(tracker, 11)
    assert not complete(tracker, 10)
    assert tracker.gradient_spike_streak == 0
    assert not complete(tracker, 11)
    assert tracker.gradient_spike_streak == 1


def test_beta_one_percent_is_allowed_two_strictly_larger_steps_stop():
    tracker = BoundedStability()
    assert not complete(tracker, 1, 10, 1000)
    assert not complete(tracker, 1, 11, 1000)
    assert tracker.observe_step(1, 11, 1000) == [
        "beta_saturated_fraction_exceeds_one_percent_twice"]


def test_nonfinite_gradient_stops_immediately_and_pending_is_enforced():
    tracker = BoundedStability()
    assert tracker.observe_step(float("inf"), 0, 10) == ["nonfinite_raw_gradient_l2"]
    with pytest.raises(ValueError, match="previous"):
        tracker.observe_step(1, 0, 10)
    with pytest.raises(ValueError, match="finite"):
        tracker.complete_step(float("inf"))


def test_probe_uses_immutable_step_zero_baseline():
    tracker = BoundedStability()
    with pytest.raises(ValueError, match="step-zero"):
        tracker.observe_probe(4, 2)
    assert not tracker.observe_probe(0, 2)
    assert not tracker.observe_probe(4, 2.09)
    assert not tracker.observe_probe(4, 2.1)
    assert tracker.observe_probe(8, 2.11)
    with pytest.raises(ValueError, match="immutable"):
        tracker.observe_probe(0, 3)
    assert tracker.observe_probe(4, float("nan")) == ["nonfinite_probe_mean_nll"]


def test_completed_gradient_cannot_be_substituted_or_observed_twice():
    tracker = BoundedStability()
    assert not tracker.observe_step(2, 0, 10)
    with pytest.raises(ValueError, match="match"):
        tracker.complete_step(1)
    with pytest.raises(ValueError, match="previous"):
        tracker.observe_step(2, 0, 10)
    tracker.complete_step(2)
    assert list(tracker.completed_gradients) == [2]

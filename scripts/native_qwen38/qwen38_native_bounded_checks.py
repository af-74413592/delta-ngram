"""Pure CPU acceptance rules for the frozen, bounded initial writer run.

No threshold fitting, retries, optimizer operations, tensor libraries or I/O.
"""
from __future__ import annotations

from collections import deque
import math
import statistics


E2E_STEPS = (2.0 ** -14, 2.0 ** -13)
E2E_RTOL = .20
E2E_ATOL = .001
E2E_MIN_ANALYTIC = .01


def evaluate_end_to_end(analytic, pairs, baseline_exact):
    """Evaluate both registered centered differences, without choosing a step.

    ``pairs`` is exactly two dicts with ``step``, ``plus``, and ``minus``.
    Losses and the analytic directional derivative use the same denominator.
    """
    if type(baseline_exact) is not bool:
        raise ValueError("baseline_exact must be a measured boolean")
    if len(pairs) != 2:
        raise ValueError("exactly two preregistered centered pairs are required")
    derivative = float(analytic)
    probes = []
    for expected, pair in zip(E2E_STEPS, pairs):
        if set(pair) != {"step", "plus", "minus"} or float(pair["step"]) != expected:
            raise ValueError("finite-difference fields or fixed step order changed")
        plus, minus = float(pair["plus"]), float(pair["minus"])
        slope = (plus - minus) / (2 * expected)
        finite = all(math.isfinite(x) for x in (plus, minus, slope, derivative))
        tolerance = E2E_ATOL + E2E_RTOL * max(abs(derivative), abs(slope))
        probes.append({
            "step": expected, "plus": plus, "minus": minus, "finite_difference": slope,
            "absolute_error": abs(slope - derivative) if finite else None,
            "tolerance": tolerance if finite else None,
            "finite": finite,
            "matches_analytic": bool(finite and abs(slope - derivative) <= tolerance),
        })
    reasons = []
    if not all(row["finite"] for row in probes):
        reasons.append("nonfinite_end_to_end_measurement")
    if not baseline_exact:
        reasons.append("analytic_and_evaluation_baselines_not_bitwise_equal")
    if math.isfinite(derivative) and abs(derivative) < E2E_MIN_ANALYTIC:
        reasons.append("analytic_direction_below_registered_informativeness_floor")
    finite = all(row["finite"] for row in probes)
    slopes = [row["finite_difference"] for row in probes]
    mutual_error = abs(slopes[0] - slopes[1]) if finite else None
    mutual_tolerance = E2E_RTOL * max(abs(x) for x in slopes) if finite else None
    stable = bool(finite and mutual_error <= mutual_tolerance)
    if finite and not stable:
        reasons.append("two_centered_slopes_disagree_by_more_than_twenty_percent")
    if finite and not all(row["matches_analytic"] for row in probes):
        reasons.append("centered_slope_disagrees_with_analytic_direction")
    status = ("rejected" if not finite else "inconclusive" if reasons else "accepted")
    return {
        "protocol": "qwen38_bounded_end_to_end_direction_v1", "status": status,
        "analytic_directional_derivative": derivative if math.isfinite(derivative) else None,
        "baseline_bitwise_equal": baseline_exact, "probes": probes,
        "mutual_slope_error": mutual_error, "mutual_slope_tolerance": mutual_tolerance,
        "mutual_slopes_stable": stable, "reasons": reasons,
        "fixed_steps": list(E2E_STEPS), "rtol": E2E_RTOL, "atol": E2E_ATOL,
        "minimum_absolute_analytic_direction": E2E_MIN_ANALYTIC,
        "interpretation": "One registered BF16 end-to-end direction only; not exact differentiability of quantized CE.",
    }


class BoundedStability:
    """Per-arm state; only completed steps enter the five-gradient reference."""

    def __init__(self):
        self.completed_gradients = deque(maxlen=5)
        self.completed_steps = 0
        self.gradient_spike_streak = 0
        self.beta_saturation_streak = 0
        self.initial_probe_nll = None
        self._pending = None
        self._pending_reasons = None
        self.last_observation = None

    def observe_step(self, raw_gradient_l2, beta_saturated, beta_events):
        if self._pending is not None:
            raise ValueError("complete the previous observed step before observing another")
        if (type(beta_saturated) is not int or type(beta_events) is not int
                or beta_events <= 0 or not 0 <= beta_saturated <= beta_events):
            raise ValueError("beta counts must be integers with 0 <= saturated <= positive events")
        gradient = float(raw_gradient_l2)
        if math.isfinite(gradient) and gradient < 0:
            raise ValueError("gradient L2 norm cannot be negative")
        reasons = []
        if not math.isfinite(gradient):
            reasons.append("nonfinite_raw_gradient_l2")
        median = statistics.median(self.completed_gradients) if len(self.completed_gradients) == 5 else None
        spike = bool(math.isfinite(gradient) and median is not None and gradient > 10 * median)
        self.gradient_spike_streak = self.gradient_spike_streak + 1 if spike else 0
        fraction = beta_saturated / beta_events
        saturated = fraction > .01
        self.beta_saturation_streak = self.beta_saturation_streak + 1 if saturated else 0
        if self.gradient_spike_streak >= 2:
            reasons.append("raw_gradient_exceeds_ten_times_previous_five_median_twice")
        if self.beta_saturation_streak >= 2:
            reasons.append("beta_saturated_fraction_exceeds_one_percent_twice")
        self.last_observation = {
            "step": self.completed_steps + 1,
            "raw_gradient_l2": gradient if math.isfinite(gradient) else None,
            "previous_five_completed_gradient_median": median,
            "gradient_spike_streak": self.gradient_spike_streak,
            "beta_saturated": beta_saturated, "beta_events": beta_events,
            "beta_saturated_fraction": fraction,
            "beta_saturation_streak": self.beta_saturation_streak,
            "stop_reasons": reasons.copy(),
        }
        self._pending, self._pending_reasons = gradient, reasons.copy()
        return reasons

    def complete_step(self, raw_gradient_l2):
        gradient = float(raw_gradient_l2)
        if self._pending is None or not math.isfinite(gradient) or gradient != self._pending:
            raise ValueError("completed gradient must match the finite pending observation")
        if self._pending_reasons:
            raise ValueError("a stopped step cannot be marked completed")
        self.completed_gradients.append(gradient)
        self.completed_steps += 1
        self._pending = self._pending_reasons = None

    def observe_probe(self, step, mean_nll):
        if type(step) is not int or step not in (0, 4, 8):
            raise ValueError("probe checkpoint must be 0, 4 or 8")
        nll = float(mean_nll)
        if not math.isfinite(nll):
            return ["nonfinite_probe_mean_nll"]
        if step == 0:
            if self.initial_probe_nll is not None:
                raise ValueError("initial probe baseline is immutable")
            self.initial_probe_nll = nll
            return []
        if self.initial_probe_nll is None:
            raise ValueError("measure step-zero probe before later checkpoints")
        return (["probe_mean_nll_increased_more_than_point_one_from_step_zero"]
                if nll > self.initial_probe_nll + .1 else [])

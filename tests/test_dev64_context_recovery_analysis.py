"""Scientific-statistic checks for the separate CPU context recovery analysis."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "dev64_context_recovery_analysis",
    Path(__file__).resolve().parents[1] / "scripts/analyze_dev64_context_recovery_20261020.py",
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_paired_ratio_keeps_exact_proportionality():
    denominator = np.linspace(.1, 2, 10000)
    stats, draws = analysis.ratio_statistics(.25, 1, denominator / 4, denominator)
    assert stats["recovery_percent"] == 25
    assert stats["recovery_percent_ci95_low"] == 25
    assert stats["recovery_percent_ci95_high"] == 25
    assert np.all(draws == .25)


def test_negative_denominator_draws_remain_and_unstable_interval_is_withheld():
    denominator = np.asarray([-1.0] * 1000 + [1.0] * 9000)
    stats, draws = analysis.ratio_statistics(.2, .8, np.full(10000, .2), denominator)
    assert stats["ratio_status"] == "denominator_ci_includes_zero"
    assert stats["denominator_nonpositive_draw_fraction"] == .1
    assert stats["denominator_negative_draws"] == 1000
    assert stats["discarded_bootstrap_draws"] == 0
    assert np.all(draws[:1000] == -.2)
    assert stats["all_draw_signed_ratio_percentile025"] == -.2
    assert stats["recovery_percent_ci95_low"] is None
    assert stats["recovery_percent_ci95_high"] is None


def test_exact_zero_draw_stays_at_its_index_and_prevents_interval():
    denominator = np.ones(10000)
    denominator[71] = 0
    stats, draws = analysis.ratio_statistics(.3, 1, np.full(10000, .3), denominator)
    assert stats["ratio_status"] == "zero_denominator_draws"
    assert stats["denominator_zero_draws"] == 1
    assert draws.shape == (10000,) and np.isnan(draws[71])
    assert np.count_nonzero(np.isnan(draws)) == 1
    assert stats["all_draw_signed_ratio_percentile025"] is None
    assert stats["recovery_percent_ci95_low"] is None
    assert stats["discarded_bootstrap_draws"] == 0


def test_nonpositive_mean_denominator_has_no_percentage():
    stats, _ = analysis.ratio_statistics(.3, -.1, np.full(10000, .3), np.full(10000, -.1))
    assert stats["ratio_status"] == "nonpositive_mean_denominator"
    assert stats["recovery_ratio"] is None and stats["recovery_percent"] is None


def test_percentage_is_not_clipped_to_accuracy_range():
    stats, _ = analysis.ratio_statistics(.3, .1, np.full(10000, .3), np.full(10000, .1))
    assert stats["recovery_percent"] == pytest.approx(300)
    stats, _ = analysis.ratio_statistics(-.2, .1, np.full(10000, -.2), np.full(10000, .1))
    assert stats["recovery_percent"] == pytest.approx(-200)


def test_family_draws_are_reproducible_and_preserve_sample_size():
    a = analysis.bootstrap_counts(64)
    b = analysis.bootstrap_counts(64)
    assert a.shape == (10000, 64)
    np.testing.assert_array_equal(a, b)
    assert np.all(a.sum(axis=1) == 64)
    assert np.any(a == 0) and np.any(a > 1)


def test_summary_uses_equal_family_ratio_of_means_and_shared_draws():
    # Half the events have a one-nat context gain, half have three nats.
    # Memory gain is one nat in every event: ratio-of-means is 1/2, whereas
    # mean-of-ratios is 2/3. Unequal target lengths must not change that result.
    families = [f"family_{i:02}" for i in range(64)]
    indexes, context, jobs = {}, {}, {}
    for nominal in analysis.NOMINAL:
        jobs[nominal] = {"label": str(nominal), "training_loss_tokens": nominal}
        indexes[nominal] = {}
        for i, family in enumerate(families):
            tokens = 10 if i < 32 else 1000
            gain = 1.0 if i < 32 else 3.0
            for distance in analysis.DISTANCES:
                empty = {"nll": 5.0, "first_target_nll": 7.0,
                         "target_tokens": tokens, "first_target_tokens": min(tokens, 384)}
                self_score = dict(empty, nll=4.0, first_target_nll=6.5)
                indexes[nominal][family, distance] = {
                    "pair_id": family, "conditions": {"empty": empty, "self": self_score}}
                context[family, distance] = {
                    "conditions": {"context": {"nll": 5.0 - gain, "first_target_nll": 7.0 - gain}}}
    counts = analysis.bootstrap_counts(64)
    rows, per_family, arrays = analysis.summarize(jobs, indexes, context, families, counts)
    assert len(rows) == 210 and len(per_family) == 13440
    assert sum(row["role"] == "primary" for row in rows) == 10
    for row in rows:
        expected = 50 if row["scope"] == "all_targets" else 25
        assert row["recovery_percent"] == expected
    # Identical gains across checkpoints/distances must have identical paired
    # draws, not merely approximately similar marginal confidence intervals.
    np.testing.assert_array_equal(arrays["recovery_ratio_draws"][:, 0],
                                  arrays["recovery_ratio_draws"][:, 10])
    np.testing.assert_array_equal(arrays["recovery_ratio_draws"][:, 0],
                                  arrays["recovery_ratio_draws"][:, 4])
    assert arrays["recovery_ratio_draws"].shape == (10000, 210)
    assert arrays["context_gain_draws"].shape == (10000, 10)

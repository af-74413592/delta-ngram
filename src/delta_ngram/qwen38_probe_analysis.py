"""CPU-only paired analysis for the fixed-RMSNorm, two-alpha stability check."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math
from numbers import Real
import random

ARMS = ("alpha100_rmsnorm", "alpha099_rmsnorm")
SOURCES = ("books", "code", "papers", "web", "news", "news_event")
STEPS = (0, 4, 8)
TARGETS = 128
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 104729


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    _require(isinstance(value, Real) and not isinstance(value, bool)
             and math.isfinite(value), f"{name} must be a finite real number")
    return float(value)


def _percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return (sorted_values[lower]
            + (sorted_values[upper] - sorted_values[lower]) * (position - lower))


def _interval(values):
    values = sorted(values)
    return [_percentile(values, .025), _percentile(values, .975)]


def _validated(rows):
    rows = list(rows)
    _require(len(rows) == 72, "requires exactly 2 arms x 3 steps x 12 samples")
    matrix, sample_sources, empty = {}, {}, {}
    for row in rows:
        _require(isinstance(row, Mapping), "each probe row must be a mapping")
        arm, step = row.get("arm"), row.get("step")
        sample, source = row.get("sample_id"), row.get("source")
        _require(arm in ARMS and type(step) is int and step in STEPS,
                 "unknown arm or checkpoint in fixed RMSNorm protocol")
        _require(isinstance(sample, str) and sample.strip() and source in SOURCES,
                 "sample_id and registered source are required")
        _require(type(row.get("loss_target_tokens")) is int
                 and row["loss_target_tokens"] == TARGETS,
                 "each sample must contain exactly 128 target tokens")
        if "query_target_tokens" in row:
            _require(type(row["query_target_tokens"]) is int
                     and row["query_target_tokens"] == TARGETS,
                     "query and loss target counts disagree")
        key = (arm, step, sample)
        _require(key not in matrix, "duplicate arm/step/sample probe row")
        loss = _finite(row.get("loss_sum"), "loss_sum")
        base = _finite(row.get("empty_loss_sum"), "empty_loss_sum")
        delta = _finite(row.get("delta_empty"), "delta_empty")
        _require(loss >= 0 and base >= 0, "NLL sums must be nonnegative")
        _require(math.isclose(delta, (base - loss) / TARGETS,
                             rel_tol=1e-7, abs_tol=1e-9),
                 "delta_empty disagrees with the paired loss sums")
        _require(sample not in sample_sources or sample_sources[sample] == source,
                 "one sample cannot change source across arms/checkpoints")
        _require(sample not in empty or empty[sample] == base,
                 "the same sample must reuse exactly the same empty baseline")
        sample_sources[sample], empty[sample] = source, base
        matrix[key] = loss
    by_source = {
        source: sorted(sample for sample, value in sample_sources.items() if value == source)
        for source in SOURCES
    }
    _require(all(len(samples) == 2 for samples in by_source.values()),
             "requires exactly two independent sample IDs per source")
    samples = [sample for source in SOURCES for sample in by_source[source]]
    _require(set(matrix) == {(arm, step, sample)
                            for arm in ARMS for step in STEPS for sample in samples},
             "probe rows do not form complete paired arms/checkpoints")
    return matrix, by_source, samples, empty


def analyze_probe_rows(probe_rows):
    """Return complete paired summaries; never select a norm or start a run.

    Input is 72 mappings with arm, step, sample_id, source, loss_sum,
    loss_target_tokens=128, empty_loss_sum and delta_empty. Extra diagnostics
    are ignored. Both alpha arms and steps 0/4/8 must share the same two
    families per each of six sources, with identical per-family empty losses.

    Bootstrap intervals cover step-4/8 learning gains and alpha contrasts.
    One local RNG draws two families with replacement within each source;
    each draw is shared across arms and checkpoints. Percentiles use linear
    interpolation at (n-1)*p. Input order and global random state do not matter.
    """
    matrix, by_source, samples, empty = _validated(probe_rows)
    scopes = {"overall": samples, **by_source}
    source_for = {sample: source for source, ids in by_source.items() for sample in ids}
    gains = {
        (arm, step): {
            sample: (matrix[(arm, 0, sample)] - matrix[(arm, step, sample)]) / TARGETS
            for sample in samples
        }
        for arm in ARMS for step in (4, 8)
    }
    contrasts = {
        step: {sample: gains[(ARMS[1], step)][sample] - gains[(ARMS[0], step)][sample]
               for sample in samples}
        for step in (4, 8)
    }
    distributions = defaultdict(list)
    rng = random.Random(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = {source: [ids[rng.randrange(2)], ids[rng.randrange(2)]]
                    for source, ids in by_source.items()}
        selected["overall"] = [sample for source in SOURCES for sample in selected[source]]
        for scope, ids in selected.items():
            denominator = len(ids)
            for step in (4, 8):
                for arm in ARMS:
                    distributions[(arm, scope, step)].append(
                        math.fsum(gains[(arm, step)][sample] for sample in ids) / denominator)
                distributions[("contrast", scope, step)].append(
                    math.fsum(contrasts[step][sample] for sample in ids) / denominator)

    def arm_scope(arm, scope, ids):
        count = TARGETS * len(ids)
        empty_sum = math.fsum(empty[sample] for sample in ids)
        checkpoints = {}
        for step in STEPS:
            total = math.fsum(matrix[(arm, step, sample)] for sample in ids)
            checkpoints[str(step)] = {
                "sample_count": len(ids), "loss_target_tokens": count,
                "loss_sum": total, "mean_loss": total / count,
                "empty_loss_sum": empty_sum, "mean_empty_loss": empty_sum / count,
                "delta_empty": (empty_sum - total) / count,
            }
        learning = {}
        for step in (4, 8):
            value = math.fsum(gains[(arm, step)][sample] for sample in ids) / len(ids)
            learning[str(step)] = {
                "gain_from_step0": value,
                "paired_family_bootstrap_ci95": _interval(distributions[(arm, scope, step)]),
            }
        return {"checkpoints": checkpoints, "learning": learning}

    per_arm = {
        arm: {
            "alpha": 1.0 if arm == ARMS[0] else .99, "norm_mode": "rmsnorm",
            "overall": arm_scope(arm, "overall", samples),
            "by_source": {source: arm_scope(arm, source, ids)
                          for source, ids in by_source.items()},
        }
        for arm in ARMS
    }

    def contrast_scope(scope, ids):
        return {
            str(step): {
                "gain_difference": math.fsum(contrasts[step][s] for s in ids) / len(ids),
                "paired_family_bootstrap_ci95": _interval(
                    distributions[("contrast", scope, step)]),
            }
            for step in (4, 8)
        }

    return {
        "protocol": "qwen38_fixed_rms_two_alpha_probe_analysis_v1",
        "norm_mode": "rmsnorm", "norm_choice_source": "prior explicit user decision",
        "automatic_norm_choice": False, "automatic_alpha_choice": False,
        "input_rows": len(matrix), "sample_count": len(samples),
        "sources": list(SOURCES), "steps": list(STEPS),
        "targets_per_sample": TARGETS,
        "self_probe_target_accesses": len(matrix) * TARGETS,
        "shared_empty_target_accesses": len(samples) * TARGETS,
        "per_arm": per_arm,
        "paired_alpha_contrast": {
            "definition": "gain(alpha0.99) - gain(alpha1.0); gain=L_self(step0)-L_self(step)",
            "overall": contrast_scope("overall", samples),
            "by_source": {source: contrast_scope(source, ids)
                          for source, ids in by_source.items()},
        },
        "paired_samples": [
            {"sample_id": sample, "source": source_for[sample],
             "empty_mean_loss": empty[sample] / TARGETS,
             "arms": {arm: {
                 "mean_loss_by_step": {str(step): matrix[(arm, step, sample)] / TARGETS
                                       for step in STEPS},
                 "gain_by_step": {str(step): gains[(arm, step)][sample] for step in (4, 8)},
             } for arm in ARMS}}
            for sample in samples
        ],
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "method": "paired family resampling within source, two draws per source",
            "interval": "95% percentile, linear interpolation at (n-1)*p",
            "paired_across_all_arms_and_checkpoints": True,
            "assumption": "each frozen sample_id represents one independent family",
            "scope": "descriptive uncertainty for twelve fixed training-side families",
        },
        "limitations": [
            "This summarizes recorded probe rows; it does not certify runtime, gradient or hash gates.",
            "No independent DEV64/FINAL performance or automatic continuation decision is inferred.",
        ],
    }

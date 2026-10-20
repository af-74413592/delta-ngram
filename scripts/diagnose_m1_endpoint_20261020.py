"""Decompose M1's endpoint donor-gap change using existing CPU-readable results.

This is a descriptive diagnosis, not a new experiment or causal identification.
It does not import torch, open model weights, read source articles, or access
sealed FINAL / incomplete DEV64 data. The previously computed 65M-to-100M CIs
are reused; one explicitly requested 95M-to-100M paired self-NLL check is added.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

import analyze_m1_results_20261020 as analysis


METRICS = ("self_nll", "empty_nll", "donor_nll", "delta_empty",
           "delta_donor", "donor_gain_over_empty")


def measures(own, empty, donor):
    return {
        "self_nll": own, "empty_nll": empty, "donor_nll": donor,
        "delta_empty": empty - own, "delta_donor": donor - own,
        "donor_gain_over_empty": empty - donor,
    }


def macro_measures(evaluation):
    values = analysis.macro_points(evaluation)
    empty_categories = defaultdict(list)
    for row in evaluation["groups"]:
        empty_categories[evaluation["macro_source_groups"][row["source"]]].append(row)
    empty = math.fsum(
        math.fsum(row["empty_loss_sum"] for row in rows) / sum(row["loss_tokens"] for row in rows)
        for _, rows in sorted(empty_categories.items())) / len(empty_categories)
    return measures(values["self_nll"], empty, values["donor_nll"])


def pooled_measures(evaluation):
    return measures(*(evaluation[key] for key in ("self_nll", "empty_nll", "donor_nll")))


def loss_sums(evaluation):
    return {condition: math.fsum(row[condition + "_loss_sum"] for row in evaluation["groups"])
            for condition in ("self", "empty", "donor")}


def event_partitions(evaluation):
    """full first-block diagnostic is a subset; after_prefix is NOT subtractable."""
    all_event = evaluation["per_source"]["news_event"]
    first = evaluation["later_article_first_block"]["full"]
    total_losses, first_losses = loss_sums(all_event), loss_sums(first)
    remainder_tokens = all_event["loss_tokens"] - first["loss_tokens"]
    analysis.require(remainder_tokens > 0, "no remaining event blocks")
    remainder_losses = {key: total_losses[key] - first_losses[key] for key in total_losses}
    rows = []
    for label, count, losses in (
        ("first_block_full", first["loss_tokens"], first_losses),
        ("remaining_blocks_algebraic_residual", remainder_tokens, remainder_losses),
        ("all_news_event_targets", all_event["loss_tokens"], total_losses),
    ):
        rows.append({
            "partition": label, "target_tokens": count,
            **measures(*(losses[key] / count for key in ("self", "empty", "donor"))),
        })
    for condition in ("self", "empty", "donor"):
        rebuilt = math.fsum(row[condition + "_nll"] * row["target_tokens"] for row in rows[:2])
        analysis.require(math.isclose(rebuilt, total_losses[condition], rel_tol=0., abs_tol=1e-8),
                         "event target partition does not reconstruct")
    return rows


def training_diagnostics(path, evaluations):
    if not path.is_file():
        return [], {
            "available": False,
            "reason": "No local M1 metrics.jsonl; training summary contains counters, not train NLL.",
        }
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "train":
                records.append(row)
    records.sort(key=lambda row: row["step"])
    analysis.require(len({r["step"] for r in records}) == len(records), "duplicate training steps")
    analysis.require([r["step"] for r in records] == list(range(1, evaluations[-1]["step"] + 1)),
                     "training log does not cover every endpoint step exactly once")
    bins, previous_step, previous_source = [], 0, {}
    for evaluation in evaluations:
        step = evaluation["step"]
        if step == 0:
            continue
        selected = [row for row in records if previous_step < row["step"] <= step]
        count = sum(row["batch_loss_tokens"] for row in selected)
        last = selected[-1]
        source_tokens = {
            source: tokens - previous_source.get(source, 0)
            for source, tokens in last.get("source_loss_tokens", {}).items()
        }
        analysis.require(sum(source_tokens.values()) == count, "training source token mismatch")
        bins.append({
            "end_step": step, "end_training_loss_tokens": evaluation["training_loss_tokens"],
            "start_step_exclusive": previous_step, "optimizer_steps": len(selected),
            "batch_loss_tokens": count,
            "token_weighted_train_nll": math.fsum(
                row["nll"] * row["batch_loss_tokens"] for row in selected) / count,
            "median_step_train_nll": statistics.median(row["nll"] for row in selected),
            "mean_grad_norm_before_clip": math.fsum(
                row["grad_norm_before_clip"] for row in selected) / len(selected),
            **{"source_tokens_" + key: value for key, value in source_tokens.items()},
        })
        previous_step = step
        previous_source = last.get("source_loss_tokens", {})
    return bins, {
        "available": True, "path": str(path), "sha256": analysis.sha256(path),
        "training_steps": len(records), "source_article_text_read": False,
        "aggregation": "Target-token-weighted training NLL over optimizer steps between adjacent evaluation milestones.",
        "caution": "Different training batches/content in each interval. These are not a "
        "fixed training-probe loss curve; source balance alone does not equalize example difficulty.",
        "selected_interval": next(row for row in bins if row["end_step"] == 2713),
        "last_interval": bins[-1],
    }


def trend_intervals(evaluations):
    by_nominal = {row["nominal_loss_tokens"]: row for row in evaluations}
    result = []
    for start, end in ((5, 25), (25, 50), (50, 75), (75, 100), (50, 100), (90, 100), (95, 100)):
        a, b = by_nominal[start * 1000000], by_nominal[end * 1000000]
        x, y = macro_measures(a), macro_measures(b)
        actual_tokens = b["training_loss_tokens"] - a["training_loss_tokens"]
        result.append({
            "nominal_start_millions": start, "nominal_end_millions": end,
            "actual_added_training_loss_tokens": actual_tokens,
            "self_nll_improvement": x["self_nll"] - y["self_nll"],
            "self_nll_improvement_per_5m_actual_tokens":
                (x["self_nll"] - y["self_nll"]) * 5000000 / actual_tokens,
            "donor_gap_change": y["delta_donor"] - x["delta_donor"],
            "confidence_interval_in_this_table": False,
        })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=analysis.LAUNCH / "final_backup_20261020")
    parser.add_argument("--training-log", type=Path,
                        default=analysis.LAUNCH / "diagnostic_logs/metrics.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        default=analysis.RUNS / "m1_analysis_20261020/diagnostics")
    args = parser.parse_args(argv)
    paths = sorted(args.m1_dir.glob("eval_step_*.json"))
    evaluations = [analysis.load_json(path) for path in paths]
    evaluations.sort(key=lambda row: row["training_loss_tokens"])
    analysis.require(len(evaluations) == 21, "expected 21 fixed M1 evaluation points")
    selected = max(
        (e for e in evaluations if e["checkpoint_selection_eligible"]),
        key=lambda e: (e["source_macro_delta_donor"], -e["training_loss_tokens"]))
    endpoint = evaluations[-1]
    series, source_series, first_series, partition_series = [], [], [], []
    for evaluation in evaluations:
        analysis.verify_pair(evaluation, endpoint)
        common = {key: evaluation[key] for key in ("step", "training_loss_tokens")}
        series.append({**common, **macro_measures(evaluation)})
        for source, values in evaluation["per_source"].items():
            source_series.append({**common, "source": source, "target_tokens": values["loss_tokens"],
                                  **pooled_measures(values)})
        for part in ("full", "after_prefix"):
            diagnostic = evaluation["later_article_first_block"][part]
            first_series.append({**common, "part": part,
                                 "target_tokens": diagnostic["loss_tokens"],
                                 "groups": diagnostic["group_count"],
                                 "articles": diagnostic["articles"],
                                 **pooled_measures(diagnostic)})
        partition_series.extend({**common, **row} for row in event_partitions(evaluation))
    selected_macro, endpoint_macro = macro_measures(selected), macro_measures(endpoint)
    changes = {key: endpoint_macro[key] - selected_macro[key] for key in METRICS}
    source_rows, category_tokens = [], defaultdict(int)
    mapping = endpoint["macro_source_groups"]
    for source, values in endpoint["per_source"].items():
        category_tokens[mapping[source]] += values["loss_tokens"]
    for source, ending in endpoint["per_source"].items():
        beginning = selected["per_source"][source]
        weight = ending["loss_tokens"] / category_tokens[mapping[source]] / len(category_tokens)
        a, b = pooled_measures(beginning), pooled_measures(ending)
        row = {
            "source": source, "macro_category": mapping[source],
            "groups": ending["group_count"], "target_tokens": ending["loss_tokens"],
            "weight_in_five_category_macro": weight,
        }
        for key in METRICS:
            row.update({
                "selected_" + key: a[key], "endpoint_" + key: b[key],
                "change_" + key: b[key] - a[key],
                "macro_contribution_" + key: (b[key] - a[key]) * weight,
            })
        source_rows.append(row)
    for key in METRICS:
        analysis.require(math.isclose(
            math.fsum(row["macro_contribution_" + key] for row in source_rows),
            changes[key], rel_tol=0., abs_tol=2e-15), "source contribution reconstruction failed")
    selected_families = {r["group_id"]: r for r in selected["groups"]}
    family_rows = []
    for end in endpoint["groups"]:
        start = selected_families[end["group_id"]]
        count = end["loss_tokens"]
        a = measures(*(start[key + "_loss_sum"] / count for key in ("self", "empty", "donor")))
        b = measures(*(end[key + "_loss_sum"] / count for key in ("self", "empty", "donor")))
        family_rows.append({
            "group_id": end["group_id"], "source": end["source"], "target_tokens": count,
            **{"change_" + key: b[key] - a[key] for key in METRICS},
        })
    family_counts = []
    for source in sorted(endpoint["per_source"]):
        rows = [r for r in family_rows if r["source"] == source]
        family_counts.append({
            "source": source, "families": len(rows),
            "self_nll_improved": sum(r["change_self_nll"] < 0 for r in rows),
            "donor_nll_improved": sum(r["change_donor_nll"] < 0 for r in rows),
            "donor_gap_contracted": sum(r["change_delta_donor"] < 0 for r in rows),
            "median_family_self_nll_change": statistics.median(r["change_self_nll"] for r in rows),
            "median_family_donor_nll_change": statistics.median(r["change_donor_nll"] for r in rows),
        })
    selected_partitions = {r["partition"]: r for r in event_partitions(selected)}
    endpoint_partitions = {r["partition"]: r for r in event_partitions(endpoint)}
    event_changes = [{
        "partition": key, "target_tokens": endpoint_partitions[key]["target_tokens"],
        **{"selected_" + metric: selected_partitions[key][metric] for metric in METRICS},
        **{"endpoint_" + metric: endpoint_partitions[key][metric] for metric in METRICS},
        **{"change_" + metric: endpoint_partitions[key][metric] -
           selected_partitions[key][metric] for metric in METRICS},
    } for key in selected_partitions]
    training_bins, training = training_diagnostics(args.training_log, evaluations)
    paired_path = analysis.RUNS / "m1_analysis_20261020/data/report_summary.json"
    paired_summary = analysis.load_json(paired_path)
    reused_ci = []
    for row in paired_summary["paired_contrasts"]:
        if row["comparison"] == "selected65M_minus_endpoint100M":
            reused_ci.append({
                "comparison": "endpoint100M_minus_selected65M", "metric": row["metric"],
                "difference": -row["difference"],
                "ci_lower": -row["ci_upper"], "ci_upper": -row["ci_lower"],
                "source": str(paired_path), "exploratory": True,
                "selection_corrected": False, "multiple_comparison_adjusted": False,
                "new_bootstrap_performed": False,
            })
    previous = evaluations[-2]
    analysis.require(previous["step"] == 3965, "expected the registered 95M predecessor")
    endpoint_vs_previous = next(
        row for row in analysis.paired_contrast(
            endpoint, previous, name="endpoint100M_minus_95M",
            seed=20261020, draws=10000)
        if row["metric"] == "self_nll")
    trends = trend_intervals(evaluations)
    source_mixture_ranges = {
        source: {
            "minimum_fraction": min(row["source_tokens_" + source] / row["batch_loss_tokens"]
                                    for row in training_bins),
            "maximum_fraction": max(row["source_tokens_" + source] / row["batch_loss_tokens"]
                                    for row in training_bins),
        }
        for source in endpoint["per_source"]
    } if training_bins else {}
    mathematical_audit = {
        "protocol": "m1_writer_overfitting_math_audit_20261020_v1",
        "estimand": "L_t = mean over five semantic categories of sum(self loss) / "
        "sum(target tokens); all checkpoints share the same 640 held-out families.",
        "hypothesis": {
            "null": "D = L_later - L_earlier >= 0 (no held-out self-NLL improvement)",
            "alternative": "D < 0 (held-out self-NLL improves)",
            "decision_rule": "A wholly negative two-sided paired 95% interval supports "
            "improvement at a nominal pointwise level. This is not a selection- or "
            "multiplicity-corrected confirmatory test.",
        },
        "65m_to_100m": next(row for row in reused_ci if row["metric"] == "self_nll"),
        "95m_to_100m": endpoint_vs_previous,
        "95m_to_100m_interval_contains_zero": (
            endpoint_vs_previous["ci_lower"] <= 0 <= endpoint_vs_previous["ci_upper"]),
        "six_sources_65m_to_100m_self_nll_changes": {
            row["source"]: row["change_self_nll"] for row in source_rows},
        "trend_intervals": trends,
        "training_source_token_fraction_ranges": source_mixture_ranges,
        "training_nll_available": training["available"],
        "training_loss_interpretation": "Training interval NLL is weighted by actual "
        "target tokens and accompanied by exact source-token proportions. The log "
        "does not expose source-specific training loss sums, so changing examples/"
        "lengths/difficulty cannot be removed. Do not infer overfitting from this "
        "mixed training curve or subtract it from a differently weighted validation curve.",
        "conclusions": [
            "65M-to-100M held-out self prediction improves in all six source-pooled "
            "point estimates, and the macro paired interval is wholly negative. "
            "This contradicts aggregate self-NLL deterioration as an explanation "
            "for the contracted donor gap over that interval.",
            "95M-to-100M has a favorable self-NLL point estimate but its paired "
            "interval includes zero. This sample does not establish positive marginal "
            "benefit for the final roughly 5M tokens.",
            "Average benefit per additional 5M declines in the later training "
            "intervals. This is evidence of diminishing measured returns, not a "
            "proof of a hard saturation boundary.",
            "Absence of aggregate overfitting evidence does not prove absence of "
            "subgroup regressions, guarantee all future updates help, or establish "
            "generalization beyond the fixed historical validation sample.",
        ],
        "limitations": {
            "one_training_seed": True, "fixed_donor_mapping": True,
            "posthoc_analysis": True, "multiple_comparison_adjusted": False,
            "checkpoint_selection_corrected": False,
            "newly_bootstrapped_comparisons": ["endpoint100M_minus_95M self_nll"],
            "capacity_checked_separately": True,
        },
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "macro_nll_series.csv": series,
        "source_nll_series.csv": source_series,
        "source_65m_to_100m_contributions.csv": source_rows,
        "first_block_series.csv": first_series,
        "event_partition_series.csv": partition_series,
        "event_65m_to_100m_partitions.csv": event_changes,
        "family_65m_to_100m_changes.csv": family_rows,
        "family_improvement_counts.csv": family_counts,
        "marginal_gain_intervals.csv": trends,
    }
    if training_bins:
        tables["training_intervals.csv"] = training_bins
    for name, rows in tables.items():
        analysis.write_csv(output / name, rows)
    summary = {
        "protocol": "m1_endpoint_donor_gap_diagnosis_20261020_v1",
        "difference_direction": "endpoint100M minus selected65M",
        "selected_step": selected["step"], "endpoint_step": endpoint["step"],
        "selected_macro": selected_macro, "endpoint_macro": endpoint_macro,
        "macro_change": changes, "reused_paired_confidence_intervals": reused_ci,
        "source_contributions": source_rows, "family_descriptive_counts": family_counts,
        "event_partitions": event_changes, "training": training,
        "mathematical_audit_file": "mathematical_audit.json",
        "evidence": [
            "All six source-pooled self NLL values improve from 65M to 100M.",
            "The donor-gap contraction equals donor NLL improvement minus self NLL "
            "improvement; it is not a rise in self prediction loss.",
            "Both full and after-prefix first-article-block donor gaps improve, "
            "while the full event-trajectory donor gap contracts.",
            "The full first-block diagnostic is measured before any current-article "
            "write. Remaining blocks can read memory written from the current article.",
            "More useful mismatched memory is consistent with increased reusable "
            "same-source information or a changed residual policy; output losses alone "
            "do not identify the precise representation mechanism.",
        ],
        "limitations": [
            "Descriptive decomposition, not an intervention on capacity, collision "
            "rate, residual norms, or the writer objective.",
            "The 65M-to-100M intervals are reused. One requested 95M-to-100M self-NLL "
            "follow-up is added in mathematical_audit.json. No interval is adjusted "
            "for checkpoint selection or multiple comparisons.",
            "Training NLL intervals use different batches and examples. A classic "
            "overfitting diagnosis requires worsening fixed held-out prediction; "
            "these aggregate self-NLL results instead continue to improve.",
            "First_block_full is an observed subset of news_event targets. The "
            "remaining-block quantities are algebraic differences of stored loss sums; "
            "small FP32/double reduction differences may remain.",
            "after_prefix uses a different local input convention from full and "
            "must not be subtracted from the event total.",
            "No claim about QA accuracy, pretraining novelty, sealed FINAL, or the "
            "incomplete new-event DEV64 run.",
        ],
    }
    analysis.write_json(output / "diagnosis_summary.json", summary)
    analysis.write_json(output / "mathematical_audit.json", mathematical_audit)
    ci65 = mathematical_audit["65m_to_100m"]
    ci95 = mathematical_audit["95m_to_100m"]
    text = [
        "M1 writer overfitting and diminishing-returns mathematical audit",
        "Difference D = held-out self NLL at later checkpoint minus earlier checkpoint.",
        "H0: D >= 0; alternative: D < 0. Negative values favor later training.",
        f"65M->100M: D={ci65['difference']:.10f}; paired 95% CI "
        f"[{ci65['ci_lower']:.10f}, {ci65['ci_upper']:.10f}].",
        f"95M->100M: D={ci95['difference']:.10f}; paired 95% CI "
        f"[{ci95['ci_lower']:.10f}, {ci95['ci_upper']:.10f}].",
        "The first interval supports improvement; the final-5M interval includes zero.",
        "All six source-pooled self-NLL point estimates improve from 65M to 100M.",
        "Later intervals show smaller average benefit per additional 5M tokens.",
        "This is not evidence of aggregate overfitting and is not a guarantee of future gain.",
        "Training NLL uses target-token weighting, but source-specific losses are unavailable.",
        "Different training examples prevent a clean fixed-training-probe comparison.",
        "All intervals are posthoc, single-seed, fixed-donor, and unadjusted for selection/multiplicity.",
    ]
    (output / "mathematical_audit.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    input_paths = [*paths, paired_path]
    if args.training_log.is_file():
        input_paths.append(args.training_log)
    analysis.write_json(output / "source_manifest.json", {
        "script": str(Path(__file__).resolve()), "script_sha256": analysis.sha256(Path(__file__)),
        "analysis_dependency": str(Path(analysis.__file__).resolve()),
        "analysis_dependency_sha256": analysis.sha256(Path(analysis.__file__)),
        "inputs": [{"path": str(p), "sha256": analysis.sha256(p)} for p in input_paths],
        "outputs": [{"file": name, "sha256": analysis.sha256(output / name)}
                    for name in [*tables, "diagnosis_summary.json", "mathematical_audit.json",
                                 "mathematical_audit.txt"]],
        "new_bootstrap_performed": True,
        "new_bootstrap_scope": "one requested endpoint100M_minus_95M self-NLL follow-up",
        "torch_imported": "torch" in sys.modules,
    })
    analysis.require("torch" not in sys.modules, "CPU analysis must not import torch")
    print(json.dumps({
        "output": str(output), "macro_change": changes, "training_available": training["available"],
        "event_partitions": event_changes, "source_contributions": [
            {"source": row["source"], "macro_delta_donor_change": row["macro_contribution_delta_donor"]}
            for row in source_rows],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

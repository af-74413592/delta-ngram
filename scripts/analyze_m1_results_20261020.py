"""CPU-only, reproducible analysis of frozen M1 and historical alpha results.

No model, source article, sealed FINAL dataset, or unfinished news run is read.
All confidence intervals added here are exploratory, pointwise, and unadjusted.
The selected milestone is chosen on this same validation sample; its intervals
are not corrected for checkpoint selection. Only the Python standard library is
used, including a direct import of the standard-library-only statistics helper.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
RUNS = REPOSITORY.parent / "delta-ngram-runs"
LAUNCH = RUNS / "multidomain_launch_20261019"
IDENTITY_KEYS = (
    "validation_protocol_sha256", "validation_sample_sha256",
    "donor_sample_sha256",
)
METRICS = ("delta_empty", "delta_donor", "self_nll", "donor_nll")
LIMITATIONS = [
    "One training seed (17); bootstrap intervals reflect fixed validation-family "
    "sampling, not training-seed variability.",
    "Intervals are conditional on the fixed donor mapping and do not include "
    "uncertainty from selecting a different set of donors.",
    "Historical held-out writer-training groups only; no pretraining-novelty, "
    "free-generation, question-answer accuracy, or sealed FINAL claim.",
    "Best milestone is selected using this same validation sample. Added "
    "candidate-minus-reference intervals are exploratory and are not corrected "
    "for checkpoint selection or multiple comparisons.",
    "news and news_event share one macro category. Within categories losses "
    "are pooled with target-token weights; the five categories have equal weight.",
    "A larger donor gap can reflect worse mismatched-memory predictions. "
    "Inspect self NLL and delta_empty as well as delta_donor.",
    "Saved pointwise intervals are copied from the original evaluation; newly "
    "computed paired contrasts use seed 20261020 and 10000 draws by default.",
    "The incomplete new-event DEV64 evaluation is excluded and remains incomplete. "
    "Cancelled alpha continuation/revision/pruning suites are not resumed.",
    "The selected milestone identifies the best evaluated point under the "
    "registered rule; this analysis does not establish that its model weights "
    "have been retained.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_bytes())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def write_csv(path, rows):
    require(bool(rows), f"no rows for {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_values(row):
    own, donor, empty = (row[key + "_loss_sum"]
                         for key in ("self", "donor", "empty"))
    return (empty - own, donor - own, own, donor)


def family_categories(evaluation):
    """Match raw_evaluation.macro_statistics's category/family aggregation."""
    mapping = evaluation["macro_source_groups"]
    grouped = defaultdict(dict)
    for row in evaluation["groups"]:
        category = mapping[row["source"]]
        family = row.get("family_id", row["group_id"])
        aggregate = grouped[category].setdefault(family, [0, 0., 0., 0., 0.])
        aggregate[0] += row["loss_tokens"]
        for index, value in enumerate(metric_values(row), start=1):
            aggregate[index] += value
    # The original helper keeps family insertion order from its input records.
    return {name: values for name, values in sorted(grouped.items())}


def macro_points(evaluation):
    categories = family_categories(evaluation)
    return {
        metric: math.fsum(
            math.fsum(row[index] for row in families.values()) /
            sum(row[0] for row in families.values())
            for families in categories.values()
        ) / len(categories)
        for index, metric in enumerate(METRICS, start=1)
    }


def verify_saved_points(evaluation):
    points = macro_points(evaluation)
    for metric in ("delta_empty", "delta_donor"):
        require(math.isclose(points[metric], evaluation["source_macro_" + metric],
                             rel_tol=0., abs_tol=1e-12),
                f"saved macro {metric} does not reproduce")
    return points


def verify_pair(candidate, reference):
    for key in IDENTITY_KEYS:
        require(candidate[key] == reference[key], f"unpaired {key}")
    for key in ("macro_source_groups", "validation_protocol", "donor_sample"):
        require(candidate[key] == reference[key], f"unpaired {key}")
    def indexed(evaluation):
        rows = evaluation["groups"]
        result = {row["group_id"]: row for row in rows}
        require(len(result) == len(rows), "nonunique group records")
        return result
    left, right = indexed(candidate), indexed(reference)
    require(left.keys() == right.keys(), "different group IDs")
    identity = []
    for group_id in sorted(left):
        a, b = left[group_id], right[group_id]
        for key in ("source", "family_id", "document_ids", "windows", "loss_tokens"):
            require(a.get(key) == b.get(key), f"unpaired {group_id}: {key}")
        identity.append({key: a.get(key) for key in (
            "group_id", "source", "family_id", "document_ids", "windows", "loss_tokens")})
    maximum_empty_difference = max(abs(
        left[key]["empty_loss_sum"] - right[key]["empty_loss_sum"]) for key in left)
    return {
        "paired_groups": len(left), "pairing_identity_sha256": json_hash(identity),
        "validation_and_donor_identities_equal": True,
        "empty_loss_sums_exact_equal": maximum_empty_difference == 0.,
        "max_abs_empty_loss_sum_difference": maximum_empty_difference,
    }


def percentile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] -
                            values[lower]) * (position - lower)


def paired_contrast(candidate, reference, *, name, seed, draws):
    verification = verify_pair(candidate, reference)
    left, right = family_categories(candidate), family_categories(reference)
    require(left.keys() == right.keys(), "different macro categories")
    strata = []
    for category in left:
        require(left[category].keys() == right[category].keys(),
                f"unpaired families: {category}")
        rows = []
        for family, a in left[category].items():
            b = right[category][family]
            require(a[0] == b[0], "different paired family token counts")
            rows.append((a[0], *(a[i] - b[i] for i in range(1, 5))))
        strata.append(rows)
    def statistic(samples):
        ratios = []
        for rows in samples:
            denominator = sum(row[0] for row in rows)
            ratios.append([math.fsum(row[i] for row in rows) / denominator
                           for i in range(1, 5)])
        return [math.fsum(row[i] for row in ratios) / len(ratios)
                for i in range(4)]
    point = statistic(strata)
    expected_left, expected_right = macro_points(candidate), macro_points(reference)
    for metric, value in zip(METRICS, point):
        require(math.isclose(value, expected_left[metric] - expected_right[metric],
                             rel_tol=0., abs_tol=1e-12), "contrast algebra mismatch")
    rng = random.Random(seed)
    distribution = [[] for _ in METRICS]
    for _ in range(draws):
        # Family samples are shared across both checkpoints and all four metrics.
        samples = [[rows[rng.randrange(len(rows))] for _ in rows] for rows in strata]
        for index, value in enumerate(statistic(samples)):
            distribution[index].append(value)
    result = []
    for index, metric in enumerate(METRICS):
        result.append({
            "comparison": name, "metric": metric, "difference": point[index],
            "ci_lower": percentile(distribution[index], .025),
            "ci_upper": percentile(distribution[index], .975),
            "difference_definition": "candidate minus reference",
            "improvement_direction": "positive" if metric.startswith("delta") else "negative",
            "bootstrap_seed": seed, "bootstrap_draws": draws, "confidence": .95,
            "exploratory": True, "multiple_comparison_adjusted": False,
            "selection_corrected": False, **verification,
        })
    return result


def original_ci(evaluation, control):
    return evaluation["source_macro_" + control + "_bootstrap"]


def summary_row(evaluation, *, label, path, tokens=None):
    points = verify_saved_points(evaluation)
    row = {
        "label": label, "file": str(path),
        "step": evaluation.get("step"),
        "training_loss_tokens": tokens if tokens is not None else evaluation.get("training_loss_tokens"),
        "target_loss_tokens": evaluation["loss_tokens"],
        "group_count": evaluation["group_count"],
        "macro_category_count": len(evaluation["macro_categories"]),
        "macro_delta_empty": points["delta_empty"],
        "macro_delta_donor": points["delta_donor"],
        "macro_self_nll": points["self_nll"],
        "macro_donor_nll": points["donor_nll"],
        "pooled_self_nll": evaluation["self_nll"],
        "pooled_empty_nll": evaluation["empty_nll"],
        "pooled_donor_nll": evaluation["donor_nll"],
        "pooled_delta_empty": evaluation["delta_empty"],
        "pooled_delta_donor": evaluation["delta_donor"],
        "checkpoint_selection_eligible": evaluation["checkpoint_selection_eligible"],
        **{key: evaluation[key] for key in IDENTITY_KEYS},
    }
    for control in ("empty", "donor"):
        ci = original_ci(evaluation, control)
        row.update({f"macro_delta_{control}_ci_lower": ci["ci_lower"],
                    f"macro_delta_{control}_ci_upper": ci["ci_upper"],
                    f"macro_delta_{control}_ci_seed": ci["seed"],
                    f"macro_delta_{control}_ci_draws": ci["replicates"]})
    return row


def source_rows(evaluation, label):
    category_tokens = defaultdict(int)
    for source, values in evaluation["per_source"].items():
        category_tokens[evaluation["macro_source_groups"][source]] += values["loss_tokens"]
    rows = []
    for source, values in sorted(evaluation["per_source"].items()):
        category = evaluation["macro_source_groups"][source]
        rows.append({
            "checkpoint": label, "source": source, "macro_category": category,
            "macro_category_weight": 1. / len(category_tokens),
            "source_token_share_within_category": values["loss_tokens"] / category_tokens[category],
            "groups": values["group_count"], "target_tokens": values["loss_tokens"],
            **{key: values[key] for key in (
                "self_nll", "empty_nll", "donor_nll", "delta_empty", "delta_donor")},
            "delta_empty_ci_lower": values["paired_group_bootstrap"]["ci_lower"],
            "delta_empty_ci_upper": values["paired_group_bootstrap"]["ci_upper"],
            "delta_donor_ci_lower": values["donor_paired_group_bootstrap"]["ci_lower"],
            "delta_donor_ci_upper": values["donor_paired_group_bootstrap"]["ci_upper"],
        })
    return rows


def first_block_rows(evaluation, label):
    result = []
    diagnostic = evaluation.get("later_article_first_block")
    if diagnostic is None:
        return result
    for part in ("full", "after_prefix"):
        item = diagnostic[part]
        result.append({
            "checkpoint": label, "part": part, "groups": item["group_count"],
            "articles": item["articles"], "target_tokens": item["loss_tokens"],
            **{key: item[key] for key in (
                "self_nll", "empty_nll", "donor_nll", "delta_empty", "delta_donor")},
            "delta_empty_ci_lower": item["paired_group_bootstrap"]["ci_lower"],
            "delta_empty_ci_upper": item["paired_group_bootstrap"]["ci_upper"],
            "delta_donor_ci_lower": item["donor_paired_group_bootstrap"]["ci_lower"],
            "delta_donor_ci_upper": item["donor_paired_group_bootstrap"]["ci_upper"],
            "current_article_memory_writes": diagnostic["protocol"]["current_article_memory_writes"],
            "checkpoint_selection_uses_this_diagnostic": False,
        })
    return result


def verify_original_helper(evaluation):
    helper = REPOSITORY / "src/delta_ngram/raw_evaluation.py"
    spec = importlib.util.spec_from_file_location("m1_cpu_statistics", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = original_ci(evaluation, "empty")
    reproduced = module.macro_statistics(
        evaluation["groups"], evaluation["macro_source_groups"],
        bootstrap_seed=settings["seed"], bootstrap_samples=settings["replicates"],
        donor=True,
    )
    for condition in ("empty", "donor"):
        for key in ("ci_lower", "ci_upper"):
            require(math.isclose(
                reproduced[f"source_macro_{condition}_bootstrap"][key],
                evaluation[f"source_macro_{condition}_bootstrap"][key],
                rel_tol=0., abs_tol=1e-12), f"original {condition} CI mismatch")
    require(reproduced["checkpoint_selection_eligible"] ==
            evaluation["checkpoint_selection_eligible"], "eligibility mismatch")
    return {"point_and_original_ci_reproduced": True, "helper_file": str(helper),
            "helper_sha256": sha256(helper), "absolute_tolerance": 1e-12}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-dir", type=Path, default=LAUNCH / "final_backup_20261020")
    parser.add_argument("--alpha-dir", type=Path, default=LAUNCH / "alpha_evaluations")
    parser.add_argument("--output-dir", type=Path, default=RUNS / "m1_analysis_20261020/data")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--allow-missing-alpha", action="store_true")
    args = parser.parse_args(argv)
    require(args.draws > 0, "bootstrap draws must be positive")
    sources = []
    def read(path, role):
        sources.append({"path": str(path.resolve()), "role": role,
                        "bytes": path.stat().st_size, "sha256": sha256(path)})
        return load_json(path)
    evaluation_paths = sorted(args.m1_dir.glob("eval_step_*.json"))
    require(len(evaluation_paths) == 21, "expected all 21 registered M1 evaluations")
    evaluations = [(path, read(path, "m1_evaluation")) for path in evaluation_paths]
    evaluations.sort(key=lambda item: item[1]["training_loss_tokens"])
    run_summary = read(args.m1_dir / "summary.json", "m1_training_summary")
    endpoint_path, endpoint = evaluations[-1]
    require(endpoint["budget_endpoint"] is True and endpoint["is_final"] is True,
            "last M1 evaluation is not the budget endpoint")
    require(run_summary["counters"]["loss_tokens"] == endpoint["training_loss_tokens"],
            "summary and endpoint token counts differ")
    base_path, base = next(item for item in evaluations if item[1]["step"] == 210)
    eligible = [item for item in evaluations if item[1]["checkpoint_selection_eligible"]]
    require(bool(eligible), "no checkpoint qualifies for the registered rule")
    selected_path, selected = max(
        eligible, key=lambda item: (item[1]["source_macro_delta_donor"],
                                   -item[1]["training_loss_tokens"]))
    series, checks = [], []
    for path, evaluation in evaluations:
        verify_pair(evaluation, base)
        series.append(summary_row(evaluation, label=f"m1_step_{evaluation['step']:06d}", path=path))
    for evaluation in (base, selected, endpoint):
        checks.append({"step": evaluation["step"], **verify_original_helper(evaluation)})
    alpha_inputs = []
    for path in sorted(args.alpha_dir.glob("*_historical.json")):
        candidate = load_json(path)
        require(candidate.get("suite") == "historical",
                f"expected a historical-only alpha evaluation: {path}")
        require(candidate.get("complete") is True, f"incomplete alpha: {path}")
        candidate = read(path, "alpha_historical_evaluation")
        prefix = candidate["m1_prefix"]
        require(prefix["exact_complete_prefix_verified"] is True
                and prefix["actual_loss_tokens"] == base["training_loss_tokens"]
                and prefix["stop_step"] == base["step"],
                "alpha did not complete the same M1 first-5M prefix")
        evaluation = candidate["results"]["historical"]
        verify_pair(evaluation, base)
        alpha_inputs.append((path, candidate, evaluation))
    require(args.allow_missing_alpha or len(alpha_inputs) == 4,
            f"expected four historical alpha files, found {len(alpha_inputs)}")
    variants = [item[1]["checkpoint"]["variant"] for item in alpha_inputs]
    require(len(variants) == len(set(variants)), "duplicate alpha variants")
    alpha_prefix_identities = [
        {key: wrapper["m1_prefix"][key] for key in (
            "manifest_sha256", "m1_checkpoint_sha256", "writer_init_sha256",
            "stop_step", "actual_loss_tokens", "nominal_loss_tokens")}
        for _, wrapper, _ in alpha_inputs
    ]
    require(len({json_hash(item) for item in alpha_prefix_identities}) <= 1,
            "alpha evaluations do not share the same prefix/init identity")
    alpha_rows = [summary_row(base, label="D1_first5M", path=base_path)]
    contrasts = paired_contrast(
        selected, endpoint, name="selected65M_minus_endpoint100M",
        seed=args.seed, draws=args.draws)
    first_blocks = []
    for label, evaluation in (("D1_first5M", base), ("selected65M", selected), ("endpoint100M", endpoint)):
        first_blocks.extend(first_block_rows(evaluation, label))
    for path, wrapper, evaluation in alpha_inputs:
        variant = wrapper["checkpoint"]["variant"]
        row = summary_row(evaluation, label=variant, path=path,
                          tokens=wrapper["m1_prefix"]["actual_loss_tokens"])
        row.update(step=wrapper["checkpoint"]["step"],
                   exact_complete_prefix_verified=True,
                   prefix_manifest_sha256=wrapper["m1_prefix"]["manifest_sha256"])
        alpha_rows.append(row)
        first_blocks.extend(first_block_rows(evaluation, variant))
        print(json.dumps({"stage": "paired_bootstrap", "variant": variant}), flush=True)
        contrasts.extend(paired_contrast(evaluation, base,
                         name=f"{variant}_minus_D1_first5M", seed=args.seed, draws=args.draws))
    endpoint_sources = source_rows(endpoint, "endpoint100M")
    selected_sources = source_rows(selected, "selected65M")
    categories = family_categories(endpoint)
    summary = {
        "protocol": "m1_results_numeric_analysis_20261020_v1",
        "scope": "M1 historical validation and four same-prefix historical alpha evaluations",
        "analysis_seed": args.seed, "analysis_bootstrap_draws": args.draws,
        "m1_evaluations": len(series), "alpha_evaluations": len(alpha_inputs),
        "run_endpoint": {
            "complete_flag": run_summary["complete"], "budget_endpoint": run_summary["budget_endpoint"],
            "stop_reason": run_summary["stop_reason"], "step": endpoint["step"],
            "training_loss_tokens": endpoint["training_loss_tokens"],
            "requested_loss_tokens": run_summary["requested_loss_tokens"],
            "unspent_loss_tokens": run_summary["unspent_loss_tokens"],
        },
        "best_by_registered_gate": summary_row(
            selected, label="selected65M", path=selected_path),
        "checkpoint_selection_rule": selected["checkpoint_selection_rule"],
        "endpoint": summary_row(endpoint, label="endpoint100M", path=endpoint_path),
        "source_counts": endpoint_sources,
        "macro_category_counts": {
            name: {"families": len(rows),
                   "target_tokens": sum(row[0] for row in rows.values()),
                   "macro_weight": 1. / len(categories)}
            for name, rows in categories.items()
        },
        "macro_definition": "Map news_event to news; sum family-cluster losses "
        "and target counts within each category; form category loss/count ratios; "
        "take the arithmetic mean of five category ratios.",
        "validation_identity": {key: base[key] for key in IDENTITY_KEYS},
        "all_21_m1_sample_protocol_donor_identities_equal": True,
        "alpha_prefix_identity": alpha_prefix_identities[0] if alpha_prefix_identities else None,
        "original_statistics_reproduction": checks,
        "alpha_first5M": alpha_rows, "first_article_block": first_blocks,
        "paired_contrasts": contrasts, "limitations": LIMITATIONS,
        "first_block_protocol": endpoint["later_article_first_block"]["protocol"],
        "gates": {"p1_gate_status": endpoint["p1_gate_status"],
                  "p6_gate_evaluated": endpoint["p6_gate_evaluated"]},
    }
    require("torch" not in sys.modules, "analysis must not import torch")
    for source in sources:
        require(sha256(Path(source["path"])) == source["sha256"],
                f"input changed during analysis: {source['path']}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "m1_series.csv": series, "endpoint_per_source.csv": endpoint_sources,
        "selected65M_per_source.csv": selected_sources,
        "first_article_block.csv": first_blocks,
        "alpha_first5M_comparison.csv": alpha_rows,
        "paired_contrasts.csv": contrasts,
    }
    for filename, rows in tables.items():
        write_csv(output / filename, rows)
    write_json(output / "report_summary.json", summary)
    manifest = {
        "analysis_script": {"path": str(Path(__file__).resolve()),
                            "sha256": sha256(Path(__file__))},
        "statistics_helper": checks[0]["helper_file"],
        "statistics_helper_sha256": checks[0]["helper_sha256"],
        "python_version": sys.version, "seed": args.seed, "draws": args.draws,
        "source_files": sources,
        "outputs": [{"file": name, "sha256": sha256(output / name),
                     "bytes": (output / name).stat().st_size}
                    for name in sorted([*tables, "report_summary.json"])],
        "model_imported": False, "source_articles_read": False,
        "sealed_final_accessed": False,
    }
    write_json(output / "source_manifest.json", manifest)
    print(json.dumps({
        "output": str(output), "m1_points": len(series), "alpha_variants": variants,
        "selected_step": selected["step"], "endpoint_step": endpoint["step"],
        "contrast_rows": len(contrasts), "checks": "passed",
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

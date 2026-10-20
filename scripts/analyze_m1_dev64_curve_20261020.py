"""CPU-only analysis of the fixed M1 0M..100M DEV64 continuation curve.

Requires all 21 completed results by default. --allow-partial writes explicitly
labelled preview outputs in separate directories. Checkpoint roles are fixed:
100M primary, 65M secondary; this script never selects a new best checkpoint.
No model, GPU, source-article body, training code, or FINAL dataset is opened.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO.parent / "delta-ngram-runs"
OPS = RUNS / "multidomain_launch_20261019"
DISTANCES = (0, 1, 4, 8, 16)
ARMS = ("self", "empty", "donor", "self_perm", "donor_perm")
CONTRASTS = ("delta_empty", "delta_donor", "direction_delta_self",
             "direction_delta_donor", "direction_difference",
             "retention_change_empty", "retention_change_donor")
COMPARISON_METRICS = ("delta_empty", "delta_donor", "self_nll", "donor_nll")
PREFIX_GROUP_COUNTS = {"restated": 34, "not_restated": 30, "unknown": 0}
EXPECTED_REGISTRATION_SHA256 = "1bad9562b3fb1ca605437859a5d65cea8c9daa7855bac822bab84355edeb4e9f"
READER_COMMIT = "c1899de289a04d12100db370d81485cdf75e47ca"
EVALUATOR_COMMIT = "167d24783161be7842bbb9aec52f2c7804191ba9"
LIMITATIONS = [
    "DEV64 only; no sealed FINAL result, QA accuracy, or free-generation claim.",
    "Equal-family mean: average target-token NLL within each family, then weight 64 families equally. This differs from the historical five-source macro.",
    "100M is primary and 65M secondary by prior user registration. The curve does not select a replacement primary checkpoint.",
    "One writer-training seed (17). Family bootstrap does not measure training-seed variability.",
    "Pointwise 95% intervals are not adjusted across 21 checkpoints, five distances, or multiple contrasts.",
    "Added checkpoint-paired contrasts are exploratory and unadjusted; direction-control seeds are averaged within each family, not treated as independent observations.",
    "Both self NLL and donor NLL must be inspected: a larger donor gap can also result from worse donor predictions.",
    "Source archives retrieved in 2026 do not certify unchanged 2025 bodies; base pretraining novelty is unknown.",
    "Sum of per-worker evaluation time is not parallel wall time or a billing measurement.",
]


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def close(a, b, message, tolerance=2e-11):
    require(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance, message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def role(nominal):
    return "primary" if nominal == 100000000 else "secondary" if nominal == 65000000 else "curve_only"


def bootstrap_weights(families, seed, samples):
    """Match the evaluator's Python RNG draws; couple distances and contrasts."""
    rng = random.Random(seed)
    weights = np.zeros((samples, families), dtype=np.float64)
    for draw in range(samples):
        counts = np.bincount([rng.randrange(families) for _ in range(families)], minlength=families)
        weights[draw] = counts / families
    return weights


def bootstrap_ci(weights, values):
    draws = weights @ np.asarray(values, dtype=np.float64)
    return np.quantile(draws, [0.025, 0.975], axis=0, method="linear")


def indexed_records(result):
    records = result["per_family"]
    indexed = {(r["event_family"], r["unrelated_article_count"]): r for r in records}
    families = sorted({r["event_family"] for r in records})
    require(len(families) == 64 and len(records) == len(indexed) == 320, "Expected 64 unique families × five distances")
    require(set(indexed) == {(f, d) for f in families for d in DISTANCES}, "Family-distance coverage differs")
    return families, indexed


def record_identity(row):
    keys = ("pair_id", "event_family", "domain", "later_source_id", "unrelated_article_count",
            "prefix_restatement_label", "revision_eligible", "revision_type")
    arm_keys = ("target_tokens", "first_target_tokens", "read_windows", "reader_input_tokens",
                "actual_written_articles", "actual_written_tokens", "actual_written_blocks",
                "address_prefix_tokens", "write_source_ids")
    return {**{k: row.get(k) for k in keys},
            "conditions": {arm: {k: row["conditions"][arm].get(k) for k in arm_keys} for arm in ARMS}}


def validate_result(path, job, reference=None):
    result = load(path)
    require(result.get("protocol") == "delta_ngram_recent_news_continuation_eval_v1", "Unexpected evaluation protocol")
    require(result.get("complete") is True and result.get("evaluation_only") is True
            and result.get("split") == "dev", f"Incomplete or non-DEV result: {path.name}")
    require(result["optimizer_steps"] == result["backward_calls"] == 0, "Evaluation performed optimization")
    require(result["independent_families"] == 64 and result["conditions"] == list(ARMS), "Evaluation conditions differ")
    cp = result["checkpoint"]
    for key in ("sha256", "step", "checkpoint"):
        actual = cp["path"] if key == "checkpoint" else cp[key]
        require(actual == job[key], f"{job['label']}: checkpoint {key} mismatch")
    require(cp["training_loss_tokens"] == job["checkpoint_training_loss_tokens"], "Checkpoint token metadata differs")
    require(result["writer_tensor_sha256"] == cp["writer_tensor_sha256"] == job["writer_tensor_sha256"],
            "Writer tensor hash differs from the CPU-audited checkpoint inventory")
    manifest = result["manifest"]
    require(canonical_sha(manifest) == result["manifest_sha256"], "Result manifest hash is invalid")
    require(manifest["distances"] == list(DISTANCES) and manifest["independent_families"] == 64, "Manifest cohort differs")
    require(manifest["write_block_tokens"] == 512 and manifest["initial_prefix_tokens"] == 128, "Scoring protocol changed")
    require(manifest["bootstrap_seed"] == 20261019 and manifest["bootstrap_samples"] == 10000,
            "Manifest bootstrap settings changed")
    require([p["seed"] for p in result["direction_control"]["plans"]] == [17, 29, 43]
            and result["direction_control"]["seed_selection_allowed"] is False,
            "Registered direction controls changed")
    require(result["bootstrap"] == {
        "unit": "event_family", "paired_across_conditions_and_distances": True,
        "seed": 20261019, "replicates": 10000, "confidence": .95,
    }, "Original bootstrap settings changed")
    require(canonical_sha(result["model"]) == result["model_identity_sha256"], "Model identity hash invalid")
    require(result["model"]["revision"] == result["model"]["resolved_commit"] == READER_COMMIT
            and result["model"]["matches_registry_reader"] is True, "Reader identity differs")
    for when in ("git_before", "git_after"):
        require(result["run"][when] == {"commit": EVALUATOR_COMMIT, "dirty": False}, "Evaluation source changed")
    require(result["run"]["threads"] == 8 and result["run"]["deterministic_algorithms"] is True,
            "Evaluation execution settings differ")
    checks = result["integrity_checks"]
    for key in ("input_files_unchanged", "writer_state_dict_exact_equal", "parameters_frozen_no_gradients",
                "target_never_written", "memory_unchanged_during_queries", "actual_write_budgets_verified"):
        require(checks.get(key) is True, "Failed runtime integrity check: " + key)
    families, indexed = indexed_records(result)
    by_pair = {p["id"]: p for p in manifest["pairs"]}
    require(len(by_pair) == 64, "Manifest pair identities are not unique")
    for (family, distance), row in indexed.items():
        require(set(row["conditions"]) == set(ARMS), "Per-family conditions differ")
        pair = by_pair[row["pair_id"]]
        require(pair["event_family"] == family and pair["later_source_id"] == row["later_source_id"],
                "Scored family/target identity differs from the frozen manifest")
        targets = row["conditions"]["self"]["target_tokens"]
        require(type(targets) is int and targets > 0, "Invalid target count")
        require(targets == pair["later_token_count"] - 128 and pair["domain"] == row["domain"],
                "Scored target count/domain differs from the frozen pair")
        for arm in ARMS:
            value = row["conditions"][arm]
            require(value["target_tokens"] == targets, "Targets differ across conditions")
            close(value["nll"], value["loss_sum"] / targets, "NLL does not match loss/token sums")
            if arm.endswith("_perm"):
                require(set(value["per_seed"]) == {"17", "29", "43"}, "Direction-control seeds differ")
                for seed in value["per_seed"].values():
                    require(seed["target_tokens"] == targets, "Direction-control target count differs")
                close(value["nll"], math.fsum(x["nll"] for x in value["per_seed"].values()) / 3,
                      "Direction seeds were not averaged within family")
        c = row["conditions"]
        for arm in ("empty", "donor"):
            close(row["delta_" + arm], c[arm]["nll"] - c["self"]["nll"], "Delta sign or value changed")
        for arm in ("self", "donor"):
            close(row["direction_delta_" + arm], c[arm + "_perm"]["nll"] - c[arm]["nll"], "Direction contrast changed")
        close(row["direction_difference"], row["direction_delta_self"] - row["direction_delta_donor"],
              "Direction-difference value changed")
        for key in ("nll", "loss_sum", "target_tokens"):
            require(c["empty"][key] == indexed[family, 0]["conditions"]["empty"][key],
                    "Empty baseline differs between interference distances")
    require(sum(indexed[f, 0]["conditions"]["self"]["target_tokens"] for f in families)
            == manifest["target_tokens_per_distance"], "Manifest target-token total differs")
    if reference is not None:
        require(result["manifest_sha256"] == reference["manifest_sha256"]
                and manifest == reference["manifest"], "Frozen input manifest changed across checkpoints")
        require(result["model_identity_sha256"] == reference["model_identity_sha256"], "Reader identity changed across checkpoints")
        rf, ri = indexed_records(reference)
        require(families == rf, "Event families differ across checkpoints")
        for key, row in indexed.items():
            require(record_identity(row) == record_identity(ri[key]), "Family, target, prefix or write identities differ")
            for field in ("nll", "loss_sum", "target_tokens"):
                require(row["conditions"]["empty"][field] == ri[key]["conditions"]["empty"][field],
                        "Empty baseline must match exactly across checkpoints: " + field)
    return result, families, indexed


def summarize_point(job, result, families, indexed, weights, preview, *, prefix_group=None):
    saved_summary = result if prefix_group is None else result["prefix_restatement_groups"][prefix_group]
    saved = {r["unrelated_article_count"]: r for r in saved_summary["retention_curve"]}
    require(set(saved) == set(DISTANCES) and len(saved_summary["retention_curve"]) == 5, "Invalid saved retention curve")
    family_count = len(families)
    require(family_count > 0 and weights.shape[1] == family_count, "Invalid summary family count")
    values = np.asarray([[indexed[f, d][key] if not key.startswith("retention_change_")
                          else indexed[f, d]["delta_" + key.removeprefix("retention_change_")]
                          - indexed[f, 0]["delta_" + key.removeprefix("retention_change_")]
                          for d in DISTANCES for key in CONTRASTS] for f in families])
    ci = bootstrap_ci(weights, values)
    output = []
    for di, distance in enumerate(DISTANCES):
        rows = [indexed[f, distance] for f in families]
        old = saved[distance]
        row = {"checkpoint": job["label"], "role": role(job["nominal_loss_tokens"]),
               "nominal_loss_tokens": job["nominal_loss_tokens"], "training_loss_tokens": job["training_loss_tokens"],
               "step": job["step"], "unrelated_article_count": distance, "families": family_count,
               "target_tokens": sum(r["conditions"]["self"]["target_tokens"] for r in rows),
               "preview_only": preview}
        if prefix_group is not None:
            row.update(prefix_restatement_label=prefix_group,
                       bootstrap_seed=saved_summary["bootstrap"]["seed"],
                       bootstrap_samples=saved_summary["bootstrap"]["replicates"])
        require(old["families"] == family_count and old["target_tokens"] == row["target_tokens"],
                "Saved retention summary has different coverage")
        for arm in ARMS:
            value = math.fsum(r["conditions"][arm]["nll"] for r in rows) / family_count
            close(value, old[arm + "_nll"], "Saved family mean differs")
            row[arm + "_nll"] = value
            row[arm + "_pooled_token_nll"] = math.fsum(r["conditions"][arm]["loss_sum"] for r in rows) / row["target_tokens"]
            row[arm + "_first_target_nll"] = math.fsum(r["conditions"][arm]["first_target_nll"] for r in rows) / family_count
        for ki, key in enumerate(CONTRASTS):
            column = di * len(CONTRASTS) + ki
            value = math.fsum(values[:, column]) / family_count
            close(value, old[key], "Saved contrast mean differs: " + key)
            require(len(old[key + "_ci95"]) == 2, "Invalid saved interval")
            for side in range(2):
                close(float(ci[side, column]), old[key + "_ci95"][side], "Saved paired interval differs: " + key)
            row[key] = value
            row[key + "_ci_lower"], row[key + "_ci_upper"] = old[key + "_ci95"]
        output.append(row)
    return output


def summarize_prefix_groups(job, result, families, indexed, weights_by_group, preview):
    """Export only the frozen annotation groups and their original intervals."""
    saved_groups = result["prefix_restatement_groups"]
    require(set(saved_groups) == set(PREFIX_GROUP_COUNTS), "Frozen prefix groups differ")
    require(all(indexed[f, d]["prefix_restatement_label"] in PREFIX_GROUP_COUNTS
                for f in families for d in DISTANCES), "Unknown prefix annotation label")
    output = []
    for label, expected_count in PREFIX_GROUP_COUNTS.items():
        selected = [f for f in families if indexed[f, 0]["prefix_restatement_label"] == label]
        require(len(selected) == saved_groups[label]["families"] == expected_count,
                "Frozen prefix group size differs: " + label)
        require(all(indexed[f, d]["prefix_restatement_label"] == label
                    for f in selected for d in DISTANCES), "Prefix group changes across distances")
        if not selected:
            require("retention_curve" not in saved_groups[label], "Empty prefix group has estimates")
            continue
        require(saved_groups[label]["bootstrap"] == result["bootstrap"],
                "Original prefix-group bootstrap settings changed")
        output.extend(summarize_point(job, result, selected, indexed,
                                      weights_by_group[label], preview, prefix_group=label))
    return output


def paired_comparison(candidate, reference, loaded, jobs, weights, preview, seed, samples):
    if candidate not in loaded or reference not in loaded:
        return []
    _, families, ci = loaded[candidate]
    _, other_families, ri = loaded[reference]
    require(families == other_families, "Paired comparison families differ")
    def value(row, key):
        return row["conditions"][key.removesuffix("_nll")]["nll"] if key.endswith("_nll") else row[key]
    matrix = np.asarray([[value(ci[f, d], key) - value(ri[f, d], key)
                          for d in DISTANCES for key in COMPARISON_METRICS] for f in families])
    intervals = bootstrap_ci(weights, matrix)
    rows = []
    for di, distance in enumerate(DISTANCES):
        for ki, key in enumerate(COMPARISON_METRICS):
            col = di * len(COMPARISON_METRICS) + ki
            rows.append({
                "comparison": f"{jobs[candidate]['label']}_minus_{jobs[reference]['label']}",
                "candidate": jobs[candidate]["label"], "reference": jobs[reference]["label"],
                "unrelated_article_count": distance, "metric": key,
                "estimate": math.fsum(matrix[:, col]) / 64,
                "ci_lower": float(intervals[0, col]), "ci_upper": float(intervals[1, col]),
                "families": 64, "bootstrap_seed": seed, "bootstrap_samples": samples,
                "interpretation": "negative improves NLL" if key.endswith("_nll") else "positive improves the control-minus-self gap",
                "inference": "exploratory_pointwise_unadjusted", "preview_only": preview,
            })
    return rows


def plot_figures(rows, directory, preview):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "figure.dpi": 120})
    suffix = " — PARTIAL PREVIEW" if preview else ""
    prefix = "dev64_preview_" if preview else "dev64_"
    colors = dict(zip(DISTANCES, ("#264653", "#287a8f", "#2a9d8f", "#e9a348", "#c85645")))
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, key in zip(axes, ("delta_empty", "delta_donor")):
        for d in DISTANCES:
            selected = sorted((r for r in rows if r["unrelated_article_count"] == d), key=lambda x: x["training_loss_tokens"])
            x = [r["training_loss_tokens"] / 1e6 for r in selected]
            ax.plot(x, [r[key] for r in selected], ".-", label=f"{d} unrelated articles", color=colors[d], linewidth=1.5)
            ax.fill_between(x, [r[key + "_ci_lower"] for r in selected],
                            [r[key + "_ci_upper"] for r in selected], color=colors[d], alpha=.08)
        ax.axhline(0, color="#777777", linewidth=.7)
        ax.set(xlabel="Actual writer-training target tokens (millions)", ylabel="NLL improvement (nats/token)",
               title="Empty − self" if key == "delta_empty" else "Donor − self")
        ax.grid(alpha=.18)
    axes[1].legend(fontsize=8)
    fig.suptitle("DEV64: equal-family training curve" + suffix)
    path = directory / (prefix + "training_curves.png")
    fig.savefig(path)
    figures.append(str(path))
    plt.close(fig)
    primary = sorted((r for r in rows if r["role"] == "primary"), key=lambda x: x["unrelated_article_count"])
    if primary:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
        for ax, retained in zip(axes, (False, True)):
            for control, color in (("empty", "#287a8f"), ("donor", "#c85645")):
                key = ("retention_change_" if retained else "delta_") + control
                ax.plot(DISTANCES, [r[key] for r in primary], "o-", color=color, label=control)
                ax.fill_between(DISTANCES, [r[key + "_ci_lower"] for r in primary],
                                [r[key + "_ci_upper"] for r in primary], color=color, alpha=.15)
            ax.axhline(0, color="#777777", linewidth=.7)
            ax.set(xlabel="Unrelated replay articles", xticks=DISTANCES,
                   ylabel="NLL difference (nats/token)",
                   title="Change relative to distance 0" if retained else "Control − self")
            ax.grid(alpha=.18)
            ax.legend()
        fig.suptitle("100M primary: DEV64 retention with paired 95% intervals" + suffix)
        path = directory / (prefix + "100m_retention.png")
        fig.savefig(path)
        figures.append(str(path))
        plt.close(fig)
    selected_points = [(0, "0M initialization", "#777777"), (65000000, "65M secondary", "#e9a348"),
                       (100000000, "100M primary", "#287a8f")]
    if any(r["nominal_loss_tokens"] == 0 for r in rows):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
        for nominal, label, color in selected_points:
            values = sorted((r for r in rows if r["nominal_loss_tokens"] == nominal), key=lambda x: x["unrelated_article_count"])
            if not values:
                continue
            for ax, key in zip(axes, ("delta_empty", "delta_donor")):
                ax.plot(DISTANCES, [r[key] for r in values], "o-", label=label, color=color)
                ax.fill_between(DISTANCES, [r[key + "_ci_lower"] for r in values],
                                [r[key + "_ci_upper"] for r in values], alpha=.1, color=color)
        for ax, title in zip(axes, ("Empty − self", "Donor − self")):
            ax.axhline(0, color="#777777", linewidth=.7)
            ax.set(xlabel="Unrelated replay articles", ylabel="NLL improvement (nats/token)", xticks=DISTANCES, title=title)
            ax.grid(alpha=.18)
            ax.legend()
        fig.suptitle("DEV64: fixed checkpoints versus the actual 0M writer" + suffix)
        path = directory / (prefix + "0m_reference.png")
        fig.savefig(path)
        figures.append(str(path))
        plt.close(fig)
    return figures


def runtime_summary(rawdir, loaded, jobs, preview):
    rows, starts, finishes = [], [], []
    for nominal, (result, _, _) in sorted(loaded.items()):
        run = result["run"]
        elapsed = run["elapsed_seconds"]
        require(math.isfinite(elapsed) and elapsed > 0, "Invalid evaluation elapsed time")
        end = datetime.fromisoformat(run["created_utc"])
        starts.append(end - timedelta(seconds=elapsed))
        finishes.append(end)
        rows.append({"checkpoint": jobs[nominal]["label"], "role": role(nominal),
                     "training_loss_tokens": jobs[nominal]["training_loss_tokens"],
                     "elapsed_seconds": elapsed, "completed_at_utc": run["created_utc"],
                     "threads": run["threads"], "device": run["device"], "torch_version": run["torch_version"],
                     "preview_only": preview})
    summary = {"completed_evaluations": len(rows), "sum_worker_evaluation_seconds": math.fsum(r["elapsed_seconds"] for r in rows),
               "observed_evaluation_span_seconds": (max(finishes) - min(starts)).total_seconds(),
               "observed_span_definition": "Latest per-result timestamp minus earliest (timestamp − elapsed); excludes queue setup.",
               "parallel_wall_seconds": None, "preview_only": preview}
    runtime_path, completion_path = rawdir / "runtime.json", rawdir / "completion.json"
    if runtime_path.is_file() and completion_path.is_file():
        start, finish = load(runtime_path), load(completion_path)
        summary["runner_complete"] = finish.get("complete", False)
        summary["parallel_wall_seconds"] = (
            datetime.fromisoformat(finish["completed_at_utc"]) - datetime.fromisoformat(start["created_at_utc"])
        ).total_seconds()
        summary["initial_workers"] = start["initial_workers"]
        summary["maximum_workers"] = start["maximum_workers"]
        summary["threads_per_worker"] = start["threads_per_worker"]
    return summary, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=OPS / "dev64_curve_results_20261020")
    parser.add_argument("--inventory", type=Path, default=OPS / "m1_dev64_curve_inventory_20261020.json")
    parser.add_argument("--registration", type=Path, default=OPS / "m1_dev64_curve_registration_20261020.json")
    parser.add_argument("--policy", type=Path, default=OPS / "m1_checkpoint_policy_20261020.json")
    parser.add_argument("--registration-sha256", default=EXPECTED_REGISTRATION_SHA256)
    parser.add_argument("--output-dir", type=Path, default=RUNS / "m1_analysis_20261020")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--comparison-bootstrap-seed", type=int, default=20261020)
    parser.add_argument("--comparison-bootstrap-samples", type=int, default=10000)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    require(args.comparison_bootstrap_samples > 0, "Bootstrap sample count must be positive")
    require(sha(args.registration) == args.registration_sha256, "Registration file changed")
    reg, inv, policy = load(args.registration), load(args.inventory), load(args.policy)
    require(reg["inventory_sha256"] == sha(args.inventory) and reg["base_policy_sha256"] == sha(args.policy),
            "Inventory/base policy differs from preregistration")
    require(reg["execute_final"] is False and reg["selection_from_curve_allowed"] is False, "Role policy changed")
    jobs = {r["nominal_loss_tokens"]: r for r in inv["checkpoints"]}
    require(len(jobs) == 21 and sorted(jobs) == list(range(0, 100000001, 5000000)), "Inventory is not the complete 21-point curve")
    for fixed_role, n in (("primary", 100000000), ("secondary", 65000000)):
        require(jobs[n]["sha256"] == policy[fixed_role]["sha256"] == reg[fixed_role + "_checkpoint_sha256"], "Registered roles changed")
    paths = {n: args.raw_dir / f"m1_{n // 1000000:03d}m_continuation.json" for n in jobs}
    missing = [n for n, path in paths.items() if not path.is_file()]
    require(args.allow_partial or not missing, "Full curve required; missing nominal checkpoints: " + ", ".join(f"{n//1000000}M" for n in missing))
    require(len(missing) < 21, "No completed checkpoint results are available")
    preview = bool(missing)
    loaded, reference = {}, None
    for n, path in sorted(paths.items()):
        if not path.is_file():
            continue
        loaded[n] = validate_result(path, jobs[n], reference)
        if reference is None:
            reference = loaded[n][0]
    require(reference["manifest"]["registry_file_sha256"] == policy["dev64_registry_sha256"], "Wrong DEV registry")
    require(reference["manifest"]["prefix_annotations"]["file_sha256"] == policy["dev64_annotations_sha256"],
            "Wrong frozen prefix annotations")
    original_weights = bootstrap_weights(64, 20261019, 10000)
    prefix_weights = {label: bootstrap_weights(count, 20261019, 10000)
                      for label, count in PREFIX_GROUP_COUNTS.items() if count}
    rows, prefix_rows = [], []
    for n, (result, families, indexed) in loaded.items():
        rows.extend(summarize_point(jobs[n], result, families, indexed, original_weights, preview))
        prefix_rows.extend(summarize_prefix_groups(jobs[n], result, families, indexed, prefix_weights, preview))
    row_index = {(r["nominal_loss_tokens"], r["unrelated_article_count"]): r for r in rows}
    for row in rows:
        initial = row_index.get((0, row["unrelated_article_count"]))
        for key in COMPARISON_METRICS:
            row[key + "_minus_0m"] = row[key] - initial[key] if initial else None
    comparison_weights = bootstrap_weights(64, args.comparison_bootstrap_seed, args.comparison_bootstrap_samples)
    comparisons = []
    for a, b in ((100000000, 65000000), (100000000, 0), (65000000, 0)):
        comparisons.extend(paired_comparison(a, b, loaded, jobs, comparison_weights, preview,
                                           args.comparison_bootstrap_seed, args.comparison_bootstrap_samples))
    runtimes, runtime_rows = runtime_summary(args.raw_dir, loaded, jobs, preview)
    datadir = args.output_dir / "dev64_data"
    if preview:
        datadir /= "preview"
    figure_dir = args.output_dir / "figures"
    datadir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    write_csv(datadir / "dev64_checkpoint_distance_curve.csv", rows)
    write_csv(datadir / "dev64_prefix_group_results.csv", prefix_rows)
    write_csv(datadir / "dev64_paired_checkpoint_comparisons.csv", comparisons)
    write_csv(datadir / "dev64_runtime.csv", runtime_rows)
    family_rows = []
    for n, (_, families, indexed) in loaded.items():
        for f in families:
            for d in DISTANCES:
                r = indexed[f, d]
                family_rows.append({"checkpoint": jobs[n]["label"], "training_loss_tokens": jobs[n]["training_loss_tokens"],
                    "event_family": f, "pair_id": r["pair_id"], "later_source_id": r["later_source_id"],
                    "domain": r["domain"], "unrelated_article_count": d,
                    "target_tokens": r["conditions"]["self"]["target_tokens"],
                    **{a + "_nll": r["conditions"][a]["nll"] for a in ARMS},
                    **{k: r[k] for k in CONTRASTS if not k.startswith("retention_change_")}, "preview_only": preview})
    write_csv(datadir / "dev64_family_distance_results.csv", family_rows)
    figures = [] if args.no_figures else plot_figures(rows, figure_dir, preview)
    summary = {
        "complete_curve": not preview, "preview_only": preview, "completed_checkpoints": len(loaded),
        "missing_nominal_loss_tokens": missing, "checkpoint_selection_performed": False,
        "aggregation": "mean target-token NLL within family, then unweighted mean over 64 event families",
        "manifest_sha256": reference["manifest_sha256"], "model_identity_sha256": reference["model_identity_sha256"],
        "target_tokens_per_distance": reference["manifest"]["target_tokens_per_distance"],
        "prefix_group_counts": dict(PREFIX_GROUP_COUNTS),
        "prefix_group_results_file": "dev64_prefix_group_results.csv",
        "prefix_group_bootstrap": dict(reference["bootstrap"]),
        "primary100M": [r for r in rows if r["nominal_loss_tokens"] == 100000000],
        "secondary65M": [r for r in rows if r["nominal_loss_tokens"] == 65000000],
        "initialization0M": [r for r in rows if r["nominal_loss_tokens"] == 0],
        "paired_comparisons": comparisons, "runtime": runtimes, "figures": figures,
        "validation": {"checkpoint_hash_step_tokens_tensorhash_verified": True, "manifest_equal": True,
                       "family_target_write_identity_equal": True, "coverage_64_by_5": True,
                       "empty_nll_loss_sum_targets_exact_equal": True,
                       "saved_means_and_bootstrap_intervals_independently_recomputed": True,
                       "prefix_group_means_and_bootstrap_intervals_independently_recomputed": True,
                       "prefix_group_membership_fixed_across_distances_and_checkpoints": True},
        "limitations": LIMITATIONS, "analysis_created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(datadir / "dev64_summary.json", summary)
    sources = [args.inventory, args.registration, args.policy, Path(__file__)]
    sources += [p for p in paths.values() if p.is_file()]
    sources += [args.raw_dir / name for name in ("runtime.json", "completion.json") if (args.raw_dir / name).is_file()]
    write_json(datadir / "dev64_source_manifest.json", {
        "files": [{"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha(p)} for p in sources],
        "reproduction_arguments": vars(args) | {"raw_dir": str(args.raw_dir), "inventory": str(args.inventory),
            "registration": str(args.registration), "policy": str(args.policy), "output_dir": str(args.output_dir)},
        "numpy_version": np.__version__, "python_version": sys.version,
    })
    print(json.dumps({"complete_curve": not preview, "preview_only": preview, "checkpoints": len(loaded),
                      "curve_rows": len(rows), "prefix_group_rows": len(prefix_rows),
                      "output_dir": str(datadir), "figures": figures}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

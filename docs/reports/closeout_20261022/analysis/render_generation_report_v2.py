"""Independently verify score aggregation and render the sealed four-condition report."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

CONDITIONS = ("self", "empty", "donor", "full_context")
LABELS = ("Self memory", "Empty memory", "Donor memory", "Full context")
COLORS = ("#B7602E", "#8B969F", "#7B6BA6", "#276286")
GROUPS = {
    "main_f1": (1, 2, 3, 4), "multi_hop": (1,), "temporal": (2,),
    "open_domain": (3,), "single_hop": (4,), "adversarial": (5,),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    if not math.isfinite(float(a)) or abs(float(a) - float(b)) > 1e-9:
        raise ValueError(f"aggregation mismatch: {a} versus {b}")


def audit(report, rows):
    import numpy as np

    if len(rows) != 1986 or len({x["qid"] for x in rows}) != 1986:
        raise ValueError("requires the complete 1986 unique questions")
    sids = sorted({x["sample_id"] for x in rows})
    if len(sids) != 10 or any(x["category"] not in (1, 2, 3, 4, 5) for x in rows):
        raise ValueError("wrong conversation/category population")
    for row in rows:
        if set(row["scores"]) != set(CONDITIONS):
            raise ValueError("incomplete condition matrix")
        for condition in CONDITIONS:
            value = row["scores"][condition]
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid score")
            close(row["official_rounded_scores"][condition], round(value, 3))
    original = report["scores"]
    boot = original["bootstrap"]
    if any(boot[k] != v for k, v in {
        "cluster": "conversation", "cluster_count": 10, "replicates": 10000,
        "seed": 20261022, "paired": True, "percentile": [2.5, 97.5],
    }.items()):
        raise ValueError("bootstrap protocol differs")
    # Convert sampled cluster indices into multiplicities and independently
    # aggregate through a matrix product, preserving question weighting.
    sampled = np.random.default_rng(20261022).integers(0, 10, (10000, 10))
    weights = np.stack([(sampled == i).sum(axis=1) for i in range(10)], axis=1)
    checks = 0
    for name, cats in GROUPS.items():
        selected = [x for x in rows if x["category"] in cats]
        group = original["metrics"][name]
        if group["question_count"] != len(selected):
            raise ValueError("group size differs")
        counts = np.array([sum(x["sample_id"] == s for x in selected) for s in sids])
        sums = np.array([[math.fsum(x["scores"][c] for x in selected if x["sample_id"] == s)
                          for c in CONDITIONS] for s in sids])
        denominator = weights @ counts
        valid = denominator > 0
        samples = (weights @ sums)[valid] / denominator[valid, None] * 100
        points = sums.sum(axis=0) / len(selected) * 100

        def check_interval(record, point, draws):
            close(record["score_percent"], point)
            bounds = np.percentile(draws, (2.5, 97.5))
            for actual, expected in zip(record["cluster_bootstrap_95ci_percent"], bounds):
                close(actual, expected)
            if record["valid_bootstrap_replicates"] != int(valid.sum()):
                raise ValueError("bootstrap valid count differs")

        for i, c in enumerate(CONDITIONS):
            check_interval(group["conditions"][c], points[i], samples[:, i])
            rounded = 100 * math.fsum(x["official_rounded_scores"][c] for x in selected) / len(selected)
            close(original["official_cli_rounded_point_estimates_percent"][name][c], rounded)
            checks += 1
        for i, c in enumerate(CONDITIONS[1:], 1):
            check_interval(group["paired_differences"]["self-" + c],
                           points[0] - points[i], samples[:, 0] - samples[:, i])
            checks += 1
        for sid, count, total in zip(sids, counts, sums):
            conversation = original["per_conversation"][sid][name]
            if conversation["question_count"] != int(count):
                raise ValueError("conversation count differs")
            for c, value in zip(CONDITIONS, total):
                if count:
                    close(conversation["score_percent"][c], value / count * 100)
                elif conversation["score_percent"][c] is not None:
                    raise ValueError("empty conversation subgroup is not null")
    for name, original_name, count in (
        ("main_results_1540_questions", "main_f1", 1540),
        ("adversarial_results_446_questions", "adversarial", 446),
    ):
        if report[name] != original["metrics"][original_name] or report[name]["question_count"] != count:
            raise ValueError("wrapper differs from original scorer")
    if any(report["output_diagnostics"][c]["questions"] != 1986 for c in CONDITIONS):
        raise ValueError("incomplete generation diagnostic counts")
    return {"status": "passed", "questions": 1986, "conditions": 4,
            "conversations": 10, "condition_and_paired_intervals_recomputed": checks,
            "bootstrap_replicates": 10000, "bootstrap_seed": 20261022,
            "scope": "Independent aggregation, bootstrap, rounding and wrapper audit; official answer matching uses the frozen upstream scorer."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--per-question", type=Path, required=True)
    parser.add_argument("--per-question-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.source) != args.source_sha256 or sha(args.per_question) != args.per_question_sha256:
        raise ValueError("input SHA mismatch")
    if args.output.exists():
        raise ValueError("output must be fresh")
    report = json.loads(args.source.read_bytes())
    rows = [json.loads(line) for line in args.per_question.read_text(encoding="utf-8").splitlines() if line]
    audit_result = audit(report, rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    plotted = {}
    for ax, key, title, metric in (
        (axes[0], "main_results_1540_questions", "Answer F1 | 1,540 questions", "Official token F1 (%)"),
        (axes[1], "adversarial_results_446_questions", "Adversarial | 446 questions", "Official refusal accuracy (%)"),
    ):
        group = report[key]
        plotted[key] = group
        for i, (condition, color) in enumerate(zip(CONDITIONS, COLORS)):
            point = group["conditions"][condition]["score_percent"]
            low, high = group["conditions"][condition]["cluster_bootstrap_95ci_percent"]
            ax.hlines(i, low, high, color=color, linewidth=2.5)
            ax.plot(point, i, "o", color=color, markersize=7, clip_on=False)
            ax.annotate(f"{point:.2f}", (point, i), xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=9, color=color)
        ax.set(title=title, xlabel=metric, xlim=(-3, 103), xticks=(0, 25, 50, 75, 100),
               yticks=range(4), yticklabels=LABELS, ylim=(3.6, -.6))
        ax.grid(axis="x", alpha=.18)
    axes[1].tick_params(labelleft=True)
    fig.suptitle("LoCoMo: four frozen generation conditions", fontsize=14)
    fig.text(.5, .025, "Bars: paired conversation-cluster bootstrap 95% CI (10 conversations, 10,000 draws).\n"
             "Fixed zero-shot prompt. Main F1 and adversarial refusal are separate metrics.",
             ha="center", fontsize=8.5, color="#46515B")
    fig.tight_layout(rect=(0, .1, 1, .94), w_pad=2)
    for ext in ("png", "svg"):
        fig.savefig(args.output / ("generation_four_conditions." + ext), dpi=190, facecolor="white")
    plt.close(fig)
    metadata = {"protocol": "qwen38_locomo_generation_figure_v2",
                "report_path": str(args.source.resolve()), "report_sha256": args.source_sha256,
                "per_question_sha256": args.per_question_sha256, "renderer_sha256": sha(__file__),
                "audit": audit_result, "plotted_values": plotted}
    (args.output / "audit_and_plotted_values.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "audit": audit_result}))


if __name__ == "__main__":
    main()

"""Plot the registered primary 100M comparison from verified CPU summaries."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with (args.analysis_dir / "context_recovery_primary100m.csv").open(
            encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["scope"] == "all_targets"]
    rows.sort(key=lambda r: int(r["unrelated_article_count"]))
    assert [int(r["unrelated_article_count"]) for r in rows] == [0, 1, 4, 8, 16]
    assert all(r["checkpoint"] == "100M" and int(r["families"]) == 64 for r in rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "dev64_context_reference_100m"
    assert args.overwrite or not output.with_suffix(".png").exists()
    assert args.overwrite or not output.with_suffix(".svg").exists()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (gain_ax, ratio_ax) = plt.subplots(1, 2, figsize=(11.8, 4.6))
    x = np.arange(5)
    for field, label, color, shift in [
        ("memory", "100M writer memory", "#007d7a", -.08),
        ("context", "Original model + full source context", "#cc7130", .08),
    ]:
        values = np.array([float(r[field + "_gain_nats"]) for r in rows])
        low = np.array([float(r[field + "_gain_ci95_low"]) for r in rows])
        high = np.array([float(r[field + "_gain_ci95_high"]) for r in rows])
        gain_ax.errorbar(x + shift, values, yerr=[values - low, high - values],
                         marker="o", lw=1.8, capsize=3, color=color, label=label)
    gain_ax.axhline(0, color="#555555", lw=.7)
    gain_ax.set(title="Absolute predictive benefit", ylabel="NLL reduction (nats / target token)")
    gain_ax.legend(fontsize=9, loc="lower left", bbox_to_anchor=(.03, .20), frameon=False)
    for i, row in enumerate(rows):
        if row["ratio_ci_available"].lower() == "true":
            value = float(row["recovery_percent"])
            low, high = (float(row["recovery_percent_ci95_" + side]) for side in ("low", "high"))
            ratio_ax.errorbar(i, value, yerr=[[value-low], [high-value]], fmt="o",
                              capsize=4, color="#007d7a", markersize=7)
            ratio_ax.annotate(f"{value:.1f}%", (i, high), xytext=(0, 8),
                              textcoords="offset points", ha="center", fontsize=10)
        else:
            ratio_ax.text(i, 0, "unstable\nratio", ha="center", va="bottom", fontsize=9)
    ratio_ax.axhline(100, color="#888888", ls="--", lw=1,
                    label="Equal observed NLL benefit")
    ratio_ax.axhline(0, color="#555555", lw=.7)
    ratio_ax.set(title="Memory benefit / context benefit", ylabel="Recovered NLL benefit (%)")
    ratio_ax.margins(y=.22)
    if all(r["ratio_ci_available"].lower() == "true" and
           0 <= float(r["recovery_percent_ci95_low"]) <=
           float(r["recovery_percent_ci95_high"]) <= 100 for r in rows):
        ratio_ax.set_ylim(0, 110)
    ratio_ax.legend(fontsize=9, loc="best", frameon=False)
    for ax in (gain_ax, ratio_ax):
        ax.set_xticks(x, ["0", "1", "4", "8", "16"])
        ax.set_xlabel("Intervening unrelated articles")
        ax.grid(axis="y", alpha=.16)
    fig.suptitle("DEV64 original-weight context reference | primary 100M", fontsize=15)
    fig.text(.5, .015, "64 equally weighted events; same target tokens; paired 95% bootstrap intervals.\n"
             "A proportion of NLL benefit, not answer accuracy or a theoretical memory ceiling.",
             ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .10, 1, .94))
    fig.savefig(output.with_suffix(".png"), dpi=190, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()

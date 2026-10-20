"""Render the saved CPU capacity audit; no inference, tokenization, or replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT = REPO.parent / "delta-ngram-runs/m1_analysis_20261020"
DISTANCES = (0, 1, 4, 8, 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPORT / "dev64_data/capacity_scope_note.json")
    parser.add_argument("--output", type=Path, default=REPORT / "figures/dev64_capacity.png")
    args = parser.parse_args()
    note = json.loads(args.input.read_bytes())
    assert note["status"] == "complete_exact_address_replay"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
        "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#A7AFBA", "axes.labelcolor": "#243247",
        "text.color": "#243247", "xtick.color": "#526175", "ytick.color": "#526175",
        "grid.color": "#DCE2EA", "grid.linewidth": 0.7,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    data = {(r["unrelated_article_count"], r["arm"]): r for r in note["per_distance_arm_summary"]}
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), sharex=True)
    colors = {"self": "#2462A8", "donor": "#CA7831"}
    labels = {"self": "Self", "donor": "Fixed donor"}
    for row, order in enumerate((2, 3)):
        for column, stem in enumerate((
                f"occupancy_U{order}_over_B",
                f"hash_merge_fraction_K{order}_minus_U{order}_over_K{order}")):
            ax = axes[row, column]
            for arm in ("self", "donor"):
                values = [data[distance, arm] for distance in DISTANCES]
                lower, middle, upper = (
                    [r[stem + "_" + level] * 100 for r in values]
                    for level in ("min", "p50", "max"))
                ax.fill_between(DISTANCES, lower, upper, color=colors[arm], alpha=.12,
                                linewidth=0)
                ax.plot(DISTANCES, middle, marker="o" if arm == "self" else "s",
                        linestyle="-" if arm == "self" else "--", markersize=4.5,
                        linewidth=1.8, color=colors[arm], label=labels[arm])
            ax.set_title(f"{order}-gram table", loc="left", fontweight="semibold", pad=10)
            ax.set_ylabel("Occupied physical rows: U / B" if column == 0
                          else "Distinct-key merge fraction: (K - U) / K")
            ax.set_xticks(DISTANCES)
            ax.set_xlim(-.5, 16.5)
            ax.set_ylim(0, 100 if column == 0 else 40)
            if column == 0:
                ax.axhline(100, color="#8795A5", linewidth=.8, linestyle=(0, (4, 3)),
                           clip_on=False, zorder=3)
                ax.text(.99, .97, "Physical row limit", transform=ax.transAxes,
                        ha="right", va="top", fontsize=9, color="#667486")
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            ax.grid(axis="y", zorder=0)
            if row == 1:
                ax.set_xlabel("Unrelated articles written after the anchor")
    fig.suptitle("DEV64: physical-row occupancy and distinct-key hash merging",
                 x=.07, y=.97, ha="left", fontsize=17, fontweight="bold")
    fig.text(.07, .922, "Exact CPU address replay  |  B = 16,384 per table  |  64 event families",
             fontsize=11, color="#526175")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(.94, .929),
               ncol=2, frameon=False, fontsize=10, handlelength=2.5)
    fig.text(.07, .055,
             "Lines: median. Shaded bands: min-max across families (not confidence intervals).",
             fontsize=10, color="#526175")
    fig.text(.07, .027,
             "K counts distinct token n-grams; U counts hash rows. Repeats E - K are excluded. "
             "(K - U) / K is not pairwise collision probability.",
             fontsize=9, color="#526175")
    fig.subplots_adjust(left=.09, right=.96, top=.85, bottom=.14, hspace=.30, wspace=.28)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

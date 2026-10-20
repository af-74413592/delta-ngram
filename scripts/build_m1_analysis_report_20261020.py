"""Render scientific plots for the Markdown M1 analysis; no model execution."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT.parent / "delta-ngram-runs/m1_analysis_20261020"
BLUE = "#126C9A"
TEAL = "#168177"
ORANGE = "#C56D32"
INK = "#1F3343"
MUTED = "#536675"
LABELS = {
    "D1_first5M": "D1 / α=1",
    "fixed_099": "固定 α=.99",
    "fixed_095": "固定 α=.95",
    "learned_global_099": "全局可学习 α",
    "evidence_block_099": "证据块 α",
}


def read_json(path):
    return json.loads(path.read_bytes())


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    return float(row[key])


def fmt(value, signed=False):
    return format(float(value), "+.6f" if signed else ".6f")


def load_data(base):
    return read_json(base / "data/report_summary.json")


def plots(base, font_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np

    font = FontProperties(fname=str(font_path))
    plt.rcParams.update({
        "font.family": font.get_name(), "axes.unicode_minus": False,
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#B4C1C8", "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "savefig.facecolor": "white",
    })
    output = base / "figures"
    output.mkdir(parents=True, exist_ok=True)
    series = read_csv(base / "data/m1_series.csv")
    x = np.array([number(row, "training_loss_tokens") / 1e6 for row in series])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), layout="constrained")
    for key, label, color in [
        ("macro_delta_empty", "正确记忆相对空记忆 Δempty", BLUE),
        ("macro_delta_donor", "正确记忆相对错配记忆 Δdonor", ORANGE),
    ]:
        y = np.array([number(row, key) for row in series])
        axes[0].plot(x, y, "o-", lw=1.6, ms=3, label=label, color=color)
        axes[0].fill_between(
            x, [number(row, key + "_ci_lower") for row in series],
            [number(row, key + "_ci_upper") for row in series], color=color, alpha=.13,
        )
    for key, label, color in [
        ("macro_self_nll", "正确记忆 self NLL", BLUE),
        ("macro_donor_nll", "错配记忆 donor NLL", ORANGE),
    ]:
        axes[1].plot(x[1:], [number(row, key) for row in series[1:]],
                     "o-", lw=1.6, ms=3, color=color, label=label)
    axes[0].set(title="全程收益：越高越好", ylabel="nats / token")
    axes[1].set(title="5M 后的预测损失：越低越好", ylabel="nats / token")
    for ax in axes:
        ax.axvline(65.009696, ls=":", color=MUTED, lw=1)
        ax.set_xlabel("训练目标 token（百万）")
        ax.grid(axis="y", alpha=.18)
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(output / "training_curve.png", dpi=190)
    plt.close(fig)

    sources = read_csv(base / "data/endpoint_per_source.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6), layout="constrained")
    y = np.arange(len(sources))
    for ax, key, label, color in zip(
        axes, ["delta_empty", "delta_donor"], ["Δempty", "Δdonor"], [BLUE, ORANGE]
    ):
        values = np.array([number(row, key) for row in sources])
        low = np.array([number(row, key + "_ci_lower") for row in sources])
        high = np.array([number(row, key + "_ci_upper") for row in sources])
        ax.errorbar(values, y, xerr=[values-low, high-values],
                    fmt="o", color=color, ecolor=color, capsize=3)
        ax.set_yticks(y, [row["source"] for row in sources])
        ax.invert_yaxis()
        ax.set(title=f"终点 {label} 与原始 95% CI", xlabel="nats / token")
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=.18)
    fig.savefig(output / "endpoint_sources.png", dpi=190)
    plt.close(fig)

    contrasts = read_csv(base / "data/paired_contrasts.csv")
    names = ["fixed_099", "fixed_095", "learned_global_099", "evidence_block_099"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.1), layout="constrained")
    for ax, metric, color in zip(axes, ["delta_empty", "delta_donor"], [BLUE, ORANGE]):
        rows = [next(row for row in contrasts
                     if row["comparison"] == name + "_minus_D1_first5M"
                     and row["metric"] == metric) for name in names]
        values = np.array([number(row, "difference") for row in rows])
        low = np.array([number(row, "ci_lower") for row in rows])
        high = np.array([number(row, "ci_upper") for row in rows])
        ax.errorbar(values * 1000, np.arange(4),
                    xerr=[(values-low)*1000, (high-values)*1000],
                    fmt="o", color=color, ecolor=color, capsize=3)
        ax.set_yticks(np.arange(4), [LABELS[name] for name in names])
        ax.invert_yaxis()
        ax.axvline(0, ls=":", color=MUTED)
        ax.set(title=f"{metric.replace('delta_', 'Δ')} 相对 D1 的变化",
               xlabel="差值（千分之一 nats / token；越高越好）")
        ax.grid(axis="x", alpha=.18)
    fig.savefig(output / "alpha_paired_contrasts.png", dpi=190)
    plt.close(fig)

    capacity = read_json(base / "diagnostics/mathematical_capacity_audit.json")
    writer = read_json(base / "diagnostics/mathematical_audit.json")
    occupancy = capacity["observed"]["overall"]["aggregate_occupancy_U_over_2B"]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.6), layout="constrained")
    levels = ["p50", "p95", "max"]
    axes[0].bar(["中位数", "p95", "最大值"],
                [occupancy[key] * 100 for key in levels], color=BLUE, width=.6)
    axes[0].axhline(100, ls=":", color=ORANGE, label="总行数容量")
    axes[0].set_ylim(0, 108)
    axes[0].set(title="完整训练轨迹的总占用", ylabel="占用率（%）")
    for i, key in enumerate(levels):
        axes[0].text(i, occupancy[key] * 100 + 3, f"{occupancy[key]*100:.2f}%",
                     ha="center", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    values = [writer["65m_to_100m"], writer["95m_to_100m"]]
    differences = np.array([row["difference"] * 1000 for row in values])
    lower = np.array([row["ci_lower"] * 1000 for row in values])
    upper = np.array([row["ci_upper"] * 1000 for row in values])
    axes[1].errorbar(differences, [0, 1],
                    xerr=[differences-lower, upper-differences],
                    fmt="o", color=TEAL, capsize=4)
    axes[1].axvline(0, ls=":", color=MUTED)
    axes[1].set_yticks([0, 1], ["65M → 终点", "95M → 终点"])
    axes[1].set_ylim(-.65, 1.65)
    axes[1].invert_yaxis()
    axes[1].set(title="留出 self NLL 变化及 95% CI",
                xlabel="千分之一 nats/token；负值为改善")
    intervals = writer["trend_intervals"][:4]
    gains = [row["self_nll_improvement_per_5m_actual_tokens"] * 1000
             for row in intervals]
    axes[2].bar(["5-25M", "25-50M", "50-75M", "75-100M"],
                gains, color=TEAL, width=.6)
    axes[2].set(title="每新增 5M token 的平均收益",
                ylabel="self NLL 改善（千分之一 nats/token）")
    for i, value in enumerate(gains):
        axes[2].text(i, value + .12, f"{value:.3f}", ha="center", fontsize=8)
    axes[2].set_ylim(0, max(gains) * 1.18)
    for ax in axes:
        ax.grid(axis="y" if ax is not axes[1] else "x", alpha=.15)
    fig.savefig(output / "mathematical_checks.png", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    args = parser.parse_args()
    plots(args.base, args.font)


if __name__ == "__main__":
    main()

"""Recompute the first forward diagnostic's denominators and trajectory table."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/qwen38_memory_forward_20261021/probe/report.json"
raw = SOURCE.read_bytes()
report = json.loads(raw)
diagnostics = report["diagnostics"]
gains = [row for row in diagnostics["encoding_replacement_records"] if row["block_g"] is not None]
denominators = np.array([row["read_state_l2"] for row in gains])
values = np.array([row["block_g"] for row in gains])
indices = [4, 6, 7, 9, 10, 11]
table = []
fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.3), sharey=True)
for index, ax in zip(indices, axes.flat):
    tracks = {eps: sorted([row for row in diagnostics["twin_trajectory_records"]
                         if row["trajectory_index"] == index and row["relative_epsilon"] == eps],
                         key=lambda row: row["write_index"]) for eps in (.001, .01)}
    small, large = tracks[.001], tracks[.01]
    reference = small[-1]["reference_state_l2"]
    table.append({"trajectory_index": index, "write_blocks": len(small),
        "repeat_relative_final": small[-1]["repeat_error_l2"] / reference,
        "eps001_relative_final": small[-1]["error_over_reference_state"],
        "eps01_relative_final": large[-1]["error_over_reference_state"],
        "repeat_seed_absolute": small[0]["repeat_error_l2"],
        "repeat_first_followup_absolute": small[1]["repeat_error_l2"],
        "final_absolute_ratio_eps01_over_eps001": large[-1]["state_error_l2"] / small[-1]["state_error_l2"]})
    x = [row["write_index"] + 1 for row in small]
    ax.plot(x, [100 * row["repeat_error_l2"] / row["reference_state_l2"] for row in small],
            color="#555555", ls="--", label="Independent repeat")
    ax.plot(x, [100 * row["error_over_reference_state"] for row in small],
            color="#2166ac", label="0.1% initial perturbation")
    ax.plot(x, [100 * row["error_over_reference_state"] for row in large],
            color="#d6604d", label="1% initial perturbation")
    ax.set_title(f"Trajectory {index}  |  {len(small)} writes", fontsize=11)
    ax.set_xlabel("Write number")
    ax.set_ylim(0, 7)
    ax.grid(alpha=.18)
axes[0, 0].set_ylabel("Error / current reference state (%)")
axes[1, 0].set_ylabel("Error / current reference state (%)")
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
fig.suptitle("80M memory-encoded writer: finite trajectory separation", fontsize=14)
fig.tight_layout(rect=(0, .06, 1, .94))
(ROOT / "figures").mkdir(exist_ok=True)
for extension in ("png", "svg"):
    fig.savefig(ROOT / f"figures/qwen38_memory_forward_twins_20261021.{extension}", dpi=180)
plt.close(fig)

analysis = {"source_sha256": hashlib.sha256(raw).hexdigest(),
    "fraction_to_percent": "multiply by 100; e.g. 0.0254 means 2.54%, not 0.0254%",
    "eligible_blocks": len(gains), "denominator_quantiles_0_25_50_75_100": np.quantile(denominators, [0, .25, .5, .75, 1]).tolist(),
    "pearson_G_vs_denominator": float(np.corrcoef(values, denominators)[0, 1]),
    "pearson_G_vs_log_denominator": float(np.corrcoef(values, np.log(denominators))[0, 1]),
    "largest_G_records": sorted(gains, key=lambda row: row["block_g"], reverse=True)[:10],
    "final_relative_errors": table,
    "interpretation": "finite observed trajectories only; autonomous repeat is not a subtractable local noise floor"}
(ROOT / "data/qwen38_memory_forward_analysis_20261021.json").write_text(
    json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in analysis.items() if key != "largest_G_records"}, indent=2))

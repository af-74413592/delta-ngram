"""CPU-only, reproducible analysis of the saved DEV64 result extract."""
from pathlib import Path
import hashlib
import json
import random

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "qwen38_native_100m_source_20261021.json"
OUT = ROOT / "qwen38_native_100m_analysis_20261021.json"
SEED, REPLICATES = 20261019, 10000
DISTANCES = [0, 1, 4, 8, 16]
METRICS = ["delta_empty", "delta_donor", "self_nll", "donor_nll", "empty_nll",
           "first_target_delta_empty", "first_target_delta_donor",
           "direction_delta_self", "direction_delta_donor", "direction_difference"]
source = json.loads(SOURCE.read_text(encoding="utf-8"))
reports = {(r["arm"], r["milestone"]): r for r in source["reports"]}
families = sorted({r["event_family"] for r in source["reports"][0]["per_family"]})
assert len(families) == 64
indexes = {}
arrays = {}
base_meta = None
for key, report in reports.items():
    assert report["complete"] and report["file_hash_verified"] and report["cursor_hash_verified"]
    assert report["evaluation_target_accesses"] == report["expected_evaluation_target_accesses"] == 2197313
    assert report["backward_calls"] == report["optimizer_steps"] == 0
    assert report["manifest"]["training_rng_restored_on_exit"]
    assert report["manifest"]["writer_and_alpha_state_unchanged"]
    index = {(r["event_family"], r["unrelated_article_count"]): r for r in report["per_family"]}
    assert len(index) == len(report["per_family"]) == 320
    assert set(index) == {(f, k) for f in families for k in DISTANCES}
    indexes[key] = index
    meta = {(f, k): (r["pair_id"], r["later_source_id"], r["prefix_restatement_label"],
                      r["revision"], [(c, r["conditions"][c]["target_tokens"],
                       r["conditions"][c]["first_target_tokens"]) for c in sorted(r["conditions"])])
            for (f, k), r in index.items()}
    if base_meta is None:
        base_meta = meta
    else:
        assert meta == base_meta, "Family/target/annotation mismatch across nodes"
    values = np.empty((64, 5, len(METRICS)))
    for i, family in enumerate(families):
        for j, distance in enumerate(DISTANCES):
            row = index[(family, distance)]
            for m, metric in enumerate(METRICS):
                values[i, j, m] = (row["conditions"][metric[:-4]]["nll"]
                                    if metric.endswith("_nll") else row[metric])
            assert abs(row["delta_empty"] - (row["conditions"]["empty"]["nll"] - row["conditions"]["self"]["nll"])) < 1e-12
            assert abs(row["delta_donor"] - (row["conditions"]["donor"]["nll"] - row["conditions"]["self"]["nll"])) < 1e-12
    arrays[key] = values

weights_cache = {}
def weights(n):
    if n not in weights_cache:
        rng = random.Random(SEED)
        indices = np.array([[rng.randrange(n) for _ in range(n)] for _ in range(REPLICATES)])
        count = np.zeros((REPLICATES, n), dtype=np.float64)
        np.add.at(count, (np.arange(REPLICATES)[:, None], indices), 1)
        weights_cache[n] = count / n
    return weights_cache[n]

def summarize(values):
    n = values.shape[0]
    flat = values.reshape(n, -1)
    boot = weights(n) @ flat
    mean = flat.mean(axis=0).reshape(values.shape[1:])
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0, method="linear").reshape((2,) + values.shape[1:])
    return [{"k": k, "n": n, **{metric: {"mean": float(mean[j, m]),
            "ci95": [float(lo[j, m]), float(hi[j, m])]}
            for m, metric in enumerate(METRICS)}} for j, k in enumerate(DISTANCES)]

def keyname(key):
    return f"{key[0]}_{key[1] // 1000000}M"

nodes = {}
for key, a in arrays.items():
    rows = summarize(a)
    report = reports[key]
    for result, original in zip(rows, report["summary"]["retention_curve"]):
        for metric in ["delta_empty", "delta_donor", "direction_delta_self", "direction_delta_donor", "direction_difference"]:
            assert abs(result[metric]["mean"] - original[metric]) < 1e-12
    nodes[keyname(key)] = {"actual_targets": report["checkpoint"]["completed_loss_targets"],
                           "step": report["checkpoint"]["optimizer_step"], "rows": rows,
                           "source_file_sha256": report["file_sha256"]}

comparisons = {}
comparison_specs = [("alpha100_rmsnorm", a, "alpha100_rmsnorm", b)
                    for a, b in [(40,20),(60,40),(80,60),(100,80),(100,40),(100,60),(80,40)]]
comparison_specs += [("alpha100_rmsnorm", m, "alpha099_rmsnorm", m) for m in [20,40]]
for ca, cm, ra, rm in comparison_specs:
    ck, rk = (ca, cm*1000000), (ra, rm*1000000)
    diff = arrays[ck] - arrays[rk]
    for m, metric in enumerate(METRICS):
        if metric.endswith("_nll"):
            diff[:, :, m] *= -1
    comparisons[f"{keyname(ck)}_vs_{keyname(rk)}"] = {
        "candidate": keyname(ck), "reference": keyname(rk),
        "sign": "NLL fields: reference minus candidate (positive means lower candidate NLL); other fields: candidate minus reference (interpret by metric)",
        "rows": summarize(diff)}

final_key = ("alpha100_rmsnorm",100000000)
final_index = indexes[final_key]
groups = {label: [i for i,f in enumerate(families) if final_index[(f,0)]["prefix_restatement_label"] == label]
          for label in ["restated","not_restated","unknown"]}
groups["revision"] = [i for i,f in enumerate(families) if final_index[(f,0)]["revision"]["eligible"]]
groups["non_revision"] = [i for i in range(64) if i not in groups["revision"]]
assert {k:len(v) for k,v in groups.items()} == {"restated":34,"not_restated":23,"unknown":7,"revision":21,"non_revision":43}
subgroups = {name: {"n":len(ids), "families":[families[i] for i in ids],
                    "rows":summarize(arrays[final_key][ids])} for name,ids in groups.items()}
retention = arrays[final_key] - arrays[final_key][:,0:1,:]
result = {
    "protocol":"qwen38_native_100m_cpu_posthoc_analysis_v1",
    "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "bootstrap":{"unit":"event_family","replicates":REPLICATES,"seed":SEED,
                 "rng":"random.Random.randrange","percentiles":"numpy linear .025/.975",
                 "paired_across_nodes_conditions_and_distances":True,
                 "multiple_comparison_correction":False,"training_seeds":1},
    "metrics":METRICS,"families":families,"nodes":nodes,"comparisons":comparisons,
    "final_subgroups":subgroups,"final_retention_change":summarize(retention),
    "verification":{"all_7_reports_complete_and_hash_verified":True,
                    "all_64_by_5_rows_unique_and_aligned":True,
                    "all_targets_annotations_and_ids_aligned":True,
                    "summary_point_estimates_reproduced":True},
}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("Saved",OUT)
print("Final 100M")
for row in nodes[keyname(final_key)]["rows"]:
    print(row["k"], {m:row[m] for m in ["delta_empty","delta_donor","direction_difference"]})
print("Comparisons k=0,16")
for key,value in comparisons.items():
    if "100M" in key or "alpha099" in key:
        print(key)
        for row in [value["rows"][0],value["rows"][-1]]:
            print(row["k"], {m:row[m] for m in ["delta_empty","delta_donor","self_nll","donor_nll"]})

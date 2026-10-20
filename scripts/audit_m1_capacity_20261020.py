"""Reproduce M1 capacity checks from frozen JSON metadata and a plain/gzip log.

Uses only Python's standard library. Does not import the model or torch, run
inference/training, access sealed FINAL data, or change source logs. Defaults
refer to the local historical M1 archive. All paths can be overridden.

Example:
    python scripts/audit_m1_capacity_20261020.py

The original uncompressed log SHA is checked while parsing every line, even
when the input is gzip. Outputs go to a separate capacity_reproduction folder.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO.parent / "delta-ngram-runs"
OPS = RUNS / "multidomain_launch_20261019"
BACKUP = OPS / "final_backup_20261020"
EXPECTED_LOG_SHA256 = "919f99f27049bd87922de48346d58ec20592717bd8aa5faab53b3ae09b0fd48a"
EXPECTED_GZIP_SHA256 = "3ff959ad3eea5ad1a6e1cfb2f59a7ead5622c0e37f36bb7264bc60d62063a00e"
BANDS = {
    "early_111_210": (111, 210),
    "near65M_2614_2713": (2614, 2713),
    "late_4074_4173": (4074, 4173),
}
SOURCE_LINES = {
    "ngram_lookup.py": [5, 12, 23, 25],
    "operators.py": [13, 14, 15, 27, 32, 35, 37, 39, 46],
    "raw_lm.py": [98, 163, 164, 227, 294, 488, 490, 498, 764, 774, 778],
    "event_lm.py": [82, 228],
    "batching.py": [31, 88, 139],
    "state.py": [27, 67, 71, 76],
    "writer.py": [47, 103, 108, 112],
    "block_monitor.py": [124, 148, 154, 155, 156, 158, 161],
    "raw_evaluation.py": [23, 30, 34, 36],
}
MATH = {
    "address": "row_o=1+slot(o)*B+hash(token_ngram) mod B; independent of writer parameters.",
    "allocation": "A_(t+1)=A_t union touched_t, including beta=0; no eviction/full trigger.",
    "occupancy": "U_o=|allocated rows in order o|; q_o=U_o/B; U=sum_o U_o, q=U/(H*B).",
    "raw_bound": "T=512*(L-1); E_o=max(0,T-o+1); U_o<=min(B,E_o).",
    "event_bound": "w_j=a_j except w_last=a_last-((a_last-1) mod 512+1); "
                   "E_o=sum_j max(0,w_j-o+1); U_o<=min(B,E_o).",
    "donor_bound": "Same source/length and target write-token budget T; article boundaries "
                   "can differ, but U_o<=min(B,max(0,T-o+1)).",
    "collision_decomposition": "E-U=(E-K)+(K-U): E occurrences, K distinct exact ngram keys, "
                               "U hashed rows. E-K is key repetition; K-U is collision slot loss.",
    "state_update": "x_next=(1-a)*x+sum_i(beta_i/m)*target_i, a=sum_i beta_i/m.",
    "conditional_bound": "For 0<=beta_i<=1, ||x_next||<=max(||x||,max_i||target_i||). "
                         "Targets must be uniformly bounded for a uniform state bound. "
                         "Target MLP has no hard output bound; no BPTT contraction follows.",
    "fixed_checkpoint_proof": "Same frozen evidence/masks/hash/write schedule implies equal "
                              "touched sets, allocation, occupancy, and counts across checkpoints. "
                              "A_0 is empty per trajectory; training steps do not share memory rows.",
}
LIMITATIONS = [
    "Log occupancy is total U across two disjoint orders, not separate U_2 and U_3.",
    "Training examples differ across time bands; temporal means are observational.",
    "True hash collisions cannot be separated from repeated identical ngrams without K.",
    "Low occupancy does not exclude representation/interference capacity limits.",
    "Finite observed norms do not establish a uniform target bound or BPTT contraction.",
    "Only historical completed M1 logs/metadata are read; no GPU or model forward pass.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_bytes())


def file_sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def describe(values):
    values = sorted(values)
    require(bool(values), "empty distribution")

    def percentile(probability):
        position = (len(values) - 1) * probability
        lower = int(position)
        return values[lower] + (values[min(lower + 1, len(values) - 1)] -
                                values[lower]) * (position - lower)

    return {
        "count": len(values), "min": values[0], "p50": percentile(.5),
        "p90": percentile(.9), "p95": percentile(.95), "p99": percentile(.99),
        "max": values[-1], "mean": math.fsum(values) / len(values),
    }


def scan_log(path, expected_sha, capacity):
    digest = hashlib.sha256()
    trajectories = {}
    writes, events = defaultdict(int), defaultdict(int)
    phases, kinds = Counter(), Counter()
    strata = defaultdict(lambda: defaultdict(float))
    first_failures = monotonic_failures = nonfinite = records = raw_bytes = 0
    beta_min, beta_max = 1., 0.
    patch_max = {"value": 0.}
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rb") as stream:
        for line in stream:
            digest.update(line)
            raw_bytes += len(line)
            record = json.loads(line)
            records += 1
            require(record["protocol"] == "delta_ngram_block_metrics_v1", "unexpected log protocol")
            phase, kind = record.get("phase"), record.get("kind")
            phases[phase] += 1
            kinds[kind] += 1
            require(phase == "train", "this audit expects training-only block metrics")
            metrics = record["metrics"]
            step, source = record["step"], record["source"]
            identity = (step, record["microbatch"], record["owner"])
            band = next((name for name, (low, high) in BANDS.items() if low <= step <= high), None)
            key = (band, source, record.get("window_blocks")) if band else None
            if kind == "write":
                rows = metrics["total_written_rows"]
                old = trajectories.get(identity)
                if old is None and rows != metrics["new_rows"]:
                    first_failures += 1
                if old is not None:
                    require(old["source"] == source, "trajectory source changed")
                    if rows < old["rows"]:
                        monotonic_failures += 1
                writes[identity] += 1
                events[identity] += metrics["beta"]["count"]
                beta_min = min(beta_min, metrics["beta"]["min"])
                beta_max = max(beta_max, metrics["beta"]["max"])
                nonfinite += metrics["beta"]["nonfinite_count"] + metrics["patch_delta_row_l2"]["nonfinite_count"]
                norm = metrics.get("patch_total_norm")
                if norm is None:
                    nonfinite += 1
                elif norm > patch_max["value"]:
                    patch_max = {"value": norm, "step": step, "source": source,
                                 "window_blocks": record.get("window_blocks"), "block": record.get("block")}
                trajectories[identity] = {
                    "step": step, "source": source, "window_blocks": record.get("window_blocks"),
                    "rows": rows, "terminal_patch_frobenius": norm,
                }
                if key:
                    item = strata[key]
                    item["write_records"] += 1
                    item["event_count"] += metrics["beta"]["count"]
                    item["beta_sum"] += metrics["beta"]["sum"]
                    histogram = metrics["beta"]["histogram"]
                    require(histogram["edges"] == [0., .001, .005, .01, .025, .05, .1, .2,
                                                  .5, .8, .9, .95, .99, 1.], "beta histogram changed")
                    item["beta_lt_001_count"] += sum(histogram["counts"][:3])
                    item["beta_ge_095_count"] += sum(histogram["counts"][11:])
                    item["touched_rows_sum"] += metrics["touched_rows"]
                    item["delta_row_l2_sum"] += metrics["patch_delta_row_l2"]["sum"]
                    item["delta_rows"] += metrics["patch_delta_row_l2"]["count"]
                    if record.get("block") == 0:
                        item["first_beta_sum"] += metrics["beta"]["sum"]
                        item["first_event_count"] += metrics["beta"]["count"]
            elif kind == "query":
                if key:
                    strata[key]["query_hits"] += metrics["all"]["hit_count"]
                    strata[key]["query_addresses"] += metrics["all"]["denominator"]
            else:
                raise ValueError(f"unexpected block kind: {kind}")
    actual_sha = digest.hexdigest()
    require(actual_sha == expected_sha, f"uncompressed log hash differs: {actual_sha}")
    require(nonfinite == 0, "nonfinite logged values; cannot reproduce finite summary")
    require(0 <= beta_min <= beta_max <= 1, "out-of-range gate")
    require(first_failures == 0 and monotonic_failures == 0, "memory lifecycle invariant failed")
    by_source, by_band = defaultdict(list), defaultdict(list)
    for identity, value in trajectories.items():
        by_source[value["source"]].append((identity, value))
        for band, (low, high) in BANDS.items():
            if low <= value["step"] <= high:
                by_band[(band, value["source"])].append((identity, value))
                item = strata[(band, value["source"], value["window_blocks"])]
                rows = value["rows"]
                item["trajectories"] += 1
                item["terminal_rows_sum"] += rows
                item["terminal_rows_max"] = max(item["terminal_rows_max"], rows)
                item["terminal_patch_norm_sum"] += value["terminal_patch_frobenius"]
                item["terminal_row_rms_sum"] += value["terminal_patch_frobenius"] / math.sqrt(rows) if rows else 0.

    def trajectory_summary(items):
        return {
            "trajectories": len(items),
            "written_unique_rows": describe([value["rows"] for _, value in items]),
            "aggregate_occupancy_U_over_2B": describe([value["rows"] / capacity for _, value in items]),
            "write_calls": describe([writes[identity] for identity, _ in items]),
            "write_event_occurrences_E": describe([events[identity] for identity, _ in items]),
            "terminal_patch_frobenius": describe([value["terminal_patch_frobenius"] for _, value in items]),
        }

    def stratum_summary(key, item):
        def ratio(numerator, denominator):
            return item[numerator] / item[denominator] if item[denominator] else None
        band, source, blocks = key
        return {
            "band": band, "source": source, "window_blocks": blocks,
            "write_records": int(item["write_records"]), "trajectories": int(item["trajectories"]),
            "event_count": int(item["event_count"]), "beta_event_mean": ratio("beta_sum", "event_count"),
            "beta_lt_001_fraction": ratio("beta_lt_001_count", "event_count"),
            "beta_ge_095_fraction": ratio("beta_ge_095_count", "event_count"),
            "first_block_beta_event_mean": ratio("first_beta_sum", "first_event_count"),
            "within_block_repeated_row_event_fraction_including_true_repeats":
                1 - ratio("touched_rows_sum", "event_count") if item["event_count"] else None,
            "delta_l2_mean_per_touched_row": ratio("delta_row_l2_sum", "delta_rows"),
            "terminal_rows_mean": ratio("terminal_rows_sum", "trajectories"),
            "terminal_rows_max": int(item["terminal_rows_max"]),
            "terminal_occupancy_fraction_mean": ratio("terminal_rows_sum", "trajectories") / capacity,
            "terminal_patch_frobenius_mean": ratio("terminal_patch_norm_sum", "trajectories"),
            "terminal_row_rms_mean": ratio("terminal_row_rms_sum", "trajectories"),
            "query_address_hit_rate": ratio("query_hits", "query_addresses"),
        }

    source_sums = defaultdict(lambda: defaultdict(float))
    for (band, source, _), values in strata.items():
        item = source_sums[(band, source)]
        for name, number in values.items():
            if name == "terminal_rows_max":
                item[name] = max(item[name], number)
            else:
                item[name] += number
    observed = {
        "log_path": str(path.resolve()), "log_bytes": raw_bytes, "log_sha256": actual_sha,
        "stored_file_bytes": path.stat().st_size, "stored_file_sha256": file_sha256(path),
        "records": records, "phases": dict(phases), "kinds": dict(kinds),
        "trajectory_count": len(trajectories), "write_record_count": sum(writes.values()),
        "first_write_nonempty_failures": first_failures,
        "nonmonotone_occupancy_failures": monotonic_failures,
        "beta_min": beta_min, "beta_max": beta_max, "nonfinite_logged_count": nonfinite,
        "maximum_patch_frobenius": patch_max,
        "overall": trajectory_summary(list(trajectories.items())),
        "by_source": {source: trajectory_summary(items) for source, items in sorted(by_source.items())},
        "by_band_source": [{"band": band, "source": source, **trajectory_summary(items)}
                           for (band, source), items in sorted(by_band.items())],
    }
    temporal = {
        "bands": BANDS, "strata": [stratum_summary(key, values) for key, values in sorted(strata.items())],
        "source_summaries": [stratum_summary((band, source, None), values)
                             for (band, source), values in sorted(source_sums.items())],
        "source_sha256": actual_sha, "limitations": LIMITATIONS,
    }
    return observed, temporal


def length_bounds(manifest, buckets, orders, block_size):
    capacity = buckets * len(orders)

    def info(lengths):
        written = list(lengths)
        written[-1] -= (written[-1] - 1) % block_size + 1
        occurrences = {str(order): sum(max(0, n - order + 1) for n in written) for order in orders}
        bounds = {order: min(buckets, number) for order, number in occurrences.items()}
        return {
            "raw_tokens": sum(lengths), "write_calls": sum(math.ceil(n / block_size) for n in lengths) - 1,
            "writer_tokens": sum(written), "occupied_rows_upper_bound_by_order": bounds,
            "occupied_rows_total_upper_bound": sum(bounds.values()),
        }

    def summarize(infos):
        result = {
            "max_raw_tokens": max(item["raw_tokens"] for item in infos),
            "max_writer_tokens": max(item["writer_tokens"] for item in infos),
            "max_write_calls": max(item["write_calls"] for item in infos),
            "max_row_bound_by_order": {str(order): max(item["occupied_rows_upper_bound_by_order"][str(order)]
                                                       for item in infos) for order in orders},
            "max_total_row_bound": max(item["occupied_rows_total_upper_bound"] for item in infos),
        }
        result["max_total_occupancy_bound"] = result["max_total_row_bound"] / capacity
        return result

    validation = []
    for source in sorted({window["source"] for window in manifest["validation_sample"]}):
        windows = [window for window in manifest["validation_sample"] if window["source"] == source]
        infos = [info(window["article_tokens"] if source == "news_event" else [window["blocks"] * block_size])
                 for window in windows]
        validation.append({"source": source, "windows": len(windows), **summarize(infos)})
    events = [info([article["tokens"] for article in document["articles"]])
              for document in manifest["event_documents"] if document["split"] == "train"]
    return validation, {"event_documents": len(events), **summarize(events)}


def render_text(report):
    observed = report["observed"]
    occupancy = observed["overall"]["aggregate_occupancy_U_over_2B"]
    lines = [
        "M1 Mathematical Capacity Audit - reproducible CPU-only aggregation",
        "", "Config: " + json.dumps(report["config"], sort_keys=True),
        f"Uncompressed log SHA256: {observed['log_sha256']}",
        f"Uncompressed bytes: {observed['log_bytes']}; trajectories: {observed['trajectory_count']}; "
        f"writes: {observed['write_record_count']}",
        "", "Terminal occupancy U/(2B), trajectory weighted:",
        *[f"  {key}: {occupancy[key]:.10%}" for key in ("min", "p50", "p90", "p95", "p99", "max", "mean")],
        f"Maximum actual U: {observed['overall']['written_unique_rows']['max']}",
        f"First-write-nonempty failures: {observed['first_write_nonempty_failures']}",
        f"Nonmonotone occupancy failures: {observed['nonmonotone_occupancy_failures']}",
        f"Observed beta range: [{observed['beta_min']}, {observed['beta_max']}]",
        f"Nonfinite logged count: {observed['nonfinite_logged_count']}",
        "", "Length-based upper bounds (not measured validation occupancy):",
        *[f"  {item['source']}: T<={item['max_writer_tokens']}, writes<={item['max_write_calls']}, "
          f"U<={item['max_total_row_bound']}, q<={item['max_total_occupancy_bound']:.10%}"
          for item in report["frozen_validation_length_bounds"]],
        "", "Mathematics:",
        *[f"  {key}: {value}" for key, value in MATH.items()],
        "", "Excluded:",
        *["  " + item for item in report["conclusions"]["excluded"]],
        "", "Unresolved:",
        *["  " + item for item in report["conclusions"]["not_excluded"]],
        "", "Limitations:", *["  " + item for item in LIMITATIONS],
        "", "Code evidence:",
        *[f"  src/delta_ngram/{name}: " + ", ".join(map(str, numbers))
          for name, numbers in SOURCE_LINES.items()],
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=OPS / "diagnostic_logs/block_metrics.jsonl.gz")
    parser.add_argument("--data-manifest", type=Path, default=BACKUP / "data_manifest.json")
    parser.add_argument("--config", type=Path, default=OPS / "first_m1_checkpoint/config.json")
    parser.add_argument("--eval-dir", type=Path, default=BACKUP)
    parser.add_argument("--training-summary", type=Path, default=BACKUP / "summary.json")
    parser.add_argument("--source-log-sha256", default=EXPECTED_LOG_SHA256)
    parser.add_argument("--output-dir", type=Path, default=RUNS / "m1_analysis_20261020/diagnostics/capacity_reproduction")
    args = parser.parse_args(argv)
    started = time.time()
    config, manifest, summary = (load_json(path) for path in
                                 (args.config, args.data_manifest, args.training_summary))
    reader, raw = config["package_config"]["reader"], config["package_config"]["raw"]
    buckets, orders, block_size = reader["buckets"], reader["orders"], raw["block_size"]
    require((buckets, orders, block_size, config["width"]) == (16384, [2, 3], 512, 1024),
            "configuration differs from the frozen M1 contract")
    require(config["package_config"]["method"]["operator"] == "D1", "audit requires D1")
    capacity = buckets * len(orders)
    observed, temporal = scan_log(args.log, args.source_log_sha256, capacity)
    require(observed["trajectory_count"] == summary["counters"]["trajectories"], "trajectory log incomplete")
    require(observed["write_record_count"] == summary["counters"]["write_blocks"], "write log incomplete")
    validation, event_pool = length_bounds(manifest, buckets, orders, block_size)
    evaluations = [load_json(path) for path in sorted(args.eval_dir.glob("eval_step_*.json"))]
    require(len(evaluations) == 21, "expected 21 registered historical M1 checkpoints")
    identity_counts = {key: len({item[key] for item in evaluations}) for key in
                       ("validation_sample_sha256", "validation_protocol_sha256", "donor_sample_sha256")}
    require(all(value == 1 for value in identity_counts.values()), "fixed evaluation identity changed")
    require(observed["overall"]["written_unique_rows"]["max"] < buckets, "a single order could be full")
    require(all(max(item["max_row_bound_by_order"].values()) < buckets for item in validation),
            "validation length bounds do not exclude a full order")
    report = {
        "protocol": "mathematical_capacity_audit_20261020_v1",
        "config": {"buckets_per_order_B": buckets, "orders": orders, "number_of_orders_H": len(orders),
                   "total_nonpadding_rows_per_stream": capacity, "width": config["width"],
                   "value_dtype": "float32", "operator": "D1", "base": "implicit_zero"},
        "observed": observed, "frozen_validation_length_bounds": validation,
        "training_event_pool_length_bound": event_pool,
        "manifest": {"path": str(args.data_manifest.resolve()), "sha256": file_sha256(args.data_manifest),
                     "validation_sample_count": len(manifest["validation_sample"])},
        "fixed_checkpoint_identities": {"evaluation_files": len(evaluations), "distinct_hash_counts": identity_counts},
        "mathematics": MATH, "source_evidence": SOURCE_LINES, "limitations": LIMITATIONS,
        "conclusions": {
            "excluded": [
                "One memory table accumulating across 100M training tokens.",
                "Any logged trajectory filling even a single order's table.",
                "Full-capacity eviction/drop as the endpoint mechanism; D1 contains no such branch.",
                "Different fixed-evaluation allocation or collision positions caused by changed writer weights.",
            ],
            "not_excluded": [
                "Existing hash collision interference at low occupancy; the log lacks exact-key count K.",
                "Representational/interference capacity limits and changing values or gate responses.",
                "Target bounds, BPTT sensitivity, overfitting, or transfer mechanisms not proved by occupancy.",
            ],
        },
        "log_completeness_check": {"training_summary_trajectories": summary["counters"]["trajectories"],
                                   "training_summary_write_blocks": summary["counters"]["write_blocks"]},
        "reproduction": {"script": str(Path(__file__).resolve()), "script_sha256": file_sha256(Path(__file__)),
                         "python_version": sys.version, "analysis_seconds": time.time() - started},
    }
    require("torch" not in sys.modules, "torch must not be imported")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "mathematical_capacity_audit.json", report)
    write_json(output / "capacity_block_log_audit.json", temporal)
    (output / "mathematical_capacity_audit.txt").write_text(render_text(report), encoding="utf-8", newline="\n")
    verification = {
        "log": str(args.log.resolve()), "stored_bytes": args.log.stat().st_size,
        "stored_sha256": observed["stored_file_sha256"],
        "uncompressed_bytes": observed["log_bytes"], "uncompressed_sha256": observed["log_sha256"],
        "original_uncompressed_sha256_verified": True,
        "original_gzip_sha256_verified": (observed["stored_file_sha256"] == EXPECTED_GZIP_SHA256)
                                          if args.log.suffix.lower() == ".gz" else None,
        "source_metadata": [{"path": str(path.resolve()), "sha256": file_sha256(path)}
                            for path in (args.data_manifest, args.config, args.training_summary)],
        "outputs": [{"file": name, "sha256": file_sha256(output / name)} for name in
                    ("mathematical_capacity_audit.json", "mathematical_capacity_audit.txt", "capacity_block_log_audit.json")],
    }
    write_json(output / "verification.json", verification)
    print(json.dumps({"output": str(output), "original_sha256_verified": True,
                      "trajectories": observed["trajectory_count"], "writes": observed["write_record_count"],
                      "max_rows": observed["overall"]["written_unique_rows"]["max"],
                      "max_occupancy": observed["overall"]["aggregate_occupancy_U_over_2B"]["max"],
                      "analysis_seconds": time.time() - started}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

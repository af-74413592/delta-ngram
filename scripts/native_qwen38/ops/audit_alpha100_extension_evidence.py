"""Recompute stored DEV64 improvement and observed training time without CE."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random


def info(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluations = args.root / "reports/native_main_training_07/evaluations"
    paths = [next(evaluations.glob(pattern)) for pattern in
             ("stage-00-alpha100*/report.json", "stage-02-alpha100*/report.json")]
    reports = [json.loads(path.read_bytes()) for path in paths]
    indexed = []
    for report in reports:
        assert report["complete"] is True and report["arm"] == "alpha100_rmsnorm"
        assert report["alpha"] == 1.0 and report["independent_families"] == 64
        index = {(row["event_family"], row["unrelated_article_count"]): row for row in report["per_family"]}
        assert len(index) == 320
        indexed.append(index)
    assert set(indexed[0]) == set(indexed[1])
    families = sorted({key[0] for key in indexed[0]})
    curves = []
    for lag in (0, 1, 4, 8, 16):
        old, new = [], []
        for family in families:
            a, b = indexed[0][family, lag], indexed[1][family, lag]
            for key in ("pair_id", "later_source_id", "native_prefix_annotation"):
                assert a.get(key) == b.get(key), (family, lag, key)
            for condition in ("self", "donor"):
                assert a["conditions"][condition]["target_tokens"] == b["conditions"][condition]["target_tokens"]
                assert a["conditions"][condition]["write_source_ids"] == b["conditions"][condition]["write_source_ids"]
            old.append(a["delta_donor"])
            new.append(b["delta_donor"])
        delta = [b - a for a, b in zip(old, new)]
        rng = random.Random(20261019)
        draws = [math.fsum(delta[rng.randrange(len(families))] for _ in families) / len(families)
                 for _ in range(10000)]
        curves.append({"lag": lag, "families": len(families),
                       "20M_donor_minus_self": math.fsum(old) / len(old),
                       "40M_donor_minus_self": math.fsum(new) / len(new),
                       "paired_40M_minus_20M": math.fsum(delta) / len(delta),
                       "paired_ci95": [percentile(draws, .025), percentile(draws, .975)]})
    timing = []
    for number in (5, 6, 7):
        run = args.root / f"reports/native_main_training_{number:02d}"
        path = run / "training.jsonl"
        groups = [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
        arms = {}
        for name in ("alpha100_rmsnorm", "alpha099_rmsnorm"):
            rows = [row for row in groups if row["arm"] == name]
            seconds = math.fsum(row["seconds"] for row in rows)
            arms[name] = {"completed_groups": len(rows),
                          "completed_targets_in_logged_groups": sum(row["loss_target_tokens"] for row in rows),
                          "group_wall_seconds_including_checkpoint": seconds,
                          "group_reserved_gpu_hours": seconds * 4 / 3600}
        evals = []
        for path in sorted((run / "evaluations").glob("*/report.json")):
            report = json.loads(path.read_bytes())
            evals.append({"arm": report["arm"], "milestone": report["milestone"],
                          "seconds": report["seconds"], "reserved_gpu_hours": report["seconds"] * 4 / 3600,
                          "report": info(path)})
        timing.append({"run": str(run), "arms": arms, "completed_DEV64": evals,
                       "incomplete_groups_and_failed_eval_time": "not available from these completed-operation logs"})
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "new_CE_targets": 0,
              "optimizer_updates": 0, "source_reports": [info(path) for path in paths],
              "comparison": curves,
              "bootstrap": {"unit": "paired event family", "seed": 20261019, "replicates": 10000,
                            "confidence": .95, "interval": "linearly interpolated percentile"},
              "timing_snapshot": timing,
              "timing_definition": "four GPUs reserved during each logged operation; not kernel-active time",
              "timing_scope": "completed operations in runs05/06/07 only; source07 can still be running",
              "timing_additivity": "group/eval operation times are components, never add them to whole-job occupancy",
              "rationale_scope": "observed DEV64 improvement; exploratory post-result budget deviation, not FINAL evidence"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": info(args.output), "lag0": curves[0]}), flush=True)


if __name__ == "__main__":
    main()

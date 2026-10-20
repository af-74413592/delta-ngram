"""CPU-only exact continuation forecast from an atomic checkpoint snapshot."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from delta_ngram.qwen38_train_data import NativeFullTrainingData

    # A single read binds the state to its digest even while latest is replaced.
    raw = args.checkpoint.read_bytes()
    source = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    data = NativeFullTrainingData(args.cache_root, verify_files=False)
    stats = data.manifest["statistics"]

    def counts(state):
        rows = []
        for cell in state["cells"]:
            capacity = stats[cell["source"]]["train"]["window_loss_tokens"][str(cell["blocks"])]
            windows = stats[cell["source"]]["train"]["window_count"][str(cell["blocks"])]
            assert len(cell["order"]) == windows and capacity > 0
            draws = cell["epoch"] * windows + cell["cursor"]
            rows.append({"source": cell["source"], "blocks": cell["blocks"],
                         "loss_tokens": cell["loss_tokens"], "capacity_loss_tokens": capacity,
                         "token_repeats": cell["loss_tokens"] / capacity,
                         "draws": draws, "windows": windows, "draw_passes": draws / windows,
                         "event_within_three_passes": (cell["loss_tokens"] <= 3 * capacity
                                                        and draws <= 3 * windows)
                         if cell["source"] == "news_event" else None})
        return rows

    results = {}
    for name, arm in source["arms"].items():
        state = arm["sampler"]
        sampler = data.sampler(seed=17)
        sampler.load_state_dict(state)
        assert state["total_loss_tokens"] == arm["completed_loss_targets"]
        targets, step = arm["completed_loss_targets"], arm["optimizer_step"]
        result = {"source_model_targets": targets, "source_optimizer_step": step,
                  "source_sampler_fingerprint": state["fingerprint"],
                  "source_sampler": copy.deepcopy(state), "source_repeats": counts(state),
                  "projections": []}
        if name == "alpha100_rmsnorm":
            for nominal in (60000000, 80000000, 100000000):
                while True:
                    before = sampler.state_dict()
                    group = sampler.next_batch(12)
                    assert len(group) == 12
                    cost = sum(item.loss_tokens for item in group)
                    after = sampler.state_dict()
                    rows = counts(after)
                    event_ok = all(row["event_within_three_passes"] is not False for row in rows)
                    if targets + cost > nominal or not event_ok:
                        sampler.load_state_dict(before)
                        result["projections"].append({
                            "nominal_model_targets": nominal, "actual_model_targets": targets,
                            "optimizer_step": step, "next_whole_group_cost": cost,
                            "stop_reason": "model_milestone" if event_ok else "event_three_pass_limit",
                            "repeats": counts(before), "sampler": before,
                            "minimum_lifetime_optimizer_accesses": targets + 1468915})
                        break
                    targets += cost
                    step += 1
                if not event_ok:
                    break
        results[name] = result
    output = {"protocol": "qwen38_native_exact_sampler_forecast_v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint": {"path": str(args.checkpoint), "bytes": len(raw),
                             "sha256": hashlib.sha256(raw).hexdigest()},
              "cache_identity": data.identity, "arms": results,
              "new_CE_targets": 0, "optimizer_updates": 0,
              "forecast_only": True, "group_size": 12,
              "assumptions": ["Existing sampler, independent arm states, unchanged full groups",
                              "DEV64 preserves sampler and training RNG",
                              "No future failed query charges included in minimum lifetime total"],
              "repeat_scope": "persisted sampler lineages; discarded historical runs are separate charges"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "arms": {name: {"source_model_targets": row["source_model_targets"],
                                       "projections": [{key: value for key, value in item.items()
                                                        if key not in ("sampler", "repeats")}
                                                       for item in row["projections"]]}
                               for name, row in results.items()}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Cross-run P1 decisions, endpoint provenance, and strict JSON rejection."""
import importlib.util
import json
from pathlib import Path
import random

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_p1.py"
SPEC = importlib.util.spec_from_file_location("summarize_p1", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_run(tmp_path, name, seed, *, direction_delta=.02, variable_groups=False):
    path = tmp_path / name
    path.mkdir()
    raw = {"block_size": 512, "prefix_tokens": 128, "block_counts": [8, 16],
           "length_weights": {"8": .5, "16": .5}, "source_weights": {"code/python": 1.},
           "batch_size": 4, "gradient_accumulation_steps": 1, "checkpoint_reader": True,
           "loss_tokens": 5_000_000, "eval_every_tokens": 2_500_000,
           "eval_seed": 104729, "eval_windows": 64, "eval_bootstrap_samples": 39, "resume": None}
    model = {"name_or_path": "Qwen/Qwen3-0.6B", "revision": "c" * 40, "smoke": False}
    package = {"seed": seed, "raw": raw, "model": model,
               "reader": {"backend": "external_hash", "buckets": 16384, "orders": [2, 3], "layer_index": 1},
               "writer": {"hidden_dim": 128, "right_window": 4}, "method": {"name": "forward", "operator": "D1"},
               "train": {"lr": .0003, "gradient_clip": 1.}, "run": {"checkpoint": None, "output_dir": str(path)}}
    protocol = {"name": "group_first_fixed_dev_paired_bootstrap_v1", "selection": "group_first_round_robin",
                "group_key": "group_id", "seed": 104729, "window_limit": 64, "bootstrap_samples": 39,
                "minimum_groups": 64, "evaluation_batching": "within_group", "batch_size": 4, "prefix_tokens": 128}
    sample = [{"group_id": f"g{i:02d}", "document_id": f"d{i}", "source": "code/python",
               "start": 0, "blocks": 8, "loss_tokens": 3584, "token_sha256": "b" * 64} for i in range(64)]
    hashes = {"validation_protocol_sha256": audit.digest(protocol), "validation_sample_sha256": audit.digest(sample)}
    tokenizer = {"class": "Qwen2Tokenizer", "vocab_sha256": "d" * 64, "pipeline_sha256": "e" * 64,
                 "special_tokens": {"pad_token_id": 1, "eos_token_id": 2, "bos_token_id": None}}
    identity = {"protocol": "delta_ngram_raw_lm_v1", "seed": seed, "dataset_sha256": "a" * 64,
                "tokenizer": tokenizer, "model": model, "resolved_model_commit": model["revision"],
                "device_type": "cuda", "dtype": "torch.bfloat16", "reader": package["reader"],
                "writer": package["writer"], "method": package["method"],
                "optimizer": {"name": "AdamW", "weight_decay": 0., **package["train"]},
                "raw": {key: raw[key] for key in audit.RAW_KEYS}, "validation": hashes}
    config = {"raw_identity": identity, "package_config": package, "protocol": identity["protocol"],
              "resolved_model_commit": model["revision"], "tokenizer_fingerprint": "sha256:" + audit.digest(tokenizer),
              "validation_protocol": protocol, **hashes}
    manifest = {"dataset_sha256": identity["dataset_sha256"], "tokenizer": tokenizer,
                "validation_protocol": protocol, "validation_sample": sample, **hashes,
                "windows_by_blocks": {"8": 1000, "16": 1000}}
    save(path / "config.json", config)
    save(path / "data_manifest.json", manifest)
    save(path.with_name(name + ".launch.json"), {"source_sha256": {"delta_ngram/raw_lm.py": "f" * 64}})
    events = [{"event": "start", "step": 0, "counters": {"loss_tokens": 0}}]
    total, step = 0, 0
    while total < audit.BUDGET:
        block_choices = ([16] * (2 if step < 35 else 1) + [8] * (2 if step < 35 else 3)) if step < 128 else [8, 16, 8, 16]
        blocks = []
        for count in block_choices:
            if total + sum((n - 1) * 512 for n in blocks) + (count - 1) * 512 <= audit.BUDGET:
                blocks.append(count)
            elif total + sum((n - 1) * 512 for n in blocks) + 3584 <= audit.BUDGET:
                blocks.append(8)
        if not blocks:
            break
        batch = sum((n - 1) * 512 for n in blocks)
        step, total = step + 1, total + batch
        events.append({"event": "train", "step": step, "batch_loss_tokens": batch,
                       "counters": {"loss_tokens": total}, "windows": [{"blocks": n} for n in blocks]})

    def evaluation(event, delta, *, final=False):
        group_deltas = [delta + (.1 if i % 2 else -.1) if variable_groups and final else delta for i in range(64)]
        groups = [{"group_id": f"g{i:02d}", "loss_tokens": 3584,
                   "self_loss_sum": 3584., "empty_loss_sum": 3584 * (1 + value)} for i, value in enumerate(group_deltas)]
        pairs = [(g["empty_loss_sum"] - g["self_loss_sum"], g["loss_tokens"]) for g in groups]
        rng = random.Random(104729)
        draws = sorted(sum(pairs[rng.randrange(64)][0] for _ in range(64)) / (64 * 3584) for _ in range(39))
        bounds = []
        for p in (.025, .975):
            location = 38 * p
            low = int(location)
            bounds.append(draws[low] + (draws[low + 1] - draws[low]) * (location - low))
        empty = sum(g["empty_loss_sum"] for g in groups) / (64 * 3584)
        value = {"step": event["step"], "training_loss_tokens": event["counters"]["loss_tokens"],
                 "evaluation_kind": "final" if final else "milestone", "nominal_loss_tokens": audit.BUDGET if final else audit.MIDPOINT,
                 "requested_loss_tokens": audit.BUDGET, "validation_protocol": protocol, **hashes,
                 "groups": groups, "group_count": 64, "loss_tokens": 64 * 3584,
                 "self_nll": 1., "empty_nll": empty, "delta_empty": empty - 1.,
                 "paired_group_bootstrap": {"method": "paired_group_percentile_ratio_of_sums", "confidence": .95,
                                            "seed": 104729, "replicates": 39, "ci_lower": bounds[0], "ci_upper": bounds[1]},
                 "p1_gate_passed": False, "p1_point_check": {"passed": False}}
        if final:
            value.update(is_final=True, budget_endpoint=True, final_nominal_loss_tokens=audit.BUDGET,
                         unspent_loss_tokens=audit.BUDGET - total)
        save(path / f"eval_step_{event['step']:06d}.json", value)
        return value

    crossing = next(e for e in events[1:] if e["counters"]["loss_tokens"] >= audit.MIDPOINT)
    assert crossing["counters"]["loss_tokens"] == 2_502_656
    midpoint = evaluation(crossing, .02)
    events.insert(events.index(crossing) + 1, {"event": "evaluation", **midpoint})
    final = evaluation(events[-1], direction_delta, final=True)
    summary = {"completed_steps": step, "counters": {"loss_tokens": total}, "unspent_loss_tokens": audit.BUDGET - total,
               "requested_loss_tokens": audit.BUDGET, "nominal_loss_tokens": audit.BUDGET, "budget_endpoint": True,
               "stop_reason": "remaining_budget_below_complete_trajectory", "complete": False, "last_evaluation": final}
    save(path / "summary.json", summary)
    events.append({"event": "final_evaluation", **final})
    (path / "metrics.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return path


def test_normal_tail_and_midpoint_overshoot_pass_with_engineering_separate(tmp_path):
    main = make_run(tmp_path, "main", 17)
    replica = make_run(tmp_path, "replica", 29)
    report = audit.summarize(main, replica)
    assert report["statistical_gate_passed"] is True
    assert report["main_run"]["midpoint"]["actual_loss_tokens"] == 2_502_656
    end = report["main_run"]["endpoint"]
    assert 0 < end["unspent_loss_tokens"] < end["minimum_trajectory_loss_tokens"]
    assert end["actual_loss_tokens"] + end["unspent_loss_tokens"] == 5_000_000
    assert report["engineering_acceptance"]["status"] == "not_evaluated"
    assert report["p1_gate_passed"] is None


def test_missing_second_seed_is_pending_and_single_point_flags_are_ignored(tmp_path):
    report = audit.summarize(make_run(tmp_path, "main", 17))
    assert report["statistical_gate_status"] == "pending"
    assert report["statistical_gate_passed"] is None
    assert report["main_run"]["midpoint"]["passed"] is True
    assert report["main_run"]["endpoint"]["passed"] is True


def test_running_main_retains_midpoint_and_reports_pending_endpoint(tmp_path):
    main = make_run(tmp_path, "main", 17)
    (main / "summary.json").unlink()
    report = audit.summarize(main)
    assert report["main_run"]["midpoint"]["passed"] is True
    assert report["main_run"]["endpoint"]["status"] == "pending"


def test_same_seed_is_invalid(tmp_path):
    report = audit.summarize(make_run(tmp_path, "main", 17), make_run(tmp_path, "replica", 17))
    assert report["statistical_gate_status"] == "invalid"
    assert "seed_must_differ" in report["cross_run_identity"]["differences"]


def test_different_fixed_sample_is_invalid_even_when_all_hashes_are_consistent(tmp_path):
    main, replica = make_run(tmp_path, "main", 17), make_run(tmp_path, "replica", 29)
    manifest = read(replica / "data_manifest.json")
    old = manifest["validation_sample_sha256"]
    manifest["validation_sample"][0]["token_sha256"] = "9" * 64
    new = audit.digest(manifest["validation_sample"])
    save(replica / "data_manifest.json", manifest)
    for path in replica.glob("*.json*"):
        path.write_text(path.read_text().replace(old, new), encoding="utf-8")
    report = audit.summarize(main, replica)
    assert report["statistical_gate_status"] == "invalid"
    assert report["replication_run"]["endpoint"]["status"] == "passed"
    assert "validation" in report["cross_run_identity"]["differences"]


@pytest.mark.parametrize("delta,expected", [(.0001, True), (0., False), (-.01, False)])
def test_replication_checks_direction_without_new_effect_or_ci_threshold(tmp_path, delta, expected):
    report = audit.summarize(make_run(tmp_path, "main", 17),
                             make_run(tmp_path, "replica", 29, direction_delta=delta, variable_groups=True))
    assert report["statistical_gate_passed"] is expected
    assert report["replication_run"]["endpoint"]["ci_lower"] < 0


@pytest.mark.parametrize("mutation", ["wall", "short", "step", "not_final", "resume", "writer_init", "ledger"])
def test_interruption_or_false_endpoint_is_rejected(tmp_path, mutation):
    path = make_run(tmp_path, "main", 17)
    if mutation in ("resume", "writer_init"):
        config = read(path / "config.json")
        section, key = ("raw", "resume") if mutation == "resume" else ("run", "checkpoint")
        config["package_config"][section][key] = "earlier.pt"
        save(path / "config.json", config)
    elif mutation == "ledger":
        events = (path / "metrics.jsonl").read_text().splitlines()
        del events[3]
        (path / "metrics.jsonl").write_text("\n".join(events), encoding="utf-8")
    else:
        summary = read(path / "summary.json")
        if mutation == "wall":
            summary["stop_reason"] = "wall_limit"
        elif mutation == "step":
            summary["stop_reason"] = "step_limit"
        elif mutation == "short":
            summary["counters"]["loss_tokens"] -= 3584
            summary["unspent_loss_tokens"] += 3584
        else:
            summary["last_evaluation"]["is_final"] = False
        save(path / "summary.json", summary)
    assert audit.summarize(path)["statistical_gate_status"] == "invalid"


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400", "9" * 400, '"0.02"', "true"])
def test_nonfinite_or_nonnumeric_metric_is_invalid(tmp_path, literal):
    path = make_run(tmp_path, "main", 17)
    point = path / "eval_step_000128.json"
    value = read(point)
    value["delta_empty"] = "REPLACE"
    point.write_text(json.dumps(value).replace('"REPLACE"', literal), encoding="utf-8")
    assert audit.summarize(path)["main_run"]["midpoint"]["status"] == "invalid"


def test_missing_metric_is_pending_and_duplicate_key_is_invalid(tmp_path):
    path = make_run(tmp_path, "main", 17)
    point = path / "eval_step_000128.json"
    value = read(point)
    del value["paired_group_bootstrap"]
    save(point, value)
    assert audit.summarize(path)["main_run"]["midpoint"]["status"] == "pending"
    point.write_text('{"delta_empty": 1, "delta_empty": 2}', encoding="utf-8")
    assert audit.summarize(path)["main_run"]["midpoint"]["status"] == "invalid"


def test_zero_recomputed_ci_cannot_pass_via_recorded_roundoff(tmp_path):
    path = make_run(tmp_path, "main", 17)
    point = path / "eval_step_000128.json"
    value = read(point)
    for i, group in enumerate(value["groups"]):
        group["empty_loss_sum"] = group["self_loss_sum"] + (3584 * .128 if i == 0 else 0)
    value.update(empty_nll=1.002, delta_empty=.002)
    rng = random.Random(104729)
    counts = sorted(sum(rng.randrange(64) == 0 for _ in range(64)) for _ in range(39))
    assert counts[0] == counts[1] == 0
    value["paired_group_bootstrap"]["ci_lower"] = 5e-13
    value["paired_group_bootstrap"]["ci_upper"] = (counts[37] * .95 + counts[38] * .05) * .002
    save(point, value)
    log = path / "metrics.jsonl"
    events = [json.loads(line) for line in log.read_text().splitlines()]
    events = [{"event": "evaluation", **value} if event["event"] == "evaluation" else event for event in events]
    log.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    result = audit.summarize(path)["main_run"]["midpoint"]
    assert result["status"] == "not_met" and result["ci_lower"] == 0


def test_midpoint_log_disagreement_and_missing_active_length_are_invalid(tmp_path):
    path = make_run(tmp_path, "main", 17)
    log = path / "metrics.jsonl"
    events = [json.loads(line) for line in log.read_text().splitlines()]
    for event in events:
        if event["event"] == "evaluation":
            event["delta_empty"] = -.01
    log.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    assert audit.summarize(path)["main_run"]["midpoint"]["status"] == "invalid"
    manifest = read(path / "data_manifest.json")
    manifest["windows_by_blocks"]["8"] = 0
    save(path / "data_manifest.json", manifest)
    assert audit.summarize(path)["main_run"]["endpoint"]["status"] == "invalid"


def test_source_drift_is_invalid_and_cli_writes_auditable_json(tmp_path, capsys):
    main, replica = make_run(tmp_path, "main", 17), make_run(tmp_path, "replica", 29)
    save(replica.with_name("replica.launch.json"), {"source_sha256": {"delta_ngram/raw_lm.py": "1" * 64}})
    output = tmp_path / "audit.json"
    assert audit.main(["--main-run", str(main), "--replication-run", str(replica), "--output", str(output)]) == 1
    report = read(output)
    assert json.loads(capsys.readouterr().out) == report
    assert report["cross_run_identity"]["differences"] == ["source_sha256"]
    assert all(audit.valid_hash(item["sha256"]) for item in report["main_run"]["inputs"].values())

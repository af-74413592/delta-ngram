"""Audit the registered P1 development checks using JSON artifacts only.

This never reads corpus/holdout data or loads model checkpoints. Statistical
success is separate from the externally assessed causal/engineering checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random


BUDGET = 5_000_000
MIDPOINT = 2_500_000
HASH_KEYS = ("validation_protocol_sha256", "validation_sample_sha256")
RAW_KEYS = ("block_size", "prefix_tokens", "block_counts", "length_weights",
            "source_weights", "batch_size", "gradient_accumulation_steps", "checkpoint_reader")


class AuditError(ValueError):
    def __init__(self, message, status="invalid"):
        super().__init__(message)
        self.status = status


def require(condition, message):
    if not condition:
        raise AuditError(message)


def field(mapping, key):
    require(isinstance(mapping, dict), f"expected object containing {key}")
    if key not in mapping:
        raise AuditError(f"missing field: {key}", "pending")
    return mapping[key]


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"{label} must be a finite number")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite(value):
        if type(value) in (int, float):
            try:
                is_finite = math.isfinite(value)
            except OverflowError:
                is_finite = False
            require(is_finite, "non-finite or unrepresentable JSON number")
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)

    value = json.loads(text, object_pairs_hook=pairs)
    finite(value)
    return value


def read_artifact(path, inputs, *, lines=False):
    try:
        content = path.read_bytes()
    except FileNotFoundError as error:
        raise AuditError(f"missing artifact: {path.name}", "pending") from error
    except OSError as error:
        raise AuditError(f"cannot read {path.name}: {error}") from error
    inputs[path.name] = {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
    try:
        text = content.decode("utf-8")
        return [strict_json(line) for line in text.splitlines() if line.strip()] if lines else strict_json(text)
    except (ValueError, UnicodeError) as error:
        raise AuditError(f"{path.name}: {error}") from error


def check_identity(config, manifest, launch):
    identity = field(config, "raw_identity")
    package = field(config, "package_config")
    raw = field(package, "raw")
    for key in ("protocol", "seed", "dataset_sha256", "tokenizer", "model", "resolved_model_commit",
                "device_type", "dtype", "reader", "writer", "method", "optimizer", "raw", "validation"):
        field(identity, key)
    seed = integer(identity["seed"], "seed")
    require(seed == field(package, "seed"), "package/identity seed mismatch")
    require(identity["protocol"] == field(config, "protocol") == "delta_ngram_raw_lm_v1", "unexpected raw protocol")
    require(valid_hash(identity["dataset_sha256"]), "invalid dataset SHA256")
    require(identity["dataset_sha256"] == field(manifest, "dataset_sha256"), "dataset identity mismatch")
    require(identity["tokenizer"] == field(manifest, "tokenizer"), "tokenizer identity mismatch")
    tokenizer = identity["tokenizer"]
    require(valid_hash(field(tokenizer, "vocab_sha256")), "missing tokenizer vocabulary fingerprint")
    require(valid_hash(field(tokenizer, "pipeline_sha256")), "missing tokenizer pipeline fingerprint")
    field(tokenizer, "class")
    field(tokenizer, "special_tokens")
    require(field(config, "tokenizer_fingerprint") == "sha256:" + digest(tokenizer), "tokenizer hash mismatch")
    model = field(package, "model")
    require(identity["model"] == {key: field(model, key) for key in ("name_or_path", "revision", "smoke")},
            "package/model identity mismatch")
    require(model["name_or_path"] == "Qwen/Qwen3-0.6B" and model["smoke"] is False,
            "P1 requires the deployment Qwen3-0.6B reader, not smoke/Base")
    require(isinstance(model["revision"], str) and len(model["revision"]) == 40 and
            identity["resolved_model_commit"] == model["revision"] == field(config, "resolved_model_commit"),
            "reader revision is not pinned/resolved consistently")
    for key in ("reader", "writer", "method"):
        require(identity[key] == field(package, key), f"package/{key} identity mismatch")
    require(field(identity["method"], "name") == "forward" and field(identity["method"], "operator") == "D1",
            "P1 requires the forward D1 method")
    require(identity["raw"] == {key: field(raw, key) for key in RAW_KEYS}, "package/raw identity mismatch")
    train = field(package, "train")
    require(identity["optimizer"] == {"name": "AdamW", "weight_decay": 0.0,
                                      "lr": field(train, "lr"), "gradient_clip": field(train, "gradient_clip")},
            "package/optimizer identity mismatch")
    require(raw["block_size"] == 512 and raw["prefix_tokens"] == 128 and raw["block_counts"] == [8, 16],
            "P1 requires the registered 512/128, 8/16-block recipe")
    require(raw["length_weights"] == {"8": .5, "16": .5}, "P1 requires equal target-token length weights")
    require(integer(field(raw, "loss_tokens"), "budget", 1) == BUDGET and
            integer(field(raw, "eval_every_tokens"), "evaluation interval", 1) == MIDPOINT,
            "P1 requires the 5M budget and prescheduled 2.5M interval")
    require(field(raw, "resume") is None and field(field(package, "run"), "checkpoint") is None,
            "P1 run must start fresh without resume or writer initialization")
    for key in ("batch_size", "gradient_accumulation_steps"):
        integer(raw[key], key, 1)
    protocol = field(config, "validation_protocol")
    expected_protocol = {"name": "group_first_fixed_dev_paired_bootstrap_v1",
                         "selection": "group_first_round_robin", "group_key": "group_id",
                         "seed": field(raw, "eval_seed"), "window_limit": field(raw, "eval_windows"),
                         "bootstrap_samples": field(raw, "eval_bootstrap_samples"), "minimum_groups": 64,
                         "evaluation_batching": "within_group", "batch_size": raw["batch_size"],
                         "prefix_tokens": raw["prefix_tokens"]}
    require(protocol == expected_protocol == field(manifest, "validation_protocol"), "validation protocol mismatch")
    integer(protocol["seed"], "evaluation seed")
    integer(protocol["bootstrap_samples"], "bootstrap samples", 1)
    sample = field(manifest, "validation_sample")
    require(isinstance(sample, list) and sample, "validation sample must be nonempty")
    hashes = {HASH_KEYS[0]: digest(protocol), HASH_KEYS[1]: digest(sample)}
    require(field(identity, "validation") == hashes, "validation identity/hash mismatch")
    for key in HASH_KEYS:
        require(field(config, key) == field(manifest, key) == hashes[key], f"{key} mismatch")
    sources = field(launch, "source_sha256")
    require(isinstance(sources, dict) and sources and all(valid_hash(v) for v in sources.values()),
            "launch requires declared source SHA256 fingerprints")
    return identity, raw, protocol, sample, sources


def check_log(events, raw):
    starts = [event for event in events if field(event, "event") == "start"]
    require(len(starts) == 1 and events[0] is starts[0] and field(starts[0], "step") == 0 and
            field(field(starts[0], "counters"), "loss_tokens") == 0, "run lacks one fresh step-zero start")
    steps, previous = {}, 0
    for event in events:
        if event["event"] != "train":
            continue
        step = integer(field(event, "step"), "training step", 1)
        actual = integer(field(field(event, "counters"), "loss_tokens"), "training loss tokens", 1)
        batch = integer(field(event, "batch_loss_tokens"), "batch loss tokens", 1)
        windows = field(event, "windows")
        require(isinstance(windows, list) and 0 < len(windows) <= raw["batch_size"] * raw["gradient_accumulation_steps"],
                "invalid optimizer-group windows")
        blocks = [integer(field(item, "blocks"), "trajectory blocks", 2) for item in windows]
        require(all(value in raw["block_counts"] for value in blocks), "unregistered trajectory length")
        require(step == len(steps) + 1 and actual - previous == batch == sum((n - 1) * raw["block_size"] for n in blocks),
                "noncontiguous training steps or inconsistent token ledger")
        require(actual <= BUDGET, "training exceeded the P1 budget")
        steps[step], previous = actual, actual
    return steps


def check_point(evaluation, identity, protocol, sample, *, direction_only=False):
    for key in HASH_KEYS:
        require(field(evaluation, key) == identity["validation"][key], f"evaluation {key} mismatch")
    require(field(evaluation, "validation_protocol") == protocol, "evaluation protocol mismatch")
    require(field(evaluation, "requested_loss_tokens") == BUDGET, "evaluation budget mismatch")
    groups = field(evaluation, "groups")
    require(isinstance(groups, list) and groups, "evaluation groups must be nonempty")
    sample_counts = {}
    for item in sample:
        group = field(item, "group_id")
        require(isinstance(group, str) and group, "invalid validation group")
        sample_counts[group] = sample_counts.get(group, 0) + integer(field(item, "loss_tokens"), "sample loss tokens", 1)
    actual_counts, paired, own, empty = {}, [], [], []
    for group in sorted(groups, key=lambda item: str(field(item, "group_id"))):
        name = field(group, "group_id")
        require(isinstance(name, str) and name not in actual_counts, "duplicate/invalid evaluation group")
        count = integer(field(group, "loss_tokens"), "group loss tokens", 1)
        left, right = (number(field(group, key), key) for key in ("self_loss_sum", "empty_loss_sum"))
        require(min(left, right) >= 0, "negative group loss")
        actual_counts[name] = count
        paired.append((right - left, count))
        own.append(left)
        empty.append(right)
    require(actual_counts == sample_counts, "evaluation groups/target counts differ from fixed sample")
    count = sum(actual_counts.values())
    require(integer(field(evaluation, "group_count"), "group_count", 1) == len(groups) and
            integer(field(evaluation, "loss_tokens"), "evaluation loss tokens", 1) == count,
            "evaluation aggregate counts mismatch")
    delta = (math.fsum(empty) - math.fsum(own)) / count
    for key, expected in (("self_nll", math.fsum(own) / count), ("empty_nll", math.fsum(empty) / count),
                          ("delta_empty", delta)):
        require(math.isclose(number(field(evaluation, key), key), expected, rel_tol=1e-10, abs_tol=1e-12),
                f"{key} disagrees with paired group sums")
    bootstrap = field(evaluation, "paired_group_bootstrap")
    require(field(bootstrap, "method") == "paired_group_percentile_ratio_of_sums" and
            number(field(bootstrap, "confidence"), "confidence") == .95 and
            field(bootstrap, "seed") == protocol["seed"] and
            field(bootstrap, "replicates") == protocol["bootstrap_samples"], "bootstrap protocol mismatch")
    rng, draws = random.Random(protocol["seed"]), []
    for _ in range(protocol["bootstrap_samples"]):
        draw = [paired[rng.randrange(len(paired))] for _ in paired]
        draws.append(math.fsum(value for value, _ in draw) / sum(tokens for _, tokens in draw))
    draws.sort()
    bounds = []
    for probability, key in ((.025, "ci_lower"), (.975, "ci_upper")):
        position = (len(draws) - 1) * probability
        low = int(position)
        fraction = position - low
        expected = draws[low] * (1 - fraction) + draws[min(low + 1, len(draws) - 1)] * fraction
        value = number(field(bootstrap, key), key)
        require(math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-12), f"{key} disagrees with paired bootstrap")
        bounds.append(expected)
    passed = delta > 0 if direction_only else delta >= .001 and bounds[0] > 0
    status = "pending" if len(groups) < 64 else "passed" if passed else "not_met"
    return {"status": status, "passed": None if status == "pending" else passed,
            "reason": "insufficient independent groups" if status == "pending" else "direction only" if direction_only else "main-run point threshold",
            "step": integer(field(evaluation, "step"), "evaluation step", 1),
            "actual_loss_tokens": integer(field(evaluation, "training_loss_tokens"), "evaluation training tokens", 1),
            "delta_empty": delta, "ci_lower": bounds[0], "ci_upper": bounds[1], "group_count": len(groups)}


def issue(error):
    return {"status": error.status, "passed": None, "reason": str(error)}


def audit_run(path, *, replication=False):
    path = Path(path).resolve()
    result = {"path": str(path), "inputs": {}, "identity_check": {"status": "pending"}}
    try:
        config = read_artifact(path / "config.json", result["inputs"])
        manifest = read_artifact(path / "data_manifest.json", result["inputs"])
        launch = read_artifact(path.with_name(path.name + ".launch.json"), result["inputs"])
        identity, raw, protocol, sample, sources = check_identity(config, manifest, launch)
        events = read_artifact(path / "metrics.jsonl", result["inputs"], lines=True)
        steps = check_log(events, raw)
        result.update(seed=identity["seed"], identity=identity, source_sha256=sources)
        result["identity_check"] = {"status": "passed", "fresh_start": True}
    except AuditError as error:
        result["identity_check"] = issue(error)
        return result
    if not replication:
        try:
            crossing = next((step for step, count in steps.items() if count >= MIDPOINT), None)
            if crossing is None:
                raise AuditError("2.5M optimizer boundary not reached", "pending")
            evaluation = read_artifact(path / f"eval_step_{crossing:06d}.json", result["inputs"])
            require(field(evaluation, "evaluation_kind") == "milestone" and
                    field(evaluation, "nominal_loss_tokens") == MIDPOINT and
                    field(evaluation, "step") == crossing and
                    field(evaluation, "training_loss_tokens") == steps[crossing], "not the first 2.5M milestone boundary")
            point = check_point(evaluation, identity, protocol, sample)
            final_keys = {"is_final", "budget_endpoint", "final_nominal_loss_tokens", "unspent_loss_tokens"}
            midpoint_events = [{k: v for k, v in event.items() if k != "event"} for event in events
                               if event["event"] == "evaluation" and field(event, "step") == crossing]
            require(midpoint_events == [{k: v for k, v in evaluation.items() if k not in final_keys}],
                    "missing/inconsistent 2.5M evaluation log evidence")
            result["midpoint"] = {**point, "nominal_loss_tokens": MIDPOINT}
        except AuditError as error:
            result["midpoint"] = issue(error)
    try:
        summary = read_artifact(path / "summary.json", result["inputs"])
        step = integer(field(summary, "completed_steps"), "completed steps", 1)
        actual = integer(field(field(summary, "counters"), "loss_tokens"), "completed loss tokens", 1)
        unspent = integer(field(summary, "unspent_loss_tokens"), "unspent tokens")
        require(field(summary, "requested_loss_tokens") == field(summary, "nominal_loss_tokens") == BUDGET and
                actual + unspent == BUDGET and field(summary, "budget_endpoint") is True,
                "summary is not a nominal 5M budget endpoint")
        available = field(manifest, "windows_by_blocks")
        for blocks in raw["block_counts"]:
            integer(field(available, str(blocks)), "active length available windows", 1)
        costs = [(blocks - 1) * raw["block_size"] for blocks in raw["block_counts"]]
        reason = field(summary, "stop_reason")
        require((reason == "loss_token_budget" and field(summary, "complete") is True and unspent == 0) or
                (reason == "remaining_budget_below_complete_trajectory" and field(summary, "complete") is False and
                 0 < unspent < min(costs)), "interrupted run or invalid unspendable tail")
        require(step == len(steps) and steps.get(step) == actual, "summary disagrees with final optimizer boundary")
        evaluation = read_artifact(path / f"eval_step_{step:06d}.json", result["inputs"])
        require(field(summary, "last_evaluation") == evaluation, "summary/final evaluation mismatch")
        require(field(evaluation, "is_final") is True and field(evaluation, "budget_endpoint") is True and
                field(evaluation, "final_nominal_loss_tokens") == BUDGET and
                field(evaluation, "nominal_loss_tokens") == BUDGET and
                field(evaluation, "evaluation_kind") in ("final", "milestone") and
                field(evaluation, "training_loss_tokens") == actual and field(evaluation, "step") == step and
                field(evaluation, "unspent_loss_tokens") == unspent, "evaluation is not the matching final 5M endpoint")
        final_events = [{k: v for k, v in event.items() if k != "event"} for event in events if event["event"] == "final_evaluation"]
        require(final_events == [evaluation], "missing/ambiguous final_evaluation log evidence")
        if not replication and result["midpoint"].get("step") is not None:
            require(step > result["midpoint"]["step"], "2.5M and 5M must be different optimizer boundaries")
        result["endpoint"] = {**check_point(evaluation, identity, protocol, sample, direction_only=replication),
                              "nominal_loss_tokens": BUDGET, "unspent_loss_tokens": unspent,
                              "minimum_trajectory_loss_tokens": min(costs), "stop_reason": reason}
    except AuditError as error:
        result["endpoint"] = issue(error)
    return result


def summarize(main_run, replication_run=None):
    main = audit_run(main_run)
    replica = audit_run(replication_run, replication=True) if replication_run else None
    comparison = {"status": "pending", "reason": "second fresh seed not supplied"}
    if replica is not None and "identity" in main and "identity" in replica:
        left = {k: v for k, v in main["identity"].items() if k != "seed"}
        right = {k: v for k, v in replica["identity"].items() if k != "seed"}
        differences = sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key))
        if main["source_sha256"] != replica["source_sha256"]:
            differences.append("source_sha256")
        if main["seed"] == replica["seed"]:
            differences.append("seed_must_differ")
        comparison = {"status": "invalid" if differences else "passed", "differences": differences,
                      "comparison": "raw_identity except training seed, plus declared source SHA256"}
    checks = [main["identity_check"], main.get("midpoint", {"status": "pending"}),
              main.get("endpoint", {"status": "pending"}), comparison]
    if replica is not None:
        checks.extend([replica["identity_check"], replica.get("endpoint", {"status": "pending"})])
    statuses = {check["status"] for check in checks}
    status = next((name for name in ("invalid", "not_met", "pending") if name in statuses), "passed")
    return {"protocol": "p1_cross_run_audit_v1", "statistical_gate_status": status,
            "statistical_gate_passed": True if status == "passed" else False if status == "not_met" else None,
            "main_run": main, "replication_run": replica, "cross_run_identity": comparison,
            "engineering_acceptance": {"status": "not_evaluated", "external_required": True,
                                       "checks": ["causal gradients", "numerical correctness", "state isolation"]},
            "p1_gate_passed": None, "p1_gate_status": "not_evaluated_here",
            "scope": "fixed development sample only; no holdout data read; source hashes are declarations, not runtime attestation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", required=True)
    parser.add_argument("--replication-run")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = summarize(args.main_run, args.replication_run)
    rendered = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if report["statistical_gate_status"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())

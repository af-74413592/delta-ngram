"""CPU analysis of the separately registered DEV64 raw-weight context reference.

The frozen empty arm anchors BOTH gains. Fresh raw empty is a numerical sanity
check, not a different denominator baseline. The primary result is fixed at
100M; the other checkpoints are descriptive and never select a replacement.
No torch/model/source-text/GPU work is performed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OPS = REPO.parent / "delta-ngram-runs" / "multidomain_launch_20261019"
DISTANCES = (0, 1, 4, 8, 16)
SCOPES = (("all_targets", "nll", "target_tokens"),
          ("first_scoring_window", "first_target_nll", "first_target_tokens"))
NOMINAL = tuple(range(0, 100000001, 5000000))
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
REGISTRY_SHA = "9ea663ad70131e272522cbad8f6c0fa20bc03f41b324fe7b71d0fb8f4fbfccc2"
ANNOTATIONS_SHA = "7164e9f8b6f23ebfeaaccc2687a6a8b6ad222803f705bacd1a203a2596fdbcce"
INVENTORY_SHA = "710686bd0c1f77be618b9a4b06c7a62ed5ac4046767c9c03af0602e5feb5d4e9"
REFERENCE_SHA = "7051e93ad164a90e26623a852b2679e1d9d619809cd14f5aabc69635402c9f0c"
CONTEXT_PROTOCOL = "recent_news_dev64_full_context_20261020_v1"
EMPTY_NLL_ATOL = 1e-4
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20261020


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def close(a, b, message, atol=2e-11):
    require(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol, message)


def indexed_records(records):
    indexed = {(r["event_family"], r["unrelated_article_count"]): r for r in records}
    families = sorted({r["event_family"] for r in records})
    require(len(records) == len(indexed) == 320 and len(families) == 64,
            "Expected 64 unique families by five distances")
    require(set(indexed) == {(f, d) for f in families for d in DISTANCES},
            "Family-distance coverage differs")
    return families, indexed


def validate_score(score, target_count):
    require(type(score["target_tokens"]) is int and score["target_tokens"] == target_count,
            "Scored target count differs")
    require(score["first_target_tokens"] == min(384, target_count),
            "First scoring window must contain B[128:min(512,len(B))]")
    require(score["read_windows"] == math.ceil(target_count / 384), "Read-window count differs")
    for nll, loss, count in (("nll", "loss_sum", "target_tokens"),
                             ("first_target_nll", "first_target_loss_sum", "first_target_tokens")):
        require(score[loss] >= 0 and math.isfinite(score[loss]), "Invalid target loss")
        close(score[nll], score[loss] / score[count], "NLL does not match loss/token sums")


def load_curve(raw_dir, inventory_path):
    """Validate identities, targets and exact empty equality for all 21 points."""
    require(sha(inventory_path) == INVENTORY_SHA, "Frozen checkpoint inventory SHA256 differs")
    inventory = load(inventory_path)
    jobs = {j["nominal_loss_tokens"]: j for j in inventory["checkpoints"]}
    require(set(jobs) == set(NOMINAL), "Expected all 21 registered checkpoint identities")
    paths = {n: raw_dir / f"m1_{n // 1000000:03d}m_continuation.json" for n in NOMINAL}
    require(all(p.is_file() for p in paths.values()), "All 21 complete DEV64 JSON files are required")
    require(sha(paths[100000000]) == REFERENCE_SHA, "Frozen primary DEV64 reference bytes differ")
    loaded, indexes = {}, {}
    reference = load(paths[100000000])
    families, ri = indexed_records(reference["per_family"])
    reference_model = reference["model"]
    require(reference_model["name_or_path"] == "Qwen/Qwen3-0.6B"
            and reference_model["revision"] == reference_model["resolved_commit"] == REVISION
            and reference_model["dtype"] == "torch.bfloat16", "Frozen reader identity differs")
    manifest = reference["manifest"]
    require(manifest["registry_file_sha256"] == REGISTRY_SHA
            and manifest["prefix_annotations"]["file_sha256"] == ANNOTATIONS_SHA,
            "Frozen DEV64 input identities differ")
    require(manifest["initial_prefix_tokens"] == 128
            and manifest["target_tokens_per_distance"] == 52914, "Frozen scoring scope differs")
    pairs = {p["id"]: p for p in manifest["pairs"]}
    empty_fields = ("nll", "loss_sum", "target_tokens", "first_target_nll",
                    "first_target_loss_sum", "first_target_tokens", "read_windows")
    for n, path in paths.items():
        result = reference if n == 100000000 else load(path)
        job = jobs[n]
        require(result["protocol"] == "delta_ngram_recent_news_continuation_eval_v1"
                and result["complete"] is True and result["evaluation_only"] is True
                and result["split"] == "dev", "Incomplete or non-DEV curve result")
        require(result["optimizer_steps"] == result["backward_calls"] == 0,
                "Old result performed optimization")
        cp = result["checkpoint"]
        for field in ("sha256", "step", "writer_tensor_sha256"):
            require(cp[field] == job[field], f"Checkpoint identity mismatch: {n}, {field}")
        require(cp["path"] == job["checkpoint"]
                and cp["training_loss_tokens"] == job["checkpoint_training_loss_tokens"]
                and result["writer_tensor_sha256"] == job["writer_tensor_sha256"],
                "Checkpoint path, training tokens or tensor hash differs")
        require(canonical_sha(result["manifest"]) == result["manifest_sha256"]
                == reference["manifest_sha256"] and result["manifest"] == manifest,
                "Frozen manifest differs between checkpoints")
        require(canonical_sha(result["model"]) == result["model_identity_sha256"]
                == reference["model_identity_sha256"], "Reader differs between checkpoints")
        fs, indexed = indexed_records(result["per_family"])
        require(fs == families, "Families differ between checkpoints")
        for (family, distance), row in indexed.items():
            pair = pairs[row["pair_id"]]
            target_count = pair["later_token_count"] - 128
            require(pair["event_family"] == family
                    and row["later_source_id"] == pair["later_source_id"], "Target source differs")
            for arm in ("empty", "self"):
                validate_score(row["conditions"][arm], target_count)
            condition = next(c for c in pair["interference_conditions"]
                             if c["unrelated_article_count"] == distance)
            expected_sources = [pair["earlier_source_id"], *condition["distractor_source_ids"]]
            require(row["conditions"]["self"]["write_source_ids"] == expected_sources,
                    "Executed self source order differs from frozen plan")
            require(row["conditions"]["empty"]["write_source_ids"] == [],
                    "Empty arm wrote an article")
            for field in empty_fields:
                require(row["conditions"]["empty"][field]
                        == ri[family, 0]["conditions"]["empty"][field],
                        f"Old empty must be exact across all checkpoints/distances: {family}, {field}")
        require(sum(indexed[f, 0]["conditions"]["self"]["target_tokens"] for f in families) == 52914,
                "Frozen total targets differ")
        loaded[n], indexes[n] = result, indexed
    return jobs, loaded, indexes, families, paths


def validate_context(context, context_path, preflight, preflight_path, reference, families, ri):
    require(context["protocol"] == CONTEXT_PROTOCOL and context["complete"] is True
            and context["evaluation_only"] is True and context["split"] == "dev",
            "A complete DEV64 raw-context result is required")
    require(context["optimizer_steps"] == context["backward_calls"] == 0, "Context performed optimization")
    require(context["preflight_sha256"] == sha(preflight_path), "Preflight file SHA256 differs")
    registration = preflight["registration"]
    require(preflight["protocol"] == CONTEXT_PROTOCOL and preflight["stage"] == "cpu_preflight"
            and preflight["model_weights_loaded"] is False, "Expected preflight before model loading")
    require(canonical_sha(registration) == preflight["registration_sha256"]
            == context["preflight_registration_sha256"], "Context registration identity differs")
    require(context["reference"]["sha256"] == registration["reference_file_sha256"] == REFERENCE_SHA
            and context["reference"]["checkpoint_sha256"]
            == registration["reference_checkpoint_sha256"] == reference["checkpoint"]["sha256"],
            "Context refers to a different primary reference")
    for field in ("name_or_path", "revision", "resolved_commit", "dtype", "tokenizer_fingerprint"):
        require(context["model"][field] == reference["model"][field], "Context reader differs: " + field)
    require(context["model"]["attn_implementation"] == "sdpa", "Context attention implementation differs")
    for field in ("registry_file_sha256", "tokenizer", "tokenizer_fingerprint", "sources", "pairs",
                  "target_tokens_per_distance"):
        require(context["manifest"][field] == reference["manifest"][field],
                "Context frozen manifest differs: " + field)
    require(context["manifest"]["prefix_annotations"]["file_sha256"] == ANNOTATIONS_SHA,
            "Context prefix annotation identity differs")
    protocol = context["context_protocol"]
    require(protocol == registration["context_protocol"], "Context protocol changed after registration")
    for key, value in {"separator_text": "\n\n", "separator_token_ids": [271],
                       "query_prefix_tokens": 128, "target_block_tokens": 384,
                       "truncation_allowed": False, "memory_injection": False,
                       "writer_checkpoint_loaded": False, "chat_template": False,
                       "special_tokens_added": False}.items():
        require(protocol[key] == value, "Context protocol changed: " + key)
    require(registration["empty_nll_absolute_tolerance"] == EMPTY_NLL_ATOL
            == context["empty_sanity"]["tolerance"], "Predeclared empty tolerance changed")
    for field in ("no_writer_checkpoint_loaded", "no_memory_adapter_constructed", "no_forward_hooks",
                  "parameters_frozen_no_gradients", "parameter_objects_and_versions_unchanged",
                  "input_files_unchanged", "targets_identical_to_frozen_dev64", "no_truncation"):
        require(context["integrity_checks"].get(field) is True, "Context integrity failed: " + field)
    require(context["empty_sanity"]["all_64_families_passed"] is True, "Raw empty sanity incomplete")
    require(all(p["nll_absolute_difference"] <= registration["logits_to_keep_nll_parity_absolute_tolerance"]
                for p in context["empty_sanity"]["selected_logits_parity"]),
            "Selected-logit parity failed")
    fs, indexed = indexed_records(context["per_family"])
    require(fs == families, "Raw context families differ")
    pfs, plans = indexed_records(registration["plans"])
    require(pfs == families, "Registered context plans differ")
    raw_empty = context["raw_empty_sanity_per_family"]
    require(set(raw_empty) == set(families), "Raw empty sanity must cover exactly 64 families")
    drift = []
    for (family, distance), row in indexed.items():
        old = ri[family, distance]
        plan = plans[family, distance]
        require(set(row["conditions"]) == {"empty", "context"}, "Unexpected raw context conditions")
        for field in ("pair_id", "later_source_id"):
            require(row[field] == old[field], "Context target identity differs: " + field)
        for field in ("target_token_ids_sha256", "target_tokens", "context_source_ids",
                      "evidence_token_ids_sha256", "source_article_tokens", "separator_tokens",
                      "evidence_tokens"):
            require(row[field] == plan[field], "Scored context differs from plan: " + field)
        require(row["context_source_ids"] == old["conditions"]["self"]["write_source_ids"],
                "Context source order differs from executed writer self arm")
        require(row["target_token_ids_sha256"] == indexed[family, 0]["target_token_ids_sha256"],
                "Context target token identity changed across distances")
        targets = old["conditions"]["empty"]["target_tokens"]
        require(row["target_tokens"] == targets, "Context target count differs from original")
        require(row["empty_baseline_source"] == "frozen_primary_100m_reference"
                and row["conditions"]["empty"] == old["conditions"]["empty"],
                "Context result must preserve the exact frozen empty anchor")
        for arm, score in (("raw_empty", raw_empty[family]), ("context", row["conditions"]["context"])):
            validate_score(score, targets)
            windows = score["windows"]
            require(len(windows) == len(plan["windows"]) == score["read_windows"],
                    "Actual window count differs")
            for wi, (window, expected) in enumerate(zip(windows, plan["windows"])):
                for field in ("query_context_start_token", "target_start_token",
                              "target_end_token_exclusive", "target_tokens",
                              "query_prefix_token_ids_sha256", "target_token_ids_sha256"):
                    require(window[field] == expected[field], "Scored target window differs: " + field)
                start = 128 + wi * 384
                require(window["query_context_start_token"] == start - 128
                        and window["target_start_token"] == start
                        and window["target_end_token_exclusive"] == min(start + 384, targets + 128),
                        "Target windows have gaps, overlap or changed prefix")
                if arm == "context":
                    for field in ("input_token_ids_sha256", "reader_input_tokens",
                                  "target_start_position", "target_end_position_exclusive"):
                        require(window[field] == expected[field], "Full context window differs: " + field)
                else:
                    require(window["reader_input_tokens"] == 128 + window["target_tokens"]
                            and window["target_start_position"] == 128,
                            "Raw empty window contains extra context")
                require(window["loss_sum"] >= 0 and math.isfinite(window["loss_sum"]),
                        "Invalid window loss")
            close(score["loss_sum"], math.fsum(w["loss_sum"] for w in windows),
                  "Window losses do not sum to article loss")
            close(score["first_target_loss_sum"], windows[0]["loss_sum"], "First window loss differs")
            require(score["reader_input_tokens"] == sum(w["reader_input_tokens"] for w in windows),
                    "Reader input token count differs from actual windows")
        if distance == 0:
            item = {"event_family": family}
            for _, metric, _ in SCOPES:
                difference = raw_empty[family][metric] - old["conditions"]["empty"][metric]
                require(abs(difference) <= EMPTY_NLL_ATOL, f"Raw empty drift exceeds tolerance: {family}, {metric}")
                item[metric + "_signed_difference"] = difference
            drift.append(item)
    require(context["execution"]["context_cells"] == 320
            and context["execution"]["context_read_windows"] == 865
            and context["execution"]["empty_families"] == 64
            and context["execution"]["empty_read_windows"] == 173, "Raw execution scope differs")
    return indexed, {
        "context_file_sha256": sha(context_path),
        "preflight_file_sha256": sha(preflight_path),
        "all_21_checkpoint_empty_exact_equal": True,
        "old_empty_nll_absolute_tolerance": 0,
        "raw_empty_nll_absolute_tolerance": EMPTY_NLL_ATOL,
        "maximum_raw_empty_nll_absolute_difference": max(abs(r["nll_signed_difference"]) for r in drift),
        "maximum_raw_empty_first_window_nll_absolute_difference":
            max(abs(r["first_target_nll_signed_difference"]) for r in drift),
        "raw_empty_family_differences": drift,
        "manifest_sources_token_hashes_and_pairs_match": True,
        "target_windows_match_registered_hashes_and_frozen_boundaries": True,
        "context_source_order_matches_all_frozen_self_arms": True,
    }


def bootstrap_counts(family_count, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    """One family resampling matrix shared by all checkpoints/distances/scopes."""
    rng = random.Random(seed)
    counts = np.zeros((samples, family_count), dtype=np.uint8)
    for i in range(samples):
        counts[i] = np.bincount([rng.randrange(family_count) for _ in range(family_count)],
                                minlength=family_count)
    return counts


def interval(values):
    return np.quantile(values, [.025, .975], method="linear").tolist()


def ratio_statistics(numerator, denominator, numerator_draws, denominator_draws):
    """Retain negative denominator draws; zero draws make a ratio CI unavailable."""
    require(numerator_draws.shape == denominator_draws.shape, "Bootstrap draws must be paired")
    require(np.isfinite(numerator_draws).all() and np.isfinite(denominator_draws).all(),
            "Gain draws must be finite")
    denominator_ci = interval(denominator_draws)
    zero = denominator_draws == 0
    negative = denominator_draws < 0
    ratio_draws = np.full(numerator_draws.shape, np.nan, dtype=np.float64)
    np.divide(numerator_draws, denominator_draws, out=ratio_draws, where=~zero)
    # This diagnostic uses ALL draws, including negative denominators. If any
    # denominator is exactly zero, no percentile interval is computed at all.
    diagnostic = interval(ratio_draws) if not zero.any() else None
    stable = denominator > 0 and denominator_ci[0] > 0 and not zero.any()
    status = ("positive_denominator_ci" if stable else
              "nonpositive_mean_denominator" if denominator <= 0 else
              "zero_denominator_draws" if zero.any() else "denominator_ci_includes_zero")
    ci = diagnostic if stable else None
    return {
        "memory_gain_nats": float(numerator),
        "memory_gain_ci95_low": interval(numerator_draws)[0],
        "memory_gain_ci95_high": interval(numerator_draws)[1],
        "context_gain_nats": float(denominator),
        "context_gain_ci95_low": denominator_ci[0],
        "context_gain_ci95_high": denominator_ci[1],
        "recovery_ratio": float(numerator / denominator) if denominator > 0 else None,
        "recovery_percent": float(100 * numerator / denominator) if denominator > 0 else None,
        "recovery_percent_ci95_low": 100 * ci[0] if ci is not None else None,
        "recovery_percent_ci95_high": 100 * ci[1] if ci is not None else None,
        "ratio_status": status,
        "ratio_ci_available": stable,
        "denominator_nonpositive_draws": int((zero | negative).sum()),
        "denominator_nonpositive_draw_fraction": float((zero | negative).mean()),
        "denominator_negative_draws": int(negative.sum()),
        "denominator_zero_draws": int(zero.sum()),
        "all_draw_signed_ratio_percentile025": diagnostic[0] if diagnostic is not None else None,
        "all_draw_signed_ratio_percentile975": diagnostic[1] if diagnostic is not None else None,
        "discarded_bootstrap_draws": 0,
    }, ratio_draws


def summarize(jobs, indexes, context_index, families, counts):
    weights = counts.astype(np.float64) / len(families)
    require(counts.shape == (BOOTSTRAP_SAMPLES, len(families))
            and np.all(counts.sum(axis=1) == len(families)), "Invalid paired bootstrap draws")
    rows, family_rows, numerators, denominators = [], [], [], []
    # Each denominator is shared by all 21 checkpoints. The original empty
    # values are exactly equal at every checkpoint, as validated above.
    reference = indexes[100000000]
    denominator_columns = {}
    for si, (scope, metric, _) in enumerate(SCOPES):
        for distance in DISTANCES:
            values = np.asarray([reference[f, distance]["conditions"]["empty"][metric]
                                 - context_index[f, distance]["conditions"]["context"][metric]
                                 for f in families])
            denominator_columns[scope, distance] = len(denominators)
            denominators.append(values)
    denominator_values = np.column_stack(denominators)
    denominator_draws = weights @ denominator_values
    for n in NOMINAL:
        for scope, metric, count_key in SCOPES:
            for distance in DISTANCES:
                idx = indexes[n]
                empty = np.asarray([idx[f, distance]["conditions"]["empty"][metric] for f in families])
                self_nll = np.asarray([idx[f, distance]["conditions"]["self"][metric] for f in families])
                context_nll = np.asarray([context_index[f, distance]["conditions"]["context"][metric]
                                          for f in families])
                values = empty - self_nll
                numerators.append(values)
                column = denominator_columns[scope, distance]
                rows.append({
                    "checkpoint": jobs[n]["label"], "nominal_loss_tokens": n,
                    "actual_training_loss_tokens": jobs[n]["training_loss_tokens"],
                    "role": "primary" if n == 100000000 else "secondary" if n == 65000000 else "descriptive_curve",
                    "scope": scope, "unrelated_article_count": distance, "families": len(families),
                    "target_tokens": sum(idx[f, distance]["conditions"]["self"][count_key] for f in families),
                    "empty_nll": float(empty.mean()), "self_nll": float(self_nll.mean()),
                    "context_nll": float(context_nll.mean()), "denominator_column": column,
                })
                for fi, family in enumerate(families):
                    family_rows.append({
                        "checkpoint": jobs[n]["label"], "nominal_loss_tokens": n, "scope": scope,
                        "event_family": family, "pair_id": idx[family, distance]["pair_id"],
                        "unrelated_article_count": distance,
                        "target_tokens": idx[family, distance]["conditions"]["self"][count_key],
                        "empty_nll": float(empty[fi]), "self_nll": float(self_nll[fi]),
                        "context_nll": float(context_nll[fi]),
                        "memory_gain_nats": float(values[fi]),
                        "context_gain_nats": float(empty[fi] - context_nll[fi]),
                    })
    numerator_values = np.column_stack(numerators)
    numerator_draws = weights @ numerator_values
    all_ratios = np.empty_like(numerator_draws)
    for ci, row in enumerate(rows):
        di = row["denominator_column"]
        stats, ratios = ratio_statistics(float(numerator_values[:, ci].mean()),
            float(denominator_values[:, di].mean()), numerator_draws[:, ci], denominator_draws[:, di])
        row.update(stats)
        all_ratios[:, ci] = ratios
    arrays = {
        "family_resample_counts": counts,
        "memory_gain_draws": numerator_draws,
        "context_gain_draws": denominator_draws,
        "recovery_ratio_draws": all_ratios,
        "denominator_column_by_curve_row": np.asarray([r["denominator_column"] for r in rows]),
    }
    return rows, family_rows, arrays


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interpretation(rows, validation):
    primary = [r for r in rows if r["role"] == "primary"]
    lines = [
        "# DEV64 full-context NLL benefit recovery",
        "",
        "Primary: fixed 100M writer; 65M is secondary. The 21-point table is descriptive.",
        "The raw-weight full-context reference is evaluated once and reused at every checkpoint.",
        "",
        "R = mean_family(frozen_empty NLL - self NLL) / "
        "mean_family(frozen_empty NLL - raw_full_context NLL).",
        "",
        "Each event family has equal weight after its target-token mean. This is a ratio of means, "
        "not a mean of event-level ratios. It measures relative NLL benefit, not accuracy, fact recall, "
        "or a theoretical memory upper bound. Percentages are not clipped to [0,100].",
        "",
        "| Scope | Unrelated articles | Memory gain (nats) | Context gain (nats) | Recovery, pointwise 95% CI | Nonpositive denominator draws |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for r in primary:
        if r["ratio_ci_available"]:
            ratio = (f"{r['recovery_percent']:.2f}% "
                     f"[{r['recovery_percent_ci95_low']:.2f}, {r['recovery_percent_ci95_high']:.2f}]")
        else:
            ratio = "Unavailable as a stable percentage: " + r["ratio_status"]
        lines.append(f"| {r['scope']} | {r['unrelated_article_count']} | "
                     f"{r['memory_gain_nats']:.6f} | {r['context_gain_nats']:.6f} | {ratio} | "
                     f"{r['denominator_nonpositive_draws']}/{BOOTSTRAP_SAMPLES} |")
    lines += [
        "",
        "The first scoring window is B[128:min(512,len(B))], at most 384 target tokens. "
        "It is not the first token or the whole first article block.",
        "",
        f"All 21 old empty baselines match exactly, including their first scoring windows. "
        f"Fresh raw empty was checked at absolute NLL tolerance {EMPTY_NLL_ATOL}; "
        f"maximum whole-target drift = {validation['maximum_raw_empty_nll_absolute_difference']:.9g}, "
        f"first-window drift = {validation['maximum_raw_empty_first_window_nll_absolute_difference']:.9g}.",
        "",
        f"Bootstrap: {BOOTSTRAP_SAMPLES:,} paired family draws, seed {BOOTSTRAP_SEED}; "
        "the same draws cover all distances, checkpoints and scopes. Negative denominator draws "
        "remain in the signed ratios. If the denominator CI includes zero, a stable ratio CI is "
        "withheld. Any exactly zero denominator draw makes the ratio interval unavailable; its "
        "NaN stays at the same draw index in the NPZ. No draws are discarded.",
        "",
        "CSV diagnostic signed-ratio percentiles are not a stable confidence interval when the "
        "denominator is uncertain. Inspect the numerator, denominator CI and nonpositive draw "
        "fraction together. All intervals are pointwise and unadjusted for multiple comparisons. "
        "One training seed and a fixed event cohort do not establish general accuracy gains.",
        "",
        "This incremental analysis does not modify the sealed M1 report or earlier analysis ZIP "
        "and does not execute FINAL, QA, training or free generation.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--curve-dir", type=Path, default=OPS / "dev64_curve_results_20261020")
    parser.add_argument("--inventory", type=Path, default=OPS / "m1_dev64_curve_inventory_20261020.json")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty directory for this separate incremental analysis")
    args = parser.parse_args()
    require(not args.output_dir.exists() or (args.output_dir.is_dir() and not any(args.output_dir.iterdir())),
            "Output directory must be new or empty; existing analyses are not overwritten")
    jobs, loaded, indexes, families, paths = load_curve(args.curve_dir, args.inventory)
    context, preflight = load(args.context), load(args.preflight)
    ci, validation = validate_context(context, args.context, preflight, args.preflight,
                                     loaded[100000000], families, indexes[100000000])
    counts = bootstrap_counts(len(families))
    rows, family_rows, arrays = summarize(jobs, indexes, ci, families, counts)
    require(len(rows) == 210 and len(family_rows) == 13440, "Analysis output coverage differs")
    primary = [r for r in rows if r["role"] == "primary"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "context_recovery_curve.csv", rows)
    write_csv(args.output_dir / "context_recovery_primary100m.csv", primary)
    write_csv(args.output_dir / "context_recovery_family.csv", family_rows)
    np.savez_compressed(args.output_dir / "context_recovery_bootstrap_draws.npz", **arrays)
    summary = {
        "protocol": "dev64_full_context_recovery_analysis_20261020_v1",
        "complete": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_nominal_loss_tokens": 100000000, "secondary_nominal_loss_tokens": 65000000,
        "checkpoint_selection_performed": False, "checkpoints": 21, "families": families,
        "distances": list(DISTANCES), "scopes": [s[0] for s in SCOPES],
        "ratio_definition": "mean_family(old_empty_nll-self_nll)/mean_family(old_empty_nll-raw_context_nll)",
        "baseline": "Same frozen old empty in numerator and denominator; new raw empty is a sanity check.",
        "aggregation": "Target-token mean within family, equal weights across 64 families, then ratio of means.",
        "interpretation": "Relative NLL benefit; not accuracy, fact-recall percentage, or theoretical upper bound.",
        "context_protocol": context["context_protocol"],
        "model": context["model"],
        "evaluation_execution": context["execution"],
        "evaluation_run": context["run"],
        "bootstrap": {"unit": "event_family", "seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES,
            "confidence": .95, "paired_across_checkpoints_distances_and_scopes": True,
            "quantile_method": "linear", "discarded_draws": 0, "negative_denominator_draws_retained": True,
            "zero_denominator_draws": "Undefined ratios are NaN in NPZ at original indices; no ratio interval.",
            "unstable_denominator_policy": "No stable ratio CI when denominator mean <=0 or CI includes zero.",
            "multiple_comparison_adjustment": False},
        "primary100M": primary,
        "secondary65M": [r for r in rows if r["role"] == "secondary"],
        "validation": validation,
        "array_axes": {
            "family_resample_counts": "draw x family in summary families order",
            "memory_gain_draws": "draw x context_recovery_curve.csv row (header excluded)",
            "context_gain_draws": "draw x denominator column; scopes then distances",
            "recovery_ratio_draws": "draw x context_recovery_curve.csv row; negative denominators included",
        },
        "limitations": ["Single training seed; bootstrap does not cover training randomness.",
                        "Pointwise exploratory intervals do not adjust 21 checkpoints or multiple scopes.",
                        "Full context can itself help or hurt; it is an observed reference, not an upper bound.",
                        "Identical source token identities and registered scoring windows checked; "
                        "this CPU analysis does not retokenize source articles or rerun model inference."],
    }
    write_json(args.output_dir / "context_recovery_summary.json", summary)
    source_files = [args.context, args.preflight, args.inventory, Path(__file__), *paths.values()]
    write_json(args.output_dir / "context_recovery_source_manifest.json", {
        "files": [{"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha(p)} for p in source_files],
        "arguments": {k: str(v.resolve()) for k, v in vars(args).items()},
        "python": sys.version, "numpy": np.__version__,
    })
    (args.output_dir / "interpretation.md").write_text(interpretation(rows, validation),
                                                     encoding="utf-8", newline="\n")
    print(json.dumps({"complete": True, "checkpoints": 21, "curve_rows": len(rows),
                      "primary_rows": len(primary), "family_rows": len(family_rows),
                      "output_dir": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Replay frozen DEV64 hash addresses on CPU; never construct model or writer.

This computes exact allocated-row counts for D1 from the verified token streams.
It does not replay neural values/gates or estimate semantic capacity.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import time


REPO = Path(__file__).resolve().parents[1]
RUNS = REPO.parent / "delta-ngram-runs"
OPS = RUNS / "multidomain_launch_20261019"
REPORT = RUNS / "m1_analysis_20261020"
DISTANCES = (0, 1, 4, 8, 16)
ORDERS = (2, 3)
BUCKETS = 16384
REGISTRY_SHA = "9ea663ad70131e272522cbad8f6c0fa20bc03f41b324fe7b71d0fb8f4fbfccc2"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
EVALUATOR_COMMIT = "167d24783161be7842bbb9aec52f2c7804191ba9"
# Git-blob SHA256 values read from the clean evaluator commit above. Local
# Windows CRLF is normalized to LF before comparing Python source semantics.
FROZEN_SOURCE_SHA = {
    "src/delta_ngram/ngram_lookup.py": "ef6ced910e521b644bf84b25ba20090d6c4b7a2d903c97d397832dc79e12d3d8",
    "src/delta_ngram/operators.py": "4a2eabfd92239bc0f8ca994807c317273abb1db1cab9072ac94e130830959105",
    "src/delta_ngram/writer.py": "0772efde1c8cff1ac25277885e363e5acf7201a07c6fe606abfa51aeeee01489",
    "src/delta_ngram/batching.py": "1ae96443d8404b95bdd3dea367fb9498c5410ae6f211ea1a07de4b872e0a8f16",
    "src/delta_ngram/recent_news_evaluation.py": "cb511e7b8ef95bd670a2f9d087e4491d463077a61a3a6fb6367f685694e5e192",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_interpretation(path, note):
    count = note["verification"]["equivalent_completed_checkpoints_verified"]
    maximum = note["overall_maximum"]
    maximum_order = max(note["maximum_per_order"].items(),
                        key=lambda item: item[1][f"occupancy_U{item[0]}_over_B"])
    order, single = maximum_order
    lines = [
        "# 新 DEV64：物理容量与哈希合并",
        "",
        f"已核验 **{count}/21 个注册检查点**的配置、冻结源/token 哈希和逐格写入预算一致。"
        "以下统计覆盖 64 个事件族 × 5 个干扰距离 × self/donor，共 640 个不同写入条件。",
        "",
        "审计使用冻结文章及同一 tokenizer，逐篇确认 token 序列 SHA256，"
        "并将每个 512-token 写入块的地址与原评测 CPU 哈希函数核对。"
        "它是**精确地址重放**，不属于评测期间的 state.rows 日志；没有调用模型、writer 或 reader。",
        "",
        f"新 DEV64 的两表总占用最高 **{maximum['U_total']:,}/32,768"
        f"＝{maximum['occupancy_U_total_over_2B']:.4%}**。单表最高为 {order}-gram 表的 "
        f"**{single[f'U{order}']:,}/16,384＝{single[f'occupancy_U{order}_over_B']:.4%}**。"
        "因此，本次实际写入条件可排除物理哈希表满表；不能据此排除哈希冲突、"
        "后续写入对旧值的覆盖干扰或表示容量限制。历史训练轨迹的 34.33% 不能外推到新 DEV64。",
        "",
        "### 两表合计占用率",
        "",
        "下表为 64 个事件族的最小值／中位数／最大值，不是置信区间。",
        "",
        "| 干扰文章数 | self 占用率 | donor 占用率 |",
        "|---:|---:|---:|",
    ]
    indexed = {(r["unrelated_article_count"], r["arm"]): r for r in note["per_distance_arm_summary"]}
    for distance in DISTANCES:
        values = []
        for arm in ("self", "donor"):
            r = indexed[distance, arm]
            values.append(" / ".join(f"{r['occupancy_U_total_over_2B_' + level]:.2%}"
                                     for level in ("min", "p50", "max")))
        lines.append(f"| {distance} | {values[0]} | {values[1]} |")
    lines += [
        "",
        "### 重复 n-gram 与哈希合并的区分",
        "",
        "- **E**：有效 n-gram 写入次数；**K**：不同 token-ID n-gram 的数量；"
        "**U**：不同物理哈希行的数量。",
        "- **E−K** 计重复出现的相同 n-gram；**K−U** 计不同 n-gram 被哈希合并后的数量差。"
        "两者是不同来源，满足 E−U＝(E−K)＋(K−U)。",
        "- 图中的 **(K−U)/K** 是不同 key 经哈希合并后的缩减比例，"
        "不是任意两个 key 碰撞的概率，也不是碰撞 key 对数。",
        "",
        f"总占用最大条件为 `{maximum['event_family']}`、self、16 篇干扰："
        f"2-gram 的 E/K/U＝{maximum['E2']:,}/{maximum['K2']:,}/{maximum['U2']:,}，"
        f"故重复次数为 {maximum['E2'] - maximum['K2']:,}，不同 key 的哈希合并数量差为 "
        f"{maximum['K2'] - maximum['U2']:,}；3-gram 的 E/K/U＝"
        f"{maximum['E3']:,}/{maximum['K3']:,}/{maximum['U3']:,}。",
        "",
        "仅凭长度得到的上界是 Uₒ ≤ min(16,384, Σ文章 max(0,L−o+1))。"
        "16 篇干扰时，部分条件的上界已达到 100%，不能用这个上界判定未满表；"
        "上面的精确地址重放补足了这一点。",
        "",
        "D1 为每个有效地址分配行，β＝0 也不跳过分配，且不逐出旧行；"
        "查询和方向置换只读。因此在已核验的同一数据、tokenizer、地址函数和写入顺序下，"
        "E/K/U 不随 writer 检查点权重改变。这不表示记忆值、门控或 NLL 不随权重改变。",
        "",
        "![每表占用与不同 key 的哈希合并](../figures/dev64_capacity.png)",
        "",
        "明细：[640 个条件](capacity_family_distance.csv)；"
        "汇总：[距离统计](capacity_distance_summary.csv)；"
        "核验依据：[审计记录](capacity_scope_note.json)。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def token_identity(tokenizer):
    return {
        "class": type(tokenizer).__name__,
        "vocab_sha256": canonical_sha(tokenizer.get_vocab()),
        "pipeline_sha256": canonical_sha(json.loads(tokenizer.backend_tokenizer.to_str())),
        "special_tokens": {name: getattr(tokenizer, name, None)
                           for name in ("pad_token_id", "eos_token_id", "bos_token_id")},
    }


def hashed_ngram(key, slot):
    value = 0
    for token in key:
        value = (value * 1000003 + token + 1) % 2147483647
    return 1 + slot * BUCKETS + value % BUCKETS


def article_sets(tokens, torch, original_rows):
    """Independent tuple/hash replay checked against the original CPU kernel."""
    keys = {order: set() for order in ORDERS}
    rows = {order: set() for order in ORDERS}
    events = {order: max(0, len(tokens) - order + 1) for order in ORDERS}
    checked_positions = 0
    # Match write_article: article-local context, two-token boundary prefix,
    # and exclude prefix positions themselves from the write events.
    for start in range(0, len(tokens), 512):
        end = min(start + 512, len(tokens))
        prefix_start = max(0, start - 2)
        ids = torch.tensor([tokens[prefix_start:end]], dtype=torch.long, device="cpu")
        result = original_rows(ids, torch.ones_like(ids), BUCKETS, ORDERS)
        result = result[0, start - prefix_start:].tolist()
        for local, actual in enumerate(result):
            position = start + local
            for slot, order in enumerate(ORDERS):
                key = tuple(tokens[position - order + 1:position + 1]) if position >= order - 1 else ()
                expected = hashed_ngram(key, slot) if key else 0
                require(actual[slot] == expected, "Python replay differs from original hash kernel")
                if key:
                    keys[order].add(key)
                    rows[order].add(expected)
                checked_positions += 1
    return {"keys": keys, "rows": rows, "events": events,
            "token_count": len(tokens), "blocks": math.ceil(len(tokens) / 512),
            "address_prefix_tokens": 2 * max(0, math.ceil(len(tokens) / 512) - 1),
            "checked_address_positions": checked_positions}


def validate_result(path):
    result = load(path)
    require(result.get("protocol") == "delta_ngram_recent_news_continuation_eval_v1",
            "Unexpected result protocol")
    require(result.get("complete") is True and result.get("evaluation_only") is True
            and result.get("split") == "dev", "Only complete evaluation-only DEV results accepted")
    require(result.get("optimizer_steps") == result.get("backward_calls") == 0, "Not read-only evaluation")
    require(result.get("independent_families") == 64 and len(result.get("per_family", [])) == 320,
            "Expected complete 64-family result")
    manifest = result["manifest"]
    require(canonical_sha(manifest) == result["manifest_sha256"], "Invalid manifest digest")
    require(manifest["registry_file_sha256"] == REGISTRY_SHA, "Unexpected registry")
    require(manifest["distances"] == list(DISTANCES) and manifest["write_block_tokens"] == 512,
            "Write schedule differs")
    config = result["run"]["config"]
    require(config["reader"]["backend"] == "external_hash"
            and config["reader"]["buckets"] == BUCKETS
            and config["reader"]["orders"] == list(ORDERS)
            and config["method"]["operator"] == "D1", "Unsupported memory configuration")
    require("memory_backend" not in result, "An alternate memory backend needs a separate replay audit")
    require(result["run"]["git_before"] == result["run"]["git_after"]
            == {"commit": EVALUATOR_COMMIT, "dirty": False}, "Scored evaluator source changed")
    for key in ("actual_write_budgets_verified", "memory_unchanged_during_queries",
                "target_never_written", "input_files_unchanged"):
        require(result["integrity_checks"].get(key) is True, "Missing runtime integrity check: " + key)
    rows = {(r["event_family"], r["unrelated_article_count"]): r for r in result["per_family"]}
    families = sorted({family for family, _ in rows})
    require(len(families) == 64 and set(rows) == {(f, d) for f in families for d in DISTANCES},
            "Duplicate/missing family-distance cells")
    return result, families, rows


def capacity_identity(result):
    """Only facts that determine address allocation, with no scored NLLs."""
    fields = ("actual_written_articles", "actual_written_tokens", "actual_written_blocks",
              "address_prefix_tokens", "write_source_ids")
    rows = sorted(result["per_family"], key=lambda r: (r["event_family"], r["unrelated_article_count"]))
    return canonical_sha({
        "reader": result["run"]["config"]["reader"], "method": result["run"]["config"]["method"],
        "manifest_sha256": result["manifest_sha256"],
        "write_budgets": [
            {"family": r["event_family"], "distance": r["unrelated_article_count"],
             "conditions": {arm: {key: r["conditions"][arm][key] for key in fields}
                            for arm in ("self", "donor")}}
            for r in rows
        ],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True, help="One complete frozen DEV64 JSON result")
    parser.add_argument("--inventory", type=Path, default=OPS / "m1_dev64_curve_inventory_20261020.json")
    parser.add_argument("--registry", type=Path, default=RUNS / "recent_news_sources/recent_news_continuation64_20261019.json")
    parser.add_argument("--output-dir", type=Path, default=REPORT / "dev64_data")
    args = parser.parse_args()
    started = time.perf_counter()
    require(not any("final_holdout" in str(p).lower() for p in (args.result, args.registry, args.output_dir)),
            "FINAL paths are forbidden")
    result, families, indexed = validate_result(args.result)
    code_verification = {}
    for relative, expected in FROZEN_SOURCE_SHA.items():
        actual = hashlib.sha256((REPO / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        require(actual == expected, "Local source differs from frozen evaluator: " + relative)
        code_verification[relative] = {"frozen_git_blob_sha256": expected,
                                       "local_lf_normalized_sha256": actual, "matches": True}
    reference_identity = capacity_identity(result)
    registered = {entry["sha256"] for entry in load(args.inventory)["checkpoints"]}
    require(len(registered) == 21, "Expected registered 21-checkpoint inventory")
    equivalent_results = []
    result_paths = {args.result.resolve(), *(path.resolve() for path in args.result.parent.glob("*continuation.json"))}
    for path in sorted(result_paths):
        candidate = load(path)
        if candidate.get("complete") is not True:
            continue
        candidate, _, _ = validate_result(path)
        require(candidate["checkpoint"]["sha256"] in registered, "Checkpoint is not registered")
        require(capacity_identity(candidate) == reference_identity,
                "Checkpoint capacity inputs differ: " + path.name)
        equivalent_results.append({"file": path.name, "sha256": sha(path),
                                   "checkpoint": candidate["checkpoint"],
                                   "capacity_inputs_sha256": reference_identity})
    require(len({item["checkpoint"]["sha256"] for item in equivalent_results}) == len(equivalent_results),
            "Duplicate checkpoint results")
    require(sha(args.registry) == REGISTRY_SHA, "Local registry differs from scored registry")
    registry = load(args.registry)
    require(registry["split"] == "dev" and registry["evaluation_only"] is True, "DEV registry required")
    require(registry["model"]["revision"] == REVISION, "Unexpected tokenizer revision")
    require(registry["tokenization"]["add_special_tokens"] is False, "Special-token policy changed")
    source_metadata = {s["id"]: s for s in result["manifest"]["sources"]}
    local_sources = {s["id"]: s for s in registry["sources"] + registry["background_sources"]}
    pairs = {p["id"]: p for p in registry["pairs"]}
    write_ids = sorted({s for r in result["per_family"] for arm in ("self", "donor")
                        for s in r["conditions"][arm]["write_source_ids"]})
    require(not set(write_ids).intersection(p["later_source_id"] for p in registry["pairs"]),
            "Target article appeared in write stream")
    # Process-local CPU/offline constraints; no global settings or model loads.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(registry["model"]["name_or_path"],
        revision=REVISION, local_files_only=True)
    identity = token_identity(tokenizer)
    kernel_path = REPO / "src/delta_ngram/ngram_lookup.py"
    spec = importlib.util.spec_from_file_location("dev64_capacity_original_hash_kernel", kernel_path)
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    require(sha(REPO / "src/delta_ngram/recent_news_evaluation.py") == result["run"]["evaluator_sha256"],
            "Local evaluation source differs from the scored evaluator")
    cache, source_receipts = {}, []
    for source_id in write_ids:
        expected, local = source_metadata[source_id], local_sources[source_id]
        text_path = (args.registry.resolve().parent / local["extracted_text_relpath"]).resolve()
        require(text_path.is_relative_to(args.registry.resolve().parent), "Source path escapes registry root")
        require(sha(text_path) == expected["extracted_text_sha256"] == local["extracted_text_sha256"],
                "Source text digest differs: " + source_id)
        text = text_path.read_bytes().decode("utf-8")
        require("\ufffd" not in text and "\r" not in text, "Invalid frozen UTF-8/LF text")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        require(len(tokens) == expected["token_count"] == local["token_count"], "Token count differs: " + source_id)
        require(canonical_sha(tokens) == expected["token_ids_sha256"], "Token sequence differs: " + source_id)
        cache[source_id] = article_sets(tokens, torch, kernel.ngram_rows)
        source_receipts.append({
            "source_id": source_id, "extracted_text_relpath": local["extracted_text_relpath"],
            "text_sha256": expected["extracted_text_sha256"], "token_count": len(tokens),
            "token_ids_sha256": expected["token_ids_sha256"],
            "all_token_ids_verified_against_scored_result": True,
            "all_block_addresses_verified_against_original_cpu_kernel": True,
        })
    rows = []
    for family in families:
        for arm in ("self", "donor"):
            keys = {order: set() for order in ORDERS}
            occupied = {order: set() for order in ORDERS}
            events = dict.fromkeys(ORDERS, 0)
            written, tokens, blocks, prefix_tokens = [], 0, 0, 0
            for distance in DISTANCES:
                original = indexed[family, distance]
                value = original["conditions"][arm]
                planned = value["write_source_ids"]
                pair = pairs[original["pair_id"]]
                anchor = pair["earlier_source_id"] if arm == "self" else pair["donor_source_id"]
                require(planned == [anchor, *pair["distractor_source_ids"][:distance]],
                        "Result write order differs from frozen registry")
                require(planned[:len(written)] == written, "Write schedule is not nested")
                for source_id in planned[len(written):]:
                    article = cache[source_id]
                    tokens += article["token_count"]
                    blocks += article["blocks"]
                    prefix_tokens += article["address_prefix_tokens"]
                    for order in ORDERS:
                        keys[order].update(article["keys"][order])
                        occupied[order].update(article["rows"][order])
                        events[order] += article["events"][order]
                    written.append(source_id)
                require(tokens == value["actual_written_tokens"] and blocks == value["actual_written_blocks"]
                        and len(written) == value["actual_written_articles"] == distance + 1
                        and prefix_tokens == value["address_prefix_tokens"], "Replay/runtime write budget mismatch")
                row = {"event_family": family, "pair_id": original["pair_id"], "arm": arm,
                       "unrelated_article_count": distance, "actual_written_articles": len(written),
                       "actual_written_tokens": tokens, "actual_written_blocks": blocks,
                       "address_prefix_tokens": prefix_tokens}
                for order in ORDERS:
                    e, k, u = events[order], len(keys[order]), len(occupied[order])
                    require(0 <= u <= k <= e and u <= BUCKETS, "Invalid E/K/U ordering")
                    row.update({
                        f"E{order}": e, f"K{order}": k, f"U{order}": u,
                        f"repeat_occurrences_E{order}_minus_K{order}": e - k,
                        f"hash_merge_excess_K{order}_minus_U{order}": k - u,
                        f"occupancy_U{order}_over_B": u / BUCKETS,
                        f"hash_merge_fraction_K{order}_minus_U{order}_over_K{order}": (k - u) / k if k else 0,
                        f"length_upper_bound_U{order}": min(BUCKETS, e),
                    })
                row.update(E_total=row["E2"] + row["E3"], K_total=row["K2"] + row["K3"],
                           U_total=row["U2"] + row["U3"])
                row["occupancy_U_total_over_2B"] = row["U_total"] / (2 * BUCKETS)
                row["length_upper_bound_U_total"] = sum(row[f"length_upper_bound_U{o}"] for o in ORDERS)
                row["length_upper_bound_occupancy"] = row["length_upper_bound_U_total"] / (2 * BUCKETS)
                rows.append(row)
    require(len(rows) == 640, "Expected 64 families x 5 distances x 2 write arms")
    metrics = [name for name in rows[0] if name not in
               {"event_family", "pair_id", "arm", "unrelated_article_count"}]
    summary = []
    for distance in DISTANCES:
        for arm in ("self", "donor"):
            selected = [r for r in rows if r["unrelated_article_count"] == distance and r["arm"] == arm]
            group = {"unrelated_article_count": distance, "arm": arm, "families": len(selected)}
            for key in metrics:
                values = [r[key] for r in selected]
                group.update({key + "_min": min(values), key + "_p50": statistics.median(values),
                              key + "_max": max(values)})
            summary.append(group)
    source_paths = (
        "scripts/audit_m1_dev64_capacity_20261020.py", "scripts/evaluate_recent_news_continuation.py",
        "src/delta_ngram/recent_news_evaluation.py", "src/delta_ngram/ngram_lookup.py",
        "src/delta_ngram/writer.py", "src/delta_ngram/batching.py", "src/delta_ngram/operators.py",
    )
    maximum = max(rows, key=lambda r: r["occupancy_U_total_over_2B"])
    note = {
        "protocol": "m1_dev64_cpu_exact_address_capacity_audit_20261020_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_exact_address_replay", "split": "dev", "evaluation_only": True,
        "result_path": str(args.result.resolve()), "result_sha256": sha(args.result),
        "checkpoint": result["checkpoint"], "scored_evaluator_sha256": result["run"]["evaluator_sha256"],
        "frozen_evaluator_commit": EVALUATOR_COMMIT,
        "frozen_source_verification": code_verification,
        "registry_sha256": REGISTRY_SHA, "manifest_sha256": result["manifest_sha256"],
        "definitions": {
            "E_o": "Number of valid n-gram occurrences written, summed over articles; article context resets.",
            "K_o": "Number of distinct exact token-ID n-gram tuples in the accumulated write stream.",
            "U_o": "Number of distinct physical hash rows addressed in that order; exact D1 allocated-row count.",
            "E_o_minus_K_o": "Repeated occurrences of already-seen exact keys; not hash collisions.",
            "K_o_minus_U_o": "Distinct-key excess after hash merging; not number of colliding key pairs.",
            "q_o": "U_o / 16384", "q_total": "(U_2 + U_3) / 32768",
            "length_bound": "U_o <= min(16384, sum_articles max(0, token_count - o + 1)).",
        },
        "memory_semantics": {
            "orders": list(ORDERS), "buckets_per_order": BUCKETS, "total_physical_rows": 2 * BUCKETS,
            "operator": "D1", "state_reset": "Fresh state for each family and each self/donor arm.",
            "between_distances": "Nested distractor prefixes continue the same state.",
            "article_boundaries": "Memory retained; n-grams never cross an article boundary.",
            "block_boundaries": "512-token blocks include up to two prior tokens for addressing only.",
            "allocation": "Every valid nonzero address is allocated even when beta is zero; no row eviction.",
            "query_and_permutation": "Queries and channel permutations do not change addressed/allocated rows.",
            "checkpoint_dependence": "Under the same verified D1 config and write/token schedule, E/K/U do not depend on writer weights.",
        },
        "scope": {
            "historical_maximum_occupancy_34_33_percent_applies_to_new_dev64": False,
            "exactness": "Exact replay of allocation addresses; not an in-run state.rows log.",
            "neural_values_gates_or_losses_replayed": False,
            "proves": "Physical hash-row occupancy and exact-key/hash merging for all scored DEV64 write schedules.",
            "does_not_prove": "Semantic storage capacity, useful memory retention, absence of interference, or a cause of an NLL change.",
            "across_checkpoints": "Applicable to a checkpoint only after its config, source/token hashes, and per-cell write budgets match this result.",
            "empty_arm_occupancy": 0,
            "self_perm_and_donor_perm": "Same allocation as self and donor, respectively; only read channels are permuted.",
        },
        "tokenizer": {
            "name_or_path": registry["model"]["name_or_path"], "revision": REVISION,
            "local_identity": identity, "scored_identity": result["manifest"]["tokenizer"],
            "identity_metadata_exact_match": identity == result["manifest"]["tokenizer"],
            "all_written_token_sequences_sha256_exact_match": True,
            "identity_note": "Exact per-source token-sequence hashes establish replay identity even if library class/pipeline metadata differs.",
        },
        "verification": {
            "unique_written_sources": len(cache), "unique_source_tokens": sum(a["token_count"] for a in cache.values()),
            "checked_kernel_address_positions": sum(a["checked_address_positions"] for a in cache.values()),
            "family_distance_arm_cells": len(rows), "all_runtime_write_budgets_match": True,
            "all_block_addresses_match_original_cpu_hash_kernel": True,
            "positive_count_rows_equal_address_union_by_D1_source": True,
            "equivalent_completed_checkpoints_verified": len(equivalent_results),
            "expected_registered_checkpoints": 21,
            "all_21_checkpoints_verified": len(equivalent_results) == 21,
        },
        "equivalent_completed_results": equivalent_results,
        "source_inputs": source_receipts,
        "code_sha256": {p: sha(REPO / p) for p in source_paths},
        "runtime": {"elapsed_seconds": time.perf_counter() - started,
                    "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
                    "torch_threads": torch.get_num_threads(), "device": "cpu",
                    "model_constructed": False, "writer_constructed": False, "reader_constructed": False,
                    "network_used": False, "final_data_opened": False},
        "overall_maximum": maximum,
        "maximum_per_order": {
            str(order): {key: value for key, value in max(rows, key=lambda r: r[f"U{order}"]).items()
                         if key in ("event_family", "pair_id", "arm", "unrelated_article_count",
                                    f"E{order}", f"K{order}", f"U{order}", f"occupancy_U{order}_over_B")}
            for order in ORDERS
        },
        "per_distance_arm_summary": summary,
        "artifacts": ["capacity_family_distance.csv", "capacity_distance_summary.csv", "capacity_scope_note.json",
                      "capacity_interpretation.md"],
        "reproduction": {
            "command": "python delta-ngram/scripts/audit_m1_dev64_capacity_20261020.py --result <complete_DEV64_JSON> --registry <frozen_DEV64_registry_JSON>",
            "inputs": "One complete DEV64 result, frozen DEV registry and write-article text files, cached matching tokenizer, original source kernel.",
            "dependencies": "torch and transformers for CPU kernel verification/tokenization; no model weights.",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "capacity_family_distance.csv", rows)
    write_csv(args.output_dir / "capacity_distance_summary.csv", summary)
    write_json(args.output_dir / "capacity_scope_note.json", note)
    write_interpretation(args.output_dir / "capacity_interpretation.md", note)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "cells": len(rows),
                      "max_occupancy": maximum["occupancy_U_total_over_2B"],
                      "max_U": maximum["U_total"], "runtime_seconds": note["runtime"]["elapsed_seconds"]}))


if __name__ == "__main__":
    main()

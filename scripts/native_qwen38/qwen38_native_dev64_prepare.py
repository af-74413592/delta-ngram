"""Derive fixed DEV64 from frozen UTF-8 originals with the native tokenizer.

CPU text preparation only: no model, reader, CE, training, or FINAL input.
The frozen 64 families, fixed donor IDs and distractor ordering are preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_qwen38_native_data import (
    MODEL_ID, REVISION, digest, file_info, local_tokenizer_receipt,
    require, tokenizer_identity, write_json,
)

REGISTRY_SHA = "9ea663ad70131e272522cbad8f6c0fa20bc03f41b324fe7b71d0fb8f4fbfccc2"
ANNOTATIONS_SHA = "7164e9f8b6f23ebfeaaccc2687a6a8b6ad222803f705bacd1a203a2596fdbcce"
TOKENIZER_FINGERPRINT = "sha256:47a5ecb7efba199d03fdebf84bfffb685da70065ff76c4a160dd08c9bc4771ec"
DISTANCES, SEEDS = (0, 1, 4, 8, 16), (17, 29, 43)
PREFIX, WRITE_BLOCK, TARGET_BLOCK, VOCABULARY = 128, 512, 384, 248320


def read_sources(registry_path, annotations_path):
    registry_path, annotations_path = Path(registry_path).resolve(), Path(annotations_path).resolve()
    require(file_info(registry_path)["sha256"] == REGISTRY_SHA, "frozen DEV64 registry differs")
    require(file_info(annotations_path)["sha256"] == ANNOTATIONS_SHA, "frozen prefix annotations differ")
    registry = json.loads(registry_path.read_bytes())
    annotations = json.loads(annotations_path.read_bytes())
    require(registry["protocol"] == "recent_news_continuation_v1"
            and registry["split"] == "dev" and registry["evaluation_only"] is True,
            "only the frozen evaluation-only DEV registry is accepted")
    require(annotations["parent_registry_file_sha256"] == REGISTRY_SHA
            and annotations["initial_prefix_tokens"] == PREFIX
            and annotations["model_results_seen"] is False, "annotation parent differs")
    sources = registry["sources"] + registry["background_sources"]
    pairs = registry["pairs"]
    require(len(pairs) == 64 and len(registry["sources"]) == 128
            and len(registry["background_sources"]) == 18
            and len({p["event_family"] for p in pairs}) == 64, "frozen membership differs")
    indexed = {source["id"]: source for source in sources}
    require(len(indexed) == len(sources), "duplicate source IDs")
    later_ids = {p["later_source_id"] for p in pairs}
    by_pair = {row["pair_id"]: row for row in annotations["annotations"]}
    require(len(by_pair) == 64 and set(by_pair) == {p["id"] for p in pairs}, "annotation coverage differs")
    texts, files = {}, {}
    for source in sources:
        for stem in ("extracted_text", "raw_response", "raw_record"):
            relative = source.get(stem + "_relpath")
            if relative is None:
                require(stem != "extracted_text", "source lacks original text")
                continue
            path = (registry_path.parent / relative).resolve()
            require(path.is_relative_to(registry_path.parent), "source path escapes frozen registry")
            actual = file_info(path)
            require(actual["sha256"] == source[stem + "_sha256"], "source hash differs: " + relative)
            require(relative not in files or files[relative] == actual, "conflicting source identity")
            files[relative] = actual
            if stem == "extracted_text":
                text = path.read_bytes().decode("utf-8")
                require(text and "\ufffd" not in text and "\r" not in text, "source must be UTF-8/LF")
                texts[source["id"]] = text
    for pair in pairs:
        require(pair["earlier_source_id"] in indexed and pair["later_source_id"] in indexed
                and pair["donor_source_id"] in indexed, "unknown pair source")
        writes = [pair["earlier_source_id"], pair["donor_source_id"], *pair["distractor_source_ids"]]
        require(not later_ids.intersection(writes) and len(pair["distractor_source_ids"]) == 16,
                "target article cannot enter any write stream")
        require(tuple(row["unrelated_article_count"] for row in pair["interference_conditions"])
                == DISTANCES, "interference distances differ")
        for row in pair["interference_conditions"]:
            require(row["distractor_source_ids"] == pair["distractor_source_ids"][:row["unrelated_article_count"]],
                    "interference ordering differs")
    return registry, annotations, texts, files


def derive_prefix(pair, original, texts, tokens, tokenizer):
    late_id, early_id = pair["later_source_id"], pair["earlier_source_id"]
    later, earlier = texts[late_id], texts[early_id]
    old_end = original["later_prefix_char_end_exclusive"]
    require(hashlib.sha256(later[:old_end].encode()).hexdigest() == original["later_prefix_text_sha256"],
            "original annotated prefix bytes differ")
    matching = []
    for fact in original["key_facts"]:
        for name, source_id, text in (("earlier_span", early_id, earlier),
                                      ("later_prefix_span", late_id, later[:old_end])):
            span = fact.get(name)
            if span is None:
                continue
            start, end = span["char_start"], span["char_end_exclusive"]
            require(span["source_id"] == source_id and 0 <= start < end <= len(text)
                    and hashlib.sha256(text[start:end].encode()).hexdigest() == span["text_sha256"],
                    "frozen fact span differs")
    prefix_ids = tokens[late_id][:PREFIX]
    prefix = tokenizer.decode(prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    exact = later.startswith(prefix) and "\ufffd" not in prefix
    new_end = len(prefix) if exact else None
    if exact:
        matching = [fact["id"] for fact in original["key_facts"]
                    if fact.get("later_prefix_span", {}).get("char_end_exclusive", len(later) + 1) <= new_end]
    if not exact:
        label, reason = "unknown", "native token boundary does not decode to an exact original-text prefix"
    elif matching:
        label, reason = "restated", "frozen positive proposition span remains wholly inside the native prefix"
    elif original["label"] == "not_restated" and new_end <= old_end:
        label, reason = "not_restated", "native prefix is a subset of the frozen negatively annotated prefix"
    else:
        label, reason = "unknown", "changed prefix has no logically sufficient inherited label; no semantic claim"
    return {"pair_id": pair["id"], "event_family": pair["event_family"], "label": label,
            "parent_label": original["label"], "rationale": reason,
            "review_method": "source-span-and-prefix-containment-only; no model scores",
            "old_prefix_char_end_exclusive": old_end, "native_prefix_char_end_exclusive": new_end,
            "native_prefix_is_exact_original_prefix": exact, "native_prefix_text": prefix,
            "native_prefix_token_ids_sha256": digest(prefix_ids),
            "native_prefix_text_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
            "retained_positive_fact_ids": matching, "parent_key_facts": copy.deepcopy(original["key_facts"]),
            "revision": copy.deepcopy(original.get("revision", {"eligible": False}))}


def prepare(registry_path, annotations_path, output, tokenizer, tokenizer_receipt):
    registry, annotations, texts, source_files = read_sources(registry_path, annotations_path)
    output = Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(Path(registry_path).resolve().parent),
            "use a new output outside the frozen source root")
    identity = tokenizer_identity(tokenizer)
    fingerprint = "sha256:" + digest(identity)
    require(fingerprint == TOKENIZER_FINGERPRINT, "DEV tokenizer differs from native training tokenizer")
    tokens, sources = {}, []
    for source in registry["sources"] + registry["background_sources"]:
        ids = tokenizer.encode(texts[source["id"]], add_special_tokens=False)
        require(ids and all(type(token) is int and 0 <= token < VOCABULARY for token in ids),
                "invalid native token IDs")
        tokens[source["id"]] = ids
        derived = {key: value for key, value in source.items() if not key.endswith("_path")}
        derived.update(parent_token_count=source["token_count"],
                       parent_token_ids_sha256=source.get("token_ids_sha256"),
                       token_count=len(ids), token_ids_sha256=digest(ids),
                       raw_text_characters=len(texts[source["id"]]),
                       raw_text_UTF8_bytes=len(texts[source["id"]].encode()),
                       write_blocks=math.ceil(len(ids) / WRITE_BLOCK))
        sources.append(derived)
    pairs, prefix_rows = copy.deepcopy(registry["pairs"]), []
    old_annotations = {row["pair_id"]: row for row in annotations["annotations"]}
    for pair in pairs:
        early, late, donor = (pair[key] for key in ("earlier_source_id", "later_source_id", "donor_source_id"))
        require(len(tokens[late]) > PREFIX, "native target is too short; frozen family must not be replaced")
        pair["parent_scoring"] = copy.deepcopy(pair["scoring"])
        pair.update(earlier_token_count=len(tokens[early]), later_token_count=len(tokens[late]),
                    earlier_write_token_count=len(tokens[early]), donor_write_token_count=len(tokens[donor]),
                    earlier_write_blocks=math.ceil(len(tokens[early]) / WRITE_BLOCK),
                    donor_write_blocks=math.ceil(len(tokens[donor]) / WRITE_BLOCK))
        pair["scoring"]["target_end_token_exclusive"] = len(tokens[late])
        prefix = derive_prefix(pair, old_annotations[pair["id"]], texts, tokens, tokenizer)
        pair["native_prefix_annotation"] = prefix
        prefix_rows.append(prefix)
        for row in pair["interference_conditions"]:
            selected = row["distractor_source_ids"]
            count, blocks = sum(len(tokens[key]) for key in selected), sum(
                math.ceil(len(tokens[key]) / WRITE_BLOCK) for key in selected)
            row.update(distractor_token_count=count, distractor_block_count=blocks)
            for arm, anchor in (("self", early), ("donor", donor)):
                row[arm + "_actual_written_articles"] = len(selected) + 1
                row[arm + "_actual_written_tokens"] = count + len(tokens[anchor])
                row[arm + "_actual_written_blocks"] = blocks + math.ceil(len(tokens[anchor]) / WRITE_BLOCK)
    targets = sum(len(tokens[pair["later_source_id"]]) - PREFIX for pair in pairs)
    plan = {
        "protocol": "qwen38_native_dev64_plan_v1", "split": "dev", "evaluation_only": True,
        "model": {"name": MODEL_ID, "revision": REVISION}, "parent_model": registry["model"],
        "source_registry_file_sha256": REGISTRY_SHA, "source_annotations_file_sha256": ANNOTATIONS_SHA,
        "source_files": source_files, "tokenizer": identity, "tokenizer_fingerprint": fingerprint,
        "tokenizer_receipt": tokenizer_receipt, "add_special_tokens": False,
        "sources": sources, "pairs": pairs, "independent_families": 64, "distances": list(DISTANCES),
        "write_block_tokens": WRITE_BLOCK, "initial_prefix_tokens": PREFIX,
        "maximum_target_tokens_per_window": TARGET_BLOCK, "target_tokens_per_distance": targets,
        "prefix_annotations": prefix_rows, "prefix_label_counts": dict(Counter(row["label"] for row in prefix_rows)),
        "source_membership_donor_and_distractor_order_unchanged": True,
        "evaluation": {"conditions": ["self", "empty", "donor", "self_perm", "donor_perm"],
                       "direction_control_seeds": list(SEEDS), "ce_target_accesses": 41 * targets,
                       "ce_multiplier": "1 empty + 5 distances * 2 memory arms * (1 + 3 permutations)",
                       "bootstrap_seed": 20261019, "bootstrap_samples": 10000,
                       "primary": "equal-family mean NLL(empty)-NLL(self) and NLL(donor)-NLL(self)",
                       "pooled": "sum loss / sum target tokens; report separately from equal-family mean",
                       "query_memory_writes": False, "optimizer_updates": 0,
                       "cross_tokenizer_nat_per_token_direct_comparison": False},
        "prepared_utc": datetime.now(timezone.utc).isoformat(), "CE_executed": False,
        "pretraining_cutoff": "unknown", "base_pretraining_novelty_certified": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(output / "tokens.json", tokens)
    write_json(output / "COMPLETE.json", {
        "protocol": "qwen38_native_dev64_seal_v1", "ready_for_evaluation": True,
        "source_registry_file_sha256": REGISTRY_SHA, "source_annotations_file_sha256": ANNOTATIONS_SHA,
        "tokenizer_fingerprint": fingerprint, "files": {
            name: file_info(output / name) for name in ("plan.json", "tokens.json")}})
    return plan


def load_native_dev64_cache(root, *, expected_complete_sha256=None):
    root = Path(root).resolve()
    if root.is_file():
        require(root.name == "plan.json", "DEV64 file argument must be plan.json")
        root = root.parent
    seal_info = file_info(root / "COMPLETE.json")
    require(expected_complete_sha256 is None or seal_info["sha256"] == expected_complete_sha256,
            "DEV64 complete seal differs")
    seal = json.loads((root / "COMPLETE.json").read_bytes())
    require(seal.get("protocol") == "qwen38_native_dev64_seal_v1"
            and seal.get("ready_for_evaluation") is True, "DEV64 cache is not sealed")
    for name in ("plan.json", "tokens.json"):
        require(file_info(root / name) == seal["files"][name], "DEV64 cache file differs: " + name)
    plan, tokens = (json.loads((root / name).read_bytes()) for name in ("plan.json", "tokens.json"))
    require(plan["protocol"] == "qwen38_native_dev64_plan_v1" and plan["independent_families"] == 64
            and plan["source_registry_file_sha256"] == REGISTRY_SHA
            and plan["source_annotations_file_sha256"] == ANNOTATIONS_SHA
            and plan["tokenizer_fingerprint"] == TOKENIZER_FINGERPRINT, "DEV64 data identity differs")
    require(set(tokens) == {row["id"] for row in plan["sources"]}, "DEV64 token membership differs")
    for source in plan["sources"]:
        ids = tokens[source["id"]]
        require(len(ids) == source["token_count"] and digest(ids) == source["token_ids_sha256"],
                "DEV64 token content differs")
    actual_targets = sum(len(tokens[pair["later_source_id"]]) - PREFIX for pair in plan["pairs"])
    require(actual_targets == plan["target_tokens_per_distance"]
            and plan["evaluation"]["ce_target_accesses"] == 41 * actual_targets, "DEV64 CE count differs")
    return {"plan": plan, "tokens": tokens, "identity": {"complete": seal_info, **seal}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registry", "annotations", "tokenizer-dir", "tokenizer-files-manifest", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    receipt = local_tokenizer_receipt(args.tokenizer_dir, expected_manifest=args.tokenizer_files_manifest)
    from transformers import AutoTokenizer
    import transformers
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_dir), local_files_only=True, trust_remote_code=False)
    receipt["transformers_version"] = transformers.__version__
    plan = prepare(args.registry, args.annotations, args.output_dir, tokenizer, receipt)
    loaded = load_native_dev64_cache(args.output_dir)
    print(json.dumps({"status": "native_DEV64_prepared_zero_CE", "output": str(args.output_dir),
                      "families": 64, "sources": len(plan["sources"]),
                      "target_tokens_per_distance": plan["target_tokens_per_distance"],
                      "evaluation_CE_targets": plan["evaluation"]["ce_target_accesses"],
                      "prefix_label_counts": plan["prefix_label_counts"], "identity": loaded["identity"]}))


if __name__ == "__main__":
    main()

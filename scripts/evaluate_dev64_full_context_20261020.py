"""Raw-weight full-source-context reference for the frozen DEV64 continuation.

This independent runner never constructs or restores a writer, memory adapter,
or memory state. Each query retains the original 128-token rolling B prefix.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
PROTOCOL = "recent_news_dev64_full_context_20261020_v1"
MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
REGISTRY_SHA = "9ea663ad70131e272522cbad8f6c0fa20bc03f41b324fe7b71d0fb8f4fbfccc2"
ANNOTATIONS_SHA = "7164e9f8b6f23ebfeaaccc2687a6a8b6ad222803f705bacd1a203a2596fdbcce"
EVALUATOR_SHA = "cb511e7b8ef95bd670a2f9d087e4491d463077a61a3a6fb6367f685694e5e192"
REFERENCE_SHA = "7051e93ad164a90e26623a852b2679e1d9d619809cd14f5aabc69635402c9f0c"
DISTANCES = (0, 1, 4, 8, 16)
EMPTY_NLL_ATOL = 0.0001
LOGIT_PARITY_ATOL = 0.0001
CONTEXT_PROTOCOL = {
    "evidence": "A followed by the same nested ordered unrelated articles as the frozen self-memory arm",
    "separator_text": "\n\n", "tokenization": "concatenate individually frozen article token IDs and separately encoded separator IDs",
    "query_prefix_tokens": 128, "target_block_tokens": 384,
    "query_window": "B[start-128:end], with start=128,512,... and end=min(start+384,len(B))",
    "causality": "standard causal attention; only each target's preceding positions predict it",
    "query_history": "No earlier B history beyond the identical local window; reset all KV state per window",
    "truncation_allowed": False, "chat_template": False, "special_tokens_added": False,
    "memory_injection": False, "writer_checkpoint_loaded": False,
    "interpretation": "Observed full-source-context NLL benefit reference, not a theoretical upper bound",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def position_policy(config):
    limit = getattr(config, "max_position_embeddings", None)
    require(type(limit) is int and limit > 0, "A finite model position limit is required")
    require(not getattr(config, "use_sliding_window", False), "Sliding-window attention is not allowed")
    require(getattr(config, "sliding_window", None) is None, "Unexpected sliding-window configuration")
    kinds = getattr(config, "layer_types", None)
    require(kinds is None or set(kinds) == {"full_attention"}, "All layers must retain full context")
    return {"max_position_embeddings": limit, "sliding_window": None,
            "use_sliding_window": False, "layer_types": kinds}


def make_plans(prepared, separator, maximum):
    from delta_ngram.recent_news_evaluation import scoring_windows
    require(bool(separator), "The fixed article separator must contain tokens")
    plans = []
    for pair in prepared.registry["pairs"]:
        later = prepared.tokens[pair["later_source_id"]]
        for condition in pair["interference_conditions"]:
            distance = condition["unrelated_article_count"]
            source_ids = [pair["earlier_source_id"], *condition["distractor_source_ids"]]
            require(source_ids[1:] == pair["distractor_source_ids"][:distance],
                    "Distractor order differs from the frozen self arm")
            require(pair["later_source_id"] not in source_ids, "Target article entered evidence")
            evidence = tuple(token for source in source_ids
                             for token in (*prepared.tokens[source], *separator))
            windows = []
            for context_start, target_start, target_end in scoring_windows(len(later)):
                query = later[context_start:target_end]
                combined = (*evidence, *query)
                require(len(combined) <= maximum, "Full context exceeds model limit; refusing to truncate")
                windows.append({
                    "query_context_start_token": context_start, "target_start_token": target_start,
                    "target_end_token_exclusive": target_end, "target_tokens": target_end - target_start,
                    "query_prefix_token_ids_sha256": canonical_sha(later[context_start:target_start]),
                    "target_token_ids_sha256": canonical_sha(later[target_start:target_end]),
                    "input_token_ids_sha256": canonical_sha(combined), "reader_input_tokens": len(combined),
                    "target_start_position": len(evidence) + 128,
                    "target_end_position_exclusive": len(combined),
                })
            require(sum(w["target_tokens"] for w in windows) == len(later) - 128,
                    "Targets are not covered exactly once")
            plans.append({
                "pair_id": pair["id"], "event_family": pair["event_family"], "domain": pair["domain"],
                "later_source_id": pair["later_source_id"], "unrelated_article_count": distance,
                "context_source_ids": source_ids, "source_article_tokens": sum(len(prepared.tokens[s]) for s in source_ids),
                "separator_tokens": len(separator) * len(source_ids), "evidence_tokens": len(evidence),
                "evidence_token_ids_sha256": canonical_sha(evidence),
                "target_token_ids_sha256": canonical_sha(later[128:]), "target_tokens": len(later) - 128,
                "windows": windows, "_evidence_ids": evidence,
            })
    return plans


def prepare(args):
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    import delta_ngram.recent_news_evaluation as news
    from delta_ngram.raw_lm import tokenizer_identity
    reference_path = Path(args.reference).resolve()
    require(sha(reference_path) == REFERENCE_SHA, "Expected the frozen primary 100M reference bytes")
    reference = json.loads(reference_path.read_bytes())
    require(reference["complete"] is True and reference["independent_families"] == 64,
            "Reference must be a complete frozen DEV64 result")
    require(reference["model"]["name_or_path"] == MODEL and reference["model"]["revision"] == REVISION
            and reference["model"]["dtype"] == "torch.bfloat16", "Reference reader identity differs")
    require(reference["manifest"]["registry_file_sha256"] == REGISTRY_SHA,
            "Reference registry identity differs")
    require(reference["manifest"]["prefix_annotations"]["file_sha256"] == ANNOTATIONS_SHA,
            "Reference annotation identity differs")
    evaluator_bytes = Path(news.__file__).read_bytes().replace(b"\r\n", b"\n")
    require(hashlib.sha256(evaluator_bytes).hexdigest() == EVALUATOR_SHA, "Frozen preparation source differs")
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    require(config.model_type == "qwen3" and getattr(config, "_commit_hash", None) == REVISION,
            "Model config revision differs")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    # Preparation needs only vocabulary bounds, never model weights.
    proxy = SimpleNamespace(tokenizer=tokenizer, reader=SimpleNamespace(model=SimpleNamespace(
        get_input_embeddings=lambda: SimpleNamespace(weight=SimpleNamespace(shape=(config.vocab_size, 1))))))
    prepared = news.prepare_continuation(args.registry, proxy, tokenizer_source=MODEL,
        tokenizer_revision=REVISION, expected_sha256=REGISTRY_SHA)
    annotations = news.prepare_prefix_annotations(args.annotations, prepared, proxy,
        expected_sha256=ANNOTATIONS_SHA)
    require(tokenizer_identity(proxy) == reference["manifest"]["tokenizer"], "Tokenizer pipeline differs")
    source_identity = lambda rows: {row["id"]: (row["token_count"], row["token_ids_sha256"],
        row["extracted_text_sha256"]) for row in rows}
    require(source_identity(prepared.manifest["sources"]) ==
            source_identity(reference["manifest"]["sources"]), "Frozen source token identity differs")
    require(prepared.manifest["pairs"] == reference["manifest"]["pairs"], "Frozen causal plans differ")
    policy = position_policy(config)
    separator = tuple(tokenizer.encode("\n\n", add_special_tokens=False))
    require(separator == (271,), "Pinned separator token IDs differ")
    plans = make_plans(prepared, separator, policy["max_position_embeddings"])
    rows = reference["per_family"]
    indexed = {(row["event_family"], row["unrelated_article_count"]): row for row in rows}
    require(len(indexed) == len(rows) == len(plans) == 320, "Expected 320 family-distance cells")
    empty_by_family = {}
    for plan in plans:
        old = indexed[plan["event_family"], plan["unrelated_article_count"]]
        require(old["pair_id"] == plan["pair_id"] and old["later_source_id"] == plan["later_source_id"],
                "Reference pair mapping differs")
        require(old["conditions"]["self"]["write_source_ids"] == plan["context_source_ids"],
                "Context source order differs from executed memory self arm")
        empty = old["conditions"]["empty"]
        require(empty["target_tokens"] == plan["target_tokens"] and
                empty["read_windows"] == len(plan["windows"]), "Reference target coverage differs")
        if plan["event_family"] in empty_by_family:
            require(empty_by_family[plan["event_family"]] == empty, "Reference empty varies by distance")
        empty_by_family[plan["event_family"]] = empty
    require(len(empty_by_family) == 64, "Expected 64 independent event families")
    require(sum(len(p["windows"]) for p in plans) == 865, "Expected 865 full-context read windows")
    targets = {str(d): sum(p["target_tokens"] for p in plans if p["unrelated_article_count"] == d)
               for d in DISTANCES}
    require(set(targets.values()) == {52914}, "Every distance must score all 52,914 original targets")
    public_plans = [{key: value for key, value in plan.items() if not key.startswith("_")} for plan in plans]
    registration = {
        "protocol": PROTOCOL, "reader_model": {"name_or_path": MODEL, "revision": REVISION,
            "dtype": "torch.bfloat16", "attn_implementation": "sdpa", **policy},
        "context_protocol": {**CONTEXT_PROTOCOL, "separator_token_ids": list(separator)},
        "registry_file_sha256": REGISTRY_SHA, "annotations_file_sha256": ANNOTATIONS_SHA,
        "tokenizer": prepared.manifest["tokenizer"], "tokenizer_fingerprint": prepared.manifest["tokenizer_fingerprint"],
        "reference_file_sha256": sha(reference_path), "reference_checkpoint_sha256": reference["checkpoint"]["sha256"],
        "evaluator_source_sha256_lf": EVALUATOR_SHA,
        "runner_source_sha256_lf": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "families": 64, "context_cells": 320, "context_read_windows": 865, "empty_read_windows": 173,
        "target_tokens_per_distance": targets, "maximum_context_tokens": max(w["reader_input_tokens"] for p in plans for w in p["windows"]),
        "total_context_reader_input_tokens": sum(w["reader_input_tokens"] for p in plans for w in p["windows"]),
        "empty_nll_absolute_tolerance": EMPTY_NLL_ATOL, "logits_to_keep_nll_parity_absolute_tolerance": LOGIT_PARITY_ATOL,
        "logit_selection": "Keep T+1 trailing logits for T targets, discard the final unused logit",
        "sanity_policy": "All 64 original empty family total and first-window NLLs must match frozen empty; coverage is exactly 173 windows, whose individual old losses were not retained",
        "ratio_empty_reference": "Use the same frozen primary-reference empty for both numerator and denominator; raw recomputed empty is a separate sanity check",
        "plans": public_plans,
    }
    preflight = {"protocol": PROTOCOL, "stage": "cpu_preflight", "model_weights_loaded": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "registration": registration,
        "registration_sha256": canonical_sha(registration),
        "runtime": {"python": sys.version, "transformers": transformers.__version__},
        "inputs": {"reference": str(reference_path), "registry": str(Path(args.registry).resolve()),
                   "annotations": str(Path(args.annotations).resolve())}}
    manifest = {**prepared.manifest, "prefix_annotations": annotations.manifest}
    return config, prepared, annotations, reference, empty_by_family, plans, manifest, preflight


def selected_loss(model, input_ids, target_count, *, full_logits=False):
    """Predict final T input tokens using positions [-T-1:-1], never the token itself."""
    import torch
    import torch.nn.functional as F
    require(0 < target_count < input_ids.shape[1], "Targets require a preceding causal prefix")
    mask = torch.ones_like(input_ids)
    output = model(input_ids=input_ids, attention_mask=mask, use_cache=False, return_dict=True,
                   logits_to_keep=0 if full_logits else target_count + 1)
    logits = output.logits[:, -target_count - 1:-1, :]
    targets = input_ids[:, -target_count:]
    require(logits.shape[:2] == targets.shape, "Causal logit/target alignment differs")
    value = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                            targets.reshape(-1), reduction="sum")
    require(bool(torch.isfinite(value)), "Nonfinite context target loss")
    return float(value)


def score_article(model, later, windows, evidence, device):
    import torch
    losses = []
    for window in windows:
        query = later[window["query_context_start_token"]:window["target_end_token_exclusive"]]
        ids = torch.tensor([(*evidence, *query)], dtype=torch.long, device=device)
        value = selected_loss(model, ids, window["target_tokens"])
        losses.append({**window, "loss_sum": value,
                       "reader_input_tokens": ids.numel(),
                       "input_token_ids_sha256": canonical_sha(tuple(evidence) + tuple(query)),
                       "target_start_position": len(evidence) + 128,
                       "target_end_position_exclusive": ids.shape[1]})
    total = sum(w["target_tokens"] for w in windows)
    return {"loss_sum": math.fsum(row["loss_sum"] for row in losses), "target_tokens": total,
        "nll": math.fsum(row["loss_sum"] for row in losses) / total,
        "read_windows": len(losses), "reader_input_tokens": sum(row["reader_input_tokens"] for row in losses),
        "first_target_loss_sum": losses[0]["loss_sum"], "first_target_tokens": windows[0]["target_tokens"],
        "first_target_nll": losses[0]["loss_sum"] / windows[0]["target_tokens"], "windows": losses}


def assert_unhooked(model):
    require(all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules()),
            "Raw model must not have forward hooks or memory adapters")
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()),
            "Raw model parameters must be frozen and gradient-free")


def evaluate(args, prepared, annotations, reference, old_empty, plans, manifest, preflight):
    import torch
    from transformers import AutoModelForCausalLM
    require(args.device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            "Recorded evaluation requires a BF16-capable CUDA device")
    require(args.preflight is not None, "Save the CPU preflight registration before GPU execution")
    saved = json.loads(Path(args.preflight).read_bytes())
    require(saved["registration_sha256"] == canonical_sha(saved["registration"]) ==
            preflight["registration_sha256"], "Inputs or runner changed after the CPU preflight")
    device = torch.device(args.device)
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    model.eval().requires_grad_(False)
    model.config.use_cache = False
    require(getattr(model.config, "_commit_hash", None) == REVISION, "Loaded model revision differs")
    require(str(next(model.parameters()).dtype) == "torch.bfloat16", "Loaded model dtype differs")
    require(getattr(model.config, "_attn_implementation", None) == "sdpa", "Loaded attention backend differs")
    require(position_policy(model.config) == {
                key: preflight["registration"]["reader_model"][key]
                for key in ("max_position_embeddings", "sliding_window", "use_sliding_window", "layer_types")},
            "Loaded model position policy differs")
    require("logits_to_keep" in inspect.signature(model.forward).parameters,
            "Installed Qwen3 model does not support selected trailing logits")
    assert_unhooked(model)
    versions = {name: (value, value._version) for name, value in model.named_parameters()}
    base_plans = [plan for plan in plans if plan["unrelated_article_count"] == 0]
    fresh_empty, sanity = {}, []
    with torch.no_grad():
        # Compare full-vocabulary-position scoring with selected tail scoring on
        # both a full-sized and a short final window, without long-context logits.
        probes = [base_plans[0]["windows"][0], base_plans[0]["windows"][-1]]
        later = prepared.tokens[base_plans[0]["later_source_id"]]
        parity = []
        for window in probes:
            ids = torch.tensor([later[window["query_context_start_token"]:window["target_end_token_exclusive"]]],
                               dtype=torch.long, device=device)
            selected = selected_loss(model, ids, window["target_tokens"])
            full = selected_loss(model, ids, window["target_tokens"], full_logits=True)
            difference = abs(selected - full) / window["target_tokens"]
            require(difference <= LOGIT_PARITY_ATOL, "Selected logits differ from full-position CE")
            parity.append({"target_tokens": window["target_tokens"], "nll_absolute_difference": difference})
        for plan in base_plans:
            family = plan["event_family"]
            score = score_article(model, prepared.tokens[plan["later_source_id"]], plan["windows"], (), device)
            expected = old_empty[family]
            require(score["target_tokens"] == expected["target_tokens"] and
                    score["read_windows"] == expected["read_windows"], "Empty target coverage differs")
            differences = {key: abs(score[key] - expected[key]) for key in ("nll", "first_target_nll")}
            require(max(differences.values()) <= EMPTY_NLL_ATOL,
                    "Raw empty differs from frozen empty: " + family + " " + json.dumps(differences))
            fresh_empty[family] = score
            sanity.append({"event_family": family, **differences, "passed": True})
            print(json.dumps({"stage": "empty_sanity", "families_complete": len(sanity), "families_total": 64,
                              "max_nll_difference": max(differences.values())}), flush=True)
        write_new(str(args.output) + ".empty_sanity.json", {
            "complete": True, "stage": "raw_empty_sanity", "protocol": PROTOCOL,
            "preflight_registration_sha256": preflight["registration_sha256"],
            "reference_sha256": preflight["registration"]["reference_file_sha256"],
            "tolerance": EMPTY_NLL_ATOL, "selected_logits_parity": parity, "families": sanity,
            "scores": fresh_empty, "elapsed_seconds": time.monotonic() - started})
        records = []
        if not args.sanity_only:
            progress_path = Path(str(args.output) + ".progress.jsonl")
            with progress_path.open("x", encoding="utf-8", newline="\n") as progress:
                for plan in plans:
                    score = score_article(model, prepared.tokens[plan["later_source_id"]],
                                          plan["windows"], plan["_evidence_ids"], device)
                    record = {key: value for key, value in plan.items()
                              if not key.startswith("_") and key != "windows"}
                    record["conditions"] = {"empty": old_empty[plan["event_family"]], "context": score}
                    record["empty_baseline_source"] = "frozen_primary_100m_reference"
                    records.append(record)
                    progress.write(json.dumps(record, allow_nan=False) + "\n")
                    progress.flush()
                    print(json.dumps({"stage": "context_cell", "cells_complete": len(records),
                        "cells_total": len(plans), "pair_id": plan["pair_id"],
                        "unrelated_article_count": plan["unrelated_article_count"],
                        "elapsed_seconds": time.monotonic() - started}), flush=True)
    assert_unhooked(model)
    current = dict(model.named_parameters())
    require(current.keys() == versions.keys() and all(current[name] is value and value._version == version
            for name, (value, version) in versions.items()), "Raw model parameters changed")
    prepared.verify_files()
    annotations.verify_file()
    require(sha(args.reference) == preflight["registration"]["reference_file_sha256"],
            "Reference changed during evaluation")
    require(hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest() ==
            preflight["registration"]["runner_source_sha256_lf"], "Runner source changed during evaluation")
    model_info = {"name_or_path": MODEL, "revision": REVISION, "resolved_commit": REVISION,
        "dtype": "torch.bfloat16", "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
        "attn_implementation": "sdpa"}
    return {"protocol": PROTOCOL, "complete": not args.sanity_only,
        "stage": "empty_sanity_complete" if args.sanity_only else "full_context_complete",
        "evaluation_only": True, "split": "dev", "independent_families": 64,
        "optimizer_steps": 0, "backward_calls": 0, "model": model_info, "manifest": manifest,
        "context_protocol": preflight["registration"]["context_protocol"],
        "preflight_registration_sha256": preflight["registration_sha256"], "preflight_sha256": sha(args.preflight),
        "reference": {"path": str(Path(args.reference).resolve()), "sha256": sha(args.reference),
                      "checkpoint_sha256": reference["checkpoint"]["sha256"]},
        "per_family": records, "raw_empty_sanity_per_family": fresh_empty,
        "empty_sanity": {"all_64_families_passed": True, "tolerance": EMPTY_NLL_ATOL,
            "maximum_nll_absolute_difference": max(row["nll"] for row in sanity),
            "selected_logits_parity": parity},
        "execution": {"empty_families": 64, "empty_read_windows": 173, "context_cells": len(records),
                      "context_read_windows": sum(row["conditions"]["context"]["read_windows"] for row in records),
                      "elapsed_seconds": time.monotonic() - started},
        "integrity_checks": {"no_writer_checkpoint_loaded": True, "no_memory_adapter_constructed": True,
            "no_forward_hooks": True, "parameters_frozen_no_gradients": True,
            "parameter_objects_and_versions_unchanged": True, "input_files_unchanged": True,
            "targets_identical_to_frozen_dev64": True, "no_truncation": True},
        "run": {"created_at_utc": datetime.now(timezone.utc).isoformat(),
            "torch_version": str(torch.__version__), "device": str(device),
            "threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "runner_sha256": sha(__file__), "note": "Original frozen reader weights only; no writer checkpoint was opened."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--reference", required=True, help="Frozen complete 100M DEV64 JSON; not a checkpoint file")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight", type=Path, help="Previously saved CPU registration JSON")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--threads", default=8, type=int)
    args = parser.parse_args()
    require(args.threads > 0, "Threads must be positive")
    require(not (args.preflight_only and args.sanity_only), "Choose one execution scope")
    require(not args.output.exists(), "Outputs are immutable; choose a new path")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    _, prepared, annotations, reference, old_empty, plans, manifest, preflight = prepare(args)
    if args.preflight_only:
        result = preflight
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = evaluate(args, prepared, annotations, reference, old_empty, plans, manifest, preflight)
    write_new(args.output, result)
    print(json.dumps({"output": str(args.output), "stage": result["stage"],
        "registration_sha256": preflight["registration_sha256"],
        "maximum_context_tokens": preflight["registration"]["maximum_context_tokens"],
        "complete": result.get("complete", False)}), flush=True)


if __name__ == "__main__":
    main()

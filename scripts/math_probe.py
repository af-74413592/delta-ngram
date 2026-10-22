"""Prepare, generate, and score fixed MATH-64 probes / MATH-500 endpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# The package __init__ exports Torch-backed adapters. CPU-only scoring must not
# execute it; generation imports model_loading lazily inside generate().
sys.path.insert(0, str(ROOT / "src" / "cl_research"))
import math_retention as math_eval
from humaneval_probe import file_hash, generation_statistics, merge_boundary_checkpoint, write_json

STOP_STRINGS = ("\nProblem:", "\nQuestion:", "\n\nUser:")


def fresh_directory(directory):
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"use a fresh output directory: {directory}")


def prepare(directory):
    fresh_directory(directory)
    with urllib.request.urlopen(math_eval.DATA_URL, timeout=60) as response:
        payload = response.read(8 * 1024 * 1024 + 1)
    if len(payload) > 8 * 1024 * 1024:
        raise ValueError("unexpected MATH-500 download size")
    rows = math_eval.validate_records([json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()])
    probe_ids, allocation = math_eval.select_probe(rows)
    prompts = [{"task_id": row["unique_id"], "problem": row["problem"],
                "subject": row["subject"], "level": row["level"]} for row in rows]
    answers = [{"task_id": row["unique_id"], "answer": row["answer"],
                "subject": row["subject"], "level": row["level"]} for row in rows]
    write_json(directory / "prompts.json", prompts)
    write_json(directory / "answers.json", answers)
    manifest = {"benchmark": "MATH-500", "source_revision": math_eval.REVISION,
                "source_url": math_eval.DATA_URL, "source_sha256": math_eval.digest(payload),
                "task_count": len(rows), "task_ids": [row["task_id"] for row in prompts],
                "purpose": "evaluation_only_never_train_or_select_hyperparameters",
                "probe_ids": probe_ids, "probe_seed": math_eval.SEED,
                "probe_selection": "Hamilton_proportional_subject_x_level_then_sha256_math-probe-v1/seed/id",
                "probe_allocation": allocation, "prepared_utc": datetime.now(timezone.utc).isoformat(),
                "files": {name: file_hash(directory / name) for name in ("prompts.json", "answers.json")}}
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_locked_data(directory, filename="prompts.json"):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_revision"] != math_eval.REVISION or manifest["task_count"] != 500:
        raise ValueError("unexpected MATH dataset revision or count")
    if filename not in ("prompts.json", "answers.json"):
        raise ValueError("unexpected data file")
    if file_hash(directory / filename) != manifest["files"][filename]:
        raise ValueError("MATH data differs from preparation manifest")
    rows = json.loads((directory / filename).read_text(encoding="utf-8"))
    if len(rows) != 500 or len(set(manifest["task_ids"])) != 500:
        raise ValueError("MATH full set must contain 500 unique IDs")
    rows = math_eval.project_rows(rows, manifest["task_ids"])
    selection, allocation = math_eval.select_probe([dict(row, unique_id=row["task_id"]) for row in rows])
    if selection != manifest["probe_ids"] or allocation != manifest["probe_allocation"]:
        raise ValueError("MATH probe selection differs from deterministic metadata allocation")
    return rows, manifest


def scope_ids(scope, manifest):
    if scope == "probe":
        return manifest["probe_ids"]
    if scope == "full":
        return manifest["task_ids"]
    raise ValueError("scope must be probe or full")


def decode_protocol(manifest, backend):
    return {"version": 1, "benchmark": "MATH-500", "source_revision": math_eval.REVISION,
            "prompts_sha256": manifest["files"]["prompts.json"], "prompt_template": math_eval.PROMPT_TEMPLATE,
            "prompt_format": "raw_text_no_chat_template_no_reference_solution",
            "max_new_tokens": math_eval.MAX_NEW_TOKENS, "samples_per_task": 1,
            "do_sample": False, "num_beams": 1, "seed": math_eval.SEED, "batch_size": 1,
            "stop_strings": list(STOP_STRINGS), "dtype": "torch.bfloat16", "attn_implementation": backend,
            "checkpoint_state": "all_adapters_merged_in_history_order"}


def trim_completion(text):
    matches = [(text.find(stop), stop) for stop in STOP_STRINGS if stop in text]
    if not matches:
        return text, None
    position, stop = min(matches)
    return text[:position], stop


def sha256_hex(value):
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise argparse.ArgumentTypeError("expected a 64-character SHA-256 hex digest")
    return value.lower()


def load_reused_samples(directory, expected_generator_sha256, generation, prompts):
    """Validate a frozen source trace, ignoring only its unterminated last line."""
    summary_bytes = (directory / "summary.json").read_bytes()
    previous = json.loads(summary_bytes)
    if previous.get("status") not in ("generating", "failed", "completed"):
        raise ValueError("reuse source must be a generation run")
    for key in ("data_manifest", "model_identity", "source_checkpoint", "source_checkpoint_sha256",
                "protocol", "protocol_sha256", "math_core_sha256"):
        if key not in previous or previous[key] != generation[key]:
            raise ValueError(f"reuse source differs in {key}")
    if previous.get("generator_sha256") != expected_generator_sha256:
        raise ValueError("reuse source differs in expected generator_sha256")
    previous_ids = scope_ids(previous.get("scope"), generation["data_manifest"])
    if previous.get("task_ids") != previous_ids or previous.get("task_count") != len(previous_ids):
        raise ValueError("reuse source scope or task count is inconsistent")
    trace_bytes = (directory / "trace.jsonl").read_bytes()
    complete_length = trace_bytes.rfind(b"\n") + 1
    rows_by_id = {}
    prompt_by_id = {row["task_id"]: row for row in prompts}
    for line_number, line in enumerate(trace_bytes[:complete_length].splitlines(), 1):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid reuse trace line {line_number}") from exc
        if not isinstance(row, dict) or row.get("task_id") not in previous_ids:
            raise ValueError(f"unexpected task ID in reuse trace line {line_number}")
        task_id = row["task_id"]
        if task_id in rows_by_id:
            raise ValueError(f"duplicate task ID in reuse trace: {task_id}")
        problem = prompt_by_id[task_id]
        prompt = math_eval.PROMPT_TEMPLATE.format(problem=problem["problem"])
        if row.get("prompt_sha256") != math_eval.digest(prompt.encode()):
            raise ValueError(f"reuse prompt hash differs for {task_id}")
        if row.get("subject") != problem["subject"] or row.get("level") != problem["level"]:
            raise ValueError(f"reuse metadata differs for {task_id}")
        for key in ("completion", "raw_completion"):
            if not isinstance(row.get(key), str):
                raise ValueError(f"invalid reuse {key} for {task_id}")
        for key in ("generated_tokens", "prompt_tokens"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f"invalid reuse {key} for {task_id}")
        for key in ("max_token_hit", "truncated"):
            if type(row.get(key)) is not bool:
                raise ValueError(f"invalid reuse {key} for {task_id}")
        elapsed = row.get("elapsed_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError(f"invalid reuse elapsed_seconds for {task_id}")
        if row.get("correct") is not None or row.get("scoring_status") != "pending_scoring":
            raise ValueError(f"reuse trace is not unscored generation for {task_id}")
        rows_by_id[task_id] = row
    reused = [rows_by_id[task_id] for task_id in generation["task_ids"] if task_id in rows_by_id]
    provenance = {"reuse_source": str(directory), "reuse_source_status": previous["status"],
                  "reuse_source_summary_sha256": math_eval.digest(summary_bytes),
                  "reuse_source_trace_sha256": math_eval.digest(trace_bytes),
                  "reuse_source_generator_sha256": expected_generator_sha256,
                  "reuse_source_complete_records": len(rows_by_id),
                  "reuse_source_ignored_trailing_bytes": len(trace_bytes) - complete_length}
    return reused, provenance


def generate_missing(args, prompts, summary):
    """Load the model only when the requested scope still has missing tasks."""
    import torch
    from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    from cl_research.model_loading import load_text_model

    random.seed(math_eval.SEED)
    torch.manual_seed(math_eval.SEED)
    torch.cuda.manual_seed_all(math_eval.SEED)
    model = load_text_model(args.model_path, device="cuda", dtype=torch.bfloat16,
                            attn_implementation=args.attn_implementation)
    history = merge_boundary_checkpoint(model, args.checkpoint, args.model_path)
    summary.update(merge_history=history, boundary_index=len(history), model_loading=model.cl_loading_info)
    model.requires_grad_(False).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    with torch.inference_mode():
        for problem in prompts:
            prompt = math_eval.PROMPT_TEMPLATE.format(problem=problem["problem"])
            batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            batch = {key: value.to("cuda") for key, value in batch.items()}
            prompt_length = batch["input_ids"].shape[1]

            class StopAtBoundary(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    continuation = tokenizer.decode(input_ids[0, prompt_length:].tolist(), skip_special_tokens=True)
                    return any(stop in continuation for stop in STOP_STRINGS)

            started = time.perf_counter()
            tokens = model.generate(**batch, do_sample=False, num_beams=1,
                                    max_new_tokens=math_eval.MAX_NEW_TOKENS, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id,
                                    stopping_criteria=StoppingCriteriaList([StopAtBoundary()]))
            generated_ids = tokens[0, prompt_length:].tolist()
            raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
            completion, stop = trim_completion(raw)
            eos = model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
            hit = len(generated_ids) >= math_eval.MAX_NEW_TOKENS
            truncated = hit and stop is None and (not generated_ids or generated_ids[-1] not in eos_ids)
            yield {"task_id": problem["task_id"], "subject": problem["subject"], "level": problem["level"],
                   "completion": completion, "raw_completion": raw, "stop_string": stop,
                   "prompt_sha256": math_eval.digest(prompt.encode()), "generated_tokens": len(generated_ids),
                   "prompt_tokens": prompt_length, "max_token_hit": hit, "truncated": truncated,
                   "elapsed_seconds": time.perf_counter() - started, "correct": None,
                   "scoring_status": "pending_scoring"}


def generate(args):
    started_run = time.perf_counter()
    prompts, manifest = load_locked_data(args.data_dir)
    selected_ids = scope_ids(args.scope, manifest)
    source, checkpoint_hash = None, None
    if args.checkpoint:
        import torch

        source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if args.checkpoint.name != "checkpoint.pt" or source["step"] != source["config"]["steps"]:
            raise ValueError("math retention uses final checkpoint.pt only")
        declared_model = source["config"]["model_path"]
        if args.model_path and args.model_path != declared_model:
            raise ValueError("checkpoint and requested model differ")
        args.model_path = declared_model
        checkpoint_hash = file_hash(args.checkpoint)
    if not args.model_path:
        raise ValueError("base generation requires --model-path")
    identity = json.loads((Path(args.model_path) / "verified_weights.json").read_text(encoding="utf-8"))
    if source and source["config"].get("verified_model_identity") != identity:
        raise ValueError("checkpoint model identity mismatch")
    protocol = decode_protocol(manifest, args.attn_implementation)
    summary = {"status": "generating", "scoring_status": "pending_scoring", "accuracy": None,
               "protocol": protocol, "protocol_sha256": math_eval.json_digest(protocol),
               "scope": args.scope, "task_ids": selected_ids, "task_count": len(selected_ids),
               "model_identity": identity, "source_checkpoint": str(args.checkpoint) if args.checkpoint else None,
               "source_checkpoint_sha256": checkpoint_hash, "source_config": source["config"] if source else None,
               "generated_tasks": 0, "started_utc": datetime.now(timezone.utc).isoformat(),
               "selection_uses_math_metrics": False, "data_manifest": manifest,
               "generator_sha256": file_hash(Path(__file__)),
               "math_core_sha256": file_hash(Path(math_eval.__file__)),
               "package_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")}}
    reuse_from = getattr(args, "reuse_from", None)
    reuse_hash = getattr(args, "reuse_generator_sha256", None)
    if bool(reuse_from) != bool(reuse_hash):
        raise ValueError("--reuse-from and --reuse-generator-sha256 must be supplied together")
    reused, provenance = [], {"reuse_source": None}
    if reuse_from:
        reused, provenance = load_reused_samples(reuse_from, sha256_hex(reuse_hash), summary, prompts)
    reused_ids = [row["task_id"] for row in reused]
    missing = [row for row in math_eval.project_rows(prompts, selected_ids) if row["task_id"] not in reused_ids]
    samples, newly_generated = list(reused), []
    summary.update(provenance, reused_task_ids=reused_ids, reused_tasks=len(reused), newly_generated_tasks=0,
                   generated_tasks=len(samples), model_loading_skipped=not missing,
                   timing_basis="per_task_generation_seconds_including_reused_source_compute",
                   elapsed_seconds_basis="current_invocation_wall_seconds",
                   reused_generation_timing=generation_statistics(reused),
                   new_generation_timing=generation_statistics(newly_generated), timing=generation_statistics(samples))
    fresh_directory(args.output)
    write_json(args.output / "summary.json", summary)
    try:
        with (args.output / "trace.jsonl").open("w", encoding="utf-8") as trace:
            for row in reused:
                trace.write(json.dumps(row, ensure_ascii=False) + "\n")
            trace.flush()
            for row in generate_missing(args, missing, summary) if missing else ():
                samples.append(row)
                newly_generated.append(row)
                trace.write(json.dumps(row, ensure_ascii=False) + "\n")
                trace.flush()
                summary.update(generated_tasks=len(samples), newly_generated_tasks=len(newly_generated),
                               timing=generation_statistics(samples),
                               new_generation_timing=generation_statistics(newly_generated))
                write_json(args.output / "summary.json", summary)
                print(json.dumps({"task_id": row["task_id"], "generated_tasks": len(samples)}), flush=True)
        if len(samples) != len(selected_ids):
            raise ValueError("generation did not produce the exact requested task count")
        samples = math_eval.project_rows(samples, selected_ids)
        final_trace = args.output / "trace.jsonl.tmp"
        final_trace.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples), encoding="utf-8")
        final_trace.replace(args.output / "trace.jsonl")
        write_json(args.output / "math_completions.json", samples)
        summary.update(status="completed", completions_sha256=file_hash(args.output / "math_completions.json"),
                       trace_sha256=file_hash(args.output / "trace.jsonl"))
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.perf_counter() - started_run, timing=generation_statistics(samples),
                       new_generation_timing=generation_statistics(newly_generated))
        write_json(args.output / "summary.json", summary)
    return summary


def _load_reference(directory, generation, score_protocol):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") not in ("completed", "completed_with_unresolved"):
        raise ValueError("reference scoring is incomplete")
    for key in ("protocol_sha256", "model_identity"):
        if summary[key] != generation[key]:
            raise ValueError(f"reference differs in {key}")
    if summary["scoring_protocol"] != score_protocol:
        raise ValueError("reference scoring protocol differs")
    if summary.get("source_checkpoint") is not None:
        raise ValueError("retention reference must be an unadapted base model")
    if file_hash(directory / "results.json") != summary["results_sha256"]:
        raise ValueError("reference results differ from their checksum")
    return json.loads((directory / "results.json").read_text(encoding="utf-8"))


def score(args):
    actual = importlib.metadata.version("math-verify")
    if actual != math_eval.MATH_VERIFY_VERSION:
        raise RuntimeError(f"math-verify=={math_eval.MATH_VERIFY_VERSION} required, found {actual}")
    answers, manifest = load_locked_data(args.data_dir, "answers.json")
    generation = json.loads((args.samples_dir / "summary.json").read_text(encoding="utf-8"))
    if generation["status"] != "completed":
        raise ValueError("generation must complete before scoring")
    for key in ("source_revision", "files", "probe_ids"):
        if generation["data_manifest"][key] != manifest[key]:
            raise ValueError("generation and scorer data manifests differ")
    if file_hash(args.samples_dir / "math_completions.json") != generation["completions_sha256"]:
        raise ValueError("generated samples differ from their checksum")
    samples = json.loads((args.samples_dir / "math_completions.json").read_text(encoding="utf-8"))
    generated_ids = scope_ids(generation["scope"], manifest)
    if len(samples) != len(generated_ids) or generation["task_ids"] != generated_ids:
        raise ValueError("generation scope or task count is inconsistent")
    math_eval.project_rows(samples, generated_ids)
    scope = args.scope or generation["scope"]
    selected_ids = scope_ids(scope, manifest)
    samples = math_eval.project_rows(samples, selected_ids)
    references = {row["task_id"]: row for row in answers}
    score_protocol = math_eval.scoring_protocol()
    reference = _load_reference(args.reference, generation, score_protocol) if args.reference else None
    if reference is not None:
        math_eval.project_rows(reference, selected_ids)
    fresh_directory(args.output)
    summary = {"status": "scoring", "accuracy": None, "scope": scope, "task_ids": selected_ids,
               "task_count": len(selected_ids), "scored_tasks": 0, "data_manifest": manifest,
               "protocol": generation["protocol"], "protocol_sha256": generation["protocol_sha256"],
               "scoring_protocol": score_protocol, "scoring_protocol_sha256": math_eval.json_digest(score_protocol),
               "model_identity": generation["model_identity"], "source_checkpoint": generation["source_checkpoint"],
               "source_checkpoint_sha256": generation.get("source_checkpoint_sha256"),
               "source_generation": str(args.samples_dir), "source_completions_sha256": generation["completions_sha256"],
               "reference": str(args.reference) if args.reference else None,
               "core_sha256": file_hash(Path(math_eval.__file__)),
               "package_versions": {name: importlib.metadata.version(name) for name in
                                    ("math-verify", "latex2sympy2-extended", "sympy", "antlr4-python3-runtime", "mpmath")},
               "started_utc": datetime.now(timezone.utc).isoformat()}
    results, started_run = [], time.perf_counter()
    write_json(args.output / "summary.json", summary)
    scorer = math_eval.MathScorer(timeout_seconds=score_protocol["deadline_seconds"])
    try:
        with (args.output / "trace.jsonl").open("w", encoding="utf-8") as trace:
            for sample in samples:
                result = dict(sample)
                result.update(scorer.score(references[sample["task_id"]]["answer"], sample["completion"]))
                result["scoring_status"] = result["status"]
                results.append(result)
                trace.write(json.dumps(result, ensure_ascii=False) + "\n")
                trace.flush()
                summary["scored_tasks"] = len(results)
                write_json(args.output / "summary.json", summary)
        metrics = math_eval.result_metrics(results, selected_ids)
        summary.update(metrics=metrics, accuracy=metrics["accuracy"],
                       status="completed_with_unresolved" if metrics["unresolved_count"] else "completed")
        if scope == "full":
            summary["probe_metrics"] = math_eval.result_metrics(results, manifest["probe_ids"])
        if reference is not None:
            summary["transitions"] = math_eval.transition_counts(reference, results, selected_ids)
            if scope == "full":
                summary["probe_transitions"] = math_eval.transition_counts(reference, results, manifest["probe_ids"])
        write_json(args.output / "results.json", results)
        summary["results_sha256"] = file_hash(args.output / "results.json")
    except Exception as exc:
        summary.update(status="failed", accuracy=None, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        scorer.close()
        summary.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.perf_counter() - started_run)
        write_json(args.output / "summary.json", summary)
    return summary


def audit_references(args):
    """Check parser coverage of fixed gold answers; this is not a model score."""
    answers, manifest = load_locked_data(args.data_dir, "answers.json")
    fresh_directory(args.output)
    results = []
    scorer = math_eval.MathScorer()
    started = time.perf_counter()
    try:
        for row in answers:
            result = scorer.score(row["answer"], r"\boxed{" + row["answer"] + "}")
            results.append({"task_id": row["task_id"], **result})
    finally:
        scorer.close()
    failed = [row for row in results if row.get("correct") is not True]
    summary = {"purpose": "reference_parser_preflight_not_model_evaluation", "n": len(results),
               "status": "passed" if not failed else "failed", "failed_count": len(failed),
               "failures": failed, "scoring_protocol": math_eval.scoring_protocol(),
               "data_manifest": manifest, "elapsed_seconds": time.perf_counter() - started}
    write_json(args.output / "results.json", results)
    write_json(args.output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prep = actions.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    gen = actions.add_parser("generate")
    gen.add_argument("--data-dir", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--model-path")
    gen.add_argument("--checkpoint", type=Path)
    gen.add_argument("--reuse-from", type=Path, help="reuse validated complete records from this generation directory")
    gen.add_argument("--reuse-generator-sha256", type=sha256_hex, help="explicit SHA-256 of the source generator script")
    gen.add_argument("--scope", choices=("probe", "full"), default="probe")
    gen.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="flash_attention_2")
    scoring = actions.add_parser("score")
    scoring.add_argument("--data-dir", type=Path, required=True)
    scoring.add_argument("--samples-dir", type=Path, required=True)
    scoring.add_argument("--output", type=Path, required=True)
    scoring.add_argument("--scope", choices=("probe", "full"))
    scoring.add_argument("--reference", type=Path)
    audit = actions.add_parser("audit-references")
    audit.add_argument("--data-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.output), indent=2))
    elif args.action == "generate":
        generate(args)
    elif args.action == "audit-references":
        result = audit_references(args)
        print(json.dumps({key: result[key] for key in ("purpose", "n", "status", "failed_count", "failures", "elapsed_seconds")}, indent=2))
        if result["status"] != "passed":
            raise SystemExit(1)
    else:
        print(json.dumps(score(args), indent=2))


if __name__ == "__main__":
    main()

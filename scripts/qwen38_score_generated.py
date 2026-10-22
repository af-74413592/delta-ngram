"""Score Qwen generation traces on the already locked 128/64/7x64 queries.

No model loading, training, sample selection, or candidate Python execution on
the host. MATH reuses MathScorer; HumanEval runs in the validated WASI worker.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "cl_research"))
import math_retention
import niche_metrics
from humaneval_probe import file_hash


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_digest(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def locked_population(args):
    if args.suite in {"math", "code"}:
        directory = args.retention_data
        manifest = read_json(directory / "manifest.json")
        if manifest.get("protocol") != "retention_reference_nll_v2_128_64":
            raise ValueError("requires the original Qwen math128/code64 reference manifest")
        source = directory / "records.jsonl"
        if file_hash(source) != manifest["files"]["records.jsonl"]:
            raise ValueError("locked math/code records changed")
        rows = [row for row in read_jsonl(source) if row["domain"] == args.suite]
        expected_ids = manifest["task_ids"][args.suite]
        expected_count = 128 if args.suite == "math" else 64
        if len(rows) != expected_count or [row["task_id"] for row in rows] != expected_ids:
            raise ValueError("math/code population/order differs from the locked manifest")
    else:
        directory = args.task_data
        manifest = read_json(directory / "manifest.json")
        if manifest.get("protocol") != "qwen38_niche_probes64_v1":
            raise ValueError("requires the original Qwen 7x64 task manifest")
        source = directory / "task_probes64.jsonl"
        if file_hash(source) != manifest["files"]["task_probes64.jsonl"]:
            raise ValueError("locked task probes changed")
        selected_tasks = args.tasks or manifest["task_order"]
        if len(set(selected_tasks)) != len(selected_tasks) or not set(selected_tasks).issubset(manifest["task_order"]):
            raise ValueError("unknown/duplicate task selection")
        rows = [row for row in read_jsonl(source) if row["task"] in selected_tasks]
        expected_ids = [item for task in manifest["task_order"] if task in selected_tasks
                        for item in manifest["sources"][task]["ids"]]
        if [row["task_id"] for row in rows] != expected_ids or any(
                sum(row["task"] == task for row in rows) != 64 for task in selected_tasks):
            raise ValueError("task probes differ from the complete locked 64-row population")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("duplicate locked query IDs")
    for row in rows:
        if row.get("prompt_sha256", digest(row["prompt"])) != digest(row["prompt"]):
            raise ValueError("locked prompt hash mismatch")
    return rows, {"manifest": str(directory / "manifest.json"), "manifest_sha256": file_hash(directory / "manifest.json"),
                  "records": str(source), "records_sha256": file_hash(source), "task_ids": expected_ids}


def load_samples(args, locked):
    directory = args.samples_dir
    generation = read_json(directory / "summary.json")
    if generation.get("status") != "completed":
        raise ValueError("generation is not completed; no partial run is reported as accuracy")
    if generation.get("suite", generation.get("domain")) != args.suite:
        raise ValueError("generation suite differs from requested scorer")
    protocol = generation.get("protocol") or {}
    if protocol.get("do_sample") is not False or protocol.get("num_beams") != 1:
        raise ValueError("requires the locked greedy decoding protocol")
    if args.suite in {"math", "code"}:
        expected_tokens = 1024 if args.suite == "math" else 256
        if protocol.get("max_new_tokens") != expected_tokens:
            raise ValueError("math/code generation token budget differs from the locked protocol")
    names = ("generations.jsonl", "trace.jsonl", "predictions.jsonl", "results.jsonl",
             "math_completions.json", "humaneval_completions.json", "generations.json")
    candidates = [directory / name for name in names if (directory / name).is_file()]
    if not candidates:
        raise ValueError("missing generation trace/completions file")
    source = candidates[0]
    checksum = file_hash(source)
    declared = generation.get("trace_sha256") if source.suffix == ".jsonl" else generation.get("completions_sha256")
    if not declared or declared != checksum:
        raise ValueError("generation trace checksum differs from summary")
    samples = read_jsonl(source) if source.suffix == ".jsonl" else read_json(source)
    expected = {row["task_id"]: row for row in locked}
    ids = [row.get("task_id") for row in samples]
    # A task-stage generation may deliberately contain the complete locked
    # 7x64 population so one HF load can be reused for multiple triangle
    # slices.  When --tasks selects a slice, validate that every requested
    # ID is present and score only that slice; retain the exact-population
    # check for math/code and unselected task runs.
    full_task_generation = args.suite == "tasks" and bool(args.tasks)
    if full_task_generation:
        if (len(set(ids)) != len(ids) or len(ids) < len(expected)
                or not set(expected).issubset(ids)):
            raise ValueError(f"generation is missing selected locked queries ({len(expected)} required; got {len(ids)})")
    elif len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError(f"generation must contain the exact locked population ({len(expected)} queries); got {len(ids)}")
    by_id = {}
    for original in samples:
        row = dict(original)
        task_id = row["task_id"]
        if task_id not in expected:
            if full_task_generation:
                continue
            raise ValueError(f"unexpected generated task ID: {task_id}")
        if not isinstance(row.get("completion"), str):
            if isinstance(row.get("prediction"), str):
                row["completion"] = row["prediction"]
            else:
                raise ValueError(f"missing completion: {task_id}")
        prompt_hash = row.get("prompt_sha256")
        if prompt_hash is None and isinstance(row.get("prompt"), str):
            prompt_hash = digest(row["prompt"])
        if prompt_hash != digest(expected[task_id]["prompt"]):
            raise ValueError(f"generated prompt differs from locked source: {task_id}")
        by_id[task_id] = row
    return [by_id[row["task_id"]] for row in locked], generation, source, checksum


def score_math(samples, args):
    from math_probe import load_locked_data
    if importlib.metadata.version("math-verify") != math_retention.MATH_VERIFY_VERSION:
        raise RuntimeError(f"math-verify=={math_retention.MATH_VERIFY_VERSION} required")
    answers, _ = load_locked_data(args.math_data, "answers.json")
    gold = {row["task_id"]: row["answer"] for row in answers}
    scorer = math_retention.MathScorer(timeout_seconds=10)
    try:
        for row in samples:
            yield {**row, **scorer.score(gold[row["task_id"]], row["completion"])}
    finally:
        scorer.close()


def score_code(samples, args):
    import evaluate_humaneval_wasi as wasi
    from humaneval_probe import load_locked_data
    if args.runtime is None or args.validation is None:
        raise ValueError("code requires --runtime and --validation (the existing validated WASI runtime)")
    problems, manifest = load_locked_data(args.code_data, "problems.json")
    wasi.verify_validation(args.validation, manifest)
    gold = {row["task_id"]: row for row in problems}
    with wasi.Worker(args.runtime) as worker:
        for row in samples:
            result = worker.check(row["task_id"], wasi.build_program(gold[row["task_id"]], row["completion"]))
            if result.get("status") != "completed" or type(result.get("passed")) is not bool:
                raise RuntimeError(f"WASI infrastructure failure: {row['task_id']}")
            yield {**row, **result, "correct": result["passed"]}


def score_tasks(samples, locked):
    gold = {row["task_id"]: row for row in locked}
    for row in samples:
        reference = gold[row["task_id"]]
        result = niche_metrics.score_prediction(row["completion"], {**reference, "metric": "inflection_exact"})
        yield {**row, "task": reference["task"], "target": reference["target"],
               "references": reference.get("references", [reference["target"]]), **result, "status": "scored"}


def metrics(rows, suite):
    ids = [row["task_id"] for row in rows]
    if suite == "math":
        return math_retention.result_metrics(rows, ids)
    correct = sum(row["correct"] is True for row in rows)
    output = {"n": len(rows), "correct_count": correct, "accuracy": correct / len(rows),
              "unresolved_count": 0}
    if suite == "code":
        from evaluate_humaneval_isolated import wilson_interval
        output.update(passed=correct, pass_at_1=correct / len(rows), wilson_95_interval=wilson_interval(correct, len(rows)))
    elif suite == "tasks":
        output["per_task"] = {}
        for task in dict.fromkeys(row["task"] for row in rows):
            selected = [row for row in rows if row["task"] == task]
            count = sum(row["correct"] is True for row in selected)
            output["per_task"][task] = {"n": len(selected), "correct_count": count, "accuracy": count / len(selected),
                                        "mean_edit_distance": sum(row["edit_distance"] for row in selected) / len(selected)}
    return output


def run(args):
    locked, population = locked_population(args)
    samples, generation, source, source_hash = load_samples(args, locked)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("fresh scoring output directory required")
    protocol = {"name": "qwen38_generated_accuracy_fixed_v1", "suite": args.suite,
                "population": population, "generation_protocol": generation.get("protocol"),
                "scorer": (math_retention.scoring_protocol() if args.suite == "math" else
                           "humaneval-wasi-v1" if args.suite == "code" else "niche_inflection_NFC_exact_references_v1")}
    summary = {"status": "scoring", "suite": args.suite, "protocol": protocol,
               "protocol_sha256": json_digest(protocol), "expected_count": len(locked), "scored_count": 0,
               "source_generation": str(args.samples_dir), "source_trace": str(source), "source_trace_sha256": source_hash,
               "source_generation_summary_sha256": file_hash(args.samples_dir / "summary.json"),
               "source_checkpoint": generation.get("source_checkpoint", generation.get("checkpoint")),
               "method": generation.get("method", generation.get("arm")), "stage": generation.get("stage", generation.get("round")),
               "scorer_sha256": file_hash(Path(__file__)), "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output / "summary.json", summary)
    results = []
    started = time.perf_counter()
    try:
        iterator = score_math(samples, args) if args.suite == "math" else score_code(samples, args) if args.suite == "code" else score_tasks(samples, locked)
        with (args.output / "trace.jsonl").open("w", encoding="utf-8") as trace:
            for result in iterator:
                results.append(result)
                trace.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                trace.flush()
                summary["scored_count"] = len(results)
                write_json(args.output / "summary.json", summary)
                print(json.dumps({"task_id": result["task_id"], "correct": result["correct"], "scored": len(results)}, ensure_ascii=False), flush=True)
        summary["metrics"] = metrics(results, args.suite)
        summary["status"] = "completed_with_unresolved" if summary["metrics"]["unresolved_count"] else "completed"
        write_json(args.output / "results.json", results)
        summary["results_sha256"] = file_hash(args.output / "results.json")
        if args.reference:
            ref_summary = read_json(args.reference / "summary.json")
            if ref_summary.get("status") not in {"completed", "completed_with_unresolved"}:
                raise ValueError("reference scoring is incomplete")
            if ref_summary.get("protocol_sha256") != summary["protocol_sha256"]:
                raise ValueError("reference population or generation/scoring protocol differs")
            if file_hash(args.reference / "results.json") != ref_summary.get("results_sha256"):
                raise ValueError("reference results checksum differs")
            reference = read_json(args.reference / "results.json")
            summary["transitions"] = math_retention.transition_counts(reference, results, population["task_ids"])
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("math", "code", "tasks"), required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retention-data", type=Path, default=ROOT / "data" / "retention_nll_v2_128_64")
    parser.add_argument("--task-data", type=Path, default=ROOT / "data" / "qwen38_eval_v1")
    parser.add_argument("--math-data", type=Path, default=ROOT / "data" / "retention_v3" / "math500")
    parser.add_argument("--code-data", type=Path, default=ROOT / "data" / "retention_v3" / "humaneval")
    parser.add_argument("--tasks", nargs="+", help="Optional task population, always all locked 64 rows per selected task")
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.tasks and args.suite != "tasks":
        parser.error("--tasks requires --suite tasks")
    print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

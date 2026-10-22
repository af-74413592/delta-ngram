"""Fixed HumanEval-32 probes and full-164 endpoints; never execute generated code."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
import urllib.request

REVISION = "6d43fb980f9fee3c892a914eda09951f772ad10d"
DATA_URL = f"https://raw.githubusercontent.com/openai/human-eval/{REVISION}/data/HumanEval.jsonl.gz"
TASK_IDS = tuple(f"HumanEval/{i}" for i in range(164))
PROBE_IDS = tuple(sorted(TASK_IDS, key=lambda task: hashlib.sha256(f"humaneval-probe-v2/17/{task}".encode()).digest())[:32])
MAX_NEW_TOKENS = 256
STOP_STRINGS = ("\nclass", "\ndef", "\n#", "\n@", "\nif", "\nprint")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def scope_ids(scope):
    if scope not in {"probe", "full"}:
        raise ValueError("scope must be probe or full")
    return PROBE_IDS if scope == "probe" else TASK_IDS


def validate_tasks(rows, expected_ids=TASK_IDS):
    expected_ids = tuple(expected_ids)
    if not expected_ids or len(set(expected_ids)) != len(expected_ids) or not set(expected_ids).issubset(TASK_IDS):
        raise ValueError("expected task IDs must be a nonempty, unique HumanEval subset")
    ids = [row["task_id"] for row in rows]
    if len(ids) != len(expected_ids) or set(ids) != set(expected_ids):
        raise ValueError("HumanEval requires exactly one record for every declared task ID")
    mapping = {row["task_id"]: row for row in rows}
    return [mapping[task] for task in expected_ids]


def project_tasks(rows, expected_ids):
    ids = [row['task_id'] for row in rows]
    if len(set(ids)) != len(ids) or not set(expected_ids).issubset(ids):
        raise ValueError('records are duplicated or missing a requested projection task')
    return validate_tasks([row for row in rows if row['task_id'] in set(expected_ids)], expected_ids)


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError("use a fresh HumanEval evaluation-only directory")
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        payload = response.read(2 * 1024 * 1024 + 1)
    if len(payload) > 2 * 1024 * 1024:
        raise ValueError("unexpected dataset download size")
    rows = validate_tasks([json.loads(line) for line in gzip.decompress(payload).decode("utf-8").splitlines() if line.strip()])
    # Canonical solutions are unnecessary. Tests are kept separate from prompts.
    problems = [{key: row[key] for key in ("task_id", "prompt", "test", "entry_point")} for row in rows]
    prompts = [{key: row[key] for key in ("task_id", "prompt")} for row in rows]
    write_json(directory / "prompts.json", prompts)
    write_json(directory / "problems.json", problems)
    manifest = {"benchmark": "HumanEval", "task_count": 164, "source_url": DATA_URL,
                "source_revision": REVISION, "source_gzip_sha256": sha256_bytes(payload),
                "purpose": "evaluation_only_never_train_or_select_hyperparameters",
                "files": {name: file_hash(directory / name) for name in ("prompts.json", "problems.json")}}
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_locked_data(directory, filename="prompts.json"):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_revision"] != REVISION or manifest["task_count"] != 164:
        raise ValueError("unexpected HumanEval revision or task count")
    if file_hash(directory / filename) != manifest["files"][filename]:
        raise ValueError("HumanEval file differs from its preparation manifest")
    return validate_tasks(json.loads((directory / filename).read_text(encoding="utf-8"))), manifest


def trim_completion(text):
    matches = [(text.find(stop), stop) for stop in STOP_STRINGS if stop in text]
    if not matches:
        return text, None
    position, stop = min(matches)
    return text[:position], stop


def decode_protocol(manifest, backend):
    # Scope lives outside this shared decoding protocol so full->probe projection is comparable.
    return {"version": 2, "benchmark": "HumanEval", "source_revision": REVISION,
            "prompts_sha256": manifest["files"]["prompts.json"],
            "prompt_format": "original_raw_function_prefix_no_chat_template", "samples_per_task": 1,
            "do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS, "seed": 17,
            "batch_size": 1, "stop_strings": list(STOP_STRINGS), "dtype": "torch.bfloat16",
            "attn_implementation": backend, "checkpoint_state": "all_adapters_merged_in_history_order"}


def generation_statistics(samples):
    count = len(samples)
    elapsed = sum(row['elapsed_seconds'] for row in samples)
    hits = sum(bool(row['max_token_hit']) for row in samples)
    truncated = sum(bool(row['truncated']) for row in samples)
    return {'measured_tasks': count, 'generation_elapsed_seconds': elapsed,
            'seconds_per_task': elapsed/count if count else None, 'max_token_hit_count': hits,
            'max_token_hit_rate': hits/count if count else None, 'truncation_count': truncated,
            'truncation_rate': truncated/count if count else None}


def merge_boundary_checkpoint(model, checkpoint, model_path):
    if checkpoint is None:
        return []
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cl_research.text_training import merge_prior_checkpoint
    return merge_prior_checkpoint(model, checkpoint, expected_model_path=model_path)


def generate(args):
    import torch
    from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from cl_research.model_loading import load_text_model

    prompts, manifest = load_locked_data(args.data_dir)
    scope = getattr(args, 'scope', 'probe')
    selected_ids = scope_ids(scope)
    prompts = project_tasks(prompts, selected_ids)
    source, checkpoint_hash = None, None
    if args.checkpoint:
        source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if args.checkpoint.name != "checkpoint.pt" or source["step"] != source["config"]["steps"]:
            raise ValueError("coding retention uses final checkpoint.pt only")
        declared_model = source["config"]["model_path"]
        if args.model_path and args.model_path != declared_model:
            raise ValueError("checkpoint and requested model differ")
        args.model_path = declared_model
        checkpoint_hash = file_hash(args.checkpoint)
    if not args.model_path:
        raise ValueError("base generation requires --model-path")
    identity_path = Path(args.model_path) / "verified_weights.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if source and source["config"].get("verified_model_identity") != identity:
        raise ValueError("checkpoint model identity mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("use a fresh coding output directory")
    protocol = decode_protocol(manifest, args.attn_implementation)
    summary = {"status": "generating", "execution_status": "pending_execution", "pass_at_1": None,
               "protocol": protocol, "protocol_sha256": sha256_bytes(json.dumps(protocol, sort_keys=True).encode()),
               "scope": scope, "task_ids": list(selected_ids), "task_count": len(selected_ids),
               "probe_selection": "first32_sha256(humaneval-probe-v2/17/task_id), no model information",
               "model_identity": identity, "source_checkpoint": str(args.checkpoint) if args.checkpoint else None,
               "source_checkpoint_sha256": checkpoint_hash, "source_config": source["config"] if source else None,
               "generated_tasks": 0, "started_utc": datetime.now(timezone.utc).isoformat(),
               "selection_uses_coding_metrics": False, "data_manifest": manifest,
               "generator_sha256": file_hash(Path(__file__)),
               "package_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")}}
    write_json(args.output / "summary.json", summary)
    run_started = time.perf_counter()
    samples = []
    try:
        random.seed(17)
        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        model = load_text_model(args.model_path, device="cuda", dtype=torch.bfloat16,
                                attn_implementation=args.attn_implementation)
        # This merges the current adapter too, reproducing a task-boundary model.
        history = merge_boundary_checkpoint(model, args.checkpoint, args.model_path)
        summary["merge_history"] = history
        summary["boundary_index"] = len(history)
        summary["model_loading"] = model.cl_loading_info
        model.requires_grad_(False).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        with (args.output / "trace.jsonl").open("w", encoding="utf-8") as trace, torch.inference_mode():
            for problem in prompts:
                prompt = problem["prompt"]
                batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
                batch = {key: value.to("cuda") for key, value in batch.items()}
                prompt_length = batch["input_ids"].shape[1]

                class StopAtBoundary(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        continuation = tokenizer.decode(input_ids[0, prompt_length:].tolist(), skip_special_tokens=True)
                        return any(stop in continuation for stop in STOP_STRINGS)

                started = time.perf_counter()
                tokens = model.generate(**batch, do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
                                        use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                        stopping_criteria=StoppingCriteriaList([StopAtBoundary()]))
                generated_ids = tokens[0, prompt_length:].tolist()
                raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
                completion, stop = trim_completion(raw)
                eos = model.generation_config.eos_token_id
                eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
                max_token_hit = len(generated_ids) >= MAX_NEW_TOKENS
                truncated = max_token_hit and stop is None and (not generated_ids or generated_ids[-1] not in eos_ids)
                row = {"task_id": problem["task_id"], "completion": completion, "raw_completion": raw,
                       "prompt_sha256": sha256_bytes(prompt.encode()), "stop_string": stop,
                       "generated_tokens": len(generated_ids), "max_token_hit": max_token_hit, "truncated": truncated,
                       "elapsed_seconds": time.perf_counter() - started,
                       "execution_status": "pending_execution", "passed": None}
                samples.append(row)
                trace.write(json.dumps(row) + "\n")
                trace.flush()
                summary["generated_tasks"] = len(samples)
                summary['timing'] = generation_statistics(samples)
                write_json(args.output / "summary.json", summary)
                print(json.dumps({"task_id": row["task_id"], "generated_tasks": len(samples)}), flush=True)
        write_json(args.output / "humaneval_completions.json", samples)
        summary.update(status="completed", execution_status="pending_execution",
                       completions_sha256=file_hash(args.output / "humaneval_completions.json"))
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary['elapsed_seconds'] = time.perf_counter() - run_started
        summary['timing'] = generation_statistics(samples)
        write_json(args.output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    gen = subparsers.add_parser("generate")
    gen.add_argument("--data-dir", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--model-path")
    gen.add_argument("--checkpoint", type=Path)
    gen.add_argument("--scope", choices=("probe", "full"), default="probe")
    gen.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="flash_attention_2")
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.output), indent=2))
    else:
        generate(args)


if __name__ == "__main__":
    main()

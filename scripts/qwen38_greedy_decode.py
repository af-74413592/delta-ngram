"""Framework-independent, forward-only greedy generation for Qwen checkpoints.

The MCore loader supplies ``forward_next(full_prefix_token_ids) -> int``.  Every
distributed rank runs this same loop and the callback must return the same
global-vocabulary token on every rank.  Only ``writer=True`` writes files.  No
model, optimizer, CUDA context, or teacher-forced loss is created here.

Math/code outputs follow the existing locked scorers. Task outputs use the
existing ``inflection_exact`` scorer, including all accepted references.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def load_math_suite(data_dir, scope="probe", backend="mcore_flash"):
    import math_probe
    data_dir = Path(data_dir)
    records, manifest = math_probe.load_locked_data(data_dir)
    ids = list(math_probe.scope_ids(scope, manifest))
    records = math_probe.math_eval.project_rows(records, ids)
    records = [dict(row, prompt=math_probe.math_eval.PROMPT_TEMPLATE.format(problem=row["problem"]))
               for row in records]
    return {"domain": "math", "scope": scope, "records": records, "task_ids": ids,
            "data_manifest": manifest, "protocol": math_probe.decode_protocol(manifest, backend),
            "filename": "math_completions.json", "math_core_sha256": _file_hash(math_probe.math_eval.__file__)}


def load_code_suite(data_dir, scope="probe", backend="mcore_flash"):
    import humaneval_probe
    records, manifest = humaneval_probe.load_locked_data(Path(data_dir))
    ids = list(humaneval_probe.scope_ids(scope))
    return {"domain": "code", "scope": scope,
            "records": humaneval_probe.project_tasks(records, ids), "task_ids": ids,
            "data_manifest": manifest, "protocol": humaneval_probe.decode_protocol(manifest, backend),
            "filename": "humaneval_completions.json"}


def load_qwen_reference_suite(records, manifest, domain, data_root=None, backend="mcore_flash"):
    """Use the actual Qwen math128/code64 population, never the old 64/32 probe.

    ``records, manifest`` come from the retention loader's verified v2 reader.
    Its reference completion is deliberately removed from generation records.
    These outputs are scored by qwen38_score_generated, not the legacy scope
    scorer whose fixed population is different.
    """
    import math_probe
    import humaneval_probe
    if domain not in {"math", "code"}:
        raise ValueError("Qwen reference domain must be math or code")
    if manifest.get("protocol") != "retention_reference_nll_v2_128_64" or manifest.get("version") != 2:
        raise ValueError("Qwen generation requires the locked v2 math128/code64 population")
    selected = [row for row in records if row["domain"] == domain]
    ids = [row["task_id"] for row in selected]
    expected = 128 if domain == "math" else 64
    if ids != manifest["task_ids"][domain] or len(ids) != expected or len(set(ids)) != expected:
        raise ValueError("Qwen generation population/order differs from the locked reference manifest")
    prompts = []
    for row in selected:
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("reference prompt hash mismatch")
        prompts.append({key: row[key] for key in ("domain", "task_id", "prompt", "prompt_sha256")})
    protocol = {"version": 1, "benchmark": "Qwen_fixed_retention_v2", "domain": domain,
                "population": "math128_code64", "data_manifest_sha256": _digest(manifest),
                "prompt_format": "original_locked_raw_prompt_no_chat_template_no_reference",
                "do_sample": False, "num_beams": 1, "samples_per_task": 1, "batch_size": 1,
                "max_new_tokens": 1024 if domain == "math" else 256,
                "stop_strings": list(math_probe.STOP_STRINGS if domain == "math" else humaneval_probe.STOP_STRINGS),
                "seed": 17, "dtype": "torch.bfloat16", "attn_implementation": backend}
    return {"domain": domain, "scope": "locked_v2_128_64", "records": prompts, "task_ids": ids,
            "data_manifest": manifest, "protocol": protocol,
            "filename": "math_completions.json" if domain == "math" else "humaneval_completions.json",
            "data_root": str(data_root) if data_root is not None else None}


def load_task_suite(path, tasks=None, probes_per_task=64, max_new_tokens=64,
                    backend="mcore_flash", prompt_format="actual_swift_generation_template"):
    """Select fixed JSONL rows in file order, never using targets for selection."""
    path = Path(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    available = list(dict.fromkeys(row["task"] for row in records))
    chosen = list(tasks) if tasks is not None else available
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(available):
        raise ValueError("task selection is empty, duplicated or absent from the fixed task file")
    if probes_per_task < 1 or max_new_tokens < 1:
        raise ValueError("task sample/token limits must be positive")
    selected = []
    for task in chosen:
        rows = [row for row in records if row["task"] == task][:probes_per_task]
        if len(rows) != probes_per_task:
            raise ValueError(f"task {task} has fewer than {probes_per_task} fixed probes")
        for row in rows:
            if not all(isinstance(row.get(key), str) and row[key] for key in ("task_id", "prompt", "target")):
                raise ValueError("invalid fixed task record")
        selected.extend(dict(row, metric=row.get("metric", "inflection_exact")) for row in rows)
    ids = [row["task_id"] for row in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task IDs in fixed task probes")
    protocol = {"version": 1, "benchmark": "fixed_niche_tasks", "prompt_format": prompt_format,
                "max_new_tokens": max_new_tokens, "do_sample": False, "num_beams": 1,
                "samples_per_task": 1, "batch_size": 1, "stop_strings": [], "seed": 17,
                "dtype": "torch.bfloat16", "attn_implementation": backend,
                "scoring": "cl_research.niche_metrics.score_prediction"}
    return {"domain": "tasks", "scope": "probe", "records": selected, "task_ids": ids,
            "data_manifest": {"source": str(path.resolve()), "source_sha256": _file_hash(path),
                              "tasks": chosen, "probes_per_task": probes_per_task},
            "protocol": protocol, "filename": "task_completions.json"}


def greedy_decode(prompt_token_ids: Sequence[int], *, forward_next: Callable[[list[int]], int],
                  decode: Callable[[list[int]], str], eos_token_ids=(), max_new_tokens=32,
                  stop_strings=(), max_context_tokens=None):
    """Generate one answer without ever supplying its reference tokens."""
    prefix = list(prompt_token_ids)
    if not prefix or any(type(token) is not int or token < 0 for token in prefix):
        raise ValueError("prompt must be a nonempty list of nonnegative token IDs")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if max_context_tokens is not None and len(prefix) >= max_context_tokens:
        raise ValueError("prompt leaves no room for generation; do not silently truncate")
    eos = set(eos_token_ids)
    generated, stop, finish = [], None, "max_new_tokens"
    started = time.perf_counter()
    raw = ""
    for _ in range(max_new_tokens):
        if max_context_tokens is not None and len(prefix) >= max_context_tokens:
            finish = "context_limit"
            break
        next_id = forward_next(prefix)
        # Permit a scalar returned by a distributed torch reduction.
        if hasattr(next_id, "item"):
            next_id = next_id.item()
        if type(next_id) is not int or next_id < 0:
            raise ValueError("forward_next must return one global nonnegative integer token ID")
        generated.append(next_id)
        prefix = prefix + [next_id]
        raw = decode(generated)
        matches = [(raw.find(value), value) for value in stop_strings if value in raw]
        if matches:
            position, stop = min(matches)
            finish = "stop_string"
            break
        if next_id in eos:
            finish = "eos"
            break
    completion = raw[:raw.find(stop)] if stop is not None else raw
    hit = len(generated) >= max_new_tokens
    return {"completion": completion, "raw_completion": raw, "stop_string": stop,
            "finish_reason": finish, "generated_token_ids": generated,
            "generated_tokens": len(generated), "prompt_tokens": len(prompt_token_ids),
            "max_token_hit": hit, "truncated": finish in ("max_new_tokens", "context_limit"),
            "elapsed_seconds": time.perf_counter() - started}


def run_suite(suite, tokenizer, forward_next, output, *, model_identity,
              source_checkpoint=None, source_checkpoint_sha256=None, writer=True,
              encode_prompt=None, before_query=None, eos_token_ids=None,
              max_context_tokens=None, checkpoint_state="native_boundary_restored"):
    """Run one locked suite; only writer rank touches output or emits progress.

    ``encode_prompt(record) -> list[int]`` is mandatory for task chat templates.
    Raw math/code defaults to tokenizer.encode(prompt, add_special_tokens=True).
    ``before_query(record, prompt_ids)`` may reset the caller's inference cache.
    The callback and tokenizer must be deterministic and shared on all ranks.
    """
    output = Path(output)
    if suite["domain"] == "tasks" and encode_prompt is None:
        raise ValueError("task generation requires the actual training-compatible encode_prompt callback")
    records = suite["records"]
    ids = [row["task_id"] for row in records]
    if not records or ids != suite["task_ids"] or len(set(ids)) != len(ids):
        raise ValueError("suite records differ from their locked task IDs")
    if writer:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError(f"use a fresh generation directory: {output}")
    protocol = dict(suite["protocol"], checkpoint_state=checkpoint_state)
    summary = {"status": "generating", "domain": suite["domain"], "suite": suite["domain"], "scope": suite["scope"],
               "protocol": protocol, "protocol_sha256": _digest(protocol),
               "data_manifest": suite["data_manifest"], "model_identity": model_identity,
               "source_checkpoint": str(source_checkpoint) if source_checkpoint is not None else None,
               "source_checkpoint_sha256": source_checkpoint_sha256,
               "task_ids": ids, "task_count": len(ids), "generated_tasks": 0,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "generator_sha256": _file_hash(__file__), "optimizer_constructed": False,
               "optimizer_steps": 0, "teacher_forced": False, "selection_uses_scores": False}
    if "math_core_sha256" in suite:
        summary["math_core_sha256"] = suite["math_core_sha256"]
    if suite["domain"] == "code":
        summary.update(execution_status="pending_execution", pass_at_1=None)
    if eos_token_ids is None:
        eos_token_ids = tokenizer.eos_token_id
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos_token_ids = [value for value in (eos_token_ids or []) if value is not None]
    summary["eos_token_ids"] = eos_token_ids
    if writer:
        _write_json(output / "summary.json", summary)
    samples, started = [], time.perf_counter()
    try:
        for record in records:
            prompt_ids = (encode_prompt(record) if encode_prompt is not None
                          else tokenizer.encode(record["prompt"], add_special_tokens=True))
            prompt_ids = list(prompt_ids)
            if before_query is not None:
                before_query(record, prompt_ids)
            result = greedy_decode(prompt_ids, forward_next=forward_next,
                                   decode=lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True),
                                   eos_token_ids=eos_token_ids, max_new_tokens=protocol["max_new_tokens"],
                                   stop_strings=protocol.get("stop_strings", []),
                                   max_context_tokens=max_context_tokens)
            row = {"task_id": record["task_id"], **result,
                   "prompt_sha256": hashlib.sha256(record["prompt"].encode()).hexdigest(),
                   "prompt_token_ids_sha256": _digest(prompt_ids)}
            if suite["domain"] == "math":
                row.update({key: record[key] for key in ("subject", "level") if key in record})
                row.update(correct=None, scoring_status="pending_scoring")
            elif suite["domain"] == "code":
                row.update(execution_status="pending_execution", passed=None)
            else:
                sys.path.insert(0, str(ROOT / "src" / "cl_research"))
                from niche_metrics import score_prediction
                row.update(task=record["task"], prediction=row["completion"], target=record["target"],
                           references=record.get("references", [record["target"]]),
                           score=score_prediction(row["completion"], record))
            samples.append(row)
            summary["generated_tasks"] = len(samples)
            if writer:
                with (output / "trace.jsonl").open("a", encoding="utf-8") as trace:
                    trace.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                _write_json(output / "summary.json", summary)
                print(json.dumps({"generation_domain": suite["domain"], "task_id": row["task_id"],
                                  "generated_tasks": len(samples), "total": len(ids),
                                  "tokens": row["generated_tokens"], "seconds": row["elapsed_seconds"]}), flush=True)
        if suite["domain"] == "tasks":
            from niche_metrics import aggregate_scores
            grouped = defaultdict(list)
            for row in samples:
                grouped[row["task"]].append(row["score"])
            summary["tasks"] = {task: aggregate_scores(scores) for task, scores in grouped.items()}
            summary["generation"] = aggregate_scores([row["score"] for row in samples])
        summary["status"] = "completed"
        if writer:
            _write_json(output / suite["filename"], samples)
            summary["completions_sha256"] = _file_hash(output / suite["filename"])
            summary["trace_sha256"] = _file_hash(output / "trace.jsonl")
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.perf_counter() - started)
        from humaneval_probe import generation_statistics
        summary["timing"] = generation_statistics(samples)
        if writer:
            _write_json(output / "summary.json", summary)
    return summary

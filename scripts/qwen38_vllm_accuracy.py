"""Cached, batched greedy generation on the locked Qwen accuracy populations.

This runner takes an HF-layout model (base or an already exported checkpoint),
not native MCore adapter payloads. It never trains or measures reference NLL.
Task prompts must be encoded separately by the original Swift template with
enable_thinking=False; Swift is deliberately not imported by this process.

Example:
  python scripts/qwen38_vllm_accuracy.py --model /workspace/models/Qwen3.8-Flash-Next \
    --output /workspace/eval/base --domains code math --method base --stage 0

Task-token manifest schema: {schema_version:1,
 prompt_format:"actual_swift_generation_template", enable_thinking:false,
 source_sha256:<task_probes64.jsonl hash>,
 tokenizer_files:{"tokenizer.json":<hash>,"tokenizer_config.json":<hash>},
 records:[{task_id,prompt_sha256,prompt_token_ids:[...]}]}.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen38_greedy_decode as locked

MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 16


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_suites(retention_data, task_data, domains, task_probe_tasks=None):
    """Validate all locked source files before creating a CUDA context."""
    suites = []
    if any(domain in {"math", "code"} for domain in domains):
        directory = Path(retention_data)
        manifest = read_json(directory / "manifest.json")
        source = directory / "records.jsonl"
        if locked._file_hash(source) != manifest["files"]["records.jsonl"]:
            raise ValueError("locked math/code source checksum changed")
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    for domain in domains:
        if domain in {"math", "code"}:
            suite = locked.load_qwen_reference_suite(records, manifest, domain,
                                                     data_root=retention_data, backend="vllm_cached")
        else:
            directory = Path(task_data)
            task_manifest = read_json(directory / "manifest.json")
            source = directory / "task_probes64.jsonl"
            if (task_manifest.get("protocol") != "qwen38_niche_probes64_v1"
                    or task_manifest.get("task_order") != ["sme", "nav", "hye", "afb", "sqi", "mkd", "amh"]
                    or locked._file_hash(source) != task_manifest["files"][source.name]):
                raise ValueError("locked seven-task manifest/source changed")
            chosen_tasks = list(task_probe_tasks) if task_probe_tasks else list(task_manifest["task_order"])
            if (not chosen_tasks or len(set(chosen_tasks)) != len(chosen_tasks)
                    or not set(chosen_tasks).issubset(task_manifest["task_order"])):
                raise ValueError("invalid task probe selection")
            suite = locked.load_task_suite(source, tasks=chosen_tasks,
                                           probes_per_task=64, max_new_tokens=128, backend="vllm_cached")
            expected = [task_id for task in chosen_tasks
                        for task_id in task_manifest["sources"][task]["ids"]]
            if suite["task_ids"] != expected or len(expected) != 64 * len(chosen_tasks):
                raise ValueError("task IDs differ from the locked task population")
        suites.append(suite)
    return suites


def check_token_ids(ids):
    if not isinstance(ids, list) or not ids or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError("prompt_token_ids must be a nonempty list of nonnegative integers")
    return ids


def load_task_tokens(path, suite, model_path):
    if path is None:
        raise ValueError("tasks require --task-prompt-tokens from the original Swift template")
    manifest = read_json(path)
    if (manifest.get("schema_version") != 1
            or manifest.get("prompt_format") != "actual_swift_generation_template"
            or manifest.get("enable_thinking") is not False
            or manifest.get("source_sha256") != suite["data_manifest"]["source_sha256"]):
        raise ValueError("task-token manifest does not bind the exact Swift non-thinking source")
    files = manifest.get("tokenizer_files", {})
    if not {"tokenizer.json", "tokenizer_config.json"}.issubset(files):
        raise ValueError("task-token manifest must bind both tokenizer files")
    for name, digest in files.items():
        if Path(name).name != name or locked._file_hash(Path(model_path) / name) != digest:
            raise ValueError(f"task tokenizer file checksum mismatch: {name}")
    rows = manifest.get("records", [])
    expected_ids = suite["task_ids"]
    if [row.get("task_id") for row in rows] != expected_ids:
        by_id = {row.get("task_id"): row for row in rows}
        if not set(expected_ids).issubset(by_id):
            raise ValueError("task-token IDs/order differ from locked 7x64 queries")
        rows = [by_id[task_id] for task_id in expected_ids]
    result = {}
    for row, original in zip(rows, suite["records"]):
        if row.get("prompt_sha256") != text_hash(original["prompt"]):
            raise ValueError(f"task-token prompt checksum mismatch: {original['task_id']}")
        result[row["task_id"]] = check_token_ids(row.get("prompt_token_ids"))
    return result, locked._file_hash(path)


def encode_suite(suite, tokenizer, *, task_tokens=None):
    encoded = []
    for row in suite["records"]:
        ids = (task_tokens[row["task_id"]] if suite["domain"] == "tasks"
               else list(tokenizer.encode(row["prompt"], add_special_tokens=True)))
        check_token_ids(ids)
        if len(ids) + suite["protocol"]["max_new_tokens"] > MAX_MODEL_LEN:
            raise ValueError(f"query exceeds fixed {MAX_MODEL_LEN} context: {row['task_id']}; no truncation")
        encoded.append(ids)
    return encoded


def hf_model_identity(model_path):
    """Bind resume to this HF directory without rehashing 360 GB on each run.

    The stat inventory is a mutation guard, not a claim of upstream SHA audit.
    An export/base verification manifest can additionally be bound by the CLI.
    """
    model_path = Path(model_path).resolve()
    config = read_json(model_path / "config.json")
    if config.get("quantization_config"):
        raise ValueError("this comparison requires unquantized BF16 weights")
    index_path = model_path / "model.safetensors.index.json"
    index = read_json(index_path)
    inventory = {}
    for name in sorted(set(index["weight_map"].values())):
        path = (model_path / name).resolve()
        if not path.is_relative_to(model_path):
            raise ValueError("model index escapes its directory")
        info = path.stat()
        inventory[name] = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
    return {"model_path": str(model_path), "config_sha256": locked._file_hash(model_path / "config.json"),
            "index_sha256": locked._file_hash(index_path), "weight_stat_inventory": inventory,
            "weight_stat_inventory_sha256": locked._digest(inventory),
            "weights_rehashed_by_generation": False,
            "tokenizer_files": {name: locked._file_hash(model_path / name)
                                for name in ("tokenizer.json", "tokenizer_config.json")}}


def engine_kwargs(model_path, *, enforce_eager=True, gpu_memory_utilization=.9,
                  tensor_parallel_size=4, moe_backend="auto"):
    if type(tensor_parallel_size) is not int or tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be a positive integer")
    return dict(model=str(model_path), tensor_parallel_size=tensor_parallel_size, pipeline_parallel_size=1,
                dtype="bfloat16", quantization=None, max_model_len=MAX_MODEL_LEN,
                max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=4096,
                language_model_only=True, seed=17, enforce_eager=enforce_eager,
                gpu_memory_utilization=gpu_memory_utilization, generation_config="vllm",
                kv_cache_dtype="auto", attention_config={"indexer_kv_dtype": "bf16"},
                moe_backend=moe_backend)


def configure_ple_cpu_offload(kwargs, engine_args_cls, vllm_envs, *, requested):
    """Only enable PLE offload through an interface this vLLM exposes."""
    if not requested:
        return
    fields = set()
    if dataclasses.is_dataclass(engine_args_cls):
        fields.update(field.name for field in dataclasses.fields(engine_args_cls))
    for cls in engine_args_cls.__mro__:
        fields.update(getattr(cls, "__annotations__", {}))
    fields.update(parameter.name for parameter in inspect.signature(engine_args_cls).parameters.values()
                  if parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL))
    if "engram_config" in fields:
        kwargs["engram_config"] = {"cpu_offload": True}
        return
    registry = vars(vllm_envs).get("environment_variables", {})
    name = "VLLM_PLE_CPU_OFFLOAD"
    if not isinstance(registry, dict) or name not in registry:
        raise RuntimeError("PLE CPU offload requested, but EngineArgs has no engram_config "
                           "and vllm.envs does not register VLLM_PLE_CPU_OFFLOAD")
    os.environ[name] = "1"
    if getattr(vllm_envs, name) not in (True, 1):
        raise RuntimeError("VLLM_PLE_CPU_OFFLOAD did not resolve to enabled")


def execution_config(kwargs, config=None, vllm_envs=None, *, ple_cpu_offload_requested=False):
    """Bind requested arguments and resolved runtime settings, without model paths.

    The no-config form is useful for CPU-only fixtures. Production always passes
    EngineArgs.create_engine_config(), and checks the live engine against it.
    """
    requested = copy.deepcopy({key: value for key, value in kwargs.items()
                               if key not in {"model", "tokenizer"}})
    result = {"requested": requested, "resolved": None,
              "ple_cpu_offload_requested": ple_cpu_offload_requested}
    if config is None:
        return result
    model = config.model_config
    parallel = config.parallel_config
    scheduler = config.scheduler_config
    cache = config.cache_config
    attention = config.attention_config
    indexer_dtype = (attention.resolve_indexer_kv_dtype("bf16")
                     if hasattr(attention, "resolve_indexer_kv_dtype") else attention.indexer_kv_dtype)
    resolved = {
        "tensor_parallel_size": parallel.tensor_parallel_size,
        "pipeline_parallel_size": parallel.pipeline_parallel_size,
        "dtype": str(model.dtype).removeprefix("torch."),
        "quantization": model.quantization,
        "max_model_len": model.max_model_len,
        "max_num_seqs": scheduler.max_num_seqs,
        "max_num_batched_tokens": scheduler.max_num_batched_tokens,
        "enforce_eager": model.enforce_eager,
        "gpu_memory_utilization": cache.gpu_memory_utilization,
        "kv_cache_dtype": cache.cache_dtype,
        "indexer_kv_dtype": indexer_dtype,
        "moe_backend": config.kernel_config.moe_backend,
    }
    engram = getattr(config, "engram_config", None)
    registry = vars(vllm_envs).get("environment_variables", {}) if vllm_envs is not None else {}
    if engram is not None:
        resolved["ple_cpu_offload"] = engram.cpu_offload
        resolved["ple_cpu_offload_interface"] = "engram_config"
    elif isinstance(registry, dict) and "VLLM_PLE_CPU_OFFLOAD" in registry:
        resolved["ple_cpu_offload"] = bool(getattr(vllm_envs, "VLLM_PLE_CPU_OFFLOAD"))
        resolved["ple_cpu_offload_interface"] = "VLLM_PLE_CPU_OFFLOAD"
    else:
        # Omission is not evidence that an unknown implementation disables it.
        resolved["ple_cpu_offload"] = None
        resolved["ple_cpu_offload_interface"] = "unexposed"
    for key in ("tensor_parallel_size", "pipeline_parallel_size", "dtype", "quantization",
                "max_model_len", "max_num_seqs", "max_num_batched_tokens", "enforce_eager",
                "gpu_memory_utilization"):
        if resolved[key] != requested[key]:
            raise ValueError(f"resolved {key} differs from the requested accuracy configuration")
    if resolved["kv_cache_dtype"] not in ("auto", "bfloat16") or indexer_dtype != "bf16":
        raise ValueError("accuracy generation requires BF16 main KV and indexer caches")
    if requested["moe_backend"] != "auto" and resolved["moe_backend"] != requested["moe_backend"]:
        raise ValueError("resolved MoE backend differs from the requested accuracy configuration")
    if ple_cpu_offload_requested and resolved["ple_cpu_offload"] is not True:
        raise RuntimeError("requested PLE CPU offload was not enabled in the resolved configuration")
    result["resolved"] = resolved
    # Freeze a plain JSON value before vLLM can mutate nested argument objects.
    return json.loads(json.dumps(result, allow_nan=False))


def protocol_for(suite, eos_ids, *, execution=None):
    execution = copy.deepcopy(execution if execution is not None else execution_config(engine_kwargs("")))
    settings = execution["requested"]
    return dict(suite["protocol"], batch_size=MAX_NUM_SEQS, temperature=0., seed=17,
                engine="vllm", kv_cache=True, max_model_len=MAX_MODEL_LEN,
                tensor_parallel_size=settings["tensor_parallel_size"], pipeline_parallel_size=1, quantization=None,
                language_model_only=True, max_num_batched_tokens=4096,
                execution_config=execution,
                eos_token_ids=eos_ids, checkpoint_state="hf_layout_weights_loaded")


def make_contract(suite, encoded, identity, *, eos_ids, method, stage,
                  source_checkpoint=None, source_checkpoint_sha256=None, task_token_sha256=None,
                  execution=None):
    return {"version": 2, "suite": suite["domain"], "task_ids": suite["task_ids"],
            "protocol": protocol_for(suite, eos_ids, execution=execution), "model_identity": identity,
            "data_manifest": suite["data_manifest"], "method": method, "stage": stage,
            "source_checkpoint": source_checkpoint, "source_checkpoint_sha256": source_checkpoint_sha256,
            "task_token_manifest_sha256": task_token_sha256,
            "prompt_bindings": [{"task_id": row["task_id"], "prompt_sha256": text_hash(row["prompt"]),
                                 "prompt_token_ids_sha256": locked._digest(ids)}
                                for row, ids in zip(suite["records"], encoded)]}


def load_resume(output, contract):
    output = Path(output)
    if not output.exists() or not any(output.iterdir()):
        return []
    summary = read_json(output / "summary.json")
    contract_hash = locked._digest(contract)
    if summary.get("run_contract_sha256") != contract_hash:
        raise ValueError(f"resume contract changed: {output}; use a new output directory")
    trace = output / "trace.jsonl"
    if not trace.exists():
        if summary.get("generated_tasks", 0):
            raise ValueError("resume trace is missing")
        return []
    if summary.get("trace_sha256") != locked._file_hash(trace):
        raise ValueError("resume trace checksum mismatch")
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [row["task_id"] for row in rows] != contract["task_ids"][:len(rows)]:
        raise ValueError("resume trace is not an ordered prefix of locked IDs")
    if len(rows) != summary.get("generated_tasks"):
        raise ValueError("resume summary/trace count differs")
    for row, expected in zip(rows, contract["prompt_bindings"]):
        if any(row.get(key) != expected[key] for key in expected):
            raise ValueError("resume prompt identity changed")
        if row.get("run_contract_sha256") != contract_hash:
            raise ValueError("resume row checkpoint/protocol identity changed")
    return rows


def sample_row(request, record, prompt_ids, tokenizer, *, max_tokens, batch_seconds, contract_hash):
    if list(request.prompt_token_ids) != prompt_ids or len(request.outputs) != 1 or not request.finished:
        raise ValueError("vLLM output does not match the requested prompt/single completed sample")
    completion = request.outputs[0]
    token_ids = [int(value) for value in completion.token_ids]
    metrics = getattr(request, "metrics", None)
    started = getattr(metrics, "arrival_time", None)
    finished = getattr(metrics, "finished_time", None)
    latency = (finished - started if started is not None and finished is not None else None)
    if latency is not None and (not math.isfinite(latency) or latency < 0):
        latency = None
    row = {"task_id": record["task_id"], "completion": completion.text,
           "raw_completion": tokenizer.decode(token_ids, skip_special_tokens=True),
           "prompt_sha256": text_hash(record["prompt"]), "prompt_token_ids": prompt_ids,
           "prompt_token_ids_sha256": locked._digest(prompt_ids),
           "generated_token_ids": token_ids, "generated_token_ids_sha256": locked._digest(token_ids),
           "generated_tokens": len(token_ids), "prompt_tokens": len(prompt_ids),
           "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason,
           "stop_string": completion.stop_reason if isinstance(completion.stop_reason, str) else None,
           "truncated": completion.finish_reason == "length", "max_token_hit": len(token_ids) >= max_tokens,
           "elapsed_seconds": latency, "batch_elapsed_seconds": batch_seconds,
           "elapsed_seconds_kind": "vllm_request_latency" if latency is not None else "unavailable",
           "run_contract_sha256": contract_hash, "generated_utc": utc_now()}
    if "task" in record:
        row.update(task=record["task"], prediction=completion.text)
    return row


def run_suite(llm, sampling_params_cls, tokenizer, suite, encoded, output, contract, *, existing=None):
    output = Path(output)
    samples = load_resume(output, contract) if existing is None else list(existing)
    output.mkdir(parents=True, exist_ok=True)
    protocol = contract["protocol"]
    contract_hash = locked._digest(contract)
    summary = {**contract, "domain": suite["domain"], "scope": suite["scope"],
               "status": "generating", "run_contract_sha256": contract_hash,
               "protocol_sha256": locked._digest(protocol), "task_count": len(suite["records"]),
               "generated_tasks": len(samples), "resumed_tasks": len(samples),
               "started_utc": utc_now(), "generator_sha256": locked._file_hash(__file__),
               "optimizer_constructed": False, "optimizer_steps": 0,
               "teacher_forced": False, "selection_uses_scores": False}
    if (output / "trace.jsonl").exists():
        summary["trace_sha256"] = locked._file_hash(output / "trace.jsonl")
    locked._write_json(output / "summary.json", summary)
    started = time.perf_counter()
    try:
        params = None
        if len(samples) < len(encoded):
            params = sampling_params_cls(n=1, temperature=0., top_p=1., top_k=-1, seed=17,
                                     max_tokens=protocol["max_new_tokens"],
                                     stop=protocol.get("stop_strings", []), stop_token_ids=protocol["eos_token_ids"],
                                     ignore_eos=False, skip_special_tokens=True,
                                     include_stop_str_in_output=False)
        batch_size = protocol["batch_size"]
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("registered batch size must be a positive integer")
        for offset in range(len(samples), len(encoded), batch_size):
            ids_batch = encoded[offset:offset + batch_size]
            batch_started = time.perf_counter()
            responses = llm.generate([{"prompt_token_ids": ids} for ids in ids_batch],
                                     sampling_params=params, use_tqdm=False)
            seconds = time.perf_counter() - batch_started
            if len(responses) != len(ids_batch):
                raise ValueError("vLLM returned a different number of requests")
            for response, record, ids in zip(responses, suite["records"][offset:offset + batch_size], ids_batch):
                row = sample_row(response, record, ids, tokenizer, max_tokens=protocol["max_new_tokens"],
                                 batch_seconds=seconds, contract_hash=contract_hash)
                samples.append(row)
                # Atomic full trace snapshot is tiny (<=448 rows), making
                # interrupted batches restartable without half-written JSONL.
                trace = output / "trace.jsonl"
                temporary = trace.with_name("trace.jsonl.tmp")
                temporary.write_text("".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
                                             for item in samples), encoding="utf-8")
                temporary.replace(trace)
                summary.update(generated_tasks=len(samples), trace_sha256=locked._file_hash(trace))
                locked._write_json(output / "summary.json", summary)
                print(json.dumps({"suite": suite["domain"], "task_id": row["task_id"],
                                  "generated": len(samples), "total": len(encoded),
                                  "tokens": row["generated_tokens"], "batch_seconds": seconds}), flush=True)
        locked._write_json(output / suite["filename"], samples)
        summary.update(status="completed", completions_sha256=locked._file_hash(output / suite["filename"]),
                       trace_sha256=locked._file_hash(output / "trace.jsonl"))
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary.update(finished_utc=utc_now(), elapsed_seconds=time.perf_counter() - started)
        locked._write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", choices=("math", "code", "tasks"), default=["code", "math", "tasks"])
    parser.add_argument("--retention-data", type=Path, default=ROOT / "data/retention_nll_v2_128_64")
    parser.add_argument("--task-data", type=Path, default=ROOT / "data/qwen38_eval_v1")
    parser.add_argument("--task-prompt-tokens", type=Path)
    parser.add_argument("--task-probe-tasks", nargs="+",
                        help="Optional ordered subset of locked language tasks for the incremental triangle")
    parser.add_argument("--method", choices=("base", "ngram", "shared", "merged"), required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--source-checkpoint-sha256")
    parser.add_argument("--identity-manifest", type=Path, help="Optional original verified base/export provenance JSON")
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--ple-cpu-offload", action="store_true",
                        help="Require PLE CPU offload through a supported vLLM interface")
    parser.add_argument("--moe-backend", choices=("auto", "triton"), default="auto")
    args = parser.parse_args()
    if (len(set(args.domains)) != len(args.domains) or not 0 <= args.stage <= 7
            or (args.method == "base") != (args.stage == 0)):
        parser.error("unique domains and consistent method/stage 0..7 required")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("GPU memory utilization must be in (0,1]")
    if args.tensor_parallel_size < 1:
        parser.error("tensor parallel size must be positive")
    suites = load_suites(args.retention_data, args.task_data, args.domains,
                         task_probe_tasks=args.task_probe_tasks)
    identity = hf_model_identity(args.model)
    identity["runtime_versions"] = {name: importlib.metadata.version(name)
                                    for name in ("vllm", "torch", "transformers")}
    if args.identity_manifest:
        identity["provenance_manifest"] = str(args.identity_manifest.resolve())
        identity["provenance_manifest_sha256"] = locked._file_hash(args.identity_manifest)
    task_tokens, task_token_hash = None, None
    if "tasks" in args.domains:
        task_tokens, task_token_hash = load_task_tokens(args.task_prompt_tokens,
            next(suite for suite in suites if suite["domain"] == "tasks"), args.model)
    # Resolve CPU-side configuration before checking resume; no model is loaded.
    # vLLM owns the TP workers and must spawn them after the offload settings exist.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import vllm.envs as vllm_envs
    from vllm.engine.arg_utils import EngineArgs
    kwargs = engine_kwargs(args.model.resolve(), enforce_eager=args.enforce_eager,
                           gpu_memory_utilization=args.gpu_memory_utilization,
                           tensor_parallel_size=args.tensor_parallel_size, moe_backend=args.moe_backend)
    configure_ple_cpu_offload(kwargs, EngineArgs, vllm_envs, requested=args.ple_cpu_offload)
    planned_config = EngineArgs(**copy.deepcopy(kwargs)).create_engine_config()
    execution = execution_config(kwargs, planned_config, vllm_envs,
                                 ple_cpu_offload_requested=args.ple_cpu_offload)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    generation_config_path = args.model / "generation_config.json"
    generation_config = read_json(generation_config_path) if generation_config_path.exists() else {}
    eos_ids = generation_config.get("eos_token_id", tokenizer.eos_token_id)
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
    prepared = []
    for suite in suites:
        encoded = encode_suite(suite, tokenizer, task_tokens=task_tokens)
        contract = make_contract(suite, encoded, identity, eos_ids=eos_ids, method=args.method, stage=args.stage,
                                 source_checkpoint=args.source_checkpoint,
                                 source_checkpoint_sha256=args.source_checkpoint_sha256,
                                 task_token_sha256=task_token_hash if suite["domain"] == "tasks" else None,
                                 execution=execution)
        existing = load_resume(args.output / suite["domain"], contract)
        prepared.append((suite, encoded, contract, existing))
    if all(len(existing) == len(suite["records"]) for suite, _, _, existing in prepared):
        for suite, encoded, contract, existing in prepared:
            run_suite(None, None, tokenizer, suite, encoded, args.output / suite["domain"], contract,
                      existing=existing)
        print(json.dumps({"status": "already_completed", "output": str(args.output)}), flush=True)
        return
    from vllm import LLM, SamplingParams
    print(json.dumps({"event": "loading_model", "model": str(args.model),
                      "tp": args.tensor_parallel_size, "pp": 1, "execution_config": execution,
                      "dtype": "bfloat16", "vllm_version": importlib.metadata.version("vllm")}), flush=True)
    llm = LLM(**copy.deepcopy(kwargs))
    actual_execution = execution_config(kwargs, llm.llm_engine.vllm_config, vllm_envs,
                                        ple_cpu_offload_requested=args.ple_cpu_offload)
    if actual_execution != execution:
        raise RuntimeError("loaded engine configuration differs from the prepared resume contract")
    for suite, encoded, contract, existing in prepared:
        run_suite(llm, SamplingParams, tokenizer, suite, encoded, args.output / suite["domain"], contract,
                  existing=existing)


if __name__ == "__main__":
    main()

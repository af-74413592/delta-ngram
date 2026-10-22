"""LoCoMo fixed full-context base generation; never loads answer annotations.

CPU validation precedes model initialization. Modes gate, pilot and formal
share one immutable contract, so the fixed conv-48 pilot can be resumed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

PROTOCOL = "qwen38_locomo_full_context_base_v3"
CONTEXT, NEW_TOKENS, PILOT = 25226, 32, "conv-48"
REQUIRED_MANIFEST_SHA = "3a5b27f86738fe6cfb1135d8bd1ef862e4aff82d07cc0f4c4dbadf1cd9250bd4"
REQUIRED_PROTOCOL_SHA = "866045ef6e8afa937b3424da2c232c20e57a4a687a1cb35c50613127bdaf791c"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def no_delta_environment():
    require(not any(value for key, value in os.environ.items() if key.startswith("DELTA_NGRAM_VLLM")),
        "full-context generation requires a process without Delta bindings")


def prompt_ids(question, prefixes):
    prefix = prefixes[question["full_context_prefix_id"]]
    ids = list(prefix["token_ids"]) + list(question["full_context_suffix_ids"])
    require(bool(ids) and all(type(token) is int and 0 <= token < 248320 for token in ids), "invalid prompt tokens")
    require(len(ids) == question["full_context_prompt_tokens"] and canonical(ids) == question["full_context_prompt_sha256"],
        "full-context prompt length/SHA differs")
    require(len(ids) + NEW_TOKENS <= CONTEXT, "full transcript cannot fit; truncation is forbidden")
    return ids


def load_population(prepared, *, expected_manifest_sha=REQUIRED_MANIFEST_SHA):
    prepared = Path(prepared)
    manifest_path = prepared / "manifest.json"
    require(digest(manifest_path) == expected_manifest_sha, "prepared manifest SHA differs")
    manifest = read_json(manifest_path)
    # Explicit allowlist: annotation and writer files are never opened here.
    names = ("protocol.json", "runtime_data/questions.jsonl", "runtime_data/full_context_prefixes.json")
    for name in names:
        path = prepared / name
        require(digest(path) == manifest["files"][name]["sha256"]
            and path.stat().st_size == manifest["files"][name]["bytes"], "runtime data changed")
    require(digest(prepared / "protocol.json") == manifest["protocol_sha256"] == REQUIRED_PROTOCOL_SHA,
        "LoCoMo protocol changed")
    protocol = read_json(prepared / "protocol.json")
    generation = protocol["generation"]
    require(protocol["protocol"] == "qwen38_locomo_zero_shot_v1" and generation["enable_thinking"] is False
        and generation["temperature"] == 0 and generation["max_new_tokens"] == NEW_TOKENS
        and generation["do_sample"] is False and generation["stop_token_ids"] == [248044, 248046]
        and protocol["timing_pilot_conversation"] == PILOT, "generation protocol differs")
    rows = [json.loads(line) for line in (prepared / "runtime_data/questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    prefixes = read_json(prepared / "runtime_data/full_context_prefixes.json")
    require(len(rows) == 1986 and len({row["qid"] for row in rows}) == 1986
        and len({row["sample_id"] for row in rows}) == 10 and len(prefixes) == 10, "LoCoMo population differs")
    for row in rows:
        require(row["sample_id"] in prefixes and row["qid"].startswith(row["sample_id"] + ":qa"), "invalid question identity")
        prompt_ids(row, prefixes)
    # This engineering order is fixed before scores and identical for all modes.
    ordered = [row for row in rows if row["sample_id"] == PILOT] + [row for row in rows if row["sample_id"] != PILOT]
    require(sum(row["sample_id"] == PILOT for row in ordered) == 239, "pilot population changed")
    counts = [row["full_context_prompt_tokens"] for row in rows]
    stats = {"questions": len(rows), "conversations": len(prefixes), "pilot_questions": 239,
        "prompt_tokens_min": min(counts), "prompt_tokens_max": max(counts), "prefill_tokens_without_prefix_reuse": sum(counts),
        "pilot_prefill_tokens": sum(row["full_context_prompt_tokens"] for row in rows if row["sample_id"] == PILOT),
        "maximum_generation_tokens": len(rows) * NEW_TOKENS, "context": CONTEXT, "truncate_prompt_tokens": None}
    require(stats["prompt_tokens_max"] == 25194 and stats["prefill_tokens_without_prefix_reuse"] == 43753517,
        "frozen token statistics differ")
    return protocol, ordered, prefixes, stats


def engine_settings(model):
    import qwen38_vllm_accuracy as accuracy
    kwargs = accuracy.engine_kwargs(model, enforce_eager=True, gpu_memory_utilization=.85,
        tensor_parallel_size=8, moe_backend="triton")
    kwargs.update(max_model_len=CONTEXT, max_num_seqs=1, max_num_batched_tokens=CONTEXT,
        enable_prefix_caching=False, enable_chunked_prefill=False, mamba_cache_mode="none", speculative_config=None,
        worker_extension_cls="qwen38_locomo_prefetch_worker.LoCoMoPrefetchAuditWorker",
        offload_backend="prefetch", offload_group_size=3, offload_num_in_group=1,
        offload_prefetch_step=1, offload_params=["experts"])
    return kwargs



def offload_execution(config):
    offload = config.offload_config
    result = {"backend": offload.offload_backend, "group_size": offload.prefetch.offload_group_size,
        "num_in_group": offload.prefetch.offload_num_in_group, "prefetch_step": offload.prefetch.offload_prefetch_step,
        "params": sorted(offload.prefetch.offload_params), "uva_cpu_offload_gb": offload.uva.cpu_offload_gb,
        "quantization": config.model_config.quantization, "model_dtype": str(config.model_config.dtype)}
    require(result == {"backend": "prefetch", "group_size": 3, "num_in_group": 1, "prefetch_step": 1,
        "params": ["experts"], "uva_cpu_offload_gb": 0, "quantization": None, "model_dtype": "torch.bfloat16"},
        "resolved native offload or base precision differs")
    return result


def cpu_offload_preflight(package):
    from vllm.config.offload import OffloadConfig, PrefetchOffloadConfig
    from vllm.engine.arg_utils import EngineArgs
    import inspect
    needed = {"offload_backend", "offload_group_size", "offload_num_in_group", "offload_prefetch_step", "offload_params"}
    require(needed <= set(inspect.signature(EngineArgs).parameters), "public native prefetch arguments unavailable")
    offload = OffloadConfig(offload_backend="prefetch", prefetch=PrefetchOffloadConfig(
        offload_group_size=3, offload_num_in_group=1, offload_prefetch_step=1, offload_params={"experts"}))
    require(offload.prefetch.offload_params == {"experts"} and offload.uva.cpu_offload_gb == 0, "CPU offload config differs")
    expected = {
        "config/offload.py": "6f2db3d9a2730588473d5155811690c6241b167a21dcb62fbbd9ebb6386de42c",
        "model_executor/models/utils.py": "570a2eb26c7274c9790b6ff42e3fd30403ad6534c68d256f4d867d22e2924919",
        "model_executor/offloader/base.py": "5157a59232715e7247761588efb88fc44e14970b580722a1f0b2e8b3a23a10fe",
        "model_executor/offloader/prefetch.py": "a64581cbb6cce0eaf3b9cfeacb216c17697a355fc6221579fc106f6546629792",
        "models/qwen4_exp/nvidia/model.py": "5c5d4bbe41222c846f50faf9075f45eebb84880e45eac835634948928bd260f1"}
    for name, sha in expected.items():
        require(digest(Path(package) / name) == sha, "audited native prefetch source changed: " + name)
    return expected


def resume_rows(output, contract, questions):
    path = Path(output) / "checkpoint.json"
    if not path.exists():
        require(not (Path(output) / "predictions.jsonl").exists(), "prediction output has lost its resume checkpoint")
        return []
    saved = read_json(path)
    require(saved["contract_sha256"] == canonical(contract), "resume contract differs")
    rows = saved["rows"]
    require(len(rows) <= len(questions), "resume has extra question rows")
    for row, question in zip(rows, questions):
        require(row["qid"] == question["qid"] and row["sample_id"] == question["sample_id"]
            and row["condition"] == "full_context" and row["prompt_sha256"] == question["full_context_prompt_sha256"]
            and row["prompt_tokens"] == question["full_context_prompt_tokens"] and isinstance(row["text"], str)
            and row["contract_sha256"] == canonical(contract)
            and canonical(row["generated_token_ids"]) == row["generated_token_ids_sha256"], "resume row identity changed")
    return rows


def generation_row(response, question, ids, tokenizer, seconds, contract_hash):
    require(response.finished and list(response.prompt_token_ids) == ids and len(response.outputs) == 1,
        "generation output does not match the full requested prompt")
    result = response.outputs[0]
    generated = list(result.token_ids)
    require(0 < len(generated) <= NEW_TOKENS and all(type(token) is int and 0 <= token < 248320 for token in generated),
        "invalid continuation tokens")
    return {"qid": question["qid"], "sample_id": question["sample_id"], "condition": "full_context",
        "prompt_sha256": question["full_context_prompt_sha256"], "prompt_tokens": len(ids),
        "source_memory_sample_id": None, "state_fingerprint_before": None, "state_fingerprint_after": None,
        "Delta_enabled": False, "memory_updates": 0, "generated_token_ids": generated,
        "generated_token_ids_sha256": canonical(generated), "text": tokenizer.decode(generated, skip_special_tokens=True),
        "vllm_text": result.text, "stop_reason": result.stop_reason, "finish_reason": result.finish_reason,
        "max_token_limit_reached": result.finish_reason == "length" or len(generated) >= NEW_TOKENS,
        "elapsed_seconds": seconds, "contract_sha256": contract_hash}


def resources(llm, *, initialize=False, reset=False):
    deadline = time.monotonic() + 30
    while True:
        left = deadline - time.monotonic()
        require(left > 0, "scheduler did not release the completed request")
        idle = llm.collective_rpc("locomo_base_status", args=(), timeout=left)
        require(len(idle) == 8 and sorted(row["rank"] for row in idle) == list(range(8)), "missing TP idle report")
        if all(row["quiescent"] for row in idle):
            break
        time.sleep(.05)
    reports = llm.collective_rpc("locomo_base_initialize" if initialize else "locomo_base_resources",
        args=() if initialize else (reset,), timeout=120)
    require(len(reports) == 8 and sorted(row["rank"] for row in reports) == list(range(8)), "missing TP resource report")
    require(all(row["kv_capacity_tokens"] >= CONTEXT and row["kv_max_concurrency"] >= 1
        and row["base_parameters_unchanged"] and not row["Delta_enabled"]
        and row["canonical_cpu_weights_unchanged"] and row["offload_module_count"] == 16
        and row["offload_net_saved_bytes"] > 0 and row["offload_host_dtypes"] == ["torch.bfloat16"] for row in reports),
        "base/cache capacity contract failed")
    expected_worker = digest(Path(__file__).with_name("qwen38_locomo_prefetch_worker.py"))
    require(all(row["worker_sha256"] == expected_worker for row in reports), "resource worker source changed")
    return reports


def gate(llm, params, tokenizer, questions, prefixes, contract_hash):
    longest = max(questions, key=lambda row: row["full_context_prompt_tokens"])
    pilot_longest = max((row for row in questions if row["sample_id"] == PILOT), key=lambda row: row["full_context_prompt_tokens"])
    observations = []
    for question in (longest, pilot_longest, longest):
        ids = prompt_ids(question, prefixes)
        started = time.perf_counter()
        response, = llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)
        observations.append(generation_row(response, question, ids, tokenizer, time.perf_counter() - started, contract_hash))
    require(observations[0]["generated_token_ids"] == observations[2]["generated_token_ids"],
        "same longest query changed after an unrelated request")
    return {"status": "passed", "max_prompt_tokens": len(prompt_ids(longest, prefixes)),
        "longest_admitted_without_truncation": True, "repeat_same_tokens_after_other_prompt": True,
        "observations": observations, "resources_after": resources(llm)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "model", "delta-repo", "base-verification", "vllm-fork-manifest", "reference-binding", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("gate", "pilot", "formal"), default="pilot")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()
    no_delta_environment()
    sys.path.insert(0, str(options.delta_repo / "src"))
    sys.path.insert(0, str(options.delta_repo / "scripts/native_qwen38"))
    import torch
    import qwen38_vllm_accuracy as accuracy
    from qwen38_native_runtime_preflight import verified_model
    from qwen38_vllm_delta_accuracy import verify_fork
    from delta_ngram import qwen38_vllm_live as live
    fork_manifest = read_json(options.vllm_fork_manifest)
    cpu_fork = verify_fork(options.vllm_fork_manifest, Path(fork_manifest["package_root"]) / "__init__.py")
    require(fork_manifest.get("protocol") == live.PROTOCOL and not live.split_enabled()
        and fork_manifest["live_helper"]["sha256"] == digest(live.__file__)
        and Path(fork_manifest["live_helper"]["path"]).resolve() == Path(live.__file__).resolve(), "base live-helper provenance differs")
    protocol, questions, prefixes, stats = load_population(options.prepared)
    identity = accuracy.hf_model_identity(options.model)
    identity["base_hash_audit"] = verified_model(options.model, options.base_verification)
    require(identity["tokenizer_files"] == protocol["tokenizer_files"], "LoCoMo tokenizer differs from base")
    reference = read_json(options.reference_binding)
    for key in ("config_sha256", "index_sha256", "tokenizer_files", "weight_stat_inventory_sha256"):
        require(reference["base_model"][key] == identity[key], "base differs from accepted A800 reader")
    eos = read_json(options.model / "generation_config.json")["eos_token_id"]
    require(sorted(eos) == sorted(protocol["generation"]["stop_token_ids"]), "generation EOS differs")
    sources = {str(path.resolve()): digest(path) for path in (Path(__file__), Path(__file__).with_name("qwen38_locomo_prefetch_worker.py"),
        Path(accuracy.__file__), Path(accuracy.locked.__file__), Path(live.__file__),
        Path(accuracy.__file__).with_name("qwen38_vllm_delta_accuracy.py"),
        options.delta_repo / "scripts/native_qwen38/qwen38_native_runtime_preflight.py")}
    contract = {"protocol": PROTOCOL, "condition": "full_context", "model_identity": identity,
        "prepared_manifest_sha256": digest(options.prepared / "manifest.json"), "protocol_sha256": REQUIRED_PROTOCOL_SHA,
        "fork_manifest_sha256": digest(options.vllm_fork_manifest), "reference_binding_sha256": digest(options.reference_binding),
        "source_sha256": sources, "statistics": stats, "requested_engine_settings": engine_settings(str(options.model.resolve())),
        "native_offload_sources": cpu_offload_preflight(fork_manifest["package_root"]),
        "fork": cpu_fork, "runtime_versions": {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")},
        "generation": protocol["generation"], "question_order": [row["qid"] for row in questions],
        "prompt_bindings": [{"qid": row["qid"], "sha256": row["full_context_prompt_sha256"],
            "tokens": row["full_context_prompt_tokens"]} for row in questions],
        "scores_read_by_generator": False, "Delta_enabled": False, "optimizer_steps": 0, "memory_updates": 0}
    require(not torch.cuda.is_initialized(), "CPU preflight initialized CUDA")
    if options.validate_only:
        print(json.dumps({"status": "CPU_preflight_passed", "contract_sha256": canonical(contract), **stats,
            "source_sha256": sources, "native_prefetch_CPU_verified": True, "cuda_initialized": False, "GPU_fit_verified": False})); return
    options.output.mkdir(parents=True, exist_ok=options.resume)
    contract_path = options.output / "contract.json"
    if options.resume:
        require(read_json(contract_path) == contract, "resume contract changed")
    else:
        require(not contract_path.exists(), "output contract already exists")
        write_json(contract_path, contract)
    rows = resume_rows(options.output, contract, questions)
    os.environ.update(VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_ENABLE_V1_MULTIPROCESSING="1",
        VLLM_USE_V2_MODEL_RUNNER="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    import vllm
    import vllm.envs as envs
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from transformers import AutoTokenizer
    fork = verify_fork(options.vllm_fork_manifest, vllm.__file__)
    kwargs = copy.deepcopy(contract["requested_engine_settings"])
    accuracy.configure_ple_cpu_offload(kwargs, EngineArgs, envs, requested=True)
    planned = EngineArgs(**copy.deepcopy(kwargs)).create_engine_config()
    require(planned.use_v2_model_runner and planned.model_config.max_model_len == CONTEXT
        and not planned.cache_config.enable_prefix_caching and not planned.scheduler_config.enable_chunked_prefill
        and planned.speculative_config is None, "resolved execution policy differs")
    offload_resolved = offload_execution(planned)
    execution = accuracy.execution_config(kwargs, planned, envs, ple_cpu_offload_requested=True)
    status = {"status": "loading", "contract_sha256": canonical(contract), "mode": options.mode, "fork": fork,
        "execution": execution, "native_offload_execution": offload_resolved, "generated_questions": len(rows), "total_questions": 1986,
        "memory_updates": 0, "optimizer_steps": 0, "backward_calls": 0}
    write_json(options.output / "STATUS.json", status)
    try:
        started = time.perf_counter()
        llm = LLM(**copy.deepcopy(kwargs))
        status["model_load_seconds"] = time.perf_counter() - started
        require(accuracy.execution_config(kwargs, llm.llm_engine.vllm_config, envs, ple_cpu_offload_requested=True) == execution,
            "actual engine settings differ")
        require(offload_execution(llm.llm_engine.vllm_config) == offload_resolved, "actual native offload differs")
        status["initial_resources"] = resources(llm, initialize=True)
        tokenizer = AutoTokenizer.from_pretrained(str(options.model), local_files_only=True)
        params = SamplingParams(n=1, temperature=0., top_p=1., top_k=-1, seed=17, max_tokens=NEW_TOKENS,
            stop_token_ids=eos, ignore_eos=False, skip_special_tokens=True, include_stop_str_in_output=False)
        gate_path = options.output / "gate.json"
        if not gate_path.exists():
            write_json(gate_path, gate(llm, params, tokenizer, questions, prefixes, canonical(contract)))
            write_json(options.output / "gate_receipt.json", {"sha256": digest(gate_path), "contract_sha256": canonical(contract)})
        else:
            require(options.resume and read_json(options.output / "gate_receipt.json") == {
                "sha256": digest(gate_path), "contract_sha256": canonical(contract)} and read_json(gate_path)["status"] == "passed",
                "prior gate changed")
        status.update(status="gate_passed", gate_sha256=digest(gate_path))
        write_json(options.output / "STATUS.json", status)
        if options.mode == "gate":
            return
        limit = 239 if options.mode == "pilot" else 1986
        require(len(rows) <= limit, "pilot mode cannot hide existing formal outputs")
        resources(llm, reset=True)
        for question in questions[len(rows):limit]:
            ids = prompt_ids(question, prefixes)
            started = time.perf_counter()
            response, = llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)
            row = generation_row(response, question, ids, tokenizer, time.perf_counter() - started, canonical(contract))
            rows.append(row)
            write_json(options.output / "checkpoint.json", {"contract_sha256": canonical(contract), "rows": rows})
            status.update(status="generating", generated_questions=len(rows), last_qid=row["qid"])
            write_json(options.output / "STATUS.json", status)
            print(json.dumps({"qid": row["qid"], "generated": len(rows), "total": limit,
                "prompt_tokens": len(ids), "new_tokens": len(row["generated_token_ids"]), "seconds": row["elapsed_seconds"]}), flush=True)
            if len(rows) == 239 or len(rows) == limit or questions[len(rows)]["sample_id"] != row["sample_id"]:
                resource_path = options.output / f"resources_after_{len(rows):04d}.json"
                write_json(resource_path, {"after_questions": len(rows), "workers": resources(llm)})
        predictions = options.output / "predictions.jsonl"
        predictions.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
        status.update(status="completed" if limit == 1986 else "pilot_completed", generated_questions=len(rows),
            predictions_sha256=digest(predictions), generated_tokens=sum(len(row["generated_token_ids"]) for row in rows),
            generation_seconds=sum(row["elapsed_seconds"] for row in rows),
            max_token_limit_hits=sum(row["max_token_limit_reached"] for row in rows), final_resources=resources(llm))
        write_json(options.output / ("COMPLETE.json" if limit == 1986 else "PILOT_COMPLETE.json"), status)
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(options.output / "STATUS.json", status)


if __name__ == "__main__":
    main()

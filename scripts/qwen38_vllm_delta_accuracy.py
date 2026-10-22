"""Fixed-FP32-delta generation using an isolated, pinned vLLM fork.

One process owns one immutable task-boundary snapshot. Prefix sharing is off;
request-local attention/GDN/PLE caches remain enabled. The old accuracy runner
supplies the locked populations, prompt tokens, stop rules, and trace format.
The optional cache check compares cached decoding with fresh one-token requests
on each complete generated prefix, in the same patched engine.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen38_vllm_accuracy as original


def digest(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def runtime_contract(identity, execution):
    """Compare serving code/configuration independently of the fixed snapshot."""
    return {"model_identity": {key: value for key, value in identity.items()
                               if key not in {"delta_snapshot_sha256", "delta_binding_sha256"}},
            "execution": execution}


def prepare_binding(snapshot, contract, output, model_identity):
    """Validate the separately registered identity before a CUDA engine exists."""
    from delta_ngram.qwen38_native import Qwen38PLELayout
    from delta_ngram.qwen38_vllm_snapshot import load_vllm_snapshot

    identity = contract["expected_snapshot_identity"]
    expected_base = identity.get("base_model")
    if not isinstance(expected_base, dict) or any(
        expected_base.get(key) != model_identity[key]
        for key in ("config_sha256", "index_sha256", "verification_sha256")
    ):
        raise ValueError("snapshot base-model identity differs from the loaded HF model")
    if identity.get("tokenizer") != model_identity["tokenizer_files"]:
        raise ValueError("snapshot tokenizer identity differs from the loaded HF model")
    layout_data = dict(contract["expected_layout"])
    for key in ("head_offsets", "head_vocab_sizes", "orders"):
        if key in layout_data:
            layout_data[key] = tuple(layout_data[key])
    layout = Qwen38PLELayout(**layout_data)
    if contract.get("owner_id") != 0 or type(contract.get("owner_id")) is not int:
        raise ValueError("one isolated engine requires the registered owner 0")
    load_vllm_snapshot(
        snapshot, expected_sha256=contract["snapshot_sha256"],
        expected_identity=identity, expected_layout=layout,
        expected_owner_id=0, device="cpu",
    )
    binding = {
        "schema_version": 1, "snapshot_path": str(Path(snapshot).resolve()),
        "snapshot_sha256": contract["snapshot_sha256"], "identity": identity,
        "layout": asdict(layout), "owner_id": 0,
    }
    path = Path(output) / "delta_binding.json"
    text = json.dumps(binding, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError("output is already bound to a different snapshot; use a new directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    return binding, path, digest(path)


def verify_fork(manifest_path, package_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = original.read_json(manifest_path)
    root = Path(manifest["package_root"]).resolve()
    if Path(package_path).resolve().parent != root:
        raise ValueError("imported vLLM is not the registered isolated fork")
    if manifest.get("upstream_version") != "0.30.0":
        raise ValueError("this integration is audited against vLLM 0.30.0")
    patched = manifest.get("patched_files", [])
    if not patched:
        raise ValueError("fork manifest has no patched source files")
    for relative in patched:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or digest(path) != manifest["files"][relative]["sha256"]:
            raise ValueError(f"fork source changed: {relative}")
    helpers = {}
    for key, module_name in (("helper", "delta_ngram.qwen38_vllm_ple"),
                             ("snapshot_helper", "delta_ngram.qwen38_vllm_snapshot")):
        record = manifest[key]
        imported_path = Path(importlib.import_module(module_name).__file__).resolve()
        if imported_path != Path(record["path"]).resolve() or digest(imported_path) != record["sha256"]:
            raise ValueError(f"fork {key} changed")
        helpers[key] = record["sha256"]
    return {"manifest_sha256": digest(manifest_path), "package_root": str(root),
            "upstream_version": manifest["upstream_version"], "helpers": helpers,
            "patched_files": {name: manifest["files"][name]["sha256"] for name in patched}}


def inspect_model_memory(model):
    """Small RPC result; runtime_identity also checks immutable tensor guards."""
    from delta_ngram.qwen38_vllm_ple import runtime_identity
    found = [runtime_identity(module) for module in model.modules()
             if type(module).__name__ == "Qwen4ExpPLELayer"]
    if len(found) != 1:
        raise ValueError(f"expected one replicated PLE per worker, got {len(found)}")
    return found[0]


def inspect_engine(llm, binding, binding_sha256, tensor_parallel_size, expected_helpers=None,
                   expected_worker_extension=None):
    records = llm.collective_rpc("delta_ngram_memory_identity")
    if len(records) != tensor_parallel_size:
        raise ValueError("not every registered TP worker reported its memory binding")
    for record in records:
        if (record.get("enabled") is not True
                or record.get("binding_sha256") != binding_sha256
                or record.get("snapshot_sha256") != binding["snapshot_sha256"]
                or record.get("cache_dtype") not in ("torch.float32", "float32")):
            raise ValueError(f"worker did not load the fixed FP32 PLE snapshot: {record}")
        if json.dumps(record.get("snapshot_identity"), sort_keys=True) != json.dumps(binding["identity"], sort_keys=True):
            raise ValueError("worker snapshot identity differs from the registered binding")
        if expected_helpers is not None:
            for key, actual_key in (("helper", "helper_source"), ("snapshot_helper", "snapshot_helper_source")):
                if record.get(actual_key, {}).get("sha256") != expected_helpers[key]:
                    raise ValueError(f"worker imported a different {key}")
        if (expected_worker_extension is not None
                and record.get("worker_extension_source", {}).get("sha256") != expected_worker_extension):
            raise ValueError("worker imported a different read-only inspection extension")
    return records


def next_token_from_fresh_request(llm, params_cls, prefix):
    params = params_cls(n=1, temperature=0., top_p=1., top_k=-1, seed=17,
                        max_tokens=1, ignore_eos=True, stop_token_ids=[], detokenize=False)
    result = llm.generate([{"prompt_token_ids": list(prefix)}], params, use_tqdm=False)
    if (len(result) != 1 or not result[0].finished
            or list(result[0].prompt_token_ids) != list(prefix)
            or len(result[0].outputs) != 1
            or len(result[0].outputs[0].token_ids) != 1):
        raise ValueError("fresh full-prefix reference did not return exactly one token")
    return int(result[0].outputs[0].token_ids[0])


def cached_batch(llm, params_cls, records, eos_ids, max_new_tokens):
    params = params_cls(n=1, temperature=0., top_p=1., top_k=-1, seed=17,
                        max_tokens=max_new_tokens, ignore_eos=False,
                        stop_token_ids=eos_ids, skip_special_tokens=True)
    started = time.perf_counter()
    results = llm.generate([{"prompt_token_ids": row["prompt_token_ids"]} for row in records],
                           params, use_tqdm=False)
    seconds = time.perf_counter() - started
    if len(results) != len(records):
        raise ValueError("cached batch response count differs")
    outputs = []
    for request, row in zip(results, records):
        if (not request.finished or list(request.prompt_token_ids) != row["prompt_token_ids"]
                or len(request.outputs) != 1):
            raise ValueError("cached batch response identity differs")
        completion = request.outputs[0]
        outputs.append({"id": row["id"], "generated_token_ids": list(completion.token_ids),
                        "text": completion.text, "finish_reason": completion.finish_reason,
                        "stop_reason": completion.stop_reason})
    return outputs, seconds


def check_cache(llm, params_cls, tokenizer, records, eos_ids, *, max_new_tokens, max_context_tokens):
    """Same-stack parity plus request reordering, without reference-answer input."""
    if max_new_tokens < 2:
        raise ValueError("cache validation requires at least two generation steps")
    if not records or len({row["id"] for row in records}) != len(records):
        raise ValueError("cache-check records require unique IDs")
    for row in records:
        original.check_token_ids(row["prompt_token_ids"])
        if len(row["prompt_token_ids"]) + max_new_tokens > max_context_tokens:
            raise ValueError("cache-check prompt exceeds the registered context; no truncation")
    cached, cached_seconds = cached_batch(llm, params_cls, records, eos_ids, max_new_tokens)
    comparisons, reference_seconds = [], 0.
    for row, candidate in zip(records, cached):
        reference = original.locked.greedy_decode(
            row["prompt_token_ids"],
            forward_next=lambda prefix: next_token_from_fresh_request(llm, params_cls, prefix),
            decode=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
            eos_token_ids=eos_ids, max_new_tokens=max_new_tokens,
            max_context_tokens=max_context_tokens,
        )
        reference_seconds += reference["elapsed_seconds"]
        same = candidate["generated_token_ids"] == reference["generated_token_ids"]
        comparisons.append({"id": row["id"], "tokens_identical": same,
                            "cached": candidate, "full_prefix": reference})
        print(json.dumps({"event": "cache_parity", "id": row["id"],
                          "tokens_identical": same,
                          "generated_tokens": len(candidate["generated_token_ids"])}), flush=True)
    reordered, reordered_seconds = cached_batch(llm, params_cls, list(reversed(records)),
                                                eos_ids, max_new_tokens)
    first = {row["id"]: row["generated_token_ids"] for row in cached}
    order_equal = all(first[row["id"]] == row["generated_token_ids"] for row in reordered)
    # Repeat after unrelated requests. Physical slot reuse is not inferred here.
    repeated, repeat_seconds = cached_batch(llm, params_cls, records[:1], eos_ids, max_new_tokens)
    repeat_equal = repeated[0]["generated_token_ids"] == cached[0]["generated_token_ids"]
    decode_exercised = any(len(row["generated_token_ids"]) > 1 for row in cached)
    passed = (all(row["tokens_identical"] for row in comparisons) and order_equal
              and repeat_equal and decode_exercised)
    return {"status": "passed" if passed else "failed", "comparisons": comparisons,
            "reordered_batch": reordered, "reordered_tokens_identical": order_equal,
            "repeat_first_request": repeated[0], "repeat_tokens_identical": repeat_equal,
            "decode_cache_exercised": decode_exercised, "physical_cache_slot_reuse_verified": False,
            "prefix_sharing": False, "request_local_cache": True,
            "reference": "fresh request for each complete generated prefix, max_tokens=1",
            "timings_seconds": {"cached_batch": cached_seconds, "full_prefix_reference": reference_seconds,
                                "reordered_batch": reordered_seconds, "repeat_first": repeat_seconds},
            "timing_scope": "small validation cohort; not a full-suite throughput estimate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-verification", type=Path, required=True,
                        help="Completed full-weight SHA audit receipt with current shard signatures")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--snapshot-contract", type=Path, required=True)
    parser.add_argument("--vllm-fork-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.90)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--cache-check-prompts", type=Path)
    parser.add_argument("--cache-check-max-new-tokens", type=int, default=8)
    parser.add_argument("--task-data", type=Path)
    parser.add_argument("--task-prompt-tokens", type=Path)
    parser.add_argument("--task-probe-tasks", nargs="+")
    parser.add_argument("--method", choices=("writer", "delta_writer"))
    parser.add_argument("--stage", type=int)
    parser.add_argument("--runtime-contract", type=Path,
                        help="Require this accepted serving identity before loading the GPU engine")
    args = parser.parse_args()
    if (args.tensor_parallel_size < 1 or not 0 < args.gpu_memory_utilization < 1
            or min(args.max_model_len, args.max_num_seqs, args.max_num_batched_tokens,
                   args.cache_check_max_new_tokens) < 1):
        parser.error("positive engine limits and GPU memory utilization in (0,1) required")
    if not args.cache_check_prompts and (
        args.task_data is None or args.task_prompt_tokens is None or args.method is None
        or args.stage is None or not 0 <= args.stage <= 7
    ):
        parser.error("accuracy requires task data/prompt tokens, method, and stage 0..7")
    args.output.mkdir(parents=True, exist_ok=True)
    identity = original.hf_model_identity(args.model)
    from qwen38_native_runtime_preflight import verified_model
    verified_base = verified_model(args.model, args.base_verification)
    identity["verification_sha256"] = verified_base["verification_sha256"]
    identity["base_hash_audit"] = verified_base
    contract = original.read_json(args.snapshot_contract)
    if not args.cache_check_prompts:
        trajectory = contract["expected_snapshot_identity"].get("trajectory")
        if (not isinstance(trajectory, dict) or trajectory.get("method") != args.method
                or type(trajectory.get("stage")) is not int or trajectory["stage"] != args.stage):
            raise ValueError("requested method/stage differs from the registered snapshot trajectory")
    binding, binding_path, binding_hash = prepare_binding(args.snapshot, contract, args.output, identity)
    os.environ["DELTA_NGRAM_VLLM_BINDING"] = str(binding_path.resolve())
    os.environ["DELTA_NGRAM_VLLM_BINDING_SHA256"] = binding_hash
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    import vllm
    import vllm.envs as vllm_envs
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from transformers import AutoTokenizer
    identity["vllm_fork"] = verify_fork(args.vllm_fork_manifest, vllm.__file__)
    identity["delta_snapshot_sha256"] = binding["snapshot_sha256"]
    identity["delta_binding_sha256"] = binding_hash
    identity["runtime_versions"] = {name: importlib.metadata.version(name)
                                    for name in ("vllm", "torch", "transformers")}
    identity["generator_sha256"] = digest(__file__)
    identity["worker_extension_sha256"] = digest(Path(__file__).with_name("qwen38_vllm_delta_worker.py"))
    identity["runtime_source_sha256"] = {name: digest(Path(__file__).with_name(name)) for name in (
        "qwen38_vllm_delta_accuracy.py", "qwen38_vllm_delta_worker.py", "qwen38_vllm_accuracy.py",
        "qwen38_greedy_decode.py", "qwen38_native_runtime_preflight.py")}
    kwargs = original.engine_kwargs(args.model.resolve(), enforce_eager=True,
                                     gpu_memory_utilization=args.gpu_memory_utilization,
                                     tensor_parallel_size=args.tensor_parallel_size, moe_backend="triton")
    kwargs.update(max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                  max_num_batched_tokens=args.max_num_batched_tokens,
                  enable_prefix_caching=False, mamba_cache_mode="none", speculative_config=None,
                  worker_extension_cls="qwen38_vllm_delta_worker.DeltaNGramWorkerExtension")
    original.configure_ple_cpu_offload(kwargs, EngineArgs, vllm_envs, requested=True)
    planned = EngineArgs(**copy.deepcopy(kwargs)).create_engine_config()
    if planned.cache_config.enable_prefix_caching:
        raise ValueError("fixed-delta first implementation disables cross-request prefix sharing")
    execution = original.execution_config(kwargs, planned, vllm_envs, ple_cpu_offload_requested=True)
    if args.runtime_contract and original.read_json(args.runtime_contract) != runtime_contract(identity, execution):
        raise ValueError("generation runtime differs from the accepted code/model/execution contract")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    generation = original.read_json(args.model / "generation_config.json")
    eos = generation.get("eos_token_id", tokenizer.eos_token_id)
    eos_ids = [eos] if isinstance(eos, int) else list(eos or [])
    prepared = []
    if not args.cache_check_prompts:
        suite, = original.load_suites(None, args.task_data, ["tasks"], args.task_probe_tasks)
        tokens, token_hash = original.load_task_tokens(args.task_prompt_tokens, suite, args.model)
        encoded = original.encode_suite(suite, tokenizer, task_tokens=tokens)
        if any(len(ids) + suite["protocol"]["max_new_tokens"] > args.max_model_len for ids in encoded):
            raise ValueError("registered task prompt exceeds engine context; no truncation")
        run_contract = original.make_contract(
            suite, encoded, identity, eos_ids=eos_ids, method=args.method, stage=args.stage,
            source_checkpoint=str(args.snapshot.resolve()), source_checkpoint_sha256=binding["snapshot_sha256"],
            task_token_sha256=token_hash, execution=execution)
        run_contract["protocol"].update(
            checkpoint_state="immutable_fp32_delta_overlay", max_model_len=args.max_model_len,
            batch_size=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens, max_num_seqs=args.max_num_seqs,
            enable_prefix_caching=False, request_local_cache=True,
            delta_binding_sha256=binding_hash)
        existing = original.load_resume(args.output / "tasks", run_contract)
        prepared.append((suite, encoded, run_contract, existing))
    status = {"status": "loading", "model_identity": identity, "binding": binding,
              "execution": execution, "started_utc": original.utc_now(), "optimizer_steps": 0}
    status_path = args.output / "runtime_status.json"
    original.locked._write_json(status_path, status)
    started = time.perf_counter()
    try:
        llm = LLM(**copy.deepcopy(kwargs))
        status["load_seconds"] = time.perf_counter() - started
        actual = original.execution_config(kwargs, llm.llm_engine.vllm_config, vllm_envs,
                                            ple_cpu_offload_requested=True)
        if actual != execution or llm.llm_engine.vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("live engine differs from the registered execution")
        status["workers_before"] = inspect_engine(llm, binding, binding_hash, args.tensor_parallel_size,
                                                  identity["vllm_fork"]["helpers"],
                                                  identity["worker_extension_sha256"])
        status["status"] = "evaluating"
        original.locked._write_json(status_path, status)
        if args.cache_check_prompts:
            prompts = original.read_json(args.cache_check_prompts)
            report = check_cache(llm, SamplingParams, tokenizer, prompts["records"], eos_ids,
                                 max_new_tokens=args.cache_check_max_new_tokens,
                                 max_context_tokens=args.max_model_len)
            report.update(model_identity=identity, binding_sha256=binding_hash,
                          prompt_manifest_sha256=digest(args.cache_check_prompts), execution=execution)
            original.locked._write_json(args.output / "cache_check.json", report)
            status["cache_check_status"] = report["status"]
            if report["status"] != "passed":
                raise ValueError("cached decoding differs from the fresh full-prefix reference")
        else:
            for suite, encoded, run_contract, existing in prepared:
                original.run_suite(llm, SamplingParams, tokenizer, suite, encoded, args.output / "tasks",
                                   run_contract, existing=existing)
        status["workers_after"] = inspect_engine(llm, binding, binding_hash, args.tensor_parallel_size,
                                                 identity["vllm_fork"]["helpers"],
                                                 identity["worker_extension_sha256"])
        status["status"] = "passed"
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status.update(finished_utc=original.utc_now(), total_seconds=time.perf_counter() - started)
        original.locked._write_json(status_path, status)


if __name__ == "__main__":
    main()

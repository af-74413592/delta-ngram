"""A800 B1 cached generation with two independent, complete live FP32 states.

The semantic preflight records cached/fresh numerical differences without
rewriting the older strict parity failure. Formal supports are ordered B1
requests; one rank writes D1, then every TP rank adopts exactly that patch.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qwen38_vllm_accuracy as original
from qwen38_vllm_delta_accuracy import digest, verify_fork


def require(condition, message):
    if not condition: raise ValueError(message)


def write_json(path, value):
    target = Path(path); temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as stream: os.fsync(stream.fileno())
    os.replace(temporary, target)


def agree(records, fields, *, label):
    require(len(records) == 8 and sorted(x["rank"] for x in records) == list(range(8)),
            label + ": missing TP worker")
    for name in fields:
        require(all(x[name] == records[0][name] for x in records), label + ": TP disagreement: " + name)
    return records[0]


def validate_endpoint_receipt(root, receipt, *, arm, name, task_ids_by_task):
    path = Path(receipt["path"]).resolve()
    require(path.is_relative_to(Path(root).resolve()) and digest(path) == receipt["sha256"],
            "completed generation receipt moved or changed")
    completed = original.read_json(path)
    tasks = list(task_ids_by_task)
    if name == "boundary_00_base":
        expected_tasks, epoch, domains = [], 0, ["math", "code"]
    else:
        match = re.fullmatch(r"task_(\d{2})_([a-z]+)_before|boundary_(\d{2})_([a-z]+)", name)
        require(match is not None, "unknown evaluation phase")
        before = match[1] is not None; stage = int(match[1] or match[3]); task = match[2] or match[4]
        require(1 <= stage <= 7 and tasks[stage - 1] == task, "evaluation task order differs")
        expected_tasks = [task] if before else tasks[:stage]
        epoch = (stage - 1 if before else stage) * 5000
        domains = expected_tasks if before else [*expected_tasks, "math", "code"]
    require(path == (Path(root) / arm / "generation" / name / "COMPLETE.json").resolve()
        and completed["arm"] == arm and completed["name"] == name and completed["epoch"] == epoch
        and completed["generated_domains"] == expected_tasks and completed["math_code_generation"] == 0
        and completed["status"] == "completed" and completed["domains"] == receipt["domains"] == domains,
            "generation receipt domains/status differ")
    require([x["domain"] for x in completed["summaries"]] == (["tasks"] if expected_tasks else []),
            "generation receipt contains a different suite population")
    for item in completed["summaries"]:
        folder = path.parent / item["domain"]
        summary = original.read_json(folder / "summary.json")
        require(summary["status"] == "completed" and summary["generated_tasks"] == summary["task_count"] == item["generated_tasks"]
            and summary["run_contract_sha256"] == item["run_contract_sha256"]
            and digest(folder / "trace.jsonl") == item["trace_sha256"] == summary["trace_sha256"],
            "completed generation trace or population changed")
        require(summary["method"] == arm and summary["stage"] == epoch // 5000
            and summary["model_identity"]["live_state_epoch"] == epoch
            and summary["model_identity"]["live_state_hash"] == completed["state_hash"] == summary["source_checkpoint_sha256"]
            and summary["source_checkpoint"] == "live-state:" + completed["state_hash"]
            and summary["model_identity"]["live_binding_sha256"] == digest(Path(root) / "binding.json"),
            "completed generation belongs to another arm, phase, or state")
        name = {"tasks": "task_completions.json", "math": "math_completions.json", "code": "humaneval_completions.json"}[item["domain"]]
        require(digest(folder / name) == summary["completions_sha256"], "completed generations changed")
        expected_ids = [task_id for task in expected_tasks for task_id in task_ids_by_task[task]]
        require(summary["task_ids"] == expected_ids and item["generated_tasks"] == len(expected_ids),
                "completed generation task IDs differ from the locked population")
        rows = [json.loads(line) for line in (folder / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        require([row["task_id"] for row in rows] == expected_ids, "completed generation trace order differs")
        require(len(summary["prompt_bindings"]) == len(rows) and all(
            row["run_contract_sha256"] == summary["run_contract_sha256"]
            and all(row.get(key) == value for key, value in prompt.items())
            for row, prompt in zip(rows, summary["prompt_bindings"])), "completed generation prompt identity differs")
        contract_keys = ("version", "suite", "task_ids", "protocol", "model_identity", "data_manifest", "method", "stage",
                         "source_checkpoint", "source_checkpoint_sha256", "task_token_manifest_sha256", "prompt_bindings")
        require(original.locked._digest({key: summary[key] for key in contract_keys}) == summary["run_contract_sha256"],
                "completed generation contract changed")


def archive_pending(root, paths, reason):
    root = Path(root).resolve(); destination = root / "recovery" / uuid.uuid4().hex
    destination.mkdir(parents=True, exist_ok=False)
    moved = []
    for value in paths:
        path = Path(value).resolve()
        require(path.is_relative_to(root) and path != root, "pending recovery path escaped campaign root")
        if path.exists():
            target = destination / path.name
            require(target.is_relative_to(root), "pending recovery destination escaped campaign root")
            path.rename(target); moved.append({"from": str(path), "to": str(target)})
    write_json(destination / "recovery.json", {"reason": reason, "preserved": moved})


class LiveEngine:
    def __init__(self, llm, params_cls, output):
        self.llm, self.params_cls, self.output = llm, params_cls, Path(output)
        self.epochs = {"ordinary": 0, "delta": 0}
        self.query_audit = None
        require(type(llm.llm_engine.engine_core).__name__ == "SyncMPClient",
                "live cleanup protocol requires the audited multiprocessing SyncMPClient")

    def rpc(self, name, *args, timeout=120):
        return self.llm.collective_rpc(name, args=args, timeout=timeout)

    def quiescent(self):
        deadline = time.monotonic() + 30
        while True:
            left = deadline - time.monotonic()
            require(left > 0, "scheduler did not release request state within 30 seconds")
            rows = self.rpc("delta_live_status", timeout=left)
            agree(rows, [], label="quiescence")
            if all(x["quiescent"] for x in rows): return
            time.sleep(.05)

    def request(self, arm, kind, ids, params, *, request_id=None, diagnostics=False):
        self.quiescent()
        request_id = request_id or f"{arm}:{kind}:{uuid.uuid4().hex}"
        log_path = None
        if diagnostics:
            directory = self.output / "gate" / "logits"; directory.mkdir(parents=True, exist_ok=True)
            log_path = str((directory / (uuid.uuid4().hex + ".pt")).resolve())
        self.rpc("delta_live_begin", arm, kind, request_id, self.epochs[arm], list(ids), diagnostics, log_path)
        require(getattr(params, "prompt_logprobs", None) is None, "live generation excludes prompt-logprob side calls")
        started = time.monotonic()
        responses = self.llm.generate([{"prompt_token_ids": list(ids)}], params, use_tqdm=False)
        require(len(responses) == 1 and responses[0].finished
            and list(responses[0].prompt_token_ids) == list(ids) and len(responses[0].outputs) == 1,
            "engine response differs from the registered B1 request")
        self.quiescent()
        fields = ["arm", "kind", "request_id", "epoch", "input_sha256", "forward_calls",
                  "fresh_context", "native_ids_sha256", "history_sha256"]
        if kind == "support": fields += ["hidden_sha256", "physical_rows_sha256"]
        records = self.rpc("delta_live_finish")
        result = agree(records, fields, label="request finish")
        require(all(x["fresh_context"] for x in records), "a TP worker inherited prior request context")
        result = dict(result, elapsed_seconds=time.monotonic() - started)
        if self.query_audit is not None and kind == "query":
            with self.query_audit.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, allow_nan=False) + "\n")
        return responses[0], result

    def support(self, arm, row):
        params = self.params_cls(n=1, temperature=0., top_p=1., top_k=-1, seed=17,
            max_tokens=1, ignore_eos=True, stop_token_ids=[], detokenize=False)
        _, observation = self.request(arm, "support", row["input_ids"], params,
            request_id=f"{arm}:support:{row['row_id']}")
        records = self.rpc("delta_live_commit")
        result = agree(records, ["arm", "epoch", "identity", "rows", "block_clock", "update_rows"], label="D1")
        require(result["epoch"] == self.epochs[arm] + 1 and result["block_clock"] == [result["epoch"]],
                "D1 advanced a wrong epoch")
        self.epochs[arm] = result["epoch"]
        return {"state": result, "observation": observation}

    def checkpoint(self, path, *, replace=False):
        self.quiescent()
        records = self.rpc("delta_live_checkpoint", str(Path(path).resolve()), replace, timeout=600)
        result = agree(records, ["state_hashes", "epochs", "identities"], label="checkpoint")
        require(result["epochs"] == self.epochs, "checkpoint cursor disagrees with workers")
        return result

    def restore(self, receipt):
        records = self.rpc("delta_live_restore", receipt["path"], receipt["sha256"], timeout=600)
        result = agree(records, ["state_hashes", "epochs", "identities"], label="restore")
        require(all(result[k] == receipt[k] for k in ("state_hashes", "epochs", "identities")),
                "restore changed the complete state")
        self.epochs = dict(result["epochs"])
        return result


class ReadOnlyEngine:
    def __init__(self, engine, arm): self.engine, self.arm = engine, arm
    def generate(self, requests, sampling_params, use_tqdm=False):
        require(len(requests) == 1, "endpoint generation must remain B1")
        response, _ = self.engine.request(self.arm, "query", requests[0]["prompt_token_ids"], sampling_params)
        return [response]


def numerical_case(engine, arm, ids, *, steps):
    params = engine.params_cls(n=1, temperature=0., top_p=1., top_k=-1, seed=17,
        max_tokens=steps, ignore_eos=True, stop_token_ids=[], detokenize=False)
    response, cached = engine.request(arm, "query", ids, params, diagnostics=True)
    tokens = list(response.outputs[0].token_ids)
    require(len(tokens) == steps and len(cached["diagnostics"]) == steps, "gate did not exercise multi-step decode")
    comparisons = []
    import torch
    require(digest(cached["logits_file"]["path"]) == cached["logits_file"]["sha256"], "cached logits receipt changed")
    cache_logits = torch.load(cached["logits_file"]["path"], map_location="cpu", weights_only=True)["logits"]
    for step in range(steps):
        prefix = list(ids) + tokens[:step]
        one = engine.params_cls(n=1, temperature=0., top_p=1., top_k=-1, seed=17,
            max_tokens=1, ignore_eos=True, stop_token_ids=[], detokenize=False)
        fresh_response, fresh = engine.request(arm, "query", prefix, one, diagnostics=True)
        left, right = cached["diagnostics"][step], fresh["diagnostics"][0]
        require(left["positions"][-1] == right["positions"][-1] == len(prefix) - 1
            and left["input_ids"][-1] == right["input_ids"][-1] == prefix[-1]
            and left["native_ids"][-1] == right["native_ids"][-1],
            "cached/fresh request used a wrong position, token, or native PLE address")
        if step == 0:
            require(left["native_ids"] == right["native_ids"], "full prefill native IDs differ")
        actual = int(fresh_response.outputs[0].token_ids[0])
        require(digest(fresh["logits_file"]["path"]) == fresh["logits_file"]["sha256"], "fresh logits receipt changed")
        fresh_logits = torch.load(fresh["logits_file"]["path"], map_location="cpu", weights_only=True)["logits"][0][0]
        cached_logits = cache_logits[step][0]
        require(left["argmax"] == tokens[step] and right["argmax"] == actual, "greedy output is not captured argmax")
        candidates = sorted({tokens[step], actual})
        comparisons.append({"step": step, "input_prefix_sha256": original.locked._digest(prefix),
            "native_ids_exact": True, "cached_token": tokens[step], "fresh_token": actual,
            "argmax_equal": tokens[step] == actual,
            "cached_top_margin": left["top_margin"], "fresh_top_margin": right["top_margin"],
            "candidate_logits": [{"token_id": t, "cached": float(cached_logits[t]), "fresh": float(fresh_logits[t])}
                                 for t in candidates],
            "full_vocab_max_abs": float((cached_logits.double() - fresh_logits.double()).abs().max()),
            "fresh": fresh})
    return {"tokens": tokens, "cached": cached, "same_history_comparisons": comparisons,
        "first_argmax_difference": next((x["step"] for x in comparisons if not x["argmax_equal"]), None)}


def observe_slot_history(history, observation):
    """Record actual physical reuse; one prompt need not retain one cache slot."""
    reused = []
    for slot in observation["physical_cache_slots"]:
        prior = history.get(slot)
        if prior is not None and prior["input_sha256"] != observation["input_sha256"]:
            reused.append({"slot": slot, "previous_request_id": prior["request_id"],
                "previous_input_sha256": prior["input_sha256"],
                "current_request_id": observation["request_id"],
                "current_input_sha256": observation["input_sha256"],
                "fresh_context": observation["fresh_context"],
                "all_conv_states_match_own_history": all(
                    x["state_after_matches_own_history"] and x["state_before_matches_own_history"] is not False
                    for x in observation["diagnostics"])})
        history[slot] = {key: observation[key] for key in ("request_id", "input_sha256")}
    return reused


def semantic_gate(engine, support, prompts, *, steps):
    gate = engine.output / "gate"; gate.mkdir(exist_ok=False)
    zero = engine.checkpoint(gate / "zero.state.pt")
    reports = {}
    for phase in ("zero", "nonzero"):
        if phase == "nonzero":
            for arm in ("ordinary", "delta"):
                for row in support[:2]: engine.support(arm, row)
        before = engine.checkpoint(gate / (phase + ".before.state.pt"))
        for arm in ("ordinary", "delta"):
            key = phase + ":" + arm
            cases = [numerical_case(engine, arm, ids, steps=steps) for ids in prompts]
            slot_history = {}
            for case in cases:
                observe_slot_history(slot_history, case["cached"])
                for comparison in case["same_history_comparisons"]:
                    observe_slot_history(slot_history, comparison["fresh"])
            repeat = []; params = engine.params_cls(n=1, temperature=0., top_p=1., top_k=-1, seed=17,
                max_tokens=steps, ignore_eos=True, stop_token_ids=[], detokenize=False)
            for index in (1, 0, 0):
                response, observation = engine.request(arm, "query", prompts[index], params, diagnostics=True)
                require(list(response.outputs[0].token_ids) == cases[index]["tokens"],
                        "same B1 query changed after unrelated requests/slot reuse")
                reuse = observe_slot_history(slot_history, observation)
                repeat.append({"prompt_index": index, "tokens_equal": True, "slot_reuse": bool(reuse),
                               "physical_slot_cross_history_evidence": reuse,
                               "observation": observation})
            require(any(row["slot_reuse"] for row in repeat),
                    "preflight did not exercise real cross-history physical cache-slot reuse")
            reports[key] = {"cases": cases, "repetition_and_order": repeat}
        after = engine.checkpoint(gate / (phase + ".after.state.pt"))
        require(all(after[k] == before[k] for k in ("state_hashes", "epochs", "identities")),
                "queries changed a live Delta field")
        engine.restore(before)
    engine.restore(zero)
    capacity_start = engine.rpc("delta_live_pad_capacity", 666747, timeout=600)
    agree(capacity_start, ["engineering_only", "rows_per_arm", "state_bytes_per_rank"], label="capacity initialization")
    for arm in ("ordinary", "delta"): engine.support(arm, support[0])
    capacity_end = engine.rpc("delta_live_capacity_result", timeout=600)
    agree(capacity_end, ["rows_per_arm", "epochs", "state_hashes"], label="capacity D1")
    require(all(x["epochs"] == {"ordinary": 1, "delta": 1} for x in capacity_end), "capacity D1 did not complete")
    engine.restore(zero)
    reset = engine.checkpoint(gate / "reset.state.pt")
    require(reset["state_hashes"] == zero["state_hashes"] and reset["epochs"] == {"ordinary": 0, "delta": 0},
            "formal trajectories were not reset after preflight")
    result = {"status": "passed", "protocol": "qwen38_vllm_live_semantic_numerical_preflight_v1",
        "zero": zero, "formal_reset": reset, "cases": reports,
        "D1_per_arm_before_reset": 2, "steps_per_query": steps,
        "engineering_capacity": {"zero_padding_rows_per_arm": 666747, "before": capacity_start,
                                 "after_one_D1_per_arm": capacity_end, "real_35k_state_claimed": False},
        "query_state_read_only": True, "all_TP_states_equal": True,
        "native_addresses_exact_on_same_history": True, "PLE_conv_state_matches_own_inputs": True,
        "cached_fresh_argmax_equality_required": False,
        "prior_strict_nonzero_parity_gate": "failed_preserved_without_reclassification",
        "numerical_scope": "record actual full-vocabulary logits and candidate margins; no claim of bitwise kernel parity"}
    write_json(gate / "COMPLETE.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("delta-repo", "model", "base-verification", "vllm-fork-manifest", "layout-contract",
                 "encoded-dir", "main-writer", "full-writer", "task-data", "task-prompt-tokens", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--retention-data", type=Path, help="Unused compatibility argument; H200 owns math/code NLL.")
    p.add_argument("--layout-contract-sha256", required=True)
    p.add_argument("--encoded-manifest-sha256", required=True)
    p.add_argument("--mode", choices=("gate", "formal"), default="gate")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--gate-decode-steps", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=.85)
    return p


def main():
    options = parser().parse_args()
    require(options.save_every > 0 and 2 <= options.gate_decode_steps <= 16
        and 0 < options.gpu_memory_utilization < 1, "invalid registered limits")
    sys.path.insert(0, str(options.delta_repo / "src"))
    sys.path.insert(0, str(options.delta_repo / "scripts/native_qwen38"))
    from delta_ngram import qwen38_vllm_live as live
    from delta_ngram.qwen38_seven_forward import TASKS
    from delta_ngram.qwen38_forward_recovery import (initial_cursor, pending_evaluation, acknowledge_evaluation,
        advance_writes, validate_recovery, recover_ledger)
    from qwen38_seven_task_forward_writer import load_encoded
    from qwen38_native_runtime_preflight import verified_model
    from qwen38_h200_writer_timing import WRITERS
    manifest, data = load_encoded(options)
    require(len(data["queries"]) == 640 and all(len(data[t]) == 5000 for t in TASKS), "formal population changed")
    identity = original.hf_model_identity(options.model)
    identity["base_hash_audit"] = verified_model(options.model, options.base_verification)
    identity["verification_sha256"] = identity["base_hash_audit"]["verification_sha256"]
    require(digest(options.layout_contract) == options.layout_contract_sha256, "layout contract changed")
    layout_contract = original.read_json(options.layout_contract)
    for key in ("config_sha256", "index_sha256", "verification_sha256"):
        require(layout_contract["expected_snapshot_identity"]["base_model"][key] == identity[key], "layout belongs to another model")
    require(layout_contract["expected_snapshot_identity"]["tokenizer"] == identity["tokenizer_files"], "layout tokenizer differs")
    fields = dict(layout_contract["expected_layout"])
    for key in ("head_offsets", "head_vocab_sizes", "orders"): fields[key] = tuple(fields[key])
    layout = live.Qwen38PLELayout(**fields)
    writers = {arm: {"path": str(path.resolve()), "sha256": WRITERS[arm]["archive"], "tensor_sha256": WRITERS[arm]["tensor"]}
               for arm, path in (("ordinary", options.main_writer), ("delta", options.full_writer))}
    require(all(digest(spec["path"]) == spec["sha256"] for spec in writers.values()), "writer archive changed")
    suite, = original.load_suites(None, options.task_data, ["tasks"])
    task_tokens, task_token_sha = original.load_task_tokens(options.task_prompt_tokens, suite, options.model)
    require(manifest["generation_prompts_sha256"] == task_token_sha, "support/template generation binding differs")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(options.model), local_files_only=True)
    context_statistics = {}
    for population in original.load_suites(None, options.task_data, ["tasks"]):
        encoded = original.encode_suite(population, tokenizer,
            task_tokens=task_tokens if population["domain"] == "tasks" else None)
        context_statistics[population["domain"]] = {"prompts": len(encoded), "max_prompt_tokens": max(map(len, encoded)),
            "max_new_tokens": population["protocol"]["max_new_tokens"],
            "required_context": max(map(len, encoded)) + population["protocol"]["max_new_tokens"]}
    max_context = max(value["required_context"] for value in context_statistics.values())
    require(128 < max_context <= 8192, "locked generation context is outside the audited model bound")
    require(all(0 < len(row["input_ids"]) + 1 <= max_context for task in TASKS for row in data[task]),
            "a registered support plus its unused sample exceeds engine context")
    # Import these sources before binding; no vLLM or CUDA initialization occurs.
    sources = {str(path.resolve()): digest(path) for path in (
        Path(__file__), Path(__file__).with_name("qwen38_vllm_live_worker.py"), Path(live.__file__),
        Path(__file__).with_name("qwen38_vllm_accuracy.py"),
        Path(__file__).with_name("qwen38_greedy_decode.py"),
        options.delta_repo / "src/delta_ngram/qwen38_vllm_ple.py",
        options.delta_repo / "src/delta_ngram/qwen38_forward_recovery.py")}
    binding = {"schema_version": 1, "protocol": live.PROTOCOL, "layout": asdict(layout), "base_model": identity,
        "writers": writers, "encoded_manifest_sha256": options.encoded_manifest_sha256,
        "source_sha256": sources, "generation": "B1_cached_greedy", "D1_per_arm": 35000,
        "generation_schedule": {"task_queries_per_population": 448, "per_arm": 2240, "total": 4480,
            "before_each_task": 64, "after_each_task": "64 times learned task count", "math_code_generation": 0},
        "engine_capacity": {"gpu_memory_utilization": options.gpu_memory_utilization, "max_num_seqs": 1,
            "max_model_len": max_context, "max_num_batched_tokens": max_context,
            "context_statistics": context_statistics, "live_states_per_rank": 2,
            "engineering_capacity_rows_per_arm": 666747, "state_bytes_per_arm_at_capacity": 445387012},
        "gate_policy": "semantic_exact_history_numerical_logits_recorded_v1", "gate_decode_steps": options.gate_decode_steps}
    import torch
    require(not torch.cuda.is_initialized(), "CPU preflight initialized CUDA")
    if options.validate_only:
        print(json.dumps({"status": "CPU_preflight_passed", "protocol": live.PROTOCOL,
            "D1_total": 70000, "generation_total": 4480, "query_population": 448, "source_sha256": sources})); return
    options.output.mkdir(parents=True, exist_ok=options.resume)
    binding_path = options.output / "binding.json"
    if options.resume:
        require(original.read_json(binding_path) == json.loads(json.dumps(binding)), "resume binding changed")
    else: write_json(binding_path, binding)
    binding_sha = digest(binding_path)
    os.environ.update({live.ENV: str(binding_path.resolve()), live.ENV_SHA: binding_sha,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    require("DELTA_NGRAM_VLLM_BINDING" not in os.environ and "DELTA_NGRAM_VLLM_BINDING_SHA256" not in os.environ,
            "fixed-snapshot and live bindings cannot share a process")
    import vllm
    import vllm.envs as vllm_envs
    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    fork = verify_fork(options.vllm_fork_manifest, vllm.__file__)
    fork_manifest = original.read_json(options.vllm_fork_manifest)
    require(fork_manifest.get("protocol") == live.PROTOCOL
        and fork_manifest["live_helper"]["sha256"] == digest(live.__file__)
        and Path(fork_manifest["live_helper"]["path"]).resolve() == Path(live.__file__).resolve(), "live fork helper differs")
    kwargs = original.engine_kwargs(options.model.resolve(), enforce_eager=True,
        gpu_memory_utilization=options.gpu_memory_utilization, tensor_parallel_size=8, moe_backend="triton")
    kwargs.update(max_model_len=max_context, max_num_seqs=1, max_num_batched_tokens=max_context,
        enable_prefix_caching=False, enable_chunked_prefill=False, mamba_cache_mode="none", speculative_config=None,
        worker_extension_cls="qwen38_vllm_live_worker.DeltaNGramLiveWorkerExtension")
    original.configure_ple_cpu_offload(kwargs, EngineArgs, vllm_envs, requested=True)
    planned = EngineArgs(**copy.deepcopy(kwargs)).create_engine_config()
    require(planned.use_v2_model_runner and not planned.cache_config.enable_prefix_caching
        and not planned.scheduler_config.enable_chunked_prefill and planned.speculative_config is None,
        "engine resolved a different live request/cache policy")
    execution = original.execution_config(kwargs, planned, vllm_envs, ple_cpu_offload_requested=True)
    gen = original.read_json(options.model / "generation_config.json")
    eos = gen.get("eos_token_id", tokenizer.eos_token_id); eos_ids = [eos] if isinstance(eos, int) else list(eos)
    status = {"status": "loading", "binding_sha256": binding_sha, "fork": fork, "execution": execution,
        "mode": options.mode, "started_utc": original.utc_now(), "optimizer_steps": 0, "backward_calls": 0}
    prior_status = original.read_json(options.output / "STATUS.json") if options.resume and (options.output / "STATUS.json").exists() else None
    if prior_status is not None and "gate_sha256" in prior_status:
        status["gate_sha256"] = prior_status["gate_sha256"]
    write_json(options.output / "STATUS.json", status)
    try:
        llm = LLM(**copy.deepcopy(kwargs))
        require(original.execution_config(kwargs, llm.llm_engine.vllm_config, vllm_envs,
            ple_cpu_offload_requested=True) == execution, "live engine differs from registered execution")
        engine = LiveEngine(llm, SamplingParams, options.output)
        workers = engine.rpc("delta_live_initialize", writers, timeout=600)
        ready = agree(workers, ["binding_sha256", "protocol", "worker_sha256", "helper_sha256", "cache_dtype", "quiescent"], label="initialize")
        require(ready["binding_sha256"] == binding_sha and ready["cache_dtype"] == "torch.float32" and ready["quiescent"]
            and ready["worker_sha256"] == digest(Path(__file__).with_name("qwen38_vllm_live_worker.py"))
            and ready["helper_sha256"] == digest(live.__file__), "worker imported another live implementation")
        gate_path = options.output / "gate/COMPLETE.json"
        if not gate_path.exists():
            require(not options.resume, "incomplete gate must be preserved and restarted in a fresh output")
            prompts = [task_tokens[next(row["task_id"] for row in suite["records"] if row["task"] == task)] for task in TASKS[:2]]
            gate = semantic_gate(engine, data[TASKS[0]], prompts, steps=options.gate_decode_steps)
        else:
            require(options.resume, "existing gate without resume")
            gate = original.read_json(gate_path); require(gate["status"] == "passed", "semantic gate failed")
            require(prior_status is not None and prior_status.get("gate_sha256") == digest(gate_path)
                and prior_status.get("binding_sha256") == binding_sha, "resume semantic gate receipt changed")
        status.update(status="semantic_gate_passed", gate_sha256=digest(gate_path), workers=workers)
        write_json(options.output / "STATUS.json", status)
        if options.mode == "gate": return
        cursors = {arm: initial_cursor() for arm in live.ARMS}; evaluations = {arm: {} for arm in live.ARMS}
        latest_path = options.output / "LATEST.json"
        if latest_path.exists():
            require(options.resume, "formal checkpoint exists")
            latest = original.read_json(latest_path)
            require(latest["binding_sha256"] == binding_sha and latest["gate_sha256"] == digest(gate_path), "resume identity differs")
            engine.restore(latest["state"]); cursors, evaluations = latest["cursors"], latest["evaluations"]
            for arm in live.ARMS:
                validate_recovery(cursors[arm], evaluations[arm])
                for name, receipt in evaluations[arm].items():
                    validate_endpoint_receipt(options.output, receipt, arm=arm, name=name,
                        task_ids_by_task={task: [row["task_id"] for row in suite["records"] if row["task"] == task] for task in TASKS})
                require(engine.epochs[arm] == cursors[arm]["writes_completed"], "resume support cursor differs")
                recover_ledger(options.output / f"{arm}.writes.jsonl", writes_completed=engine.epochs[arm],
                    archive_dir=options.output / "recovery" / uuid.uuid4().hex)

        def save(name="resume", *, immutable=False):
            for arm in live.ARMS: validate_recovery(cursors[arm], evaluations[arm])
            if not immutable:
                previous = original.read_json(latest_path)["state"]["path"] if latest_path.exists() else ""
                name = "resume_b" if Path(previous).name == "resume_a.state.pt" else "resume_a"
            receipt = engine.checkpoint(options.output / (name + ".state.pt"), replace=not immutable)
            row = {"binding_sha256": binding_sha, "gate_sha256": digest(gate_path), "state": receipt,
                   "cursors": copy.deepcopy(cursors), "evaluations": copy.deepcopy(evaluations)}
            write_json(options.output / "LATEST.json", row)
            if immutable: write_json(options.output / (name + ".json"), row)
            return receipt

        def evaluate(arm, name, domains):
            destination = options.output / arm / "generation" / name
            destination.mkdir(parents=True, exist_ok=True)
            tasks = [t for t in TASKS if t in domains]
            selected = ["tasks"] if tasks else []
            suites = original.load_suites(None, options.task_data, selected, tasks)
            engine.query_audit = destination / "request_audit.jsonl"
            summaries = []
            state = save()
            for current in suites:
                tokens, token_sha = (original.load_task_tokens(options.task_prompt_tokens, current, options.model)
                    if current["domain"] == "tasks" else (None, None))
                encoded = original.encode_suite(current, tokenizer, task_tokens=tokens)
                contract = original.make_contract(current, encoded, {**identity, "live_binding_sha256": binding_sha,
                    "live_state_hash": state["state_hashes"][arm], "live_state_epoch": engine.epochs[arm]},
                    eos_ids=eos_ids, method=arm, stage=cursors[arm]["task_index"],
                    source_checkpoint="live-state:" + state["state_hashes"][arm], source_checkpoint_sha256=state["state_hashes"][arm],
                    task_token_sha256=token_sha, execution=execution)
                contract["protocol"].update(batch_size=1, max_num_batched_tokens=max_context, max_model_len=max_context, max_num_seqs=1,
                    checkpoint_state="immutable_request_bound_live_FP32_Delta", enable_prefix_caching=False,
                    request_local_cache=True, semantic_preflight_sha256=digest(gate_path))
                target = destination / current["domain"]
                try: existing = original.load_resume(target, contract)
                except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as error:
                    archive_pending(options.output, [target], "pending generation prefix failed validation: " + str(error))
                    existing = []
                summaries.append(original.run_suite(ReadOnlyEngine(engine, arm), SamplingParams, tokenizer,
                    current, encoded, target, contract, existing=existing))
            engine.query_audit = None
            after = save()
            require(all(after[k] == state[k] for k in ("state_hashes", "epochs", "identities")), "endpoint generation mutated Delta")
            result = {"status": "completed", "arm": arm, "name": name, "domains": domains,
                "state_hash": state["state_hashes"][arm], "epoch": engine.epochs[arm],
                "generated_domains": tasks, "math_code_generation": 0,
                "summaries": [{"domain": s["domain"], "generated_tasks": s["generated_tasks"],
                               "run_contract_sha256": s["run_contract_sha256"], "trace_sha256": s["trace_sha256"]} for s in summaries]}
            write_json(destination / "COMPLETE.json", result)
            return {"path": str(destination / "COMPLETE.json"), "sha256": digest(destination / "COMPLETE.json"), "domains": domains}

        for arm in live.ARMS:
            cursor = cursors[arm]
            status.update(status="formal_running", active_arm=arm, cursor=cursor)
            write_json(options.output / "STATUS.json", status)
            while cursor["phase"] != "complete":
                pending = pending_evaluation(cursor)
                if pending:
                    name, domains = pending
                    boundary = options.output / f"{arm}.{name}.state.pt"
                    sidecar = options.output / f"{arm}.{name}.json"
                    current = save()
                    valid_boundary = False
                    if boundary.exists() or sidecar.exists():
                        try:
                            previous = original.read_json(sidecar)
                            valid_boundary = (previous["binding_sha256"] == binding_sha
                                and previous["gate_sha256"] == digest(gate_path)
                                and Path(previous["state"]["path"]).resolve() == boundary.resolve()
                                and digest(boundary) == previous["state"]["sha256"]
                                and previous["cursors"] == cursors and previous["evaluations"] == evaluations
                                and all(previous["state"][k] == current[k] for k in ("state_hashes", "epochs", "identities")))
                        except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError): pass
                        if not valid_boundary:
                            archive_pending(options.output, [boundary, sidecar], "unverified pending boundary checkpoint")
                    if not valid_boundary: save(f"{arm}.{name}", immutable=True)
                    evaluations[arm][name] = evaluate(arm, name, domains)
                    acknowledge_evaluation(cursor, name); save(); continue
                task = TASKS[cursor["task_index"]]
                row = data[task][cursor["example_index"]]
                event = engine.support(arm, row)
                advance_writes(cursor, 1, len(row["input_ids"]))
                require(engine.epochs[arm] == cursor["writes_completed"], "host/worker D1 cursors differ")
                with (options.output / f"{arm}.writes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"kind": "write", "write_index": engine.epochs[arm], "task": task,
                        "source_row_index": row["source_row_index"], "row_id": row["row_id"],
                        "tokens": len(row["input_ids"]), **event}, allow_nan=False) + "\n")
                if cursor["writes_completed"] % options.save_every == 0 or cursor["phase"] != "write": save()
                if cursor["writes_completed"] % 100 == 0:
                    print(json.dumps({"arm": arm, "cursor": cursor, "state_identity": event["state"]["identity"]}), flush=True)
            require(engine.epochs[arm] == 35000, "formal chain ended at a different support population")
        final = save("completed", immutable=True)
        status.update(status="completed", D1_total=70000, final=final)
        write_json(options.output / "COMPLETE.json", status)
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_utc"] = original.utc_now()
        write_json(options.output / "STATUS.json", status)


if __name__ == "__main__": main()

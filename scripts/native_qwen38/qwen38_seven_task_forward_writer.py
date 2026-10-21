"""Independent seven-task A800 forward writer: prepare on CPU, execute explicitly.

Frozen alpha=1 100M writer. Native evidence H reads T0 only; each independent
support example causes exactly one chronological D1 update of a shared Delta.
Native TF queries read immutable split-FP32 Delta. Generation is a separate
consumer of the immutable endpoint snapshots, never a surrogate TF accuracy.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from delta_ngram.qwen38_seven_forward import (
    TASKS, PROTOCOL, require, digest, canonical_hash, evidence_batch,
    state_payload, restore_state, state_fingerprint, validate_encoded_record,
)
from delta_ngram.qwen38_forward_recovery import (
    RECOVERY_PROTOCOL, initial_cursor, evaluation_policy, evaluation_name,
    validate_recovery, pending_evaluation, acknowledge_evaluation, advance_writes,
    recover_ledger, archive_existing, query_metadata, select_queries, validate_evaluation_artifacts,
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_rows(path, rows):
    raw = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(raw, encoding="utf-8")
    return {"path": path.name, "sha256": digest(path), "rows": len(rows)}


def setup_cl(path):
    for directory in (Path(path) / "scripts", Path(path) / "src"):
        sys.path.insert(0, str(directory.resolve()))


def prepare(options):
    """No backbone load, no shuffle, no truncation, no support example merging."""
    setup_cl(options.cl_repo)
    from qwen38_mcore_retention_baseline import (
        register_qwen4_exp_swift_compat, load_task_probes,
        encode_task_reference, load_math_code_references,
    )
    from cl_research import retention_nll
    register_qwen4_exp_swift_compat()
    from swift.model import get_model_processor
    from swift.template import get_template
    model, processor = get_model_processor(str(options.model_path), model_type="qwen4_exp",
                                           load_model=False, download_model=False)
    require(model is None, "CPU preparation must not instantiate a model")
    template = get_template(processor, template_type="qwen3_8", enable_thinking=False,
                            max_length=8192, truncation_strategy="raise")
    template.set_mode("train")
    historical = json.loads(options.historical_args.read_bytes())
    require(historical.get("template") == "qwen3_8" and historical.get("enable_thinking") is False
            and historical.get("max_length") == 128, "historical training template differs")
    generation = json.loads(options.generation_prompts.read_bytes())
    require(generation["source_sha256"] == digest(options.task_probes)
            and generation["enable_thinking"] is False, "fixed generation population differs")
    for name, expected in generation["tokenizer_files"].items():
        require(digest(options.model_path / name) == expected, "fixed generation tokenizer differs")
    prompt_by_id = {row["task_id"]: row for row in generation["records"]}
    require(len(prompt_by_id) == 448, "fixed generation manifest must contain 448 unique task IDs")
    options.encoded_dir.mkdir(parents=True, exist_ok=False)
    files, sources, statistics = {}, {}, {}
    for task in TASKS:
        source = options.train_dir / f"{task}_train.jsonl"
        rows = read_rows(source)
        require(len(rows) == 5000, f"{task}: exactly 5000 independent supports are required")
        encoded = []
        for index, row in enumerate(rows):
            item = template.encode(row)
            require(isinstance(item, dict), "training template split/deleted a sample")
            ids = list(item["input_ids"])
            require(0 < len(ids) <= 128, f"{task}:{index}: cannot silently drop/truncate an overlength sample")
            record = {"task": task, "row_id": f"{task}:export-row:{index}", "source_row_index": index,
                      "source_record_sha256": canonical_hash(row), "input_ids": ids,
                      "input_token_ids_sha256": canonical_hash(ids), "write_events": 1,
                      "training_labels_sha256": canonical_hash(list(item["labels"]))}
            encoded.append(validate_encoded_record(record, evidence=True))
        files[task] = save_rows(options.encoded_dir / f"{task}.jsonl", encoded)
        sources[task] = {"path": str(source.resolve()), "sha256": digest(source)}
        statistics[task] = {"examples": 5000, "D1_updates": 5000,
                            "real_tokens": sum(len(row["input_ids"]) for row in encoded),
                            "max_tokens": max(len(row["input_ids"]) for row in encoded)}
    tasks, _ = load_task_probes(options.task_probes)
    references, reference_manifest = load_math_code_references(options.reference_dir, "v2_128_64")
    queries = []
    for row in tasks:
        item = encode_task_reference(row, template, 8192)
        fixed = prompt_by_id[row["task_id"]]
        require(fixed["prompt_sha256"] == row["prompt_sha256"]
                and item["input_ids"][:item["prompt_length"]] == fixed["prompt_token_ids"],
                "native TF prompt differs from the locked greedy-generation prefix")
        queries.append(validate_encoded_record({**row, **item}, evidence=False))
    for row in references:
        item = retention_nll.encode_reference(row, template.tokenizer, 8192)
        queries.append(validate_encoded_record({**row, **item}, evidence=False))
    files["queries"] = save_rows(options.encoded_dir / "queries.jsonl", queries)
    manifest = {
        "protocol": PROTOCOL, "created_utc": datetime.now(timezone.utc).isoformat(),
        "task_order": list(TASKS), "files": files, "support_sources": sources,
        "statistics": statistics, "D1_updates": 35000, "distinct_support_examples": 35000,
        "template": "qwen3_8", "enable_thinking": False, "max_support_tokens": 128,
        "historical_args": {"path": str(options.historical_args), "sha256": digest(options.historical_args),
                            "dataset_shuffle": historical.get("dataset_shuffle"),
                            "data_seed": historical.get("data_seed"), "seed": historical.get("seed"),
                            "truncation_strategy": historical.get("truncation_strategy")},
        "support_order": "exported_file_order; historical_actual_sampler_order_not_recovered",
        "same_support_set_and_exposure_count": True, "historical_sampler_order_claimed": False,
        "independent_documents": True, "one_D1_per_document": True,
        "context_reset_between_examples": ["attention", "positions", "PLE_ngram_text_context", "local_convolution"],
        "continuous_across_examples_and_tasks": ["Delta", "counts", "block_clock"],
        "task_probe_sha256": digest(options.task_probes),
        "generation_prompts_sha256": digest(options.generation_prompts),
        "reference_manifest_sha256": digest(options.reference_dir / "manifest.json"),
        "reference_records_sha256": digest(options.reference_dir / "records.jsonl"),
        "reference_protocol": reference_manifest["protocol"],
        "tokenizer_files": generation["tokenizer_files"],
        "query_population": {**{task: 64 for task in TASKS}, "math": 128, "code": 64},
        "prepare_script_sha256": digest(__file__),
    }
    write_json(options.encoded_dir / "manifest.json", manifest)
    print(json.dumps({"status": "prepared_CPU_only", "path": str(options.encoded_dir.resolve()),
                      "manifest_sha256": digest(options.encoded_dir / "manifest.json"),
                      "support_examples": 35000, "queries": len(queries)}, ensure_ascii=False), flush=True)


def load_encoded(options):
    path = options.encoded_dir / "manifest.json"
    require(digest(path) == options.encoded_manifest_sha256, "encoded manifest is not the registered file")
    manifest = json.loads(path.read_bytes())
    require(manifest["protocol"] == PROTOCOL and manifest["task_order"] == list(TASKS)
            and manifest["D1_updates"] == 35000, "encoded chain protocol differs")
    result = {}
    for key, info in manifest["files"].items():
        source = (options.encoded_dir / info["path"]).resolve()
        require(source.is_relative_to(options.encoded_dir.resolve()) and digest(source) == info["sha256"],
                "encoded input file escaped its directory or changed")
        rows = read_rows(source)
        require(len(rows) == info["rows"], "encoded row count differs")
        result[key] = [validate_encoded_record(row, evidence=key != "queries") for row in rows]
    return manifest, result


def writer_source(options):
    import torch
    from delta_ngram.qwen38_probe import tensor_state_hash
    require(digest(options.writer_checkpoint) == options.writer_sha256, "source writer file hash differs")
    saved = torch.load(options.writer_checkpoint, map_location="cpu", weights_only=False)
    candidates = list(saved["arms"].values()) if "arms" in saved else [saved]
    selected = [arm for arm in candidates if arm.get("alpha") == 1.]
    require(len(selected) == 1, "checkpoint must identify one alpha=1 writer")
    source = selected[0]
    require(source["optimizer_step"] == 4172 and source["completed_loss_targets"] == 99997072,
            "requires the registered main alpha=1 nominal-100M endpoint")
    require(tensor_state_hash(source["writer"]) == source["writer_sha256"], "writer tensor digest differs")
    return source


def execute(options):
    import torch
    from qwen38_native_a800_session import run_a800_session
    manifest, data = load_encoded(options)
    source = writer_source(options)
    if options.resume:
        require(options.output.is_dir(), "resume output directory is missing")
    else:
        # torchrun starts all ranks concurrently; only rank0 creates run output.
        if int(os.environ.get("RANK", "0")) == 0:
            options.output.mkdir(parents=True, exist_ok=False)
        else:
            deadline = time.monotonic() + 120
            while not options.output.is_dir():
                require(time.monotonic() < deadline, "rank0 did not create output directory")
                time.sleep(.1)
    setup_cl(options.cl_repo)

    def run(session):
        import torch.distributed as dist
        from megatron.core import parallel_state as ps
        from delta_ngram.alpha_memory import LazyDecayState
        from delta_ngram.qwen38_native import make_qwen38_writer
        from delta_ngram.qwen38_memory_writer import write_memory_aware_native_block
        from delta_ngram.qwen38_training import writer_evidence_features
        from delta_ngram.qwen38_vllm_snapshot import save_vllm_snapshot
        from delta_ngram.qwen38_probe import tensor_state_hash
        from qwen38_native_a800_forward import native_forward, native_dispatch, sync_written_rows
        from qwen38_native_a800_session import runtime_phase
        from qwen38_mcore_retention_baseline import LayerCapture, reference_batch
        rank, device = dist.get_rank(), session.device
        control = session.control_group
        runtime_phase("callback_state_setup_begin")
        first = ps.is_pipeline_first_stage()
        require((rank == 0) == (first and ps.get_tensor_model_parallel_rank() == 0),
                "canonical writer owner is not global rank0")
        code_paths = [Path(__file__), Path(__file__).with_name("qwen38_native_a800_session.py"),
                      Path(__file__).with_name("qwen38_native_a800_forward.py"),
                      Path(__file__).with_name("qwen38_native_a800_dispatch.py"),
                      Path(__file__).with_name("qwen38_native_a800_host.py")]
        code_paths += [REPO / "src" / "delta_ngram" / name for name in (
            "qwen38_seven_forward.py", "qwen38_forward_recovery.py", "qwen38_native.py", "qwen38_adapter.py", "qwen38_precision.py",
            "qwen38_writer.py", "qwen38_memory_writer.py", "qwen38_training.py", "qwen38_vllm_snapshot.py",
            "alpha_memory.py", "batching.py", "state.py", "operators.py")]
        identity = {"base_model": session.identity, "tokenizer": manifest["tokenizer_files"],
                    "addressing": "native_PLE_ids_plus_one;one_owner;no_external_rehash",
                    "writer": {"checkpoint_sha256": options.writer_sha256,
                               "tensor_sha256": source["writer_sha256"], "alpha": 1.,
                               "optimizer_step": 4172, "completed_loss_targets": 99997072},
                    "operator": "D1;one_independent_example_per_update;FP32",
                    "trajectory": {"protocol": PROTOCOL, "method": "writer",
                                   "encoded_manifest_sha256": options.encoded_manifest_sha256,
                                   "task_order": list(TASKS), "H_encoding": "native_T0_only",
                                   "gate_reads_current_Delta": True, "evidence_batch_size": options.evidence_batch_size},
                    "native_runtime": session.versions,
                    "evaluation_policy": evaluation_policy(options.query_limit, options.capture_layers),
                    "recovery_protocol": RECOVERY_PROTOCOL,
                    "code_sha256": {str(path.relative_to(REPO)): digest(path) for path in code_paths}}
        cursor, evaluation_receipts = initial_cursor(), {}
        state = LazyDecayState.from_batch(session.layout.empty_state(1, device)) if first else None
        saved_signature = None
        if options.resume:
            saved = torch.load(options.resume, map_location="cpu", weights_only=True)
            require(saved["protocol"] == PROTOCOL and saved["identity"] == identity,
                    "resume source/data/reader/encoding identity differs")
            cursor = dict(saved["cursor"])
            require(saved.get("recovery_protocol") == RECOVERY_PROTOCOL, "resume checkpoint lacks pending-evaluation phases")
            evaluation_receipts = dict(saved["evaluation_receipts"])
            saved_signature = saved["state_fingerprint"]
            if first:
                state = restore_state(saved["state"], device=device)
                require(state.row_capacity == session.layout.row_capacity and state.width == session.layout.row_width,
                        "resume state layout differs")
                require(state_fingerprint(state) == saved["state_fingerprint"], "resume state fingerprint differs")
        validate_recovery(cursor, evaluation_receipts)
        if first:
            require(int(state.block_clock[0]) == cursor["writes_completed"], "state clock and resume cursor differ")
        writer = alpha = None
        if rank == 0:
            writer, alpha = make_qwen38_writer(alpha=1.)
            writer.load_state_dict(source["writer"], strict=True)
            writer = writer.to(device).float().eval().requires_grad_(False)
            alpha = alpha.to(device).eval().requires_grad_(False)
        initial_parameters = [(p, p._version) for p in writer.parameters()] if writer is not None else []
        runtime_phase("callback_state_writer_setup_complete")
        dist.barrier(group=control)
        ledger = options.output / "write_events.jsonl"
        completed_this_run = 0
        attempt = [uuid.uuid4().hex if rank == 0 else None]
        dist.broadcast_object_list(attempt, src=0, group=control)
        attempt_id = attempt[0]
        audit = options.output / "audit" / attempt_id
        resume_audit = None
        if rank == 0 and options.resume:
            for name, receipt in evaluation_receipts.items():
                domains = receipt["domains"]
                expected = [query_metadata(row) for row in select_queries(data["queries"], domains, options.query_limit)]
                validate_evaluation_artifacts(options.output, name, receipt, identity=identity,
                                              expected_records=expected, capture_layers=options.capture_layers)
            resume_audit = recover_ledger(ledger, writes_completed=cursor["writes_completed"], archive_dir=audit)
            for name in ("COMPLETE.json", "STOPPED.json", "registration.json"):
                archive_existing(options.output / name, audit)
        dist.barrier(group=control)

        def log(value):
            if rank == 0:
                value = {"attempt_id": attempt_id, **value}
                with ledger.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)

        def save(name, *, endpoint=False):
            nonlocal saved_signature
            validate_recovery(cursor, evaluation_receipts)
            runtime_phase("save_begin:" + name)
            signature = state_fingerprint(state) if first else None
            signatures = [None] * dist.get_world_size()
            dist.all_gather_object(signatures, signature, group=control)
            require(signatures[0] == signatures[1] and signatures[0] is not None
                    and all(value is None for value in signatures[2:]),
                    "the canonical Delta was not replicated exactly to TP1")
            saved_signature = signatures[0]
            receipt = None
            if rank == 0:
                require(tensor_state_hash(writer.state_dict()) == source["writer_sha256"], "frozen writer changed")
                path = options.output / f"{name}.state.pt"
                temporary = path.with_suffix(".pt.partial")
                torch.save({"protocol": PROTOCOL, "identity": identity, "cursor": dict(cursor),
                            "recovery_protocol": RECOVERY_PROTOCOL, "evaluation_receipts": evaluation_receipts,
                            "state": state_payload(state), "state_fingerprint": signature,
                            "writer_frozen": True, "optimizer_steps": 0,
                            "layout": asdict(session.layout), "checkpoint_boundary": "complete_D1_and_explicit_evaluation_phase"}, temporary)
                os.replace(temporary, path)
                receipt = {"path": str(path), "sha256": digest(path), "cursor": dict(cursor),
                           "state_fingerprint": signature, "rows": state.rows.numel(),
                           "state_l2": float(torch.linalg.vector_norm(state.values.double())),
                           "max_block_count": float(state.counts.max()) if state.counts.numel() else 0.}
                if endpoint:
                    snapshot = options.output / f"{name}.delta.pt"
                    temporary = snapshot.with_suffix(".pt.partial")
                    is_partial = cursor["example_index"] != 0
                    snapshot_identity = {**identity,
                        "trajectory": {**identity["trajectory"], "stage": cursor["task_index"],
                                       "partial": is_partial,
                                       "purpose": "bounded_diagnostic" if is_partial else "task_boundary"},
                        "endpoint": {"name": name, "cursor": dict(cursor)}}
                    snapshot_receipt = save_vllm_snapshot(temporary, state=state, layout=session.layout,
                                                          identity=snapshot_identity)
                    os.replace(temporary, snapshot)
                    snapshot_receipt["path"] = str(snapshot.resolve())
                    receipt["snapshot"] = snapshot_receipt
                    write_json(options.output / f"{name}.delta_contract.json", snapshot_receipt)
                write_json(options.output / f"{name}.json", receipt)
                write_json(options.output / "LATEST.json", receipt)
            dist.barrier(group=control)
            runtime_phase("save_complete:" + name)
            return receipt

        def evaluate(name, domains):
            runtime_phase("evaluate_begin:" + name)
            rows = select_queries(data["queries"], domains, options.query_limit)
            destination = options.output / "tf" / name
            if rank == 0:
                # A pending phase always replays. Preserve its prior attempt;
                # no stale completed marker remains visible while replaying.
                archive_existing(destination, audit / "tf")
                destination.mkdir(parents=True, exist_ok=False)
                (destination / "predictions.jsonl").write_text("", encoding="utf-8")
            dist.barrier(group=control)
            checkpoint = {"path": name, "sha256": canonical_hash({"identity": identity, "cursor": cursor})}
            capture = LayerCapture(session.models, SimpleNamespace(sequence_parallel=True), destination,
                dtype="bfloat16", last_prompt_only=False, checkpoint=checkpoint,
                checkpoint_kind="frozen_writer_fixed_FP32_Delta") if options.capture_layers else None
            metrics, encoded_records, local_capture_files = [], [], []
            receipt = None
            try:
                for index, row in enumerate(rows):
                    batch, _ = reference_batch(row, pad_token_id=int(session.tokenizer.pad_token_id),
                                               pad_multiple=2, device=device)
                    valid = torch.arange(batch["input_ids"].shape[1], device=device)[None] < len(row["input_ids"])
                    metadata = query_metadata(row)
                    encoded_records.append(metadata)
                    if capture:
                        capture.begin(index, metadata, row, batch["input_ids"].shape[1])
                    tick = time.monotonic()
                    result = native_forward(session, batch, valid, state=state, evidence=False)
                    require(result["token_count"] == row["target_token_count"], "TF target count changed")
                    timings = capture.finish() if capture else [0., 0., 0.]
                    if capture:
                        path = destination / "representations" / f"query-{index:04d}-rank-{rank:02d}.pt"
                        local_capture_files.append({"path": path.relative_to(destination).as_posix(),
                                                    "sha256": digest(path), "bytes": path.stat().st_size})
                    dist.barrier(group=control)
                    metrics.append({**metadata, **result, "seconds": time.monotonic() - tick,
                                    "capture_seconds": timings})
                    if rank == 0:
                        with (destination / "predictions.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(metrics[-1], allow_nan=False) + "\n")
                capture_files = [None] * dist.get_world_size()
                dist.all_gather_object(capture_files, local_capture_files, group=control)
                if rank == 0:
                    grouped = {}
                    for domain in domains:
                        chosen = [r for r in metrics if r["domain"] == domain]
                        total = sum(r["token_count"] for r in chosen)
                        grouped[domain] = {"n": len(chosen), "token_count": total,
                                           "loss_sum": math.fsum(r["loss_sum"] for r in chosen),
                                           "answer_nll": math.fsum(r["loss_sum"] for r in chosen) / total,
                                           "row_ids": [r["task_id"] for r in chosen]}
                    write_json(destination / "summary.json", {"status": "completed", "identity": identity,
                        "cursor": dict(cursor), "domains": grouped, "capture_layers": options.capture_layers,
                        "capture_checkpoint": checkpoint, "encoded_records": encoded_records,
                        "state_fingerprint": saved_signature,
                        "predictions_sha256": digest(destination / "predictions.jsonl"),
                        "capture_files": [item for group in capture_files for item in group],
                        "topology": {"tp": 2, "pp": 4, "ep": 2, "cp": 1, "dp": 1},
                        "task_count": len(rows), "completed_tasks": len(metrics),
                        "state_read_only": True, "generation_accuracy_measured_here": False})
                    receipt = {"summary_path": f"tf/{name}/summary.json", "summary_sha256": digest(destination / "summary.json"),
                               "cursor": dict(cursor), "domains": list(domains), "query_count": len(rows)}
                    validate_evaluation_artifacts(options.output, name, receipt, identity=identity,
                                                  expected_records=encoded_records, capture_layers=options.capture_layers)
            finally:
                if capture:
                    capture.close()
            dist.barrier(group=control)
            receipts = [receipt]
            dist.broadcast_object_list(receipts, src=0, group=control)
            runtime_phase("evaluate_complete:" + name)
            return receipts[0]

        with native_dispatch(session) as dispatch, torch.no_grad():
            dist.barrier(group=control)
            if rank == 0:
                write_json(options.output / "registration.json", {"protocol": PROTOCOL, "identity": identity,
                    "dispatch": dispatch, "TP_PP_EP": [2, 4, 2], "reader_parameters_frozen": True,
                    "writer_parameters_frozen": True, "single_canonical_writer_rank": 0,
                    "initial_cursor": dict(cursor), "max_writes_this_run": options.max_writes,
                    "query_limit_per_domain": options.query_limit, "source_manifest": manifest,
                    "attempt_id": attempt_id, "resume_audit": resume_audit,
                    "H200_numerical_equivalence_claimed": False})
            while cursor["phase"] != "complete":
                validate_recovery(cursor, evaluation_receipts)
                phase, task_index = cursor["phase"], cursor["task_index"]
                stop_budget = bool(options.max_writes and completed_this_run >= options.max_writes)
                if phase in ("pre_eval", "write") and (task_index >= options.max_tasks or stop_budget):
                    if phase == "write" and cursor["example_index"]:
                        save("partial", endpoint=True)
                        evaluate("partial", [TASKS[task_index]])
                    else:
                        save("resume")
                    break
                pending = pending_evaluation(cursor)
                if pending is not None:
                    name, domains = pending
                    if phase == "base_eval":
                        save("boundary_00", endpoint=True)
                    elif phase == "boundary_eval":
                        save(name, endpoint=True)
                    else:
                        save("resume")
                    evaluation_receipts[name] = evaluate(name, domains)
                    acknowledge_evaluation(cursor, name)
                    save("resume")
                    continue
                task = TASKS[task_index]
                count = min(options.evidence_batch_size, 5000 - cursor["example_index"])
                if options.max_writes:
                    count = min(count, options.max_writes - completed_this_run)
                index = cursor["example_index"]
                rows = data[task][index:index + count]
                mb = evidence_batch(rows, pad_token_id=int(session.tokenizer.pad_token_id), device=device)
                tick = time.monotonic()
                result = native_forward(session, mb["batch"], mb["valid_mask"], evidence=True)
                dirty = []
                if rank == 0:
                    for lane, row in enumerate(rows):
                        length = len(row["input_ids"])
                        h = writer_evidence_features(result["hidden"][lane:lane + 1, :length], mode="rmsnorm", eps=1e-6)
                        physical = result["physical_rows"][lane:lane + 1, :length]
                        observation = {}
                        def observe(events):
                            beta = events.beta.detach().double()
                            observation.update(event_count=events.rows.numel(), beta_mean=float(beta.mean()),
                                beta_min=float(beta.min()), beta_max=float(beta.max()),
                                target_rms=float(events.target.detach().double().square().mean().sqrt()))
                        state = write_memory_aware_native_block(writer, state, h, physical,
                            torch.ones(h.shape[:2], dtype=torch.bool, device=device), alpha, event_observer=observe)
                        require(bool(state.values.isfinite().all()), "nonfinite Delta after D1")
                        touched = torch.unique(physical[physical.ne(0)], sorted=True)
                        dirty.append(touched)
                        log({"kind": "write", "task": task, "source_row_index": row["source_row_index"],
                             "write_index": cursor["writes_completed"] + lane + 1,
                             "tokens": length, "unique_touched_rows": touched.numel(), **observation})
                if first:
                    touched = torch.unique(torch.cat(dirty), sorted=True) if rank == 0 else None
                    state = sync_written_rows(state, touched, device=device)
                advance_writes(cursor, len(rows), mb["real_tokens"])
                completed_this_run += len(rows)
                del result
                dist.barrier(group=control)
                if cursor["phase"] == "write" and cursor["example_index"] % options.save_every < len(rows):
                    save("resume")
                if cursor["example_index"] % 100 < len(rows):
                    log({"kind": "progress", **cursor, "task": task,
                         "batch_seconds": time.monotonic() - tick})
            session.assert_frozen()
            require(all(p._version == version and not p.requires_grad and p.grad is None
                        for p, version in initial_parameters), "writer parameter mutation or gradients detected")
            validate_recovery(cursor, evaluation_receipts)
            status = "completed" if cursor["phase"] == "complete" else "bounded_stop"
            if rank == 0:
                write_json(options.output / ("COMPLETE.json" if status == "completed" else "STOPPED.json"),
                    {"status": status, "cursor": cursor, "evaluation_receipts": evaluation_receipts,
                    "identity": identity, "backward_calls": 0, "optimizer_steps": 0,
                    "generation_pending_separate_snapshot_consumer": True})
            return {"status": status, "cursor": cursor}
    return run_a800_session(options, run)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--execute", action="store_true")
    p.add_argument("--cl-repo", type=Path, required=True)
    p.add_argument("--delta-repo", type=Path, default=REPO)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--encoded-dir", type=Path, required=True)
    p.add_argument("--train-dir", type=Path)
    p.add_argument("--task-probes", type=Path)
    p.add_argument("--reference-dir", type=Path)
    p.add_argument("--generation-prompts", type=Path)
    p.add_argument("--historical-args", type=Path)
    p.add_argument("--encoded-manifest-sha256")
    p.add_argument("--writer-checkpoint", type=Path)
    p.add_argument("--writer-sha256")
    p.add_argument("--verification", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--evidence-batch-size", type=int, choices=(1, 2, 4, 8, 12), default=4)
    p.add_argument("--max-writes", type=int, default=0, help="Explicit bounded smoke limit; 0 executes the registered chain.")
    p.add_argument("--max-tasks", type=int, default=7)
    p.add_argument("--query-limit", type=int, default=0, help="Per-domain bounded smoke limit; 0 keeps all locked queries.")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--capture-layers", action=argparse.BooleanOptionalAction, default=True)
    return p


def main():
    options = parser().parse_args()
    options.tokens = 8192
    require(0 <= options.max_writes <= 35000 and 1 <= options.max_tasks <= 7
            and options.query_limit >= 0 and options.save_every > 0, "invalid bounded limits")
    if options.prepare:
        require(all(getattr(options, key) is not None for key in
                    ("train_dir", "task_probes", "reference_dir", "generation_prompts", "historical_args")),
                "prepare requires all locked support/reference/template sources")
        prepare(options)
    else:
        require(all(getattr(options, key) is not None for key in
                    ("encoded_manifest_sha256", "writer_checkpoint", "writer_sha256", "verification", "output")),
                "execute requires all writer/data/base hashes and output path")
        execute(options)


if __name__ == "__main__":
    main()

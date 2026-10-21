"""Bounded A800 memory-aware two-task chain; no optimizer or generation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from delta_ngram.qwen38_seven_forward import (
    require, digest, canonical_hash, evidence_batch, state_payload, restore_state,
    state_fingerprint, immutable_guard, assert_unchanged,
)
from delta_ngram.qwen38_forward_recovery import query_metadata, select_queries
from delta_ngram.qwen38_shortchain_protocol import (
    SHORTCHAIN_PROTOCOL, shortchain_plan, validate_shortchain_cursor,
    validate_state_payload, assert_state_payload_equal,
)
from qwen38_seven_task_forward_writer import setup_cl, load_encoded, write_json


def load_full_writer(options):
    import torch
    from delta_ngram.qwen38_probe import tensor_state_hash
    expected = "51d5977845e9930f48732c73ae2503511d8976b1201f139a2be46bfab9f57308"
    require(options.writer_sha256 == expected and digest(options.writer_checkpoint) == expected,
            "full100M checkpoint file hash differs")
    saved = torch.load(options.writer_checkpoint, map_location="cpu", weights_only=False)
    require(saved["protocol"] == "qwen38_native_memory_fork_checkpoint_v1"
            and saved["policy"]["branch"] == "full" and saved["completed_groups"] == 835
            and saved["completed_targets"] == 20015113, "full-memory endpoint differs")
    branch = saved["branch_state"]
    require(branch["model_optimizer_step"] == 4172 and branch["completed_loss_targets"] == 99997072
            and tensor_state_hash(branch["writer"]) == branch["writer_sha256"], "writer tensors/endpoint differ")
    return branch["writer"], {"sha256": expected, "writer_tensor_state_hash": branch["writer_sha256"]}


def execute(options):
    import torch
    from qwen38_native_a800_session import run_a800_session
    manifest, data = load_encoded(options)
    tensors, source = load_full_writer(options)
    require(source["sha256"] == options.writer_sha256, "full writer registration differs")
    plan = shortchain_plan(tasks=("sme", "nav"), frames_per_task=32, query_limit=1, restore_after=16)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
    else:
        deadline = time.monotonic() + 120
        while not options.output.is_dir():
            require(time.monotonic() < deadline, "rank0 did not create output")
            time.sleep(.1)
    setup_cl(options.cl_repo)

    def run(session):
        import torch.distributed as dist
        from megatron.core import parallel_state as ps
        from delta_ngram.alpha_memory import LazyDecayState
        from delta_ngram.qwen38_native import make_qwen38_writer
        from delta_ngram.qwen38_memory_writer import write_memory_aware_native_block
        from delta_ngram.qwen38_training import writer_evidence_features
        from delta_ngram.qwen38_probe import tensor_state_hash
        from qwen38_native_a800_forward import native_forward, native_dispatch, sync_written_rows
        from qwen38_mcore_retention_baseline import LayerCapture, reference_batch
        rank, device, control = dist.get_rank(), session.device, session.control_group
        first = ps.is_pipeline_first_stage()
        state = LazyDecayState.from_batch(session.layout.empty_state(1, device)) if first else None
        writer = alpha = None
        if rank == 0:
            writer, alpha = make_qwen38_writer(alpha=1.)
            writer.load_state_dict(tensors, strict=True)
            writer = writer.to(device).float().eval().requires_grad_(False)
            alpha = alpha.to(device).eval().requires_grad_(False)
        initial_parameters = [(p, p._version) for p in writer.parameters()] if writer is not None else []
        code = [Path(__file__), Path(__file__).with_name("qwen38_native_a800_forward.py"),
                Path(__file__).with_name("qwen38_native_a800_session.py"),
                REPO / "src/delta_ngram/qwen38_shortchain_protocol.py"]
        identity = {
            "base_model": session.identity, "tokenizer": manifest["tokenizer_files"],
            "addressing": "native_PLE_ids_plus_one;one_owner;no_external_rehash",
            "writer": {"checkpoint_sha256": options.writer_sha256, "tensor_sha256": source["writer_tensor_state_hash"],
                       "alpha": 1., "optimizer_step": 4172, "completed_loss_targets": 99997072, "branch": "full"},
            "operator": "unchanged_D1;one_independent_example_per_update;FP32",
            "trajectory": {"protocol": SHORTCHAIN_PROTOCOL, "method": "writer", "task_order": ["sme", "nav"],
                           "H_encoding": "memory_aware_detached_current_prewrite_Delta", "evidence_batch_size": 1,
                           "encoded_manifest_sha256": options.encoded_manifest_sha256},
            "shortchain_plan": plan, "native_runtime": session.versions,
            "code_sha256": {p.relative_to(REPO).as_posix(): digest(p) for p in code},
        }
        writes, real_tokens = 0, 0
        evaluations, events, resume_receipt, feedback_receipt = {}, [], None, None
        restore_candidate = None
        replay_pending = False
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()

        def cursor():
            value = {"writes_completed": writes, "task_index": writes // 32, "example_index": writes % 32,
                     "real_evidence_tokens": real_tokens, "phase": "bounded_shortchain"}
            validate_shortchain_cursor(value, plan)
            return value

        def log(value):
            if rank == 0:
                with (options.output / "events.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(value, allow_nan=False) + "\n")
                print(json.dumps(value, allow_nan=False), flush=True)

        def fingerprints(current):
            local = state_fingerprint(current) if first else None
            all_values = [None] * dist.get_world_size()
            dist.all_gather_object(all_values, local, group=control)
            require(all_values[0] == all_values[1] and all_values[0] is not None
                    and all(x is None for x in all_values[2:]), "TP state replicas disagree")
            return all_values[0]

        def save(name):
            fingerprint = fingerprints(state)
            if rank == 0:
                payload = state_payload(state)
                details = validate_state_payload(payload, expected_writes=writes)
                path = options.output / (name + ".state.pt")
                require(not path.exists(), "refusing to overwrite shortchain checkpoint")
                torch.save({"protocol": SHORTCHAIN_PROTOCOL, "identity": identity, "cursor": cursor(),
                            "state": payload, "state_fingerprint": fingerprint,
                            "evaluation_receipts": evaluations.copy(), "writer_frozen": True,
                            "optimizer_steps": 0, "layout": asdict(session.layout)}, path)
                write_json(options.output / (name + ".json"),
                           {"path": str(path), "sha256": digest(path), "state_fingerprint": fingerprint,
                            "cursor": cursor(), "state_summary": details})
            dist.barrier(group=control)
            return fingerprint

        def evaluate(name, domains, *, longest=False):
            signature = save(name)
            rows = ([max((r for r in data["queries"] if r["domain"] == domain),
                         key=lambda r: len(r["input_ids"])) for domain in domains] if longest
                    else select_queries(data["queries"], domains, 1))
            destination = options.output / "tf" / name
            if rank == 0:
                destination.mkdir(parents=True, exist_ok=False)
            dist.barrier(group=control)
            checkpoint = {"path": name, "sha256": canonical_hash({"identity": identity, "cursor": cursor()})}
            capture = LayerCapture(session.models, SimpleNamespace(sequence_parallel=True), destination,
                dtype="bfloat16", last_prompt_only=False, checkpoint=checkpoint,
                checkpoint_kind="frozen_writer_fixed_FP32_Delta")
            metrics, records, local_files = [], [], []
            before = immutable_guard(state) if first else None
            try:
                for index, row in enumerate(rows):
                    batch, _ = reference_batch(row, pad_token_id=int(session.tokenizer.pad_token_id),
                                               pad_multiple=2, device=device)
                    valid = torch.arange(batch["input_ids"].shape[1], device=device)[None] < len(row["input_ids"])
                    metadata = query_metadata(row)
                    records.append(metadata)
                    capture.begin(index, metadata, row, batch["input_ids"].shape[1])
                    tick = time.monotonic()
                    result = native_forward(session, batch, valid, state=state)
                    require(result["token_count"] == row["target_token_count"], "TF target count changed")
                    capture.finish()
                    path = destination / "representations" / f"query-{index:04d}-rank-{rank:02d}.pt"
                    local_files.append({"path": path.relative_to(destination).as_posix(),
                                        "sha256": digest(path), "bytes": path.stat().st_size})
                    metrics.append({**metadata, **result, "seconds": time.monotonic() - tick,
                                    "input_tokens": len(row["input_ids"])})
                if before is not None:
                    assert_unchanged(before)
                require(fingerprints(state) == signature, "TF changed full state fingerprint")
                files = [None] * dist.get_world_size()
                dist.all_gather_object(files, local_files, group=control)
                if rank == 0:
                    prediction = destination / "predictions.jsonl"
                    prediction.write_text("".join(json.dumps(x, allow_nan=False) + "\n" for x in metrics), encoding="utf-8")
                    grouped = {}
                    for domain in domains:
                        chosen = [r for r in metrics if r["domain"] == domain]
                        total = sum(r["token_count"] for r in chosen)
                        loss = math.fsum(r["loss_sum"] for r in chosen)
                        grouped[domain] = {"n": len(chosen), "token_count": total, "loss_sum": loss,
                                           "answer_nll": loss / total, "row_ids": [r["task_id"] for r in chosen]}
                    write_json(destination / "summary.json", {"status": "completed", "identity": identity,
                        "cursor": cursor(), "domains": grouped, "capture_layers": True,
                        "capture_checkpoint": checkpoint, "encoded_records": records,
                        "state_fingerprint": signature, "state_fingerprint_after": signature,
                        "predictions_sha256": digest(prediction), "capture_files": [x for group in files for x in group],
                        "topology": {"tp": 2, "pp": 4, "ep": 2, "cp": 1, "dp": 1},
                        "task_count": len(rows), "completed_tasks": len(metrics), "state_read_only": True,
                        "generation_accuracy_measured_here": False})
            finally:
                capture.close()
            dist.barrier(group=control)
            evaluations[name] = {"summary_sha256": digest(destination / "summary.json"),
                                 "domains": list(domains), "query_count": len(rows), "writes_completed": writes}
            log({"kind": "evaluation_complete", "name": name, **evaluations[name]})

        def one_write(current, row, *, ordinal, feedback=False):
            nonlocal feedback_receipt
            tick = time.monotonic()
            mb = evidence_batch([row], pad_token_id=int(session.tokenizer.pad_token_id), device=device)
            result = native_forward(session, mb["batch"], mb["valid_mask"], state=current,
                                    evidence=True, encoding_kind="memory")
            length = mb["lengths"][0]
            h = result["hidden"][:, :length].detach()
            require(bool(h.isfinite().all()), "nonfinite memory-aware evidence hidden")
            h_sha = tensor_state_hash({"H": h})
            base_difference = None
            if feedback:
                base = native_forward(session, mb["batch"], mb["valid_mask"], evidence=True, encoding_kind="base")
                difference = h.float() - base["hidden"][:, :length].float()
                base_difference = {"max_abs": float(difference.abs().max()),
                                   "relative_l2": float(torch.linalg.vector_norm(difference.double()) /
                                                        torch.linalg.vector_norm(base["hidden"][:, :length].double()))}
                require(base_difference["max_abs"] > 0., "nonempty memory had no observed encoding effect")
            record = None
            touched = None
            if rank == 0:
                physical = result["physical_rows"][:, :length]
                prior, counts = current.lookup(physical)
                read_nonzero = int(prior.ne(0).any(dim=-1).sum())
                features = writer_evidence_features(h, mode="rmsnorm", eps=1e-6)
                current = write_memory_aware_native_block(writer, current, features, physical,
                    torch.ones(features.shape[:2], dtype=torch.bool, device=device), alpha)
                require(bool(current.values.isfinite().all()) and int(current.block_clock[0]) == ordinal,
                        "D1 state/clock invalid")
                touched = torch.unique(physical[physical.ne(0)], sorted=True)
                record = {"kind": "write", "write_index": ordinal, "task": row["task"],
                          "source_row_index": row["source_row_index"], "input_token_ids_sha256": row["input_token_ids_sha256"],
                          "tokens": length, "padded_tokens": mb["padded_tokens"], "H_sha256": h_sha,
                          "read_nonzero_row_positions": read_nonzero, "rows": current.rows.numel(),
                          "state_l2": float(torch.linalg.vector_norm(current.values.double())),
                          "counts_max": float(current.counts.max()), "block_clock": int(current.block_clock[0])}
                if feedback:
                    require(read_nonzero > 0, "feedback probe did not read nonzero Delta")
                    feedback_receipt = {"ordinal": ordinal, "read_nonzero_row_positions": read_nonzero,
                                        "memory_H_sha256": h_sha, **base_difference, "state_read_only_during_encoding": True}
            if first:
                current = sync_written_rows(current, touched, device=device)
            dist.barrier(group=control)
            if rank == 0:
                record["seconds"] = time.monotonic() - tick
            return current, record, h_sha

        with native_dispatch(session) as dispatch, torch.no_grad():
            if rank == 0:
                write_json(options.output / "registration.json", {"identity": identity, "dispatch": dispatch,
                    "plan": plan, "extra_long_context_probes": ["longest_math", "longest_code"],
                    "canonical_writes": 64, "recovery_replay_writes": 1, "encoding_feedback_base_calls": 1,
                    "numerical_scope": "bounded_A800_path; H200_comparison_is_separate"})
            evaluate("base", ["math", "code"])
            evaluate("long_context_base", ["math", "code"], longest=True)
            for task_index, task in enumerate(("sme", "nav")):
                evaluate(f"task_{task_index + 1:02d}_{task}_before", [task])
                for row in data[task][:32]:
                    ordinal = writes + 1
                    state, event, h_sha = one_write(state, row, ordinal=ordinal, feedback=ordinal == 2)
                    if replay_pending:
                        repeated, repeat_event, repeat_h = one_write(restore_candidate, row, ordinal=ordinal)
                        if first:
                            assert_state_payload_equal(state_payload(state), state_payload(repeated))
                        require(h_sha == repeat_h, "reloaded checkpoint changed the next evidence H")
                        equal = fingerprints(state) == fingerprints(repeated)
                        require(equal, "reloaded checkpoint changed next D1")
                        if rank == 0:
                            resume_receipt = {"saved_after": writes, "replayed_next_write": ordinal,
                                "full_state_bitwise_equal": True, "evidence_H_bitwise_equal": True,
                                "next_D1_bitwise_equal": True, "state_fingerprint": state_fingerprint(state),
                                "diagnostic_replay_seconds": repeat_event["seconds"]}
                        restore_candidate = None
                        replay_pending = False
                    writes += 1
                    real_tokens += len(row["input_ids"])
                    if rank == 0:
                        events.append(event)
                    log(event)
                    if writes == 16:
                        signature = save("resume_probe_16")
                        saved = torch.load(options.output / "resume_probe_16.state.pt", map_location="cpu", weights_only=True)
                        require(saved["identity"] == identity and saved["cursor"] == cursor()
                                and saved["state_fingerprint"] == signature, "checkpoint identity/cursor changed")
                        if first:
                            restore_candidate = restore_state(saved["state"], device=device)
                            assert_state_payload_equal(state_payload(state), state_payload(restore_candidate))
                        replay_pending = True
                evaluate(f"after_task_{task_index + 1:02d}_{task}", [*("sme", "nav")[:task_index + 1], "math", "code"])
            final_signature = save("final")
            session.assert_frozen()
            require(all(p._version == version and not p.requires_grad and p.grad is None
                        for p, version in initial_parameters), "writer changed or acquired gradients")
            gpu = {"rank": rank, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
            peaks = [None] * dist.get_world_size()
            dist.all_gather_object(peaks, gpu, group=control)
            if rank == 0:
                require(tensor_state_hash(writer.state_dict()) == source["writer_tensor_state_hash"], "writer tensors changed")
                require(writes == 64 and resume_receipt is not None and feedback_receipt is not None,
                        "bounded acceptance obligations are incomplete")
                require(sum(x["query_count"] for x in evaluations.values()) == 13, "query budget changed")
                write_json(options.output / "COMPLETE.json", {"protocol": SHORTCHAIN_PROTOCOL, "status": "completed",
                    "identity": identity, "cursor": cursor(), "evaluation_receipts": evaluations,
                    "final_state_fingerprint": final_signature, "checkpoint_recovery": resume_receipt,
                    "memory_encoding_feedback": feedback_receipt, "gpu_peaks": peaks,
                    "wall_seconds_callback": time.monotonic() - started, "canonical_writes": 64,
                    "diagnostic_replay_writes": 1, "encoding_forward_calls": 66, "TF_queries": 13,
                    "write_seconds_sum": math.fsum(x["seconds"] for x in events),
                    "backward_calls": 0, "optimizer_steps": 0, "generation_calls": 0})

    return run_a800_session(options, run)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cl-repo", "delta-repo", "model-path", "verification", "encoded-dir", "writer-checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--encoded-manifest-sha256", required=True)
    p.add_argument("--writer-sha256", required=True)
    options = p.parse_args()
    options.tokens = 8192
    execute(options)


if __name__ == "__main__":
    main()

"""Bounded ordinary main100M writer B1/B12 timing and numerical comparison.

Three independent zero-start states consume the same 24 supports in order.
Static T0 evidence may be encoded together, but every D1 update is B1 and
uses true tokens only. Differences are reported, never silently accepted.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from delta_ngram.qwen38_seven_forward import require, digest, state_payload, state_fingerprint, STATE_FIELDS
from delta_ngram.qwen38_writer_batch_benchmark import (
    BENCHMARK_PROTOCOL, MODES, select_frames, make_jobs, padded_batch, compare_tensor,
)
from qwen38_seven_task_forward_writer import load_encoded, writer_source, setup_cl, write_json

MAIN_ARCHIVE_SHA = "523b951487cee0ec726077db0104a8fc929f4629bdb4ca46604295a4b78fbfe3"
MAIN_WRITER_SHA = "d83d2ba750913ec5c9e1a3fc43385a7c16fbe683a106934d7f8f21f853fdb460"


def execute(options):
    import torch
    from qwen38_native_a800_session import run_a800_session
    require(options.writer_sha256 == MAIN_ARCHIVE_SHA, "ordinary benchmark needs the main100M checkpoint")
    manifest, data = load_encoded(options)
    rows = select_frames(data)
    source = writer_source(options)
    require(source["writer_sha256"] == MAIN_WRITER_SHA, "ordinary writer parameters differ")
    plans = {mode: make_jobs(rows, mode) for mode in MODES}
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
        from delta_ngram.qwen38_native import make_qwen38_writer, write_native_block
        from delta_ngram.qwen38_training import writer_evidence_features
        from delta_ngram.qwen38_probe import tensor_state_hash
        from qwen38_native_a800_forward import native_forward, native_dispatch, sync_written_rows
        rank, device, control = dist.get_rank(), session.device, session.control_group
        require(session.world_communication_preflight["status"] == "passed", "world bootstrap missing")
        first = ps.is_pipeline_first_stage()
        writer = alpha = None
        if rank == 0:
            writer, alpha = make_qwen38_writer(alpha=1.)
            writer.load_state_dict(source["writer"], strict=True)
            writer = writer.to(device).float().eval().requires_grad_(False)
            alpha = alpha.to(device).eval().requires_grad_(False)
        versions = [(p, p._version) for p in writer.parameters()] if writer is not None else []
        captures, states, summaries = {}, {}, {}
        warmups = []
        native_calls = 0
        started = time.monotonic()

        def encode(job):
            mb = padded_batch(job["rows"], job["sequence_length"],
                              pad_token_id=int(session.tokenizer.pad_token_id), device=device)
            result = native_forward(session, mb["batch"], mb["valid_mask"],
                                    evidence=True, encoding_kind="base")
            return mb, result

        with native_dispatch(session), torch.no_grad():
            # Warm up every used B,S shape, without advancing any memory.
            seen = set()
            for mode in MODES:
                for job in plans[mode]:
                    key = (len(job["rows"]), job["sequence_length"])
                    if key in seen:
                        continue
                    seen.add(key)
                    tick = time.monotonic()
                    mb, result = encode(job)
                    torch.cuda.synchronize(device)
                    dist.barrier(group=control)
                    warmups.append({"batch_size": key[0], "sequence_length": key[1],
                                    "seconds": time.monotonic() - tick, "D1_writes": 0})
                    native_calls += 1
                    del mb, result

            for mode in MODES:
                state = LazyDecayState.from_batch(session.layout.empty_state(1, device)) if first else None
                records, samples = [], []
                torch.cuda.reset_peak_memory_stats(device)
                for job_index, job in enumerate(plans[mode]):
                    dist.barrier(group=control)
                    torch.cuda.synchronize(device)
                    tick = time.monotonic()
                    mb, result = encode(job)
                    torch.cuda.synchronize(device)
                    encoded_at = time.monotonic()
                    native_calls += 1
                    phases = {"encoding": encoded_at - tick, "features": 0., "D1": 0., "TP_sync": 0.}
                    for lane, index in enumerate(job["indices"]):
                        length = len(rows[index]["input_ids"])
                        touched = None
                        if rank == 0:
                            h = result["hidden"][lane:lane + 1, :length]
                            physical = result["physical_rows"][lane:lane + 1, :length]
                            require(h.shape == (1, length, 2560) and physical.shape[:2] == (1, length),
                                    "D1 evidence layout differs")
                            feature_tick = time.monotonic()
                            features = writer_evidence_features(h, mode="rmsnorm", eps=1e-6)
                            torch.cuda.synchronize(device)
                            feature_done = time.monotonic()
                            state = write_native_block(writer, state, features, physical,
                                torch.ones((1, length), dtype=torch.bool, device=device), alpha)
                            torch.cuda.synchronize(device)
                            write_done = time.monotonic()
                            phases["features"] += feature_done - feature_tick
                            phases["D1"] += write_done - feature_done
                            touched = torch.unique(physical[physical.ne(0)], sorted=True)
                            require(int(state.block_clock[0]) == index + 1, "chronological D1 count changed")
                        sync_tick = time.monotonic()
                        if first:
                            state = sync_written_rows(state, touched, device=device)
                        torch.cuda.synchronize(device)
                        phases["TP_sync"] += time.monotonic() - sync_tick
                    dist.barrier(group=control)
                    elapsed = time.monotonic() - tick
                    # All CPU capture/hash/telemetry costs are outside measured work.
                    capture_tick = time.monotonic()
                    if rank == 0:
                        for lane, index in enumerate(job["indices"]):
                            length = len(rows[index]["input_ids"])
                            h = result["hidden"][lane:lane + 1, :length].detach().cpu().clone()
                            physical = result["physical_rows"][lane:lane + 1, :length].detach().cpu().clone()
                            require(bool(h.isfinite().all()), "nonfinite evidence H")
                            samples.append({"index": index, "input_sha256": rows[index]["input_token_ids_sha256"],
                                            "H": h, "physical_rows": physical})
                    gathered = [None] * dist.get_world_size()
                    dist.all_gather_object(gathered, {**phases, "total": elapsed}, group=control)
                    overhead = time.monotonic() - capture_tick
                    if rank == 0:
                        record = {"mode": mode, "job": job_index, "indices": job["indices"],
                                  "input_tokens": mb["real_tokens"], "padded_tokens": mb["padded_tokens"],
                                  "batch_size": len(job["rows"]), "sequence_length": job["sequence_length"],
                                  "seconds_rank0": {**phases, "total": elapsed},
                                  "seconds_max_across_ranks": {k: max(x[k] for x in gathered) for k in gathered[0]},
                                  "capture_and_telemetry_seconds": overhead}
                        records.append(record)
                        print(json.dumps(record), flush=True)
                    del mb, result
                signature = state_fingerprint(state) if first else None
                signatures = [None] * dist.get_world_size()
                dist.all_gather_object(signatures, signature, group=control)
                require(signatures[0] == signatures[1] and signatures[0] is not None
                        and all(x is None for x in signatures[2:]), "TP state replicas differ")
                peaks = [None] * dist.get_world_size()
                dist.all_gather_object(peaks, {"rank": rank, "allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "reserved_bytes": torch.cuda.max_memory_reserved(device)}, group=control)
                if rank == 0:
                    captures[mode], states[mode] = samples, state_payload(state)
                    seconds = math.fsum(r["seconds_max_across_ranks"]["total"] for r in records)
                    summaries[mode] = {"D1_writes": 24, "encoding_calls": len(records), "records": records,
                        "measured_seconds": seconds, "seconds_per_frame": seconds / 24,
                        "real_tokens_per_second": sum(len(r["input_ids"]) for r in rows) / seconds,
                        "state_fingerprint": signatures[0], "gpu_peaks": peaks,
                        "phase_seconds_rank0": {key: math.fsum(r["seconds_rank0"][key] for r in records)
                                                for key in ("encoding", "features", "D1", "TP_sync")}}
                del state
                dist.barrier(group=control)

        if rank == 0:
            comparisons = {}
            for candidate, reference in (("b1_matched_padding", "b1_canonical"),
                                         ("b12", "b1_matched_padding"), ("b12", "b1_canonical")):
                per_sample = []
                for actual, expected in zip(captures[candidate], captures[reference]):
                    require(actual["index"] == expected["index"] and actual["input_sha256"] == expected["input_sha256"]
                            and torch.equal(actual["physical_rows"], expected["physical_rows"]),
                            "batching changed true-token native addresses or input order")
                    per_sample.append({"index": actual["index"], **compare_tensor(actual["H"], expected["H"])})
                state_errors = {field: compare_tensor(states[candidate][field], states[reference][field])
                                for field in STATE_FIELDS}
                require(all(state_errors[field]["bitwise_equal"] for field in STATE_FIELDS if field != "values"),
                        "batching changed state addresses, counts or lazy clocks")
                comparisons[candidate + "_versus_" + reference] = {
                    "H_per_sample": per_sample, "state_fields": state_errors,
                    "all_H_bitwise_equal": all(r["bitwise_equal"] for r in per_sample),
                    "all_state_fields_bitwise_equal": all(r["bitwise_equal"] for r in state_errors.values()),
                    "candidate_speedup": summaries[reference]["measured_seconds"] / summaries[candidate]["measured_seconds"]}
            require(all(p._version == version and p.grad is None for p, version in versions)
                    and tensor_state_hash(writer.state_dict()) == MAIN_WRITER_SHA, "frozen ordinary writer changed")
            require(native_calls == len(warmups) + 50 and native_calls <= 100, "forward-call budget differs")
            torch.save({"protocol": BENCHMARK_PROTOCOL, "captures": captures, "states": states},
                       options.output / "numerical_evidence.pt")
            write_json(options.output / "COMPLETE.json", {
                "protocol": BENCHMARK_PROTOCOL, "status": "completed", "scientific_endpoint": False,
                "writer": {"checkpoint_sha256": MAIN_ARCHIVE_SHA, "tensor_sha256": MAIN_WRITER_SHA,
                           "optimizer_step": 4172, "completed_loss_targets": 99997072, "alpha": 1.},
                "encoding": "static_T0_only; independent causal documents; positions reset per lane",
                "D1": "one true-length frame at a time; shared chronological state within each mode; FP32",
                "support_selection": [{"task": r["task"], "source_row_index": r["source_row_index"],
                                       "tokens": len(r["input_ids"]), "sha256": r["input_token_ids_sha256"]} for r in rows],
                "encoded_manifest_sha256": options.encoded_manifest_sha256, "warmups": warmups,
                "total_D1_writes": 72, "native_encoding_calls": native_calls, "modes": summaries,
                "comparisons": comparisons, "automatic_B12_acceptance": False,
                "timing_notes": ["B1 canonical uses each frame's minimum even padding, matching prior delta B1 padding policy.",
                    "B1 matched padding and B12 use the same per-group sequence length; this is the controlled batch-shape comparison.",
                    "Times include native H transport, RMS features, sequential D1, TP row sync, and final control barrier.",
                    "CPU evidence capture, hashes, telemetry, checkpoint serialization, warmups, and model loading are excluded.",
                    "This starts from empty state and does not measure 666747-row capacity or long-run throughput.",
                    "Different main/full writer weights and T0/memory encodings prevent attributing timing differences solely to batching."],
                "numerical_evidence_sha256": digest(options.output / "numerical_evidence.pt"),
                "base_identity": session.identity, "runtime": session.versions,
                "world_communication_preflight": session.world_communication_preflight,
                "callback_seconds": time.monotonic() - started, "TF_queries": 0, "generation_calls": 0,
                "backward_calls": 0, "optimizer_steps": 0, "H200_numerical_equivalence_claimed": False,
                "code_sha256": {"runner": digest(__file__), "protocol": digest(REPO / "src/delta_ngram/qwen38_writer_batch_benchmark.py")}})
        dist.barrier(group=control)
    return run_a800_session(options, run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cl-repo", "delta-repo", "model-path", "verification", "encoded-dir", "writer-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("encoded-manifest-sha256", "writer-sha256"):
        parser.add_argument("--" + name, required=True)
    options = parser.parse_args()
    options.tokens = 8192
    execute(options)

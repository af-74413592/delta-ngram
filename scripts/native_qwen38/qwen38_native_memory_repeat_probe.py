"""Independent repeat-only controls on the immutable 80M source checkpoint.

This separate diagnostic is not a training gate. It preserves completed V2
costs, injects no perturbations, and scores four final reference/repeat states.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import random
import sys
import time

import torch

from qwen38_native_norm_probe import canonical_hash, file_info, require
from qwen38_native_train_staged_alpha100 import clone_cpu, write_json
from qwen38_native_train_batched import restore_rng_state
from qwen38_native_alpha100_policy import state_digest
from qwen38_native_train_memory_fork import comparable_reader_runtime
import qwen38_native_memory_fork_policy as fork
from qwen38_native_memory_forward_probe import prepare as prepare_v1

PROTOCOL = "qwen38_native_memory_repeat_probe_v3"


def execution_protocol():
    from qwen38_native_memory_repeat_schedule import repeat_probe_protocol
    return repeat_probe_protocol()


def diagnostic_protocol():
    return {
        "protocol": PROTOCOL,
        "source": "immutable_actual80M_alpha100_writer_no_residual",
        "source_optimizer_step": fork.SOURCE_STEP,
        "source_model_targets": fork.SOURCE_TARGETS,
        "sampling": "same registered next 12 complete trajectories as V1 and V2",
        "base_control": "two independently evolved states sharing actual legacy H12 base features",
        "memory_control": "two independent reference/repeat trajectories with causal H_mem from empty",
        "state_perturbations_injected": 0,
        "final_query_conditions": ["memory_ref", "memory_repeat", "base_ref", "base_repeat"],
        "final_query_scope": "same final unseen query of six trajectories with more than one write",
        "final_query_batch_limit": 3,
        "CE_targets_per_condition": 3050, "CE_targets": 12200,
        "backward_calls": 0, "optimizer_updates": 0,
        "training_gate": False, "new_stability_threshold": False,
        "nonfinite": "fail_immediately_no_retry", "automatic_retry": False,
        "automatic_training_start": False,
        "execution": execution_protocol(),
        "execution_protocol_sha256": canonical_hash(execution_protocol()),
    }


@contextmanager
def forward_only_guard():
    blocked = []
    originals = [
        (torch.optim.Optimizer, "__init__", torch.optim.Optimizer.__init__),
        (torch.Tensor, "backward", torch.Tensor.backward),
        (torch.autograd, "backward", torch.autograd.backward),
        (torch.autograd, "grad", torch.autograd.grad),
    ]
    def forbid(name):
        def fail(*args, **kwargs):
            blocked.append(name)
            raise RuntimeError(f"supplemental probe forbids {name}")
        return fail
    try:
        for owner, name, original in originals:
            setattr(owner, name, forbid("optimizer_construction" if owner is torch.optim.Optimizer else name))
        with torch.no_grad():
            yield blocked
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def prepare(options):
    """Reuse v1's immutable source and sampler validation, changing only protocol."""
    prepared = prepare_v1(options)
    manifest = prepared["manifest"]
    manifest["protocol"] = PROTOCOL
    manifest["diagnostic_protocol"] = diagnostic_protocol()
    manifest["final_query_registration"] = register_final_queries(prepared["plans"])
    manifest["encoding_registration"] = register_encoding_calls(prepared["plans"], manifest["selection"]["assignments"])
    require(options.prior_supplemental_output != options.output, "V3 output cannot replace completed V2 output")
    manifest["prior_supplemental"] = prior_supplemental_receipt(options.prior_supplemental_output, manifest)
    manifest["run_id"] = canonical_hash({
        "output": str(options.output),
        "source": manifest["source80_checkpoint"],
        "code": manifest["code_manifest_sha256"],
        "protocol": diagnostic_protocol(),
        "selection": manifest["selection"],
        "final_query_registration": manifest["final_query_registration"],
        "encoding_registration": manifest["encoding_registration"],
        "prior_supplemental": manifest["prior_supplemental"],
    })
    return prepared


def prior_supplemental_receipt(directory, manifest):
    """Read sealed V2 results without changing or refunding their budget."""
    report_path, complete_path = directory / "report.json", directory / "COMPLETE.json"
    report = json.loads(report_path.read_bytes())
    complete = json.loads(complete_path.read_bytes())
    require(complete.get("complete") is True
            and complete.get("protocol") == report.get("protocol") == "qwen38_native_memory_supplemental_probe_v2"
            and complete["report"] == file_info(report_path), "prior V2 report receipt does not verify")
    source = manifest["source80_checkpoint"]
    require(all(complete["source80_checkpoint"][key] == source[key] for key in ("bytes", "sha256"))
            and report["selection"] == manifest["selection"]
            and report["final_query_registration"]["records"] == manifest["final_query_registration"]["records"],
            "V3 must replay V2's immutable source, selected trajectories and final query frames")
    budget = report["budget"]
    require(budget == complete["budget"] and budget["CE_targets"] == complete["CE_targets"] == 12200
            and budget["encoding_calls"] == 92 and budget["backward_calls"] == budget["optimizer_updates"] == 0,
            "completed V2 historical budget differs from its registered 92 encodings and 12200 CE")
    return {"directory": str(directory), "report": file_info(report_path), "complete": file_info(complete_path),
            "budget": budget, "preserved_without_refund": True}


def register_encoding_calls(plans, assignments):
    require(len(assignments) == 4 and sorted(index for values in assignments for index in values) == list(range(12)),
            "repeat probe requires the original four-rank trajectory assignment")
    max_writes = max(sum(step["evidence"] is not None for index in values for step in plans[index]) for values in assignments)
    h12_calls = (max_writes + 11) // 12
    waves = max(map(len, plans))
    calls_per_memory_track = 0
    for ordinal in range(waves):
        ready = max(sum(ordinal < len(plans[index]) and plans[index][ordinal]["evidence"] is not None
                        for index in values) for values in assignments)
        calls_per_memory_track += (ready + 2) // 3
    total = h12_calls + 2 * calls_per_memory_track
    require(h12_calls == 2 and calls_per_memory_track == 15 and total == 32,
            "registered repeat-only plan must require exactly 32 encoding calls")
    return {"H12_base_collective_calls": h12_calls, "memory_reference_collective_calls": calls_per_memory_track,
            "memory_repeat_collective_calls": calls_per_memory_track, "total_encoding_calls": total,
            "base_twin_extra_encoding_calls": 0, "G_feature_comparison_calls": 0, "perturbation_calls": 0}


def register_final_queries(plans):
    from delta_ngram.qwen38_staged_batching import collate_staged_frames
    from qwen38_native_memory_repeat_schedule import FINAL_CONDITIONS as CONDITIONS
    records = []
    for index, plan in enumerate(plans):
        if sum(step["evidence"] is not None for step in plan) <= 1:
            continue
        require(plan[-1]["evidence"] is None and plan[-1]["query"] is not None,
                "supplemental final query must follow all writes and never be written")
        frame = plan[-1]["query"]
        mb = collate_staged_frames([frame], kind="query", device="cpu")
        records.append({"trajectory_index": index, "frame_index": len(plan) - 1,
                        "frame_sha256": canonical_hash(frame),
                        "target_tokens": mb["loss_target_tokens"]})
    require(len(records) == 6, "registered first group must have six evolved trajectories")
    per_condition = sum(row["target_tokens"] for row in records)
    total = per_condition * len(CONDITIONS)
    require(per_condition == 3050 and total == 12200, "repeat-only final-query registration must be exactly 3050 targets per condition")
    return {"records": records, "conditions": list(CONDITIONS),
            "CE_targets_per_condition": per_condition, "CE_targets": total,
            "query_scope": "final unseen block of each evolved training-corpus trajectory"}


def precharge_score(ledger, registration, *, kind, ce_targets, real_input_tokens, padded_input_tokens):
    require(kind in registration["conditions"] and type(ce_targets) is int and ce_targets > 0,
            "invalid global final-query CE charge")
    require(all(type(value) is int and value >= 0 for value in (real_input_tokens, padded_input_tokens))
            and real_input_tokens <= padded_input_tokens and ce_targets <= real_input_tokens,
            "invalid final-query input token charge")
    by_kind = ledger.setdefault("final_query_by_kind", {})
    cell = by_kind.get(kind, {"calls": 0, "CE_targets": 0,
                             "real_input_tokens": 0, "padded_input_tokens": 0})
    require(cell["CE_targets"] + ce_targets <= registration["CE_targets_per_condition"]
            and ledger["CE_targets"] + ce_targets <= registration["CE_targets"],
            "supplemental final-query CE reservation exceeded")
    by_kind[kind] = cell
    ledger["CE_targets"] += ce_targets
    ledger["final_query_calls"] += 1
    ledger["final_query_real_input_tokens"] += real_input_tokens
    ledger["final_query_padded_input_tokens"] += padded_input_tokens
    cell["calls"] += 1
    cell["CE_targets"] += ce_targets
    cell["real_input_tokens"] += real_input_tokens
    cell["padded_input_tokens"] += padded_input_tokens


def execute(options, prepared):
    from qwen38_native_ep4_session import run_ep4_session
    from delta_ngram.qwen38_native import make_qwen38_writer
    from delta_ngram.qwen38_precision import Qwen38SplitPLEPrecision
    from delta_ngram.qwen38_probe import tensor_state_hash
    from qwen38_native_dispatch import native_hc_dispatch
    from qwen38_native_router import native_router_dispatch
    from qwen38_native_ep4_session import runtime_identity
    from qwen38_native_memory_probe_runtime import collect_probe_h12_base
    from qwen38_native_memory_repeat_schedule import run_memory_repeat_probe, score_repeat_query_states
    from delta_ngram.alpha_memory import LazyDecayState

    rank = int(os.environ.get("RANK", "-1"))
    require(0 <= rank < 4 and int(os.environ.get("WORLD_SIZE", "0")) == 4,
            "supplemental probe requires four torchrun ranks")
    manifest = prepared["manifest"]
    marker = options.output / "input_manifest.json"
    ledger = {
        "protocol": "qwen38_memory_repeat_probe_ledger_v3",
        "CE_targets": 0, "backward_calls": 0, "optimizer_updates": 0,
        "encoding_calls": 0, "encoding_real_input_tokens": 0,
        "encoding_padded_input_tokens": 0, "by_kind": {},
        "final_query_calls": 0, "final_query_real_input_tokens": 0,
        "final_query_padded_input_tokens": 0,
        "reserved_final_query_CE_targets": manifest["final_query_registration"]["CE_targets"],
        "failed_or_partial_accesses_are_not_refunded": True,
        "prior_completed_supplemental_v2": manifest["prior_supplemental"],
        "reserved_encoding_calls": manifest["encoding_registration"]["total_encoding_calls"],
    }
    started = time.monotonic()
    status = "running"
    immutable_before = state_digest(prepared["source_arm"])

    def save_ledger():
        prior = manifest["prior_supplemental"]["budget"]
        ledger["cumulative_v2_v3"] = {key: prior[key] + ledger[key]
                                      for key in ("CE_targets", "encoding_calls", "encoding_real_input_tokens",
                                                  "encoding_padded_input_tokens", "backward_calls", "optimizer_updates")}
        if rank == 0:
            write_json(options.output / "budget_ledger.json", ledger)

    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        manifest["diagnostic_protocol"] = diagnostic_protocol()
        write_json(marker, manifest)
        write_json(options.output / "diagnostic_protocol.json", diagnostic_protocol())
        save_ledger()
    else:
        deadline = time.monotonic() + 300
        while not marker.is_file():
            require(time.monotonic() < deadline, "rank0 did not create supplemental manifest")
            time.sleep(.1)
    options.tokens = 640

    def run(session):
        nonlocal status
        import numpy as np
        import torch.distributed as dist
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        device = session.device
        def gather(value):
            values = [None] * 4
            dist.all_gather_object(values, value)
            return values
        def agree(value, message):
            values = gather(value)
            require(all(item == values[0] for item in values), message)
            return values[0]
        def capture_rng():
            return {"rank": rank, "python": random.getstate(), "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state(device).cpu(),
                    "tensor_parallel": clone_cpu(get_cuda_rng_tracker().get_states())}
        def restore_rng(value):
            restore_rng_state(value, rank=rank, device=device,
                              tensor_parallel_tracker=get_cuda_rng_tracker())
        def charge_encoding(row):
            require(row.get("ce_targets") == 0, "supplemental evidence encoding cannot charge CE")
            kind = row["kind"]
            agree(kind, "ranks disagree on supplemental encoding kind")
            costs = gather((row["real_input_tokens"], row["padded_input_tokens"]))
            real, padded = sum(x[0] for x in costs), sum(x[1] for x in costs)
            require(ledger["encoding_calls"] + 1 <= ledger["reserved_encoding_calls"],
                    "repeat-only encoding reservation exceeded")
            ledger["encoding_calls"] += 1
            ledger["encoding_real_input_tokens"] += real
            ledger["encoding_padded_input_tokens"] += padded
            cell = ledger["by_kind"].setdefault(kind, {"calls": 0, "real_input_tokens": 0,
                                                         "padded_input_tokens": 0})
            cell["calls"] += 1; cell["real_input_tokens"] += real; cell["padded_input_tokens"] += padded
            save_ledger()
        def charge_score(row):
            require(type(row.get("ce_targets")) is int and row["ce_targets"] >= 0,
                    "rank-local supplemental query CE count must be nonnegative")
            agree(row["kind"], "ranks disagree on final-query condition")
            costs = gather((row["ce_targets"], row["real_input_tokens"], row["padded_input_tokens"]))
            ce = sum(x[0] for x in costs); real = sum(x[1] for x in costs); padded = sum(x[2] for x in costs)
            precharge_score(ledger, manifest["final_query_registration"], kind=row["kind"],
                            ce_targets=ce, real_input_tokens=real, padded_input_tokens=padded)
            save_ledger()

        with ExitStack() as scope:
            hc = scope.enter_context(native_hc_dispatch(unwrap=True))
            router = scope.enter_context(native_router_dispatch(session.raw))
            precision = scope.enter_context(Qwen38SplitPLEPrecision(session.adapter))
            runtime = {"protocol": PROTOCOL, "native": runtime_identity(session),
                       "HC": hc.identity, "router": router["identity"], "precision": precision.identity(),
                       "code_sha256": manifest["code_manifest_sha256"],
                       "diagnostic_protocol": diagnostic_protocol()}
            require(comparable_reader_runtime(runtime) == comparable_reader_runtime(manifest["source_runtime_identity"]),
                    "supplemental reader/topology/HC/router/precision differs from source")
            runtime_hash = agree(canonical_hash(runtime), "rank runtime identities differ")
            if rank == 0:
                write_json(options.output / "runtime_identity.json", runtime)
            saved = prepared["source_arm"]
            with torch.random.fork_rng(devices=[]):
                writer, alpha = make_qwen38_writer(alpha=1.)
                writer.load_state_dict(saved["writer"], strict=True)
            writer = writer.to(device).float().requires_grad_(False).eval()
            alpha = alpha.to(device).requires_grad_(False).eval()
            require(tensor_state_hash(writer.state_dict()) == saved["writer_sha256"], "supplemental writer changed")
            local = [{"index": index, "plan": prepared["plans"][index],
                      "state": LazyDecayState.from_batch(session.layout.empty_state(1, device))}
                     for index in manifest["selection"]["assignments"][rank]]
            restore_rng(next(row for row in saved["rng_by_rank"] if row["rank"] == rank))
            before_rng = capture_rng()
            def on_final_states(tasks):
                return score_repeat_query_states(session, tasks, gather=gather, on_score=charge_score,
                                                 batch_limit=3)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            probe_started = time.monotonic()
            try:
                h12 = collect_probe_h12_base(session, local, gather=gather,
                                            on_encoding=charge_encoding, batch_limit=12)
                result = run_memory_repeat_probe(
                    session, writer, alpha, local, gather=gather, on_encoding=charge_encoding,
                    on_final_states=on_final_states,
                    h12_base_blocks=h12["blocks"], protocol=execution_protocol())
                require(result["protocol"] == execution_protocol(),
                        "supplemental execution protocol differs from registration")
                registration = manifest["final_query_registration"]
                require(result["ce_targets"] == ledger["CE_targets"] == registration["CE_targets"]
                        and result["final_query_trajectory_indices"] == sorted(
                            row["trajectory_index"] for row in registration["records"]),
                        "supplemental final-query CE or trajectory coverage differs")
                require(ledger["encoding_calls"] == ledger["reserved_encoding_calls"]
                        and result["state_perturbations_injected"] == 0,
                        "repeat-only encoding coverage differs or a perturbation was introduced")
                require(all(ledger["final_query_by_kind"][kind]["CE_targets"]
                            == registration["CE_targets_per_condition"]
                            for kind in registration["conditions"]),
                        "final-query condition coverage differs")
                torch.cuda.synchronize(device)
                require(state_digest(capture_rng()) == state_digest(before_rng),
                        "supplemental diagnostics advanced training RNG")
            finally:
                restore_rng(before_rng)
            require(tensor_state_hash(writer.state_dict()) == saved["writer_sha256"]
                    and all(not p.requires_grad and p.grad is None for p in writer.parameters()),
                    "supplemental diagnostics changed writer or acquired gradients")
            require(state_digest(prepared["source_arm"]) == immutable_before,
                    "loaded source training state changed")
            session.assert_frozen()
            status = "complete_repeat_forward_with_query_ce"
            timing_ranks = gather({"rank": rank, "wall_seconds": time.monotonic() - started,
                                   "probe_seconds": time.monotonic() - probe_started,
                                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
            timing = {"ranks": timing_ranks,
                      "wall_seconds_max_rank": max(row["wall_seconds"] for row in timing_ranks),
                      "probe_seconds_max_rank": max(row["probe_seconds"] for row in timing_ranks),
                      "GPU_occupied_hours": max(row["wall_seconds"] for row in timing_ranks) * 4 / 3600,
                      "diagnostic_GPU_occupied_hours": max(row["probe_seconds"] for row in timing_ranks) * 4 / 3600,
                      "definition": "rank max elapsed times four; not a GPU utilization measurement",
                      "CE_targets": ledger["CE_targets"], "backward_calls": 0, "optimizer_updates": 0}
            if rank == 0:
                require(file_info(options.source_checkpoint) == {
                    key: manifest["source80_checkpoint"][key] for key in ("bytes", "sha256")},
                    "source checkpoint file changed during supplemental diagnostics")
                report = {"protocol": PROTOCOL, "status": status, "diagnostics": result,
                          "h12_cache": ledger["by_kind"]["supplemental_h12_base"],
                          "final_query_registration": registration,
                          "selection": manifest["selection"], "budget": ledger, "timing": timing,
                          "source_writer_unchanged": True, "source_arm_state_unchanged": True,
                          "all_rank_RNG_unchanged": True, "runtime_identity_sha256": runtime_hash,
                          "diagnostic_protocol": diagnostic_protocol(), "automatic_training_start": False}
                write_json(options.output / "report.json", report)
                write_json(options.output / "timing_ledger.json", timing)
                write_json(options.output / "COMPLETE.json", {
                    "protocol": PROTOCOL, "status": status, "complete": True,
                    "report": file_info(options.output / "report.json"),
                    "source80_checkpoint": manifest["source80_checkpoint"],
                    "input_manifest": file_info(marker), "budget": ledger,
                    "CE_targets": ledger["CE_targets"], "backward_calls": 0,
                    "optimizer_updates": 0, "automatic_training_start": False})
                print(json.dumps({"status": status, "output": str(options.output), "budget": ledger}), flush=True)
            dist.barrier()
    try:
        with forward_only_guard() as blocked:
            run_ep4_session(options, run)
        require(not blocked, "supplemental forward-only guard observed forbidden calls")
    except BaseException as error:
        status = "failed_repeat_probe"
        save_ledger()
        if rank == 0:
            write_json(options.output / "FAILURE.json", {"protocol": PROTOCOL, "status": status,
                "error_type": type(error).__name__, "message": str(error), "budget": ledger,
                "automatic_retry": False, "automatic_training_start": False})
        raise
    finally:
        write_json(options.output / f"timing-rank-{rank}.json", {
            "rank": rank, "status": status, "wall_seconds": time.monotonic() - started,
            "CE_targets": ledger["CE_targets"], "backward_calls": 0, "optimizer_updates": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source-checkpoint", "source-input-manifest", "source-runtime-identity",
                 "source-evaluation-report", "source-evaluation-complete", "cache-root", "delta-repo", "output",
                 "prior-supplemental-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("model-path", "verification", "reference-training-log"):
        parser.add_argument("--" + name, type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--validate", action="store_true")
    actions.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    require(not options.execute or options.model_path and options.verification,
            "execute requires model-path and verification")
    for key, value in vars(options).items():
        if isinstance(value, Path):
            setattr(options, key, value.resolve())
    sys.path.insert(0, str(options.delta_repo / "src"))
    with forward_only_guard():
        prepared = prepare(options)
    if options.validate:
        print(json.dumps({"status": "repeat_probe_CPU_valid", "manifest": prepared["manifest"],
                          "CE_targets": "runtime exact final query", "backward_calls": 0,
                          "optimizer_updates": 0}, indent=2))
        return
    execute(options, prepared)


if __name__ == "__main__":
    main()

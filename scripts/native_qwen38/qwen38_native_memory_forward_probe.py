"""Forward-only diagnostics at the immutable alpha=1 80M writer.

CPU --validate selects the next complete 12-trajectory group from a cloned
sampler and prepares its manifest. --execute uses four EP4 ranks, no CE, no
backward and no optimizer; it never starts a training process afterwards.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

from qwen38_native_norm_probe import canonical_hash, file_info, require
from qwen38_native_train_staged_alpha100 import assign_trajectories, clone_cpu, write_json
from qwen38_native_train_batched import restore_rng_state, validate_frame_plan
from qwen38_native_alpha100_policy import state_digest
from qwen38_native_train_memory_fork import read_checkpoint, code_manifest, comparable_reader_runtime
import qwen38_native_memory_fork_policy as fork

PROTOCOL = "qwen38_native_memory_forward_probe_v1"


def execution_protocol():
    return {"protocol": PROTOCOL, "relative_epsilons": [1e-3, 1e-2],
            "perturbation_seed": 17, "absolute_floor": 1e-8, "batch_limit": 3,
            "expected_global_trajectories": 12, "stop_g_p99": 1.,
            "retain_row_gain_samples": False}


def diagnostic_protocol():
    return {
        "protocol": PROTOCOL,
        "source": "immutable_actual80M_alpha100_writer_no_residual",
        "source_optimizer_step": fork.SOURCE_STEP,
        "source_model_targets": fork.SOURCE_TARGETS,
        "sampling": "next_12_complete_trajectories_from_CPU_cloned_source_sampler",
        "relative_epsilons": [1e-3, 1e-2], "perturbation_seed": 17,
        "absolute_floor": 1e-8, "batch_limit": 3,
        "G": "matched_frame_lane_shape_Hbase_vs_Hmem_on_base_write_trajectory",
        "twins": "own_prewrite_memory_encoding_reference_repeat_and_two_value_perturbations",
        "perturbation": "after_first_nonzero_write; same_seed_plus_global_trajectory_index; values_only",
        "repeated_reference": "unperturbed_repeat_measures_numeric_noise_floor",
        "matched_native_calls_per_ready_wave": 6,
        "CE_targets": 0, "backward_calls": 0, "optimizer_updates": 0,
        "nonfinite": "fail_immediately_no_retry",
        "G_p99_above_one": "risk_flag_only_complete_this_bounded_forward_probe",
        "interpretation": "finite_sample_diagnostics_not_a_global_stability_certificate",
        "automatic_retry": False, "automatic_training_start": False,
        "execution": execution_protocol(),
        "execution_protocol_sha256": canonical_hash(execution_protocol()),
    }


@contextmanager
def forward_only_guard():
    """Fail closed if runtime code tries optimization or autograd derivatives."""
    import torch
    blocked = []
    originals = [(torch.optim.Optimizer, "__init__", torch.optim.Optimizer.__init__),
                 (torch.Tensor, "backward", torch.Tensor.backward),
                 (torch.autograd, "backward", torch.autograd.backward),
                 (torch.autograd, "grad", torch.autograd.grad)]
    def forbid(name):
        def fail(*args, **kwargs):
            blocked.append(name)
            raise RuntimeError(f"forward-only probe forbids {name}")
        return fail
    try:
        for owner, name, original in originals:
            setattr(owner, name, forbid("optimizer_construction" if owner is torch.optim.Optimizer else name))
        with torch.no_grad():
            yield blocked
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def select_probe_group(data, source_sampler):
    """No sampling mutation escapes this function, including validation errors."""
    sampler = data.sampler(seed=17)
    sampler.load_state_dict(copy.deepcopy(source_sampler))
    require(sampler.state_dict() == source_sampler, "source sampler clone changed")
    try:
        group = sampler.next_batch(12)
        require(len(group) == 12, "diagnostic requires exactly 12 complete trajectories")
        plans = [data.frames(item) for item in group]
        for item, plan in zip(group, plans):
            validate_frame_plan(plan, item.loss_tokens)
        after = copy.deepcopy(sampler.state_dict())
    finally:
        sampler.load_state_dict(source_sampler)
        require(sampler.state_dict() == source_sampler, "CPU probe did not restore its cloned sampler")
    descriptors = [{"global_trajectory_index": index,
                    "frame_plan_sha256": canonical_hash(plan),
                    "frames": len(plan), "write_blocks": sum(frame["evidence"] is not None for frame in plan),
                    "available_query_targets_not_executed": item.loss_tokens}
                   for index, (item, plan) in enumerate(zip(group, plans))]
    return group, plans, {
        "group": fork.group_identity(group), "group_identity_sha256": canonical_hash(fork.group_identity(group)),
        "assignments": assign_trajectories(plans), "trajectories": descriptors,
        "source_sampler_sha256": canonical_hash(source_sampler),
        "selected_group_next_sampler_sha256": canonical_hash(after),
        "source_sampler_restored": True, "complete_trajectories": len(group),
        "available_query_targets_not_executed": sum(item.loss_tokens for item in group),
        "CE_targets": 0,
    }


def prepare(options):
    from delta_ngram.qwen38_train_data import NativeFullTrainingData
    source, source_info = read_checkpoint(options.source_checkpoint)
    require(source_info["sha256"] == fork.SOURCE_CHECKPOINT_SHA256,
            "diagnostics require the registered immutable80M archive")
    saved = fork.validate_source80(source)
    original_manifest = json.loads(options.source_input_manifest.read_bytes())
    original_runtime = json.loads(options.source_runtime_identity.read_bytes())
    require(file_info(options.source_input_manifest)["sha256"] == source["registration_sha256"]
            and canonical_hash(original_runtime) == source["runtime_identity_sha256"]
            and canonical_hash(original_manifest["code"]) == original_manifest["code_manifest_sha256"]
            == source["code_manifest_sha256"] == original_runtime["code_sha256"],
            "source registration/runtime/code identity chain differs")
    receipt = fork.evaluation_receipt(source, options.source_evaluation_report, options.source_evaluation_complete)
    data = NativeFullTrainingData(
        options.cache_root, expected_complete_sha256=original_manifest["cache_complete"]["sha256"],
        expected_manifest_sha256=original_manifest["cache_manifest"]["sha256"])
    require(data.identity == source["data_identity"], "source training data identity differs")
    group, plans, selection = select_probe_group(data, saved["sampler"])
    codes = code_manifest(options.delta_repo)
    manifest = {
        "protocol": PROTOCOL, "diagnostic_protocol": diagnostic_protocol(),
        "source80_checkpoint": source_info,
        "source_input_manifest": file_info(options.source_input_manifest),
        "source_runtime_identity": original_runtime,
        "source_evaluation_receipt": receipt,
        "source_writer_sha256": saved["writer_sha256"],
        "source_arm_state_sha256": state_digest(saved),
        "source_rank_RNG_sha256": state_digest(saved["rng_by_rank"]),
        "code": codes, "code_manifest_sha256": canonical_hash(codes),
        "data_identity": data.identity, "selection": selection,
        "output": str(options.output), "automatic_training_start": False,
    }
    if options.reference_training_log is not None:
        records = fork.reference_groups(options.reference_training_log)
        fork.validate_replay_group(group, records[0], 0)
        manifest["reference_first_mainline_group"] = records[0]
        manifest["reference_training_log"] = file_info(options.reference_training_log)
    manifest["run_id"] = canonical_hash({"output": str(options.output), "source": source_info,
                                         "code": manifest["code_manifest_sha256"],
                                         "protocol": diagnostic_protocol(), "selection": selection})
    require(options.output != options.source_checkpoint.parent, "source output must remain immutable")
    if os.environ.get("RANK") in (None, "0"):
        require(not options.output.exists(), "probe output must be a new directory")
    return {"source": source, "source_arm": saved, "plans": plans, "manifest": manifest}


def execute(options, prepared):
    from qwen38_native_ep4_session import run_ep4_session
    rank = int(os.environ.get("RANK", "-1"))
    require(0 <= rank < 4 and int(os.environ.get("WORLD_SIZE", "0")) == 4,
            "execute the forward probe with four torchrun ranks")
    manifest = prepared["manifest"]
    marker = options.output / "input_manifest.json"
    ledger = {"protocol": "qwen38_memory_forward_probe_ledger_v1", "CE_targets": 0,
              "backward_calls": 0, "optimizer_updates": 0, "encoding_calls": 0,
              "encoding_real_input_tokens": 0, "encoding_padded_input_tokens": 0,
              "by_kind": {}, "failed_or_partial_accesses_are_not_refunded": True}
    started = time.monotonic()
    status, timing_record = "running", None
    immutable_before = state_digest(prepared["source_arm"])
    def save_ledger():
        if rank == 0:
            write_json(options.output / "budget_ledger.json", ledger)
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        write_json(marker, manifest)
        write_json(options.output / "diagnostic_protocol.json", diagnostic_protocol())
        save_ledger()
    else:
        deadline = time.monotonic() + 300
        while not marker.is_file():
            require(time.monotonic() < deadline, "rank0 did not create the probe manifest")
            time.sleep(.1)
    options.tokens = 640

    def run(session):
        nonlocal status, timing_record
        import numpy as np
        import torch
        import torch.distributed as dist
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        from delta_ngram.alpha_memory import LazyDecayState
        from delta_ngram.qwen38_native import make_qwen38_writer
        from delta_ngram.qwen38_precision import Qwen38SplitPLEPrecision
        from delta_ngram.qwen38_probe import tensor_state_hash
        from qwen38_native_dispatch import native_hc_dispatch
        from qwen38_native_router import native_router_dispatch
        from qwen38_native_ep4_session import runtime_identity
        from qwen38_native_memory_forward_schedule import run_memory_forward_probe

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
        def on_encoding(row):
            require(row.get("ce_targets") == 0, "forward probe must not dispatch CE")
            real, padded = map(sum, zip(*gather((row["real_input_tokens"], row["padded_input_tokens"]))))
            kind = row["kind"]
            agree(kind, "ranks disagree on diagnostic encoding kind")
            ledger["encoding_calls"] += 1
            ledger["encoding_real_input_tokens"] += real
            ledger["encoding_padded_input_tokens"] += padded
            cell = ledger["by_kind"].setdefault(kind, {"calls": 0, "real_input_tokens": 0, "padded_input_tokens": 0})
            cell["calls"] += 1
            cell["real_input_tokens"] += real
            cell["padded_input_tokens"] += padded
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
                    "forward probe reader/topology/HC/router/precision differ from source")
            runtime_hash = agree(canonical_hash(runtime), "rank runtime identities differ")
            if rank == 0:
                write_json(options.output / "runtime_identity.json", runtime)
            saved = prepared["source_arm"]
            with torch.random.fork_rng(devices=[]):
                writer, alpha = make_qwen38_writer(alpha=1.)
                writer.load_state_dict(saved["writer"], strict=True)
            writer = writer.to(device).float().requires_grad_(False).eval()
            alpha = alpha.to(device).requires_grad_(False).eval()
            require(tensor_state_hash(writer.state_dict()) == saved["writer_sha256"], "diagnostic writer changed")
            local = [{"index": index, "plan": prepared["plans"][index],
                      "state": LazyDecayState.from_batch(session.layout.empty_state(1, device))}
                     for index in manifest["selection"]["assignments"][rank]]
            restore_rng(next(row for row in saved["rng_by_rank"] if row["rank"] == rank))
            before_rng = capture_rng()
            writer_before = tensor_state_hash(writer.state_dict())
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            probe_started = time.monotonic()
            result = None
            try:
                result = run_memory_forward_probe(session, writer, alpha, local, gather=gather,
                    on_encoding=on_encoding, protocol=execution_protocol())
                require(result["protocol"] == execution_protocol(),
                        "forward core execution protocol differs from the input manifest")
                torch.cuda.synchronize(device)
                require(state_digest(capture_rng()) == state_digest(before_rng),
                        "forward diagnostics advanced training RNG")
            finally:
                restore_rng(before_rng)
            require(tensor_state_hash(writer.state_dict()) == writer_before
                    and all(not p.requires_grad and p.grad is None for p in writer.parameters()),
                    "forward diagnostics changed writer or acquired gradients")
            require(state_digest(prepared["source_arm"]) == immutable_before,
                    "loaded source training state changed")
            session.assert_frozen()
            ranks = gather({"rank": rank, "probe_seconds": time.monotonic() - probe_started,
                            "wall_seconds": time.monotonic() - started,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                            "RNG_restored": True, "writer_unchanged": True})
            timing_record = {"ranks": ranks,
                "probe_seconds_max_rank": max(row["probe_seconds"] for row in ranks),
                "wall_seconds_max_rank": max(row["wall_seconds"] for row in ranks),
                "GPU_occupied_hours": max(row["wall_seconds"] for row in ranks) * 4 / 3600,
                "diagnostic_GPU_occupied_hours": max(row["probe_seconds"] for row in ranks) * 4 / 3600,
                "definition": "rank_max_wall_times_four; wall includes model load; no utilization claim",
                "wall_scope": "execute_entry_to_forward_checks; final per-rank files include reader teardown"}
            status = "complete_forward_only"
            if rank == 0:
                require(file_info(options.source_checkpoint) == {
                    key: manifest["source80_checkpoint"][key] for key in ("bytes", "sha256")},
                    "source checkpoint file changed during diagnostics")
                report = {"protocol": PROTOCOL, "status": status, "diagnostics": result,
                          "selection": manifest["selection"], "budget": ledger, "timing": timing_record,
                          "source_writer_unchanged": True, "source_arm_state_unchanged": True,
                          "all_rank_RNG_unchanged": True, "runtime_identity_sha256": runtime_hash,
                          "diagnostic_protocol": diagnostic_protocol(), "automatic_training_start": False}
                write_json(options.output / "report.json", report)
                write_json(options.output / "timing_ledger.json", timing_record)
                write_json(options.output / "COMPLETE.json", {
                    "protocol": PROTOCOL, "status": status, "complete": True,
                    "report": file_info(options.output / "report.json"),
                    "source80_checkpoint": manifest["source80_checkpoint"],
                    "input_manifest": file_info(marker), "budget": ledger,
                    "CE_targets": 0, "backward_calls": 0, "optimizer_updates": 0,
                    "automatic_training_start": False})
                print(json.dumps({"status": status, "output": str(options.output), "budget": ledger}), flush=True)
            dist.barrier()
    try:
        with forward_only_guard() as blocked:
            run_ep4_session(options, run)
        require(not blocked, "forward-only guard observed forbidden calls")
    except BaseException as error:
        status = "failed_forward_only"
        save_ledger()
        if rank == 0:
            write_json(options.output / "FAILURE.json", {
                "protocol": PROTOCOL, "status": status, "error_type": type(error).__name__,
                "message": str(error), "budget": ledger, "automatic_retry": False,
                "automatic_training_start": False})
        raise
    finally:
        write_json(options.output / f"timing-rank-{rank}.json", {
            "rank": rank, "status": status, "wall_seconds": time.monotonic() - started,
            "scope": "execute_entry_including_model_load", "CE_targets": 0,
            "backward_calls": 0, "optimizer_updates": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source-checkpoint", "source-input-manifest", "source-runtime-identity",
                 "source-evaluation-report", "source-evaluation-complete", "cache-root", "delta-repo", "output"):
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
        print(json.dumps({"status": "forward_probe_CPU_valid", "manifest": prepared["manifest"],
                          "CE_targets": 0, "backward_calls": 0, "optimizer_updates": 0}, indent=2))
        return
    execute(options, prepared)


if __name__ == "__main__":
    main()

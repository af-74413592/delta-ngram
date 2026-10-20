"""Four exact-input native B3 reader calls on the first step48 query wave.

No optimizer update, checkpoints, or tensor artifacts. Calls 1/2/3 reuse the
same packed leaf object and bytes. Call 3 follows a base-only evidence call.
Call 4 rebuilds the writer from the identical cached H. All query CE is charged
to the diagnostic ledger before dispatch; every call restores the same RNG.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from qwen38_native_norm_probe import canonical_hash, file_info, require, write_json
from qwen38_native_ep4_trial import assign_trajectories, prepare as prepare_source
from qwen38_native_staged_diagnostic import TensorComparison
from qwen38_native_train_batched import restore_rng_state, validate_frame_plan


def prepare(options):
    validation = copy.copy(options)
    validation.max_groups = 3
    source, prior, manifest = prepare_source(validation)
    require(source["optimizer_step"] == 47 and source["completed_loss_targets"] == 1107675,
            "reader diagnostic requires the frozen step47 checkpoint")
    require(prior["total_optimizer_target_accesses"] == 1468915
            and prior["total_diagnostic_target_accesses"] == 14564,
            "reader diagnostic must carry the completed A/B/C ledger")
    for name in ("qwen38_native_reader_repeat_diagnostic.py", "qwen38_native_staged_diagnostic.py",
                 "qwen38_native_staged_trial.py", "qwen38_native_staged_schedule.py"):
        path = Path(__file__).with_name(name)
        manifest["code"][path.relative_to(options.delta_repo).as_posix()] = file_info(path)
    path = options.delta_repo / "src/delta_ngram/qwen38_staged_batching.py"
    manifest["code"][path.relative_to(options.delta_repo).as_posix()] = file_info(path)
    manifest.update({
        "protocol": "qwen38_native_exact_B3_reader_repeat_diagnostic_v1",
        "groups": 0, "optimizer_updates": 0, "sampling": "first_query_wave_of_next_step48_group",
        "registered_group_denominator": 33397, "query_calls": 4, "local_reader_batch": 3,
        "calls": ["fixed_leaf_same_RNG", "same_leaf_same_RNG",
                  "insert_same_B3_base_evidence_then_same_leaf_same_RNG",
                  "rebuild_writer_from_same_H_then_same_RNG"],
        "exact_input_requirement": "calls1_2_3_same_leaf_object_and_bytes; all4_same_tokens_rows_counts",
        "diagnostic_CE_targets": "4_times_actual_global_targets_in_first_query_wave",
        "diagnostic_target_cap": 100000, "new_optimizer_targets": 0,
        "tensor_capture": "per_token_CE_final_H_leaf_cotangent_router_maps_in_RAM_only",
        "tensor_artifacts_saved": False, "training_state_saved": False,
        "comparison": "report_exact_and_FP64_metrics; existing_gradient_relL2_0.02_cosine_0.999_unchanged",
        "interpretation": "recomputed_delta_call4_requires_first3_reader_repeats_stable",
    })
    return source, prior, manifest


def execute(options, source, prior, manifest):
    from qwen38_native_ep4_session import run_ep4_session
    rank = int(os.environ.get("RANK", "-1"))
    require(0 <= rank < 4 and int(os.environ.get("WORLD_SIZE", "0")) == 4, "use four torchrun ranks")
    initial_budget = {
        "protocol": "qwen38_native_reader_diagnostic_ledger_v1",
        "prior_optimizer_target_accesses": prior["total_optimizer_target_accesses"],
        "prior_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"],
        "total_optimizer_target_accesses": prior["total_optimizer_target_accesses"],
        "total_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"],
        "trial_optimizer_targets_dispatched": 0, "trial_diagnostic_targets_dispatched": 0,
        "trial_diagnostic_targets_completed": 0, "optimizer_updates": 0,
        "failed_or_partial_dispatches_are_not_refunded": True,
    }
    marker = options.output / "input_manifest.json"
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        write_json(marker, manifest)
        write_json(options.output / "budget_ledger.json", initial_budget)
    else:
        deadline = time.monotonic() + 300
        while not marker.is_file():
            require(time.monotonic() < deadline, "manifest creation timed out")
            time.sleep(.1)
    options.tokens = 640

    def run(session):
        import numpy as np
        import torch
        import torch.distributed as dist
        from megatron.core import parallel_state as ps
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        from delta_ngram.alpha_memory import LazyDecayState
        from delta_ngram.qwen38_native import make_qwen38_writer, write_native_block
        from delta_ngram.qwen38_precision import Qwen38SplitPLEPrecision
        from delta_ngram.qwen38_probe import tensor_state_hash
        from delta_ngram.qwen38_training import NativeTrajectoryGradients, writer_evidence_features
        from delta_ngram.qwen38_train_data import NativeFullTrainingData
        from delta_ngram.qwen38_train_batching import crop_evidence_lanes
        from delta_ngram.qwen38_ep4_batching import (
            collate_ep4_frames, pack_ep4_query_tickets, SingleRankGradientPolicy)
        from delta_ngram.qwen38_staged_batching import materialize_staged_snapshot
        from qwen38_native_dispatch import native_hc_dispatch
        from qwen38_native_router import native_router_dispatch
        from qwen38_native_runtime_preflight import language_model
        from qwen38_native_ep4_session import runtime_identity
        from qwen38_native_ep4_schedule import collect_evidence_microbatches, schedule_query_microbatches

        device = session.device
        dispatched = completed = 0
        reports, recording = [], {}
        baseline = None
        handles = []

        def gather(value):
            copies = [None] * 4
            dist.all_gather_object(copies, value)
            return copies

        def agree(value, message):
            copies = gather(value)
            require(all(value == copies[0] for value in copies), message)
            return copies[0]

        def ledger():
            return {**initial_budget, "trial_diagnostic_targets_dispatched": dispatched,
                    "trial_diagnostic_targets_completed": completed,
                    "total_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"] + dispatched}

        def save_ledger():
            if rank == 0:
                write_json(options.output / "budget_ledger.json", ledger())

        def rng_state():
            return {"rank": rank, "python": random.getstate(), "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state(device).cpu(),
                    "tensor_parallel": copy.deepcopy(get_cuda_rng_tracker().get_states())}

        def comparison(reference, candidate):
            value = TensorComparison()
            value.add(reference, candidate)
            return TensorComparison.merge(gather(value.state()))

        def fingerprint(state, mb, *, include_values):
            tensors = {"input_ids": mb["batch"]["input_ids"], "labels": mb["batch"]["labels"],
                       "position_ids": mb["batch"]["position_ids"], "valid_mask": mb["valid_mask"],
                       "owner_ids": mb["owner_ids"], "rows": state.rows, "counts": state.counts}
            if include_values:
                tensors["values"] = state.values
            return tensor_state_hash(tensors)

        with ExitStack() as scope:
            try:
                hc = scope.enter_context(native_hc_dispatch(unwrap=True))
                router = scope.enter_context(native_router_dispatch(session.raw))
                precision = scope.enter_context(Qwen38SplitPLEPrecision(session.adapter))
                config = session.models[0].config
                required = {"moe_input_jitter_eps": None, "moe_expert_capacity_factor": None,
                            "attention_dropout": 0., "hidden_dropout": 0.}
                require(all(hasattr(config, key) and getattr(config, key) == value
                            for key, value in required.items()), "dropout/jitter/capacity configuration changed")
                router_config = {key: getattr(config, key, "missing") for key in (
                    "moe_router_force_load_balancing", "moe_router_force_biased", "moe_enable_routing_replay")}
                execution_config = {key: getattr(config, key, "missing") for key in (
                    "moe_token_dispatcher_type", "moe_permute_fusion", "moe_router_fusion",
                    "moe_grouped_gemm", "moe_shared_expert_overlap", "deterministic_mode",
                    "sequence_parallel", "fp8")}
                routers = [(name, module) for name, module in session.raw.named_modules()
                           if type(module).__name__ == "TopKRouter"
                           and type(module).__module__ == "mcore_bridge.model.modules.topk_router"]
                router_flags = {name: {key: getattr(module, key, "missing") for key in (
                    "enable_expert_bias", "frozen_expert_bias")}
                                | {"router_replay_type": type(getattr(module, "router_replay", None)).__name__}
                                for name, module in routers}
                runtime = {"native": runtime_identity(session), "HC": hc.identity, "router": router["identity"],
                           "precision": precision.identity(), "code_sha256": canonical_hash(manifest["code"]),
                           "randomness_and_capacity": required, "additional_router_config": router_config,
                           "execution_config": execution_config,
                           "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                           "cudnn_deterministic": torch.backends.cudnn.deterministic,
                           "cudnn_benchmark": torch.backends.cudnn.benchmark,
                           "router_flags": router_flags, "router_map_hooks": len(routers)}
                runtime_hash = agree(canonical_hash(runtime), "runtime differs across ranks")
                if rank == 0:
                    write_json(options.output / "runtime_identity.json", runtime)
                policy = SingleRankGradientPolicy(
                    runtime_hash, tensor_parallel_world_size=ps.get_tensor_model_parallel_world_size())
                data = NativeFullTrainingData(
                    options.cache_root, expected_complete_sha256=manifest["cache_complete"]["sha256"],
                    expected_manifest_sha256=manifest["cache_manifest"]["sha256"])
                require(data.identity == source["data_identity"], "source corpus changed")
                sampler = data.sampler(seed=17)
                sampler.load_state_dict(source["sampler"])
                group = sampler.next_batch(12)
                require(len(group) == 12 and sum(item.loss_tokens for item in group) == 33397,
                        "source sampler selected a different optimizer group")
                plans = [data.frames(item) for item in group]
                for item, plan in zip(group, plans):
                    validate_frame_plan(plan, item.loss_tokens)
                assignments = assign_trajectories(plans)
                identity = [{"id": item.document_id, "group_id": item.group_id, "start": item.start,
                             "blocks": item.blocks, "loss_tokens": item.loss_tokens} for item in group]
                agree(canonical_hash({"identity": identity, "assignment": assignments}), "group differs across ranks")
                first_wave = next(ordinal for ordinal in range(max(map(len, plans)))
                                  if any(ordinal < len(plan) and plan[ordinal]["query"] is not None for plan in plans))
                require(first_wave > 0, "first scored wave must follow real warmup evidence")
                writer, alpha = make_qwen38_writer(alpha=1.)
                writer, alpha = writer.to(device).float().train(), alpha.to(device)
                writer.load_state_dict(source["writer"], strict=True)
                writer_hash = agree(tensor_state_hash(writer.state_dict()), "writer replicas differ")
                restore_rng_state(next(item for item in source["rng_by_rank"] if item["rank"] == rank),
                                  rank=rank, device=device, tensor_parallel_tracker=get_cuda_rng_tracker())
                evidence = {}
                local_indices = assignments[rank]

                def empty_states():
                    return {index: LazyDecayState.from_batch(session.layout.empty_state(1, device))
                            for index in local_indices}

                def write(states, index, block):
                    hidden = writer_evidence_features(block["H_new"].to(device), mode="rmsnorm", eps=1e-6)
                    states[index] = write_native_block(
                        writer, states[index], hidden, block["physical_rows_new"].to(device),
                        torch.ones(hidden.shape[:2], dtype=torch.bool, device=device), alpha)

                states = empty_states()
                repeat_evidence_mb = None
                for ordinal in range(first_wave):
                    frames = [plans[index][ordinal]["evidence"] if ordinal < len(plans[index]) else None
                              for index in local_indices]
                    mb = collate_ep4_frames(frames, kind="evidence", device=device)
                    result = collect_evidence_microbatches(session, [mb])["microbatches"][0]
                    for index, block in zip(local_indices, crop_evidence_lanes(
                            mb, result["hidden"], result["physical_rows"])):
                        if block is not None:
                            evidence[index, ordinal] = block
                            write(states, index, block)
                    repeat_evidence_mb = mb
                query_frames = [plans[index][first_wave]["query"] if first_wave < len(plans[index]) else None
                                for index in local_indices]
                query_mb = collate_ep4_frames(query_frames, kind="query", device=device)
                global_targets = sum(gather(query_mb["loss_target_tokens"]))
                require(0 < global_targets <= 6144
                        and prior["total_diagnostic_target_accesses"] + 4 * global_targets <= 100000,
                        "four repeats exceed registered diagnostic ceiling")

                def pack_states(values):
                    tickets = []
                    for index, frame in zip(local_indices, query_frames):
                        if frame is None:
                            tickets.append(None)
                            continue
                        bridge = NativeTrajectoryGradients(
                            total_loss_tokens=33397, expected_loss_tokens=frame["loss_target_tokens"],
                            policy=policy, runtime_identity_sha256=runtime_hash)
                        tickets.append(bridge.prepare_query(materialize_staged_snapshot(values[index])))
                    return pack_ep4_query_tickets(tickets, row_capacity=session.layout.row_capacity, device=device)

                fixed = pack_states(states)
                fixed_leaf = fixed.reader_state.values
                fixed_leaf_bytes = fixed_leaf.detach().cpu().clone()
                fixed_input_sha = fingerprint(fixed.reader_state, query_mb, include_values=True)
                structure_sha = fingerprint(fixed.reader_state, query_mb, include_values=False)
                common_rng = rng_state()

                small_buffers, buffer_metadata = {}, {}
                remaining_bytes = 16 * 1024 * 1024
                for name, value in session.raw.named_buffers():
                    size = value.numel() * value.element_size()
                    buffer_metadata[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                                             "device": str(value.device), "bytes": size}
                    if size <= 1024 * 1024 and size <= remaining_bytes:
                        small_buffers[name] = value
                        remaining_bytes -= size

                def buffer_state():
                    current = dict(session.raw.named_buffers())
                    hashes, remaining = {}, 16 * 1024 * 1024
                    for name in small_buffers:
                        value = current.get(name)
                        size = value.numel() * value.element_size() if value is not None else 0
                        if value is None:
                            hashes[name] = "buffer_removed"
                        elif size > 1024 * 1024 or size > remaining:
                            hashes[name] = "buffer_now_exceeds_hash_limit"
                        else:
                            hashes[name] = tensor_state_hash({name: value.reshape(-1)})
                            remaining -= size
                    return {"small_buffer_sha": hashes,
                            "versions": {name: value._version for name, value in current.items()},
                            "objects": {name: id(value) for name, value in current.items()}}

                initial_buffers = buffer_state()

                def capture_loss(module, args, output):
                    if recording.get("active"):
                        require(torch.is_tensor(output) and tuple(output.shape) == (3, 640),
                                "native CE capture must remain per-token [3,640]")
                        require("CE" not in recording, "duplicate CE capture")
                        recording["CE"] = output.detach().float().cpu().clone()

                def capture_H(module, args, kwargs):
                    if recording.get("active"):
                        value = args[0] if args else kwargs.get("hidden_states", kwargs.get("input_"))
                        require(torch.is_tensor(value) and tuple(value.shape) == (640, 3, 2560),
                                "native final H capture shape changed")
                        require("H" not in recording, "duplicate native final H capture")
                        recording["H"] = value.detach().transpose(0, 1).contiguous().cpu().clone()

                def router_hook(name):
                    def hook(module, args, output):
                        if recording.get("active"):
                            require(isinstance(output, tuple) and len(output) == 2
                                    and torch.is_tensor(output[1]) and output[1].dtype == torch.bool
                                    and output[1].ndim == 2, "native router map shape changed")
                            key = name + "#" + str(sum(key.startswith(name + "#") for key in recording["routes"]))
                            recording["routes"][key] = output[1].detach().cpu().contiguous().clone()
                    return hook

                handles.append(session.models[0].register_forward_hook(capture_loss))
                handles.append(language_model(session.raw).output_layer.register_forward_pre_hook(
                    capture_H, with_kwargs=True))
                handles += [module.register_forward_hook(router_hook(name)) for name, module in routers]
                mask_cpu = query_mb["batch"]["labels"].ne(-100).cpu()
                valid_cpu = query_mb["valid_mask"].cpu()

                for call in range(1, 5):
                    if call == 3:
                        # Observed reader state is deliberately not restored.
                        # Restore RNG *after* the interleaved evidence call.
                        collect_evidence_microbatches(session, [repeat_evidence_mb])
                    packed = fixed
                    if call == 4:
                        rebuilt = empty_states()
                        for ordinal in range(first_wave):
                            for index in local_indices:
                                if (index, ordinal) in evidence:
                                    write(rebuilt, index, evidence[index, ordinal])
                        packed = pack_states(rebuilt)
                    state = packed.reader_state
                    state.values.grad = None
                    current_structure = fingerprint(state, query_mb, include_values=False)
                    current_input = fingerprint(state, query_mb, include_values=True)
                    require(current_structure == structure_sha, "token/row/count input identity changed")
                    if call <= 3:
                        require(state.values is fixed_leaf and current_input == fixed_input_sha,
                                "exact reader repeats changed the fixed leaf object or bytes")
                    restore_rng_state(common_rng, rank=rank, device=device,
                                      tensor_parallel_tracker=get_cuda_rng_tracker())
                    before_buffers = buffer_state()
                    query_mb["reader_state"] = state
                    recording.clear()
                    recording.update({"active": True, "routes": {}})
                    dispatched += global_targets
                    save_ledger()
                    result = schedule_query_microbatches(
                        session, [query_mb], group_total_target_tokens=33397)["microbatches"][0]
                    recording["active"] = False
                    require("CE" in recording and "H" in recording and state.values.grad is not None,
                            "native query did not produce all diagnostic captures")
                    cotangent = state.values.grad.detach().cpu().clone()
                    after_buffers = buffer_state()
                    require(fingerprint(state, query_mb, include_values=True) == current_input,
                            "native reader mutated its input leaf or batch")
                    completed += global_targets
                    save_ledger()
                    session.assert_frozen()
                    require(tensor_state_hash(writer.state_dict()) == writer_hash
                            and all(value.grad is None for value in writer.parameters()),
                            "reader diagnostic unexpectedly updated the writer")
                    current = {"CE": recording["CE"], "H": recording["H"], "cotangent": cotangent,
                               "routes": recording["routes"], "leaf": state.values.detach().cpu().clone()}
                    local_routes = {}
                    for name, value in current["routes"].items():
                        entry = {"sha256": tensor_state_hash({name: value}), "shape": list(value.shape)}
                        if baseline is not None:
                            reference = baseline["routes"][name]
                            require(value.shape == reference.shape, "router shape changed")
                            changed = value != reference
                            entry.update({"changed_bits": int(changed.sum()),
                                          "changed_token_rows": int(changed.any(dim=1).sum())})
                        local_routes[name] = entry
                    route_reports = gather({"rank": rank, "maps": local_routes})
                    changed_buffers = gather({
                        "rank": rank,
                        "changed_during_call": [name for name in small_buffers
                                                if before_buffers["small_buffer_sha"][name]
                                                != after_buffers["small_buffer_sha"][name]],
                        "changed_from_first_call_start": [name for name in small_buffers
                            if initial_buffers["small_buffer_sha"][name] != after_buffers["small_buffer_sha"][name]],
                        "version_changed": [name for name in after_buffers["versions"]
                                            if before_buffers["versions"].get(name) != after_buffers["versions"][name]],
                        "buffer_object_replaced": [name for name in after_buffers["objects"]
                            if before_buffers["objects"].get(name) != after_buffers["objects"][name]]})
                    row = {"call": call, "global_targets": global_targets,
                           "global_loss_sum": math.fsum(gather(result["loss_sum"])),
                           "input_fingerprints": gather({"rank": rank, "input_leaf_sha256": current_input,
                                                       "structure_sha256": current_structure,
                                                       "same_original_leaf_object": state.values is fixed_leaf}),
                           "buffer_changes": changed_buffers, "routing": route_reports}
                    if baseline is None:
                        baseline = current
                    else:
                        row["versus_call1"] = {
                            "selected_per_token_CE": comparison(baseline["CE"][mask_cpu], current["CE"][mask_cpu]),
                            "leaf_cotangent": comparison(baseline["cotangent"], current["cotangent"]),
                            "full_final_H": comparison(baseline["H"], current["H"]),
                            "valid_final_H": comparison(baseline["H"][valid_cpu], current["H"][valid_cpu]),
                            "leaf_values": comparison(fixed_leaf_bytes, current["leaf"])}
                        cot = row["versus_call1"]["leaf_cotangent"]
                        row["cotangent_within_existing_gradient_tolerance"] = (
                            cot["relative_l2"] is not None and cot["relative_l2"] <= .02 and cot["cosine"] >= .999)
                    reports.append(row)
                    if rank == 0:
                        write_json(options.output / ("call_" + str(call) + ".json"), row)
                        write_json(options.output / "progress.json", {
                            "status": "reader_diagnostic_running", "completed_call": call, "budget": ledger()})
                        print(json.dumps({"event": "reader_repeat_complete", "call": call,
                                          "versus_call1": row.get("versus_call1"), "budget": ledger()}), flush=True)
                    state.values.grad = None
                    recording.clear()
                    dist.barrier()
                require(completed == dispatched == 4 * global_targets, "reader diagnostic accounting differs")
                if rank == 0:
                    stable = all(row["cotangent_within_existing_gradient_tolerance"] for row in reports[1:3])
                    exact = all(all(row["versus_call1"][key]["bitwise_equal"] for key in (
                        "selected_per_token_CE", "leaf_cotangent", "full_final_H")) for row in reports[1:3])
                    write_json(options.output / "COMPLETE.json", {
                        "protocol": manifest["protocol"], "status": "four_reader_calls_completed",
                        "first_query_wave": first_wave, "group_denominator": 33397,
                        "global_targets_per_query": global_targets, "assignment": assignments,
                        "group_identity_sha256": canonical_hash(identity), "calls": reports, "budget": ledger(),
                        "first_three_bitwise_repeatable": exact,
                        "first_three_cotangents_within_existing_tolerance": stable,
                        "fourth_call_delta_sensitivity_interpretable": exact,
                        "small_buffer_hash_limit_each_bytes": 1024 * 1024,
                        "small_buffer_hash_limit_total_bytes": 16 * 1024 * 1024,
                        "small_buffers_hashed": list(small_buffers), "buffer_metadata": buffer_metadata,
                        "optimizer_updates": 0, "training_state_saved": False,
                        "tensor_artifacts_saved": False, "formal_checkpoint_unchanged": True})
                dist.barrier()
            except BaseException as error:
                save_ledger()
                if rank == 0:
                    write_json(options.output / "FAILURE.json", {
                        "status": "reader_diagnostic_stopped", "error_type": type(error).__name__,
                        "message": str(error), "completed_calls": len(reports), "budget": ledger(),
                        "automatic_retry": False, "training_state_saved": False})
                raise
            finally:
                for handle in handles:
                    handle.remove()

    try:
        run_ep4_session(options, run)
    except BaseException as error:
        failure = options.output / "FAILURE.json"
        if rank == 0 and not failure.exists():
            write_json(failure, {"status": "reader_diagnostic_stopped", "error_type": type(error).__name__,
                                "message": str(error), "automatic_retry": False,
                                "budget": json.loads((options.output / "budget_ledger.json").read_bytes())})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("resume-checkpoint", "source-budget-ledger", "source-runtime-identity", "cache-root",
                 "model-path", "verification", "delta-repo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--validate", action="store_true")
    options = parser.parse_args()
    for key, value in vars(options).items():
        if isinstance(value, Path):
            setattr(options, key, value.resolve())
    sys.path.insert(0, str(options.delta_repo / "src"))
    source, prior, manifest = prepare(options)
    if options.validate:
        print(json.dumps({"status": "reader_repeat_inputs_valid", "input_manifest": manifest}, indent=2))
        return
    execute(options, source, prior, manifest)


if __name__ == "__main__":
    main()

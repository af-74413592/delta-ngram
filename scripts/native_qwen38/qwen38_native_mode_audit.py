"""Compare training-dispatch and no-grad logits on six frozen training families.

Both paths use eval mode and PLE_FUSED_KERNEL=0. Each query is run with the
same initialized RMS-normalized writer memory, and again with the same rows
but zero delta. A repeated grad-enabled forward measures replay equality.
Optional flags capture PLE inputs or a raw reader backward without an optimizer
update. Precision paths can share one frozen-model load.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

from qwen38_native_runtime_preflight import (
    digest, first_tensor, language_model, schedule_query, write_json,
)
from qwen38_native_norm_probe import file_info, require
from qwen38_native_frozen_session import run_frozen_session
from qwen38_native_paired_norm_probe import prepare_frames, verify_observation


def compare_logits(reference, observed):
    """FP32 probability metrics on complete, aligned vocabulary logits."""
    import torch
    require(reference.shape == observed.shape and reference.ndim == 2,
            "logit shapes must agree")
    a, b = reference.float(), observed.float()
    require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "nonfinite query logits")
    difference = b - a
    log_a, log_b = a.log_softmax(-1), b.log_softmax(-1)
    kl = (log_a.exp() * (log_a - log_b)).sum(-1)
    return {
        "bitwise_equal": bool(torch.equal(reference, observed)),
        "positions": a.shape[0], "vocabulary": a.shape[1],
        "reference_logits_rms": float(a.square().mean().sqrt()),
        "difference_max_abs": float(difference.abs().max()),
        "difference_mean_abs": float(difference.abs().mean()),
        "difference_rms": float(difference.square().mean().sqrt()),
        "top1_changed_tokens": int(a.argmax(-1).ne(b.argmax(-1)).sum()),
        "KL_reference_to_observed_mean": float(kl.mean()),
        "KL_reference_to_observed_max": float(kl.max()),
    }


def quantization_visibility(base, delta, observed=None):
    """Measure actual BF16 rounding on selected coordinates; no model labels."""
    import torch
    require(base.shape == delta.shape and base.dtype == torch.bfloat16,
            "visibility requires aligned native BF16 base and FP32 delta")
    raw = delta.float()
    rounded = (base.float() + raw).to(base.dtype)
    if observed is not None:
        require(torch.equal(rounded, observed), "observed native overlay differs from measured rounding")
    effective = rounded.float() - base.float()
    support = raw.ne(0)
    count, energy = int(support.sum()), float(raw.double().square().sum())
    effective_energy = float(effective.double().square().sum())
    dot = float((raw.double() * effective.double()).sum())
    error_energy = float((effective.double() - raw.double()).square().sum())
    return {
        "coordinates": raw.numel(), "nonzero_delta_coordinates": count,
        "changed_coordinates": int(effective.ne(0).sum()),
        "changed_fraction_of_nonzero_delta": float(effective[support].ne(0).float().mean()) if count else None,
        "swallowed_delta_energy_fraction": float(
            raw.double().masked_fill(effective.ne(0), 0).square().sum()) / energy if energy else None,
        "rounding_error_energy_over_delta_energy": error_energy / energy if energy else None,
        "effective_delta_energy_over_requested_energy": effective_energy / energy if energy else None,
        "requested_effective_dot": dot,
        "requested_effective_cosine": dot / (energy * effective_energy) ** .5
        if energy and effective_energy else None,
        "relative_l2_error": (error_energy / energy) ** .5 if energy else None,
        "requested_delta_rms": float(raw.square().mean().sqrt()),
        "effective_delta_rms": float(effective.square().mean().sqrt()),
        "base_rms": float(base.float().square().mean().sqrt()),
        "all_coordinates_unchanged": bool(torch.equal(base, rounded)),
    }


def query_modes(*, hc_mode, compare_compiled, with_backward, source, state_case):
    modes = [{"name": name, "grad": name != "no_grad", "backward": False, "hc": hc_mode}
             for name in ("grad", "no_grad", "grad_repeat")]
    if compare_compiled:
        require(hc_mode == "eager", "--compare-compiled requires selected --hc-helper-mode eager")
        modes += [{"name": "compiled_grad", "grad": True, "backward": False, "hc": "compiled"},
                  {"name": "compiled_no_grad", "grad": False, "backward": False, "hc": "compiled"}]
    if with_backward and source == "books" and state_case == "calibrated_initial_writer":
        modes.append({"name": "train_backward", "grad": True, "backward": True, "hc": hc_mode})
    return modes


def precision_modes(selected):
    require(selected in ("native", "split", "both"), "unknown PLE precision")
    return ("native", "split") if selected == "both" else (selected,)


def runtime_identity_fields(by_mode):
    """Name the primary grad dispatch explicitly, regardless of the last query."""
    require("grad" in by_mode, "missing primary grad runtime identity")
    return {"runtime_identity_sha256": by_mode["grad"],
            "runtime_identity_sha256_by_mode": dict(by_mode)}


def ple_effect(actual, zero, valid, target):
    """Aligned full-sequence native PLE output; separately count selected positions."""
    import torch
    require(actual.shape == zero.shape and actual.ndim == 3
            and actual.shape[:2] == valid.transpose(0, 1).shape, "PLE effect shape mismatch")
    change = actual.float() - zero.float()

    def stats(mask):
        value, baseline = change[mask], zero.float()[mask]
        return {"positions": int(mask.sum()), "difference_max_abs": float(value.abs().max()),
                "difference_rms": float(value.square().mean().sqrt()),
                "zero_delta_output_rms": float(baseline.square().mean().sqrt())}

    return {"bitwise_equal": bool(torch.equal(actual, zero)),
            "scope": "full sequence gathered from both native TP sequence shards",
            "sequence_positions": actual.shape[0] * actual.shape[1],
            "valid_positions": stats(valid.transpose(0, 1)),
            "loss_target_positions": stats(target.transpose(0, 1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "verification", "delta-repo", "cache-root",
                 "observation-root", "registration", "scale-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prior-diagnostic-targets", type=int, required=True)
    parser.add_argument("--hc-helper-mode", choices=("compiled", "eager"), default="compiled")
    parser.add_argument("--router-topk", choices=("native", "sorted"), default="native")
    parser.add_argument("--ple-precision", choices=("native", "split", "both"), default="native")
    parser.add_argument("--source", choices=("all", "books"), default="all")
    parser.add_argument("--trace-hc", action="store_true")
    parser.add_argument("--trace-modules", action="store_true",
                        help="trace native layers 1/2, including MoE router/experts/dispatch")
    parser.add_argument("--capture-ple", action="store_true",
                        help="save both TP replicas of the existing books initial/zero grad passes")
    parser.add_argument("--compare-compiled", action="store_true",
                        help="compare compiled and selected eager HC on the same states in this load")
    parser.add_argument("--with-backward", action="store_true",
                        help="one extra books initial-state native backward; save raw delta cotangent")
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    for name in ("model_path", "verification", "delta_repo", "cache_root",
                 "observation_root", "registration", "scale_receipt", "output"):
        setattr(options, name, getattr(options, name).resolve())
    sys.path.insert(0, str(options.delta_repo / "src"))
    registration = json.loads(options.registration.read_bytes())
    frames = prepare_frames(options, registration)
    items = [item for item in frames["samples"]
             if item["sample"]["initial_perturbation_member"]]
    require(len(items) == 6 and len({item["sample"]["source"] for item in items}) == 6,
            "requires the six fixed source-rank-zero training families")
    if options.source != "all":
        items = [item for item in items if item["sample"]["source"] == options.source]
        require(len(items) == 1, "missing fixed books family")
    observation, initial, history = verify_observation(options, frames)
    scale = json.loads(options.scale_receipt.read_bytes())
    require(scale["rho"] == .01, "only frozen rho=.01 is supported")
    scales = scale["target_scale_float32_by_mode"]["rmsnorm"]
    require(len(scales) == 16 and all(0 < x < 1 for x in scales),
            "invalid registered RMSNorm target scales")
    require(observation["identity"]["verification_sha256"] == digest(options.verification),
            "evidence and query model receipts differ")
    selected_precisions = precision_modes(options.ple_precision)
    planned_targets = len(selected_precisions) * sum(len(query_modes(
        hc_mode=options.hc_helper_mode, compare_compiled=options.compare_compiled,
        with_backward=options.with_backward, source=item["sample"]["source"], state_case=state))
        * 128 for item in items for state in ("calibrated_initial_writer", "same_rows_zero_delta"))
    require(0 <= options.prior_diagnostic_targets <= 100000 - planned_targets,
            "mode audit exceeds the global diagnostic budget")
    contract = {
        "protocol": "qwen38_native_training_inference_mode_audit_v1",
        "source_script_sha256": digest(__file__),
        "registration": file_info(options.registration),
        "scale_receipt": file_info(options.scale_receipt),
        "observation_receipt": file_info(options.observation_root / "COMPLETE.json"),
        "model_receipt": file_info(options.verification),
        "sample_ids": [item["sample"]["sample_id"] for item in items],
        "query_frame_hashes": [item["query"]["frame_sha256"] for item in items],
        "alpha": .99, "writer_norm": "rmsnorm", "rmsnorm_epsilon": 1e-6,
        "state_cases": ["calibrated_initial_writer", "same_rows_zero_delta"],
        "query_modes": ["grad", "no_grad", "grad_repeat"],
        "reader_eval": True, "PLE_FUSED_KERNEL": "0",
        "hc_helper_mode": options.hc_helper_mode, "source_subset": options.source,
        "router_topk": options.router_topk, "ple_precision": list(selected_precisions),
        "trace_hc": options.trace_hc,
        "trace_modules": options.trace_modules,
        "capture_ple": options.capture_ple, "compare_compiled": options.compare_compiled,
        "with_backward": options.with_backward,
        "trace_layer_numbers": [1, 2] if options.trace_modules else [],
        "evidence_cache_status": "historical H is reused only for this dispatch diagnostic; recompute after final dispatch selection",
        "maximum_loss_target_accesses": planned_targets,
        "prior_global_diagnostic_target_accesses": options.prior_diagnostic_targets,
        "optimizer_steps": 0, "backward_passes": int(options.with_backward) * len(selected_precisions),
        "ple_snapshot_scope": "native precision only; legacy snapshot formula does not describe split PLE",
        "visibility": "Actual initial delta and zero delta; synthetic multipliers are sensitivity probes, not trained scales.",
        "interpretation": "Measures dispatch parity; does not validate a gradient or select a norm.",
    }
    if not options.execute:
        print(json.dumps(contract, indent=2))
        return
    rank = int(os.environ.get("RANK", "-1"))
    require(rank >= 0 and int(os.environ.get("WORLD_SIZE", "0")) == 4, "requires torchrun4")
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        write_json(options.output / "registration.json", contract)
    else:
        deadline = time.monotonic() + 300
        while not (options.output / "registration.json").is_file():
            require(time.monotonic() < deadline, "rank0 did not prepare the audit")
            time.sleep(.1)
    options.tokens = 640
    hc_reports = []

    def audit(session):
        import torch
        import torch.distributed as dist
        from megatron.core import parallel_state as ps
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        from delta_ngram.qwen38_native import make_qwen38_writer
        from delta_ngram.qwen38_probe import build_history_memory
        from qwen38_native_dispatch import native_hc_dispatch
        from qwen38_native_router import native_router_dispatch
        from qwen38_native_tp_gradient_probe import canonical_hash, live_runtime_identity
        from qwen38_native_ple_snapshot import capture_native_ple_snapshot, save_native_ple_snapshot
        from delta_ngram import qwen38_precision as precision_module
        from delta_ngram.qwen38_precision import Qwen38SplitPLEPrecision
        device = session.device
        reports, calls, targets = [], [], 0
        snapshot_receipts, anchor_receipts = [], []
        base_runtime_identity = live_runtime_identity(session.models)
        writer = alpha = None
        if session.adapter is not None:
            writer, alpha = make_qwen38_writer(alpha=.99)
            writer.load_state_dict(initial, strict=True)
            writer = writer.to(device).float().eval().requires_grad_(False)
            alpha = alpha.to(device)
            writer.target_scale.copy_(torch.tensor(scales, device=device))
        last = ps.is_pipeline_last_stage()
        tp = ps.get_tensor_model_parallel_rank()
        tp_group = ps.get_tensor_model_parallel_group()
        current_mask = capture = None
        handle = None
        overlay_original = None
        overlay_report = None
        memory = valid = None
        mode = state_case = selected_precision = None
        precision_observations = []
        precision_reference = {}
        router_reports = []
        ple_outputs = {}
        ple_handle = None
        from qwen38_native_mode_trace import native_module_trace
        module_trace_context = (native_module_trace(session.raw, layer_numbers=(1, 2))
                                if options.trace_modules else nullcontext(None))
        module_trace = module_trace_context.__enter__()
        trace_pairs = []
        if session.adapter is not None:
            overlay_original = session.adapter._overlay

            def measure_overlay(native_ids, base):
                nonlocal overlay_report
                output = overlay_original(native_ids, base)
                if mode == "grad" and selected_precision == "native":
                    with torch.no_grad():
                        physical = session.layout.physical_rows(native_ids, valid)
                        delta, _ = memory.lookup(physical)
                        shaped_base = base.reshape(*native_ids.shape, 160)
                        shaped_output = output.reshape_as(shaped_base)
                        # Padding is excluded; missing rows remain explicit zero increments.
                        b, d, o = shaped_base[valid], delta[valid], shaped_output[valid]
                        overlay_report = {
                            "actual": quantization_visibility(b, d, o),
                            "synthetic_amplitude_sensitivity": {
                                str(multiplier): quantization_visibility(b, d * multiplier)
                                for multiplier in (.25, 4., 16.)},
                            "sensitivity_is_training_observation": False,
                        }
                return output

            session.adapter._overlay = measure_overlay

            def save_ple(module, args, output):
                if mode == "grad":
                    ple_outputs[state_case] = first_tensor(output, torch).detach().float().clone()

            ple_handle = session.adapter.ple_layer.register_forward_hook(save_ple)
        if last:
            def save_logits(module, args, output):
                nonlocal capture
                value = first_tensor(output, torch)
                require(capture is None and value.ndim == 3 and value.shape[1:] == (1, 124160),
                        "requires one native TP2 vocabulary-sharded LM-head output")
                require(value.shape[0] == current_mask.numel(),
                        "LM-head logits do not cover the complete query")
                # Clone before the native cross-entropy implementation can reuse a buffer.
                capture = value[:, 0][current_mask].detach().clone()
            handle = language_model(session.raw).output_layer.register_forward_hook(save_logits)
        try:
            for item, selected_precision in (
                    (item, precision) for item in items for precision in selected_precisions):
                ple_outputs.clear()
                sample, query = item["sample"], item["query"]
                ids = torch.tensor([query["input_ids"]], device=device, dtype=torch.long)
                labels = torch.tensor([query["labels"]], device=device, dtype=torch.long)
                valid = torch.tensor([query["encoder_valid_mask"]], device=device, dtype=torch.bool)
                owners = torch.zeros(1, device=device, dtype=torch.long)
                current_mask = labels[0].ne(-100)
                batch = {"input_ids": ids, "position_ids": torch.arange(
                    ids.shape[1], device=device)[None], "labels": labels, "attention_mask": None}
                state = None
                if session.adapter is not None:
                    with torch.no_grad():
                        state = build_history_memory(
                            writer, alpha, history[sample["sample_id"]], norm_mode="rmsnorm",
                            row_capacity=session.layout.row_capacity, device=device)
                for state_case in contract["state_cases"]:
                    measurements, logits = {}, {}
                    dispatch_identities = {}
                    overlay_report = None
                    precision_observations = []

                    def observe_precision(report):
                        precision_observations.append({
                            "global_rank": rank, "sample_id": sample["sample_id"],
                            "state": state_case, "query_mode": mode, **report})
                    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
                    tracker_rng = copy.deepcopy(get_cuda_rng_tracker().get_states())
                    modes = query_modes(
                        hc_mode=options.hc_helper_mode, compare_compiled=options.compare_compiled,
                        with_backward=options.with_backward, source=sample["source"], state_case=state_case)
                    for mode_spec in modes:
                        mode = mode_spec["name"]
                        targets += 128
                        require(targets <= planned_targets, "mode-audit dispatch cap exceeded")
                        calls.append({"sample_id": sample["sample_id"], "source": sample["source"],
                                      "state": state_case, "mode": mode, "targets": 128,
                                      "ple_precision": selected_precision})
                        if rank == 0:
                            write_json(options.output / "progress.json", {
                                "dispatched_targets": targets,
                                "global_diagnostic_target_accesses": options.prior_diagnostic_targets + targets,
                                "calls": calls, "optimizer_steps": 0})
                        torch.set_rng_state(cpu_rng)
                        torch.cuda.set_rng_state(gpu_rng, device)
                        get_cuda_rng_tracker().set_states(copy.deepcopy(tracker_rng))
                        requires_grad = mode_spec["grad"]
                        memory = None
                        if state is not None:
                            values = state.values.detach().clone()
                            if state_case == "same_rows_zero_delta":
                                values.zero_()
                            memory = state.with_values(values.requires_grad_(requires_grad))
                        capture = None
                        if module_trace is not None:
                            module_trace.set_phase(
                                f"{sample['source']}:{selected_precision}:{state_case}:{mode}"
                                if mode != "grad_repeat" else None)
                        precision_context = (
                            Qwen38SplitPLEPrecision(
                                session.adapter, observer=observe_precision, compare_native_zero=True)
                            if selected_precision == "split" and session.adapter is not None
                            else nullcontext(None))
                        router_context = (
                            native_router_dispatch(session.raw) if options.router_topk == "sorted"
                            else nullcontext({"identity": {"protocol": "unchanged_native_router"},
                                              "restored": True, "routers": []}))
                        with precision_context, router_context as router, native_hc_dispatch(
                                unwrap=mode_spec["hc"] == "eager", trace=options.trace_hc) as hc:
                            hc.set_phase(f"{sample['source']}:{selected_precision}:{state_case}:{mode}"
                                         if options.trace_hc and mode != "grad_repeat" else None)
                            precision_identity = {
                                "mode": selected_precision,
                                "implementation_sha256": digest(precision_module.__file__)
                                if selected_precision == "split" else None,
                                "protocol": precision_module.PROTOCOL if selected_precision == "split"
                                else "unchanged_native_ple",
                            }
                            dispatch_sha = canonical_hash(
                                {"native_runtime": base_runtime_identity, "hc_dispatch": hc.identity,
                                 "router_dispatch": router["identity"], "ple_precision": precision_identity})
                            dispatch_identities[mode] = dispatch_sha
                            snapshot_context = (
                                capture_native_ple_snapshot(
                                    session.adapter, state=memory, valid_mask=valid, owner_ids=owners,
                                    runtime_identity_sha256=dispatch_sha,
                                    ple_source_sha256=base_runtime_identity["sources"]["mcore_bridge.model.modules.ple"],
                                    metadata={"delta_origin": "calibrated_initial_writer",
                                              "sample_id": sample["sample_id"], "source": sample["source"],
                                              "state_case": state_case, "mode": mode, "alpha": .99,
                                              "H_cache": "historical diagnostic only", "hc": mode_spec["hc"]})
                                if options.capture_ple and selected_precision == "native"
                                and sample["source"] == "books" and mode == "grad"
                                and session.adapter is not None else nullcontext(None))
                            with snapshot_context as snapshot:
                                result = schedule_query(
                                    session.models, batch, valid_mask=valid, owner_ids=owners,
                                    adapter=session.adapter, reader_state=memory, backward=mode_spec["backward"],
                                    query_grad_enabled=requires_grad, group_total_target_tokens=128)
                            if snapshot is not None:
                                snapshot_receipts.append(save_native_ple_snapshot(
                                    snapshot, options.output / f"ple-books-{state_case}-rank{rank}.pt"))
                        hc_reports.append(hc.report())
                        router_reports.append(router)
                        if mode_spec["backward"] and session.adapter is not None:
                            require(memory.values.grad is not None
                                    and bool(torch.isfinite(memory.values.grad).all()),
                                    "native training backward did not reach the delta leaf")
                            anchor_path = options.output / f"reader-anchor-books-{selected_precision}-rank{rank}.pt"
                            require(not anchor_path.exists(), "reader anchor already exists")
                            torch.save({
                                "protocol": "qwen38_raw_native_reader_anchor_v1",
                                "sample_id": sample["sample_id"], "state_case": state_case,
                                "ple_precision": precision_identity,
                                "runtime_identity_sha256": dispatch_sha,
                                "rows": memory.rows.detach().cpu(), "values": memory.values.detach().cpu(),
                                "counts": memory.counts.detach().cpu(),
                                "raw_stored_value_cotangent": memory.values.grad.detach().cpu(),
                                "lazy_metadata": {name: getattr(memory, name).detach().cpu()
                                    for name in ("block_clock", "cumulative_log_alpha",
                                                 "row_last_written_block", "row_log_alpha_at_write")},
                                "writer": {name: value.detach().cpu()
                                           for name, value in writer.state_dict().items()},
                                "query": query, "alpha": .99, "norm_mode": "rmsnorm",
                                "reader_loss_denominator": 128, "loss_sum": result["loss_sum"],
                                "tp_policy_applied": False, "writer_bptt_performed": False,
                                "scope": "raw native cotangent; requires matching TP/runtime acceptance before writer use",
                                "evidence_cache_status": contract["evidence_cache_status"],
                            }, anchor_path)
                            anchor_receipts.append({"rank": rank, **file_info(anchor_path),
                                                    "raw_gradient_l2": float(memory.values.grad.norm()),
                                                    "tp_policy_applied": False})
                        measurements[mode] = result["normalized_loss"]
                        if last:
                            require(capture is not None and capture.shape == (128, 124160),
                                    "missing aligned target logits")
                            parts = [torch.empty_like(capture) for _ in range(2)]
                            dist.all_gather(parts, capture, group=tp_group)
                            if tp == 0:
                                logits[mode] = torch.cat(parts, -1)
                            del parts
                        del result, memory
                        dist.barrier()
                    record = None
                    visibility_ranks = [None] * dist.get_world_size()
                    dist.all_gather_object(visibility_ranks, overlay_report)
                    precision_ranks = [None] * dist.get_world_size()
                    dist.all_gather_object(precision_ranks, precision_observations)
                    full_ple_effect = None
                    if session.adapter is not None and state_case == "same_rows_zero_delta":
                        full_outputs = {}
                        for case in contract["state_cases"]:
                            local_output = ple_outputs[case].contiguous()
                            parts = [torch.empty_like(local_output) for _ in range(2)]
                            dist.all_gather(parts, local_output, group=tp_group)
                            full_outputs[case] = torch.cat(parts, dim=0)
                        full_ple_effect = ple_effect(
                            full_outputs["calibrated_initial_writer"], full_outputs["same_rows_zero_delta"],
                            valid, labels.ne(-100))
                    if last and tp == 0:
                        with torch.no_grad():
                            record = {
                                "sample_id": sample["sample_id"], "source": sample["source"],
                                "ple_precision": selected_precision,
                                **runtime_identity_fields(dispatch_identities),
                                "state": state_case, "loss_nat_per_token": measurements,
                                "loss_difference_no_grad_minus_grad": measurements["no_grad"] - measurements["grad"],
                                "grad_replay_loss_bitwise_equal": measurements["grad_repeat"] == measurements["grad"],
                                "train_inference": compare_logits(logits["grad"], logits["no_grad"]),
                                "grad_replay": compare_logits(logits["grad"], logits["grad_repeat"]),
                            }
                            if options.compare_compiled:
                                record["compiled_vs_selected_eager"] = {
                                    "grad": compare_logits(logits["grad"], logits["compiled_grad"]),
                                    "no_grad": compare_logits(logits["no_grad"], logits["compiled_no_grad"]),
                                    "compiled_grad_vs_no_grad": compare_logits(
                                        logits["compiled_grad"], logits["compiled_no_grad"]),
                                    "loss_compiled_grad_minus_eager_grad": measurements["compiled_grad"] - measurements["grad"],
                                    "loss_compiled_no_grad_minus_eager_no_grad": measurements["compiled_no_grad"] - measurements["no_grad"]}
                            if "train_backward" in logits:
                                record["training_backward_forward_vs_eval_grad"] = compare_logits(
                                    logits["grad"], logits["train_backward"])
                                record["training_backward_forward_vs_no_grad"] = compare_logits(
                                    logits["no_grad"], logits["train_backward"])
                                record["loss_training_forward_minus_eval_grad"] = (
                                    measurements["train_backward"] - measurements["grad"])
                            key = (sample["sample_id"], state_case)
                            if options.ple_precision == "both":
                                if selected_precision == "native":
                                    precision_reference[key] = (
                                        logits["grad"].detach().cpu(), measurements["grad"])
                                else:
                                    native_logits, native_loss = precision_reference.pop(key)
                                    record["whole_reader_split_vs_native_grad"] = compare_logits(
                                        native_logits.to(logits["grad"].device), logits["grad"])
                                    record["whole_reader_loss_split_minus_native_grad"] = (
                                        measurements["grad"] - native_loss)
                    delivered = [record]
                    dist.broadcast_object_list(delivered, src=ps.get_pipeline_model_parallel_last_rank(),
                                               group=ps.get_pipeline_model_parallel_group())
                    if rank == 0:
                        require(delivered[0] is not None, "missing mode metrics from final-stage TP0")
                        reports.append(delivered[0])
                        reports[-1]["native_lookup_visibility_by_tp_rank"] = [
                            {"global_rank": index, "visibility": value}
                            for index, value in enumerate(visibility_ranks) if value is not None]
                        reports[-1]["split_precision_visibility_by_tp_rank"] = [
                            {"global_rank": index, "observations": value}
                            for index, value in enumerate(precision_ranks) if value]
                        if state_case == "same_rows_zero_delta":
                            reports[-1]["initial_writer_PLE_effect_full_positions"] = full_ple_effect
                        write_json(options.output / "metrics.json", reports)
                        print(json.dumps({"event": "mode_audit_case", "source": sample["source"],
                                          "state": state_case,
                                          "ple_precision": selected_precision,
                                          "loss_delta": delivered[0]["loss_difference_no_grad_minus_grad"],
                                          "logit_max_abs": delivered[0]["train_inference"]["difference_max_abs"],
                                          "replay_equal": delivered[0]["grad_replay"]["bitwise_equal"]}), flush=True)
                    del logits
                    if module_trace is not None:
                        trace_pairs.append((f"{sample['source']}:{selected_precision}:{state_case}:grad",
                                            f"{sample['source']}:{selected_precision}:{state_case}:no_grad"))
                    session.assert_frozen()
            require(targets == planned_targets, "mode audit did not consume its fixed plan")
            all_artifacts = [None] * dist.get_world_size()
            dist.all_gather_object(all_artifacts, {"rank": rank, "snapshots": snapshot_receipts,
                                                 "reader_anchors": anchor_receipts})
            if rank == 0:
                write_json(options.output / "COMPLETE.json", {
                    "protocol": contract["protocol"], "status": "complete", "registration": contract,
                    "identity": session.identity, "versions": session.versions, "cases": reports,
                    "loss_target_accesses": targets, "optimizer_steps": 0,
                    "backward_passes": contract["backward_passes"], "artifacts_by_rank": all_artifacts,
                    "reader_unchanged": True,
                    "all_grad_replays_bitwise_equal": all(row["grad_replay"]["bitwise_equal"]
                                                         and row["grad_replay_loss_bitwise_equal"] for row in reports),
                    "train_inference_all_logits_bitwise_equal": all(
                        row["train_inference"]["bitwise_equal"] for row in reports),
                })
        finally:
            if handle is not None:
                handle.remove()
            if ple_handle is not None:
                ple_handle.remove()
            if overlay_original is not None:
                session.adapter._overlay = overlay_original
            module_trace_context.__exit__(*sys.exc_info())
            if module_trace is not None:
                trace_report = module_trace.report(comparisons=trace_pairs)
                trace_report["model_config"] = {
                    name: getattr(session.models[0].config, name, None) for name in (
                        "num_layers", "num_moe_experts", "moe_layer_freq",
                        "moe_grouped_gemm", "moe_token_dispatcher_type",
                        "moe_shared_expert_intermediate_size", "moe_shared_expert_overlap",
                        "moe_router_dtype", "moe_router_pre_softmax", "moe_router_score_function",
                        "moe_router_fusion", "moe_router_num_groups", "moe_router_group_topk",
                        "moe_router_topk_scaling_factor",
                        "transformer_impl", "sequence_parallel", "fp8")}
                write_json(options.output / f"module_trace_rank{rank}.json", trace_report)
            write_json(options.output / f"router_dispatch_rank{rank}.json", router_reports)
        return {"mode_audit": "complete"}

    run_frozen_session(options, audit)
    hc_report = {
        "identity": hc_reports[0]["identity"], "dispatches": [x["identity"] for x in hc_reports],
        "trace": [row for report in hc_reports for row in report["trace"]],
        "restored_original_function_objects": all(
            x["restored_original_function_objects"] for x in hc_reports)}
    write_json(options.output / f"hc_dispatch_rank{rank}.json", hc_report)
    if rank == 0:
        receipt = json.loads((options.output / "COMPLETE.json").read_bytes())
        receipt["hc_dispatch_identity"] = hc_report["identity"]
        receipt["hc_helpers_restored"] = hc_report["restored_original_function_objects"]
        write_json(options.output / "COMPLETE.json", receipt)


if __name__ == "__main__":
    main()

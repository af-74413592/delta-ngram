"""Prepare the selected native runtime before separately registered main training.

--prepare-frames is CPU-only and binds every training/probe ID, token and mask.
--execute requires an explicitly authorized preparation candidate. Fresh H and
scales are collected with the final runtime before any optimizer is constructed.
All query phases use the SAME grad-enabled native forward; only training and the
registered writer finite difference invoke backward. Evidence alone is no_grad.
Current execution requires --prepare-only and exits before the historical
eight-step stage. That separate short stage is no longer authorized.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from qwen38_native_norm_probe import (
    canonical_hash, digest, file_info, observation_frames, read_index,
    read_tokens, require, verify_cache, write_json,
)
from qwen38_native_runtime_preflight import language_model, schedule_query
from qwen38_native_frozen_session import run_frozen_session


def code_manifest(delta_repo):
    scripts = Path(__file__).resolve().parent
    paths = [scripts / name for name in (
        "qwen38_native_paired_norm_probe.py", "qwen38_native_norm_probe.py",
        "qwen38_native_runtime_preflight.py", "qwen38_native_frozen_session.py",
        "qwen38_native_tp_gradient_probe.py", "qwen38_native_tp_structure.py",
        "qwen38_native_writer_adjoint_probe.py", "qwen38_native_router.py",
        "qwen38_native_bounded_checks.py", "qwen38_native_dispatch.py")]
    paths += [Path(delta_repo) / "src" / "delta_ngram" / name for name in (
        "qwen38_probe.py", "qwen38_probe_analysis.py", "qwen38_training.py", "qwen38_native.py",
        "qwen38_adapter.py", "qwen38_writer.py", "alpha_memory.py", "batching.py",
        "state.py", "operators.py", "qwen38_precision.py")]
    return {"protocol": "qwen38_native_runner_code_manifest_v1",
            "files": {path.name: file_info(path) for path in paths}}


def prepare_frames(options, registration):
    from delta_ngram.qwen38_probe import query_frame, validate_sample_registration
    frozen = validate_sample_registration(registration)
    root, cache, selected = verify_cache(options.cache_root)
    selection = json.loads((root / "calibration_selection.json").read_bytes())
    original = {sample["sample_id"]: sample for sample in selection["samples"]}
    for sample in frozen:
        require(all(sample.get(key) == value for key, value in original[sample["sample_id"]].items()),
                "registration changes sealed calibration membership")
    training = [sample for sample in frozen if sample["role"] == "training_and_scale_calibration"]
    probes = [sample for sample in frozen if sample["role"] == "fixed_train_side_probe"]
    require([s["sample_id"] for s in training] == [s["sample_id"] for s in selected],
            "training order differs from lossless observation")
    evidence = observation_frames(root, training) + observation_frames(root, probes)
    by_sample = {sample["sample_id"]: [] for sample in frozen}
    for frame in evidence:
        by_sample[frame["sample_id"]].append(frame)
    documents, events = read_index(root / "document_index.jsonl"), read_index(root / "event_index.jsonl")
    samples = []
    for sample in frozen:
        if sample["kind"] == "ordinary":
            entry = documents[sample["id"]]
        else:
            entry = events[sample["id"]]["articles"][1]
        require(entry["split"] == "train" and entry["group_id"] == sample["group_id"]
                and entry["source"] == sample["source"], "query cache source/owner/split mismatch")
        ids = read_tokens(root / "tokens.uint32", entry)
        samples.append({"sample": sample, "evidence": by_sample[sample["sample_id"]],
                        "query": query_frame(sample, ids)})
    return {"protocol": "qwen38_native_normshort_frames_v1",
            "cache_identity": cache, "samples": samples,
            "query_targets_per_sample": 128,
            "training_sample_ids": [s["sample_id"] for s in training],
            "probe_sample_ids": [s["sample_id"] for s in probes]}


def collect_native_evidence(session, frame):
    """One fresh no-label native forward; return cropped real H/rows on PP0."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from swift.megatron.utils import forward_step_helper
    device = session.device
    ids = torch.tensor([frame["input_ids"]], dtype=torch.long, device=device)
    mask = torch.tensor([frame["write_mask"]], dtype=torch.bool, device=device)
    head, handle = None, None
    if ps.is_pipeline_last_stage():
        lm = language_model(session.raw)

        def capture(module, args, kwargs):
            nonlocal head
            value = args[0] if args else kwargs.get("hidden_states", kwargs.get("input_"))
            require(head is None and tuple(value.shape) == (ids.shape[1] // 2, 1, 2560),
                    "unexpected native final-H shape or repeated head")
            head = value.detach().clone()

        handle = lm.output_layer.register_forward_pre_hook(capture, with_kwargs=True)
    inputs = {"input_ids": ids, "position_ids": torch.arange(
        ids.shape[1], device=device).unsqueeze(0), "labels": None, "attention_mask": None}
    try:
        with torch.no_grad():
            if session.adapter is not None:
                output, captured = session.adapter.run_native(
                    lambda: forward_step_helper(session.models[0], inputs),
                    input_ids=ids, valid_mask=mask,
                    owner_ids=torch.zeros(1, dtype=torch.long, device=device),
                    mode="evidence", state=None)
            else:
                output, captured = forward_step_helper(session.models[0], inputs), None
            del output
            full = torch.empty((ids.shape[1], 1, 2560), dtype=torch.bfloat16, device=device)
            if ps.is_pipeline_last_stage():
                require(head is not None, "native head was not observed")
                parts = [torch.empty_like(head) for _ in range(2)]
                dist.all_gather(parts, head.contiguous(), group=ps.get_tensor_model_parallel_group())
                full.copy_(torch.cat(parts))
            dist.broadcast(full, src=ps.get_pipeline_model_parallel_last_rank(),
                           group=ps.get_pipeline_model_parallel_group())
            if session.adapter is None:
                return None
            start, length = frame["prefix_tokens"], frame["new_tokens"]
            return {"frame": frame, "H_new": full[start:start + length].transpose(0, 1).cpu(),
                    "physical_rows_new": captured.physical_rows[:, start:start + length].cpu(),
                    "T0_head_rms_new": captured.base_head_rms.cpu(),
                    "native_ids": captured.native_ids.cpu(), "loss_labels": None}
    finally:
        if handle is not None:
            handle.remove()


def verify_initial_parameters(options, frames):
    """Reuse only immutable seed-17 parameters; historical H is never loaded."""
    import torch
    root = options.observation_root
    report = json.loads((root / "COMPLETE.json").read_bytes())
    require(report.get("status") == "complete" and report["optimizer_steps"] == 0
            and report["loss_target_tokens"] == 0, "requires completed zero-update H observation")
    initial = None
    for artifact in report["artifacts"]:
        if Path(artifact["path"]).name != "writer_initial_state.pt":
            continue
        path = (root / artifact["path"]).resolve()
        require(path.is_relative_to(root.resolve())
                and file_info(path) == {k: artifact[k] for k in ("bytes", "sha256")},
                "observation artifact differs from completed receipt")
        value = torch.load(path, map_location="cpu", weights_only=True)
        initial = value
    require(initial is not None and torch.equal(initial["target_scale"], torch.ones(16)),
            "missing initial unit-scale writer state")
    return report, initial


def verify_observation(options, frames):
    """Historical audit compatibility; bounded training never uses these H."""
    import torch
    report, initial = verify_initial_parameters(options, frames)
    root, by_frame = options.observation_root, {}
    for artifact in report["artifacts"]:
        if Path(artifact["path"]).name == "writer_initial_state.pt":
            continue
        path = (root / artifact["path"]).resolve()
        require(path.is_relative_to(root.resolve())
                and file_info(path) == {key: artifact[key] for key in ("bytes", "sha256")},
                "historical observation artifact differs from completed receipt")
        value = torch.load(path, map_location="cpu", weights_only=True)
        by_frame[value["frame"]["frame_sha256"]] = value
    history = {}
    for item in frames["samples"]:
        if item["sample"]["role"] == "training_and_scale_calibration":
            blocks = [by_frame[frame["frame_sha256"]] for frame in item["evidence"]]
            require(all(block["frame"] == frame for block, frame in zip(blocks, item["evidence"])),
                    "historical observation frame changed")
            history[item["sample"]["sample_id"]] = blocks
    return report, initial, history


def preparation_contract():
    """Exact preregistration payload; CPU callers may import before freezing."""
    return {
        "protocol": "qwen38_split_precision_bounded_preparation_v1",
        "diagnostic_target_allocation": 1024, "fresh_training_H_and_scale": True,
        "reuse_TP_proof_for_unchanged_mapping_only": True,
        "router_policy": "always_sorted_topk", "HC_policy": "eager",
        "direction_seed": 20261020,
        "direction_distribution": "Rademacher", "direction_rms": 1.0,
        "direction_coordinates": "valid_materialized_memory",
        "absolute_steps": [2.0 ** -14, 2.0 ** -13], "rtol": .20, "atol": .001,
        "minimum_absolute_analytic": .01, "exact_baseline_repeat": True,
        "automatic_retry": False,
    }


def tp_session_receipt_path(options):
    explicit = getattr(options, "tp_session_receipt", None)
    return explicit if explicit is not None else options.tp_gradient_receipt.parent.parent / "COMPLETE.json"


def tp_mapping_provenance(options, registration, tp_report):
    """Read source pins from the SAME accepted TP run, before loading a model."""
    path = tp_session_receipt_path(options)
    expected_sha = registration["required_evidence"].get("native_TP_session_receipt_sha256")
    require(isinstance(expected_sha, str) and digest(path) == expected_sha,
            "accepted TP session receipt is missing or differs from its registered SHA")
    outer = json.loads(path.read_bytes())
    require(outer.get("protocol") == "native_tp_structure_only_v1"
            and outer.get("optimizer_steps") == 0
            and outer.get("result") == tp_report
            and tp_report.get("complete") is True and tp_report.get("status") == "accepted",
            "TP session receipt does not contain the exact accepted structural result")
    identity = outer.get("identity", {})
    require(identity.get("verification_sha256") == digest(options.verification),
            "accepted TP run and current checkpoint verification differ")
    sources = {
        "megatron.core.tensor_parallel.mappings":
            tp_report["selected_policy_receipt"]["operator_source_sha256"],
        "delta_ngram.qwen38_adapter": identity.get("adapter_sha256"),
        "delta_ngram.qwen38_native": identity.get("native_boundary_sha256"),
    }
    require(all(isinstance(value, str) and len(value) == 64
                and all(char in "0123456789abcdef" for char in value) for value in sources.values()),
            "accepted TP session lacks valid mapping source SHA256 pins")
    for module, filename in (("delta_ngram.qwen38_adapter", "qwen38_adapter.py"),
                             ("delta_ngram.qwen38_native", "qwen38_native.py")):
        require(digest(options.delta_repo / "src" / "delta_ngram" / filename) == sources[module],
                f"{module} source changed since the accepted TP structure proof")
    return {"session_receipt_sha256": expected_sha, "sources": sources}


def validate_candidate(options, registration, frames, codes, observation, tp_report, ple_report):
    from delta_ngram.qwen38_probe import policy_from_receipt, validate_numeric_registration
    validate_numeric_registration(registration)
    require(registration.get("status") == "PREPARATION_FROZEN_CONDITIONAL"
            and registration.get("conditional_execution_authorized") is True
            and registration.get("diagnostic_execution_allowed") is True
            and registration.get("optimizer_execution_allowed") is False,
            "requires explicit frozen preparation authorization; drafts cannot dispatch")
    require(options.prepare_only
            and registration.get("execution_scope") == "preparation_only_for_main_training",
            "the separate eight-step stage is canceled; only main-training preparation is authorized")
    required = registration["required_evidence"]
    actual = {
        "derived_complete_seal_sha256": frames["cache_identity"]["complete"]["sha256"],
        "derived_data_manifest_sha256": frames["cache_identity"]["data_manifest"]["sha256"],
        "calibration24_selection_manifest_sha256": frames["cache_identity"]["calibration_selection"]["sha256"],
        "source_data_manifest_sha256": frames["cache_identity"]["source_manifest_sha256"],
        "frame_manifest_sha256": canonical_hash(frames),
        "runner_code_manifest_sha256": canonical_hash(codes),
        "model_weights_receipt_sha256": digest(options.verification),
        "writer_initial_parameters_sha256": digest(options.observation_root / "writer_initial_state.pt"),
        "native_TP_structural_adjoint_receipt_sha256": digest(options.tp_gradient_receipt),
        "native_TP_session_receipt_sha256": digest(tp_session_receipt_path(options)),
        "native_PLE_FP64_derivative_receipt_sha256": digest(options.ple_reference_receipt),
        "parent_budget_ledger_sha256": digest(options.parent_budget_ledger),
        "bounded_stability_registration_sha256": canonical_hash(
            registration["current_execution_policy"]["stability_thresholds_and_stop_rules"]),
    }
    require(all(required.get(key) == value for key, value in actual.items()),
            "candidate evidence/code/frame/scale binding differs")
    require(tp_report.get("status") == "accepted", "native TP structure proof is not accepted")
    policy = policy_from_receipt(tp_report["selected_policy_receipt"])
    require(policy.receipt()["protocol"] == "qwen38_native_tp_structure_v1"
            and policy.reduction == "replica", "requires accepted structural TP replica provenance")
    tp_mapping_provenance(options, registration, tp_report)
    require(registration["scale_calibration"]["rho"] == .01,
            "only the single root-frozen rho=.01 candidate is authorized")
    require(registration["scale_calibration"]["target_scales_by_mode"] is None
            and required["H_T0_raw_target_observation_sha256"] is None
            and required["scale_calibration_receipt_sha256"] is None,
            "preparation must regenerate H/scales; historical calibration is inactive")
    require(ple_report.get("status") == "reference_fd_passed"
            and ple_report.get("complete") is True, "PLE FP64 reference derivative check failed")
    contract = registration["preparation_contract"]
    require(contract == preparation_contract(), "preparation/E2E contract differs from preregistration")
    parent = json.loads(options.parent_budget_ledger.read_bytes())
    budgets = registration["budgets"]
    require(all(parent.get(name) == budgets[name] for name in (
        "prior_diagnostic_target_accesses", "prior_writer_optimizer_target_accesses",
        "prior_normshort_target_accesses")), "prior consumption does not match parent ledger")
    require(registration["current_execution_policy"]["A_no_grad_parity_is_hard_training_block"] is False,
            "no-grad parity must remain a parallel investigation")
    require(observation["identity"]["verification_sha256"] == actual["model_weights_receipt_sha256"],
            "observation and execution checkpoint receipts differ")
    return policy


def execute(options, registration, frames, codes, initial, prior_tp_policy, observation):
    mapping_provenance = tp_mapping_provenance(
        options, registration, json.loads(options.tp_gradient_receipt.read_bytes()))
    rank = int(os.environ.get("RANK", "-1"))
    require(rank >= 0 and int(os.environ.get("WORLD_SIZE", "0")) == 4, "execution requires torchrun4")
    marker = options.output / "candidate_registration.json"
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        write_json(marker, registration)
        write_json(options.output / "frames.json", frames)
        write_json(options.output / "code_manifest.json", codes)
    else:
        deadline = time.monotonic() + 300
        while not marker.is_file():
            require(time.monotonic() < deadline, "rank0 output preparation timed out")
            time.sleep(.1)
    options.tokens = 640

    def run(session):
        import torch
        import torch.distributed as dist
        from megatron.core import parallel_state as ps
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        from delta_ngram.qwen38_native import make_qwen38_writer
        from delta_ngram.qwen38_precision import Qwen38SplitPLEPrecision
        from delta_ngram.qwen38_training import NativeTrajectoryGradients, distributed_tp_gradient
        from delta_ngram.qwen38_probe import (
            ARMS, NativeProbeLedger, build_history_memory, parameter_change_norm, tensor_state_hash)
        from qwen38_native_tp_gradient_probe import live_runtime_identity, tensor_hash
        from qwen38_native_writer_adjoint_probe import run_fixed_cotangent_writer_probe
        from qwen38_native_router import native_router_dispatch
        from qwen38_native_dispatch import native_hc_dispatch
        from qwen38_native_bounded_checks import BoundedStability, evaluate_end_to_end
        started = time.monotonic()
        first = session.adapter is not None
        device = session.device
        history, scales = {}, {}
        runtime_scope = ExitStack()
        hc_report = runtime_scope.enter_context(native_hc_dispatch(unwrap=True))
        router_report = runtime_scope.enter_context(native_router_dispatch(session.raw))
        visibility = []
        precision = Qwen38SplitPLEPrecision(session.adapter) if first else None
        if first:
            runtime_scope.enter_context(precision)
        base_runtime = live_runtime_identity(session.models)
        unchanged_sources = mapping_provenance["sources"]
        require(all(base_runtime["sources"].get(key) == value for key, value in unchanged_sources.items())
                and base_runtime["topology"] == {"tp": 2, "pp": 2, "ep": 2, "etp": 1, "cp": 1, "dp": 1}
                and base_runtime["reader"]["sequence_parallel"] is True
                and base_runtime["reader"]["recompute_granularity"] is None
                and base_runtime["reader"]["deallocate_pipeline_outputs"] is False,
                "TP mapping source/topology differs from accepted structural provenance")
        budgets = registration["budgets"]
        ledger = NativeProbeLedger(
            prior_diagnostic=budgets["prior_diagnostic_target_accesses"],
            prior_optimizer=budgets["prior_writer_optimizer_target_accesses"],
            prior_normshort=budgets["prior_normshort_target_accesses"],
            save=lambda value: write_json(options.output / "budget_ledger.json", value) if rank == 0 else None)
        planned_targets = 2304 + 1024
        require(ledger.prior_normshort + planned_targets <= 75000
                and ledger.prior_diagnostic + planned_targets <= 100000
                and ledger.prior_optimizer <= 100000000,
                "cannot reserve the complete preparation within remaining budgets")
        ledger.save(ledger.report())
        samples = {item["sample"]["sample_id"]: item for item in frames["samples"]}
        train_ids, probe_ids = frames["training_sample_ids"], frames["probe_sample_ids"]
        perturb_ids = [key for key in train_ids if samples[key]["sample"]["initial_perturbation_member"]]
        require(len(perturb_ids) == 6, "initial perturbation needs one training family per source")
        first_event = next(key for key in train_ids if samples[key]["sample"]["source"] == "news_event")
        require(len(samples[first_event]["evidence"]) >= 2, "writer FD requires at least two history writes")
        probe_rows, training_rows, checkpoints = [], [], []
        completed_optimizer_updates = 0

        def gathered(value):
            result = [None] * dist.get_world_size()
            dist.all_gather_object(result, value)
            return result

        def collectively(condition, message):
            failures = [value for value in gathered(None if condition else f"rank {rank}: {message}")
                        if value is not None]
            require(not failures, "; ".join(failures))

        precision_identities = [value for value in gathered(
            precision.identity() if first else None) if value is not None]
        require(len(precision_identities) == 2 and precision_identities[0] == precision_identities[1],
                "split precision identity differs across TP replicas")
        runtime_identity = {
            "native": base_runtime, "precision": precision_identities[0],
            "router": router_report["identity"],
            "HC_dispatch": hc_report.identity,
            "TP_structure_provenance": {
                "receipt_sha256": digest(options.tp_gradient_receipt),
                "session_receipt_sha256": mapping_provenance["session_receipt_sha256"],
                "original_runtime_sha256": prior_tp_policy.runtime_identity_sha256,
                "unchanged_mapping_sources": unchanged_sources,
                "scope": "mapping/scatter scaling only; new local arithmetic has separate gates"},
            "memory_boundary": "materialized FP32 BatchMemoryState; all history writes remain differentiable",
        }
        runtime_hashes = gathered(canonical_hash(runtime_identity))
        require(len(set(runtime_hashes)) == 1, "runtime identities differ across model ranks")
        # This rebinding carries the old structural proof only over unchanged
        # TP mappings. It does not assert that the new PLE arithmetic was tested.
        tp_policy = replace(prior_tp_policy, runtime_identity_sha256=runtime_hashes[0])
        if rank == 0:
            write_json(options.output / "runtime_identity.json", runtime_identity)

        def rng_state():
            return (torch.get_rng_state(), torch.cuda.get_rng_state(device),
                    copy.deepcopy(get_cuda_rng_tracker().get_states()))

        def replay_rng(state):
            torch.set_rng_state(state[0])
            torch.cuda.set_rng_state(state[1], device)
            get_cuda_rng_tracker().set_states(copy.deepcopy(state[2]))

        def materialized_history(writer, alpha, key, *, diagnostics=None, writer_adjoint_view=False):
            lazy = build_history_memory(
                writer, alpha, history[key], norm_mode="rmsnorm",
                row_capacity=session.layout.row_capacity, device=device, diagnostics=diagnostics)
            effective = lazy._lookup_packed(lazy.rows)[0]
            if writer_adjoint_view:
                return SimpleNamespace(rows=lazy.rows, values=effective, counts=lazy.counts,
                                       block_clock=lazy.block_clock)
            return lazy.memory.with_values(effective)

        def check_replicas(writer, tag):
            hashes = [value for value in gathered(
                tensor_state_hash(writer.state_dict()) if first else None) if value is not None]
            require(len(hashes) == 2 and len(set(hashes)) == 1,
                    f"native writer TP replicas differ at {tag}")
            return hashes[0]

        def new_writer(arm):
            if not first:
                return None, None
            writer, alpha = make_qwen38_writer(alpha=arm[1])
            writer.load_state_dict(initial, strict=True)
            writer = writer.to(device).float().train()
            with torch.no_grad():
                writer.target_scale.copy_(torch.tensor(scales[arm[2]], dtype=torch.float32, device=device))
            alpha = alpha.to(device)
            require(all(value.requires_grad for value in writer.parameters())
                    and not any(value.requires_grad for value in alpha.parameters()),
                    "only native writer parameters may be trainable")
            return writer, alpha

        def query(key, writer, alpha, mode, *, tag, backward=False, denominator=128,
                  optimizer_targets=False, empty=False, diagnostics=False, return_cotangent=False,
                  fixed_state=None, capture_logits=False, observe_precision=False):
            item, frame = samples[key], samples[key]["query"]
            stats = {} if diagnostics and first else None
            ticket = bridge = state = None
            actual_cotangent = actual_rows = None
            logits = None
            head_handle = None
            gate_handle = None
            gate_counts = [0, 0]
            delta_packets = []
            if diagnostics and first:
                def gates(module, args, output):
                    beta = output.detach().float().sigmoid()
                    gate_counts[0] += int(((beta < .001) | (beta > .999)).sum())
                    gate_counts[1] += beta.numel()
                gate_handle = writer.gate_head[2].register_forward_hook(gates)
            if capture_logits and ps.is_pipeline_last_stage():
                def capture(module, args, output):
                    nonlocal logits
                    value = output[0] if isinstance(output, tuple) else output
                    require(logits is None and value.ndim == 3
                            and value.shape[1:] == (1, 124160),
                            "unexpected TP-sharded native logits")
                    selection = torch.tensor(frame["labels"], device=device).ne(-100)
                    logits = value[:, 0][selection].detach().clone()
                head_handle = language_model(session.raw).output_layer.register_forward_hook(capture)
            if first:
                precision.observer = (lambda report: visibility.append({
                    "rank": rank, "tag": tag, "sample_id": key, **report})) if observe_precision else None
                precision.compare_native_zero = observe_precision
                if return_cotangent:
                    def retain_delta(packet):
                        require(packet["mode"] == "query" and packet["dense_delta"].requires_grad,
                                "anchor requires the differentiable FP32 split-delta tensor")
                        packet["dense_delta"].retain_grad()
                        delta_packets.append(packet)
                    precision.delta_observer = retain_delta
            # In every query phase, retain the same grad-enabled graph route.
            with torch.enable_grad():
                if first:
                    if fixed_state is not None:
                        state = fixed_state.with_values(
                            fixed_state.values.detach().clone().requires_grad_(True))
                    else:
                        if empty:
                            zero_writer, zero_alpha = new_writer(("empty", .99, "rmsnorm"))
                            with torch.no_grad():
                                state = materialized_history(zero_writer, zero_alpha, key)
                            state = state.with_values(torch.zeros_like(state.values).requires_grad_(True))
                            del zero_writer, zero_alpha
                        else:
                            state = materialized_history(writer, alpha, key, diagnostics=stats)
                        if backward:
                            bridge = NativeTrajectoryGradients(
                                total_loss_tokens=denominator, expected_loss_tokens=128,
                                policy=tp_policy, runtime_identity_sha256=tp_policy.runtime_identity_sha256)
                            ticket = bridge.prepare_query(state)
                            state = ticket.reader_state
                        elif not empty:
                            state = state.with_values(state.values.detach().clone().requires_grad_(True))
                if gate_handle is not None:
                    gate_handle.remove()
                    gate_handle = None
                collectively(not first or bool(state.values.isfinite().all()),
                             "nonfinite materialized memory")
                batch = {
                    "input_ids": torch.tensor([frame["input_ids"]], dtype=torch.long, device=device),
                    "labels": torch.tensor([frame["labels"]], dtype=torch.long, device=device),
                    "position_ids": torch.tensor([frame["position_ids"]], dtype=torch.long, device=device),
                    "attention_mask": None,
                }
                valid = torch.tensor([frame["encoder_valid_mask"]], dtype=torch.bool, device=device)
                owners = torch.zeros(1, dtype=torch.long, device=device)
                ledger.reserve(tag, targets=128, inputs=len(frame["input_ids"]),
                               optimizer=optimizer_targets)
                try:
                    result = schedule_query(
                        session.models, batch, valid_mask=valid, owner_ids=owners,
                        adapter=session.adapter, reader_state=state, backward=backward,
                        mode="query", group_total_target_tokens=denominator, query_grad_enabled=True)
                finally:
                    if head_handle is not None:
                        head_handle.remove()
                    if first:
                        precision.observer = None
                        precision.compare_native_zero = False
                        precision.delta_observer = None
                if bridge is not None:
                    if return_cotangent:
                        actual_cotangent = distributed_tp_gradient(
                            ticket.leaf_values.grad, tp_policy,
                            group=ps.get_tensor_model_parallel_group())
                        actual_rows = state.rows.detach().clone()
                    bridge.record_query(
                        ticket, loss_sum=result["loss_sum"], loss_tokens=128,
                        reader_loss_divisor=denominator,
                        reducer=lambda gradient, policy: distributed_tp_gradient(
                            gradient, policy, group=ps.get_tensor_model_parallel_group()))
                    bridge.backward_writer()
                if return_cotangent:
                    lookup_adjoint = None
                    if first:
                        from delta_ngram.qwen38_training import (
                            manual_split_delta_adjoint, split_delta_adjoint_report)
                        require(len(delta_packets) == 1, "anchor must observe exactly one split delta")
                        packet = delta_packets[0]
                        dense_grad = packet["dense_delta"].grad
                        require(dense_grad is not None and dense_grad.dtype == torch.float32
                                and tuple(dense_grad.shape) == (1, len(frame["input_ids"]), 2560)
                                and bool(packet["owner_ids"].eq(0).all()),
                                "split-delta anchor shape, owner or gradient differs")
                        expected = manual_split_delta_adjoint(
                            state, physical_rows=packet["physical_rows"],
                            dense_delta_cotangent=dense_grad, valid_mask=packet["valid_mask"],
                            owner_ids=packet["owner_ids"])
                        replicas = [torch.empty_like(actual_cotangent) for _ in range(2)]
                        dist.all_gather(replicas, actual_cotangent.contiguous(),
                                        group=ps.get_tensor_model_parallel_group())
                        lookup_adjoint = split_delta_adjoint_report(
                            expected, actual_cotangent, replica_gradients=replicas)
                        left, right = (value.detach().double() for value in replicas)
                        difference = left - right
                        left_norm, right_norm = float(left.norm()), float(right.norm())
                        difference_norm = float(difference.norm())
                        local_error = (expected - actual_cotangent).abs()
                        lookup_adjoint.update(
                            rank=rank,
                            dense_cotangent_sha256=tensor_hash(dense_grad),
                            dense_shape=list(dense_grad.shape),
                            local_mismatched_components=int((
                                local_error > 1e-7 + 1e-6 * expected.abs()).sum()),
                            tp_pair_max_abs=float(difference.abs().max()),
                            tp_pair_difference_l2=difference_norm,
                            tp_pair_relative_l2=(difference_norm / left_norm if left_norm else None),
                            tp_pair_cosine=(float((left * right).sum()) / (left_norm * right_norm)
                                            if left_norm and right_norm else None),
                            tp_pair_mismatched_components=int(left.ne(right).sum()),
                            memory_boundary="materialized FP32; lazy decay differentiated upstream")
                        if lookup_adjoint["status"] != "accepted":
                            path = options.output / f"split_lookup_adjoint-rank-{rank}.pt"
                            torch.save({
                                "rows": state.rows.detach().cpu(),
                                "physical_rows": packet["physical_rows"].detach().cpu(),
                                "valid_mask": packet["valid_mask"].detach().cpu(),
                                "owner_ids": packet["owner_ids"].detach().cpu(),
                                "dense_delta_cotangent": dense_grad.detach().cpu(),
                                "manual_reference": expected.detach().cpu(),
                                "observed_raw_cotangent": actual_cotangent.detach().cpu(),
                                "row_capacity": state.row_capacity,
                                "batch_size": state.batch_size,
                                "memory_boundary": "materialized FP32",
                            }, path)
                            lookup_adjoint["failure_tensors"] = {"path": path.name, **file_info(path)}
                    lookup_adjoints = [value for value in gathered(lookup_adjoint) if value is not None]
                    if rank == 0:
                        write_json(options.output / "split_lookup_adjoints.json", {
                            "protocol": "qwen38_split_lookup_anchor_evidence_v1",
                            "tag": tag, "sample_id": key, "runtime_identity_sha256": runtime_hashes[0],
                            "observations": lookup_adjoints, "budget": ledger.report(),
                            "optimizer_steps": 0, "extra_CE_targets": 0,
                            "acceptance_thresholds_changed": False})
                    collectively(len(lookup_adjoints) == 2 and all(
                        value["status"] == "accepted" for value in lookup_adjoints),
                        "actual split-delta manual lookup adjoint failed")
                    hashes = [value for value in gathered(
                        tensor_hash(actual_cotangent) if first else None) if value is not None]
                    collectively(len(hashes) == 2 and len(set(hashes)) == 1,
                                 "actual anchor FP32 cotangent TP replicas differ")
            session.assert_frozen()
            collectively(math.isfinite(result["loss_sum"]), "nonfinite loss")
            summary = {
                "sample_id": key, "source": item["sample"]["source"], "tag": tag,
                "loss_sum": result["loss_sum"], "query_target_tokens": 128,
                "loss_target_tokens": 128, "normalized_loss": result["loss_sum"] / 128,
                "reader_loss_denominator": denominator, "query_grad_enabled": True,
                "write_blocks": 0 if empty else len(item["evidence"]),
            }
            if diagnostics:
                reports = [value for value in gathered(stats if first else None) if value is not None]
                require(len(reports) == 2, "missing writer diagnostics from TP replicas")
                summary["writer_diagnostics"] = reports[0]
                counts = [value for value in gathered(gate_counts if first else None) if value is not None]
                require(len(counts) == 2 and counts[0] == counts[1], "beta counters differ across TP")
                summary["beta_saturated_events"], summary["beta_valid_events"] = counts[0]
            if capture_logits:
                summary["logits_sha256_by_rank"] = gathered(tensor_hash(logits) if logits is not None else None)
            if return_cotangent:
                require(backward and not empty, "actual cotangent requires a nonempty backward query")
                summary["_reader_cotangent"] = actual_cotangent
                summary["_reader_rows"] = actual_rows
                summary["_reader_state"] = state
                summary["split_lookup_adjoints"] = lookup_adjoints
                summary["memory_boundary"] = "materialized effective FP32 delta; full lazy history differentiated upstream"
            return summary

        def save_checkpoint(writer, optimizer, arm, step):
            signature = check_replicas(writer, f"{arm[0]}:{step}")
            if rank == 0:
                path = options.output / f"{arm[0]}-step-{step:02d}.pt"
                torch.save({"arm": arm, "optimizer_step": step,
                            "writer": {name: value.detach().cpu() for name, value in writer.state_dict().items()},
                            "optimizer": optimizer.state_dict(), "writer_sha256": signature,
                            "final_registration_file_sha256": digest(options.output / "final_registration.json"),
                            "reader_saved": False}, path)
                record = {"arm": arm[0], "step": step, "writer_sha256": signature,
                          "path": path.name, **file_info(path)}
                checkpoints.append(record)
                write_json(path.with_suffix(".json"), record)

        try:
            # H and scale statistics are generated only after both numerical
            # policies are installed. The old observation supplies parameters.
            artifacts = []
            for key in train_ids:
                parts = []
                for frame in samples[key]["evidence"]:
                    ledger.reserve("training-evidence:" + frame["frame_sha256"],
                                   targets=0, inputs=len(frame["input_ids"]))
                    block = collect_native_evidence(session, frame)
                    if first:
                        parts.append(block)
                    if rank == 0:
                        directory = options.output / "training_evidence"
                        directory.mkdir(exist_ok=True)
                        path = directory / f'{frame["frame_sha256"]}.pt'
                        torch.save(block, path)
                        artifacts.append({"path": str(path.relative_to(options.output)), **file_info(path)})
                if first:
                    history[key] = parts
            calibration = None
            if first:
                from delta_ngram.qwen38_training import writer_evidence_features
                from delta_ngram.state import SparseMemoryState
                unit_writer, _ = make_qwen38_writer(alpha=1.)
                unit_writer.load_state_dict(initial, strict=True)
                unit_writer = unit_writer.to(device).float().eval().requires_grad_(False)
                raw_square = torch.zeros(16, dtype=torch.float64, device=device)
                base_square = torch.zeros_like(raw_square)
                token_count = 0
                with torch.no_grad():
                    for key in train_ids:
                        for block in history[key]:
                            h = block["H_new"].to(device)
                            rows = block["physical_rows_new"].to(device)
                            mask = torch.ones(h.shape[:2], dtype=torch.bool, device=device)
                            events = unit_writer.events(
                                writer_evidence_features(h, mode="rmsnorm", eps=1e-6),
                                rows, mask, SparseMemoryState.empty(160, device))
                            target = events.target.double().reshape(-1, 16, 160)
                            require(bool(target.isfinite().all()), "nonfinite raw calibration target")
                            raw_square += target.square().sum((0, 2))
                            count = h.shape[1]
                            base_square += block["T0_head_rms_new"].to(device).double().square() * count
                            token_count += count
                raw_rms = (raw_square / (token_count * 160)).sqrt()
                base_rms = (base_square / token_count).sqrt()
                require(bool(raw_rms.gt(0).all()) and bool(raw_rms.isfinite().all())
                        and bool(base_rms.gt(0).all()) and bool(base_rms.isfinite().all()),
                        "zero/nonfinite scale calibration; no clamp or fallback authorized")
                vector = (.01 * base_rms / raw_rms).float()
                calibration = {
                    "protocol": "qwen38_selected_runtime_scale_v1", "status": "complete",
                    "runtime_identity_sha256": runtime_hashes[0],
                    "frame_manifest_sha256": canonical_hash(frames), "rho": .01,
                    "raw_target_rms_by_head": raw_rms.cpu().tolist(),
                    "T0_rms_by_head": base_rms.cpu().tolist(),
                    "target_scale_float32_by_mode": {"rmsnorm": vector.cpu().tolist()},
                    "new_evidence_tokens": token_count, "loss_target_tokens": 0,
                    "optimizer_steps": 0, "historical_H_or_scales_reused": False,
                }
                del unit_writer
            calibrations = [value for value in gathered(calibration) if value is not None]
            require(len(calibrations) == 2 and calibrations[0] == calibrations[1],
                    "fresh calibration differs across TP replicas")
            calibration = calibrations[0]
            scales = calibration["target_scale_float32_by_mode"]
            if rank == 0:
                write_json(options.output / "fresh_observation.json", {
                    "status": "complete", "runtime_identity_sha256": runtime_hashes[0],
                    "artifacts": artifacts, "loss_target_tokens": 0, "optimizer_steps": 0,
                    "writer_initial_parameters_sha256": digest(
                        options.observation_root / "writer_initial_state.pt")})
                write_json(options.output / "fresh_scale_calibration.json", calibration)
            dist.barrier()
            session.assert_frozen()

            # One baseline per frozen training calibration member; shared by all arms.
            empty_perturb = {}
            empty_anchor = empty_rng = None
            for key in perturb_ids:
                saved_rng = rng_state()
                row = query(key, None, None, None, tag="initial-empty:" + key, empty=True,
                            capture_logits=key == first_event, observe_precision=key == first_event)
                empty_perturb[key] = row["loss_sum"]
                if key == first_event:
                    empty_anchor, empty_rng = row, saved_rng
            initial_rows = []
            for arm in ARMS:
                writer, alpha = new_writer(arm)
                check_replicas(writer, "initial:" + arm[0])
                for key in perturb_ids:
                    row = query(key, writer, alpha, arm[2], tag="initial-self:" + arm[0] + ":" + key,
                                observe_precision=True)
                    row.update(arm=arm[0], empty_loss_sum=empty_perturb[key],
                               delta_empty=(empty_perturb[key] - row["loss_sum"]) / 128)
                    initial_rows.append(row)
                del writer, alpha
            initial_receipt = {"protocol": "qwen38_native_initial_perturbation_v1",
                               "status": "observed",
                               "target_accesses": 2304, "rows": initial_rows,
                               "hard_gate": False, "rho": .01,
                               "criterion": "report empty/self changes; no amplitude rejection or automatic retuning"}
            if rank == 0:
                write_json(options.output / "initial_perturbation.json", initial_receipt)
            dist.barrier()
            require(empty_anchor is not None, "first event absent from initial empty allocation")
            replay_rng(empty_rng)
            if first:
                precision.__exit__(None, None, None)
            try:
                native_empty = query(first_event, None, None, None,
                                     tag="diagnostic:historical-native-empty",
                                     empty=True, capture_logits=True)
            finally:
                if first:
                    precision.__enter__()
            empty_comparison = {
                "scope": "historical native PLE versus split PLE; same sorted-router policy and RNG",
                "native": native_empty, "split": empty_anchor,
                "split_minus_native_nll": (empty_anchor["loss_sum"] - native_empty["loss_sum"]) / 128,
                "same_logits": empty_anchor["logits_sha256_by_rank"] == native_empty["logits_sha256_by_rank"],
                "identity_required": False, "additional_target_accesses": 128,
            }
            if rank == 0:
                write_json(options.output / "empty_runtime_comparison.json", empty_comparison)

            fd_arm = ARMS[1]  # fixed alpha .99, RMSNorm, first training news_event.
            writer, alpha = new_writer(fd_arm)
            if first:
                writer.zero_grad(set_to_none=True)
            anchor_rng = rng_state()
            anchor = query(first_event, writer, alpha, fd_arm[2],
                           tag="writer-adjoint:real-reader-anchor", backward=True,
                           return_cotangent=True, capture_logits=True, observe_precision=True)
            actual_cotangent = anchor.pop("_reader_cotangent")
            actual_rows = anchor.pop("_reader_rows")
            anchor_state = anchor.pop("_reader_state")
            replay_rng(anchor_rng)
            matched = query(first_event, writer, alpha, fd_arm[2],
                            tag="diagnostic:actual-training-forward-comparison",
                            fixed_state=anchor_state, capture_logits=True)
            matched_exact = (anchor["loss_sum"] == matched["loss_sum"]
                             and anchor["logits_sha256_by_rank"] == matched["logits_sha256_by_rank"])
            comparison = {"status": "accepted" if matched_exact else "rejected",
                          "training_forward": anchor, "main_evaluation": matched,
                          "loss_and_logits_bitwise_equal": matched_exact,
                          "same_grad_enabled_requires_grad_leaf": True}
            if rank == 0:
                write_json(options.output / "actual_training_forward_comparison.json", comparison)
            require(matched_exact, "actual training forward differs from main evaluation forward")
            writer_fd = run_fixed_cotangent_writer_probe(
                writer=writer, sample_id=first_event, write_blocks=len(samples[first_event]["evidence"]),
                cotangent=actual_cotangent, rows=actual_rows, anchor_receipt=anchor,
                output=options.output / "writer_fd",
                build_state=lambda: materialized_history(
                    writer, alpha, first_event, writer_adjoint_view=True) if first else None)
            writer_fd["state_coordinate_system"] = "materialized effective FP32 delta"
            if rank == 0:
                write_json(options.output / "writer_fd" / "COMPLETE.json", writer_fd)
            require(writer_fd.get("status") == "accepted", "native writer full-BPTT finite difference failed")
            direction = analytic = direction_metadata = None
            if first:
                generator = torch.Generator(device="cpu").manual_seed(20261020)
                direction = (torch.randint(0, 2, anchor_state.values.shape,
                                           generator=generator, dtype=torch.int64) * 2 - 1).float().to(device)
                analytic = float((actual_cotangent.double() * direction.double()).sum())
                direction_metadata = {
                    "seed": 20261020, "distribution": "Rademacher",
                    "coordinates": "valid_materialized_memory", "shape": list(direction.shape),
                    "rms": float(direction.double().square().mean().sqrt()),
                    "l2": float(direction.double().norm()), "sha256": tensor_hash(direction),
                    "rows_sha256": tensor_hash(anchor_state.rows),
                }
                if rank == 0:
                    torch.save({"direction": direction.cpu(), "rows": anchor_state.rows.cpu(),
                                "baseline_values": anchor_state.values.detach().cpu()},
                               options.output / "end_to_end_direction.pt")
            directions = [value for value in gathered(direction_metadata) if value is not None]
            derivatives = [value for value in gathered(analytic) if value is not None]
            require(len(directions) == 2 and directions[0] == directions[1]
                    and len(derivatives) == 2 and derivatives[0] == derivatives[1],
                    "registered E2E direction/derivative differs across TP replicas")
            pairs = []
            for step in (2.0 ** -14, 2.0 ** -13):
                losses = []
                for sign in (1, -1):
                    replay_rng(anchor_rng)
                    perturbed = anchor_state.with_values(
                        anchor_state.values.detach() + sign * step * direction) if first else None
                    row = query(first_event, writer, alpha, fd_arm[2],
                                tag=f"diagnostic:e2e:{step}:{sign}", fixed_state=perturbed)
                    losses.append(row["normalized_loss"])
                pairs.append({"step": step, "plus": losses[0], "minus": losses[1]})
            replay_rng(anchor_rng)
            repeat = query(first_event, writer, alpha, fd_arm[2],
                           tag="diagnostic:e2e:exact-baseline-repeat",
                           fixed_state=anchor_state, capture_logits=True)
            exact_repeat = (repeat["loss_sum"] == matched["loss_sum"]
                            and repeat["logits_sha256_by_rank"] == matched["logits_sha256_by_rank"])
            end_to_end = evaluate_end_to_end(derivatives[0], pairs, exact_repeat)
            end_to_end.update(direction=directions[0], baseline=matched, baseline_repeat=repeat,
                              runtime_identity_sha256=runtime_hashes[0],
                              scope="frozen-reader BF16 surrogate; this does not establish smoothness")
            visibility_by_rank = [value for value in gathered(visibility if first else None)
                                  if value is not None]
            if rank == 0:
                write_json(options.output / "end_to_end_directional.json", end_to_end)
                write_json(options.output / "quantization_visibility.json", {
                    "reports": [row for values in visibility_by_rank for row in values],
                    "extra_CE_targets": 0, "runtime_identity_sha256": runtime_hashes[0]})
            require(end_to_end["status"] == "accepted",
                    "registered end-to-end directional check failed or was inconclusive")
            del writer, alpha, anchor_state, actual_cotangent, direction
            session.assert_frozen()
            if rank == 0:
                final = copy.deepcopy(registration)
                final.update(status="PREPARATION_COMPLETE_MAIN_TRAINING_PENDING", execution_allowed=False,
                             optimizer_execution_allowed=False, diagnostic_execution_allowed=True)
                final["required_evidence"]["initial_perturbation_receipt_sha256"] = digest(
                    options.output / "initial_perturbation.json")
                final["required_evidence"].update({
                    "runtime_identity_sha256": runtime_hashes[0],
                    "H_T0_raw_target_observation_sha256": digest(options.output / "fresh_observation.json"),
                    "scale_calibration_receipt_sha256": digest(options.output / "fresh_scale_calibration.json"),
                    "writer_fixed_cotangent_gradient_receipt_sha256": digest(
                        options.output / "writer_fd" / "COMPLETE.json"),
                    "end_to_end_directional_receipt_sha256": digest(options.output / "end_to_end_directional.json"),
                    "actual_training_forward_comparison_receipt_sha256": digest(
                        options.output / "actual_training_forward_comparison.json"),
                    "quantization_visibility_receipt_sha256": digest(
                        options.output / "quantization_visibility.json")})
                final["scale_calibration"]["target_scales_by_mode"] = scales
                final["scale_calibration"]["status"] = "FROZEN_FOR_SELECTED_RUNTIME"
                final["freeze"].update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                                       frozen_by="conditionally_authorized_fixed_protocol_acceptance",
                                       final_run_registration_sha256=None)
                final["freeze"]["candidate_file_sha256"] = digest(marker)
                write_json(options.output / "final_registration.json", final)
                write_json(options.output / "final_registration_receipt.json", {
                    "final_registration": file_info(options.output / "final_registration.json"),
                    "self_hash_convention": "file SHA is external; internal self-hash remains null"})
            dist.barrier()
            if options.prepare_only:
                require(ledger.optimizer_targets == 0 and ledger.targets == planned_targets,
                        "actual preparation consumption differs from the registered 3328 targets")
                if rank == 0:
                    write_json(options.output / "PREPARATION_COMPLETE.json", {
                        "status": "preparation_complete", "protocol": "qwen38_main_training_preparation_v1",
                        "runtime_identity_sha256": runtime_hashes[0],
                        "final_registration": file_info(options.output / "final_registration.json"),
                        "budget": ledger.report(), "optimizer_steps": 0,
                        "separate_eight_step_stage_executed": False,
                        "writer_fd": writer_fd, "end_to_end_directional": end_to_end,
                        "actual_training_forward_comparison": comparison,
                        "fresh_scale_calibration": calibration,
                        "reader_unchanged": True, "seconds": time.monotonic() - started,
                        "next_stage": "separately registered main training"})
                dist.barrier()
                return

            # Probe evidence is observed only after calibration/acceptance is frozen.
            for key in probe_ids:
                parts = []
                for frame in samples[key]["evidence"]:
                    ledger.reserve("probe-evidence:" + frame["frame_sha256"],
                                   targets=0, inputs=len(frame["input_ids"]))
                    block = collect_native_evidence(session, frame)
                    if first:
                        parts.append(block)
                    if rank == 0:
                        directory = options.output / "probe_evidence"
                        directory.mkdir(exist_ok=True)
                        path = directory / f'{frame["frame_sha256"]}.pt'
                        torch.save(block, path)
                        write_json(path.with_suffix(".json"), file_info(path))
                if first:
                    history[key] = parts
            empty_probe = {key: query(key, None, None, None,
                                     tag="probe-empty:" + key, empty=True)["loss_sum"] for key in probe_ids}
            if rank == 0:
                write_json(options.output / "probe_empty_baseline.json", empty_probe)

            for arm in ARMS:
                writer, alpha = new_writer(arm)
                stability = BoundedStability()
                optimizer = None
                if first:
                    optimizer = torch.optim.AdamW(
                        writer.parameters(), lr=3e-4, betas=(.9, .999), eps=1e-8,
                        weight_decay=0., amsgrad=False, maximize=False, foreach=False,
                        fused=False, capturable=False, differentiable=False)
                for step in range(9):
                    if step > 0:
                        if first:
                            optimizer.zero_grad(set_to_none=True)
                        step_losses = []
                        step_started = time.monotonic()
                        for key in train_ids:
                            result = query(key, writer, alpha, arm[2],
                                           tag=f"train:{arm[0]}:{step}:" + key, backward=True,
                                           denominator=1536, optimizer_targets=True, diagnostics=True)
                            step_losses.append(result)
                        diagnostics = None
                        if first:
                            params = list(writer.parameters())
                            nonzero = sum(int(value.grad is not None and bool(value.grad.ne(0).any()))
                                          for value in params)
                            finite = all(value.grad is None or bool(value.grad.isfinite().all())
                                         for value in params)
                            norm = math.sqrt(sum(float(value.grad.detach().double().square().sum())
                                                 for value in params if value.grad is not None))
                            diagnostics = {"pre_clip_gradient_l2": float(norm), "clipped": float(norm) > 1.,
                                           "parameters_with_nonzero_gradient": nonzero,
                                           "parameter_tensor_count": len(params),
                                           "nonfinite_count": 0 if finite else 1}
                        replica_diagnostics = [value for value in gathered(diagnostics) if value is not None]
                        require(len(replica_diagnostics) == 2
                                and replica_diagnostics[0] == replica_diagnostics[1],
                                "writer gradient statistics differ across TP")
                        gradient = replica_diagnostics[0]
                        require(gradient["nonfinite_count"] == 0
                                and math.isfinite(gradient["pre_clip_gradient_l2"]),
                                "nonfinite writer gradient")
                        raw_norm = gradient["pre_clip_gradient_l2"]
                        reasons = stability.observe_step(
                            raw_norm, sum(row["beta_saturated_events"] for row in step_losses),
                            sum(row["beta_valid_events"] for row in step_losses))
                        if rank == 0:
                            write_json(options.output / f"stability-{arm[0]}-{step:02d}.json",
                                       stability.last_observation)
                        require(not reasons, "registered stability stop: " + "; ".join(reasons))
                        if first:
                            torch.nn.utils.clip_grad_norm_(
                                params, max_norm=1., norm_type=2., error_if_nonfinite=True, foreach=False)
                            optimizer.step()
                        completed_optimizer_updates += 1
                        stability.complete_step(raw_norm)
                        collectively(not first or all(bool(value.isfinite().all())
                                     for value in writer.parameters()), "nonfinite writer state after update")
                        session.assert_frozen()
                        changes = [value for value in gathered(
                            parameter_change_norm(writer, initial) if first else None) if value is not None]
                        gradient["parameter_change_l2"] = changes[0]
                        signature = check_replicas(writer, f"{arm[0]}:step{step}")
                        row = {"arm": arm[0], "step": step, "loss_sum": sum(r["loss_sum"] for r in step_losses),
                               "loss_target_tokens": 1536, "writer_sha256": signature,
                               "gradient": gradient, "stability": stability.last_observation,
                               "samples": step_losses,
                               "seconds": time.monotonic() - step_started}
                        row["mean_loss"] = row["loss_sum"] / 1536
                        training_rows.append(row)
                        if rank == 0:
                            write_json(options.output / f"train-{arm[0]}-{step:02d}.json", row)
                            print(json.dumps({"event": "writer_step_complete", "arm": arm[0],
                                              "step": step, "loss": row["mean_loss"],
                                              "targets_charged": ledger.report()["normshort_target_accesses"]}),
                                  flush=True)
                    if step in (0, 4, 8):
                        current_probe = []
                        for key in probe_ids:
                            row = query(key, writer, alpha, arm[2],
                                        tag=f"probe:{arm[0]}:{step}:" + key, diagnostics=True)
                            row.update(arm=arm[0], step=step, empty_loss_sum=empty_probe[key],
                                       delta_empty=(empty_probe[key] - row["loss_sum"]) / 128)
                            probe_rows.append(row)
                            current_probe.append(row)
                        save_checkpoint(writer, optimizer, arm, step)
                        if rank == 0:
                            write_json(options.output / f"probe-{arm[0]}-{step:02d}.json",
                                       [row for row in probe_rows if row["arm"] == arm[0] and row["step"] == step])
                        mean_probe = sum(row["loss_sum"] for row in current_probe) / 1536
                        reasons = stability.observe_probe(step, mean_probe)
                        if rank == 0:
                            write_json(options.output / f"probe-stability-{arm[0]}-{step:02d}.json", {
                                "mean_nll": mean_probe, "step_zero_mean_nll": stability.initial_probe_nll,
                                "stop_reasons": reasons, "automatic_extension": False})
                        require(not reasons, "registered probe stop: " + "; ".join(reasons))
                del writer, alpha, optimizer
                session.assert_frozen()
            require(ledger.optimizer_targets == 24576
                    and ledger.targets == 37632 + 1024,
                    "actual dispatch counts differ from complete fixed protocol")
            if rank == 0:
                from delta_ngram.qwen38_probe_analysis import analyze_probe_rows
                analysis = analyze_probe_rows(probe_rows)
                write_json(options.output / "analysis.json", analysis)
                write_json(options.output / "COMPLETE.json", {
                    "status": "complete", "protocol": "qwen38_native_bounded_training_v3",
                    "identity": session.identity, "versions": session.versions,
                    "final_registration": file_info(options.output / "final_registration.json"),
                    "budget": ledger.report(), "optimizer_steps_per_arm": 8,
                    "initial_perturbation": initial_receipt, "writer_fd": writer_fd,
                    "end_to_end_directional": end_to_end,
                    "actual_training_forward_comparison": comparison,
                    "runtime_identity_sha256": runtime_hashes[0],
                    "training": training_rows, "probe": probe_rows, "analysis": analysis,
                    "writer_checkpoints": checkpoints, "seconds": time.monotonic() - started,
                    "reader_unchanged": True, "native_T0_unchanged": True,
                    "automatic_pilot_or_extension": False,
                    "final_norm_mode": "rmsnorm", "norm_epsilon": 1e-6,
                    "norm_choice_source": "explicit user decision", "automatic_norm_choice": False})
            dist.barrier()
        except BaseException as error:
            if rank == 0:
                write_json(options.output / "FAILURE.json", {
                    "status": "stopped", "type": type(error).__name__, "message": str(error),
                    "budget": ledger.report(), "optimizer_updates_completed": completed_optimizer_updates,
                    "automatic_retry": False})
            raise
        finally:
            runtime_scope.close()
    run_frozen_session(options, run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registration", "cache-root", "delta-repo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("model-path", "verification", "observation-root",
                 "tp-gradient-receipt", "tp-session-receipt",
                 "ple-reference-receipt", "parent-budget-ledger"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--prepare-frames", action="store_true")
    parser.add_argument("--validate-preparation", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="run 3328-target preparation and exit before any optimizer/probe stage")
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    require(sum((options.execute, options.prepare_frames, options.validate_preparation)) <= 1,
            "choose prepare-frames, validate-preparation or execute")
    require(not (options.execute or options.validate_preparation) or options.prepare_only,
            "current execution requires --prepare-only; the separate eight-step stage is canceled")
    sys.path.insert(0, str(options.delta_repo / "src"))
    registration = json.loads(options.registration.read_bytes())
    frames = prepare_frames(options, registration)
    codes = code_manifest(options.delta_repo)
    if not (options.execute or options.validate_preparation):
        result = {"frame_manifest_sha256": canonical_hash(frames),
                  "runner_code_manifest_sha256": canonical_hash(codes),
                  "samples": len(frames["samples"]), "optimizer_steps": 0,
                  "loss_target_tokens": 0}
        if options.prepare_frames:
            options.output.mkdir(parents=True, exist_ok=False)
            write_json(options.output / "frames.json", frames)
            write_json(options.output / "code_manifest.json", codes)
            write_json(options.output / "COMPLETE.json", result)
        print(json.dumps(result, indent=2))
        return
    require(all(getattr(options, name) is not None for name in (
        "model_path", "verification", "observation_root", "tp_gradient_receipt",
        "ple_reference_receipt", "parent_budget_ledger")),
        "execution needs verified model, immutable initial parameters, TP/PLE receipts and prior ledger")
    observation, initial = verify_initial_parameters(options, frames)
    tp_report = json.loads(options.tp_gradient_receipt.read_bytes())
    ple_report = json.loads(options.ple_reference_receipt.read_bytes())
    policy = validate_candidate(options, registration, frames, codes, observation, tp_report, ple_report)
    if options.validate_preparation:
        print(json.dumps({
            "status": "preparation_candidate_validated", "optimizer_steps": 0,
            "loss_target_tokens": 0, "registration": file_info(options.registration),
            "frame_manifest_sha256": canonical_hash(frames),
            "runner_code_manifest_sha256": canonical_hash(codes),
            "preparation_contract": preparation_contract(),
            "planned_initial_observation_targets": 2304, "planned_diagnostic_targets": 1024,
            "planned_total_targets": 3328, "planned_writer_optimizer_targets": 0,
        }, indent=2))
        return
    execute(options, registration, frames, codes, initial, policy, observation)


if __name__ == "__main__":
    main()

"""One EP4 group isolating writer scheduling, query pooling, and evidence B12.

A executes the original B3 waves and keeps H/query references only in CPU RAM.
B reuses A-H, builds all writer histories, then replays original B3 query waves.
C reuses A-H and pools query times at Q4. Each branch starts from the same
writer/AdamW/RNG and performs one real update. H12 is a no-CE comparison only.
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
from qwen38_native_ep4_trial import assign_trajectories, optimizer_hash, prepare as prepare_source
from qwen38_native_staged_trial import clone_cpu, compare_results, global_frame_batches
from qwen38_native_train_batched import restore_checkpoint_state, restore_rng_state, validate_frame_plan


class TensorComparison:
    """Small additive statistics; full reference tensors never enter JSON."""

    def __init__(self):
        self.parts = 0
        self.elements = 0
        self.reference_squared = self.candidate_squared = self.error_squared = self.dot = 0.
        self.max_abs = 0.
        self.bitwise = self.close = True

    def add(self, reference, candidate):
        import torch
        reference, candidate = reference.detach().cpu(), candidate.detach().cpu()
        require(reference.shape == candidate.shape and bool(reference.isfinite().all())
                and bool(candidate.isfinite().all()), "comparison requires finite matching tensors")
        self.bitwise &= torch.equal(reference, candidate)
        self.close &= torch.allclose(reference, candidate, rtol=1e-6, atol=1e-7)
        a, b = reference.double().reshape(-1), candidate.double().reshape(-1)
        error = b - a
        self.parts += 1
        self.elements += a.numel()
        self.reference_squared += float(a.square().sum())
        self.candidate_squared += float(b.square().sum())
        self.error_squared += float(error.square().sum())
        self.dot += float(torch.dot(a, b))
        self.max_abs = max(self.max_abs, float(error.abs().max()) if error.numel() else 0.)

    def state(self):
        return dict(vars(self))

    @classmethod
    def merge(cls, states):
        result = cls()
        for state in states:
            for name in ("parts", "elements", "reference_squared", "candidate_squared", "error_squared", "dot"):
                setattr(result, name, getattr(result, name) + state[name])
            result.max_abs = max(result.max_abs, state["max_abs"])
            result.bitwise &= state["bitwise"]
            result.close &= state["close"]
        return result.report()

    def report(self):
        a, b, e = map(math.sqrt, (self.reference_squared, self.candidate_squared, self.error_squared))
        cosine = self.dot / (a * b) if a and b else (1. if a == b == 0 else 0.)
        return {"parts": self.parts, "elements": self.elements, "reference_l2": a,
                "candidate_l2": b, "difference_l2": e,
                "relative_l2": e / a if a else (0. if e == 0 else None),
                "cosine": max(-1., min(1., cosine)), "max_abs": self.max_abs,
                "bitwise_equal": self.bitwise, "allclose_rtol1e-6_atol1e-7": self.close}


def prepare(options):
    # Reuse only the source validator; its fixed three-group execution is not called.
    validation_options = copy.copy(options)
    validation_options.max_groups = 3
    source, prior, manifest = prepare_source(validation_options)
    require(source["optimizer_step"] == 47 and source["completed_loss_targets"] == 1107675,
            "this diagnostic is registered for the frozen step47 source")
    require(prior["total_optimizer_target_accesses"] == 1368724,
            "carry the complete staged comparison ledger without refunds")
    for path in (Path(__file__).resolve(),
                 Path(__file__).with_name("qwen38_native_staged_trial.py"),
                 Path(__file__).with_name("qwen38_native_staged_schedule.py"),
                 options.delta_repo / "src/delta_ngram/qwen38_staged_batching.py"):
        manifest["code"][path.relative_to(options.delta_repo).as_posix()] = file_info(path)
    manifest.update({
        "protocol": "qwen38_native_staged_single_group_diagnostic_v1",
        "groups": 1, "sampling": "next_one_whole_group_from_frozen_step47_sampler",
        "expected_group_targets": 33397, "expected_new_optimizer_targets": 100191,
        "branches": {"A": "original_interleaved_B3_capture_H_delta_cotangent",
                     "B": "same_A_H_all_writer_histories_then_original_wave_B3_queries",
                     "C": "same_A_H_all_writer_histories_then_cross_time_Q4_queries"},
        "evidence_comparison": "fresh_H12_with_head_skip_against_A_H; no_CE_no_writer_update",
        "head_skip_control": "first_A_evidence_wave_repeat_same_B3_with_head_skip; full_H_and_rows; zero_CE",
        "branch_initialization": "identical_writer_AdamW_all_rank_RNG",
        "loss_comparison": "absolute_mean_NLL<=0.001 OR relative_mean_NLL<=0.001",
        "gradient_comparison": "global_preclip_relative_L2<=0.02 AND cosine>=0.999",
        "snapshot_comparison": "rows_counts_identity_exact; values_report_bitwise_and_rtol1e-6_atol1e-7",
        "interpretation": "B_must_pass_before_C_can_support_query_pooling_equivalence",
        "references_saved": False, "model_state_saved": False,
        "performance_measurement": "diagnostic_capture_overhead_included; not_a_speed_benchmark",
    })
    return source, prior, manifest


def execute(options, source, prior, manifest):
    from qwen38_native_ep4_session import run_ep4_session
    rank = int(os.environ.get("RANK", "-1"))
    require(0 <= rank < 4 and int(os.environ.get("WORLD_SIZE", "0")) == 4, "use four torchrun ranks")
    initial_budget = {
        "protocol": "qwen38_native_performance_optimizer_ledger_v1",
        "prior_optimizer_target_accesses": prior["total_optimizer_target_accesses"],
        "prior_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"],
        "trial_optimizer_targets_dispatched": 0, "trial_optimizer_targets_completed": 0,
        "total_optimizer_target_accesses": prior["total_optimizer_target_accesses"],
        "total_diagnostic_target_accesses": prior["total_diagnostic_target_accesses"],
        "trial_diagnostic_target_accesses": 0,
        "failed_or_partial_dispatches_are_not_refunded": True,
        "formal_model_progress_unchanged": True,
    }
    marker = options.output / "input_manifest.json"
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        write_json(marker, manifest)
        write_json(options.output / "budget_ledger.json", initial_budget)
    else:
        deadline = time.monotonic() + 300
        while not marker.is_file():
            require(time.monotonic() < deadline, "rank0 manifest creation timed out")
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
            collate_ep4_frames, pack_ep4_query_tickets, SingleRankGradientPolicy, sum_ep4_writer_gradients)
        from delta_ngram.qwen38_staged_batching import (
            collate_staged_frames, pack_staged_query_tickets, materialize_staged_snapshot,
            release_staged_reader_leaf)
        from qwen38_native_bounded_checks import BoundedStability
        from qwen38_native_dispatch import native_hc_dispatch
        from qwen38_native_router import native_router_dispatch
        from qwen38_native_ep4_session import runtime_identity
        import qwen38_native_ep4_schedule as reference_schedule
        import qwen38_native_staged_schedule as staged_schedule

        device, started = session.device, time.monotonic()
        dispatched = completed = 0
        branch_dispatches = {branch: 0 for branch in "ABC"}
        results, branch_reports = {}, {}
        evidence, query_reference, head_probe = {}, {}, {}
        shapes = []

        def gather(value):
            copies = [None] * 4
            dist.all_gather_object(copies, value)
            return copies

        def agree(value, message):
            copies = gather(value)
            require(all(item == copies[0] for item in copies), message)
            return copies[0]

        def budget():
            return {**initial_budget, "trial_optimizer_targets_dispatched": dispatched,
                    "trial_optimizer_targets_completed": completed,
                    "total_optimizer_target_accesses": prior["total_optimizer_target_accesses"] + dispatched,
                    "optimizer_dispatches_by_branch": dict(branch_dispatches)}

        def save_budget():
            if rank == 0:
                write_json(options.output / "budget_ledger.json", budget())

        def reserve(branch, targets):
            nonlocal dispatched
            require(prior["total_optimizer_target_accesses"] + dispatched + targets <= 100000000,
                    "diagnostic exceeds shared optimizer target budget")
            dispatched += targets
            branch_dispatches[branch] += targets
            save_budget()

        def rng_state():
            return {"rank": rank, "python": random.getstate(), "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state(device).cpu(),
                    "tensor_parallel": clone_cpu(get_cuda_rng_tracker().get_states())}

        def timed(label, function):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            begin = time.monotonic()
            value = function()
            torch.cuda.synchronize(device)
            copies = gather({"rank": rank, "seconds": time.monotonic() - begin,
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)})
            return value, {"phase": label, "seconds_max_rank": max(item["seconds"] for item in copies),
                           "peak_allocated_bytes": max(item["peak_allocated_bytes"] for item in copies),
                           "ranks": copies}

        with ExitStack() as scope:
            try:
                hc = scope.enter_context(native_hc_dispatch(unwrap=True))
                router = scope.enter_context(native_router_dispatch(session.raw))
                precision = scope.enter_context(Qwen38SplitPLEPrecision(session.adapter))
                config = session.models[0].config
                deterministic_fields = ("moe_input_jitter_eps", "moe_expert_capacity_factor",
                                        "attention_dropout", "hidden_dropout")
                require(all(hasattr(config, name) for name in deterministic_fields),
                        "native configuration is missing a required randomness/capacity field")
                deterministic_config = {name: getattr(config, name) for name in deterministic_fields}
                require(deterministic_config == {
                    "moe_input_jitter_eps": None, "moe_expert_capacity_factor": None,
                    "attention_dropout": 0., "hidden_dropout": 0.},
                    "diagnostic requires no router jitter/capacity truncation or dropout")
                runtime = {"native": runtime_identity(session), "HC": hc.identity, "router": router["identity"],
                           "precision": precision.identity(), "code_sha256": canonical_hash(manifest["code"]),
                           "diagnostic": manifest["protocol"], "A_B_query_batch": 3, "C_query_batch_max": 4,
                           "H_comparison_batch_max": 12, "randomness_and_capacity": deterministic_config}
                runtime_hash = agree(canonical_hash(runtime), "runtime differs across ranks")
                policy = SingleRankGradientPolicy(
                    runtime_hash, tensor_parallel_world_size=ps.get_tensor_model_parallel_world_size())
                if rank == 0:
                    write_json(options.output / "runtime_identity.json", runtime)
                data = NativeFullTrainingData(
                    options.cache_root, expected_complete_sha256=manifest["cache_complete"]["sha256"],
                    expected_manifest_sha256=manifest["cache_manifest"]["sha256"])
                require(data.identity == source["data_identity"], "source corpus identity changed")
                sampler = data.sampler(seed=17)
                writer, alpha = make_qwen38_writer(alpha=1.)
                writer, alpha = writer.to(device).float().train(), alpha.to(device)
                optimizer = torch.optim.AdamW(writer.parameters(), lr=.0003, betas=(.9, .999),
                                              eps=1e-8, weight_decay=0., foreach=False, fused=False)
                stability = BoundedStability(hard_stop_gradient_spikes=False, hard_stop_beta_saturation=False)
                source_step, source_targets = restore_checkpoint_state(
                    source, writer=writer, optimizer=optimizer, sampler=sampler, stability=stability)
                restore_rng_state(next(item for item in source["rng_by_rank"] if item["rank"] == rank),
                                  rank=rank, device=device, tensor_parallel_tracker=get_cuda_rng_tracker())
                agree(tensor_state_hash(writer.state_dict()), "restored writer differs across ranks")
                agree(optimizer_hash(optimizer), "restored AdamW differs across ranks")
                group = sampler.next_batch(12)
                require(len(group) == 12, "source sampler needs 12 whole trajectories")
                plans = [data.frames(item) for item in group]
                for item, plan in zip(group, plans):
                    validate_frame_plan(plan, item.loss_tokens)
                group_targets = sum(item.loss_tokens for item in group)
                require(group_targets == 33397, "selected group differs from registered step48 group")
                require(prior["total_optimizer_target_accesses"] + 3 * group_targets <= 100000000,
                        "complete diagnostic exceeds shared budget")
                identity = [{"id": item.document_id, "group_id": item.group_id, "source": item.source,
                             "blocks": item.blocks, "start": item.start, "loss_tokens": item.loss_tokens}
                            for item in group]
                assignments = assign_trajectories(plans)
                agree(canonical_hash({"group": identity, "assignment": assignments}), "group differs across ranks")
                common = {"writer": clone_cpu(writer.state_dict()),
                          "optimizer": clone_cpu(optimizer.state_dict()), "rng": rng_state()}

                def new_local():
                    return [{"index": index, "plan": plans[index],
                             "state": LazyDecayState.from_batch(session.layout.empty_state(1, device)),
                             "bridge": NativeTrajectoryGradients(
                                 total_loss_tokens=group_targets, expected_loss_tokens=group[index].loss_tokens,
                                 policy=policy, runtime_identity_sha256=runtime_hash),
                             "loss_sum": 0., "loss_targets": 0} for index in assignments[rank]]

                def write(item, block):
                    hidden = writer_evidence_features(block["H_new"].to(device), mode="rmsnorm", eps=1e-6)
                    item["state"] = write_native_block(
                        writer, item["state"], hidden, block["physical_rows_new"].to(device),
                        torch.ones(hidden.shape[:2], dtype=torch.bool, device=device), alpha)
                    require(bool(item["state"].values.isfinite().all()), "nonfinite writer state")

                def query_call(branch, tasks, *, pooled, comparisons):
                    frames = [task["frame"] if task is not None else None for task in tasks]
                    collate = collate_staged_frames if pooled else collate_ep4_frames
                    mb = collate(frames, kind="query", device=device)
                    reserve(branch, sum(gather(mb["loss_target_tokens"])))
                    shapes.append({"branch": branch, "kind": "query", "batch": len(tasks)})
                    tickets = [task["owner"]["bridge"].prepare_query(
                        materialize_staged_snapshot(task["snapshot"])) if task is not None else None
                               for task in tasks]
                    pack = pack_staged_query_tickets if pooled else pack_ep4_query_tickets
                    packed = pack(tickets, row_capacity=session.layout.row_capacity, device=device)
                    mb["reader_state"] = packed.reader_state
                    schedule = staged_schedule if pooled else reference_schedule
                    result = schedule.schedule_query_microbatches(
                        session, [mb], group_total_target_tokens=group_targets)["microbatches"][0]
                    require(tuple(result["loss_target_tokens_by_lane"]) == mb["loss_target_tokens_by_lane"],
                            "query target counts differ")
                    packed.scatter_cotangent()
                    for lane, (task, ticket) in enumerate(zip(tasks, tickets)):
                        if task is None:
                            continue
                        item, key = task["owner"], task["key"]
                        item["bridge"].record_query(
                            ticket, loss_sum=result["loss_sum_by_lane"][lane],
                            loss_tokens=task["frame"]["loss_target_tokens"], reader_loss_divisor=group_targets)
                        item["loss_sum"] += result["loss_sum_by_lane"][lane]
                        item["loss_targets"] += task["frame"]["loss_target_tokens"]
                        values = {
                            "rows": ticket.reader_state.rows.detach().cpu().clone(),
                            "counts": ticket.reader_state.counts.detach().cpu().clone(),
                            "delta": ticket._source_values.detach().cpu().clone(),
                            "cotangent": (ticket._cotangent.detach().cpu().clone()
                                          if ticket._cotangent is not None else None),
                            "frame_sha256": canonical_hash(task["frame"]),
                            "loss_sum": result["loss_sum_by_lane"][lane],
                        }
                        if branch == "A":
                            require(key not in query_reference, "duplicate reference query")
                            query_reference[key] = values
                        else:
                            ref = query_reference[key]
                            mapping_equal = (values["frame_sha256"] == ref["frame_sha256"]
                                             and torch.equal(values["rows"], ref["rows"])
                                             and torch.equal(values["counts"], ref["counts"])
                                             and (values["cotangent"] is None) == (ref["cotangent"] is None))
                            require(mapping_equal, f"{branch}: query mapping/rows/counts differ at {key}")
                            comparisons["delta"].add(ref["delta"], values["delta"])
                            if ref["cotangent"] is not None:
                                comparisons["cotangent"].add(ref["cotangent"], values["cotangent"])
                            single_delta, single_cotangent = TensorComparison(), TensorComparison()
                            single_delta.add(ref["delta"], values["delta"])
                            if ref["cotangent"] is not None:
                                single_cotangent.add(ref["cotangent"], values["cotangent"])
                            comparisons["queries"].append({
                                "trajectory_index": key[0], "ordinal": key[1], "mapping_exact": True,
                                "delta": single_delta.report(), "cotangent": single_cotangent.report(),
                                "loss_sum_absolute_difference": abs(values["loss_sum"] - ref["loss_sum"])})
                        # B/C mirror the staged lifecycle; A retains the original lifecycle.
                        if branch != "A":
                            release_staged_reader_leaf(ticket)
                        task["snapshot"] = None

                def reference_forward(local):
                    for ordinal in range(max(map(len, plans))):
                        for kind in ("query", "evidence"):
                            global_frames = [plan[ordinal][kind] for plan in plans if ordinal < len(plan)]
                            if not any(frame is not None for frame in global_frames):
                                continue
                            frames = [item["plan"][ordinal][kind] if ordinal < len(item["plan"]) else None
                                      for item in local]
                            if kind == "query":
                                tasks = [{"owner": item, "key": (item["index"], ordinal),
                                          "frame": frame, "snapshot": item["state"]} if frame is not None else None
                                         for item, frame in zip(local, frames)]
                                query_call("A", tasks, pooled=False, comparisons=None)
                            else:
                                mb = collate_ep4_frames(frames, kind="evidence", device=device)
                                shapes.append({"branch": "A", "kind": "evidence", "batch": 3})
                                result = reference_schedule.collect_evidence_microbatches(
                                    session, [mb])["microbatches"][0]
                                blocks = crop_evidence_lanes(mb, result["hidden"], result["physical_rows"])
                                if not head_probe:
                                    control = staged_schedule.collect_evidence_microbatches(
                                        session, [mb])["microbatches"][0]
                                    addresses_equal = torch.equal(result["physical_rows"], control["physical_rows"])
                                    require(all(gather(addresses_equal)),
                                            "same-B head skip changed native physical addresses")
                                    comparison = TensorComparison()
                                    comparison.add(result["hidden"], control["hidden"])
                                    head_probe.update({
                                        "ordinal": ordinal, "local_batch": 3, "CE_targets": 0,
                                        "physical_rows_exact": True,
                                        "full_native_H": TensorComparison.merge(gather(comparison.state())),
                                        "only_change": "skip_unused_LM_head_after_capture"})
                                    del control, comparison
                                for item, block in zip(local, blocks):
                                    if block is not None:
                                        key = (item["index"], ordinal)
                                        require(key not in evidence, "duplicate reference H")
                                        evidence[key] = block
                                        write(item, block)
                        for item in local:
                            if ordinal + 1 == len(item["plan"]):
                                item["bridge"].backward_writer()
                                item["bridge"] = item["state"] = None

                def delayed_forward(branch, local, comparisons):
                    tasks, by_key = [], {}
                    for item in local:
                        for ordinal, frame in enumerate(item["plan"]):
                            key = (item["index"], ordinal)
                            if frame["query"] is not None:
                                task = {"owner": item, "key": key, "frame": frame["query"],
                                        "snapshot": item["state"]}
                                tasks.append(task)
                                by_key[key] = task
                            if frame["evidence"] is not None:
                                write(item, evidence[key])
                        item["state"] = None
                    if branch == "B":
                        for ordinal in range(max(map(len, plans))):
                            if not any(ordinal < len(plan) and plan[ordinal]["query"] is not None for plan in plans):
                                continue
                            query_call(branch, [by_key.get((item["index"], ordinal)) for item in local],
                                       pooled=False, comparisons=comparisons)
                    else:
                        for start, batch, _ in global_frame_batches(gather(len(tasks)), 4):
                            active = tasks[start:start + batch]
                            query_call(branch, active + [None] * (batch - len(active)),
                                       pooled=True, comparisons=comparisons)
                    require(all(task["snapshot"] is None for task in tasks), "unconsumed delayed query")

                def finish(branch, local):
                    nonlocal completed
                    if branch != "A":
                        for item in local:
                            item["bridge"].backward_writer()
                            item["bridge"] = item["state"] = None
                    require(all(item["loss_targets"] == group[item["index"]].loss_tokens for item in local),
                            "trajectory target accounting differs")
                    reduction = sum_ep4_writer_gradients(writer, group=dist.group.WORLD)
                    norm = reduction["global_gradient_l2"]
                    require(math.isfinite(norm) and norm > 0, "zero/nonfinite writer gradient")
                    gradients = ({key: value.grad.detach().cpu().clone()
                                  for key, value in writer.named_parameters() if value.grad is not None}
                                 if rank == 0 else {})
                    agree(tensor_state_hash({key: value.grad for key, value in writer.named_parameters()
                                             if value.grad is not None}), "writer gradients differ across ranks")
                    torch.nn.utils.clip_grad_norm_(writer.parameters(), 1., error_if_nonfinite=True, foreach=False)
                    optimizer.step()
                    require(all(bool(value.isfinite().all()) for value in writer.parameters()), "nonfinite update")
                    signature = agree(tensor_state_hash(writer.state_dict()), "updated writer replicas differ")
                    adam_signature = agree(optimizer_hash(optimizer), "updated AdamW replicas differ")
                    session.assert_frozen()
                    loss_sum = math.fsum(gather(math.fsum(item["loss_sum"] for item in local)))
                    completed += group_targets
                    require(completed == dispatched and branch_dispatches[branch] == group_targets,
                            "branch did not cover every registered query target")
                    save_budget()
                    return {"mean_nll": loss_sum / group_targets, "loss_sum": loss_sum,
                            "raw_gradient_l2": norm, "writer_sha256": signature,
                            "optimizer_sha256": adam_signature, "gradients": gradients,
                            "parameters": ({key: value.detach().cpu().clone()
                                            for key, value in writer.named_parameters()} if rank == 0 else {})}

                for branch in "ABC":
                    writer.load_state_dict(common["writer"], strict=True)
                    optimizer.load_state_dict(clone_cpu(common["optimizer"]))
                    restore_rng_state(common["rng"], rank=rank, device=device,
                                      tensor_parallel_tracker=get_cuda_rng_tracker())
                    optimizer.zero_grad(set_to_none=True)
                    agree(tensor_state_hash(writer.state_dict()), "branch initial writer replicas differ")
                    agree(optimizer_hash(optimizer), "branch initial AdamW replicas differ")
                    local = new_local()
                    comparisons = {"delta": TensorComparison(), "cotangent": TensorComparison(), "queries": []}
                    if branch == "A":
                        _, forward_time = timed("A_original_B3_with_capture", lambda: reference_forward(local))
                    else:
                        _, forward_time = timed(
                            branch + "_same_H_delayed_queries", lambda: delayed_forward(branch, local, comparisons))
                    result, update_time = timed(branch + "_VJP_SUM_clip_AdamW", lambda: finish(branch, local))
                    results[branch] = result
                    report = {"branch": branch, "phases": [forward_time, update_time],
                              "result": {key: value for key, value in result.items()
                                         if key not in ("gradients", "parameters")}}
                    if branch != "A":
                        pair = compare_results(results["A"], result, common["writer"]) if rank == 0 else None
                        report["against_A"] = gather(pair)[0]
                        report["effective_delta"] = TensorComparison.merge(gather(comparisons["delta"].state()))
                        report["reader_cotangent"] = TensorComparison.merge(gather(comparisons["cotangent"].state()))
                        report["queries"] = [query for items in gather(comparisons["queries"]) for query in items]
                        report["all_query_mapping_exact"] = all(query["mapping_exact"] for query in report["queries"])
                        report["schedule_isolation_pass"] = (
                            report["against_A"]["within_registered_tolerances"]
                            and report["effective_delta"]["allclose_rtol1e-6_atol1e-7"]
                            and report["all_query_mapping_exact"])
                    branch_reports[branch] = report
                    if rank == 0:
                        write_json(options.output / ("branch_" + branch + ".json"), report)
                        write_json(options.output / "progress.json", {
                            "status": "diagnostic_running", "completed_branch": branch, "budget": budget()})
                        print(json.dumps({"event": "diagnostic_branch_complete", "branch": branch,
                                          "against_A": report.get("against_A"),
                                          "effective_delta": report.get("effective_delta"),
                                          "reader_cotangent": report.get("reader_cotangent")}), flush=True)
                    del local, comparisons
                    dist.barrier()

                def evidence_comparison():
                    hidden_comparison, normalized_comparison = TensorComparison(), TensorComparison()
                    frame_reports = []
                    tasks = [(index, ordinal, frame["evidence"])
                             for index in assignments[rank] for ordinal, frame in enumerate(plans[index])
                             if frame["evidence"] is not None]
                    for start, batch, _ in global_frame_batches(gather(len(tasks)), 12):
                        active = tasks[start:start + batch]
                        frames = [task[2] for task in active] + [None] * (batch - len(active))
                        shapes.append({"branch": "H_only", "kind": "evidence", "batch": batch})
                        mb = collate_staged_frames(frames, kind="evidence", device=device)
                        result = staged_schedule.collect_evidence_microbatches(
                            session, [mb])["microbatches"][0]
                        blocks = crop_evidence_lanes(mb, result["hidden"], result["physical_rows"])
                        for task, block in zip(active, blocks):
                            ref = evidence[task[:2]]
                            require(torch.equal(ref["physical_rows_new"], block["physical_rows_new"]),
                                    "H12 physical addresses differ from reference")
                            hidden_comparison.add(ref["H_new"], block["H_new"])
                            ref_normalized = writer_evidence_features(ref["H_new"], mode="rmsnorm", eps=1e-6)
                            new_normalized = writer_evidence_features(block["H_new"], mode="rmsnorm", eps=1e-6)
                            normalized_comparison.add(ref_normalized, new_normalized)
                            single = TensorComparison()
                            single.add(ref["H_new"], block["H_new"])
                            frame_reports.append({"trajectory_index": task[0], "ordinal": task[1],
                                                  "physical_rows_exact": True, "H": single.report()})
                    return {"raw_H": TensorComparison.merge(gather(hidden_comparison.state())),
                            "RMS_normalized_writer_H": TensorComparison.merge(gather(normalized_comparison.state())),
                            "frames": [row for rows in gather(frame_reports) for row in rows],
                            "CE_targets": 0, "writer_updates": 0,
                            "difference_includes": ["evidence_batch_reorder", "batch_size", "skipped_LM_head"]}

                restore_rng_state(common["rng"], rank=rank, device=device,
                                  tensor_parallel_tracker=get_cuda_rng_tracker())
                h_report, h_timing = timed("H12_only_no_CE_comparison", evidence_comparison)
                session.assert_frozen()
                require(dispatched == completed == 100191, "diagnostic target total differs")
                if rank == 0:
                    write_json(options.output / "COMPLETE.json", {
                        "protocol": manifest["protocol"], "status": "one_group_diagnostic_completed",
                        "source_optimizer_step": source_step, "source_model_targets": source_targets,
                        "group_loss_targets": group_targets, "group_identity_sha256": canonical_hash(identity),
                        "trajectories": identity, "assignment": assignments, "branches": branch_reports,
                        "same_B3_head_skip_control": head_probe,
                        "H12_comparison": h_report, "H12_timing": h_timing,
                        "actual_batch_shapes_rank0": shapes, "budget": budget(),
                        "B_schedule_isolation_pass": branch_reports["B"]["schedule_isolation_pass"],
                        "C_query_pooling_supported": (branch_reports["B"]["schedule_isolation_pass"]
                                                      and branch_reports["C"]["schedule_isolation_pass"]),
                        "training_state_saved": False, "tensor_references_saved": False,
                        "formal_checkpoint_unchanged": True, "reader_unchanged": True,
                        "seconds_including_capture_overhead": time.monotonic() - started})
                dist.barrier()
            except BaseException as error:
                save_budget()
                if rank == 0:
                    write_json(options.output / "FAILURE.json", {
                        "status": "diagnostic_stopped", "error_type": type(error).__name__, "message": str(error),
                        "budget": budget(), "completed_branches": list(branch_reports),
                        "automatic_retry": False, "training_state_saved": False})
                raise

    try:
        run_ep4_session(options, run)
    except BaseException as error:
        failure = options.output / "FAILURE.json"
        if rank == 0 and not failure.exists():
            write_json(failure, {"status": "diagnostic_stopped", "error_type": type(error).__name__,
                                "message": str(error), "automatic_retry": False, "training_state_saved": False,
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
        print(json.dumps({"status": "single_group_diagnostic_inputs_valid", "input_manifest": manifest}, indent=2))
        return
    execute(options, source, prior, manifest)


if __name__ == "__main__":
    main()

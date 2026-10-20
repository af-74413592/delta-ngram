"""Frozen native DEV64 continuation at an alternating-training checkpoint.

The original 64 families, nested interference schedules, five conditions and
three direction seeds remain fixed. Originals are retokenized by the separate
native preparation tool. Every dispatched native CE target is charged through
the caller's evaluation ledger; this module never performs backward or updates
an optimizer, sampler, checkpoint, alpha, or writer parameter.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from qwen38_native_norm_probe import canonical_hash, require


PROTOCOL = "qwen38_native_dev64_continuation_v1"
DISTANCES = (0, 1, 4, 8, 16)
SEEDS = (17, 29, 43)
CONDITIONS = ("self", "empty", "donor", "self_perm", "donor_perm")
PREFIX, TARGET_BLOCK, WRITE_BLOCK = 128, 384, 512


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _module_guard(module):
    """Remember tensors, flags and existing gradients without clearing them."""
    tensors = {
        name: (value, value._version, value.detach().cpu().clone())
        for name, value in module.state_dict(keep_vars=True).items()
    }
    parameters = {
        name: (value, value.requires_grad, value.grad,
               None if value.grad is None else value.grad._version)
        for name, value in module.named_parameters()
    }
    modes = [(child, child.training) for child in module.modules()]
    return tensors, parameters, modes


def _check_module(module, guard, label):
    import torch
    tensors, parameters, _ = guard
    current = module.state_dict(keep_vars=True)
    require(current.keys() == tensors.keys(), f"{label} state keys changed during evaluation")
    for name, (original, version, saved) in tensors.items():
        value = current[name]
        require(value is original and value._version == version
                and torch.equal(value.detach().cpu(), saved),
                f"{label} tensor changed during evaluation: {name}")
    current_parameters = dict(module.named_parameters())
    require(current_parameters.keys() == parameters.keys(),
            f"{label} parameter keys changed during evaluation")
    for name, (original, requires_grad, gradient, gradient_version) in parameters.items():
        value = current_parameters[name]
        require(value is original and value.requires_grad == requires_grad
                and value.grad is gradient
                and (gradient is None or gradient._version == gradient_version),
                f"{label} parameter or existing gradient changed during evaluation: {name}")


def _rng_snapshot(device, tracker):
    import numpy as np
    import torch
    return {
        "python": random.getstate(), "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state(device).clone(),
        "tensor_parallel": copy.deepcopy(tracker.get_states()),
    }


def _restore_rng(saved, device, tracker):
    import numpy as np
    import torch
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch_cpu"])
    torch.cuda.set_rng_state(saved["torch_cuda"], device=device)
    tracker.set_states(copy.deepcopy(saved["tensor_parallel"]))


def _frames(tokens, source_id, *, query):
    """Use the existing native shifted-label and article-prefix convention."""
    from delta_ngram.qwen38_train_data import _frame
    initial, step = (PREFIX, TARGET_BLOCK) if query else (0, WRITE_BLOCK)
    require(len(tokens) > initial, "DEV64 article has no usable native tokens")
    return [
        _frame({"document_id": source_id, "source": "dev64",
                "article_id": source_id, "evaluation_only": True},
               tokens, start, min(start + step, len(tokens)), query=query)
        for start in range(initial, len(tokens), step)
    ]


def _score_summary(losses, frames):
    require(len(losses) == len(frames) > 0, "incomplete DEV64 query windows")
    count = sum(frame["loss_target_tokens"] for frame in frames)
    total = math.fsum(losses)
    require(count > 0 and math.isfinite(total), "invalid DEV64 loss or target count")
    first_count = frames[0]["loss_target_tokens"]
    return {
        "loss_sum": total, "target_tokens": count, "nll": total / count,
        "read_windows": len(frames),
        "reader_input_tokens": sum(sum(frame["encoder_valid_mask"]) for frame in frames),
        "first_target_loss_sum": losses[0], "first_target_tokens": first_count,
        "first_target_nll": losses[0] / first_count,
    }


def evaluate_dev64(session, writer, alpha, *, plan_path, arm, milestone, output,
                   charge_targets, checkpoint_metadata):
    """All four ranks evaluate one frozen checkpoint and return the same report.

    ``charge_targets(global_actual_targets)`` is called on EVERY rank before
    each collective-bearing native CE forward. Its ledger must persist charges
    before returning. Evidence forwards do not execute the LM head or CE.
    ``output`` is a fresh attempt directory, including on resumed evaluation.
    """
    import torch
    import torch.distributed as dist
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    from delta_ngram.alpha_memory import LazyDecayState
    from delta_ngram.batching import BatchMemoryState
    from delta_ngram.direction_control import make_direction_permutation, permute_state_for_read
    from delta_ngram.p2_initialization import writer_tensor_sha256
    from delta_ngram.qwen38_native import write_native_block
    from delta_ngram.qwen38_staged_batching import collate_staged_frames, materialize_staged_snapshot
    from delta_ngram.qwen38_train_batching import crop_evidence_lanes
    from delta_ngram.qwen38_training import writer_evidence_features
    from delta_ngram.recent_news_evaluation import summarize_continuation
    from delta_ngram.state import SparseMemoryState
    from qwen38_native_dev64_prepare import load_native_dev64_cache
    from qwen38_native_ep4_schedule import _runtime
    from qwen38_native_staged_schedule import collect_evidence_microbatches

    _runtime(session)
    require(dist.get_world_size() == 4 and callable(charge_targets),
            "DEV64 evaluation requires EP4 and a durable target-charge callback")
    require(alpha.kind == "fixed" and alpha.initial_alpha in (1., .99),
            "DEV64 evaluation uses the checkpoint's fixed alpha")
    rank, device = dist.get_rank(), session.device
    output, plan_path = Path(output), Path(plan_path)
    tracker = get_cuda_rng_tracker()
    saved_rng = _rng_snapshot(device, tracker)
    writer_guard, alpha_guard = _module_guard(writer), _module_guard(alpha)
    started = time.monotonic()
    charged = 0
    execution = {
        "native_evidence_calls": 0, "native_query_calls": 0,
        "evidence_input_tokens": 0, "query_input_tokens": 0,
        "query_windows": 0, "write_calls": 0, "writer_tokens": 0,
    }

    def gather(value):
        result = [None] * 4
        dist.all_gather_object(result, value)
        return result

    def agree(value, message):
        values = gather(value)
        require(all(item == values[0] for item in values), message)
        return values[0]

    def batches(local_count, limit):
        maximum = max(gather(local_count))
        for start in range(0, maximum, limit):
            yield start, min(limit, maximum - start)

    try:
        prepared = load_native_dev64_cache(plan_path.parent if plan_path.is_file() else plan_path)
        plan, tokens = prepared["plan"], prepared["tokens"]
        plan_identity = agree(prepared["identity"], "ranks loaded different DEV64 prepared inputs")
        require(tuple(plan["distances"]) == DISTANCES
                and tuple(plan["evaluation"]["direction_control_seeds"]) == SEEDS
                and tuple(plan["evaluation"]["conditions"]) == CONDITIONS,
                "DEV64 distances, conditions or direction-control seeds differ")
        pairs = plan["pairs"]
        require(len(pairs) == 64 and len({pair["event_family"] for pair in pairs}) == 64,
                "DEV64 evaluation requires exactly 64 frozen independent families")
        target_total = sum(len(tokens[pair["later_source_id"]]) - PREFIX for pair in pairs)
        expected_cost = 41 * target_total
        require(expected_cost > 0 and plan["evaluation"]["ce_target_accesses"] == expected_cost,
                "DEV64 expected CE accesses differ from native target coverage")
        require(all(tuple(item["unrelated_article_count"] for item in pair["interference_conditions"])
                    == DISTANCES for pair in pairs), "DEV64 pair interference distances differ")
        writer_sha = agree(writer_tensor_sha256(writer.state_dict()),
                           "writer replicas differ before DEV64 evaluation")
        agree({"arm": arm, "milestone": milestone, "alpha": alpha.initial_alpha,
               "checkpoint": checkpoint_metadata}, "ranks disagree on evaluated checkpoint")
        permutations = tuple(make_direction_permutation(session.layout.row_width, seed) for seed in SEEDS)
        manifest = {
            "protocol": PROTOCOL, "arm": arm, "nominal_model_target_milestone": milestone,
            "alpha": alpha.initial_alpha, "checkpoint": checkpoint_metadata,
            "prepared_data_identity": plan_identity,
            "writer_sha256": writer_sha, "independent_families": 64,
            "distances": list(DISTANCES), "conditions": list(CONDITIONS),
            "direction_control": {
                "plans": [item.metadata() for item in permutations],
                "scope": "one shared 160-channel permutation per seed across every family, lag and self/donor",
                "application": "materialize effective lazy values; permute read view only; continue unpermuted writes",
                "aggregation": "arithmetic mean across three fixed seeds within family before paired bootstrap",
            },
            "native_framing": {
                "evidence_new_tokens": WRITE_BLOCK, "evidence_local_prefix_tokens": PREFIX,
                "query_local_prefix_tokens": PREFIX, "query_maximum_target_tokens": TARGET_BLOCK,
                "native_padded_sequence_length": 640,
                "evidence_article_boundaries_reset_encoder_context": True,
                "labels": "native logits aligned to the following token; initial 128 later tokens unscored",
                "adaptation": "original raw UTF-8 retokenized with pinned Qwen3.8 tokenizer; evidence uses native 128-token local prefix",
                "later_article_written": False, "query_changes_memory_or_alpha_clock": False,
            },
            "evaluation_budget": {
                "unique_later_target_tokens": target_total, "target_scoring_multiplicity": 41,
                "expected_ce_target_accesses": expected_cost,
                "account": "separate evaluation_target_accesses; every dispatch precharged; no refunds",
                "empty_scored_once_per_family_and_reused_at_all_distances": True,
            },
            "scope": "fixed continuation benefit and direction controls",
            "raw_text_long_context_control_included": False,
            "optimizer_steps": 0, "backward_calls": 0,
            "query_execution": "grad-enabled native reader with detached FP32 memory leaf; no backward; discard graph immediately",
            "evidence_and_writer_execution": "no_grad",
            "writer_and_alpha_state_unchanged": True, "training_rng_restored_on_exit": True,
            "topology": {"TP": 1, "PP": 1, "EP": 4, "ETP": 1, "DP": 4},
            "query_batch_limit": 4, "evidence_batch_limit": 12,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            require(not (output / "manifest.json").exists(),
                    "DEV64 evaluation output must be a fresh attempt directory")
            _write_json(output / "manifest.json", manifest)
        dist.barrier()
        writer.eval()
        alpha.eval()
        session.assert_frozen()
        records = []

        def score(state, frames):
            nonlocal charged
            losses = []
            # Every frame gets its own local packed namespace, even when four
            # different windows read the same immutable family snapshot.
            for start, size in batches(len(frames), 4):
                active = frames[start:start + size]
                mb = collate_staged_frames(active + [None] * (size - len(active)),
                                           kind="query", device=device)
                empty = session.layout.empty_state(1, device)
                packed = BatchMemoryState.from_states(
                    [state.memory for _ in active] + [empty.memory] * (size - len(active)),
                    session.layout.row_capacity)
                require(not packed.values.requires_grad, "evaluation memory acquired a writer graph")
                if packed.values.numel():
                    packed = packed.with_values(packed.values.detach().requires_grad_(True))
                else:
                    # Same zero sentinel as the staged training packer. This
                    # keeps empty/inactive native forwards on its grad-enabled
                    # dispatch without changing any PLE row value or count.
                    sentinel = SparseMemoryState(
                        torch.tensor([1], dtype=torch.long, device=device),
                        torch.zeros((1, session.layout.row_width), dtype=torch.float32,
                                    device=device, requires_grad=True),
                        torch.zeros(1, dtype=torch.float32, device=device))
                    packed = BatchMemoryState(sentinel, size, session.layout.row_capacity)
                saved_versions = tuple(value._version for value in (packed.rows, packed.values, packed.counts))
                costs = gather((mb["loss_target_tokens"], mb["real_input_tokens"], len(active)))
                cost = sum(item[0] for item in costs)
                require(cost > 0 and charged + cost <= expected_cost,
                        "DEV64 dispatch target coverage exceeds the frozen evaluation plan")
                charge_targets(cost)
                charged += cost
                execution["native_query_calls"] += 1
                execution["query_input_tokens"] += sum(item[1] for item in costs)
                execution["query_windows"] += sum(item[2] for item in costs)
                if rank == 0:
                    _write_json(output / "budget_ledger.json", {
                        "protocol": PROTOCOL, "evaluation_target_accesses": charged,
                        "expected_evaluation_target_accesses": expected_cost,
                        "charge_timing": "before_native_query_dispatch", "refunds": False,
                    })
                # The caller's callback persisted the charge before this
                # original native CE call; no separate loss implementation.
                batch = mb["batch"]
                with torch.enable_grad():
                    native_losses, capture = session.adapter.run_native(
                        lambda: session.models[0](**batch),
                        input_ids=batch["input_ids"], valid_mask=mb["valid_mask"],
                        owner_ids=mb["owner_ids"], mode="query", state=packed)
                labels = batch["labels"]
                require(isinstance(native_losses, torch.Tensor)
                        and native_losses.shape == labels.shape and native_losses.requires_grad,
                        "DEV64 native forward must retain the registered grad-enabled CE dispatch")
                # Reading metrics never invokes backward or joins the detached
                # reader leaf to the writer. Release every native graph now.
                losses_detached = native_losses.detach()
                del native_losses, capture
                mask = labels.ne(-100)
                require(bool(losses_detached[mask].isfinite().all()), "nonfinite native DEV64 target loss")
                lane_sums = torch.where(mask, losses_detached.float(),
                                        torch.zeros_like(losses_detached, dtype=torch.float32)).sum(1)
                counts = mask.sum(1).cpu().tolist()
                require(tuple(counts) == mb["loss_target_tokens_by_lane"],
                        "DEV64 native query target counts differ")
                losses.extend(lane_sums[:len(active)].cpu().tolist())
                require(saved_versions == tuple(
                    value._version for value in (packed.rows, packed.values, packed.counts)),
                    "DEV64 native query mutated its packed memory")
                require(packed.values.grad is None, "DEV64 query unexpectedly performed backward")
                del losses_detached, lane_sums, packed, mb, batch
            return _score_summary(losses, frames)

        with torch.no_grad():
            # Fixed round-robin families keep independent memories across ranks.
            # All ranks still enter every native EP collective with matching B.
            for round_index in range(16):
                pair = pairs[4 * round_index + rank]
                later_id = pair["later_source_id"]
                query_frames = _frames(tokens[later_id], later_id, query=True)
                schedule = pair["interference_conditions"]
                source_ids = list(dict.fromkeys([
                    pair["earlier_source_id"], pair["donor_source_id"],
                    *schedule[-1]["distractor_source_ids"]]))
                require(later_id not in source_ids, "later target article cannot become evidence")
                evidence_frames = {source_id: _frames(tokens[source_id], source_id, query=False)
                                   for source_id in source_ids}
                tasks = [(source_id, index, frame) for source_id in source_ids
                         for index, frame in enumerate(evidence_frames[source_id])]
                evidence = {}
                for start, size in batches(len(tasks), 12):
                    active = tasks[start:start + size]
                    mb = collate_staged_frames([item[2] for item in active]
                                               + [None] * (size - len(active)),
                                               kind="evidence", device=device)
                    execution["evidence_input_tokens"] += sum(gather(mb["real_input_tokens"]))
                    execution["native_evidence_calls"] += 1
                    captured = collect_evidence_microbatches(session, [mb])["microbatches"][0]
                    cropped = crop_evidence_lanes(mb, captured["hidden"], captured["physical_rows"])
                    for task, block in zip(active, cropped):
                        require(block is not None, "DEV64 evidence crop dropped a real source block")
                        evidence[task[:2]] = block
                    del captured, cropped, mb

                empty = score(session.layout.empty_state(1, device), query_frames)
                empty.update(actual_written_articles=0, actual_written_tokens=0,
                             actual_written_blocks=0, address_prefix_tokens=0,
                             evidence_encoder_prefix_tokens=0, write_source_ids=[])
                arms = {distance: {"empty": empty.copy()} for distance in DISTANCES}
                local_write_calls, local_writer_tokens = 0, 0
                for mode, anchor in (("self", pair["earlier_source_id"]),
                                     ("donor", pair["donor_source_id"])):
                    state = LazyDecayState.from_batch(session.layout.empty_state(1, device))
                    written = []
                    totals = {"articles": 0, "tokens": 0, "blocks": 0,
                              "address_prefix_tokens": 0, "evidence_encoder_prefix_tokens": 0}
                    for condition in schedule:
                        distance = condition["unrelated_article_count"]
                        planned = [anchor, *condition["distractor_source_ids"]]
                        require(planned[:len(written)] == written,
                                "DEV64 interference schedule must retain nested evidence prefixes")
                        for source_id in planned[len(written):]:
                            require(source_id != later_id, "later target cannot be written")
                            for index, frame in enumerate(evidence_frames[source_id]):
                                block = evidence[(source_id, index)]
                                hidden = writer_evidence_features(
                                    block["H_new"].to(device), mode="rmsnorm", eps=1e-6)
                                state = write_native_block(
                                    writer, state, hidden, block["physical_rows_new"].to(device),
                                    torch.ones(hidden.shape[:2], dtype=torch.bool, device=device), alpha)
                                require(bool(state.values.isfinite().all()), "nonfinite DEV64 writer memory")
                                totals["tokens"] += frame["new_tokens"]
                                totals["blocks"] += 1
                                totals["address_prefix_tokens"] += min(2, frame["article_token_start"])
                                totals["evidence_encoder_prefix_tokens"] += frame["prefix_tokens"]
                                local_write_calls += 1
                                local_writer_tokens += frame["new_tokens"]
                            totals["articles"] += 1
                            written.append(source_id)
                        for unit in ("articles", "tokens", "blocks"):
                            require(totals[unit] == condition[mode + "_actual_written_" + unit],
                                    "DEV64 native write budget differs from its prepared plan")
                        materialized = materialize_staged_snapshot(state)
                        original = score(materialized, query_frames)
                        original.update({"actual_written_" + unit: totals[unit]
                                         for unit in ("articles", "tokens", "blocks")})
                        original.update(address_prefix_tokens=totals["address_prefix_tokens"],
                                        evidence_encoder_prefix_tokens=totals["evidence_encoder_prefix_tokens"],
                                        write_source_ids=written.copy())
                        arms[distance][mode] = original
                        seed_scores = {}
                        for permutation in permutations:
                            view = permute_state_for_read(materialized, permutation)
                            seed_scores[str(permutation.seed)] = score(view, query_frames)
                            del view
                        averaged = {
                            key: math.fsum(value[key] for value in seed_scores.values()) / len(SEEDS)
                            for key in ("loss_sum", "nll", "first_target_loss_sum", "first_target_nll")
                        }
                        averaged.update({
                            key: original[key] for key in (
                                "target_tokens", "read_windows", "reader_input_tokens", "first_target_tokens",
                                "actual_written_articles", "actual_written_tokens", "actual_written_blocks",
                                "address_prefix_tokens", "evidence_encoder_prefix_tokens", "write_source_ids")
                        })
                        averaged.update(per_seed=seed_scores,
                                        aggregation="arithmetic mean across fixed seeds within family")
                        arms[distance][mode + "_perm"] = averaged
                        del materialized
                    del state

                local_records = []
                annotation = pair.get("native_prefix_annotation", {})
                for distance in DISTANCES:
                    conditions = arms[distance]
                    record = {
                        "pair_id": pair["id"], "event_family": pair["event_family"],
                        "domain": pair["domain"], "unrelated_article_count": distance,
                        "later_source_id": later_id, "conditions": conditions,
                        "prefix_restatement_label": annotation.get("label", "unknown"),
                        "native_prefix_annotation": annotation,
                    }
                    for control in ("empty", "donor"):
                        record["delta_" + control] = conditions[control]["nll"] - conditions["self"]["nll"]
                        record["first_target_delta_" + control] = (
                            conditions[control]["first_target_nll"] - conditions["self"]["first_target_nll"])
                    for mode in ("self", "donor"):
                        record["direction_delta_" + mode] = (
                            conditions[mode + "_perm"]["nll"] - conditions[mode]["nll"])
                    record["direction_difference"] = (
                        record["direction_delta_self"] - record["direction_delta_donor"])
                    local_records.append(record)
                records.extend(row for group in gather(local_records) for row in group)
                counts = gather((local_write_calls, local_writer_tokens))
                execution["write_calls"] += sum(item[0] for item in counts)
                execution["writer_tokens"] += sum(item[1] for item in counts)
                session.assert_frozen()
                if rank == 0:
                    _write_json(output / "per_family_partial.json", records)
                    _write_json(output / "progress.json", {
                        "protocol": PROTOCOL, "status": "evaluating", "arm": arm,
                        "milestone": milestone, "families_complete": 4 * (round_index + 1),
                        "families_total": 64, "evaluation_target_accesses": charged,
                        "expected_evaluation_target_accesses": expected_cost,
                        "seconds": time.monotonic() - started,
                    })
                    print(json.dumps({
                        "event": "native_dev64_families_complete", "arm": arm, "milestone": milestone,
                        "families_complete": 4 * (round_index + 1), "evaluation_target_accesses": charged,
                    }, sort_keys=True), flush=True)
                del evidence, evidence_frames, tasks, query_frames, arms, local_records

        require(charged == expected_cost and len(records) == 64 * len(DISTANCES),
                "DEV64 did not complete its exact frozen condition/target coverage")
        require(load_native_dev64_cache(
            plan_path.parent if plan_path.is_file() else plan_path)["identity"] == plan_identity,
            "DEV64 prepared inputs changed during evaluation")
        _check_module(writer, writer_guard, "writer")
        _check_module(alpha, alpha_guard, "alpha")
        session.assert_frozen()
        # Summary uses a private seeded Python Random, so bootstrap does not
        # consume the training RNG. Compute once, then share the same result.
        summary = summarize_continuation(records) if rank == 0 else None
        summary = gather(summary)[0]
        report = {
            "protocol": PROTOCOL, "complete": True, "evaluation_only": True, "split": "dev",
            "arm": arm, "milestone": milestone, "alpha": alpha.initial_alpha,
            "checkpoint": checkpoint_metadata, "writer_sha256": writer_sha,
            "independent_families": 64, "conditions": list(CONDITIONS),
            "evaluation_target_accesses": charged, "expected_evaluation_target_accesses": expected_cost,
            "optimizer_steps": 0, "backward_calls": 0, "execution": execution,
            "manifest": manifest, "manifest_sha256": canonical_hash(manifest),
            "per_family": records, "summary": summary,
            "seconds": max(gather(time.monotonic() - started)),
        }
        if rank == 0:
            _write_json(output / "report.json", report)
            _write_json(output / "COMPLETE.json", {
                "protocol": PROTOCOL, "complete": True, "arm": arm, "milestone": milestone,
                "evaluation_target_accesses": charged, "writer_sha256": writer_sha,
                "report_sha256": hashlib.sha256((output / "report.json").read_bytes()).hexdigest(),
            })
        dist.barrier()
        return report
    except BaseException as error:
        if rank == 0 and output.is_dir():
            _write_json(output / "FAILURE.json", {
                "protocol": PROTOCOL, "complete": False, "arm": arm, "milestone": milestone,
                "evaluation_target_accesses": charged, "error_type": type(error).__name__,
                "message": str(error), "refunds": False, "automatic_retry": False,
            })
        raise
    finally:
        # No optimizer/sampler receives this module's temporary evaluation state.
        # Restore even after a failed forward; all consumed CE remains charged.
        _restore_rng(saved_rng, device, tracker)
        for guard in (writer_guard, alpha_guard):
            for module, was_training in guard[2]:
                module.training = was_training
        _check_module(writer, writer_guard, "writer")
        _check_module(alpha, alpha_guard, "alpha")

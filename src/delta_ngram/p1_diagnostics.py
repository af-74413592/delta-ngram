"""Frozen, causally ordered P1 diagnostics on an immutable development cohort.

All conditions use B=1. Donors change only the prior written state; context
changes only the visible true history. Target-token losses use the same shift.
The existing training objective and its saved development metrics are untouched.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from .batching import BatchMemoryState, write_batch
from .block_monitor import query_hit_summary
from .checkpoints import load_checkpoint, restore_writer
from .diagnostic_manifest import validate_diagnostic_manifest
from .raw_lm import (PROTOCOL as RAW_PROTOCOL, VALIDATION_PROTOCOL, _identity,
                     _json_hash, summarize_group_losses, tokenizer_identity)
from .runtime import synchronize


PROTOCOL = "delta_ngram_p1_frozen_diagnostics_v1"
RECURRENCE_LABELS = ("present_before_local_prefix", "absent_before_local_prefix")


def _json_value(value):
    # Exported torch checkpoints retain integer YAML keys; config.json does not.
    return json.loads(json.dumps(value, allow_nan=False))


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_diagnostic_checkpoint(method, config, checkpoint_path, *, expected_identity=None):
    """Load a raw writer with its training identity, then freeze all parameters.

``expected_identity`` should be the selected training run's config.json
raw_identity. Evaluating at B=1 does not change the original training config.
"""
    path = Path(checkpoint_path).resolve()
    payload = load_checkpoint(path)
    metadata = payload["config"]
    identity = metadata.get("raw_identity")
    if payload.get("protocol") != RAW_PROTOCOL or not isinstance(identity, dict):
        raise ValueError("diagnostics require an identified raw-LM writer checkpoint")
    _positive_integer(payload["step"], "checkpoint step")
    tokens = _positive_integer(metadata.get("training_loss_tokens"), "checkpoint training_loss_tokens")
    current_tokenizer = tokenizer_identity(method)
    if _json_value(identity.get("tokenizer")) != _json_value(current_tokenizer):
        raise ValueError("checkpoint tokenizer identity differs from the diagnostic runtime")
    current = _identity(config, method, identity.get("dataset_sha256"), current_tokenizer)
    current["validation"] = identity.get("validation")
    if _json_value(current) != _json_value(identity):
        raise ValueError("checkpoint raw identity differs from the configured runtime/training recipe")
    if expected_identity is not None and _json_value(identity) != _json_value(expected_identity):
        raise ValueError("checkpoint raw identity differs from the selected training run")
    raw = config.values["raw"]
    validation = {"name": VALIDATION_PROTOCOL, "selection": "group_first_round_robin",
                  "group_key": "group_id", "seed": raw["eval_seed"], "window_limit": raw["eval_windows"],
                  "bootstrap_samples": raw["eval_bootstrap_samples"], "minimum_groups": 64,
                  "evaluation_batching": "within_group", "batch_size": raw["batch_size"],
                  "prefix_tokens": raw["prefix_tokens"]}
    if (not isinstance(current["validation"], dict) or
            current["validation"].get("validation_protocol_sha256") != _json_hash(validation)):
        raise ValueError("checkpoint fixed development protocol differs from the configuration")
    fingerprint = "sha256:" + _json_hash(current_tokenizer)
    if metadata.get("tokenizer_fingerprint") != fingerprint:
        raise ValueError("checkpoint tokenizer fingerprint is inconsistent")
    restore_writer(method, payload, config)
    method.train(False)
    method.writer.requires_grad_(False)
    method.reader.model.requires_grad_(False)
    method.writer.zero_grad(set_to_none=True)
    method.reader.model.zero_grad(set_to_none=True)
    return {"checkpoint_path": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint_step": payload["step"], "training_loss_tokens": tokens,
            "raw_identity": _json_value(identity), "tokenizer_fingerprint": fingerprint,
            "original_evaluation_batch_size": validation["batch_size"],
            "original_validation_protocol": validation}


def per_target_cross_entropy(logits, labels):
    """Return [B,L-1] CE and mask: logit t predicts label t+1.

Ignored positions contain zero, including prefix/padding. A target at the first
new-token position is scored from the preceding prefix's final logit.
"""
    if (logits.ndim != 3 or labels.shape != logits.shape[:2] or labels.dtype != torch.long
            or labels.device != logits.device or logits.shape[1] < 2):
        raise ValueError("logits/labels must be aligned [B,L,V] / int64 [B,L]")
    valid = labels[:, 1:].ne(-100)
    losses = logits.new_zeros(valid.shape, dtype=torch.float32)
    if bool(valid.any()):
        losses[valid] = F.cross_entropy(logits[:, :-1][valid].float(), labels[:, 1:][valid], reduction="none")
        if not bool(torch.isfinite(losses[valid]).all()):
            raise RuntimeError("diagnostic target cross entropy is non-finite")
    return losses, valid


def recurrence_mask(tokens, written_blocks, block_size, prefix_tokens):
    """Exact token-ID membership in written history before the direct prefix."""
    _positive_integer(written_blocks, "written_blocks")
    _positive_integer(block_size, "block_size")
    if type(prefix_tokens) is not int or not 1 <= prefix_tokens <= block_size:
        raise ValueError("prefix_tokens must lie between 1 and block_size")
    boundary = written_blocks * block_size
    if boundary + block_size > len(tokens):
        raise ValueError("recurrence requires the complete next natural block")
    history = set(tokens[:boundary - prefix_tokens])
    return tuple(token in history for token in tokens[boundary:boundary + block_size])


def _pair_summary(records, left, right, *, seed, samples):
    selected = [record for record in records if left in record["loss_sums"] and right in record["loss_sums"]]
    grouped = defaultdict(list)
    for record in selected:
        grouped[record["group_id"]].append(record)
    if not grouped:
        return {"status": "unavailable", "loss_tokens": 0, "group_count": 0,
                "delta": None, "ci_lower": None, "ci_upper": None,
                "definition": f"L_{left} - L_{right}"}
    groups = [{"group_id": group, "loss_tokens": sum(r["loss_tokens"] for r in items),
               "self_loss_sum": math.fsum(r["loss_sums"][right] for r in items),
               "empty_loss_sum": math.fsum(r["loss_sums"][left] for r in items)}
              for group, items in sorted(grouped.items())]
    result = summarize_group_losses(groups, bootstrap_seed=seed, bootstrap_samples=samples)
    bootstrap = result["paired_group_bootstrap"]
    return {"status": "evaluated" if len(groups) >= 64 else "insufficient_independent_groups",
            "definition": f"L_{left} - L_{right}", "loss_tokens": result["loss_tokens"],
            "group_count": result["group_count"], "delta": result["delta_empty"],
            "left_nll": result["empty_nll"], "right_nll": result["self_nll"],
            "ci_lower": bootstrap["ci_lower"], "ci_upper": bootstrap["ci_upper"],
            "bootstrap": bootstrap,
            "groups": [{"group_id": item["group_id"], "loss_tokens": item["loss_tokens"],
                        "left_loss_sum": item["empty_loss_sum"], "right_loss_sum": item["self_loss_sum"]}
                       for item in groups]}


def _summarize(records, *, seed, samples):
    conditions = sorted({key for record in records for key in record["loss_sums"]})
    totals = {}
    for condition in conditions:
        selected = [record for record in records if condition in record["loss_sums"]]
        count = sum(record["loss_tokens"] for record in selected)
        total = math.fsum(record["loss_sums"][condition] for record in selected)
        totals[condition] = {"loss_tokens": count, "loss_sum": total, "nll": total / count if count else None}
    comparisons = {}
    for name, left, right in (("delta_empty", "empty", "self"), ("delta_donor", "donor_mean", "self"),
                               ("delta_context", "context", "self"),
                               ("empty_minus_context", "empty", "context")):
        comparisons[name] = _pair_summary(records, left, right, seed=seed, samples=samples)
    for condition in conditions:
        if condition.startswith("donor_") and condition != "donor_mean":
            comparisons[f"delta_{condition}"] = _pair_summary(records, condition, "self", seed=seed, samples=samples)
    numerator, denominator = comparisons["delta_empty"], comparisons["empty_minus_context"]
    ratio = None
    ratio_reason = "requires empty-minus-context > 0.01 and its group CI lower bound > 0"
    if (numerator["delta"] is not None and denominator["delta"] is not None
            and denominator["delta"] > .01 and denominator["ci_lower"] > 0):
        ratio = numerator["delta"] / denominator["delta"]
        ratio_reason = "prespecified denominator and CI conditions met; ratio is not clipped"
    return {"windows": len({record["window_id"] for record in records}),
            "group_count": len({record["group_id"] for record in records}),
            "loss_tokens": sum(record["loss_tokens"] for record in records),
            "conditions": totals, "comparisons": comparisons, "R": ratio, "R_reason": ratio_reason}


def _stratified(records, *, seed, samples):
    by_source = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(record)
    return {source: _summarize(items, seed=seed, samples=samples) for source, items in sorted(by_source.items())}


def _check_bound_identity(method, manifest, checkpoint, baseline):
    fingerprint = "sha256:" + _json_hash(tokenizer_identity(method))
    if manifest["tokenizer_identity"] != fingerprint:
        raise ValueError("diagnostic manifest tokenizer differs from the active reader")
    if checkpoint is not None:
        identity = checkpoint["raw_identity"]
        expected = identity["validation"]
        if (checkpoint["tokenizer_fingerprint"] != fingerprint or
                manifest["dataset_sha256"] != identity["dataset_sha256"] or
                manifest["baseline_validation_sample_sha256"] != expected["validation_sample_sha256"] or
                manifest["baseline_validation_protocol_sha256"] != expected["validation_protocol_sha256"]):
            raise ValueError("diagnostic manifest differs from the checkpoint corpus/fixed development identity")
        if (manifest["protocol"]["block_size"] != identity["raw"]["block_size"] or
                manifest["protocol"]["prefix_tokens"] != identity["raw"]["prefix_tokens"]):
            raise ValueError("diagnostic block/prefix recipe differs from checkpoint")
    if baseline is not None:
        for key, value in (("validation_sample_sha256", manifest["baseline_validation_sample_sha256"]),
                           ("validation_protocol_sha256", manifest["baseline_validation_protocol_sha256"])):
            if baseline.get(key) != value:
                raise ValueError("baseline evaluation does not use the fixed diagnostic cohort/protocol")
        if checkpoint is not None and (baseline.get("step") != checkpoint["checkpoint_step"] or
                                       baseline.get("training_loss_tokens") != checkpoint["training_loss_tokens"]):
            raise ValueError("baseline evaluation is not from the loaded writer checkpoint")


class _Execution:
    """One evaluation's transient state, callbacks, and actual logical costs."""

    def __init__(self, method, block_size, prefix_tokens, callback):
        self.method, self.block_size, self.prefix_tokens = method, block_size, prefix_tokens
        self.callback, self.ledger, self.records = callback, {}, 0

    def empty(self):
        reader = self.method.reader
        return BatchMemoryState.empty(1, reader.width, reader.buckets * len(reader.orders), self.method.device)

    def emit(self, record):
        self.records += 1
        if self.callback is not None:
            self.callback(record)

    def cost(self, cohort, condition):
        return self.ledger.setdefault(cohort, {}).setdefault(condition, {
            "scored_targets": 0, "reader_input_tokens": 0, "evidence_encoded_tokens": 0,
            "written_tokens": 0, "write_blocks": 0, "score_calls": 0,
            "evidence_calls": 0, "elapsed_seconds": 0.0})

    def score(self, state, tokens, block, condition, metadata):
        boundary = block * self.block_size
        start = 0 if condition == "context" else boundary - self.prefix_tokens
        values = tokens[start:boundary + self.block_size]
        maximum = getattr(self.method.reader.model.config, "max_position_embeddings", None)
        if maximum is not None and len(values) > maximum:
            raise ValueError("diagnostic context exceeds the configured reader length; refusing to truncate")
        ids = torch.tensor([values], dtype=torch.long, device=self.method.device)
        mask = torch.ones_like(ids)
        labels = torch.full_like(ids, -100)
        labels[:, -self.block_size:] = ids[:, -self.block_size:]
        synchronize(self.method.device)
        started = time.perf_counter()
        addresses = self.method.reader.adapter.addresses(ids, mask)
        query = query_hit_summary(state, addresses, labels, mask)[0]
        logits = self.method.reader(ids, mask, state).logits
        losses, valid = per_target_cross_entropy(logits, labels)
        selected = losses[valid].detach().cpu().tolist()
        if len(selected) != self.block_size:
            raise RuntimeError("diagnostic conditions scored different target tokens")
        synchronize(self.method.device)
        cost = self.cost(metadata["cohort"], condition)
        cost["score_calls"] += 1
        cost["scored_targets"] += len(selected)
        cost["reader_input_tokens"] += len(values)
        cost["elapsed_seconds"] += time.perf_counter() - started
        self.emit({**metadata, "kind": "query", "condition": condition, "block": block,
                   "written_blocks": block if condition not in ("empty", "context") else 0,
                   "target_block_number": block + 1, "loss_tokens": len(selected),
                   "loss_sum": math.fsum(selected), "reader_input_tokens": len(values),
                   "target_token_sha256": _json_hash(tokens[boundary:boundary + self.block_size]),
                   "metrics": query})
        return selected

    def write(self, state, tokens, block, condition, metadata, *, evidence_document_id, evidence_start):
        begin = block * self.block_size
        current = torch.tensor([tokens[begin:begin + self.block_size]], dtype=torch.long, device=self.method.device)
        if current.shape[1] != self.block_size:
            raise ValueError("diagnostic write requires a complete evidence block")
        mask = torch.ones_like(current)
        synchronize(self.method.device)
        started = time.perf_counter()
        hidden, _ = self.method.reader.encode_evidence(current, mask)
        boundary = max(self.method.reader.orders) - 1
        if begin and boundary:
            history = torch.tensor([tokens[max(0, begin - boundary):begin]], dtype=torch.long, device=self.method.device)
            address_ids = torch.cat((history, current), dim=1)
            rows = self.method.reader.adapter.addresses(address_ids, torch.ones_like(address_ids))[:, history.shape[1]:]
        else:
            rows = self.method.reader.adapter.addresses(current, mask)
        summaries = []
        state = write_batch(self.method.writer, state, hidden, rows, mask, observer=summaries.extend)
        synchronize(self.method.device)
        cost = self.cost(metadata["cohort"], condition)
        cost["evidence_calls"] += 1
        cost["write_blocks"] += 1
        cost["evidence_encoded_tokens"] += self.block_size
        cost["written_tokens"] += self.block_size
        cost["elapsed_seconds"] += time.perf_counter() - started
        if len(summaries) != 1:
            raise RuntimeError("diagnostic B=1 write emitted an unexpected owner count")
        self.emit({**metadata, "kind": "write", "condition": condition, "block": block,
                   "written_blocks": block + 1, "evidence_document_id": evidence_document_id,
                   "evidence_start": evidence_start + begin, "metrics": summaries[0]})
        return state


def _record(metadata, losses, mask, *, bucket=None):
    selected = range(len(mask)) if bucket is None else [i for i, present in enumerate(mask) if present == bucket]
    selected = tuple(selected)
    if not selected:
        return None
    sums = {condition: math.fsum(values[i] for i in selected) for condition, values in losses.items()}
    donors = [name for name in sums if name.startswith("donor_")]
    if donors:
        sums["donor_mean"] = math.fsum(sums[name] for name in donors) / len(donors)
    return {**metadata, "loss_tokens": len(selected), "loss_sums": sums}


@torch.no_grad()
def evaluate_diagnostics(method, documents, manifest, *, bootstrap_seed=104729,
                         bootstrap_samples=2000, block_callback=None, progress_callback=None,
                         baseline_evaluation=None, checkpoint_metadata=None):
    """Evaluate frozen self/empty/fixed donors/context and next-block curves.

Callbacks receive JSON-only records and may raise to abort on an external time
budget. Partial logs never produce a complete result. Missing donor coverage
does not remove any primary self/empty/context target or substitute one donor.
"""
    _positive_integer(bootstrap_samples, "bootstrap_samples")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    documents = tuple(documents)
    validate_diagnostic_manifest(manifest, documents)
    _check_bound_identity(method, manifest, checkpoint_metadata, baseline_evaluation)
    if not method.supports_outer_training or method.reader.backend != "external_hash":
        raise ValueError("diagnostics require the learned external-hash writer")
    method.train(False)
    method.writer.requires_grad_(False)
    method.reader.model.requires_grad_(False)
    method.writer.zero_grad(set_to_none=True)
    method.reader.model.zero_grad(set_to_none=True)
    indexed = {document.id: document for document in documents}
    protocol = manifest["protocol"]
    block_size, prefix = protocol["block_size"], protocol["prefix_tokens"]
    execution = _Execution(method, block_size, prefix, block_callback)
    primary, all_dev, recurring = [], [], {label: [] for label in RECURRENCE_LABELS}
    started = time.perf_counter()

    def progress(stage, completed, total, **extra):
        if progress_callback is not None:
            progress_callback({"stage": stage, "completed": completed, "total": total,
                               "elapsed_seconds": time.perf_counter() - started, **extra})

    windows = manifest["primary_windows"]
    progress("primary", 0, len(windows))
    for position, window in enumerate(windows):
        metadata = {key: window[key] for key in ("window_id", "document_id", "group_id", "source", "start")}
        metadata["cohort"] = "primary"
        tokens = indexed[window["document_id"]].token_ids[window["start"]:window["start"] + window["blocks"] * block_size]
        donor_entries = window["donors"] if window["donor_status"] == "matched" else []
        if donor_entries and len(donor_entries) != protocol["donors_per_target"]:
            raise ValueError("matched primary window lacks the fixed number of donors")
        histories = {f"donor_{i + 1}": indexed[donor["document_id"]].token_ids[
            donor["start"]:donor["start"] + donor["history_blocks"] * block_size]
                     for i, donor in enumerate(donor_entries)}
        states = {condition: execution.empty() for condition in ("self", *histories)}
        empty = execution.empty()
        window_all_dev = bool(donor_entries) and all(donor["split"] == "validation" for donor in donor_entries)
        for block in range(window["blocks"]):
            if block:
                losses = {condition: execution.score(states[condition], tokens, block, condition, metadata)
                          for condition in states}
                losses["empty"] = execution.score(empty, tokens, block, "empty", metadata)
                losses["context"] = execution.score(empty, tokens, block, "context", metadata)
                mask = recurrence_mask(tokens, block, block_size, prefix)
                record = _record(metadata, losses, mask)
                primary.append(record)
                if window_all_dev:
                    all_dev.append(record)
                for present, label in zip((True, False), RECURRENCE_LABELS):
                    bucket_record = _record(metadata, losses, mask, bucket=present)
                    if bucket_record is not None:
                        recurring[label].append(bucket_record)
                execution.emit({**metadata, "kind": "paired_score", "block": block,
                                "written_blocks": block, "target_block_number": block + 1,
                                "loss_tokens": block_size, "loss_sums": record["loss_sums"],
                                "recurrence_tokens": {RECURRENCE_LABELS[0]: sum(mask),
                                                      RECURRENCE_LABELS[1]: len(mask) - sum(mask)}})
            if block == window["blocks"] - 1:
                continue
            states["self"] = execution.write(states["self"], tokens, block, "self", metadata,
                                               evidence_document_id=window["document_id"], evidence_start=window["start"])
            for i, donor in enumerate(donor_entries):
                condition = f"donor_{i + 1}"
                states[condition] = execution.write(states[condition], histories[condition], block, condition, metadata,
                                                    evidence_document_id=donor["document_id"], evidence_start=donor["start"])
        progress("primary", position + 1, len(windows), window_id=window["window_id"])

    after_records = defaultdict(list)
    origins = defaultdict(list)
    for target in manifest["after_targets"]:
        origins[target["document_id"], target["start"]].append(target)
    progress("after_write", 0, len(origins))
    for position, ((document_id, start), targets) in enumerate(sorted(origins.items())):
        wanted = {target["n_written"]: target for target in targets}
        maximum = max(wanted)
        tokens = indexed[document_id].token_ids[start:start + (maximum + 1) * block_size]
        state, empty = execution.empty(), execution.empty()
        base = {"cohort": "after_write", "window_id": f"after-origin:{document_id}:{start}",
                "document_id": document_id, "group_id": indexed[document_id].group_id,
                "source": indexed[document_id].source, "start": start}
        for block in range(maximum + 1):
            if block in wanted:
                metadata = {**base, "target_id": wanted[block]["target_id"]}
                losses = {"self": execution.score(state, tokens, block, "self", metadata),
                          "empty": execution.score(empty, tokens, block, "empty", metadata)}
                record = _record(metadata, losses, recurrence_mask(tokens, block, block_size, prefix))
                after_records[block].append(record)
                execution.emit({**metadata, "kind": "paired_score", "block": block,
                                "written_blocks": block, "target_block_number": block + 1,
                                "loss_tokens": block_size, "loss_sums": record["loss_sums"]})
            if block < maximum:
                state = execution.write(state, tokens, block, "self", base,
                                          evidence_document_id=document_id, evidence_start=start)
        progress("after_write", position + 1, len(origins), document_id=document_id, start=start)

    settings = {"seed": bootstrap_seed, "samples": bootstrap_samples}
    primary_result = _summarize(primary, **settings)
    expected_targets = sum(window["loss_tokens"] for window in windows)
    if primary_result["loss_tokens"] != expected_targets:
        raise RuntimeError("diagnostics did not preserve every original primary target token")
    baseline_comparison = {"status": "not_supplied"}
    if baseline_evaluation is not None:
        if baseline_evaluation.get("loss_tokens") != expected_targets:
            raise ValueError("baseline target count differs from the diagnostic primary cohort")
        baseline_comparison = {"status": "evaluated", "interpretation": "same checkpoint/targets; numerical batch/reduction difference, not training improvement"}
        for condition in ("self", "empty"):
            old = baseline_evaluation.get(f"{condition}_nll")
            if type(old) not in (int, float) or not math.isfinite(old):
                raise ValueError("baseline NLL must be finite")
            new = primary_result["conditions"][condition]["nll"]
            baseline_comparison[condition] = {"original_nll": old, "recomputed_nll": new, "difference": new - old}
    result = {"protocol": PROTOCOL, "complete": True, "evaluation_batch_size": 1,
              "evaluation_batching": "one window at a time, identical across memory conditions",
              "original_evaluation_batch_size": (checkpoint_metadata.get("original_evaluation_batch_size")
                                                   if checkpoint_metadata else None),
              "baseline_numerical_comparison": baseline_comparison,
              "checkpoint_identity_verified": checkpoint_metadata is not None,
              "manifest_sha256": manifest["manifest_sha256"], "checkpoint": checkpoint_metadata,
              "baseline_validation_sample_sha256": manifest["baseline_validation_sample_sha256"],
              "baseline_validation_protocol_sha256": manifest["baseline_validation_protocol_sha256"],
              "bootstrap": {"seed": bootstrap_seed, "samples": bootstrap_samples, "unit": "paired document group"},
              "primary": primary_result, "primary_coverage": manifest["coverage"]["primary"],
              "all_dev_donors": _summarize(all_dev, **settings),
              "by_language": _stratified(primary, **settings),
              "by_recurrence": {label: _summarize(items, **settings) for label, items in recurring.items()},
              "recurrence_definition": protocol["recurrence_definition"],
              "after_write": {str(n): {"coverage": manifest["coverage"]["after"][str(n)],
                                        "metrics": _summarize(after_records[n], **settings),
                                        "by_language": _stratified(after_records[n], **settings)}
                              for n in protocol["after_write_blocks"]},
              "after_unavailable_targets": manifest["after_unavailable_targets"],
              "budget": {"by_cohort_and_condition": execution.ledger,
                         "elapsed_seconds": time.perf_counter() - started,
                         "definition": "actual logical reader/evidence inputs; elapsed includes instrumentation; context full history counted separately"},
              "block_records_emitted": execution.records,
              "claims": {"holdout_used": False, "pretraining_novelty_tested": False,
                         "literal_recurrence_is_semantic_memory": False, "p1_gate_redefined": False,
                         "p6_gate_evaluated": False}}
    json.dumps(result, allow_nan=False)
    progress("complete", 1, 1, primary_loss_tokens=expected_targets)
    return result

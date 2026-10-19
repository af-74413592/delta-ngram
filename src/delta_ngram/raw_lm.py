"""Causal document-block training; streams batch, state updates stay ordered.

The writer and row operators are shared with the established evidence/query
path. Only the supervision schedule changes. No block is written before its
own loss is computed. The first block warms memory without a language loss.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .batching import BatchMemoryState, write_batch
from .block_monitor import MONITOR_PROTOCOL, monitor_source_identity, query_hit_summary
from .checkpoints import load_checkpoint, restore_writer, save_checkpoint
from .data.raw_text import RawWindowSampler, build_windows, load_documents
from .runtime import log_event, save_json, synchronize
from .train import create_runtime, initialize_output

PROTOCOL = "delta_ngram_raw_lm_v1"
RESUME_FORMAT = "delta_ngram_raw_resume_v1"
VALIDATION_PROTOCOL = "group_first_fixed_dev_paired_bootstrap_v1"
MINIMUM_EVALUATION_GROUPS = 64


@dataclass
class TrajectoryLoss:
    loss_sum: torch.Tensor
    loss_tokens: int
    state: BatchMemoryState
    counters: dict[str, int]
    block_loss_sums: tuple[torch.Tensor, ...]

    @property
    def mean(self):
        return self.loss_sum / self.loss_tokens


def _pad_windows(windows, device, pad_token_id):
    if not windows:
        raise ValueError("raw trajectory batch must not be empty")
    block_size = windows[0].block_size
    if any(window.block_size != block_size or window.blocks < 2
           or len(window.token_ids) != window.blocks * block_size for window in windows):
        raise ValueError("windows must have complete blocks of a common size")
    tokens = torch.full((len(windows), max(w.blocks for w in windows), block_size),
                        pad_token_id, dtype=torch.long, device=device)
    lengths = torch.tensor([w.blocks for w in windows], dtype=torch.long, device=device)
    for owner, window in enumerate(windows):
        tokens[owner, :window.blocks] = torch.tensor(window.token_ids, dtype=torch.long,
                                                    device=device).reshape(window.blocks, block_size)
    return tokens, lengths


def score_block(reader, state, input_ids, attention_mask, labels, *, checkpoint_reader=False):
    """Return a token-sum CE, replaying the exact snapshot if recomputed.

    The whole score call owns its scoped hook, including backward recomputation.
    This does not enable Hugging Face's layer checkpointing, whose execution
    outside the hook lifetime is explicitly rejected by FrozenMemoryReader.
    """
    valid = labels[:, 1:].ne(-100)
    count = int(valid.sum().item())
    if count == 0:
        raise ValueError("a scoring block needs at least one valid shifted target")
    targets = labels[:, 1:][valid]

    def score(values):
        snapshot = state.with_values(values)
        logits = reader(input_ids, attention_mask, snapshot).logits[:, :-1, :]
        # Do not promote/store ignored prefix and padding logits in FP32.
        return F.cross_entropy(logits[valid].float(), targets, reduction="sum")

    if checkpoint_reader and torch.is_grad_enabled() and state.memory.values.requires_grad:
        loss = checkpoint(score, state.memory.values, use_reentrant=False)
    else:
        loss = score(state.memory.values)
    return loss, count


def trajectory_loss(method, windows, *, prefix_tokens=128, checkpoint_reader=False,
                    memory_mode="self", observer=None, block_observer=None, block_context=None):
    """Run full BPTT over complete windows, with one namespace per window.

    ``observer`` is an optional correctness/debug callback. It receives the
    fixed snapshot BEFORE a read or write; the trainer does not retain it.
    Empty-memory scoring leaves the reader and writer parameters intact.
    ``block_observer`` instead receives detached JSON records before each
    scored read and after each write. It runs outside score checkpointing.
    """
    if method.reader.backend != "external_hash":
        raise NotImplementedError("raw LM currently supports the external hash adapter only")
    if memory_mode not in ("self", "empty"):
        raise ValueError("memory_mode must be self or empty")
    if not method.supports_outer_training:
        raise ValueError("raw LM requires a learned forward writer")
    block_size = windows[0].block_size if windows else 0
    if type(prefix_tokens) is not int or not 1 <= prefix_tokens <= block_size:
        raise ValueError("prefix_tokens must lie between 1 and block_size")
    pad = getattr(method.tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(method.tokenizer, "eos_token_id", None)
    if type(pad) is not int or pad < 0:
        raise ValueError("tokenizer needs an integer padding or EOS token ID")
    tokens, lengths = _pad_windows(windows, method.device, pad)
    capacity = method.reader.buckets * len(method.reader.orders)
    state = BatchMemoryState.empty(len(windows), method.reader.width, capacity, method.device)
    losses = []
    counters = {"loss_tokens": 0, "raw_tokens_consumed": sum(len(w.token_ids) for w in windows),
                "writer_new_tokens": 0, "encoder_input_tokens": 0,
                "reader_input_tokens": 0, "address_prefix_tokens": 0,
                "write_blocks": 0, "trajectories": len(windows)}
    boundary = max(method.reader.orders) - 1

    def emit_block(kind, block, records, eligible):
        for record in records:
            owner = record["owner"]
            if not bool(eligible[owner]):
                continue
            window = windows[owner]
            block_observer({**(block_context or {}), "event": "block_metrics",
                            "protocol": MONITOR_PROTOCOL, "kind": kind, "owner": owner,
                            "document_id": window.document_id, "group_id": window.group_id,
                            "source": window.source, "window_start": window.start,
                            "window_blocks": window.blocks, "block": block,
                            "block_start": window.start + block * block_size,
                            "memory_mode": memory_mode, "metrics": record})

    for block in range(tokens.shape[1]):
        current = tokens[:, block]
        active = lengths > block
        current_mask = active[:, None].expand_as(current).to(torch.long)
        if block:
            prefix = tokens[:, block - 1, -prefix_tokens:]
            ids = torch.cat((prefix, current), dim=1)
            mask = active[:, None].expand_as(ids).to(torch.long)
            labels = torch.full_like(ids, -100)
            labels[:, prefix_tokens:] = torch.where(current_mask.bool(), current, -100)
            if observer is not None:
                observer("read", block, state, ids, mask, labels)
            if block_observer is not None:
                addresses = method.reader.adapter.addresses(ids, mask)
                emit_block("query", block, query_hit_summary(state, addresses, labels, mask), active)
            loss, count = score_block(method.reader, state, ids, mask, labels,
                                      checkpoint_reader=checkpoint_reader)
            losses.append(loss)
            counters["loss_tokens"] += count
            counters["reader_input_tokens"] += int(mask.sum().item())

        # Last blocks are never written: there is no later supervised use.
        writing = lengths > block + 1
        if memory_mode == "empty" or not bool(writing.any()):
            continue
        write_mask = writing[:, None].expand_as(current).to(torch.long)
        if observer is not None:
            observer("write", block, state, current, write_mask, None)
        hidden, _ = method.reader.encode_evidence(current, write_mask)
        if block and boundary:
            old_tokens = tokens[:, :block].reshape(len(windows), -1)[:, -boundary:]
            address_ids = torch.cat((old_tokens, current), dim=1)
            address_mask = writing[:, None].expand_as(address_ids).to(torch.long)
            rows = method.reader.adapter.addresses(address_ids, address_mask)[:, old_tokens.shape[1]:]
            counters["address_prefix_tokens"] += int(writing.sum().item()) * old_tokens.shape[1]
        else:
            rows = method.reader.adapter.addresses(current, write_mask)
        # Old tokens above are addressing-only; they are neither pooled nor
        # written again, and do not enter the evidence encoder a second time.
        if block_observer is None:
            state = write_batch(method.writer, state, hidden, rows, write_mask)
        else:
            state = write_batch(method.writer, state, hidden, rows, write_mask,
                                observer=lambda records: emit_block("write", block, records, writing))
        encoded = int(write_mask.sum().item())
        counters["writer_new_tokens"] += encoded
        counters["encoder_input_tokens"] += encoded
        counters["write_blocks"] += int(writing.sum().item())

    expected = sum((w.blocks - 1) * w.block_size for w in windows)
    if not losses or counters["loss_tokens"] != expected:
        raise RuntimeError("raw trajectory target accounting mismatch")
    return TrajectoryLoss(torch.stack(losses).sum(), expected, state, counters, tuple(losses))


def _json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tokenizer_identity(method):
    tokenizer = method.tokenizer
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else tokenizer.mapping
    return {"class": type(tokenizer).__name__, "vocab_sha256": _json_hash(vocab),
            "pipeline_sha256": (_json_hash(json.loads(tokenizer.backend_tokenizer.to_str()))
                                if hasattr(tokenizer, "backend_tokenizer") else None),
            "special_tokens": {name: getattr(tokenizer, name, None)
                               for name in ("pad_token_id", "eos_token_id", "bos_token_id")}}


def _identity(config, method, dataset_sha256, token_identity):
    values, raw = config.values, config.values["raw"]
    # Stop budgets and output paths may change when continuing a run. The
    # mathematical experiment, stream order and batch boundaries may not.
    identity = {"protocol": PROTOCOL, "seed": values["seed"], "dataset_sha256": dataset_sha256,
            "tokenizer": token_identity,
            "model": {key: values["model"][key] for key in ("name_or_path", "revision", "smoke")},
            "resolved_model_commit": getattr(method.reader.model.config, "_commit_hash", None),
            "device_type": method.device.type, "dtype": str(next(method.reader.model.parameters()).dtype),
            "reader": values["reader"], "writer": values["writer"], "method": values["method"],
            "optimizer": {"lr": values["train"]["lr"], "gradient_clip": values["train"]["gradient_clip"],
                          "name": "AdamW", "weight_decay": 0.0},
            "raw": {key: raw[key] for key in ("block_size", "prefix_tokens", "block_counts",
                                               "length_weights", "source_weights", "batch_size",
                                               "gradient_accumulation_steps", "checkpoint_reader")}}
    # Omit new inactive options so pre-existing resume identities stay valid.
    if raw.get("source_length_weights") is not None:
        identity["raw"]["source_length_weights"] = raw["source_length_weights"]
    if raw.get("eval_min_groups_per_source", 0):
        identity["raw"]["eval_min_groups_per_source"] = raw["eval_min_groups_per_source"]
    return identity


def _writer_metadata(config, method, args, identity):
    return {**vars(args), "out": str(args.out), "device": str(method.device), "protocol": PROTOCOL,
            "package_config": config.values, "config_source": str(config.source),
            "width": method.reader.width, "orders": list(method.reader.orders),
            "raw_identity": identity,
            "tokenizer_fingerprint": "sha256:" + _json_hash(identity["tokenizer"]),
            "resolved_model_commit": getattr(method.reader.model.config, "_commit_hash", None),
            "interface": "external zero-base hash rows; native pretrained table is not implemented"}


def _save_resume(path, method, optimizer, sampler, identity, step, counters):
    payload = {"format": RESUME_FORMAT, "identity": identity, "step": step, "counters": counters,
               "writer_state_dict": method.writer.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
               "sampler_state": sampler.state_dict(), "torch_rng": torch.get_rng_state(),
               "python_rng": random.getstate(),
               "cuda_rng": torch.cuda.get_rng_state_all() if method.device.type == "cuda" else []}
    temporary = Path(path).with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_resume(path, method, optimizer, sampler, identity):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != RESUME_FORMAT:
        raise ValueError("raw.resume requires a raw-training checkpoint, not an exported writer")
    if payload.get("identity") != identity:
        raise ValueError("raw resume identity differs (model, tokenizer, corpus or training configuration)")
    step, counters = payload.get("step"), payload.get("counters")
    if type(step) is not int or step < 0 or not isinstance(counters, dict):
        raise ValueError("malformed raw resume counters")
    if any(type(v) is not int or v < 0 for v in counters.values()) or "loss_tokens" not in counters:
        raise ValueError("malformed raw resume token counts")
    sampler.load_state_dict(payload["sampler_state"])
    if sampler.total_loss_tokens != counters["loss_tokens"]:
        raise ValueError("raw resume sampler and loss token counts disagree")
    method.writer.load_state_dict(payload["writer_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["python_rng"])
    if method.device.type == "cuda":
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return step, counters


def select_validation_windows(windows, *, limit, seed, min_groups_per_source=0,
                              required_sources=None):
    """Cover distinct groups first, then take additional windows round-robin.

    Sorting before shuffling makes the selection independent of input order.
    This private RNG never consumes the training sampler or global RNG state.
    A positive source minimum reserves one window from that many distinct
    groups in every required source, failing before selection if coverage is
    impossible. Zero retains the original globally group-first selection.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("validation window limit must be a positive integer")
    if type(min_groups_per_source) is not int or min_groups_per_source < 0:
        raise ValueError("min_groups_per_source must be a nonnegative integer")
    groups = {}
    for window in windows:
        groups.setdefault(window.group_id, []).append(window)
    group_sources = {}
    if min_groups_per_source:
        for group_id, items in groups.items():
            sources = {item.source for item in items}
            if len(sources) != 1:
                raise ValueError("per-source validation requires one source per group_id")
            group_sources[group_id] = next(iter(sources))
        required = list(group_sources.values()) if required_sources is None else list(required_sources)
        if not required or any(not isinstance(source, str) or not source for source in required):
            raise ValueError("required validation sources must be nonempty source names")
        required = sorted(set(required))
        if limit < len(required) * min_groups_per_source:
            raise ValueError("validation window limit cannot cover minimum groups per source")
        for source in required:
            available = sum(value == source for value in group_sources.values())
            if available < min_groups_per_source:
                raise ValueError(f"validation source {source!r} has {available} groups; "
                                 f"requires {min_groups_per_source}")
    rng = random.Random(seed)
    group_ids = sorted(groups)
    rng.shuffle(group_ids)
    for group_id in group_ids:
        groups[group_id].sort(key=lambda item: (item.document_id, item.start, item.blocks))
        rng.shuffle(groups[group_id])
    selected, reserved_groups = [], set()
    if min_groups_per_source:
        reserved_counts = dict.fromkeys(required, 0)
        for group_id in group_ids:
            source = group_sources[group_id]
            if source in reserved_counts and reserved_counts[source] < min_groups_per_source:
                selected.append(groups[group_id][0])
                reserved_groups.add(group_id)
                reserved_counts[source] += 1
        if len(selected) == limit:
            return selected
    for index in range(max((len(items) for items in groups.values()), default=0)):
        for group_id in group_ids:
            if index < len(groups[group_id]) and not (index == 0 and group_id in reserved_groups):
                selected.append(groups[group_id][index])
                if len(selected) == limit:
                    return selected
    return selected


def _validation_sample(windows):
    return [{"group_id": item.group_id, "document_id": item.document_id, "source": item.source,
             "start": item.start, "blocks": item.blocks, "loss_tokens": item.loss_tokens,
             "token_sha256": _json_hash(item.token_ids)} for item in windows]


def summarize_group_losses(groups, *, bootstrap_seed=104729, bootstrap_samples=2000):
    """Paired cluster bootstrap of the ratio of loss sums to target counts.

    Groups, not overlapping windows or tokens, are the independent resampling
    units. Each draw includes both memory conditions and the same token count.
    This reports one checkpoint's check, never the multi-seed P1 or P6 gate.
    """
    groups = sorted(groups, key=lambda item: item["group_id"])
    if not groups or len({item["group_id"] for item in groups}) != len(groups):
        raise ValueError("evaluation requires nonempty, unique group records")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    for item in groups:
        if type(item["loss_tokens"]) is not int or item["loss_tokens"] < 1:
            raise ValueError("evaluation group needs positive target tokens")
        if not all(math.isfinite(item[key]) for key in ("self_loss_sum", "empty_loss_sum")):
            raise ValueError("evaluation group losses must be finite")
    count = sum(item["loss_tokens"] for item in groups)
    own = math.fsum(item["self_loss_sum"] for item in groups)
    empty = math.fsum(item["empty_loss_sum"] for item in groups)
    delta = (empty - own) / count
    paired = [(item["empty_loss_sum"] - item["self_loss_sum"], item["loss_tokens"])
              for item in groups]
    rng = random.Random(bootstrap_seed)
    draws = []
    for _ in range(bootstrap_samples):
        sample = [paired[rng.randrange(len(paired))] for _ in paired]
        draws.append(math.fsum(value for value, _ in sample) / sum(tokens for _, tokens in sample))
    draws.sort()

    def percentile(probability):
        position = (len(draws) - 1) * probability
        lower = int(position)
        fraction = position - lower
        return draws[lower] * (1 - fraction) + draws[min(lower + 1, len(draws) - 1)] * fraction

    lower, upper = percentile(.025), percentile(.975)
    enough_groups = len(groups) >= MINIMUM_EVALUATION_GROUPS
    point_passed = delta >= .001 and lower > 0 if enough_groups else None
    return {"loss_tokens": count, "self_nll": own / count, "empty_nll": empty / count,
            "delta_empty": delta, "group_count": len(groups), "groups": groups,
            "paired_group_bootstrap": {
                "method": "paired_group_percentile_ratio_of_sums", "confidence": .95,
                "seed": bootstrap_seed, "replicates": bootstrap_samples,
                "ci_lower": lower, "ci_upper": upper},
            "p1_point_check": {
                "status": ("insufficient_groups" if not enough_groups else
                           "passed" if point_passed else "not_met"),
                "passed": point_passed, "minimum_groups": MINIMUM_EVALUATION_GROUPS,
                "delta_threshold": .001, "requires_positive_ci_lower": True},
            "p1_gate_passed": False,
            "p1_gate_status": "not_evaluated_requires_two_milestones_and_second_seed",
            "p6_gate_evaluated": False}


@torch.no_grad()
def evaluate_raw(method, windows, *, prefix_tokens, batch_size,
                 bootstrap_seed=104729, bootstrap_samples=2000, report_per_source=False):
    """Score paired losses by group, with source summaries for mixed corpora."""
    if type(report_per_source) is not bool:
        raise ValueError("report_per_source must be boolean")
    method.eval()
    report_per_source = report_per_source or len({item.source for item in windows}) > 1
    grouped = {}
    for window in windows:
        grouped.setdefault(window.group_id, []).append(window)
    records = []
    for group_id in sorted(grouped):
        items = grouped[group_id]
        sources = {item.source for item in items}
        if report_per_source and len(sources) != 1:
            raise ValueError("per-source evaluation requires one source per group_id")
        own_sum = empty_sum = 0.0
        count = 0
        # Homogeneous group batches give exact group attribution without
        # changing the tested training loss/scoring implementation.
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            own = trajectory_loss(method, batch, prefix_tokens=prefix_tokens)
            empty = trajectory_loss(method, batch, prefix_tokens=prefix_tokens, memory_mode="empty")
            if own.loss_tokens != empty.loss_tokens:
                raise RuntimeError("paired evaluation target counts differ")
            own_sum += float(own.loss_sum)
            empty_sum += float(empty.loss_sum)
            count += own.loss_tokens
        record = {"group_id": group_id, "document_ids": sorted({item.document_id for item in items}),
                  "windows": len(items), "self_loss_sum": own_sum,
                  "empty_loss_sum": empty_sum, "loss_tokens": count}
        if report_per_source:
            record["source"] = next(iter(sources))
        records.append(record)
    summary = summarize_group_losses(records, bootstrap_seed=bootstrap_seed,
                                     bootstrap_samples=bootstrap_samples)
    if report_per_source:
        summary["per_source"] = {
            source: summarize_group_losses([record for record in records if record["source"] == source],
                                            bootstrap_seed=bootstrap_seed,
                                            bootstrap_samples=bootstrap_samples)
            for source in sorted({record["source"] for record in records})}
        summary["source_macro_delta_empty"] = math.fsum(
            result["delta_empty"] for result in summary["per_source"].values()) / len(summary["per_source"])
    return {**summary, "windows": len(windows),
            "scope": "held-out writer-training document groups; no pretraining-novelty claim"}


def train_raw(config, *, monitor_blocks=True, p2_initialization=None):
    """Train raw trajectories, logging detached per-block metrics by default."""
    if type(monitor_blocks) is not bool:
        raise ValueError("monitor_blocks must be boolean")
    raw = config.values["raw"]
    if raw["dataset"] is None:
        raise ValueError("train-raw requires raw.dataset (document JSONL)")
    if config.values["method"]["name"] != "forward":
        raise ValueError("train-raw requires method.name: forward")
    if raw["resume"] and config.values["run"]["checkpoint"]:
        raise ValueError("choose raw.resume or run.checkpoint initialization, not both")
    if p2_initialization is not None and (raw["resume"] or config.values["run"]["checkpoint"]):
        raise ValueError("P2 initialization cannot be combined with resume or normal checkpoint loading")
    if config.values["reader"]["backend"] != "external_hash":
        raise NotImplementedError("train-raw native table integration is not implemented")
    started = time.monotonic()
    method, args = create_runtime(config)
    token_identity = tokenizer_identity(method)
    token_fingerprint = "sha256:" + _json_hash(token_identity)
    documents = load_documents(raw["dataset"], method.tokenizer, tokenizer_identity=token_fingerprint,
                               require_record_identity=True)
    vocabulary_size = method.reader.model.get_input_embeddings().weight.shape[0]
    for document in documents:
        if max(document.token_ids) >= vocabulary_size:
            raise ValueError(f"document {document.id} has token IDs outside the reader vocabulary")
    sampler_options, window_options = {}, {}
    source_lengths = raw.get("source_length_weights")
    if source_lengths is not None:
        sampler_options["source_length_weights"] = source_lengths
        window_options["source_block_counts"] = {
            source: [count for count, weight in weights.items() if weight > 0]
            for source, weights in source_lengths.items()}
    windows = build_windows(documents, raw["block_size"], tuple(raw["block_counts"]), split="train",
                            **window_options)
    sampler = RawWindowSampler(windows, seed=config.values["seed"],
                               length_weights=raw["length_weights"], source_weights=raw["source_weights"],
                               **sampler_options)
    validation_pool = build_windows(documents, raw["block_size"], tuple(raw["block_counts"]), split="validation",
                                    **window_options)
    validation = [window for block_count in raw["block_counts"] for window in validation_pool[block_count]]
    minimum_source_groups = raw.get("eval_min_groups_per_source", 0)
    if source_lengths is not None or minimum_source_groups:
        validation = [window for window in validation if window.source in sampler.source_weights
                      and (source_lengths is None or source_lengths[window.source].get(window.blocks, 0) > 0)]
    if not validation:
        raise ValueError("raw.dataset needs a disjoint validation split with complete windows")
    available_validation_groups = len({window.group_id for window in validation})
    available_source_groups = {
        source: len({window.group_id for window in validation if window.source == source})
        for source in sorted(sampler.source_weights)}
    validation = select_validation_windows(validation, limit=raw["eval_windows"], seed=raw["eval_seed"],
                                           min_groups_per_source=minimum_source_groups,
                                           required_sources=sorted(sampler.source_weights))
    validation_sample = _validation_sample(validation)
    validation_protocol = {
        "name": VALIDATION_PROTOCOL, "selection": "group_first_round_robin",
        "group_key": "group_id", "seed": raw["eval_seed"], "window_limit": raw["eval_windows"],
        "bootstrap_samples": raw["eval_bootstrap_samples"], "minimum_groups": MINIMUM_EVALUATION_GROUPS,
        "evaluation_batching": "within_group", "batch_size": raw["batch_size"],
        "prefix_tokens": raw["prefix_tokens"]}
    if minimum_source_groups:
        validation_protocol.update(selection="source_quota_then_group_first_round_robin",
                                   minimum_groups_per_source=minimum_source_groups,
                                   required_sources=sorted(sampler.source_weights))
    if source_lengths is not None:
        validation_protocol["source_length_weights"] = source_lengths
    validation_metadata = {"validation_protocol": validation_protocol,
                           "validation_protocol_sha256": _json_hash(validation_protocol),
                           "validation_sample_sha256": _json_hash(validation_sample)}
    dataset_sha = hashlib.sha256(Path(raw["dataset"]).read_bytes()).hexdigest()
    identity = _identity(config, method, dataset_sha, token_identity)
    identity["validation"] = {key: validation_metadata[key] for key in
                              ("validation_protocol_sha256", "validation_sample_sha256")}
    optimizer = torch.optim.AdamW(method.parameters(), lr=args.lr, weight_decay=0.0)
    step, counters = 0, {"loss_tokens": 0, "raw_tokens_consumed": 0, "writer_new_tokens": 0,
                         "encoder_input_tokens": 0, "reader_input_tokens": 0,
                         "address_prefix_tokens": 0, "write_blocks": 0, "trajectories": 0}
    if raw["resume"]:
        step, counters = _restore_resume(raw["resume"], method, optimizer, sampler, identity)
    elif config.values["run"]["checkpoint"]:
        restore_writer(method, load_checkpoint(config.values["run"]["checkpoint"]), config)
    p2_provenance = None
    if p2_initialization is not None:
        from .p2_initialization import initialize_p2_writer
        p2_provenance = initialize_p2_writer(method, config, p2_initialization, identity)
    if counters["loss_tokens"] > raw["loss_tokens"]:
        raise ValueError("raw.loss_tokens is below the already-consumed resume budget")
    initialize_output(args)
    metadata = _writer_metadata(config, method, args, identity)
    if p2_provenance is not None:
        metadata["p2_initialization"] = p2_provenance
    # Observability is recorded separately so existing writer/resume identities
    # remain evaluable and monitor toggles do not alter the experiment contract.
    metadata["block_monitor"] = {"enabled": monitor_blocks,
                                 "path": "block_metrics.jsonl" if monitor_blocks else None,
                                 **monitor_source_identity()}
    metadata["data_scope"] = "in-memory document JSONL training; corpus preparation and audit are external"
    metadata.update(validation_metadata)
    save_json(args.out / "config.json", metadata)
    save_json(args.out / "data_manifest.json", {
        "dataset_sha256": dataset_sha, "tokenizer": token_identity, "tokenizer_fingerprint": token_fingerprint,
        "documents": [{"id": d.id, "group_id": d.group_id, "source": d.source, "split": d.split,
                       "tokens": len(d.token_ids), "token_sha256": d.token_sha256,
                       "text_sha256": d.text_sha256, "metadata": d.metadata} for d in documents],
        "unique_raw_tokens": sum(len(d.token_ids) for d in documents),
        "windows_by_blocks": {str(n): len(pool) for n, pool in windows.items()},
        **validation_metadata, "validation_sample": validation_sample,
        "available_validation_groups": available_validation_groups,
        "selected_validation_groups": len({window.group_id for window in validation}),
        **({"available_validation_groups_per_source": available_source_groups,
            "selected_validation_groups_per_source": {
                source: len({window.group_id for window in validation if window.source == source})
                for source in sorted(sampler.source_weights)}}
           if minimum_source_groups or source_lengths is not None else {}),
        "overlap_policy": "nonoverlapping within a length bucket; lengths/epochs can reuse source positions"})
    save_checkpoint(args.out / "writer_init.pt", method.writer, metadata, step, protocol=PROTOCOL)
    log_event(args.out / "metrics.jsonl", {"event": "start", "protocol": PROTOCOL,
                                          "step": step, "counters": counters.copy(),
                                          "block_monitor": metadata["block_monitor"]})
    block_observer = ((lambda record: log_event(args.out / "block_metrics.jsonl", record))
                      if monitor_blocks else None)
    def save_progress(*, snapshot=False):
        # Latest resume is atomic, including when a later evaluation fails.
        _save_resume(args.out / "training.pt", method, optimizer, sampler, identity, step, counters)
        if snapshot:
            current_metadata = {**metadata, "training_loss_tokens": counters["loss_tokens"]}
            save_checkpoint(args.out / "writer.pt", method.writer, current_metadata, step, protocol=PROTOCOL)
            suffix = f"step_{step:06d}_tokens_{counters['loss_tokens']:012d}.pt"
            writer_path, training_path = args.out / f"writer_{suffix}", args.out / f"training_{suffix}"
            # A final evaluation may share a step with a periodic evaluation.
            # Preserve that step's existing immutable snapshot rather than overwrite it.
            if not writer_path.exists():
                save_checkpoint(writer_path, method.writer, current_metadata, step, protocol=PROTOCOL)
            if not training_path.exists():
                _save_resume(training_path, method, optimizer, sampler, identity, step, counters)
        log_event(args.out / "metrics.jsonl", {"event": "checkpoint", "step": step,
                                               "training_loss_tokens": counters["loss_tokens"],
                                               "milestone_snapshot": snapshot})

    def run_evaluation(kind, nominal_loss_tokens):
        result = evaluate_raw(method, validation, prefix_tokens=raw["prefix_tokens"],
                              batch_size=raw["batch_size"], bootstrap_seed=raw["eval_seed"],
                              bootstrap_samples=raw["eval_bootstrap_samples"],
                              report_per_source=bool(minimum_source_groups))
        result.update(validation_metadata)
        result.update(step=step, training_loss_tokens=counters["loss_tokens"],
                      evaluation_kind=kind, nominal_loss_tokens=nominal_loss_tokens,
                      requested_loss_tokens=raw["loss_tokens"])
        save_json(args.out / f"eval_step_{step:06d}.json", result)
        log_event(args.out / "metrics.jsonl", {"event": "evaluation", **result})
        return result

    save_progress()
    last_evaluation = run_evaluation("initial", counters["loss_tokens"])
    next_evaluation = ((counters["loss_tokens"] // raw["eval_every_tokens"]) + 1) * raw["eval_every_tokens"]
    checkpoint_interval = raw["checkpoint_every_tokens"] or raw["eval_every_tokens"]
    next_checkpoint = ((counters["loss_tokens"] // checkpoint_interval) + 1) * checkpoint_interval
    stop_reason = "loss_token_budget"

    while counters["loss_tokens"] < raw["loss_tokens"]:
        if time.monotonic() - started >= args.max_wall_seconds:
            stop_reason = "wall_limit"
            break
        if raw["max_steps"] is not None and step >= raw["max_steps"]:
            stop_reason = "step_limit"
            break
        batch = sampler.next_batch(raw["batch_size"] * raw["gradient_accumulation_steps"],
                                   remaining_loss_tokens=raw["loss_tokens"] - counters["loss_tokens"])
        if not batch:
            stop_reason = "remaining_budget_below_complete_trajectory"
            break
        method.train()
        optimizer.zero_grad(set_to_none=True)
        synchronize(method.device)
        step_started = time.perf_counter()
        # Normalize by the whole optimizer group's targets, including a shorter
        # final microbatch. Backward releases each independent trajectory graph.
        group_tokens = sum(window.loss_tokens for window in batch)
        group_loss_sum = 0.0
        group_counters = dict.fromkeys(counters, 0)
        for start in range(0, len(batch), raw["batch_size"]):
            result = trajectory_loss(method, batch[start:start + raw["batch_size"]],
                                     prefix_tokens=raw["prefix_tokens"],
                                     checkpoint_reader=raw["checkpoint_reader"],
                                     block_observer=block_observer,
                                     block_context={"step": step + 1,
                                                    "microbatch": start // raw["batch_size"],
                                                    "phase": "train", "seed": config.values["seed"]})
            loss = result.loss_sum / group_tokens
            if not loss.requires_grad or not bool(torch.isfinite(loss)):
                raise RuntimeError("raw LM loss is disconnected from writer or non-finite")
            loss.backward()
            group_loss_sum += float(result.loss_sum.detach())
            for key, value in result.counters.items():
                group_counters[key] += value
            del loss, result
        if any(parameter.grad is not None for parameter in method.reader.model.parameters()):
            raise RuntimeError("frozen raw LM reader accumulated parameter gradients")
        grad_norm = torch.nn.utils.clip_grad_norm_(method.parameters(), config.values["train"]["gradient_clip"],
                                                  error_if_nonfinite=True)
        if float(grad_norm) == 0:
            raise RuntimeError("raw LM writer gradient is zero; inspect causal read/write coverage")
        optimizer.step()
        step += 1
        for key, value in group_counters.items():
            counters[key] += value
        synchronize(method.device)
        elapsed = time.perf_counter() - step_started
        log_event(args.out / "metrics.jsonl", {
            "event": "train", "step": step, "nll": group_loss_sum / group_tokens, "batch_loss_tokens": group_tokens,
            "counters": counters.copy(), "grad_norm_before_clip": float(grad_norm),
            "step_seconds": elapsed, "loss_tokens_per_second": group_tokens / max(elapsed, 1e-9),
            "windows": [{"document_id": w.document_id, "start": w.start, "blocks": w.blocks} for w in batch]})
        if counters["loss_tokens"] >= next_evaluation:
            save_progress(snapshot=True)
            last_evaluation = run_evaluation("milestone", next_evaluation)
            next_evaluation = ((counters["loss_tokens"] // raw["eval_every_tokens"]) + 1) * raw["eval_every_tokens"]
            next_checkpoint = ((counters["loss_tokens"] // checkpoint_interval) + 1) * checkpoint_interval
        elif counters["loss_tokens"] >= next_checkpoint:
            save_progress()
            next_checkpoint = ((counters["loss_tokens"] // checkpoint_interval) + 1) * checkpoint_interval

    budget_endpoint = stop_reason in ("loss_token_budget", "remaining_budget_below_complete_trajectory")
    final_nominal_tokens = raw["loss_tokens"] if budget_endpoint else counters["loss_tokens"]
    save_progress(snapshot=True)
    if last_evaluation["step"] != step:
        last_evaluation = run_evaluation("final", final_nominal_tokens)
    last_evaluation.update(is_final=True, budget_endpoint=budget_endpoint,
                           final_nominal_loss_tokens=final_nominal_tokens,
                           unspent_loss_tokens=raw["loss_tokens"] - counters["loss_tokens"])
    save_json(args.out / f"eval_step_{step:06d}.json", last_evaluation)
    log_event(args.out / "metrics.jsonl", {"event": "final_evaluation", **last_evaluation})
    _save_resume(args.out / "training.pt", method, optimizer, sampler, identity, step, counters)
    summary = {"protocol": PROTOCOL, "complete": stop_reason == "loss_token_budget",
               "stop_reason": stop_reason, "completed_steps": step, "counters": counters,
               "requested_loss_tokens": raw["loss_tokens"], "unspent_loss_tokens": raw["loss_tokens"] - counters["loss_tokens"],
               "budget_endpoint": budget_endpoint, "nominal_loss_tokens": final_nominal_tokens,
               "last_evaluation": last_evaluation, "elapsed_seconds": time.monotonic() - started,
               "scope": "CPU plumbing only" if args.smoke else "single-device raw LM writer training",
               "native_adapter": False, "ddp": False, "time_parallel_scan": False,
               "block_monitor": metadata["block_monitor"],
               "checkpoint": "writer.pt", "resume_checkpoint": "training.pt"}
    save_json(args.out / "summary.json", summary)
    log_event(args.out / "metrics.jsonl", {"event": "complete", **summary})
    return summary

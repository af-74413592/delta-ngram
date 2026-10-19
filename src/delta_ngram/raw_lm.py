"""Causal document-block training; streams batch, state updates stay ordered.

The writer and row operators are shared with the established evidence/query
path. Only the supervision schedule changes. No block is written before its
own loss is computed. The first block warms memory without a language loss.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .batching import BatchMemoryState, write_batch
from .checkpoints import load_checkpoint, restore_writer, save_checkpoint
from .data.raw_text import RawWindowSampler, build_windows, load_documents
from .runtime import log_event, save_json, synchronize
from .train import create_runtime, initialize_output

PROTOCOL = "delta_ngram_raw_lm_v1"
RESUME_FORMAT = "delta_ngram_raw_resume_v1"


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
                    memory_mode="self", observer=None):
    """Run full BPTT over complete windows, with one namespace per window.

    ``observer`` is an optional correctness/debug callback. It receives the
    fixed snapshot BEFORE a read or write; the trainer does not retain it.
    Empty-memory scoring leaves the reader and writer parameters intact.
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
        state = write_batch(method.writer, state, hidden, rows, write_mask)
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
    return {"protocol": PROTOCOL, "seed": values["seed"], "dataset_sha256": dataset_sha256,
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


@torch.no_grad()
def evaluate_raw(method, windows, *, prefix_tokens, batch_size):
    method.eval()
    self_sum = empty_sum = 0.0
    count = 0
    for start in range(0, len(windows), batch_size):
        batch = windows[start:start + batch_size]
        own = trajectory_loss(method, batch, prefix_tokens=prefix_tokens)
        empty = trajectory_loss(method, batch, prefix_tokens=prefix_tokens, memory_mode="empty")
        self_sum += float(own.loss_sum)
        empty_sum += float(empty.loss_sum)
        count += own.loss_tokens
    if not count:
        raise ValueError("raw validation needs nonempty complete windows")
    return {"loss_tokens": count, "windows": len(windows), "self_nll": self_sum / count,
            "empty_nll": empty_sum / count, "delta_empty": (empty_sum - self_sum) / count,
            "scope": "held-out writer-training documents; no pretraining-novelty claim",
            "p6_gate_evaluated": False}


def train_raw(config):
    raw = config.values["raw"]
    if raw["dataset"] is None:
        raise ValueError("train-raw requires raw.dataset (document JSONL)")
    if config.values["method"]["name"] != "forward":
        raise ValueError("train-raw requires method.name: forward")
    if raw["resume"] and config.values["run"]["checkpoint"]:
        raise ValueError("choose raw.resume or run.checkpoint initialization, not both")
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
    windows = build_windows(documents, raw["block_size"], tuple(raw["block_counts"]), split="train")
    validation_pool = build_windows(documents, raw["block_size"], tuple(raw["block_counts"]), split="validation")
    validation = [window for block_count in raw["block_counts"] for window in validation_pool[block_count]]
    if not validation:
        raise ValueError("raw.dataset needs a disjoint validation split with complete windows")
    # Fixed, deterministic validation sample; independent of the training RNG.
    validation_rng = random.Random(config.values["seed"] + 104729)
    validation_rng.shuffle(validation)
    validation = validation[:raw["eval_windows"]]
    sampler = RawWindowSampler(windows, seed=config.values["seed"],
                               length_weights=raw["length_weights"], source_weights=raw["source_weights"])
    dataset_sha = hashlib.sha256(Path(raw["dataset"]).read_bytes()).hexdigest()
    identity = _identity(config, method, dataset_sha, token_identity)
    optimizer = torch.optim.AdamW(method.parameters(), lr=args.lr, weight_decay=0.0)
    step, counters = 0, {"loss_tokens": 0, "raw_tokens_consumed": 0, "writer_new_tokens": 0,
                         "encoder_input_tokens": 0, "reader_input_tokens": 0,
                         "address_prefix_tokens": 0, "write_blocks": 0, "trajectories": 0}
    if raw["resume"]:
        step, counters = _restore_resume(raw["resume"], method, optimizer, sampler, identity)
    elif config.values["run"]["checkpoint"]:
        restore_writer(method, load_checkpoint(config.values["run"]["checkpoint"]), config)
    if counters["loss_tokens"] > raw["loss_tokens"]:
        raise ValueError("raw.loss_tokens is below the already-consumed resume budget")
    initialize_output(args)
    metadata = _writer_metadata(config, method, args, identity)
    metadata["data_scope"] = "in-memory document JSONL pilot; not the production 200M-token corpus builder"
    save_json(args.out / "config.json", metadata)
    save_json(args.out / "data_manifest.json", {
        "dataset_sha256": dataset_sha, "tokenizer": token_identity, "tokenizer_fingerprint": token_fingerprint,
        "documents": [{"id": d.id, "group_id": d.group_id, "source": d.source, "split": d.split,
                       "tokens": len(d.token_ids), "token_sha256": d.token_sha256,
                       "text_sha256": d.text_sha256, "metadata": d.metadata} for d in documents],
        "unique_raw_tokens": sum(len(d.token_ids) for d in documents),
        "windows_by_blocks": {str(n): len(pool) for n, pool in windows.items()},
        "overlap_policy": "nonoverlapping within a length bucket; lengths/epochs can reuse source positions"})
    save_checkpoint(args.out / "writer_init.pt", method.writer, metadata, step, protocol=PROTOCOL)
    log_event(args.out / "metrics.jsonl", {"event": "start", "protocol": PROTOCOL,
                                          "step": step, "counters": counters.copy()})
    last_evaluation = evaluate_raw(method, validation, prefix_tokens=raw["prefix_tokens"],
                                   batch_size=raw["batch_size"])
    last_evaluation.update(step=step, training_loss_tokens=counters["loss_tokens"])
    save_json(args.out / f"eval_step_{step:06d}.json", last_evaluation)
    log_event(args.out / "metrics.jsonl", {"event": "evaluation", **last_evaluation})
    next_evaluation = ((counters["loss_tokens"] // raw["eval_every_tokens"]) + 1) * raw["eval_every_tokens"]
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
                                     checkpoint_reader=raw["checkpoint_reader"])
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
            last_evaluation = evaluate_raw(method, validation, prefix_tokens=raw["prefix_tokens"],
                                           batch_size=raw["batch_size"])
            last_evaluation.update(step=step, training_loss_tokens=counters["loss_tokens"])
            save_json(args.out / f"eval_step_{step:06d}.json", last_evaluation)
            log_event(args.out / "metrics.jsonl", {"event": "evaluation", **last_evaluation})
            save_checkpoint(args.out / "writer.pt", method.writer, metadata, step, protocol=PROTOCOL)
            _save_resume(args.out / "training.pt", method, optimizer, sampler, identity, step, counters)
            next_evaluation = ((counters["loss_tokens"] // raw["eval_every_tokens"]) + 1) * raw["eval_every_tokens"]

    if last_evaluation["step"] != step:
        last_evaluation = evaluate_raw(method, validation, prefix_tokens=raw["prefix_tokens"],
                                       batch_size=raw["batch_size"])
        last_evaluation.update(step=step, training_loss_tokens=counters["loss_tokens"])
        save_json(args.out / f"eval_step_{step:06d}.json", last_evaluation)
        log_event(args.out / "metrics.jsonl", {"event": "evaluation", **last_evaluation})
    save_checkpoint(args.out / "writer.pt", method.writer, metadata, step, protocol=PROTOCOL)
    _save_resume(args.out / "training.pt", method, optimizer, sampler, identity, step, counters)
    summary = {"protocol": PROTOCOL, "complete": stop_reason == "loss_token_budget",
               "stop_reason": stop_reason, "completed_steps": step, "counters": counters,
               "requested_loss_tokens": raw["loss_tokens"], "unspent_loss_tokens": raw["loss_tokens"] - counters["loss_tokens"],
               "last_evaluation": last_evaluation, "elapsed_seconds": time.monotonic() - started,
               "scope": "CPU plumbing only" if args.smoke else "single-device raw LM writer training",
               "native_adapter": False, "ddp": False, "time_parallel_scan": False,
               "checkpoint": "writer.pt", "resume_checkpoint": "training.pt"}
    save_json(args.out / "summary.json", summary)
    log_event(args.out / "metrics.jsonl", {"event": "complete", **summary})
    return summary

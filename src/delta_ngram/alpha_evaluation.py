"""Frozen joint writer/alpha evaluation, independent of training snapshots.

Historical trajectories keep the trainer's scoring/write schedule. Recent
continuation queries never write their target article. Revision is a separate
old-answer/new-answer test before and after writing explicit new evidence.
Pruning creates read snapshots and measures serialized size; it does not free
the live lazy state's GPU memory or produce a resumable training checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import random

import torch

from .alpha_memory import (BlockAlpha, LazyDecayState, PRUNING_RMS_THRESHOLDS,
                           make_alpha_policy, prune_alpha_for_read, write_alpha_block_online)
from .batching import BatchMemoryState
from .block_monitor import MONITOR_PROTOCOL, query_hit_summary
from .checkpoints import load_checkpoint, restore_writer
from .data.event_streams import EventTrajectory, event_donor_compatible
from .direction_control import permute_state_for_read
from .event_lm import _chunks, _donor_segments
from .raw_lm import (TrajectoryLoss, _json_hash, _pad_windows, score_block,
                     summarize_group_losses, tokenizer_identity)
from .recent_news_evaluation import (PREFIX, WRITE_BLOCK, _check_parameters, _parameters,
                                    score_later_article, scoring_windows)
from .train import create_runtime


PROTOCOL = "delta_ngram_frozen_alpha_evaluation_v1"
ALPHA_FORMAT = "delta_ngram_alpha_policy_v1"


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _safe_load(data):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(data, map_location="cpu", weights_only=True)


def _canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _freeze(method, alpha):
    method.eval()
    for module in (method.writer, method.reader.model, alpha):
        module.eval().requires_grad_(False)
        module.zero_grad(set_to_none=True)


class FrozenGuard:
    """Exact writer/alpha tensors plus object/version checks on the reader."""

    def __init__(self, method, alpha):
        _freeze(method, alpha)
        self.modules = {"writer": method.writer, "alpha": alpha, "reader": method.reader.model}
        self.parameters = {name: _parameters(module) for name, module in self.modules.items()}
        self.tensors = {name: {key: value.detach().clone() for key, value in module.state_dict().items()}
                        for name, module in self.modules.items() if name != "reader"}

    def verify(self):
        for name, module in self.modules.items():
            _check_parameters(module, self.parameters[name], name)
            if name in self.tensors:
                actual, expected = module.state_dict(), self.tensors[name]
                if actual.keys() != expected.keys() or any(not torch.equal(value, actual[key])
                                                          for key, value in expected.items()):
                    raise RuntimeError(f"{name} state_dict changed during frozen evaluation")


@dataclass(frozen=True)
class JointAlphaRuntime:
    method: object
    alpha: BlockAlpha
    manifest: dict
    files: tuple

    def verify_files(self):
        for path, digest in self.files:
            if file_sha256(path) != digest:
                raise RuntimeError(f"joint alpha evaluation input changed: {path}")


def load_joint_alpha(config, writer_path, alpha_path):
    """Validate both immutable payloads before constructing the model runtime."""
    writer_path, alpha_path = Path(writer_path).resolve(), Path(alpha_path).resolve()
    writer_bytes, alpha_bytes = writer_path.read_bytes(), alpha_path.read_bytes()
    writer_hash, alpha_hash = hashlib.sha256(writer_bytes).hexdigest(), hashlib.sha256(alpha_bytes).hexdigest()
    writer = load_checkpoint(io.BytesIO(writer_bytes))
    payload = _safe_load(io.BytesIO(alpha_bytes))
    metadata = writer["config"]
    if (not isinstance(payload, dict) or payload.get("format") != ALPHA_FORMAT
            or metadata.get("requires_alpha_payload") is not True
            or writer.get("protocol") != "delta_ngram_alpha_lm_v1"):
        raise ValueError("joint evaluation requires a marked alpha writer and alpha-policy payload")
    if payload.get("writer_sha256") != writer_hash:
        raise ValueError("alpha payload is paired with a different writer file")
    identity = payload.get("identity")
    if not isinstance(identity, dict) or _canonical(identity) != _canonical(metadata.get("alpha_identity")):
        raise ValueError("writer/alpha identities disagree")
    if (payload.get("step") != writer["step"] or payload.get("variant") != identity.get("variant")
            or payload.get("policy") != identity.get("alpha")
            or payload.get("online_requires_frozen_writer_and_alpha") is not True):
        raise ValueError("writer/alpha step, variant, or policy identity disagrees")
    config_hash = file_sha256(config.source)
    if identity.get("config_sha256") != config_hash:
        raise ValueError("use the exact frozen training configuration for joint alpha evaluation")
    method, _ = create_runtime(config)
    restore_writer(method, writer, config)
    alpha = make_alpha_policy(payload["variant"], method.reader.width).to(method.device)
    if alpha.metadata() != payload["policy"]:
        raise ValueError("serialized alpha policy differs from the registered variant")
    fixed_log_alpha = alpha.fixed_log_alpha.clone()
    alpha.load_state_dict(payload["state_dict"], strict=True)
    if not all(bool(torch.isfinite(t).all()) for t in alpha.state_dict().values()):
        raise ValueError("alpha payload contains nonfinite parameters")
    if not torch.equal(alpha.fixed_log_alpha, fixed_log_alpha):
        raise ValueError("serialized fixed_log_alpha differs from the registered policy")
    tokenizer = tokenizer_identity(method)
    fingerprint = "sha256:" + _json_hash(tokenizer)
    if (metadata.get("tokenizer_fingerprint") != fingerprint
            or metadata.get("raw_identity", {}).get("tokenizer") != tokenizer):
        raise ValueError("joint checkpoint tokenizer differs from the runtime")
    resolved = getattr(method.reader.model.config, "_commit_hash", None)
    if not config.values["model"]["smoke"] and resolved != config.values["model"]["revision"]:
        raise ValueError("runtime reader commit differs from the pinned configuration")
    _freeze(method, alpha)
    manifest = {"writer_path": str(writer_path), "writer_sha256": writer_hash,
        "alpha_path": str(alpha_path), "alpha_sha256": alpha_hash, "step": writer["step"],
        "training_loss_tokens": metadata.get("training_loss_tokens"), "identity": identity,
        "policy": payload["policy"], "variant": payload["variant"],
        "tokenizer_fingerprint": fingerprint, "raw_identity": metadata.get("raw_identity"),
        "joint_identity_verified": True, "parameters_frozen": True}
    return JointAlphaRuntime(method, alpha, manifest,
        ((writer_path, writer_hash), (alpha_path, alpha_hash), (config.source, config_hash)))


def _memory_snapshot(state):
    names = ("rows", "values", "counts", "precision", "block_clock", "cumulative_log_alpha",
             "row_last_written_block", "row_log_alpha_at_write")
    return {name: None if getattr(state, name) is None else getattr(state, name).clone() for name in names}


def _check_memory(state, before):
    for name, previous in before.items():
        current = getattr(state, name)
        if (current is None) != (previous is None) or (current is not None and not torch.equal(current, previous)):
            raise RuntimeError(f"read-only query changed lazy memory {name}")


class AlphaEvaluationBackend:
    """Explicit lazy-memory adapter for the shared continuation evaluator."""

    def __init__(self, alpha, *, pruning=False, alpha_observer=None):
        if not isinstance(alpha, BlockAlpha) or type(pruning) is not bool:
            raise TypeError("alpha backend requires BlockAlpha and boolean pruning")
        self.alpha, self.pruning, self.alpha_observer = alpha, pruning, alpha_observer
        self.guard = None
        self.applied = []
        self.context = {}

    def start(self, method):
        self.guard, self.applied = FrozenGuard(method, self.alpha), []
        self.context = {}

    def set_context(self, **context):
        self.context = context

    def empty(self, method, batch_size=1):
        return LazyDecayState.empty(batch_size, method.reader.width,
                                   method.reader.buckets * len(method.reader.orders), method.device)

    def _observe(self, row):
        self.applied.extend(row["alpha"])
        if self.alpha_observer is not None:
            self.alpha_observer({**self.context, **row})

    def write_block(self, method, state, hidden, rows, mask, *, observer=None, context=None):
        if observer is not None:
            raise ValueError("alpha historical block monitor is not supported; use alpha_observer")
        def observe(row):
            self._observe({**(context or {}), **row})
        return write_alpha_block_online(method.writer, state, hidden, rows, mask, self.alpha,
                                         alpha_observer=observe)

    @torch.no_grad()
    def write_article(self, method, state, token_ids, *, source_id, observer=None):
        counters = {"articles": 1, "tokens": 0, "blocks": 0, "address_prefix_tokens": 0}
        boundary = max(method.reader.orders) - 1
        for start in range(0, len(token_ids), WRITE_BLOCK):
            block = token_ids[start:start + WRITE_BLOCK]
            ids = torch.tensor([block], dtype=torch.long, device=method.device)
            mask = torch.ones_like(ids)
            hidden, _ = method.reader.encode_evidence(ids, mask)
            address_start = max(0, start - boundary)
            address_ids = torch.tensor([token_ids[address_start:start + len(block)]], dtype=torch.long, device=method.device)
            rows = method.reader.adapter.addresses(address_ids, torch.ones_like(address_ids))[:, start - address_start:]
            if observer is not None:
                observer({"kind": "write", "source_id": source_id, "article_token_start": start,
                          "ids": ids, "rows": rows, "state": state})
            state = self.write_block(method, state, hidden, rows, mask,
                context={"source_id": source_id, "article_token_start": start,
                         "actual_new_tokens": int(mask.sum())})
            counters["tokens"] += len(block)
            counters["blocks"] += 1
            counters["address_prefix_tokens"] += start - address_start
        return state, counters

    @staticmethod
    def permute_read(state, plan):
        # A row's lazy factor is scalar, so the shared channel permutation
        # commutes with decay. The original state remains the writable state.
        index = torch.tensor(plan.indices, dtype=torch.long, device=state.values.device)
        if state.width != plan.width:
            raise ValueError("permutation width differs from alpha memory")
        return state.with_values(state.values.index_select(1, index))

    @torch.no_grad()
    def after_query(self, method, state, tokens, *, source_id, original_score):
        result = {"owner_block_clock": state.block_clock.tolist(),
                  "active_rows": state.rows.numel(), "pruning_measured": self.pruning}
        if self.pruning:
            result["pruning"] = pruning_query_report(method, state, tokens, source_id=source_id,
                                                       original_score=original_score)
        return result

    def finish(self, method):
        if self.guard is None:
            raise RuntimeError("alpha backend has not started")
        self.guard.verify()
        return {"name": "lazy_delta_alpha", "policy": self.alpha.metadata(),
            "applied_blocks": len(self.applied),
            "applied_alpha": {"min": min(self.applied), "max": max(self.applied),
                              "mean": math.fsum(self.applied) / len(self.applied)} if self.applied else None,
            "alpha_state_dict_exact_equal": True, "alpha_parameters_frozen_no_gradients": True,
            "query_advances_clock": False, "decay_applies_to": "delta patches only",
            "untouched_row_semantics": "Stored deltas stay stale; effective read values decay with the owner clock.",
            "pruning_scope": "Read snapshots only; no claim of releasing live online RAM."}


@torch.no_grad()
def pruning_query_report(method, state, tokens, *, source_id, original_score=None):
    before = _memory_snapshot(state)
    original = dict(original_score) if original_score is not None else score_later_article(
        method, state, tokens, source_id=source_id)
    addresses = []
    for start, _, end in scoring_windows(len(tokens)):
        ids = torch.tensor([tokens[start:end]], dtype=torch.long, device=method.device)
        addresses.append(method.reader.adapter.addresses(ids, torch.ones_like(ids)))
    query_rows = torch.cat(addresses, dim=1)
    reports = []
    for threshold in PRUNING_RMS_THRESHOLDS:
        snapshot = prune_alpha_for_read(state, threshold, query_rows)
        score = score_later_article(method, snapshot.state, tokens, source_id=source_id)
        snapshot.report["readout"]["nll_measured"] = True
        reports.append({**snapshot.report, "query_score": score,
            "nll_difference_from_unpruned": score["nll"] - original["nll"],
            "loss_sum_difference_from_unpruned": score["loss_sum"] - original["loss_sum"],
            "residual_scope": "All reader input positions, including the repeated fixed prefix.",
            "online_ram_freed": False})
    _check_memory(state, before)
    return {"thresholds": list(PRUNING_RMS_THRESHOLDS), "reports": reports,
            "original_query_score": original, "live_lazy_state_unchanged": True,
            "size_measurement": "len(torch.save archive bytes), including reported clock/index/count metadata",
            "online_ram_freed": False, "resumable": False}


# These evaluation-only schedules intentionally mirror raw_lm/event_lm.
# Keeping them here leaves both immutable M1 and alpha-training sources alone.
# Unit-alpha parity tests cover masks, counters, donor segments and statistics.
@torch.no_grad()
def _historical_trajectory(method, windows, *, backend, prefix_tokens=128, checkpoint_reader=False,
                    memory_mode="self", observer=None, block_observer=None, block_context=None,
                    donor_windows=None):
    """Score the historical schedule without gradients, one namespace per window.

    ``observer`` is an optional correctness/debug callback. It receives the
    fixed snapshot BEFORE a read or write; the trainer does not retain it.
    Empty-memory scoring leaves the reader and writer parameters intact.
    ``block_observer`` instead receives detached JSON records before each
    scored read and after each write. It runs outside score checkpointing.
    """
    event_flags = [getattr(window, "variable_cost", False) for window in windows]
    if any(event_flags):
        if all(event_flags):
            return _event_historical_trajectory(method, windows, backend=backend, prefix_tokens=prefix_tokens,
                checkpoint_reader=checkpoint_reader, memory_mode=memory_mode, observer=observer,
                block_observer=block_observer, block_context=block_context, donor_trajectories=donor_windows)
        # The types have different supervision masks. Each subgroup batches
        # independent streams; summation preserves exact target accounting.
        if observer is not None:
            raise ValueError("mixed trajectory debug observer requires homogeneous subgroups")
        if donor_windows is not None and len(donor_windows) != len(windows):
            raise ValueError("donor windows must match the target batch size")
        results, states = [], [None] * len(windows)
        for is_event in (False, True):
            owners = [i for i, flag in enumerate(event_flags) if flag == is_event]
            callback = None if block_observer is None else lambda row, owners=owners: block_observer(
                {**row, "owner": owners[row["owner"]]})
            result = _historical_trajectory(method, [windows[i] for i in owners], prefix_tokens=prefix_tokens, backend=backend,
                checkpoint_reader=checkpoint_reader, memory_mode=memory_mode, block_observer=callback,
                block_context=block_context,
                donor_windows=None if donor_windows is None else [donor_windows[i] for i in owners])
            for owner, state in zip(owners, result.state.unpack()):
                states[owner] = state
            results.append(result)
        combined = LazyDecayState.from_states(states)
        diagnostics = [result.first_article_block for result in results if result.first_article_block is not None]
        return TrajectoryLoss(torch.stack([result.loss_sum for result in results]).sum(),
            sum(result.loss_tokens for result in results), combined,
            {key: sum(result.counters[key] for result in results) for key in results[0].counters},
            tuple(loss for result in results for loss in result.block_loss_sums),
            diagnostics[0] if diagnostics else None)
    if method.reader.backend != "external_hash":
        raise NotImplementedError("raw LM currently supports the external hash adapter only")
    if memory_mode not in ("self", "empty", "donor"):
        raise ValueError("memory_mode must be self, empty or donor")
    if (memory_mode == "donor") != (donor_windows is not None):
        raise ValueError("donor mode requires explicit donor_windows and other modes forbid them")
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
    evidence_tokens = tokens
    if donor_windows is not None:
        if len(donor_windows) != len(windows) or any(
                target.group_id == donor.group_id or target.source != donor.source
                or target.blocks != donor.blocks or target.block_size != donor.block_size
                for target, donor in zip(windows, donor_windows)):
            raise ValueError("donors must match source and history length and use different groups")
        evidence_tokens, _ = _pad_windows(donor_windows, method.device, pad)
    capacity = method.reader.buckets * len(method.reader.orders)
    state = backend.empty(method, len(windows))
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
        evidence = evidence_tokens[:, block]
        write_mask = writing[:, None].expand_as(current).to(torch.long)
        if observer is not None:
            observer("write", block, state, evidence, write_mask, None)
        hidden, _ = method.reader.encode_evidence(evidence, write_mask)
        if block and boundary:
            old_tokens = evidence_tokens[:, :block].reshape(len(windows), -1)[:, -boundary:]
            address_ids = torch.cat((old_tokens, evidence), dim=1)
            address_mask = writing[:, None].expand_as(address_ids).to(torch.long)
            rows = method.reader.adapter.addresses(address_ids, address_mask)[:, old_tokens.shape[1]:]
            counters["address_prefix_tokens"] += int(writing.sum().item()) * old_tokens.shape[1]
        else:
            rows = method.reader.adapter.addresses(evidence, write_mask)
        # Old tokens above are addressing-only; they are neither pooled nor
        # written again, and do not enter the evidence encoder a second time.
        backend.set_context(evaluation="historical", memory_mode=memory_mode, block=block,
            owners=[{"owner": i, "document_id": w.document_id, "group_id": w.group_id,
                "source": w.source, "window_start": w.start,
                "evidence_document_id": (donor_windows or windows)[i].document_id,
                "actual_new_tokens": int(write_mask[i].sum())} for i, w in enumerate(windows)])
        if block_observer is None:
            state = backend.write_block(method, state, hidden, rows, write_mask)
        else:
            state = backend.write_block(method, state, hidden, rows, write_mask,
                                observer=lambda records: emit_block("write", block, records, writing))
        encoded = int(write_mask.sum().item())
        counters["writer_new_tokens"] += encoded
        counters["encoder_input_tokens"] += encoded
        counters["write_blocks"] += int(writing.sum().item())

    expected = sum((w.blocks - 1) * w.block_size for w in windows)
    if not losses or counters["loss_tokens"] != expected:
        raise RuntimeError("raw trajectory target accounting mismatch")
    return TrajectoryLoss(torch.stack(losses).sum(), expected, state, counters, tuple(losses))



@torch.no_grad()
def _event_historical_trajectory(method, trajectories, *, backend, prefix_tokens=128, checkpoint_reader=False,
                          memory_mode="self", observer=None, block_observer=None,
                          block_context=None, donor_trajectories=None):
    """One independent namespace per event, retained across its complete articles.

    Donor controls consume an equally sized prefix of an unrelated event. Their
    article boundaries remain local; all read prefixes and labels stay target
    identical. A donor must contain enough real tokens for every target write.
    """
    # Lazy import permits raw_lm to dispatch events through this module.
    from .raw_lm import TrajectoryLoss, score_block
    if not trajectories or any(not isinstance(w, EventTrajectory) for w in trajectories):
        raise ValueError("event trajectory batch must contain event trajectories")
    if method.reader.backend != "external_hash" or not method.supports_outer_training:
        raise ValueError("event LM requires an external-hash learned forward writer")
    if memory_mode not in ("self", "empty", "donor"):
        raise ValueError("memory_mode must be self, empty or donor")
    if donor_trajectories is not None and memory_mode == "self":
        memory_mode = "donor"
    if memory_mode == "donor":
        if donor_trajectories is None or len(donor_trajectories) != len(trajectories):
            raise ValueError("donor trajectories must match target batch size")
        if any(not event_donor_compatible(t, d) for t, d in zip(trajectories, donor_trajectories)):
            raise ValueError("donor must be independent with matching bucket and sufficient real history")
    elif donor_trajectories is not None:
        raise ValueError("donors are only valid for donor memory")
    block_size = trajectories[0].block_size
    if any(t.block_size != block_size for t in trajectories):
        raise ValueError("event trajectories need a common block size")
    if type(prefix_tokens) is not int or not 1 <= prefix_tokens <= block_size:
        raise ValueError("prefix_tokens must lie between 1 and block_size")
    pad = getattr(method.tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(method.tokenizer, "eos_token_id", None)
    bos = getattr(method.tokenizer, "bos_token_id", None)
    if bos is None:
        bos = getattr(method.tokenizer, "eos_token_id", None)
    if type(pad) is not int or pad < 0 or type(bos) is not int or bos < 0:
        raise ValueError("event tokenizer requires pad/EOS and BOS/EOS IDs")
    batch_size = len(trajectories)
    device = method.device
    state = backend.empty(method, batch_size)
    plans = [_chunks(t) for t in trajectories]
    donor_offsets = [0] * batch_size
    losses = []
    first_article_block = ({name: {"loss_sum": 0.0, "loss_tokens": 0, "articles": 0}
                            for name in ("full", "after_prefix")} if not torch.is_grad_enabled() else None)
    counters = {"loss_tokens": 0, "raw_tokens_consumed": sum(t.raw_tokens for t in trajectories),
                "writer_new_tokens": 0, "encoder_input_tokens": 0,
                "reader_input_tokens": 0, "address_prefix_tokens": 0,
                "write_blocks": 0, "trajectories": batch_size}
    boundary = max(method.reader.orders) - 1

    def emit(kind, step, records, eligible):
        for record in records:
            owner = record["owner"]
            if not eligible[owner]:
                continue
            trajectory = trajectories[owner]
            article, offset, _ = plans[owner][step]
            block_observer({**(block_context or {}), "event": "block_metrics", "protocol": MONITOR_PROTOCOL,
                "kind": kind, "owner": owner, "document_id": trajectory.document_id,
                "group_id": trajectory.group_id, "source": trajectory.source,
                "window_start": 0, "window_blocks": trajectory.blocks, "block": step,
                "article_index": article, "article_id": trajectory.articles[article].id,
                "article_token_start": offset, "block_start": sum(len(a.token_ids) for a in trajectory.articles[:article]) + offset,
                "memory_mode": memory_mode, "metrics": record})

    for step in range(max(map(len, plans))):
        current = [plan[step] if step < len(plan) else None for plan in plans]
        reading = [chunk is not None and chunk[0] > 0 for chunk in current]
        if any(reading):
            sequences, prefixes = [], []
            for owner, chunk in enumerate(current):
                if not reading[owner]:
                    sequences.append(())
                    prefixes.append(0)
                    continue
                article, start, tokens = chunk
                local = trajectories[owner].articles[article].token_ids
                prefix = local[max(0, start - prefix_tokens):start] if start else (bos,)
                sequences.append(prefix + tokens)
                prefixes.append(len(prefix))
            size = max(map(len, sequences))
            ids = torch.full((batch_size, size), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            labels = torch.full_like(ids, -100)
            for owner, sequence in enumerate(sequences):
                if sequence:
                    ids[owner, :len(sequence)] = torch.tensor(sequence, device=device)
                    mask[owner, :len(sequence)] = 1
                    labels[owner, prefixes[owner]:len(sequence)] = ids[owner, prefixes[owner]:len(sequence)]
            if observer is not None:
                observer("read", step, state, ids, mask, labels)
            if block_observer is not None:
                addresses = method.reader.adapter.addresses(ids, mask)
                emit("query", step, query_hit_summary(state, addresses, labels, mask), reading)
            first = [reading[owner] and chunk[1] == 0 for owner, chunk in enumerate(current)]
            token_observer = None
            if first_article_block is not None and any(first):
                def token_observer(token_losses, valid):
                    owners = torch.nonzero(valid, as_tuple=True)[0]
                    selected = torch.tensor(first, device=device)[owners]
                    item = first_article_block['full']
                    item['loss_sum'] += float(token_losses[selected].double().sum())
                    item['loss_tokens'] += int(selected.sum())
                    item['articles'] += sum(first)
            loss, count = score_block(method.reader, state, ids, mask, labels,
                                      checkpoint_reader=checkpoint_reader, token_loss_observer=token_observer)
            losses.append(loss)
            counters["loss_tokens"] += count
            counters["reader_input_tokens"] += int(mask.sum().item())
            if first_article_block is not None and any(first):
                # Same first block with a real article prefix, matching the new-event
                # continuation convention. No BOS and no current-article write.
                diagnostic_ids = torch.full((batch_size, block_size), pad, dtype=torch.long, device=device)
                diagnostic_mask = torch.zeros_like(diagnostic_ids)
                diagnostic_labels = torch.full_like(diagnostic_ids, -100)
                diagnostic_articles = 0
                for owner, chunk in enumerate(current):
                    if not first[owner] or len(chunk[2]) <= prefix_tokens:
                        continue
                    tokens = chunk[2]
                    size = len(tokens)
                    diagnostic_ids[owner, :size] = torch.tensor(tokens, device=device)
                    diagnostic_mask[owner, :size] = 1
                    diagnostic_labels[owner, prefix_tokens:size] = diagnostic_ids[owner, prefix_tokens:size]
                    diagnostic_articles += 1
                if diagnostic_articles:
                    value, tokens = score_block(method.reader, state, diagnostic_ids, diagnostic_mask, diagnostic_labels)
                    item = first_article_block['after_prefix']
                    item['loss_sum'] += float(value)
                    item['loss_tokens'] += tokens
                    item['articles'] += diagnostic_articles
                    counters['reader_input_tokens'] += int(diagnostic_mask.sum())

        # A trajectory's final block is never written. All warmup blocks are.
        writing = [step + 1 < len(plan) for plan in plans]
        if memory_mode == "empty" or not any(writing):
            continue
        segments = []
        for owner, chunk in enumerate(current):
            if not writing[owner]:
                segments.append([])
            elif memory_mode == "donor":
                length = len(chunk[2])
                segments.append(_donor_segments(donor_trajectories[owner], donor_offsets[owner], length))
                donor_offsets[owner] += length
            else:
                segments.append([chunk])
        write_sources = donor_trajectories if memory_mode == "donor" else trajectories
        for segment in range(max(map(len, segments))):
            ids = torch.full((batch_size, block_size), pad, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            address_sequences, address_prefixes = [], []
            eligible = [segment < len(s) for s in segments]
            for owner, owner_segments in enumerate(segments):
                if not eligible[owner]:
                    address_sequences.append(())
                    address_prefixes.append(0)
                    continue
                article, start, tokens = owner_segments[segment]
                ids[owner, :len(tokens)] = torch.tensor(tokens, device=device)
                mask[owner, :len(tokens)] = 1
                local = write_sources[owner].articles[article].token_ids
                prefix = local[max(0, start - boundary):start] if boundary else ()
                address_sequences.append(prefix + tokens)
                address_prefixes.append(len(prefix))
            if observer is not None:
                observer("write", step, state, ids, mask, None)
            hidden, _ = method.reader.encode_evidence(ids, mask)
            address_size = max(map(len, address_sequences))
            address_ids = torch.full((batch_size, address_size), pad, dtype=torch.long, device=device)
            address_mask = torch.zeros_like(address_ids)
            for owner, sequence in enumerate(address_sequences):
                if sequence:
                    address_ids[owner, :len(sequence)] = torch.tensor(sequence, device=device)
                    address_mask[owner, :len(sequence)] = 1
            full_rows = method.reader.adapter.addresses(address_ids, address_mask)
            rows = torch.zeros((batch_size, block_size, *full_rows.shape[2:]), dtype=full_rows.dtype, device=device)
            for owner, sequence in enumerate(address_sequences):
                start = address_prefixes[owner]
                count = len(sequence) - start
                if count:
                    rows[owner, :count] = full_rows[owner, start:start + count]
            backend.set_context(evaluation="historical", memory_mode=memory_mode, block=step,
                segment=segment, owners=[{"owner": i, "document_id": t.document_id,
                    "group_id": t.group_id, "source": t.source,
                    "evidence_document_id": write_sources[i].document_id,
                    "evidence_article_id": (write_sources[i].articles[segments[i][segment][0]].id
                                            if eligible[i] else None),
                    "actual_new_tokens": int(mask[i].sum())} for i, t in enumerate(trajectories)])
            if block_observer is None:
                state = backend.write_block(method, state, hidden, rows, mask)
            else:
                state = backend.write_block(method, state, hidden, rows, mask,
                                    observer=lambda records: emit("write", step, records, eligible))
            encoded = int(mask.sum().item())
            counters["writer_new_tokens"] += encoded
            counters["encoder_input_tokens"] += encoded
            counters["address_prefix_tokens"] += sum(address_prefixes)
            counters["write_blocks"] += sum(eligible)
    expected = sum(t.loss_tokens for t in trajectories)
    if not losses or counters["loss_tokens"] != expected:
        raise RuntimeError("event trajectory actual target accounting mismatch")
    if memory_mode != "empty" and counters["writer_new_tokens"] != sum(t.writer_tokens for t in trajectories):
        raise RuntimeError("event trajectory actual write accounting mismatch")
    return TrajectoryLoss(torch.stack(losses).sum(), expected, state, counters, tuple(losses), first_article_block)


@torch.no_grad()
def evaluate_alpha_historical(method, windows, backend, *, prefix_tokens, batch_size,
                 bootstrap_seed=104729, bootstrap_samples=2000, report_per_source=False,
                 report_donor=False, macro_source_groups=None, fixed_donors=None):
    """Score paired losses by group, with source summaries for mixed corpora."""
    if backend.pruning:
        raise ValueError("historical pruning is not measured; use the registered continuation query snapshots")
    if type(report_per_source) is not bool:
        raise ValueError("report_per_source must be boolean")
    if type(report_donor) is not bool:
        raise ValueError("report_donor must be boolean")
    from .raw_evaluation import (select_donors, window_key, donor_manifest,
                                 donor_statistics, macro_statistics)
    if fixed_donors is not None and not report_donor:
        raise ValueError("fixed_donors requires report_donor")
    donors = (select_donors(windows, bootstrap_seed) if fixed_donors is None else fixed_donors) if report_donor else None
    if report_donor and not all(window_key(window) in donors for window in windows):
        raise ValueError("fixed donor map is missing a target")
    backend.start(method)
    method.eval()
    report_per_source = report_per_source or report_donor or len({item.source for item in windows}) > 1
    grouped = {}
    for window in windows:
        grouped.setdefault(window.group_id, []).append(window)
    records = []
    first_records = {name: [] for name in ('full', 'after_prefix')}
    for group_id in sorted(grouped):
        items = grouped[group_id]
        sources = {item.source for item in items}
        if report_per_source and len(sources) != 1:
            raise ValueError("per-source evaluation requires one source per group_id")
        own_sum = empty_sum = donor_sum = 0.0
        count = 0
        first_group = {name: {'group_id': group_id, 'source': next(iter(sources)),
            'self_loss_sum': 0.0, 'empty_loss_sum': 0.0, 'donor_loss_sum': 0.0,
            'loss_tokens': 0, 'articles': 0} for name in first_records}
        # Homogeneous group batches give exact group attribution without
        # changing the tested training loss/scoring implementation.
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            own = _historical_trajectory(method, batch, backend=backend, prefix_tokens=prefix_tokens)
            empty = _historical_trajectory(method, batch, backend=backend, prefix_tokens=prefix_tokens, memory_mode="empty")
            if own.loss_tokens != empty.loss_tokens:
                raise RuntimeError("paired evaluation target counts differ")
            own_sum += float(own.loss_sum)
            empty_sum += float(empty.loss_sum)
            conditions = {'self': own, 'empty': empty}
            if report_donor:
                swapped = _historical_trajectory(method, batch, backend=backend, prefix_tokens=prefix_tokens, memory_mode="donor",
                                          donor_windows=[donors[window_key(item)] for item in batch])
                if swapped.loss_tokens != own.loss_tokens:
                    raise RuntimeError("donor evaluation target counts differ")
                donor_sum += float(swapped.loss_sum)
                conditions['donor'] = swapped
            first = getattr(own, 'first_article_block', None)
            if first is not None:
                for name, base in first.items():
                    first_group[name]['loss_tokens'] += base['loss_tokens']
                    first_group[name]['articles'] += base['articles']
                    for condition, result in conditions.items():
                        diagnostic = result.first_article_block[name]
                        if (diagnostic['loss_tokens'], diagnostic['articles']) != (base['loss_tokens'], base['articles']):
                            raise RuntimeError('first-article paired diagnostic targets differ')
                        first_group[name][condition + '_loss_sum'] += diagnostic['loss_sum']
            count += own.loss_tokens
        record = {"group_id": group_id, "document_ids": sorted({item.document_id for item in items}),
                  "windows": len(items), "self_loss_sum": own_sum,
                  "empty_loss_sum": empty_sum, "loss_tokens": count}
        if report_per_source:
            record["source"] = next(iter(sources))
        if report_donor:
            record["donor_loss_sum"] = donor_sum
        records.append(record)
        for name, item in first_group.items():
            if item['loss_tokens']:
                first_records[name].append(item)
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
    if report_donor:
        summary.update(donor_statistics(records, bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples))
        for source, result in summary["per_source"].items():
            result.update(donor_statistics([record for record in records if record["source"] == source],
                                          bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples))
        mapping = donor_manifest(windows, donors)
        summary.update(donor_sample=mapping, donor_sample_sha256=_json_hash(mapping),
                       evaluation_only=True, optimizer_steps=0, backward_calls=0,
                       conditions=["self", "empty", "donor"])
    if report_donor or macro_source_groups is not None:
        summary.update(macro_statistics(records, macro_source_groups, bootstrap_seed=bootstrap_seed,
                                         bootstrap_samples=bootstrap_samples, donor=report_donor))
    if first_records['full']:
        from .raw_evaluation import first_article_block_protocol
        first_summary = {'protocol': first_article_block_protocol(prefix_tokens)}
        for name, items in first_records.items():
            if not items:
                first_summary[name] = {'group_count': 0, 'loss_tokens': 0, 'articles': 0,
                                       'status': 'no_targets_after_fixed_prefix'}
                continue
            result = summarize_group_losses(items, bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples)
            if report_donor:
                result.update(donor_statistics(items, bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples))
            result.update(articles=sum(item['articles'] for item in items),
                          conditions=['self', 'empty', 'donor'] if report_donor else ['self', 'empty'])
            first_summary[name] = result
        summary['later_article_first_block'] = first_summary
    return {**summary, "windows": len(windows), "memory_backend": backend.finish(method),
            "scope": "held-out writer-training document groups; no pretraining-novelty claim"}


@torch.no_grad()
def score_revision_candidates(method, state, query, candidates, *, observer=None):
    """Use the established answer-token scorer and guard every lazy clock."""
    from .news_eval import score_candidates
    before = _memory_snapshot(state)
    if observer is not None:
        observer({"kind": "revision_read", "query": query, "candidates": candidates, "state": state})
    result = score_candidates(method, state, query + "\nAnswer:", candidates)
    _check_memory(state, before)
    old, new = result["candidates"]
    margin = old["nll"] - new["nll"]
    return {**result, "old_answer_nll": old["nll"], "new_answer_nll": new["nll"],
        "new_preference_margin": margin, "new_answer_selected": margin > 0,
        "old_answer_selected": margin < 0, "tied": margin == 0,
        "owner_block_clock": state.block_clock.tolist(), "query_never_written": True}


def _revision_summary(records, bootstrap_seed, bootstrap_samples):
    from .recent_news_evaluation import DISTANCES, _percentile
    curve = {}
    for distance in DISTANCES:
        selected = [r for r in records if r["unrelated_article_count"] == distance]
        if not selected:
            continue
        families = sorted({r["event_family"] for r in selected})
        if len(families) != len(selected):
            raise ValueError("revision summary requires one pair per independent family")
        metrics = {}
        for row in selected:
            values = {}
            for arm in ("self", "empty", "donor"):
                old = row["stages"]["old_evidence"][arm]
                new = row["stages"]["old_plus_new_evidence"][arm]
                for stage, score in (("old_evidence", old), ("old_plus_new_evidence", new)):
                    for key in ("old_answer_nll", "new_answer_nll", "new_preference_margin",
                                "new_answer_selected", "old_answer_selected", "tied"):
                        values[f"{arm}.{stage}.{key}"] = float(score[key])
                values[arm + ".margin_change"] = new["new_preference_margin"] - old["new_preference_margin"]
                values[arm + ".old_selection_reduction"] = float(old["old_answer_selected"]) - float(new["old_answer_selected"])
            for arm in ("empty", "donor"):
                values["margin_change_over_" + arm] = values["self.margin_change"] - values[arm + ".margin_change"]
            metrics[row["event_family"]] = values
        keys = tuple(next(iter(metrics.values())))
        samples = {key: [] for key in keys}
        rng = random.Random(bootstrap_seed)
        for _ in range(bootstrap_samples):
            draw = [metrics[rng.choice(families)] for _ in families]
            for key in keys:
                samples[key].append(math.fsum(row[key] for row in draw) / len(draw))
        curve[str(distance)] = {"families": len(families), "metrics": {
            key: {"mean": math.fsum(row[key] for row in metrics.values()) / len(families),
                  "paired_family_bootstrap_ci95": [_percentile(sorted(samples[key]), .025),
                                                    _percentile(sorted(samples[key]), .975)]}
            for key in keys}}
    return {"retention_curve": curve, "bootstrap_unit": "independent event family; paired arms and stages",
            "bootstrap_seed": bootstrap_seed, "bootstrap_samples": bootstrap_samples}


@torch.no_grad()
def evaluate_revision(method, prepared, prefix_annotations, backend=None, *, bootstrap_seed=20261019,
                      bootstrap_samples=10000, progress_callback=None, observer=None):
    """Explicit curator queries before/after evidence, separate from continuation.

    Later articles are evidence only in this separate revision task. Neither
    query nor candidate text ever enters a writer. A missing backend selects
    unit alpha, whose D1 equivalence is covered by the historical parity tests.
    """
    from .recent_news_evaluation import DISTANCES
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if prefix_annotations is None:
        raise ValueError("revision evaluation requires frozen annotated queries and answer spans")
    prepared.verify_files()
    prefix_annotations.verify_file()
    selected = []
    for pair in prepared.registry["pairs"]:
        revision = prefix_annotations.by_pair[pair["id"]].get("revision", {"eligible": False})
        if not revision["eligible"]:
            continue
        query = revision.get("revision_query")
        candidates = [revision["earlier_answer"], revision["later_answer"]]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("eligible revision needs a frozen curator-written revision_query")
        if any(not isinstance(c, str) or not c or c != c.strip() for c in candidates) or candidates[0] == candidates[1]:
            raise ValueError("revision candidates must be distinct nonempty curated answers without outer whitespace")
        selected.append((pair, revision, query, candidates))
    is_d1 = backend is None
    backend = backend or AlphaEvaluationBackend(BlockAlpha(method.reader.width, initial_alpha=1).to(method.device))
    backend.start(method)
    records = []
    execution = {"write_calls": 0, "writer_tokens": 0, "candidate_query_calls": 0}
    for pair, revision, query, candidates in selected:
        empty = score_revision_candidates(method, backend.empty(method), query, candidates, observer=observer)
        empty.update(actual_written_articles=0, actual_written_tokens=0, actual_written_blocks=0,
                     address_prefix_tokens=0, write_source_ids=[])
        execution["candidate_query_calls"] += 1
        stages = {distance: {} for distance in DISTANCES}
        for stage, include_new in (("old_evidence", False), ("old_plus_new_evidence", True)):
            for distance in DISTANCES:
                stages[distance][stage] = {"empty": dict(empty)}
            for arm, anchor in (("self", pair["earlier_source_id"]), ("donor", pair["donor_source_id"])):
                backend.set_context(evaluation="revision", pair_id=pair["id"], event_family=pair["event_family"],
                                    stage=stage, memory_mode=arm)
                state = backend.empty(method)
                anchors = [anchor]
                if include_new and arm == "self":
                    anchors.append(pair["later_source_id"])
                totals = dict.fromkeys(("articles", "tokens", "blocks", "address_prefix_tokens"), 0)
                written = []
                for condition in pair["interference_conditions"]:
                    distance = condition["unrelated_article_count"]
                    planned = [*anchors, *condition["distractor_source_ids"]]
                    if planned[:len(written)] != written:
                        raise ValueError("revision distractor schedule must stay nested")
                    for source_id in planned[len(written):]:
                        state, counts = backend.write_article(method, state, prepared.tokens[source_id],
                                                              source_id=source_id, observer=observer)
                        for key, value in counts.items():
                            totals[key] += value
                        written.append(source_id)
                        execution["write_calls"] += counts["blocks"]
                        execution["writer_tokens"] += counts["tokens"]
                    score = score_revision_candidates(method, state, query, candidates, observer=observer)
                    execution["candidate_query_calls"] += 1
                    score.update({"actual_written_" + unit: totals[unit] for unit in ("articles", "tokens", "blocks")})
                    score.update(write_source_ids=written.copy(), address_prefix_tokens=totals["address_prefix_tokens"])
                    stages[distance][stage][arm] = score
        for distance in DISTANCES:
            records.append({"pair_id": pair["id"], "event_family": pair["event_family"],
                "revision_type": revision["type"], "property": revision["property"],
                "query": query, "candidates": candidates, "unrelated_article_count": distance,
                "stages": stages[distance]})
        if progress_callback is not None:
            progress_callback({"stage": "revision_pair_complete", "pair_id": pair["id"],
                "pairs_complete": len(records) // len(DISTANCES), "pairs_total": len(selected)})
    memory = backend.finish(method)
    prepared.verify_files()
    prefix_annotations.verify_file()
    result = {"protocol": "delta_ngram_frozen_news_revision_candidates_v1", "complete": True,
        "evaluation_only": True, "split": "dev", "optimizer_steps": 0, "backward_calls": 0,
        "independent_families": len(selected), "per_family": records, "execution": execution,
        "conditions": ["self", "empty", "donor"], "memory_backend": memory,
        "baseline_unit_alpha_D1": is_d1,
        "annotation_sha256": prefix_annotations.sha256,
        "registry_file_sha256": prepared.manifest["registry_file_sha256"],
        "query_protocol": "Frozen revision_query + newline Answer:; leading-space candidates; mean answer-token NLL, no EOS loss.",
        "write_protocol": "Self reads old evidence, or old then explicit new evidence, followed by the same nested distractors. Both stages reuse the fixed registered independent donor article; no unregistered donor later article is read.",
        "control_scope": "Empty and donor controls are unchanged between stages. Their actual write budgets are reported; they are not matched to the extra self revision article.",
        "revision_causal_limitation": "Without a matched independent donor update, the before/after self change alone does not isolate revision-specific benefit from the effect of an extra evidence write.",
        "interpretation": "Candidate ranking diagnostic, not free generation; positive margin favors new over old; ties reported separately.",
        "integrity_checks": {"query_and_candidates_never_written": True, "lazy_memory_unchanged_during_queries": True,
                             "parameters_frozen_no_gradients": True, "input_files_unchanged": True},
        "status": "evaluated" if selected else "no eligible annotated revision families; no estimate",
        **_revision_summary(records, bootstrap_seed, bootstrap_samples)}
    result["by_revision_type"] = {kind: _revision_summary([r for r in records if r["revision_type"] == kind],
        bootstrap_seed, bootstrap_samples) for kind in sorted({r["revision_type"] for r in records})}
    json.dumps(result, allow_nan=False)
    return result


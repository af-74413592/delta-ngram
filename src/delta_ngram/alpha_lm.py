"""Isolated alpha-ablation trajectories; the M1 path imports none of this.

The causal masks, block batching, article boundaries and target counts mirror
raw_lm/event_lm. Only a NEW evidence write advances the lazy owner clock. This
module is for offline training; online evaluation must freeze writer and alpha.
"""
from __future__ import annotations

import torch

from .alpha_memory import LazyDecayState, write_alpha_block
from .data.event_streams import EventTrajectory
from .event_lm import _chunks
from .raw_lm import TrajectoryLoss, _pad_windows, score_block


def _validate(method, windows, policy, prefix_tokens):
    if not windows or method.reader.backend != "external_hash" or not method.supports_outer_training:
        raise ValueError("alpha LM requires nonempty external-hash learned-writer trajectories")
    if method.writer.operator.name != "D1" or policy.width != method.reader.width:
        raise ValueError("alpha LM requires D1 and a matching alpha width")
    size = windows[0].block_size
    if any(w.block_size != size for w in windows) or type(prefix_tokens) is not int or not 1 <= prefix_tokens <= size:
        raise ValueError("alpha trajectories require common block size and a valid prefix")
    pad = getattr(method.tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(method.tokenizer, "eos_token_id", None)
    if type(pad) is not int or pad < 0:
        raise ValueError("tokenizer requires a pad/EOS token")
    return size, pad


def _empty(method, windows):
    return LazyDecayState.empty(len(windows), method.reader.width,
                               method.reader.buckets * len(method.reader.orders), method.device)


def _counters(windows):
    return {"loss_tokens": 0, "raw_tokens_consumed": sum(
        w.raw_tokens if isinstance(w, EventTrajectory) else len(w.token_ids) for w in windows),
        "writer_new_tokens": 0, "encoder_input_tokens": 0, "reader_input_tokens": 0,
        "address_prefix_tokens": 0, "write_blocks": 0, "trajectories": len(windows)}


def _write(method, policy, state, hidden, rows, mask, block, windows, callback):
    def observe(record):
        callback({**record, "block": block,
                  "document_ids": [windows[i].document_id for i in record["owner_ids"]],
                  "group_ids": [windows[i].group_id for i in record["owner_ids"]],
                  "sources": [windows[i].source for i in record["owner_ids"]]})
    return write_alpha_block(method.writer, state, hidden, rows, mask, policy,
                             alpha_observer=None if callback is None else observe)


def alpha_trajectory_loss(method, windows, policy, *, prefix_tokens=128,
                          checkpoint_reader=False, observer=None, alpha_observer=None):
    """Full BPTT, with raw/event subgrouping identical to the M1 training path."""
    _validate(method, windows, policy, prefix_tokens)
    flags = [isinstance(w, EventTrajectory) for w in windows]
    if all(flags):
        return alpha_event_trajectory_loss(method, windows, policy, prefix_tokens=prefix_tokens,
            checkpoint_reader=checkpoint_reader, observer=observer, alpha_observer=alpha_observer)
    if any(flags):
        if observer is not None:
            raise ValueError("mixed trajectory debug observer requires homogeneous subgroups")
        results, states = [], [None] * len(windows)
        for flag in (False, True):
            owners = [i for i, value in enumerate(flags) if value == flag]
            callback = None if alpha_observer is None else lambda row, owners=owners: alpha_observer(
                {**row, "owner_ids": [owners[i] for i in row["owner_ids"]]})
            result = alpha_trajectory_loss(method, [windows[i] for i in owners], policy,
                prefix_tokens=prefix_tokens, checkpoint_reader=checkpoint_reader, alpha_observer=callback)
            for owner, state in zip(owners, result.state.unpack()):
                states[owner] = state
            results.append(result)
        return TrajectoryLoss(torch.stack([r.loss_sum for r in results]).sum(),
            sum(r.loss_tokens for r in results), LazyDecayState.from_states(states),
            {key: sum(r.counters[key] for r in results) for key in results[0].counters},
            tuple(loss for r in results for loss in r.block_loss_sums))
    size, pad = _validate(method, windows, policy, prefix_tokens)
    tokens, lengths = _pad_windows(windows, method.device, pad)
    state, counters, losses = _empty(method, windows), _counters(windows), []
    boundary = max(method.reader.orders) - 1
    for block in range(tokens.shape[1]):
        current, active = tokens[:, block], lengths > block
        if block:
            ids = torch.cat((tokens[:, block - 1, -prefix_tokens:], current), dim=1)
            mask = active[:, None].expand_as(ids).long()
            labels = torch.full_like(ids, -100)
            labels[:, prefix_tokens:] = torch.where(active[:, None], current, -100)
            if observer is not None:
                observer("read", block, state, ids, mask, labels)
            loss, count = score_block(method.reader, state, ids, mask, labels,
                                      checkpoint_reader=checkpoint_reader)
            losses.append(loss)
            counters["loss_tokens"] += count
            counters["reader_input_tokens"] += int(mask.sum())
        writing = lengths > block + 1
        if not bool(writing.any()):
            continue
        mask = writing[:, None].expand_as(current).long()
        if observer is not None:
            observer("write", block, state, current, mask, None)
        hidden, _ = method.reader.encode_evidence(current, mask)
        if block and boundary:
            prefix = tokens[:, :block].reshape(len(windows), -1)[:, -boundary:]
            ids = torch.cat((prefix, current), dim=1)
            addresses = method.reader.adapter.addresses(ids, writing[:, None].expand_as(ids).long())
            rows = addresses[:, prefix.shape[1]:]
            counters["address_prefix_tokens"] += int(writing.sum()) * prefix.shape[1]
        else:
            rows = method.reader.adapter.addresses(current, mask)
        state = _write(method, policy, state, hidden, rows, mask, block, windows, alpha_observer)
        counters["writer_new_tokens"] += int(mask.sum())
        counters["encoder_input_tokens"] += int(mask.sum())
        counters["write_blocks"] += int(writing.sum())
    expected = sum(w.loss_tokens for w in windows)
    if not losses or counters["loss_tokens"] != expected:
        raise RuntimeError("alpha raw target accounting mismatch")
    return TrajectoryLoss(torch.stack(losses).sum(), expected, state, counters, tuple(losses))


def alpha_event_trajectory_loss(method, trajectories, policy, *, prefix_tokens=128,
                                checkpoint_reader=False, observer=None, alpha_observer=None):
    """Whole first-article warmup; later articles score every real token."""
    size, pad = _validate(method, trajectories, policy, prefix_tokens)
    if any(not isinstance(w, EventTrajectory) for w in trajectories):
        raise ValueError("alpha event batch requires EventTrajectory objects")
    bos = getattr(method.tokenizer, "bos_token_id", None)
    if bos is None:
        bos = getattr(method.tokenizer, "eos_token_id", None)
    if type(bos) is not int or bos < 0:
        raise ValueError("event tokenizer requires BOS/EOS")
    device, batch = method.device, len(trajectories)
    state, counters, losses = _empty(method, trajectories), _counters(trajectories), []
    plans, boundary = [_chunks(w) for w in trajectories], max(method.reader.orders) - 1
    for step in range(max(map(len, plans))):
        current = [p[step] if step < len(p) else None for p in plans]
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
            ids = torch.full((batch, max(map(len, sequences))), pad, dtype=torch.long, device=device)
            mask, labels = torch.zeros_like(ids), torch.full_like(ids, -100)
            for owner, sequence in enumerate(sequences):
                if sequence:
                    ids[owner, :len(sequence)] = torch.tensor(sequence, device=device)
                    mask[owner, :len(sequence)] = 1
                    labels[owner, prefixes[owner]:len(sequence)] = ids[owner, prefixes[owner]:len(sequence)]
            if observer is not None:
                observer("read", step, state, ids, mask, labels)
            loss, count = score_block(method.reader, state, ids, mask, labels,
                                      checkpoint_reader=checkpoint_reader)
            losses.append(loss)
            counters["loss_tokens"] += count
            counters["reader_input_tokens"] += int(mask.sum())
        writing = [step + 1 < len(p) for p in plans]
        if not any(writing):
            continue
        ids = torch.full((batch, size), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        sequences, prefixes = [], []
        for owner, chunk in enumerate(current):
            if not writing[owner]:
                sequences.append(())
                prefixes.append(0)
                continue
            article, start, tokens = chunk
            ids[owner, :len(tokens)] = torch.tensor(tokens, device=device)
            mask[owner, :len(tokens)] = 1
            local = trajectories[owner].articles[article].token_ids
            prefix = local[max(0, start - boundary):start] if boundary else ()
            sequences.append(prefix + tokens)
            prefixes.append(len(prefix))
        if observer is not None:
            observer("write", step, state, ids, mask, None)
        hidden, _ = method.reader.encode_evidence(ids, mask)
        address_ids = torch.full((batch, max(map(len, sequences))), pad, dtype=torch.long, device=device)
        address_mask = torch.zeros_like(address_ids)
        for owner, sequence in enumerate(sequences):
            if sequence:
                address_ids[owner, :len(sequence)] = torch.tensor(sequence, device=device)
                address_mask[owner, :len(sequence)] = 1
        full_rows = method.reader.adapter.addresses(address_ids, address_mask)
        rows = torch.zeros((batch, size, *full_rows.shape[2:]), dtype=full_rows.dtype, device=device)
        for owner, sequence in enumerate(sequences):
            count, start = len(sequence) - prefixes[owner], prefixes[owner]
            if count:
                rows[owner, :count] = full_rows[owner, start:start + count]
        state = _write(method, policy, state, hidden, rows, mask, step, trajectories, alpha_observer)
        counters["writer_new_tokens"] += int(mask.sum())
        counters["encoder_input_tokens"] += int(mask.sum())
        counters["address_prefix_tokens"] += sum(prefixes)
        counters["write_blocks"] += sum(writing)
    expected = sum(w.loss_tokens for w in trajectories)
    if (not losses or counters["loss_tokens"] != expected
            or counters["writer_new_tokens"] != sum(w.writer_tokens for w in trajectories)):
        raise RuntimeError("alpha event target/write accounting mismatch")
    return TrajectoryLoss(torch.stack(losses).sum(), expected, state, counters, tuple(losses))

"""Cross-article BPTT with whole-article warmup and article-local boundaries.

Padding never enters loss, evidence pooling, or row writes. The first token of
each later article is predicted from a local BOS (EOS fallback) and memory.
The preceding article is never used as an LM prefix or n-gram boundary.
"""
from __future__ import annotations

import torch

from .batching import BatchMemoryState, write_batch
from .block_monitor import MONITOR_PROTOCOL, query_hit_summary
from .data.event_streams import EventTrajectory, event_donor_compatible


def _chunks(trajectory):
    return [(article_index, start, article.token_ids[start:start + trajectory.block_size])
            for article_index, article in enumerate(trajectory.articles)
            for start in range(0, len(article.token_ids), trajectory.block_size)]


def _donor_segments(donor, offset, count):
    """Consume exact real tokens, splitting at article and encoder boundaries."""
    result, position = [], 0
    for article_index, article in enumerate(donor.articles):
        end = position + len(article.token_ids)
        if offset < end and count:
            start = max(0, offset - position)
            while start < len(article.token_ids) and count:
                size = min(count, donor.block_size, len(article.token_ids) - start)
                result.append((article_index, start, article.token_ids[start:start + size]))
                start += size
                count -= size
                offset += size
        position = end
    if count:
        raise ValueError("donor has insufficient chronological history tokens")
    return result


def event_trajectory_loss(method, trajectories, *, prefix_tokens=128, checkpoint_reader=False,
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
    state = BatchMemoryState.empty(batch_size, method.reader.width,
                                   method.reader.buckets * len(method.reader.orders), device)
    plans = [_chunks(t) for t in trajectories]
    donor_offsets = [0] * batch_size
    losses = []
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
            loss, count = score_block(method.reader, state, ids, mask, labels,
                                      checkpoint_reader=checkpoint_reader)
            losses.append(loss)
            counters["loss_tokens"] += count
            counters["reader_input_tokens"] += int(mask.sum().item())

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
            if block_observer is None:
                state = write_batch(method.writer, state, hidden, rows, mask)
            else:
                state = write_batch(method.writer, state, hidden, rows, mask,
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
    return TrajectoryLoss(torch.stack(losses).sum(), expected, state, counters, tuple(losses))

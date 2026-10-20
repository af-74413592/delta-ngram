"""Independent query-time namespaces for the isolated four-stage schedule.

The caller retains functional writer snapshots and the complete writer graph.
Different times of ONE trajectory may share a native query batch, but each gets
its own packed owner namespace and its own original bridge ticket. This module
does not update memory, change labels, normalize loss, or step an optimizer.
"""
from __future__ import annotations

from typing import Sequence

import torch

from .alpha_memory import LazyDecayState
from .batching import BatchMemoryState
from .qwen38_train_batching import PackedNativeQueries, collate_native_frames
from .qwen38_training import NativeQueryTicket
from .state import SparseMemoryState


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def collate_staged_frames(
        frames: Sequence[dict | None], *, kind: str, device="cpu") -> dict:
    """Collate any positive B through the unchanged B4 frame validator.

    Every original frame retains its prefix, positions, shifted labels, and
    write mask. All-None batches are allowed for collective-bearing EP calls.
    """
    frames = tuple(frames)
    _require(bool(frames), "a staged batch requires at least one lane")
    chunks = []
    for offset in range(0, len(frames), 4):
        chunk = frames[offset:offset + 4]
        chunks.append((collate_native_frames(
            (*chunk, *((None,) * (4 - len(chunk)))), kind=kind, device=device), len(chunk)))

    def concatenate(key, *, batch=False):
        return torch.cat([
            (item["batch"] if batch else item)[key][:count] for item, count in chunks], dim=0)

    counts = tuple(value for item, count in chunks
                   for value in item["loss_target_tokens_by_lane"][:count])
    inputs = tuple(value for item, count in chunks for value in item["input_tokens_by_lane"][:count])
    return {
        "kind": kind, "frames": frames,
        "batch": {
            "input_ids": concatenate("input_ids", batch=True),
            "labels": concatenate("labels", batch=True) if kind == "query" else None,
            "position_ids": concatenate("position_ids", batch=True), "attention_mask": None,
        },
        "valid_mask": concatenate("valid_mask"),
        "encoder_valid_mask": concatenate("encoder_valid_mask"),
        "write_mask": concatenate("write_mask"),
        "owner_ids": torch.arange(len(frames), dtype=torch.long, device=device),
        "active_mask": concatenate("active_mask"),
        "loss_target_tokens_by_lane": counts, "loss_target_tokens": sum(counts),
        "input_tokens_by_lane": inputs, "real_input_tokens": sum(inputs),
        "padded_input_tokens": len(frames) * 640,
    }


def materialize_staged_snapshot(state: BatchMemoryState | LazyDecayState) -> BatchMemoryState:
    """Expose current effective values while retaining the entire writer graph.

    Save LazyDecayState objects during the chronological writer pass, then call
    this only when a query microbatch is ready. A subsequent functional write
    returns a new state; it must never mutate one of these saved snapshots.
    """
    _require(isinstance(state, (BatchMemoryState, LazyDecayState))
             and state.batch_size == 1 and state.precision is None,
             "staged queries need a one-owner native D1 snapshot")
    if isinstance(state, LazyDecayState):
        return state.memory.with_values(state._lookup_packed(state.rows)[0])
    return state


def pack_staged_query_tickets(
        tickets: Sequence[NativeQueryTicket | None], *, row_capacity: int,
        width: int = 160, device=None) -> PackedNativeQueries:
    """Pack independent query TIMES, allowing several tickets from one bridge.

    A physical row r at lane k is stored at k*(row_capacity+1)+r, regardless of
    the ticket's chronological trajectory owner. The detached packed leaf is
    only a reader input; scatter_cotangent routes each slice back to the exact
    original ticket. Do not use this packed state for future writer writes.
    """
    tickets = tuple(tickets)
    _require(bool(tickets), "a staged query batch requires at least one lane")
    present = [ticket for ticket in tickets if ticket is not None]
    _require(all(isinstance(ticket, NativeQueryTicket) for ticket in present),
             "staged query lanes must be original bridge tickets or None")
    _require(len({id(ticket) for ticket in present}) == len(present),
             "one original query ticket cannot occur twice in the same batch")
    _require(type(row_capacity) is int and row_capacity > 0
             and type(width) is int and width > 0, "invalid staged memory dimensions")
    if device is None:
        device = present[0].leaf_values.device if present else "cpu"
    device = torch.device(device)
    states, spans, cursor = [], [], 0
    for ticket in tickets:
        if ticket is None:
            memory = SparseMemoryState.empty(width, device)
        else:
            state = ticket.reader_state
            _require(isinstance(state, BatchMemoryState) and state.batch_size == 1
                     and state.row_capacity == row_capacity and state.width == width
                     and state.values.device == device and state.values.dtype == torch.float32
                     and state.precision is None and not ticket._recorded,
                     "prepare a matching materialized FP32 snapshot before staged packing")
            ticket._check_immutable()
            memory = state.memory
        states.append(memory)
        spans.append(slice(cursor, cursor + memory.rows.numel()))
        cursor += memory.rows.numel()
    if cursor:
        packed = BatchMemoryState.from_states(states, row_capacity)
        # from_states owns new cat storage. Original snapshot values keep their
        # graph; they are not replaced or detached by this reader-only boundary.
        leaf = packed.values.detach().requires_grad_(True)
        packed = packed.with_values(leaf)
    else:
        # An all-inactive EP rank still executes a graph-connected native
        # backward. The zero sentinel is never mapped to an original ticket.
        sentinel = SparseMemoryState(
            torch.tensor([1], dtype=torch.long, device=device),
            torch.zeros((1, width), dtype=torch.float32, device=device, requires_grad=True),
            torch.zeros(1, dtype=torch.float32, device=device))
        packed = BatchMemoryState(sentinel, len(tickets), row_capacity)
    return PackedNativeQueries(packed, tickets, tuple(spans))


def release_staged_reader_leaf(ticket: NativeQueryTicket) -> int:
    """Drop a recorded disposable leaf while retaining source graph/cotangent.

    Call after scatter_cotangent and bridge.record_query, once the native
    reader backward has completed. The bridge's immutable source-version
    guards, original source_values, and cotangent remain untouched. They are
    still consumed by its later joint backward_writer call.
    """
    _require(isinstance(ticket, NativeQueryTicket) and ticket._recorded
             and not getattr(ticket, "_staged_reader_released", False),
             "release each staged reader leaf once, after recording its query")
    ticket._check_immutable()
    _require(ticket.leaf_values.grad is None and ticket._source_values is not None,
             "release only consumed reader gradients before the writer VJP")
    state = ticket.reader_state
    _require(isinstance(state, BatchMemoryState), "release requires a materialized reader state")
    released_bytes = ticket.leaf_values.numel() * ticket.leaf_values.element_size()
    empty = BatchMemoryState.empty(1, state.width, state.row_capacity, state.values.device)
    ticket.reader_state = empty
    ticket.leaf_values = empty.values
    ticket._staged_reader_released = True
    return released_bytes

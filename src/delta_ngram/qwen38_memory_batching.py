"""Detached read snapshots for the independent memory-encoding experiment.

The evidence frame (including its BOS/prefix convention) is unchanged. Reading
memory covers the whole real input, while writing still covers only new tokens.
None of these snapshots may replace the original differentiable writer state.
"""
from __future__ import annotations

from typing import Sequence

import torch

from .alpha_memory import LazyDecayState
from .batching import BatchMemoryState
from .qwen38_staged_batching import collate_staged_frames, materialize_staged_snapshot
from .state import SparseMemoryState


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@torch.no_grad()
def detached_encoding_snapshot(state: BatchMemoryState | LazyDecayState) -> BatchMemoryState:
    """Own all read-only bytes at the current clock, with no autograd history."""
    materialized = materialize_staged_snapshot(state)
    return BatchMemoryState(SparseMemoryState(
        materialized.rows.detach().clone(), materialized.values.detach().clone(),
        materialized.counts.detach().clone()), 1, materialized.row_capacity)


def collate_memory_encoding_frames(
        frames: Sequence[dict | None], states: Sequence[BatchMemoryState | LazyDecayState | None] | None = None,
        *, row_capacity: int, width: int = 160, device="cpu") -> dict:
    """Collate unlabelled evidence and optional independent pre-write states.

    Passing states selects the memory path, even when every state is empty.
    Omitting states selects the base path with exactly the same token layout.
    """
    frames = tuple(frames)
    item = collate_staged_frames(frames, kind="evidence", device=device)
    item["valid_mask"] = item["encoder_valid_mask"]
    item["encoding_protocol"] = "qwen38_memory_encoding_frames_v1"
    item["encoding_kind"] = "base" if states is None else "memory"
    if states is None:
        return item
    states = tuple(states)
    _require(len(states) == len(frames), "each encoding lane requires its own pre-write state")
    memories = []
    for frame, state in zip(frames, states):
        if frame is None:
            _require(state is None, "inactive encoding lanes cannot own a state")
            memories.append(SparseMemoryState.empty(width, device))
            continue
        _require(isinstance(state, (BatchMemoryState, LazyDecayState)),
                 "active memory-encoding lanes require a pre-write snapshot")
        _require(state.row_capacity == row_capacity and state.width == width
                 and state.rows.device == torch.device(device),
                 "encoding state dimensions or device do not match its lane")
        memories.append(detached_encoding_snapshot(state).memory)
    # cat allocates independent lane storage, including counts/addresses.
    item["reader_state"] = BatchMemoryState.from_states(memories, row_capacity)
    return item

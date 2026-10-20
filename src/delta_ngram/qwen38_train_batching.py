"""Independent native trajectory lanes for true pipeline microbatches.

Sampling and chronological read/write order remain the caller's responsibility.
This module only pads frames, packs already materialized query snapshots, and
routes one checked reader cotangent back to each original trajectory ticket.
It never updates a writer, advances an alpha clock, or changes a loss divisor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .batching import BatchMemoryState
from .qwen38_train_data import PAD
from .qwen38_training import NativeQueryTicket
from .state import SparseMemoryState


SEQUENCE_LENGTH = 640


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def collate_native_frames(
        frames: Sequence[dict | None], *, kind: str, device="cpu",
        sequence_length: int = SEQUENCE_LENGTH) -> dict:
    """Pad B2/B4 independent frames to 640 without inventing targets or writes.

    An absent lane is fully inactive. Query masks include real local prefixes;
    evidence capture masks include only new tokens, matching the existing
    native adapter's evidence/T0-statistic contract. The encoder mask is also
    returned explicitly. The caller skips microbatches with no active lanes.
    """
    frames = tuple(frames)
    _require(len(frames) in (2, 4), "native pipeline microbatches require B2 or B4")
    _require(kind in ("query", "evidence"), "frame kind must be query or evidence")
    _require(sequence_length == SEQUENCE_LENGTH, "native batching uses fixed sequence length 640")
    batch_size = len(frames)
    ids = torch.full((batch_size, sequence_length), PAD, dtype=torch.long, device=device)
    labels = (torch.full_like(ids, -100) if kind == "query" else None)
    encoder_valid = torch.zeros_like(ids, dtype=torch.bool)
    write_mask = torch.zeros_like(encoder_valid)
    positions = torch.arange(sequence_length, device=device).expand(batch_size, -1).clone()
    active = torch.zeros(batch_size, dtype=torch.bool, device=device)
    target_counts, input_counts = [], []
    for lane, frame in enumerate(frames):
        if frame is None:
            target_counts.append(0)
            input_counts.append(0)
            continue
        length = len(frame["input_ids"])
        _require(0 < length <= sequence_length
                 and all(len(frame[name]) == length for name in (
                     "encoder_valid_mask", "write_mask", "position_ids")),
                 "native frame inputs and masks have inconsistent lengths")
        values = torch.tensor(frame["input_ids"], dtype=torch.long, device=device)
        valid = torch.tensor(frame["encoder_valid_mask"], dtype=torch.bool, device=device)
        writes = torch.tensor(frame["write_mask"], dtype=torch.bool, device=device)
        _require(bool(valid.any()) and not bool((writes & ~valid).any()),
                 "an active frame must contain real tokens and cannot write padding")
        ids[lane, :length] = values
        encoder_valid[lane, :length] = valid
        write_mask[lane, :length] = writes
        positions[lane, :length] = torch.tensor(frame["position_ids"], dtype=torch.long, device=device)
        active[lane] = True
        if kind == "query":
            _require("labels" in frame and len(frame["labels"]) == length
                     and not bool(writes.any()), "query frames cannot write memory")
            supplied = torch.tensor(frame["labels"], dtype=torch.long, device=device)
            count = int(supplied.ne(-100).sum())
            _require(count == frame["loss_target_tokens"] and count > 0
                     and not bool((supplied.ne(-100) & ~valid).any()),
                     "native query labels include padding or have a different target count")
            labels[lane, :length] = supplied
        else:
            count = 0
            prefix, new = frame["prefix_tokens"], frame["new_tokens"]
            expected = torch.zeros(length, dtype=torch.bool, device=device)
            expected[prefix:prefix + new] = True
            _require(frame["loss_target_tokens"] == 0 and frame.get("loss_labels") is None
                     and "labels" not in frame and new > 0 and prefix + new <= length
                     and torch.equal(writes, expected),
                     "evidence must write each new real token exactly once")
        target_counts.append(count)
        input_counts.append(int(valid.sum()))
    return {
        "kind": kind, "frames": frames,
        "batch": {"input_ids": ids, "labels": labels, "position_ids": positions, "attention_mask": None},
        "valid_mask": encoder_valid if kind == "query" else write_mask,
        "encoder_valid_mask": encoder_valid, "write_mask": write_mask,
        "owner_ids": torch.arange(batch_size, dtype=torch.long, device=device),
        "active_mask": active,
        "loss_target_tokens_by_lane": tuple(target_counts),
        "loss_target_tokens": sum(target_counts),
        "input_tokens_by_lane": tuple(input_counts),
        "real_input_tokens": sum(input_counts),
        "padded_input_tokens": batch_size * sequence_length,
    }


@dataclass
class PackedNativeQueries:
    """One detached reader leaf with a retained mapping to trajectory tickets."""

    reader_state: BatchMemoryState
    tickets: tuple[NativeQueryTicket | None, ...]
    row_slices: tuple[slice, ...]
    _scattered: bool = False

    @property
    def owner_ids(self):
        return torch.arange(self.reader_state.batch_size, dtype=torch.long,
                            device=self.reader_state.values.device)

    @torch.no_grad()
    def scatter_cotangent(self, gradient: Tensor | None = None) -> None:
        """Install a checked complete MB gradient on the original ticket leaves.

        Pass the canonical TP replica gradient after validating it once for the
        combined leaf. The original NativeTrajectoryGradients objects then use
        record_query(..., reducer=None) with the same whole-group denominator.
        """
        _require(not self._scattered, "this microbatch cotangent was already scattered")
        if gradient is None:
            gradient = self.reader_state.values.grad
        _require(isinstance(gradient, Tensor)
                 and gradient.shape == self.reader_state.values.shape
                 and gradient.dtype == torch.float32
                 and gradient.device == self.reader_state.values.device
                 and not gradient.requires_grad and bool(torch.isfinite(gradient).all()),
                 "microbatch cotangent must be a finite aligned detached FP32 tensor")
        for ticket, span in zip(self.tickets, self.row_slices):
            if ticket is None:
                _require(span.start == span.stop, "absent query lane unexpectedly owns memory rows")
                continue
            ticket._check_immutable()
            _require(not ticket._recorded and ticket.leaf_values.grad is None,
                     "query ticket has already received or consumed a cotangent")
            _require(gradient[span].shape == ticket.leaf_values.shape,
                     "packed row slice differs from its original trajectory ticket")
        for ticket, span in zip(self.tickets, self.row_slices):
            if ticket is not None and ticket.requires_reader_backward:
                ticket.leaf_values.grad = gradient[span].detach().clone()
        self.reader_state.values.grad = None
        self._scattered = True


def pack_query_tickets(
        tickets: Sequence[NativeQueryTicket | None], *, row_capacity: int,
        width: int = 160, device=None) -> PackedNativeQueries:
    """Pack independent one-owner materialized snapshots without joining BPTT.

    The caller retains each original lazy history. A new packed FP32 leaf is
    used only inside the native reader. Its gradient is split by storage-row
    slices, never by token counts or head counts.
    """
    tickets = tuple(tickets)
    _require(len(tickets) in (2, 4), "native query packing requires B2 or B4 lanes")
    present = [ticket for ticket in tickets if ticket is not None]
    _require(bool(present) and all(isinstance(ticket, NativeQueryTicket) for ticket in present),
             "a query microbatch needs at least one original trajectory ticket")
    _require(len({id(ticket) for ticket in present}) == len(present)
             and len({id(ticket._owner) for ticket in present}) == len(present),
             "one microbatch must contain independent trajectories, not successive queries of one owner")
    if device is None:
        device = present[0].leaf_values.device
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
                     "pack only matching one-owner materialized FP32 query tickets")
            ticket._check_immutable()
            memory = state.memory
        states.append(memory)
        spans.append(slice(cursor, cursor + memory.rows.numel()))
        cursor += memory.rows.numel()
    packed = BatchMemoryState.from_states(states, row_capacity)
    # torch.cat in from_states allocated independent storage. Detaching that
    # storage leaves each ticket's source writer graph intact and unconsumed.
    leaf = packed.values.detach().requires_grad_(bool(packed.values.numel()))
    return PackedNativeQueries(packed.with_values(leaf), tickets, tuple(spans))


def crop_evidence_lanes(
        collated: dict, hidden: Tensor, physical_rows: Tensor) -> list[dict | None]:
    """Crop real new tokens from full native H/addresses; inactive lanes stay None.

    Tensors may already be on CPU after the PP/TP collection. Returned slices
    own their storage, so a later pipeline wave cannot overwrite writer inputs.
    This helper performs no memory writes or alpha-clock updates.
    """
    _require(collated.get("kind") == "evidence", "only evidence frames can be cropped for writing")
    frames = collated["frames"]
    shape = (len(frames), SEQUENCE_LENGTH)
    _require(isinstance(hidden, Tensor) and hidden.ndim == 3
             and hidden.shape[:2] == shape and hidden.shape[-1] == 2560
             and isinstance(physical_rows, Tensor)
             and physical_rows.shape == (*shape, 16) and physical_rows.dtype == torch.long,
             "native evidence must contain full [B,640,2560] H and [B,640,16] physical rows")
    result = []
    for lane, frame in enumerate(frames):
        if frame is None:
            result.append(None)
            continue
        start, count = frame["prefix_tokens"], frame["new_tokens"]
        selected = physical_rows[lane:lane + 1, start:start + count]
        _require(count > 0 and selected.shape[1] == count and bool(selected.gt(0).all()),
                 "new evidence tokens need valid physical native rows")
        result.append({
            "frame": frame,
            "H_new": hidden[lane:lane + 1, start:start + count].detach().clone(),
            "physical_rows_new": selected.detach().clone(),
            "loss_labels": None,
        })
    return result

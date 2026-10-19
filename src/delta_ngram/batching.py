"""Independent sparse memories for streams at one logical time step.

This module changes addressing, not the writer or update equations. It must not
be used to merge successive time steps of one stream into a batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor

from .state import SparseMemoryState
from .writer import ForwardWriter


def _validate_namespace(batch_size: int, row_capacity: int) -> None:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(row_capacity) is not int or row_capacity < 1:
        raise ValueError("row_capacity must be a positive integer")
    maximum_key = batch_size * (row_capacity + 1) - 1
    if maximum_key > torch.iinfo(torch.int64).max:
        raise ValueError("packed row namespace would overflow int64")


@dataclass(frozen=True)
class BatchMemoryState:
    """Pack physical row r of stream b as b * (row_capacity + 1) + r.

    Physical padding row zero always remains zero, including for later owners.
    ``rows`` exposes packed storage keys; ``lookup`` accepts physical IDs whose
    first dimension is the batch. Values and optional precision retain their
    autograd history. The precision flags preserve per-stream None semantics;
    placeholder precision entries for such streams are not observable on unpack.
    """

    memory: SparseMemoryState
    batch_size: int
    row_capacity: int
    _precision_owners: tuple[bool, ...] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_namespace(self.batch_size, self.row_capacity)
        if not isinstance(self.memory, SparseMemoryState):
            raise ValueError("memory must be a SparseMemoryState")
        maximum_key = self.batch_size * (self.row_capacity + 1) - 1
        if bool((self.memory.rows > maximum_key).any()):
            raise ValueError("packed rows exceed the batch namespace")
        # B=1 permits row_capacity == int64.max without representing cap + 1
        # in an int64 tensor operation.
        if self.batch_size > 1 and bool(
                (self.memory.rows.remainder(self.row_capacity + 1) == 0).any()):
            raise ValueError("packed state contains an owner padding row")
        flags = self._precision_owners
        if flags is None:
            flags = (self.memory.precision is not None,) * self.batch_size
            object.__setattr__(self, "_precision_owners", flags)
        if (not isinstance(flags, tuple) or len(flags) != self.batch_size
                or any(type(value) is not bool for value in flags)):
            raise ValueError("precision owner flags must match the batch")
        if any(flags) != (self.memory.precision is not None):
            raise ValueError("precision owner flags disagree with packed precision")

    @property
    def width(self) -> int:
        return self.memory.width

    @property
    def rows(self) -> Tensor:
        return self.memory.rows

    @property
    def values(self) -> Tensor:
        return self.memory.values

    @property
    def counts(self) -> Tensor:
        return self.memory.counts

    @property
    def precision(self) -> Tensor | None:
        return self.memory.precision

    @classmethod
    def empty(cls, batch_size: int, width: int, row_capacity: int,
              device: torch.device | str = "cpu") -> BatchMemoryState:
        _validate_namespace(batch_size, row_capacity)
        return cls(SparseMemoryState.empty(width, device), batch_size, row_capacity)

    @classmethod
    def from_states(cls, states: Sequence[SparseMemoryState],
                    row_capacity: int) -> BatchMemoryState:
        states = tuple(states)
        _validate_namespace(len(states), row_capacity)
        if any(not isinstance(state, SparseMemoryState) for state in states):
            raise ValueError("states must contain SparseMemoryState objects")
        width, device = states[0].width, states[0].rows.device
        for state in states:
            if state.width != width or state.rows.device != device:
                raise ValueError("all streams must have the same width and device")
            if bool((state.rows > row_capacity).any()):
                raise ValueError("physical rows exceed row_capacity")
        flags = tuple(state.precision is not None for state in states)
        precision = None
        if any(flags):
            # Missing precision has no fixed prior until an operator is chosen.
            # Store positive placeholders, and restore the operator's prior only
            # when D3 touches a stream that previously had no precision.
            precision = torch.cat([
                state.precision if state.precision is not None
                else torch.ones_like(state.counts) for state in states])
        memory = SparseMemoryState(
            torch.cat([state.rows + owner * (row_capacity + 1)
                       for owner, state in enumerate(states)]),
            torch.cat([state.values for state in states]),
            torch.cat([state.counts for state in states]),
            precision,
        )
        return cls(memory, len(states), row_capacity, flags)

    def _pack_rows(self, rows: Tensor) -> Tensor:
        if not isinstance(rows, Tensor) or rows.dtype not in (torch.int32, torch.int64):
            raise ValueError("physical rows must be int32 or int64 tensors")
        if rows.ndim < 1 or rows.shape[0] != self.batch_size:
            raise ValueError("physical rows must have leading dimension batch_size")
        if rows.device != self.rows.device:
            raise ValueError("physical rows and memory must share a device")
        rows = rows.to(dtype=torch.long)
        if bool(((rows < 0) | (rows > self.row_capacity)).any()):
            raise ValueError("physical rows must be between zero and row_capacity")
        if self.batch_size == 1:
            return rows
        offsets = torch.arange(self.batch_size, device=rows.device, dtype=torch.long)
        offsets = (offsets * (self.row_capacity + 1)).reshape(
            self.batch_size, *([1] * (rows.ndim - 1)))
        return torch.where(rows.ne(0), rows + offsets, 0)

    def lookup(self, rows: Tensor) -> tuple[Tensor, Tensor]:
        """Read physical IDs [B, ...], without allocating missing rows."""
        return self.memory.lookup(self._pack_rows(rows))

    def lookup_precision(self, rows: Tensor, initial: float = 1.0) -> Tensor:
        packed = self._pack_rows(rows)
        precision = self.memory.lookup_precision(packed, initial=initial)
        flags = torch.tensor(self._precision_owners, device=rows.device).reshape(
            self.batch_size, *([1] * (rows.ndim - 1)))
        return torch.where(flags, precision, precision.new_full((), initial))

    def unpack(self) -> tuple[SparseMemoryState, ...]:
        """Recover physical per-stream states without detaching any tensor."""
        result = []
        for owner in range(self.batch_size):
            offset = owner * (self.row_capacity + 1)
            selected = (self.rows > offset) & (self.rows <= offset + self.row_capacity)
            precision = (self.memory.precision[selected]
                         if self._precision_owners[owner] else None)
            result.append(SparseMemoryState(
                self.rows[selected] - offset, self.values[selected],
                self.counts[selected], precision))
        return tuple(result)

    def with_values(self, values: Tensor) -> BatchMemoryState:
        """Replace only values, e.g. for an explicit checkpoint tensor input."""
        if not isinstance(values, Tensor) or values.shape != self.values.shape:
            raise ValueError("replacement values must preserve the packed value shape")
        memory = SparseMemoryState(self.rows, values, self.counts, self.precision)
        return BatchMemoryState(memory, self.batch_size, self.row_capacity,
                                self._precision_owners)

    def _owner_ids(self, packed_rows: Tensor) -> Tensor:
        if self.batch_size == 1:
            return torch.zeros_like(packed_rows)
        return packed_rows.div(self.row_capacity + 1, rounding_mode="floor")


def write_batch(writer: ForwardWriter, state: BatchMemoryState, hidden: Tensor,
                physical_rows: Tensor, mask: Tensor) -> BatchMemoryState:
    """Write once per independent stream at the same logical time step.

    The existing writer computes every gate from the common old packed state.
    The existing operator then aggregates events by isolated physical row. D4
    exclusions are defined in physical space, so they apply to every owner.
    """
    if not isinstance(state, BatchMemoryState):
        raise ValueError("state must be a BatchMemoryState")
    if hidden.ndim != 3 or hidden.shape[0] != state.batch_size:
        raise ValueError("hidden must be [batch_size, T, width]")
    if hidden.shape[-1] != state.width:
        raise ValueError("hidden width differs from memory")
    if physical_rows.shape != (*hidden.shape[:2], len(writer.orders)):
        raise ValueError("physical_rows must be [batch_size, T, K]")
    if mask.shape != hidden.shape[:2]:
        raise ValueError("mask must be [batch_size, T]")
    packed_rows = state._pack_rows(physical_rows)
    operator = writer.operator
    if operator.name == "D4" and any(row > state.row_capacity for row in operator.frozen_rows):
        raise ValueError("D4 frozen physical rows exceed row_capacity")
    events = writer.events(hidden, packed_rows, mask, state.memory)
    event_rows, target, beta = events.rows, events.target, events.beta
    if operator.name == "D4" and operator.frozen_rows:
        valid = mask.bool()[:, :, None] & physical_rows.ne(0)
        physical_events = physical_rows[valid].to(dtype=torch.long)
        frozen = physical_events.new_tensor(operator.frozen_rows)
        allowed = ~torch.isin(physical_events, frozen)
        event_rows, target, beta = event_rows[allowed], target[allowed], beta[allowed]
    if not event_rows.numel():
        return state

    memory = state.memory
    flags = state._precision_owners
    if operator.name == "D3":
        if memory.precision is not None and not all(flags):
            present = torch.tensor(flags, device=state.rows.device)[state._owner_ids(state.rows)]
            precision = torch.where(
                present, memory.precision,
                memory.precision.new_full((), operator.precision_init))
            memory = SparseMemoryState(memory.rows, memory.values, memory.counts, precision)
        touched = torch.bincount(state._owner_ids(event_rows), minlength=state.batch_size).bool().tolist()
        flags = tuple(old or active for old, active in zip(flags, touched))

    updated = operator(memory, event_rows, target, beta)
    return BatchMemoryState(updated, state.batch_size, state.row_capacity, flags)

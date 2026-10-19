"""Isolated lazy forgetting ablations for D1 delta memory.

This module is not imported by the M1 runner and changes no default write path.
Only delta patches are stored here; a pretrained base table T0 is not an input.
Each active evidence block advances its owner's clock before its D1 write.
Queries only call lookup and never advance clocks or modify stored rows.

The four experimental policies share M1's complete-batch prefix through its
first approximately 5M-target checkpoint. The separate alpha_training module
locks that prefix; do not independently truncate a stream at 5M target tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .batching import BatchMemoryState
from .operators import functional_d1
from .state import SparseMemoryState
from .writer import ForwardWriter


ALPHA_VARIANTS = ("fixed_099", "fixed_095", "learned_global_099", "evidence_block_099")
PRUNING_RMS_THRESHOLDS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)


class BlockAlpha(nn.Module):
    """One alpha per owner/block from fixed, global, or evidence-only inputs.

    The evidence head sees a masked mean of detached, unpatched frozen-encoder
    hidden states for NEW tokens in the arrived evidence block. It has width+1
    learned parameters, zero initial weights, and logit(initial_alpha) bias.
    The global policy has one learned parameter. Fixed policies have none.
    Forward execution never updates parameters; offline BPTT may optimize them.
    """

    def __init__(self, width: int, kind: str = "fixed", initial_alpha: float = 0.99):
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("width must be a positive integer")
        if kind not in {"fixed", "global", "evidence"}:
            raise ValueError("alpha kind must be fixed, global, or evidence")
        if (isinstance(initial_alpha, bool) or not math.isfinite(initial_alpha)
                or not 0 < initial_alpha <= 1):
            raise ValueError("initial_alpha must be finite and in (0,1]")
        if kind != "fixed" and initial_alpha == 1:
            raise ValueError("learned alpha requires initial_alpha < 1")
        self.width, self.kind, self.initial_alpha = width, kind, float(initial_alpha)
        self.register_buffer("fixed_log_alpha", torch.tensor(math.log(initial_alpha), dtype=torch.float64))
        self.logit = None
        self.head = None
        self.initial_raw_logit = None
        if kind == "global":
            self.logit = nn.Parameter(torch.tensor(math.log(initial_alpha / (1 - initial_alpha))))
            self.initial_raw_logit = self.logit.detach().item()
        elif kind == "evidence":
            self.head = nn.Linear(width, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, math.log(initial_alpha / (1 - initial_alpha)))
            self.initial_raw_logit = self.head.bias.detach().item()

    def metadata(self) -> dict:
        return {"kind": self.kind, "initial_alpha": self.initial_alpha,
                "initial_raw_logit": self.initial_raw_logit,
                "hidden_width": self.width,
                "learned_parameter_count": sum(p.numel() for p in self.parameters()),
                "features": "masked_mean_detached_unpatched_evidence_hidden"
                if self.kind == "evidence" else "none",
                "clock_unit": "one_active_new_evidence_block_per_owner",
                "decay_timing": "advance_before_D1_write; query_does_not_advance",
                "cumulative_log_dtype": "float64"}

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """Return log(alpha) [B] in FP64 without touching memory or a clock."""
        return self.log_alpha_and_logit(hidden, mask)[0]

    def log_alpha_and_logit(self, hidden: Tensor, mask: Tensor):
        """Return the applied log(alpha) and raw learned logits for audit."""
        if hidden.ndim != 3 or hidden.shape[-1] != self.width or mask.shape != hidden.shape[:2]:
            raise ValueError("hidden/mask must be [B,T,width] and [B,T]")
        if hidden.device != mask.device or hidden.device != self.fixed_log_alpha.device:
            raise ValueError("alpha policy, hidden and mask must share a device")
        if self.kind == "fixed":
            return self.fixed_log_alpha.double().expand(hidden.shape[0]), None
        if self.kind == "global":
            raw = self.logit.expand(hidden.shape[0])
            return F.logsigmoid(raw.double()), raw
        valid = mask.bool()
        features = hidden.detach().to(dtype=self.head.weight.dtype)
        # where also prevents masked non-finite padding from entering the pool.
        features = torch.where(valid[..., None], features, 0.0)
        pooled = features.sum(dim=1) / valid.sum(dim=1).clamp_min(1)[:, None]
        raw = self.head(pooled).squeeze(-1)
        return F.logsigmoid(raw.double()), raw


def make_alpha_policy(variant: str, width: int) -> BlockAlpha:
    """Construct one of the four frozen experiment variants, without a runner."""
    choices = {"fixed_099": ("fixed", 0.99), "fixed_095": ("fixed", 0.95),
               "learned_global_099": ("global", 0.99), "evidence_block_099": ("evidence", 0.99)}
    if variant not in choices:
        raise ValueError(f"variant must be one of {ALPHA_VARIANTS}")
    kind, initial = choices[variant]
    return BlockAlpha(width=width, kind=kind, initial_alpha=initial)


def _alpha_observation(policy, log_alpha, raw_logit, active, clock):
    """Detached JSON data from the actual applied block policy, no extra pass."""
    owners = active.nonzero(as_tuple=True)[0]
    logs = log_alpha[owners].detach().cpu().tolist()
    alphas = [math.exp(value) for value in logs]
    raw = raw_logit[owners].detach().cpu().tolist() if raw_logit is not None else None
    half_lives = []
    for value in logs:
        half = math.log(0.5) / value if value < 0 else None
        half_lives.append(half if half is not None and math.isfinite(half) else None)
    return {"kind": policy.kind, "initial_alpha": policy.initial_alpha,
            "initial_raw_logit": policy.initial_raw_logit,
            "owner_ids": owners.detach().cpu().tolist(),
            "owner_block_clock": clock[owners].detach().cpu().tolist(),
            "log_alpha": logs, "alpha": alphas, "raw_logit": raw,
            "delta_alpha_from_initial": [value - policy.initial_alpha for value in alphas],
            "delta_logit_from_initial": [value - policy.initial_raw_logit for value in raw]
            if raw is not None else None,
            "equivalent_constant_alpha_half_life_blocks": half_lives,
            "unit_alpha_blocks": sum(value == 0 for value in logs),
            "half_life_note": "Equivalent if this alpha remains constant; not the actual future half-life of a varying-alpha stream. Null means infinite or not representable.",
            "interpretation_note": "A value near the .99 initialization alone is not evidence of a learned preference; compare movement, gradients/updates and alpha=1 baseline performance."}


@dataclass(frozen=True)
class LazyDecayState:
    """Stored deltas at their last write, plus per-owner cumulative log clocks.

    Physical row addressing is exactly BatchMemoryState's owner namespace.
    ``values`` are stored, potentially stale deltas; ``lookup`` is the effective
    read interface. All tensors follow SparseMemoryState's immutable convention.
    Counts are touched-block counts and are never decayed.
    """

    memory: BatchMemoryState
    block_clock: Tensor
    cumulative_log_alpha: Tensor
    row_last_written_block: Tensor
    row_log_alpha_at_write: Tensor

    def __post_init__(self):
        if not isinstance(self.memory, BatchMemoryState):
            raise TypeError("memory must be BatchMemoryState")
        b, r, device = self.memory.batch_size, self.memory.rows.numel(), self.memory.rows.device
        for tensor, shape, dtype in (
            (self.block_clock, (b,), torch.long),
            (self.cumulative_log_alpha, (b,), torch.float64),
            (self.row_last_written_block, (r,), torch.long),
            (self.row_log_alpha_at_write, (r,), torch.float64),
        ):
            if not isinstance(tensor, Tensor) or tensor.shape != shape or tensor.dtype != dtype or tensor.device != device:
                raise ValueError("lazy clock shapes, dtypes, or devices disagree")
        if bool((self.block_clock < 0).any()) or bool((self.row_last_written_block < 0).any()):
            raise ValueError("block clocks must be nonnegative")
        if (not bool(torch.isfinite(self.cumulative_log_alpha).all())
                or not bool(torch.isfinite(self.row_log_alpha_at_write).all())
                or bool((self.cumulative_log_alpha > 0).any())
                or bool((self.row_log_alpha_at_write > 0).any())):
            raise ValueError("cumulative log alpha must be finite and nonpositive")
        owners = self.memory._owner_ids(self.memory.rows)
        if bool((self.row_last_written_block > self.block_clock[owners]).any()):
            raise ValueError("a row write cannot be in its owner's future")
        if bool((self.cumulative_log_alpha[owners] > self.row_log_alpha_at_write).any()):
            raise ValueError("cumulative alpha cannot increase after a row write")

    @property
    def width(self):
        return self.memory.width

    @property
    def rows(self):
        return self.memory.rows

    @property
    def values(self):
        return self.memory.values

    @property
    def counts(self):
        return self.memory.counts

    @property
    def precision(self):
        return self.memory.precision

    @property
    def batch_size(self):
        return self.memory.batch_size

    @property
    def row_capacity(self):
        return self.memory.row_capacity

    @classmethod
    def from_batch(cls, memory: BatchMemoryState):
        """Start at time zero from deltas valid NOW; not a resume-file loader."""
        return cls(memory, torch.zeros(memory.batch_size, dtype=torch.long, device=memory.rows.device),
                   torch.zeros(memory.batch_size, dtype=torch.float64, device=memory.rows.device),
                   torch.zeros_like(memory.rows),
                   torch.zeros(memory.rows.numel(), dtype=torch.float64, device=memory.rows.device))

    @classmethod
    def empty(cls, batch_size: int, width: int, row_capacity: int, device="cpu"):
        return cls.from_batch(BatchMemoryState.empty(batch_size, width, row_capacity, device))

    def _lookup_packed(self, packed_rows: Tensor):
        values, counts = self.memory.memory.lookup(packed_rows)
        if not self.rows.numel():
            return values, counts
        flat = packed_rows.reshape(-1).long()
        positions = torch.searchsorted(self.rows, flat)
        safe = positions.clamp_max(self.rows.numel() - 1)
        hit = (flat > 0) & (positions < self.rows.numel()) & (self.rows[safe] == flat)
        owners = self.memory._owner_ids(self.rows[safe])
        log_factor = self.cumulative_log_alpha[owners] - self.row_log_alpha_at_write[safe]
        factor = torch.where(hit, log_factor.exp(), 1.0).to(values.dtype)
        return values * factor.reshape(*packed_rows.shape, 1), counts

    def lookup(self, physical_rows: Tensor):
        """Read effective physical rows [B,...]; never store the decayed view."""
        return self._lookup_packed(self.memory._pack_rows(physical_rows))

    def lookup_precision(self, physical_rows: Tensor, initial: float = 1.0):
        return self.memory.lookup_precision(physical_rows, initial)

    def with_values(self, values: Tensor):
        """Replay a reader snapshot without detaching either lazy-log path."""
        return LazyDecayState(self.memory.with_values(values), self.block_clock,
                              self.cumulative_log_alpha, self.row_last_written_block,
                              self.row_log_alpha_at_write)

    def unpack(self):
        """Independent one-owner lazy states, retaining stored values and graphs."""
        owners = self.memory._owner_ids(self.rows)
        return tuple(LazyDecayState(BatchMemoryState.from_states([memory], self.row_capacity),
                     self.block_clock[i:i + 1], self.cumulative_log_alpha[i:i + 1],
                     self.row_last_written_block[owners == i], self.row_log_alpha_at_write[owners == i])
                     for i, memory in enumerate(self.memory.unpack()))

    @classmethod
    def from_states(cls, states):
        """Combine one-owner lazy snapshots without materializing their decay."""
        if not states or any(s.batch_size != 1 for s in states):
            raise ValueError("from_states requires nonempty one-owner lazy states")
        capacity = states[0].row_capacity
        if any(s.row_capacity != capacity for s in states):
            raise ValueError("lazy states must share row capacity")
        memory = BatchMemoryState.from_states([s.memory.memory for s in states], capacity)
        return cls(memory, torch.cat([s.block_clock for s in states]),
                   torch.cat([s.cumulative_log_alpha for s in states]),
                   torch.cat([s.row_last_written_block for s in states]),
                   torch.cat([s.row_log_alpha_at_write for s in states]))

    def detach(self):
        """Explicit TBPTT boundary, including both cumulative-log graph paths."""
        memory = BatchMemoryState(self.memory.memory.detach(), self.batch_size, self.row_capacity,
                                  self.memory._precision_owners)
        return LazyDecayState(memory, self.block_clock, self.cumulative_log_alpha.detach(),
                              self.row_last_written_block, self.row_log_alpha_at_write.detach())


def write_alpha_block(writer: ForwardWriter, state: LazyDecayState, hidden: Tensor,
                      physical_rows: Tensor, mask: Tensor, policy: BlockAlpha, *,
                      alpha_observer=None) -> LazyDecayState:
    """Advance active owner clocks, decay touched rows, then run unchanged D1.

    Mask only NEW, already-arrived evidence tokens. Padding owners do not tick;
    a valid block with no complete n-gram still ticks. Every event in the block
    sees the same decayed snapshot. Untouched stored rows and last-write clocks
    remain unchanged. This function retains graphs for offline BPTT. An optional
    observer receives detached JSON for the alpha actually applied to active
    owners, once after a successful block; the runner can add step/token IDs and
    post-backward gradient/update metrics without changing this function.
    """
    if not isinstance(state, LazyDecayState) or not isinstance(policy, BlockAlpha):
        raise TypeError("state/policy must be LazyDecayState and BlockAlpha")
    if writer.operator.name != "D1":
        raise ValueError("alpha ablations support the D1 operator only")
    if hidden.ndim != 3 or hidden.shape[0] != state.batch_size or hidden.shape[-1] != state.width:
        raise ValueError("hidden must be [batch_size,T,width]")
    if (mask.shape != hidden.shape[:2]
            or physical_rows.shape != (*hidden.shape[:2], len(writer.orders))):
        raise ValueError("mask or physical row shape differs from evidence")
    if (writer.width != state.width or policy.width != state.width
            or hidden.device != state.rows.device or mask.device != state.rows.device):
        raise ValueError("writer, policy, evidence and state widths/devices must agree")
    packed_rows = state.memory._pack_rows(physical_rows)
    active = mask.bool().any(dim=1)
    if not bool(active.any()):
        return state
    log_alpha, raw_logit = policy.log_alpha_and_logit(hidden, mask)
    if not bool(torch.isfinite(log_alpha).all()) or bool((log_alpha > 0).any()):
        raise ValueError("policy log alpha must be finite and nonpositive")
    clock = state.block_clock + active.long()
    log_prefix = state.cumulative_log_alpha + torch.where(active, log_alpha, 0.0)
    advanced = LazyDecayState(state.memory, clock, log_prefix,
                               state.row_last_written_block, state.row_log_alpha_at_write)
    live = mask.bool()[:, :, None] & packed_rows.ne(0)
    touched = torch.unique(packed_rows[live], sorted=True)
    if not touched.numel():
        if alpha_observer is not None:
            alpha_observer(_alpha_observation(policy, log_alpha, raw_logit, active, clock))
        return advanced

    old_values, old_counts = advanced._lookup_packed(touched)
    precision = state.memory.memory.lookup_precision(touched) if state.precision is not None else None
    snapshot = SparseMemoryState(touched, old_values, old_counts, precision)
    events = writer.events(hidden, packed_rows, mask, snapshot)
    updated = functional_d1(snapshot, events.rows, events.target, events.beta)

    all_rows = torch.unique(torch.cat((state.rows, touched)), sorted=True)
    # Original stored values remain stale for every untouched row: true lazy decay.
    stored_values, counts = state.memory.memory.lookup(all_rows)
    positions = torch.searchsorted(all_rows, touched)
    old_positions = torch.searchsorted(all_rows, state.rows)
    stored_values = stored_values.index_copy(0, positions, updated.values)
    counts = counts.index_copy(0, positions, updated.counts)
    last_written = torch.zeros_like(all_rows).index_copy(0, old_positions, state.row_last_written_block)
    last_log = log_prefix.new_zeros(all_rows.shape).index_copy(0, old_positions, state.row_log_alpha_at_write)
    owners = state.memory._owner_ids(touched)
    last_written = last_written.index_copy(0, positions, clock[owners])
    last_log = last_log.index_copy(0, positions, log_prefix[owners])
    precision = state.memory.memory.lookup_precision(all_rows) if state.precision is not None else None
    memory = BatchMemoryState(SparseMemoryState(all_rows, stored_values, counts, precision),
                              state.batch_size, state.row_capacity, state.memory._precision_owners)
    result = LazyDecayState(memory, clock, log_prefix, last_written, last_log)
    if alpha_observer is not None:
        alpha_observer(_alpha_observation(policy, log_alpha, raw_logit, active, clock))
    return result


@torch.no_grad()
def write_alpha_block_online(writer: ForwardWriter, state: LazyDecayState, hidden: Tensor,
                             physical_rows: Tensor, mask: Tensor, policy: BlockAlpha, *,
                             alpha_observer=None) -> LazyDecayState:
    """Forward-only online entry point. No optimizer or parameter update occurs."""
    return write_alpha_block(writer, state, hidden, physical_rows, mask, policy,
                             alpha_observer=alpha_observer)


@dataclass(frozen=True)
class PrunedReadSnapshot:
    """Materialized read state and audit archive; explicitly NOT a resume state."""

    state: BatchMemoryState
    report: dict
    archive_bytes: bytes


def _bytes(tensor):
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def _cpu(tensor):
    return None if tensor is None else tensor.detach().cpu().contiguous()


def _metadata_cost(state: LazyDecayState, selected: Tensor) -> dict:
    rows = int(selected.sum())
    return {"row_index_bytes": rows * state.rows.element_size(),
            "row_count_bytes": rows * state.counts.element_size(),
            "row_last_written_block_bytes": rows * state.row_last_written_block.element_size(),
            "row_log_alpha_at_write_bytes": rows * state.row_log_alpha_at_write.element_size(),
            "row_precision_bytes": rows * state.precision.element_size() if state.precision is not None else 0,
            "owner_block_clock_bytes": _bytes(state.block_clock),
            "owner_cumulative_log_alpha_bytes": _bytes(state.cumulative_log_alpha),
            "owner_precision_flags_bytes": state.batch_size}


@torch.no_grad()
def prune_alpha_for_read(state: LazyDecayState, threshold: float,
                         query_rows: Tensor | None = None) -> PrunedReadSnapshot:
    """Keep rows with FP64 effective-delta RMS strictly above a fixed threshold.

    The original lazy state is untouched. This export has materialized values
    and may discard counts/clocks of dropped rows, so it cannot resume training.
    Keeping all-row metadata is necessary to preserve count/clock semantics of
    future writes, but is not sufficient for exact resume after nonzero deltas
    or an autograd graph have been discarded.
    """
    if not isinstance(state, LazyDecayState):
        raise TypeError("state must be LazyDecayState")
    if isinstance(threshold, bool) or threshold not in PRUNING_RMS_THRESHOLDS:
        raise ValueError(f"threshold must be one of {PRUNING_RMS_THRESHOLDS}")
    effective, counts = state._lookup_packed(state.rows)
    if not bool(torch.isfinite(effective).all()):
        raise ValueError("effective deltas must be finite for pruning")
    rms = effective.double().square().mean(dim=1).sqrt()
    keep = rms > threshold
    precision = state.precision[keep] if state.precision is not None else None
    read_state = BatchMemoryState(SparseMemoryState(state.rows[keep], effective[keep], counts[keep], precision),
                                  state.batch_size, state.row_capacity, state.memory._precision_owners)
    metadata_bytes = _metadata_cost(state, keep)
    all_metadata_bytes = _metadata_cost(state, torch.ones_like(keep))
    owner_flags = torch.tensor(state.memory._precision_owners, dtype=torch.bool)
    payload = {"format": "delta-ngram-alpha-read-snapshot", "schema_version": 1,
               "resumable": False, "threshold_rms": float(threshold), "rms_dtype": "float64",
               "batch_size": state.batch_size, "row_capacity": state.row_capacity,
               "state": {"rows": _cpu(read_state.rows), "values": _cpu(read_state.values),
                         "counts": _cpu(read_state.counts), "precision": _cpu(read_state.precision)},
               "audit_clock": {"owner_block_clock": _cpu(state.block_clock),
                               "owner_cumulative_log_alpha": _cpu(state.cumulative_log_alpha),
                               "row_last_written_block": _cpu(state.row_last_written_block[keep]),
                               "row_log_alpha_at_write": _cpu(state.row_log_alpha_at_write[keep]),
                               "owner_precision_flags": owner_flags}}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    archive = buffer.getvalue()
    readout = {"measured": False}
    if query_rows is not None:
        if query_rows.ndim != 3 or query_rows.shape[-1] < 1:
            raise ValueError("query_rows must be [B,T,K] with K >= 1")
        scale = math.sqrt(query_rows.shape[-1])
        before = state.lookup(query_rows)[0].sum(dim=-2) / scale
        after = read_state.lookup(query_rows)[0].sum(dim=-2) / scale
        difference = (before - after).double()
        norms = torch.linalg.vector_norm(difference, dim=-1)
        readout = {"measured": True, "positions": norms.numel(),
                   "residual_difference_l2_mean": norms.mean().item() if norms.numel() else 0.0,
                   "residual_difference_l2_max": norms.max().item() if norms.numel() else 0.0,
                   "residual_difference_rms": difference.square().mean().sqrt().item() if difference.numel() else 0.0,
                   "full_residual_l2_mean": torch.linalg.vector_norm(before.double(), dim=-1).mean().item()
                   if norms.numel() else 0.0,
                   "pruned_residual_l2_mean": torch.linalg.vector_norm(after.double(), dim=-1).mean().item()
                   if norms.numel() else 0.0,
                   "nll_measured": False}
    tensor_bytes = _bytes(read_state.values) + sum(metadata_bytes.values())
    report = {"threshold_rms": float(threshold), "comparison": "effective_delta_rms > threshold",
              "rms_dtype": "float64", "original_rows": state.rows.numel(),
              "retained_rows": read_state.rows.numel(), "dropped_rows": int((~keep).sum()),
              "dropped_nonzero_rows": int(((~keep) & (rms > 0)).sum()),
              "effective_delta_rms_mean": rms.mean().item() if rms.numel() else 0.0,
              "effective_delta_rms_max": rms.max().item() if rms.numel() else 0.0,
              "delta_payload_bytes": _bytes(read_state.values),
              "unpruned_delta_payload_bytes": _bytes(effective),
              "metadata_bytes": metadata_bytes, "raw_tensor_bytes": tensor_bytes,
              "serialized_bytes": len(archive), "serialization_overhead_bytes": len(archive) - tensor_bytes,
              "all_row_metadata_for_future_write_semantics": all_metadata_bytes,
              "all_row_metadata_bytes": sum(all_metadata_bytes.values()),
              "resumable": False,
              "resume_limitation": "Read-only export. All-row metadata is necessary, not sufficient: exact resume also needs all nonzero deltas and the relevant autograd history.",
              "readout": readout}
    return PrunedReadSnapshot(read_state, report, archive)

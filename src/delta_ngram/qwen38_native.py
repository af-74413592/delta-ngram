"""CPU-testable Qwen3.8-Flash-Next writer and native PLE boundaries.

The model runtime must supply its own native hash IDs and already-dequantized
PLE lookup. This module never rehashes tokens or substitutes an external reader.
It does not load the full model or install a distributed runtime hook.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor

from .alpha_memory import BlockAlpha, LazyDecayState
from .batching import BatchMemoryState
from .operators import functional_d1
from .state import SparseMemoryState
from .qwen38_writer import NativeForwardWriter


@dataclass(frozen=True)
class Qwen38PLELayout:
    """Pinned single-PLE layout; offsets/sizes must come from native buffers."""

    head_offsets: tuple[int, ...]
    head_vocab_sizes: tuple[int, ...]
    input_dim: int = 2560
    row_width: int = 160
    heads_per_order: int = 8
    orders: tuple[int, ...] = (2, 3)
    hc_count: int = 4
    ple_layer_id: int = 2

    def __post_init__(self):
        if (self.input_dim, self.row_width, self.heads_per_order, self.orders,
                self.hc_count, self.ple_layer_id) != (2560, 160, 8, (2, 3), 4, 2):
            raise ValueError("layout differs from the audited Qwen3.8-Flash-Next text config")
        if len(self.head_offsets) != 16 or len(self.head_vocab_sizes) != 16:
            raise ValueError("native PLE requires 16 head offsets and sizes")
        end = 0
        for offset, size in zip(self.head_offsets, self.head_vocab_sizes):
            if type(offset) is not int or type(size) is not int or offset != end or size < 1:
                raise ValueError("head regions must be positive, contiguous, and order-major")
            end = offset + size

    @classmethod
    def from_native(cls, config: Mapping, offsets: Tensor, sizes: Tensor):
        text = config.get("text_config", config)
        expected = {"hidden_size": 2560, "ple_embed_dim": 2560, "ngram_size": 3,
                    "heads_per_ngram": 8, "hc_count": 4, "ple_layer_ids": [2]}
        if any(text.get(key) != value for key, value in expected.items()):
            raise ValueError("model config does not match the audited native PLE interface")
        if offsets.dtype != torch.long or sizes.dtype != torch.long:
            raise ValueError("native addressing buffers must be int64")
        if offsets.ndim != 1 or sizes.ndim != 1:
            raise ValueError("native addressing buffers must be vectors")
        return cls(tuple(offsets.cpu().tolist()), tuple(sizes.cpu().tolist()))

    @property
    def row_capacity(self):
        return self.head_offsets[-1] + self.head_vocab_sizes[-1]

    def physical_rows(self, native_ids: Tensor, mask: Tensor) -> Tensor:
        """Map native row 0 (valid) to state row 1; only masked tokens become 0."""
        if (native_ids.dtype != torch.long or native_ids.ndim != 3
                or native_ids.shape[-1] != 16 or mask.shape != native_ids.shape[:2]
                or mask.device != native_ids.device):
            raise ValueError("native IDs/mask must be int64 [B,T,16] and [B,T] on one device")
        offsets = native_ids.new_tensor(self.head_offsets)
        ends = offsets + native_ids.new_tensor(self.head_vocab_sizes)
        valid = mask.bool()[..., None]
        if bool((valid & ((native_ids < offsets) | (native_ids >= ends))).any()):
            raise ValueError("native ID lies outside its checkpoint-defined head region")
        return torch.where(valid, native_ids + 1, 0)

    def empty_state(self, batch_size: int, device="cpu") -> BatchMemoryState:
        return BatchMemoryState.from_states(
            [SparseMemoryState.empty(self.row_width, device) for _ in range(batch_size)],
            self.row_capacity,
        )


def make_qwen38_writer(*, alpha: float = 1.0) -> tuple[NativeForwardWriter, BlockAlpha]:
    """The only native decay arms are fixed alpha=1 and fixed alpha=0.99."""
    if isinstance(alpha, bool) or alpha not in (1.0, 0.99):
        raise ValueError("native comparison is restricted to fixed alpha 1 or 0.99")
    writer = NativeForwardWriter(
        width=160, input_dim=2560, hidden_dim=256, orders=(2, 3),
        heads_per_order=8, right_window=4, operator="D1",
        target_scale=torch.ones(16),
    )
    return writer, BlockAlpha(width=2560, kind="fixed", initial_alpha=alpha)


def write_native_block(writer: NativeForwardWriter, state: LazyDecayState, hidden: Tensor,
                       physical_rows: Tensor, mask: Tensor, policy: BlockAlpha) -> LazyDecayState:
    """Native-only fixed-alpha entry; executed M1 write paths remain untouched.

    Advance each active evidence owner's clock, materialize only touched old
    rows at that clock, then call the unchanged functional_d1. Untouched rows
    retain their lazy representation and are decayed when next read.
    """
    if not isinstance(writer, NativeForwardWriter) or not isinstance(state, LazyDecayState):
        raise TypeError("native writes require NativeForwardWriter and LazyDecayState")
    if (writer.operator.name != "D1" or policy.kind != "fixed"
            or policy.initial_alpha not in (1.0, 0.99)):
        raise ValueError("native writes require D1 and fixed alpha 1 or 0.99")
    if state.precision is not None:
        raise ValueError("native D1 does not accept precision-bearing states")
    if (hidden.ndim != 3 or hidden.shape[0] != state.batch_size
            or hidden.shape[-1] != writer.input_dim
            or mask.shape != hidden.shape[:2]
            or physical_rows.shape != (*hidden.shape[:2], writer.num_heads)):
        raise ValueError("native evidence, mask, and head shapes disagree")
    if (writer.width != state.width or policy.width != writer.input_dim
            or hidden.device != state.rows.device or mask.device != state.rows.device):
        raise ValueError("native writer, state, and policy dimensions/devices disagree")
    packed = state.memory._pack_rows(physical_rows)
    active = mask.bool().any(dim=1)
    if not bool(active.any()):
        return state
    log_alpha = policy(hidden, mask)
    if not bool(torch.isfinite(log_alpha).all()) or bool((log_alpha > 0).any()):
        raise ValueError("log alpha must be finite and nonpositive")
    clock = state.block_clock + active.long()
    log_prefix = state.cumulative_log_alpha + torch.where(active, log_alpha, 0.0)
    advanced = LazyDecayState(state.memory, clock, log_prefix,
                              state.row_last_written_block, state.row_log_alpha_at_write)
    touched = torch.unique(packed[mask.bool()[..., None] & packed.ne(0)], sorted=True)
    if not touched.numel():
        return advanced
    old_values, old_counts = advanced._lookup_packed(touched)
    snapshot = SparseMemoryState(touched, old_values, old_counts)
    events = writer.events(hidden, packed, mask, snapshot)
    updated = functional_d1(snapshot, events.rows, events.target, events.beta)
    rows = torch.unique(torch.cat((state.rows, touched)), sorted=True)
    values, counts = state.memory.memory.lookup(rows)
    positions = torch.searchsorted(rows, touched)
    previous_positions = torch.searchsorted(rows, state.rows)
    values = values.index_copy(0, positions, updated.values)
    counts = counts.index_copy(0, positions, updated.counts)
    last_written = torch.zeros_like(rows).index_copy(
        0, previous_positions, state.row_last_written_block)
    last_log = log_prefix.new_zeros(rows.shape).index_copy(
        0, previous_positions, state.row_log_alpha_at_write)
    owners = state.memory._owner_ids(touched)
    last_written = last_written.index_copy(0, positions, clock[owners])
    last_log = last_log.index_copy(0, positions, log_prefix[owners])
    memory = BatchMemoryState(SparseMemoryState(rows, values, counts),
                              state.batch_size, state.row_capacity)
    return LazyDecayState(memory, clock, log_prefix, last_written, last_log)


def add_native_ple_delta(base_lookup: Tensor, native_ids: Tensor, mask: Tensor,
                         state, layout: Qwen38PLELayout) -> Tensor:
    """Add sparse FP32 delta after native lookup, before PLE projections.

    Input/output are [B,T,2560], with 16 concatenated 160-dimensional heads.
    Frozen T0 is detached; gradients through delta are retained for offline
    training. Query calls do not write rows or advance the evidence-block clock.
    The caller passes the result to the unchanged native key/value projections,
    HC gates, normalization, and causal convolution.
    """
    physical = layout.physical_rows(native_ids, mask)
    if (base_lookup.shape != (*native_ids.shape[:2], layout.input_dim)
            or not base_lookup.is_floating_point()
            or base_lookup.device != native_ids.device
            or state.width != layout.row_width
            or state.batch_size != native_ids.shape[0]):
        raise ValueError("native lookup, state, and addressing layout disagree")
    delta, _ = state.lookup(physical)
    return (base_lookup.detach().float() + delta.flatten(-2)).to(base_lookup.dtype)


@torch.no_grad()
def calibrate_qwen38_target_scale(writer: NativeForwardWriter, hidden: Tensor,
                                  rows: Tensor, mask: Tensor, base_head_rms: Tensor,
                                  *, target_to_base_rms: float) -> dict:
    """Set per-head target RMS using arrived evidence and measured T0 RMS only.

    Run before optimization, identically for both alpha arms. T0 statistics
    must use dequantized native rows, including the checkpoint weight_scale.
    This calibrates targets, not the nonlinear downstream PLE response.
    """
    if (writer.input_dim, writer.width, writer.num_heads) != (2560, 160, 16):
        raise ValueError("calibration requires the native 2560/160/16 writer")
    if writer.target_scale is None:
        raise ValueError("writer must have a persistent per-head scale buffer")
    ratio = float(target_to_base_rms)
    if not 0 < ratio <= 1:
        raise ValueError("target_to_base_rms must be finite and in (0,1]")
    rms = torch.as_tensor(base_head_rms, dtype=torch.float32, device=hidden.device)
    if rms.shape != (16,) or not bool(torch.isfinite(rms).all()) or bool((rms <= 0).any()):
        raise ValueError("base_head_rms must contain 16 positive measured values")
    events = writer.events(hidden, rows, mask, SparseMemoryState.empty(160, hidden.device))
    heads = (mask.bool()[..., None] & rows.ne(0)).nonzero(as_tuple=True)[2]
    counts = torch.bincount(heads, minlength=16)
    if bool((counts == 0).any()):
        raise ValueError("calibration evidence must cover every native head")
    squares = torch.zeros_like(rms).index_add(0, heads, events.target.square().sum(-1))
    observed = (squares / (counts * writer.width)).sqrt()
    if not bool(torch.isfinite(observed).all()) or bool((observed <= 0).any()):
        raise ValueError("target calibration has zero or nonfinite RMS")
    writer.target_scale.mul_(ratio * rms / observed)
    return {"target_to_base_rms": ratio, "base_head_rms": rms.cpu().tolist(),
            "previous_target_head_rms": observed.cpu().tolist(),
            "target_scale": writer.target_scale.cpu().tolist(),
            "events_per_head": counts.cpu().tolist(),
            "ple_response_calibrated": False}

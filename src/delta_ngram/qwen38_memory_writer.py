"""Detached memory-encoded writer forks, without changing native reader math."""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .alpha_memory import BlockAlpha, LazyDecayState
from .batching import BatchMemoryState
from .operators import functional_d1
from .qwen38_native import write_native_block
from .qwen38_writer import (
    NativeForwardWriter, NativeWriterFeatures, WriteEventComponents, WriteEvents,
)
from .state import SparseMemoryState


class NativeMemoryResidualWriter(nn.Module):
    """Frozen base on H_base plus a zero-initialized memory-aware residual.

    Both hidden tensors must use the same framing and external normalization.
    Feature construction detaches them, but the base gate keeps its direct
    autograd path through old memory values. The passed base writer is owned
    by this module and frozen in place. Only ``residual_head`` is trainable.

    The default input is [x_base; x_mem - x_base], with the base writer's pooling,
    right-context window, and head embeddings. A shared 128-dimensional hidden
    layer jointly emits width target coordinates and one gate-logit residual.
    For the native 2560/160 layout this has exactly 1,995,169 parameters.

    With ``difference_baseline='matched_base'`` the input is instead
    [x_base; x_mem - x_base_matched]. The frozen base and first input half still
    use the actual legacy H12 base. Only the difference half uses the explicit
    same-batch base encoding. The caller must persist this policy separately
    from the tensor state_dict, whose keys and shapes do not change.
    """

    def __init__(self, base_writer: NativeForwardWriter, hidden_dim: int = 128, *,
                 difference_baseline: str = "actual_base") -> None:
        super().__init__()
        if not isinstance(base_writer, NativeForwardWriter):
            raise TypeError("base_writer must be a NativeForwardWriter")
        if base_writer.operator.name != "D1":
            raise ValueError("memory-aware residual writes require D1")
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if difference_baseline not in ("actual_base", "matched_base"):
            raise ValueError("difference_baseline must be actual_base or matched_base")
        self.difference_baseline = difference_baseline
        self.base_writer = base_writer
        self.base_writer.requires_grad_(False)
        for parameter in self.base_writer.parameters():
            parameter.grad = None
        self.hidden_dim = hidden_dim
        features = base_writer.target_head[0].in_features
        factory = {"device": base_writer.pool_score.weight.device,
                   "dtype": base_writer.pool_score.weight.dtype}
        self.residual_head = nn.Sequential(
            nn.Linear(2 * features, hidden_dim, **factory), nn.SiLU(),
            nn.Linear(hidden_dim, base_writer.width + 1, **factory),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @property
    def width(self):
        return self.base_writer.width

    @property
    def input_dim(self):
        return self.base_writer.input_dim

    @property
    def num_heads(self):
        return self.base_writer.num_heads

    @property
    def heads_per_order(self):
        return self.base_writer.heads_per_order

    @property
    def orders(self):
        return self.base_writer.orders

    @property
    def right_window(self):
        return self.base_writer.right_window

    @property
    def operator(self):
        return self.base_writer.operator

    @property
    def target_scale(self):
        return self.base_writer.target_scale

    def _residual_output(self, base: NativeWriterFeatures,
                         memory: NativeWriterFeatures,
                         difference_base: NativeWriterFeatures) -> Tensor:
        if base.dense is not None:
            inputs = torch.cat((base.dense, memory.dense - difference_base.dense), dim=-1)
            return self.residual_head(inputs)
        # Factor the first affine layer exactly as the native writer does.
        # No [events, 15424] feature tensor is constructed for sixteen heads.
        first = self.residual_head[0]
        evidence_dim = 3 * self.input_dim
        feature_dim = self.base_writer.target_head[0].in_features
        shared = F.linear(base.evidence, first.weight[:, :evidence_dim], first.bias)
        shared = shared + F.linear(
            memory.evidence - difference_base.evidence,
            first.weight[:, feature_dim:feature_dim + evidence_dim])
        per_head = F.linear(base.head_features,
                            first.weight[:, evidence_dim:feature_dim])
        per_head = per_head + F.linear(
            memory.head_features - difference_base.head_features,
            first.weight[:, feature_dim + evidence_dim:])
        batch, token, _ = base.indices.unbind(dim=1)
        return self.residual_head[2](
            self.residual_head[1](shared[batch, token] + per_head))

    def _difference_hidden(self, hidden_base: Tensor, hidden_mem: Tensor,
                           hidden_base_matched: Tensor | None) -> Tensor:
        if hidden_mem.shape != hidden_base.shape or hidden_mem.device != hidden_base.device:
            raise ValueError("base and memory hidden tensors must share shape and device")
        if self.difference_baseline == "matched_base":
            if hidden_base_matched is None:
                raise ValueError("matched_base residual requires hidden_base_matched")
            if (hidden_base_matched.shape != hidden_base.shape
                    or hidden_base_matched.device != hidden_base.device):
                raise ValueError("matched base hidden must share base shape and device")
            return hidden_base_matched
        if hidden_base_matched is not None:
            raise ValueError("hidden_base_matched requires difference_baseline=matched_base")
        return hidden_base

    def event_components(self, hidden_base: Tensor, rows: Tensor, mask: Tensor,
                         state: SparseMemoryState, *,
                         hidden_mem: Tensor,
                         hidden_base_matched: Tensor | None = None) -> WriteEventComponents:
        difference_hidden = self._difference_hidden(hidden_base, hidden_mem, hidden_base_matched)
        # Do not put the frozen base in no_grad: its gate reads differentiable
        # state values, which remain part of the full writer/D1 state chain.
        base = self.base_writer.event_components(hidden_base, rows, mask, state)
        if not base.rows.numel():
            return base
        memory = self.base_writer.event_features(hidden_mem, rows, mask)
        difference_base = (base.features if self.difference_baseline == "actual_base" else
                           self.base_writer.event_features(difference_hidden, rows, mask))
        residual = self._residual_output(base.features, memory, difference_base)
        delta_target = residual[:, :self.width].float()
        if self.target_scale is not None:
            heads = base.features.indices[:, 2]
            delta_target = delta_target * self.target_scale[heads, None].float()
        target = base.target + delta_target
        logits = base.gate_logits + residual[:, self.width]
        return WriteEventComponents(base.rows, target, logits, base.features)

    def events(self, hidden_base: Tensor, rows: Tensor, mask: Tensor,
               state: SparseMemoryState, *, hidden_mem: Tensor,
               hidden_base_matched: Tensor | None = None) -> WriteEvents:
        components = self.event_components(hidden_base, rows, mask, state,
                                           hidden_mem=hidden_mem, hidden_base_matched=hidden_base_matched)
        return WriteEvents(components.rows, components.target,
                           torch.sigmoid(components.gate_logits).float())

    def forward(self, hidden_base: Tensor, rows: Tensor, mask: Tensor,
                state: SparseMemoryState, *, hidden_mem: Tensor,
                hidden_base_matched: Tensor | None = None) -> SparseMemoryState:
        events = self.events(hidden_base, rows, mask, state, hidden_mem=hidden_mem,
                             hidden_base_matched=hidden_base_matched)
        return self.operator(state, events.rows, events.target, events.beta)


def write_memory_aware_native_block(
    writer: NativeForwardWriter | NativeMemoryResidualWriter,
    state: LazyDecayState, hidden_mem: Tensor, physical_rows: Tensor,
    mask: Tensor, policy: BlockAlpha, *, hidden_base: Tensor | None = None,
    hidden_base_matched: Tensor | None = None,
    event_observer: Callable[[WriteEvents], None] | None = None,
) -> LazyDecayState:
    """Write either fork with the native lazy-clock and D1 update semantics.

    Full finetuning uses the unchanged native helper with H_mem. The residual
    fork requires H_base as well and only changes event construction; clocks,
    touched-row materialization, aggregation and untouched-row storage follow
    ``write_native_block`` in the same order. Encoding happens in the caller,
    on the detached state immediately before this write.

    ``event_observer`` sees the final combined events before D1. It must only
    retain detached statistics, rather than storing the live autograd graph.
    """
    if event_observer is not None and not callable(event_observer):
        raise TypeError("event_observer must be callable")
    if isinstance(writer, NativeForwardWriter) and hidden_base_matched is not None:
        raise ValueError("hidden_base_matched is only valid for a matched-base residual writer")
    if isinstance(writer, NativeForwardWriter) and event_observer is None:
        return write_native_block(writer, state, hidden_mem, physical_rows, mask, policy)
    if (not isinstance(writer, (NativeForwardWriter, NativeMemoryResidualWriter))
            or not isinstance(state, LazyDecayState)):
        raise TypeError("memory-aware writes require a native writer and LazyDecayState")
    residual = isinstance(writer, NativeMemoryResidualWriter)
    if residual and hidden_base is None:
        raise ValueError("residual writes require hidden_base")
    if residual:
        writer._difference_hidden(hidden_base, hidden_mem, hidden_base_matched)
    if (writer.operator.name != "D1" or policy.kind != "fixed"
            or policy.initial_alpha not in (1.0, 0.99)):
        raise ValueError("native writes require D1 and fixed alpha 1 or 0.99")
    if state.precision is not None:
        raise ValueError("native D1 does not accept precision-bearing states")
    if (hidden_mem.ndim != 3 or hidden_mem.shape[0] != state.batch_size
            or hidden_mem.shape[-1] != writer.input_dim
            or mask.shape != hidden_mem.shape[:2]
            or physical_rows.shape != (*hidden_mem.shape[:2], writer.num_heads)):
        raise ValueError("native evidence, mask, and head shapes disagree")
    if (writer.width != state.width or policy.width != writer.input_dim
            or hidden_mem.device != state.rows.device or mask.device != state.rows.device):
        raise ValueError("native writer, state, and policy dimensions/devices disagree")
    packed = state.memory._pack_rows(physical_rows)
    active = mask.bool().any(dim=1)
    if not bool(active.any()):
        return state
    log_alpha = policy(hidden_base if residual else hidden_mem, mask)
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
    events = (writer.events(hidden_base, packed, mask, snapshot, hidden_mem=hidden_mem,
                            hidden_base_matched=hidden_base_matched)
              if residual else writer.events(hidden_mem, packed, mask, snapshot))
    if event_observer is not None:
        event_observer(events)
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

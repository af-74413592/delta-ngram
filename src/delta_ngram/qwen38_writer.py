"""Isolated native writer prototype; the executed M1 writer stays unchanged."""
from __future__ import annotations

import math
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .state import SparseMemoryState
from .operators import WriteOperator


@dataclass(frozen=True)
class WriteEvents:
    """Actual evidence events before aggregation or D4 physical-row exclusion."""

    rows: Tensor
    target: Tensor
    beta: Tensor

    @property
    def targets(self):
        return self.target


@dataclass(frozen=True)
class NativeWriterFeatures:
    """The original event features, without expanding shared tokens per head.

    ``dense`` is populated only for the legacy single-head execution. The
    multi-head execution keeps ``evidence`` [B,T,3*input_dim] separate from
    ``head_features`` [events,head_dim]. ``as_tensor`` is for small diagnostics;
    training should project these two pieces before gathering event rows.
    """

    rows: Tensor
    indices: Tensor
    head_features: Tensor
    evidence: Tensor | None
    dense: Tensor | None

    def as_tensor(self) -> Tensor:
        if self.dense is not None:
            return self.dense
        batch, token, _ = self.indices.unbind(dim=1)
        return torch.cat((self.evidence[batch, token], self.head_features), dim=-1)


@dataclass(frozen=True)
class WriteEventComponents:
    """Original scaled targets and pre-sigmoid gate logits for composition."""

    rows: Tensor
    target: Tensor
    gate_logits: Tensor
    features: NativeWriterFeatures


class NativeForwardWriter(nn.Module):
    """Map a complete, already-arrived evidence block to D1 write events.

    Hidden features are constant LM-head inputs from the unpatched backbone.
    The attention pool and right-context mean only see this evidence tensor;
    future questions, answers, and audit labels are absent from the interface.
    """

    def __init__(self, width: int, hidden_dim: int = 128,
                 orders: tuple[int, ...] = (2, 3), right_window: int = 4, *,
                 operator: WriteOperator | str = "D1", input_dim: int | None = None,
                 heads_per_order: int = 1, target_scale: Tensor | None = None) -> None:
        super().__init__()
        input_dim = width if input_dim is None else input_dim
        if width < 1 or hidden_dim < 1 or right_window < 0:
            raise ValueError("invalid writer dimensions")
        if type(input_dim) is not int or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if type(heads_per_order) is not int or heads_per_order < 1:
            raise ValueError("heads_per_order must be a positive integer")
        if not orders or any(order < 1 for order in orders):
            raise ValueError("orders must contain positive n-gram lengths")
        self.width, self.orders, self.right_window = width, tuple(orders), right_window
        self.input_dim, self.heads_per_order = input_dim, heads_per_order
        self.num_heads = len(self.orders) * heads_per_order
        self.operator = WriteOperator(operator) if isinstance(operator, str) else operator
        if not isinstance(self.operator, WriteOperator):
            raise ValueError("operator must be a WriteOperator or its name")
        head_dim = min(32, hidden_dim)
        self.pool_score = nn.Linear(input_dim, 1, bias=False)
        # Order-major channels: (order_0, head_0..H-1), then order_1.
        self.head_embedding = nn.Embedding(self.num_heads, head_dim)
        features = 3 * input_dim + head_dim
        self.target_head = nn.Sequential(nn.Linear(features, hidden_dim), nn.SiLU(),
                                         nn.Linear(hidden_dim, width))
        gate_width = features if self.operator.evidence_only_gate else features + 2 * width + 1
        self.gate_head = nn.Sequential(nn.Linear(gate_width, hidden_dim),
                                       nn.SiLU(), nn.Linear(hidden_dim, 1))
        # Nonzero targets and small open gates provide a gradient on step one.
        nn.init.zeros_(self.pool_score.weight)
        nn.init.normal_(self.head_embedding.weight, std=0.02)
        nn.init.normal_(self.target_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.target_head[-1].bias)
        nn.init.normal_(self.gate_head[-1].weight, std=1e-3)
        nn.init.constant_(self.gate_head[-1].bias, math.log(0.05 / 0.95))
        if target_scale is not None:
            target_scale = torch.as_tensor(target_scale, dtype=torch.float32).detach().clone()
            if (target_scale.shape != (self.num_heads,)
                    or not bool(torch.isfinite(target_scale).all())
                    or bool((target_scale <= 0).any())):
                raise ValueError("target_scale must contain one positive finite value per head")
        # None preserves the exact legacy state_dict and initialization RNG.
        self.register_buffer("target_scale", target_scale)

    def forward(self, hidden: Tensor, rows: Tensor, mask: Tensor,
                state: SparseMemoryState) -> SparseMemoryState:
        events = self.events(hidden, rows, mask, state)
        return self.operator(state, events.rows, events.target, events.beta)

    def events(self, hidden: Tensor, rows: Tensor, mask: Tensor,
               state: SparseMemoryState) -> WriteEvents:
        """Expose targets/gates for diagnostics; no write math is repeated here."""
        components = self.event_components(hidden, rows, mask, state)
        beta = torch.sigmoid(components.gate_logits).float()
        return WriteEvents(components.rows, components.target, beta)

    def event_components(self, hidden: Tensor, rows: Tensor, mask: Tensor,
                         state: SparseMemoryState) -> WriteEventComponents:
        """Keep the original arithmetic while exposing logits and features.

        Freezing the writer's parameters does not detach the direct state
        inputs to the gate. No inverse sigmoid or alternate projection is
        used to recover the base logits.
        """
        if hidden.ndim != 3 or hidden.shape[-1] != self.input_dim:
            raise ValueError("hidden must be [B, T, input_dim]")
        if rows.shape != (*hidden.shape[:2], self.num_heads) or mask.shape != hidden.shape[:2]:
            raise ValueError("rows/mask must have shapes [B, T, K] and [B, T]")
        if rows.dtype != torch.long or state.width != self.width:
            raise ValueError("row dtype or state width mismatch")
        if not (hidden.device == rows.device == mask.device == state.rows.device
                == self.pool_score.weight.device):
            raise ValueError("writer, evidence, and state must share a device")
        features = self.event_features(hidden, rows, mask)
        if not features.rows.numel():
            return WriteEventComponents(
                features.rows, state.values.new_empty((0, self.width)),
                state.counts.new_empty(0), features)
        batch, token, head = features.indices.unbind(dim=1)
        head_features = features.head_features
        if self.heads_per_order == 1:
            target = self.target_head(features.dense).float()
        else:
            evidence = features.evidence
            evidence_dim = 3 * self.input_dim
            target_first = self.target_head[0]
            shared = F.linear(evidence, target_first.weight[:, :evidence_dim],
                              target_first.bias)
            per_head = F.linear(head_features, target_first.weight[:, evidence_dim:])
            target = self.target_head[2](
                self.target_head[1](shared[batch, token] + per_head)).float()
        if self.target_scale is not None:
            target = target * self.target_scale[head, None].float()
        event_rows = features.rows
        state_features = None
        if not self.operator.evidence_only_gate:
            old_values, old_counts = state.lookup(event_rows)
            dtype = self.pool_score.weight.dtype
            state_features = torch.cat((old_values.to(dtype),
                                        (target - old_values).to(dtype),
                                        old_counts.log1p()[:, None].to(dtype)), dim=-1)
        if self.heads_per_order == 1:
            gate_features = features.dense if state_features is None else torch.cat(
                (features.dense, state_features), dim=-1)
            logits = self.gate_head(gate_features)
        else:
            gate_first = self.gate_head[0]
            shared = F.linear(evidence, gate_first.weight[:, :evidence_dim], gate_first.bias)
            head_end = evidence_dim + head_features.shape[-1]
            gate = shared[batch, token] + F.linear(
                head_features, gate_first.weight[:, evidence_dim:head_end])
            if state_features is not None:
                gate = gate + F.linear(state_features, gate_first.weight[:, head_end:])
            logits = self.gate_head[2](self.gate_head[1](gate))
        return WriteEventComponents(event_rows, target, logits.squeeze(-1), features)

    def event_features(self, hidden: Tensor, rows: Tensor, mask: Tensor) -> NativeWriterFeatures:
        """Build the same detached evidence features used by targets and gates."""
        if hidden.ndim != 3 or hidden.shape[-1] != self.input_dim:
            raise ValueError("hidden must be [B, T, input_dim]")
        if rows.shape != (*hidden.shape[:2], self.num_heads) or mask.shape != hidden.shape[:2]:
            raise ValueError("rows/mask must have shapes [B, T, K] and [B, T]")
        if rows.dtype != torch.long:
            raise ValueError("row dtype must be int64")
        if not (hidden.device == rows.device == mask.device == self.pool_score.weight.device):
            raise ValueError("writer and evidence must share a device")
        valid = mask.bool()
        events = valid[:, :, None] & rows.ne(0)
        indices = events.nonzero(as_tuple=False)
        if not indices.numel():
            return NativeWriterFeatures(
                rows.new_empty(0), indices,
                self.head_embedding.weight.new_empty((0, self.head_embedding.embedding_dim)),
                None, self.pool_score.weight.new_empty((0, self.target_head[0].in_features)))

        hidden = hidden.detach().to(dtype=self.pool_score.weight.dtype)
        scores = self.pool_score(hidden).squeeze(-1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1) * valid.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pooled = (weights[:, :, None] * hidden).sum(dim=1)

        right_sum = torch.zeros_like(hidden)
        right_count = hidden.new_zeros(hidden.shape[:2])
        for offset in range(1, min(self.right_window, hidden.shape[1] - 1) + 1):
            # Only valid positions to the right, within this arrived block.
            active = valid[:, offset:].to(dtype=hidden.dtype)
            right_sum[:, :-offset] = right_sum[:, :-offset] + hidden[:, offset:] * active[:, :, None]
            right_count[:, :-offset] = right_count[:, :-offset] + active
        right = right_sum / right_count.clamp_min(1.0)[:, :, None]

        batch, token, head = indices.unbind(dim=1)
        head_features = self.head_embedding(head)
        if self.heads_per_order == 1:
            # Preserve the executed single-head path, including floating-point order.
            features = torch.cat((hidden[batch, token], right[batch, token], pooled[batch],
                                  head_features), dim=-1)
            evidence = None
        else:
            # Both first layers are affine in the shared evidence. Project once
            # per token rather than copying 3*input_dim features to every head.
            evidence = torch.cat((hidden, right, pooled[:, None].expand_as(hidden)), dim=-1)
            features = None
        event_rows = rows[batch, token, head]
        return NativeWriterFeatures(event_rows, indices, head_features, evidence, features)

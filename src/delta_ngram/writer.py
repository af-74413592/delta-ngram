from __future__ import annotations

import math
from dataclasses import dataclass
import torch
from torch import Tensor, nn
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

class ForwardWriter(nn.Module):
    """Map a complete, already-arrived evidence block to D1 write events.

    Hidden features are constant LM-head inputs from the unpatched backbone.
    The attention pool and right-context mean only see this evidence tensor;
    future questions, answers, and audit labels are absent from the interface.
    """

    def __init__(self, width: int, hidden_dim: int = 128,
                 orders: tuple[int, ...] = (2, 3), right_window: int = 4, *,
                 operator: WriteOperator | str = "D1") -> None:
        super().__init__()
        if width < 1 or hidden_dim < 1 or right_window < 0:
            raise ValueError("invalid writer dimensions")
        if not orders or any(order < 1 for order in orders):
            raise ValueError("orders must contain positive n-gram lengths")
        self.width, self.orders, self.right_window = width, tuple(orders), right_window
        self.operator = WriteOperator(operator) if isinstance(operator, str) else operator
        if not isinstance(self.operator, WriteOperator):
            raise ValueError("operator must be a WriteOperator or its name")
        head_dim = min(32, hidden_dim)
        self.pool_score = nn.Linear(width, 1, bias=False)
        self.head_embedding = nn.Embedding(len(self.orders), head_dim)
        features = 3 * width + head_dim
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

    def forward(self, hidden: Tensor, rows: Tensor, mask: Tensor,
                state: SparseMemoryState) -> SparseMemoryState:
        events = self.events(hidden, rows, mask, state)
        return self.operator(state, events.rows, events.target, events.beta)

    def events(self, hidden: Tensor, rows: Tensor, mask: Tensor,
               state: SparseMemoryState) -> WriteEvents:
        """Expose targets/gates for diagnostics; no write math is repeated here."""
        if hidden.ndim != 3 or hidden.shape[-1] != self.width:
            raise ValueError("hidden must be [B, T, width]")
        if rows.shape != (*hidden.shape[:2], len(self.orders)) or mask.shape != hidden.shape[:2]:
            raise ValueError("rows/mask must have shapes [B, T, K] and [B, T]")
        if rows.dtype != torch.long or state.width != self.width:
            raise ValueError("row dtype or state width mismatch")
        if not (hidden.device == rows.device == mask.device == state.rows.device
                == self.pool_score.weight.device):
            raise ValueError("writer, evidence, and state must share a device")
        valid = mask.bool()
        events = valid[:, :, None] & rows.ne(0)
        indices = events.nonzero(as_tuple=False)
        if not indices.numel():
            return WriteEvents(rows.new_empty(0), state.values.new_empty((0, self.width)),
                               state.counts.new_empty(0))

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
        features = torch.cat((hidden[batch, token], right[batch, token], pooled[batch],
                              self.head_embedding(head)), dim=-1)
        target = self.target_head(features).float()
        event_rows = rows[batch, token, head]
        if self.operator.evidence_only_gate:
            gate_features = features
        else:
            old_values, old_counts = state.lookup(event_rows)
            gate_features = torch.cat((features, old_values.to(features.dtype),
                                       (target - old_values).to(features.dtype),
                                       old_counts.log1p()[:, None].to(features.dtype)), dim=-1)
        beta = torch.sigmoid(self.gate_head(gate_features).squeeze(-1)).float()
        return WriteEvents(event_rows, target, beta)

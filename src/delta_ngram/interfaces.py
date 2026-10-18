"""The method contract consumed by the one training/evaluation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .state import SparseMemoryState


@dataclass(frozen=True)
class WriteResult:
    state: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)


class MemoryMethod:
    """An evidence-only writer and a read-only view of its explicit state.

    A method may implement an online optimizer internally. It must return an
    independent readable snapshot and must never receive a query or its label
    in ``write``. Ordinary contexts are materialized only by ``render_prompt``.
    """

    supports_outer_training = False
    supports_row_diagnostics = True
    name = "abstract"

    def __init__(self, reader, tokenizer, device, max_evidence_tokens: int = 128):
        self.reader = reader
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_evidence_tokens = max_evidence_tokens

    def empty_state(self):
        raise NotImplementedError

    def write(self, state, evidence: str) -> WriteResult:
        raise NotImplementedError

    def memory(self, state) -> SparseMemoryState:
        return state

    def render_prompt(self, state, question: str) -> str:
        return question

    def counterfactual(self, reference, alternative):
        """Keep reference addresses/counts while replacing stored content."""
        values, _ = self.memory(alternative).lookup(self.memory(reference).rows)
        old = self.memory(reference)
        return SparseMemoryState(old.rows, values, old.counts, old.precision)

    def replace_memory(self, state, memory: SparseMemoryState):
        """Install a diagnostic snapshot without updating an optimizer."""
        return memory

    def parameters(self):
        return ()

    def train(self, mode: bool = True):
        return self

    def eval(self):
        return self.train(False)

    def size_report(self, state) -> dict[str, int]:
        memory = self.memory(state)
        tensors = (memory.rows, memory.values, memory.counts, memory.precision)
        size = sum(x.numel() * x.element_size() for x in tensors if x is not None)
        return {"state_rows": int(memory.rows.numel()), "table_tensor_bytes": size,
                "optimizer_tensor_bytes": 0, "persistent_tensor_bytes": size}

    def state_bytes(self, state) -> int:
        return self.size_report(state)["persistent_tensor_bytes"]


"""Three online optimizers sharing one evidence-only causal-loss/write loop."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..interfaces import MemoryMethod, WriteResult
from ..ngram_lookup import ngram_rows
from ..runtime import token_ids
from ..state import SparseMemoryState
from .gradient_memory import OnlineGradientMemory, OnlineSparseAdamMemory


def evidence_loss(reader, ids, mask, state, extra_residual=None):
    """Mean next-token loss on the arrived evidence, shifted exactly once.

    This preserves the executed SparseAdam baseline's FP32 conversion and
    cross-entropy operation ordering. No query or scoring-label argument exists.
    """
    output = (reader(ids, mask, state) if extra_residual is None
              else reader.forward_with_residual(ids, mask, state, extra_residual))
    logits = output.logits[:, :-1, :].float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))


@dataclass(frozen=True)
class GradientState:
    """Read an immutable endpoint; only the latest owner revision may be written."""

    owner: OnlineGradientMemory | None
    snapshot: SparseMemoryState
    revision: int
    optimizer_tensor_bytes: int = 0


class OnlineGradientMethod(MemoryMethod):
    """Evidence-only Adam, SGD, or occurrence-normalized SGD.

    Optimizer moments persist within a trajectory. ``empty_state`` constructs
    a fresh optimizer, while old states retain independent readable snapshots.
    Counterfactual/diagnostic states are read-only and cannot resume optimization.
    """

    def __init__(self, reader, tokenizer, device, max_evidence_tokens=128, *,
                 kind="adam", lr=0.01, online_steps=10):
        super().__init__(reader, tokenizer, device, max_evidence_tokens)
        if kind not in {"adam", "sgd", "read_normalized_sgd"}:
            raise ValueError(f"Unknown online gradient method: {kind}")
        if lr <= 0 or online_steps < 1:
            raise ValueError("learning rate and online_steps must be positive")
        self.name, self.kind, self.lr, self.online_steps = kind, kind, lr, online_steps

    def empty_state(self):
        owner = OnlineGradientMemory(self.reader.width, self.device, self.lr, self.kind)
        return GradientState(owner, owner.snapshot(), owner.revision)

    def memory(self, state):
        return state.snapshot

    def write(self, state, evidence: str) -> WriteResult:
        owner = state.owner
        if owner is None or state.revision != owner.revision:
            raise RuntimeError("Only the latest live gradient state can be written; diagnostic/old snapshots are read-only")
        ids = token_ids(self.tokenizer, evidence, self.device)
        if not 2 <= ids.shape[1] <= self.max_evidence_tokens:
            raise ValueError("evidence must fit the configured length and contain at least two tokens")
        mask = torch.ones_like(ids)
        event_rows = ngram_rows(ids, mask, self.reader.buckets, self.reader.orders)
        touched = owner.observe(event_rows)
        if not touched.numel():
            raise ValueError("Evidence contains no writable n-gram rows")
        # Invalidate the previous writable handle even if a later optimizer step fails.
        owner.revision += 1
        read_counts = None
        if self.kind == "read_normalized_sgd":
            unique, read_counts = torch.unique(event_rows[event_rows.ne(0)], sorted=True, return_counts=True)
            if not torch.equal(unique, touched):
                raise RuntimeError("read-count rows differ from the observed rows")
        losses, norms = [], []
        # The shared evaluator can disable autograd around read operations. An
        # online-gradient method explicitly opens it only for this E-only update.
        with torch.enable_grad():
            for _ in range(self.online_steps):
                owner.optimizer.zero_grad(set_to_none=True)
                loss = evidence_loss(self.reader, ids, mask, owner.state())
                if not loss.requires_grad or not bool(torch.isfinite(loss)):
                    raise RuntimeError("evidence loss is non-finite or disconnected from the table")
                loss.backward()
                if any(parameter.grad is not None for parameter in self.reader.model.parameters()):
                    raise RuntimeError("frozen reader accumulated a parameter gradient")
                losses.append(float(loss.detach()))
                norms.append(owner.step_on(touched, read_counts))
                del loss
        sizes = owner.size_report()
        result = GradientState(owner, owner.snapshot(), owner.revision, sizes["optimizer_tensor_bytes"])
        diagnostics = {
            "evidence_tokens": int(ids.numel()), "prediction_tokens_per_step": int(ids.shape[1] - 1),
            "online_steps": self.online_steps, "touched_rows": int(touched.numel()),
            "causal_evidence_nll": losses, "gradient_norm": norms, **sizes,
            "optimizer": "SparseAdam" if self.kind == "adam" else "SGD(momentum=0)",
            "gradient_normalization": ("mean evidence-loss gradient / current-block row occurrences"
                                       if self.kind == "read_normalized_sgd" else "mean evidence-loss gradient"),
        }
        return WriteResult(result, diagnostics)

    def replace_memory(self, state, memory):
        return GradientState(None, memory.detach(), -1)

    def counterfactual(self, reference, alternative):
        old = self.memory(reference)
        values, _ = self.memory(alternative).lookup(old.rows)
        return self.replace_memory(reference, SparseMemoryState(old.rows, values, old.counts, old.precision))

    def size_report(self, state):
        report = super().size_report(state)
        report["optimizer_tensor_bytes"] = state.optimizer_tensor_bytes
        report["persistent_tensor_bytes"] += state.optimizer_tensor_bytes
        # Match the executed baseline's table+optimizer accounting. The live
        # optimizer table and this independent evaluator endpoint coexist;
        # endpoint copies are disclosed separately, not hidden in model weights.
        report["endpoint_snapshot_tensor_bytes"] = report["table_tensor_bytes"]
        return report


__all__ = ["OnlineGradientMethod", "OnlineGradientMemory", "OnlineSparseAdamMemory",
           "GradientState", "evidence_loss"]

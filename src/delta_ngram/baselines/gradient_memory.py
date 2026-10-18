"""Compact online gradient state with exact row-stable SparseAdam semantics."""
from __future__ import annotations

import torch
from torch import nn

from ..state import SparseMemoryState


class OnlineGradientMemory:
    """Allocate evidence rows only; preserve optimizer state within a trajectory.

    ``adam`` follows the executed prototype's SparseAdam operation ordering.
    SGD has no momentum. ``read_normalized_sgd`` divides the gradient of the
    mean evidence loss by this block's physical-row occurrence count, including
    occurrences with zero gradient. It does not undo the loss's token mean.
    """

    def __init__(self, width: int, device: torch.device | str, lr: float,
                 kind: str = "adam") -> None:
        if kind not in {"adam", "sgd", "read_normalized_sgd"}:
            raise ValueError(f"Unknown online gradient method: {kind}")
        if width < 1 or lr <= 0:
            raise ValueError("width and learning rate must be positive")
        self.kind = kind
        self.rows = torch.empty(0, dtype=torch.long, device=device)
        self.values = nn.Parameter(torch.empty((0, width), dtype=torch.float32, device=device))
        self.counts = torch.empty(0, dtype=torch.float32, device=device)
        self.optimizer = (torch.optim.SparseAdam([self.values], lr=lr) if kind == "adam"
                          else torch.optim.SGD([self.values], lr=lr))
        self.revision = 0

    @torch.no_grad()
    def observe(self, event_rows: torch.Tensor) -> torch.Tensor:
        """Expand the table and advance observation counts once per block."""
        if event_rows.dtype != torch.long or event_rows.device != self.rows.device:
            raise ValueError("event row IDs must be int64 on the memory device")
        if bool((event_rows < 0).any()):
            raise ValueError("negative rows are invalid")
        if self.values.grad is not None:
            raise RuntimeError("row expansion requires the previous gradient to be cleared")
        touched = torch.unique(event_rows[event_rows.ne(0)], sorted=True)
        combined = torch.unique(torch.cat((self.rows, touched)), sorted=True)
        if combined.numel() != self.rows.numel():
            old_values, old_optimizer = self.values, self.optimizer
            old_positions = torch.searchsorted(combined, self.rows)
            new_values = old_values.new_zeros((combined.numel(), old_values.shape[1]))
            new_values.index_copy_(0, old_positions, old_values.detach())
            new_counts = self.counts.new_zeros(combined.numel())
            new_counts.index_copy_(0, old_positions, self.counts)
            self.values = nn.Parameter(new_values)
            group = old_optimizer.param_groups[0]
            if self.kind == "adam":
                self.optimizer = torch.optim.SparseAdam(
                    [self.values], lr=group["lr"], betas=group["betas"], eps=group["eps"],
                    maximize=group.get("maximize", False))
                previous = old_optimizer.state.get(old_values, {})
                if previous:
                    if set(previous) != {"step", "exp_avg", "exp_avg_sq"}:
                        raise RuntimeError("unexpected SparseAdam state; refusing an incomplete migration")
                    migrated = {}
                    for key in ("exp_avg", "exp_avg_sq"):
                        tensor = previous[key]
                        if tensor.shape != old_values.shape:
                            raise RuntimeError("SparseAdam moment shape differs from its parameter")
                        expanded = tensor.new_zeros(self.values.shape)
                        expanded.index_copy_(0, old_positions, tensor)
                        migrated[key] = expanded
                    old_step = previous["step"]
                    migrated["step"] = old_step.clone() if torch.is_tensor(old_step) else old_step
                    self.optimizer.state[self.values] = migrated
            else:
                if old_optimizer.state:
                    raise RuntimeError("momentum-free SGD must not carry optimizer state")
                self.optimizer = torch.optim.SGD([self.values], lr=group["lr"])
            self.rows, self.counts = combined, new_counts
        if touched.numel():
            positions = torch.searchsorted(self.rows, touched)
            self.counts.index_add_(0, positions, torch.ones_like(positions, dtype=torch.float32))
        return touched

    def state(self) -> SparseMemoryState:
        return SparseMemoryState(self.rows, self.values, self.counts)

    @torch.no_grad()
    def snapshot(self) -> SparseMemoryState:
        return SparseMemoryState(self.rows.detach().clone(), self.values.detach().clone(),
                                 self.counts.detach().clone())

    def step_on(self, touched: torch.Tensor, read_counts: torch.Tensor | None = None) -> float:
        gradient = self.values.grad
        if gradient is None or gradient.is_sparse or gradient.shape != self.values.shape:
            raise RuntimeError("expected a dense gradient over the compact live table")
        positions = torch.searchsorted(self.rows, touched)
        if (not touched.numel() or bool((positions >= self.rows.numel()).any())
                or not torch.equal(self.rows[positions], touched)):
            raise RuntimeError("step requested an unobserved or empty row set")
        current_gradient = gradient.index_select(0, positions)
        if not bool(torch.isfinite(current_gradient).all()):
            raise RuntimeError("non-finite evidence gradient")
        if self.kind == "read_normalized_sgd":
            if (read_counts is None or read_counts.shape != touched.shape
                    or read_counts.device != touched.device or bool((read_counts <= 0).any())):
                raise ValueError("read-normalized SGD needs positive current-block occurrence counts")
            current_gradient = current_gradient / read_counts[:, None].to(current_gradient.dtype)
        norm = float(current_gradient.norm())
        self.values.grad = torch.sparse_coo_tensor(
            positions[None, :], current_gradient, self.values.shape,
            dtype=self.values.dtype, device=self.values.device).coalesce()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return norm

    def size_report(self) -> dict[str, int]:
        tensors = (self.rows, self.values, self.counts)
        table_bytes = sum(t.numel() * t.element_size() for t in tensors)
        optimizer_bytes = sum(value.numel() * value.element_size()
                              for value in self.optimizer.state.get(self.values, {}).values()
                              if torch.is_tensor(value))
        return {"state_rows": int(self.rows.numel()), "table_tensor_bytes": table_bytes,
                "optimizer_tensor_bytes": optimizer_bytes,
                "persistent_tensor_bytes": table_bytes + optimizer_bytes}


class OnlineSparseAdamMemory(OnlineGradientMemory):
    """Compatibility name for the executed Adam baseline's state manager."""

    def __init__(self, width: int, device: torch.device | str, lr: float) -> None:
        super().__init__(width, device, lr, kind="adam")

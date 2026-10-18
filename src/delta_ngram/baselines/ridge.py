"""Evidence-only optimized-residual target plus a reuse-penalized ridge write.

This is an EngramEdit-style baseline variant, not an exact reproduction of that
method. The target is optimized on the current evidence at this prototype's own
injection point; no future question or scoring answer enters either stage.
"""
from __future__ import annotations

import math

import torch

from ..interfaces import MemoryMethod, WriteResult
from ..ngram_lookup import ngram_rows
from ..runtime import timed, token_ids
from ..state import SparseMemoryState
from .gradient import evidence_loss


def read_design_matrix(event_rows: torch.Tensor):
    """Return current-block rows and the exact sum-heads/sqrt(K) read matrix.

    Positions with no valid row are retained as zero rows. Repeated physical
    rows at one token are coalesced by summing their coefficients.
    """
    if event_rows.dtype != torch.long or event_rows.ndim != 3 or event_rows.shape[0] != 1:
        raise ValueError("a single evidence block needs int64 addresses [1,T,K]")
    if bool((event_rows < 0).any()) or event_rows.shape[-1] < 1:
        raise ValueError("addresses must be nonnegative and have at least one head")
    touched = torch.unique(event_rows[event_rows.ne(0)], sorted=True)
    length, heads = event_rows.shape[1:]
    design = torch.zeros((length, touched.numel()), dtype=torch.float64, device=event_rows.device)
    flat = event_rows.reshape(-1)
    live = flat.ne(0)
    positions = torch.arange(length, device=event_rows.device).repeat_interleave(heads)[live]
    columns = torch.searchsorted(touched, flat[live])
    coefficients = design.new_full((int(live.sum()),), 1.0 / math.sqrt(heads))
    design.index_put_((positions, columns), coefficients, accumulate=True)
    return touched, design


def solve_reuse_ridge(design, residual, old_counts, ridge_lambda, reuse_penalty):
    """Solve min_D ||X D-R||_F^2 + lambda sum_j (1+rho*c_j)||D_j||^2."""
    if ridge_lambda <= 0 or reuse_penalty < 0:
        raise ValueError("ridge_lambda must be positive and reuse_penalty nonnegative")
    if (design.ndim != 2 or residual.ndim != 2 or design.shape[0] != residual.shape[0]
            or old_counts.shape != (design.shape[1],)):
        raise ValueError("ridge matrix, residual, and count shapes differ")
    if bool((old_counts < 0).any()):
        raise ValueError("reuse counts must be nonnegative")
    design, residual = design.double(), residual.double()
    penalty = ridge_lambda * (1.0 + reuse_penalty * old_counts.double())
    system = design.T @ design + torch.diag(penalty)
    right = design.T @ residual
    solution = torch.linalg.solve(system, right)
    if not bool(torch.isfinite(solution).all()):
        raise RuntimeError("non-finite ridge solution")
    return solution.float()


class EvidenceRidgeMethod(MemoryMethod):
    name = "engram_edit_style"
    variant = "evidence_optimized_residual_target_reuse_ridge_v1"

    def __init__(self, reader, tokenizer, device, max_evidence_tokens=128, *,
                 target_steps=10, target_lr=0.05, target_l2=0.001,
                 ridge_lambda=1.0, reuse_penalty=1.0):
        super().__init__(reader, tokenizer, device, max_evidence_tokens)
        if target_steps < 1 or target_lr <= 0 or target_l2 < 0 or ridge_lambda <= 0 or reuse_penalty < 0:
            raise ValueError("invalid residual-target or ridge configuration")
        self.target_steps, self.target_lr, self.target_l2 = target_steps, target_lr, target_l2
        self.ridge_lambda, self.reuse_penalty = ridge_lambda, reuse_penalty

    def empty_state(self):
        return SparseMemoryState.empty(self.reader.width, self.device)

    def write(self, state, evidence: str):
        ids = token_ids(self.tokenizer, evidence, self.device)
        if not 2 <= ids.shape[1] <= self.max_evidence_tokens:
            raise ValueError("evidence must fit the configured length and contain at least two tokens")
        if state.precision is not None:
            raise ValueError("this ridge variant uses observation counts, not RLS precision state")
        mask = torch.ones_like(ids)
        event_rows = ngram_rows(ids, mask, self.reader.buckets, self.reader.orders)
        touched, design = read_design_matrix(event_rows)
        if not touched.numel():
            raise ValueError("Evidence contains no writable n-gram rows")
        snapshot = state.detach()

        def optimize_target():
            with torch.enable_grad():
                residual = torch.zeros((*ids.shape, self.reader.width), dtype=torch.float32,
                                       device=self.device, requires_grad=True)
                optimizer = torch.optim.Adam([residual], lr=self.target_lr)
                losses, objectives = [], []
                for _ in range(self.target_steps):
                    optimizer.zero_grad(set_to_none=True)
                    loss = evidence_loss(self.reader, ids, mask, snapshot, residual)
                    objective = loss + self.target_l2 * residual.square().mean()
                    if not bool(torch.isfinite(objective)):
                        raise RuntimeError("non-finite evidence target objective")
                    objective.backward()
                    if residual.grad is None or not bool(torch.isfinite(residual.grad).all()):
                        raise RuntimeError("non-finite or missing target-residual gradient")
                    if any(parameter.grad is not None for parameter in self.reader.model.parameters()):
                        raise RuntimeError("frozen reader accumulated a parameter gradient")
                    losses.append(float(loss.detach()))
                    objectives.append(float(objective.detach()))
                    optimizer.step()
                if not bool(torch.isfinite(residual).all()):
                    raise RuntimeError("non-finite optimized target residual")
                return residual.detach(), losses, objectives

        (target, losses, objectives), target_seconds = timed(optimize_target, self.device)

        def solve_and_update():
            with torch.no_grad():
                old_values, old_counts = snapshot.lookup(touched)
                increments = solve_reuse_ridge(design, target[0], old_counts,
                                               self.ridge_lambda, self.reuse_penalty)
                rows = torch.unique(torch.cat((snapshot.rows, touched)), sorted=True)
                values, counts = snapshot.lookup(rows)
                positions = torch.searchsorted(rows, touched)
                result = SparseMemoryState(rows, values.index_copy(0, positions, old_values + increments),
                                           counts.index_copy(0, positions, old_counts + 1.0))
                fit_error = float((design @ increments.double() - target[0].double()).norm())
                penalty = self.ridge_lambda * (1.0 + self.reuse_penalty * old_counts.double())
                penalty_value = float((penalty[:, None] * increments.double().square()).sum())
                return result, increments, fit_error, penalty_value

        (result, increments, fit_error, penalty_value), solve_seconds = timed(solve_and_update, self.device)
        diagnostics = {
            "variant": self.variant,
            "scope": "EngramEdit-style E-only optimized-target ridge variant; not an exact original reproduction",
            "evidence_tokens": int(ids.numel()), "prediction_tokens_per_target_step": int(ids.shape[1] - 1),
            "touched_rows": int(touched.numel()), "target_steps": self.target_steps,
            "target_optimizer": "Adam; zero-initialized per-token FP32 extra residual; reset per block",
            "target_lr": self.target_lr, "target_l2": self.target_l2,
            "target_evidence_nll": losses, "target_objective": objectives,
            "target_optimization_seconds": target_seconds, "ridge_solve_and_update_seconds": solve_seconds,
            "target_residual_l2": float(target.norm()), "row_increment_l2": float(increments.norm()),
            "ridge_fit_error_l2": fit_error, "ridge_weighted_penalty": penalty_value,
            "ridge_lambda": self.ridge_lambda, "reuse_penalty": self.reuse_penalty,
            "reuse_count_timing": "old block counts; increment once after solve",
            **self.size_report(result),
        }
        return WriteResult(result, diagnostics)


__all__ = ["EvidenceRidgeMethod", "read_design_matrix", "solve_reuse_ridge"]

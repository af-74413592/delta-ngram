from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import Tensor
from .state import SparseMemoryState

def functional_d1(state: SparseMemoryState, event_rows: Tensor,
                  target: Tensor, beta: Tensor) -> SparseMemoryState:
    """Update each touched row from one common old snapshot, without mutation.

    delta'[j] = delta[j] + mean_a(beta[a] * (target[a] - delta[j])).
    The mean includes all non-padding events for j, including beta=0 events.
    Counts advance once per touched physical row per block, even for beta=0.
    """
    if event_rows.dtype != torch.long or event_rows.ndim != 1:
        raise ValueError("event_rows must be int64 [E]")
    if target.shape != (event_rows.numel(), state.width) or beta.shape != event_rows.shape:
        raise ValueError("target/beta must have shapes [E, width] and [E]")
    if not (event_rows.device == target.device == beta.device == state.rows.device):
        raise ValueError("events and state must share a device")
    if not target.is_floating_point() or not beta.is_floating_point():
        raise ValueError("target and beta must be floating point")
    if bool((event_rows < 0).any()):
        raise ValueError("negative event rows are invalid")
    live = event_rows.ne(0)
    if not bool(live.any()):
        return state
    event_rows = event_rows[live]
    target, beta = target[live].float(), beta[live].float()
    touched, inverse, multiplicity = torch.unique(
        event_rows, sorted=True, return_inverse=True, return_counts=True)
    old_values, old_counts = state.lookup(touched)
    changes = beta[:, None] * (target - old_values[inverse])
    summed = old_values.new_zeros(old_values.shape).index_add(0, inverse, changes)
    updated = old_values + summed / multiplicity[:, None].to(dtype=torch.float32)

    rows = torch.unique(torch.cat((state.rows, touched)), sorted=True)
    values, counts = state.lookup(rows)
    positions = torch.searchsorted(rows, touched)
    if state.precision is not None:
        return SparseMemoryState(rows, values.index_copy(0, positions, updated),
                                 counts.index_copy(0, positions, old_counts + 1.0),
                                 state.lookup_precision(rows))
    return SparseMemoryState(rows, values.index_copy(0, positions, updated),
                             counts.index_copy(0, positions, old_counts + 1.0))


def _events(state, event_rows, target, beta):
    if event_rows.dtype != torch.long or event_rows.ndim != 1:
        raise ValueError("event_rows must be int64 [E]")
    if target.shape != (event_rows.numel(), state.width) or beta.shape != event_rows.shape:
        raise ValueError("target/beta must have shapes [E, width] and [E]")
    if not (event_rows.device == target.device == beta.device == state.rows.device):
        raise ValueError("events and state must share a device")
    if not target.is_floating_point() or not beta.is_floating_point():
        raise ValueError("target and beta must be floating point")
    if bool((event_rows < 0).any()):
        raise ValueError("negative event rows are invalid")
    live = event_rows.ne(0)
    return event_rows[live], target[live].float(), beta[live].float()


def _replace(state, touched, updated, old_counts, *, new_precision=None, precision_init=1.0):
    rows = torch.unique(torch.cat((state.rows, touched)), sorted=True)
    values, counts = state.lookup(rows)
    positions = torch.searchsorted(rows, touched)
    precision = None
    if state.precision is not None or new_precision is not None:
        precision = state.lookup_precision(rows, initial=precision_init)
        if new_precision is not None:
            precision = precision.index_copy(0, positions, new_precision)
    return SparseMemoryState(rows, values.index_copy(0, positions, updated),
                             counts.index_copy(0, positions, old_counts + 1.0), precision)


def functional_direct(state, event_rows, target, beta):
    """C: overwrite by the unweighted target mean; beta is intentionally ignored."""
    event_rows, target, beta = _events(state, event_rows, target, beta)
    if not event_rows.numel():
        return state
    touched, inverse, multiplicity = torch.unique(
        event_rows, sorted=True, return_inverse=True, return_counts=True)
    old_values, old_counts = state.lookup(touched)
    updated = old_values.new_zeros(old_values.shape).index_add(0, inverse, target)
    updated = updated / multiplicity[:, None].to(dtype=torch.float32)
    return _replace(state, touched, updated, old_counts)


def functional_d2(state, event_rows, target, beta, *, kappa=1.0):
    """Observation-count decay; every block uses its old count, including beta=0."""
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    event_rows, target, beta = _events(state, event_rows, target, beta)
    if not event_rows.numel():
        return state
    touched, inverse, multiplicity = torch.unique(
        event_rows, sorted=True, return_inverse=True, return_counts=True)
    old_values, old_counts = state.lookup(touched)
    changes = beta[:, None] * (target - old_values[inverse])
    summed = old_values.new_zeros(old_values.shape).index_add(0, inverse, changes)
    residual_mean = summed / multiplicity[:, None].to(dtype=torch.float32)
    updated = old_values + residual_mean / (1.0 + kappa * old_counts)[:, None]
    return _replace(state, touched, updated, old_counts)


def functional_d3(state, event_rows, target, beta, *, precision_init=1.0):
    """Scalar RLS with differentiable precision; lambda is the positive prior."""
    if not math.isfinite(precision_init) or precision_init <= 0:
        raise ValueError("precision_init must be finite and positive")
    event_rows, target, beta = _events(state, event_rows, target, beta)
    if not event_rows.numel():
        return state
    if not bool(torch.isfinite(beta).all()) or bool(((beta < 0) | (beta > 1)).any()):
        raise ValueError("D3 weights must be finite and in [0, 1]")
    touched, inverse, multiplicity = torch.unique(
        event_rows, sorted=True, return_inverse=True, return_counts=True)
    old_values, old_counts = state.lookup(touched)
    denominator = multiplicity.to(dtype=torch.float32)
    weighted = old_values.new_zeros(old_values.shape).index_add(0, inverse, beta[:, None] * target)
    A = weighted / denominator[:, None]
    w = old_counts.new_zeros(old_counts.shape).index_add(0, inverse, beta) / denominator
    old_precision = state.lookup_precision(touched, initial=precision_init)
    new_precision = old_precision + w
    updated = old_values + (A - w[:, None] * old_values) / new_precision[:, None]
    return _replace(state, touched, updated, old_counts,
                    new_precision=new_precision, precision_init=precision_init)


def functional_d4(state, event_rows, target, beta, *, frozen_rows=()):
    """D1 after excluding entire fixed physical rows, including counts and q.

    frozen_rows must be fixed from tokenizer/template addressing before the
    experiment. No query, label, or measured-effect argument is accepted here.
    """
    event_rows, target, beta = _events(state, event_rows, target, beta)
    frozen = torch.as_tensor(frozen_rows, device=event_rows.device)
    if (frozen.ndim != 1 or (frozen.numel() and frozen.dtype not in (torch.int32, torch.int64))
            or bool((frozen <= 0).any())):
        raise ValueError("frozen_rows must be positive physical row IDs")
    frozen = frozen.to(dtype=torch.long)
    allowed = ~torch.isin(event_rows, frozen)
    return functional_d1(state, event_rows[allowed], target[allowed], beta[allowed])


def functional_d0(state, event_rows, target, beta):
    """Same delta equation as D1; the D0 writer supplies evidence-only gates."""
    return functional_d1(state, event_rows, target, beta)


@dataclass(frozen=True)
class WriteOperator:
    """Stateless operator configuration; it never adds writer parameters."""

    name: str = "D1"
    kappa: float = 1.0
    precision_init: float = 1.0
    frozen_rows: tuple[int, ...] = ()

    def __post_init__(self):
        name = self.name.upper()
        if name in ("DIRECT", "OVERWRITE"):
            name = "C"
        if name not in ("C", "D0", "D1", "D2", "D3", "D4"):
            raise ValueError(f"unsupported write operator: {self.name}")
        if not math.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("kappa must be finite and nonnegative")
        if not math.isfinite(self.precision_init) or self.precision_init <= 0:
            raise ValueError("precision_init must be finite and positive")
        if any(not isinstance(row, int) or isinstance(row, bool) or row <= 0 for row in self.frozen_rows):
            raise ValueError("frozen_rows must contain positive integer physical row IDs")
        if self.frozen_rows and name != "D4":
            raise ValueError("frozen_rows are only used by D4")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "frozen_rows", tuple(sorted(set(self.frozen_rows))))

    @property
    def evidence_only_gate(self):
        return self.name == "D0"

    def config(self):
        return {"name": self.name, "kappa": self.kappa, "precision_init": self.precision_init,
                "frozen_rows": list(self.frozen_rows)}

    def __call__(self, state, event_rows, target, beta):
        if self.name in ("D0", "D1"):
            return functional_d1(state, event_rows, target, beta)
        if self.name == "C":
            return functional_direct(state, event_rows, target, beta)
        if self.name == "D2":
            return functional_d2(state, event_rows, target, beta, kappa=self.kappa)
        if self.name == "D3":
            return functional_d3(state, event_rows, target, beta, precision_init=self.precision_init)
        return functional_d4(state, event_rows, target, beta, frozen_rows=self.frozen_rows)


def apply_write_operator(state, event_rows, target, beta, operator: WriteOperator | str = "D1"):
    """Public functional dispatch; D0 differs only in the writer's gate inputs."""
    if isinstance(operator, str):
        operator = WriteOperator(operator)
    return operator(state, event_rows, target, beta)

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

STATE_SCHEMA_VERSION = 1
IDENTITY_FIELDS = frozenset({"base_model", "tokenizer", "addressing", "writer", "operator", "trajectory"})


def _identity(metadata: Mapping[str, Any]) -> dict:
    """Canonical JSON metadata: every compatibility dimension is explicit."""
    if not isinstance(metadata, Mapping) or not IDENTITY_FIELDS.issubset(metadata):
        raise ValueError(f"identity requires fields: {sorted(IDENTITY_FIELDS)}")
    if any(metadata[key] is None or metadata[key] == "" or metadata[key] == {} for key in IDENTITY_FIELDS):
        raise ValueError("identity fields must be nonempty")
    try:
        return json.loads(json.dumps(dict(metadata), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("identity must contain finite JSON-compatible metadata") from exc

@dataclass(frozen=True)
class SparseMemoryState:
    """Sorted non-padding rows, FP32 deltas, and block-level observation counts.

    Treat the tensors as immutable. Updates below return new tensors and retain
    autograd history; serialization or an explicit TBPTT boundary may use detach.
    """

    rows: Tensor
    values: Tensor
    counts: Tensor
    precision: Tensor | None = None

    def __post_init__(self) -> None:
        if self.rows.dtype != torch.long or self.rows.ndim != 1:
            raise ValueError("rows must be int64 [R]")
        if self.values.dtype != torch.float32 or self.values.ndim != 2:
            raise ValueError("values must be float32 [R, width]")
        if self.counts.dtype != torch.float32 or self.counts.ndim != 1:
            raise ValueError("counts must be float32 [R]")
        if (self.values.shape[0] != self.rows.numel()
                or self.counts.shape != self.rows.shape or self.values.shape[1] < 1):
            raise ValueError("state shapes disagree")
        if not (self.rows.device == self.values.device == self.counts.device):
            raise ValueError("all state tensors must share a device")
        if self.precision is not None:
            if (self.precision.dtype != torch.float32 or self.precision.shape != self.rows.shape
                    or self.precision.device != self.rows.device):
                raise ValueError("precision must be float32 [R] on the state device")
            if not bool(torch.isfinite(self.precision).all()) or bool((self.precision <= 0).any()):
                raise ValueError("precision must be finite and positive")
        if self.rows.numel() and (
            bool((self.rows <= 0).any())
            or bool((self.rows[1:] <= self.rows[:-1]).any())
        ):
            raise ValueError("state rows must be sorted, unique, and positive")

    @property
    def width(self) -> int:
        return self.values.shape[1]

    @classmethod
    def empty(cls, width: int, device: torch.device | str = "cpu", *,
              with_precision: bool = False) -> SparseMemoryState:
        if width < 1:
            raise ValueError("width must be positive")
        return cls(torch.empty(0, dtype=torch.long, device=device),
                   torch.empty((0, width), dtype=torch.float32, device=device),
                   torch.empty(0, dtype=torch.float32, device=device),
                   torch.empty(0, dtype=torch.float32, device=device) if with_precision else None)

    def lookup(self, row_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Return values [*ids, width], counts [*ids]; misses never add rows."""
        if row_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("lookup row IDs must be integer tensors")
        if row_ids.device != self.rows.device:
            raise ValueError("lookup IDs and state must share a device")
        flat = row_ids.reshape(-1).to(dtype=torch.long)
        if not self.rows.numel():
            return (self.values.new_zeros((*row_ids.shape, self.width)),
                    self.counts.new_zeros(row_ids.shape))
        positions = torch.searchsorted(self.rows, flat)
        safe = positions.clamp_max(self.rows.numel() - 1)
        hit = (flat > 0) & (positions < self.rows.numel()) & (self.rows[safe] == flat)
        values = torch.where(hit[:, None], self.values[safe], 0.0)
        counts = torch.where(hit, self.counts[safe], 0.0)
        return values.reshape(*row_ids.shape, self.width), counts.reshape(row_ids.shape)

    def detach(self) -> SparseMemoryState:
        """Explicit gradient boundary; use outside a full-BPTT trajectory."""
        return SparseMemoryState(self.rows, self.values.detach(), self.counts.detach(),
                                 self.precision.detach() if self.precision is not None else None)

    def lookup_precision(self, row_ids: Tensor, initial: float = 1.0) -> Tensor:
        """Read differentiable q; unseen rows use lambda without allocation."""
        if initial <= 0 or not torch.isfinite(torch.tensor(initial)):
            raise ValueError("initial precision must be finite and positive")
        if row_ids.dtype not in (torch.int32, torch.int64) or row_ids.device != self.rows.device:
            raise ValueError("precision row IDs must be integer tensors on the state device")
        if self.precision is None or not self.rows.numel():
            return self.counts.new_full(row_ids.shape, initial)
        flat = row_ids.reshape(-1).to(dtype=torch.long)
        positions = torch.searchsorted(self.rows, flat)
        safe = positions.clamp_max(self.rows.numel() - 1)
        hit = (flat > 0) & (positions < self.rows.numel()) & (self.rows[safe] == flat)
        return torch.where(hit, self.precision[safe], initial).reshape(row_ids.shape)

    def save(self, path: str | Path, *, identity: Mapping[str, Any]) -> None:
        """Persist a detached snapshot; the live trajectory retains its graph."""
        metadata = _identity(identity)
        tensors = {name: value.detach().cpu().clone() if value is not None else None
                   for name, value in (("rows", self.rows), ("values", self.values),
                                       ("counts", self.counts), ("precision", self.precision))}
        torch.save({"format": "delta-ngram-state", "schema_version": STATE_SCHEMA_VERSION,
                    "identity": metadata, "state": tensors}, Path(path))

    @classmethod
    def load(cls, path: str | Path, *, expected_identity: Mapping[str, Any],
             device: torch.device | str = "cpu") -> SparseMemoryState:
        """Reject unknown schemas and every identity mismatch before restoring."""
        expected = _identity(expected_identity)
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        if (not isinstance(payload, dict) or payload.get("format") != "delta-ngram-state"
                or type(payload.get("schema_version")) is not int
                or payload.get("schema_version") != STATE_SCHEMA_VERSION):
            raise ValueError("unsupported state persistence schema")
        if _identity(payload.get("identity")) != expected:
            raise ValueError("state identity is incompatible with the requested reader/writer/trajectory")
        tensors = payload.get("state")
        if not isinstance(tensors, dict) or set(tensors) != {"rows", "values", "counts", "precision"}:
            raise ValueError("malformed state tensor payload")
        if any(not torch.is_tensor(tensors[name]) for name in ("rows", "values", "counts")):
            raise ValueError("malformed state tensors")
        if tensors["precision"] is not None and not torch.is_tensor(tensors["precision"]):
            raise ValueError("malformed precision tensor")
        state = cls(**tensors)
        if (not bool(torch.isfinite(state.values).all()) or not bool(torch.isfinite(state.counts).all())
                or bool((state.counts < 0).any()) or bool((state.counts != state.counts.floor()).any())):
            raise ValueError("persisted state contains invalid values or observation counts")
        return state

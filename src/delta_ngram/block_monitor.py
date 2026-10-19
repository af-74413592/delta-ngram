"""Detached block diagnostics shared by training and checkpoint evaluation.

The monitoring path never calls the writer, changes a state, or draws random
numbers. Fixed histogram bins and sum/sumsq/count can be merged across blocks;
the per-block quantiles cannot be averaged into a population quantile.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import Tensor

from .state import SparseMemoryState


MONITOR_PROTOCOL = "delta_ngram_block_metrics_v1"
BETA_HISTOGRAM_EDGES = (0., .001, .005, .01, .025, .05, .1, .2, .5, .8, .9, .95, .99, 1.)
ROW_L2_HISTOGRAM_EDGES = (0., 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, .1, 1., 10., 100., 1e3, 1e6)
QUANTILES = {"p01": .01, "p05": .05, "p25": .25, "p50": .5,
             "p75": .75, "p95": .95, "p99": .99}


def _finite_scalar(value):
    value = float(value)
    return value if math.isfinite(value) else None


@torch.no_grad()
def distribution_summary(values: Tensor, histogram_edges=BETA_HISTOGRAM_EDGES) -> dict:
    """Return bounded JSON statistics, with nonfinite inputs counted explicitly.

Moments and quantiles describe finite entries only. Histogram bins are
left-closed/right-open except the final bin, which includes its right edge.
Empty distributions have null extrema/mean/quantiles and zero moments.
"""
    edges = tuple(float(edge) for edge in histogram_edges)
    if (len(edges) < 2 or not all(math.isfinite(edge) for edge in edges)
            or any(left >= right for left, right in zip(edges, edges[1:]))):
        raise ValueError("histogram edges must be finite and strictly increasing")
    flat = values.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    finite = flat[torch.isfinite(flat)]
    count = finite.numel()
    histogram = torch.histogram(finite, bins=torch.tensor(edges, dtype=torch.float64)).hist
    quantiles = ([_finite_scalar(value) for value in torch.quantile(
        finite, torch.tensor(list(QUANTILES.values()), dtype=torch.float64)).tolist()]
        if count else [None] * len(QUANTILES))
    return {"count": flat.numel(), "finite_count": count, "nonfinite_count": flat.numel() - count,
            "sum": _finite_scalar(finite.sum()), "sumsq": _finite_scalar(finite.square().sum()),
            "min": _finite_scalar(finite.min()) if count else None,
            "max": _finite_scalar(finite.max()) if count else None,
            "mean": _finite_scalar(finite.mean()) if count else None,
            "quantiles": dict(zip(QUANTILES, quantiles)),
            "histogram": {"edges": list(edges), "counts": [int(value) for value in histogram.tolist()],
                          "underflow": int((finite < edges[0]).sum()),
                          "overflow": int((finite > edges[-1]).sum())}}


def _written_hits(memory: SparseMemoryState, rows: Tensor) -> Tensor:
    """Membership with positive write count, independent of patch magnitude."""
    if not memory.rows.numel():
        return torch.zeros_like(rows, dtype=torch.bool)
    flat = rows.reshape(-1).long()
    positions = torch.searchsorted(memory.rows, flat)
    safe = positions.clamp_max(memory.rows.numel() - 1)
    return ((flat > 0) & (positions < memory.rows.numel())
            & memory.rows[safe].eq(flat) & memory.counts[safe].gt(0)).reshape(rows.shape)


def _hit_counts(hits: Tensor, valid: Tensor) -> dict:
    numerator, denominator = int((hits & valid).sum()), int(valid.sum())
    return {"hit_count": numerator, "denominator": denominator,
            "hit_rate": numerator / denominator if denominator else None}


@torch.no_grad()
def query_hit_summary(state, addresses: Tensor, labels: Tensor,
                      attention_mask: Tensor | None = None) -> list[dict]:
    """Measure addresses at logits that actually predict scored target labels.

``addresses`` is [B,T,K] for the exact reader input, including its prefix.
Logit t predicts label t+1: the first target uses the prefix's final address,
not the first target token's own address. Padding row zero never contributes
to the denominator. Token-any-hit uses scored tokens with at least one valid
address; ``scored_tokens`` also exposes targets whose every head is invalid.
``state`` may be BatchMemoryState, or a SparseMemoryState for B=1.
"""
    if (addresses.ndim != 3 or addresses.dtype not in (torch.int32, torch.int64)
            or labels.shape != addresses.shape[:2] or addresses.shape[-1] < 1):
        raise ValueError("addresses/labels must be integer [B,T,K] / [B,T]")
    if attention_mask is not None and attention_mask.shape != labels.shape:
        raise ValueError("attention_mask must match labels")
    if labels.device != addresses.device or (attention_mask is not None
                                             and attention_mask.device != addresses.device):
        raise ValueError("query tensors must share a device")
    if isinstance(state, SparseMemoryState):
        if addresses.shape[0] != 1 or addresses.device != state.rows.device:
            raise ValueError("sparse state queries require B=1 on the state device")
        packed, memory = addresses, state
        owner_rows = torch.zeros_like(memory.rows)
    else:
        packed, memory = state._pack_rows(addresses), state.memory
        owner_rows = state._owner_ids(memory.rows)
    scored = labels[:, 1:].ne(-100)
    valid = scored[:, :, None] & addresses[:, :-1].ne(0)
    if attention_mask is not None:
        valid = valid & attention_mask[:, :-1, None].bool()
    hits = _written_hits(memory, packed[:, :-1]) & valid
    records = []
    for owner in range(addresses.shape[0]):
        eligible_tokens = valid[owner].any(dim=-1)
        token_hits = hits[owner].any(dim=-1)
        token_denominator = int(eligible_tokens.sum())
        token_count = int(token_hits.sum())
        records.append({
            "owner": owner, "all": _hit_counts(hits[owner], valid[owner]),
            "per_head": [{"head": head, **_hit_counts(hits[owner, :, head], valid[owner, :, head])}
                         for head in range(addresses.shape[-1])],
            "token_any_hit_count": token_count, "token_denominator": token_denominator,
            "token_any_hit_rate": token_count / token_denominator if token_denominator else None,
            "scored_tokens": int(scored[owner].sum()),
            "written_rows": int((owner_rows.eq(owner) & memory.counts.gt(0)).sum()),
        })
    return records


@torch.no_grad()
def summarize_write(old_state, new_state, event_rows: Tensor, event_beta: Tensor) -> list[dict]:
    """Summarize the actual applied write events once, without writer replay.

Rows use BatchMemoryState's packed owner namespace. For D4 the caller passes
the events remaining after physical-row exclusion. Delta norms cover unique
touched rows, comparing missing old rows to zero; the total norm covers the
whole resulting owner patch. An inactive owner receives zero/empty metrics.
"""
    if (old_state.batch_size != new_state.batch_size or old_state.row_capacity != new_state.row_capacity
            or old_state.width != new_state.width or event_rows.ndim != 1
            or event_rows.shape != event_beta.shape):
        raise ValueError("write summary requires matching states and one beta per event row")
    rows, beta = event_rows.detach(), event_beta.detach()
    event_owners = old_state._owner_ids(rows)
    new_owners = new_state._owner_ids(new_state.rows)
    result = []
    for owner in range(old_state.batch_size):
        selected = event_owners.eq(owner)
        touched = torch.unique(rows[selected], sorted=True)
        old_values, _ = old_state.memory.lookup(touched)
        new_values, _ = new_state.memory.lookup(touched)
        row_l2 = torch.linalg.vector_norm(new_values.detach() - old_values.detach(), dim=-1)
        owner_values = new_state.values.detach()[new_owners.eq(owner)]
        result.append({
            "owner": owner, "beta": distribution_summary(beta[selected]),
            "touched_rows": touched.numel(),
            "new_rows": int((~_written_hits(old_state.memory, touched)
                             & _written_hits(new_state.memory, touched)).sum()),
            "total_written_rows": int((new_owners.eq(owner) & new_state.counts.gt(0)).sum()),
            "patch_delta_frobenius": _finite_scalar(torch.linalg.vector_norm(row_l2)),
            "patch_delta_row_l2": distribution_summary(row_l2, ROW_L2_HISTOGRAM_EDGES),
            "patch_total_norm": _finite_scalar(torch.linalg.vector_norm(owner_values)),
        })
    return result


def monitor_source_identity() -> dict:
    """Record source independently of mathematical checkpoint compatibility."""
    root = Path(__file__).resolve().parent
    files = {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(root.rglob("*.py"))}
    return {"protocol": MONITOR_PROTOCOL, "source_sha256": files}

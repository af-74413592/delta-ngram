"""B3 owner packing and SUM-only writer gradients for the isolated EP4 trial.

Every rank owns three of the twelve optimizer-group trajectories. Reader
losses already use the GLOBAL target-token denominator. The writer is replicated
across four ranks, so its local full-history gradients are SUMMED, never averaged.
This module does not change native model arithmetic or authorize a TP2 policy.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Sequence

import torch

from .batching import BatchMemoryState
from .qwen38_train_batching import (
    PackedNativeQueries, collate_native_frames, pack_query_tickets,
)
from .qwen38_training import NativeQueryTicket, TPGradientPolicy
from .state import SparseMemoryState


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class SingleRankGradientPolicy(TPGradientPolicy):
    """Explicit identity at TP1; no fabricated numerical probes or TP2 receipt.

    This specialization retains the bridge's policy interface. TP1 has no
    inter-rank TP reduction to choose. EP4 input gradients still come from the
    actual distributed model's backward; this receipt makes no derivative claim.
    """

    def __init__(self, runtime_identity_sha256: str, *, tensor_parallel_world_size: int):
        _require(type(tensor_parallel_world_size) is int and tensor_parallel_world_size == 1,
                 "single-rank identity requires the actual TP world size to be one")
        super().__init__(
            reduction="replica", world_size=1,
            runtime_identity_sha256=runtime_identity_sha256,
            direction_id="identity_at_TP1_no_probe", probes=(), rtol=0., atol=0.)

    def __post_init__(self):
        _require(self.world_size == 1 and self.reduction == "replica" and self.probes == (),
                 "TP1 identity cannot claim a multi-rank reduction or numerical probe")
        _require(isinstance(self.runtime_identity_sha256, str)
                 and re.fullmatch(r"[0-9a-f]{64}", self.runtime_identity_sha256) is not None,
                 "runtime identity must be a lowercase SHA256")

    def receipt(self) -> dict:
        result = {
            "protocol": "qwen38_tp1_identity_policy_v1", "accepted": True,
            "world_size": 1, "reduction": "replica",
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "scope": "identity only: one tensor-parallel rank needs no TP reduction",
            "numerical_probes": [],
            "uses_TP2_proof": False, "validates_reader_derivative": False,
            "loss_contract": "local CE sum / global optimizer-group target tokens",
            "writer_replica_reduction": "separate four-rank SUM after local full BPTT",
        }
        result["receipt_sha256"] = hashlib.sha256(
            json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return result


def collate_ep4_frames(
        frames: Sequence[dict | None], *, kind: str, device="cpu") -> dict:
    """Reuse frozen frame validation, then remove the fourth inactive lane."""
    frames = tuple(frames)
    _require(len(frames) == 3, "EP4 trial requires exactly three local trajectory lanes")
    result = collate_native_frames((*frames, None), kind=kind, device=device)
    for key, value in result["batch"].items():
        if isinstance(value, torch.Tensor):
            result["batch"][key] = value[:3].contiguous()
    for key in ("valid_mask", "encoder_valid_mask", "write_mask", "owner_ids", "active_mask"):
        result[key] = result[key][:3].contiguous()
    for key in ("frames", "loss_target_tokens_by_lane", "input_tokens_by_lane"):
        result[key] = result[key][:3]
    result["padded_input_tokens"] = 3 * result["batch"]["input_ids"].shape[1]
    return result


def pack_ep4_query_tickets(
        tickets: Sequence[NativeQueryTicket | None], *, row_capacity: int,
        width: int = 160, device=None) -> PackedNativeQueries:
    """Pack B3 independent histories, including a fully inactive local rank.

    If all stored memories are empty, one zero sentinel value keeps the actual
    reader graph differentiable. Its cotangent is discarded, never assigned to
    a trajectory. All inactive inputs MUST have false valid masks and -100
    labels; no synthetic scored token or writer event is created.
    """
    tickets = tuple(tickets)
    _require(len(tickets) == 3, "EP4 query packing requires three lanes")
    present = [ticket for ticket in tickets if ticket is not None]
    if present:
        packed = pack_query_tickets(
            (*tickets, None), row_capacity=row_capacity, width=width, device=device)
        memory = BatchMemoryState(packed.reader_state.memory, 3, row_capacity)
        result = PackedNativeQueries(memory, tickets, packed.row_slices[:3])
        device = memory.values.device
        if memory.values.numel():
            return result
    else:
        device = torch.device("cpu" if device is None else device)
        result = None
    _require(type(row_capacity) is int and row_capacity > 0
             and type(width) is int and width > 0, "invalid sentinel memory dimensions")
    sentinel = SparseMemoryState(
        torch.tensor([1], dtype=torch.long, device=device),
        torch.zeros((1, width), dtype=torch.float32, device=device, requires_grad=True),
        torch.zeros(1, dtype=torch.float32, device=device))
    return PackedNativeQueries(
        BatchMemoryState(sentinel, 3, row_capacity), tickets,
        result.row_slices if result is not None else (slice(0, 0),) * 3)


@torch.no_grad()
def sum_ep4_writer_gradients(writer, *, group) -> dict:
    """SUM global-denominator local gradients before clipping and AdamW.

    All ranks call once per completed optimizer group, including ranks without
    local gradients. Globally unused parameters retain grad=None. No DDP
    average, TP-replica selection, optimizer step, or second normalization occurs.
    """
    import torch.distributed as dist

    _require(dist.is_initialized() and group is not None and dist.get_world_size(group) == 4,
             "EP4 writer synchronization requires an explicit initialized four-rank group")
    params = [value for value in writer.parameters() if value.requires_grad]
    _require(bool(params) and all(value.dtype == torch.float32
                                 and value.device == params[0].device for value in params),
             "EP4 writer parameters must be nonempty FP32 tensors on one device")
    invalid = any(value.grad is not None and (
        value.grad.is_sparse or value.grad.shape != value.shape
        or value.grad.dtype != torch.float32 or value.grad.device != value.device
        or not bool(torch.isfinite(value.grad).all())) for value in params)
    flags = torch.tensor(
        [int(value.grad is not None) for value in params] + [int(invalid)],
        dtype=torch.int64, device=params[0].device)
    dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=group)
    _require(int(flags[-1]) == 0, "one or more EP4 ranks have invalid writer gradients")
    flat = torch.cat([
        torch.zeros_like(value).reshape(-1) if value.grad is None else value.grad.reshape(-1)
        for value in params])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
    _require(bool(torch.isfinite(flat).all()), "global EP4 writer gradient is nonfinite")
    offset = 0
    for index, value in enumerate(params):
        next_offset = offset + value.numel()
        if int(flags[index]):
            value.grad = flat[offset:next_offset].view_as(value).clone()
        else:
            value.grad = None
        offset = next_offset
    return {
        "protocol": "qwen38_ep4_writer_gradient_sum_v1", "world_size": 4,
        "reduction": "SUM", "additional_divisor": 1,
        "local_losses_used_global_target_denominator": True,
        "global_gradient_l2": float(flat.double().norm()),
        "parameters_with_grad": int(flags[:-1].sum()),
    }

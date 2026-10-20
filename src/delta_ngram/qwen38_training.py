"""Writer full-BPTT with one disposable native-reader graph per query.

The real Megatron schedule is supplied by the caller, never replaced here.
For each query, pass ticket.reader_state into that schedule and backward its
explicit scalar loss_sum / reader_loss_divisor. Record the resulting leaf
gradient. After all writes/queries in a trajectory, backward_writer performs
one joint VJP into the ORIGINAL state-value tensors. Only the reader boundary
is detached; writes must continue from the original snapshots.

TP scaling has no default. A policy includes directional finite differences or
an explicitly scoped native SP/lookup adjoint proof for a runtime identity.
A CPU fixture receipt cannot
authorize a different GPU runtime. This module does not choose TP/DP/EP groups,
transport final H across PP, step an optimizer, or run a model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Callable, Literal

import torch
from torch import Tensor

from .alpha_memory import LazyDecayState
from .batching import BatchMemoryState


Memory = BatchMemoryState | LazyDecayState


@dataclass(frozen=True)
class GradientScaleProbe:
    """One central difference of the SAME logical direction on every TP rank.

    finite_difference and each rank_directional_gradients entry refer to the
    same scalar loss_sum / explicit_divisor. TP copies are not separate losses.
    """

    step: float
    finite_difference: float
    rank_directional_gradients: tuple[float, ...]


@dataclass(frozen=True)
class TPGradientPolicy:
    """Explicit reduction, accepted only against supplied numerical evidence.

    ``replica`` uses each rank's already-complete gradient without a collective.
    ``sum`` and ``mean`` require that named TP reduction before the writer VJP.
    Persist receipt() next to runtime/code identities and calibration results.
    """

    reduction: Literal["replica", "sum", "mean"]
    world_size: int
    runtime_identity_sha256: str
    direction_id: str
    probes: tuple[GradientScaleProbe, ...]
    rtol: float
    atol: float

    def __post_init__(self):
        if self.reduction not in ("replica", "sum", "mean"):
            raise ValueError("explicit TP reduction must be replica, sum, or mean")
        if type(self.world_size) is not int or self.world_size < 1:
            raise ValueError("TP world_size must be a positive integer")
        if not re.fullmatch(r"[0-9a-f]{64}", self.runtime_identity_sha256):
            raise ValueError("runtime identity must be a lowercase SHA256")
        if not isinstance(self.direction_id, str) or not self.direction_id.strip():
            raise ValueError("finite-difference direction identity is required")
        if (not math.isfinite(self.rtol) or not 0 <= self.rtol < 1
                or not math.isfinite(self.atol) or self.atol < 0):
            raise ValueError("finite-difference tolerances must be finite and explicit")
        if (not isinstance(self.probes, tuple) or len(self.probes) < 2
                or any(not isinstance(probe, GradientScaleProbe) for probe in self.probes)):
            raise ValueError("at least two finite-difference step sizes are required")
        steps = []
        for probe in self.probes:
            if not math.isfinite(probe.step) or probe.step <= 0:
                raise ValueError("finite-difference steps must be finite and positive")
            steps.append(probe.step)
            gradients = probe.rank_directional_gradients
            if (not isinstance(gradients, tuple) or len(gradients) != self.world_size
                    or not all(math.isfinite(value) for value in gradients)
                    or not math.isfinite(probe.finite_difference)
                    or abs(probe.finite_difference) <= 10 * self.atol):
                raise ValueError("finite-difference evidence is nonfinite or uninformative")
            if self.reduction == "replica":
                candidates = gradients
            else:
                value = sum(gradients)
                candidates = (value / self.world_size if self.reduction == "mean" else value,)
            tolerance = self.atol + self.rtol * abs(probe.finite_difference)
            if any(abs(value - probe.finite_difference) > tolerance for value in candidates):
                raise ValueError("selected TP reduction disagrees with finite differences")
        if len(set(steps)) != len(steps):
            raise ValueError("finite-difference steps must be distinct")
        reference = self.probes[0].finite_difference
        if any(abs(probe.finite_difference - reference)
               > self.atol + self.rtol * abs(reference) for probe in self.probes[1:]):
            raise ValueError("finite-difference estimates do not agree across step sizes")

    def receipt(self) -> dict:
        result = {
            "protocol": "qwen38_tp_gradient_scale_v1",
            "accepted": True,
            "loss_contract": "loss_sum / explicit_divisor; one logical loss, not a sum of TP copies",
            **asdict(self),
        }
        result["receipt_sha256"] = hashlib.sha256(
            json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return result


@dataclass(frozen=True)
class NativeTPStructurePolicy:
    """Replica-only TP scaling proven at the actual SP and lookup boundaries.

    The finite differences below concern a FP64 LINEAR SP operator, not CE.
    The native query supplies independent gather/index-add adjoint comparisons.
    This receipt does not certify smoothness of the quantized reader loss.
    """

    reduction: Literal["replica"]
    world_size: int
    runtime_identity_sha256: str
    operator_source_sha256: str
    proof_source_sha256: str
    direction_id: str
    operator_probes: tuple[GradientScaleProbe, ...]
    sp_adjoint_max_abs_errors: tuple[float, ...]
    overlay_adjoint_max_abs_errors: tuple[float, ...]
    overlay_reference_max_abs: tuple[float, ...]
    replicated_gradient_sha256: tuple[str, ...]
    duplicated_readout_sha256: tuple[str, ...]
    rtol: float = 1e-6
    atol: float = 1e-7

    def __post_init__(self):
        if self.reduction != "replica" or self.world_size != 2:
            raise ValueError("native structure proof authorizes TP2 replica only")
        for value in (self.runtime_identity_sha256, self.operator_source_sha256,
                      self.proof_source_sha256, self.direction_id):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("native proof identities must be lowercase SHA256")
        if self.rtol != 1e-6 or self.atol != 1e-7:
            raise ValueError("native lookup adjoint tolerances are fixed by the protocol")
        # This validates only the small FP64 scatter operator's linear objective.
        TPGradientPolicy("replica", 2, self.runtime_identity_sha256, self.direction_id,
                         self.operator_probes, 1e-8, 1e-9)
        for values in (self.sp_adjoint_max_abs_errors, self.overlay_adjoint_max_abs_errors,
                       self.overlay_reference_max_abs):
            if (not isinstance(values, tuple) or len(values) != 2
                    or any(not math.isfinite(x) or x < 0 for x in values)):
                raise ValueError("two finite nonnegative native adjoint observations are required")
        if any(self.sp_adjoint_max_abs_errors):
            raise ValueError("native SP adjoint must equal the independent TP gather exactly")
        for error, reference in zip(self.overlay_adjoint_max_abs_errors,
                                    self.overlay_reference_max_abs):
            if reference <= 10 * self.atol or error > self.atol + self.rtol * reference:
                raise ValueError("native overlay adjoint is uninformative or disagrees with index-add")
        for values in (self.replicated_gradient_sha256, self.duplicated_readout_sha256):
            if (not isinstance(values, tuple) or len(values) != 2
                    or any(not isinstance(x, str) or not re.fullmatch(r"[0-9a-f]{64}", x)
                           for x in values) or len(set(values)) != 1):
                raise ValueError("native gradients and duplicated readout must match across TP")

    def receipt(self) -> dict:
        result = {
            "protocol": "qwen38_native_tp_structure_v1", "accepted": True,
            "scope": "TP scaling only: FP64 SP linear FD plus native SP/lookup adjoints",
            "does_not_validate_bf16_ce_finite_differences": True,
            "loss_contract": "one logical CE sum / explicit divisor; TP copies are not summed",
            **asdict(self),
        }
        result["receipt_sha256"] = hashlib.sha256(
            json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return result


TPPolicy = TPGradientPolicy | NativeTPStructurePolicy


@torch.no_grad()
def distributed_tp_gradient(gradient: Tensor, policy: TPPolicy, *, group) -> Tensor:
    """Apply an accepted policy to exactly the caller-supplied TP process group."""
    import torch.distributed as dist

    if not dist.is_initialized() or group is None:
        raise RuntimeError("an initialized, explicit TP process group is required")
    if dist.get_world_size(group) != policy.world_size:
        raise ValueError("TP process group differs from the validated policy")
    result = gradient.detach().clone()
    if policy.reduction != "replica":
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        if policy.reduction == "mean":
            result.div_(policy.world_size)
    return result


def writer_evidence_features(
    hidden: Tensor, *, mode: Literal["none", "rmsnorm"], eps: float,
) -> Tensor:
    """Detached FP32 H; use identically for target calibration and evidence writes.

    RMSNorm is parameter-free, across H channels separately for every token.
    The caller registers mode/epsilon and supplies base-only post-HC-mixer H.
    This does not normalize delta, change the native reader, or add parameters.
    """
    if (not isinstance(hidden, Tensor) or hidden.ndim != 3
            or not hidden.is_floating_point() or hidden.shape[-1] < 1):
        raise ValueError("evidence H must be floating [B,T,H]")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("register a finite positive RMSNorm epsilon")
    # Own the evidence bytes: a subsequent PP receive or inference workspace
    # may reuse its buffer while writer autograd still needs these features.
    value = hidden.detach().to(dtype=torch.float32, copy=True)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("evidence H is nonfinite")
    if mode == "none":
        return value
    if mode == "rmsnorm":
        return value * (value.square().mean(-1, keepdim=True) + eps).rsqrt()
    raise ValueError("H preprocessing mode must be none or rmsnorm")


class NativeQueryTicket:
    """One reader invocation; never use reader_state for subsequent writer writes."""

    def __init__(self, owner, snapshot: Memory):
        self._owner = owner
        self._source_values = snapshot.values
        tensors = [snapshot.rows, snapshot.values, snapshot.counts]
        if isinstance(snapshot, LazyDecayState):
            tensors += [snapshot.block_clock, snapshot.cumulative_log_alpha,
                        snapshot.row_last_written_block, snapshot.row_log_alpha_at_write]
        self._versions = [(value, value._version) for value in tensors]
        self.requires_reader_backward = bool(snapshot.values.numel())
        self.leaf_values = snapshot.values.detach().clone().requires_grad_(
            self.requires_reader_backward)
        self.reader_state = snapshot.with_values(self.leaf_values)
        self._recorded = False
        self._cotangent = None

    def _check_immutable(self):
        if any(value._version != version for value, version in self._versions):
            raise RuntimeError("a writer snapshot was modified in place")
        if self.leaf_values._version != 0:
            raise RuntimeError("reader delta leaf was modified in place")


class NativeTrajectoryGradients:
    """Collect query cotangents, then apply a single full-trajectory writer VJP.

    total_loss_tokens is the entire optimizer group's denominator.
    expected_loss_tokens is this trajectory's share of that group. Separate
    trajectories may call backward_writer before the optimizer group ends.
    The caller alone handles optimizer zero/clip/step and DP synchronization.
    """

    def __init__(
        self, *, total_loss_tokens: int, expected_loss_tokens: int,
        policy: TPPolicy, runtime_identity_sha256: str,
    ):
        if (type(total_loss_tokens) is not int or type(expected_loss_tokens) is not int
                or not 0 < expected_loss_tokens <= total_loss_tokens):
            raise ValueError("positive trajectory/group token budgets are required")
        if not isinstance(policy, (TPGradientPolicy, NativeTPStructurePolicy)):
            raise TypeError("an explicitly validated TP gradient policy is required")
        if runtime_identity_sha256 != policy.runtime_identity_sha256:
            raise ValueError("TP validation receipt belongs to a different runtime")
        self.total_loss_tokens = total_loss_tokens
        self.expected_loss_tokens = expected_loss_tokens
        self.policy = policy
        self._tickets = []
        self._observed_tokens = 0
        self._loss_sum = 0.
        self._finished = False

    def prepare_query(self, snapshot: Memory) -> NativeQueryTicket:
        if self._finished:
            raise RuntimeError("trajectory backward has already finished")
        if not isinstance(snapshot, (BatchMemoryState, LazyDecayState)):
            raise TypeError("query snapshot must be a native batch or lazy state")
        if snapshot.precision is not None or snapshot.counts.requires_grad:
            raise ValueError("native D1 counts are constants and precision is unsupported")
        if isinstance(snapshot, LazyDecayState) and (
                snapshot.cumulative_log_alpha.requires_grad
                or snapshot.row_log_alpha_at_write.requires_grad):
            raise ValueError("this bridge supports fixed-alpha clocks only")
        if snapshot.values.numel() and not snapshot.values.requires_grad:
            raise ValueError("nonempty writer snapshot is detached from its write history")
        ticket = NativeQueryTicket(self, snapshot)
        self._tickets.append(ticket)
        return ticket

    def record_query(
        self, ticket: NativeQueryTicket, *, loss_sum: float | Tensor, loss_tokens: int,
        reader_loss_divisor: float,
        reducer: Callable[[Tensor, TPPolicy], Tensor] | None = None,
    ):
        """Save d(loss_sum / reader_loss_divisor)/d(reader_leaf), then release it.

        Native schedules must expose the divisor actually used in backward.
        Recovery to loss_sum and normalization by the optimizer-group token
        budget happen exactly once here. No implicit microbatch/DP scale exists.
        """
        if self._finished or not isinstance(ticket, NativeQueryTicket) or ticket._owner is not self:
            raise ValueError("query ticket does not belong to this active trajectory")
        if ticket._recorded:
            raise RuntimeError("query ticket was already recorded")
        if (type(loss_tokens) is not int or loss_tokens < 1
                or self._observed_tokens + loss_tokens > self.expected_loss_tokens):
            raise ValueError("query token count exceeds the registered trajectory budget")
        divisor = float(reader_loss_divisor)
        scalar = float(loss_sum.detach()) if isinstance(loss_sum, Tensor) else float(loss_sum)
        if not math.isfinite(divisor) or divisor <= 0 or not math.isfinite(scalar):
            raise ValueError("loss and explicit reader divisor must be finite")
        ticket._check_immutable()
        if ticket.requires_reader_backward:
            gradient = ticket.leaf_values.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("native reader did not return a finite leaf gradient")
            gradient = gradient.detach().clone()
            if self.policy.world_size > 1 and self.policy.reduction != "replica" and reducer is None:
                raise ValueError("the validated TP policy requires an explicit reducer")
            if reducer is not None:
                with torch.no_grad():
                    gradient = reducer(gradient, self.policy)
            if (not isinstance(gradient, Tensor) or gradient.shape != ticket.leaf_values.shape
                    or gradient.device != ticket.leaf_values.device
                    or gradient.dtype != ticket.leaf_values.dtype
                    or gradient.requires_grad or not bool(torch.isfinite(gradient).all())):
                raise ValueError("TP reducer returned an invalid cotangent")
            ticket._cotangent = gradient.detach().clone() * (divisor / self.total_loss_tokens)
        elif ticket.leaf_values.grad is not None:
            raise RuntimeError("empty memory unexpectedly received a reader gradient")
        ticket.leaf_values.grad = None
        ticket._recorded = True
        self._observed_tokens += loss_tokens
        self._loss_sum += scalar

    def backward_writer(self) -> dict:
        """Apply every saved query VJP jointly, retaining the complete write chain."""
        if self._finished:
            raise RuntimeError("trajectory backward has already finished")
        if self._observed_tokens != self.expected_loss_tokens or any(
                not ticket._recorded for ticket in self._tickets):
            raise RuntimeError("all registered query tokens/tickets must finish before writer backward")
        for ticket in self._tickets:
            ticket._check_immutable()
        active = [ticket for ticket in self._tickets if ticket._cotangent is not None]
        self._finished = True  # A partial autograd failure cannot safely be replayed.
        if active:
            torch.autograd.backward(
                tuple(ticket._source_values for ticket in active),
                tuple(ticket._cotangent for ticket in active),
            )
        report = {
            "loss_sum": self._loss_sum,
            "loss_tokens": self._observed_tokens,
            "optimizer_group_loss_tokens": self.total_loss_tokens,
            "normalized_loss": self._loss_sum / self.total_loss_tokens,
            "queries": len(self._tickets),
            "writer_vjp_queries": len(active),
            "tp_policy_receipt_sha256": self.policy.receipt()["receipt_sha256"],
        }
        for ticket in self._tickets:
            ticket._source_values = None
            ticket._cotangent = None
            ticket._versions.clear()
        self._tickets.clear()
        return report

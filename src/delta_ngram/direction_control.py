"""Preregistered shared-channel direction controls for read-only evaluation.

Use the same plan for every event, interference level, and self/donor condition.
Only pass the returned state to the reader. Continue future writes from the
original state so that this control cannot change the writer's gate trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import random
from typing import overload

import torch

from .batching import BatchMemoryState
from .state import SparseMemoryState


DIRECTION_CONTROL_SEEDS = (17, 29, 43)
DIRECTION_CONTROL_SCHEME = "shared_channel_permutation_v1"


@dataclass(frozen=True)
class DirectionPermutation:
    """One auditable permutation of the value channels, shared by all rows.

    The full mapping is stored in metadata because Python RNG implementations
    are not assumed interchangeable forever. Fixed points are retained; there
    is no result-dependent resampling or enforcement of a derangement.
    """

    width: int
    seed: int
    indices: tuple[int, ...] = field(init=False)
    permutation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 1:
            raise ValueError("width must be a positive integer")
        if type(self.seed) is not int or self.seed not in DIRECTION_CONTROL_SEEDS:
            raise ValueError(f"seed must be one of {DIRECTION_CONTROL_SEEDS}")
        indices = list(range(self.width))
        random.Random(self.seed).shuffle(indices)
        encoded = json.dumps(indices, separators=(",", ":")).encode("ascii")
        object.__setattr__(self, "indices", tuple(indices))
        object.__setattr__(self, "permutation_sha256", hashlib.sha256(encoded).hexdigest())

    def metadata(self) -> dict:
        """JSON-safe preregistration data; hash covers compact ASCII JSON of P."""
        fixed_points = sum(i == channel for i, channel in enumerate(self.indices))
        return {
            "scheme": DIRECTION_CONTROL_SCHEME,
            "method": "python.random.Random(seed).shuffle(list(range(width)))",
            "seed": self.seed,
            "width": self.width,
            "permutation": list(self.indices),
            "permutation_sha256": self.permutation_sha256,
            "hash_encoding": "sha256(compact_ascii_json(permutation))",
            "fixed_points": fixed_points,
            "is_identity": fixed_points == self.width,
        }


def make_direction_permutation(width: int, seed: int) -> DirectionPermutation:
    """Build one fixed-seed plan; construct once and reuse for every read."""
    return DirectionPermutation(width=width, seed=seed)


@overload
def permute_state_for_read(state: SparseMemoryState,
                          plan: DirectionPermutation) -> SparseMemoryState: ...


@overload
def permute_state_for_read(state: BatchMemoryState,
                          plan: DirectionPermutation) -> BatchMemoryState: ...


def permute_state_for_read(state, plan):
    """Return a state with values[:, P], preserving the input state unchanged.

    This is a read-only evaluation view by convention, not a writable successor
    state. Column selection allocates new values and preserves each row's exact
    element multiset (including signed zero). Rows, counts, optional precision,
    and batch owner namespaces are retained. All row norms and the shared-order
    readout norm are mathematically invariant; floating reductions can round.
    No model, writer, or parameter tensor is accepted or changed by this API.
    """
    if not isinstance(state, (SparseMemoryState, BatchMemoryState)):
        raise TypeError("state must be SparseMemoryState or BatchMemoryState")
    if not isinstance(plan, DirectionPermutation):
        raise TypeError("plan must be a DirectionPermutation")
    if state.width != plan.width:
        raise ValueError("permutation width differs from the memory value width")
    index = torch.tensor(plan.indices, dtype=torch.long, device=state.values.device)
    values = state.values.index_select(1, index)
    if isinstance(state, BatchMemoryState):
        return state.with_values(values)
    return SparseMemoryState(state.rows, values, state.counts, state.precision)

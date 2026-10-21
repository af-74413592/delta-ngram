"""Pinned, read-only FP32 Delta snapshots for the local Qwen3.8 vLLM reader.

These files are inference views, not resumable writer checkpoints. Lazy decay
is materialized once during export. No row pruning, BF16 merge, or projection
of the sparse table happens here. Each engine serves exactly one owner (0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .alpha_memory import LazyDecayState
from .batching import BatchMemoryState
from .qwen38_native import Qwen38PLELayout
from .state import _identity


SNAPSHOT_FORMAT = "delta-ngram-qwen38-vllm-delta"
SNAPSHOT_SCHEMA_VERSION = 1
_LAYOUT_FIELDS = frozenset({
    "head_offsets", "head_vocab_sizes", "input_dim", "row_width",
    "heads_per_order", "orders", "hc_count", "ple_layer_id",
})
_PAYLOAD_FIELDS = frozenset({
    "format", "schema_version", "identity", "layout", "owner_id", "batch_size",
    "row_format", "decay_materialized", "source_kind", "rows", "values",
})


def _owner(owner_id: int) -> None:
    if type(owner_id) is not int or owner_id != 0:
        raise ValueError("the fixed-Delta reader requires integer owner_id=0")


def _layout(layout: Qwen38PLELayout | Mapping[str, Any]) -> Qwen38PLELayout:
    if isinstance(layout, Qwen38PLELayout):
        fields = {key: getattr(layout, key) for key in _LAYOUT_FIELDS}
    elif isinstance(layout, Mapping) and set(layout) == _LAYOUT_FIELDS:
        fields = dict(layout)
    else:
        raise ValueError("layout must contain the complete Qwen38PLELayout fields")
    for key in ("head_offsets", "head_vocab_sizes", "orders"):
        if (not isinstance(fields[key], (tuple, list))
                or any(type(value) is not int for value in fields[key])):
            raise ValueError("layout sequence fields must contain integers")
        fields[key] = tuple(fields[key])
    for key in _LAYOUT_FIELDS - {"head_offsets", "head_vocab_sizes", "orders"}:
        if type(fields[key]) is not int:
            raise ValueError("layout dimensions must be integers")
    result = Qwen38PLELayout(**fields)
    if result.row_capacity > torch.iinfo(torch.int64).max:
        raise ValueError("layout row capacity exceeds int64")
    return result


def _layout_dict(layout: Qwen38PLELayout) -> dict:
    return {key: list(value) if isinstance(value, tuple) else value
            for key in sorted(_LAYOUT_FIELDS)
            for value in (getattr(layout, key),)}


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _sha256(value: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in value)):
        raise ValueError("expected_sha256 must be a pinned 64-character SHA256")
    return value.lower()


def _device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("snapshot device must be CPU or CUDA")
    return result


def _validate_tensors(rows: Tensor, values: Tensor, layout: Qwen38PLELayout) -> None:
    if (not isinstance(rows, Tensor) or rows.layout != torch.strided
            or rows.dtype != torch.int64 or rows.ndim != 1):
        raise ValueError("snapshot rows must be int64 [R]")
    if (not isinstance(values, Tensor) or values.layout != torch.strided
            or values.dtype != torch.float32
            or values.shape != (rows.numel(), layout.row_width)):
        raise ValueError("snapshot values must be float32 [R,160]")
    if rows.device != values.device or rows.device.type not in {"cpu", "cuda"}:
        raise ValueError("snapshot tensors must share a CPU or CUDA device")
    if (bool(((rows <= 0) | (rows > layout.row_capacity)).any())
            or bool((rows[1:] <= rows[:-1]).any())):
        raise ValueError("snapshot rows must be sorted, unique physical rows in range")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("snapshot values must be finite")


@dataclass(frozen=True)
class FrozenVLLMDelta:
    """Immutable API over owned sparse tensors, constructed by the loader.

    Private tensors are never returned by the public API. Version guards catch
    ordinary in-place mutation of private storage; they are an integrity check,
    not a security boundary against arbitrary Python code or ``Tensor.data``.
    The file SHA is checked once at load, never during token lookup.
    """

    _rows: Tensor = field(repr=False)
    _values: Tensor = field(repr=False)
    _offsets: Tensor = field(repr=False)
    _ends: Tensor = field(repr=False)
    _identity_json: str = field(repr=False)
    _layout_json: str = field(repr=False)
    sha256: str
    source_kind: str
    owner_id: int = 0
    _versions: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _owner(self.owner_id)
        object.__setattr__(self, "_versions", tuple(
            value._version for value in self._tensors()))

    def _tensors(self) -> tuple[Tensor, ...]:
        return self._rows, self._values, self._offsets, self._ends

    def _guard(self) -> None:
        if tuple(value._version for value in self._tensors()) != self._versions:
            raise RuntimeError("fixed-Delta snapshot tensors were mutated after loading")

    @property
    def identity(self) -> dict:
        return json.loads(self._identity_json)

    @property
    def layout(self) -> Qwen38PLELayout:
        return _layout(json.loads(self._layout_json))

    @property
    def row_count(self) -> int:
        return self._rows.numel()

    @property
    def device(self) -> torch.device:
        return self._rows.device

    def to(self, device: torch.device | str) -> FrozenVLLMDelta:
        """Return the same immutable snapshot or an independent device copy."""
        self._guard()
        target = _device(device)
        if target == self.device:
            return self
        # vLLM can load from inside inference_mode. Ordinary owned tensors keep
        # version counters even when their subsequent readers use that mode.
        with torch.inference_mode(False), torch.no_grad():
            tensors = tuple(value.to(device=target, copy=True) for value in self._tensors())
        return FrozenVLLMDelta(*tensors, self._identity_json, self._layout_json,
                               self.sha256, self.source_kind, self.owner_id)

    def lookup(self, native_ids: Tensor, *, owner_id: int = 0) -> Tensor:
        """Read FP32 [tokens,2560] from native global IDs [tokens,16].

        Native row zero is valid and maps to physical row one. Every head must
        use its own checkpoint-defined region. CUDA range checks are device
        assertions to avoid a host synchronization on every generated token.
        Empty Delta follows this same reader and returns fresh FP32 zeros.
        """
        _owner(owner_id)
        if owner_id != self.owner_id:
            raise ValueError("lookup owner does not match the snapshot")
        self._guard()
        if (not isinstance(native_ids, Tensor) or native_ids.layout != torch.strided
                or native_ids.dtype not in (torch.int32, torch.int64)
                or native_ids.ndim != 2 or native_ids.shape[1] != 16):
            raise ValueError("native IDs must be integer [tokens,16]")
        if native_ids.device != self.device:
            raise ValueError("native IDs and snapshot must share a device")
        valid = ((native_ids >= self._offsets) & (native_ids < self._ends)).all()
        message = "native ID lies outside its checkpoint-defined head region"
        if self.device.type == "cuda":
            torch._assert_async(valid, message)
        elif not bool(valid):
            raise ValueError(message)
        with torch.no_grad():
            physical = native_ids.reshape(-1).to(dtype=torch.int64) + 1
            if not self.row_count:
                return self._values.new_zeros((native_ids.shape[0], 2560))
            positions = torch.searchsorted(self._rows, physical)
            safe = positions.clamp_max(self.row_count - 1)
            hit = (positions < self.row_count) & (self._rows[safe] == physical)
            values = torch.where(hit[:, None], self._values[safe], 0.0)
            return values.reshape(native_ids.shape[0], 2560)


def save_vllm_snapshot(path: str | Path, *, state: BatchMemoryState | LazyDecayState,
                       layout: Qwen38PLELayout | Mapping[str, Any],
                       identity: Mapping[str, Any], owner_id: int = 0) -> dict:
    """Export effective Delta rows without changing the live evidence state."""
    _owner(owner_id)
    native_layout = _layout(layout)
    metadata = _identity(identity)
    if not isinstance(state, (BatchMemoryState, LazyDecayState)):
        raise TypeError("snapshot export requires BatchMemoryState or LazyDecayState")
    if state.batch_size != 1:
        raise ValueError("snapshot export requires exactly one owner; unpack first")
    if state.row_capacity != native_layout.row_capacity or state.width != native_layout.row_width:
        raise ValueError("state capacity or width differs from the pinned layout")
    if state.precision is not None:
        raise ValueError("the Qwen3.8 D1 snapshot does not accept precision-bearing states")
    _validate_tensors(state.rows, state.values, native_layout)
    counts = state.counts
    if (counts.dtype != torch.float32 or counts.shape != state.rows.shape
            or counts.device != state.rows.device or not bool(torch.isfinite(counts).all())
            or bool((counts < 0).any()) or bool((counts != counts.floor()).any())):
        raise ValueError("source state observation counts are invalid")
    with torch.inference_mode(False), torch.no_grad():
        if isinstance(state, LazyDecayState):
            # Recheck clocks, which callers might have mutated since construction.
            LazyDecayState(state.memory, state.block_clock, state.cumulative_log_alpha,
                           state.row_last_written_block, state.row_log_alpha_at_write)
            effective, _ = state._lookup_packed(state.rows)
            source_kind = "lazy_decay"
        else:
            effective = state.values
            source_kind = "batch"
        rows = state.rows.detach().to(device="cpu", copy=True).contiguous()
        values = effective.detach().to(device="cpu", copy=True).contiguous()
    _validate_tensors(rows, values, native_layout)
    payload = {
        "format": SNAPSHOT_FORMAT, "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "identity": metadata, "layout": _layout_dict(native_layout),
        "owner_id": owner_id, "batch_size": 1,
        "row_format": "physical_native_plus_one", "decay_materialized": True,
        "source_kind": source_kind, "rows": rows, "values": values,
    }
    target = Path(path)
    torch.save(payload, target)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(target.resolve()), "sha256": digest.hexdigest(),
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "snapshot_sha256": digest.hexdigest(),
            "expected_snapshot_identity": metadata,
            "expected_layout": _layout_dict(native_layout),
            "size_bytes": target.stat().st_size, "row_count": rows.numel(),
            "owner_id": owner_id, "source_kind": source_kind,
            "decay_materialized": True, "identity": metadata,
            "layout": _layout_dict(native_layout)}


def load_vllm_snapshot(path: str | Path, *, expected_sha256: str,
                       expected_identity: Mapping[str, Any],
                       expected_layout: Qwen38PLELayout | Mapping[str, Any],
                       expected_owner_id: int = 0,
                       device: torch.device | str = "cpu") -> FrozenVLLMDelta:
    """Check the pinned bytes and complete CPU payload before any device copy."""
    _owner(expected_owner_id)
    expected_digest = _sha256(expected_sha256)
    identity = _identity(expected_identity)
    native_layout = _layout(expected_layout)
    target = _device(device)
    # Hash and deserialize the same bytes, so pathname replacement cannot race
    # the integrity check. weights_only avoids unpickling arbitrary objects.
    data = Path(path).read_bytes()
    actual_digest = hashlib.sha256(data).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("snapshot SHA256 differs from the pinned digest")
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    del data
    if (not isinstance(payload, dict) or set(payload) != _PAYLOAD_FIELDS
            or payload.get("format") != SNAPSHOT_FORMAT
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != SNAPSHOT_SCHEMA_VERSION):
        raise ValueError("unsupported or malformed vLLM snapshot format")
    if (_json(_identity(payload["identity"])) != _json(identity)
            or _layout(payload["layout"]) != native_layout):
        raise ValueError("snapshot identity or layout differs from the pinned reader")
    _owner(payload["owner_id"])
    if (payload["owner_id"] != expected_owner_id
            or type(payload["batch_size"]) is not int or payload["batch_size"] != 1
            or payload["row_format"] != "physical_native_plus_one"
            or payload["decay_materialized"] is not True
            or payload["source_kind"] not in ("batch", "lazy_decay")):
        raise ValueError("snapshot owner, addressing, or materialization is invalid")
    _validate_tensors(payload["rows"], payload["values"], native_layout)
    with torch.inference_mode(False), torch.no_grad():
        rows = payload["rows"].detach().clone().contiguous()
        values = payload["values"].detach().clone().contiguous()
        offsets = torch.tensor(native_layout.head_offsets, dtype=torch.int64)
        ends = offsets + torch.tensor(native_layout.head_vocab_sizes, dtype=torch.int64)
    snapshot = FrozenVLLMDelta(rows, values, offsets, ends, _json(identity),
                              _json(_layout_dict(native_layout)), actual_digest,
                              payload["source_kind"], expected_owner_id)
    del payload
    return snapshot.to(target)

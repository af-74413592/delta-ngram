"""Scoped lookup overlay for the audited Megatron-SWIFT Qwen4Exp PLE.

No Transformers model loader, replacement hash, or replacement PLE is used.
The native embedding computes its IDs and finishes dequantized/TP-reduced T0
lookup. We add sparse delta at that boundary, before the native projections.

Attach only on the PP stage holding PLE. Every run must describe the exact
unpacked, sequence/context-parallel-gathered [B,T] input seen by PLE.compute.
Prefix tokens stay in input_ids; valid_mask selects positions allowed to read
delta (and physical_rows returned for a subsequent, separately masked write).
Rows are independent sequence/conv lanes; owner_ids map them to memory owners.
Evidence must also use a base-only runtime cache: removing this lookup overlay
cannot undo delta already encoded in a reused query KV/conv cache. Native hash
EOS resets are preserved; they do not create a new owner or reset SWIFT's
training convolution. Distinct owners therefore need separate native lanes.

Internal activation recomputation must be disabled for the initial runtime
acceptance. A supported checkpoint closure must call run_native itself with
the same immutable state snapshot on every replay. An unbound replay fails
instead of silently reading T0. This adapter does not implement TP/PP writer
gradient communication or cache ownership; those require runtime acceptance.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Literal, TypeVar

import torch
from torch import Tensor, nn

from .alpha_memory import LazyDecayState
from .batching import BatchMemoryState
from .qwen38_native import Qwen38PLELayout, add_native_ple_delta


Memory = BatchMemoryState | LazyDecayState
Result = TypeVar("Result")


@dataclass(frozen=True)
class NativePLECapture:
    """Native IDs and physical rows in *runtime lane* order, not owner order."""

    native_ids: Tensor
    physical_rows: Tensor
    owner_ids: Tensor
    valid_mask: Tensor
    base_head_rms: Tensor
    lookup_backend: str


@dataclass
class _Read:
    input_ids: Tensor
    valid_mask: Tensor
    owner_ids: Tensor
    mode: str
    state: Memory | None
    in_embedding: bool = False
    capture: NativePLECapture | None = None
    ple_completed: bool = False


@dataclass(frozen=True)
class _OwnerView:
    """Read selected/reordered owners without copying or detaching their state."""

    state: Memory
    owners: Tensor

    @property
    def width(self):
        return self.state.width

    @property
    def batch_size(self):
        return self.owners.numel()

    def lookup(self, physical_rows: Tensor):
        offsets = self.owners[:, None, None] * (self.state.row_capacity + 1)
        packed = torch.where(physical_rows.ne(0), physical_rows + offsets, 0)
        if isinstance(self.state, LazyDecayState):
            return self.state._lookup_packed(packed)
        return self.state.memory.lookup(packed)


class Qwen38NativePLEAdapter:
    """Install reversible hooks on one frozen, audited SWIFT PLE instance.

    Example::

        with Qwen38NativePLEAdapter(ple, layout) as adapter:
            result, addresses = adapter.run_native(
                lambda: model(...), input_ids=logical_tokens,
                valid_mask=logical_mask, owner_ids=owners, mode="query",
                state=memory)
            loss = result.loss
            loss.backward()  # ordinary autograd retains delta's history

    Evidence calls use mode="evidence", state=None. The backbone must already
    be frozen; the adapter verifies PLE parameters and never changes flags.
    Its hooks do not add parameters, buffers, or state_dict entries.
    """

    _marker = "_delta_ngram_qwen38_adapter"

    def __init__(self, ple_layer: nn.Module, layout: Qwen38PLELayout):
        if not isinstance(ple_layer, nn.Module) or not isinstance(layout, Qwen38PLELayout):
            raise TypeError("expected native PLE module and Qwen38PLELayout")
        embedding = getattr(ple_layer, "ple_embedding", None)
        if (not isinstance(embedding, nn.Module)
                or not callable(getattr(ple_layer, "compute", None))
                or getattr(ple_layer, "hidden_size", None) != layout.input_dim
                or getattr(ple_layer, "hc_count", None) != layout.hc_count):
            raise ValueError("module does not expose the audited SWIFT PLE interface")
        if any(parameter.requires_grad for parameter in ple_layer.parameters()):
            raise ValueError("freeze native PLE parameters before installing the adapter")
        for name, expected in (
            ("ngram_heads_offsets", layout.head_offsets),
            ("ngram_heads_vocab_sizes", layout.head_vocab_sizes),
        ):
            value = getattr(embedding, name, None)
            if (not isinstance(value, Tensor) or value.dtype != torch.long
                    or value.ndim != 1 or tuple(value.cpu().tolist()) != expected):
                raise ValueError(f"native {name} does not match the pinned layout")
        if (getattr(embedding, "ngram_size", None) != 3
                or getattr(embedding, "heads_per_ngram", None) != 8):
            raise ValueError("native ngram/head ordering differs from the pinned layout")
        if hasattr(embedding, self._marker):
            raise RuntimeError("this native embedding already has a delta adapter")
        self.ple_layer = ple_layer
        self.embedding = embedding
        self.layout = layout
        self._read: ContextVar[_Read | None] = ContextVar(
            f"qwen38_ple_read_{id(self)}", default=None)
        self._handles = []
        self._host_original = None
        self._host_had_instance_attribute = False
        self._host_instance_attribute = None
        self._closed = False
        self.backend = "host" if getattr(embedding, "cpu_offload", False) else "device"
        setattr(embedding, self._marker, self)
        try:
            self._handles.append(embedding.register_forward_pre_hook(
                self._embedding_enter, with_kwargs=True))
            self._handles.append(embedding.register_forward_hook(
                self._embedding_exit, with_kwargs=True, always_call=True))
            self._handles.append(ple_layer.register_forward_hook(self._ple_exit))
            if self.backend == "host":
                host_lookup = getattr(embedding, "_host_lookup", None)
                if not callable(host_lookup):
                    raise ValueError("CPU offload module has no native _host_lookup")
                self._host_original = host_lookup
                self._host_had_instance_attribute = "_host_lookup" in embedding.__dict__
                self._host_instance_attribute = embedding.__dict__.get("_host_lookup")
                embedding._host_lookup = self._host_lookup
            else:
                table = getattr(embedding, "ngram_embedding", None)
                if not isinstance(table, nn.Module):
                    raise ValueError("device PLE has no native ngram_embedding module")
                self._handles.append(table.register_forward_hook(
                    self._device_lookup, with_kwargs=True))
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        if self._closed:
            raise RuntimeError("adapter is closed")
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        """Remove hooks and restore the exact previous host lookup binding."""
        if self._read.get() is not None:
            raise RuntimeError("cannot close the adapter during a native read")
        if self._closed:
            return
        for handle in self._handles:
            handle.remove()
        if self._host_original is not None:
            if self._host_had_instance_attribute:
                self.embedding._host_lookup = self._host_instance_attribute
            elif "_host_lookup" in self.embedding.__dict__:
                delattr(self.embedding, "_host_lookup")
        if getattr(self.embedding, self._marker, None) is self:
            delattr(self.embedding, self._marker)
        self._closed = True

    def run_native(
        self, call: Callable[[], Result], *, input_ids: Tensor, valid_mask: Tensor,
        owner_ids: Tensor, mode: Literal["evidence", "query"], state: Memory | None = None,
    ) -> tuple[Result, NativePLECapture]:
        """Run one complete native PLE invocation, including the surrounding model.

        Keep backward outside this call. A checkpoint may replay this whole
        function; a native-internal replay without this binding fails closed.
        No writes, owner-clock advancement, or evidence feature capture happen
        here. Use capture.physical_rows with an explicit new-evidence write mask.
        """
        if self._closed:
            raise RuntimeError("adapter is closed")
        if self._read.get() is not None:
            raise RuntimeError("nested reads on one PLE adapter are unsupported")
        if (not isinstance(input_ids, Tensor) or input_ids.dtype != torch.long
                or input_ids.ndim != 2 or min(input_ids.shape) < 1):
            raise ValueError("input_ids must be nonempty int64 [B,T]")
        if (not isinstance(valid_mask, Tensor) or valid_mask.dtype != torch.bool
                or valid_mask.shape != input_ids.shape
                or valid_mask.device != input_ids.device):
            raise ValueError("valid_mask must be boolean [B,T] on the token device")
        if (not isinstance(owner_ids, Tensor) or owner_ids.dtype != torch.long
                or owner_ids.shape != (input_ids.shape[0],)
                or owner_ids.device != input_ids.device or bool((owner_ids < 0).any())):
            raise ValueError("owner_ids must be nonnegative int64 [B] on the token device")
        if mode == "evidence":
            if state is not None:
                raise ValueError("evidence must not receive delta state")
        elif mode == "query":
            if not isinstance(state, (BatchMemoryState, LazyDecayState)):
                raise TypeError("query requires BatchMemoryState or LazyDecayState")
            if (state.row_capacity != self.layout.row_capacity
                    or state.width != self.layout.row_width
                    or state.rows.device != input_ids.device
                    or bool((owner_ids >= state.batch_size).any())
                    or state.precision is not None):
                raise ValueError("query state namespace, owner, device, or width mismatch")
            if state.values.requires_grad and (
                    not torch.is_grad_enabled() or torch.is_inference_mode_enabled()):
                raise RuntimeError("training delta requires a grad-enabled native read")
        else:
            raise ValueError("mode must be evidence or query")
        read = _Read(input_ids.detach().clone(), valid_mask.detach().clone(),
                     owner_ids.detach().clone(), mode, state)
        token = self._read.set(read)
        try:
            result = call()
            if read.capture is None or not read.ple_completed:
                raise RuntimeError("call must execute exactly one complete native PLE forward")
            return result, read.capture
        finally:
            self._read.reset(token)

    def _current(self) -> _Read:
        read = self._read.get()
        if read is None:
            raise RuntimeError(
                "unbound native PLE read/recompute; replay a run_native closure")
        return read

    def _embedding_enter(self, module, args, kwargs):
        read = self._current()
        actual = args[0] if args else kwargs.get("input_ids")
        if (not isinstance(actual, Tensor) or actual.dtype != torch.long
                or actual.device != read.input_ids.device
                or not torch.equal(actual, read.input_ids)):
            raise ValueError(
                "native token layout differs from bound unpacked/gathered input_ids")
        if read.in_embedding or read.capture is not None:
            raise RuntimeError("one run_native call must have one native lookup")
        read.in_embedding = True

    def _embedding_exit(self, module, args, kwargs, output):
        read = self._read.get()
        if read is not None:
            read.in_embedding = False

    def _ple_exit(self, module, args, output):
        read = self._current()
        if read.capture is None or read.ple_completed:
            raise RuntimeError("native PLE did not execute exactly one bound lookup")
        if (read.mode == "query" and read.state.values.requires_grad
                and (not isinstance(output, Tensor) or not output.requires_grad)):
            raise RuntimeError("native PLE output lost the training delta gradient")
        read.ple_completed = True

    def _host_lookup(self, native_ids):
        # _host_lookup finishes native CPU transfer and TP all_reduce first.
        self._current()
        base = self._host_original(native_ids)
        return self._overlay(native_ids, base)

    def _device_lookup(self, module, args, kwargs, output):
        native_ids = args[0] if args else kwargs.get("input_")
        if native_ids is None:
            native_ids = kwargs.get("input")
        return self._overlay(native_ids, output)

    def _overlay(self, native_ids: Tensor, base: Tensor) -> Tensor:
        read = self._current()
        if not read.in_embedding or read.capture is not None:
            raise RuntimeError("lookup must occur once inside native embedding.forward")
        if (not isinstance(native_ids, Tensor)
                or native_ids.device != read.input_ids.device
                or native_ids.shape != (*read.input_ids.shape, 16)):
            raise ValueError("native lookup must expose global IDs [B,T,16]")
        physical = self.layout.physical_rows(native_ids, read.valid_mask)
        expected = (*read.input_ids.shape, self.layout.input_dim)
        headed = (*read.input_ids.shape, 16, self.layout.row_width)
        if (not isinstance(base, Tensor) or base.device != native_ids.device
                or base.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or tuple(base.shape) not in (expected, headed)):
            raise ValueError("lookup must return dequantized complete T0 heads")
        flat = base.detach().reshape(expected)
        if read.mode == "evidence":
            value = flat
        else:
            value = add_native_ple_delta(
                flat, native_ids, read.valid_mask,
                _OwnerView(read.state, read.owner_ids), self.layout)
        with torch.no_grad():
            selected = flat.reshape(headed)[read.valid_mask].float()
            rms = (selected.square().mean((0, 2)).sqrt() if selected.shape[0]
                   else flat.new_zeros(16, dtype=torch.float32))
        read.capture = NativePLECapture(
            native_ids.detach().clone(), physical.detach().clone(),
            read.owner_ids, read.valid_mask, rms, self.backend)
        return value.reshape_as(base)

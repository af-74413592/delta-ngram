from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol
import torch
from torch import Tensor, nn
from .state import SparseMemoryState
from .ngram_lookup import ngram_rows


class AddressReadAdapter(Protocol):
    """Boundary for addressing and delta readout, separate from write policy.

    A future native adapter must specify pretrained table addressing/readout;
    this external residual prototype is not a native-table implementation.
    """

    def addresses(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor: ...
    def read_residual(self, state: SparseMemoryState, rows: Tensor) -> Tensor: ...
    def config(self) -> dict: ...


@dataclass(frozen=True)
class ExternalHashAdapter:
    buckets: int
    orders: tuple[int, ...]

    def addresses(self, input_ids, attention_mask):
        return ngram_rows(input_ids.to(dtype=torch.long), attention_mask, self.buckets, self.orders)

    def read_residual(self, state, rows):
        values, _ = state.lookup(rows)
        return values.sum(dim=-2) / math.sqrt(len(self.orders))

    def config(self):
        return {"backend": "external_hash", "hash_version": 1,
                "buckets": self.buckets, "orders": list(self.orders),
                "implicit_base": "zero", "scale": 1 / math.sqrt(len(self.orders))}

class FrozenMemoryReader(nn.Module):
    """Frozen HF reader with per-call, fixed before-block residual injection.

    No request-specific state or addresses persist on the module. The initial
    implementation disables activation checkpointing and ordinary KV caching.
    Query forwards retain autograd through delta; evidence encoding does not.
    """

    def __init__(self, model: nn.Module, buckets: int = 16384,
                 orders: tuple[int, ...] = (2, 3), layer_index: int = 1, *,
                 backend: str = "external_hash") -> None:
        super().__init__()
        if backend == "native":
            raise NotImplementedError("native table addressing/readout is not implemented")
        if backend != "external_hash":
            raise ValueError(f"unsupported reader backend: {backend}")
        if buckets < 1 or not orders or any(order < 1 for order in orders):
            raise ValueError("invalid hash configuration")
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None or not 0 <= layer_index < len(layers):
            raise ValueError("reader requires model.model.layers[layer_index]")
        output = model.get_output_embeddings()
        if output is None or not hasattr(output, "weight") or output.weight.ndim != 2:
            raise ValueError("reader requires an explicit output embedding module")
        self.model = model
        self.buckets, self.orders, self.layer_index = buckets, tuple(orders), layer_index
        self.backend = backend
        self.adapter: AddressReadAdapter = ExternalHashAdapter(buckets, self.orders)
        self.width = int(output.weight.shape[1])
        self.model.requires_grad_(False)
        self.model.eval()
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = False

    def train(self, mode: bool = True) -> FrozenMemoryReader:
        super().train(mode)
        self.model.eval()
        return self

    def _inputs(self, input_ids: Tensor, attention_mask: Tensor | None) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must be integer [B, T]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape or attention_mask.device != input_ids.device:
            raise ValueError("attention mask must match input IDs")
        if getattr(self.model, "is_gradient_checkpointing", False):
            raise RuntimeError("scoped memory hooks require activation checkpointing disabled")
        self.model.eval()
        # Hash arithmetic must stay int64 even when a caller supplies int32 IDs.
        rows = self.adapter.addresses(input_ids, attention_mask)
        return attention_mask, rows

    def encode_evidence(self, input_ids: Tensor,
                        attention_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return actual LM-head inputs and addresses from the unpatched model."""
        attention_mask, rows = self._inputs(input_ids, attention_mask)
        captured: list[Tensor] = []

        def capture(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("input", kwargs.get("hidden_states"))
            if hidden is None:
                raise RuntimeError("could not capture the output embedding input")
            captured.append(hidden.detach())

        handle = self.model.get_output_embeddings().register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with torch.no_grad():
                self.model(input_ids=input_ids, attention_mask=attention_mask,
                           use_cache=False, return_dict=True)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape != (*input_ids.shape, self.width):
            raise RuntimeError("output embedding must receive the full evidence sequence exactly once")
        return captured[0], rows

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None,
                state: SparseMemoryState):
        return self.forward_with_residual(input_ids, attention_mask, state, None)

    def forward_with_residual(self, input_ids: Tensor, attention_mask: Tensor | None,
                              state: SparseMemoryState, extra_residual: Tensor | None):
        """External-backend diagnostic residual at the existing injection site.

        The default path retains the executed D1 addition/cast order. Additional
        residuals keep their gradient and are not stored on the reader or state.
        """
        attention_mask, rows = self._inputs(input_ids, attention_mask)
        if state.width != self.width or state.rows.device != input_ids.device:
            raise ValueError("state width/device differs from query")
        if extra_residual is not None:
            if self.backend != "external_hash":
                raise NotImplementedError("extra residual injection requires external_hash")
            if (extra_residual.shape != (*input_ids.shape, self.width)
                    or extra_residual.dtype != torch.float32 or extra_residual.device != input_ids.device):
                raise ValueError("extra_residual must be float32 [B, T, width] on the input device")

        def inject(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            if hidden.shape != (*input_ids.shape, self.width):
                raise RuntimeError("decoder hidden shape differs from the query addresses")
            residual = self.adapter.read_residual(state, rows)
            updated = hidden + residual.to(dtype=hidden.dtype)
            if extra_residual is not None:
                updated = updated + extra_residual.to(dtype=hidden.dtype)
            if args:
                return (updated, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": updated}

        handle = self.model.model.layers[self.layer_index].register_forward_pre_hook(inject, with_kwargs=True)
        try:
            return self.model(input_ids=input_ids, attention_mask=attention_mask,
                              use_cache=False, return_dict=True)
        finally:
            handle.remove()

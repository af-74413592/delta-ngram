"""Frozen A800 PLE host transport for the source-pinned nested TP embedding.

The A800 bridge has no native CPU-offload branch. Only its nested vocabulary
lookup is replaced after checkpoint loading; the original native hash, head
order, PLE/SP wrapper, and projection math stay in their installed modules.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

VOCAB_SOURCE_SHA256 = "fd9c4191a59d6b9a181eec1e208de56eae4678536f290c6d6aff8a353992e638"
PLE_SOURCE_SHA256 = "94d709eba089228ed07b7325dea75427189fe7fd395d82f6e79c0614eeaa2a71"


def require(condition, message):
    if not condition:
        raise ValueError(message)


class FrozenHostVocabShard(nn.Module):
    """Exact BF16 CPU rows, selected-row transfer, and the original TP sum."""

    def __init__(self, weight, *, start, end, total, reduce_sum: Callable):
        super().__init__()
        require(isinstance(weight, torch.Tensor) and weight.device.type == "cpu"
                and weight.dtype == torch.bfloat16 and weight.ndim == 2
                and not weight.requires_grad and weight.grad is None,
                "host shard must be the already-loaded frozen BF16 CPU table")
        require(0 <= start < end <= total and end - start == weight.shape[0], "invalid native TP vocab partition")
        # Deliberately unregistered: nn.Module._apply must never move this table
        # during Swift's m.cuda() / Float16Module wrapping.
        object.__setattr__(self, "host_table", weight.detach())
        self.vocab_start_index, self.vocab_end_index = int(start), int(end)
        self.num_embeddings, self.num_embeddings_per_partition = int(total), int(end - start)
        self.embedding_dim = int(weight.shape[1])
        self.reduce_sum = reduce_sum
        self._identity = (self.host_table.data_ptr(), self.host_table._version)

    def assert_frozen(self):
        require(not list(self.parameters()) and not list(self.buffers()),
                "host T0 must not be registered for CUDA movement")
        require(self.host_table.device.type == "cpu" and self.host_table.dtype == torch.bfloat16
                and not self.host_table.requires_grad and self.host_table.grad is None
                and (self.host_table.data_ptr(), self.host_table._version) == self._identity,
                "frozen BF16 CPU T0 changed or moved")

    @torch.no_grad()
    def forward(self, input_):
        self.assert_frozen()
        require(input_.dtype == torch.long, "native PLE IDs must remain int64")
        ids = input_.detach().to(device="cpu")
        mask = (ids < self.vocab_start_index) | (ids >= self.vocab_end_index)
        local = (ids - self.vocab_start_index).masked_fill(mask, 0)
        selected = F.embedding(local, self.host_table)
        selected[mask, :] = 0
        selected = selected.to(device=input_.device)
        result = self.reduce_sum(selected)
        require(result.shape == (*input_.shape, self.embedding_dim) and result.dtype == torch.bfloat16
                and result.device == input_.device, "native host lookup TP output changed")
        return result


@dataclass
class HostTableBinding:
    embedding: nn.Module
    original: nn.Module
    replacement: FrozenHostVocabShard
    identity: dict

    def assert_frozen(self):
        require(self.embedding.ngram_embedding is self.replacement, "native PLE host transport was replaced")
        self.replacement.assert_frozen()
        require(self.original.weight.device.type == "cpu" and not self.original.weight.requires_grad
                and self.original.weight.data_ptr() == self.replacement.host_table.data_ptr(),
                "loaded T0 storage was replaced or copied")

    def restore(self):
        self.assert_frozen()
        self.embedding.ngram_embedding = self.original


def install_a800_host_table(ple):
    """Install after all pretrained weights load and before any model .cuda()."""
    from megatron.core.tensor_parallel import VocabParallelEmbedding
    from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region

    embedding, table = ple.ple_embedding, ple.ple_embedding.ngram_embedding
    require(type(table) is VocabParallelEmbedding
            and hashlib.sha256(Path(inspect.getfile(VocabParallelEmbedding)).read_bytes()).hexdigest() == VOCAB_SOURCE_SHA256
            and hashlib.sha256(Path(inspect.getfile(type(embedding))).read_bytes()).hexdigest() == PLE_SOURCE_SHA256,
            "A800 native PLE/vocabulary implementation differs from the audited source")
    require(not table.reduce_scatter_embeddings and table.tp_group.size() == 2
            and not hasattr(embedding, "cpu_offload"), "expected original A800 TP2 embedding without a host branch")
    group = table.tp_group
    replacement = FrozenHostVocabShard(table.weight, start=table.vocab_start_index, end=table.vocab_end_index,
        total=table.num_embeddings,
        reduce_sum=lambda selected: reduce_from_tensor_model_parallel_region(selected, group=group))
    embedding.ngram_embedding = replacement
    identity = {"protocol": "qwen38_a800_frozen_host_vocab_v1", "original_vocab_source_sha256": VOCAB_SOURCE_SHA256,
                "original_ple_source_sha256": PLE_SOURCE_SHA256,
                "transport": "CPU_BF16_selected_rows_H2D_then_original_TP_sum",
                "native_hash_unchanged": True, "host_storage_reused_without_copy": True,
                "host_table_registered_parameter_or_buffer": False,
                "H200_numerical_equivalence_claimed": False}
    binding = HostTableBinding(embedding, table, replacement, identity)
    binding.assert_frozen()
    return binding

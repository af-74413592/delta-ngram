from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/native_qwen38"))
from qwen38_native_a800_host import FrozenHostVocabShard


def test_host_tp_shards_sum_to_full_native_embedding_and_reuse_storage():
    torch.manual_seed(29)
    full = torch.randn(20, 160, dtype=torch.bfloat16)
    ids = torch.tensor([[[0, 9, 10, 19], [3, 17, 3, 2]], [[19, 0, 11, 8], [10, 10, 9, 9]]])
    pieces = []
    for start in (0, 10):
        weight = torch.nn.Parameter(full[start:start + 10], requires_grad=False)
        shard = FrozenHostVocabShard(weight, start=start, end=start + 10, total=20, reduce_sum=lambda value: value)
        assert shard.host_table.data_ptr() == weight.data_ptr()
        assert not list(shard.parameters()) and not list(shard.buffers())
        pieces.append(shard(ids))
    assert torch.equal(pieces[0] + pieces[1], torch.nn.functional.embedding(ids, full))


def test_module_apply_does_not_touch_host_table():
    weight = torch.nn.Parameter(torch.randn(10, 160, dtype=torch.bfloat16), requires_grad=False)
    shard = FrozenHostVocabShard(weight, start=0, end=10, total=20, reduce_sum=lambda value: value)
    parent = torch.nn.ModuleDict({"host": shard, "projection": torch.nn.Linear(160, 4, dtype=torch.bfloat16)})
    pointer, version = weight.data_ptr(), weight._version
    parent.to(dtype=torch.float32)
    assert parent["projection"].weight.dtype == torch.float32
    assert shard.host_table.dtype == torch.bfloat16 and shard.host_table.device.type == "cpu"
    assert shard.host_table.data_ptr() == pointer and shard.host_table._version == version
    shard.assert_frozen()


def test_mutated_host_storage_is_rejected():
    weight = torch.nn.Parameter(torch.ones(10, 160, dtype=torch.bfloat16), requires_grad=False)
    shard = FrozenHostVocabShard(weight, start=0, end=10, total=20, reduce_sum=lambda value: value)
    with torch.no_grad():
        weight[0, 0] = 2
    with pytest.raises(ValueError, match="changed or moved"):
        shard(torch.zeros((1, 1, 16), dtype=torch.long))


def test_trainable_host_table_is_rejected():
    with pytest.raises(ValueError, match="frozen"):
        FrozenHostVocabShard(torch.nn.Parameter(torch.ones(10, 160, dtype=torch.bfloat16)),
                            start=0, end=10, total=20, reduce_sum=lambda value: value)

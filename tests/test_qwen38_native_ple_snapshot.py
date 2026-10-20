from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.state import SparseMemoryState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
from qwen38_native_ple_snapshot import capture_native_ple_snapshot, save_native_ple_snapshot


class Layout:
    @staticmethod
    def physical_rows(ids, valid):
        return torch.where(valid[..., None], ids, 0)


class Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(8, dtype=torch.bfloat16), requires_grad=False)
        self.eps, self.group_size = 1e-6, 4


class PLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size, self.hc_count, self.short_conv_state_len = 4, 2, 2
        self.key_proj = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
        self.value_proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.norm_key, self.norm_query, self.norm_conv = Norm(), Norm(), Norm()
        self.conv1d = nn.Conv1d(8, 8, 3, groups=8, bias=False, dtype=torch.bfloat16)
        self.calls = 0
        self.requires_grad_(False)

    def compute(self, hidden_states, input_ids):
        self.calls += 1
        embedding = self.adapter._overlay(input_ids, self.base)
        result = hidden_states + self.key_proj(embedding)
        self.last_return = result
        return result


class Adapter:
    def __init__(self, ple, state, valid):
        self.ple_layer, self.state, self.valid_mask = ple, state, valid
        self.layout = Layout()
        ple.adapter = self

    def _overlay(self, native_ids, base):
        delta, _ = self.state.lookup(self.layout.physical_rows(native_ids, self.valid_mask))
        return (base.float() + delta.flatten(-2)).bfloat16()


def fixture():
    valid = torch.tensor([[True, True, False]])
    ids = torch.tensor([[[1, 2], [2, 9], [1, 2]]])
    values = torch.tensor([[.004, .008], [.012, .016]], requires_grad=True)
    batch = BatchMemoryState(SparseMemoryState(torch.tensor([1, 2]), values, torch.ones(2)), 1, 100)
    state = LazyDecayState(batch, torch.tensor([1]), torch.tensor([-0.6931471805599453], dtype=torch.float64),
                           torch.tensor([0, 0]), torch.zeros(2, dtype=torch.float64))
    ple = PLE()
    ple.base = torch.full((1, 3, 4), .25, dtype=torch.bfloat16)
    adapter = Adapter(ple, state, valid)
    return adapter, state, valid, ids, torch.ones((1, 3, 8), dtype=torch.bfloat16)


def capture(adapter, state, valid):
    return capture_native_ple_snapshot(
        adapter, state=state, valid_mask=valid, owner_ids=torch.tensor([0]),
        runtime_identity_sha256="a" * 64, ple_source_sha256="b" * 64,
        metadata={"delta_origin": "calibrated_initial_writer", "source": "books"})


def test_snapshot_uses_effective_lazy_delta_and_preserves_output_and_graph():
    adapter, state, valid, ids, hidden = fixture()
    original_compute, original_overlay = adapter.ple_layer.compute, adapter._overlay
    with capture(adapter, state, valid) as snap:
        result = adapter.ple_layer.compute(hidden, ids)
        assert result is adapter.ple_layer.last_return
    assert adapter.ple_layer.calls == 1
    assert result.requires_grad
    assert adapter.ple_layer.compute == original_compute
    assert adapter._overlay == original_overlay
    assert "compute" not in adapter.ple_layer.__dict__
    assert "_overlay" not in adapter.__dict__
    assert snap["capture_complete"]
    assert not snap["cotangents_captured"]
    assert snap["rows"].tolist() == [1, 2, 9]
    torch.testing.assert_close(snap["delta"][:2], state.values.detach() * .5)
    assert not bool(snap["delta"][2].ne(0).any())
    assert not bool(snap["physical_rows"][0, 2].ne(0).any())
    result.float().sum().backward()
    assert state.values.grad is not None
    assert not snap["delta"].requires_grad


def test_existing_overlay_wrapper_is_restored_by_identity():
    adapter, state, valid, ids, hidden = fixture()
    original = adapter._overlay
    wrapper = lambda native_ids, base: original(native_ids, base)
    adapter._overlay = wrapper
    with capture(adapter, state, valid):
        adapter.ple_layer.compute(hidden, ids)
    assert adapter._overlay is wrapper


def test_repeated_forward_and_exception_restore_hooks():
    adapter, state, valid, ids, hidden = fixture()
    original = adapter.ple_layer.compute
    with pytest.raises(ValueError, match="multiple"):
        with capture(adapter, state, valid):
            adapter.ple_layer.compute(hidden, ids)
            adapter.ple_layer.compute(hidden, ids)
    assert adapter.ple_layer.compute == original
    assert "_overlay" not in adapter.__dict__
    with pytest.raises(RuntimeError, match="user failure"):
        with capture(adapter, state, valid):
            raise RuntimeError("user failure")
    assert adapter.ple_layer.compute == original


def test_save_roundtrips_and_refuses_overwrite(tmp_path):
    adapter, state, valid, ids, hidden = fixture()
    with capture(adapter, state, valid) as snap:
        adapter.ple_layer.compute(hidden, ids)
    path = tmp_path / "snapshot.pt"
    receipt = save_native_ple_snapshot(snap, path)
    assert receipt["bytes"] > 0 and len(receipt["sha256"]) == 64
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["metadata"]["source"] == "books"
    torch.testing.assert_close(loaded["delta"], snap["delta"])
    with pytest.raises(ValueError, match="overwrite"):
        save_native_ple_snapshot(snap, path)


def test_missing_forward_or_wrong_owner_is_rejected():
    adapter, state, valid, ids, hidden = fixture()
    with pytest.raises(ValueError, match="exactly one"):
        with capture(adapter, state, valid):
            pass
    with pytest.raises(ValueError, match="owner zero"):
        with capture_native_ple_snapshot(
            adapter, state=state, valid_mask=valid, owner_ids=torch.tensor([1]),
            runtime_identity_sha256="a" * 64, ple_source_sha256="b" * 64,
            metadata={"delta_origin": "calibrated_initial_writer"}):
            pass

from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from delta_ngram.batching import BatchMemoryState
from delta_ngram.state import SparseMemoryState
from delta_ngram.qwen38_adapter import _OwnerView
from delta_ngram.qwen38_precision import (
    Qwen38SplitPLEPrecision, _FrozenFP32Linear, _ieee_matmul, _native_body_from_projections,
    depthwise_causal_fp32, grouped_rms_fp32, signal_statistics, split_projections,
)


class Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(8, dtype=torch.bfloat16), requires_grad=False)
        self.eps, self.group_size = 1e-6, 4

    def forward(self, x):
        return grouped_rms_fp32(x, self).to(x.dtype)


class Linear(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight.bfloat16(), requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight), None


class Embedding(nn.Module):
    def forward(self, input_ids):
        return self.adapter._overlay(self.ids, self.base)


class Layout:
    @staticmethod
    def physical_rows(ids, mask):
        return torch.where(mask[..., None], ids, 0)


class PLE(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size, self.hc_count, self.short_conv_state_len = 4, 2, 2
        self.key_proj = Linear(torch.cat([torch.eye(4), torch.eye(4) * .5]))
        self.value_proj = Linear(torch.eye(4) * .75)
        self.norm_key, self.norm_query, self.norm_conv = Norm(), Norm(), Norm()
        self.conv1d = nn.Conv1d(8, 8, 3, groups=8, bias=False, dtype=torch.bfloat16)
        self.conv1d.weight.data.fill_(.002)
        self.ple_embedding = Embedding()
        self.requires_grad_(False)

    def _short_conv(self, x):
        return F.silu(self.conv1d(F.pad(x.transpose(1, 2), (2, 0)))).transpose(1, 2)

    def compute(self, hidden_states, input_ids):
        embeddings = self.ple_embedding(input_ids)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)
        return _native_body_from_projections(hidden_states, key, value, self)


class Adapter:
    def __init__(self, state, owner_ids=None):
        self.ple_layer = PLE()
        self.layout = Layout()
        self.ple_layer.ple_embedding.adapter = self
        self.ple_layer.ple_embedding.ids = torch.tensor([[[1, 2], [2, 3], [3, 1]]])
        self.ple_layer.ple_embedding.base = torch.tensor(
            [[[.25, .5, .25, .5], [.5, .25, .5, .25], [.25, .25, .5, .5]]], dtype=torch.bfloat16)
        self.read = SimpleNamespace(
            state=state, owner_ids=torch.tensor([0]) if owner_ids is None else owner_ids,
            valid_mask=torch.tensor([[True, True, False]]), mode="query")
        self.calls = 0

    def _current(self):
        return self.read

    def _overlay(self, ids, base):
        self.calls += 1
        if self.read.mode == "evidence":
            return base
        physical = self.layout.physical_rows(ids, self.read.valid_mask)
        delta, _ = _OwnerView(self.read.state, self.read.owner_ids).lookup(physical)
        return (base.float() + delta.flatten(-2)).bfloat16()


def fixture(scale=1e-5):
    values = torch.tensor([[1., -2.], [-1., 2.], [3., 1.]]) * scale
    values.requires_grad_()
    state = BatchMemoryState(SparseMemoryState(torch.tensor([1, 2, 3]), values, torch.ones(3)), 1, 100)
    adapter = Adapter(state)
    hidden = torch.tensor(
        [[[1., 2., 3., 4., 2., 3., 4., 1.],
          [3., 2., 1., 4., 1., 3., 2., 4.],
          [1., 4., 2., 3., 3., 1., 4., 2.]]], dtype=torch.bfloat16)
    return adapter, state, hidden, torch.tensor([[4, 5, 6]])


def test_split_projection_keeps_sub_bf16_delta_and_backpropagates_before_cast():
    adapter, state, hidden, ids = fixture()
    ple = adapter.ple_layer
    base = ple.ple_embedding.base
    dense, _ = state.lookup(ple.ple_embedding.ids)
    assert torch.equal((base.float() + dense.flatten(-2)).bfloat16(), base)
    key0, _ = ple.key_proj(base)
    value0, _ = ple.value_proj(base)
    key, value, key_delta, value_delta = split_projections(key0, value0, dense.flatten(-2), ple)
    assert key.dtype == value.dtype == torch.float32
    assert bool((key - key0.float()).ne(0).any())
    (key.square().sum() + value.square().sum()).backward()
    assert state.values.grad is not None and bool(state.values.grad.ne(0).any())
    assert not any(p.grad is not None for p in ple.parameters())
    assert signal_statistics(key_delta, key - key0.float())["cosine"] > .99999


def test_context_preserves_native_lookup_and_final_dtype_and_restores_methods():
    adapter, state, hidden, ids = fixture()
    ple = adapter.ple_layer
    compute, overlay = ple.compute, adapter._overlay
    before = {name: p.detach().clone() for name, p in ple.named_parameters()}
    cutpoints, reports = [], []
    with Qwen38SplitPLEPrecision(
            adapter, observer=reports.append, compare_native_zero=True,
            delta_observer=cutpoints.append) as precision:
        result = ple.compute(hidden, ids)
        assert result.dtype == torch.bfloat16 and result.requires_grad
        assert adapter.calls == 1
        assert precision.identity()["protocol"].endswith("_v1")
        assert cutpoints[0]["dense_delta"].requires_grad
        cutpoints[0]["dense_delta"].retain_grad()
        result.float().square().sum().backward()
        assert cutpoints[0]["dense_delta"].grad is not None
    assert ple.compute == compute and adapter._overlay == overlay
    assert "compute" not in ple.__dict__ and "_overlay" not in adapter.__dict__
    assert reports[0]["fp32_body_dtype"] == "torch.float32"
    assert "empty_baseline_comparison" in reports[0]
    assert reports[0]["final_output_cast"]["relative_l2_error"] is not None
    assert state.values.grad is not None
    for name, p in ple.named_parameters():
        assert torch.equal(p, before[name]) and p.grad is None


def test_zero_query_and_evidence_use_same_numerical_path():
    adapter, state, hidden, ids = fixture(0)
    with Qwen38SplitPLEPrecision(adapter):
        query = adapter.ple_layer.compute(hidden, ids)
        adapter.read.mode, adapter.read.state = "evidence", None
        evidence = adapter.ple_layer.compute(hidden, ids)
    assert torch.equal(query, evidence)
    assert query.requires_grad and not evidence.requires_grad


def test_empty_native_identity_is_not_falsely_assumed():
    adapter, state, hidden, ids = fixture(0)
    native = adapter.ple_layer.compute(hidden, ids)
    reports = []
    with Qwen38SplitPLEPrecision(adapter, observer=reports.append, compare_native_zero=True):
        changed = adapter.ple_layer.compute(hidden, ids)
    assert reports[0]["empty_baseline_comparison"]["bitwise_equal"] == torch.equal(native, changed)
    assert not torch.equal(native, changed)


def test_direct_depthwise_conv_matches_independent_fp32_conv():
    adapter, state, hidden, ids = fixture()
    ple = adapter.ple_layer
    x = hidden.float() / 7
    expected = F.conv1d(F.pad(x.transpose(1, 2), (2, 0)), ple.conv1d.weight.float(), groups=8).transpose(1, 2)
    torch.testing.assert_close(depthwise_causal_fp32(x, ple), expected, atol=1e-8, rtol=1e-6)


def test_exception_and_duplicate_install_restore_original_bindings():
    adapter, state, hidden, ids = fixture()
    original = adapter.ple_layer.compute
    with pytest.raises(RuntimeError, match="test failure"):
        with Qwen38SplitPLEPrecision(adapter):
            with pytest.raises(ValueError, match="already installed"):
                with Qwen38SplitPLEPrecision(adapter):
                    pass
            raise RuntimeError("test failure")
    assert adapter.ple_layer.compute == original
    assert not hasattr(adapter.ple_layer, Qwen38SplitPLEPrecision._marker)


def test_owner_selection_and_mask_match_sparse_memory_semantics():
    first = SparseMemoryState(torch.tensor([1, 2, 3]), torch.full((3, 2), .1), torch.ones(3))
    second_values = torch.tensor([[.001, .002], [.003, .004], [.005, .006]], requires_grad=True)
    second = SparseMemoryState(torch.tensor([1, 2, 3]), second_values, torch.ones(3))
    state = BatchMemoryState.from_states([first, second], 100)
    adapter, _, hidden, ids = fixture()
    adapter.read.state, adapter.read.owner_ids = state, torch.tensor([1])
    observations = []
    with Qwen38SplitPLEPrecision(adapter, delta_observer=observations.append):
        result = adapter.ple_layer.compute(hidden, ids)
    dense = observations[0]["dense_delta"]
    torch.testing.assert_close(dense[0, 0], second_values[:2].flatten())
    assert not bool(dense[0, -1].ne(0).any())
    result.float().sum().backward()
    assert second_values.grad is not None and bool(second_values.grad.ne(0).any())


def test_cpu_autocast_cannot_downcast_delta_branch():
    adapter, state, hidden, ids = fixture()
    base_key, _ = adapter.ple_layer.key_proj(adapter.ple_layer.ple_embedding.base)
    base_value, _ = adapter.ple_layer.value_proj(adapter.ple_layer.ple_embedding.base)
    dense, _ = state.lookup(adapter.ple_layer.ple_embedding.ids)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        outputs = split_projections(base_key, base_value, dense.flatten(-2), adapter.ple_layer)
    assert all(value.dtype == torch.float32 for value in outputs)


def test_frozen_fp32_linear_backward_matches_independent_matrix_adjoint():
    generator = torch.Generator().manual_seed(43)
    x = torch.randn((2, 3, 7), generator=generator, requires_grad=True)
    weight = torch.randn((5, 7), generator=generator)
    cotangent = torch.randn((2, 3, 5), generator=generator)
    output = _FrozenFP32Linear.apply(x, weight)
    gradient, = torch.autograd.grad((output * cotangent).sum(), (x,))
    expected = torch.einsum("bto,oi->bti", cotangent.double(), weight.double()).float()
    torch.testing.assert_close(gradient, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(output, x.detach() @ weight.T)


@pytest.mark.parametrize("modern", [False, True])
def test_ieee_dispatch_uses_one_backend_api_and_restores_after_exception(monkeypatch, modern):
    backend = (SimpleNamespace(fp32_precision="tf32") if modern
               else SimpleNamespace(allow_tf32=True))
    monkeypatch.setattr(torch.backends.cuda, "matmul", backend)
    with pytest.raises(RuntimeError, match="dispatch"):
        with _ieee_matmul(torch.device("cuda")):
            if modern:
                assert backend.fp32_precision == "ieee" and not hasattr(backend, "allow_tf32")
            else:
                assert backend.allow_tf32 is False and not hasattr(backend, "fp32_precision")
            raise RuntimeError("dispatch")
    assert (backend.fp32_precision == "tf32") if modern else backend.allow_tf32

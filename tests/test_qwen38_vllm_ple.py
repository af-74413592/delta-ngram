"""CPU checks for the split numerical boundary and native-cache integration."""
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from delta_ngram.qwen38_native import Qwen38PLELayout
from delta_ngram.qwen38_precision import ple_body_fp32
import delta_ngram.qwen38_vllm_ple as split


class TinyNorm(nn.Module):
    def __init__(self, width=8, group_size=4):
        super().__init__()
        self.group_size, self.eps = group_size, 1e-6
        self.weight = nn.Parameter(torch.linspace(-.2, .2, width).bfloat16(), requires_grad=False)


class TinyProjection(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(81)
        self.weight = nn.Parameter(torch.randn(12, 6, generator=generator).bfloat16() / 7,
                                   requires_grad=False)
        self.output_sizes = [8, 4]
        self.inputs = []

    def forward(self, inputs):
        self.inputs.append(inputs)
        return F.linear(inputs, self.weight), None


class TinyPLE(nn.Module):
    """A small independent causal-conv backend with request slot routing."""
    def __init__(self):
        super().__init__()
        self.hidden_size, self.hc_count = 4, 2
        self.kv_proj = TinyProjection()
        self.norm_key, self.norm_query, self.norm_conv = TinyNorm(), TinyNorm(), TinyNorm()
        self.conv1d = nn.Conv1d(8, 8, 3, groups=8, dilation=3, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.conv1d.weight.copy_(torch.linspace(-.1, .1, 24).reshape(8, 1, 3))
        self.conv1d.requires_grad_(False)
        self.short_conv_state_len = 6
        self.kv_cache = (torch.zeros(3, 8, 6),)
        self.segments = [(0, 0, 1, True)]
        self.last_inputs = self.last_residual = self.last_outer = None

    def _short_conv(self, inputs, residual, outer_residual):
        self.last_inputs = inputs.clone()
        self.last_residual = residual.clone()
        self.last_outer = outer_residual.clone()
        for slot, start, end, reset in self.segments:
            history = self.kv_cache[0][slot]
            if reset:
                history = torch.zeros_like(history)
            sequence = torch.cat([history, inputs[start:end].T], dim=1)
            # Independent torch convolution; the training reference uses an
            # explicit cross-correlation, and the production backend is Triton.
            convolved = F.conv1d(sequence[None], self.conv1d.weight.float(),
                                 groups=8, dilation=3)[0].T
            residual[start:end].add_(F.silu(convolved))
            self.kv_cache[0][slot].copy_(sequence[:, -6:])
        residual.add_(outer_residual)


def tensors(length):
    generator = torch.Generator().manual_seed(19 + length)
    return (torch.randn(length, 8, generator=generator).bfloat16(),
            torch.randn(length, 6, generator=generator).bfloat16(),
            torch.randn(length, 6, generator=generator) * .0001)


def reference(ple, hidden, base, delta):
    merged = F.linear(base, ple.kv_proj.weight).float() + F.linear(delta, ple.kv_proj.weight.float())
    key, value = merged.split([8, 4], dim=-1)
    body = ple_body_fp32(hidden[None], key[None], value[None], ple)[0]
    return hidden + body.bfloat16()


def test_full_prefill_matches_independent_split_formula_and_cast_boundary():
    ple = TinyPLE()
    hidden, base, delta = tensors(11)
    ple.segments = [(0, 0, 11, True)]
    output = split.split_cached_body(ple, hidden, base, delta, ple.kv_proj.weight.float())
    torch.testing.assert_close(output, reference(ple, hidden, base, delta), atol=0, rtol=0)
    assert ple.kv_proj.inputs[0] is base
    assert ple.last_inputs.dtype == ple.last_residual.dtype == ple.kv_cache[0].dtype == torch.float32
    assert not ple.last_outer.any()
    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize("lengths", [(5, 2, 1, 3), (1,) * 11, (3, 4, 4)])
def test_chunked_prefill_and_decode_keep_the_same_conv_history(lengths):
    ple = TinyPLE()
    hidden, base, delta = tensors(sum(lengths))
    outputs, offset = [], 0
    for length in lengths:
        ple.segments = [(2, 0, length, offset == 0)]
        stop = offset + length
        outputs.append(split.split_cached_body(ple, hidden[offset:stop], base[offset:stop],
                                                delta[offset:stop], ple.kv_proj.weight.float()))
        offset = stop
    torch.testing.assert_close(torch.cat(outputs), reference(ple, hidden, base, delta), atol=0, rtol=0)


def test_request_reordering_and_slot_reuse_do_not_mix_conv_histories():
    ple = TinyPLE()
    first = tensors(7)
    second = tensors(4)
    # Packed prefill A(3), B(2), in slots 1 and 0.
    ple.segments = [(1, 0, 3, True), (0, 3, 5, True)]
    prefill = split.split_cached_body(ple, *[torch.cat([a[:3], b[:2]]) for a, b in zip(first, second)],
                                     ple.kv_proj.weight.float())
    # Reverse the request order during the next chunk: B(2), A(4).
    ple.segments = [(0, 0, 2, False), (1, 2, 6, False)]
    continuation = split.split_cached_body(
        ple, *[torch.cat([b[2:], a[3:]]) for a, b in zip(first, second)], ple.kv_proj.weight.float())
    torch.testing.assert_close(torch.cat([prefill[:3], continuation[2:]]), reference(ple, *first), atol=0, rtol=0)
    torch.testing.assert_close(torch.cat([prefill[3:], continuation[:2]]), reference(ple, *second), atol=0, rtol=0)
    # A new request reuses B's old slot, explicitly with no initial history.
    ple.segments = [(0, 0, 4, True)]
    restarted = split.split_cached_body(ple, *second, ple.kv_proj.weight.float())
    torch.testing.assert_close(restarted, reference(ple, *second), atol=0, rtol=0)


def test_sub_bf16_delta_survives_until_the_fp32_gate_and_conv():
    ple = TinyPLE()
    hidden, _, _ = tensors(1)
    base = torch.full((1, 6), .5, dtype=torch.bfloat16)
    delta = torch.full((1, 6), 1e-4)
    assert torch.equal((base.float() + delta).bfloat16(), base)
    split.split_cached_body(ple, hidden, base, torch.zeros_like(delta), ple.kv_proj.weight.float())
    zero_inputs, zero_gated = ple.last_inputs.clone(), ple.last_residual.clone()
    split.split_cached_body(ple, hidden, base, delta, ple.kv_proj.weight.float())
    assert torch.count_nonzero(ple.last_inputs - zero_inputs) > 0
    assert torch.count_nonzero(ple.last_residual - zero_gated) > 0


def test_bf16_convolution_cache_is_rejected_before_it_loses_delta_precision():
    ple = TinyPLE()
    ple.kv_cache = (ple.kv_cache[0].bfloat16(),)
    with pytest.raises(ValueError, match="cache must be allocated in FP32"):
        split.split_cached_body(ple, *tensors(1), ple.kv_proj.weight.float())


def test_forward_consumes_native_embedding_and_hash_metadata_unchanged(monkeypatch):
    ple = TinyPLE()
    hidden, base, delta = tensors(2)
    ple.segments = [(0, 0, 2, True)]
    token_ids = torch.tensor([8, 9])
    qsl = torch.tensor([0, 2], dtype=torch.int32)
    context = torch.tensor([[4, 7]], dtype=torch.int32)
    native_ids = torch.tensor([[0, 15], [4, 19]])
    observed = []

    class NativeEmbedding(nn.Module):
        def forward(self, actual_hidden, ids, actual_qsl, actual_context):
            observed.append(("lookup", ids, actual_qsl, actual_context))
            assert actual_hidden is hidden
            return base

        def compute_ngram_ids(self, ids, actual_qsl, actual_context):
            observed.append(("hash", ids, actual_qsl, actual_context))
            return native_ids

    class Snapshot:
        def lookup(self, ids, *, owner_id):
            assert ids is native_ids and owner_id == 0
            return delta

    ple.ple_embedding = NativeEmbedding()
    ple._dequantize_embeddings = lambda embeddings, dtype: embeddings
    runtime = SimpleNamespace(snapshot=Snapshot(), fp32_weight=ple.kv_proj.weight.float())
    monkeypatch.setattr(split, "_runtime", lambda _: runtime)
    actual = split.split_ple_forward(ple, hidden, token_ids, qsl, context)
    assert [item[0] for item in observed] == ["lookup", "hash"]
    assert all(item[2] is qsl and item[3] is context for item in observed)
    torch.testing.assert_close(actual, reference(ple, hidden, base, delta), atol=0, rtol=0)


@pytest.fixture
def binding_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(split, "_environment", split._UNSET)
    monkeypatch.setattr(split, "_binding", None)
    monkeypatch.delenv(split.BINDING_ENV, raising=False)
    monkeypatch.delenv(split.BINDING_SHA_ENV, raising=False)
    data = {"schema_version": 1, "snapshot_path": str(tmp_path / "snapshot.pt"),
            "snapshot_sha256": "a" * 64, "owner_id": 0,
            "identity": {"base_model": "expected"},
            "layout": asdict(Qwen38PLELayout(tuple(range(0, 32, 2)), (2,) * 16))}
    path = tmp_path / "binding.json"

    def install(**updates):
        data.update(updates)
        path.write_text(json.dumps(data), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        monkeypatch.setenv(split.BINDING_ENV, str(path))
        monkeypatch.setenv(split.BINDING_SHA_ENV, digest)
        return path, digest

    return install


def test_binding_is_hash_checked_and_cannot_be_replaced(binding_environment, monkeypatch):
    _, digest = binding_environment()
    assert split.split_enabled()
    assert split._binding["_binding_sha256"] == digest
    monkeypatch.setenv(split.BINDING_SHA_ENV, "b" * 64)
    with pytest.raises(ValueError, match="environment changed"):
        split.split_enabled()


def test_binding_cannot_be_enabled_after_disabled_mode_was_observed(binding_environment):
    assert not split.split_enabled()
    binding_environment()
    with pytest.raises(ValueError, match="environment changed"):
        split.split_enabled()


def test_binding_sha_mismatch_is_rejected(binding_environment, monkeypatch):
    binding_environment()
    monkeypatch.setenv(split.BINDING_SHA_ENV, "b" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        split.split_enabled()


@pytest.mark.parametrize("owner", [True, 1, -1, "0"])
def test_binding_rejects_nonzero_or_ambiguous_owner(binding_environment, owner):
    binding_environment(owner_id=owner)
    with pytest.raises(ValueError, match="owner_id=0"):
        split.split_enabled()


def test_binding_requires_both_environment_values(binding_environment, monkeypatch):
    binding_environment()
    monkeypatch.delenv(split.BINDING_SHA_ENV)
    with pytest.raises(ValueError, match="must be set together"):
        split.split_enabled()

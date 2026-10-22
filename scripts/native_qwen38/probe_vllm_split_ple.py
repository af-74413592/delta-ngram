"""Small CUDA-only PLE/cache probe; never constructs or loads the language model."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from delta_ngram.qwen38_precision import _ieee_matmul, depthwise_causal_fp32, ple_body_fp32
from delta_ngram.qwen38_vllm_ple import split_cached_body
from vllm.models.qwen4_exp.nvidia import ple_layer as native
from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding
from vllm.v1.attention.backends.short_conv_attn import PleShortConvAttentionMetadata


class Projection(nn.Module):
    def __init__(self, generator, device):
        super().__init__()
        self.output_sizes = [128, 32]
        self.weight = nn.Parameter(
            (torch.randn(160, 64, generator=generator) / 16).to(device=device, dtype=torch.bfloat16),
            requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight), None


class TinyNativePLE(nn.Module):
    _short_conv = native.Qwen4ExpPLELayer._short_conv
    _short_conv_dilated_dispatch = native.Qwen4ExpPLELayer._short_conv_dilated_dispatch

    def __init__(self, device):
        super().__init__()
        generator = torch.Generator().manual_seed(408)
        self.hidden_size, self.hc_count, self.hc_hidden_size = 32, 4, 128
        self.prefix = "tiny_native.ple"
        self.conv_state_len = self.short_conv_state_len = 9
        self.short_conv_dilation, self.num_spec_tokens = 3, 0
        self.kv_proj = Projection(generator, device)
        self.norm_key = native.Qwen4ExpPLEGroupedNorm(128, 1e-6, 32, torch.bfloat16).to(device)
        self.norm_query = native.Qwen4ExpPLEGroupedNorm(128, 1e-6, 32, torch.bfloat16).to(device)
        self.norm_conv = native.Qwen4ExpPLEGroupedNorm(128, 1e-6, 32, torch.bfloat16).to(device)
        self.conv1d = nn.Conv1d(128, 128, 4, groups=128, dilation=3, bias=False,
                                dtype=torch.bfloat16, device=device)
        with torch.no_grad():
            self.conv1d.weight.copy_(torch.randn(128, 1, 4, generator=generator).to(device) / 32)
        self.requires_grad_(False)
        self.reset_cache()

    def reset_cache(self):
        canonical = torch.zeros(3, 128, 9, device=self.kv_proj.weight.device, dtype=torch.float32)
        self.kv_cache = (canonical if native.is_conv_state_dim_first()
                         else canonical.transpose(-1, -2).contiguous(),)

    def canonical_cache(self):
        return self.kv_cache[0] if native.is_conv_state_dim_first() else self.kv_cache[0].transpose(-1, -2)


def metadata(device, lengths, slots, initial, num_decodes=0):
    query_start = [0]
    for length in lengths:
        query_start.append(query_start[-1] + length)
    qsl = torch.tensor(query_start, device=device, dtype=torch.int32)
    state = torch.tensor(slots, device=device, dtype=torch.int32)
    has_initial = torch.tensor(initial, device=device, dtype=torch.bool)
    kwargs = dict(
        num_prefills=len(lengths) - num_decodes, num_prefill_tokens=sum(lengths[num_decodes:]),
        num_decodes=num_decodes, num_decode_tokens=sum(lengths[:num_decodes]), num_reqs=len(lengths),
        has_initial_states_p=has_initial[num_decodes:], query_start_loc_p=None,
        num_computed_tokens_p=None, state_indices_tensor_p=None, state_indices_tensor_d=None,
        query_start_loc_d=None, num_accepted_tokens=None, block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None, block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        seq_lens=torch.tensor(lengths, device=device, dtype=torch.int32),
        num_actual_tokens=sum(lengths), query_start_loc=qsl, state_indices_tensor=state,
        has_initial_states_d=has_initial[:num_decodes], non_spec_query_start_loc=qsl,
    )
    return PleShortConvAttentionMetadata(**kwargs)


@contextmanager
def route(ple, meta):
    original = native.get_forward_context
    native.get_forward_context = lambda: SimpleNamespace(attn_metadata={ple.prefix: meta})
    try:
        yield
    finally:
        native.get_forward_context = original


def stats(actual, expected):
    difference = actual.float() - expected.float()
    return {"shape": list(actual.shape), "dtype": str(actual.dtype),
            "max_abs": float(difference.abs().max()),
            "rms_error": float(difference.square().mean().sqrt()),
            "bitwise_equal": bool(torch.equal(actual, expected))}


def data(length, seed, device):
    generator = torch.Generator().manual_seed(seed)
    return tuple(value.to(device) for value in (
        torch.randn(length, 128, generator=generator).bfloat16(),
        torch.randn(length, 64, generator=generator).bfloat16(),
        torch.randn(length, 64, generator=generator) * .0001))


def reference(ple, packet):
    hidden, base, delta = packet
    with _ieee_matmul(hidden.device):
        projected = F.linear(base, ple.kv_proj.weight).float() + F.linear(delta, ple.kv_proj.weight.float())
    key, value = projected.split([128, 32], dim=-1)
    result = ple_body_fp32(hidden[None], key[None], value[None], ple)[0]
    return hidden + result.bfloat16()


def run_body(ple, packet, lengths, slots, initial, num_decodes=0):
    with route(ple, metadata(ple.kv_proj.weight.device, lengths, slots, initial, num_decodes)):
        return split_cached_body(ple, *packet, ple.kv_proj.weight.float())


def hash_probe(device):
    # Initialize only hash buffers; no embedding table is constructed.
    embedding = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(embedding)
    embedding.ngram_size, embedding.heads_per_ngram, embedding.eos_token_id = 3, 8, 0
    embedding.register_buffer("layer_multipliers", torch.tensor([31, 37, 43], dtype=torch.long))
    sizes = torch.arange(17, 33, dtype=torch.long)
    embedding.register_buffer("ngram_heads_vocab_sizes", sizes)
    embedding.register_buffer("ngram_heads_offsets", torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)[:-1]]))
    tokens = torch.tensor([3, 4, 5, 0, 6, 1, 8, 9, 11, 12, 13, 14, 15], dtype=torch.long)
    full_qsl = torch.tensor([0, len(tokens)], dtype=torch.int32)
    cpu = embedding.compute_ngram_ids(tokens, full_qsl, torch.zeros(1, 2, dtype=torch.int32))
    embedding.to(device)
    full = embedding.compute_ngram_ids(tokens.to(device), full_qsl.to(device),
                                       torch.zeros(1, 2, device=device, dtype=torch.int32))
    chunks, offset = [], 0
    for length in (4, 3, 1, 1, 4):
        context = torch.zeros(2, dtype=torch.long)
        history = tokens[max(0, offset - 2):offset]
        if len(history):
            context[-len(history):] = history
        chunks.append(embedding.compute_ngram_ids(tokens[offset:offset + length].to(device),
                                                   torch.tensor([0, length], device=device, dtype=torch.int32),
                                                   context[None].to(device=device, dtype=torch.int32)))
        offset += length
    assert torch.equal(full.cpu(), cpu)
    assert torch.equal(torch.cat(chunks), full)
    return {"cpu_gpu_equal": True, "full_chunked_equal": True,
            "tokens": len(tokens), "heads": 16, "crosses_eos_and_single_token_decode": True}


@torch.no_grad()
def probe(device):
    ple = TinyNativePLE(device)
    report = {"ngram": hash_probe(device), "cache_dtype": str(ple.kv_cache[0].dtype),
              "native_cache_layout_dim_first": native.is_conv_state_dim_first()}
    packet = data(19, 981, device)
    expected = reference(ple, packet)
    full = run_body(ple, packet, [19], [2], [False])
    report["full_prefill_vs_explicit_reference"] = stats(full, expected)
    torch.testing.assert_close(full, expected, atol=.015625, rtol=.01)
    ple.reset_cache()
    outputs, offset = [], 0
    for length in (7, 3, 1, 1, 7):
        chunk = tuple(value[offset:offset + length] for value in packet)
        outputs.append(run_body(ple, chunk, [length], [2], [offset > 0], int(length == 1)))
        offset += length
    chunked = torch.cat(outputs)
    report["chunked_and_decode_vs_full"] = stats(chunked, full)
    torch.testing.assert_close(chunked, full, atol=.015625, rtol=.01)
    # Direct FP32 conv comparison prevents the BF16 return cast from masking
    # cache rounding loss. It also measures the native kernel's FMA differences.
    ple.reset_cache()
    inputs = packet[0].float() * .1234567
    conv_expected = F.silu(depthwise_causal_fp32(inputs[None], ple))[0]
    conv_chunks, offset = [], 0
    for length in (6, 1, 1, 11):
        residual = torch.zeros(length, 128, device=device)
        with route(ple, metadata(device, [length], [1], [offset > 0], int(length == 1))):
            ple._short_conv(inputs[offset:offset + length], residual, torch.zeros_like(residual))
        conv_chunks.append(residual)
        offset += length
    conv_actual = torch.cat(conv_chunks)
    report["fp32_native_conv_vs_explicit"] = stats(conv_actual, conv_expected)
    torch.testing.assert_close(conv_actual, conv_expected, atol=1e-6, rtol=2e-5)
    assert torch.equal(ple.canonical_cache()[1], inputs[-9:].T)
    report["cached_history_retains_exact_fp32_inputs"] = True
    # Mixed decode+prefill and changed request order reuse native slot routing.
    ple.reset_cache()
    first, second = data(13, 79, device), data(10, 80, device)
    initial = tuple(torch.cat([a[:4], b[:3]]) for a, b in zip(first, second))
    # Native vLLM slot 0 is NULL_BLOCK_ID, so use two real slots (1 and 2).
    prefix = run_body(ple, initial, [4, 3], [2, 1], [False, False])
    middle = tuple(torch.cat([b[3:4], a[4:9]]) for a, b in zip(first, second))
    middle_result = run_body(ple, middle, [1, 5], [1, 2], [True, True], num_decodes=1)
    tail = tuple(torch.cat([b[4:], a[9:]]) for a, b in zip(first, second))
    tail_result = run_body(ple, tail, [6, 4], [1, 2], [True, True])
    actual_first = torch.cat([prefix[:4], middle_result[1:], tail_result[6:]])
    actual_second = torch.cat([prefix[4:], middle_result[:1], tail_result[:6]])
    report["mixed_request_a"] = stats(actual_first, reference(ple, first))
    report["mixed_request_b"] = stats(actual_second, reference(ple, second))
    torch.testing.assert_close(actual_first, reference(ple, first), atol=.015625, rtol=.01)
    torch.testing.assert_close(actual_second, reference(ple, second), atol=.015625, rtol=.01)
    report["status"] = "passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert importlib.metadata.version("vllm") == "0.30.0"
    assert torch.cuda.device_count() == 1, "set CUDA_VISIBLE_DEVICES to one GPU for this small probe"
    result = probe(torch.device("cuda:0"))
    torch.cuda.synchronize()
    result.update(vllm=importlib.metadata.version("vllm"), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  full_model_loaded=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

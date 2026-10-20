"""CPU contracts at actual SWIFT hash/lookup/PLE boundaries, not model acceptance.

The hash, EOS shift and readout below transcribe the audited local
cl/artifacts/qwen38_swift_source/ple.py:321-374,505-535,646-679.
Audited source SHA256:
dbd02fac13bfb2134d3ae0110a56a1a9581f18b01a1efe3dcdbc9515de3f4b86.
Only expensive/native dependencies are replaced: tiny table regions, frozen
structured projections, ordinary grouped RMSNorm, and no distributed kernels.
The adapter itself never sees/reimplements the fixture's hashing algorithm.
"""
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.qwen38_adapter import Qwen38NativePLEAdapter
from delta_ngram.qwen38_native import Qwen38PLELayout, make_qwen38_writer, write_native_block
from delta_ngram.state import SparseMemoryState


class NativeEmbedding(nn.Module):
    def __init__(self, host):
        super().__init__()
        self.ngram_size, self.heads_per_ngram, self.context_len = 3, 8, 2
        self.eos_token_id, self.cpu_offload = 99, host
        self.register_buffer("ngram_heads_offsets", torch.arange(16) * 5)
        self.register_buffer("ngram_heads_vocab_sizes", torch.full((16,), 5))
        self.register_buffer("layer_multipliers", torch.tensor([3, 7, 11]))
        # The native SWIFT loader dequantizes FP8 with global weight_scale
        # before storing the host/device table. No scale is reapplied by hooks.
        raw = (torch.randn(80, 160) * 2).to(torch.float8_e4m3fn)
        self.register_buffer("raw_fp8", raw)
        self.register_buffer("weight_scale", torch.tensor(.125))
        table = raw.float() * self.weight_scale
        self.register_buffer("host_table", table)
        self.ngram_embedding = nn.Embedding.from_pretrained(table.clone(), freeze=True)

    def _shift_right_ignore_eos(self, token_ids, shift):
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1),
             previous_eos_inclusive[:, :-1]], dim=1)
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def forward(self, input_ids):
        input_ids = input_ids.long()
        previous_context = input_ids.new_full(
            (input_ids.shape[0], self.context_len), self.eos_token_id)
        token_history = torch.cat([previous_context, input_ids], dim=-1)
        shifted_tokens = [self._shift_right_ignore_eos(token_history, shift)
                          for shift in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start_idx = (ngram - 2) * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed_ids = torch.bitwise_xor(
                    mixed_ids, shifted_tokens[position] * self.layer_multipliers[position])
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
            blocks.append(ngram_ids + head_offsets.view(1, 1, -1))
        ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1]:]
        if self.cpu_offload:
            return self._host_lookup(ngram_ids)
        return self.ngram_embedding(ngram_ids).flatten(-2)

    def _host_lookup(self, ngram_ids):
        return self.host_table[ngram_ids].flatten(-2)


class Projection(nn.Module):
    """Small frozen linear maps with TELinear's (output, bias) return contract."""

    def __init__(self, branches):
        super().__init__()
        self.branches = branches
        self.weight = nn.Parameter(torch.linspace(.7, 1.3, 2560), requires_grad=False)

    def forward(self, x):
        return torch.cat([x.roll(i, dims=-1) * self.weight
                          for i in range(self.branches)], dim=-1), None


class GroupedNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(.8, 1.2, 10240), requires_grad=False)

    def forward(self, x):
        grouped = x.unflatten(-1, (4, 2560))
        return (grouped * (grouped.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
                ).flatten(-2) * self.weight


class NativePLE(nn.Module):
    def __init__(self, host=False):
        super().__init__()
        self.hidden_size, self.hc_count = 2560, 4
        self.ple_embedding = NativeEmbedding(host)
        self.key_proj, self.value_proj = Projection(4), Projection(1)
        self.norm_key, self.norm_query, self.norm_conv = (GroupedNorm() for _ in range(3))
        self.short_conv_state_len = 9
        self.conv1d = nn.Conv1d(10240, 10240, 4, groups=10240, dilation=3, bias=False)
        self.requires_grad_(False)

    def _short_conv(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)
        sequence_length = hidden_states.shape[-1]
        conv_input = F.pad(hidden_states, (self.short_conv_state_len, 0))
        conv_input = conv_input[..., -(self.short_conv_state_len + sequence_length):]
        return F.silu(self.conv1d(conv_input)).transpose(1, 2)

    def compute(self, hidden_states, input_ids):
        embeddings = self.ple_embedding(input_ids)
        return self.readout(hidden_states, embeddings)

    def readout(self, hidden_states, embeddings):
        key_normed = self.norm_key(self.key_proj(embeddings)[0]).unflatten(
            -1, (self.hc_count, self.hidden_size))
        value = self.value_proj(embeddings)[0]
        query_normed = self.norm_query(hidden_states).unflatten(
            -1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        return gated_value + self._short_conv(gated_value_normed)

    def forward(self, hidden_states, input_ids, packed_seq_params=None):
        if packed_seq_params is not None:
            cu = packed_seq_params.cu_seqlens_q
            hid = hidden_states.new_zeros((len(cu) - 1, packed_seq_params.max_seqlen_q, 10240))
            toks = input_ids.new_full(hid.shape[:2], self.ple_embedding.eos_token_id)
            for i in range(len(cu) - 1):
                start, end = int(cu[i]), int(cu[i + 1])
                hid[i, :end-start] = hidden_states[start:end, 0]
                toks[i, :end-start] = input_ids[0, start:end]
            res = self.compute(hid, toks)
            return torch.cat([res[i, :int(cu[i+1] - cu[i])] for i in range(len(cu)-1)])[:, None]
        return self.compute(hidden_states.transpose(0, 1), input_ids).transpose(0, 1).contiguous()


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def setup(host=False):
    torch.manual_seed(53)
    ple = NativePLE(host)
    layout = Qwen38PLELayout(tuple(5 * i for i in range(16)), (5,) * 16)
    tokens = torch.tensor([[4, 5, 99, 7, 8], [10, 11, 12, 99, 99]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    hidden = torch.randn(5, 2, 10240)
    return ple, layout, tokens, mask, hidden


def run(adapter, ple, tokens, mask, hidden, mode="evidence", state=None, owners=None):
    if owners is None:
        owners = torch.arange(tokens.shape[0])
    return adapter.run_native(lambda: ple(hidden, tokens), input_ids=tokens,
                              valid_mask=mask, owner_ids=owners, mode=mode, state=state)


@pytest.mark.parametrize("host", [False, True])
def test_zero_overlay_actual_hash_eos_prefix_and_restore(host):
    ple, layout, tokens, mask, hidden = setup(host)
    original = ple(hidden, tokens)
    state_keys = tuple(ple.state_dict())
    with Qwen38NativePLEAdapter(ple, layout) as adapter:
        evidence, address = run(adapter, ple, tokens, mask, hidden)
        empty, query = run(adapter, ple, tokens, mask, hidden, "query", layout.empty_state(2))
        torch.testing.assert_close(evidence, original, rtol=0, atol=0)
        torch.testing.assert_close(empty, original, rtol=0, atol=0)
        assert address.lookup_backend == ("host" if host else "device")
        assert address.physical_rows[1, -2:].count_nonzero() == 0
        assert torch.equal(address.native_ids, query.native_ids)
        assert tuple(ple.state_dict()) == state_keys
        # First token following EOS hashes identically to a fresh request.
        short_tokens, short_hidden = tokens[:1, 3:], hidden[3:, :1]
        _, fresh = run(adapter, ple, short_tokens, torch.ones_like(short_tokens, dtype=torch.bool),
                       short_hidden)
        assert torch.equal(address.native_ids[0, 3:], fresh.native_ids[0])
        # Explicit prefix affects the native hash; it is never chopped by adapter.
        _, no_prefix = run(adapter, ple, tokens[:1, 1:2], mask[:1, 1:2], hidden[1:2, :1])
        assert not torch.equal(address.native_ids[0, 1], no_prefix.native_ids[0, 0])
        # Capture T0 RMS uses true scaled FP8 values, not raw stored FP8.
        expected = ple.ple_embedding.host_table[address.native_ids][mask]
        torch.testing.assert_close(address.base_head_rms, expected.square().mean((0, 2)).sqrt())
    torch.testing.assert_close(ple(hidden, tokens), original, rtol=0, atol=0)
    assert "_host_lookup" not in ple.ple_embedding.__dict__


@pytest.mark.parametrize("host", [False, True])
def test_nonzero_overlay_matches_full_readout_and_owner_gradient(host):
    ple, layout, tokens, mask, hidden = setup(host)
    rows = torch.arange(1, 81)
    values = [torch.randn(80, 160, requires_grad=True) * .03 for _ in range(3)]
    for value in values:
        value.retain_grad()
    memory = BatchMemoryState.from_states(
        [SparseMemoryState(rows, value, torch.ones(80)) for value in values], 80)
    owners = torch.tensor([2, 0])  # Reordered subset; owner 1 must be untouched.
    base = ple.ple_embedding(tokens)
    with Qwen38NativePLEAdapter(ple, layout) as adapter:
        output, capture = run(adapter, ple, tokens, mask, hidden, "query", memory, owners)
        manual_delta = torch.stack([memory.unpack()[owner].lookup(capture.physical_rows[lane])[0]
                                    for lane, owner in enumerate(owners)])
        expected = ple.readout(hidden.transpose(0, 1), base + manual_delta.flatten(-2)).transpose(0, 1)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        actual_grad = torch.autograd.grad(output.square().mean(), values, retain_graph=True)
        expected_grad = torch.autograd.grad(expected.square().mean(), values, retain_graph=True)
        for actual, expected_value in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected_value, rtol=0, atol=0)
        output.square().mean().backward()
        assert values[2].grad.abs().sum() > 0 and values[0].grad.abs().sum() > 0
        assert values[1].grad is None or values[1].grad.count_nonzero() == 0
        assert all(parameter.grad is None for parameter in ple.parameters())
        with pytest.raises(ValueError, match="evidence must not"):
            run(adapter, ple, tokens, mask, hidden, "evidence", memory)


@pytest.mark.parametrize("alpha", [1., .99])
def test_real_readout_backpropagates_to_writer_through_lazy_snapshot(alpha):
    ple, layout, tokens, mask, hidden = setup(True)
    writer, policy = make_qwen38_writer(alpha=alpha)
    feature = torch.randn(2, 5, 2560, requires_grad=True)
    with Qwen38NativePLEAdapter(ple, layout) as adapter:
        _, address = run(adapter, ple, tokens, mask, hidden)
        memory = write_native_block(writer, LazyDecayState.empty(2, 160, 80),
                                    feature, address.physical_rows, mask, policy)
        # Advance again on other rows so the read must materialize lazy decay.
        shifted = torch.where(address.physical_rows > 0,
                              (address.physical_rows - 1) // 5 * 5
                              + address.physical_rows.remainder(5) + 1, 0)
        memory = write_native_block(writer, memory, feature, shifted, mask, policy)
        before = (memory.values.clone(), memory.block_clock.clone(), memory.counts.clone())
        output, _ = run(adapter, ple, tokens, mask, hidden, "query", memory)
        output.square().mean().backward()
        assert writer.target_head[-1].weight.grad.abs().sum() > 0
        assert writer.gate_head[-1].weight.grad.abs().sum() > 0
        assert feature.grad is None
        for actual, saved in zip((memory.values, memory.block_clock, memory.counts), before):
            assert torch.equal(actual, saved)


def test_packed_rows_match_separate_native_sequences_and_do_not_mix_owners():
    ple, layout, tokens, mask, hidden = setup()
    tokens = tokens[:, :3]
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    tokens[1, 2] = 99
    hidden = hidden[:3]
    packed_tokens = torch.cat((tokens[0], tokens[1, :2]))[None]
    packed_hidden = torch.cat((hidden[:, 0], hidden[:2, 1]))[:, None]
    psp = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3, 5]), max_seqlen_q=3)
    expected = ple(packed_hidden, packed_tokens, psp)
    with Qwen38NativePLEAdapter(ple, layout) as adapter:
        output, address = adapter.run_native(
            lambda: ple(packed_hidden, packed_tokens, psp),
            input_ids=tokens, valid_mask=mask, owner_ids=torch.tensor([7, 2]), mode="evidence")
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        _, single = run(adapter, ple, tokens[1:2, :2], mask[1:2, :2], hidden[:2, 1:2])
        assert torch.equal(address.native_ids[1, :2], single.native_ids[0])
        assert address.physical_rows[1, 2].count_nonzero() == 0


def test_boundaries_fail_closed_and_cleanup_after_error():
    ple, layout, tokens, mask, hidden = setup(True)
    with Qwen38NativePLEAdapter(ple, layout) as adapter:
        with pytest.raises(RuntimeError, match="already has"):
            Qwen38NativePLEAdapter(ple, layout)
        with pytest.raises(RuntimeError, match="unbound"):
            ple(hidden, tokens)
        with pytest.raises(ValueError, match="token layout"):
            adapter.run_native(lambda: ple(hidden, tokens.flip(0)), input_ids=tokens,
                               valid_mask=mask, owner_ids=torch.arange(2), mode="evidence")
        # A failed call cannot leave an old state/owner binding installed.
        run(adapter, ple, tokens, mask, hidden)
        with pytest.raises(RuntimeError, match="complete native"):
            adapter.run_native(lambda: torch.tensor(0), input_ids=tokens,
                               valid_mask=mask, owner_ids=torch.arange(2), mode="evidence")
        with pytest.raises(ValueError, match="namespace"):
            run(adapter, ple, tokens, mask, hidden, "query", layout.empty_state(1))
    ple(hidden, tokens)


def test_explicit_checkpoint_replay_keeps_snapshot_and_internal_replay_fails():
    ple, layout, tokens, mask, hidden = setup()
    rows = torch.arange(1, 81)
    values = torch.randn(80, 160, requires_grad=True)
    memory = BatchMemoryState.from_states(
        [SparseMemoryState(rows, values, torch.ones(80))], 80)
    owners = torch.tensor([0, 0])
    with Qwen38NativePLEAdapter(ple, layout) as adapter:
        def replay(value):
            state = memory.with_values(value)
            return run(adapter, ple, tokens, mask, hidden, "query", state, owners)[0]
        expected = torch.autograd.grad(replay(values).square().mean(), values)[0]
        actual = torch.autograd.grad(
            checkpoint(replay, values, use_reentrant=False).square().mean(), values)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with torch.no_grad(), pytest.raises(RuntimeError, match="grad-enabled"):
            replay(values)
        with pytest.raises(RuntimeError, match="unbound"):
            output, _ = adapter.run_native(
                lambda: checkpoint(lambda h: ple(h, tokens), hidden, use_reentrant=False),
                input_ids=tokens, valid_mask=mask, owner_ids=owners, mode="query", state=memory)
            output.sum().backward()


def test_directional_finite_difference_through_gate_norm_and_causal_convolution():
    ple, layout, tokens, mask, hidden = setup(True)
    rows = torch.arange(1, 81)
    values = (torch.randn(80, 160) * .02).requires_grad_()
    memory = BatchMemoryState.from_states(
        [SparseMemoryState(rows, values, torch.ones(80))], 80)
    owners = torch.tensor([0, 0])
    weights = torch.randn_like(hidden)
    with Qwen38NativePLEAdapter(ple, layout) as adapter:
        def objective(value):
            output, _ = run(adapter, ple, tokens, mask, hidden, "query",
                            memory.with_values(value), owners)
            # Accumulate in double to avoid cancellation in the FD loss, while
            # retaining the actual FP32 overlay/readout being checked.
            return (output.double() * weights.double()).sum() / 10240
        _, addresses = run(adapter, ple, tokens, mask, hidden)
        direction = torch.zeros_like(values)
        touched = int(addresses.physical_rows[0, 0, 0]) - 1
        direction[touched] = torch.randn(160)
        direction /= direction.norm()
        derivative = (torch.autograd.grad(objective(values), values)[0] * direction).sum()
        assert abs(float(derivative)) > 1e-5
        for step in (.02, .01):
            numerical = (objective(values.detach() + step * direction)
                         - objective(values.detach() - step * direction)) / (2 * step)
            torch.testing.assert_close(derivative.double(), numerical, rtol=.02, atol=2e-6)

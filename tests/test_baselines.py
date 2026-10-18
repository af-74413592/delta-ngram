"""CPU checks for online optimizer continuity, evidence boundaries, and ridge writes."""
from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_ngram.baselines import (
    BM25RetrievalMethod, EvidenceRidgeMethod, FullContextMethod, OnlineGradientMethod,
    OnlineGradientMemory, OnlineSparseAdamMemory,
)
from delta_ngram.baselines.context import bm25_indices
from delta_ngram.baselines.gradient import evidence_loss
from delta_ngram.baselines.ridge import read_design_matrix, solve_reuse_ridge
from delta_ngram.ngram_lookup import ngram_rows
from delta_ngram.runtime import CharacterTokenizer, token_ids
from delta_ngram.state import SparseMemoryState


class ToyReader:
    """Small deterministic causal reader; no pretrained downloads or GPU access."""

    def __init__(self, vocab_size, width=8):
        self.width, self.buckets, self.orders = width, 97, (2, 3)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            self.model = nn.ModuleDict({"embedding": nn.Embedding(vocab_size, width),
                                        "head": nn.Linear(width, vocab_size, bias=False)})
        self.model.requires_grad_(False).eval()

    def __call__(self, ids, mask, state):
        return self.forward_with_residual(ids, mask, state, None)

    def forward_with_residual(self, ids, mask, state, extra_residual):
        rows = ngram_rows(ids, mask, self.buckets, self.orders)
        residual = state.lookup(rows)[0].sum(dim=-2) / math.sqrt(len(self.orders))
        hidden = self.model["embedding"](ids) + residual
        if extra_residual is not None:
            hidden = hidden + extra_residual
        hidden = hidden.cumsum(dim=1).tanh()
        return SimpleNamespace(logits=self.model["head"](hidden))


@pytest.fixture
def setup():
    torch.set_num_threads(2)
    tokenizer = CharacterTokenizer()
    return ToyReader(tokenizer.vocab_size), tokenizer


def test_adam_expansion_preserves_moments_and_unread_rows():
    memory = OnlineSparseAdamMemory(2, "cpu", 0.01)
    touched = memory.observe(torch.tensor([2, 5, 5]))
    memory.values.grad = torch.tensor([[1., 2.], [3., 4.]])
    memory.step_on(touched)
    endpoint = memory.snapshot()
    before = {key: value.clone() if torch.is_tensor(value) else value
              for key, value in memory.optimizer.state[memory.values].items()}
    touched = memory.observe(torch.tensor([5, 9]))
    migrated = memory.optimizer.state[memory.values]
    assert migrated["step"] == before["step"]
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(migrated[key][:2], before[key])
        assert not migrated[key][2].any()
    assert torch.equal(memory.values[:2].detach(), endpoint.values)
    memory.values.grad = torch.ones_like(memory.values)
    memory.step_on(touched)
    after = memory.optimizer.state[memory.values]
    assert torch.equal(memory.values[0].detach(), endpoint.values[0])
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(after[key][0], before[key][0])
    assert memory.counts.tolist() == [1., 2., 1.]
    assert torch.equal(endpoint.counts, torch.tensor([1., 1.]))
    assert after["step"] == before["step"] + 1


def test_adam_k10_bitwise_parity_with_fixed_universe_sparseadam(setup):
    """An independent, nonexpanding SparseAdam table is the arithmetic oracle."""
    reader, tokenizer = setup
    evidence = ("abacb", "abdab")
    ids = [token_ids(tokenizer, text, torch.device("cpu")) for text in evidence]
    events = [ngram_rows(item, torch.ones_like(item), reader.buckets, reader.orders) for item in ids]
    all_rows = torch.unique(torch.cat([item[item.ne(0)] for item in events]), sorted=True)
    values = nn.Parameter(torch.zeros(all_rows.numel(), reader.width))
    counts = torch.zeros(all_rows.numel())
    optimizer = torch.optim.SparseAdam([values], lr=0.01)
    method = OnlineGradientMethod(reader, tokenizer, "cpu", kind="adam", lr=0.01, online_steps=10)
    state = method.empty_state()
    for text, current_ids, event_rows in zip(evidence, ids, events):
        actual = method.write(state, text)
        state = actual.state
        touched = torch.unique(event_rows[event_rows.ne(0)], sorted=True)
        positions = torch.searchsorted(all_rows, touched)
        counts[positions] += 1
        reference_losses = []
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            output = reader(current_ids, torch.ones_like(current_ids), SparseMemoryState(all_rows, values, counts))
            logits = output.logits[:, :-1, :].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), current_ids[:, 1:].reshape(-1))
            loss.backward()
            reference_losses.append(float(loss.detach()))
            gradient = values.grad.index_select(0, positions)
            values.grad = torch.sparse_coo_tensor(positions[None, :], gradient, values.shape).coalesce()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        snapshot = method.memory(state)
        live_positions = torch.searchsorted(all_rows, snapshot.rows)
        assert actual.diagnostics["causal_evidence_nll"] == reference_losses
        assert torch.equal(snapshot.values, values.detach()[live_positions])
        assert torch.equal(snapshot.counts, counts[live_positions])
        actual_optimizer = state.owner.optimizer.state[state.owner.values]
        expected_optimizer = optimizer.state[values]
        assert actual_optimizer["step"] == expected_optimizer["step"]
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(actual_optimizer[key], expected_optimizer[key][live_positions])


@pytest.mark.parametrize("kind", ["adam", "sgd", "read_normalized_sgd"])
def test_gradient_snapshots_reset_and_counterfactual_are_read_only(setup, kind):
    reader, tokenizer = setup
    before_rng = torch.random.get_rng_state().clone()
    method = OnlineGradientMethod(reader, tokenizer, "cpu", kind=kind, online_steps=2)
    initial = method.empty_state()
    assert torch.equal(before_rng, torch.random.get_rng_state())
    with torch.no_grad():
        first = method.write(initial, "abacb").state
        saved = tuple(value.clone() for value in (first.snapshot.rows, first.snapshot.values, first.snapshot.counts))
        second = method.write(first, "abdab").state
    for expected, actual in zip(saved, (first.snapshot.rows, first.snapshot.values, first.snapshot.counts)):
        assert torch.equal(expected, actual)
    with pytest.raises(RuntimeError, match="latest live"):
        method.write(first, "abdab")
    fresh = method.empty_state()
    assert fresh.snapshot.rows.numel() == 0 and not fresh.owner.optimizer.state
    other = method.write(fresh, "abacx").state
    counterfactual = method.counterfactual(first, other)
    assert torch.equal(counterfactual.snapshot.rows, first.snapshot.rows)
    assert torch.equal(counterfactual.snapshot.counts, first.snapshot.counts)
    assert torch.equal(counterfactual.snapshot.values, other.snapshot.lookup(first.snapshot.rows)[0])
    with pytest.raises(RuntimeError, match="read-only"):
        method.write(counterfactual, "abdab")
    assert all(parameter.grad is None for parameter in reader.model.parameters())
    assert method.parameters() == () and not method.supports_outer_training
    assert list(inspect.signature(method.write).parameters) == ["state", "evidence"]
    assert method.state_bytes(second) > 0


def test_sgd_and_read_normalized_sgd_have_distinct_exact_updates():
    plain = OnlineGradientMemory(2, "cpu", 0.25, "sgd")
    normalized = OnlineGradientMemory(2, "cpu", 0.25, "read_normalized_sgd")
    for memory in (plain, normalized):
        touched = memory.observe(torch.tensor([2, 2, 2, 7]))
        memory.values.grad = torch.tensor([[6., 12.], [4., 8.]])
        memory.step_on(touched, torch.tensor([3, 1]))
    assert torch.equal(plain.values.detach(), torch.tensor([[-1.5, -3.], [-1., -2.]]))
    assert torch.equal(normalized.values.detach(), torch.tensor([[-0.5, -1.], [-1., -2.]]))
    assert plain.counts.tolist() == normalized.counts.tolist() == [1., 1.]


def test_context_and_bm25_keep_evidence_history_and_use_query_only_for_retrieval(setup):
    reader, tokenizer = setup
    context = FullContextMethod(reader, tokenizer, "cpu")
    first = context.write(context.empty_state(), "Alice likes red.\n").state
    second = context.write(first, "Boris likes blue.\n").state
    assert first.evidence == ("Alice likes red.\n",)
    query = "Alice likes which color?"
    assert context.render_prompt(second, query) == "Alice likes red.\n\nBoris likes blue.\n\n" + query
    retrieval = BM25RetrievalMethod(reader, tokenizer, "cpu", top_k=1)
    assert retrieval.selected_evidence(second, query) == (first.evidence[0],)
    assert retrieval.selected_evidence(second, "Boris color") == (second.evidence[1],)
    assert second.evidence == ("Alice likes red.\n", "Boris likes blue.\n")
    assert bm25_indices(("same", "same", "same"), "missing", 2) == [0, 1]
    assert context.memory(second).rows.numel() == 0
    assert context.state_bytes(second) == sum(len(text.encode("utf-8")) for text in second.evidence)
    assert retrieval.counterfactual(second, first).evidence == first.evidence
    assert not retrieval.supports_row_diagnostics


def test_read_matrix_coalesces_duplicate_physical_rows_and_ignores_padding():
    rows = torch.tensor([[[0, 0], [2, 2], [2, 9]]])
    touched, design = read_design_matrix(rows)
    assert touched.tolist() == [2, 9]
    expected = torch.tensor([[0., 0.], [2 / math.sqrt(2), 0.],
                             [1 / math.sqrt(2), 1 / math.sqrt(2)]], dtype=torch.float64)
    torch.testing.assert_close(design, expected, rtol=0, atol=0)


def test_reuse_ridge_matches_analytic_solution_and_normal_equations():
    design = torch.eye(2, dtype=torch.float64)
    residual = torch.tensor([[2., 4.], [6., 8.]])
    counts = torch.tensor([0., 3.])
    increments = solve_reuse_ridge(design, residual, counts, 1.0, 1.0)
    torch.testing.assert_close(increments, torch.tensor([[1., 2.], [1.2, 1.6]]))
    arbitrary = torch.tensor([[1., 0.5], [0., 1.], [1., 1.]], dtype=torch.float64)
    target = torch.tensor([[2., 3.], [1., -1.], [0., 5.]])
    solved = solve_reuse_ridge(arbitrary, target, counts, 0.5, 2.0).double()
    normal_residual = arbitrary.T @ (arbitrary @ solved - target.double())
    normal_residual += (0.5 * (1 + 2 * counts.double()))[:, None] * solved
    torch.testing.assert_close(normal_residual, torch.zeros_like(normal_residual), rtol=0, atol=5e-7)


def test_optimized_target_ridge_is_evidence_only_and_preserves_unobserved_rows(setup):
    reader, tokenizer = setup
    method = EvidenceRidgeMethod(reader, tokenizer, "cpu", target_steps=3, target_lr=0.05)
    empty = method.empty_state()
    with torch.no_grad():
        first_result = method.write(empty, "abacb")
        first = first_result.state
        saved = first.values.clone()
        second_result = method.write(first, "abdab")
    second = second_result.state
    e2 = token_ids(tokenizer, "abdab", torch.device("cpu"))
    rows2 = ngram_rows(e2, torch.ones_like(e2), reader.buckets, reader.orders)
    untouched = first.rows[~torch.isin(first.rows, rows2)]
    assert untouched.numel() > 0
    assert torch.equal(first.values, saved)
    assert torch.equal(first.lookup(untouched)[0], second.lookup(untouched)[0])
    assert torch.equal(first.lookup(untouched)[1], second.lookup(untouched)[1])
    assert empty.rows.numel() == 0 and not second.values.requires_grad
    assert first_result.diagnostics["target_residual_l2"] > 0
    losses = first_result.diagnostics["target_evidence_nll"]
    assert losses[-1] < losses[0]
    assert first_result.diagnostics["target_optimization_seconds"] >= 0
    assert first_result.diagnostics["ridge_solve_and_update_seconds"] >= 0
    assert "not an exact original reproduction" in first_result.diagnostics["scope"]
    assert list(inspect.signature(method.write).parameters) == ["state", "evidence"]
    assert all(parameter.grad is None for parameter in reader.model.parameters())
    counterfactual = method.counterfactual(first, second)
    assert torch.equal(counterfactual.rows, first.rows) and torch.equal(counterfactual.counts, first.counts)


def test_all_gradient_kinds_use_the_same_mean_evidence_loss(setup):
    reader, tokenizer = setup
    evidence = "abacb"
    ids = token_ids(tokenizer, evidence, torch.device("cpu"))
    empty = SparseMemoryState.empty(reader.width)
    expected = float(evidence_loss(reader, ids, torch.ones_like(ids), empty))
    initial_losses = []
    for kind in ("adam", "sgd", "read_normalized_sgd"):
        method = OnlineGradientMethod(reader, tokenizer, "cpu", kind=kind, online_steps=1)
        result = method.write(method.empty_state(), evidence)
        initial_losses.append(result.diagnostics["causal_evidence_nll"][0])
    assert initial_losses == [expected] * 3

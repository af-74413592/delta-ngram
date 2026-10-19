"""CPU parity and gradient checks for isolated lazy-alpha trajectories."""
import pytest
import torch

from delta_ngram.alpha_lm import alpha_trajectory_loss
from delta_ngram.alpha_memory import BlockAlpha, make_alpha_policy
from delta_ngram.data.event_streams import EventArticle, EventTrajectory
from delta_ngram.data.raw_text import TrajectoryWindow
from delta_ngram.raw_lm import trajectory_loss
from delta_ngram.train import create_runtime
from test_raw_lm import config_for, assert_grads


def items(method, kind):
    tokens = tuple(method.tokenizer.encode("abc xyz\n" * 60))
    if kind == "raw":
        return [TrajectoryWindow("a", "a", "code", 0, tokens[:32], 4),
                TrajectoryWindow("b", "b", "code", 0, tokens[:16], 2)]
    events = [EventTrajectory(name, name, "news_event", tuple(
        EventArticle(f"{name}-{i}", tokens[:length], {}) for i, length in enumerate(lengths)), 8, 128)
        for name, lengths in (("a", (129, 130, 131)), ("b", (128, 128)))]
    if kind == "mixed":
        return [TrajectoryWindow("c", "c", "code", 0, tokens[:256], 2), *events]
    return events


def gradients(parameters):
    return [None if p.grad is None else p.grad.detach().clone() for p in parameters]


@pytest.mark.parametrize("kind", ["raw", "event", "mixed"])
def test_unit_alpha_matches_core_masks_counts_state_loss_gradients(tmp_path, kind):
    method, _ = create_runtime(config_for(tmp_path))
    windows = items(method, kind)
    expected = trajectory_loss(method, windows, prefix_tokens=4)
    expected.mean.backward()
    wanted = gradients(method.parameters())
    method.writer.zero_grad(set_to_none=True)
    alpha = BlockAlpha(method.reader.width, initial_alpha=1)
    actual = alpha_trajectory_loss(method, windows, alpha, prefix_tokens=4)
    actual.mean.backward()
    assert actual.counters == expected.counters
    torch.testing.assert_close(actual.mean, expected.mean, rtol=0, atol=0)
    torch.testing.assert_close(actual.state.values, expected.state.values, rtol=0, atol=0)
    torch.testing.assert_close(actual.state.counts, expected.state.counts, rtol=0, atol=0)
    assert_grads(gradients(method.parameters()), wanted, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("kind", ["raw", "event", "mixed"])
@pytest.mark.parametrize("variant", ["learned_global_099", "evidence_block_099"])
def test_reader_checkpoint_preserves_both_lazy_log_gradient_paths(tmp_path, kind, variant):
    method, _ = create_runtime(config_for(tmp_path))
    alpha = make_alpha_policy(variant, method.reader.width)
    params = list(method.parameters()) + list(alpha.parameters())
    windows = items(method, kind)
    ordinary = alpha_trajectory_loss(method, windows, alpha, prefix_tokens=4)
    ordinary.mean.backward()
    wanted = gradients(params)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in alpha.parameters())
    for p in params:
        p.grad = None
    replayed = alpha_trajectory_loss(method, windows, alpha, prefix_tokens=4, checkpoint_reader=True)
    replayed.mean.backward()
    torch.testing.assert_close(ordinary.mean, replayed.mean, rtol=0, atol=0)
    assert_grads(gradients(params), wanted, rtol=0, atol=0)
    assert all(p.grad is None for p in method.reader.model.parameters())


def test_event_clock_warmup_padding_and_final_write_boundary(tmp_path):
    method, _ = create_runtime(config_for(tmp_path))
    alpha = make_alpha_policy("learned_global_099", method.reader.width)
    trace, observed = [], []
    result = alpha_trajectory_loss(method, items(method, "event"), alpha, prefix_tokens=4,
        alpha_observer=trace.append,
        observer=lambda kind, block, state, *_: observed.append((kind, block, state)))
    assert result.state.block_clock.tolist() == [5, 1]
    assert [r["owner_ids"] for r in trace] == [[0, 1], [0], [0], [0], [0]]
    assert not any(kind == "write" and block == 5 for kind, block, _ in observed)
    # Every later read uses the existing clock; no query itself advances it.
    for kind, block, state in observed:
        if kind == "read":
            assert state.block_clock[0] == block
    result.block_loss_sums[-1].backward()
    assert alpha.logit.grad is not None and alpha.logit.grad.abs() > 0


def test_mixed_owner_mapping_and_graph_preserving_repack(tmp_path):
    method, _ = create_runtime(config_for(tmp_path))
    alpha = make_alpha_policy("evidence_block_099", method.reader.width)
    windows, trace = items(method, "mixed"), []
    batch = alpha_trajectory_loss(method, windows, alpha, prefix_tokens=4, alpha_observer=trace.append)
    for row in trace:
        assert row["document_ids"] == [windows[i].document_id for i in row["owner_ids"]]
    singles = [alpha_trajectory_loss(method, [w], alpha, prefix_tokens=4) for w in windows]
    torch.testing.assert_close(batch.loss_sum, sum(r.loss_sum for r in singles))
    for packed, single in zip(batch.state.unpack(), singles):
        torch.testing.assert_close(packed.values, single.state.values)
        torch.testing.assert_close(packed.cumulative_log_alpha, single.state.cumulative_log_alpha)
        assert packed.cumulative_log_alpha.grad_fn is not None

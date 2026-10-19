"""Tiny CPU checks of event boundaries, causal BPTT and donor accounting."""
import pytest
import torch
import yaml

from delta_ngram.config import ExperimentConfig
from delta_ngram.data.event_streams import EventArticle, EventTrajectory
from delta_ngram.event_lm import event_trajectory_loss
from delta_ngram.train import create_runtime


@pytest.fixture
def method(tmp_path):
    path = tmp_path / "event.yaml"
    path.write_text(yaml.safe_dump({"seed": 17, "model": {"smoke": True, "device": "cpu"},
        "reader": {"buckets": 17}, "writer": {"hidden_dim": 12}}), encoding="utf-8")
    model, _ = create_runtime(ExperimentConfig.load(path))
    return model


def event(method, lengths=(129, 130, 131), name="event", text="abc xyz\n"):
    source = tuple(method.tokenizer.encode(text * 80))
    articles = tuple(EventArticle(f"{name}-{i}", source[:n], {"publication_date_original": f"2017-01-{i+1:02d}"})
                     for i, n in enumerate(lengths))
    chunks = sum((n + 127) // 128 for n in lengths)
    return EventTrajectory(name, name, "news_event", articles, max(8, chunks), 128)


def gradients(method):
    return [None if p.grad is None else p.grad.clone() for p in method.parameters()]


def test_whole_first_article_warmup_real_tokens_only_and_final_no_write(method):
    item = event(method)
    observations = []
    def observe(kind, step, state, ids, mask, labels):
        observations.append((kind, step, int(mask.sum()), None if labels is None else int(labels.ne(-100).sum())))
        if kind == "read" and step in (2, 4):
            assert ids.shape[1] == 129
            assert labels[0, 0] == -100
            assert int(labels.ne(-100).sum()) == 128
    result = event_trajectory_loss(method, [item], prefix_tokens=8, observer=observe)
    assert [(kind, step) for kind, step, _, _ in observations] == [
        ("write", 0), ("write", 1), ("read", 2), ("write", 2),
        ("read", 3), ("write", 3), ("read", 4), ("write", 4), ("read", 5)]
    assert result.loss_tokens == 261
    assert result.counters["raw_tokens_consumed"] == 390
    assert result.counters["writer_new_tokens"] == 387
    assert result.counters["encoder_input_tokens"] == 387
    assert all(count == 128 for kind, _, count, _ in observations if kind == "write" and count > 3)


def test_first_article_diagnostics_use_frozen_pre_article_state_and_real_prefix(method):
    from delta_ngram.raw_lm import score_block
    item = event(method)
    captured = []
    def observe(kind, step, state, ids, mask, labels):
        if kind == 'read' and step in (2, 4):
            captured.append((state, ids.clone(), mask.clone(), labels.clone()))
    with torch.no_grad():
        result = event_trajectory_loss(method, [item], prefix_tokens=8, observer=observe)
        full = aligned = 0.0
        for index, (state, ids, mask, labels) in enumerate(captured):
            value, count = score_block(method.reader, state, ids, mask, labels)
            assert count == 128
            full += float(value)
            # Independent scoring reconstructs only this article's real first block;
            # the state is exactly the snapshot before any of its tokens were written.
            real_ids = torch.tensor([item.articles[index + 1].token_ids[:128]])
            real_labels = real_ids.clone()
            real_labels[:, :8] = -100
            value, count = score_block(method.reader, state, real_ids, torch.ones_like(real_ids), real_labels)
            assert count == 120
            aligned += float(value)
    assert result.first_article_block['full']['loss_tokens'] == 256
    assert result.first_article_block['after_prefix']['loss_tokens'] == 240
    assert result.first_article_block['full']['articles'] == result.first_article_block['after_prefix']['articles'] == 2
    assert result.first_article_block['full']['loss_sum'] == pytest.approx(full, abs=1e-4)
    assert result.first_article_block['after_prefix']['loss_sum'] == aligned
    assert not result.loss_sum.requires_grad
    # Offline training does not add diagnostic reads or alter its loss objective.
    trained = event_trajectory_loss(method, [item], prefix_tokens=8)
    assert trained.first_article_block is None
    torch.testing.assert_close(result.loss_sum, trained.loss_sum, rtol=0, atol=0)
    torch.testing.assert_close(result.state.memory.values, trained.state.memory.values, rtol=0, atol=0)


def test_first_article_prefix_excludes_zero_target_short_articles(method):
    with torch.no_grad():
        result = event_trajectory_loss(method, [event(method, lengths=(128, 128))], prefix_tokens=128)
    assert result.first_article_block['full']['loss_tokens'] == 128
    assert result.first_article_block['after_prefix'] == {'loss_sum': 0.0, 'loss_tokens': 0, 'articles': 0}


def test_last_article_loss_has_gradient_to_first_article_state(method):
    states = {}
    def observe(kind, step, state, *_):
        if kind == "read":
            states[step] = state
    result = event_trajectory_loss(method, [event(method)], prefix_tokens=8, observer=observe)
    first_values = states[2].memory.values
    gradient, = torch.autograd.grad(result.block_loss_sums[-1], first_values, retain_graph=True)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    result.mean.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in method.parameters())
    assert all(p.grad is None for p in method.reader.model.parameters())


def test_future_perturbation_does_not_change_earlier_article_loss(method):
    original = event(method)
    changed = event(method, text="QQQ ZZZ\n")
    changed = EventTrajectory("changed", "changed", "news_event", original.articles[:2] + changed.articles[2:], 8, 128)
    first = event_trajectory_loss(method, [original], prefix_tokens=8)
    later = event_trajectory_loss(method, [changed], prefix_tokens=8)
    for before, after in zip(first.block_loss_sums[:2], later.block_loss_sums[:2]):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_ngram_boundaries_reset_at_each_article_and_padding_is_zero(method, monkeypatch):
    import delta_ngram.event_lm as module
    item = event(method)
    seen = []
    original = module.write_batch
    def capture(writer, state, hidden, rows, mask, **kwargs):
        seen.append((rows.detach().clone(), mask.detach().clone()))
        assert not rows[mask == 0].any()
        return original(writer, state, hidden, rows, mask, **kwargs)
    monkeypatch.setattr(module, "write_batch", capture)
    event_trajectory_loss(method, [item], prefix_tokens=8)
    for index, article_index in ((0, 0), (2, 1), (4, 2)):
        ids = torch.tensor([item.articles[article_index].token_ids[:128]])
        expected = method.reader.adapter.addresses(ids, torch.ones_like(ids))
        torch.testing.assert_close(seen[index][0], expected)


def test_owner_isolation_matches_independent_gradients(method):
    items = [event(method, name="a"), event(method, lengths=(128, 128), name="b", text="xyz abc\n")]
    batch = event_trajectory_loss(method, items, prefix_tokens=8)
    batch.mean.backward()
    batch_grad = gradients(method)
    method.writer.zero_grad(set_to_none=True)
    separate = [event_trajectory_loss(method, [item], prefix_tokens=8) for item in items]
    mean = sum(r.loss_sum for r in separate) / sum(r.loss_tokens for r in separate)
    mean.backward()
    torch.testing.assert_close(batch.mean, mean, rtol=1e-5, atol=1e-6)
    for actual, expected in zip(batch_grad, gradients(method)):
        if actual is None or expected is None:
            assert actual is expected
        else:
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=2e-6)
    assert len(batch.state.unpack()) == 2


def test_checkpoint_recomputation_and_empty_memory(method):
    item = event(method)
    ordinary = event_trajectory_loss(method, [item], prefix_tokens=8)
    ordinary.mean.backward()
    expected = gradients(method)
    method.writer.zero_grad(set_to_none=True)
    replayed = event_trajectory_loss(method, [item], prefix_tokens=8, checkpoint_reader=True)
    replayed.mean.backward()
    torch.testing.assert_close(ordinary.mean, replayed.mean, rtol=0, atol=0)
    for actual, wanted in zip(gradients(method), expected):
        if actual is not None:
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    with torch.no_grad():
        empty = event_trajectory_loss(method, [item], prefix_tokens=8, memory_mode="empty")
    assert empty.loss_tokens == item.loss_tokens and empty.state.rows.numel() == 0
    assert empty.counters["writer_new_tokens"] == 0


def test_event_block_monitor_reports_only_active_reads_and_writes(method):
    item = event(method)
    records = []
    event_trajectory_loss(method, [item], prefix_tokens=8, block_observer=records.append)
    queries = [row for row in records if row["kind"] == "query"]
    writes = [row for row in records if row["kind"] == "write"]
    assert [row["block"] for row in queries] == [2, 3, 4, 5]
    assert [row["block"] for row in writes] == [0, 1, 2, 3, 4]
    assert {row["article_index"] for row in queries} == {1, 2}
    assert all(row["group_id"] == item.group_id for row in records)


def test_donor_matches_real_write_budget_without_changing_target_reads(method):
    target = event(method)
    donor = event(method, lengths=(128, 133, 140), name="donor", text="xyz abc\n")
    observed = []
    def capture(kind, step, state, ids, mask, labels):
        if kind == "read":
            observed.append((step, ids.clone(), mask.clone(), labels.clone()))
    own = event_trajectory_loss(method, [target], prefix_tokens=8, observer=capture)
    expected, observed[:] = observed[:], []
    foreign = event_trajectory_loss(method, [target], prefix_tokens=8, memory_mode="donor",
                                    donor_trajectories=[donor], observer=capture)
    assert foreign.counters["writer_new_tokens"] == own.counters["writer_new_tokens"] == target.writer_tokens
    assert foreign.loss_tokens == own.loss_tokens
    for actual, wanted in zip(observed, expected):
        assert actual[0] == wanted[0]
        for left, right in zip(actual[1:], wanted[1:]):
            assert torch.equal(left, right)
    with pytest.raises(ValueError, match="independent"):
        event_trajectory_loss(method, [target], memory_mode="donor", donor_trajectories=[target])
    too_short = event(method, lengths=(128, 128), name="short")
    with pytest.raises(ValueError, match="sufficient real history"):
        event_trajectory_loss(method, [target], memory_mode="donor", donor_trajectories=[too_short])


@pytest.mark.parametrize("raw", [{"eval_donor": 1}, {"macro_source_groups": []},
    {"macro_source_groups": {"news_event": ""}},
    {"macro_source_groups": {"news_event": "news"}, "source_weights": {"news": 1}}])
def test_event_config_rejects_invalid_flags_and_macro_groups(tmp_path, raw):
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump({"raw": raw}), encoding="utf-8")
    with pytest.raises(ValueError, match="raw\\."):
        ExperimentConfig.load(path)


def test_event_config_path_and_macro_mapping(tmp_path):
    path = tmp_path / "good.yaml"
    path.write_text(yaml.safe_dump({"raw": {"event_dataset": "events.jsonl", "eval_donor": True,
        "source_weights": {"news": .15, "news_event": .05},
        "macro_source_groups": {"news": "news", "news_event": "news"}}}), encoding="utf-8")
    config = ExperimentConfig.load(path)
    assert config.values["raw"]["event_dataset"] == str((tmp_path / "events.jsonl").resolve())

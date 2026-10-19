import copy
import json

import pytest
import yaml

from delta_ngram.config import ExperimentConfig
from delta_ngram.data.raw_text import RawDocument, RawWindowSampler, build_windows


def _pool():
    docs = [RawDocument(f"{source}{i}", f"{source}-family{i}", source, "train", tuple(range(96)))
            for source in ("code", "news") for i in range(5)]
    return build_windows(docs, 2, (2, 4, 8, 16), source_block_counts={"code": [8, 16], "news": [2, 4]})


def _sampler(seed=17):
    return RawWindowSampler(_pool(), seed, source_weights={"code": .8, "news": .2},
                            source_length_weights={"code": {8: .5, 16: .5}, "news": {2: .9, 4: .1}})


def test_conditional_schedule_tracks_joint_loss_tokens_not_trajectory_counts():
    sampler = _sampler()
    batch = sampler.next_batch(10000)
    total = sum(window.loss_tokens for window in batch)
    assert sampler.total_loss_tokens == total
    for cell, share in {(8, "code"): .4, (16, "code"): .4,
                        (2, "news"): .18, (4, "news"): .02}.items():
        assert abs(sampler.cell_loss_tokens[cell] - share * total) < 100
    assert abs(sampler.source_loss_tokens["news"] - total * .2) < 100
    assert all((window.blocks, window.source) in sampler.cell_weights for window in batch)
    assert sum(window.source == "news" for window in batch) > len(batch) / 2


def test_conditional_resume_roundtrip_preserves_future_windows_and_all_counters():
    original = _sampler()
    original.next_batch(217)
    saved = json.loads(json.dumps(original.state_dict()))
    assert saved["version"] == 2
    restored = _sampler(seed=918)
    restored.load_state_dict(saved)
    assert restored.next_batch(450) == original.next_batch(450)
    assert restored.state_dict() == original.state_dict()
    corrupted = copy.deepcopy(saved)
    corrupted["cell_loss_tokens"][0][2] += 2
    with pytest.raises(ValueError, match="cell token counters"):
        _sampler().load_state_dict(corrupted)


def test_short_news_can_fill_budget_tail_without_consuming_unaffordable_window():
    sampler = _sampler()
    batch = sampler.next_batch(100, remaining_loss_tokens=17)
    assert sum(window.loss_tokens for window in batch) == 16
    assert sampler.last_status["minimum_loss_tokens"] == 2
    state = sampler.state_dict()
    assert sampler.next_window(1) is None
    assert sampler.state_dict() == state


def test_source_window_filter_does_not_build_inactive_lengths():
    windows = _pool()
    assert {window.source for window in windows[2] + windows[4]} == {"news"}
    assert {window.source for window in windows[8] + windows[16]} == {"code"}
    with pytest.raises(ValueError, match="source_block_counts"):
        build_windows([], 2, (2, 4), source_block_counts={"news": [8]})


def test_conditional_requires_all_positive_cells_and_rejects_ambiguous_weights():
    with pytest.raises(ValueError, match="mutually exclusive"):
        RawWindowSampler(_pool(), 1, length_weights={2: 1}, source_length_weights={"news": {2: 1}})
    with pytest.raises(ValueError, match="exactly the active"):
        RawWindowSampler(_pool(), 1, source_length_weights={"news": {2: 1}})
    with pytest.raises(ValueError, match="nonzero quota has no windows"):
        RawWindowSampler(_pool(), 1, source_weights={"news": 1}, source_length_weights={"news": {16: 1}})


@pytest.mark.parametrize("change", [
    {"eval_min_groups_per_source": True},
    {"eval_min_groups_per_source": -1},
    {"length_weights": {2: 1, 4: 1, 8: 1, 16: 1}},
    {"source_length_weights": {"news": {2: 1}}},
    {"source_length_weights": {"news": {2: 1}, "code": {32: 1}}},
    {"source_length_weights": {"news": {2: float("nan")}, "code": {8: 1}}},
    {"source_length_weights": {"news": {"2": 1}, "code": {8: 1}}},
])
def test_config_rejects_invalid_conditional_recipe(tmp_path, change):
    raw = {"block_counts": [2, 4, 8, 16], "source_weights": {"code": .8, "news": .2},
           "source_length_weights": {"code": {8: .5, 16: .5}, "news": {2: .9, 4: .1}}}
    raw.update(change)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"raw": raw}), encoding="utf-8")
    with pytest.raises(ValueError, match="raw\\."):
        ExperimentConfig.load(path)

"""Offline checks of legacy parity, trajectory order, splits, and label isolation."""
from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import random
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delta_ngram.data import (
    AffixEpisode, AffixRule, Episode, EpisodeProtocol, Query, WriteEvent,
    affix_family_pools, build_manifest, entity_pools, episode_count, episode_events,
    episode_queries, from_manifest, legacy_fact_manifest, manifest_sha256,
    sample_episode, sample_fact_episode, sample_training_episode, to_manifest,
    validation_episodes,
)


def _legacy_fixtures():
    """Checked-in outputs extracted once from the archived executed pilot."""
    fixture = json.loads((Path(__file__).parent / "fixtures/legacy_fact_seed_fixtures.json").read_text(encoding="utf-8"))
    assert fixture["fixture_schema_version"] == 1
    assert fixture["source_sha256"] == "6220ab4cf610c51a1e1923b0db221b8284889a8a6e0b3ba19fa7613b6fbc7d04"
    assert fixture["validation_episodes_per_seed"] == 16
    assert fixture["training_steps"] == fixture["training_batch_episodes"] == 3
    assert [case["seed"] for case in fixture["cases"]] == [0, 17, 29]
    return fixture["cases"]


def _assert_legacy_text(episode, expected):
    assert episode.episode_id == expected["episode_id"]
    assert [item["index"] for item in expected["items"]] == [0, 1]
    for item in expected["items"]:
        index = item["index"]
        assert episode.question(index) == item["question"]
        assert episode.answer(index) == item["answer"]
        assert episode.evidence(index) == item["evidence"]
        assert episode.evidence(index, wrong=True) == item["wrong_evidence"]
        assert episode.evidence_color(index) == item["evidence_color"]
        assert episode.evidence_color(index, wrong=True) == item["wrong_evidence_color"]


def test_legacy_manifest_golden_hash_and_exact_record_shape():
    manifest = legacy_fact_manifest(17, 16)
    assert manifest_sha256(manifest) == "af2b73afa4fb721a03eb3f375fb0f9995b0cdf1bc4ac323dcee0e6a0d9a9058e"
    assert [field.name for field in fields(Episode)] == ["episode_id", "names", "colors"]
    assert set(manifest) == {"train_entity_pool", "validation_entity_pool", "validation_episodes",
                             "training_rng_seed", "worlds_generated_online", "metadata_is_not_model_input"}
    loaded = json.loads(json.dumps(manifest))
    assert manifest_sha256(loaded) == manifest_sha256(manifest)
    assert to_manifest(validation_episodes(loaded)[0]) == manifest["validation_episodes"][0]


def test_archived_pilot_rng_text_and_semantic_json_parity():
    for case in _legacy_fixtures():
        seed, expected = case["seed"], case["manifest"]
        old_train, old_validation = expected["train_entity_pool"], expected["validation_entity_pool"]
        assert entity_pools(seed) == (old_train, old_validation)
        new_rng = random.Random(seed + 29009)
        assert len(case["validation_text"]) == len(expected["validation_episodes"]) == 16
        for index in range(16):
            episode_id = f"validation-{index:06d}"
            new_episode = sample_episode(new_rng, old_validation, episode_id)
            assert json.loads(json.dumps(asdict(new_episode))) == expected["validation_episodes"][index]
            _assert_legacy_text(new_episode, case["validation_text"][index])
        assert json.loads(json.dumps(new_rng.getstate())) == case["validation_rng_state_after"]
        actual = legacy_fact_manifest(seed, 16)
        assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
        assert manifest_sha256(actual) == case["manifest_sha256"]
        assert case["manifest_sha256"] == hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
        new_rng = random.Random(seed + 39019)
        assert [(item["step"], item["item"]) for item in case["training"]] == [
            (step, item) for step in range(1, 4) for item in range(3)]
        for item in case["training"]:
            episode_id = f"train-{item['step']:06d}-{item['item']}"
            episode = sample_training_episode(actual, new_rng, episode_id)
            assert json.loads(json.dumps(asdict(episode))) == item["record"]
            _assert_legacy_text(episode, item["text"])
        assert json.loads(json.dumps(new_rng.getstate())) == case["training_rng_state_after"]


def test_scalable_fact_event_order_and_no_future_queries():
    names, _ = entity_pools(17)
    episode = sample_fact_episode(random.Random(2), names, "private-audit-identifier", facts=4)
    assert isinstance(episode, EpisodeProtocol)
    assert episode_count(episode) == len(episode) == 4
    assert [query.index for query in episode_queries(episode, 1)] == [0]
    assert [query.index for query in episode_queries(episode, 2)] == [1, 0]
    assert [query.index for query in episode_queries(episode, 4)] == [3, 0, 1, 2]
    events = list(episode_events(episode))
    seen = set()
    for event in events:
        if isinstance(event, WriteEvent):
            seen.add(event.index)
            assert event.evidence == episode.evidence(event.index)
            assert episode.episode_id not in event.evidence
            assert not hasattr(event, "answer")
        else:
            assert isinstance(event, Query) and event.index in seen
            assert event.question == episode.question(event.index)
            assert event.answer == episode.answer(event.index)
    assert len(seen) == 4
    assert episode_queries(episode, 0) == ()
    for stage in (-1, 5):
        with pytest.raises(ValueError, match="stage"):
            episode_queries(episode, stage)


@pytest.mark.parametrize("task", ["fact", "affix", "mixed"])
def test_manifest_roundtrip_determinism_and_split_isolation(task):
    manifest = build_manifest(task=task, seed=17, eval_episodes=8, facts_per_episode=4)
    assert manifest == build_manifest(task=task, seed=17, eval_episodes=8, facts_per_episode=4)
    loaded = json.loads(json.dumps(manifest))
    validation = validation_episodes(loaded)
    assert len(validation) == 8
    assert not set(manifest["train_entity_pool"]) & set(manifest["validation_entity_pool"])
    rng = random.Random(manifest["training_rng_seed"])
    training = [sample_training_episode(loaded, rng, f"train-{index}") for index in range(8)]
    for split, episodes in (("train", training), ("validation", validation)):
        for episode in episodes:
            assert from_manifest(json.loads(json.dumps(to_manifest(episode)))) == episode
            assert len(episode) == 4
            names = episode.names if isinstance(episode, Episode) else episode.languages
            assert set(names) <= set(manifest[f"{split}_entity_pool"])
            if isinstance(episode, AffixEpisode):
                assert {rule.family for rule in episode.rules} <= set(manifest[f"{split}_rule_families"])
                stems = {stem for support in episode.support_stems for stem in support} | set(episode.query_stems)
                assert stems <= set(manifest[f"{split}_stem_pool"])
    if task != "fact":
        assert not set(manifest["train_rule_families"]) & set(manifest["validation_rule_families"])
        assert not set(manifest["train_stem_pool"]) & set(manifest["validation_stem_pool"])
        assert (manifest["train_rule_families"], manifest["validation_rule_families"]) == affix_family_pools(17)


def test_fact_wrapper_preserves_legacy_samples_and_training_stream():
    legacy, wrapped = legacy_fact_manifest(17, 16), build_manifest(seed=17, eval_episodes=16)
    for key, value in legacy.items():
        assert wrapped[key] == value
    left, right = random.Random(legacy["training_rng_seed"]), random.Random(wrapped["training_rng_seed"])
    for index in range(12):
        assert sample_training_episode(legacy, left, str(index)) == sample_training_episode(wrapped, right, str(index))


def test_affix_queries_are_unseen_and_evidence_excludes_audit_metadata_and_gold():
    manifest = build_manifest(task="affix", eval_episodes=12, facts_per_episode=3)
    for episode in validation_episodes(manifest):
        evidence = "\n".join(episode.evidence(index) for index in range(len(episode)))
        support = {stem for stems in episode.support_stems for stem in stems}
        assert not support & set(episode.query_stems)
        assert episode.episode_id not in evidence
        for index, rule in enumerate(episode.rules):
            assert rule.family not in evidence
            assert episode.query_stems[index] not in evidence
            assert episode.answer(index).strip() not in evidence
            assert episode.answer(index) == " " + rule.apply(episode.query_stems[index])
            assert episode.corrupted_answer(index) != episode.answer(index)
            assert episode.corrupted_evidence(index) != episode.evidence(index)
            assert episode.corrupted_evidence(index) == episode.evidence(index, wrong=True)
            for stem in episode.support_stems[index]:
                assert f"{stem} -> {rule.corrupted().apply(stem)}\n" in episode.corrupted_evidence(index)


def test_multiple_worlds_change_answers_without_changing_question_or_audit_leak():
    first = Episode("audit-a", ("NameA", "NameB"), ("red", "blue"))
    second = Episode("audit-b", first.names, ("green", "yellow"))
    assert first.question(0) == second.question(0)
    assert first.answer(0) != second.answer(0)
    assert first.evidence(0) != second.evidence(0)
    rule_a, rule_b = AffixRule("prefix", "za"), AffixRule("prefix", "lu")
    first = AffixEpisode("audit-a", ("LanguageA",), (rule_a,), (("supporta", "supportb"),), ("querystem",))
    second = AffixEpisode("audit-b", first.languages, (rule_b,), first.support_stems, first.query_stems)
    assert first.question(0) == second.question(0)
    assert first.answer(0) != second.answer(0)
    assert first.evidence(0) != second.evidence(0)


def test_mixed_draw_selects_whole_episode_and_is_seed_reproducible():
    manifest = build_manifest(task="mixed", eval_episodes=32)
    episodes = validation_episodes(manifest)
    assert {type(episode) for episode in episodes} == {Episode, AffixEpisode}
    for probability, expected in ((0.0, AffixEpisode), (1.0, Episode)):
        selected = build_manifest(task="mixed", eval_episodes=5, fact_probability=probability)
        assert all(isinstance(episode, expected) for episode in validation_episodes(selected))
    first, second = random.Random(23), random.Random(23)
    for index in range(20):
        assert sample_training_episode(manifest, first, str(index)) == sample_training_episode(manifest, second, str(index))


def test_invalid_generator_settings_and_future_manifest_versions_fail_loudly():
    for kwargs in ({"task": "unknown"}, {"facts_per_episode": 0}, {"support_examples": 0},
                   {"eval_episodes": -1}, {"fact_probability": 1.1},
                   {"task": "affix", "facts_per_episode": 63},
                   {"task": "mixed", "facts_per_episode": 63, "fact_probability": 1.0}):
        with pytest.raises(ValueError):
            build_manifest(**kwargs)
    future = {**legacy_fact_manifest(17, 1), "manifest_version": 999}
    with pytest.raises(ValueError, match="version"):
        validation_episodes(future)
    with pytest.raises(ValueError, match="version"):
        sample_training_episode(future, random.Random(0), "audit")
    with pytest.raises(ValueError, match="disjoint"):
        AffixEpisode("audit", ("LanguageA",), (AffixRule("suffix", suffix="ek"),), (("same",),), ("same",))

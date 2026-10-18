"""Versioned fact probes must never treat an intentionally replaced value as forgotten."""
from copy import deepcopy
from dataclasses import asdict
import json
import random

import pytest

from delta_ngram.data import (
    ATTRIBUTE_VALUES, COLORS, E1_MANIFEST_VERSION, MANIFEST_VERSION, FRUITS,
    AffixEpisode, AffixRule, E1Episode, Episode, EpisodeProtocol, FactEvent, Query,
    WriteEvent, build_manifest, entity_pools, episode_count, episode_events,
    episode_queries, from_manifest, legacy_fact_manifest, manifest_sha256,
    question_for_style, sample_e1_episode, sample_training_episode, to_manifest,
    validate_e1_scenario, validate_manifest, validation_episodes,
)


def _revision_episode():
    return E1Episode("private-audit-id", (
        FactEvent(("NameA", "favorite_color"), "red", "acquisition"),
        FactEvent(("NameA", "favorite_fruit"), "pear", "acquisition"),
        FactEvent(("NameA", "favorite_color"), "blue", "revision"),
        FactEvent(("NameB", "favorite_color"), "green", "acquisition"),
    ))


def test_exact_text_boundaries_and_static_gold_for_each_fact_version():
    episode = _revision_episode()
    assert isinstance(episode, EpisodeProtocol)
    assert episode_count(episode) == len(episode) == 4
    assert episode.evidence(0) == "Profile\nName: NameA\nFavorite color: red\n"
    assert episode.evidence(1) == "Profile\nName: NameA\nFavorite fruit: pear\n"
    assert episode.evidence(2) == "Update\nName: NameA\nFavorite color: blue\n"
    assert episode.question(0) == episode.question(2) == "Profile\nName: NameA\nFavorite color:"
    assert episode.question(1) == "Profile\nName: NameA\nFavorite fruit:"
    assert episode.answer(0) == " red" and episode.answer(2) == " blue"
    assert episode.fact_key(0) == episode.fact_key(2) == ("NameA", "favorite_color")
    assert episode.fact_key(1) == ("NameA", "favorite_fruit")
    assert [episode.event_kind(index) for index in range(4)] == [
        "acquisition", "acquisition", "revision", "acquisition"]
    for index in range(len(episode)):
        assert episode.episode_id not in episode.evidence(index)
        assert "favorite_" not in episode.evidence(index)
        assert "acquisition" not in episode.evidence(index)
        assert "revision" not in episode.evidence(index)
        assert episode.question(index, "named_question") == question_for_style(episode, index, "named_question")
    assert episode.question(2, "named_question") == "What is NameA's favorite color? Answer with one word:"
    assert episode.question(1, "named_question") == "What is NameA's favorite fruit? Answer with one word:"


def test_active_versions_supersession_and_probe_abilities_have_exact_stage_semantics():
    episode = _revision_episode()
    assert [episode.active_query_indices(stage) for stage in range(5)] == [
        (), (0,), (0, 1), (1, 2), (1, 2, 3)]
    assert [episode.superseded_at(index) for index in range(4)] == [3, None, None, None]
    assert [episode.previous_version(index) for index in range(4)] == [None, None, 0, None]
    expected = [[], [(0, "acquisition")], [(1, "acquisition"), (0, "retention")],
                [(2, "revision"), (1, "retention")],
                [(3, "acquisition"), (1, "retention"), (2, "retention")]]
    for stage, pairs in enumerate(expected):
        queries = episode_queries(episode, stage)
        assert [(query.index, query.ability) for query in queries] == pairs
        assert all(query.index < stage and query.answer == episode.answer(query.index) for query in queries)
    assert episode.answer(0) == " red"  # Historical gold is not silently rewritten.
    events = list(episode_events(episode))
    stage = 0
    for item in events:
        if isinstance(item, WriteEvent):
            stage += 1
            assert item.evidence == episode.evidence(stage - 1)
            assert not hasattr(item, "answer")
        else:
            assert isinstance(item, Query)
            assert item.index in episode.active_query_indices(stage)
    assert stage == 4


def test_multiple_revision_chain_tracks_immediate_versions_and_latest_return_to_old_value():
    original = _revision_episode()
    episode = E1Episode(original.episode_id, (*original.events,
        FactEvent(("NameA", "favorite_color"), "red", "revision")))
    assert episode.active_query_indices(5) == (1, 3, 4)
    assert episode.superseded_at(0) == 3 and episode.superseded_at(2) == 5
    assert episode.previous_version(4) == 2
    assert episode.answer(4) == episode.answer(0)
    assert [query.index for query in episode_queries(episode, 5)] == [4, 1, 3]


def test_question_variants_are_pure_and_legacy_canonical_remains_exact():
    names, _ = entity_pools(17)
    rng = random.Random(23)
    episode = sample_e1_episode(rng, names, "audit", scenario="revision")
    record = asdict(episode)
    evidence = [episode.evidence(index) for index in range(len(episode))]
    local_state, global_state = rng.getstate(), random.getstate()
    for index in range(len(episode)):
        for style in ("canonical", "named_question"):
            question_for_style(episode, index, style)
    assert rng.getstate() == local_state and random.getstate() == global_state
    assert asdict(episode) == record
    assert [episode.evidence(index) for index in range(len(episode))] == evidence
    legacy = Episode("legacy", ("NameA", "NameB"), ("red", "blue"))
    assert question_for_style(legacy, 0) == legacy.question(0)
    assert question_for_style(legacy, 0, "named_question") == "What is NameA's favorite color? Answer with one word:"
    assert [query.index for query in episode_queries(legacy, 2)] == [1, 0]
    affix = AffixEpisode("affix", ("LanguageA",), (AffixRule("prefix", "za"),), (("support",),), ("query",))
    assert question_for_style(affix, 0) == affix.question(0)
    with pytest.raises(ValueError, match="unsupported"):
        question_for_style(affix, 0, "named_question")
    for item in (legacy, episode):
        with pytest.raises(ValueError, match="style"):
            question_for_style(item, 0, "pronoun")


def test_wrong_evidence_corruption_preserves_keys_attributes_and_revision_inequality():
    episode = _revision_episode()
    assert ATTRIBUTE_VALUES == {"favorite_color": COLORS, "favorite_fruit": FRUITS}
    assert FRUITS == ("apple", "pear", "mango", "peach", "plum", "grape", "lemon", "melon")
    for index, event in enumerate(episode.events):
        answer = episode.corrupted_answer(index)
        assert answer != episode.answer(index)
        assert answer.strip() in ATTRIBUTE_VALUES[event.key[1]]
        assert episode.corrupted_evidence(index) == episode.evidence(index, wrong=True)
        assert episode.corrupted_evidence(index) == episode.evidence(index).replace(
            ": " + event.value + "\n", ":" + answer + "\n")
    assert episode.corrupted_answer(0) != episode.corrupted_answer(2)
    assert episode.corrupted_evidence(2).startswith("Update\n")


@pytest.mark.parametrize("scenario", ["independent", "revision"])
@pytest.mark.parametrize("writes", [4, 16, 256])
def test_e1_sampling_roundtrip_determinism_and_split_membership(scenario, writes):
    manifest = build_manifest(task="fact_e1", seed=17, facts_per_episode=writes,
                              eval_episodes=3, e1_scenario=scenario)
    assert manifest == build_manifest(task="fact_e1", seed=17, facts_per_episode=writes,
                                     eval_episodes=3, e1_scenario=scenario)
    assert manifest["manifest_version"] == E1_MANIFEST_VERSION == 2
    assert manifest["generator_config"]["e1_scenario"] == scenario
    train_names, validation_names = entity_pools(17)
    assert manifest["train_entity_pool"] == train_names
    assert manifest["validation_entity_pool"] == validation_names
    assert not set(train_names) & set(validation_names)
    before = deepcopy(manifest)
    validate_manifest(manifest, expected_task="fact_e1", expected_eval_episodes=3,
                      expected_facts_per_episode=writes, expected_e1_scenario=scenario)
    loaded = json.loads(json.dumps(manifest))
    assert manifest_sha256(loaded) == manifest_sha256(manifest)
    validation = validation_episodes(loaded)
    left, right = random.Random(manifest["training_rng_seed"]), random.Random(manifest["training_rng_seed"])
    training = [sample_training_episode(manifest, left, f"train-{index}") for index in range(3)]
    assert training == [sample_training_episode(loaded, right, f"train-{index}") for index in range(3)]
    assert left.getstate() == right.getstate()
    for split, episodes in (("train", training), ("validation", validation)):
        for episode in episodes:
            assert isinstance(episode, E1Episode) and len(episode) == writes
            validate_e1_scenario(episode, scenario)
            assert {event.key[0] for event in episode.events} <= set(manifest[f"{split}_entity_pool"])
            assert from_manifest(json.loads(json.dumps(to_manifest(episode)))) == episode
            if scenario == "revision":
                assert episode.events[0].key == episode.events[2].key
                assert episode.events[0].value != episode.events[2].value
                assert episode.events[1].key[0] == episode.events[0].key[0]
                assert len(episode.active_query_indices(writes)) == writes - 1
            else:
                assert len(episode.active_query_indices(writes)) == writes
    assert manifest == before


def test_e1_addition_does_not_upgrade_or_extend_legacy_or_v1_manifests():
    assert MANIFEST_VERSION == 1
    assert manifest_sha256(legacy_fact_manifest(17, 16)) == "af2b73afa4fb721a03eb3f375fb0f9995b0cdf1bc4ac323dcee0e6a0d9a9058e"
    for task in ("fact", "affix", "mixed"):
        manifest = build_manifest(task=task, eval_episodes=2)
        assert manifest["manifest_version"] == 1
        assert "e1_scenario" not in manifest["generator_config"]
        assert all("events" not in record for record in manifest["validation_episodes"])
        validate_manifest(manifest)


@pytest.mark.parametrize("expected", [
    {"expected_task": "fact"}, {"expected_eval_episodes": 3},
    {"expected_facts_per_episode": 16}, {"expected_e1_scenario": "independent"},
])
def test_e1_saved_manifest_configuration_expectations_are_checked(expected):
    manifest = build_manifest(task="fact_e1", facts_per_episode=4, eval_episodes=2,
                              e1_scenario="revision")
    with pytest.raises(ValueError, match="differs from the configured data"):
        validate_manifest(manifest, **expected)


@pytest.mark.parametrize("mutation", [
    "version_one", "version_two_other_task", "missing_version", "missing_scenario", "unknown_scenario",
    "too_few_writes", "different_scenario", "record_task", "missing_record_task", "hybrid_record",
    "event_count", "event_not_mapping", "key_shape", "key_attribute", "unknown_value", "cross_split_entity",
    "unknown_kind", "first_event_revision", "reacquisition", "unchanged_revision", "different_entity_fruit",
    "noncolor_later_write", "same_entity_later_write",
])
def test_e1_replay_rejects_illegal_versions_splits_keys_values_and_scenarios(mutation):
    manifest = json.loads(json.dumps(build_manifest(task="fact_e1", facts_per_episode=4,
                                    eval_episodes=2, e1_scenario="revision")))
    config = manifest["generator_config"]
    record = manifest["validation_episodes"][0]
    events = record["events"]
    if mutation == "version_one":
        manifest["manifest_version"] = 1
    elif mutation == "version_two_other_task":
        manifest["task"] = "fact"
    elif mutation == "missing_version":
        del manifest["manifest_version"]
    elif mutation == "missing_scenario":
        del config["e1_scenario"]
    elif mutation == "unknown_scenario":
        config["e1_scenario"] = "pronoun"
    elif mutation == "too_few_writes":
        config["facts_per_episode"] = 2
    elif mutation == "different_scenario":
        config["e1_scenario"] = "independent"
    elif mutation == "record_task":
        record["task"] = "fact"
    elif mutation == "missing_record_task":
        del record["task"]
    elif mutation == "hybrid_record":
        record["names"] = [events[0]["key"][0]]
    elif mutation == "event_count":
        events.pop()
    elif mutation == "event_not_mapping":
        events[0] = "event"
    elif mutation == "key_shape":
        events[0]["key"] = events[0]["key"][0]
    elif mutation == "key_attribute":
        events[0]["key"][1] = "favorite_animal"
    elif mutation == "unknown_value":
        events[1]["value"] = "red"
    elif mutation == "cross_split_entity":
        events[0]["key"][0] = manifest["train_entity_pool"][0]
    elif mutation == "unknown_kind":
        events[2]["kind"] = "forgetting"
    elif mutation == "first_event_revision":
        events[0]["kind"] = "revision"
    elif mutation == "reacquisition":
        events[2]["kind"] = "acquisition"
    elif mutation == "unchanged_revision":
        events[2]["value"] = events[0]["value"]
    elif mutation == "different_entity_fruit":
        events[1]["key"][0] = events[3]["key"][0]
    elif mutation == "noncolor_later_write":
        events[3]["key"][1] = "favorite_fruit"
        events[3]["value"] = "apple"
    else:
        events[3]["key"] = list(events[0]["key"])
        events[3]["kind"] = "revision"
        events[3]["value"] = next(color for color in COLORS if color != events[2]["value"])
    with pytest.raises(ValueError, match="manifest"):
        validation_episodes(manifest)


def test_revision_capacity_counts_unique_entities_instead_of_writes():
    manifest = build_manifest(task="fact_e1", facts_per_episode=3, eval_episodes=1,
                              e1_scenario="revision")
    name = manifest["validation_episodes"][0]["events"][0]["key"][0]
    manifest["validation_entity_pool"] = [name]
    manifest["train_entity_pool"] = manifest["train_entity_pool"][:1]
    validate_manifest(manifest)
    assert len(sample_training_episode(manifest, random.Random(0), "train")) == 3
    empty = build_manifest(task="fact_e1", facts_per_episode=4, eval_episodes=0,
                           e1_scenario="revision")
    assert validation_episodes(empty) == []


@pytest.mark.parametrize("stage", [-1, 5, True, 1.5])
def test_e1_invalid_stages_are_rejected(stage):
    with pytest.raises(ValueError, match="stage"):
        episode_queries(_revision_episode(), stage)


@pytest.mark.parametrize("index", [-1, 4, True, 0.5])
def test_e1_invalid_event_indices_are_rejected(index):
    episode = _revision_episode()
    for method in (episode.question, episode.answer, episode.evidence, episode.fact_key,
                   episode.event_kind, episode.superseded_at, episode.previous_version):
        with pytest.raises(ValueError, match="index"):
            method(index)


@pytest.mark.parametrize("kwargs", [
    {"scenario": "unknown"}, {"writes": 0}, {"writes": True}, {"writes": 1.5},
    {"writes": 2, "scenario": "revision"}, {"writes": 4},
])
def test_e1_invalid_generator_settings_fail(kwargs):
    with pytest.raises(ValueError):
        sample_e1_episode(random.Random(0), ["NameA", "NameB", "NameC"], "audit", **kwargs)

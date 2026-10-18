"""Replay manifests retain the generator's split and task boundaries."""
from copy import deepcopy
import json
import random

import pytest
import torch

from delta_ngram.data import (
    AffixRule, build_manifest, legacy_fact_manifest, manifest_sha256,
    sample_training_episode, validate_manifest, validation_episodes,
)


def test_legacy_validation_preserves_exact_dictionary_hash_and_all_rng_states():
    manifest = legacy_fact_manifest(17, 16)
    before = deepcopy(manifest)
    serialized = json.dumps(manifest, sort_keys=True)
    python_rng = random.getstate()
    torch_rng = torch.random.get_rng_state().clone()
    validate_manifest(manifest, expected_task="fact", expected_eval_episodes=16,
                      expected_facts_per_episode=2, expected_support_examples=99)
    assert len(validation_episodes(manifest)) == 16
    assert manifest == before and json.dumps(manifest, sort_keys=True) == serialized
    assert manifest_sha256(manifest) == "af2b73afa4fb721a03eb3f375fb0f9995b0cdf1bc4ac323dcee0e6a0d9a9058e"
    assert random.getstate() == python_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)


@pytest.mark.parametrize("task", ["fact", "affix", "mixed"])
def test_correct_versioned_manifests_roundtrip_with_matching_config(task):
    manifest = build_manifest(task=task, facts_per_episode=3, support_examples=2, eval_episodes=7)
    before = deepcopy(manifest)
    validate_manifest(manifest, expected_task=task, expected_eval_episodes=7,
                      expected_facts_per_episode=3, expected_support_examples=2)
    reloaded = json.loads(json.dumps(manifest))
    validate_manifest(reloaded)
    assert validation_episodes(manifest) == validation_episodes(reloaded)
    assert manifest == before
    left, right = random.Random(manifest["training_rng_seed"]), random.Random(manifest["training_rng_seed"])
    # Validation must not change the exact online stream, including mixed draws.
    for index in range(3):
        assert sample_training_episode(manifest, left, str(index)) == sample_training_episode(before, right, str(index))


@pytest.mark.parametrize("suffix", ["entity_pool", "stem_pool", "rule_families"])
def test_split_intersections_are_rejected_by_public_replay_entrypoint(suffix):
    manifest = build_manifest(task="affix", eval_episodes=2)
    manifest[f"train_{suffix}"] = list(manifest[f"validation_{suffix}"])
    with pytest.raises(ValueError, match=f"{suffix} must be disjoint"):
        validation_episodes(manifest)


@pytest.mark.parametrize("field", ["names", "languages", "support_stems", "query_stems", "rules"])
def test_each_validation_record_must_belong_to_its_declared_split(field):
    manifest = json.loads(json.dumps(build_manifest(task="fact" if field == "names" else "affix", eval_episodes=2)))
    record = manifest["validation_episodes"][0]
    if field in ("names", "languages"):
        record[field][0] = manifest["train_entity_pool"][0]
    elif field == "support_stems":
        record[field][0][0] = manifest["train_stem_pool"][0]
    elif field == "query_stems":
        record[field][0] = manifest["train_stem_pool"][0]
    else:
        family = manifest["train_rule_families"][0]
        rule = AffixRule(family, "za" if family in ("prefix", "prefix_suffix") else "",
                        "ek" if family in ("suffix", "prefix_suffix") else "")
        record[field][0] = {"family": rule.family, "prefix": rule.prefix, "suffix": rule.suffix}
    with pytest.raises(ValueError, match="outside the validation"):
        validate_manifest(manifest)


def test_record_type_and_manifest_task_must_agree():
    fact = build_manifest(task="fact", eval_episodes=1)
    affix = build_manifest(task="affix", eval_episodes=1)
    fact["validation_episodes"] = affix["validation_episodes"]
    with pytest.raises(ValueError, match="task/type"):
        validate_manifest(fact)
    affix["validation_episodes"] = legacy_fact_manifest(17, 1)["validation_episodes"]
    with pytest.raises(ValueError, match="task/type"):
        validate_manifest(affix)
    malformed = build_manifest(task="affix", eval_episodes=1)
    malformed["task"] = "fact"
    with pytest.raises(ValueError, match="fact task cannot contain affix"):
        validate_manifest(malformed)


@pytest.mark.parametrize("expected", [
    {"expected_task": "mixed"}, {"expected_eval_episodes": 4},
    {"expected_facts_per_episode": 4}, {"expected_support_examples": 4},
])
def test_yaml_expectations_are_checked_without_reinterpreting_saved_data(expected):
    manifest = build_manifest(task="affix", eval_episodes=2, facts_per_episode=2, support_examples=3)
    with pytest.raises(ValueError, match="differs from the configured data"):
        validate_manifest(manifest, **expected)


@pytest.mark.parametrize("mutation", [
    "version", "boolean_version", "missing_version", "missing_config", "boolean_count",
    "wrong_rng_seed", "wrong_eval_count", "wrong_fact_count", "wrong_support_count",
    "duplicate_id", "duplicate_entity", "unknown_color", "unknown_family",
    "support_query_overlap", "invalid_rule_parameters", "mixed_nonfinite_probability",
])
def test_malformed_or_self_inconsistent_schema_is_rejected(mutation):
    manifest = json.loads(json.dumps(build_manifest(task="mixed", eval_episodes=16)))
    fact = next(record for record in manifest["validation_episodes"] if record.get("task", "fact") == "fact")
    affix = next(record for record in manifest["validation_episodes"] if record.get("task") == "affix")
    if mutation == "version":
        manifest["manifest_version"] = 999
    elif mutation == "boolean_version":
        manifest["manifest_version"] = True
    elif mutation == "missing_version":
        del manifest["manifest_version"]
    elif mutation == "missing_config":
        del manifest["generator_config"]
    elif mutation == "boolean_count":
        manifest["generator_config"]["facts_per_episode"] = True
    elif mutation == "wrong_rng_seed":
        manifest["training_rng_seed"] += 1
    elif mutation == "wrong_eval_count":
        manifest["validation_episodes"].pop()
    elif mutation == "wrong_fact_count":
        fact["names"].pop()
    elif mutation == "wrong_support_count":
        affix["support_stems"][0].pop()
    elif mutation == "duplicate_id":
        manifest["validation_episodes"][1]["episode_id"] = manifest["validation_episodes"][0]["episode_id"]
    elif mutation == "duplicate_entity":
        fact["names"][1] = fact["names"][0]
    elif mutation == "unknown_color":
        fact["colors"][0] = "not-a-color"
    elif mutation == "unknown_family":
        manifest["train_rule_families"].append("unknown")
    elif mutation == "support_query_overlap":
        affix["query_stems"][0] = affix["support_stems"][0][0]
    elif mutation == "invalid_rule_parameters":
        affix["rules"][0]["prefix"] = "invalid"
    else:
        manifest["generator_config"]["fact_probability"] = float("nan")
    with pytest.raises(ValueError, match="manifest"):
        validate_manifest(manifest)


def test_zero_validation_episodes_remains_valid_at_the_data_layer():
    for manifest in (legacy_fact_manifest(17, 0), build_manifest(task="affix", eval_episodes=0)):
        validate_manifest(manifest, expected_eval_episodes=0)
        assert validation_episodes(manifest) == []

"""Versioned reproducible manifests and exact legacy-fact compatibility."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import random
from typing import Any, Mapping

from .affix import (AFFIX_FAMILIES, AffixEpisode, AffixRule, affix_family_pools,
                    affix_stem_pools, sample_affix_episode)
from .facts import COLORS, Episode, entity_pools, sample_episode, sample_fact_episode


MANIFEST_VERSION = 1
GeneratedEpisode = Episode | AffixEpisode


def _manifest_integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"manifest {label} must be an integer{suffix}")
    return value


def _manifest_strings(value: Any, label: str, *, unique: bool = True) -> tuple[str, ...]:
    if (not isinstance(value, (list, tuple)) or not value
            or any(not isinstance(item, str) or not item for item in value)):
        raise ValueError(f"manifest {label} must be a nonempty sequence of nonempty strings")
    if unique and len(set(value)) != len(value):
        raise ValueError(f"manifest {label} must contain unique values")
    return tuple(value)


def _manifest_split(manifest: Mapping[str, Any], suffix: str) -> tuple[set[str], set[str]]:
    train = set(_manifest_strings(manifest.get(f"train_{suffix}"), f"train_{suffix}"))
    validation = set(_manifest_strings(manifest.get(f"validation_{suffix}"), f"validation_{suffix}"))
    if train & validation:
        raise ValueError(f"manifest train/validation {suffix} must be disjoint")
    return train, validation


def validate_manifest(manifest: Mapping[str, Any], *, expected_task: str | None = None,
                      expected_eval_episodes: int | None = None,
                      expected_facts_per_episode: int | None = None,
                      expected_support_examples: int | None = None) -> None:
    """Validate replayed data, split membership, and optional caller expectations.

    This is a load-time check; it does not mutate the manifest or consume RNG.
    The unversioned historical schema means two facts per episode. Versioned
    schemas declare their generator configuration explicitly. Support-example
    expectations apply only to affix/mixed tasks, since facts have no supports.
    Metadata can contain extra audit fields, but reserved episode fields cannot
    contradict the declared task. Online sampling assumes this check ran once.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    if "manifest_version" not in manifest:
        if "task" in manifest or "generator_config" in manifest:
            raise ValueError("manifest task/generator_config require manifest_version")
        task, facts, supports, declared_evaluation_count = "fact", 2, None, None
    else:
        version = _manifest_integer(manifest["manifest_version"], "manifest_version", minimum=1)
        if version != MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest version: {version}")
        task = manifest.get("task")
        if task not in ("fact", "affix", "mixed"):
            raise ValueError("manifest task must be 'fact', 'affix', or 'mixed'")
        config = manifest.get("generator_config")
        if not isinstance(config, Mapping):
            raise ValueError("manifest generator_config must be a mapping")
        seed = _manifest_integer(config.get("seed"), "generator_config.seed")
        facts = _manifest_integer(config.get("facts_per_episode"), "generator_config.facts_per_episode", minimum=1)
        supports = _manifest_integer(config.get("support_examples"), "generator_config.support_examples", minimum=1)
        declared_evaluation_count = _manifest_integer(config.get("eval_episodes"), "generator_config.eval_episodes", minimum=0)
        if manifest.get("training_rng_seed") != seed + 39019:
            raise ValueError("manifest training_rng_seed disagrees with generator_config.seed")
        if task == "mixed":
            probability = config.get("fact_probability")
            if (type(probability) not in (int, float) or not 0.0 <= probability <= 1.0):
                raise ValueError("manifest generator_config.fact_probability must be finite and between zero and one")
    _manifest_integer(manifest.get("training_rng_seed"), "training_rng_seed")
    for field in ("worlds_generated_online", "metadata_is_not_model_input"):
        if manifest.get(field) is not True:
            raise ValueError(f"manifest {field} must be true for this generator schema")
    train_names, validation_names = _manifest_split(manifest, "entity_pool")
    if min(len(train_names), len(validation_names)) < facts:
        raise ValueError("manifest entity pools cannot supply facts_per_episode unique names")
    train_families = validation_families = train_stems = validation_stems = set()
    affix_pool_fields = {f"{split}_{suffix}" for split in ("train", "validation")
                         for suffix in ("rule_families", "stem_pool")}
    if task in ("affix", "mixed"):
        train_families, validation_families = _manifest_split(manifest, "rule_families")
        train_stems, validation_stems = _manifest_split(manifest, "stem_pool")
        if not (train_families | validation_families) <= set(AFFIX_FAMILIES):
            raise ValueError("manifest contains unknown semantic rule families")
        if min(len(train_stems), len(validation_stems)) < facts * (supports + 1):
            raise ValueError("manifest stem pools cannot supply disjoint support and query stems")
    elif affix_pool_fields & manifest.keys():
        raise ValueError("manifest fact task cannot contain affix split pools")
    records = manifest.get("validation_episodes")
    if not isinstance(records, (list, tuple)):
        raise ValueError("manifest validation_episodes must be a sequence")
    if declared_evaluation_count is not None and len(records) != declared_evaluation_count:
        raise ValueError("manifest validation count disagrees with generator_config.eval_episodes")
    expected_values = (("task", task, expected_task),
                       ("eval_episodes", len(records), expected_eval_episodes),
                       ("facts_per_episode", facts, expected_facts_per_episode))
    if task in ("affix", "mixed"):
        expected_values += (("support_examples", supports, expected_support_examples),)
    for field, actual, expected in expected_values:
        if expected is not None and (actual != expected or isinstance(actual, bool) != isinstance(expected, bool)):
            raise ValueError(f"manifest {field} differs from the configured data.{field}")
    seen_ids = set()
    for index, record in enumerate(records):
        label = f"validation_episodes[{index}]"
        if not isinstance(record, Mapping):
            raise ValueError(f"manifest {label} must be a mapping")
        episode_id = record.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id or episode_id in seen_ids:
            raise ValueError(f"manifest {label}.episode_id must be a unique nonempty string")
        seen_ids.add(episode_id)
        record_task = record.get("task", "fact")
        if record_task not in ("fact", "affix") or (task != "mixed" and record_task != task):
            raise ValueError(f"manifest {label} task/type disagrees with manifest task")
        if record_task == "fact":
            if {"languages", "rules", "support_stems", "query_stems"} & record.keys():
                raise ValueError(f"manifest {label} fact record contains affix fields")
            names = _manifest_strings(record.get("names"), f"{label}.names")
            colors = _manifest_strings(record.get("colors"), f"{label}.colors", unique=facts == 2)
            if len(names) != facts or len(colors) != facts:
                raise ValueError(f"manifest {label} fact count disagrees with facts_per_episode")
            if not set(names) <= validation_names:
                raise ValueError(f"manifest {label} names are outside the validation entity split")
            if not set(colors) <= set(COLORS):
                raise ValueError(f"manifest {label} contains unknown colors")
        else:
            if {"names", "colors"} & record.keys():
                raise ValueError(f"manifest {label} affix record contains fact fields")
            names = _manifest_strings(record.get("languages"), f"{label}.languages")
            queries = _manifest_strings(record.get("query_stems"), f"{label}.query_stems")
            if len(names) != facts or len(queries) != facts:
                raise ValueError(f"manifest {label} rule count disagrees with facts_per_episode")
            if not set(names) <= validation_names:
                raise ValueError(f"manifest {label} languages are outside the validation entity split")
            support_sets = record.get("support_stems")
            rules = record.get("rules")
            if (not isinstance(support_sets, (list, tuple)) or len(support_sets) != facts
                    or not isinstance(rules, (list, tuple)) or len(rules) != facts):
                raise ValueError(f"manifest {label} needs one support set and rule per language")
            for support in support_sets:
                stems = _manifest_strings(support, f"{label}.support_stems")
                if len(stems) != supports:
                    raise ValueError(f"manifest {label} support count disagrees with support_examples")
                if not set(stems) <= validation_stems:
                    raise ValueError(f"manifest {label} support stems are outside the validation stem split")
            if not set(queries) <= validation_stems:
                raise ValueError(f"manifest {label} query stems are outside the validation stem split")
            for rule in rules:
                if (not isinstance(rule, Mapping) or not isinstance(rule.get("family"), str)
                        or rule["family"] not in validation_families):
                    raise ValueError(f"manifest {label} rule family is outside the validation family split")
        # Reuse the episode's own semantic invariants, including globally
        # disjoint support/query stems and valid prefix/suffix parameters.
        try:
            from_manifest(record)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"manifest {label} is malformed: {error}") from error


def to_manifest(episode: GeneratedEpisode) -> dict[str, Any]:
    """Legacy facts retain their exact record shape; other task types are tagged."""
    if isinstance(episode, Episode):
        return asdict(episode)
    if isinstance(episode, AffixEpisode):
        return {"task": "affix", **asdict(episode)}
    raise TypeError(f"unsupported episode type: {type(episode).__name__}")


def from_manifest(record: Mapping[str, Any]) -> GeneratedEpisode:
    task = record.get("task", "fact")
    if task == "fact":
        names, colors = tuple(record["names"]), tuple(record["colors"])
        if not names or len(names) != len(colors):
            raise ValueError("fact records need equally sized nonempty names and colors")
        return Episode(str(record["episode_id"]), names, colors)
    if task == "affix":
        return AffixEpisode(str(record["episode_id"]), tuple(record["languages"]),
                            tuple(AffixRule(**rule) for rule in record["rules"]),
                            tuple(tuple(stems) for stems in record["support_stems"]),
                            tuple(record["query_stems"]))
    raise ValueError(f"unknown episode task: {task}")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Match the archived pilot's sorted semantic-JSON hash, including spacing."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def legacy_fact_manifest(seed: int, eval_episodes: int) -> dict[str, Any]:
    if eval_episodes < 0:
        raise ValueError("eval_episodes cannot be negative")
    train_names, validation_names = entity_pools(seed)
    validation_rng = random.Random(seed + 29009)
    validation = [sample_episode(validation_rng, validation_names, f"validation-{index:06d}")
                  for index in range(eval_episodes)]
    return {"train_entity_pool": train_names, "validation_entity_pool": validation_names,
            "validation_episodes": [asdict(episode) for episode in validation],
            "training_rng_seed": seed + 39019, "worlds_generated_online": True,
            "metadata_is_not_model_input": True}


def _sample(manifest: Mapping[str, Any], rng: random.Random, episode_id: str,
            *, validation: bool) -> GeneratedEpisode:
    version = manifest.get("manifest_version", MANIFEST_VERSION)
    if version != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version: {version}")
    config = manifest.get("generator_config", {})
    task = manifest.get("task", "fact")
    if task == "mixed":
        # Make one choice for the whole episode; never splice unrelated writes.
        task = "fact" if rng.random() < config.get("fact_probability", 0.5) else "affix"
    split = "validation" if validation else "train"
    names = manifest[f"{split}_entity_pool"]
    count = config.get("facts_per_episode", 2)
    if task == "fact":
        return sample_fact_episode(rng, names, episode_id, facts=count)
    if task == "affix":
        return sample_affix_episode(rng, names, episode_id,
                                    families=manifest[f"{split}_rule_families"],
                                    stems=manifest[f"{split}_stem_pool"], rules=count,
                                    support_examples=config.get("support_examples", 3))
    raise ValueError(f"unknown generator task: {task}")


def build_manifest(*, task: str = "fact", seed: int = 17, eval_episodes: int = 32,
                   facts_per_episode: int = 2, support_examples: int = 3,
                   fact_probability: float = 0.5) -> dict[str, Any]:
    """Fix all split pools before drawing held-out worlds or online training data.

    Use legacy_fact_manifest for a byte-equivalent legacy JSON representation;
    this wrapper deliberately records the generator version and configuration.
    """
    if task not in ("fact", "affix", "mixed"):
        raise ValueError("task must be 'fact', 'affix', or 'mixed'")
    if eval_episodes < 0 or facts_per_episode < 1 or support_examples < 1:
        raise ValueError("counts must be positive (eval_episodes may be zero)")
    if not 0.0 <= fact_probability <= 1.0:
        raise ValueError("fact_probability must be between zero and one")
    train_names, validation_names = entity_pools(seed)
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "task": task,
        "generator_config": {"seed": seed, "eval_episodes": eval_episodes,
                             "facts_per_episode": facts_per_episode,
                             "support_examples": support_examples},
        "train_entity_pool": train_names,
        "validation_entity_pool": validation_names,
        "training_rng_seed": seed + 39019,
        "worlds_generated_online": True,
        "metadata_is_not_model_input": True,
    }
    if task == "mixed":
        manifest["generator_config"]["fact_probability"] = fact_probability
    if task in ("affix", "mixed"):
        train_families, validation_families = affix_family_pools(seed)
        train_stems, validation_stems = affix_stem_pools(seed)
        manifest.update({"train_rule_families": train_families,
                         "validation_rule_families": validation_families,
                         "train_stem_pool": train_stems, "validation_stem_pool": validation_stems})
    # Validate both split capacities even when evaluation is disabled. These
    # independent RNGs cannot perturb the actual training/evaluation streams.
    for validation in (False, True):
        for component in (("fact", "affix") if task == "mixed" else (task,)):
            _sample({**manifest, "task": component}, random.Random(0), "capacity-check",
                    validation=validation)
    validation_rng = random.Random(seed + 29009)
    manifest["validation_episodes"] = [to_manifest(_sample(
        manifest, validation_rng, f"validation-{index:06d}", validation=True))
        for index in range(eval_episodes)]
    return manifest


def sample_training_episode(manifest: Mapping[str, Any], rng: random.Random,
                            episode_id: str) -> GeneratedEpisode:
    return _sample(manifest, rng, episode_id, validation=False)


def validation_episodes(manifest: Mapping[str, Any]) -> list[GeneratedEpisode]:
    validate_manifest(manifest)
    return [from_manifest(record) for record in manifest["validation_episodes"]]

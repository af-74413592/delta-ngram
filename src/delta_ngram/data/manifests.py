"""Versioned reproducible manifests and exact legacy-fact compatibility."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import random
from typing import Any, Mapping

from .affix import AffixEpisode, AffixRule, affix_family_pools, affix_stem_pools, sample_affix_episode
from .facts import Episode, entity_pools, sample_episode, sample_fact_episode


MANIFEST_VERSION = 1
GeneratedEpisode = Episode | AffixEpisode


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
    version = manifest.get("manifest_version", MANIFEST_VERSION)
    if version != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version: {version}")
    return [from_manifest(record) for record in manifest["validation_episodes"]]

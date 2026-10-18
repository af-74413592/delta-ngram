"""Deterministic episode generators with explicit evidence/query separation."""
from .affix import (AFFIX_FAMILIES, PREFIXES, SUFFIXES, AffixEpisode, AffixRule,
                    affix_family_pools, affix_stem_pools, sample_affix_episode)
from .episodes import (EpisodeProtocol, Query, WriteEvent, episode_count,
                       episode_events, episode_queries)
from .facts import COLORS, Episode, FactEpisode, entity_pools, sample_episode, sample_fact_episode
from .manifests import (MANIFEST_VERSION, GeneratedEpisode, build_manifest,
                        from_manifest, legacy_fact_manifest, manifest_sha256,
                        sample_training_episode, to_manifest, validation_episodes)

__all__ = ["AFFIX_FAMILIES", "PREFIXES", "SUFFIXES", "AffixEpisode", "AffixRule",
           "affix_family_pools", "affix_stem_pools", "sample_affix_episode",
           "EpisodeProtocol", "Query", "WriteEvent", "episode_count", "episode_events",
           "episode_queries", "COLORS", "Episode", "FactEpisode", "entity_pools",
           "sample_episode", "sample_fact_episode", "MANIFEST_VERSION", "GeneratedEpisode",
           "build_manifest", "from_manifest", "legacy_fact_manifest", "manifest_sha256",
           "sample_training_episode", "to_manifest", "validation_episodes"]

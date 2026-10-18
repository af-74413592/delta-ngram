"""Synthetic rule-induction worlds with held-out structural rule families.

The three semantic families are prefix, suffix, and prefix-plus-suffix. A
deterministic split holds out one entire family, including all its lexical
parameters. Each evidence block gives examples for one named lexicon; its query
uses a stem absent from every support example in the episode.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence


AFFIX_FAMILIES = ("prefix", "suffix", "prefix_suffix")
PREFIXES = ("za", "lu", "fi", "vo", "ne", "tu", "ki", "po")
SUFFIXES = ("ek", "um", "ar", "is", "ot", "en", "ul", "as")


@dataclass(frozen=True)
class AffixRule:
    family: str
    prefix: str = ""
    suffix: str = ""

    def __post_init__(self) -> None:
        expected = {"prefix": (True, False), "suffix": (False, True),
                    "prefix_suffix": (True, True)}
        if self.family not in expected or (bool(self.prefix), bool(self.suffix)) != expected[self.family]:
            raise ValueError("affix parameters do not match the semantic rule family")
        if (self.prefix and self.prefix not in PREFIXES) or (self.suffix and self.suffix not in SUFFIXES):
            raise ValueError("unknown affix parameter")

    def apply(self, stem: str) -> str:
        return self.prefix + stem + self.suffix

    def corrupted(self) -> AffixRule:
        prefix = PREFIXES[(PREFIXES.index(self.prefix) + 1) % len(PREFIXES)] if self.prefix else ""
        suffix = SUFFIXES[(SUFFIXES.index(self.suffix) + 1) % len(SUFFIXES)] if self.suffix else ""
        return AffixRule(self.family, prefix, suffix)


@dataclass(frozen=True)
class AffixEpisode:
    episode_id: str
    languages: tuple[str, ...]
    rules: tuple[AffixRule, ...]
    support_stems: tuple[tuple[str, ...], ...]
    query_stems: tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.languages)
        if count == 0 or any(len(values) != count for values in
                             (self.rules, self.support_stems, self.query_stems)):
            raise ValueError("each lexicon needs one rule, support set, and query stem")
        if len(set(self.languages)) != count or any(not stems for stems in self.support_stems):
            raise ValueError("lexicon names must be unique and support sets nonempty")
        support = [stem for stems in self.support_stems for stem in stems]
        if len(set(support)) != len(support) or len(set(self.query_stems)) != count:
            raise ValueError("support/query stems must be unique within the episode")
        if set(support) & set(self.query_stems):
            raise ValueError("support and query stems must be disjoint")

    @property
    def evidence_count(self) -> int:
        return len(self.languages)

    def __len__(self) -> int:
        return self.evidence_count

    def evidence(self, index: int, *, wrong: bool = False) -> str:
        rule = self.rules[index].corrupted() if wrong else self.rules[index]
        examples = "".join(f"{stem} -> {rule.apply(stem)}\n" for stem in self.support_stems[index])
        return f"Lexicon\nLanguage: {self.languages[index]}\nExamples:\n{examples}"

    def question(self, index: int) -> str:
        return f"Lexicon\nLanguage: {self.languages[index]}\nInput: {self.query_stems[index]}\nOutput:"

    def answer(self, index: int) -> str:
        return " " + self.rules[index].apply(self.query_stems[index])

    def corrupted_evidence(self, index: int) -> str:
        return self.evidence(index, wrong=True)

    def corrupted_answer(self, index: int) -> str:
        return " " + self.rules[index].corrupted().apply(self.query_stems[index])


def affix_family_pools(seed: int) -> tuple[list[str], list[str]]:
    families = list(AFFIX_FAMILIES)
    random.Random(seed + 47017).shuffle(families)
    return families[:2], families[2:]


def affix_stem_pools(seed: int) -> tuple[list[str], list[str]]:
    # Independent vocabulary/shuffle from the entity labels. Split before draws.
    syllables = ("bem", "dak", "fip", "gom", "het", "jup", "lom", "nus", "rep", "wug")
    stems = [a + b + c for a in syllables for b in syllables for c in syllables]
    random.Random(seed + 57037).shuffle(stems)
    return stems[:750], stems[750:]


def sample_affix_episode(rng: random.Random, names: Sequence[str], episode_id: str,
                         *, families: Sequence[str], stems: Sequence[str],
                         rules: int = 2, support_examples: int = 3) -> AffixEpisode:
    if not families or any(family not in AFFIX_FAMILIES for family in families):
        raise ValueError("a nonempty pool of known semantic rule families is required")
    if rules < 1 or support_examples < 1:
        raise ValueError("rules and support_examples must be positive")
    if len(set(names)) != len(names) or rules > len(names):
        raise ValueError("rules must fit a pool of unique lexicon names")
    if len(set(stems)) != len(stems) or rules * (support_examples + 1) > len(stems):
        raise ValueError("stem pool cannot provide disjoint support and query stems")
    languages = tuple(rng.sample(names, rules))
    selected_stems = rng.sample(stems, rules * (support_examples + 1))
    chosen_rules = []
    support, queries = [], []
    for index in range(rules):
        family = rng.choice(families)
        prefix = rng.choice(PREFIXES) if family in ("prefix", "prefix_suffix") else ""
        suffix = rng.choice(SUFFIXES) if family in ("suffix", "prefix_suffix") else ""
        chosen_rules.append(AffixRule(family, prefix, suffix))
        offset = index * (support_examples + 1)
        support.append(tuple(selected_stems[offset:offset + support_examples]))
        queries.append(selected_stems[offset + support_examples])
    return AffixEpisode(episode_id, languages, tuple(chosen_rules), tuple(support), tuple(queries))

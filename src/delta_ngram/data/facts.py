"""Fact worlds; the two-fact path exactly preserves the archived pilot RNG."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence


COLORS = ("red", "blue", "green", "yellow", "purple", "orange", "black", "white")


@dataclass(frozen=True)
class Episode:
    # Keep exactly these three fields: legacy manifest hashes use asdict(Episode).
    episode_id: str
    names: tuple[str, ...]
    colors: tuple[str, ...]

    def evidence_color(self, index: int, *, wrong: bool = False) -> str:
        color = self.colors[index]
        if wrong:
            color = COLORS[(COLORS.index(color) + 3) % len(COLORS)]
        return color

    def evidence(self, index: int, *, wrong: bool = False) -> str:
        color = self.evidence_color(index, wrong=wrong)
        return f"Profile\nName: {self.names[index]}\nFavorite color: {color}\n"

    def question(self, index: int) -> str:
        return f"Profile\nName: {self.names[index]}\nFavorite color:"

    def answer(self, index: int) -> str:
        return " " + self.colors[index]

    @property
    def evidence_count(self) -> int:
        return len(self.names)

    def __len__(self) -> int:
        return self.evidence_count

    def corrupted_evidence(self, index: int) -> str:
        return self.evidence(index, wrong=True)

    def corrupted_answer(self, index: int) -> str:
        return " " + self.evidence_color(index, wrong=True)


FactEpisode = Episode


def entity_pools(seed: int) -> tuple[list[str], list[str]]:
    syllables = ("ba", "be", "bi", "bo", "da", "de", "di", "do", "ka", "ke",
                 "ki", "ko", "la", "le", "li", "lo", "ma", "me", "mi", "mo")
    names = [a.capitalize() + b + c for a in syllables for b in syllables for c in syllables]
    random.Random(seed + 17011).shuffle(names)
    train, validation = names[:512], names[512:768]
    assert not set(train) & set(validation)
    return train, validation


def sample_episode(rng: random.Random, names: list[str], episode_id: str) -> Episode:
    return Episode(episode_id, tuple(rng.sample(names, 2)), tuple(rng.sample(COLORS, 2)))


def sample_fact_episode(rng: random.Random, names: Sequence[str], episode_id: str,
                        *, facts: int = 2) -> Episode:
    """Scale the number of writes; retain pilot sampling exactly when facts=2.

    Longer worlds use independent color draws, allowing repeated colors. Entity
    names remain unique within each world and are sampled without replacement.
    """
    if not 1 <= facts <= len(names) or len(set(names)) != len(names):
        raise ValueError("facts must be positive and fit a pool of unique names")
    if facts == 2:
        return sample_episode(rng, list(names), episode_id)
    chosen_names = tuple(rng.sample(names, facts))
    return Episode(episode_id, chosen_names, tuple(rng.choice(COLORS) for _ in range(facts)))

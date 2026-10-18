"""Explicit fact versions for E1 acquisition, retention, and revision probes.

Event indices are zero based. Stages count completed writes, so an event at
index 2 supersedes a previous version at stage 3. Gold answers remain attached
to their event; ``active_query_indices`` excludes obsolete versions from probes.
Only evidence text crosses the writer boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from .facts import COLORS


FRUITS = ("apple", "pear", "mango", "peach", "plum", "grape", "lemon", "melon")
ATTRIBUTE_VALUES = {"favorite_color": COLORS, "favorite_fruit": FRUITS}
ATTRIBUTE_LABELS = {"favorite_color": "Favorite color", "favorite_fruit": "Favorite fruit"}
E1_SCENARIOS = ("independent", "revision")


@dataclass(frozen=True)
class FactEvent:
    key: tuple[str, str]
    value: str
    kind: str

    def __post_init__(self) -> None:
        if (not isinstance(self.key, tuple) or len(self.key) != 2
                or any(not isinstance(part, str) or not part for part in self.key)):
            raise ValueError("fact event key must be a nonempty (entity, attribute) tuple")
        if self.key[1] not in ATTRIBUTE_VALUES:
            raise ValueError("fact event has an unknown attribute")
        if self.value not in ATTRIBUTE_VALUES[self.key[1]]:
            raise ValueError("fact event value is outside its attribute vocabulary")
        if self.kind not in ("acquisition", "revision"):
            raise ValueError("fact event kind must be acquisition or revision")


@dataclass(frozen=True)
class E1Episode:
    episode_id: str
    events: tuple[FactEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("E1 episode_id must be a nonempty string")
        if (not isinstance(self.events, tuple) or not self.events
                or any(not isinstance(event, FactEvent) for event in self.events)):
            raise ValueError("E1 episodes need a nonempty tuple of FactEvent values")
        current: dict[tuple[str, str], str] = {}
        for event in self.events:
            if event.kind == "acquisition" and event.key in current:
                raise ValueError("acquisition must introduce a new fact key")
            if event.kind == "revision":
                if event.key not in current:
                    raise ValueError("revision requires a previously acquired fact key")
                if event.value == current[event.key]:
                    raise ValueError("revision must change the current fact value")
            current[event.key] = event.value

    def _event(self, index: int) -> FactEvent:
        if type(index) is not int or not 0 <= index < self.evidence_count:
            raise ValueError("event index must be within the episode")
        return self.events[index]

    @property
    def evidence_count(self) -> int:
        return len(self.events)

    def __len__(self) -> int:
        return self.evidence_count

    def event_kind(self, index: int) -> str:
        return self._event(index).kind

    def fact_key(self, index: int) -> tuple[str, str]:
        return self._event(index).key

    def _value(self, index: int, *, wrong: bool = False) -> str:
        event = self._event(index)
        if not wrong:
            return event.value
        values = ATTRIBUTE_VALUES[event.key[1]]
        # A fixed bijection preserves both changed-value revisions and key identity.
        return values[(values.index(event.value) + 3) % len(values)]

    def evidence(self, index: int, *, wrong: bool = False) -> str:
        event = self._event(index)
        name, attribute = event.key
        header = "Update" if event.kind == "revision" else "Profile"
        return f"{header}\nName: {name}\n{ATTRIBUTE_LABELS[attribute]}: {self._value(index, wrong=wrong)}\n"

    def question(self, index: int, style: str = "canonical") -> str:
        name, attribute = self._event(index).key
        label = ATTRIBUTE_LABELS[attribute]
        if style == "canonical":
            return f"Profile\nName: {name}\n{label}:"
        if style == "named_question":
            return f"What is {name}'s {label.lower()}? Answer with one word:"
        raise ValueError(f"unknown E1 question style: {style}")

    def answer(self, index: int) -> str:
        return " " + self._value(index)

    def corrupted_evidence(self, index: int) -> str:
        return self.evidence(index, wrong=True)

    def corrupted_answer(self, index: int) -> str:
        return " " + self._value(index, wrong=True)

    def active_query_indices(self, stage: int) -> tuple[int, ...]:
        if type(stage) is not int or not 0 <= stage <= self.evidence_count:
            raise ValueError("stage must be between zero and the episode's write count")
        latest = {event.key: index for index, event in enumerate(self.events[:stage])}
        return tuple(sorted(latest.values()))

    def superseded_at(self, index: int) -> int | None:
        key = self.fact_key(index)
        for later in range(index + 1, self.evidence_count):
            if self.events[later].key == key:
                return later + 1
        return None

    def previous_version(self, index: int) -> int | None:
        key = self.fact_key(index)
        for earlier in range(index - 1, -1, -1):
            if self.events[earlier].key == key:
                return earlier
        return None


def validate_e1_scenario(episode: E1Episode, scenario: str) -> None:
    """Enforce the frozen E1 generator layout in addition to event legality."""
    if scenario not in E1_SCENARIOS:
        raise ValueError(f"unknown E1 scenario: {scenario}")
    events = episode.events
    if scenario == "independent":
        if (any(event.kind != "acquisition" or event.key[1] != "favorite_color" for event in events)
                or len({event.key[0] for event in events}) != len(events)):
            raise ValueError("independent E1 scenario requires distinct entity color acquisitions")
        return
    if len(events) < 3:
        raise ValueError("revision E1 scenario requires at least three writes")
    name = events[0].key[0]
    if (events[0].key != (name, "favorite_color") or events[0].kind != "acquisition"
            or events[1].key != (name, "favorite_fruit") or events[1].kind != "acquisition"
            or events[2].key != (name, "favorite_color") or events[2].kind != "revision"):
        raise ValueError("revision E1 scenario must begin with A.color, A.fruit, then revised A.color")
    later = events[3:]
    later_names = [event.key[0] for event in later]
    if (any(event.kind != "acquisition" or event.key[1] != "favorite_color" for event in later)
            or name in later_names or len(set(later_names)) != len(later_names)):
        raise ValueError("revision E1 later writes must acquire colors for distinct new entities")


def sample_e1_episode(rng: random.Random, names: Sequence[str], episode_id: str,
                      *, writes: int = 4, scenario: str = "independent") -> E1Episode:
    """Sample a whole E1 world without mixing training and validation entities."""
    if scenario not in E1_SCENARIOS:
        raise ValueError(f"unknown E1 scenario: {scenario}")
    minimum = 3 if scenario == "revision" else 1
    if type(writes) is not int or writes < minimum:
        raise ValueError(f"E1 {scenario} scenario needs at least {minimum} writes")
    name_count = writes - 2 if scenario == "revision" else writes
    if name_count > len(names) or len(set(names)) != len(names):
        raise ValueError("E1 writes must fit a pool of unique names")
    chosen_names = tuple(rng.sample(names, name_count))
    if scenario == "independent":
        events = tuple(FactEvent((name, "favorite_color"), rng.choice(COLORS), "acquisition")
                       for name in chosen_names)
    else:
        name = chosen_names[0]
        old_color = rng.choice(COLORS)
        fruit = rng.choice(FRUITS)
        new_color = rng.choice(tuple(color for color in COLORS if color != old_color))
        events = (FactEvent((name, "favorite_color"), old_color, "acquisition"),
                  FactEvent((name, "favorite_fruit"), fruit, "acquisition"),
                  FactEvent((name, "favorite_color"), new_color, "revision"),
                  *(FactEvent((later, "favorite_color"), rng.choice(COLORS), "acquisition")
                    for later in chosen_names[1:]))
    return E1Episode(episode_id, events)

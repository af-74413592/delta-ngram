"""Typed evidence/query boundaries shared by generated task families.

Episode metadata and gold answers are audit/outer-objective data. A writer must
receive only ``episode.evidence(index)`` (or a WriteEvent's evidence field).
Stages count completed writes and are one based; evidence/query indices are zero
based. Every stage queries its new item first, then all still-active older items.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable

from .e1 import E1Episode
from .facts import Episode


@runtime_checkable
class EpisodeProtocol(Protocol):
    episode_id: str

    @property
    def evidence_count(self) -> int: ...

    def evidence(self, index: int, *, wrong: bool = False) -> str: ...

    def question(self, index: int) -> str: ...

    def answer(self, index: int) -> str: ...

    def corrupted_evidence(self, index: int) -> str: ...

    def corrupted_answer(self, index: int) -> str: ...


@dataclass(frozen=True)
class Query:
    index: int
    question: str
    answer: str
    ability: str


@dataclass(frozen=True)
class WriteEvent:
    index: int
    evidence: str


def episode_count(episode: EpisodeProtocol) -> int:
    return episode.evidence_count


def question_for_style(episode: EpisodeProtocol, index: int, style: str = "canonical") -> str:
    """Render a probe without changing evidence or drawing any random values."""
    if style == "canonical":
        return episode.question(index)
    if isinstance(episode, E1Episode):
        return episode.question(index, style=style)
    if isinstance(episode, Episode) and style == "named_question":
        return f"What is {episode.names[index]}'s favorite color? Answer with one word:"
    raise ValueError(f"unsupported question style {style!r} for {type(episode).__name__}")


def episode_queries(episode: EpisodeProtocol, stage: int) -> tuple[Query, ...]:
    """Return the new write's probe, then retention of still-active versions."""
    if isinstance(episode, E1Episode):
        active = episode.active_query_indices(stage)
        if stage == 0:
            return ()
        indices = (stage - 1, *(index for index in active if index != stage - 1))
        return tuple(Query(index, episode.question(index), episode.answer(index),
                           episode.event_kind(index) if index == stage - 1 else "retention")
                     for index in indices)
    if not 0 <= stage <= episode_count(episode):
        raise ValueError("stage must be between zero and the episode's write count")
    if stage == 0:
        return ()
    indices = (stage - 1, *range(stage - 1))
    return tuple(Query(index, episode.question(index), episode.answer(index),
                       "acquisition" if index == stage - 1 else "retention")
                 for index in indices)


def episode_events(episode: EpisodeProtocol) -> Iterator[WriteEvent | Query]:
    """Yield evidence writes interleaved with their outer-objective queries."""
    for index in range(episode_count(episode)):
        yield WriteEvent(index, episode.evidence(index))
        yield from episode_queries(episode, index + 1)

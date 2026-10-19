"""Offline document tokenization, complete trajectories, and resumable sampling.

Windows do not overlap within one document/length bucket. Different length
buckets deliberately reuse document spans; document IDs and offsets expose this.
Sampling repeats shuffled windows after each cell is exhausted. Neither document
loading nor sampling downloads data, decodes tokens, or truncates a trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


SPLITS = ("train", "validation")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _tokens(values: Any) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("token_ids must be a nonempty list or tuple")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("token_ids must contain nonnegative integers")
    return tuple(values)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RawDocument:
    id: str
    group_id: str
    source: str
    split: str
    token_ids: tuple[int, ...]
    tokenizer_identity: str | None = None
    text_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    token_sha256: str = field(init=False)

    def __post_init__(self):
        for name in ("id", "group_id", "source"):
            _text(getattr(self, name), name)
        if self.split not in SPLITS:
            raise ValueError("split must be train or validation")
        object.__setattr__(self, "token_ids", _tokens(self.token_ids))
        object.__setattr__(self, "token_sha256", _digest(self.token_ids))
        if self.tokenizer_identity is not None:
            _text(self.tokenizer_identity, "tokenizer_identity")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")


def _validate_documents(documents: Sequence[RawDocument]) -> None:
    ids, groups = set(), {}
    identities = set()
    for document in documents:
        if document.id in ids:
            raise ValueError(f"duplicate document id: {document.id}")
        ids.add(document.id)
        previous = groups.setdefault(document.group_id, document.split)
        if previous != document.split:
            raise ValueError(f"group_id crosses splits: {document.group_id}")
        if document.tokenizer_identity is not None:
            identities.add(document.tokenizer_identity)
    if len(identities) > 1:
        raise ValueError("documents use different tokenizer identities")


def load_documents(path: str | Path, tokenizer=None, *,
                   tokenizer_identity: str | None = None,
                   require_record_identity: bool = False) -> list[RawDocument]:
    """Read JSONL and encode each text document exactly once, without special IDs.

    Every record requires id/group_id/source/split and exactly one of text or
    token_ids. Pre-tokenized records require an explicit tokenizer_identity in
    the record or this function's keyword argument. With require_record_identity,
    every token_ids record must carry its own identity for comparison. Metadata lives in the
    optional metadata (or meta) object. Text is hashed before tokenization.
    """
    if tokenizer_identity is not None:
        _text(tokenizer_identity, "tokenizer_identity")
    inferred = getattr(tokenizer, "name_or_path", None)
    if not isinstance(inferred, str) or not inferred.strip():
        inferred = None
    expected_identity = tokenizer_identity or inferred
    documents = []
    required = {"id", "group_id", "source", "split"}
    allowed = required | {"text", "token_ids", "tokenizer_identity", "metadata", "meta"}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
                missing, unknown = required - record.keys(), record.keys() - allowed
                if missing or unknown:
                    raise ValueError(f"invalid fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
                for name in ("id", "group_id", "source"):
                    _text(record[name], name)
                if record["split"] not in SPLITS:
                    raise ValueError("split must be train or validation")
                if ("text" in record) == ("token_ids" in record):
                    raise ValueError("record requires exactly one of text or token_ids")
                if "meta" in record and "metadata" in record:
                    raise ValueError("use only one of meta or metadata")
                metadata = record.get("metadata", record.get("meta", {}))
                if not isinstance(metadata, dict):
                    raise ValueError("metadata must be an object")
                identity = record.get("tokenizer_identity", expected_identity)
                if identity is not None:
                    _text(identity, "tokenizer_identity")
                if expected_identity is not None and identity != expected_identity:
                    raise ValueError("tokenizer_identity differs from configured tokenizer")
                text_sha256 = None
                if "text" in record:
                    content = _text(record["text"], "text")
                    if tokenizer is None:
                        raise ValueError("text records require a tokenizer")
                    token_ids = _tokens(tokenizer.encode(content, add_special_tokens=False))
                    text_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
                else:
                    if require_record_identity and record.get("tokenizer_identity") is None:
                        raise ValueError("pre-tokenized records must carry tokenizer_identity")
                    if record.get("tokenizer_identity", tokenizer_identity) is None:
                        raise ValueError("pre-tokenized records require explicit tokenizer_identity")
                    token_ids = _tokens(record["token_ids"])
                documents.append(RawDocument(
                    record["id"], record["group_id"], record["source"], record["split"],
                    token_ids, identity, text_sha256, dict(metadata)))
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not documents:
        raise ValueError("document file is empty")
    _validate_documents(documents)
    return documents


@dataclass(frozen=True)
class TrajectoryWindow:
    document_id: str
    group_id: str
    source: str
    start: int
    token_ids: tuple[int, ...]
    blocks: int

    def __post_init__(self):
        for name in ("document_id", "group_id", "source"):
            _text(getattr(self, name), name)
        if type(self.start) is not int or self.start < 0:
            raise ValueError("start must be a nonnegative integer")
        if type(self.blocks) is not int or self.blocks < 2:
            raise ValueError("blocks must be an integer >= 2")
        object.__setattr__(self, "token_ids", _tokens(self.token_ids))
        if len(self.token_ids) % self.blocks:
            raise ValueError("window must contain complete equal-sized blocks")

    @property
    def block_size(self) -> int:
        return len(self.token_ids) // self.blocks

    @property
    def loss_tokens(self) -> int:
        """Targets after the first, write-only warm-up block."""
        return (self.blocks - 1) * self.block_size

    @property
    def end(self) -> int:
        return self.start + len(self.token_ids)


def build_windows(documents: Sequence[RawDocument], block_size: int,
                  block_counts=(8, 16), split: str = "train") -> dict[int, list[TrajectoryWindow]]:
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    counts = tuple(block_counts)
    if not counts or any(type(count) is not int or count < 2 for count in counts):
        raise ValueError("block_counts must contain integers >= 2")
    if len(set(counts)) != len(counts):
        raise ValueError("block_counts must be unique")
    if split not in SPLITS:
        raise ValueError("split must be train or validation")
    documents = list(documents)
    _validate_documents(documents)
    windows = {count: [] for count in counts}
    for document in documents:
        if document.split != split:
            continue
        for count in counts:
            size = count * block_size
            for start in range(0, len(document.token_ids) - size + 1, size):
                windows[count].append(TrajectoryWindow(
                    document.id, document.group_id, document.source, start,
                    document.token_ids[start:start + size], count))
    return windows


def _weights(supplied: Mapping | None, available: Sequence, name: str) -> dict:
    values = dict.fromkeys(available, 1.0) if supplied is None else dict(supplied)
    if not values:
        raise ValueError(f"{name} must have positive total weight")
    result = {}
    for key, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} values must be finite and nonnegative")
        if value > 0:
            if key not in available:
                raise ValueError(f"{name} requests unavailable quota: {key}")
            result[key] = float(value)
    total = sum(result.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"{name} must have positive finite total weight")
    return {key: result[key] / total for key in sorted(result)}


def _nonnegative_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class RawWindowSampler:
    """Token-fair length/source scheduling with shuffled per-cell epochs.

    The next length minimizes (served_tokens + trajectory_cost) / target_share.
    This weighted-fair schedule approaches the same token mixture as drawing
    lengths with probability proportional to share / trajectory_cost, without
    random quota swings in short runs. Sources use the same rule. Each active
    length/source pair must have windows; absent quotas never silently fall back.
    """
    def __init__(self, windows: Mapping[int, Sequence[TrajectoryWindow]], seed: int,
                 length_weights: Mapping[int, float] | None = None,
                 source_weights: Mapping[str, float] | None = None):
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self._windows = {count: tuple(items) for count, items in windows.items()}
        for count, items in self._windows.items():
            if type(count) is not int or count < 2 or any(item.blocks != count for item in items):
                raise ValueError("window bucket keys must match blocks >= 2")
        self.length_weights = _weights(length_weights, tuple(self._windows), "length_weights")
        active_windows = [item for count in self.length_weights for item in self._windows[count]]
        sizes = {item.block_size for item in active_windows}
        if len(sizes) != 1:
            raise ValueError("active windows must have one common block_size and cannot all be empty")
        self.block_size = sizes.pop()
        sources = sorted({item.source for item in active_windows})
        self.source_weights = _weights(source_weights, sources, "source_weights")
        self._cells = {}
        for count in self.length_weights:
            for source in self.source_weights:
                items = tuple(item for item in self._windows[count] if item.source == source)
                if not items:
                    raise ValueError(f"nonzero quota has no windows: blocks={count}, source={source}")
                self._cells[count, source] = items
        self._rng = random.Random(seed)
        self._orders, self._cursors, self._epochs = {}, {}, {}
        for key, items in self._cells.items():
            order = list(range(len(items)))
            self._rng.shuffle(order)
            self._orders[key], self._cursors[key], self._epochs[key] = order, 0, 0
        self.length_loss_tokens = dict.fromkeys(self.length_weights, 0)
        self.source_loss_tokens = dict.fromkeys(self.source_weights, 0)
        self.total_loss_tokens = 0
        self.last_status = {"reason": "ready"}
        self.fingerprint = _digest({
            "length_weights": self.length_weights, "source_weights": self.source_weights,
            "cells": [[count, source, [[item.document_id, item.group_id, item.start,
                        item.blocks, _digest(item.token_ids)] for item in items]]
                      for (count, source), items in self._cells.items()]})

    def next_window(self, remaining_loss_tokens: int | None = None) -> TrajectoryWindow | None:
        if remaining_loss_tokens is not None:
            _nonnegative_integer(remaining_loss_tokens, "remaining_loss_tokens")
        costs = {count: (count - 1) * self.block_size for count in self.length_weights}
        eligible = [count for count, cost in costs.items()
                    if remaining_loss_tokens is None or cost <= remaining_loss_tokens]
        if not eligible:
            self.last_status = {"reason": "insufficient_budget", "remaining_loss_tokens": remaining_loss_tokens,
                                "minimum_loss_tokens": min(costs.values())}
            return None
        count = min(eligible, key=lambda value: (
            (self.length_loss_tokens[value] + costs[value]) / self.length_weights[value], value))
        cost = costs[count]
        source = min(self.source_weights, key=lambda value: (
            (self.source_loss_tokens[value] + cost) / self.source_weights[value], value))
        key = count, source
        if self._cursors[key] == len(self._orders[key]):
            self._rng.shuffle(self._orders[key])
            self._cursors[key] = 0
            self._epochs[key] += 1
        window = self._cells[key][self._orders[key][self._cursors[key]]]
        self._cursors[key] += 1
        self.length_loss_tokens[count] += cost
        self.source_loss_tokens[source] += cost
        self.total_loss_tokens += cost
        self.last_status = {"reason": "sampled", "loss_tokens": cost}
        return window

    def next_batch(self, count: int, remaining_loss_tokens: int | None = None) -> list[TrajectoryWindow]:
        _nonnegative_integer(count, "count")
        if remaining_loss_tokens is not None:
            _nonnegative_integer(remaining_loss_tokens, "remaining_loss_tokens")
        batch = []
        for _ in range(count):
            window = self.next_window(remaining_loss_tokens)
            if window is None:
                break
            batch.append(window)
            if remaining_loss_tokens is not None:
                remaining_loss_tokens -= window.loss_tokens
        return batch

    def state_dict(self) -> dict:
        """Return a JSON-serializable snapshot, including shuffle RNG and cursors."""
        return json.loads(json.dumps({
            "version": 1, "fingerprint": self.fingerprint, "rng_state": self._rng.getstate(),
            "total_loss_tokens": self.total_loss_tokens,
            "length_loss_tokens": [[key, value] for key, value in self.length_loss_tokens.items()],
            "source_loss_tokens": [[key, value] for key, value in self.source_loss_tokens.items()],
            "cells": [{"blocks": key[0], "source": key[1], "order": self._orders[key],
                       "cursor": self._cursors[key], "epoch": self._epochs[key]} for key in self._cells],
            "last_status": self.last_status}))

    def load_state_dict(self, state: Mapping) -> None:
        if state.get("version") != 1 or state.get("fingerprint") != self.fingerprint:
            raise ValueError("sampler state version or corpus/config fingerprint differs")
        try:
            lengths, sources = dict(state["length_loss_tokens"]), dict(state["source_loss_tokens"])
            if set(lengths) != set(self.length_weights) or set(sources) != set(self.source_weights):
                raise ValueError("sampler quota keys differ")
            total = _nonnegative_integer(state["total_loss_tokens"], "total_loss_tokens")
            for key, value in lengths.items():
                _nonnegative_integer(value, "length_loss_tokens")
                if value % ((key - 1) * self.block_size):
                    raise ValueError("length token count is not a complete trajectory multiple")
            for value in sources.values():
                _nonnegative_integer(value, "source_loss_tokens")
            if sum(lengths.values()) != total or sum(sources.values()) != total:
                raise ValueError("sampler token counters disagree")
            orders, cursors, epochs = {}, {}, {}
            for cell in state["cells"]:
                key = cell["blocks"], cell["source"]
                if key not in self._cells or key in orders:
                    raise ValueError("invalid or duplicate sampler cell")
                order = list(cell["order"])
                if any(type(value) is not int for value in order) or sorted(order) != list(range(len(self._cells[key]))):
                    raise ValueError("sampler order must be a permutation")
                cursor = _nonnegative_integer(cell["cursor"], "cursor")
                if cursor > len(order):
                    raise ValueError("sampler cursor is out of range")
                orders[key], cursors[key] = order, cursor
                epochs[key] = _nonnegative_integer(cell["epoch"], "epoch")
            if orders.keys() != self._cells.keys():
                raise ValueError("sampler state is missing cells")
            rng_state = state["rng_state"]
            restored_rng = random.Random()
            restored_rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
            status = dict(state["last_status"])
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("malformed sampler state") from error
        self._rng = restored_rng
        self._orders, self._cursors, self._epochs = orders, cursors, epochs
        self.length_loss_tokens, self.source_loss_tokens = lengths, sources
        self.total_loss_tokens, self.last_status = total, status

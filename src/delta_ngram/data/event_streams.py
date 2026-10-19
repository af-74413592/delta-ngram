"""Complete, dated news articles and variable-cost, exactly resumable sampling."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import json
import random
import re

from .raw_text import _digest, _text, _tokens, _weights, _nonnegative_integer

NEWS_REVISION = "81eb2ce0d2a9dad6ad16b68ef750ec290880fa36"
NEWS_SHARDS = {
    f"plain_text/train-{index:05d}-of-00005.parquet": checksum for index, checksum in enumerate((
        "5cda9a2b0b40a9c9438ae02168b16beb61e52449a895869515501962c5aeb977",
        "1d33583fdd82fa28a7aa961a633aeb63be90fdac8f47c1893daf630c9a5f074a",
        "78d59696b61fa9789879836a303c7a04beb52107c6678a444b98ce4e8b97b1ae",
        "53b35fa4b5d82a0308ddaba3d55033c49a1f1d4cca45d88d2633495cab4767db",
        "e0869a42852439aa408030de51464d0a8cbe5608ff1a7ba9d165e10964c6fdef"))}


def publication_time(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?", value):
        raise ValueError("explicit publication date required")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return stamp.date(), stamp.astimezone(timezone.utc) if stamp.tzinfo is not None else None


def date_split(day):
    for split, begin, end in (("train", date(2017, 1, 1), date(2017, 12, 31)),
                              ("validation", date(2018, 1, 1), date(2018, 4, 30)),
                              ("holdout", date(2018, 5, 1), date(2018, 7, 31))):
        if begin <= day <= end:
            return split
    raise ValueError("publication date outside old-news partitions")


def validate_chronology(articles, split):
    times = [publication_time(article.metadata["publication_date_original"]) for article in articles]
    if any(date_split(day) != split for day, _ in times):
        raise ValueError("article publication date crosses stream split")
    for index, (day, utc) in enumerate(times):
        for later_day, later_utc in times[index + 1:]:
            if abs((later_day - day).days) > 30:
                raise ValueError("event span exceeds 30 days")
            if utc is not None and later_utc is not None:
                seconds = (later_utc - utc).total_seconds()
                if seconds <= 0:
                    raise ValueError("UTC publication chronology must be strictly increasing")
                if seconds > 30 * 86400:
                    raise ValueError("UTC event span exceeds 30 days")
            elif later_day <= day:
                raise ValueError("ambiguous or reversed publication chronology")


@dataclass(frozen=True)
class EventArticle:
    id: str
    token_ids: tuple[int, ...]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        _text(self.id, "article id")
        object.__setattr__(self, "token_ids", _tokens(self.token_ids))
        if not 128 <= len(self.token_ids) <= 16384:
            raise ValueError("complete event article must have 128..16384 tokens")
        if not isinstance(self.metadata, dict):
            raise ValueError("article metadata must be an object")


@dataclass(frozen=True)
class EventDocument:
    id: str
    group_id: str
    source: str
    split: str
    articles: tuple[EventArticle, ...]
    tokenizer_identity: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for key in ("id", "group_id", "tokenizer_identity"):
            _text(getattr(self, key), key)
        if self.source != "news_event" or self.split not in ("train", "validation", "holdout"):
            raise ValueError("event source/split is invalid")
        object.__setattr__(self, "articles", tuple(self.articles))
        if not 2 <= len(self.articles) <= 4 or not all(isinstance(a, EventArticle) for a in self.articles):
            raise ValueError("event stream requires 2..4 complete articles")
        if len({a.id for a in self.articles}) != len(self.articles):
            raise ValueError("duplicate article in event stream")
        if sum(len(a.token_ids) for a in self.articles) > 16384 or sum((len(a.token_ids) + 511) // 512 for a in self.articles) > 32:
            raise ValueError("complete event exceeds 16384 real tokens or 32 article chunks")
        validate_chronology(self.articles, self.split)


def iter_event_documents(path, *, tokenizer_identity, require_record_identity=True, allow_holdout=False):
    """Validate one stream at a time; holdout opt-in is for corpus auditing only."""
    from pathlib import Path
    count, ids, groups, owners, member_owners = 0, set(), {}, {}, {}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                required = {"id", "group_id", "source", "split", "articles", "metadata", "tokenizer_identity"}
                if not isinstance(row, dict) or set(row) != required:
                    raise ValueError("invalid event record fields")
                if row["split"] == "holdout" and not allow_holdout:
                    raise ValueError("holdout is excluded from runtime event loading")
                if row["tokenizer_identity"] != tokenizer_identity:
                    raise ValueError("tokenizer_identity differs from configured tokenizer")
                meta = row["metadata"]
                if not isinstance(meta, dict):
                    raise ValueError("event metadata must be an object")
                if meta.get("dataset_revision", NEWS_REVISION) != NEWS_REVISION:
                    raise ValueError("event dataset revision mismatch")
                members = meta["family_members"]
                if (not isinstance(members, list) or not members or any(not isinstance(x, str) for x in members)
                        or len(set(members)) != len(members)):
                    raise ValueError("invalid family_members")
                if row["group_id"] != "news-family:" + _digest(sorted(members)):
                    raise ValueError("family membership does not match group_id")
                for member in members:
                    if member_owners.setdefault(member, row["group_id"]) != row["group_id"]:
                        raise ValueError("recovered family membership overlaps another family")
                articles = []
                for supplied in row["articles"]:
                    if not isinstance(supplied, dict) or set(supplied) != {"id", "token_ids", "metadata"}:
                        raise ValueError("invalid article fields")
                    article = EventArticle(**supplied)
                    am = article.metadata
                    if am.get("truncated", False) is not False or am.get("complete_extracted_article", True) is not True:
                        raise ValueError("event articles must be complete and untruncated")
                    if article.id not in members or am["dataset_revision"] != NEWS_REVISION:
                        raise ValueError("article provenance/family mismatch")
                    origin = {name: am[name] for name in ("source_shard", "source_row", "source_shard_sha256")}
                    if type(origin["source_row"]) is not int or origin["source_row"] < 1:
                        raise ValueError("invalid source row")
                    if NEWS_SHARDS.get(origin["source_shard"]) != origin["source_shard_sha256"]:
                        raise ValueError("source shard differs from pinned news corpus")
                    if article.id != "cc-news:" + _digest([NEWS_REVISION, origin]):
                        raise ValueError("article ID does not match source provenance")
                    for key in ("text_sha256", "normalized_text_sha256", "token_sha256", "source_shard_sha256"):
                        if not isinstance(am[key], str) or not re.fullmatch("[0-9a-f]{64}", am[key]):
                            raise ValueError("invalid article hash")
                    if am["token_sha256"] != _digest(article.token_ids):
                        raise ValueError("article token hash mismatch")
                    day, _ = publication_time(am["publication_date_original"])
                    if am["publication_date"] != day.isoformat():
                        raise ValueError("article publication date mismatch")
                    for key in (article.id, am["normalized_text_sha256"], am["canonical_url"]):
                        old = owners.setdefault(key, (row["split"], row["group_id"]))
                        if old != (row["split"], row["group_id"]):
                            raise ValueError("article identity crosses event families or splits")
                    articles.append(article)
                if len({a.metadata["normalized_text_sha256"] for a in articles}) != len(articles):
                    raise ValueError("exact duplicate bodies in event stream")
                document = EventDocument(row["id"], row["group_id"], row["source"], row["split"],
                                         tuple(articles), row["tokenizer_identity"], meta)
                if meta.get("stream_article_ids") != [a.id for a in articles]:
                    raise ValueError("ordered article IDs do not match stream metadata")
                offset, boundaries = 0, []
                for index, article in enumerate(articles):
                    end = offset + len(article.token_ids)
                    boundaries.append({"article_id": article.id, "start": offset, "end": end,
                                       "role": "history" if index == 0 else "target"})
                    offset = end
                if "article_boundaries" in meta and meta["article_boundaries"] != boundaries:
                    raise ValueError("article boundaries do not match complete token lengths")
                costs = {"raw_tokens": offset, "target_tokens": sum(len(a.token_ids) for a in articles[1:]),
                         "article_chunk_count": sum((len(a.token_ids) + 511) // 512 for a in articles)}
                for key, expected in costs.items():
                    if key in meta and (type(meta[key]) is not int or meta[key] != expected):
                        raise ValueError("declared event token/chunk accounting mismatch")
                if "block_size" in meta and meta["block_size"] != 512:
                    raise ValueError("prepared event block size must be 512")
                if "block_bucket" in meta:
                    bucket = next(n for n in (4, 8, 16, 32) if n >= costs["article_chunk_count"])
                    if type(meta["block_bucket"]) is not int or meta["block_bucket"] != bucket:
                        raise ValueError("declared event bucket mismatch")
                if document.id in ids or document.group_id in groups:
                    raise ValueError("duplicate event ID/family")
                ids.add(document.id)
                groups[document.group_id] = document.split
                count += 1
                yield document
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
    if not count:
        raise ValueError("event document file is empty")


def load_event_documents(path, *, tokenizer_identity, require_record_identity=True, allow_holdout=False):
    """Read the separate event corpus, refusing holdout during normal runtime."""
    return list(iter_event_documents(path, tokenizer_identity=tokenizer_identity,
        require_record_identity=require_record_identity, allow_holdout=allow_holdout))


@dataclass(frozen=True)
class EventTrajectory:
    document_id: str
    group_id: str
    source: str
    articles: tuple[EventArticle, ...]
    blocks: int
    block_size: int = 512
    start: int = 0
    variable_cost: bool = field(default=True, init=False)

    def __post_init__(self):
        for key in ("document_id", "group_id"):
            _text(getattr(self, key), key)
        object.__setattr__(self, "articles", tuple(self.articles))
        if not all(isinstance(a, EventArticle) for a in self.articles):
            raise ValueError("event articles must be complete EventArticle records")
        if type(self.block_size) is not int or self.block_size < 1:
            raise ValueError("event block_size must be positive")
        if type(self.blocks) is not int or self.blocks < self.actual_blocks or self.blocks < 2:
            raise ValueError("event bucket cannot truncate article chunks")
        if not 2 <= len(self.articles) <= 4 or self.source != "news_event" or self.start != 0:
            raise ValueError("invalid event trajectory")
        if self.raw_tokens > 16384 or self.actual_blocks > 32:
            raise ValueError("complete event exceeds 16384 real tokens or 32 article chunks")

    @property
    def token_ids(self):
        return tuple(token for article in self.articles for token in article.token_ids)

    @property
    def actual_blocks(self):
        return sum((len(a.token_ids) + self.block_size - 1) // self.block_size for a in self.articles)

    @property
    def raw_tokens(self):
        return sum(len(a.token_ids) for a in self.articles)

    @property
    def loss_tokens(self):
        return sum(len(a.token_ids) for a in self.articles[1:])

    @property
    def writer_tokens(self):
        last = (len(self.articles[-1].token_ids) - 1) % self.block_size + 1
        return self.raw_tokens - last


def build_event_trajectories(documents, block_size=512, split="train", block_counts=(4, 8, 16, 32)):
    if type(block_size) is not int or block_size < 1:
        raise ValueError("event block_size must be positive")
    counts = sorted(set(block_counts))
    if not counts or any(type(n) is not int or n < 2 for n in counts):
        raise ValueError("event block counts must be integers >= 2")
    result = {count: [] for count in counts}
    for document in documents:
        if document.split != split:
            continue
        chunks = sum((len(a.token_ids) + block_size - 1) // block_size for a in document.articles)
        bucket = next((n for n in counts if n >= chunks), None)
        if bucket is None:
            raise ValueError("event stream exceeds configured buckets; complete articles cannot be truncated")
        result[bucket].append(EventTrajectory(document.id, document.group_id, document.source,
                                              document.articles, bucket, block_size))
    return result


def event_donor_compatible(target, donor):
    return (isinstance(donor, EventTrajectory) and donor.source == target.source
            and donor.blocks == target.blocks and donor.block_size == target.block_size
            and donor.group_id != target.group_id and donor.raw_tokens >= target.writer_tokens)


class VariableCostSampler:
    """V3 token-fair scheduler: queued heads cost their actual target counts.

    An exhausted cell is shuffled immediately. No head is skipped to fit the
    remaining budget, so pausing never consumes RNG or changes future order.
    """
    def __init__(self, windows, seed, length_weights=None, source_weights=None, source_length_weights=None):
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self._variable_cost = True
        self._windows = {count: tuple(items) for count, items in windows.items()}
        if any(type(count) is not int or count < 2 or any(w.blocks != count for w in items)
               for count, items in self._windows.items()):
            raise ValueError("window bucket keys must match blocks")
        if source_length_weights is not None and length_weights is not None:
            raise ValueError("source_length_weights and length_weights are mutually exclusive")
        self.length_weights = _weights(length_weights, tuple(windows), "length_weights")
        all_items = [w for items in self._windows.values() for w in items]
        sizes = {w.block_size for w in all_items}
        if len(sizes) != 1 or any(type(w.loss_tokens) is not int or w.loss_tokens < 1 for w in all_items):
            raise ValueError("variable trajectories require one block size and positive actual costs")
        self.block_size = sizes.pop()
        self.source_weights = _weights(source_weights, sorted({w.source for w in all_items}), "source_weights")
        if source_length_weights is None:
            self.source_length_weights = {source: self.length_weights for source in self.source_weights}
        else:
            if set(source_length_weights) != set(self.source_weights):
                raise ValueError("source_length_weights must cover exactly active sources")
            self.source_length_weights = {source: _weights(weights, tuple(windows), "source_length_weights")
                                          for source, weights in source_length_weights.items()}
        self.cell_weights = {(count, source): self.source_weights[source] * share
                             for source, lengths in self.source_length_weights.items() for count, share in lengths.items()}
        self.length_weights = {count: sum(share for (n, _), share in self.cell_weights.items() if n == count)
                               for count in sorted({n for n, _ in self.cell_weights})}
        self._cells = {key: tuple(w for w in self._windows[key[0]] if w.source == key[1])
                       for key in sorted(self.cell_weights)}
        if any(not items for items in self._cells.values()):
            raise ValueError("nonzero quota has no windows")
        self._rng = random.Random(seed)
        self._orders, self._cursors, self._epochs = {}, {}, {}
        for key, items in self._cells.items():
            order = list(range(len(items)))
            self._rng.shuffle(order)
            self._orders[key], self._cursors[key], self._epochs[key] = order, 0, 0
        self.cell_loss_tokens = dict.fromkeys(self._cells, 0)
        self.length_loss_tokens = dict.fromkeys(self.length_weights, 0)
        self.source_loss_tokens = dict.fromkeys(self.source_weights, 0)
        self.total_loss_tokens = 0
        self.last_status = {"reason": "ready"}
        self.fingerprint = _digest({"scheduling": "variable_target_tokens_v3", "weights": [[*k, v] for k, v in self.cell_weights.items()],
            "cells": [[*key, [[w.document_id, w.group_id, w.start, w.blocks, w.loss_tokens,
                        _digest(w.token_ids), [_digest(a.token_ids) for a in w.articles] if getattr(w, "variable_cost", False) else None]
                        for w in items]] for key, items in self._cells.items()]})

    def next_window(self, remaining_loss_tokens=None):
        if remaining_loss_tokens is not None:
            _nonnegative_integer(remaining_loss_tokens, "remaining_loss_tokens")
        heads = {key: items[self._orders[key][self._cursors[key]]] for key, items in self._cells.items()}
        eligible = [key for key, w in heads.items() if remaining_loss_tokens is None or w.loss_tokens <= remaining_loss_tokens]
        if not eligible:
            self.last_status = {"reason": "insufficient_budget", "remaining_loss_tokens": remaining_loss_tokens,
                                "minimum_loss_tokens": min(w.loss_tokens for w in heads.values())}
            return None
        key = min(eligible, key=lambda k: ((self.cell_loss_tokens[k] + heads[k].loss_tokens) / self.cell_weights[k], k))
        window, cost = heads[key], heads[key].loss_tokens
        self._cursors[key] += 1
        if self._cursors[key] == len(self._orders[key]):
            self._rng.shuffle(self._orders[key])
            self._cursors[key] = 0
            self._epochs[key] += 1
        self.cell_loss_tokens[key] += cost
        self.length_loss_tokens[key[0]] += cost
        self.source_loss_tokens[key[1]] += cost
        self.total_loss_tokens += cost
        self.last_status = {"reason": "sampled", "loss_tokens": cost}
        return window

    def next_batch(self, count, remaining_loss_tokens=None):
        _nonnegative_integer(count, "count")
        result = []
        for _ in range(count):
            item = self.next_window(remaining_loss_tokens)
            if item is None:
                break
            result.append(item)
            if remaining_loss_tokens is not None:
                remaining_loss_tokens -= item.loss_tokens
        return result

    def state_dict(self):
        return json.loads(json.dumps({"version": 3, "fingerprint": self.fingerprint,
            "rng_state": self._rng.getstate(), "total_loss_tokens": self.total_loss_tokens,
            "length_loss_tokens": list(self.length_loss_tokens.items()), "source_loss_tokens": list(self.source_loss_tokens.items()),
            "cells": [{"blocks": k[0], "source": k[1], "order": self._orders[k], "cursor": self._cursors[k],
                       "epoch": self._epochs[k], "loss_tokens": self.cell_loss_tokens[k]} for k in self._cells],
            "last_status": self.last_status}))

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise ValueError("malformed variable sampler state")
        if state.get("version") != 3 or state.get("fingerprint") != self.fingerprint:
            raise ValueError("sampler state version or corpus/config fingerprint differs")
        try:
            orders, cursors, epochs, served = {}, {}, {}, {}
            for cell in state["cells"]:
                key = cell["blocks"], cell["source"]
                if key not in self._cells or key in orders:
                    raise ValueError("invalid duplicate sampler cell")
                order = cell["order"]
                if any(type(i) is not int for i in order) or sorted(order) != list(range(len(self._cells[key]))):
                    raise ValueError("sampler order must be a permutation")
                cursor = _nonnegative_integer(cell["cursor"], "cursor")
                epoch = _nonnegative_integer(cell["epoch"], "epoch")
                if cursor >= len(order):
                    raise ValueError("sampler cursor is out of range")
                cost = epoch * sum(w.loss_tokens for w in self._cells[key]) + sum(self._cells[key][i].loss_tokens for i in order[:cursor])
                if _nonnegative_integer(cell["loss_tokens"], "cell loss tokens") != cost:
                    raise ValueError("actual variable costs disagree with cursor")
                orders[key], cursors[key], epochs[key], served[key] = list(order), cursor, epoch, cost
            if set(orders) != set(self._cells):
                raise ValueError("missing sampler cell")
            lengths = {n: sum(v for (k, _), v in served.items() if k == n) for n in self.length_weights}
            sources = {s: sum(v for (_, k), v in served.items() if k == s) for s in self.source_weights}
            for name in ("length_loss_tokens", "source_loss_tokens"):
                pairs = state[name]
                if len(dict(pairs)) != len(pairs):
                    raise ValueError("duplicate sampler quota counter")
                for _, value in pairs:
                    _nonnegative_integer(value, name)
            if (dict(state["length_loss_tokens"]) != lengths or dict(state["source_loss_tokens"]) != sources
                    or _nonnegative_integer(state["total_loss_tokens"], "total") != sum(served.values())):
                raise ValueError("sampler token counters disagree with actual costs")
            rng = random.Random()
            rs = state["rng_state"]
            rng.setstate((rs[0], tuple(rs[1]), rs[2]))
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("malformed variable sampler state") from error
        self._orders, self._cursors, self._epochs, self._rng = orders, cursors, epochs, rng
        self.cell_loss_tokens, self.length_loss_tokens, self.source_loss_tokens = served, lengths, sources
        self.total_loss_tokens, self.last_status = sum(served.values()), dict(state["last_status"])

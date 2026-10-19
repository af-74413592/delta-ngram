"""Strict news-event provenance, chronological whole articles and v3 sampling."""
import copy
from datetime import date
import json

import pytest

from delta_ngram.data.event_streams import (EventArticle, EventDocument, EventTrajectory,
    NEWS_REVISION, build_event_trajectories, event_donor_compatible, load_event_documents, date_split)
from delta_ngram.data.raw_text import RawWindowSampler, TrajectoryWindow, _digest


def article(index, length=128, day=None, token=3):
    metadata = {"publication_date_original": day or f"2017-01-{index + 1:02d}",
        "publication_date": (day or f"2017-01-{index + 1:02d}")[:10],
        "dataset_revision": NEWS_REVISION, "source_shard": "plain_text/train-00000-of-00005.parquet",
        "source_row": index + 1, "source_shard_sha256": "5cda9a2b0b40a9c9438ae02168b16beb61e52449a895869515501962c5aeb977",
        "text_sha256": _digest([index, "raw"]), "normalized_text_sha256": _digest([index, "normal"]),
        "canonical_url": f"https://example.org/{index}"}
    tokens = tuple([token] * length)
    metadata["token_sha256"] = _digest(tokens)
    origin = {key: metadata[key] for key in ("source_shard", "source_row", "source_shard_sha256")}
    return EventArticle("cc-news:" + _digest([NEWS_REVISION, origin]), tokens, metadata)


def document(lengths=(128, 129), days=None):
    articles = tuple(article(i, length, None if days is None else days[i]) for i, length in enumerate(lengths))
    members = [a.id for a in articles]
    return EventDocument("event", "news-family:" + _digest(sorted(members)), "news_event", "train", articles,
                         "sha256:test", {"family_members": members, "stream_article_ids": members})


def serialized(doc):
    return {"id": doc.id, "group_id": doc.group_id, "source": doc.source, "split": doc.split,
            "tokenizer_identity": doc.tokenizer_identity, "metadata": doc.metadata,
            "articles": [{"id": a.id, "token_ids": a.token_ids, "metadata": a.metadata} for a in doc.articles]}


def test_complete_articles_buckets_and_actual_cost():
    doc = document((129, 513, 130))
    windows = build_event_trajectories([doc], block_size=512, block_counts=(2, 4, 8))
    item = windows[4][0]
    assert item.actual_blocks == 4 and item.blocks == 4
    assert item.loss_tokens == 643 and len(item.token_ids) == 772
    assert item.writer_tokens == 642
    assert not windows[2] and not windows[8]
    with pytest.raises(ValueError, match="cannot be truncated"):
        build_event_trajectories([doc], block_counts=(2,))
    with pytest.raises(ValueError, match="128..16384"):
        article(0, 127)


@pytest.mark.parametrize("day,split", [("2017-01-01", "train"), ("2017-12-31", "train"),
    ("2018-01-01", "validation"), ("2018-04-30", "validation"),
    ("2018-05-01", "holdout"), ("2018-07-31", "holdout")])
def test_registered_news_split_boundaries(day, split):
    assert date_split(date.fromisoformat(day)) == split


@pytest.mark.parametrize("day", ["2016-12-31", "2018-08-01", "2019-01-01", "2026-10-19"])
def test_outside_registered_news_windows_rejected(day):
    with pytest.raises(ValueError, match="outside"):
        date_split(date.fromisoformat(day))


@pytest.mark.parametrize("days", [
    ["2017-01-01", "2017-01-01"],
    ["2017-01-01 01:00:00", "2017-01-01 02:00:00"],
    ["2017-01-01T01:00:00Z", "2017-01-01T02:00:00+01:00"],
    ["2017-01-01T23:00:00-12:00", "2017-01-02T00:00:00+14:00"],
    ["2017-01-01", "2017-02-01"],
    ["2017-12-31", "2018-01-01"],
])
def test_ambiguous_reversed_long_and_cross_split_dates_fail(days):
    with pytest.raises(ValueError):
        document(days=days)


def test_same_day_reliable_utc_order_and_full_provenance_load(tmp_path):
    doc = document(days=["2017-01-01T01:00:00+01:00", "2017-01-01T01:00:01Z"])
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(serialized(doc)) + "\n", encoding="utf-8")
    assert load_event_documents(path, tokenizer_identity="sha256:test") == [doc]
    with pytest.raises(ValueError, match="tokenizer_identity"):
        load_event_documents(path, tokenizer_identity="sha256:other")


def test_aware_utc_order_can_reverse_calendar_days_and_span_checks_all_pairs():
    doc = document(days=["2017-01-02T00:00:00+14:00", "2017-01-01T23:00:00-12:00"])
    assert len(doc.articles) == 2
    # First source calendar day is not necessarily the earliest source day.
    with pytest.raises(ValueError, match="30 days"):
        document((128, 128, 128), ["2017-01-02T00:00:00+14:00", "2017-01-01T23:00:00-12:00",
                                  "2017-02-01T00:00:00+14:00"])


def test_whole_stream_token_and_chunk_caps_are_independent():
    with pytest.raises(ValueError, match="16384 real tokens"):
        document((8192, 8193))
    # 16384 real tokens but separate article tails require 33 encoder chunks.
    with pytest.raises(ValueError, match="32 article chunks"):
        document((8191, 8193))
    assert document((8192, 8192))


@pytest.mark.parametrize("mutation", ["tokens", "group", "member", "date", "duplicate", "holdout", "truncated", "cost"])
def test_loader_rejects_corrupt_provenance_or_split(tmp_path, mutation):
    row = serialized(document())
    row = json.loads(json.dumps(row))
    if mutation == "tokens":
        row["articles"][1]["token_ids"][0] += 1
    elif mutation == "group":
        row["group_id"] += "x"
    elif mutation == "member":
        row["articles"][1]["id"] += "x"
    elif mutation == "date":
        row["articles"][1]["metadata"]["publication_date"] = "2017-09-01"
    elif mutation == "duplicate":
        row["articles"][1]["metadata"]["normalized_text_sha256"] = row["articles"][0]["metadata"]["normalized_text_sha256"]
    elif mutation == "truncated":
        row["articles"][1]["metadata"]["truncated"] = True
    elif mutation == "cost":
        row["metadata"]["target_tokens"] = 999
    else:
        row["split"] = "holdout"
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError):
        load_event_documents(path, tokenizer_identity="sha256:test")


def pool():
    first = EventTrajectory("a", "family-a", "news_event", (article(0), article(1, 129)), 4, 128)
    second = EventTrajectory("b", "family-b", "news_event", (article(2, 129), article(3, 259)), 8, 128)
    raw = TrajectoryWindow("raw", "raw-family", "code", 0, (4,) * 512, 4)
    return {4: [first, raw], 8: [second]}


def sampler():
    return RawWindowSampler(pool(), 17, source_weights={"code": .5, "news_event": .5},
        source_length_weights={"code": {4: 1.}, "news_event": {4: .5, 8: .5}})


def test_v3_actual_cost_fairness_and_exact_resume():
    first = sampler()
    assert first.state_dict()["version"] == 3
    sampled = first.next_batch(37)
    assert first.total_loss_tokens == sum(w.loss_tokens for w in sampled)
    for key, share in first.cell_weights.items():
        assert abs(first.cell_loss_tokens[key] - first.total_loss_tokens * share) <= 2 * max(w.loss_tokens for w in sampled)
    snapshot = first.state_dict()
    second = sampler()
    second.load_state_dict(snapshot)
    assert first.next_batch(80) == second.next_batch(80)
    assert first.state_dict() == second.state_dict()
    broken = copy.deepcopy(snapshot)
    broken["cells"][0]["loss_tokens"] += 1
    with pytest.raises(ValueError, match="actual variable costs"):
        sampler().load_state_dict(broken)


def test_no_budget_overshoot_or_partial_event_and_legacy_version():
    stream = sampler()
    result = stream.next_batch(30, 1000)
    assert sum(w.loss_tokens for w in result) <= 1000
    state = stream.state_dict()
    assert stream.next_window(1) is None
    after = stream.state_dict()
    assert after["cells"] == state["cells"] and after["rng_state"] == state["rng_state"]
    raw = pool()[4][1]
    assert RawWindowSampler({4: [raw]}, 17).state_dict()["version"] == 1
    assert RawWindowSampler({4: [raw]}, 17, source_weights={"code": 1},
        source_length_weights={"code": {4: 1}}).state_dict()["version"] == 2


def test_donor_capacity_and_independence():
    target = pool()[4][0]
    donor = EventTrajectory("donor", "another", "news_event", target.articles, 4, 128)
    assert event_donor_compatible(target, donor)
    assert not event_donor_compatible(target, target)


@pytest.mark.parametrize("mutation", ["duplicate", "counter", "order", "fingerprint", "version"])
def test_v3_rejects_corrupt_resume(mutation):
    stream = sampler()
    stream.next_batch(17)
    state = stream.state_dict()
    if mutation == "duplicate":
        state["length_loss_tokens"].append(state["length_loss_tokens"][0])
    elif mutation == "counter":
        state["total_loss_tokens"] += 1
    elif mutation == "order":
        state["cells"][0]["order"] = [999]
    elif mutation == "fingerprint":
        state["fingerprint"] = "wrong"
    else:
        state["version"] = 2
    with pytest.raises(ValueError):
        sampler().load_state_dict(state)

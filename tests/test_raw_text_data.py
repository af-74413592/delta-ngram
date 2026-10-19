"""Offline raw-text boundaries, target-token quotas, and exact resume contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delta_ngram.data.raw_text import (
    RawDocument, RawWindowSampler, TrajectoryWindow, build_windows, load_documents,
)


class CountingTokenizer:
    name_or_path = "fixture-tokenizer-v1"

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        self.calls.append(text)
        return [ord(character) for character in text]

    def decode(self, *args, **kwargs):
        raise AssertionError("the data path must not decode tokens")


def _write(tmp_path, records):
    path = tmp_path / "documents.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    return path


def _record(id="doc-a", group="group-a", split="train", **extra):
    return {"id": id, "group_id": group, "source": "books", "split": split,
            "text": "abcdefgh" * 10, **extra}


def _windows():
    documents = [RawDocument(f"{source}-{index}", f"{source}-group-{index}", source,
                             "train", tuple(range(128)))
                 for source in ("books", "papers") for index in range(3)]
    return build_windows(documents, block_size=2)


def test_tokenize_whole_documents_once_preserve_provenance_and_never_decode(tmp_path):
    tokenizer = CountingTokenizer()
    records = [_record(meta={"license": "CC0"}),
               _record("doc-b", "group-b", "validation", text="heldout text")]
    documents = load_documents(_write(tmp_path, records), tokenizer)
    assert tokenizer.calls == [record["text"] for record in records]
    assert documents[0].metadata == {"license": "CC0"}
    assert documents[0].text_sha256 == hashlib.sha256(records[0]["text"].encode()).hexdigest()
    assert documents[0].tokenizer_identity == tokenizer.name_or_path
    assert len(documents[0].token_sha256) == 64
    windows = build_windows(documents, block_size=2)
    sampler = RawWindowSampler(windows, seed=8)
    sampler.next_batch(10)
    assert tokenizer.calls == [record["text"] for record in records]
    assert all(window.document_id == "doc-a" for bucket in windows.values() for window in bucket)


def test_complete_nonoverlapping_windows_per_bucket_never_cross_documents():
    docs = [RawDocument("a", "ga", "books", "train", tuple(range(67))),
            RawDocument("b", "gb", "books", "train", tuple(range(100, 130)))]
    windows = build_windows(docs, block_size=2)
    lookup = {document.id: document for document in docs}
    for blocks, bucket in windows.items():
        previous_end = {}
        for window in bucket:
            assert len(window.token_ids) == 2 * blocks
            assert window.token_ids == lookup[window.document_id].token_ids[window.start:window.end]
            assert window.start >= previous_end.get(window.document_id, 0)
            previous_end[window.document_id] = window.end
            assert window.loss_tokens == (blocks - 1) * 2
    assert len(windows[8]) == 5 and len(windows[16]) == 2
    assert windows[8][0].start == windows[16][0].start == 0
    assert windows[8][0].document_id == windows[16][0].document_id


@pytest.mark.parametrize("records,match", [
    ([_record(), _record()], "duplicate document id"),
    ([_record(), _record("other", "group-a", "validation")], "crosses splits"),
    ([_record(text=" ")], "text must be"),
    ([_record(id="")], "id must be"),
    ([_record(split="test")], "split must be"),
    ([_record(unexpected=True)], "invalid fields"),
    ([_record(token_ids=[1, 2])], "exactly one"),
    ([_record(meta=[])], "metadata must be"),
])
def test_reject_invalid_records(tmp_path, records, match):
    with pytest.raises(ValueError, match=match):
        load_documents(_write(tmp_path, records), CountingTokenizer())


def test_tokenized_input_requires_explicit_matching_identity(tmp_path):
    record = _record()
    del record["text"]
    record["token_ids"] = list(range(32))
    with pytest.raises(ValueError, match="explicit tokenizer_identity"):
        load_documents(_write(tmp_path, [record]), CountingTokenizer())
    document = load_documents(_write(tmp_path, [record]), tokenizer_identity="fixture-tokenizer-v1")[0]
    assert document.token_ids == tuple(record["token_ids"])
    assert document.text_sha256 is None
    record["tokenizer_identity"] = "wrong-tokenizer"
    with pytest.raises(ValueError, match="differs"):
        load_documents(_write(tmp_path, [record]), CountingTokenizer())


@pytest.mark.parametrize("tokens", [[], [True], [-1], [1.5], "1 2"])
def test_reject_invalid_token_ids(tmp_path, tokens):
    record = _record()
    del record["text"]
    record.update(token_ids=tokens, tokenizer_identity="fixture-tokenizer-v1")
    with pytest.raises(ValueError, match="token_ids"):
        load_documents(_write(tmp_path, [record]))


def test_token_fair_mix_uses_target_tokens_instead_of_equal_trajectory_counts():
    sampler = RawWindowSampler(_windows(), seed=9, length_weights={8: 1, 16: 1},
                               source_weights={"books": 3, "papers": 1})
    batch = sampler.next_batch(80)
    tokens = sum(window.loss_tokens for window in batch)
    assert abs(sampler.length_loss_tokens[8] - tokens * .5) <= 30
    assert abs(sampler.source_loss_tokens["books"] - tokens * .75) <= 30
    assert sum(window.blocks == 8 for window in batch) > sum(window.blocks == 16 for window in batch)
    assert sampler.total_loss_tokens == tokens


def test_resume_is_exact_after_json_round_trip_across_shuffle_epochs():
    sampler = RawWindowSampler(_windows(), seed=19, length_weights={8: 2, 16: 3})
    sampler.next_batch(77)
    checkpoint = json.loads(json.dumps(sampler.state_dict()))
    expected = sampler.next_batch(120)
    restored = RawWindowSampler(_windows(), seed=999, length_weights={8: 2, 16: 3})
    restored.load_state_dict(checkpoint)
    assert restored.next_batch(120) == expected
    assert restored.state_dict() == sampler.state_dict()


def test_resume_rejects_changed_corpus_or_quota():
    windows = _windows()
    checkpoint = RawWindowSampler(windows, seed=2).state_dict()
    changed = dict(windows)
    changed[8] = changed[8][1:]
    with pytest.raises(ValueError, match="fingerprint"):
        RawWindowSampler(changed, seed=2).load_state_dict(checkpoint)
    with pytest.raises(ValueError, match="fingerprint"):
        RawWindowSampler(windows, seed=2, length_weights={8: 2, 16: 1}).load_state_dict(checkpoint)


def test_budget_stops_at_full_trajectory_and_does_not_consume_failed_draw():
    sampler = RawWindowSampler(_windows(), seed=7)
    before = sampler.state_dict()
    assert sampler.next_batch(4, remaining_loss_tokens=13) == []
    after = sampler.state_dict()
    assert sampler.last_status == {"reason": "insufficient_budget", "remaining_loss_tokens": 13,
                                   "minimum_loss_tokens": 14}
    for key in before.keys() - {"last_status"}:
        assert after[key] == before[key]
    batch = sampler.next_batch(100, remaining_loss_tokens=101)
    spent = sum(window.loss_tokens for window in batch)
    assert spent <= 101 and 101 - spent < 14
    assert all(len(window.token_ids) == window.blocks * 2 for window in batch)
    assert sampler.total_loss_tokens == spent


def test_insufficient_long_budget_can_still_select_a_complete_short_trajectory():
    sampler = RawWindowSampler(_windows(), seed=7, length_weights={8: 1, 16: 100})
    window = sampler.next_window(remaining_loss_tokens=14)
    assert window.blocks == 8 and window.loss_tokens == 14
    assert sampler.next_window(remaining_loss_tokens=0) is None


def test_nonzero_unavailable_source_or_length_quota_is_rejected():
    windows = _windows()
    with pytest.raises(ValueError, match="unavailable quota"):
        RawWindowSampler(windows, seed=0, source_weights={"missing": 1})
    with pytest.raises(ValueError, match="unavailable quota"):
        RawWindowSampler(windows, seed=0, length_weights={32: 1})
    windows[16] = [item for item in windows[16] if item.source == "books"]
    with pytest.raises(ValueError, match="nonzero quota has no windows"):
        RawWindowSampler(windows, seed=0)
    sampler = RawWindowSampler(windows, seed=0, source_weights={"books": 1, "papers": 0})
    assert all(window.source == "books" for window in sampler.next_batch(10))


def test_direct_document_and_window_construction_enforces_boundaries():
    with pytest.raises(ValueError, match="crosses splits"):
        build_windows([RawDocument("a", "same", "books", "train", tuple(range(32))),
                       RawDocument("b", "same", "books", "validation", tuple(range(32)))], 2)
    with pytest.raises(ValueError, match="complete equal-sized blocks"):
        TrajectoryWindow("a", "ga", "books", 0, (1, 2, 3), 2)

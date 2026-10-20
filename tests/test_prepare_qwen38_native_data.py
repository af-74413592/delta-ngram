"""Frozen-text retokenization: identities, chronological events and cache safety."""
from __future__ import annotations

import copy
import gzip
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "prepare_qwen38_native_data", ROOT / "scripts/prepare_qwen38_native_data.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)

from delta_ngram.data.raw_text import load_documents
from delta_ngram.data.event_streams import NEWS_REVISION, NEWS_SHARDS, load_event_documents


OLD_IDENTITY = "sha256:" + "a" * 64
RECEIPT = {"model_id": prepare.MODEL_ID, "revision": prepare.REVISION,
           "test_fixture": True}


class NativeTokenizerFixture:
    pad_token_id, eos_token_id, bos_token_id = 0, 1, None

    def __init__(self):
        self.calls = []

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        assert "DO NOT TOKENIZE" not in text
        self.calls.append(text)
        return [ord(character) for character in text]

    def decode(self, *args, **kwargs):
        raise AssertionError("never recover original text by decoding old tokens")

    def get_vocab(self):
        return {"fixture-native": 7, "字": ord("字")}


def lines(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def compressed_lines(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_hash(text):
    return prepare.hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest(directory):
    supplied = {
        "format": "delta_ngram_event_mixture_v1", "ready": True,
        "tokenizer": {"fingerprint": OLD_IDENTITY},
        "files": {name: prepare.file_info(directory / name) for name in prepare.INPUT_FILES},
        "training_source_weights": {"code": .2, "papers": .25, "books": .2,
                                    "web": .15, "news": .175, "news_event": .025},
        "resolved_raw_config_proposal": {"source_length_weights": {
            source: {str(n): 1 / len(lengths) for n in lengths}
            for source, lengths in prepare.LENGTHS.items()}},
    }
    path = directory / "data_manifest.json"
    path.write_text(json.dumps(supplied), encoding="utf-8")
    return prepare.file_info(path)["sha256"]


def corpus(tmp_path):
    directory = tmp_path / "frozen"
    directory.mkdir()
    documents, text_rows = [], []
    for source in prepare.SOURCES[:-1]:
        for ordinal, split in ((0, "train"), (1, "train"), (2, "validation")):
            content = "字" + source + str(ordinal) + "x" * 8192
            meta = {"text_sha256": text_hash(content), "token_sha256": prepare.digest([11, 12]),
                    "token_count": 2, "audit_component_id": f"component:{source}:{ordinal}",
                    "license": "fixture", "nested_provenance": {"unchanged": [1, 2]}}
            row = {"id": f"{source}:{ordinal}", "group_id": f"group:{source}:{ordinal}",
                   "source": source, "split": split, "token_ids": [11, 12],
                   "tokenizer_identity": OLD_IDENTITY, "metadata": meta}
            documents.append(row)
            text_rows.append({"id": row["id"], "text": content, "metadata": meta})
    events, event_texts = [], []
    source_row = 0
    # Native chunks move these formerly short streams into all four frozen buckets.
    for ordinal, (lengths, split) in enumerate((
        ((256, 256), "train"), ((1024, 2048), "train"),
        ((2048, 4096), "train"), ((4096, 8192), "train"),
        ((256, 512), "validation"),
    )):
        articles = []
        for position, length in enumerate(lengths):
            source_row += 1
            shard = next(iter(NEWS_SHARDS))
            origin = {"source_shard": shard, "source_row": source_row,
                      "source_shard_sha256": NEWS_SHARDS[shard]}
            article_id = "cc-news:" + prepare.digest([NEWS_REVISION, origin])
            content = chr(65 + ordinal) * (length - 1) + str(position)
            day = f"{2017 if split == 'train' else 2018}-01-{position + 1:02}"
            meta = {**origin, "dataset_revision": NEWS_REVISION,
                    "text_sha256": text_hash(content), "normalized_text_sha256": text_hash(content),
                    "token_sha256": prepare.digest([position + 1] * 128),
                    "canonical_url": "https://fixture.invalid/" + article_id,
                    "publication_date_original": day, "publication_date": day,
                    "complete_extracted_article": True, "truncated": False}
            articles.append({"id": article_id, "token_ids": [position + 1] * 128,
                             "metadata": meta})
            event_texts.append({"id": article_id, "text": content, "metadata": meta})
        members = [article["id"] for article in articles]
        meta = {"family_members": members, "stream_article_ids": members,
                "dataset_revision": NEWS_REVISION, "audit_component_id": f"event-component:{ordinal}",
                "article_boundaries": [
                    {"article_id": members[0], "start": 0, "end": 128, "role": "history"},
                    {"article_id": members[1], "start": 128, "end": 256, "role": "target"}],
                "raw_tokens": 256, "target_tokens": 128, "article_chunk_count": 2,
                "block_bucket": 4, "block_size": 512}
        events.append({"id": f"event:{ordinal}",
                       "group_id": "news-family:" + prepare.digest(sorted(members)),
                       "source": "news_event", "split": split, "articles": articles,
                       "tokenizer_identity": OLD_IDENTITY, "metadata": meta})
    lines(directory / prepare.INPUT_FILES[0], documents)
    lines(directory / prepare.INPUT_FILES[1], events)
    skipped = {"id": "nonselected-old-holdout", "text": "DO NOT TOKENIZE", "metadata": {}}
    compressed_lines(directory / prepare.INPUT_FILES[2], [skipped, *text_rows])
    compressed_lines(directory / prepare.INPUT_FILES[3], [skipped, *event_texts])
    # A sealed FINAL file is deliberately invalid and must never be opened.
    (directory / "sealed_FINAL.jsonl").write_bytes(b"DO NOT READ: INVALID JSON")
    return directory, manifest(directory), documents, events


def run(directory, checksum, output, tokenizer=None):
    return prepare.prepare(
        directory, output, expected_manifest_sha256=checksum,
        tokenizer=tokenizer or NativeTokenizerFixture(), tokenizer_receipt=RECEIPT)


def test_complete_derivation_preserves_membership_and_loads_in_existing_runtime(tmp_path):
    source, checksum, documents, events = corpus(tmp_path)
    before = {p.name: prepare.file_info(p) for p in source.iterdir()}
    tokenizer = NativeTokenizerFixture()
    output = tmp_path / "native"
    result = run(source, checksum, output, tokenizer)
    assert result["derived_cache_complete"] and result["ready_for_training"]
    assert result["native_runtime_accepted"] is False
    assert len(tokenizer.calls) == len(documents) + sum(len(e["articles"]) for e in events)
    assert result["retokenized_documents"] == 15
    assert result["retokenized_event_streams"] == 5
    assert result["retokenized_event_articles"] == 10
    assert all(n == 1 for n in result["skipped_nonselected_text_rows"].values())
    assert {p.name: prepare.file_info(p) for p in source.iterdir()} == before
    assert not (output / "INCOMPLETE.json").exists()
    seal = json.loads((output / "COMPLETE.json").read_text())
    assert seal["data_manifest"] == prepare.file_info(output / "data_manifest.json")
    for name, expected in seal["files"].items():
        assert prepare.file_info(output / name) == expected

    identity = result["tokenizer"]["fingerprint"]
    loaded = load_documents(output / prepare.INPUT_FILES[0], tokenizer_identity=identity,
                            require_record_identity=True)
    assert [(d.id, d.group_id, d.source, d.split) for d in loaded] == [
        (d["id"], d["group_id"], d["source"], d["split"]) for d in documents]
    assert loaded[0].token_ids[:1] == (ord("字"),)
    assert loaded[0].metadata["parent_token_count"] == 2
    assert loaded[0].metadata["token_count"] == len(tokenizer.calls[0])
    assert loaded[0].metadata["nested_provenance"] == {"unchanged": [1, 2]}
    event_loaded = load_event_documents(output / prepare.INPUT_FILES[1], tokenizer_identity=identity)
    assert [e.id for e in event_loaded] == [e["id"] for e in events]
    assert [e.metadata["block_bucket"] for e in event_loaded] == [4, 8, 16, 32, 4]
    assert event_loaded[1].metadata["article_boundaries"][1]["start"] == 1024
    assert event_loaded[1].metadata["target_tokens"] == 2048
    assert event_loaded[1].metadata["article_chunk_count"] == 6

    index = list(prepare.json_lines(output / "document_index.jsonl"))
    assert index[0]["parent_metadata"] == documents[0]["metadata"]
    assert prepare.read_cached_tokens(output / "tokens.uint32", index[0]) == list(loaded[0].token_ids)
    selection = json.loads((output / "calibration_selection.json").read_text())
    assert selection["ready"] and len(selection["samples"]) == 12
    assert all(s["split"] == "train" for s in selection["samples"])
    assert len({s["group_id"] for s in selection["samples"]}) == 12


def test_hash_failure_precedes_tokenization_and_output_creation(tmp_path):
    source, checksum, _, _ = corpus(tmp_path)
    tokenizer = NativeTokenizerFixture()
    with (source / prepare.INPUT_FILES[0]).open("ab") as stream:
        stream.write(b"\n")
    output = tmp_path / "native"
    with pytest.raises(ValueError, match="integrity mismatch"):
        run(source, checksum, output, tokenizer)
    assert not tokenizer.calls and not output.exists()


def test_calibration_only_derives_twelve_fixed_train_families_and_no_validation(tmp_path):
    source, checksum, _, _ = corpus(tmp_path)
    tokenizer = NativeTokenizerFixture()
    output = tmp_path / "calibration"
    result = prepare.prepare(source, output, expected_manifest_sha256=checksum,
                             tokenizer=tokenizer, tokenizer_receipt=RECEIPT,
                             calibration_only=True)
    assert result["derivation_scope"] == "calibration_subset_only"
    assert result["ready_for_calibration"] and not result["ready_for_training"]
    assert result["parent_membership_counts"] == {
        "documents": 15, "event_streams": 5, "event_articles": 10}
    assert result["retokenized_documents"] == 10
    assert result["retokenized_event_streams"] == 2
    assert len(tokenizer.calls) == 14
    selection = json.loads((output / "calibration_selection.json").read_text())
    assert len(selection["samples"]) == 12
    assert all(row["split"] == "train" for row in selection["samples"])
    assert all(row["split"] == "train" for row in prepare.json_lines(output / prepare.INPUT_FILES[0]))
    assert all(row["split"] == "train" for row in prepare.json_lines(output / prepare.INPUT_FILES[1]))
    assert selection["selection_scope"] == "fixed_parent_training_family_hash_prefix_no_replacement"


def test_original_text_mismatch_fails_without_a_complete_seal(tmp_path):
    source, _, _, _ = corpus(tmp_path)
    with gzip.open(source / prepare.INPUT_FILES[2], "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    rows[1]["text"] = "changed original"
    compressed_lines(source / prepare.INPUT_FILES[2], rows)
    output = tmp_path / "native"
    with pytest.raises(ValueError, match="original text SHA256"):
        run(source, manifest(source), output)
    assert (output / "FAILURE.json").exists()
    assert (output / "INCOMPLETE.json").exists()
    assert not (output / "COMPLETE.json").exists()


def test_new_event_limit_violation_is_reported_without_dropping_or_truncating(tmp_path):
    source, _, _, events = corpus(tmp_path)
    with gzip.open(source / prepare.INPUT_FILES[3], "rt", encoding="utf-8") as stream:
        texts = [json.loads(line) for line in stream]
    content = "z" * 17000
    texts[1]["text"] = content
    texts[1]["metadata"]["text_sha256"] = text_hash(content)
    events[0]["articles"][0]["metadata"]["text_sha256"] = text_hash(content)
    lines(source / prepare.INPUT_FILES[1], events)
    compressed_lines(source / prepare.INPUT_FILES[3], texts)
    output = tmp_path / "native"
    result = run(source, manifest(source), output)
    assert result["derived_cache_complete"] and not result["ready_for_training"]
    assert any(v.get("id") == "event:0" for v in result["violations"])
    native_events = list(prepare.json_lines(output / prepare.INPUT_FILES[1]))
    assert len(native_events) == len(events)
    assert len(native_events[0]["articles"][0]["token_ids"]) == 17000
    assert native_events[0]["metadata"]["block_bucket"] is None


@pytest.mark.parametrize("mutation,match", [
    ("split", "source/split"),
    ("duplicate", "duplicate/invalid"),
    ("group_leak", "crosses splits"),
    ("old_identity", "old tokenizer"),
])
def test_reject_invalid_frozen_membership(tmp_path, mutation, match):
    source, _, documents, _ = corpus(tmp_path)
    if mutation == "split":
        documents[0]["split"] = "holdout"
    elif mutation == "duplicate":
        documents.append(copy.deepcopy(documents[0]))
    elif mutation == "group_leak":
        documents[2]["group_id"] = documents[0]["group_id"]
    else:
        documents[0]["tokenizer_identity"] = "sha256:" + "b" * 64
    lines(source / prepare.INPUT_FILES[0], documents)
    with pytest.raises(ValueError, match=match):
        run(source, manifest(source), tmp_path / "native")


def test_calibration_selection_is_order_independent_and_excludes_validation():
    rows = [{"id": str(i), "source": source, "group_id": source + str(i // 2),
             "split": "train" if i < 8 else "validation", "eligible": True}
            for source in prepare.SOURCES for i in range(10)]
    first = prepare.calibration_selection(rows, 17, 2)
    second = prepare.calibration_selection(list(reversed(rows)), 17, 2)
    assert first == second
    assert len(first["samples"]) == 12
    assert all(s["split"] == "train" for s in first["samples"])
    for row in rows:
        if row["source"] == "books":
            row["eligible"] = False
    limited = prepare.calibration_selection(rows, 17, 2)
    assert not limited["ready"] and limited["deficits"] == {"books": 2}


def test_output_is_exclusive_and_cannot_be_inside_frozen_source(tmp_path):
    source, checksum, _, _ = corpus(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        run(source, checksum, source / "native")
    output = tmp_path / "native"
    output.mkdir()
    marker = output / "user-file"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="output exists"):
        run(source, checksum, output)
    assert marker.read_text() == "keep"


def test_tokenizer_file_receipt_is_pinned_and_checks_actual_bytes(tmp_path):
    directory = tmp_path / "snapshots" / prepare.REVISION
    directory.mkdir(parents=True)
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    observed = prepare.local_tokenizer_receipt(directory)
    assert observed["revision"] == prepare.REVISION
    receipt = tmp_path / "tokenizer-receipt.json"
    receipt.write_text(json.dumps(observed), encoding="utf-8")
    assert prepare.local_tokenizer_receipt(directory, expected_manifest=receipt)["files"] == observed["files"]
    (directory / "tokenizer.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="hashes differ"):
        prepare.local_tokenizer_receipt(directory, expected_manifest=receipt)

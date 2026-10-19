"""Offline temporal-leakage and integrity tests for dated old-news preparation."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_news_corpus.py"
SPEC = importlib.util.spec_from_file_location("prepare_news_corpus", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def source_record(index=0, day="2018-12-31 23:59:59", text=None):
    return {"title": f"Unique subject {index}", "url": f"https://example.org/article/{index}",
            "domain": "example.org", "date": day, "description": "source description",
            "image_url": f"https://example.org/{index}.jpg",
            "text": text if text is not None else " ".join(f"word_{index}_{n}" for n in range(160))}


def candidate(index=0, day="2018-12-31 23:59:59", text=None):
    return prepare.candidate_record(source_record(index, day, text), prepare.expected_files()[0], index + 1)


@pytest.mark.parametrize("value,split", [
    ("2017-01-01", "train"), ("2018-12-31 23:59:59", "train"),
    ("2019-01-01T00:00:00Z", "validation"), ("2019-06-30", "validation"),
    ("2019-07-01", "holdout"), ("2019-12-31 23:59:59", "holdout"),
])
def test_date_boundaries_use_explicit_publication_date(value, split):
    assert prepare.temporal_split(prepare.publication_day(value)) == split


@pytest.mark.parametrize("value", [None, "", "2019", "2019-02-29", "2018-01-01garbage", "2018-01-01 77:00:00", "01/02/2018"])
def test_malformed_and_missing_dates_are_never_inferred(value):
    with pytest.raises(ValueError):
        prepare.publication_day(value)


@pytest.mark.parametrize("value", ["2016-12-31", "2020-01-01", "2025-10-19", "2026-10-19"])
def test_recent_news_cannot_enter_any_old_corpus_split(value):
    with pytest.raises(ValueError, match="outside registered"):
        candidate(day=value)


def test_url_canonicalization_preserves_identity_query_and_rejects_credentials():
    assert prepare.canonical_url("http://www.Example.org:80/a/?b=2&utm_source=test&a=1#frag") == "https://example.org/a?a=1&b=2"
    assert prepare.canonical_url("https://example.org/?id=1") != prepare.canonical_url("https://example.org/?id=2")
    with pytest.raises(ValueError):
        prepare.canonical_url("https://user:password@example.org/a")


def test_manifest_requires_all_exact_pins_and_enforces_download_budget():
    good = {"dataset_id": prepare.DATASET, "dataset_revision": prepare.REVISION, "files": prepare.expected_files()}
    assert sum(entry["size"] for entry in prepare.validate_download_manifest(good, 1_200_000_000)) == 1_122_805_586
    bad = copy.deepcopy(good)
    bad["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exactly the five pinned"):
        prepare.validate_download_manifest(bad, 1_200_000_000)
    with pytest.raises(ValueError, match="budget"):
        prepare.validate_download_manifest(good, 1_000_000_000)
    bad = copy.deepcopy(good)
    bad["files"][0]["path"] = "../../outside.parquet"
    with pytest.raises(ValueError):
        prepare.validate_download_manifest(bad, 1_200_000_000)


def test_complete_source_and_metadata_are_preserved_without_mutation():
    record = source_record()
    original = copy.deepcopy(record)
    item = prepare.candidate_record(record, prepare.expected_files()[0], 1)
    assert record == original and item["text"] == record["text"]
    assert item["upstream_metadata"]["date"] == record["date"]
    assert prepare.normalize("  x  \r\n y \r\n") == "x\n y"


def test_reservoir_has_order_independent_hard_bound():
    records = [candidate(index) for index in range(40)]
    first, second = prepare.Reservoir(5, 17), prepare.Reservoir(5, 17)
    for item in records:
        first.add(item)
    for item in reversed(records):
        second.add(item)
    assert first.selected() == second.selected()
    assert len(first.selected()) == 5


def test_exact_duplicate_crossing_time_quarantines_entire_transitive_component():
    first, second = candidate(0), candidate(1)
    third, unrelated = candidate(2, "2019-01-01"), candidate(3, "2019-08-01")
    second["canonical_url"] = first["canonical_url"]
    third["text"] = second["text"] + "\n"
    kept, stats = prepare.form_families([first, second, third, unrelated])
    assert [item["document_id"] for item in kept] == [unrelated["document_id"]]
    assert stats["cross_time_quarantined_families"] == 1
    assert stats["cross_time_quarantined_documents"] == 3


def test_near_duplicate_crossing_validation_holdout_is_quarantined():
    text = " ".join(f"river{i} valley{i} report{i}" for i in range(150))
    first = candidate(0, "2019-06-30", text)
    second = candidate(1, "2019-07-01", text.replace("river50", "changed_once"))
    kept, stats = prepare.form_families([first, second])
    assert not kept and stats["confirmed_near_edges"] == 1
    assert stats["cross_time_quarantined_documents"] == 2


def test_equal_headline_links_different_updates_before_temporal_assignment():
    first, second = candidate(0), candidate(1, "2019-02-01")
    for item in (first, second):
        item["upstream_metadata"]["title"] = "Example council announces substantial river bridge funding"
    kept, stats = prepare.form_families([first, second])
    assert not kept and stats["headline_edges"] == 1


def test_length_report_counts_short_articles_and_never_packs_or_truncates():
    records = [candidate(0), candidate(1)]
    recorded = []

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            recorded.append((texts, kwargs))
            return {"input_ids": [[1] * 700, [2] * 2100]}

    kept, stats = prepare.tokenize(records, Tokenizer())
    assert len(kept) == 1 and len(kept[0]["token_ids"]) == 2100
    assert recorded == [([item["text"] for item in records], {"add_special_tokens": False, "truncation": False})]
    assert stats["train"]["at_least"]["512"] == 2
    assert stats["train"]["at_least"]["1024"] == 1
    assert stats["train"]["below_minimum"] == 1


def tokenized_documents():
    days = ["2018-06-01", "2019-05-01", "2019-08-01"]
    docs = [candidate(index, day) for index, day in enumerate(days)]
    docs, _ = prepare.form_families(docs)
    for item in docs:
        item["token_ids"] = [1] * 2100
    return docs


def test_budget_selection_reports_insufficient_pool_and_counts_tails():
    docs = tokenized_documents()
    selected = prepare.choose_documents(docs, train_tokens=2000, eval_tokens=2000, minimum_eval_groups=1, seed=17)
    stats = prepare.validate_and_summarize(selected, minimum_eval_groups=1)
    assert stats["splits"]["train"]["window_loss_tokens"]["2"] == 1024
    assert stats["splits"]["train"]["unused_tail_tokens"]["2"] == 52
    with pytest.raises(ValueError, match="insufficient train corpus"):
        prepare.choose_documents(docs, train_tokens=5000, eval_tokens=2000, minimum_eval_groups=1, seed=17)


def test_export_preserves_dated_provenance_and_loader_contract(tmp_path):
    from delta_ngram.data.raw_text import load_documents
    item = tokenized_documents()[0]
    record = prepare.export_document(item, "sha256:test")
    metadata = record["metadata"]
    assert metadata["publication_date_original"] == "2018-06-01"
    assert metadata["source_shard_sha256"] == prepare.FILES[0][1]
    assert metadata["license"] == "unknown"
    assert metadata["text_sha256"] == prepare.text_hash(item["text"])
    assert metadata["token_sha256"] == prepare.digest(item["token_ids"])
    path = tmp_path / "news.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    loaded = load_documents(path, tokenizer_identity="sha256:test", require_record_identity=True)
    assert loaded[0].source == "news" and loaded[0].token_ids == tuple(item["token_ids"])
    record["split"] = "holdout"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split must be"):
        load_documents(path, tokenizer_identity="sha256:test", require_record_identity=True)


def test_final_audit_rejects_forged_time_and_exact_cross_split_hash():
    docs = tokenized_documents()
    docs[0]["publication_date"] = "2026-10-19"
    with pytest.raises(ValueError):
        prepare.validate_and_summarize(docs, minimum_eval_groups=1)
    docs = tokenized_documents()
    docs[1]["normalized_text_sha256"] = docs[0]["normalized_text_sha256"]
    with pytest.raises(ValueError, match="normalized_text_sha256 crosses splits"):
        prepare.validate_and_summarize(docs, minimum_eval_groups=1)

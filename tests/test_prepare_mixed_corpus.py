"""Offline provenance, independence and resource-bound checks for mixed data."""
from argparse import Namespace
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_mixed_corpus.py"
SPEC = importlib.util.spec_from_file_location("prepare_mixed_corpus", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def source_record(source="papers", number=0):
    text = "\n".join(f"The analysis of result_{i} explains independent experiment_{i} and observation_{i}." for i in range(400))
    if source == "papers":
        return {"id": "2101.12345v2", "text": text,
                "metadata": {"license": "CC-BY-4.0", "url": "https://arxiv.org/abs/2101.12345"}}
    if source == "books":
        return {"id": str(number + 1), "text": text, "metadata": {"language": "en", "license": "Public Domain",
                "title": "A Tale of Two Cities", "url": f"https://www.gutenberg.org/ebooks/{number + 1}"}}
    return {"id": f"uuid-{number}", "text": text, "url": f"https://example.org/page/{number}?utm_source=tracking",
            "language": "en", "language_score": 0.99, "file_path": "s3://crawl/path", "dump": "CC-MAIN-2020-01"}


def item(number, source="papers", tokens=8192, split=None):
    value = {"id": f"{source}-{number}", "document_id": str(number), "source": source,
        "group_id": f"family-{source}-{number}", "work_keys": [f"work-{source}-{number}"],
        "token_count": tokens, "metadata": {"text_sha256": prepare.text_hash(f"raw-{source}-{number}"),
            "normalized_text_sha256": prepare.text_hash(f"norm-{source}-{number}")}}
    if split:
        value["split"] = split
    return value


def test_work_aliases_group_arxiv_versions_and_book_editions():
    record = source_record()
    assert prepare.work_keys("papers", record) == ["arxiv:2101.12345"]
    record["id"] = "hep-th/9901001v5"
    assert prepare.work_keys("papers", record) == ["arxiv:hep-th/9901001"]
    a, b = source_record("books", 2), source_record("books", 3)
    b["metadata"]["title"] = "A TALE of Two Cities!"
    assert set(prepare.work_keys("books", a)) & set(prepare.work_keys("books", b))
    assert prepare.canonical_url("http://Example.ORG/a/?utm_source=x&v=2#foo") == "https://example.org/a?v=2"
    assert prepare.canonical_url("http://www.Example.org:80/a?mc_cid=tracking") == "https://example.org/a"
    assert prepare.canonical_url("https://www.Example.org:443/a") == "https://example.org/a"


@pytest.mark.parametrize("source", ["papers", "books", "web"])
def test_metadata_filter_keeps_original_text_and_rights(source):
    record = source_record(source)
    before = deepcopy(record)
    assert prepare.eligible_record(source, record) is None
    assert record == before
    if source != "web":
        del record["metadata"]["license"]
        assert prepare.eligible_record(source, record) == "missing_record_license"
    else:
        record["language_score"] = 0.1
        assert prepare.eligible_record(source, record) == "non_english"


def test_download_manifest_rejects_path_escape_and_revision_change():
    source, spec = "papers", prepare.SOURCES["papers"]
    entry = {"source": source, "dataset_id": spec["dataset_id"], "dataset_revision": spec["revision"],
        "path": "arxiv-papers-0000.json.gz", "local_path": "raw/papers/arxiv-papers-0000.json.gz",
        "size": 10, "sha256": "a" * 64,
        "url": f"https://huggingface.co/datasets/{spec['dataset_id']}/resolve/{spec['revision']}/arxiv-papers-0000.json.gz"}
    assert prepare.validate_download_manifest({"files": [entry]}, 20) == [entry]
    for field, value in [("local_path", "../outside"), ("dataset_revision", "main"), ("url", "https://example.org/a")]:
        invalid = {**entry, field: value}
        with pytest.raises(ValueError, match="invalid"):
            prepare.validate_download_manifest({"files": [invalid]}, 20)
    with pytest.raises(ValueError, match="budget"):
        prepare.validate_download_manifest({"files": [entry]}, 5)


def test_disk_reservoir_is_bounded_deterministic_and_tokens_roundtrip(tmp_path):
    records = [item(index) for index in range(30)]
    selected = []
    for reverse in (False, True):
        store = prepare.CandidateStore(tmp_path / f"store-{reverse}.sqlite")
        reservoir = prepare.DiskReservoir(4, 17, store)
        for value in reversed(records) if reverse else records:
            reservoir.add(value, value["id"] * 100)
        kept = reservoir.selected()
        selected.append([value["id"] for value in kept])
        assert store.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 4
        store.set_tokens(kept[0]["id"], [0, 1, 151645, 2 ** 32 - 1])
        store.update_text(kept[0]["id"], "replaced full text")
        assert store.tokens(kept[0]["id"]) == [0, 1, 151645, 2 ** 32 - 1]
        assert store.text(kept[0]["id"]) == "replaced full text"
        store.close()
    assert selected[0] == selected[1]


def test_families_union_aliases_exact_and_confirmed_near_duplicates():
    records = [item(index) for index in range(4)]
    records[3]["work_keys"] = records[1]["work_keys"]
    base = " ".join(f"independent_word_{i}" for i in range(300))
    texts = {records[0]["id"]: base, records[1]["id"]: base + "\n",
             records[2]["id"]: base.replace("word_150", "renamed_150"), records[3]["id"]: "separate book edition"}
    stats, edges = prepare.form_families(records, texts.__getitem__)
    assert stats["exact_edges"] == 1 and stats["near_edges"] >= 1
    assert len({record["group_id"] for record in records}) == 1
    assert edges


def test_split_preserves_independent_groups_and_does_not_claim_per_length_64():
    records = [item(index, tokens=8192 if index < 3 else 4096) for index in range(192)]
    prepare.assign_splits(records, 17)
    stats = prepare.validate_and_summarize(records, {"papers": [8, 16]})
    for split in ("train", "validation", "holdout"):
        assert stats["sources"]["papers"][split]["groups"] == 64
        assert stats["sources"]["papers"][split]["eligible_groups"]["16"] == 1
    bad = deepcopy(records)
    bad[0]["group_id"] = next(record["group_id"] for record in records if record["split"] != bad[0]["split"])
    with pytest.raises(ValueError, match="group_id crosses"):
        prepare.validate_and_summarize(bad, {"papers": [8, 16]})


def test_missing_long_window_cell_and_insufficient_pool_fail():
    records = [item(index, tokens=4096) for index in range(12)]
    with pytest.raises(ValueError, match="longest-window"):
        prepare.assign_splits(records, 17, minimum_groups=1)
    with pytest.raises(ValueError, match="exhausted"):
        prepare.choose_documents(records, 999999, 999999, 1, 17)


def test_cross_split_families_quarantined_without_reassigning_splits():
    records = [item(0, split="train"), item(1, "web", split="validation"), item(2, split="train"), item(3, "code", split="train")]
    records[0]["group_id"] = records[1]["group_id"] = "conflict"
    records[2]["group_id"] = records[3]["group_id"] = "duplicate"
    kept, audit = prepare.quarantine_and_deduplicate(records, [(0, 1, 1), (2, 3, 0.95)])
    assert [value["id"] for value in kept] == [records[3]["id"]]
    assert kept[0]["split"] == "train" and audit["conflicted_families"] == 1
    assert len(audit["removed_documents"]) == 3


def create_prepared(tmp_path, source, metadata=None):
    directory = tmp_path / source
    directory.mkdir()
    store = prepare.CandidateStore(directory / "build.sqlite")
    records = []
    for number, split in enumerate(("train", "validation", "holdout")):
        value = item(number, source, split=split)
        text = " ".join(f"{source}_{number}_word_{i}" for i in range(100))
        value["metadata"].update(text_sha256=prepare.text_hash(text), normalized_text_sha256=prepare.text_hash(text))
        if metadata:
            value["metadata"].update(metadata(number))
        store.put(value["id"], text)
        store.set_tokens(value["id"], [number + 1] * 8192)
        records.append(value)
    fingerprint = "sha256:" + "f" * 64
    files = prepare.export_files(directory, records, store, fingerprint)
    store.close()
    tokenizer = {"name": prepare.TOKENIZER, "revision": prepare.TOKENIZER_REVISION, "fingerprint": fingerprint}
    prepare.save_json(directory / "data_manifest.json", {"ready": True, "tokenizer": tokenizer, "files": files})
    return directory


def test_merge_end_to_end_is_immutable_and_holds_out_complete_documents(tmp_path):
    a, b = create_prepared(tmp_path, "papers"), create_prepared(tmp_path, "books")
    args = Namespace(output_dir=str(tmp_path / "mixed"), input_dirs=[str(a), str(b)], minimum_groups=1,
                     source_lengths='{"papers":[8,16],"books":[8,16]}', source_weights='{"papers":0.5,"books":0.5}', near_threshold=0.85,
                     source_length_weights=None, planned_loss_tokens=10000, maximum_cell_passes=3.0)
    manifest = prepare.run_merge(args)
    assert manifest["ready"] and manifest["statistics"]["documents"] == 6
    records = [json.loads(line) for line in (tmp_path / "mixed" / "train_validation.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4 and {row["split"] for row in records} == {"train", "validation"}
    assert all(len(row["token_ids"]) == 8192 for row in records)
    from delta_ngram.data.raw_text import load_documents
    loaded = load_documents(tmp_path / "mixed" / "train_validation.jsonl", require_record_identity=True)
    assert len(loaded) == 4 and {document.source for document in loaded} == {"papers", "books"}
    with pytest.raises(FileExistsError):
        prepare.run_merge(args)


def test_import_rejects_corrupted_manifest_file_and_mismatched_tokenizer(tmp_path):
    source = create_prepared(tmp_path, "papers")
    store = prepare.CandidateStore(tmp_path / "import.sqlite")
    with pytest.raises(ValueError, match="tokenizer identities differ"):
        prepare.import_prepared(source, store, "sha256:" + "a" * 64)
    with (source / "train_validation.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="integrity"):
        prepare.import_prepared(source, store)
    store.close()


def test_capacity_gate_reports_cross_length_overlap_and_rejects_over_three_passes():
    records = [item(index, split=split) for index, split in enumerate(("train", "validation", "holdout"))]
    statistics = prepare.validate_and_summarize(records, {"papers": [8, 16]}, minimum_groups=1)
    report = prepare.budget_capacity(statistics, {"papers": 1.0}, {"papers": [8, 16]}, {"papers": {"8": 0.5, "16": 0.5}}, 10000, 3.0)
    assert report["passed"] and report["sources"]["papers"]["cross_length_overlap_lower_bound"] == 8192
    with pytest.raises(ValueError, match="capacity gate"):
        prepare.budget_capacity(statistics, {"papers": 1.0}, {"papers": [8, 16]}, {"papers": {"8": 0.5, "16": 0.5}}, 50000, 3.0)


def test_date_bounds_keep_crawl_publication_addition_and_missing_values_separate():
    records = [item(0, "web", split="train"), item(1, "web", split="train")]
    records[0]["metadata"].update(date="2024-04-01T00:00:00Z", dump="CC-MAIN-2024-18", added="2025-01-03")
    records[1]["metadata"].update(date=None, dump="CC-MAIN-2023-01")
    fields = prepare.selected_date_bounds(records)["web"]["train"]["fields"]
    assert fields["date"]["observed"] == 1 and fields["date"]["missing"] == 1
    assert fields["added"]["max"] == "2025-01-03" and fields["dump"]["min"] == "CC-MAIN-2023-01"
    assert fields["publication_date"]["observed"] == 0


def test_shared_news_web_url_links_changed_articles_before_cross_split_quarantine(tmp_path):
    news = create_prepared(tmp_path, "news", lambda number: {"canonical_url": "https://example.org/shared" if number == 0 else f"https://example.org/news/{number}"})
    web = create_prepared(tmp_path, "web", lambda number: {"url": "http://www.example.org:80/shared/?utm_source=x" if number == 1 else f"https://example.org/web/{number}"})
    store = prepare.CandidateStore(tmp_path / "merge.sqlite")
    first, _, _ = prepare.import_prepared(news, store)
    second, _, _ = prepare.import_prepared(web, store)
    records = first + second
    stats, edges = prepare.form_families(records, store.text)
    kept, audit = prepare.quarantine_and_deduplicate(records, edges)
    assert stats["work_edges"] == 1 and stats.get("near_edges", 0) == 0
    assert audit["conflicted_families"] == 1
    assert len(kept) == 4
    assert {row["id"] for row in audit["removed_documents"]} == {"news-0", "web-1"}
    store.close()


def test_same_split_cross_source_family_retains_specific_source_for_runtime_independence():
    news, web = item(0, "news", split="validation"), item(1, "web", split="validation")
    news["group_id"] = web["group_id"] = "same-article"
    kept, audit = prepare.quarantine_and_deduplicate([news, web], [])
    assert kept == [news] and audit["cross_source_families"] == 1
    assert audit["removed_documents"][0]["reason"] == "cross_source_family"
    with pytest.raises(ValueError, match="multiple sources"):
        prepare.validate_and_summarize([news, web], {"news": [2, 4], "web": [8, 16]}, minimum_groups=1)

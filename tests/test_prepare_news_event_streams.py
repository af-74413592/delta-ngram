"""Offline recovery, chronology, whole-article and capacity regression checks."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_news_event_streams.py"
SPEC = importlib.util.spec_from_file_location("prepare_news_event_streams", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)
news = prepare.news


def make_family(days=("2017-03-01", "2017-03-02"), *, offset=0, texts=None):
    items, rows = [], []
    shard = news.expected_files()[0]
    for index, day in enumerate(days):
        source = {"title": "Council publishes evolving bridge construction report",
                  "url": f"https://example.org/event/{offset}", "date": day,
                  "text": texts[index] if texts else " ".join(f"word_{offset}_{index}_{n}" for n in range(100))}
        row = offset + index + 1
        item = news.candidate_record(source, shard, row)
        item["text_sha256"] = news.text_hash(item["text"])
        item["normalized_text_sha256"] = news.text_hash(news.normalize(item["text"]))
        items.append(item)
        rows.append((shard, row, source))
    members = sorted(item["document_id"] for item in items)
    group = "news-family:" + news.digest(members)
    dates = sorted(item["publication_date"] for item in items)
    for item in items:
        item.update(group_id=group, family_members=members, family_date_min=dates[0], family_date_max=dates[-1])
    family = {"group_id": group, "split": items[0]["split"], "members": members,
              "family_date_min": dates[0], "family_date_max": dates[-1], "representative_id": items[0]["document_id"]}
    items[0]["token_ids"] = [1] * 1024
    representative = news.export_document(items[0], "sha256:test")
    family["representative_metadata"] = representative["metadata"]
    selected = {"id": representative["id"], "text": items[0]["text"], "metadata": representative["metadata"]}
    return family, items, rows, selected


def source_manifest():
    return {"format": "delta_ngram_news_corpus_v1", "ready": True,
            "dataset_id": news.DATASET, "dataset_revision": news.REVISION, "temporal_splits": news.TEMPORAL_SPLITS}


def write_selected(tmp_path, values):
    path = tmp_path / "selected_texts.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as output:
        for value in values:
            output.write(json.dumps(value) + "\n")
    return path


def test_selected_members_reconstruct_exact_group_without_top_level_group_or_split(tmp_path):
    family, items, rows, selected = make_family()
    assert "group_id" not in selected and "split" not in selected
    path = write_selected(tmp_path, [selected])
    families, owner, representatives = prepare.read_selected_families(path, source_manifest())
    assert list(families) == [family["group_id"]]
    recovered, stats = prepare.recover_members(rows, families, owner, representatives)
    assert {item["document_id"] for item in recovered[family["group_id"]]} == set(family["members"])
    assert stats["recovered_documents"] == 2
    assert recovered[family["group_id"]][1]["text"] == items[1]["text"]


@pytest.mark.parametrize("mutation,message", [
    (lambda value: value["metadata"].update(family_members=[]), "invalid family"),
    (lambda value: value["metadata"]["family_members"].append(value["id"]), "invalid family"),
    (lambda value: value.update(text="forged text"), "checksum"),
    (lambda value: value["metadata"].update(dataset_revision="main"), "revision"),
    (lambda value: value["metadata"].update(publication_date="2017-03-20"), "date mismatch"),
])
def test_selected_input_integrity_is_checked(tmp_path, mutation, message):
    selected = make_family()[3]
    mutation(selected)
    with pytest.raises(ValueError, match=message):
        prepare.read_selected_families(write_selected(tmp_path, [selected]), source_manifest())


def test_unfinished_original_corpus_cannot_supply_streams(tmp_path):
    manifest = source_manifest()
    manifest["ready"] = False
    with pytest.raises(ValueError, match="completed"):
        prepare.read_selected_families(write_selected(tmp_path, [make_family()[3]]), manifest)


def test_old_unsupported_temporal_registration_cannot_be_silently_remapped(tmp_path):
    manifest = source_manifest()
    manifest["temporal_splits"] = {"train": ["2017-01-01", "2018-12-31"]}
    with pytest.raises(ValueError, match="temporal splits differ"):
        prepare.read_selected_families(write_selected(tmp_path, [make_family()[3]]), manifest)


def test_recovery_requires_every_member_once_and_rechecks_representative(tmp_path):
    family, items, rows, selected = make_family()
    args = prepare.read_selected_families(write_selected(tmp_path, [selected]), source_manifest())
    with pytest.raises(ValueError, match="failed to recover 1"):
        prepare.recover_members(rows[:1], *args)
    with pytest.raises(ValueError, match="more than once"):
        prepare.recover_members(rows + rows, *args)
    forged = copy.deepcopy(rows)
    forged[0][2]["text"] += " a changed fact"
    with pytest.raises(ValueError, match="checksum|sha256"):
        prepare.recover_members(forged, *args)


def test_entire_family_time_partition_is_checked_before_duplicates_or_stream_selection(tmp_path):
    text = " ".join(f"river_{n}" for n in range(100))
    family, items, rows, selected = make_family(("2017-12-30", "2018-01-02"), texts=[text, text])
    args = prepare.read_selected_families(write_selected(tmp_path, [selected]), source_manifest())
    with pytest.raises(ValueError, match="crosses source time"):
        prepare.recover_members(rows, *args)


def test_family_date_range_must_match_all_recovered_members(tmp_path):
    family, items, rows, selected = make_family()
    selected["metadata"]["family_date_max"] = "2017-03-03"
    args = prepare.read_selected_families(write_selected(tmp_path, [selected]), source_manifest())
    with pytest.raises(ValueError, match="date range"):
        prepare.recover_members(rows, *args)


@pytest.mark.parametrize("days", [
    ("2017-03-01", "2017-03-01"),
    ("2017-03-01 00:01:00", "2017-03-01 23:59:00"),
    ("2017-03-01T01:00:00Z", "2017-03-01 23:59:00"),
    ("2017-03-01T01:00:00Z", "2017-03-01T02:00:00+01:00"),
])
def test_same_day_without_two_distinct_aware_instants_cannot_form_pair(days):
    assert prepare.choose_chain(make_family(days)[1]) == []


def test_same_day_explicit_offsets_order_by_utc_not_wall_clock():
    family, items, rows, selected = make_family(("2017-03-01T09:00:00+09:00", "2017-03-01T08:00:00Z"))
    chain = prepare.choose_chain(list(reversed(items)))
    assert [item["publication_date_original"] for item in chain] == [item["publication_date_original"] for item in items]
    assert prepare.strictly_before(*chain)


def test_timezone_offsets_reversing_calendar_order_are_handled_in_utc():
    family, items, rows, selected = make_family(("2017-03-01T23:00:00-10:00", "2017-03-02T01:00:00+14:00"))
    chain = prepare.choose_chain(items)
    assert chain == list(reversed(items))
    assert prepare.strictly_before(*chain)


def test_unknown_zone_separate_calendar_days_have_explicitly_limited_order_basis():
    chain = prepare.choose_chain(make_family(("2017-03-01 23:59:00", "2017-03-02 00:01:00"))[1])
    assert len(chain) == 2
    assert "extractor metadata" in prepare.ORDERING and "calendar dates" in prepare.ORDERING


def test_30_day_limit_and_four_article_cap_do_not_invent_or_repeat_articles():
    assert len(prepare.choose_chain(make_family(("2017-03-01", "2017-03-31"))[1])) == 2
    assert prepare.choose_chain(make_family(("2017-03-01", "2017-04-01"))[1]) == []
    items = make_family(tuple(f"2017-03-{day:02d}" for day in range(1, 8)))[1]
    chain = prepare.choose_chain(items)
    assert len(chain) == 4 and len({item["document_id"] for item in chain}) == 4
    assert prepare.choose_chain(list(reversed(items))) == chain


def test_exact_body_only_dedup_preserves_same_url_title_with_changed_fact():
    original = " ".join(f"river_{n}" for n in range(100)) + " The count is 15."
    update = original.replace("count is 15", "count is 16")
    items = make_family(("2017-03-01", "2017-03-02", "2017-03-03"), texts=[original, original + "\n", update])[1]
    retained, duplicates = prepare.deduplicate_exact(items)
    assert duplicates == 1 and [item["text"] for item in retained] == [original, update]
    assert len({item["canonical_url"] for item in retained}) == 1


class WordsTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {"input_ids": [[index + 1] * len(text.split()) for index, text in enumerate(texts)]}


def test_whole_articles_tokenized_without_truncation_and_exported_with_separate_boundaries():
    family, items, rows, selected = make_family()
    tokenizer = WordsTokenizer()
    streams, stats = prepare.make_streams({family["group_id"]: items}, {family["group_id"]: family},
                        tokenizer, "sha256:test", minimum_tokens=10)
    stream = streams[0]
    assert "token_ids" not in stream and stream["source"] == "news_event"
    assert len(stream["articles"]) == 2
    assert all(call[1] == {"add_special_tokens": False, "truncation": False} for call in tokenizer.calls)
    assert [len(article["token_ids"]) for article in stream["articles"]] == [100, 100]
    assert stream["metadata"]["article_boundaries"] == [
        {"article_id": items[0]["document_id"], "start": 0, "end": 100, "role": "history"},
        {"article_id": items[1]["document_id"], "start": 100, "end": 200, "role": "target"}]
    assert stream["metadata"]["human_verified_event"] is False
    assert stream["metadata"]["family_members"] == family["members"]
    for article, item in zip(stream["articles"], items):
        assert article["metadata"]["text_sha256"] == news.text_hash(item["text"])
        assert article["metadata"]["token_sha256"] == news.digest(article["token_ids"])
        assert article["metadata"]["truncated"] is False


def test_long_or_short_articles_are_excluded_whole_and_bias_counted():
    texts = [" ".join(f"word_{n}" for n in range(size)) for size in (50, 100, 150, 200)]
    family, items, rows, selected = make_family(("2017-03-01", "2017-03-02", "2017-03-03", "2017-03-04"), texts=texts)
    streams, stats = prepare.make_streams({family["group_id"]: items}, {family["group_id"]: family},
                        WordsTokenizer(), "sha256:test", minimum_tokens=75, maximum_tokens=175)
    assert [len(article["token_ids"]) for article in streams[0]["articles"]] == [100, 150]
    assert stats["train"]["short_articles_excluded"] == 1 and stats["train"]["long_articles_excluded"] == 1
    assert stats["train"]["long_article_tokens_excluded"] == 200


def streams_three_splits():
    streams = []
    for offset, dates in enumerate([("2017-03-01", "2017-03-02"), ("2018-03-01", "2018-03-02"), ("2018-06-01", "2018-06-02")]):
        family, items, _, _ = make_family(dates, offset=offset * 10)
        for item in items:
            item["token_ids"] = [1] * 100
        streams.append(prepare.export_stream(family, items, "sha256:test", 30))
    return streams


def test_independent_family_gate_includes_training_split_and_budget_counts_real_later_tokens():
    streams = streams_three_splits()
    summary = prepare.summarize(streams, total_training_tokens=5000)
    assert len(summary["readiness_failures"]) == 3
    assert summary["splits"]["train"]["history_tokens"] == 100
    assert summary["splits"]["train"]["later_article_tokens"] == 100
    assert summary["budget_options"]["0.05"]["fits_target_token_budget"] is True
    assert summary["budget_options"]["0.1"]["fits_target_token_budget"] is False
    assert summary["budget_options"]["0.05"]["maximum_available_target_tokens"] == 300
    assert prepare.summarize(streams, minimum_families=1)["readiness_failures"] == []
    with pytest.raises(ValueError, match="one stream"):
        prepare.summarize(streams + streams[:1])


def write_single_corpus(tmp_path, streams):
    records = []
    for index, stream in enumerate(streams):
        for suffix, count in (("", 2048), (":remaining", 4096)):
            records.append({"id": f"single:{index}{suffix}", "group_id": stream["group_id"] + suffix,
                            "split": stream["split"], "token_ids": [1] * count})
    manifest = {"files": {}}
    for name in ("train_validation.jsonl", "holdout.jsonl"):
        selected = [record for record in records if (record["split"] == "holdout") == (name == "holdout.jsonl")]
        path = tmp_path / name
        path.write_text("".join(json.dumps(record) + "\n" for record in selected), encoding="utf-8")
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": news.file_sha256(path)}
    return manifest


def test_single_news_remainder_capacity_and_eval_group_gate(tmp_path):
    streams = streams_three_splits()
    manifest = write_single_corpus(tmp_path, streams)
    remainder = prepare.single_news_remainder(manifest, tmp_path, streams)
    assert len(remainder["readiness_failures"]) == 2
    for split in news.SPLITS:
        cell = remainder["splits"][split]
        assert cell["independent_families"] == 1 and cell["removed_families"] == 1
        assert cell["raw_tokens"] == 4096 and cell["removed_raw_tokens"] == 2048
        assert cell["window_loss_tokens"]["2"] == 2048
        assert cell["window_loss_tokens"]["4"] == 3072
    assert "not additive" in remainder["capacity_note"]


def test_single_news_remainder_requires_verified_files_and_all_removed_families(tmp_path):
    streams = streams_three_splits()
    manifest = write_single_corpus(tmp_path, streams)
    forged = copy.deepcopy(streams)
    forged[0]["group_id"] = "news-family:missing"
    with pytest.raises(ValueError, match="missing from"):
        prepare.single_news_remainder(manifest, tmp_path, forged)
    (tmp_path / "holdout.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity mismatch"):
        prepare.single_news_remainder(manifest, tmp_path, streams)


def test_article_partial_chunks_have_independent_memory_budget_from_raw_tokens():
    items = make_family()[1]
    for item, count in zip(items, (8193, 8191)):
        item["token_ids"] = [1] * count
    assert prepare.chain_cost(items) == {"raw_tokens": 16384, "article_chunk_count": 33, "target_tokens": 8191}
    assert prepare.choose_chain(items, maximum_stream_tokens=16384, maximum_stream_chunks=32) == []
    items[0]["token_ids"] = [1] * 8192
    assert len(prepare.choose_chain(items, maximum_stream_tokens=16384, maximum_stream_chunks=32)) == 2


def test_chain_selection_keeps_feasible_whole_articles_when_intermediate_update_is_too_large():
    items = make_family(("2017-03-01", "2017-03-02", "2017-03-03"))[1]
    for item, count in zip(items, (100, 16300, 100)):
        item["token_ids"] = [1] * count
    chain = prepare.choose_chain(items, maximum_stream_tokens=16384, maximum_stream_chunks=32)
    assert chain == [items[0], items[2]]
    assert len(items[1]["token_ids"]) == 16300


def test_stream_bucket_counts_real_targets_without_padding_cost():
    stream = streams_three_splits()[0]
    stream["articles"][0]["token_ids"] = [1] * 513
    stream["articles"][1]["token_ids"] = [1] * 1025
    summary = prepare.summarize([stream], minimum_families=1)
    cell = summary["splits"]["train"]["buckets"]["8"]
    assert cell == {"families": 1, "actual_chunks": 5, "raw_tokens": 1538, "target_tokens": 1025}
    assert summary["splits"]["train"]["later_article_tokens"] == 1025


def test_summary_rejects_any_export_exceeding_registered_stream_memory_budget():
    stream = streams_three_splits()[0]
    for article, count in zip(stream["articles"], (8193, 8191)):
        article["token_ids"] = [1] * count
    with pytest.raises(ValueError, match="token or article-chunk"):
        prepare.summarize([stream])


def test_span_limit_checks_all_calendar_dates_when_aware_order_reverses_dates():
    items = make_family(("2017-03-02T00:00:00+14:00", "2017-03-01T23:00:00-12:00", "2017-04-01T00:00:00+14:00"))[1]
    # UTC order is the listed order. First->last calendar span=30, but middle
    # ->last spans 31 source days and must not be hidden by endpoint-only checks.
    assert prepare.strictly_before(items[0], items[1]) and prepare.strictly_before(items[1], items[2])
    chain = prepare.choose_chain(items)
    assert len(chain) == 2
    assert all(prepare.within_span(left, right, 30) for index, left in enumerate(chain) for right in chain[index + 1:])


def test_export_contract_loads_into_event_runtime_and_preserves_actual_loss_tokens(tmp_path):
    from delta_ngram.data.event_streams import load_event_documents, build_event_trajectories
    family, items, _, _ = make_family()
    for item, count in zip(items, (513, 1025)):
        item["token_ids"] = [1] * count
    stream = prepare.export_stream(family, items, "sha256:test", 30)
    path = tmp_path / "event.jsonl"
    path.write_text(json.dumps(stream) + "\n", encoding="utf-8")
    documents = load_event_documents(path, tokenizer_identity="sha256:test")
    trajectories = build_event_trajectories(documents)
    assert len(trajectories[8]) == 1
    trajectory = trajectories[8][0]
    assert trajectory.actual_blocks == 5 and trajectory.loss_tokens == 1025
    assert stream["metadata"]["block_bucket"] == trajectory.blocks
    assert stream["metadata"]["target_tokens"] == trajectory.loss_tokens
    stream["split"] = "holdout"
    path.write_text(json.dumps(stream) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="holdout"):
        load_event_documents(path, tokenizer_identity="sha256:test")


def test_failed_run_keeps_family_and_candidate_audit_without_publishing_ready_corpus(tmp_path, monkeypatch):
    rows, selections, streams = [], [], []
    for offset, days in enumerate([("2017-03-01", "2017-03-02"), ("2018-03-01", "2018-03-02"), ("2018-06-01", "2018-06-02")]):
        family, items, source_rows, selected = make_family(days, offset=offset * 10)
        for item in items:
            item["token_ids"] = [1] * 128
        rows.extend(source_rows)
        selections.append(selected)
        streams.append(prepare.export_stream(family, items, "sha256:test", 30))
    selected_path = write_selected(tmp_path, selections)
    manifest = {**source_manifest(), **write_single_corpus(tmp_path, streams),
                "download": {"files": news.expected_files()},
                "tokenizer": {"fingerprint": "sha256:" + news.digest({"test": "identity"})}}
    manifest["files"][selected_path.name] = {"bytes": selected_path.stat().st_size, "sha256": news.file_sha256(selected_path)}
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    download_path = tmp_path / "download_manifest.json"
    download_path.write_text(json.dumps({"dataset_id": news.DATASET, "dataset_revision": news.REVISION,
                                         "files": news.expected_files()}), encoding="utf-8")

    class FixedTokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1] * 128 for _ in texts]}

    monkeypatch.setattr(prepare, "verified_source_rows", lambda path, cap: iter(rows))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: FixedTokenizer())))
    monkeypatch.setitem(sys.modules, "delta_ngram.raw_lm", SimpleNamespace(tokenizer_identity=lambda value: {"test": "identity"}))
    output = tmp_path / "output"
    args = prepare.parser().parse_args(["--selected-texts", str(selected_path), "--download-manifest", str(download_path), "--output-dir", str(output)])
    with pytest.raises(ValueError, match="insufficient independent event-stream"):
        prepare.run(args)
    audit = json.loads((output / "event_stream_candidate_manifest.json").read_text(encoding="utf-8"))
    families = json.loads((output / "event_stream_families.json").read_text(encoding="utf-8"))
    assert audit["ready"] is False
    assert len(audit["statistics"]["readiness_failures"]) == 5
    assert all(len(families["splits"][split]) == 1 for split in news.SPLITS)
    assert not (output / "event_stream_manifest.json").exists()
    assert not (output / "train_validation_event_streams.jsonl").exists()


def test_selected_event_text_export_includes_every_selected_article_once_with_full_body(tmp_path):
    family, items, _, _ = make_family(("2017-03-01", "2017-03-02", "2017-03-03"))
    for item in items:
        item["token_ids"] = [1] * 128
    stream = prepare.export_stream(family, items[:2], "sha256:test", 30)
    path = tmp_path / "selected_event_texts.jsonl.gz"
    stats = prepare.export_event_texts(path, [stream, stream], {family["group_id"]: items})
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        exported = [json.loads(line) for line in handle]
    assert stats["articles"] == 2 and stats["sha256"] == news.file_sha256(path)
    assert [row["text"] for row in exported] == [item["text"] for item in items[:2]]
    assert [row["id"] for row in exported] == [item["document_id"] for item in items[:2]]
    assert all(row["metadata"]["text_sha256"] == news.text_hash(row["text"]) for row in exported)

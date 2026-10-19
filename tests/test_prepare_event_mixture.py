"""Whole-event cross-source audit, quarantine and actual-target budget gates."""
from argparse import Namespace
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SPEC = importlib.util.spec_from_file_location("prepare_event_mixture", Path(__file__).resolve().parents[1] / "scripts" / "prepare_event_mixture.py")
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)
mixed = prepare.mixed


def ordinary(name, source="news", split="train", group=None, text=None):
    text = text or " ".join(f"{name}_different_{n}" for n in range(120))
    group = group or "family:" + name
    day = {"train": "2017-03-01", "validation": "2018-03-01", "holdout": "2018-06-01"}[split]
    return {"id": name, "document_id": name, "source": source, "split": split, "group_id": group,
        "work_keys": ["imported-family:" + group], "token_count": 8192,
        "metadata": {"text_sha256": mixed.text_hash(text), "normalized_text_sha256": mixed.text_hash(mixed.normalize(text)),
                     "original_group_id": group, **({"publication_date": day, "publication_date_original": day}
                                                     if source == "news" else {})}}, text


def event(name="a", split="train", lengths=(128, 129)):
    group = "news-family:" + name
    stream = {"id": "event:" + name, "group_id": group, "source": "news_event", "split": split,
              "tokenizer_identity": "sha256:test", "metadata": {}, "articles": []}
    items, texts = [], {}
    for index, length in enumerate(lengths):
        item, text = ordinary(f"{name}-article-{index}", "news_event", split, group)
        item["token_count"] = length
        item["metadata"]["event_stream_id"] = stream["id"]
        items.append(item)
        texts[item["id"]] = text
        stream["articles"].append({"id": "original:" + item["id"], "audit_id": item["id"],
                                   "token_count": length, "metadata": dict(item["metadata"])})
    return stream, items, texts


def joint(items, texts, streams, seed=17):
    _, edges = mixed.form_families(items, texts.__getitem__)
    return prepare.select_joint_components(items, edges, streams, seed=seed)


def test_near_updates_inside_event_remain_a_complete_original_stream():
    stream, items, texts = event()
    base = " ".join(f"substantial_unique_{n}" for n in range(300))
    texts[items[0]["id"]] = base
    texts[items[1]["id"]] = base.replace("unique_20", "updated_20")
    kept, streams, audit = joint(items, texts, {stream["id"]: stream})
    assert not kept and streams == [stream] and len(streams[0]["articles"]) == 2
    assert streams[0]["group_id"] == "news-family:a"
    assert not audit["removed_event_streams"]


def test_later_nonrepresentative_article_cross_split_link_quarantines_whole_event():
    stream, items, texts = event()
    web, text = ordinary("web-copy", "web", "validation", text=texts[items[1]["id"]])
    texts[web["id"]] = text
    kept, streams, audit = joint(items + [web], texts, {stream["id"]: stream})
    assert not kept and not streams
    assert len(audit["quarantined_components"]) == 1
    assert audit["removed_event_streams"][0]["article_ids"] == [a["id"] for a in stream["articles"]]
    assert audit["removed_event_streams"][0]["reason"] == "family_crosses_imported_splits"


def test_removed_single_representative_remains_a_transitive_audit_bridge():
    stream, events, texts = event()
    anchor, anchor_text = ordinary("old-representative", group=stream["group_id"])
    other, _ = ordinary("code-copy", "code", "validation", text=anchor_text)
    texts.update({anchor["id"]: anchor_text, other["id"]: anchor_text})
    reserved = prepare.reserve_event_families([anchor, other], {stream["id"]: stream})
    assert anchor["audit_only"] and reserved[0]["reason"] == "event_family_reserved"
    kept, selected, audit = joint([anchor, other, *events], texts, {stream["id"]: stream})
    assert not kept and not selected and audit["quarantined_components"]


@pytest.mark.parametrize("source,keeps_event", [("code", False), ("papers", False), ("books", False), ("web", True)])
def test_same_split_source_priority_drops_whole_event_or_whole_other_source(source, keeps_event):
    stream, items, texts = event()
    other, text = ordinary("other", source, text=texts[items[1]["id"]])
    texts[other["id"]] = text
    kept, selected, audit = joint(items + [other], texts, {stream["id"]: stream})
    assert bool(selected) is keeps_event
    assert len(selected[0]["articles"]) == 2 if keeps_event else len(audit["removed_event_streams"]) == 1
    assert [row["id"] for row in kept] == ([] if keeps_event else ["other"])


def test_joined_event_families_choose_one_complete_stream_deterministically():
    first, a, ta = event("a")
    second, b, tb = event("b", lengths=(131, 132, 133))
    tb[b[1]["id"]] = ta[a[1]["id"]]
    originals = {row["id"]: row for row in (first, second)}
    _, selected, audit = joint(a + b, {**ta, **tb}, originals)
    assert len(selected) == 1 and selected[0] is originals[selected[0]["id"]]
    assert len(selected[0]["articles"]) in (2, 3)
    assert audit["removed_event_streams"][0]["reason"] == "joined_event_family"
    assert len(set(audit["event_component_mapping"].values())) == 1


def test_ordinary_aliases_do_not_delete_distinct_updates_but_confirmed_pairs_do():
    first, text = ordinary("first", "papers", group="work")
    second, other_text = ordinary("second", "papers", group="work")
    third, _ = ordinary("copy", "papers", group="work", text=text)
    kept, events, _ = joint([first, second, third], {"first": text, "second": other_text, "copy": text}, {})
    assert not events and {row["id"] for row in kept} == {"copy", "second"}


def test_event_statistics_count_components_not_articles_and_fail_closed():
    streams = []
    for split in prepare.SPLITS:
        value = event(split, split)[0]
        value["metadata"]["audit_component_id"] = "audit:" + split
        streams.append(value)
    stats = prepare.event_statistics(streams, minimum_groups=1)
    assert stats["train"]["groups"] == 1 and stats["train"]["articles"] == 2
    assert stats["train"]["loss_tokens"] == 129
    assert stats["train"]["padding_tokens"] == 1024 - 257
    with pytest.raises(ValueError, match="news_event/train"):
        prepare.event_statistics(streams, minimum_groups=64)


def test_event_capacity_uses_real_later_targets_and_equal_bucket_reuse():
    ordinary_stats = {"sources": {source: {"train": {"raw_tokens": 100000000,
        "window_loss_tokens": {str(n): 100000000 for n in counts},
        "unused_tail_tokens": {str(n): 0 for n in counts}, "eligible_groups": {str(n): 64 for n in counts}}}
        for source, counts in mixed.DEFAULT_LENGTHS.items()}}
    event_stats = {split: {"raw_tokens": 2000000, "loss_tokens": 1000000, "warmup_tokens": 1000000,
        "padding_tokens": 500000, "buckets": {"4": {"groups": 64, "loss_tokens": 250000},
                                                   "8": {"groups": 64, "loss_tokens": 750000}}} for split in prepare.SPLITS}
    report = prepare.capacity_report(ordinary_stats, event_stats, 10000000)
    assert report["conditional_length_weights"]["news_event"] == {"4": .25, "8": .75}
    assert {cell["expected_cell_passes"] for cell in report["sources"]["news_event"]["cells"].values()} == {.5}
    assert report["passed"]
    too_much = prepare.capacity_report(ordinary_stats, event_stats, 100000000)
    assert not too_much["passed"] and any("news_event/4" in value for value in too_much["violations"])
    reduced = prepare.capacity_report(ordinary_stats, event_stats, 100000000, event_weight=.025)
    assert reduced["passed"]
    assert reduced["training_source_weights"]["news"] == .175
    assert reduced["training_source_weights"]["news_event"] == .025
    assert {cell["expected_cell_passes"] for cell in reduced["sources"]["news_event"]["cells"].values()} == {2.5}
    assert reduced["sources"]["news"]["planned_loss_tokens"] == 17500000


@pytest.mark.parametrize("weight", [0, -.01, .2, float("nan"), float("inf")])
def test_event_mix_rejects_invalid_or_missing_news_shares(weight):
    with pytest.raises(ValueError, match="event weight"):
        prepare.training_weights(weight)


def test_event_donor_coverage_rejects_rare_bucket_and_insufficient_history():
    streams = []
    for split in ("validation", "holdout"):
        for name, lengths in (("short", (128, 128)), ("long", (1024, 1024)), ("rare", (2048, 2048))):
            streams.append(event(name + split, split, lengths)[0])
    report = prepare.donor_coverage([], streams, minimum_groups=1)
    for split in ("validation", "holdout"):
        # Short can use the long donor in bucket4; long cannot use short; rare bucket8 has no donor.
        assert report["splits"][split]["news_event"]["supported_groups"] == 1
        assert len(report["splits"][split]["news_event"]["unsupported_groups"]) == 2


def test_reservation_recovers_news_family_after_mixed_group_rename_and_requires_anchor():
    stream, _, _ = event()
    item, _ = ordinary("renamed", group="mixed-family:relabelled")
    item["metadata"]["family_work_keys"] = ["imported-family:" + stream["group_id"]]
    prepare.reserve_event_families([item], {stream["id"]: stream})
    assert item["audit_only"]
    with pytest.raises(ValueError, match="audit bridge"):
        prepare.reserve_event_families([], {stream["id"]: stream})


def test_old_ready_news_temporal_split_cannot_be_imported():
    row = ordinary("old", split="train")[0]
    row["metadata"].update(publication_date="2018-03-01", publication_date_original="2018-03-01")
    with pytest.raises(ValueError, match="temporal split"):
        prepare.validate_imported_news_dates([row])


def test_identical_full_body_cannot_have_different_pretokenized_content(tmp_path):
    a, text = ordinary("a")
    b, _ = ordinary("b", text=text)
    store = mixed.CandidateStore(tmp_path / "tokens.sqlite")
    try:
        for row, tokens in ((a, [1] * 128), (b, [2] * 128)):
            row["token_count"] = len(tokens)
            store.put(row["id"], text)
            store.set_tokens(row["id"], tokens)
        with pytest.raises(ValueError, match="inconsistent token IDs"):
            prepare.validate_same_text_tokens([a, b], store)
    finally:
        store.close()


def source_fixture(tmp_path, groups=64):
    """Small texts but complete synthetic token spans exercise actual I/O gates."""
    from test_prepare_news_event_streams import make_family, prepare as builder
    tokenizer = {"name": mixed.TOKENIZER, "revision": mixed.TOKENIZER_REVISION, "fingerprint": "sha256:test"}
    event_dir = tmp_path / "events"
    event_dir.mkdir()
    streams, recovered, anchors = [], {}, []
    for split_index, (split, month) in enumerate(zip(prepare.SPLITS, ("2017-03", "2018-03", "2018-06"))):
        for index in range(groups):
            family, members, _, _ = make_family(tuple(f"{month}-{day:02d}" for day in (1, 2, 3)),
                                                 offset=(split_index * groups + index) * 3)
            for member in members[1:]:
                member["token_ids"] = [7] * 128
            stream = builder.export_stream(family, members[1:], tokenizer["fingerprint"], 30)
            streams.append(stream)
            recovered[family["group_id"]] = members
            representative = builder.news.export_document(members[0], tokenizer["fingerprint"])
            anchors.append((representative, members[0]["text"]))
    files = {}
    for name in prepare.EVENT_FILES[:2]:
        path = event_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for stream in streams:
                if (stream["split"] == "holdout") == (name == prepare.EVENT_FILES[1]):
                    handle.write(json.dumps(stream) + "\n")
        info = mixed.file_info(path)
        files[name] = {"bytes": info["size"], "sha256": info["sha256"]}
    files[prepare.EVENT_FILES[2]] = builder.export_event_texts(event_dir / prepare.EVENT_FILES[2], streams, recovered)
    mixed.save_json(event_dir / "event_stream_manifest.json", {"ready": True, "tokenizer": tokenizer, "files": files})
    directories = []
    for source in mixed.DEFAULT_LENGTHS:
        directory = tmp_path / source
        directory.mkdir()
        directories.append(str(directory))
        store = mixed.CandidateStore(directory / "build.sqlite")
        items = []
        for split in prepare.SPLITS:
            for index in range(groups):
                value, text = ordinary(f"{source}-{split}-{index}", source, split)
                value["token_count"] = 2048 if source == "news" else 8192
                store.put(value["id"], text)
                store.set_tokens(value["id"], [4] * value["token_count"])
                items.append(value)
        if source == "news":
            for row, text in anchors:
                value = {"id": row["id"], "document_id": row["id"], "group_id": row["group_id"],
                    "source": "news", "split": row["split"], "token_count": len(row["token_ids"]),
                    "work_keys": ["imported-family:" + row["group_id"]], "metadata": row["metadata"]}
                store.put(value["id"], text)
                store.set_tokens(value["id"], row["token_ids"])
                items.append(value)
        source_files = mixed.export_files(directory, items, store, tokenizer["fingerprint"])
        store.close()
        mixed.save_json(directory / "data_manifest.json", {"ready": True, "tokenizer": tokenizer, "files": source_files})
    return directories, event_dir


def test_complete_finalizer_exports_six_independent_strata_with_registered_capacity(tmp_path):
    directories, event_dir = source_fixture(tmp_path)
    output = tmp_path / "final"
    args = Namespace(input_dirs=directories, event_dir=str(event_dir), output_dir=str(output),
                     minimum_groups=64, planned_loss_tokens=10000, maximum_cell_passes=3, near_threshold=.85, seed=17)
    result = prepare.run(args)
    assert result["ready"] and result["training_budget_capacity"]["passed"]
    assert result["resolved_raw_config_proposal"]["source_length_weights"]["news_event"] == {"4": 1.}
    for source in mixed.DEFAULT_LENGTHS:
        assert all(result["statistics"]["sources"][source][split]["groups"] == 64 for split in prepare.SPLITS)
    assert all(result["event_statistics"][split]["groups"] == 64 for split in prepare.SPLITS)
    from delta_ngram.data.event_streams import load_event_documents
    loaded = load_event_documents(output / prepare.EVENT_FILES[0], tokenizer_identity="sha256:test")
    assert len(loaded) == 128 and {record.split for record in loaded} == {"train", "validation"}
    assert all(record.group_id.startswith("news-family:") and record.metadata["audit_component_id"].startswith("mixed-family:") for record in loaded)
    audit = json.loads((output / "dedup_audit.json").read_text())
    assert audit["all_event_articles_audited"] == 384
    assert len(audit["reserved_single_news"]) == 192
    with pytest.raises(FileExistsError):
        prepare.run(args)


@pytest.mark.parametrize("corruption", ["hash", "missing_text", "not_ready"])
def test_event_import_requires_full_verified_article_sidecar(tmp_path, corruption):
    _, event_dir = source_fixture(tmp_path, groups=1)
    manifest_path = event_dir / "event_stream_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if corruption == "not_ready":
        manifest["ready"] = False
    elif corruption == "hash":
        manifest["files"][prepare.EVENT_FILES[2]]["sha256"] = "0" * 64
    else:
        text_path = event_dir / prepare.EVENT_FILES[2]
        with gzip.open(text_path, "rt", encoding="utf-8") as handle:
            rows = handle.readlines()
        with gzip.open(text_path, "wt", encoding="utf-8") as handle:
            handle.writelines(rows[:-1])
        info = mixed.file_info(text_path)
        manifest["files"][prepare.EVENT_FILES[2]] = {"bytes": info["size"], "sha256": info["sha256"]}
    mixed.save_json(manifest_path, manifest)
    store = mixed.CandidateStore(tmp_path / "bad.sqlite")
    try:
        with pytest.raises(ValueError):
            prepare.import_events(event_dir, store, "sha256:test")
    finally:
        store.close()


def test_finalizer_refuses_weakened_group_gate_before_any_artifact(tmp_path):
    args = Namespace(minimum_groups=63, planned_loss_tokens=100000000, maximum_cell_passes=3)
    with pytest.raises(ValueError, match=">=64"):
        prepare.run(args)

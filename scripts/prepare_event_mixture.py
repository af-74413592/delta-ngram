#!/usr/bin/env python
"""Audit five complete-document sources together with every event article.

All imported splits are immutable. Cross-split components are quarantined in
full. Event articles are audit nodes, never independent training examples:
selection keeps one original complete stream or drops that whole stream.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import importlib.util
import json
from pathlib import Path
import sys


def sibling(name):
    spec = importlib.util.spec_from_file_location("event_mixture_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mixed = sibling("prepare_mixed_corpus")
SPLITS = ("train", "validation", "holdout")
WEIGHTS = {"code": .20, "papers": .25, "books": .20, "web": .15, "news": .15, "news_event": .05}
PRIORITY = {source: rank for rank, source in enumerate(("code", "papers", "books", "news_event", "news", "web"))}
BUCKETS = (4, 8, 16, 32)
EVENT_FILES = ("train_validation_event_streams.jsonl", "holdout_event_streams.jsonl", "selected_event_texts.jsonl.gz")


def verified_file(directory, manifest, name):
    expected = manifest.get("files", {}).get(name, {})
    actual = mixed.file_info(directory / name)
    if actual["sha256"] != expected.get("sha256") or actual["size"] != expected.get("bytes", expected.get("size")):
        raise ValueError(f"event file integrity mismatch: {directory / name}")
    return actual


def original_news_groups(item):
    """Recover the inherited news family even after a prior mixed import."""
    if item["source"] != "news":
        return set()
    meta = item["metadata"]
    groups = {value for value in (item.get("group_id"), meta.get("original_group_id"))
              if isinstance(value, str) and value.startswith("news-family:")}
    members = meta.get("family_members")
    if isinstance(members, list) and members and all(isinstance(x, str) for x in members):
        groups.add("news-family:" + mixed.digest(sorted(members)))
    for alias in (*item["work_keys"], *meta.get("family_work_keys", [])):
        if alias.startswith("imported-family:news-family:"):
            groups.add(alias.removeprefix("imported-family:"))
    return groups


def validate_imported_news_dates(items):
    """An old ready manifest cannot silently retain the superseded date split."""
    from delta_ngram.data.event_streams import publication_time, date_split
    for item in items:
        if item["source"] != "news":
            continue
        meta = item["metadata"]
        day, _ = publication_time(meta.get("publication_date_original"))
        if meta.get("publication_date") != day.isoformat() or date_split(day) != item["split"]:
            raise ValueError("single-news publication date contradicts registered temporal split")
        for key in ("family_date_min", "family_date_max"):
            if key in meta and date_split(publication_time(meta[key])[0]) != item["split"]:
                raise ValueError("single-news full family crosses registered temporal split")


def validate_same_text_tokens(items, store):
    """An identical full body and tokenizer must yield identical full tokens."""
    owners = {}
    for item in items:
        previous = owners.setdefault(item["metadata"]["text_sha256"], item)
        if previous is item:
            continue
        if previous["token_count"] != item["token_count"]:
            raise ValueError("identical full text has different token counts under the same tokenizer")
        def token_hash(row):
            return row["metadata"].get("token_sha256") or mixed.digest(store.tokens(row["id"]))
        if token_hash(previous) != token_hash(item):
            raise ValueError("identical full text has inconsistent token IDs under the same tokenizer")


def import_events(directory, store, fingerprint, existing_ids=()):
    """Verified event tokens/texts enter the disk store one article at a time."""
    from delta_ngram.data.event_streams import iter_event_documents
    directory = Path(directory).resolve()
    manifest_path = directory / "event_stream_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer = manifest.get("tokenizer", {})
    if manifest.get("ready") is not True:
        raise ValueError("event corpus is not structurally ready")
    if (tokenizer.get("fingerprint") != fingerprint or tokenizer.get("name") != mixed.TOKENIZER
            or tokenizer.get("revision") != mixed.TOKENIZER_REVISION):
        raise ValueError("event tokenizer differs from pinned single-document sources")
    for name in EVENT_FILES:
        verified_file(directory, manifest, name)
    streams, items, by_article, groups = {}, [], {}, set()
    used_ids = set(existing_ids)
    member_owners = {}
    for name in EVENT_FILES[:2]:
        for document in iter_event_documents(directory / name, tokenizer_identity=fingerprint, allow_holdout=True):
            if (name == EVENT_FILES[1]) != (document.split == "holdout"):
                raise ValueError("event holdout file/split mismatch")
            if document.id in streams or document.group_id in groups:
                raise ValueError("duplicate imported event ID/family")
            groups.add(document.group_id)
            for member in document.metadata["family_members"]:
                if member_owners.setdefault(member, document.group_id) != document.group_id:
                    raise ValueError("event original family membership overlaps")
            record = {"id": document.id, "group_id": document.group_id, "source": document.source,
                      "split": document.split, "tokenizer_identity": document.tokenizer_identity,
                      "metadata": dict(document.metadata), "articles": []}
            for article in document.articles:
                if article.id in by_article:
                    raise ValueError("article is reused by distinct imported events")
                audit_id = "event-article:" + mixed.digest([document.id, article.id])
                if audit_id in used_ids:
                    raise ValueError("event audit ID overlaps imported document ID")
                used_ids.add(audit_id)
                meta = {**article.metadata, "original_document_id": article.id,
                        "original_group_id": document.group_id, "event_stream_id": document.id}
                aliases = ["imported-family:" + document.group_id, "news-article:" + article.id]
                for key in ("canonical_url", "url"):
                    if meta.get(key):
                        aliases.append("source-url:" + mixed.canonical_url(meta[key]))
                item = {"id": audit_id, "document_id": article.id, "source": "news_event",
                        "split": document.split, "group_id": document.group_id,
                        "token_count": len(article.token_ids), "work_keys": sorted(set(aliases)), "metadata": meta}
                store.put(audit_id, "")
                store.set_tokens(audit_id, list(article.token_ids))
                items.append(item)
                by_article[article.id] = item
                record["articles"].append({"id": article.id, "audit_id": audit_id,
                                           "token_count": len(article.token_ids), "metadata": dict(article.metadata)})
            streams[document.id] = record
    found = set()
    with gzip.open(directory / EVENT_FILES[2], "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            article_id = row.get("id", row.get("article_id"))
            if article_id not in by_article or article_id in found:
                raise ValueError("unexpected or duplicate event article text")
            item = by_article[article_id]
            text = row["text"]
            if (mixed.text_hash(text) != item["metadata"]["text_sha256"]
                    or mixed.text_hash(mixed.normalize(text)) != item["metadata"]["normalized_text_sha256"]):
                raise ValueError("event full-article text hash mismatch")
            store.update_text(item["id"], text)
            found.add(article_id)
    if found != set(by_article):
        raise ValueError("missing complete event article texts")
    store.db.commit()
    return streams, items, {"directory": str(directory), "manifest_sha256": mixed.file_info(manifest_path)["sha256"],
                            "files": manifest["files"], "streams": len(streams), "articles": len(items)}


def reserve_event_families(items, streams):
    """Keep original representatives as audit bridges, but never train on them."""
    reserved = {stream["group_id"]: stream["split"] for stream in streams.values()}
    seen, removed = set(), []
    for item in items:
        groups = original_news_groups(item)
        item["work_keys"] = sorted(set(item["work_keys"]) | {"imported-family:" + group for group in groups})
        if item["source"] == "news":
            item["work_keys"] = sorted(set(item["work_keys"]) | {"news-article:" + item["id"],
                "news-article:" + item["metadata"].get("original_document_id", item["id"])})
        matches = groups & reserved.keys()
        if matches:
            if any(reserved[group] != item["split"] for group in matches):
                raise ValueError("event family/source representative split mismatch")
            item["audit_only"] = True
            seen.update(matches)
            removed.append({"id": item["id"], "source": "news", "split": item["split"],
                            "original_event_groups": sorted(matches), "reason": "event_family_reserved"})
    if seen != reserved.keys():
        raise ValueError("some event families lack their original single-news representative audit bridge")
    return removed


def select_joint_components(items, edges, streams, *, seed=17):
    """Quarantine first, then choose a source and preserve complete streams."""
    components = defaultdict(list)
    for index, item in enumerate(items):
        components[item["group_id"]].append(index)
    duplicates = mixed.Families(len(items))
    for left, right, _ in edges:
        duplicates.join(left, right)
    kept_indices, kept_streams, removed_streams, removed_documents = set(), [], [], []
    mapping, quarantined = {}, []
    for component, indices in sorted(components.items()):
        splits = {items[index]["split"] for index in indices}
        event_ids = {items[index]["metadata"]["event_stream_id"] for index in indices
                     if items[index]["source"] == "news_event"}
        if len(splits) > 1:
            quarantined.append(component)
            reason, chosen = "family_crosses_imported_splits", None
        else:
            sources = {items[index]["source"] for index in indices if not items[index].get("audit_only")}
            chosen = min(sources, key=lambda source: PRIORITY[source]) if sources else None
            reason = "cross_source_family"
        selected_event = None
        if chosen == "news_event":
            selected_event = min(event_ids, key=lambda key: (mixed.priority(seed, "event-family-selection", key), key))
            stream = streams[selected_event]
            expected_ids = {article["audit_id"] for article in stream["articles"]}
            actual_ids = {items[index]["id"] for index in indices
                          if items[index]["source"] == "news_event" and items[index]["metadata"]["event_stream_id"] == selected_event}
            if expected_ids != actual_ids:
                raise ValueError("event family component is missing a whole article")
            stream["metadata"]["audit_component_id"] = component
            stream["metadata"]["audit_component_event_groups"] = sorted(streams[key]["group_id"] for key in event_ids)
            kept_streams.append(stream)
        elif chosen is not None:
            by_duplicate = defaultdict(list)
            for index in indices:
                if items[index]["source"] == chosen and not items[index].get("audit_only"):
                    by_duplicate[duplicates.find(index)].append(index)
            for choices in by_duplicate.values():
                kept_indices.add(min(choices, key=lambda index: items[index]["id"]))
        for event_id in sorted(event_ids):
            mapping[event_id] = component
            if event_id != selected_event:
                removed_streams.append({"id": event_id, "group_id": streams[event_id]["group_id"],
                    "split": streams[event_id]["split"], "audit_component_id": component,
                    "article_ids": [a["id"] for a in streams[event_id]["articles"]],
                    "reason": "joined_event_family" if chosen == "news_event" else reason})
        for index in indices:
            item = items[index]
            if item["source"] == "news_event":
                continue
            if index in kept_indices:
                item["metadata"]["audit_component_id"] = component
            else:
                why = ("family_crosses_imported_splits" if len(splits) > 1 else
                       "event_family_reserved" if item.get("audit_only") else
                       "confirmed_duplicate" if item["source"] == chosen else "cross_source_family")
                removed_documents.append({"id": item["id"], "source": item["source"], "split": item["split"],
                                          "audit_component_id": component, "reason": why})
    kept = [item for index, item in enumerate(items) if index in kept_indices]
    return kept, kept_streams, {"quarantined_components": quarantined, "removed_event_streams": removed_streams,
        "removed_documents": removed_documents, "event_component_mapping": mapping,
        "removed_by_source_reason": dict(Counter(f"{row['source']}/{row['reason']}" for row in removed_documents)),
        "source_priority": list(PRIORITY), "event_selection": "one unchanged original stream per joint component"}


def event_statistics(streams, minimum_groups=64):
    result = {split: {"documents": 0, "groups": 0, "articles": 0, "raw_tokens": 0, "loss_tokens": 0,
        "warmup_tokens": 0, "actual_chunks": 0, "padding_tokens": 0, "buckets": {}} for split in SPLITS}
    groups, components, article_ids = set(), set(), set()
    for stream in streams:
        if stream["group_id"] in groups or stream["metadata"]["audit_component_id"] in components:
            raise ValueError("selected event families are not independent joint components")
        groups.add(stream["group_id"])
        components.add(stream["metadata"]["audit_component_id"])
        lengths = [article["token_count"] for article in stream["articles"]]
        raw, targets = sum(lengths), sum(lengths[1:])
        chunks = sum((count + 511) // 512 for count in lengths)
        if not 2 <= len(lengths) <= 4 or any(not 128 <= count <= 16384 for count in lengths) or raw > 16384 or chunks > 32:
            raise ValueError("selected event violates complete-article/runtime bounds")
        bucket = str(next(count for count in BUCKETS if count >= chunks))
        cell = result[stream["split"]]
        cell["documents"] += 1
        cell["groups"] += 1
        cell["articles"] += len(lengths)
        cell["raw_tokens"] += raw
        cell["loss_tokens"] += targets
        cell["warmup_tokens"] += lengths[0]
        cell["actual_chunks"] += chunks
        cell["padding_tokens"] += chunks * 512 - raw
        slot = cell["buckets"].setdefault(bucket, {"groups": 0, "raw_tokens": 0, "loss_tokens": 0})
        slot["groups"] += 1
        slot["raw_tokens"] += raw
        slot["loss_tokens"] += targets
        for article in stream["articles"]:
            if article["id"] in article_ids:
                raise ValueError("article reused across selected event streams")
            article_ids.add(article["id"])
    for split, cell in result.items():
        if cell["groups"] < minimum_groups:
            raise ValueError(f"insufficient independent groups: news_event/{split} ({cell['groups']} < {minimum_groups})")
    return result


def audit_selected(items, streams, audit_items):
    selected = list(items)
    by_id = {item["id"]: item for item in audit_items}
    for stream in streams:
        selected.extend(by_id[article["audit_id"]] for article in stream["articles"])
    owners, sources = {}, {}
    for item in selected:
        component = item["group_id"]
        if sources.setdefault(component, item["source"]) != item["source"]:
            raise ValueError("selected joint component crosses training strata")
        values = ["id:" + item["id"], "family:" + component,
                  "raw:" + item["metadata"]["text_sha256"], "normal:" + item["metadata"]["normalized_text_sha256"],
                  *item["work_keys"]]
        for value in values:
            if owners.setdefault(value, item["split"]) != item["split"]:
                raise ValueError("selected article/work/text identity crosses splits")


def capacity_report(ordinary, events, planned_loss_tokens, maximum_passes=3):
    """Event bucket shares use unique later-article targets, never padded size."""
    ordinary_weights = {source: WEIGHTS[source] for source in mixed.DEFAULT_LENGTHS}
    conditional = {source: ({"2": .9, "4": .1} if source == "news" else {str(n): 1 / len(counts) for n in counts})
                   for source, counts in mixed.DEFAULT_LENGTHS.items()}
    try:
        report = mixed.budget_capacity(ordinary, ordinary_weights, mixed.DEFAULT_LENGTHS, conditional,
                                       planned_loss_tokens, maximum_passes)
    except mixed.CapacityError as error:
        report = error.report
    total = events["train"]["loss_tokens"]
    active = events["train"]["buckets"]
    conditional["news_event"] = {key: cell["loss_tokens"] / total for key, cell in sorted(active.items(), key=lambda pair: int(pair[0]))} if total else {}
    cells = {}
    for key, fraction in conditional["news_event"].items():
        available = active[key]["loss_tokens"]
        target = planned_loss_tokens * WEIGHTS["news_event"] * fraction
        passes = target / available
        cells[key] = {"planned_loss_tokens": target, "available_loss_tokens_per_pass": available,
                      "expected_cell_passes": passes, "independent_train_groups": active[key]["groups"]}
        if passes > maximum_passes:
            report["violations"].append(f"news_event/{key}: {passes:.6f} passes exceeds {maximum_passes}")
    if not total:
        report["violations"].append("news_event: no training targets")
    for split in ("validation", "holdout"):
        missing = set(events[split]["buckets"]) - set(active)
        if missing:
            report["violations"].append(f"news_event/{split}: canonical buckets absent from training: {sorted(missing)}")
    report["sources"]["news_event"] = {"raw_tokens": events["train"]["raw_tokens"],
        "planned_loss_tokens": planned_loss_tokens * WEIGHTS["news_event"], "cells": cells,
        "unique_later_article_targets": total, "warmup_tokens_per_pass": events["train"]["warmup_tokens"],
        "padding_tokens_per_pass": events["train"]["padding_tokens"],
        "overlap_note": "Each stream belongs to one bucket; later real tokens are targets once per pass. Warmup and padding are not targets."}
    report["conditional_length_weights"] = conditional
    report["training_source_weights"] = WEIGHTS
    report["event_weight_basis"] = "final train bucket later-article targets / all final train later-article targets; equal expected reuse"
    report["passed"] = not report["violations"]
    return report


def donor_coverage(items, streams, minimum_groups=64):
    """Check frozen-control eligibility using shapes, without loading any model."""
    coverage, violations = {}, []
    for split in ("validation", "holdout"):
        coverage[split] = {}
        for source, counts in mixed.DEFAULT_LENGTHS.items():
            groups_by_bucket = {count: {row["group_id"] for row in items if row["source"] == source
                and row["split"] == split and row["token_count"] >= count * 512} for count in counts}
            supported = set().union(*(groups for groups in groups_by_bucket.values() if len(groups) >= 2))
            all_groups = {row["group_id"] for row in items if row["source"] == source and row["split"] == split}
            coverage[split][source] = {"available_groups": len(all_groups), "supported_groups": len(supported),
                                      "unsupported_groups": sorted(all_groups - supported)}
        by_bucket = defaultdict(list)
        for stream in streams:
            if stream["split"] != split:
                continue
            lengths = [a["token_count"] for a in stream["articles"]]
            chunks = sum((count + 511) // 512 for count in lengths)
            bucket = next(count for count in BUCKETS if count >= chunks)
            raw = sum(lengths)
            required = raw - ((lengths[-1] - 1) % 512 + 1)
            by_bucket[bucket].append((raw, stream["group_id"], required))
        supported, unsupported = [], []
        for values in by_bucket.values():
            largest = sorted(values, reverse=True)[:2]
            for _, group, required in values:
                candidates = [raw for raw, other, _ in largest if other != group]
                (supported if candidates and max(candidates) >= required else unsupported).append(group)
        coverage[split]["news_event"] = {"available_groups": len(supported) + len(unsupported),
            "supported_groups": len(supported), "unsupported_groups": sorted(unsupported)}
        for source, cell in coverage[split].items():
            if cell["supported_groups"] < minimum_groups:
                violations.append(f"{source}/{split}: {cell['supported_groups']} donor-supported groups < {minimum_groups}")
    return {"minimum_groups": minimum_groups, "splits": coverage, "violations": violations,
            "passed": not violations, "basis": "same source/bucket, different joint family; event donor real tokens cover all target history writes"}


def export_events(output, streams, store):
    temporary = {name: output / (name + ".tmp") for name in EVENT_FILES}
    with temporary[EVENT_FILES[0]].open("w", encoding="utf-8", newline="\n") as train, \
         temporary[EVENT_FILES[1]].open("w", encoding="utf-8", newline="\n") as holdout, \
         temporary[EVENT_FILES[2]].open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as texts:
        for stream in sorted(streams, key=lambda row: (row["split"], row["group_id"], row["id"])):
            articles = []
            for supplied in stream["articles"]:
                metadata = {**supplied["metadata"], "audit_component_id": stream["metadata"]["audit_component_id"]}
                articles.append({"id": supplied["id"], "token_ids": store.tokens(supplied["audit_id"]), "metadata": metadata})
                texts.write((json.dumps({"id": supplied["id"], "text": store.text(supplied["audit_id"]),
                                         "metadata": metadata}, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
            record = {**stream, "articles": articles}
            (holdout if stream["split"] == "holdout" else train).write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    for name, path in temporary.items():
        path.replace(output / name)
    files = {}
    for name in EVENT_FILES:
        info = mixed.file_info(output / name)
        files[name] = {"bytes": info["size"], "sha256": info["sha256"]}
    return files


def run(args):
    if args.minimum_groups < 64 or args.planned_loss_tokens < 1 or not 1 <= args.maximum_cell_passes <= 3:
        raise ValueError("requires >=64 groups per stratum/split, positive budget and at most three passes")
    if not 0 < args.near_threshold <= 1:
        raise ValueError("near threshold must lie in (0,1]")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ("data_manifest.json", "event_stream_manifest.json", "candidates.sqlite", "train_validation.jsonl",
             "holdout.jsonl", "selected_texts.jsonl.gz", "dedup_audit.json", "training_capacity_report.json",
             "resolved_raw_config_proposal.json", "retention_report.json", "failure_manifest.json",
             "donor_coverage_report.json", *EVENT_FILES)
    if any((output / name).exists() for name in names):
        raise FileExistsError("refusing to replace an existing event mixture")
    store = mixed.CandidateStore(output / "candidates.sqlite")
    try:
        items, imports, ids, tokenizer = [], [], set(), None
        for directory in args.input_dirs:
            imported, identity, provenance = mixed.import_prepared(directory, store, tokenizer["fingerprint"] if tokenizer else None)
            new_ids = {item["id"] for item in imported}
            if ids & new_ids:
                raise ValueError("document IDs overlap across source corpora")
            ids.update(new_ids)
            items.extend(imported)
            imports.append(provenance)
            tokenizer = identity
        if {item["source"] for item in items} != set(mixed.DEFAULT_LENGTHS):
            raise ValueError("final mixture requires exactly code, papers, books, web and single news")
        validate_imported_news_dates(items)
        streams, event_items, event_import = import_events(args.event_dir, store, tokenizer["fingerprint"], ids)
        reserved = reserve_event_families(items, streams)
        audit_items = items + event_items
        validate_same_text_tokens(audit_items, store)
        families, edges = mixed.form_families(audit_items, store.text, args.near_threshold)
        kept, selected_events, audit = select_joint_components(audit_items, edges, streams, seed=args.seed)
        audit.update(statistics=families, reserved_single_news=reserved,
                     all_event_articles_audited=len(event_items), original_representatives_kept_as_audit_bridges=True)
        mixed.save_json(output / "dedup_audit.json", audit)
        audit_selected(kept, selected_events, audit_items)
        retained_groups = {source: {split: len({item["group_id"] for item in kept
            if item["source"] == source and item["split"] == split}) for split in SPLITS} for source in mixed.DEFAULT_LENGTHS}
        retained_groups["news_event"] = {split: len({row["metadata"]["audit_component_id"] for row in selected_events
                                                    if row["split"] == split}) for split in SPLITS}
        mixed.save_json(output / "retention_report.json", {"minimum_groups": args.minimum_groups,
            "independent_groups_by_source_split": retained_groups,
            "violations": [f"{source}/{split}: {count} groups < {args.minimum_groups}"
                           for source, splits in retained_groups.items() for split, count in splits.items() if count < args.minimum_groups],
            "original_event_streams": len(streams), "selected_complete_event_streams": len(selected_events),
            "all_event_articles_audited": len(event_items)})
        statistics = mixed.validate_and_summarize(kept, mixed.DEFAULT_LENGTHS, args.minimum_groups)
        event_stats = event_statistics(selected_events, args.minimum_groups)
        capacity = capacity_report(statistics, event_stats, args.planned_loss_tokens, args.maximum_cell_passes)
        mixed.save_json(output / "training_capacity_report.json", capacity)
        if not capacity["passed"]:
            raise mixed.CapacityError(capacity)
        donors = donor_coverage(kept, selected_events, args.minimum_groups)
        mixed.save_json(output / "donor_coverage_report.json", donors)
        if not donors["passed"]:
            raise ValueError("fixed donor coverage gate failed: " + "; ".join(donors["violations"]))
        ordinary_files = mixed.export_files(output, kept, store, tokenizer["fingerprint"])
        event_files = export_events(output, selected_events, store)
        event_manifest = {"format": "delta_ngram_news_event_streams_v1", "ready": True, "tokenizer": tokenizer,
            "files": event_files, "statistics": {"splits": event_stats}, "joint_audit": "dedup_audit.json",
            "training_budget_capacity": capacity["sources"]["news_event"], "source_import": event_import,
            "limitations": ["Automatic lexical/news families are not human-verified semantic event labels.",
                            "Near duplicate retrieval is approximate; inherited dates are source metadata."]}
        mixed.save_json(output / "event_stream_manifest.json", event_manifest)
        proposal = {"dataset": str(output / "train_validation.jsonl"),
            "event_dataset": str(output / EVENT_FILES[0]), "block_size": 512, "block_counts": [2, 4, 8, 16, 32],
            "source_weights": WEIGHTS, "source_length_weights": capacity["conditional_length_weights"],
            "eval_min_groups_per_source": args.minimum_groups, "eval_donor": True,
            "macro_source_groups": {source: "news" if source == "news_event" else source for source in WEIGHTS},
            "loss_tokens": args.planned_loss_tokens}
        mixed.save_json(output / "resolved_raw_config_proposal.json", proposal)
        manifest = {"format": "delta_ngram_event_mixture_v1", "ready": True, "tokenizer": tokenizer,
            "imports": imports, "event_import": event_import, "statistics": statistics,
            "event_statistics": event_stats, "files": {**ordinary_files, **event_files},
            "training_source_weights": WEIGHTS, "source_block_counts": {**mixed.DEFAULT_LENGTHS, "news_event": list(map(int, capacity["conditional_length_weights"]["news_event"]))},
            "training_budget_capacity": capacity, "donor_coverage": donors, "resolved_raw_config_proposal": proposal,
            "selected_date_bounds": mixed.selected_date_bounds(kept), "deduplication": {**families,
                "threshold": args.near_threshold, "quarantined_components": len(audit["quarantined_components"]),
                "dropped_complete_events": len(audit["removed_event_streams"]), "source_priority": list(PRIORITY)},
            "quality_checks": {"all_event_articles_audited": True, "original_representative_bridges_retained": True,
                "cross_split_components_quarantined_before_selection": True, "event_updates_never_deduplicated_inside_selected_stream": True,
                "one_original_event_per_joint_component": True, "minimum_independent_groups_per_stratum_split": args.minimum_groups,
                "complete_articles_and_documents": True, "holdout_excluded_from_training_files": True},
            "source_script_sha256": mixed.file_info(__file__)["sha256"],
            "dedup_primitives_script_sha256": mixed.file_info(Path(__file__).with_name("prepare_mixed_corpus.py"))["sha256"],
            "limitations": ["LSH retrieval is approximate; no guarantee of semantic event completeness.",
                "Ordinary length-cell capacities overlap and are not additive unique data.",
                "Event bucket shares are registered from final training data availability, before model evaluation."]}
        mixed.save_json(output / "data_manifest.json", manifest)
        mixed.log("event_mixture_complete", output=str(output), ordinary_documents=len(kept), event_streams=len(selected_events))
        return manifest
    except Exception as error:
        mixed.save_json(output / "failure_manifest.json", {"ready": False, "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        store.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-dirs", nargs="+", required=True)
    result.add_argument("--event-dir", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--minimum-groups", type=int, default=64)
    result.add_argument("--planned-loss-tokens", type=int, default=100_000_000)
    result.add_argument("--maximum-cell-passes", type=float, default=3.0)
    result.add_argument("--near-threshold", type=float, default=.85)
    result.add_argument("--seed", type=int, default=17)
    return result


if __name__ == "__main__":
    run(parser().parse_args())

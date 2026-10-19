#!/usr/bin/env python
"""Recover chronological, whole-article news streams from existing CC-News families.

This does not rerun approximate deduplication, infer semantic event labels, or
concatenate articles into fake documents. It consumes a completed old-news build.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

_SPEC = importlib.util.spec_from_file_location("event_stream_news_source", Path(__file__).with_name("prepare_news_corpus.py"))
news = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = news
_SPEC.loader.exec_module(news)

FORMAT = "delta_ngram_news_event_streams_v1"
BLOCK_BUCKETS = (4, 8, 16, 32)
ORDERING = ("Strictly increasing explicit source calendar dates, except that two timezone-aware "
            "timestamps use strictly increasing UTC; same-day pairs require both timestamps "
            "to be timezone-aware. Source timestamps are extractor metadata, not publisher-verified.")


def source_time(value):
    day = news.publication_day(value)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    aware = parsed.tzinfo is not None and parsed.utcoffset() is not None
    return {"day": day, "datetime": parsed, "aware": aware,
            "utc": parsed.astimezone(timezone.utc) if aware else None,
            "precision": "day" if len(value) == 10 else "subsecond" if "." in value else "second"}


def strictly_before(first, second):
    left, right = source_time(first["publication_date_original"]), source_time(second["publication_date_original"])
    if left["aware"] and right["aware"]:
        return left["utc"] < right["utc"]
    return left["day"] < right["day"]


def within_span(first, last, maximum_days):
    left, right = source_time(first["publication_date_original"]), source_time(last["publication_date_original"])
    if abs((right["day"] - left["day"]).days) > maximum_days:
        return False
    if left["aware"] and right["aware"]:
        return 0 < (right["utc"] - left["utc"]).total_seconds() <= maximum_days * 86400
    return left["day"] < right["day"]


def ordered(items):
    if items and all(source_time(item["publication_date_original"])["aware"] for item in items):
        return sorted(items, key=lambda item: (source_time(item["publication_date_original"])["utc"], item["document_id"]))
    # A stable traversal is not itself evidence of order: every selected pair
    # must separately satisfy strictly_before. Naive intraday clocks never order.
    return sorted(items, key=lambda item: (item["publication_date"],
                  source_time(item["publication_date_original"])["utc"].isoformat()
                  if source_time(item["publication_date_original"])["aware"] else "", item["document_id"]))


def read_selected_families(path, source_manifest):
    if not source_manifest.get("ready") or source_manifest.get("format") != "delta_ngram_news_corpus_v1":
        raise ValueError("source corpus must have its completed, ready old-news manifest")
    if source_manifest.get("dataset_id") != news.DATASET or source_manifest.get("dataset_revision") != news.REVISION:
        raise ValueError("source corpus dataset revision mismatch")
    if source_manifest.get("temporal_splits") != news.TEMPORAL_SPLITS:
        raise ValueError("source corpus temporal splits differ from the registered 2017/2018 windows")
    families, member_owner, representatives = {}, {}, {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record["metadata"]
            members = metadata.get("family_members")
            if (not isinstance(members, list) or not members or
                    any(not isinstance(value, str) or not value.startswith("cc-news:") for value in members) or
                    len(members) != len(set(members)) or record["id"] not in members):
                raise ValueError(f"invalid family membership at selected line {line_number}")
            members = sorted(members)
            group = "news-family:" + news.digest(members)
            if group in families:
                raise ValueError("source selection contains more than one representative per family")
            if metadata.get("dataset_revision") != news.REVISION or metadata.get("dataset_id") != news.DATASET:
                raise ValueError("selected article dataset revision mismatch")
            if news.text_hash(record["text"]) != metadata.get("text_sha256"):
                raise ValueError("selected representative text checksum mismatch")
            if news.text_hash(news.normalize(record["text"])) != metadata.get("normalized_text_sha256"):
                raise ValueError("selected representative normalized checksum mismatch")
            split = news.temporal_split(news.publication_day(metadata["publication_date_original"]))
            if metadata.get("publication_date") != metadata["publication_date_original"][:10]:
                raise ValueError("selected representative date mismatch")
            family = {"group_id": group, "split": split, "members": members,
                      "family_date_min": metadata["family_date_min"], "family_date_max": metadata["family_date_max"],
                      "representative_id": record["id"], "representative_metadata": metadata}
            families[group] = family
            representatives[record["id"]] = metadata
            for member in members:
                if member in member_owner:
                    raise ValueError("family member belongs to more than one selected family")
                member_owner[member] = group
    if not families:
        raise ValueError("empty source selection")
    return families, member_owner, representatives


def verified_source_rows(download_manifest_path, maximum_bytes):
    import pyarrow.parquet as pq
    path = Path(download_manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = news.validate_download_manifest(manifest, maximum_bytes)
    for entry in entries:
        shard_path = path.parent / "raw" / entry["path"]
        if shard_path.stat().st_size != entry["size"] or news.file_sha256(shard_path) != entry["sha256"]:
            raise ValueError(f"local shard integrity mismatch: {entry['path']}")
        row = 0
        for batch in pq.ParquetFile(shard_path).iter_batches(batch_size=2048):
            for record in batch.to_pylist():
                row += 1
                yield entry, row, record
        news.log("event_stream_shard_scanned", path=entry["path"], rows=row)


def recover_members(rows, families, member_owner, representatives, *, minimum_chars=300, maximum_chars=100000):
    recovered, found, stats = defaultdict(list), set(), Counter()
    for shard, row, record in rows:
        stats["scanned_records"] += 1
        origin = {"source_shard": shard["path"], "source_row": row, "source_shard_sha256": shard["sha256"]}
        document_id = "cc-news:" + news.digest([news.REVISION, origin])
        if document_id not in member_owner:
            continue
        if document_id in found:
            raise ValueError("requested member encountered more than once")
        found.add(document_id)
        item = news.candidate_record(record, shard, row, minimum_chars=minimum_chars, maximum_chars=maximum_chars)
        item["text_sha256"] = news.text_hash(item["text"])
        item["normalized_text_sha256"] = news.text_hash(news.normalize(item["text"]))
        group = member_owner[document_id]
        item["group_id"] = group
        if item["split"] != families[group]["split"]:
            raise ValueError("recovered family crosses source time partitions")
        if document_id in representatives:
            expected = representatives[document_id]
            for key in ("source_shard", "source_row", "source_shard_sha256", "canonical_url", "text_sha256",
                        "normalized_text_sha256", "publication_date_original"):
                if item[key] != expected[key]:
                    raise ValueError(f"recovered representative {key} mismatch")
        recovered[group].append(item)
    missing = set(member_owner) - found
    if missing:
        raise ValueError(f"failed to recover {len(missing)} requested source members")
    owners = {key: {} for key in ("canonical_url", "normalized_text_sha256")}
    for group, family in families.items():
        items = recovered[group]
        dates = sorted(item["publication_date"] for item in items)
        if dates[0] != family["family_date_min"] or dates[-1] != family["family_date_max"]:
            raise ValueError("recovered family date range mismatch")
        for item in items:
            for key, seen in owners.items():
                previous = seen.setdefault(item[key], (group, item["split"]))
                if previous != (group, item["split"]):
                    raise ValueError(f"recovered {key} shared across different families")
    stats["recovered_documents"] = len(found)
    stats["recovered_families"] = len(recovered)
    stats["multi_member_families"] = sum(len(items) >= 2 for items in recovered.values())
    return recovered, dict(stats)


def deduplicate_exact(items):
    retained, seen = [], set()
    for item in ordered(items):
        key = item["normalized_text_sha256"]
        if key not in seen:
            retained.append(item)
            seen.add(key)
    return retained, len(items) - len(retained)


def chain_cost(chain):
    return {"raw_tokens": sum(len(item["token_ids"]) for item in chain),
            "article_chunk_count": sum((len(item["token_ids"]) + 511) // 512 for item in chain),
            "target_tokens": sum(len(item["token_ids"]) for item in chain[1:])}


def choose_chain(items, *, maximum_articles=4, maximum_days=30, seed=17,
                 maximum_stream_tokens=None, maximum_stream_chunks=None):
    """Preregistered deterministic earliest-feasible traversal from each start.

    Prefer more articles, then seeded chain hash. Not a claim of a globally
    maximal chain in mixed/unknown timezone data; every pair is validated.
    """
    pool, best, best_rank = ordered(items), [], None
    for index, first in enumerate(pool):
        chain = [first]
        for following in pool[index + 1:]:
            if all(within_span(previous, following, maximum_days) and strictly_before(previous, following)
                   for previous in chain):
                if maximum_stream_tokens is not None or maximum_stream_chunks is not None:
                    cost = chain_cost(chain + [following])
                    if ((maximum_stream_tokens is not None and cost["raw_tokens"] > maximum_stream_tokens) or
                            (maximum_stream_chunks is not None and cost["article_chunk_count"] > maximum_stream_chunks)):
                        continue
                chain.append(following)
                if len(chain) == maximum_articles:
                    break
        rank = (-len(chain), news.priority(seed, [item["document_id"] for item in chain]))
        if best_rank is None or rank < best_rank:
            best, best_rank = chain, rank
    return best if len(best) >= 2 else []


def export_stream(family, chain, fingerprint, maximum_days):
    articles, boundaries, offset = [], [], 0
    for index, item in enumerate(chain):
        timing = source_time(item["publication_date_original"])
        metadata = {**item["upstream_metadata"], "dataset_id": news.DATASET, "dataset_revision": news.REVISION,
                    **{key: item[key] for key in ("publication_date", "publication_date_original", "canonical_url", "source_shard",
                        "source_row", "source_shard_sha256", "text_sha256", "normalized_text_sha256")},
                    "publication_date_basis": "upstream date field; extractor metadata, not publisher verified",
                    "timestamp_precision": timing["precision"], "timezone_status": "explicit" if timing["aware"] else "unspecified",
                    "publication_timestamp_utc": timing["utc"].isoformat() if timing["aware"] else None,
                    "token_sha256": news.digest(item["token_ids"]), "license": "unknown",
                    "complete_extracted_article": True, "truncated": False}
        articles.append({"id": item["document_id"], "token_ids": item["token_ids"], "metadata": metadata})
        end = offset + len(item["token_ids"])
        boundaries.append({"article_id": item["document_id"], "start": offset, "end": end,
                           "role": "history" if index == 0 else "target"})
        offset = end
    ids = [item["document_id"] for item in chain]
    cost = chain_cost(chain)
    days = [source_time(item["publication_date_original"])["day"] for item in chain]
    bucket = next((value for value in BLOCK_BUCKETS if value >= cost["article_chunk_count"]), None)
    return {"id": "news-event-stream:" + news.digest([family["group_id"], ids]), "group_id": family["group_id"],
            "source": "news_event", "split": family["split"], "tokenizer_identity": fingerprint, "articles": articles,
            "metadata": {"family_members": family["members"], "stream_article_ids": ids, "article_boundaries": boundaries,
                         "ordering_basis": ORDERING, "max_span_days": maximum_days,
                         "selected_calendar_span_days": (max(days) - min(days)).days,
                         "family_date_min": family["family_date_min"], "family_date_max": family["family_date_max"],
                         "dataset_revision": news.REVISION, "automatic_family": True, "human_verified_event": False,
                         **cost, "block_bucket": bucket, "block_size": 512,
                         "family_basis": "inherited canonical URL, exact body/headline, confirmed near-text component",
                         "deduplication": "normalized body exact equality only; near variants retained",
                         "article_boundaries_basis": "cumulative lengths for audit only; no joined model input",
                         "delimiter_tokens": 0, "first_article_write_only": True}}


def make_streams(recovered, families, tokenizer, fingerprint, *, minimum_tokens=128, maximum_tokens=16384,
                 maximum_articles=4, maximum_days=30, seed=17, maximum_stream_tokens=16384, maximum_stream_chunks=32):
    streams, stats = [], {split: Counter() for split in news.SPLITS}
    pending = []
    for group in sorted(recovered):
        items = recovered[group]
        cell = stats[families[group]["split"]]
        cell["recovered_families"] += 1
        cell["recovered_articles"] += len(items)
        unique, duplicates = deduplicate_exact(items)
        cell["exact_duplicate_articles_removed"] += duplicates
        if len(unique) < 2:
            cell["families_with_fewer_than_two_distinct_bodies"] += 1
            continue
        cell["families_before_length_and_order_filters"] += 1
        pending.extend(unique)
    eligible = defaultdict(list)
    for start in range(0, len(pending), 32):
        batch = pending[start:start + 32]
        encoded = tokenizer([item["text"] for item in batch], add_special_tokens=False, truncation=False)["input_ids"]
        if len(encoded) != len(batch):
            raise ValueError("tokenizer returned wrong batch size")
        for item, ids in zip(batch, encoded):
            if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError("invalid tokenizer token IDs")
            count, cell = len(ids), stats[item["split"]]
            cell["articles_tokenized"] += 1
            cell["tokens_before_length_filter"] += count
            if count < minimum_tokens:
                cell["short_articles_excluded"] += 1
                cell["short_article_tokens_excluded"] += count
            elif count > maximum_tokens:
                cell["long_articles_excluded"] += 1
                cell["long_article_tokens_excluded"] += count
            else:
                item["token_ids"] = ids
                eligible[item["group_id"]].append(item)
                cell["length_eligible_articles"] += 1
        if start % 2048 == 0:
            news.log("event_stream_tokenize", processed=min(start + 32, len(pending)), total=len(pending))
    for group in sorted(families):
        family, pool = families[group], eligible[group]
        cell = stats[family["split"]]
        if not pool:
            continue
        unrestricted = choose_chain(pool, maximum_articles=maximum_articles, maximum_days=maximum_days, seed=seed)
        if unrestricted:
            cost = chain_cost(unrestricted)
            if cost["raw_tokens"] > maximum_stream_tokens or cost["article_chunk_count"] > maximum_stream_chunks:
                cell["families_with_unrestricted_chain_exceeding_stream_budget"] += 1
        chain = choose_chain(pool, maximum_articles=maximum_articles, maximum_days=maximum_days, seed=seed,
                             maximum_stream_tokens=maximum_stream_tokens, maximum_stream_chunks=maximum_stream_chunks)
        if not chain:
            cell["families_excluded_by_stream_budget" if unrestricted else "families_without_eligible_chronological_pair"] += 1
            cell["articles_excluded_by_order_or_span"] += len(pool)
            continue
        cell["articles_not_selected_by_order_span_or_four_article_cap"] += len(pool) - len(chain)
        streams.append(export_stream(family, chain, fingerprint, maximum_days))
    return streams, {split: dict(cell) for split, cell in stats.items()}


def summarize(streams, minimum_families=64, total_training_tokens=100_000_000, maximum_reuse=3):
    cells = {split: {"independent_families": 0, "articles": 0, "history_tokens": 0, "later_article_tokens": 0,
                     "article_count_histogram": Counter(),
                     "calendar_span_days_histogram": Counter(), "publisher_articles": Counter(),
                     "buckets": {str(value): {"families": 0, "actual_chunks": 0, "raw_tokens": 0, "target_tokens": 0}
                                 for value in BLOCK_BUCKETS}} for split in news.SPLITS}
    seen, article_owners = set(), {}
    for stream in streams:
        group, split, articles = stream["group_id"], stream["split"], stream["articles"]
        if group in seen:
            raise ValueError("more than one stream per independent family")
        seen.add(group)
        if split not in cells or not 2 <= len(articles) <= 4:
            raise ValueError("invalid stream split or article count")
        cell = cells[split]
        cell["independent_families"] += 1
        cell["articles"] += len(articles)
        cell["history_tokens"] += len(articles[0]["token_ids"])
        cell["later_article_tokens"] += sum(len(article["token_ids"]) for article in articles[1:])
        cell["article_count_histogram"][str(len(articles))] += 1
        cell["calendar_span_days_histogram"][str(stream["metadata"]["selected_calendar_span_days"])] += 1
        raw = sum(len(article["token_ids"]) for article in articles)
        chunks = sum((len(article["token_ids"]) + 511) // 512 for article in articles)
        if raw > 16384 or chunks > 32:
            raise ValueError("exported stream exceeds registered token or article-chunk budget")
        bucket = str(next(value for value in BLOCK_BUCKETS if value >= chunks))
        cell["buckets"][bucket]["families"] += 1
        cell["buckets"][bucket]["actual_chunks"] += chunks
        cell["buckets"][bucket]["raw_tokens"] += raw
        cell["buckets"][bucket]["target_tokens"] += sum(len(article["token_ids"]) for article in articles[1:])
        for article in articles:
            old = article_owners.setdefault(article["id"], group)
            if old != group:
                raise ValueError("article reused across event families")
            cell["publisher_articles"][news.urlsplit(article["metadata"]["canonical_url"]).hostname] += 1
    failures = [f"{split}: {cell['independent_families']} independent families < {minimum_families}"
                for split, cell in cells.items() if cell["independent_families"] < minimum_families]
    unique = cells["train"]["later_article_tokens"]
    budget = {}
    for fraction in (0.05, 0.10):
        requested = int(total_training_tokens * fraction)
        budget[str(fraction)] = {"requested_target_tokens": requested, "unique_target_tokens": unique,
                               "maximum_reuse": maximum_reuse, "maximum_available_target_tokens": unique * maximum_reuse,
                               "required_average_reuse": requested / unique if unique else None,
                               "fits_target_token_budget": requested <= unique * maximum_reuse}
    return {"splits": cells, "readiness_failures": failures, "budget_options": budget,
            "budget_basis": "all later article tokens; first article write-only; each article starts with local BOS/EOS input so its first real token is scored; padding and BOS/EOS not targets",
            "training_budget_tokens": total_training_tokens, "maximum_reuse": maximum_reuse}


def single_news_remainder(source_manifest, source_directory, streams, *, minimum_eval_families=64):
    """Audit all original exported documents after removing selected stream families.

    Report each window cell independently: 2/4/8/16-block windows overlap and
    their token counts must never be summed into an apparent unique capacity.
    """
    removed = {stream["group_id"] for stream in streams}
    stream_splits = {stream["group_id"]: stream["split"] for stream in streams}
    seen, removed_found = set(), set()
    cells = {split: {"independent_families": 0, "raw_tokens": 0, "removed_families": 0, "removed_raw_tokens": 0,
                     "windows": Counter(), "window_loss_tokens": Counter(), "unused_tail_tokens": Counter()}
             for split in news.SPLITS}
    for name in ("train_validation.jsonl", "holdout.jsonl"):
        path = Path(source_directory) / name
        expected = source_manifest.get("files", {}).get(name, {})
        if path.stat().st_size != expected.get("bytes") or news.file_sha256(path) != expected.get("sha256"):
            raise ValueError(f"source single-news {name} integrity mismatch")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                group, split = record["group_id"], record["split"]
                if split not in cells or group in seen:
                    raise ValueError("invalid or duplicate source single-news family")
                if (name == "holdout.jsonl") != (split == "holdout"):
                    raise ValueError("source single-news file and split mismatch")
                seen.add(group)
                count, cell = len(record["token_ids"]), cells[split]
                if group in removed:
                    if split != stream_splits[group]:
                        raise ValueError("event-stream and single-news source split mismatch")
                    removed_found.add(group)
                    cell["removed_families"] += 1
                    cell["removed_raw_tokens"] += count
                else:
                    cell["independent_families"] += 1
                    cell["raw_tokens"] += count
                    for blocks in (2, 4, 8, 16):
                        windows = count // (512 * blocks)
                        cell["windows"][str(blocks)] += windows
                        cell["window_loss_tokens"][str(blocks)] += windows * (blocks - 1) * 512
                        cell["unused_tail_tokens"][str(blocks)] += count % (512 * blocks)
    if removed_found != removed:
        raise ValueError("selected stream family missing from original single-news exports")
    failures = [f"remaining single-news {split}: {cells[split]['independent_families']} families < {minimum_eval_families}"
                for split in ("validation", "holdout") if cells[split]["independent_families"] < minimum_eval_families]
    return {"splits": cells, "readiness_failures": failures,
            "capacity_note": "Window cells overlap. Counts are separate capacities and are not additive unique targets.",
            "removal_basis": "all exported event-stream families removed from the corresponding single-news split"}


def export_event_texts(path, streams, recovered):
    """Full selected article bodies for cross-source leakage auditing, one per ID."""
    path = Path(path)
    temporary, seen = path.with_suffix(path.suffix + ".tmp"), set()
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for stream in sorted(streams, key=lambda value: (value["split"], value["group_id"])):
            bodies = {item["document_id"]: item["text"] for item in recovered[stream["group_id"]]}
            for article in stream["articles"]:
                if article["id"] in seen:
                    continue
                seen.add(article["id"])
                text = bodies[article["id"]]
                if news.text_hash(text) != article["metadata"]["text_sha256"]:
                    raise ValueError("selected event article text checksum mismatch")
                handle.write(json.dumps({"id": article["id"], "text": text, "metadata": article["metadata"]},
                                        ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return {"bytes": path.stat().st_size, "sha256": news.file_sha256(path), "articles": len(seen)}


def run(args):
    if args.minimum_families < 64:
        raise ValueError("at least 64 independent families are required in every split")
    if not 2 <= args.maximum_articles <= 4 or not 1 <= args.maximum_span_days <= 30:
        raise ValueError("registered streams require 2-4 articles and at most 30 days")
    if not 128 <= args.minimum_article_tokens <= args.maximum_article_tokens <= 16384:
        raise ValueError("event runtime requires article token bounds inside 128..16384")
    if not 1 <= args.maximum_stream_tokens <= 16384 or not 2 <= args.maximum_stream_chunks <= 32:
        raise ValueError("whole streams require at most 16384 tokens and 32 article-local chunks")
    if args.total_training_tokens < 1 or not 1 <= args.maximum_reuse <= 3:
        raise ValueError("positive training budget and reuse at most three are required")
    selected = Path(args.selected_texts).resolve()
    source_path = Path(args.source_manifest).resolve() if args.source_manifest else selected.parent / "data_manifest.json"
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    expected = source_manifest.get("files", {}).get("selected_texts.jsonl.gz", {})
    if selected.stat().st_size != expected.get("bytes") or news.file_sha256(selected) != expected.get("sha256"):
        raise ValueError("selected_texts integrity does not match completed source manifest")
    download_path = Path(args.download_manifest).resolve()
    download_manifest = json.loads(download_path.read_text(encoding="utf-8"))
    news.validate_download_manifest(download_manifest, args.max_compressed_bytes)
    if source_manifest.get("download", {}).get("files") != download_manifest.get("files"):
        raise ValueError("source and rescan download shards differ")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ("train_validation_event_streams.jsonl", "holdout_event_streams.jsonl", "event_stream_manifest.json", "event_stream_families.json",
             "selected_event_texts.jsonl.gz")
    if any((output / name).exists() for name in (*names, "event_stream_candidate_manifest.json")):
        raise FileExistsError("refusing to replace an existing event-stream build")
    families, owner, representatives = read_selected_families(selected, source_manifest)
    parameters = source_manifest.get("parameters", {})
    recovered, recovery_stats = recover_members(verified_source_rows(download_path, args.max_compressed_bytes), families, owner,
                        representatives, minimum_chars=parameters.get("minimum_chars", 300), maximum_chars=parameters.get("maximum_chars", 100000))
    from transformers import AutoTokenizer
    from delta_ngram.raw_lm import tokenizer_identity
    tokenizer = AutoTokenizer.from_pretrained(news.TOKENIZER, revision=news.TOKENIZER_REVISION)
    identity = tokenizer_identity(SimpleNamespace(tokenizer=tokenizer))
    fingerprint = "sha256:" + news.digest(identity)
    if fingerprint != source_manifest.get("tokenizer", {}).get("fingerprint"):
        raise ValueError("tokenizer identity differs from source corpus")
    streams, filtering = make_streams(recovered, families, tokenizer, fingerprint,
                  minimum_tokens=args.minimum_article_tokens, maximum_tokens=args.maximum_article_tokens,
                  maximum_articles=args.maximum_articles, maximum_days=args.maximum_span_days, seed=args.seed,
                  maximum_stream_tokens=args.maximum_stream_tokens, maximum_stream_chunks=args.maximum_stream_chunks)
    statistics = summarize(streams, args.minimum_families, args.total_training_tokens, args.maximum_reuse)
    remainder = single_news_remainder(source_manifest, source_path.parent, streams)
    statistics["single_news_after_family_removal"] = remainder
    statistics["readiness_failures"].extend(remainder["readiness_failures"])
    manifest = {"format": FORMAT, "ready": not statistics["readiness_failures"], "evaluation_only": False,
                "readiness_scope": "structural family and remaining single-news evaluation-group gates; training-budget fit is reported separately",
                "dataset_id": news.DATASET, "dataset_revision": news.REVISION, "parameters": vars(args),
                "source_manifest_sha256": news.file_sha256(source_path), "selected_texts_sha256": news.file_sha256(selected),
                "download_manifest_sha256": news.file_sha256(download_path), "source_script_sha256": news.file_sha256(Path(__file__)),
                "source_family_builder_sha256": news.file_sha256(Path(__file__).with_name("prepare_news_corpus.py")),
                "tokenizer": {"name": news.TOKENIZER, "revision": news.TOKENIZER_REVISION, "identity": identity, "fingerprint": fingerprint},
                "recovery": recovery_stats, "filtering": filtering, "statistics": statistics,
                "temporal_splits": source_manifest["temporal_splits"],
                "selection": {"one_stream_per_family": True, "minimum_articles": 2, "maximum_articles": args.maximum_articles,
                              "maximum_span_days": args.maximum_span_days, "exact_body_dedup_only": True,
                              "maximum_stream_tokens": args.maximum_stream_tokens, "maximum_article_chunks": args.maximum_stream_chunks,
                              "block_buckets": list(BLOCK_BUCKETS), "block_size": 512,
                              "same_day_order_requires_explicit_timezone": True, "ordering_basis": ORDERING,
                              "chain_rule": "earliest-feasible traversal from each start; prefer count then seeded chain hash",
                              "complete_extracted_articles": True, "cross_article_concatenation": False},
                "limitations": ["Inherited families are automatic lexical/URL groups, not human-verified semantic events.",
                    "Only families represented in the previous selected corpus can be recovered; its >=1024-token representative bias remains.",
                    "Original family grouping was approximate and bounded; absent event relations cannot be recovered from membership IDs.",
                    "Unknown timezone timestamps order only by different source calendar dates; this is not proof of UTC or publication availability order.",
                    "Mixed timezone chain traversal is conservative, not a guaranteed maximum-length ordering.",
                    "Whole articles outside token limits and nonselected family members are excluded; no article is truncated.",
                    "Long-running families may contribute one <=30-day stream; full family date extent remains in metadata.",
                    "Corpus-era holdout is not recent-news FINAL evaluation; unknown article licenses remain unknown.",
                    "Reported budget feasibility must be checked against runtime loss masks and actual unique-target reuse."]}
    family_path = output / "event_stream_families.json"
    news.save_json(family_path, {"format": "delta_ngram_news_event_family_selection_v1",
                   "source_manifest_sha256": manifest["source_manifest_sha256"],
                   "splits": {split: sorted(stream["group_id"] for stream in streams if stream["split"] == split)
                              for split in news.SPLITS}})
    manifest["family_selection"] = {"path": family_path.name, "bytes": family_path.stat().st_size,
                                    "sha256": news.file_sha256(family_path)}
    news.save_json(output / "event_stream_candidate_manifest.json", manifest)
    if not manifest["ready"]:
        news.log("event_stream_insufficient", statistics=statistics)
        raise ValueError("insufficient independent event-stream families: " + "; ".join(statistics["readiness_failures"]))
    paths = {name: output / name for name in names[:2]}
    temporary = {name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items()}
    with temporary[names[0]].open("w", encoding="utf-8") as train, temporary[names[1]].open("w", encoding="utf-8") as holdout:
        for stream in sorted(streams, key=lambda value: (value["split"], value["group_id"])):
            target = holdout if stream["split"] == "holdout" else train
            target.write(json.dumps(stream, ensure_ascii=False, separators=(",", ":")) + "\n")
    for name, path in paths.items():
        temporary[name].replace(path)
    paths[family_path.name] = family_path
    manifest["files"] = {name: {"bytes": path.stat().st_size, "sha256": news.file_sha256(path)} for name, path in paths.items()}
    manifest["files"]["selected_event_texts.jsonl.gz"] = export_event_texts(output / "selected_event_texts.jsonl.gz", streams, recovered)
    news.save_json(output / names[2], manifest)
    news.log("event_stream_complete", output=str(output), statistics=statistics, files=manifest["files"])
    return manifest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--selected-texts", required=True)
    result.add_argument("--source-manifest")
    result.add_argument("--download-manifest", required=True)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--max-compressed-bytes", type=int, default=1_200_000_000)
    result.add_argument("--minimum-article-tokens", type=int, default=128)
    result.add_argument("--maximum-article-tokens", type=int, default=16384)
    result.add_argument("--maximum-articles", type=int, default=4)
    result.add_argument("--maximum-span-days", type=int, default=30)
    result.add_argument("--maximum-stream-tokens", type=int, default=16384)
    result.add_argument("--maximum-stream-chunks", type=int, default=32)
    result.add_argument("--minimum-families", type=int, default=64)
    result.add_argument("--total-training-tokens", type=int, default=100_000_000)
    result.add_argument("--maximum-reuse", type=int, default=3)
    result.add_argument("--seed", type=int, default=17)
    return result


if __name__ == "__main__":
    run(parser().parse_args())

#!/usr/bin/env python
"""Build bounded, publication-dated old-news training data from pinned CC-News.

Download is explicit and capped; source text is data and is never executed.
Complete extracted articles are encoded once, without cross-article packing.
Temporal partitions: 2017-18 train, 2019 H1 validation, 2019 H2 old holdout.
This corpus never supplies the separate, recent-event final evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import heapq
import json
from pathlib import Path
import re
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import urlopen

DATASET = "vblagoje/cc_news"
REVISION = "81eb2ce0d2a9dad6ad16b68ef750ec290880fa36"
TOKENIZER = "Qwen/Qwen3-0.6B"
TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FILES = (
    (211044294, "5cda9a2b0b40a9c9438ae02168b16beb61e52449a895869515501962c5aeb977"),
    (233804414, "1d33583fdd82fa28a7aa961a633aeb63be90fdac8f47c1893daf630c9a5f074a"),
    (218914067, "78d59696b61fa9789879836a303c7a04beb52107c6678a444b98ce4e8b97b1ae"),
    (244540595, "53b35fa4b5d82a0308ddaba3d55033c49a1f1d4cca45d88d2633495cab4767db"),
    (214502216, "e0869a42852439aa408030de51464d0a8cbe5608ff1a7ba9d165e10964c6fdef"),
)
SPLITS = ("train", "validation", "holdout")
LENGTHS = (512, 1024, 2048, 4096, 8192)
SOURCE_LENGTH_SAMPLE = {
    "dataset_revision": REVISION,
    "source_shard": "plain_text/train-00000-of-00005.parquet",
    "source_shard_sha256": FILES[0][1],
    "row_group": 0, "source_rows_one_based": [1, 1000], "documents": 1000,
    "selection": "first parquet row group, not a random corpus sample; no text/date/length filter",
    "tokenizer": TOKENIZER, "tokenizer_revision": TOKENIZER_REVISION,
    "add_special_tokens": False, "truncation": False,
    "token_percentiles": {"min": 21, "p25": 174, "p50": 453, "p75": 837.75,
                          "p90": 1175.7, "p95": 1436.5, "max": 7096},
    "at_least": {"512": 442, "1024": 155, "2048": 14, "4096": 5, "8192": 0},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text):
    """Same hash-only normalization as the existing code and mixed builders."""
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip()


def priority(seed, value):
    return int(digest([seed, value]), 16)


def log(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=True), flush=True)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def expected_files():
    return [{"path": f"plain_text/train-{i:05d}-of-00005.parquet", "size": size,
             "sha256": checksum} for i, (size, checksum) in enumerate(FILES)]


def validate_download_manifest(manifest, maximum_bytes):
    if manifest.get("dataset_id") != DATASET or manifest.get("dataset_revision") != REVISION:
        raise ValueError("download manifest must pin the required CC-News revision")
    entries = manifest.get("files")
    if entries != expected_files():
        raise ValueError("download manifest must contain exactly the five pinned shards, sizes and SHA256 values")
    if sum(entry["size"] for entry in entries) > maximum_bytes:
        raise ValueError("compressed download budget exceeded")
    return entries


def download(directory, maximum_bytes):
    """Acquire only known immutable shards; atomically publish verified files."""
    manifest = {"format": "delta_ngram_news_download_v1", "dataset_id": DATASET,
                "dataset_revision": REVISION, "files": expected_files()}
    entries = validate_download_manifest(manifest, maximum_bytes)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        path = directory / "raw" / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.stat().st_size != entry["size"] or file_sha256(path) != entry["sha256"]:
                raise ValueError(f"existing shard failed integrity verification: {path}")
            log("download_cached", path=entry["path"], bytes=entry["size"])
            continue
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{entry['path']}"
        partial = path.with_suffix(path.suffix + ".part")
        size, checksum = 0, hashlib.sha256()
        with urlopen(url, timeout=60) as response, partial.open("wb") as destination:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > entry["size"]:
                    raise ValueError("download exceeded pinned shard size")
                destination.write(chunk)
                checksum.update(chunk)
        if size != entry["size"] or checksum.hexdigest() != entry["sha256"]:
            raise ValueError(f"download failed integrity verification: {entry['path']}")
        partial.replace(path)
        log("download_complete", path=entry["path"], bytes=size, sha256=checksum.hexdigest())
    manifest["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    save_json(directory / "download_manifest.json", manifest)
    return directory / "download_manifest.json"


def publication_day(value):
    """Accept an explicit ISO date/timestamp only; never infer dates from URLs."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?", value):
        raise ValueError("missing or malformed publication date")
    # Validate the entire timestamp, not merely its date prefix. Preserve the
    # source calendar date because the upstream timezone is often unspecified.
    datetime.fromisoformat(value.replace("Z", "+00:00"))
    return date.fromisoformat(value[:10])


def temporal_split(day):
    if date(2017, 1, 1) <= day <= date(2018, 12, 31):
        return "train"
    if date(2019, 1, 1) <= day <= date(2019, 6, 30):
        return "validation"
    if date(2019, 7, 1) <= day <= date(2019, 12, 31):
        return "holdout"
    raise ValueError("publication date outside registered 2017-2019 corpus")


def canonical_url(value):
    if not isinstance(value, str):
        raise ValueError("missing article URL")
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("invalid article URL")
    host = parsed.hostname.casefold().removeprefix("www.")
    port = parsed.port
    if port and not (parsed.scheme == "http" and port == 80 or parsed.scheme == "https" and port == 443):
        host += f":{port}"
    query = [(key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=True)
             if not key.casefold().startswith("utm_") and key.casefold() not in ("fbclid", "gclid", "mc_cid", "mc_eid")]
    return urlunsplit(("https", host, parsed.path.rstrip("/") or "/", urlencode(sorted(query)), ""))


def candidate_record(record, shard, row, *, minimum_chars=300, maximum_chars=100000):
    if not isinstance(record, dict):
        raise ValueError("invalid_record")
    text = record.get("text")
    if not isinstance(text, str) or not minimum_chars <= len(text) <= maximum_chars:
        raise ValueError("character_length")
    if "\x00" in text or "\ufffd" in text:
        raise ValueError("binary_or_replacement_character")
    day = publication_day(record.get("date"))
    split = temporal_split(day)
    url = canonical_url(record.get("url"))
    substantial = [line.strip() for line in text.splitlines() if len(line.strip()) >= 20]
    if len(substantial) >= 40 and Counter(substantial).most_common(1)[0][1] / len(substantial) > 0.25:
        raise ValueError("mechanical_line_repetition")
    origin = {"source_shard": shard["path"], "source_row": row, "source_shard_sha256": shard["sha256"]}
    return {"document_id": "cc-news:" + digest([REVISION, origin]), "text": text,
            "publication_date": day.isoformat(), "publication_date_original": record["date"],
            "canonical_url": url, "split": split, "upstream_metadata": {key: val for key, val in record.items() if key != "text"},
            **origin}


class Reservoir:
    """Deterministic smallest-hash sample with a hard document count bound."""
    def __init__(self, capacity, seed):
        if capacity < 1:
            raise ValueError("candidate capacity must be positive")
        self.capacity, self.seed, self.heap = capacity, seed, []

    def add(self, item):
        rank = priority(self.seed, item["document_id"])
        value = (-rank, item["document_id"], item)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, value)
        elif value[:2] > self.heap[0][:2]:
            heapq.heapreplace(self.heap, value)

    def selected(self):
        return [entry[2] for entry in sorted(self.heap, key=lambda entry: (-entry[0], entry[1]))]


def scan(download_manifest, args):
    import pyarrow.parquet as pq
    manifest_path = Path(download_manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = validate_download_manifest(manifest, args.max_compressed_bytes)
    pools = {split: Reservoir(args.train_candidates if split == "train" else args.eval_candidates, args.seed) for split in SPLITS}
    stats, rejected, dates = Counter(), Counter(), Counter()
    for entry in entries:
        path = manifest_path.parent / "raw" / entry["path"]
        if path.stat().st_size != entry["size"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"local shard integrity mismatch: {entry['path']}")
        row = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            for record in batch.to_pylist():
                row += 1
                stats["scanned_records"] += 1
                try:
                    item = candidate_record(record, entry, row, minimum_chars=args.minimum_chars, maximum_chars=args.maximum_chars)
                except ValueError as error:
                    rejected[str(error)] += 1
                    continue
                dates[item["publication_date"][:7]] += 1
                stats[item["split"] + "_eligible_before_sampling"] += 1
                pools[item["split"]].add(item)
        log("shard_scanned", path=entry["path"], rows=row, records=stats["scanned_records"])
    candidates = [item for pool in pools.values() for item in pool.selected()]
    return candidates, {**stats, "rejected": dict(rejected), "publication_months": dict(sorted(dates.items())),
                        "bounded_candidates": {split: len(pool.heap) for split, pool in pools.items()}}, manifest


def form_families(candidates, threshold=0.85):
    """Join URL, text, headline and confirmed near-text edges before selection.

    LSH is approximate. A detected component spanning date partitions is fully
    quarantined. No semantic event-family completeness is claimed.
    """
    # Share the tested lexical-shingle/minhash primitives without importing any
    # downloaded content. This file sits beside prepare_code_pilot.py.
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("news_code_primitives", Path(__file__).with_name("prepare_code_pilot.py"))
    primitives = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = primitives
    spec.loader.exec_module(primitives)
    families = primitives.Families(len(candidates))
    seen = {key: {} for key in ("canonical_url", "normalized_text_sha256", "headline")}
    bands, stats = defaultdict(list), Counter()

    @lru_cache(maxsize=128)
    def features(index):
        return primitives.shingles(candidates[index]["text"].casefold())

    for index, item in enumerate(candidates):
        item["text_sha256"] = text_hash(item["text"])
        item["normalized_text_sha256"] = text_hash(normalize(item["text"]))
        title = item["upstream_metadata"].get("title") or ""
        words = re.findall(r"\w+", title.casefold())
        # Exact substantial headlines conservatively bind syndication/updates.
        headline = " ".join(words) if len(words) >= 5 else None
        for key in seen:
            value = headline if key == "headline" else item[key]
            if value is None:
                continue
            if value in seen[key]:
                families.join(index, seen[key][value])
                stats[key + "_edges"] += 1
            else:
                seen[key][value] = index
        signature = primitives.minhash(features(index), permutations=64)
        keys = [(band, signature[4 * band:4 * band + 4]) for band in range(16)]
        matches = {other for key in keys for other in bands[key]}
        for other in sorted(matches):
            if families.find(index) == families.find(other):
                continue
            stats["near_candidate_comparisons"] += 1
            if primitives.jaccard(features(index), features(other)) >= threshold:
                families.join(index, other)
                stats["confirmed_near_edges"] += 1
        for key in keys:
            bands[key].append(index)
        if (index + 1) % 2000 == 0:
            log("dedup_progress", processed=index + 1, total=len(candidates), **stats)
    members = defaultdict(list)
    for index in range(len(candidates)):
        members[families.find(index)].append(index)
    retained = []
    for indices in members.values():
        items = [candidates[index] for index in indices]
        member_ids = sorted(item["document_id"] for item in items)
        group = "news-family:" + digest(member_ids)
        splits = {item["split"] for item in items}
        if len(splits) != 1:
            stats["cross_time_quarantined_families"] += 1
            stats["cross_time_quarantined_documents"] += len(items)
            continue
        family_dates = sorted(item["publication_date"] for item in items)
        for item in items:
            item.update(group_id=group, family_members=member_ids,
                        family_date_min=family_dates[0], family_date_max=family_dates[-1])
            retained.append(item)
    stats.update({"candidate_documents": len(candidates), "detected_families": len(members),
                  "retained_documents_before_tokenization": len(retained)})
    return retained, dict(stats)


def tokenize(candidates, tokenizer, minimum_tokens=1024, maximum_tokens=16384):
    retained, stats = [], {split: {"documents": 0, "raw_tokens": 0, "at_least": Counter(),
                                   "below_minimum": 0, "above_maximum": 0} for split in SPLITS}
    for start in range(0, len(candidates), 64):
        batch = candidates[start:start + 64]
        encoded = tokenizer([item["text"] for item in batch], add_special_tokens=False, truncation=False)["input_ids"]
        for item, ids in zip(batch, encoded):
            count, cell = len(ids), stats[item["split"]]
            cell["documents"] += 1
            cell["raw_tokens"] += count
            for length in LENGTHS:
                cell["at_least"][str(length)] += int(count >= length)
            if count < minimum_tokens:
                cell["below_minimum"] += 1
            elif count > maximum_tokens:
                cell["above_maximum"] += 1
            else:
                item["token_ids"] = ids
                retained.append(item)
        if start % 2048 == 0:
            log("tokenize_progress", processed=min(start + 64, len(candidates)), total=len(candidates), retained=len(retained))
    return retained, stats


def choose_documents(candidates, *, train_tokens, eval_tokens, minimum_eval_groups, seed):
    selected = []
    for split in SPLITS:
        target = train_tokens if split == "train" else eval_tokens
        minimum_groups = 1 if split == "train" else minimum_eval_groups
        pool = sorted((item for item in candidates if item["split"] == split),
                      key=lambda item: priority(seed, item["document_id"]))
        used, total = set(), 0
        has_four = False
        for item in pool:
            if item["group_id"] in used:
                continue
            if total >= target and len(used) >= minimum_groups and has_four:
                break
            used.add(item["group_id"])
            selected.append(item)
            count = len(item["token_ids"])
            total += count
            has_four |= count >= 2048
        if total < target or len(used) < minimum_groups or not has_four:
            raise ValueError(f"insufficient {split} corpus: raw_tokens={total}, groups={len(used)}, has_4_blocks={has_four}; increase candidate cap or revise registered target")
    return selected


def validate_and_summarize(documents, minimum_eval_groups=64):
    stats = {split: {"documents": 0, "raw_tokens": 0, "groups": 0, "publishers": Counter(),
                     "windows": Counter(), "window_loss_tokens": Counter(), "unused_tail_tokens": Counter()} for split in SPLITS}
    ownership = {key: {} for key in ("group_id", "canonical_url", "text_sha256", "normalized_text_sha256")}
    groups = set()
    dates = defaultdict(list)
    for item in documents:
        split = item["split"]
        if temporal_split(publication_day(item["publication_date"])) != split:
            raise ValueError("publication date contradicts exported split")
        if item["group_id"] in groups:
            raise ValueError("must export exactly one article per detected family")
        groups.add(item["group_id"])
        for key, seen in ownership.items():
            old = seen.setdefault(item[key], split)
            if old != split:
                raise ValueError(f"{key} crosses splits")
        count, cell = len(item["token_ids"]), stats[split]
        cell["documents"] += 1
        cell["groups"] += 1
        cell["raw_tokens"] += count
        cell["publishers"][urlsplit(item["canonical_url"]).hostname] += count
        dates[split].append(item["publication_date"])
        for blocks in (2, 4, 8, 16):
            windows = count // (512 * blocks)
            cell["windows"][str(blocks)] += windows
            cell["window_loss_tokens"][str(blocks)] += windows * (blocks - 1) * 512
            cell["unused_tail_tokens"][str(blocks)] += count % (512 * blocks)
    for split, cell in stats.items():
        if not cell["groups"] or (split != "train" and cell["groups"] < minimum_eval_groups):
            raise ValueError(f"insufficient independent groups for {split}")
        if not all(cell["windows"][str(blocks)] for blocks in (2, 4)):
            raise ValueError(f"missing required 2/4-block length cell for {split}")
        cell["publication_date_min"], cell["publication_date_max"] = min(dates[split]), max(dates[split])
        cell["publisher_token_fractions"] = {name: count / cell["raw_tokens"] for name, count in cell.pop("publishers").items()}
    return {"splits": stats, "raw_tokens": sum(cell["raw_tokens"] for cell in stats.values())}


def export_document(item, fingerprint):
    metadata = {**item["upstream_metadata"], "dataset_id": DATASET, "dataset_revision": REVISION,
                "source_group": "news", "publication_date": item["publication_date"],
                "publication_date_original": item["publication_date_original"],
                "publication_date_basis": "upstream date field; metadata extraction, not independently verified",
                "canonical_url": item["canonical_url"], "source_shard": item["source_shard"],
                "source_row": item["source_row"], "source_shard_sha256": item["source_shard_sha256"],
                "text_sha256": item["text_sha256"], "normalized_text_sha256": item["normalized_text_sha256"],
                "token_sha256": digest(item["token_ids"]), "license": "unknown",
                "license_basis": "upstream dataset card: unknown; no per-article license supplied",
                "family_members": item["family_members"], "family_date_min": item["family_date_min"],
                "family_date_max": item["family_date_max"],
                "family_basis": "canonical URL, exact normalized text/headline, confirmed 5-shingle near text; not semantic event labels"}
    return {"id": item["document_id"], "group_id": item["group_id"], "source": "news", "split": item["split"],
            "token_ids": item["token_ids"], "tokenizer_identity": fingerprint, "metadata": metadata}


def run(args):
    if args.download_dir:
        download_path = download(args.download_dir, args.max_compressed_bytes)
    elif args.download_manifest:
        download_path = Path(args.download_manifest)
    else:
        raise ValueError("specify --download-dir or --download-manifest")
    if args.download_only:
        supplied = json.loads(download_path.read_text(encoding="utf-8"))
        for entry in validate_download_manifest(supplied, args.max_compressed_bytes):
            path = download_path.parent / "raw" / entry["path"]
            if path.stat().st_size != entry["size"] or file_sha256(path) != entry["sha256"]:
                raise ValueError(f"local shard integrity mismatch: {entry['path']}")
        log("download_only_complete", manifest=str(download_path))
        return
    if not args.output_dir:
        raise ValueError("--output-dir is required for preparation")
    if args.minimum_tokens != 1024 or args.maximum_tokens < 2048:
        raise ValueError("registered 512-token 2/4-block corpus requires minimum_tokens=1024 and maximum_tokens>=2048")
    if not 0 < args.near_threshold <= 1:
        raise ValueError("near threshold must lie in (0, 1]")
    if args.minimum_eval_groups < 64 or min(args.target_train_tokens, args.target_eval_tokens) < 1:
        raise ValueError("prepared corpus requires at least 64 groups per evaluation split and positive token targets")
    if not 0 < args.minimum_chars <= args.maximum_chars:
        raise ValueError("invalid character-length bounds")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ("train_validation.jsonl", "holdout.jsonl", "selected_texts.jsonl.gz", "data_manifest.json")
    if any((output / name).exists() for name in names):
        raise FileExistsError("refusing to replace an existing exported corpus")
    candidates, scan_stats, download_manifest = scan(download_path, args)
    save_json(output / "scan_manifest.json", {"download": download_manifest, "parameters": vars(args), "scan": scan_stats})
    candidates, dedup_stats = form_families(candidates, threshold=args.near_threshold)
    from transformers import AutoTokenizer
    from delta_ngram.raw_lm import tokenizer_identity
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    identity = tokenizer_identity(SimpleNamespace(tokenizer=tokenizer))
    fingerprint = "sha256:" + digest(identity)
    candidates, length_stats = tokenize(candidates, tokenizer, args.minimum_tokens, args.maximum_tokens)
    save_json(output / "candidate_manifest.json", {"deduplication": dedup_stats, "length_retention": length_stats})
    selected = choose_documents(candidates, train_tokens=args.target_train_tokens, eval_tokens=args.target_eval_tokens,
                                minimum_eval_groups=args.minimum_eval_groups, seed=args.seed)
    stats = validate_and_summarize(selected, args.minimum_eval_groups)
    paths = {name: output / name for name in names[:3]}
    temporary = {name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items()}
    with temporary["train_validation.jsonl"].open("w", encoding="utf-8") as train, temporary["holdout.jsonl"].open("w", encoding="utf-8") as holdout, gzip.open(temporary["selected_texts.jsonl.gz"], "wt", encoding="utf-8") as texts:
        for item in sorted(selected, key=lambda item: (item["split"], item["group_id"])):
            record = export_document(item, fingerprint)
            destination = holdout if item["split"] == "holdout" else train
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            texts.write(json.dumps({"id": record["id"], "text": item["text"], "metadata": record["metadata"]}, ensure_ascii=False, separators=(",", ":")) + "\n")
    for name, path in paths.items():
        temporary[name].replace(path)
    manifest = {"format": "delta_ngram_news_corpus_v1", "ready": True, "dataset_id": DATASET,
                "dataset_revision": REVISION, "download": download_manifest, "parameters": vars(args),
                "tokenizer": {"name": TOKENIZER, "revision": TOKENIZER_REVISION, "identity": identity, "fingerprint": fingerprint},
                "statistics": stats, "scan": scan_stats, "length_retention": length_stats,
                "length_retention_denominator": "all bounded, quality/date-eligible candidates after cross-time family quarantine, before token-length selection",
                "unfiltered_source_length_sample": SOURCE_LENGTH_SAMPLE,
                "deduplication": {**dedup_stats, "threshold": args.near_threshold,
                    "scope": "all bounded candidates before token filtering; approximate 64-minhash, 16 bands x 4 rows retrieval; all cross-time components excluded"},
                "temporal_splits": {"train": ["2017-01-01", "2018-12-31"], "validation": ["2019-01-01", "2019-06-30"], "holdout": ["2019-07-01", "2019-12-31"]},
                "quality_checks": {"chronological_splits": True, "detected_duplicate_families_split_disjoint": True,
                    "complete_extracted_articles_tokenized_without_truncation": True, "cross_article_concatenation": False,
                    "minimum_64_eval_groups": args.minimum_eval_groups >= 64, "recent_news_evaluation_excluded": True},
                "training_source_weights": {"news": 1.0}, "length_weights": {"2": 0.9, "4": 0.1},
                "files": {name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)} for name, path in paths.items()},
                "source_script_sha256": file_sha256(Path(__file__)),
                "dedup_primitives_script_sha256": file_sha256(Path(__file__).with_name("prepare_code_pilot.py")),
                "limitations": ["Source publication dates are extractor metadata, not publisher-verified timestamps; original crawl/update dates are absent.",
                    "The upstream dataset reports unknown license and supplies no article-specific license; this does not establish reuse rights.",
                    "Length>=1024 selection strongly favors long news; measured length retention is reported by split.",
                    "Near-duplicate retrieval is approximate and bounded; it is not a full semantic event-family detector or global deduplication guarantee.",
                    "Date holdout evaluates corpus-era generalization; recent-event final evaluation needs an independently curated event-family manifest.",
                    "Complete means unchanged extracted text, not guaranteed complete publisher HTML; no cross-article packing is performed."]}
    save_json(output / "data_manifest.json", manifest)
    log("complete", output=str(output), statistics=stats, files=manifest["files"])
    return manifest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    location = result.add_mutually_exclusive_group(required=True)
    location.add_argument("--download-dir")
    location.add_argument("--download-manifest")
    result.add_argument("--download-only", action="store_true")
    result.add_argument("--output-dir")
    result.add_argument("--max-compressed-bytes", type=int, default=1_200_000_000)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--train-candidates", type=int, default=400000)
    result.add_argument("--eval-candidates", type=int, default=20000)
    result.add_argument("--minimum-chars", type=int, default=300)
    result.add_argument("--maximum-chars", type=int, default=100000)
    result.add_argument("--minimum-tokens", type=int, default=1024)
    result.add_argument("--maximum-tokens", type=int, default=16384)
    result.add_argument("--target-train-tokens", type=int, default=50_000_000)
    result.add_argument("--target-eval-tokens", type=int, default=250_000)
    result.add_argument("--minimum-eval-groups", type=int, default=64)
    result.add_argument("--near-threshold", type=float, default=0.85)
    return result


if __name__ == "__main__":
    try:
        run(parser().parse_args())
    except Exception as error:
        log("failed", error_type=type(error).__name__, error=str(error))
        raise

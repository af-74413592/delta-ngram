#!/usr/bin/env python
"""Build bounded, traceable mixed-domain corpora without concatenating documents.

Download is independently runnable with only the Python standard library. All
corpus revisions and download hashes are pinned; source text is never executed.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import time
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit, quote
from urllib.request import Request, urlopen
import zlib


SOURCES = {
    "papers": {"dataset_id": "common-pile/arxiv_papers_filtered",
        "revision": "033cf7f53f9b348deec868c1a5a48484f3ee9e52", "directory": "",
        "pattern": r"arxiv-papers-\d{4}\.json\.gz", "shards": 2},
    "books": {"dataset_id": "common-pile/project_gutenberg_filtered",
        "revision": "3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7", "directory": "",
        "pattern": r"project_gutenberg-dolma-\d{4}\.json\.gz", "shards": 2},
    "web": {"dataset_id": "HuggingFaceFW/fineweb-edu",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "directory": "sample/10BT",
        "pattern": r"sample/10BT/\d+_\d+\.parquet", "shards": 1},
}
TOKENIZER = "Qwen/Qwen3-0.6B"
TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DEFAULT_TARGETS = {"papers": 50_000_000, "books": 40_000_000, "web": 30_000_000}
DEFAULT_LENGTHS = {"code": [8, 16], "papers": [8, 16], "books": [8, 16], "web": [8, 16], "news": [2, 4]}
DEFAULT_WEIGHTS = {"code": 0.20, "papers": 0.25, "books": 0.20, "web": 0.15, "news": 0.20}


def source_snapshot(dataset_id, revision):
    """Verify when this exact immutable source revision was publicly committed."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", dataset_id) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("invalid source snapshot identity")
    url = f"https://huggingface.co/api/datasets/{dataset_id}/commits/{revision}?limit=1"
    with urlopen(url, timeout=120) as response:
        commits = json.load(response)
    if not commits or commits[0].get("id") != revision:
        raise ValueError("source commit evidence does not match pinned revision")
    stamp = commits[0]["date"]
    datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return {"dataset_id": dataset_id, "dataset_revision": revision, "commit_date": stamp,
            "evidence_url": url, "verified_utc": datetime.now(timezone.utc).isoformat(),
            "meaning": "Repository snapshot availability, not article first-publication or model pretraining cutoff."}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text):
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip()


def priority(seed, *values):
    return int(digest([seed, *values]), 16)


def log(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=True), flush=True)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_info(path):
    path = Path(path)
    with path.open("rb") as stream:
        return {"size": path.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def discover_shards(source, count, seed=17):
    """Choose a deterministic random subset of the pinned repository tree."""
    spec = SOURCES[source]
    suffix = "/" + spec["directory"] if spec["directory"] else ""
    url = f"https://huggingface.co/api/datasets/{spec['dataset_id']}/tree/{spec['revision']}{suffix}?limit=1000"
    with urlopen(url, timeout=120) as response:
        entries = json.load(response)
        if response.headers.get("Link"):
            raise ValueError("source tree is paginated; refusing incomplete shard selection")
    eligible = [entry for entry in entries if entry.get("type") == "file"
                and re.fullmatch(spec["pattern"], entry.get("path", ""))]
    if not 1 <= count <= len(eligible):
        raise ValueError(f"invalid shard count for {source}: {count}/{len(eligible)}")
    eligible.sort(key=lambda entry: priority(seed, source, entry["path"]))
    result = []
    for entry in eligible[:count]:
        sha = entry.get("lfs", {}).get("oid", "")
        if not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise ValueError("upstream shard lacks a SHA256 content identity")
        result.append({"source": source, "dataset_id": spec["dataset_id"], "dataset_revision": spec["revision"],
            "path": entry["path"], "size": entry["size"], "sha256": sha,
            "local_path": f"raw/{source}/{entry['path']}",
            "url": f"https://huggingface.co/datasets/{spec['dataset_id']}/resolve/{spec['revision']}/{entry['path']}"})
    return result


def validate_download_manifest(manifest, maximum_bytes):
    entries = manifest.get("files", [])
    if not entries or sum(entry["size"] for entry in entries) > maximum_bytes:
        raise ValueError("empty manifest or compressed shard budget exceeded")
    seen = set()
    for entry in entries:
        source = entry["source"]
        spec = SOURCES[source]
        if (entry["dataset_id"] != spec["dataset_id"] or entry["dataset_revision"] != spec["revision"]
                or not re.fullmatch(spec["pattern"], entry["path"])
                or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
                or entry["local_path"] != f"raw/{source}/{entry['path']}"
                or type(entry["size"]) is not int or entry["size"] < 1):
            raise ValueError("invalid pinned shard identity")
        expected_url = f"https://huggingface.co/datasets/{spec['dataset_id']}/resolve/{spec['revision']}/{entry['path']}"
        if entry.get("url") != expected_url or (source, entry["path"]) in seen:
            raise ValueError("invalid or duplicate shard URL")
        seen.add((source, entry["path"]))
    return entries


def download_file(entry, root):
    target = root / entry["local_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_info(target) != {"size": entry["size"], "sha256": entry["sha256"]}:
            raise ValueError(f"existing shard failed integrity check: {target}")
        log("download_cached", source=entry["source"], path=entry["path"])
        return
    partial = target.with_suffix(target.suffix + ".partial")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with urlopen(Request(entry["url"], headers=headers), timeout=120) as response:
                resumed = offset > 0 and response.status == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("unexpected download range")
                mode = "ab" if resumed else "wb"
                count = offset if resumed else 0
                report_at = count + 128 * 1024 * 1024
                with partial.open(mode) as stream:
                    while chunk := response.read(4 * 1024 * 1024):
                        stream.write(chunk)
                        count += len(chunk)
                        if count > entry["size"]:
                            raise ValueError("download exceeds pinned size")
                        if count >= report_at:
                            log("download_progress", source=entry["source"], path=entry["path"], bytes=count, total=entry["size"])
                            report_at = count + 128 * 1024 * 1024
            if file_info(partial) != {"size": entry["size"], "sha256": entry["sha256"]}:
                raise ValueError(f"download integrity check failed: {target}")
            partial.replace(target)
            log("download_verified", source=entry["source"], path=entry["path"], size=entry["size"], sha256=entry["sha256"])
            return
        except (OSError, TimeoutError) as error:
            log("download_retry", attempt=attempt + 1, path=entry["path"], error=str(error))
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def run_download(args):
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "download_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        requested = {name: getattr(args, name + "_shards") for name in args.sources}
        if manifest.get("seed") != args.seed or manifest.get("requested_shards") != requested:
            raise ValueError("existing download plan differs; use a new directory")
    else:
        entries = []
        for source in args.sources:
            entries.extend(discover_shards(source, getattr(args, source + "_shards"), args.seed))
        manifest = {"format": "delta_ngram_mixed_download_v1", "seed": args.seed,
            "requested_shards": {name: getattr(args, name + "_shards") for name in args.sources},
            "created_utc": datetime.now(timezone.utc).isoformat(), "files": entries,
            "complete": False}
        validate_download_manifest(manifest, args.max_compressed_bytes)
        save_json(path, manifest)
    entries = validate_download_manifest(manifest, args.max_compressed_bytes)
    log("download_plan", manifest=str(path), compressed_bytes=sum(entry["size"] for entry in entries), files=entries)
    if args.plan_only:
        return manifest
    for entry in entries:
        download_file(entry, root)
    manifest["complete"] = True
    save_json(path, manifest)
    return manifest


def canonical_url(value):
    if not isinstance(value, str):
        raise ValueError("missing valid source URL")
    parts = urlsplit(value.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise ValueError("missing valid source URL")
    host = parts.hostname.casefold().removeprefix("www.")
    if parts.port and not (parts.scheme == "http" and parts.port == 80 or parts.scheme == "https" and parts.port == 443):
        host += f":{parts.port}"
    query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
             if not key.lower().startswith("utm_") and key.lower() not in {"fbclid", "gclid", "mc_cid", "mc_eid"}]
    return urlunsplit(("https", host, parts.path.rstrip("/") or "/", urlencode(sorted(query)), ""))


def work_keys(source, record):
    """Conservative work/version aliases; no inferred cross-document sequences."""
    meta = record.get("metadata", {})
    document_id = str(record.get("id", "")).strip()
    if not document_id:
        raise ValueError("missing_document_id")
    if source == "papers":
        match = re.fullmatch(r"(?:arxiv:)?((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7}))(?:v\d+)?", document_id, re.I)
        if not match:
            raise ValueError("invalid_arxiv_id")
        return ["arxiv:" + match.group(1).casefold()]
    if source == "books":
        if not document_id.isdigit():
            raise ValueError("invalid_gutenberg_id")
        title = meta.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("missing_book_title")
        # Same title across editions (or authors) stays together, deliberately
        # overgrouping ambiguous titles instead of claiming independent books.
        title_key = " ".join(re.findall(r"\w+", title.casefold()))
        return ["gutenberg:" + document_id, "book-title:" + title_key]
    if source == "web":
        return ["web-url:" + canonical_url(record.get("url", ""))]
    raise ValueError(f"unknown source: {source}")


def eligible_record(source, record, minimum_chars=8000, maximum_chars=2_000_000):
    if not isinstance(record, dict):
        return "invalid_record"
    text, meta = record.get("text"), record.get("metadata", {})
    if not isinstance(text, str) or not minimum_chars <= len(text) <= maximum_chars:
        return "character_length"
    if not isinstance(meta, dict):
        return "missing_metadata"
    try:
        work_keys(source, record)
    except (ValueError, TypeError, AttributeError):
        return "invalid_work_identity"
    if "\x00" in text or "\ufffd" in text:
        return "binary_or_replacement_character"
    if source in ("papers", "books"):
        if not isinstance(meta.get("license"), str) or not meta["license"].strip():
            return "missing_record_license"
        if not isinstance(meta.get("url"), str) or not meta["url"].startswith(("https://", "http://")):
            return "missing_source_url"
    if source == "books" and meta.get("language") not in ("en", "eng", "English"):
        return "non_english"
    if source == "web":
        if record.get("language") != "en" or float(record.get("language_score", 0)) < 0.8:
            return "non_english"
        if not record.get("file_path") or not record.get("dump"):
            return "missing_crawl_provenance"
    letters = [char for char in text[:20_000] if char.isalpha()]
    if not letters or sum(char.isascii() for char in letters) / len(letters) < 0.9:
        return "non_latin_text"
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 20]
    if len(lines) >= 40 and Counter(lines).most_common(1)[0][1] / len(lines) > 0.25:
        return "mechanical_line_repetition"
    return None


class CandidateStore:
    """Disk-backed text and uint32 token storage; never retain corpus token lists."""
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA cache_size=-16384")
        self.db.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, text BLOB NOT NULL, tokens BLOB)")

    def put(self, key, text):
        self.db.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, NULL)",
                        (key, zlib.compress(text.encode("utf-8"), 1)))

    def update_text(self, key, text):
        self.db.execute("UPDATE documents SET text = ? WHERE id = ?", (zlib.compress(text.encode("utf-8"), 1), key))

    def delete(self, key):
        self.db.execute("DELETE FROM documents WHERE id = ?", (key,))

    def text(self, key):
        row = self.db.execute("SELECT text FROM documents WHERE id = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return zlib.decompress(row[0]).decode("utf-8")

    def set_tokens(self, key, values):
        if any(type(value) is not int or not 0 <= value < 2 ** 32 for value in values):
            raise ValueError("token_ids must contain uint32 integers")
        data = array("I", values)
        if sys.byteorder != "little":
            data.byteswap()
        self.db.execute("UPDATE documents SET tokens = ? WHERE id = ?", (zlib.compress(data.tobytes(), 1), key))

    def tokens(self, key):
        blob = self.db.execute("SELECT tokens FROM documents WHERE id = ?", (key,)).fetchone()[0]
        result = array("I")
        result.frombytes(zlib.decompress(blob))
        if sys.byteorder != "little":
            result.byteswap()
        return result.tolist()

    def close(self):
        self.db.commit()
        self.db.close()


class DiskReservoir:
    """Order-independent priority reservoir, with only index metadata in memory."""
    def __init__(self, capacity, seed, store):
        if capacity < 1:
            raise ValueError("candidate capacity must be positive")
        self.capacity, self.seed, self.store = capacity, seed, store
        self.items, self.heap = {}, []

    def add(self, item, text):
        key = item["work_keys"][0]
        rank = priority(self.seed, item["id"])
        old = self.items.get(key)
        if old and old[0] <= rank:
            return False
        while self.heap and (self.heap[0][1] not in self.items or self.items[self.heap[0][1]][0] != -self.heap[0][0]):
            heapq.heappop(self.heap)
        if old is None and len(self.items) >= self.capacity:
            if rank >= -self.heap[0][0]:
                return False
            _, evicted = heapq.heappop(self.heap)
            self.store.delete(self.items.pop(evicted)[1]["id"])
        if old:
            self.store.delete(old[1]["id"])
        self.store.put(item["id"], text)
        self.items[key] = rank, item
        heapq.heappush(self.heap, (-rank, key))
        if len(self.heap) > max(16, 3 * self.capacity):
            self.heap = [(-value[0], key) for key, value in self.items.items()]
            heapq.heapify(self.heap)
        return True

    def selected(self):
        return [value[1] for value in sorted(self.items.values(), key=lambda value: value[0])]


def read_shard(path):
    if str(path).endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                yield number, json.loads(line)
    else:
        import pyarrow.parquet as pq
        number = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for record in batch.to_pylist():
                number += 1
                yield number, record


def scan_candidates(source, entries, root, store, args):
    reservoir = DiskReservoir(args.candidate_documents, args.seed, store)
    rejected, scans = Counter(), []
    for entry in entries:
        path = root / entry["local_path"]
        if file_info(path) != {"size": entry["size"], "sha256": entry["sha256"]}:
            raise ValueError(f"unverified source shard: {path}")
        seen, eligible = 0, 0
        for number, record in read_shard(path):
            seen += 1
            reason = eligible_record(source, record, args.minimum_chars, args.maximum_chars)
            if reason:
                rejected[reason] += 1
                continue
            eligible += 1
            origin = {"dataset_id": entry["dataset_id"], "dataset_revision": entry["dataset_revision"],
                "source_shard": entry["path"], "source_line": number, "source_shard_sha256": entry["sha256"]}
            meta = dict(record.get("metadata", {}))
            meta.update({key: value for key, value in record.items() if key not in ("text", "metadata")})
            meta.update(origin)
            if source == "web":
                meta["rights_status"] = "Original webpage rights not established by dataset license; URL and crawl provenance retained."
                meta["dataset_license"] = "ODC-BY-1.0"
            item = {"id": source + ":" + digest([origin, record["id"]]), "source": source,
                "document_id": str(record["id"]), "work_keys": work_keys(source, record), "metadata": meta}
            reservoir.add(item, record["text"])
            if seen % 10000 == 0:
                store.db.commit()
                log("scan_progress", source=source, shard=entry["path"], seen=seen,
                    eligible=eligible, retained=len(reservoir.items))
        scans.append({"path": entry["path"], "records": seen, "eligible": eligible})
        store.db.commit()
        log("shard_scanned", source=source, **scans[-1], retained=len(reservoir.items))
    return reservoir.selected(), {"shards": scans, "rejected": dict(rejected)}


class Families:
    def __init__(self, count):
        self.parent = list(range(count))

    def find(self, index):
        while index != self.parent[index]:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def join(self, first, second):
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parent[max(first, second)] = min(first, second)


def shingle_features(text):
    import numpy as np
    words = re.findall(r"\w+|[^\w\s]", text)
    if len(words) < 5:
        return np.asarray([int(text_hash(text)[:16], 16)], dtype=np.uint64)
    values = np.asarray([zlib.crc32(word.encode("utf-8")) for word in words], dtype=np.uint64)
    result = np.zeros(len(values) - 4, dtype=np.uint64)
    for offset in range(5):
        result = result * np.uint64(1099511628211) + values[offset:offset + len(result)]
    return np.unique(result)


def minhash(values):
    import numpy as np
    result = []
    for index in range(32):
        mixed = values ^ np.uint64(int(digest(["minhash", index])[:16], 16))
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        mixed ^= mixed >> np.uint64(31)
        result.append(int(mixed.min()))
    return tuple(result)


def form_families(items, text_getter, threshold=0.85):
    """Union all observed work, exact and LSH-confirmed near duplicate edges."""
    import numpy as np
    families, aliases, exact, bands = Families(len(items)), {}, {}, defaultdict(list)
    stats, edges = Counter(), []

    @lru_cache(maxsize=16)
    def features(index):
        return shingle_features(text_getter(items[index]["id"]))

    for index, item in enumerate(items):
        for alias in item["work_keys"]:
            if alias in aliases:
                families.join(index, aliases[alias])
                stats["work_edges"] += 1
            aliases[alias] = index
        text = text_getter(item["id"])
        sha = text_hash(normalize(text))
        item["metadata"].update(text_sha256=text_hash(text), normalized_text_sha256=sha)
        if sha in exact:
            families.join(index, exact[sha])
            edges.append((index, exact[sha], 1.0))
            stats["exact_edges"] += 1
        else:
            exact[sha] = index
        signature = minhash(features(index))
        keys = [(band, signature[4 * band:4 * band + 4]) for band in range(8)]
        for other in sorted({other for key in keys for other in bands[key]}):
            if item["metadata"]["normalized_text_sha256"] == items[other]["metadata"]["normalized_text_sha256"]:
                continue
            stats["near_comparisons"] += 1
            a, b = features(index), features(other)
            common = len(np.intersect1d(a, b, assume_unique=True))
            similarity = common / (len(a) + len(b) - common)
            if similarity >= threshold:
                families.join(index, other)
                edges.append((index, other, similarity))
                stats["near_edges"] += 1
        for key in keys:
            bands[key].append(index)
        if (index + 1) % 250 == 0:
            log("dedup_progress", documents=index + 1, **stats)
    members = defaultdict(list)
    for index in range(len(items)):
        members[families.find(index)].append(index)
    for indices in members.values():
        names = sorted({key for index in indices for key in items[index]["work_keys"]})
        group_id = "mixed-family:" + digest(names)
        for index in indices:
            items[index]["group_id"] = group_id
            items[index]["metadata"]["family_work_keys"] = names
    stats["families"] = len(members)
    return dict(stats), edges


def choose_documents(items, target_tokens, maximum_tokens, minimum_groups, seed):
    ordered = sorted(items, key=lambda item: priority(seed, "selection", item["group_id"], item["id"]))
    selected, used, total = [], set(), 0
    for item in ordered:
        if item["group_id"] in used or total + item["token_count"] > maximum_tokens:
            continue
        selected.append(item)
        used.add(item["group_id"])
        total += item["token_count"]
        if total >= target_tokens and len(selected) >= minimum_groups:
            return selected
    raise ValueError(f"candidate pool exhausted: tokens={total}/{target_tokens}, families={len(selected)}/{minimum_groups}")


def assign_splits(items, seed, minimum_groups=64, block_counts=(8, 16), block_size=512):
    if len({item["group_id"] for item in items}) != len(items):
        raise ValueError("split allocation requires one representative per family")
    ordered = sorted(items, key=lambda item: priority(seed, "split", item["group_id"]))
    count = max(minimum_groups, round(0.025 * len(items)))
    if len(items) < 2 * count + minimum_groups:
        raise ValueError("insufficient independent families for all three splits")
    # Reserve one longest-window document per split, then fill by deterministic
    # family rank. This reports, rather than hides, long-stratum sample size.
    long = [item for item in ordered if item["token_count"] >= max(block_counts) * block_size]
    if len(long) < 3:
        raise ValueError("fewer than three independent longest-window families")
    reserved = {long[index]["id"]: split for index, split in enumerate(("train", "validation", "holdout"))}
    assignments = Counter(reserved.values())
    for item in ordered:
        split = reserved.get(item["id"])
        if split is None:
            split = "validation" if assignments["validation"] < count else "holdout" if assignments["holdout"] < count else "train"
            assignments[split] += 1
        item["split"] = split
    return items


def validate_and_summarize(items, source_lengths, minimum_groups=64, block_size=512):
    stats, ownership = {}, {name: {} for name in ("id", "group_id", "text_sha256", "normalized_text_sha256", "work_key")}
    group_sources = {}
    if len({item["id"] for item in items}) != len(items):
        raise ValueError("duplicate document id")
    for item in items:
        split, source = item["split"], item["source"]
        if split not in ("train", "validation", "holdout") or source not in source_lengths:
            raise ValueError("invalid split/source")
        if group_sources.setdefault(item["group_id"], source) != source:
            raise ValueError("one family spans multiple sources; per-source evaluation would not be independent")
        meta = item["metadata"]
        for key in ownership:
            values = item.get("work_keys", meta.get("family_work_keys", [])) if key == "work_key" else [item[key] if key in ("id", "group_id") else meta[key]]
            for value in values:
                previous = ownership[key].setdefault(value, split)
                if previous != split:
                    raise ValueError(f"{key} crosses splits")
        cell = stats.setdefault(source, {}).setdefault(split, {"documents": 0, "groups": set(), "raw_tokens": 0,
            "windows": Counter(), "window_loss_tokens": Counter(), "eligible_groups": defaultdict(set), "unused_tail_tokens": Counter()})
        count = item["token_count"]
        cell["documents"] += 1
        cell["groups"].add(item["group_id"])
        cell["raw_tokens"] += count
        for blocks in source_lengths[source]:
            windows, tail = divmod(count, blocks * block_size)
            cell["windows"][str(blocks)] += windows
            cell["window_loss_tokens"][str(blocks)] += windows * (blocks - 1) * block_size
            cell["unused_tail_tokens"][str(blocks)] += tail
            if windows:
                cell["eligible_groups"][str(blocks)].add(item["group_id"])
    for source, lengths in source_lengths.items():
        for split in ("train", "validation", "holdout"):
            cell = stats.get(source, {}).get(split)
            if cell is None or len(cell["groups"]) < minimum_groups:
                raise ValueError(f"insufficient independent groups: {source}/{split}")
            for blocks in lengths:
                if cell["windows"][str(blocks)] < 1:
                    raise ValueError(f"missing source/length window cell: {source}/{split}/{blocks}")
            cell["groups"] = len(cell["groups"])
            cell["eligible_groups"] = {key: len(value) for key, value in cell["eligible_groups"].items()}
            for key in ("windows", "window_loss_tokens", "unused_tail_tokens"):
                cell[key] = dict(cell[key])
    return {"sources": stats, "raw_tokens": sum(item["token_count"] for item in items), "documents": len(items),
            "independent_groups": len({item["group_id"] for item in items})}


def selected_date_bounds(items):
    """Keep publication/addition/crawl evidence separate; missing dates stay missing."""
    result = {}
    fields = ("created", "added", "date", "publication_date", "publication_date_original", "dump")
    for item in items:
        source = result.setdefault(item["source"], {})
        split = source.setdefault(item["split"], {"documents": 0, "fields": {}})
        split["documents"] += 1
        for name in fields:
            value = item["metadata"].get(name)
            cell = split["fields"].setdefault(name, {"observed": 0, "missing": 0, "invalid": 0, "min": None, "max": None})
            if value is None or value == "":
                cell["missing"] += 1
                continue
            value = str(value)
            valid = bool(re.fullmatch(r"CC-MAIN-\d{4}-\d{2}", value)) if name == "dump" else bool(re.match(r"^\d{4}-\d{2}-\d{2}", value))
            if valid and name != "dump":
                try:
                    datetime.fromisoformat(value[:10])
                except ValueError:
                    valid = False
            if not valid:
                cell["invalid"] += 1
                continue
            cell["observed"] += 1
            cell["min"] = value if cell["min"] is None else min(cell["min"], value)
            cell["max"] = value if cell["max"] is None else max(cell["max"], value)
    return result


class CapacityError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("training cell capacity gate failed: " + "; ".join(report["violations"]))


def budget_capacity(statistics, source_weights, source_lengths, conditional_weights, planned_loss_tokens,
                    maximum_cell_passes, block_size=512):
    """Audit expected per-cell reuse and distinct token-position overlap.

    Different trajectory lengths tile prefixes of the same complete document.
    Therefore the union of usable raw positions equals the largest tiled prefix
    per document, and is bounded by the source raw count reported here. Cell
    capacity alone is not a claim that mixed-length passes never reuse tokens.
    """
    if type(planned_loss_tokens) is not int or planned_loss_tokens < 1 or not math.isfinite(maximum_cell_passes) or maximum_cell_passes < 1:
        raise ValueError("invalid registered training budget/reuse bound")
    result, violations = {}, []
    for source, share in source_weights.items():
        cells = {str(key): value for key, value in conditional_weights[source].items()}
        if set(cells) != {str(value) for value in source_lengths[source]} or any(not math.isfinite(value) or value <= 0 for value in cells.values()) or abs(sum(cells.values()) - 1) > 1e-9:
            raise ValueError(f"conditional length weights do not match source lengths: {source}")
        train = statistics["sources"][source]["train"]
        records, raw_consumption = {}, 0
        for blocks, fraction in cells.items():
            target = planned_loss_tokens * share * fraction
            available = train["window_loss_tokens"][blocks]
            passes = target / available if available else float("inf")
            raw_consumption += target * int(blocks) / (int(blocks) - 1)
            records[blocks] = {"planned_loss_tokens": target, "available_loss_tokens_per_pass": available,
                "expected_cell_passes": passes, "independent_train_groups": train["eligible_groups"].get(blocks, 0)}
            if passes > maximum_cell_passes:
                violations.append(f"{source}/{blocks}: {passes:.6f} passes exceeds {maximum_cell_passes}")
        covered_prefix_tokens = {blocks: train["raw_tokens"] - train["unused_tail_tokens"][blocks] for blocks in cells}
        result[source] = {"raw_tokens": train["raw_tokens"], "planned_loss_tokens": planned_loss_tokens * share,
            "planned_raw_token_consumption": raw_consumption, "raw_consumption_over_unique_pool": raw_consumption / train["raw_tokens"],
            "cells": records, "covered_raw_positions_per_cell": covered_prefix_tokens,
            "sum_cell_raw_positions": sum(covered_prefix_tokens.values()),
            "cross_length_overlap_lower_bound": max(0, sum(covered_prefix_tokens.values()) - train["raw_tokens"]),
            "overlap_note": "Length cells draw from the same document prefixes; sum of capacities is not unique raw data."}
    report = {"planned_loss_tokens": planned_loss_tokens, "maximum_train_cell_passes": maximum_cell_passes,
              "conditional_length_weights": conditional_weights, "sources": result,
              "violations": violations, "passed": not violations}
    if violations:
        raise CapacityError(report)
    return report


def export_document(item, store, fingerprint):
    return {"id": item["id"], "group_id": item["group_id"], "source": item["source"], "split": item["split"],
        "token_ids": store.tokens(item["id"]), "tokenizer_identity": fingerprint,
        "metadata": {**item["metadata"], "source_group": item["source"], "original_document_id": item["document_id"]}}


def export_files(output, items, store, fingerprint):
    paths = {name: output / name for name in ("train_validation.jsonl", "holdout.jsonl", "selected_texts.jsonl.gz")}
    temp = {name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items()}
    with temp["train_validation.jsonl"].open("w", encoding="utf-8", newline="\n") as train, temp["holdout.jsonl"].open("w", encoding="utf-8", newline="\n") as holdout:
        # Explicit mtime makes compressed provenance reproducible too.
        with temp["selected_texts.jsonl.gz"].open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as texts:
            for item in sorted(items, key=lambda item: (item["split"], item["source"], item["group_id"], item["id"])):
                record = export_document(item, store, fingerprint)
                destination = holdout if item["split"] == "holdout" else train
                destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                texts.write((json.dumps({"id": item["id"], "text": store.text(item["id"]), "metadata": record["metadata"]},
                                       ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    for name, path in paths.items():
        temp[name].replace(path)
    result = {}
    for name, path in paths.items():
        info = file_info(path)
        result[name] = {"bytes": info["size"], "sha256": info["sha256"]}
    return result


def run_prepare(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("data_manifest.json", "train_validation.jsonl", "candidates.sqlite")):
        raise FileExistsError("refusing to replace an existing prepared corpus; use a fresh directory")
    download_path = Path(args.download_manifest).resolve()
    download = json.loads(download_path.read_text(encoding="utf-8"))
    entries = [entry for entry in validate_download_manifest(download, args.max_compressed_bytes) if entry["source"] == args.source]
    if not entries:
        raise ValueError("download manifest lacks requested source")
    snapshot = source_snapshot(SOURCES[args.source]["dataset_id"], SOURCES[args.source]["revision"])
    started = time.monotonic()
    store = CandidateStore(output / "candidates.sqlite")
    try:
        items, scan = scan_candidates(args.source, entries, download_path.parent, store, args)
        save_json(output / "scan_manifest.json", {"source": args.source, "scan": scan, "candidate_documents": len(items)})
        from transformers import AutoTokenizer
        from delta_ngram.raw_lm import tokenizer_identity
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
        identity = tokenizer_identity(SimpleNamespace(tokenizer=tokenizer))
        fingerprint = "sha256:" + digest(identity)
        retained, rejected = [], Counter()
        for index, item in enumerate(items):
            ids = tokenizer.encode(store.text(item["id"]), add_special_tokens=False, truncation=False)
            if not args.minimum_tokens <= len(ids) <= args.maximum_tokens:
                rejected["token_length"] += 1
                store.delete(item["id"])
                continue
            store.set_tokens(item["id"], ids)
            item["token_count"] = len(ids)
            item["metadata"]["token_sha256"] = digest(ids)
            retained.append(item)
            if (index + 1) % 100 == 0:
                store.db.commit()
                log("tokenize_progress", source=args.source, processed=index + 1, retained=len(retained))
        del items
        store.db.commit()
        families, edges = form_families(retained, store.text, args.near_threshold)
        target = args.target_raw_tokens or DEFAULT_TARGETS[args.source]
        maximum = args.maximum_raw_tokens or int(target * 1.15)
        selected = choose_documents(retained, target, maximum, 3 * args.minimum_groups, args.seed)
        assign_splits(selected, args.seed, args.minimum_groups, tuple(args.block_counts))
        stats = validate_and_summarize(selected, {args.source: args.block_counts}, args.minimum_groups)
        files = export_files(output, selected, store, fingerprint)
        manifest = {"format": "delta_ngram_mixed_source_v1", "ready": True, "source": args.source,
            "dataset_id": SOURCES[args.source]["dataset_id"], "dataset_revision": SOURCES[args.source]["revision"],
            "source_snapshot": snapshot, "selected_date_bounds": selected_date_bounds(selected),
            "download": {"files": entries, "manifest_sha256": file_info(download_path)["sha256"]},
            "parameters": {key: value for key, value in vars(args).items() if key != "run"},
            "tokenizer": {"name": TOKENIZER, "revision": TOKENIZER_REVISION, "identity": identity, "fingerprint": fingerprint},
            "statistics": stats, "scan": scan, "token_rejections": dict(rejected), "files": files,
            "deduplication": {**families, "threshold": args.near_threshold, "confirmed_edges": len(edges),
                "method": "work aliases plus exact SHA256 and lexical 5-shingle Jaccard; 32 minhash, 8x4 LSH; one representative per detected family"},
            "quality_checks": {"split_disjoint_families": True, "whole_documents_encoded_once": True,
                "no_cross_document_concatenation": True, "minimum_groups_all_splits": args.minimum_groups,
                "nonempty_configured_source_length_cells": True, "holdout_excluded_from_trainer_file": True},
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "source_script_sha256": file_info(__file__)["sha256"],
            "limitations": ["Bounded deterministic shard sample, not a full-source representative sample.",
                "LSH near-duplicate retrieval can miss pairs; work aliases conservatively overgroup identical book titles.",
                "Complete-document length caps and one representative per family alter the sampled distribution.",
                "Source-level validation group count does not imply 64 independent groups in each length stratum.",
                "FineWeb dataset licensing does not establish original webpage rights."]}
        save_json(output / "data_manifest.json", manifest)
        log("prepare_complete", output=str(output), statistics=stats)
        return manifest
    finally:
        store.close()


def import_prepared(directory, store, expected_identity=None):
    """Validate manifest files and import one document at a time; preserve splits."""
    directory = Path(directory).resolve()
    path = directory / "data_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("ready") is not True:
        raise ValueError(f"source corpus is not marked ready: {directory}")
    tokenizer = manifest.get("tokenizer", {})
    fingerprint = tokenizer.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:"):
        raise ValueError("source manifest lacks explicit tokenizer fingerprint")
    if expected_identity is not None and fingerprint != expected_identity:
        raise ValueError("source tokenizer identities differ")
    if tokenizer.get("name") != TOKENIZER or tokenizer.get("revision") != TOKENIZER_REVISION:
        raise ValueError("source tokenizer is not the pinned deployment tokenizer")
    files = ("train_validation.jsonl", "holdout.jsonl", "selected_texts.jsonl.gz")
    for name in files:
        pinned = manifest["files"][name]
        actual = file_info(directory / name)
        if actual["sha256"] != pinned["sha256"] or actual["size"] != pinned.get("bytes", pinned.get("size")):
            raise ValueError(f"source file integrity check failed: {directory / name}")
    items, by_id = [], {}
    for name in files[:2]:
        with (directory / name).open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                split = record["split"]
                if (name == "holdout.jsonl") != (split == "holdout") or split not in ("train", "validation", "holdout"):
                    raise ValueError("holdout incorrectly included in train/validation file")
                if record.get("tokenizer_identity") != fingerprint:
                    raise ValueError("record tokenizer identity differs from source manifest")
                ids = record["token_ids"]
                if not isinstance(ids, list) or not ids:
                    raise ValueError("invalid pre-tokenized document")
                meta = dict(record["metadata"])
                if not all(re.fullmatch(r"[a-f0-9]{64}", meta.get(key, "")) for key in ("text_sha256", "normalized_text_sha256")):
                    raise ValueError("missing original text hashes")
                if meta.get("token_sha256") and meta["token_sha256"] != digest(ids):
                    raise ValueError("record token hash differs")
                source = "code" if record["source"].startswith("code/") else record["source"]
                meta.update(original_training_source=record["source"], original_group_id=record["group_id"])
                aliases = list(meta.get("family_work_keys", []))
                aliases.append("imported-family:" + record["group_id"])
                if source in ("web", "news"):
                    # A later revision of the same article may share too little
                    # text to pass near-duplicate detection. URL ownership is
                    # therefore an independent cross-source family edge.
                    for key in ("canonical_url", "url"):
                        if meta.get(key):
                            aliases.append("source-url:" + canonical_url(meta[key]))
                if source == "code":
                    aliases.extend("code-repo:" + value for value in meta.get("family_repositories", []))
                item = {"id": record["id"], "source": source, "split": split, "group_id": record["group_id"],
                    "document_id": meta.get("original_document_id", record["id"]), "token_count": len(ids),
                    "work_keys": sorted(set(aliases)), "metadata": meta}
                if item["id"] in by_id:
                    raise ValueError("duplicate source document id")
                by_id[item["id"]] = item
                items.append(item)
                store.put(item["id"], "")
                store.set_tokens(item["id"], ids)
    present = set()
    with gzip.open(directory / "selected_texts.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["id"] not in by_id or record["id"] in present:
                raise ValueError("unexpected or duplicate source text id")
            item = by_id[record["id"]]
            text = record["text"]
            if text_hash(text) != item["metadata"]["text_sha256"] or text_hash(normalize(text)) != item["metadata"]["normalized_text_sha256"]:
                raise ValueError("source text integrity check failed")
            store.update_text(item["id"], text)
            present.add(item["id"])
    if present != set(by_id):
        raise ValueError("missing complete source texts")
    store.db.commit()
    snapshot = manifest.get("source_snapshot")
    if snapshot is None and manifest.get("dataset_id") and manifest.get("dataset_revision"):
        snapshot = source_snapshot(manifest["dataset_id"], manifest["dataset_revision"])
    imported = {"directory": str(directory), "manifest_sha256": file_info(path)["sha256"],
                "source_snapshot": snapshot,
                "format": manifest.get("format"), "files": manifest["files"], "documents": len(items)}
    return items, tokenizer, imported


def quarantine_and_deduplicate(items, edges):
    """Drop all conflicted families; never relabel an imported split to fix leakage.

    Multiple genuinely different documents may remain in one family. Only
    confirmed exact/near duplicates select a single representative, preferring
    existing code so its original cohort is retained where possible.
    """
    family_splits = defaultdict(set)
    for item in items:
        family_splits[item["group_id"]].add(item["split"])
    conflicted = {key for key, splits in family_splits.items() if len(splits) > 1}
    # Per-source validation needs one source per independent family. Preserve
    # the more specific origin when news/papers/books also occur as web copies.
    source_priority = {"code": 0, "papers": 1, "books": 2, "news": 3, "web": 4}
    family_sources = defaultdict(set)
    for item in items:
        family_sources[item["group_id"]].add(item["source"])
    chosen_source = {key: min(values, key=lambda value: (source_priority.get(value, 99), value))
                     for key, values in family_sources.items()}
    duplicates = Families(len(items))
    for left, right, _ in edges:
        duplicates.join(left, right)
    components = defaultdict(list)
    for index, item in enumerate(items):
        if item["group_id"] not in conflicted and item["source"] == chosen_source[item["group_id"]]:
            components[duplicates.find(index)].append(index)
    kept_indices = set()
    for indices in components.values():
        representative = min(indices, key=lambda index: (items[index]["source"] != "code", items[index]["source"], items[index]["id"]))
        kept_indices.add(representative)
    kept, removed = [], []
    for index, item in enumerate(items):
        if index in kept_indices:
            kept.append(item)
        else:
            reason = "family_crosses_imported_splits" if item["group_id"] in conflicted else "cross_source_family" if item["source"] != chosen_source[item["group_id"]] else "confirmed_duplicate"
            removed.append({"id": item["id"], "source": item["source"], "split": item["split"], "group_id": item["group_id"], "reason": reason})
    return kept, {"conflicted_families": len(conflicted), "removed_documents": removed,
                  "cross_source_families": sum(len(values) > 1 for values in family_sources.values()),
                  "cross_source_policy": "Prefer code, papers, books, news, then web; retain only one source per independent family.",
                  "removed_by_source_reason": dict(Counter(f"{row['source']}/{row['reason']}" for row in removed))}


def run_merge(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("data_manifest.json", "train_validation.jsonl", "candidates.sqlite")):
        raise FileExistsError("refusing to replace an existing mixed corpus; use a fresh directory")
    lengths = json.loads(args.source_lengths) if args.source_lengths else DEFAULT_LENGTHS
    weights = json.loads(args.source_weights) if args.source_weights else DEFAULT_WEIGHTS
    if set(weights) != set(lengths) or abs(sum(weights.values()) - 1) > 1e-9 or any(value <= 0 for value in weights.values()):
        raise ValueError("source weights must be positive, sum to one and match source length keys")
    if any(not counts or any(type(value) is not int or value < 2 for value in counts) or len(set(counts)) != len(counts) for counts in lengths.values()):
        raise ValueError("invalid source block counts")
    length_weights = json.loads(args.source_length_weights) if args.source_length_weights else {
        source: ({"2": 0.9, "4": 0.1} if source == "news" and counts == [2, 4]
                 else {str(count): 1 / len(counts) for count in counts}) for source, counts in lengths.items()}
    store = CandidateStore(output / "candidates.sqlite")
    started = time.monotonic()
    try:
        items, imports, ids, tokenizer = [], [], set(), None
        for directory in args.input_dirs:
            imported, identity, provenance = import_prepared(directory, store, tokenizer["fingerprint"] if tokenizer else None)
            new_ids = {item["id"] for item in imported}
            if ids & new_ids:
                raise ValueError("document ids overlap across source corpora")
            ids.update(new_ids)
            tokenizer = identity
            items.extend(imported)
            imports.append(provenance)
            log("source_imported", directory=str(directory), documents=len(imported), raw_tokens=sum(item["token_count"] for item in imported))
        if set(item["source"] for item in items) != set(lengths):
            raise ValueError("imported sources do not exactly match the requested mixture")
        families, edges = form_families(items, store.text, args.near_threshold)
        kept, quarantine = quarantine_and_deduplicate(items, edges)
        save_json(output / "dedup_audit.json", {"statistics": families, **quarantine})
        statistics = validate_and_summarize(kept, lengths, args.minimum_groups)
        try:
            capacity = budget_capacity(statistics, weights, lengths, length_weights,
                                       args.planned_loss_tokens, args.maximum_cell_passes)
        except CapacityError as error:
            save_json(output / "training_capacity_report.json", error.report)
            raise
        save_json(output / "training_capacity_report.json", capacity)
        files = export_files(output, kept, store, tokenizer["fingerprint"])
        manifest = {"format": "delta_ngram_mixed_corpus_v1", "ready": True, "tokenizer": tokenizer,
            "imports": imports, "statistics": statistics, "training_source_weights": weights,
            "training_budget_capacity": capacity, "selected_date_bounds": selected_date_bounds(kept),
            "source_snapshots": [entry["source_snapshot"] for entry in imports if entry["source_snapshot"] is not None],
            "source_block_counts": lengths, "files": files, "deduplication": {**families,
                "threshold": args.near_threshold, "conflicted_families": quarantine["conflicted_families"],
                "cross_source_families": quarantine["cross_source_families"], "cross_source_policy": quarantine["cross_source_policy"],
                "removed_by_source_reason": quarantine["removed_by_source_reason"],
                "policy": "Union imported work families and cross-source exact/near duplicates; quarantine every family spanning imported splits; deduplicate confirmed pairs without relabeling splits."},
            "quality_checks": {"minimum_groups_all_sources_all_splits": args.minimum_groups,
                "all_configured_source_length_cells_nonempty": True, "no_cross_document_concatenation": True,
                "imported_split_assignments_preserved": True, "whole_source_text_hashes_verified": True,
                "holdout_excluded_from_trainer_file": True, "all_tokenizer_identities_match": True},
            "elapsed_seconds": round(time.monotonic() - started, 3), "source_script_sha256": file_info(__file__)["sha256"],
            "limitations": ["Approximate lexical LSH detection can miss near duplicates; no guarantee against semantic or event overlap.",
                "Raw pool proportions are not the loss-token mixture; sampler source and conditional length weights must be configured.",
                "Source-level development groups do not imply equal evidence in each length stratum.",
                "Training pipeline can hold multiple copies of token lists/windows; budget RAM separately from this disk-backed builder."]}
        save_json(output / "data_manifest.json", manifest)
        log("merge_complete", output=str(output), statistics=statistics, removed=quarantine["removed_by_source_reason"])
        return manifest
    finally:
        store.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="pin and verify a bounded source-shard subset")
    download.add_argument("--output-dir", required=True)
    download.add_argument("--sources", nargs="+", choices=tuple(SOURCES), default=list(SOURCES))
    download.add_argument("--seed", type=int, default=17)
    for source, spec in SOURCES.items():
        download.add_argument("--" + source + "-shards", type=int, default=spec["shards"])
    download.add_argument("--max-compressed-bytes", type=int, default=6_000_000_000)
    download.add_argument("--plan-only", action="store_true")
    download.set_defaults(run=run_download)
    prepare = commands.add_parser("prepare", help="prepare one domain from verified local shards")
    prepare.add_argument("--source", choices=tuple(SOURCES), required=True)
    prepare.add_argument("--download-manifest", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--seed", type=int, default=17)
    prepare.add_argument("--candidate-documents", type=int, default=6000)
    prepare.add_argument("--minimum-chars", type=int, default=8000)
    prepare.add_argument("--maximum-chars", type=int, default=2_000_000)
    prepare.add_argument("--minimum-tokens", type=int, default=4096)
    prepare.add_argument("--maximum-tokens", type=int, default=131072)
    prepare.add_argument("--target-raw-tokens", type=int)
    prepare.add_argument("--maximum-raw-tokens", type=int)
    prepare.add_argument("--minimum-groups", type=int, default=64)
    prepare.add_argument("--block-counts", nargs="+", type=int, default=[8, 16])
    prepare.add_argument("--near-threshold", type=float, default=0.85)
    prepare.add_argument("--max-compressed-bytes", type=int, default=6_000_000_000)
    prepare.set_defaults(run=run_prepare)
    merge = commands.add_parser("merge", help="verify prepared corpora, quarantine cross-split duplicates, freeze the mixture")
    merge.add_argument("--input-dirs", nargs="+", required=True)
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--minimum-groups", type=int, default=64)
    merge.add_argument("--source-lengths", help="JSON source -> block-count list; default news2/4, other8/16")
    merge.add_argument("--source-weights", help="JSON source -> loss-token share")
    merge.add_argument("--source-length-weights", help="JSON source -> block count -> conditional loss-token share")
    merge.add_argument("--planned-loss-tokens", type=int, default=100_000_000)
    merge.add_argument("--maximum-cell-passes", type=float, default=3.0)
    merge.add_argument("--near-threshold", type=float, default=0.85)
    merge.set_defaults(run=run_merge)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    try:
        arguments.run(arguments)
    except Exception as error:
        log("failed", error_type=type(error).__name__, error=str(error))
        raise

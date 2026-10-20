#!/usr/bin/env python3
"""Derive a local native-tokenizer cache from a verified frozen M1 corpus.

No network, model, optimizer, DEV64, or sealed FINAL input is used. The two
train_validation files contain OLD token IDs; original text comes from their
SHA-verified selected_texts archives. Old IDs are never decoded to recover text.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
INPUT_FILES = ("train_validation.jsonl", "train_validation_event_streams.jsonl",
               "selected_texts.jsonl.gz", "selected_event_texts.jsonl.gz")
SOURCES = ("books", "code", "papers", "web", "news", "news_event")
LENGTHS = {"books": (8, 16), "code": (8, 16), "papers": (8, 16),
           "web": (8, 16), "news": (2, 4), "news_event": (4, 8, 16, 32)}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "added_tokens.json", "special_tokens_map.json",
                   "chat_template.jinja")
FIRST_ID = re.compile(r'^\s*\{\s*"id"\s*:\s*("(?:[^"\\]|\\.)*")\s*[,}]')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_info(path):
    path = Path(path)
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": result.hexdigest()}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                            allow_nan=False) + "\n")


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def json_lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                require(isinstance(row, dict), f"{path}:{number}: object required")
                yield row


def token_ids(values):
    require(isinstance(values, (list, tuple)) and bool(values), "empty or invalid token IDs")
    require(all(type(value) is int and 0 <= value < 2 ** 32 for value in values),
            "token IDs must be uint32 integers")
    return values


def tokenizer_identity(tokenizer):
    """Same identity fields/hash convention as the existing raw-LM loader."""
    return {"class": type(tokenizer).__name__, "vocab_sha256": digest(tokenizer.get_vocab()),
            "pipeline_sha256": (digest(json.loads(tokenizer.backend_tokenizer.to_str()))
                                if hasattr(tokenizer, "backend_tokenizer") else None),
            "special_tokens": {key: getattr(tokenizer, key, None)
                               for key in ("pad_token_id", "eos_token_id", "bos_token_id")}}


def verify_source(directory, expected_manifest_sha256):
    manifest_path = directory / "data_manifest.json"
    require(valid_hash(expected_manifest_sha256), "expected source manifest SHA256 is required")
    actual = file_info(manifest_path)
    require(actual["sha256"] == expected_manifest_sha256, "source manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("format") == "delta_ngram_event_mixture_v1"
            and manifest.get("ready") is True, "source must be the frozen ready event mixture")
    fingerprint = manifest.get("tokenizer", {}).get("fingerprint")
    require(isinstance(fingerprint, str) and fingerprint.startswith("sha256:")
            and valid_hash(fingerprint[7:]), "source lacks its old tokenizer identity")
    files = {}
    for name in INPUT_FILES:
        pinned = manifest.get("files", {}).get(name, {})
        info = file_info(directory / name)
        require(info["sha256"] == pinned.get("sha256")
                and info["bytes"] == pinned.get("bytes", pinned.get("size")),
                f"source member integrity mismatch: {name}")
        files[name] = info
    return manifest, {"data_manifest.json": actual, **files}


def register_group(groups, identity, split):
    require(isinstance(identity, str) and identity, "missing family/component identity")
    require(groups.setdefault(identity, split) == split, "family/component crosses splits")


def load_frozen_records(directory, old_identity):
    """Keep metadata in memory, but discard old token arrays one record at a time."""
    ordinary, groups, all_ids = {}, {}, set()
    for row in json_lines(directory / INPUT_FILES[0]):
        required = {"id", "group_id", "source", "split", "token_ids",
                    "tokenizer_identity", "metadata"}
        require(set(row) == required, "unexpected frozen ordinary schema")
        require(row["source"] in SOURCES[:-1] and row["split"] in ("train", "validation"),
                "ordinary source/split is outside train/validation")
        require(row["tokenizer_identity"] == old_identity, "old tokenizer identity mismatch")
        require(isinstance(row["id"], str) and row["id"] and row["id"] not in all_ids,
                "duplicate/invalid document ID")
        all_ids.add(row["id"])
        register_group(groups, row["group_id"], row["split"])
        meta = row["metadata"]
        require(isinstance(meta, dict) and valid_hash(meta.get("text_sha256")),
                "missing original text hash")
        if meta.get("audit_component_id"):
            register_group(groups, meta["audit_component_id"], row["split"])
        old_tokens = token_ids(row.pop("token_ids"))
        old_hash = digest(old_tokens)
        require(meta.get("token_sha256", old_hash) == old_hash, "old token hash mismatch")
        row["parent_token_count"], row["parent_token_sha256"] = len(old_tokens), old_hash
        ordinary[row["id"]] = row
    require(bool(ordinary), "ordinary train/validation file is empty")

    # Reuse the existing strict provenance, chronology and boundary validation.
    from delta_ngram.data.event_streams import iter_event_documents
    events, articles = {}, {}
    for event in iter_event_documents(directory / INPUT_FILES[1], tokenizer_identity=old_identity):
        require(event.split in ("train", "validation"), "holdout event is forbidden")
        require(event.id not in all_ids, "event/document ID overlap")
        all_ids.add(event.id)
        register_group(groups, event.group_id, event.split)
        if event.metadata.get("audit_component_id"):
            register_group(groups, event.metadata["audit_component_id"], event.split)
        article_ids = []
        for article in event.articles:
            require(article.id not in all_ids, "article is reused or overlaps a document")
            all_ids.add(article.id)
            article_ids.append(article.id)
            articles[article.id] = {
                "id": article.id, "group_id": event.group_id, "source": "news_event",
                "split": event.split, "event_id": event.id,
                "metadata": copy.deepcopy(article.metadata),
                "parent_token_count": len(article.token_ids),
                "parent_token_sha256": digest(article.token_ids),
            }
        events[event.id] = {
            "id": event.id, "group_id": event.group_id, "source": event.source,
            "split": event.split, "tokenizer_identity": event.tokenizer_identity,
            "metadata": copy.deepcopy(event.metadata), "article_ids": article_ids}
    require(bool(events), "event train/validation file is empty")
    return ordinary, events, articles


def selected_texts(path, wanted):
    """Ignore nonselected archive rows without parsing their text or metadata."""
    seen, skipped = set(), 0
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            match = FIRST_ID.match(line)
            require(match is not None, f"frozen text export must start with id: line {line_number}")
            key = json.loads(match.group(1))
            if key not in wanted:
                skipped += 1
                continue
            require(key not in seen, "duplicate selected text ID")
            row = json.loads(line)
            require(set(row) == {"id", "text", "metadata"}, "invalid selected-text schema")
            text = row["text"]
            require(isinstance(text, str) and text.strip(), "missing complete original text")
            expected = wanted[key]["metadata"]["text_sha256"]
            require(hashlib.sha256(text.encode("utf-8")).hexdigest() == expected,
                    "original text SHA256 mismatch")
            require(isinstance(row["metadata"], dict)
                    and row["metadata"].get("text_sha256") == expected,
                    "text export metadata mismatch")
            seen.add(key)
            yield key, text
    require(seen == set(wanted), "missing selected original texts")
    yield None, skipped


def append_tokens(stream, ids):
    values = array("I", ids)
    require(values.itemsize == 4, "platform uint32 array width is not four")
    if sys.byteorder != "little":
        values.byteswap()
    payload = values.tobytes()
    offset = stream.tell() // 4
    stream.write(payload)
    return {"offset_tokens": offset, "token_count": len(ids),
            "token_bytes_sha256": hashlib.sha256(payload).hexdigest(),
            "token_sha256": digest(ids)}


def read_cached_tokens(path, entry):
    """Read one document/article without loading the complete token cache."""
    with Path(path).open("rb") as stream:
        stream.seek(entry["offset_tokens"] * 4)
        payload = stream.read(entry["token_count"] * 4)
    require(len(payload) == entry["token_count"] * 4, "truncated token cache")
    require(hashlib.sha256(payload).hexdigest() == entry["token_bytes_sha256"],
            "cached token bytes changed")
    values = array("I")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def new_metadata(parent, entry, old_identity):
    meta = copy.deepcopy(parent["metadata"])
    meta.update(token_count=entry["token_count"], token_sha256=entry["token_sha256"],
                parent_tokenizer_identity=old_identity,
                parent_token_count=parent["parent_token_count"],
                parent_token_sha256=parent["parent_token_sha256"])
    return meta


def family_priority(seed, source, group_id):
    return digest([seed, source, group_id])


def calibration_selection(candidates, seed, groups_per_source):
    chosen, deficits = [], {}
    for source in SOURCES:
        by_group = defaultdict(list)
        for row in candidates:
            if row["source"] == source and row["split"] == "train" and row["eligible"]:
                by_group[row["group_id"]].append(row)
        ordered = sorted(by_group, key=lambda group: (family_priority(seed, source, group), group))
        for group in ordered[:groups_per_source]:
            row = min(by_group[group], key=lambda r: (digest([seed, r["id"]]), r["id"]))
            chosen.append({**row, "sample_id": "calibration:" + digest(
                [seed, source, group, row["id"]]), "selection_hash": family_priority(seed, source, group)})
        if len(ordered) < groups_per_source:
            deficits[source] = groups_per_source - len(ordered)
    return {"protocol": "qwen38_native_train_calibration_selection_v1", "split": "train",
            "seed": seed, "groups_per_source_requested": groups_per_source,
            "policy": "Hash-rank eligible training families, then hash-rank one document/stream "
                      "per family. Ordinary samples use the first complete registered window.",
            "samples": chosen, "deficits": deficits, "ready": not deficits,
            "model_results_seen": False, "calibration_executed": False}


def prepare(source_dir, output_dir, *, expected_manifest_sha256, tokenizer,
            tokenizer_receipt, seed=17, calibration_groups=2, calibration_only=False,
            progress=None):
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    require(source != output and source not in output.parents,
            "derived output must be outside the frozen source directory")
    require(not output.exists(), "output exists; use a new derived root")
    require(type(seed) is int and type(calibration_groups) is int and calibration_groups > 0,
            "invalid calibration selection parameters")
    require(tokenizer_receipt.get("model_id") == MODEL_ID
            and tokenizer_receipt.get("revision") == REVISION, "native tokenizer revision mismatch")
    parent, input_files = verify_source(source, expected_manifest_sha256)
    ordinary, events, articles = load_frozen_records(source, parent["tokenizer"]["fingerprint"])
    parent_counts = {"documents": len(ordinary), "event_streams": len(events),
                     "event_articles": len(articles)}
    if calibration_only:
        # Freeze families before any new-tokenizer or model observation. Failure
        # to fit a native window is reported, not used to silently replace one.
        parent_candidates = [
            {"id": row["id"], "group_id": row["group_id"], "source": row["source"],
             "split": row["split"], "eligible": True}
            for row in (*ordinary.values(), *events.values())]
        initial_selection = calibration_selection(parent_candidates, seed, calibration_groups)
        require(initial_selection["ready"], "parent training pool lacks calibration families")
        selected_ids = {row["id"] for row in initial_selection["samples"]}
        ordinary = {key: row for key, row in ordinary.items() if key in selected_ids}
        events = {key: row for key, row in events.items() if key in selected_ids}
        selected_articles = {key for event in events.values() for key in event["article_ids"]}
        articles = {key: row for key, row in articles.items() if key in selected_articles}
    identity = tokenizer_identity(tokenizer)
    fingerprint = "sha256:" + digest(identity)
    old_identity = parent["tokenizer"]["fingerprint"]
    require(fingerprint != old_identity, "native tokenizer identity equals the old tokenizer")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "INCOMPLETE.json", {"status": "preparing", "source_files": input_files,
                                         "tokenizer": tokenizer_receipt})
    indexed, skipped = {}, {}
    source_stats = defaultdict(lambda: {"documents": 0, "raw_tokens": 0,
                                       "parent_raw_tokens": 0, "groups": set()})
    candidates, violations = [], []
    processed = 0
    try:
        with (output / "tokens.uint32").open("xb") as cache:
            for text_file, records in ((INPUT_FILES[2], ordinary), (INPUT_FILES[3], articles)):
                for key, text in selected_texts(source / text_file, records):
                    if key is None:
                        skipped[text_file] = text
                        continue
                    parent_row = records[key]
                    ids = token_ids(tokenizer.encode(text, add_special_tokens=False))
                    entry = append_tokens(cache, ids)
                    entry.update(id=key, group_id=parent_row["group_id"], source=parent_row["source"],
                                 split=parent_row["split"], text_sha256=parent_row["metadata"]["text_sha256"],
                                 raw_text_characters=len(text), parent_metadata=parent_row["metadata"])
                    if "event_id" in parent_row:
                        entry["event_id"] = parent_row["event_id"]
                    entry["metadata"] = new_metadata(parent_row, entry, old_identity)
                    indexed[key] = entry
                    processed += 1
                    if progress and processed % 500 == 0:
                        progress({"stage": "tokenize", "documents_and_articles": processed})
        cache_path = output / "tokens.uint32"
        with (output / "document_index.jsonl").open("x", encoding="utf-8", newline="\n") as index, \
             (output / INPUT_FILES[0]).open("x", encoding="utf-8", newline="\n") as runtime, \
             (output / "windows.jsonl").open("x", encoding="utf-8", newline="\n") as windows:
            for key, parent_row in ordinary.items():
                entry = indexed[key]
                write_line(index, entry)
                ids = read_cached_tokens(cache_path, entry)
                write_line(runtime, {key: parent_row[key] for key in
                                     ("id", "group_id", "source", "split")} |
                           {"tokenizer_identity": fingerprint, "token_ids": ids,
                            "metadata": entry["metadata"]})
                cell = source_stats[(entry["source"], entry["split"])]
                cell["documents"] += 1
                cell["raw_tokens"] += entry["token_count"]
                cell["parent_raw_tokens"] += parent_row["parent_token_count"]
                cell["groups"].add(entry["group_id"])
                for blocks in LENGTHS[entry["source"]]:
                    size = 512 * blocks
                    count = entry["token_count"] // size
                    cell.setdefault("window_count", Counter())[str(blocks)] += count
                    cell.setdefault("window_loss_tokens", Counter())[str(blocks)] += count * (blocks - 1) * 512
                    cell.setdefault("unused_tail_tokens", Counter())[str(blocks)] += entry["token_count"] % size
                    for start in range(0, count * size, size):
                        write_line(windows, {"kind": "ordinary", "id": key, "group_id": entry["group_id"],
                                            "source": entry["source"], "split": entry["split"],
                                            "start": start, "end": start + size, "blocks": blocks,
                                            "loss_tokens": (blocks - 1) * 512,
                                            "cache_offset_tokens": entry["offset_tokens"] + start})
                first_blocks = min(LENGTHS[entry["source"]])
                candidates.append({"id": key, "group_id": entry["group_id"], "source": entry["source"],
                                   "split": entry["split"], "kind": "ordinary",
                                   "eligible": entry["token_count"] >= first_blocks * 512,
                                   "text_sha256": entry["text_sha256"], "token_sha256": entry["token_sha256"],
                                   "offset_tokens": entry["offset_tokens"], "start": 0,
                                   "end": first_blocks * 512, "blocks": first_blocks,
                                   "raw_document_characters": entry["raw_text_characters"]})
            with (output / "event_index.jsonl").open("x", encoding="utf-8", newline="\n") as event_index, \
                 (output / INPUT_FILES[1]).open("x", encoding="utf-8", newline="\n") as event_runtime:
                for key, event in events.items():
                    members = [indexed[aid] for aid in event["article_ids"]]
                    lengths = [row["token_count"] for row in members]
                    chunks = sum((length + 511) // 512 for length in lengths)
                    bucket = next((n for n in LENGTHS["news_event"] if n >= chunks), None)
                    eligible = (all(128 <= length <= 16384 for length in lengths)
                                and sum(lengths) <= 16384 and bucket is not None)
                    if not eligible:
                        violations.append({"id": key, "split": event["split"],
                                           "reason": "native_event_length_outside_frozen_runtime_limits",
                                           "article_token_counts": lengths, "article_chunks": chunks})
                    offset, boundaries = 0, []
                    for i, member in enumerate(members):
                        end = offset + member["token_count"]
                        boundaries.append({"article_id": member["id"], "start": offset, "end": end,
                                           "role": "history" if i == 0 else "target"})
                        offset = end
                    meta = copy.deepcopy(event["metadata"])
                    meta.update(article_boundaries=boundaries, raw_tokens=sum(lengths),
                                target_tokens=sum(lengths[1:]), article_chunk_count=chunks,
                                block_bucket=bucket, block_size=512)
                    derived = {name: event[name] for name in ("id", "group_id", "source", "split")}
                    derived.update(tokenizer_identity=fingerprint, metadata=meta)
                    write_line(event_index, {**derived, "articles": members,
                                             "parent_metadata": event["metadata"], "eligible": eligible})
                    write_line(event_runtime, {**derived, "articles": [
                        {"id": member["id"], "token_ids": read_cached_tokens(cache_path, member),
                         "metadata": member["metadata"]} for member in members]})
                    write_line(windows, {"kind": "event", "id": key, "group_id": event["group_id"],
                                        "source": "news_event", "split": event["split"],
                                        "eligible": eligible, "blocks": bucket, "actual_blocks": chunks,
                                        "loss_tokens": sum(lengths[1:]), "article_ids": event["article_ids"]})
                    cell = source_stats[("news_event", event["split"])]
                    cell["documents"] += 1
                    cell["raw_tokens"] += sum(lengths)
                    cell["parent_raw_tokens"] += sum(articles[m["id"]]["parent_token_count"] for m in members)
                    cell["groups"].add(event["group_id"])
                    if eligible:
                        cell.setdefault("window_count", Counter())[str(bucket)] += 1
                        cell.setdefault("window_loss_tokens", Counter())[str(bucket)] += sum(lengths[1:])
                    candidates.append({"id": key, "group_id": event["group_id"], "source": "news_event",
                                       "split": event["split"], "kind": "event", "eligible": eligible,
                                       "blocks": bucket, "article_ids": event["article_ids"],
                                       "article_text_sha256": [m["text_sha256"] for m in members],
                                       "article_token_sha256": [m["token_sha256"] for m in members]})
        selection = calibration_selection(candidates, seed, calibration_groups)
        selection.update(source_manifest_sha256=expected_manifest_sha256,
                         tokenizer_fingerprint=fingerprint,
                         selection_scope=("fixed_parent_training_family_hash_prefix_no_replacement"
                                          if calibration_only else "complete_derived_training_pool"))
        write_json(output / "calibration_selection.json", selection)
        stats = {}
        for (source_name, split), cell in sorted(source_stats.items()):
            count = len(cell.pop("groups"))
            stats.setdefault(source_name, {})[split] = {**cell, "independent_groups": count}
        if not calibration_only:
            for source_name in SOURCES:
                for blocks in LENGTHS[source_name]:
                    cell = stats.get(source_name, {}).get("train", {})
                    if cell.get("window_loss_tokens", {}).get(str(blocks), 0) == 0:
                        violations.append({"source": source_name, "blocks": blocks,
                                           "reason": "registered_training_cell_has_no_native_targets"})
        files = {p.name: file_info(p) for p in sorted(output.iterdir())
                 if p.is_file() and p.name != "INCOMPLETE.json"}
        manifest = {
            "format": "qwen38_native_derived_data_v1", "derived_cache_complete": True,
            "ready_for_training": not calibration_only and not violations and selection["ready"],
            "ready_for_calibration": not violations and selection["ready"],
            "derivation_scope": "calibration_subset_only" if calibration_only else "full_train_validation",
            "parent_membership_counts": parent_counts,
            "native_runtime_accepted": False, "model": {"name": MODEL_ID, "revision": REVISION},
            "source_directory": str(source), "source_files": input_files,
            "source_manifest_sha256": expected_manifest_sha256,
            "tokenizer": {**tokenizer_receipt, "identity": identity, "fingerprint": fingerprint},
            "token_cache": {"file": "tokens.uint32", "dtype": "uint32", "byte_order": "little",
                            "offset_unit": "tokens", "special_tokens_added": False},
            "files": files, "statistics": stats, "violations": violations,
            "source_weights": parent.get("training_source_weights"),
            "source_length_weights": parent.get("resolved_raw_config_proposal", {}).get("source_length_weights"),
            "retokenized_documents": len(ordinary), "retokenized_event_streams": len(events),
            "retokenized_event_articles": len(articles),
            "skipped_nonselected_text_rows": skipped,
            "scope": "Frozen train/validation document and event membership only. Text archives "
                     "are streamed; nonselected rows are not JSON-parsed or tokenized. "
                     "Sealed FINAL and DEV64 are not inputs.",
            "frozen_membership_modified": False, "subset_selected": calibration_only,
            "old_tokens_modified": False, "source_fetches": 0,
            "split_policy": "Preserve input IDs, source, group, audit component and split.",
            "limitations": ["No native model forward or calibration has run.",
                           "Length/capacity violations are reported without dropping or truncating records.",
                           "Training budgets and repeat-consumption capacity must be registered separately.",
                           "This tool does not reclassify DEV64 prefixes or certify pretraining novelty."],
            "source_script_sha256": file_info(__file__)["sha256"],
        }
        write_json(output / "data_manifest.json", manifest)
        # Catch source replacement during preparation before sealing any result.
        _, final_inputs = verify_source(source, expected_manifest_sha256)
        require(final_inputs == input_files, "frozen source changed during preparation")
        write_json(output / "COMPLETE.json", {
            "format": "qwen38_native_derived_seal_v1", "data_manifest": file_info(output / "data_manifest.json"),
            "files": files, "ready_for_training": manifest["ready_for_training"],
            "ready_for_calibration": manifest["ready_for_calibration"],
            "derivation_scope": manifest["derivation_scope"]})
        (output / "INCOMPLETE.json").unlink()
        return manifest
    except Exception as error:
        write_json(output / "FAILURE.json", {"status": "incomplete", "error_type": type(error).__name__,
                                          "message": str(error), "source_manifest_sha256": expected_manifest_sha256})
        raise


def local_tokenizer_receipt(directory, *, expected_manifest=None):
    """Bind local tokenizer files to a pinned snapshot or explicit download receipt."""
    directory = Path(directory).resolve()
    require(directory.is_dir(), "tokenizer directory is missing")
    files = {name: file_info(directory / name) for name in TOKENIZER_FILES if (directory / name).is_file()}
    require("tokenizer_config.json" in files and (
        "tokenizer.json" in files or {"vocab.json", "merges.txt"} <= files.keys()),
        "local tokenizer files are incomplete")
    if expected_manifest is not None:
        supplied = json.loads(Path(expected_manifest).read_text(encoding="utf-8"))
        require(supplied.get("model_id") == MODEL_ID and supplied.get("revision") == REVISION,
                "tokenizer download receipt identifies a different model/revision")
        require(supplied.get("files") == files, "tokenizer file hashes differ from download receipt")
        binding = {"kind": "explicit_download_receipt", "sha256": file_info(expected_manifest)["sha256"]}
    else:
        require(directory.name == REVISION and directory.parent.name == "snapshots",
                "use the pinned HF snapshots/<revision> directory or --tokenizer-files-manifest")
        binding = {"kind": "hf_snapshot_path"}
    return {"model_id": MODEL_ID, "revision": REVISION, "files": files,
            "download_identity_binding": binding}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", default=REVISION, choices=(REVISION,))
    parser.add_argument("--tokenizer-files-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--calibration-groups-per-source", type=int, default=2)
    parser.add_argument("--calibration-only", action="store_true",
                        help="derive only the fixed train family hash prefix, not the full pool")
    args = parser.parse_args()
    receipt = local_tokenizer_receipt(args.tokenizer_dir, expected_manifest=args.tokenizer_files_manifest)
    from transformers import AutoTokenizer
    import transformers
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_dir), local_files_only=True, trust_remote_code=False)
    receipt["transformers_version"] = transformers.__version__
    started = time.monotonic()
    result = prepare(args.source_dir, args.output_dir,
                     expected_manifest_sha256=args.source_manifest_sha256,
                     tokenizer=tokenizer, tokenizer_receipt=receipt, seed=args.seed,
                     calibration_groups=args.calibration_groups_per_source,
                     calibration_only=args.calibration_only,
                     progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({"output": str(args.output_dir), "ready_for_training": result["ready_for_training"],
                      "ready_for_calibration": result["ready_for_calibration"],
                      "derivation_scope": result["derivation_scope"],
                      "documents": result["retokenized_documents"],
                      "event_streams": result["retokenized_event_streams"],
                      "elapsed_seconds": time.monotonic() - started}))
    return 0 if result["ready_for_calibration" if args.calibration_only else "ready_for_training"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

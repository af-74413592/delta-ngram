"""Native causal frames over the existing frozen M1 corpus and sampler.

This module does not load a model, tokenize text, or implement a new sampling
strategy. The sealed native-tokenizer JSONL files are consumed by the existing
raw_corpora loaders and RawWindowSampler (including its variable-cost event
path). A trajectory is one independent memory owner.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .data.event_streams import EventTrajectory
from .data.raw_text import RawWindowSampler, TrajectoryWindow
from .raw_corpora import load_corpora, trajectory_pools


MODEL = "Qwen/Qwen3.8-Flash-Next"
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
BLOCK = 512
PREFIX = 128
BOS = PAD = 248044
VOCABULARY = 248320
SOURCE_LENGTHS = {
    "books": (8, 16), "code": (8, 16), "papers": (8, 16),
    "web": (8, 16), "news": (2, 4), "news_event": (4, 8, 16, 32),
}
SOURCE_WEIGHTS = {
    "code": .20, "papers": .25, "books": .20, "web": .15,
    "news": .175, "news_event": .025,
}
REQUIRED_FILES = {
    "tokens.uint32", "document_index.jsonl", "event_index.jsonl",
    "windows.jsonl", "train_validation.jsonl",
    "train_validation_event_streams.jsonl", "calibration_selection.json",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()


def file_info(path):
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": hasher.hexdigest()}


def verify_full_cache(root, *, expected_complete_sha256=None,
                      expected_manifest_sha256=None, verify_files=True):
    """Verify the full train/validation seal without opening DEV64 or FINAL.

    ``verify_files=False`` is a size-only worker-side check, for callers that
    separately verify all file bytes. The returned identity says so explicitly.
    Default verification hashes every file listed by the immutable seal.
    """
    root = Path(root).resolve()
    _require(type(verify_files) is bool, "verify_files must be boolean")
    _require(not (root / "INCOMPLETE.json").exists(), "native cache is incomplete")
    complete_info = file_info(root / "COMPLETE.json")
    manifest_info = file_info(root / "data_manifest.json")
    for supplied, actual, name in (
        (expected_complete_sha256, complete_info["sha256"], "complete seal"),
        (expected_manifest_sha256, manifest_info["sha256"], "data manifest"),
    ):
        _require(supplied is None or supplied == actual, f"native {name} SHA256 differs")
    complete = json.loads((root / "COMPLETE.json").read_bytes())
    manifest = json.loads((root / "data_manifest.json").read_bytes())
    _require(complete.get("format") == "qwen38_native_derived_seal_v1"
             and complete.get("ready_for_training") is True
             and complete.get("derivation_scope") == "full_train_validation"
             and complete.get("data_manifest") == manifest_info,
             "requires the sealed full native training cache")
    _require(manifest.get("format") == "qwen38_native_derived_data_v1"
             and manifest.get("derived_cache_complete") is True
             and manifest.get("ready_for_training") is True
             and manifest.get("derivation_scope") == "full_train_validation"
             and manifest.get("model") == {"name": MODEL, "revision": REVISION}
             and manifest.get("frozen_membership_modified") is False
             and manifest.get("subset_selected") is False
             and not manifest.get("violations"),
             "full native cache model, scope or readiness differs")
    _require(complete.get("files") == manifest.get("files")
             and isinstance(manifest.get("files"), dict)
             and REQUIRED_FILES.issubset(manifest["files"]),
             "native cache seal file list differs")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        _require(path.is_relative_to(root) and path.is_file()
                 and isinstance(expected, dict) and set(expected) == {"bytes", "sha256"}
                 and type(expected["bytes"]) is int and expected["bytes"] >= 0
                 and isinstance(expected["sha256"], str)
                 and re.fullmatch("[0-9a-f]{64}", expected["sha256"])
                 and path.stat().st_size == expected["bytes"],
                 f"native cache file identity differs: {name}")
        if verify_files:
            _require(file_info(path) == expected, f"native cache file hash differs: {name}")
    tokenizer = manifest.get("tokenizer", {})
    fingerprint = tokenizer.get("fingerprint")
    _require(isinstance(fingerprint, str)
             and re.fullmatch("sha256:[0-9a-f]{64}", fingerprint),
             "native tokenizer fingerprint is missing")
    _require(manifest.get("source_weights") == SOURCE_WEIGHTS,
             "native source weights differ from the inherited six-source mixture")
    supplied_lengths = manifest.get("source_length_weights")
    _require(isinstance(supplied_lengths, dict)
             and set(supplied_lengths) == set(SOURCE_LENGTHS),
             "native source-length weights are missing")
    source_lengths = {}
    for source, expected_counts in SOURCE_LENGTHS.items():
        pairs = supplied_lengths[source]
        _require(isinstance(pairs, dict), "source-length weights must be mappings")
        converted = {int(key): value for key, value in pairs.items()}
        _require(len(converted) == len(pairs) and set(converted) == set(expected_counts),
                 f"native length cells differ: {source}")
        source_lengths[source] = converted
    identity = {
        "protocol": "qwen38_native_full_training_data_v1",
        "complete": complete_info, "data_manifest": manifest_info,
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "tokenizer_fingerprint": fingerprint,
        "source_weights": manifest["source_weights"],
        "source_length_weights": source_lengths,
        "all_file_hashes_verified": verify_files,
        "train_split_only": True,
        "sampling": "existing RawWindowSampler / VariableCostSampler v3",
        "frame_protocol": "qwen38_native_full_trajectory_frames_v1",
        "block_size": BLOCK, "prefix_tokens": PREFIX,
        "BOS_and_padding_id": BOS,
    }
    return manifest, identity


class NativeFullTrainingData:
    """Existing complete-window sampler plus native model-ready frame plans."""

    def __init__(self, root, *, expected_complete_sha256=None,
                 expected_manifest_sha256=None, verify_files=True):
        self.root = Path(root).resolve()
        self.manifest, self.identity = verify_full_cache(
            self.root, expected_complete_sha256=expected_complete_sha256,
            expected_manifest_sha256=expected_manifest_sha256, verify_files=verify_files)
        self.raw = {
            "dataset": str(self.root / "train_validation.jsonl"),
            "event_dataset": str(self.root / "train_validation_event_streams.jsonl"),
            "block_size": BLOCK, "prefix_tokens": PREFIX,
            "block_counts": [2, 4, 8, 16, 32], "length_weights": None,
            "source_weights": self.manifest["source_weights"],
            "source_length_weights": self.identity["source_length_weights"],
        }
        # All records already contain native IDs. No tokenizer is supplied.
        self.documents, self.events = load_corpora(
            self.raw, None, self.identity["tokenizer_fingerprint"],
            vocabulary_size=VOCABULARY)
        _require(len(self.documents) == self.manifest["retokenized_documents"]
                 and len(self.events) == self.manifest["retokenized_event_streams"],
                 "native corpus membership counts differ from its manifest")
        self.pools = trajectory_pools(self.raw, self.documents, self.events, split="train")
        self._training_documents = {document.id for document in self.documents if document.split == "train"}
        self._training_events = {event.id for event in self.events if event.split == "train"}

    def sampler(self, seed=17):
        return RawWindowSampler(
            self.pools, seed=seed, length_weights=None,
            source_weights=self.raw["source_weights"],
            source_length_weights=self.raw["source_length_weights"])

    def frames(self, trajectory):
        allowed = (self._training_events if isinstance(trajectory, EventTrajectory)
                   else self._training_documents)
        _require(trajectory.document_id in allowed, "trajectory is outside the training split")
        return trajectory_frames(trajectory)


def _frame(metadata, article_tokens, start, end, *, query):
    """Build one SP-even frame, with labels already aligned to native logits."""
    current = list(article_tokens[start:end])
    if start:
        prefix_start = max(0, start - PREFIX)
        prefix = list(article_tokens[prefix_start:start])
    else:
        prefix_start = 0
        prefix = [BOS] if query else []
    ids = prefix + current
    unpadded = len(ids)
    padding = unpadded % 2
    ids += [PAD] * padding
    valid = [True] * unpadded + [False] * padding
    write = ([False] * len(ids) if query else
             [False] * len(prefix) + [True] * len(current) + [False] * padding)
    result = {
        **metadata, "owner_id": 0,
        "article_token_start": start, "article_token_end": end,
        "prefix_token_start": prefix_start, "prefix_tokens": len(prefix),
        "new_tokens": len(current), "padding_tokens": padding,
        "input_ids": ids, "encoder_valid_mask": valid, "write_mask": write,
        "position_ids": list(range(len(ids))),
        "article_initial_chunk": start == 0,
        "loss_target_tokens": len(current) if query else 0,
    }
    if query:
        _require(bool(prefix), "a scored block needs a local prediction prefix")
        labels = [-100] * len(ids)
        labels[len(prefix) - 1:len(prefix) - 1 + len(current)] = current
        result["labels"] = labels
    else:
        result["loss_labels"] = None
    _require(0 < len(ids) <= BLOCK + PREFIX and len(ids) % 2 == 0
             and all(type(token) is int and 0 <= token < VOCABULARY for token in ids)
             and (sum(label != -100 for label in result["labels"]) == len(current)
                  if query else sum(write) == len(current)),
             "native causal frame alignment or token IDs are invalid")
    result["frame_sha256"] = canonical_hash(result)
    return result


def trajectory_frames(trajectory):
    """Return chronological ``query`` then ``evidence`` actions for one owner.

    Ordinary block zero is warmup. An event's entire first article is warmup.
    Each later block is scored before it is written, and the last block is never
    written. Event article changes reset local prefix/hash context, retaining
    the same memory owner. Padding and old prefix positions are never written.
    """
    _require(isinstance(trajectory, (TrajectoryWindow, EventTrajectory))
             and trajectory.block_size == BLOCK, "native training requires complete 512-token trajectories")
    is_event = isinstance(trajectory, EventTrajectory)
    kind = "event" if is_event else "ordinary"
    token_hash = canonical_hash(trajectory.token_ids)
    trajectory_identity = {
        "kind": kind, "document_id": trajectory.document_id,
        "group_id": trajectory.group_id, "source": trajectory.source,
        "window_start": trajectory.start, "window_blocks": trajectory.blocks,
        "token_sha256": token_hash,
    }
    sample_id = "train:" + canonical_hash(trajectory_identity)
    articles = ([(article.id, article.token_ids) for article in trajectory.articles]
                if is_event else [(trajectory.document_id, trajectory.token_ids)])
    chunks = [(article_index, article_id, tokens, start, min(start + BLOCK, len(tokens)))
              for article_index, (article_id, tokens) in enumerate(articles)
              for start in range(0, len(tokens), BLOCK)]
    steps = []
    for ordinal, (article_index, article_id, tokens, start, end) in enumerate(chunks):
        scored = article_index > 0 if is_event else ordinal > 0
        should_write = ordinal + 1 < len(chunks)
        metadata = {
            **trajectory_identity, "sample_id": sample_id, "frame_index": ordinal,
            "article_index": article_index, "article_id": article_id,
        }
        steps.append({
            "query": _frame(metadata, tokens, start, end, query=True) if scored else None,
            "evidence": _frame(metadata, tokens, start, end, query=False) if should_write else None,
            "metadata": {**metadata, "query_before_write": True,
                         "memory_owner_reset": ordinal == 0,
                         "article_boundary_reset": start == 0,
                         "loss_target_tokens": end - start if scored else 0,
                         "write_tokens": end - start if should_write else 0},
        })
    _require(sum(step["metadata"]["loss_target_tokens"] for step in steps) == trajectory.loss_tokens,
             "native trajectory target count differs from the existing sampler")
    expected_writes = trajectory.writer_tokens if is_event else len(trajectory.token_ids) - BLOCK
    _require(sum(step["metadata"]["write_tokens"] for step in steps) == expected_writes,
             "native trajectory write count differs from complete chronological history")
    return steps

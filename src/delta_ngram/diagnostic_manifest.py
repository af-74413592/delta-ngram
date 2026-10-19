"""Freeze loss-independent, document-grouped P1 diagnostic targets and donors.

The saved raw-LM validation sample defines the primary cohort. Donor matching
never changes it. After-write targets only extend those same document/start
pairs to the next complete natural block; unavailable points remain explicit.
This module uses token IDs already loaded from the corpus and never tokenizes,
decodes, reads holdout data, evaluates a model, or consumes training RNG state.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from typing import Mapping, Sequence

from .data.raw_text import RawDocument


MANIFEST_FORMAT = "delta_ngram_p1_diagnostics_manifest_v1"
_SAMPLE_KEYS = {"document_id", "group_id", "source", "start", "blocks",
                "loss_tokens", "token_sha256"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _optional_hash(value, name):
    if value is not None and (not isinstance(value, str) or len(value) != 64
                              or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{name} must be null or a lowercase SHA256 hex digest")


def _document_index(documents):
    indexed, group_splits, identities = {}, {}, set()
    for document in documents:
        if document.id in indexed:
            raise ValueError(f"duplicate document id: {document.id}")
        if document.split not in ("train", "validation"):
            raise ValueError("diagnostics accept train/validation documents only; never holdout")
        if not document.tokenizer_identity:
            raise ValueError("diagnostic documents require explicit tokenizer_identity")
        identities.add(document.tokenizer_identity)
        if group_splits.setdefault(document.group_id, document.split) != document.split:
            raise ValueError("document family crosses train/validation splits")
        indexed[document.id] = document
    if not indexed or len(identities) != 1:
        raise ValueError("diagnostics require nonempty documents with one tokenizer_identity")
    return indexed, next(iter(identities))


def _family_repositories(document):
    values = document.metadata.get("family_repositories", [])
    if not isinstance(values, (tuple, list)):
        raise ValueError("family_repositories must be a list when supplied")
    result = {value.strip().casefold() for value in values if isinstance(value, str) and value.strip()}
    own = document.metadata.get("repo_normalized")
    if isinstance(own, str) and own.strip():
        result.add(own.strip().casefold())
    return result


def _related(left, right):
    """Exclude known families and exact copies, without claiming global dedup."""
    if left.group_id == right.group_id or left.token_sha256 == right.token_sha256:
        return True
    if _family_repositories(left) & _family_repositories(right):
        return True
    for key in ("text_sha256", "normalized_text_sha256"):
        a, b = left.metadata.get(key), right.metadata.get(key)
        if a and a == b:
            return True
    return False


def _match_donors(recipient, window, documents, *, block_size, required, seed,
                  max_document_length_ratio, donor_pool_policy):
    history_blocks = window["blocks"] - 1
    history_tokens = history_blocks * block_size
    eligible = []
    for document in documents:
        if (document.source != recipient.source or _related(recipient, document)
                or len(document.token_ids) < history_tokens):
            continue
        if document.split != "validation" and donor_pool_policy == "validation_only":
            continue
        ratio = len(document.token_ids) / len(recipient.token_ids)
        if not 1 / max_document_length_ratio <= ratio <= max_document_length_ratio:
            continue
        eligible.append((document, ratio))
    # Validation is preferred categorically; only a shortage uses train donors.
    # Within a split, nearest whole-document length wins before a seeded tie.
    eligible.sort(key=lambda item: (
        item[0].split != "validation", abs(math.log(item[1])),
        _digest([seed, "donor-choice", window["window_id"], item[0].group_id, item[0].id])))
    chosen, donors = [], []
    for document, ratio in eligible:
        if any(_related(document, previous) for previous in chosen):
            continue
        available_starts = (len(document.token_ids) - history_tokens) // block_size + 1
        start_index = int(_digest([seed, "donor-start", window["window_id"], document.id]), 16) % available_starts
        start = start_index * block_size
        span = document.token_ids[start:start + history_tokens]
        donors.append({"document_id": document.id, "group_id": document.group_id,
                       "source": document.source, "split": document.split, "start": start,
                       "history_blocks": history_blocks, "token_sha256": _digest(span),
                       "document_tokens": len(document.token_ids), "document_length_ratio": ratio})
        chosen.append(document)
        if len(donors) == required:
            break
    fallback = sum(donor["split"] == "train" for donor in donors)
    matched = len(donors) == required
    return {"donors": donors, "donors_required": required,
            "donor_status": "matched" if matched else "insufficient_donors",
            "eligible_donor_groups": len({item.group_id for item, _ in eligible}),
            "eligible_validation_donor_groups": len({item.group_id for item, _ in eligible
                                                       if item.split == "validation"}),
            "all_validation_donors": matched and fallback == 0,
            "donor_train_fallback_count": fallback,
            "donor_fallback_reason": ("fewer_than_required_eligible_validation_donor_families"
                                      if fallback else None)}


def _primary_counts(items):
    matched = [item for item in items if item["donor_status"] == "matched"]
    all_dev = [item for item in items if item["all_validation_donors"]]
    return {"windows": len(items), "groups": len({item["group_id"] for item in items}),
            "documents": len({item["document_id"] for item in items}),
            "loss_tokens": sum(item["loss_tokens"] for item in items),
            "donor_matched_windows": len(matched), "donor_unmatched_windows": len(items) - len(matched),
            "donor_matched_loss_tokens": sum(item["loss_tokens"] for item in matched),
            "donor_train_fallback_slots": sum(item["donor_train_fallback_count"] for item in items),
            "donor_train_fallback_windows": sum(item["donor_train_fallback_count"] > 0 for item in items),
            "all_validation_donor_windows": len(all_dev),
            "all_validation_donor_groups": len({item["group_id"] for item in all_dev}),
            "all_validation_donor_loss_tokens": sum(item["loss_tokens"] for item in all_dev)}


def _after_counts(items, block_size):
    available = [item for item in items if item["status"] == "available"]
    return {"requested_targets": len(items), "available_targets": len(available),
            "unavailable_targets": len(items) - len(available),
            "requested_groups": len({item["group_id"] for item in items}),
            "available_groups": len({item["group_id"] for item in available}),
            "loss_tokens": len(available) * block_size,
            "target_coverage_fraction": len(available) / len(items) if items else 0.0}


def _stratified(items, summarize):
    result = summarize(items)
    for key, label in (("source", "by_source"), ("group_id", "by_group")):
        groups = defaultdict(list)
        for item in items:
            groups[item[key]].append(item)
        result[label] = {name: summarize(groups[name]) for name in sorted(groups)}
    return result


def build_diagnostic_manifest(
        documents: Sequence[RawDocument], validation_sample: Sequence[Mapping], *,
        block_size: int = 512, prefix_tokens: int = 128, seed: int = 104729,
        donors_per_target: int = 2, after_write_blocks: Sequence[int] = (4, 8, 16),
        max_document_length_ratio: float = 2.0, donor_pool_policy: str = "validation_first",
        dataset_sha256: str | None = None,
        baseline_validation_protocol_sha256: str | None = None) -> dict:
    """Build a frozen manifest without selecting targets or donors using losses.

    ``validation_first`` fills only missing donor slots from the training split;
    ``validation_only`` explicitly leaves unsupported primary donor results
    unavailable. All other primary conditions still use the complete baseline
    cohort. Whole donor-document length must be within the fixed ratio bound;
    the actual written donor span is always exactly (T-1) complete blocks.

    ``dataset_sha256`` is an optional externally verified JSONL byte hash.
    ``source_documents_sha256`` in the result is a separate logical identity of
    sorted document metadata and full token hashes, never a file-byte hash.
    """
    _integer(block_size, "block_size", 1)
    _integer(prefix_tokens, "prefix_tokens", 1)
    _integer(seed, "seed", 0)
    _integer(donors_per_target, "donors_per_target", 2)
    if prefix_tokens > block_size:
        raise ValueError("prefix_tokens must not exceed block_size")
    points = list(after_write_blocks)
    if not points or len(set(points)) != len(points):
        raise ValueError("after_write_blocks must be nonempty and unique")
    for point in points:
        _integer(point, "after_write_blocks", 1)
    points.sort()
    if (type(max_document_length_ratio) not in (int, float)
            or not math.isfinite(max_document_length_ratio) or max_document_length_ratio < 1):
        raise ValueError("max_document_length_ratio must be finite and >= 1")
    if donor_pool_policy not in ("validation_only", "validation_first"):
        raise ValueError("donor_pool_policy must be validation_only or validation_first")
    _optional_hash(dataset_sha256, "dataset_sha256")
    _optional_hash(baseline_validation_protocol_sha256, "baseline_validation_protocol_sha256")
    indexed, tokenizer_identity = _document_index(documents)
    ordered_documents = [indexed[key] for key in sorted(indexed)]
    samples = [dict(item) for item in validation_sample]
    if not samples:
        raise ValueError("diagnostics require a nonempty saved validation sample")
    primary, seen, origins = [], set(), defaultdict(list)
    for sample in samples:
        if sample.keys() != _SAMPLE_KEYS:
            raise ValueError("saved validation sample fields do not match the raw-LM manifest")
        if sample["document_id"] not in indexed:
            raise ValueError("saved validation document is absent from the supplied corpus")
        document = indexed[sample["document_id"]]
        if (document.split != "validation" or sample["group_id"] != document.group_id
                or sample["source"] != document.source):
            raise ValueError("saved validation document split/group/source differs")
        start = _integer(sample["start"], "validation start", 0)
        blocks = _integer(sample["blocks"], "validation blocks", 2)
        if start % block_size:
            raise ValueError("validation start must be block aligned")
        key = (document.id, start, blocks)
        if key in seen:
            raise ValueError("duplicate primary validation window")
        seen.add(key)
        end = start + blocks * block_size
        if end > len(document.token_ids) or sample["loss_tokens"] != (blocks - 1) * block_size:
            raise ValueError("saved validation window is not a complete matching trajectory")
        if sample["token_sha256"] != _digest(document.token_ids[start:end]):
            raise ValueError("saved validation window token hash differs from corpus")
        window = {**sample, "window_id": "primary:" + _digest(key),
                  "document_tokens": len(document.token_ids)}
        window.update(_match_donors(
            document, window, ordered_documents, block_size=block_size,
            required=donors_per_target, seed=seed,
            max_document_length_ratio=max_document_length_ratio, donor_pool_policy=donor_pool_policy))
        primary.append(window)
        origins[document.id, start].append(window["window_id"])

    after = []
    for (document_id, start), baseline_ids in sorted(origins.items()):
        document = indexed[document_id]
        for point in points:
            target_start = start + point * block_size
            end = target_start + block_size
            entry = {"target_id": "after:" + _digest([document_id, start, point]),
                     "document_id": document_id, "group_id": document.group_id,
                     "source": document.source, "start": start, "n_written": point,
                     "target_start": target_start, "loss_tokens": block_size,
                     "document_tokens": len(document.token_ids),
                     "baseline_window_ids": sorted(baseline_ids)}
            if end > len(document.token_ids):
                entry.update(status="insufficient_following_tokens",
                             required_end=end, available_end=len(document.token_ids),
                             missing_tokens=end - len(document.token_ids))
            else:
                entry.update(status="available",
                             history_token_sha256=_digest(document.token_ids[start:target_start]),
                             target_token_sha256=_digest(document.token_ids[target_start:end]))
            after.append(entry)

    protocol = {"block_size": block_size, "prefix_tokens": prefix_tokens, "seed": seed,
                "donors_per_target": donors_per_target, "after_write_blocks": points,
                "max_document_length_ratio": float(max_document_length_ratio),
                "donor_pool_policy": donor_pool_policy,
                "target_policy": "all_saved_primary_windows; after points deduplicated by document/start/n_written",
                "donor_policy": "same source/language; different known families and exact copies; complete equal history length",
                "donor_ranking": "validation before train, then abs(log(full-document length ratio)), then seeded hash",
                "donor_start": "seeded hash selects a block-aligned complete natural span",
                "recurrence_definition": "target token-ID membership in prior written history excluding its last prefix_tokens; not semantic novelty",
                "logical_document_identity": "SHA256 of sorted id/group_id/source/split/token_sha256 records; not JSONL byte hash",
                "holdout_used": False}
    coverage = {"primary": _stratified(primary, _primary_counts),
                "all_validation_donor_window_ids": [item["window_id"] for item in primary
                                                       if item["all_validation_donors"]],
                "after": {str(point): _stratified(
                    [item for item in after if item["n_written"] == point],
                    lambda items: _after_counts(items, block_size)) for point in points}}
    manifest = {"format": MANIFEST_FORMAT, "protocol": protocol,
                "tokenizer_identity": tokenizer_identity, "dataset_sha256": dataset_sha256,
                "baseline_validation_protocol_sha256": baseline_validation_protocol_sha256,
                "baseline_validation_sample": samples,
                "baseline_validation_sample_sha256": _digest(samples),
                "source_documents_sha256": _digest([
                    {"id": item.id, "group_id": item.group_id, "source": item.source,
                     "split": item.split, "token_sha256": item.token_sha256}
                    for item in ordered_documents]),
                "primary_windows": primary,
                "after_targets": [item for item in after if item["status"] == "available"],
                "after_unavailable_targets": [item for item in after if item["status"] != "available"],
                "coverage": coverage}
    manifest["manifest_sha256"] = _digest(manifest)
    return manifest


def validate_diagnostic_manifest(manifest: Mapping, documents: Sequence[RawDocument]) -> None:
    """Reject changed spans, donors, cohort, coverage, or identities on replay.

    External JSONL byte and baseline-protocol hashes remain caller assertions;
    the runner must compare those with the checkpoint and actual source file.
    All token spans and deterministic matching choices are independently rebuilt.
    """
    if not isinstance(manifest, Mapping) or manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("unsupported diagnostic manifest format")
    supplied = deepcopy(dict(manifest))
    supplied_hash = supplied.pop("manifest_sha256", None)
    if supplied_hash != _digest(supplied):
        raise ValueError("diagnostic manifest content hash differs")
    try:
        protocol = manifest["protocol"]
        expected = build_diagnostic_manifest(
            documents, manifest["baseline_validation_sample"],
            **{key: protocol[key] for key in (
                "block_size", "prefix_tokens", "seed", "donors_per_target", "after_write_blocks",
                "max_document_length_ratio", "donor_pool_policy")},
            dataset_sha256=manifest["dataset_sha256"],
            baseline_validation_protocol_sha256=manifest["baseline_validation_protocol_sha256"])
    except KeyError as error:
        raise ValueError("diagnostic manifest is missing a required field") from error
    if expected != manifest:
        raise ValueError("diagnostic manifest does not match the corpus and fixed pairing protocol")

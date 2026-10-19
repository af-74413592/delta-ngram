"""Frozen original-news continuation: earlier articles write, targets only read.

This deliberately does not call event_trajectory_loss: that training routine
writes later articles after scoring them. Here even teacher-forced past target
tokens are reader context only, and memory stays fixed during a whole query.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

import torch

from .batching import BatchMemoryState, write_batch
from .raw_lm import _json_hash, score_block, tokenizer_identity

PROTOCOL = "delta_ngram_recent_news_continuation_eval_v1"
FROZEN_DEV_SHA256 = "708e2ff104ff4e0541ac6c0092026b5921a2632e04c049422f54866a890cd667"
DISTANCES = (0, 1, 4, 8, 16)
WRITE_BLOCK = 512
PREFIX = 128
TARGET_BLOCK = WRITE_BLOCK - PREFIX
CONDITIONS = ("self", "empty", "donor", "self_perm", "donor_perm")


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _date(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


@dataclass(frozen=True)
class PreparedContinuation:
    registry: dict
    tokens: dict[str, tuple[int, ...]]
    manifest: dict
    files: tuple[tuple[Path, str], ...]

    def verify_files(self):
        for path, expected in self.files:
            if file_sha256(path) != expected:
                raise RuntimeError(f"evaluation input changed: {path}")


@dataclass(frozen=True)
class PrefixAnnotations:
    by_pair: dict
    manifest: dict
    path: Path
    sha256: str

    def verify_file(self):
        if file_sha256(self.path) != self.sha256:
            raise RuntimeError("prefix annotation file changed during evaluation")


def prepare_prefix_annotations(path, prepared, method, *, expected_sha256):
    """Validate frozen original-text spans; labels are never inferred from scores."""
    path = Path(path).resolve()
    _require(file_sha256(path) == expected_sha256, "prefix annotation file SHA256 mismatch")
    annotations = json.loads(path.read_bytes())
    _require(annotations.get("protocol") == "recent_news_prefix_annotations_v1"
             and annotations.get("evaluation_only") is True and annotations.get("split") == "dev",
             "invalid DEV prefix annotations")
    _require(annotations.get("parent_registry_file_sha256") == prepared.manifest["registry_file_sha256"]
             and annotations.get("tokenizer_revision") == prepared.registry["tokenization"]["revision"]
             and annotations.get("initial_prefix_tokens") == PREFIX, "prefix annotation parent identity mismatch")
    _require(annotations.get("model_results_seen") is False, "annotations must be frozen without model results")
    rows = annotations["annotations"]
    by_pair = {row["pair_id"]: row for row in rows}
    pairs = prepared.registry["pairs"]
    _require(len(by_pair) == len(rows) == len(pairs) and set(by_pair) == {p["id"] for p in pairs},
             "prefix annotations must cover every pair exactly once")
    sources = {s["id"]: s for s in prepared.manifest["sources"]}
    base = Path(prepared.manifest["registry_path"]).parent
    for pair in pairs:
        row = by_pair[pair["id"]]
        _require(row["event_family"] == pair["event_family"] and row.get("label") in
                 ("restated", "not_restated", "unknown") and bool(row.get("rationale")), "invalid prefix annotation label")
        early_id, late_id = pair["earlier_source_id"], pair["later_source_id"]
        earlier = (base / sources[early_id]["extracted_text_relpath"]).read_bytes().decode("utf-8")
        later = (base / sources[late_id]["extracted_text_relpath"]).read_bytes().decode("utf-8")
        prefix_ids = prepared.tokens[late_id][:PREFIX]
        try:
            prefix = method.tokenizer.decode(prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except TypeError:  # Offline character tokenizer in CPU plumbing tests.
            prefix = method.tokenizer.decode(prefix_ids, skip_special_tokens=False)
        _require(row.get("later_prefix_token_ids_sha256") == _json_hash(prefix_ids), "annotated prefix token hash mismatch")
        _require(row.get("later_prefix_text_sha256") == hashlib.sha256(prefix.encode()).hexdigest(),
                 "annotated prefix text hash mismatch")
        # If a tokenizer ends inside a Unicode character, a negative/positive
        # textual judgement cannot be grounded in an exact original prefix.
        _require(later.startswith(prefix) or row["label"] == "unknown", "annotated prefix is not an exact source prefix")
        facts = row.get("key_facts", [])
        _require(bool(facts) or row["label"] == "unknown", "decided prefix annotations need earlier key-fact spans")
        matches = 0
        for fact in facts:
            _require(bool(fact.get("fact")), "annotation fact must state a proposition")
            span = fact["earlier_span"]
            _require(span.get("source_id") == early_id, "annotation earlier span has wrong source")
            start, end = span["char_start"], span["char_end_exclusive"]
            _require(type(start) is int and type(end) is int and 0 <= start < end <= len(earlier), "invalid earlier fact span")
            _require(hashlib.sha256(earlier[start:end].encode()).hexdigest() == span["text_sha256"], "earlier fact span SHA256 mismatch")
            if fact.get("later_prefix_span") is not None:
                span = fact["later_prefix_span"]
                start, end = span["char_start"], span["char_end_exclusive"]
                _require(span.get("source_id") == late_id and type(start) is int and type(end) is int
                         and 0 <= start < end <= len(prefix), "fact restatement lies outside fixed prefix")
                _require(hashlib.sha256(prefix[start:end].encode()).hexdigest() == span["text_sha256"], "prefix fact span SHA256 mismatch")
                matches += 1
        _require(row["label"] != "restated" or matches > 0, "restated label needs a prefix fact span")
        _require(row["label"] != "not_restated" or matches == 0, "not_restated label conflicts with matching span")
        revision = row.get("revision", {"eligible": False})
        _require(type(revision.get("eligible")) is bool, "revision eligibility must be an explicit boolean")
        if revision["eligible"]:
            _require(revision.get("type") in ("status_update", "version_update", "factual_correction", "numeric_revision")
                     and bool(revision.get("property")), "revision must name its type and changing property")
            _require(bool(revision.get("earlier_answer")) and bool(revision.get("later_answer"))
                     and revision["earlier_answer"] != revision["later_answer"], "revision needs distinct old/new answers")
            for role, source_id, original in (("earlier", early_id, earlier), ("later", late_id, later)):
                selected = revision[role + "_span"]
                start, end = selected["char_start"], selected["char_end_exclusive"]
                _require(selected["source_id"] == source_id and type(start) is int and type(end) is int
                         and 0 <= start < end <= len(original), "invalid revision answer span")
                content = original[start:end]
                _require(hashlib.sha256(content.encode()).hexdigest() == selected["text_sha256"]
                         and revision[role + "_answer"] in content, "revision answer span/hash mismatch")
    manifest = {"path": str(path), "file_sha256": expected_sha256, **annotations}
    return PrefixAnnotations(by_pair, manifest, path, expected_sha256)


def _source_file(base, source, stem):
    # Relative paths make a transferred archive independent of its collector host.
    relative = source.get(stem + "_relpath")
    _require(isinstance(relative, str) and relative, f"missing portable {stem} path: {source['id']}")
    path = (base / relative).resolve()
    _require(path.is_relative_to(base), f"source path escapes registry directory: {source['id']}")
    expected = source.get(stem + "_sha256")
    _require(isinstance(expected, str) and len(expected) == 64, f"missing {stem} SHA256: {source['id']}")
    _require(file_sha256(path) == expected, f"{stem} SHA256 mismatch: {source['id']}")
    return path, expected


def prepare_continuation(path, method, *, tokenizer_source, tokenizer_revision,
                         expected_sha256=FROZEN_DEV_SHA256):
    """Load DEV originals only; reject altered bytes, timing, tokens, or plans.

The caller supplies the effective pinned tokenizer binding from its config.
The actual vocabulary/pipeline fingerprint is also recorded and compared with
the checkpoint by the CLI. No training or final-holdout data is opened here.
"""
    path = Path(path).resolve()
    payload = path.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    _require(actual_hash == expected_sha256, "registry file SHA256 mismatch")
    registry = json.loads(payload)
    _require(registry.get("protocol") == "recent_news_continuation_v1", "unsupported continuation registry")
    _require(registry.get("evaluation_only") is True and registry.get("split") == "dev",
             "only evaluation-only DEV registries are accepted")
    model, tokenization = registry["model"], registry["tokenization"]
    _require(tokenizer_source == model["name_or_path"] and tokenizer_revision == model["revision"]
             and tokenizer_revision == tokenization["revision"], "tokenizer source/revision differs from registry")
    _require(tokenization.get("add_special_tokens") is False
             and tokenization.get("write_block_tokens") == WRITE_BLOCK
             and tokenization.get("maximum_tokens_per_write_article") is None,
             "registry requires full-article 512-token writes without special tokens")
    invariants = registry["online_invariants"]
    _require(invariants.get("parameters_fixed") is True and all(invariants.get(k) is False for k in
             ("optimizer_allowed", "backward_allowed", "autograd_gradient_updates_allowed")),
             "registry must require fixed forward-only parameters")
    sources = registry["sources"] + registry["background_sources"]
    indexed = {source["id"]: source for source in sources}
    _require(len(indexed) == len(sources), "duplicate original source ID")
    files, tokens, source_manifest = [(path, actual_hash)], {}, []
    vocabulary_size = method.reader.model.get_input_embeddings().weight.shape[0]
    for source in sources:
        original, text_hash = _source_file(path.parent, source, "extracted_text")
        files.append((original, text_hash))
        for stem in ("raw_response", "raw_record"):
            if source.get(stem + "_relpath"):
                files.append(_source_file(path.parent, source, stem))
        text = original.read_bytes().decode("utf-8")
        _require("\ufffd" not in text and "\r" not in text, f"original must be UTF-8/LF: {source['id']}")
        ids = tuple(method.tokenizer.encode(text, add_special_tokens=False))
        _require(ids and all(type(token) is int and 0 <= token < vocabulary_size for token in ids),
                 f"invalid original token IDs: {source['id']}")
        _require(len(ids) == source["token_count"], f"original tokenizer count mismatch: {source['id']}")
        tokens[source["id"]] = ids
        published = _date(source["published_at"])
        available = max(published, _date(source.get("updated_at", source["published_at"])))
        _require(available == _date(source["available_at"]) <= _date(source["retrieved_at"]),
                 f"invalid source availability: {source['id']}")
        source_manifest.append({key: source[key] for key in
            ("id", "url", "published_at", "updated_at", "available_at", "retrieved_at",
             "extracted_text_relpath", "extracted_text_sha256", "raw_response_sha256", "raw_record_sha256")
            if key in source} | {"token_count": len(ids), "token_ids_sha256": _json_hash(ids)})
    pairs = registry["pairs"]
    _require(bool(pairs), "continuation registry has no pairs")
    _require(len({p["id"] for p in pairs}) == len(pairs)
             and len({p["event_family"] for p in pairs}) == len(pairs), "pairs require independent unique families")
    earlier_families = {p["earlier_source_id"]: p["event_family"] for p in pairs}
    later_ids = {p["later_source_id"] for p in pairs}
    background = {s["id"]: s["event_family"] for s in registry["background_sources"]}
    allowed_writes = earlier_families | background
    _require(len(earlier_families) == len(later_ids) == len(pairs), "paired sources must be unique")
    _require(not later_ids.intersection(allowed_writes), "later target source cannot be any write source")
    for pair in pairs:
        earlier, later = indexed[pair["earlier_source_id"]], indexed[pair["later_source_id"]]
        _require(_date(earlier["available_at"]) < _date(later["published_at"]),
                 f"earlier original is not strictly causal: {pair['id']}")
        _require(pair["query_at"] == later["available_at"], "query time must equal later availability")
        for source in (earlier, later):
            _require(_date("2025-08-01") <= _date(source["available_at"]) < _date("2026-01-01"),
                     "scored source outside DEV availability window")
        _require(earlier["extracted_text_sha256"] != later["extracted_text_sha256"], "identical paired originals")
        scoring = pair["scoring"]
        _require(scoring.get("text_source_id") == later["id"]
                 and scoring.get("initial_prefix_tokens") == scoring.get("target_start_token") == PREFIX
                 and scoring.get("maximum_read_window_tokens") == WRITE_BLOCK
                 and scoring.get("maximum_target_tokens_per_window") == TARGET_BLOCK
                 and scoring.get("target_end_token_exclusive") == len(tokens[later["id"]]) > PREFIX
                 and scoring.get("score_all_remaining_tokens") is True
                 and scoring.get("article_memory_write_allowed") is False, "invalid target scoring interval")
        donor_id = pair["donor_source_id"]
        _require(donor_id in allowed_writes and allowed_writes[donor_id] == pair["donor_event_family"]
                 and pair["donor_event_family"] != pair["event_family"], "donor must be another available write family")
        distractors = pair["distractor_source_ids"]
        _require(len(distractors) == len(set(distractors)) == 16 and set(distractors) <= background.keys()
                 and donor_id not in distractors, "invalid fixed unrelated distractor pool")
        for source_id in [donor_id, *distractors]:
            _require(_date(indexed[source_id]["available_at"]) < _date(pair["query_at"])
                     and allowed_writes[source_id] != pair["event_family"], "noncausal or same-family control")
        for arm, source_id in (("earlier", earlier["id"]), ("donor", donor_id)):
            _require(pair[arm + "_write_token_count"] == len(tokens[source_id])
                     and pair[arm + "_write_blocks"] == math.ceil(len(tokens[source_id]) / WRITE_BLOCK),
                     "anchor write budget mismatch")
        conditions = pair["interference_conditions"]
        _require(tuple(c["unrelated_article_count"] for c in conditions) == DISTANCES, "invalid interference distances")
        for condition in conditions:
            count = condition["unrelated_article_count"]
            selected = distractors[:count]
            _require(condition["distractor_source_ids"] == selected, "interference prefixes must be nested")
            filler_tokens = sum(len(tokens[s]) for s in selected)
            filler_blocks = sum(math.ceil(len(tokens[s]) / WRITE_BLOCK) for s in selected)
            _require(condition["distractor_token_count"] == filler_tokens
                     and condition["distractor_block_count"] == filler_blocks, "distractor write budget mismatch")
            for arm, anchor in (("self", earlier["id"]), ("donor", donor_id)):
                _require(condition[arm + "_actual_written_articles"] == count + 1
                         and condition[arm + "_actual_written_tokens"] == filler_tokens + len(tokens[anchor])
                         and condition[arm + "_actual_written_blocks"] == filler_blocks + math.ceil(len(tokens[anchor]) / WRITE_BLOCK),
                         f"{arm} write budget mismatch")
            _require(all(condition["empty_actual_written_" + key] == 0 for key in ("articles", "tokens", "blocks")),
                     "empty condition must have no writes")
    manifest = {"registry_path": str(path), "registry_file_sha256": actual_hash,
        "parent_qa_manifest_file_sha256": registry.get("parent_qa_manifest_file_sha256"),
        "registry_model": model, "tokenizer": tokenizer_identity(method),
        "tokenizer_binding": {"name_or_path": tokenizer_source, "revision": tokenizer_revision},
        "sources": source_manifest, "pairs": pairs, "independent_families": len(pairs),
        "write_block_tokens": WRITE_BLOCK, "initial_prefix_tokens": PREFIX,
        "target_tokens_per_distance": sum(len(tokens[p["later_source_id"]]) - PREFIX for p in pairs),
        "distances": list(DISTANCES), "archive_files_verified": len(files) - 1,
        "stage": registry.get("stage", "development engineering validation")}
    manifest["tokenizer_fingerprint"] = "sha256:" + _json_hash(manifest["tokenizer"])
    return PreparedContinuation(registry, tokens, manifest, tuple(files))


def scoring_windows(token_count):
    """Yield (reader-start, target-start, target-end) with exact target coverage."""
    _require(type(token_count) is int and token_count > PREFIX, "later body must exceed the fixed prefix")
    for start in range(PREFIX, token_count, TARGET_BLOCK):
        yield start - PREFIX, start, min(start + TARGET_BLOCK, token_count)


def _empty(method):
    return BatchMemoryState.empty(1, method.reader.width,
                                  method.reader.buckets * len(method.reader.orders), method.device)


@torch.no_grad()
def write_article(method, state, token_ids, *, source_id, observer=None):
    """Sequential complete-article writes with article-local encoder/hash context."""
    counters = {"articles": 1, "tokens": 0, "blocks": 0, "address_prefix_tokens": 0}
    boundary = max(method.reader.orders) - 1
    for start in range(0, len(token_ids), WRITE_BLOCK):
        block = token_ids[start:start + WRITE_BLOCK]
        ids = torch.tensor([block], dtype=torch.long, device=method.device)
        mask = torch.ones_like(ids)
        hidden, _ = method.reader.encode_evidence(ids, mask)
        address_start = max(0, start - boundary)
        address_ids = torch.tensor([token_ids[address_start:start + len(block)]], dtype=torch.long, device=method.device)
        rows = method.reader.adapter.addresses(address_ids, torch.ones_like(address_ids))[:, start - address_start:]
        if observer is not None:
            observer({"kind": "write", "source_id": source_id, "article_token_start": start,
                      "ids": ids, "rows": rows, "state": state})
        state = write_batch(method.writer, state, hidden, rows, mask)
        counters["tokens"] += len(block)
        counters["blocks"] += 1
        counters["address_prefix_tokens"] += start - address_start
    return state, counters


@torch.no_grad()
def score_later_article(method, state, token_ids, *, source_id, observer=None):
    """Teacher forcing uses only reader calls and never modifies memory."""
    guarded = ("rows", "values", "counts", "precision", "block_clock",
               "cumulative_log_alpha", "row_last_written_block", "row_log_alpha_at_write")
    before = {name: None if value is None else value.clone() for name, value in
              ((name, getattr(state, name)) for name in guarded if hasattr(state, name))}
    losses, counts = [], []
    reader_tokens = 0
    for context_start, target_start, target_end in scoring_windows(len(token_ids)):
        ids = torch.tensor([token_ids[context_start:target_end]], dtype=torch.long, device=method.device)
        mask = torch.ones_like(ids)
        labels = torch.full_like(ids, -100)
        labels[:, PREFIX:] = ids[:, PREFIX:]
        if observer is not None:
            observer({"kind": "read", "source_id": source_id, "target_start": target_start,
                      "target_end_exclusive": target_end, "ids": ids, "labels": labels, "state": state})
        loss, count = score_block(method.reader, state, ids, mask, labels, checkpoint_reader=False)
        _require(count == target_end - target_start, "runtime target count mismatch")
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite continuation target loss")
        losses.append(float(loss))
        counts.append(count)
        reader_tokens += ids.numel()
    for name, saved in before.items():
        current = getattr(state, name)
        if (saved is None) != (current is None) or (saved is not None and not torch.equal(saved, current)):
            raise RuntimeError("query reader mutated memory")
    _require(sum(counts) == len(token_ids) - PREFIX, "incomplete or duplicated target coverage")
    return {"loss_sum": math.fsum(losses), "target_tokens": sum(counts),
            "nll": math.fsum(losses) / sum(counts), "read_windows": len(losses),
            "reader_input_tokens": reader_tokens, "first_target_loss_sum": losses[0],
            "first_target_tokens": counts[0], "first_target_nll": losses[0] / counts[0]}


def _percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def summarize_continuation(records, *, bootstrap_seed=20261019, bootstrap_samples=10000):
    """Equal-family paired bootstrap; keep distances coupled within each draw."""
    _require(type(bootstrap_samples) is int and bootstrap_samples > 0, "bootstrap_samples must be positive")
    families = sorted({r["event_family"] for r in records})
    _require(bool(families), "no scored families")
    indexed = {(r["event_family"], r["unrelated_article_count"]): r for r in records}
    _require(len(indexed) == len(records) == len(families) * len(DISTANCES), "incomplete family-distance results")
    series = {distance: [indexed[family, distance] for family in families] for distance in DISTANCES}
    contrasts = {}
    for distance, rows in series.items():
        keys = ["delta_empty", "delta_donor"]
        keys += [key for key in ("direction_delta_self", "direction_delta_donor", "direction_difference") if key in rows[0]]
        contrasts[distance] = {key: [row[key] for row in rows] for key in keys}
        for control in ("empty", "donor"):
            contrasts[distance]["retention_change_" + control] = [
                row["delta_" + control] - indexed[row["event_family"], 0]["delta_" + control] for row in rows]
    draws = {distance: {key: [] for key in values} for distance, values in contrasts.items()}
    rng = random.Random(bootstrap_seed)
    for _ in range(bootstrap_samples):
        sampled = [rng.randrange(len(families)) for _ in families]
        for distance, values in contrasts.items():
            for key, numbers in values.items():
                draws[distance][key].append(math.fsum(numbers[i] for i in sampled) / len(sampled))
    summaries = []
    for distance, rows in series.items():
        summary = {"unrelated_article_count": distance, "families": len(rows),
                   "target_tokens": sum(row["conditions"]["self"]["target_tokens"] for row in rows)}
        for arm in (arm for arm in CONDITIONS if arm in rows[0]["conditions"]):
            summary[arm + "_nll"] = math.fsum(row["conditions"][arm]["nll"] for row in rows) / len(rows)
            summary[arm + "_pooled_token_nll"] = math.fsum(row["conditions"][arm]["loss_sum"] for row in rows) / summary["target_tokens"]
            summary[arm + "_first_target_nll"] = math.fsum(row["conditions"][arm]["first_target_nll"] for row in rows) / len(rows)
            for unit in ("articles", "tokens", "blocks"):
                values = [row["conditions"][arm]["actual_written_" + unit] for row in rows]
                summary[arm + "_actual_written_" + unit] = {"min": min(values), "max": max(values),
                                                            "mean": math.fsum(values) / len(values)}
        if "self_perm" in rows[0]["conditions"]:
            summary["direction_by_seed"] = {}
            for seed in rows[0]["conditions"]["self_perm"]["per_seed"]:
                summary["direction_by_seed"][seed] = {
                    arm + "_nll": math.fsum(row["conditions"][arm]["per_seed"][seed]["nll"] for row in rows) / len(rows)
                    for arm in ("self_perm", "donor_perm")}
                for arm in ("self", "donor"):
                    summary["direction_by_seed"][seed]["direction_delta_" + arm] = (
                        summary["direction_by_seed"][seed][arm + "_perm_nll"] - summary[arm + "_nll"])
        for key, values in contrasts[distance].items():
            summary[key] = math.fsum(values) / len(values)
            summary[key + "_ci95"] = [_percentile(draws[distance][key], .025), _percentile(draws[distance][key], .975)]
        summaries.append(summary)
    return {"aggregation": "mean target NLL within family, then equal-family macro",
            "delta_definition": "control NLL minus self NLL; positive means improvement",
            "retention_definition": "delta at distance minus paired delta at zero; zero means no change",
            "direction_definition": "mean of three permuted-read NLLs minus original NLL within each family; positive means aligned direction helps",
            "direction_difference_definition": "direction_delta_self minus direction_delta_donor",
            "bootstrap": {"unit": "event_family", "paired_across_conditions_and_distances": True,
                          "seed": bootstrap_seed, "replicates": bootstrap_samples, "confidence": .95},
            "retention_curve": summaries}


def _parameters(module):
    return {name: (value, value._version) for name, value in module.named_parameters()}


def _check_parameters(module, previous, label):
    current = dict(module.named_parameters())
    if (current.keys() != previous.keys() or any(current[name] is not value or value._version != version
            for name, (value, version) in previous.items())
            or any(value.requires_grad or value.grad is not None for value in current.values())):
        raise RuntimeError(f"{label} parameters changed or received gradients")


@torch.no_grad()
def evaluate_continuation(method, prepared, *, bootstrap_seed=20261019, bootstrap_samples=10000,
                          progress_callback=None, observer=None, prefix_annotations=None,
                          memory_backend=None):
    """All five distances, three arms, frozen parameters, and no target writes.

    For an arm we retain nested distractor prefixes between checkpoints. Query
    calls are read-only, so this equals replaying each prefix from a fresh state.
    Empty scores are computed once because its state/prefix/targets never vary.
    An explicit backend may supply empty/write_article/permute_read/start/finish
    and after_query. The default is the original D1 state and write path.
    """
    _require(method.reader.backend == "external_hash" and method.name == "forward"
             and method.supports_outer_training, "continuation requires a learned external-hash forward writer")
    prepared.verify_files()
    if prefix_annotations is not None:
        prefix_annotations.verify_file()
    method.eval()
    for module in (method.writer, method.reader.model):
        module.eval()
        module.requires_grad_(False)
        module.zero_grad(set_to_none=True)
    if memory_backend is not None:
        memory_backend.start(method)
    writer_tensors = {name: tensor.detach().clone() for name, tensor in method.writer.state_dict().items()}
    writer_parameters, reader_parameters = _parameters(method.writer), _parameters(method.reader.model)
    from .p2_initialization import writer_tensor_sha256
    from .direction_control import DIRECTION_CONTROL_SEEDS, make_direction_permutation, permute_state_for_read
    writer_hash = writer_tensor_sha256(writer_tensors)
    plans = tuple(make_direction_permutation(method.reader.width, seed) for seed in DIRECTION_CONTROL_SEEDS)
    empty_state = _empty if memory_backend is None else memory_backend.empty
    write_state = write_article if memory_backend is None else memory_backend.write_article
    permute_read = permute_state_for_read if memory_backend is None else memory_backend.permute_read
    records = []
    execution = {"write_calls": 0, "writer_tokens": 0, "query_calls": 0}
    for pair in prepared.registry["pairs"]:
        later_id = pair["later_source_id"]
        targets = prepared.tokens[later_id]
        empty = score_later_article(method, empty_state(method), targets, source_id=later_id, observer=observer)
        execution["query_calls"] += empty["read_windows"]
        empty.update(actual_written_articles=0, actual_written_tokens=0, actual_written_blocks=0,
                     address_prefix_tokens=0, write_source_ids=[])
        arms = {distance: {"empty": empty.copy()} for distance in DISTANCES}
        for arm, anchor in (("self", pair["earlier_source_id"]), ("donor", pair["donor_source_id"])):
            if memory_backend is not None:
                memory_backend.set_context(evaluation="continuation", pair_id=pair["id"],
                    event_family=pair["event_family"], memory_mode=arm)
            state = empty_state(method)
            totals = dict.fromkeys(("articles", "tokens", "blocks", "address_prefix_tokens"), 0)
            written = []
            for condition in pair["interference_conditions"]:
                distance = condition["unrelated_article_count"]
                planned = [anchor, *condition["distractor_source_ids"]]
                _require(planned[:len(written)] == written, "non-nested runtime write schedule")
                for source_id in planned[len(written):]:
                    _require(source_id != later_id, "later target cannot be written")
                    state, counts = write_state(method, state, prepared.tokens[source_id],
                                                  source_id=source_id, observer=observer)
                    for key, value in counts.items():
                        totals[key] += value
                    written.append(source_id)
                    execution["write_calls"] += counts["blocks"]
                    execution["writer_tokens"] += counts["tokens"]
                for unit in ("articles", "tokens", "blocks"):
                    _require(totals[unit] == condition[arm + "_actual_written_" + unit], "actual runtime write budget mismatch")
                scored = score_later_article(method, state, targets, source_id=later_id, observer=observer)
                execution["query_calls"] += scored["read_windows"]
                scored.update({"actual_written_" + unit: totals[unit] for unit in ("articles", "tokens", "blocks")})
                scored.update(address_prefix_tokens=totals["address_prefix_tokens"], write_source_ids=written.copy())
                if memory_backend is not None:
                    scored["memory_diagnostics"] = memory_backend.after_query(
                        method, state, targets, source_id=later_id, original_score=scored)
                arms[distance][arm] = scored
                seed_scores = {}
                for plan in plans:
                    # The read view is discarded; the original D1 state continues writing.
                    view = permute_read(state, plan)
                    seed_score = score_later_article(method, view, targets, source_id=later_id, observer=observer)
                    del view
                    execution["query_calls"] += seed_score["read_windows"]
                    seed_scores[str(plan.seed)] = seed_score
                averaged = {key: math.fsum(item[key] for item in seed_scores.values()) / len(seed_scores)
                            for key in ("loss_sum", "nll", "first_target_loss_sum", "first_target_nll")}
                averaged.update({key: scored[key] for key in ("target_tokens", "read_windows", "reader_input_tokens",
                    "first_target_tokens", "actual_written_articles", "actual_written_tokens", "actual_written_blocks",
                    "address_prefix_tokens", "write_source_ids")})
                averaged.update(per_seed=seed_scores, aggregation="arithmetic mean across fixed seeds within family")
                arms[distance][arm + "_perm"] = averaged
        for distance in DISTANCES:
            conditions = arms[distance]
            record = {"pair_id": pair["id"], "event_family": pair["event_family"], "domain": pair["domain"],
                      "unrelated_article_count": distance, "later_source_id": later_id, "conditions": conditions,
                      "prefix_restatement_label": (prefix_annotations.by_pair[pair["id"]]["label"]
                                                   if prefix_annotations is not None else "unknown")}
            revision = (prefix_annotations.by_pair[pair["id"]].get("revision", {"eligible": False})
                        if prefix_annotations is not None else {"eligible": False})
            record.update(revision_eligible=revision["eligible"], revision_type=revision.get("type"))
            for control in ("empty", "donor"):
                record["delta_" + control] = conditions[control]["nll"] - conditions["self"]["nll"]
                record["first_target_delta_" + control] = conditions[control]["first_target_nll"] - conditions["self"]["first_target_nll"]
            for arm in ("self", "donor"):
                record["direction_delta_" + arm] = conditions[arm + "_perm"]["nll"] - conditions[arm]["nll"]
            record["direction_difference"] = record["direction_delta_self"] - record["direction_delta_donor"]
            records.append(record)
        if progress_callback is not None:
            progress_callback({"stage": "pair_complete", "pair_id": pair["id"],
                               "pairs_complete": len(records) // len(DISTANCES),
                               "pairs_total": len(prepared.registry["pairs"]), "target_tokens": len(targets) - PREFIX})
    after = method.writer.state_dict()
    if writer_tensors.keys() != after.keys() or any(not torch.equal(value, after[name])
                                                   for name, value in writer_tensors.items()):
        raise RuntimeError("writer state_dict changed during continuation evaluation")
    _check_parameters(method.writer, writer_parameters, "writer")
    _check_parameters(method.reader.model, reader_parameters, "reader")
    backend_metadata = None if memory_backend is None else memory_backend.finish(method)
    prepared.verify_files()
    if prefix_annotations is not None:
        prefix_annotations.verify_file()
    direction_control = {"plans": [plan.metadata() for plan in plans],
        "scope": "one shared channel permutation per seed across all events, distances, self and donor",
        "application": "read view only; unpermuted state retained for all subsequent writes",
        "primary_seed_aggregation": "mean within family before paired event-family bootstrap",
        "seed_selection_allowed": False}
    manifest = {**prepared.manifest, "direction_control": direction_control,
        "bootstrap_seed": bootstrap_seed, "bootstrap_samples": bootstrap_samples,
        "prefix_annotations": (prefix_annotations.manifest if prefix_annotations is not None
                               else {"status": "not supplied; all labels explicitly unknown"})}
    result = {"protocol": PROTOCOL, "complete": True, "evaluation_only": True, "split": "dev",
              "optimizer_steps": 0, "backward_calls": 0, "conditions": list(CONDITIONS),
              "independent_families": len(prepared.registry["pairs"]), "per_family": records,
              "manifest": manifest, "manifest_sha256": _json_hash(manifest),
              "execution": execution, "writer_tensor_sha256": writer_hash,
              "direction_control": direction_control,
              "integrity_checks": {"input_files_unchanged": True, "writer_state_dict_exact_equal": True,
                  "writer_parameter_objects_and_versions_unchanged": True,
                  "reader_parameter_objects_and_versions_unchanged": True, "parameters_frozen_no_gradients": True,
                  "target_never_written": True, "memory_unchanged_during_queries": True,
                  "actual_write_budgets_verified": True, "reader_weight_content_hash_verified": False},
              "execution_policy": "Incremental nested replay; empty scored once and reused at identical distances.",
              "limitations": [f"Development continuation evaluation with {len(prepared.registry['pairs'])} independent families; not a FINAL evaluation.",
                  "Teacher forcing predicts fixed original targets; no target token enters the writer.",
                  "Source archives retrieved in 2026 are not certified unchanged 2025 historical snapshots.",
                  "Base pretraining novelty is unknown; report both empty and fixed-donor controls.",
                  "Reader parameter version checks do not detect every .data or buffer mutation.",
                  "Auxiliary free generation is not performed by this NLL evaluator."]}
    result.update(summarize_continuation(records, bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples))
    if backend_metadata is not None:
        result["memory_backend"] = backend_metadata
    result["prefix_restatement_groups"] = {}
    for label in ("restated", "not_restated", "unknown"):
        group = [row for row in records if row["prefix_restatement_label"] == label]
        result["prefix_restatement_groups"][label] = {"families": len(group) // len(DISTANCES),
            **(summarize_continuation(group, bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples)
               if group else {"status": "no families; no estimate"})}
    revision_records = [row for row in records if row["revision_eligible"]]
    result["revision_subset"] = {"families": len(revision_records) // len(DISTANCES),
        "scope": "Explicit old/new original spans; status/version updates distinguished from factual corrections. No claim that every later article is a revision.",
        "types": {kind: len({row["event_family"] for row in revision_records if row["revision_type"] == kind})
                  for kind in sorted({row["revision_type"] for row in revision_records})},
        **(summarize_continuation(revision_records, bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples)
           if revision_records else {"status": "no registered revision families; no estimate"})}
    json.dumps(result, allow_nan=False)
    return result

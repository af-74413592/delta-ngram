"""Evaluation-only news episodes for a frozen forward writer.

No current-news record is sent to the optimizer. Questions and candidate labels
are never writer inputs; each query sees only its causally preceding history.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from urllib.parse import urlsplit

import torch
import torch.nn.functional as F

from .runtime import answer_batch, assert_read_only, state_guard

PROTOCOL = "news_forward_v1"
SPLITS = ("dev", "final_holdout")
KINDS = ("real", "synthetic_swap", "synthetic_correction")
CONDITIONS = ("self", "empty", "donor", "context")


def manifest_sha256(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _date(value, name):
    _text(value, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO date or timestamp") from error
    # Date-only values deliberately mean date-granularity provenance.
    if len(value) != 10 and parsed.tzinfo is None:
        raise ValueError(f"{name} timestamps must include a timezone")
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _list(value, name, *, empty=False):
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError(f"{name} must be a {'possibly empty ' if empty else 'nonempty '}list")
    return value


def _unique(items, name):
    if any(not isinstance(item, dict) for item in items):
        raise ValueError(f"{name} entries must be objects")
    keys = [_text(item.get("id"), name + ".id") for item in items]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate {name} id")
    return dict(zip(keys, items))


def validate_manifest(manifest):
    """Reject event-family/source leakage, future evidence, and stale labels.

Event-family membership and source factual accuracy still need human review.
Dates/revisions document boundaries; they do not certify pretraining novelty.
"""
    if not isinstance(manifest, dict) or manifest.get("protocol") != PROTOCOL or manifest.get("evaluation_only") is not True:
        raise ValueError("requires news_forward_v1 and evaluation_only=true")
    model = manifest.get("model", {})
    for key in ("name_or_path", "revision"):
        _text(model.get(key), "model." + key)
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", model["revision"]):
        raise ValueError("model.revision must be an immutable hexadecimal commit, not a branch or tag")
    _date(model.get("revision_date"), "model.revision_date")
    if model.get("training_cutoff") != "unknown":
        _date(model.get("training_cutoff"), "model.training_cutoff")
    training = _list(manifest.get("training_event_families"), "training_event_families", empty=True)
    for family in training:
        _text(family, "training_event_families[]")
    episodes = _list(manifest.get("episodes"), "episodes")
    indexed = _unique(episodes, "episode")
    families, source_splits, text_splits = {}, {}, {}
    for episode in episodes:
        family = _text(episode.get("event_family"), "event_family")
        split = episode.get("split")
        if split not in SPLITS:
            raise ValueError("episode split must be dev or final_holdout; news is evaluation-only")
        if family in training:
            raise ValueError("training event family overlaps evaluation")
        if families.setdefault(family, split) != split:
            raise ValueError("event family crosses dev/final_holdout split")
        kind = episode.get("kind")
        if kind not in KINDS or episode.get("synthetic") is not (kind != "real"):
            raise ValueError("kind and explicit synthetic flag disagree")
        if kind != "real":
            _text(episode.get("transformation"), "synthetic transformation")
        sources = _unique(_list(episode.get("sources"), "sources"), "source")
        for source in sources.values():
            url = _text(source.get("url"), "source.url")
            parsed = urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("source.url must be an HTTP(S) URL")
            _text(source.get("title"), "source.title")
            published = _date(source.get("published_at"), "source.published_at")
            retrieved = _date(source.get("retrieved_at"), "source.retrieved_at")
            if published > retrieved:
                raise ValueError("source publication is later than retrieval")
            if "updated_at" in source:
                updated = _date(source["updated_at"], "source.updated_at")
                if not published <= updated <= retrieved:
                    raise ValueError("source updated_at must lie between publication and retrieval")
            canonical = parsed._replace(fragment="").geturl()
            if source_splits.setdefault(canonical, split) != split:
                raise ValueError("source URL crosses dev/final_holdout split")
        history = _list(episode.get("history"), "history")
        _unique(history, "history")
        facts, last_date = {}, None
        for index, block in enumerate(history, 1):
            text = _text(block.get("text"), "history.text")
            normalized = " ".join(text.split()).casefold()
            if text_splits.setdefault(normalized, split) != split:
                raise ValueError("duplicate evidence crosses dev/final_holdout split")
            available = _date(block.get("available_at"), "history.available_at")
            if last_date is not None and available < last_date:
                raise ValueError("history must be chronologically ordered")
            last_date = available
            source_ids = _list(block.get("source_ids"), "history.source_ids")
            if any(source_id not in sources for source_id in source_ids):
                raise ValueError("history has unknown source id")
            if any(_date(sources[s]["published_at"], "published_at") > available for s in source_ids):
                raise ValueError("history is available before its source publication")
            if any("updated_at" in sources[s] and _date(sources[s]["updated_at"], "updated_at") > available
                   for s in source_ids):
                raise ValueError("history is available before its source update")
            for fact in _list(block.get("facts"), "history.facts"):
                fact_id = _text(fact.get("id"), "fact.id")
                if fact_id in facts:
                    raise ValueError("duplicate fact id")
                for key in ("subject", "relation", "value"):
                    _text(fact.get(key), "fact." + key)
                if fact["value"].casefold() not in text.casefold():
                    raise ValueError("fact value must occur in its evidence text")
                facts[fact_id] = (index, fact)
        questions = _list(episode.get("questions"), "questions")
        _unique(questions, "question")
        for question in questions:
            prompt = _text(question.get("question"), "question.question")
            # Metadata labels are not part of the evidence payload.
            if any(prompt.casefold() in block["text"].casefold() for block in history):
                raise ValueError("question text leaked into writer evidence")
            after = question.get("after_history")
            if type(after) is not int or not 1 <= after <= len(history):
                raise ValueError("after_history must select a nonempty prior history")
            asked = _date(question.get("asked_at"), "question.asked_at")
            if _date(history[after - 1]["available_at"], "available_at") > asked:
                raise ValueError("query uses future history")
            fact_id = question.get("fact_id")
            if fact_id not in facts or facts[fact_id][0] > after:
                raise ValueError("query fact must already be written before asking")
            _, fact = facts[fact_id]
            later = [f for position, f in facts.values() if position <= after
                     and (f["subject"], f["relation"]) == (fact["subject"], fact["relation"])]
            if later[-1]["id"] != fact_id:
                raise ValueError("query answer references a superseded fact")
            candidates = _list(question.get("candidates"), "question.candidates")
            if len(candidates) < 2:
                raise ValueError("at least two candidates are required")
            for candidate in candidates:
                _text(candidate, "candidate")
                if candidate != candidate.strip():
                    raise ValueError("candidate whitespace is supplied by the scorer")
            if len({candidate.casefold() for candidate in candidates}) != len(candidates):
                raise ValueError("duplicate candidate answers")
            answer = question.get("answer_index")
            if type(answer) is not int or not 0 <= answer < len(candidates):
                raise ValueError("answer_index must select a candidate")
            if candidates[answer] != fact["value"]:
                raise ValueError("correct candidate must equal the supported fact value")
    for episode in episodes:
        donor = indexed.get(episode.get("donor_episode_id"))
        if donor is None or donor["event_family"] == episode["event_family"]:
            raise ValueError("fixed donor must belong to a different event family")
        if donor["split"] != episode["split"]:
            raise ValueError("fixed donor must stay within the target split")
        if donor["kind"] != episode["kind"]:
            raise ValueError("fixed donor must have the same real/synthetic kind")
        if len(donor["history"]) < max(q["after_history"] for q in episode["questions"]):
            raise ValueError("fixed donor lacks enough matched history blocks")
    return {"manifest_sha256": manifest_sha256(manifest), "episodes": len(episodes),
            "event_families": len(families),
            "by_split": {split: sum(e["split"] == split for e in episodes) for split in SPLITS}}


def question_prefix(question):
    return question["question"] + "\nAnswer:"


def _position_limit(method):
    limit = getattr(getattr(method.reader.model, "config", None), "max_position_embeddings", None)
    # Missing/malformed metadata and enormous sentinel values are not a bound.
    return limit if type(limit) is int and 0 < limit <= 2**31 - 1 else None


@torch.no_grad()
def score_candidates(method, state, prefix, candidates):
    """Teacher-forced candidate NLL, masking every visible-prefix position.

Primary candidate choice minimizes mean answer-token NLL. Sum NLL is also
reported; multi-token answers include all tokens and have no implicit EOS loss.
"""
    before = state_guard(method, state)
    scores, reference_prefix = [], None
    limit, batches = _position_limit(method), []
    # Validate every full completion before any candidate gets a reader forward.
    for candidate in candidates:
        ids, mask, labels = answer_batch(method.tokenizer, prefix, " " + candidate, method.device)
        if limit is not None and ids.shape[1] > limit:
            raise ValueError(f"Scoring input has {ids.shape[1]} tokens including candidate, "
                             f"exceeds max_position_embeddings={limit}; not truncating")
        prefix_ids = ids[labels.eq(-100)].tolist()
        if reference_prefix is None:
            reference_prefix = prefix_ids
        elif prefix_ids != reference_prefix:
            raise RuntimeError("candidate visible prefixes differ")
        batches.append((ids, mask, labels))
    for candidate, (ids, mask, labels) in zip(candidates, batches):
        output = method.reader(ids, mask, method.memory(state))
        valid = labels[:, 1:].ne(-100)
        losses = F.cross_entropy(output.logits[:, :-1, :].float().reshape(-1, output.logits.shape[-1]),
                                 labels[:, 1:].reshape(-1), reduction="none", ignore_index=-100)
        count = int(valid.sum())
        total = float(losses.reshape_as(valid)[valid].sum())
        if count < 1 or not math.isfinite(total):
            raise ValueError("candidate has no finite answer-token loss")
        target_ids = labels[:, 1:][valid].tolist()
        scores.append({"answer": candidate, "answer_tokens": count, "nll_sum": total, "nll": total / count,
                       "answer_token_sha256": hashlib.sha256(json.dumps(target_ids).encode()).hexdigest()})
    assert_read_only(method, state, before)
    return {"candidates": scores, "predicted_index": min(range(len(scores)), key=lambda i: scores[i]["nll"]),
            "prefix_tokens": len(reference_prefix), "model_position_limit": limit,
            "prefix_token_sha256": hashlib.sha256(json.dumps(reference_prefix).encode()).hexdigest()}


def _summary(records):
    conditions = {}
    for condition in CONDITIONS:
        selected = [row["conditions"][condition] for row in records]
        tokens = sum(row["gold_answer_tokens"] for row in selected)
        total = math.fsum(row["gold_nll_sum"] for row in selected)
        conditions[condition] = {"questions": len(selected), "gold_answer_tokens": tokens,
                                 "gold_nll": total / tokens if tokens else None,
                                 "candidate_accuracy": sum(row["correct"] for row in selected) / len(selected)
                                 if selected else None}
    empty, own, donor = (conditions[name] for name in ("empty", "self", "donor"))
    return {"conditions": conditions,
            "delta_nll_self_minus_empty": own["gold_nll"] - empty["gold_nll"] if records else None,
            "delta_nll_self_minus_donor": own["gold_nll"] - donor["gold_nll"] if records else None}


@torch.no_grad()
def evaluate_news(method, manifest, *, split="dev", checkpoint_metadata=None, progress_callback=None):
    """Freeze parameters; replay evidence before queries; keep conditions paired."""
    audit = validate_manifest(manifest)
    if checkpoint_metadata is not None and checkpoint_metadata.get("model") != manifest["model"]:
        raise ValueError("checkpoint metadata does not bind the manifest model boundary")
    if split not in SPLITS:
        raise ValueError("select dev or final_holdout explicitly")
    if method.name != "forward" or not method.supports_outer_training:
        raise ValueError("news evaluation requires a learned forward-only writer")
    method.eval()
    method.writer.requires_grad_(False)
    method.reader.model.requires_grad_(False)
    method.writer.zero_grad(set_to_none=True)
    method.reader.model.zero_grad(set_to_none=True)
    versions = [(parameter, parameter._version) for module in (method.writer, method.reader.model)
                for parameter in module.parameters()]
    writer_before = {name: value.detach().clone() for name, value in method.writer.state_dict().items()}
    episodes = [episode for episode in manifest["episodes"] if episode["split"] == split]
    if not episodes:
        raise ValueError("requested split has no episodes")
    indexed = {episode["id"]: episode for episode in manifest["episodes"]}
    records, write_tokens = [], {"self": 0, "donor": 0}
    for episode in episodes:
        donor = indexed[episode["donor_episode_id"]]
        states = {name: method.empty_state() for name in ("self", "empty", "donor")}
        queries = defaultdict(list)
        for question in episode["questions"]:
            queries[question["after_history"]].append(question)
        for index in range(max(queries)):
            for name, source in (("self", episode), ("donor", donor)):
                evidence = source["history"][index]["text"]
                # This call receives evidence alone, never question/candidate metadata.
                states[name] = method.write(states[name], evidence).state
                write_tokens[name] += len(method.tokenizer.encode(evidence, add_special_tokens=False))
            for question in queries[index + 1]:
                prefix = question_prefix(question)
                context = "\n\n".join(block["text"] for block in episode["history"][:index + 1])
                row = {"episode_id": episode["id"], "event_family": episode["event_family"],
                       "kind": episode["kind"], "synthetic": episode["synthetic"],
                       "question_id": question["id"], "after_history": index + 1,
                       "donor_episode_id": donor["id"], "conditions": {}}
                for condition in CONDITIONS:
                    state = states["empty" if condition == "context" else condition]
                    visible = context + "\n\n" + prefix if condition == "context" else prefix
                    scored = score_candidates(method, state, visible, question["candidates"])
                    gold = scored["candidates"][question["answer_index"]]
                    row["conditions"][condition] = {**scored,
                        "correct": scored["predicted_index"] == question["answer_index"],
                        "gold_answer_tokens": gold["answer_tokens"], "gold_nll_sum": gold["nll_sum"]}
                if len({row["conditions"][c]["prefix_token_sha256"] for c in ("self", "empty", "donor")}) != 1:
                    raise RuntimeError("memory conditions must have the same visible prefix")
                counts = [row["conditions"][c]["gold_answer_tokens"] for c in CONDITIONS]
                if len(set(counts)) != 1:
                    raise RuntimeError("all conditions must preserve the answer-token denominator")
                for candidate_index in range(len(question["candidates"])):
                    if len({row["conditions"][c]["candidates"][candidate_index]["answer_token_sha256"]
                            for c in CONDITIONS}) != 1:
                        raise RuntimeError("all conditions must score identical candidate answer tokens")
                records.append(row)
        if progress_callback is not None:
            progress_callback({"episode_id": episode["id"], "questions_complete": len(records)})
    writer_after = method.writer.state_dict()
    if writer_before.keys() != writer_after.keys() or any(
            old.shape != writer_after[name].shape or old.dtype != writer_after[name].dtype
            or old.device != writer_after[name].device or not torch.equal(old, writer_after[name])
            for name, old in writer_before.items()):
        raise RuntimeError("writer state_dict changed during frozen evaluation")
    if (any(parameter._version != version for parameter, version in versions)
            or any(parameter.requires_grad or parameter.grad is not None
                   for module in (method.writer, method.reader.model) for parameter in module.parameters())):
        raise RuntimeError("frozen parameters changed or received gradients")
    result = {"protocol": PROTOCOL, "complete": True, "split": split, **audit,
              "checkpoint": checkpoint_metadata, "checkpoint_identity_supplied": checkpoint_metadata is not None,
              "model": manifest["model"],
              "scoring": "minimum mean answer-token NLL; gold NLL pooled by answer-token count; first index wins exact ties",
              "context_scope": "same past evidence visible with empty memory; separate capability reference",
              "summary_by_kind": {kind: _summary([r for r in records if r["kind"] == kind]) for kind in KINDS
                                  if any(r["kind"] == kind for r in records)},
              "by_event_family": {family: _summary([r for r in records if r["event_family"] == family])
                                  for family in sorted({r["event_family"] for r in records})},
              "written_tokens": write_tokens, "records": records,
              "integrity_checks": {"writer_state_dict_exact_equal": True,
                                   "reader_parameter_versions_unchanged": True,
                                   "reader_content_hash_verified": False,
                                   "parameters_frozen_and_no_gradients": True,
                                   "scoring_max_position_embeddings": _position_limit(method)},
              "claims": {"training_on_news": False, "optimizer_steps": 0, "pretraining_novelty_certified": False,
                         "synthetic_is_real_news": False, "event_family_disjointness_requires_source_review": True}}
    json.dumps(result, allow_nan=False)
    return result

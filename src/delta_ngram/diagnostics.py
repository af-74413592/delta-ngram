"""Token provenance and joint row interventions for the legacy fact template.

Attribution uses actual tokenizer offsets and the reader's actual addresses.
Labels are used only for scoring; the selected rows are fixed by the original
old-query prefix, the first written state, and later evidence candidates.
"""
from __future__ import annotations

from collections import defaultdict

import torch

from .data import episode_count
from .evaluate import diagnose_rows, replace_rows, score_text
from .runtime import (CharacterTokenizer, assert_read_only, greedy_generate,
                      state_guard)


CATEGORIES = ("template", "name_internal", "name_boundary", "answer", "other")
GROUPS = (*CATEGORIES, "literal_shared", "hash_collision")


class _Unsupported(ValueError):
    pass


def _tokenization(tokenizer, text):
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if getattr(tokenizer, "is_fast", False):
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            actual = list(encoded["input_ids"])
            offsets = [list(pair) for pair in encoded["offset_mapping"]]
        except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
            raise _Unsupported(f"fast tokenizer did not provide usable offsets: {exc}") from exc
        if actual != ids:
            raise _Unsupported("offset tokenization differs from the generation tokenization")
        source = "fast_tokenizer_offsets"
    elif isinstance(tokenizer, CharacterTokenizer):
        if len(ids) != len(text):
            raise _Unsupported("CharacterTokenizer is not one token per character")
        offsets = [[index, index + 1] for index in range(len(text))]
        source = "CharacterTokenizer_character_offsets"
    else:
        raise _Unsupported("tokenizer needs fast offsets or the offline CharacterTokenizer")
    if len(offsets) != len(ids) or any(
            len(pair) != 2 or any(type(value) is not int for value in pair)
            or not 0 <= pair[0] <= pair[1] <= len(text) for pair in offsets):
        raise _Unsupported("tokenizer returned invalid character offsets")
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    tokens = list(convert(ids)) if convert else [tokenizer.decode([value]) for value in ids]
    if len(tokens) != len(ids):
        raise _Unsupported("tokenizer token strings do not align with token IDs")
    return {"text": text, "token_ids": ids, "tokens": tokens,
            "offsets": offsets, "offset_source": source}


def _sources(episode):
    count = episode_count(episode)
    names = getattr(episode, "names", None)
    if not names or len(names) != count or not all(isinstance(name, str) and name for name in names):
        raise _Unsupported("episode has no aligned fact entity names")
    if count < 2:
        raise _Unsupported("row interference diagnostics need at least two evidence blocks")
    result = []
    for index, name in enumerate(names):
        question = f"Profile\nName: {name}\nFavorite color:"
        evidence = episode.evidence(index)
        if episode.question(index) != question or not evidence.startswith(question + " ") or not evidence.endswith("\n"):
            raise _Unsupported("episode does not use the supported legacy fact template")
        value = evidence[len(question):-1]
        if not value.strip() or "\n" in value:
            raise _Unsupported("fact evidence needs one nonempty answer line")
        name_start = len("Profile\nName: ")
        name_span = [name_start, name_start + len(name)]
        answer_span = [len(question) + len(value) - len(value.lstrip()),
                       len(question) + len(value.rstrip())]
        result.append({"source": f"evidence_{index + 1}", "kind": "evidence", "index": index,
                       "text": evidence, "name_span": name_span, "answer_span": answer_span})
        if index in (0, count - 1):
            result.append({"source": f"query_{index + 1}", "kind": "query", "index": index,
                           "text": question, "name_span": name_span, "answer_span": None})
    return result


def _category(text, offsets, name_span, answer_span):
    # Ignore boundary whitespace, but do not guess where a zero-offset token came from.
    if any(start == end for start, end in offsets):
        return "other"
    positions = {position for start, end in offsets for position in range(start, end)
                 if not text[position].isspace()}
    if not positions:
        return "other"
    if answer_span and any(answer_span[0] <= position < answer_span[1] for position in positions):
        return "answer"
    name_positions = {position for position in positions if name_span[0] <= position < name_span[1]}
    if name_positions:
        return "name_internal" if name_positions == positions else "name_boundary"
    return "template"


def _occurrences(method, source):
    tokenized = _tokenization(method.tokenizer, source["text"])
    ids = torch.tensor([tokenized["token_ids"]], dtype=torch.long, device=method.device)
    _, addresses = method.reader._inputs(ids, torch.ones_like(ids))
    orders = tuple(method.reader.orders)
    if addresses.shape != (*ids.shape, len(orders)):
        raise _Unsupported("reader addresses do not align with its n-gram orders")
    addresses = addresses[0].cpu().tolist()
    events = []
    for position, row_slots in enumerate(addresses):
        for slot, row in enumerate(row_slots):
            if row == 0:
                continue
            order = orders[slot]
            start = position + 1 - order
            if start < 0:
                raise _Unsupported("reader supplied a nonzero incomplete n-gram")
            window = slice(start, position + 1)
            offsets = tokenized["offsets"][window]
            span = [min(pair[0] for pair in offsets), max(pair[1] for pair in offsets)]
            events.append({"row": row, "source": source["source"], "kind": source["kind"],
                           "index": source["index"], "token_position": position,
                           "token_positions": list(range(start, position + 1)),
                           "slot": slot, "order": order,
                           "token_ids": tokenized["token_ids"][window],
                           "tokens": tokenized["tokens"][window], "offsets": offsets,
                           "text_span": span, "text": source["text"][span[0]:span[1]],
                           "category": _category(source["text"], offsets, source["name_span"], source["answer_span"])})
    return {**source, **tokenized}, events


def _signature(event):
    return event["slot"], event["order"], tuple(event["token_ids"])


def _signature_json(signature):
    slot, order, ids = signature
    return {"slot": slot, "order": order, "token_ids": list(ids)}


def _assert_replacement(original, replacement, rows, values):
    if not torch.equal(original.rows, replacement.rows) or not torch.equal(original.counts, replacement.counts):
        raise RuntimeError("row intervention changed addresses or counts")
    if ((original.precision is None) != (replacement.precision is None)
            or (original.precision is not None and not torch.equal(original.precision, replacement.precision))):
        raise RuntimeError("row intervention changed precision")
    outside = ~torch.isin(original.rows, rows)
    if not torch.equal(original.values[outside], replacement.values[outside]):
        raise RuntimeError("row intervention changed values outside its group")
    if not torch.equal(replacement.lookup(rows)[0], values):
        raise RuntimeError("row intervention did not install its specified values")


def _effect(before, after):
    result = {"raw_changed": before["raw"] != after["raw"]}
    for key in ("first_word_correct", "strict_first_line_correct"):
        result[key] = {"accuracy_delta": int(after[key]) - int(before[key]),
                       "correct_to_wrong": bool(before[key] and not after[key]),
                       "wrong_to_correct": bool(not before[key] and after[key])}
    return result


@torch.no_grad()
def detailed_row_diagnostics(method, episode, first_state, final_state, max_new_tokens,
                             *, shared_diagnostic=None):
    """Attribute rows and intervene on each fixed group without making writes.

    ``shared_diagnostic`` may be the evaluator's already computed ``diagnose_rows``
    result. It is returned unchanged and avoids repeating its six generations.
    Categories describe any relevant occurrence of a row, so group effects are
    overlapping joint interventions and must not be added or read as per-row effects.
    """
    unsupported = {"supported": False, "shared_group": shared_diagnostic}
    if not getattr(method, "supports_row_diagnostics", False):
        return {**unsupported, "unsupported_reason": "method does not support row diagnostics"}
    if getattr(method.reader, "backend", "external_hash") != "external_hash":
        return {**unsupported, "unsupported_reason": "token provenance requires the external hash reader"}
    try:
        sources, occurrences = [], []
        for source in _sources(episode):
            tokenized, events = _occurrences(method, source)
            sources.append(tokenized)
            occurrences.extend(events)
    except _Unsupported as exc:
        return {**unsupported, "unsupported_reason": str(exc)}

    before_first, before_final = state_guard(method, first_state), state_guard(method, final_state)
    first, final = method.memory(first_state), method.memory(final_state)
    by_row = defaultdict(list)
    for event in occurrences:
        by_row[event["row"]].append(event)
    q1_rows = {event["row"] for event in occurrences if event["kind"] == "query" and event["index"] == 0}
    later_rows = {event["row"] for event in occurrences if event["kind"] == "evidence" and event["index"] > 0}
    selected = set(first.rows.cpu().tolist()) & q1_rows & later_rows
    if not selected.issubset(set(final.rows.cpu().tolist())):
        raise ValueError("final memory is missing a selected first-state row")
    if shared_diagnostic is None:
        shared_diagnostic = diagnose_rows(method, episode, first_state, final_state, max_new_tokens)
    if sorted(selected) != shared_diagnostic["old_overlap_rows"]:
        raise ValueError("detailed and shared diagnostics select different physical rows")

    grouped = {name: [] for name in GROUPS}
    rows = []
    for row, events in sorted(by_row.items()):
        old = [event for event in events if event["kind"] == "query" and event["index"] == 0]
        later = [event for event in events if event["kind"] == "evidence" and event["index"] > 0]
        old_keys, later_keys = {_signature(event) for event in old}, {_signature(event) for event in later}
        literal = old_keys & later_keys
        collision = bool(old_keys and later_keys and len(old_keys | later_keys) > 1)
        categories = sorted({event["category"] for event in old + later})
        rows.append({"row": row, "selected": row in selected, "categories": categories,
                     "q1_categories": sorted({event["category"] for event in old}),
                     "later_evidence_categories": sorted({event["category"] for event in later}),
                     "literal_shared": bool(literal), "hash_collision": collision,
                     "literal_shared_tuples": [_signature_json(key) for key in sorted(literal)],
                     "q1_distinct_tuples": [_signature_json(key) for key in sorted(old_keys)],
                     "later_evidence_distinct_tuples": [_signature_json(key) for key in sorted(later_keys)],
                     "occurrences": events})
        if row in selected:
            for category in categories:
                grouped[category].append(row)
            if literal:
                grouped["literal_shared"].append(row)
            if collision:
                grouped["hash_collision"].append(row)

    baseline = shared_diagnostic["predictions"]
    base_keys = ("write1_q1", "final_q1", "final_new")
    cache = {(): {"write1_zero_group_q1": baseline["write1_q1"],
                  "final_restore_group_q1": baseline["final_q1"],
                  "final_restore_group_new": baseline["final_new"]}}
    if selected:
        cache[tuple(sorted(selected))] = {
            "write1_zero_group_q1": baseline["write1_zero_shared_q1"],
            "final_restore_group_q1": baseline["final_restore_shared_q1"],
            "final_restore_group_new": baseline["final_restore_shared_new"]}
    interventions = {}
    for name, row_list in grouped.items():
        key = tuple(row_list)
        chosen = first.rows.new_tensor(row_list)
        old_values = first.lookup(chosen)[0]
        new_values = final.lookup(chosen)[0]
        reused = key in cache
        if not reused:
            zero_memory = replace_rows(first, chosen, torch.zeros_like(old_values))
            restore_memory = replace_rows(final, chosen, old_values)
            _assert_replacement(first, zero_memory, chosen, torch.zeros_like(old_values))
            _assert_replacement(final, restore_memory, chosen, old_values)
            zero = method.replace_memory(first_state, zero_memory)
            restored = method.replace_memory(final_state, restore_memory)
            _assert_replacement(first, method.memory(zero), chosen, torch.zeros_like(old_values))
            _assert_replacement(final, method.memory(restored), chosen, old_values)
            predictions = {}
            for endpoint, state, index in (("write1_zero_group_q1", zero, 0),
                                           ("final_restore_group_q1", restored, 0),
                                           ("final_restore_group_new", restored, episode_count(episode) - 1)):
                guard = state_guard(method, state)
                raw = greedy_generate(method, state, episode.question(index), max_new_tokens)
                assert_read_only(method, state, guard)
                predictions[endpoint] = score_text(raw, episode.answer(index))
            cache[key] = predictions
        predictions = {**{endpoint: baseline[endpoint] for endpoint in base_keys}, **cache[key]}
        interventions[name] = {
            "rows": row_list, "row_count": len(row_list), "intervention_applied": bool(row_list),
            "reused_identical_group_predictions": reused,
            "shared_change_frobenius_norm": float((new_values - old_values).norm()),
            "predictions": predictions,
            "effects": {"zero_at_write1_q1": _effect(predictions["write1_q1"], predictions["write1_zero_group_q1"]),
                        "restore_at_final_q1": _effect(predictions["final_q1"], predictions["final_restore_group_q1"]),
                        "restore_at_final_new": _effect(predictions["final_new"], predictions["final_restore_group_new"])}}
    assert_read_only(method, first_state, before_first)
    assert_read_only(method, final_state, before_final)
    return {"supported": True, "schema_version": 1, "shared_group": shared_diagnostic,
            "selection": {"rows": sorted(selected), "row_count": len(selected),
                          "q1_prefix_rows": sorted(q1_rows), "later_candidate_rows": sorted(later_rows),
                          "definition": "first-state rows intersect original old-query prefix rows intersect later evidence candidate rows",
                          "correctness_filter": False, "update_size_filter": False},
            "tokenization": sources, "rows": rows, "groups": interventions,
            "category_scope": "any original old-query or later-evidence occurrence; E1 and final new-query occurrences are provenance only",
            "literal_identity": "same n-gram slot, order, and token-ID tuple across old query and later evidence",
            "collision_identity": "same physical row with at least one differing tuple across old query and later evidence; may coexist with literal sharing",
            "scope": "overlapping joint physical-row group interventions; groups are not additive and do not establish individual-row effects",
            "invariants": {"addresses_unchanged": True, "counts_unchanged": True,
                           "precision_unchanged": True, "original_states_unchanged": True}}

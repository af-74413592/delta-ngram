"""The sole evaluator: generation, acquisition/retention, controls, and row interventions."""
from __future__ import annotations

from dataclasses import asdict
import time

import torch

from .data import E1Episode, episode_count, episode_queries, question_for_style
from .interfaces import MemoryMethod
from .runtime import (greedy_generate, log_event, mean, normalized_answer,
                      normalized_first_line, save_json, timed, token_ids)
from .state import SparseMemoryState


def physical_rows(method, text):
    ids = token_ids(method.tokenizer, text, method.device)
    _, rows = method.reader._inputs(ids, torch.ones_like(ids))
    return torch.unique(rows[rows.ne(0)], sorted=True)


def intersect(left, right):
    return left[torch.isin(left, right)]


def replace_rows(state, rows, values):
    if not rows.numel():
        return state
    positions = torch.searchsorted(state.rows, rows)
    if bool((positions >= state.rows.numel()).any()) or not torch.equal(state.rows[positions], rows):
        raise ValueError("diagnostic rows must already exist")
    return SparseMemoryState(state.rows, state.values.index_copy(0, positions, values),
                             state.counts, state.precision)


def score_text(raw, answer):
    return {"raw": raw, "target": answer.strip(),
            "first_word_correct": normalized_answer(raw) == normalized_answer(answer),
            "strict_first_line_correct": normalized_first_line(raw) == normalized_first_line(answer)}


def coverage(method, state, question):
    rows = physical_rows(method, question)
    memory = method.memory(state)
    hits = intersect(rows, memory.rows)
    return {"query_rows": int(rows.numel()), "written_query_rows": int(hits.numel()),
            "written_fraction": hits.numel() / rows.numel() if rows.numel() else None}


def styled_key(key, style):
    return key if style == "canonical" else f"{key}__{style}"


def _first_item_metrics(records, score_key="correct", style="canonical"):
    if not records:
        return {"A0": None, "A1": None, "AT": None, "conditional_retention": None,
                "conditional_retention_denominator": 0, "acquired_and_retained_all_samples": None}
    rows = [(r[score_key][styled_key("empty_q1", style)], r[score_key][styled_key("write1_q1", style)],
             r[score_key].get(styled_key(f"write{r['evidence_count']}_q1", style))) for r in records]
    eligible = [(c0, c1, ct) for c0, c1, ct in rows if ct is not None]
    denominator = sum(c1 for _, c1, _ in eligible)
    return {"A0": mean([c0 for c0, _, _ in rows]), "A1": mean([c1 for _, c1, _ in rows]),
            "AT": mean([ct for _, _, ct in eligible]),
            "conditional_retention": sum(c1 and ct for _, c1, ct in eligible) / denominator if denominator else None,
            "conditional_retention_denominator": denominator,
            "acquired_and_retained_all_samples": mean([not c0 and c1 and ct for c0, c1, ct in eligible]),
            **({"retention_eligible": len(eligible), "superseded_excluded": len(rows)-len(eligible)}
               if any("events" in r for r in records) else {})}


def version_metrics(records, style, score_key="correct"):
    """Aggregate only final active versions; revision success is scored separately."""
    suffix = "endpoints_by_style" if score_key == "correct" else "strict_endpoints_by_style"
    endpoints = [e for r in records for e in r[suffix][style]]
    eligible = [e for e in endpoints if e["retention_eligible"]]
    denominator = sum(e["C1"] for e in eligible)
    retained = [e for e in eligible if e["intervening_writes"] > 0]
    retained_denominator = sum(e["C1"] for e in retained)
    revision_key = "revision_probes" if score_key == "correct" else "strict_revision_probes"
    revisions = [e for r in records for e in r[revision_key][style]]
    final_revisions = [e for e in revisions if e["final_success"] is not None]
    collateral = [e for r in revisions for e in r["collateral"]]
    collateral_denominator = sum(e["before"] for e in collateral)
    return {
        "active_final": {"count": len(eligible), "superseded_excluded": len(endpoints)-len(eligible),
            "C0": mean([e["C0"] for e in eligible]), "C1": mean([e["C1"] for e in eligible]),
            "CT": mean([e["CT"] for e in eligible]),
            "conditional_retention": sum(e["C1"] and e["CT"] for e in eligible)/denominator if denominator else None,
            "conditional_retention_denominator": denominator,
            "acquired_and_retained_all_samples": mean([not e["C0"] and e["C1"] and e["CT"] for e in eligible])},
        "with_later_writes": {"count": len(retained), "CT": mean([e["CT"] for e in retained]),
            "conditional_retention": sum(e["C1"] and e["CT"] for e in retained)/retained_denominator if retained_denominator else None,
            "conditional_retention_denominator": retained_denominator},
        "revisions": {"count": len(revisions), "immediate_success": mean([e["immediate_success"] for e in revisions]),
            "immediate_still_previous": mean([e["immediate_still_previous"] for e in revisions]),
            "final_count": len(final_revisions), "final_success": mean([e["final_success"] for e in final_revisions]),
            "final_still_previous": mean([e["final_still_previous"] for e in final_revisions])},
        "collateral_same_entity": {"count": len(collateral), "before": mean([e["before"] for e in collateral]),
            "after": mean([e["after"] for e in collateral]), "previously_correct_denominator": collateral_denominator,
            "loss_among_previously_correct": sum(e["before"] and not e["after"] for e in collateral)/collateral_denominator if collateral_denominator else None}}


@torch.no_grad()
def evaluate(method: MemoryMethod, episodes, args, *, step=0, deadline=float("inf"),
             counterfactuals=True, row_diagnostics=False, save=True,
             query_styles=("canonical",), detailed_rows=False):
    """Replay any method through the same write/query schedule and scorer.

    Online optimizers reopen autograd inside their evidence-only ``write``.
    A0/A1/AT refer to the first item's empty/immediate/final endpoints. Per-item
    endpoints are also retained, including the number of intervening writes.
    """
    if (not query_styles or "canonical" not in query_styles or len(set(query_styles)) != len(query_styles)
            or any(s not in ("canonical", "named_question") for s in query_styles)):
        raise ValueError("query_styles must contain canonical and unique supported styles")
    if detailed_rows and not row_diagnostics:
        raise ValueError("detailed_rows requires row_diagnostics")
    method.eval()
    records = []
    write_seconds, query_seconds, writes, queries = 0.0, 0.0, 0, 0
    started = time.perf_counter()
    for episode in episodes:
        if time.monotonic() >= deadline:
            break
        n = episode_count(episode)
        is_e1 = isinstance(episode, E1Episode)
        empty = method.empty_state()
        predictions, correct, strict_correct, row_coverage = {}, {}, {}, {}
        write_details = []

        def write(state, index, wrong=False):
            nonlocal write_seconds, writes
            result, duration = timed(lambda: method.write(state, episode.evidence(index, wrong=wrong)), method.device)
            write_seconds += duration
            writes += 1
            if not wrong:
                write_details.append({"block": index + 1, "write_seconds": duration, **result.diagnostics})
            return result.state

        def query(key, state, index, prompt=None, query_method=None, style="canonical"):
            nonlocal query_seconds, queries
            active = query_method or method
            prompt = prompt or question_for_style(episode, index, style)
            key = styled_key(key, style)
            prediction, duration = timed(lambda: greedy_generate(
                active, state, prompt, args.max_new_tokens), method.device)
            query_seconds += duration
            queries += 1
            predictions[key] = prediction
            correct[key] = normalized_answer(prediction) == normalized_answer(episode.answer(index))
            strict_correct[key] = normalized_first_line(prediction) == normalized_first_line(episode.answer(index))
            if row_diagnostics and active.supports_row_diagnostics:
                row_coverage[key] = coverage(active, state, prompt)

        for index in range(n):
            for style in query_styles:
                query(f"empty_q{index+1}", empty, index, style=style)
        state = empty
        first = None
        for index in range(n):
            state = write(state, index)
            if index == 0:
                first = state
            indices = [q.index for q in episode_queries(episode, index+1)] if is_e1 else range(index+1)
            for query_index in indices:
                for style in query_styles:
                    query(f"write{index+1}_q{query_index+1}", state, query_index, style=style)
        final = state
        for style in query_styles:
            query("zero_after_write_q1", method.empty_state(), 0, style=style)
            if predictions[styled_key("zero_after_write_q1", style)] != predictions[styled_key("empty_q1", style)]:
                raise RuntimeError("Zeroing explicit memory failed to restore deterministic baseline")
        final_indices = episode.active_query_indices(n) if is_e1 else range(n)
        wrong_follow = {}
        if counterfactuals:
            wrong_state = method.empty_state()
            for index in range(n):
                wrong_state = write(wrong_state, index, wrong=True)
            wrong_state = method.counterfactual(final, wrong_state)
            for index in final_indices:
                for style in query_styles:
                    query(f"wrong_values_q{index+1}", wrong_state, index, style=style)
                    wrong_follow[styled_key(f"q{index+1}", style)] = (normalized_answer(predictions[styled_key(f"wrong_values_q{index+1}", style)])
                                                    == normalized_answer(episode.corrupted_answer(index)))
        if len(records) < args.openbook_episodes:
            # This reference uses the same decoder with an explicit empty table.
            from .methods import EmptyMemoryMethod
            reference = EmptyMemoryMethod(method.reader, method.tokenizer, method.device, method.max_evidence_tokens)
            context = "\n".join(episode.evidence(index) for index in range(n)) + "\n"
            for index in final_indices:
                for style in query_styles:
                    query(f"openbook_q{index+1}", reference.empty_state(), index,
                          context + question_for_style(episode, index, style), query_method=reference, style=style)
        diagnostic = None
        detailed = None
        first_is_current = not is_e1 or episode.superseded_at(0) is None
        if row_diagnostics and method.supports_row_diagnostics and n >= 2 and first_is_current:
            diagnostic = diagnose_rows(method, episode, first, final, args.max_new_tokens)
            if detailed_rows:
                from .diagnostics import detailed_row_diagnostics
                detailed = detailed_row_diagnostics(method, episode, first, final, args.max_new_tokens,
                                                    shared_diagnostic=diagnostic)
        elif detailed_rows:
            detailed = {"supported": False, "reason": "first version superseded; old-version intervention is not a retention test"
                        if not first_is_current else "method or episode does not support row interventions"}
        memory = method.memory(final)
        table_bytes = sum(x.numel() * x.element_size() for x in
                          (memory.rows, memory.values, memory.counts, memory.precision) if x is not None)
        def make_endpoints(scores, style):
            return [{"query_index": i, "intervening_writes": n-i-1,
                     "C0": scores[styled_key(f"empty_q{i+1}", style)],
                     "C1": scores[styled_key(f"write{i+1}_q{i+1}", style)],
                     "CT": scores.get(styled_key(f"write{n}_q{i+1}", style)),
                     **({"retention_eligible": episode.superseded_at(i) is None,
                         "superseded_at": episode.superseded_at(i), "event_kind": episode.event_kind(i),
                         "fact_key": episode.fact_key(i)} if is_e1 else {})} for i in range(n)]

        endpoints = make_endpoints(correct, "canonical")
        record = {**asdict(episode), "predictions": predictions, "correct": correct,
                  "strict_first_line_correct": strict_correct,
                  "wrong_evidence_colors": [episode.corrupted_answer(i).strip() for i in range(n)],
                  "wrong_values_follow_corrupted_evidence": wrong_follow,
                  "state_rows": int(memory.rows.numel()), "state_tensor_bytes": table_bytes,
                  "evidence_count": n, "endpoints": endpoints, "writes": write_details,
                  "size_report": method.size_report(final), "persistent_state_bytes": method.state_bytes(final)}
        if row_diagnostics:
            record.update({"row_coverage": row_coverage, "row_diagnostic": diagnostic})
        if detailed_rows:
            record["detailed_row_diagnostic"] = detailed
        if is_e1 or len(query_styles) > 1:
            record["endpoints_by_style"] = {s: make_endpoints(correct, s) for s in query_styles}
            record["strict_endpoints_by_style"] = {s: make_endpoints(strict_correct, s) for s in query_styles}
        if is_e1:
            for field, scoring, normalize in (("revision_probes", correct, normalized_answer),
                                               ("strict_revision_probes", strict_correct, normalized_first_line)):
                record[field] = {}
                for style in query_styles:
                    probes = []
                    for i in range(n):
                        previous = episode.previous_version(i)
                        if previous is None:
                            continue
                        immediate = styled_key(f"write{i+1}_q{i+1}", style)
                        final_key = styled_key(f"write{n}_q{i+1}", style)
                        collateral = [j for j in episode.active_query_indices(i)
                                      if episode.fact_key(j)[0] == episode.fact_key(i)[0]
                                      and episode.fact_key(j) != episode.fact_key(i)]
                        probes.append({"query_index": i, "previous_version": previous,
                            "immediate_success": scoring[immediate],
                            "immediate_still_previous": normalize(predictions[immediate]) == normalize(episode.answer(previous)),
                            "final_success": scoring.get(final_key),
                            "final_still_previous": normalize(predictions[final_key]) == normalize(episode.answer(previous)) if final_key in predictions else None,
                            "collateral": [{"query_index": j, "before": scoring[styled_key(f"write{i}_q{j+1}", style)],
                                            "after": scoring[styled_key(f"write{i+1}_q{j+1}", style)]} for j in collateral]})
                    record[field][style] = probes
        records.append(record)
    keys = sorted({key for record in records for key in record["correct"]})
    scores = {key: mean([record["correct"][key] for record in records if key in record["correct"]]) for key in keys}
    strict_scores = {key: mean([record["strict_first_line_correct"][key] for record in records
                               if key in record["strict_first_line_correct"]]) for key in keys}
    endpoints = _first_item_metrics(records)
    scores.update({key: value for key, value in endpoints.items() if key not in ("A0", "A1", "AT")})
    for key in sorted({key for record in records for key in record["wrong_values_follow_corrupted_evidence"]}):
        scores[f"wrong_values_{key}_follow_corrupted_evidence"] = mean([
            r["wrong_values_follow_corrupted_evidence"][key] for r in records
            if key in r["wrong_values_follow_corrupted_evidence"]])
    result = {"event": "evaluation", "step": step, "episodes": len(records),
              "requested_episodes": len(episodes), "complete": len(records) == len(episodes),
              "scores": scores, "strict_first_line_exact": strict_scores,
              "acquisition_retention": endpoints, "strict_acquisition_retention": _first_item_metrics(records, "strict_first_line_correct"),
              "evaluation_seconds": time.perf_counter()-started, "write_seconds_total": write_seconds,
              "write_count": writes, "query_seconds_total": query_seconds, "query_count": queries,
              "write_seconds_mean": write_seconds/writes if writes else None,
              "query_seconds_mean": query_seconds/queries if queries else None,
              "state_tensor_bytes_mean": mean([r["state_tensor_bytes"] for r in records]),
              "persistent_state_bytes_mean": mean([r["persistent_state_bytes"] for r in records]),
              "scoring": "greedy free generation; normalized first answer word; no choices or substring matching",
              "wrong_state_protocol": "original row addresses and counts; values looked up from wrong-evidence state"
              if method.supports_row_diagnostics else "same evidence-block positions; text replaced by corrupted evidence",
              "openbook_scope": "full two-evidence context; separate capability reference, not an upper bound"
              if all(r["evidence_count"] == 2 for r in records) else
              "full evidence context; separate capability reference, not an upper bound"}
    if row_diagnostics:
        result["row_diagnostics"] = summarize_diagnostics([r["row_diagnostic"] for r in records if r["row_diagnostic"] is not None])
    if any("events" in r for r in records) or len(query_styles) > 1:
        result["query_style_metrics"] = {style: {
            "acquisition_retention": _first_item_metrics(records, style=style),
            "strict_acquisition_retention": _first_item_metrics(records, "strict_first_line_correct", style),
            **({"version_metrics": version_metrics(records, style),
                "strict_version_metrics": version_metrics(records, style, "strict_first_line_correct")}
               if records and all("events" in r for r in records) else {})} for style in query_styles}
    if save:
        save_json(args.out / f"eval_step_{step:06d}.json", {**result, "records": records})
        log_event(args.out / "metrics.jsonl", {**result, "examples": records[:2]})
    return {**result, "records": records} if not save else result


@torch.no_grad()
def diagnose_rows(method, episode, first_state, final_state, max_new_tokens):
    """One group intervention implementation for forward and gradient tables."""
    first, final = method.memory(first_state), method.memory(final_state)
    read_rows = physical_rows(method, episode.question(0))
    writes = [physical_rows(method, episode.evidence(i)) for i in range(1, episode_count(episode))]
    later_rows = torch.unique(torch.cat(writes), sorted=True)
    overlap = intersect(intersect(first.rows, read_rows), later_rows)
    old, _ = first.lookup(overlap)
    new, _ = final.lookup(overlap)
    zeroed = method.replace_memory(first_state, replace_rows(first, overlap, torch.zeros_like(old)))
    restored = method.replace_memory(final_state, replace_rows(final, overlap, old))
    last = episode_count(episode) - 1
    predictions = {}
    for key, state, index in (("write1_q1", first_state, 0), ("final_q1", final_state, 0),
                              ("final_new", final_state, last), ("write1_zero_shared_q1", zeroed, 0),
                              ("final_restore_shared_q1", restored, 0), ("final_restore_shared_new", restored, last)):
        predictions[key] = score_text(greedy_generate(method, state, episode.question(index), max_new_tokens), episode.answer(index))
    return {"old_overlap_rows": overlap.cpu().tolist(), "q1_prefix_rows": read_rows.cpu().tolist(),
            "later_candidate_rows": later_rows.cpu().tolist(),
            "old_overlap_fraction": overlap.numel()/read_rows.numel() if read_rows.numel() else None,
            "shared_change_frobenius_norm": float((new-old).norm()), "predictions": predictions,
            "definition": "first-state rows intersect old-query prefix rows intersect later evidence candidate rows",
            "scope": "joint physical-row group interventions; no correctness or update-size filtering"}


def summarize_diagnostics(records):
    keys = sorted({key for row in records for key in row["predictions"]})
    return {"episodes": len(records), "mean_old_overlap_fraction": mean([r["old_overlap_fraction"] for r in records
                                                                           if r["old_overlap_fraction"] is not None]),
            "first_word": {key: mean([r["predictions"][key]["first_word_correct"] for r in records]) for key in keys},
            "strict_first_line": {key: mean([r["predictions"][key]["strict_first_line_correct"] for r in records]) for key in keys}}

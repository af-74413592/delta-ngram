"""The sole evaluator: generation, acquisition/retention, controls, and row interventions."""
from __future__ import annotations

from dataclasses import asdict
import time

import torch

from .data import episode_count
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


def _first_item_metrics(records, score_key="correct"):
    if not records:
        return {"A0": None, "A1": None, "AT": None, "conditional_retention": None,
                "conditional_retention_denominator": 0, "acquired_and_retained_all_samples": None}
    rows = [(r[score_key]["empty_q1"], r[score_key]["write1_q1"],
             r[score_key][f"write{r['evidence_count']}_q1"]) for r in records]
    denominator = sum(c1 for _, c1, _ in rows)
    return {"A0": mean([c0 for c0, _, _ in rows]), "A1": mean([c1 for _, c1, _ in rows]),
            "AT": mean([ct for _, _, ct in rows]),
            "conditional_retention": sum(c1 and ct for _, c1, ct in rows) / denominator if denominator else None,
            "conditional_retention_denominator": denominator,
            "acquired_and_retained_all_samples": mean([not c0 and c1 and ct for c0, c1, ct in rows])}


@torch.no_grad()
def evaluate(method: MemoryMethod, episodes, args, *, step=0, deadline=float("inf"),
             counterfactuals=True, row_diagnostics=False, save=True):
    """Replay any method through the same write/query schedule and scorer.

    Online optimizers reopen autograd inside their evidence-only ``write``.
    A0/A1/AT refer to the first item's empty/immediate/final endpoints. Per-item
    endpoints are also retained, including the number of intervening writes.
    """
    method.eval()
    records = []
    write_seconds, query_seconds, writes, queries = 0.0, 0.0, 0, 0
    started = time.perf_counter()
    for episode in episodes:
        if time.monotonic() >= deadline:
            break
        n = episode_count(episode)
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

        def query(key, state, index, prompt=None, query_method=None):
            nonlocal query_seconds, queries
            active = query_method or method
            prediction, duration = timed(lambda: greedy_generate(
                active, state, prompt or episode.question(index), args.max_new_tokens), method.device)
            query_seconds += duration
            queries += 1
            predictions[key] = prediction
            correct[key] = normalized_answer(prediction) == normalized_answer(episode.answer(index))
            strict_correct[key] = normalized_first_line(prediction) == normalized_first_line(episode.answer(index))
            if row_diagnostics and active.supports_row_diagnostics:
                row_coverage[key] = coverage(active, state, prompt or episode.question(index))

        for index in range(n):
            query(f"empty_q{index+1}", empty, index)
        state = empty
        first = None
        for index in range(n):
            state = write(state, index)
            if index == 0:
                first = state
            for query_index in range(index + 1):
                query(f"write{index+1}_q{query_index+1}", state, query_index)
        final = state
        query("zero_after_write_q1", method.empty_state(), 0)
        if predictions["zero_after_write_q1"] != predictions["empty_q1"]:
            raise RuntimeError("Zeroing explicit memory failed to restore deterministic baseline")
        wrong_follow = {}
        if counterfactuals:
            wrong_state = method.empty_state()
            for index in range(n):
                wrong_state = write(wrong_state, index, wrong=True)
            wrong_state = method.counterfactual(final, wrong_state)
            for index in range(n):
                query(f"wrong_values_q{index+1}", wrong_state, index)
                wrong_follow[f"q{index+1}"] = (normalized_answer(predictions[f"wrong_values_q{index+1}"])
                                                == normalized_answer(episode.corrupted_answer(index)))
        if len(records) < args.openbook_episodes:
            # This reference uses the same decoder with an explicit empty table.
            from .methods import EmptyMemoryMethod
            reference = EmptyMemoryMethod(method.reader, method.tokenizer, method.device, method.max_evidence_tokens)
            context = "\n".join(episode.evidence(index) for index in range(n)) + "\n"
            for index in range(n):
                query(f"openbook_q{index+1}", reference.empty_state(), index,
                      context + episode.question(index), query_method=reference)
        diagnostic = None
        if row_diagnostics and method.supports_row_diagnostics and n >= 2:
            diagnostic = diagnose_rows(method, episode, first, final, args.max_new_tokens)
        memory = method.memory(final)
        table_bytes = sum(x.numel() * x.element_size() for x in
                          (memory.rows, memory.values, memory.counts, memory.precision) if x is not None)
        endpoints = [{"query_index": i, "intervening_writes": n-i-1,
                      "C0": correct[f"empty_q{i+1}"], "C1": correct[f"write{i+1}_q{i+1}"],
                      "CT": correct[f"write{n}_q{i+1}"]} for i in range(n)]
        record = {**asdict(episode), "predictions": predictions, "correct": correct,
                  "strict_first_line_correct": strict_correct,
                  "wrong_evidence_colors": [episode.corrupted_answer(i).strip() for i in range(n)],
                  "wrong_values_follow_corrupted_evidence": wrong_follow,
                  "state_rows": int(memory.rows.numel()), "state_tensor_bytes": table_bytes,
                  "evidence_count": n, "endpoints": endpoints, "writes": write_details,
                  "size_report": method.size_report(final), "persistent_state_bytes": method.state_bytes(final)}
        if row_diagnostics:
            record.update({"row_coverage": row_coverage, "row_diagnostic": diagnostic})
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

"""CPU-only first-token and continuation decomposition of sealed NLL rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from render_nll_report import CONDITIONS, PAIRS, audit

PROTOCOL_SHA = "495e2164aa512bd9da0ebd10537815d43dc32408c83007810c802f108ec21b68"
SEED, REPLICATES = 20261022, 10000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def split_questions(rows):
    """Keep paired records; answers of length one have no continuation record."""
    lookup = {(r["qid"], r["condition"]): r for r in rows}
    require(len(lookup) == len(rows), "duplicate paired row")
    questions = []
    for qid in sorted({r["qid"] for r in rows}):
        triple = [lookup[qid, c] for c in CONDITIONS]
        base = triple[0]
        require(all(r["target_tokens"] == base["target_tokens"] and r["sample_id"] == base["sample_id"]
                    for r in triple), "paired population differs")
        count = base["target_tokens"]
        require(type(count) is int and count >= 1, "invalid target length")
        result = {"qid": qid, "sample_id": base["sample_id"], "target_tokens": count, "groups": {}}
        for name, start, stop in (("first_token", 0, 1), ("continuation", 1, count)):
            if start == stop:
                continue
            records = {}
            for condition, row in zip(CONDITIONS, triple):
                logs = row["token_logprobs"]
                require(len(logs) == count and all(math.isfinite(v) and v <= 1e-6 for v in logs),
                        "invalid token log probabilities")
                losses = [-v for v in logs[start:stop]]
                total = math.fsum(losses)
                records[condition] = {"loss_sum": total, "mean_loss": total / len(losses)}
            result["groups"][name] = {"token_count": stop - start, "conditions": records}
        questions.append(result)
    return questions


def aggregate(questions, sampled=None):
    """Resample conversations, retaining all questions and their multiplicity."""
    sids = sorted({q["sample_id"] for q in questions})
    if sampled is None:
        sampled = np.random.default_rng(SEED).integers(0, len(sids), (REPLICATES, len(sids)))
    sampled = np.asarray(sampled)
    require(sampled.ndim == 2 and sampled.shape[1] == len(sids)
            and np.issubdtype(sampled.dtype, np.integer)
            and bool(((sampled >= 0) & (sampled < len(sids))).all()), "invalid cluster draws")
    weights = np.stack([(sampled == i).sum(axis=1) for i in range(len(sids))], axis=1)
    groups = {}
    for group in ("first_token", "continuation"):
        selected = [q for q in questions if group in q["groups"]]
        by_sid = [[q["groups"][group] for q in selected if q["sample_id"] == sid] for sid in sids]
        counts = np.array([len(qs) for qs in by_sid], dtype=np.int64)
        tokens = np.array([sum(q["token_count"] for q in qs) for qs in by_sid], dtype=np.int64)
        require(bool((counts > 0).all()) and bool((tokens > 0).all()),
                "every conversation must have nonempty observations for both registered groups")
        metrics, per_conversation = {}, {}
        for sid, count, token_count in zip(sids, counts, tokens):
            per_conversation[sid] = {"question_count": int(count), "token_count": int(token_count)}
        for metric, field, denominator in (("query_mean_nll", "mean_loss", counts),
                                           ("token_pooled_nll", "loss_sum", tokens)):
            sums = np.array([[math.fsum(q["conditions"][c][field] for q in qs)
                              for c in CONDITIONS] for qs in by_sid], dtype=np.float64)
            points = sums.sum(axis=0) / denominator.sum()
            draws = (weights @ sums) / (weights @ denominator)[:, None]

            def record(point, samples):
                require(bool(np.isfinite(samples).all()), "nonfinite bootstrap draw")
                return {"estimate": float(point),
                        "cluster_bootstrap_95ci": np.percentile(samples, (2.5, 97.5)).tolist(),
                        "valid_bootstrap_replicates": len(samples)}

            metrics[metric] = {"conditions": {c: record(points[i], draws[:, i])
                                               for i, c in enumerate(CONDITIONS)},
                               "paired_differences": {}}
            for a, b in PAIRS:
                ia, ib = CONDITIONS.index(a), CONDITIONS.index(b)
                metrics[metric]["paired_differences"][a + "-" + b] = record(
                    points[ia] - points[ib], draws[:, ia] - draws[:, ib])
            for sid, num, den in zip(sids, sums, denominator):
                per_conversation[sid][metric] = {c: float(num[i] / den) for i, c in enumerate(CONDITIONS)}
        groups[group] = {"question_count": len(selected), "token_count_per_condition": int(tokens.sum()),
                         "excluded_question_count": len(questions) - len(selected),
                         "metrics": metrics, "per_conversation": per_conversation}
    return {"groups": groups, "conversations": sids,
            "bootstrap": {"cluster": "conversation", "cluster_count": len(sids), "paired": True,
                          "replicates": len(sampled), "seed": SEED, "percentile": [2.5, 97.5],
                          "same_draws_across_conditions_and_groups": True,
                          "draws_int64_le_sha256": hashlib.sha256(sampled.astype("<i8").tobytes()).hexdigest()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rows", "summary", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("rows-sha256", "summary-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    require(digest(args.rows) == args.rows_sha256 and digest(args.summary) == args.summary_sha256,
            "sealed NLL input SHA mismatch")
    require(digest(args.protocol) == PROTOCOL_SHA, "pre-registered position protocol changed")
    require(not args.output.exists(), "output must be new")
    protocol = json.loads(args.protocol.read_bytes())
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line]
    original_summary = json.loads(args.summary.read_bytes())
    total_audit = audit(original_summary, rows)
    questions = split_questions(rows)
    result = aggregate(questions)
    for group, expected in protocol["groups"].items():
        observed = result["groups"][group]
        require(observed["question_count"] == expected["questions"]
                and observed["token_count_per_condition"] == expected["tokens_per_condition"],
                "position population differs from pre-registration")
    require(result["groups"]["continuation"]["excluded_question_count"] == 104,
            "single-token answers incorrectly included")
    result.update(protocol="qwen38_locomo_nll_position_analysis_v1", unit="nats/token",
                  preregistration_sha256=PROTOCOL_SHA, rows_sha256=args.rows_sha256,
                  original_summary_sha256=args.summary_sha256, source_sha256=digest(__file__),
                  original_total_nll_audit=total_audit, continuation_empty_imputation=False,
                  additional_GPU_forwards=0, interpretation=protocol["interpretation"])
    args.output.mkdir(parents=True, exist_ok=False)
    per_question_path = args.output / "per_question.jsonl"
    per_question_path.write_text("".join(json.dumps(q, ensure_ascii=False, allow_nan=False) + "\n"
                                         for q in questions), encoding="utf-8")
    result["per_question_sha256"] = digest(per_question_path)
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output),
                      "groups": {k: {f: v[f] for f in ("question_count", "token_count_per_condition", "excluded_question_count")}
                                 for k, v in result["groups"].items()}}, allow_nan=False))


if __name__ == "__main__":
    main()

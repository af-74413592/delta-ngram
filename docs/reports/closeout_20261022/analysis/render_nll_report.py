"""Audit the sealed reference-NLL rows and render paired uncertainty in nats/token."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

CONDITIONS = ("self", "empty", "donor")
PAIRS = (("self", "empty"), ("donor", "empty"), ("self", "donor"))
COLORS = ("#B7602E", "#8B969F", "#7B6BA6")
PARENT = "37826dd71c6fd3798fa5aa4ddf68031571461d6fcb376023ac2e0e3155b575d7"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected):
    if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > 1e-9:
        raise ValueError("NLL summary differs from independent recomputation")


def audit(report, rows):
    import numpy as np

    lookup = {(r["qid"], r["condition"]): r for r in rows}
    qids = sorted({r["qid"] for r in rows})
    if len(rows) != 4620 or len(lookup) != 4620 or len(qids) != 1540:
        raise ValueError("requires the complete 1540 by 3 paired matrix")
    bindings = {r["nll_binding_sha256"] for r in rows}
    if len(bindings) != 1 or any(len(v) != 64 for v in bindings):
        raise ValueError("mixed NLL source bindings")
    questions = []
    for qid in qids:
        triple = [lookup[qid, c] for c in CONDITIONS]
        first = triple[0]
        invariants = ("sample_id", "category", "prompt_sha256", "target_sha256", "reference_sha256",
                      "prompt_tokens", "target_tokens", "nll_binding_sha256")
        if any(any(r[k] != first[k] for k in invariants) for r in triple):
            raise ValueError("paired prompt/reference identities differ")
        if first["category"] not in (1, 2, 3, 4):
            raise ValueError("adversarial questions entered main NLL")
        means, sums = [], []
        for row in triple:
            if (row["parent_binding_sha256"] != PARENT or row["memory_updates"] != 0
                    or row["teacher_forced"] is not True or row["eos_loss_added"] is not False
                    or row["discarded_sample_tokens"] != 1 or row["native_forward_calls"] != 1
                    or row["state_fingerprint_before"] != row["state_fingerprint_after"]):
                raise ValueError("read-only teacher-forcing contract differs")
            count = row["target_tokens"]
            total = row["prompt_tokens"] + count
            logs = row["token_logprobs"]
            if (type(count) is not int or count <= 0 or len(logs) != count
                    or row["prompt_tokens"] <= 0 or total > 640
                    or row["head_rows"] != [1, total]
                    or any(not math.isfinite(v) or v > 1e-6 for v in logs)):
                raise ValueError("invalid token loss, length, or native head rows")
            loss = -math.fsum(logs)
            close(row["nll_sum"], loss)
            close(row["nll_token_mean"], loss / count)
            means.append(loss / count)
            sums.append(loss)
        questions.append((first["sample_id"], first["target_tokens"], means, sums))
    sids = sorted({q[0] for q in questions})
    if len(sids) != 10:
        raise ValueError("expected ten conversations")
    counts = np.array([sum(q[0] == s for q in questions) for s in sids])
    tokens = np.array([sum(q[1] for q in questions if q[0] == s) for s in sids])
    for key, expected in {"question_count": 1540, "evaluation_count": 4620,
                          "target_token_count_per_condition": int(tokens.sum()), "excluded_category5": 446}.items():
        if report[key] != expected:
            raise ValueError("summary population differs")
    for key, expected in {"cluster": "conversation", "cluster_count": 10, "replicates": 10000,
                          "seed": 20261022, "paired": True, "percentile": [2.5, 97.5]}.items():
        if report["bootstrap"][key] != expected:
            raise ValueError("bootstrap protocol differs")
    sampled = np.random.default_rng(20261022).integers(0, 10, (10000, 10))
    weights = np.stack([(sampled == i).sum(axis=1) for i in range(10)], axis=1)
    verified = 0
    for metric, index, denominator in (("query_mean_nll", 2, counts), ("token_pooled_nll", 3, tokens)):
        numerator = np.array([[math.fsum(q[index][i] for q in questions if q[0] == s)
                               for i in range(3)] for s in sids])
        points = numerator.sum(axis=0) / denominator.sum()
        draws = (weights @ numerator) / (weights @ denominator)[:, None]

        def check(record, point, sample):
            close(record["estimate"], point)
            for value, expected in zip(record["cluster_bootstrap_95ci"], np.percentile(sample, (2.5, 97.5))):
                close(value, expected)
            if record["valid_bootstrap_replicates"] != 10000:
                raise ValueError("bootstrap replicate count differs")

        for i, condition in enumerate(CONDITIONS):
            check(report["metrics"][metric]["conditions"][condition], points[i], draws[:, i])
            verified += 1
        for a, b in PAIRS:
            ia, ib = CONDITIONS.index(a), CONDITIONS.index(b)
            check(report["metrics"][metric]["paired_differences"][a + "-" + b],
                  points[ia] - points[ib], draws[:, ia] - draws[:, ib])
            verified += 1
        for sid, count, token_count, den, num in zip(sids, counts, tokens, denominator, numerator):
            record = report["per_conversation"][sid]
            if record["question_count"] != int(count) or record["target_token_count"] != int(token_count):
                raise ValueError("per-conversation count differs")
            for i, condition in enumerate(CONDITIONS):
                close(record[metric][condition], num[i] / den)
    return {"status": "passed", "questions": 1540, "evaluations": 4620, "conversations": 10,
            "target_tokens_per_condition": int(tokens.sum()), "intervals_recomputed": verified,
            "nll_binding_sha256": next(iter(bindings)),
            "scope": "Raw token loss, paired input identity, read-only metadata, population and independent cluster-bootstrap aggregation audit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--rows-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.summary) != args.summary_sha256 or sha(args.rows) != args.rows_sha256:
        raise ValueError("input SHA mismatch")
    if args.output.exists():
        raise ValueError("output must be fresh")
    report = json.loads(args.summary.read_bytes())
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line]
    checked = audit(report, rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    args.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axrow, metric, title in zip(axes, ("query_mean_nll", "token_pooled_nll"),
                                  ("Mean of question NLLs", "NLL over all target tokens")):
        group = report["metrics"][metric]
        for column, (ax, records, labels) in enumerate(zip(axrow,
                (group["conditions"], group["paired_differences"]),
                (CONDITIONS, tuple(a + "-" + b for a, b in PAIRS)))):
            maximum = 0
            for i, (label, color) in enumerate(zip(labels, COLORS)):
                value = records[label]["estimate"]
                low, high = records[label]["cluster_bootstrap_95ci"]
                maximum = max(maximum, abs(low), abs(high), abs(value))
                ax.hlines(i, low, high, color=color, linewidth=2.5)
                ax.plot(value, i, "o", color=color, markersize=6, clip_on=False)
                ax.annotate(f"{value:.5g}", (value, i), xytext=(0, 11), textcoords="offset points",
                            ha="center", color=color, fontsize=9)
            ax.set(yticks=range(3), yticklabels=[x.replace("-", " minus ") for x in labels],
                   ylim=(2.6, -.6), title=title if column == 0 else "Paired differences",
                   xlabel="Nats/token (lower is better)" if column == 0 else "Difference in nats/token (negative is better)")
            ax.grid(axis="x", alpha=.18)
            if column:
                radius = max(maximum * 1.3, 1e-6)
                ax.set_xlim(-radius, radius)
                ax.axvline(0, color="#69747D", linewidth=.8, linestyle="--")
                ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
            else:
                ax.set_xlim(-maximum * .03, maximum * 1.25 if maximum else 1)
    fig.suptitle("LoCoMo: probability of fixed reference answers", fontsize=14)
    fig.text(.5, .022, "1,540 questions, three read-only conditions; category 5 excluded. Bars: paired conversation-cluster 95% CI.\n"
             "10 conversations; 10,000 bootstrap draws. Reference-answer probability does not establish generated-answer accuracy.",
             ha="center", fontsize=8.5, color="#46515B")
    fig.tight_layout(rect=(0, .075, 1, .95), h_pad=2, w_pad=2)
    for ext in ("png", "svg"):
        fig.savefig(args.output / ("reference_nll." + ext), dpi=190, facecolor="white")
    plt.close(fig)
    metadata = {"protocol": "qwen38_locomo_nll_figure_v1", "summary_sha256": args.summary_sha256,
                "rows_sha256": args.rows_sha256, "renderer_sha256": sha(__file__),
                "audit": checked, "plotted_values": report["metrics"]}
    (args.output / "audit_and_plotted_values.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "audit": checked}))


if __name__ == "__main__":
    main()

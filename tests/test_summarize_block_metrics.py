"""Weighted aggregation of public block metrics without model dependencies."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_block_metrics.py"
SPEC = importlib.util.spec_from_file_location("summarize_block_metrics", SCRIPT)
aggregate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate)


def distribution(values, edges=(0., .5, 1.)):
    finite = [value for value in values if value is not None]
    bins = [sum(left <= value < right or (i == len(edges) - 2 and value == right)
                for value in finite) for i, (left, right) in enumerate(zip(edges, edges[1:]))]
    return {"count": len(values), "finite_count": len(finite), "nonfinite_count": len(values) - len(finite),
            "sum": math.fsum(finite), "sumsq": math.fsum(value * value for value in finite),
            "min": min(finite) if finite else None, "max": max(finite) if finite else None,
            "mean": math.fsum(finite) / len(finite) if finite else None,
            "quantiles": {"p50": 999},  # intentionally useless for merging
            "histogram": {"edges": list(edges), "counts": bins,
                          "underflow": sum(value < edges[0] for value in finite),
                          "overflow": sum(value > edges[-1] for value in finite)}}


def write(beta=(.1,), row_l2=(1.,), **tags):
    return {"kind": "write", "cohort": "primary", "condition": "self", "written_blocks": 1,
            "metrics": {"beta": distribution(beta), "patch_delta_row_l2": distribution(row_l2, (0., 1., 2., 10.)),
                        "touched_rows": len(row_l2), "new_rows": 1, "total_written_rows": 4,
                        "patch_delta_frobenius": 3., "patch_total_norm": 5.}, **tags}


def query(heads=((1, 1), (1, 1)), token_hits=1, token_denominator=1, **tags):
    hit_count, denominator = sum(pair[0] for pair in heads), sum(pair[1] for pair in heads)
    return {"kind": "query", "cohort": "primary", "condition": "self", "written_blocks": 1,
            "metrics": {"all": {"hit_count": hit_count, "denominator": denominator, "hit_rate": 999},
                        "per_head": [{"head": i, "hit_count": pair[0], "denominator": pair[1], "hit_rate": 999}
                                     for i, pair in enumerate(heads)],
                        "token_any_hit_count": token_hits, "token_denominator": token_denominator,
                        "token_any_hit_rate": 999, "scored_tokens": token_denominator}, **tags}


def save(tmp_path, records):
    path = tmp_path / "block_metrics.jsonl"
    path.write_text("\n".join(json.dumps(record, allow_nan=False) for record in records) + "\n", encoding="utf-8")
    return path


def test_beta_and_query_are_weighted_by_events_not_blocks(tmp_path):
    first, second = write(), write(beta=(.9,) * 9, row_l2=(2., 2., 2.))
    first["metrics"]["total_written_rows"] = 1
    second["metrics"]["new_rows"] = 2
    path = save(tmp_path, [first, second, query(), query(heads=((0, 8), (0, 10)), token_hits=0,
                                                       token_denominator=10), {"kind": "paired_score"}])
    report = aggregate.summarize(path)
    result, = report["totals"]
    beta = result["write"]["beta"]
    assert beta["count"] == beta["finite_count"] == 10 and beta["nonfinite_count"] == 0
    assert beta["mean"] == pytest.approx(.82)
    assert beta["std"] == pytest.approx(.24)
    assert beta["sum"] == pytest.approx(8.2) and beta["sumsq"] == pytest.approx(7.3)
    assert beta["histogram"]["counts"] == [1, 9]
    assert beta["quantiles"] is None
    scalar = result["write"]["block_scalars"]["touched_rows"]
    assert scalar["median"] == scalar["mean"] == 2 and scalar["p95"] == pytest.approx(2.9)
    assert result["write"]["patch_delta_row_l2"]["mean"] == pytest.approx(1.75)
    queries = result["query"]
    assert queries["all"] == {"hit_count": 2, "denominator": 20, "hit_rate": .1}
    assert queries["token_any"]["hit_rate"] == pytest.approx(1 / 11)
    assert queries["per_head"][0]["hit_rate"] == pytest.approx(1 / 9)
    assert report["source"]["paired_score_records_skipped"] == 1
    assert report["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    json.dumps(report, allow_nan=False)


def test_training_block_numbers_and_explicit_diagnostic_numbers(tmp_path):
    records = []
    for kind, block in (("write", 0), ("query", 1), ("write", 1), ("query", 2)):
        record = write() if kind == "write" else query()
        for field in ("cohort", "condition", "written_blocks"):
            record.pop(field)
        record.update(event="block_metrics", protocol="delta_ngram_block_metrics_v1",
                      block=block, memory_mode="self", phase="train")
        records.append(record)
    records.append(query(condition="context", block=8, written_blocks=0))
    report = aggregate.summarize(save(tmp_path, records))
    train = [item for item in report["groups"] if item["cohort"] == "train"]
    assert [item["written_blocks"] for item in train] == [1, 2]
    assert all(item["write"]["records"] == item["query"]["records"] == 1 for item in train)
    context, = [item for item in report["groups"] if item["condition"] == "context"]
    assert context["written_blocks"] == 0 and context["query"]["records"] == 1
    assert len(report["totals"]) == 2


def test_nonfinite_underflow_overflow_and_empty_denominators_are_explicit(tmp_path):
    record = write(beta=(None, -.1, 1.1), row_l2=(None,))
    record["metrics"]["patch_total_norm"] = None
    report = aggregate.summarize(save(tmp_path, [record, query(heads=((0, 0), (0, 0)),
                                                              token_hits=0, token_denominator=0)]))
    result = report["totals"][0]
    beta = result["write"]["beta"]
    assert beta["count"] == 3 and beta["finite_count"] == 2 and beta["nonfinite_count"] == 1
    assert beta["histogram"]["underflow"] == beta["histogram"]["overflow"] == 1
    assert result["write"]["block_scalars"]["patch_total_norm"]["nonfinite_count"] == 1
    assert result["write"]["patch_delta_row_l2"]["nonfinite_count"] == 1
    assert result["query"]["all"]["hit_rate"] is result["query"]["token_any"]["hit_rate"] is None
    json.dumps(report, allow_nan=False)


def test_inconsistent_histograms_are_rejected_even_across_conditions(tmp_path):
    first, second = write(), write(condition="donor_1")
    second["metrics"]["beta"]["histogram"]["edges"] = [0., .4, 1.]
    with pytest.raises(ValueError, match="inconsistent histogram edges"):
        aggregate.summarize(save(tmp_path, [first, second]))
    second = deepcopy(first)
    second["metrics"]["beta"]["histogram"]["counts"] = [0, 0]
    with pytest.raises(ValueError, match="cover finite_count"):
        aggregate.summarize(save(tmp_path, [second]))


def test_cli_saves_versioned_summary_and_refuses_overwriting_input(tmp_path):
    path = save(tmp_path, [write(), query()])
    output = tmp_path / "summary.json"
    assert aggregate.main(["--input", str(path), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["protocol"] == aggregate.PROTOCOL
    assert result["source"]["summarized_records"] == 2
    with pytest.raises(SystemExit):
        aggregate.main(["--input", str(path), "--output", str(path)])

"""Aggregate training or diagnostic block logs without replaying model work.

Read JSONL incrementally. Only per-block scalar observations and mergeable
distribution statistics are retained, never complete event records. This
summarizes the records present in a file; it does not certify run completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


PROTOCOL = "delta_ngram_block_metrics_summary_v1"
SCALAR_FIELDS = ("touched_rows", "new_rows", "total_written_rows",
                 "patch_delta_frobenius", "patch_total_norm")


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _label(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def _percentile(ordered, probability):
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = int(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def _scalar_summary(values):
    finite = sorted(value for value in values if value is not None)
    return {"count": len(values), "finite_count": len(finite),
            "nonfinite_count": len(values) - len(finite),
            "mean": math.fsum(finite) / len(finite) if finite else None,
            "min": finite[0] if finite else None, "max": finite[-1] if finite else None,
            "median": _percentile(finite, .5), "p95": _percentile(finite, .95)}


class _Distribution:
    def __init__(self):
        self.blocks = self.count = self.finite = self.nonfinite = 0
        self.edges, self.bins = None, []
        self.underflow = self.overflow = 0
        self.sums, self.squares = [], []
        self.invalid_moment_blocks = 0
        self.minimum, self.maximum = None, None

    def add(self, value):
        count = _integer(value["count"], "distribution count")
        finite = _integer(value["finite_count"], "distribution finite_count")
        nonfinite = _integer(value["nonfinite_count"], "distribution nonfinite_count")
        if count != finite + nonfinite:
            raise ValueError("distribution count does not equal finite + nonfinite")
        histogram = value["histogram"]
        edges = tuple(_number(edge, "histogram edge") for edge in histogram["edges"])
        if len(edges) < 2 or any(a >= b for a, b in zip(edges, edges[1:])):
            raise ValueError("histogram edges must be strictly increasing")
        counts = [_integer(number, "histogram count") for number in histogram["counts"]]
        underflow = _integer(histogram["underflow"], "histogram underflow")
        overflow = _integer(histogram["overflow"], "histogram overflow")
        if len(counts) != len(edges) - 1 or sum(counts) + underflow + overflow != finite:
            raise ValueError("histogram bins/underflow/overflow do not cover finite_count")
        if self.edges is not None and edges != self.edges:
            raise ValueError("inconsistent histogram edges; distributions cannot be merged")
        if self.edges is None:
            self.edges, self.bins = edges, [0] * len(counts)
        self.blocks += 1
        self.count += count
        self.finite += finite
        self.nonfinite += nonfinite
        self.bins = [a + b for a, b in zip(self.bins, counts)]
        self.underflow += underflow
        self.overflow += overflow
        if value["sum"] is None or value["sumsq"] is None:
            # A source monitor encodes an overflowing moment as null. Keep
            # counts/histogram valid, but never invent a weighted mean/std.
            self.invalid_moment_blocks += 1
        else:
            total = _number(value["sum"], "distribution sum")
            squares = _number(value["sumsq"], "distribution sumsq")
            if squares < 0 or (not finite and (total != 0 or squares != 0)):
                raise ValueError("invalid distribution moments")
            self.sums.append(total)
            self.squares.append(squares)
        if finite:
            low, high = _number(value["min"], "distribution min"), _number(value["max"], "distribution max")
            if low > high:
                raise ValueError("distribution min exceeds max")
            self.minimum = low if self.minimum is None else min(low, self.minimum)
            self.maximum = high if self.maximum is None else max(high, self.maximum)
        elif value["min"] is not None or value["max"] is not None:
            raise ValueError("empty finite distribution must have null extrema")

    def summary(self):
        total = math.fsum(self.sums) if not self.invalid_moment_blocks else None
        squares = math.fsum(self.squares) if not self.invalid_moment_blocks else None
        mean = total / self.finite if total is not None and self.finite else None
        variance = squares / self.finite - mean * mean if mean is not None else None
        return {"blocks": self.blocks, "count": self.count, "finite_count": self.finite,
                "nonfinite_count": self.nonfinite, "invalid_moment_blocks": self.invalid_moment_blocks,
                "sum": total, "sumsq": squares, "mean": mean,
                "std": math.sqrt(max(0., variance)) if variance is not None else None,
                "std_definition": "population std over finite events; denominator finite_count",
                "min": self.minimum, "max": self.maximum,
                "histogram": {"edges": list(self.edges) if self.edges is not None else [],
                              "counts": self.bins, "underflow": self.underflow, "overflow": self.overflow},
                "quantiles": None, "quantiles_note": "per-block quantiles are not merged or averaged"}


def _counts(value):
    hits, denominator = _integer(value["hit_count"], "hit_count"), _integer(value["denominator"], "denominator")
    if hits > denominator:
        raise ValueError("hit_count exceeds denominator")
    return hits, denominator


def _ratio(hits, denominator):
    return {"hit_count": hits, "denominator": denominator,
            "hit_rate": hits / denominator if denominator else None}


class _Group:
    def __init__(self):
        self.write_records = self.query_records = 0
        self.beta, self.row_l2 = _Distribution(), _Distribution()
        self.scalars = {name: [] for name in SCALAR_FIELDS}
        self.all_hits = self.all_denominator = 0
        self.heads = {}
        self.token_hits = self.token_denominator = self.scored_tokens = 0

    def add(self, kind, metrics):
        if kind == "write":
            self.beta.add(metrics["beta"])
            self.row_l2.add(metrics["patch_delta_row_l2"])
            for name in SCALAR_FIELDS:
                value = metrics[name]
                if name.endswith("rows"):
                    value = _integer(value, name)
                elif value is not None:
                    value = _number(value, name)
                    if value < 0:
                        raise ValueError("patch norm cannot be negative")
                self.scalars[name].append(value)
            if (metrics["new_rows"] > metrics["touched_rows"] or
                    metrics["new_rows"] > metrics["total_written_rows"] or
                    metrics["patch_delta_row_l2"]["count"] != metrics["touched_rows"]):
                raise ValueError("inconsistent write row counts")
            self.write_records += 1
            return
        hits, denominator = _counts(metrics["all"])
        per_head = {}
        for head in metrics["per_head"]:
            index = _integer(head["head"], "head")
            if index in per_head:
                raise ValueError("duplicate query head")
            per_head[index] = _counts(head)
        if not per_head or sorted(per_head) != list(range(len(per_head))):
            raise ValueError("query heads must be contiguous from zero")
        if self.heads and self.heads.keys() != per_head.keys():
            raise ValueError("inconsistent query head layout")
        if (sum(pair[0] for pair in per_head.values()), sum(pair[1] for pair in per_head.values())) != (hits, denominator):
            raise ValueError("per-head query counts disagree with all-head counts")
        token_hits = _integer(metrics["token_any_hit_count"], "token_any_hit_count")
        token_denominator = _integer(metrics["token_denominator"], "token_denominator")
        scored = _integer(metrics["scored_tokens"], "scored_tokens")
        if not token_hits <= token_denominator <= scored or token_hits > hits:
            raise ValueError("inconsistent token-any-hit counts")
        self.all_hits += hits
        self.all_denominator += denominator
        for index, pair in per_head.items():
            old = self.heads.setdefault(index, [0, 0])
            old[0] += pair[0]
            old[1] += pair[1]
        self.token_hits += token_hits
        self.token_denominator += token_denominator
        self.scored_tokens += scored
        self.query_records += 1

    def summary(self):
        return {"write": {"records": self.write_records,
                          "beta": self.beta.summary(), "patch_delta_row_l2": self.row_l2.summary(),
                          "block_scalars": {key: _scalar_summary(values) for key, values in self.scalars.items()}},
                "query": {"records": self.query_records,
                          "all": _ratio(self.all_hits, self.all_denominator),
                          "per_head": [{"head": key, **_ratio(*pair)} for key, pair in sorted(self.heads.items())],
                          "token_any": _ratio(self.token_hits, self.token_denominator),
                          "scored_tokens": self.scored_tokens}}


def summarize(input_path):
    """Aggregate exactly the valid records read, preserving a byte-level hash."""
    path = Path(input_path).resolve()
    digest, size, records, used, skipped = hashlib.sha256(), 0, 0, 0, 0
    groups, totals, fixed_edges, protocols = {}, {}, {}, set()
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            digest.update(raw)
            size += len(raw)
            if not raw.strip():
                continue
            try:
                record = _json(raw.decode("utf-8"))
                if not isinstance(record, dict):
                    raise ValueError("record must be a JSON object")
                records += 1
                kind = record.get("kind")
                if kind == "paired_score":
                    skipped += 1
                    continue
                if kind not in ("query", "write"):
                    raise ValueError("expected query, write, or paired_score kind")
                cohort = _label(record.get("cohort", record.get("phase", "train")), "cohort/phase")
                condition = _label(record.get("condition", record.get("memory_mode")), "condition/memory_mode")
                written = (record["written_blocks"] if "written_blocks" in record else
                           _integer(record.get("block"), "block") + int(kind == "write"))
                written = _integer(written, "written_blocks")
                metrics = record["metrics"]
                if kind == "write":
                    for name in ("beta", "patch_delta_row_l2"):
                        edges = tuple(metrics[name]["histogram"]["edges"])
                        if fixed_edges.setdefault(name, edges) != edges:
                            raise ValueError(f"inconsistent histogram edges for {name}")
                group_key, total_key = (cohort, condition, written), (cohort, condition)
                if group_key not in groups:
                    groups[group_key] = _Group()
                if total_key not in totals:
                    totals[total_key] = _Group()
                groups[group_key].add(kind, metrics)
                totals[total_key].add(kind, metrics)
                protocols.add(_label(record.get("protocol", "unspecified_in_source_record"), "protocol"))
                used += 1
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{path.name}:{line_number}: {error}") from error
    result = {
        "protocol": PROTOCOL, "source": {"path": str(path), "sha256": digest.hexdigest(), "bytes": size,
                                          "nonempty_records": records, "summarized_records": used,
                                          "paired_score_records_skipped": skipped,
                                          "record_protocols": sorted(protocols)},
        "scope": "logged records only; run completion and missing records are not inferred",
        "definitions": {
            "grouping": "cohort (fallback phase/train) x condition (fallback memory_mode) x written_blocks",
            "training_written_blocks": "query: block; write: block+1; explicit diagnostic written_blocks takes precedence",
            "distribution_weighting": "finite event count, not equal weight per block; no averaged block quantiles",
            "scalar_weighting": "one observation per stream/window write block; exact linear-interpolated median and p95",
            "row_count_scope": "touched/new/total_written_rows are per-block state snapshots, not global distinct-row totals",
            "query_weighting": "sum hit counts / sum valid nonzero-address denominators; token-any uses eligible target tokens",
            "nonfinite": "source nonfinite_count retained; null patch norms counted as nonfinite; null moments invalidate merged mean/std",
            "histogram": "fixed source bins; left-closed/right-open except the last bin includes its upper edge",
        },
        "groups": [{"cohort": key[0], "condition": key[1], "written_blocks": key[2], **value.summary()}
                   for key, value in sorted(groups.items())],
        "totals": [{"cohort": key[0], "condition": key[1], **value.summary()}
                   for key, value in sorted(totals.items())],
    }
    json.dumps(result, allow_nan=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    temporary = output.with_suffix(output.suffix + ".tmp")
    if Path(args.input).resolve() in (output, temporary.resolve()):
        parser.error("--output and its temporary file must differ from --input")
    result = summarize(args.input)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "source_sha256": result["source"]["sha256"],
                      "records": result["source"]["summarized_records"], "groups": len(result["groups"])}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

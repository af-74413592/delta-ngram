"""Strict post-hoc accuracy scoring and aggregation for Qwen3.8 runs.

The Qwen3.8 campaign currently records teacher-forced NLL and hidden-state
captures.  This module deliberately keeps *generation* separate from scoring:

* ``score-math`` delegates to the locked MATH-500 scorer in ``math_probe``;
* ``score-code`` delegates to the locked HumanEval WASI scorer;
* ``aggregate`` joins already completed scoring summaries across checkpoints.

No model is loaded and no training is performed by this script.  A generation
directory is accepted only when its summary and completion trace checksums
match.  Therefore a NLL summary can never silently become an accuracy score.
Generation for Qwen3.8 still needs an MCore generation runner (the existing
MCore retention runner is forward-only); this file is the stable scoring and
reporting interface that runner should write to.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def fresh_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"use a fresh output directory: {path}")
    return path


def _check_generation(samples_dir: Path, filename: str, *, expected_version: int | None = None) -> dict:
    summary_path = samples_dir / "summary.json"
    samples_path = samples_dir / filename
    if not summary_path.is_file() or not samples_path.is_file():
        raise ValueError(f"missing generation summary or {filename}: {samples_dir}")
    summary = read_json(summary_path)
    if summary.get("status") != "completed":
        raise ValueError(f"generation is not completed: {samples_dir}")
    checksum_key = "completions_sha256"
    declared = summary.get(checksum_key)
    if not isinstance(declared, str) or sha256(samples_path) != declared:
        raise ValueError(f"{filename} changed after generation or checksum is missing")
    protocol = summary.get("protocol") or {}
    if expected_version is not None and protocol.get("version") != expected_version:
        raise ValueError(f"unsupported generation protocol version: {protocol.get('version')}")
    if summary.get("task_count") != len(summary.get("task_ids", [])):
        raise ValueError("generation task count and task IDs disagree")
    return summary


def score_math(args: argparse.Namespace) -> dict:
    import math_probe

    samples_dir = Path(args.samples_dir)
    generation = _check_generation(samples_dir, "math_completions.json")
    output = fresh_directory(args.output)
    result = math_probe.score(SimpleNamespace(
        data_dir=Path(args.data_dir), samples_dir=samples_dir, output=output,
        scope=args.scope, reference=Path(args.reference) if args.reference else None))
    # Add an explicit Qwen campaign marker without altering the locked scorer's
    # protocol fields or checksums.
    result["evaluator"] = {"name": "qwen38_accuracy_eval", "domain": "math",
                           "script_sha256": sha256(__file__),
                           "source_generation_summary_sha256": sha256(samples_dir / "summary.json"),
                           "source_checkpoint": generation.get("source_checkpoint"),
                           "scoring_only": True}
    write_json(output / "summary.json", result)
    return result


def score_code(args: argparse.Namespace) -> dict:
    import evaluate_humaneval_wasi

    samples_dir = Path(args.samples_dir)
    generation = _check_generation(samples_dir, "humaneval_completions.json", expected_version=2)
    output = fresh_directory(args.output)
    result = evaluate_humaneval_wasi.score(SimpleNamespace(
        data_dir=Path(args.data_dir), samples_dir=samples_dir, output=output,
        scope=args.scope, reference=Path(args.reference) if args.reference else None,
        runtime=Path(args.runtime), validation=Path(args.validation)))
    result["evaluator"] = {"name": "qwen38_accuracy_eval", "domain": "code",
                           "script_sha256": sha256(__file__),
                           "source_generation_summary_sha256": sha256(samples_dir / "summary.json"),
                           "source_checkpoint": generation.get("source_checkpoint"),
                           "scoring_only": True}
    write_json(output / "summary.json", result)
    return result


def _metric(summary: dict, domain: str) -> tuple[int | None, int | None, float | None]:
    if domain == "math":
        metrics = summary.get("metrics") or {}
        return metrics.get("correct_count"), metrics.get("n"), metrics.get("accuracy")
    metrics = summary
    return metrics.get("passed"), metrics.get("n"), metrics.get("pass_at_1")


def _row_from_entry(entry: dict) -> dict:
    if not isinstance(entry, dict) or not entry.get("label"):
        raise ValueError("each manifest entry needs a nonempty label")
    row = {"label": str(entry["label"]), "arm": entry.get("arm"),
           "stage": entry.get("stage"), "checkpoint": entry.get("checkpoint")}
    for domain, field in (("math", "math_summary"), ("code", "code_summary")):
        if not entry.get(field):
            raise ValueError(f"{row['label']}: missing {field}")
        summary = read_json(entry[field])
        if domain == "math":
            if summary.get("status") not in ("completed", "completed_with_unresolved"):
                raise ValueError(f"{row['label']}: incomplete math score")
        elif summary.get("execution_status") != "completed":
            raise ValueError(f"{row['label']}: incomplete code score")
        count, total, value = _metric(summary, domain)
        if not isinstance(total, int) or total <= 0 or not isinstance(value, (int, float)):
            raise ValueError(f"{row['label']}: invalid {domain} score")
        row[f"{domain}_correct"] = count
        row[f"{domain}_n"] = total
        row[f"{domain}_accuracy"] = float(value)
        row[f"{domain}_summary"] = str(Path(entry[field]).resolve())
        transitions = summary.get("transitions") or {}
        row[f"{domain}_lost"] = transitions.get("lost")
        row[f"{domain}_gained"] = transitions.get("gained")
    # If the caller supplies a checkpoint path, bind both domains to it when
    # available. This catches accidentally mixing stage-1 math and stage-7 code.
    source_checkpoints = {read_json(entry[field]).get("source_checkpoint")
                          for field in ("math_summary", "code_summary")}
    source_checkpoints.discard(None)
    if len(source_checkpoints) > 1:
        raise ValueError(f"{row['label']}: math/code source checkpoints differ")
    row["source_checkpoint"] = next(iter(source_checkpoints), row["checkpoint"])
    return row


def aggregate(args: argparse.Namespace) -> dict:
    manifest = read_json(args.manifest)
    entries = manifest.get("entries") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must contain a nonempty entries list")
    rows = [_row_from_entry(entry) for entry in entries]
    labels = [row["label"] for row in rows]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate labels in accuracy manifest")
    output = fresh_directory(args.output)
    payload = {"protocol": "qwen38_exact_accuracy_aggregate_v1",
               "generated_utc": datetime.now(timezone.utc).isoformat(),
               "script_sha256": sha256(__file__),
               "manifest": str(Path(args.manifest).resolve()), "rows": rows,
               "note": "Greedy generation accuracy only; NLL and representation drift are separate metrics."}
    write_json(output / "accuracy_aggregate.json", payload)
    fields = ["label", "arm", "stage", "source_checkpoint", "math_correct", "math_n", "math_accuracy",
              "math_lost", "math_gained", "code_correct", "code_n", "code_accuracy", "code_lost", "code_gained"]
    with (output / "accuracy_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Qwen3.8 生成准确率汇总", "",
             "本表只接收已完成的贪心生成评分；NLL、表示漂移和准确率分开记录。", "",
             "|标签|方法|阶段|数学|代码|数学丢失/新增|代码丢失/新增|", "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        def pct(name):
            value = row[name]
            return "—" if value is None else f"{value:.2%} ({row[name.replace('accuracy', 'correct')]}/{row[name.replace('accuracy', 'n')]})"
        def trans(domain):
            lost, gained = row[f"{domain}_lost"], row[f"{domain}_gained"]
            return "—" if lost is None or gained is None else f"{lost}/{gained}"
        lines.append(f"|{row['label']}|{row.get('arm') or '—'}|{row.get('stage') or '—'}|{pct('math_accuracy')}|{pct('code_accuracy')}|{trans('math')}|{trans('code')}|")
    (output / "accuracy_aggregate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    math = actions.add_parser("score-math", help="score an existing math_probe generation")
    math.add_argument("--data-dir", type=Path, required=True); math.add_argument("--samples-dir", type=Path, required=True)
    math.add_argument("--output", type=Path, required=True); math.add_argument("--scope", choices=("probe", "full"))
    math.add_argument("--reference", type=Path)
    code = actions.add_parser("score-code", help="score an existing humaneval_probe generation")
    for name in ("data-dir", "samples-dir", "output", "runtime", "validation"):
        code.add_argument("--" + name, type=Path, required=True)
    code.add_argument("--scope", choices=("probe", "full"), default="probe"); code.add_argument("--reference", type=Path)
    agg = actions.add_parser("aggregate", help="join completed math/code summaries")
    agg.add_argument("--manifest", type=Path, required=True); agg.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "score-math": result = score_math(args)
    elif args.action == "score-code": result = score_code(args)
    else: result = aggregate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

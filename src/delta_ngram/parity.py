"""Exact artifact parity against the archived two-fact, seed-17 pilot.

No numeric tolerance is used. Runtime measurements, paths, checkpoint envelope
metadata, and candidate-only diagnostic fields are explicitly out of scope.
The command fails unless both runs are complete; its default contract is 500
training steps and seed 17. Small CPU fixtures can override the expected steps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import torch
from torch.torch_version import TorchVersion


TRAIN_FIELDS = ("step", "episodes", "answer_nll", "write1_q1_nll", "write2_q2_nll",
                "write2_q1_nll", "grad_norm_before_clip")
EVAL_EXCLUSIONS = frozenset({
    "evaluation_seconds", "write_seconds_total", "query_seconds_total",
    "write_seconds_mean", "query_seconds_mean", "elapsed_seconds", "step_seconds",
    "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "peak_memory_bytes",
    "protocol", "interface", "scope", "config", "checkpoint", "output_dir", "out",
})
BASE_QUERY_KEYS = frozenset({"empty_q1", "empty_q2", "write1_q1", "write2_q1", "write2_q2",
                             "zero_after_write_q1", "wrong_values_q1", "wrong_values_q2"})
RETENTION_SCORE_KEYS = frozenset({"conditional_retention", "conditional_retention_denominator",
                                  "acquired_and_retained_all_samples",
                                  "wrong_values_q1_follow_corrupted_evidence",
                                  "wrong_values_q2_follow_corrupted_evidence"})
RECORD_FIELDS = frozenset({"episode_id", "names", "colors", "predictions", "correct",
                           "strict_first_line_correct", "wrong_evidence_colors",
                           "wrong_values_follow_corrupted_evidence", "state_rows", "state_tensor_bytes"})
_MISSING = object()


def _brief(value: Any) -> Any:
    if value is _MISSING:
        return "<missing>"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= 500 else text[:497] + "..."


class _Comparison:
    def __init__(self, limit: int):
        self.limit = limit
        self.mismatches: list[dict[str, Any]] = []
        self.mismatch_count = 0
        self.compared_values = 0
        self.files: dict[str, dict[str, dict[str, Any]]] = {"reference": {}, "candidate": {}}
        self.ignored_candidate_fields: list[str] = []
        self.ignored_candidate_field_count = 0

    def fail(self, path: str, reason: str, reference: Any = _MISSING,
             candidate: Any = _MISSING) -> None:
        self.mismatch_count += 1
        if len(self.mismatches) < self.limit:
            self.mismatches.append({"path": path, "reason": reason,
                                    "reference": _brief(reference), "candidate": _brief(candidate)})

    def require(self, actual: Any, expected: Any, path: str, reason: str) -> None:
        if actual != expected or isinstance(actual, bool) != isinstance(expected, bool):
            self.fail(path, reason, expected, actual)

    def compare(self, reference: Any, candidate: Any, path: str, *,
                additions: bool = True, exclude: frozenset[str] = frozenset()) -> None:
        if isinstance(reference, dict):
            if not isinstance(candidate, dict):
                self.fail(path, "expected an object", reference, candidate)
                return
            for key in sorted(reference):
                if key in exclude:
                    continue
                child_path = f"{path}.{key}"
                if key not in candidate:
                    self.fail(child_path, "missing reference field", reference[key])
                else:
                    self.compare(reference[key], candidate[key], child_path, additions=additions)
            for key in sorted(candidate.keys() - reference.keys()):
                if additions:
                    self.ignored_candidate_field_count += 1
                    if len(self.ignored_candidate_fields) < 100:
                        self.ignored_candidate_fields.append(f"{path}.{key}")
                else:
                    self.fail(f"{path}.{key}", "unexpected field", candidate=candidate[key])
        elif isinstance(reference, list):
            if not isinstance(candidate, list):
                self.fail(path, "expected a list", reference, candidate)
                return
            self.require(len(candidate), len(reference), f"{path}.length", "list lengths differ")
            for index, (left, right) in enumerate(zip(reference, candidate)):
                self.compare(left, right, f"{path}[{index}]", additions=additions)
        else:
            self.compared_values += 1
            if reference != candidate or isinstance(reference, bool) != isinstance(candidate, bool):
                self.fail(path, "values differ (no numeric tolerance)", reference, candidate)

    def fingerprint(self, path: Path, side: str) -> bool:
        if not path.is_file():
            self.fail(f"{side}/{path.name}", "required artifact is missing")
            return False
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            self.files[side][path.name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
        except OSError as error:
            self.fail(f"{side}/{path.name}", f"cannot read artifact: {error}")
            return False
        return True

    def json_file(self, root: Path, name: str, side: str) -> Any:
        path = root / name
        if not self.fingerprint(path, side):
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
        except (OSError, ValueError) as error:
            self.fail(f"{side}/{name}", f"invalid JSON: {error}")
            return None


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _config_value(config: Mapping[str, Any], key: str) -> Any:
    if key in config:
        return config[key]
    for group in ("train", "training", "data"):
        section = config.get(group, {})
        if isinstance(section, dict) and key in section:
            return section[key]
    package = config.get("package_config")
    return _config_value(package, key) if isinstance(package, dict) else None


def _mapping(value: Any, path: str, comparison: _Comparison) -> dict[str, Any]:
    if not isinstance(value, dict):
        comparison.fail(path, "expected a JSON object", candidate=value)
        return {}
    return value


def _require_fields(value: Mapping[str, Any], keys, path: str, comparison: _Comparison) -> None:
    for key in sorted(keys):
        if key not in value:
            comparison.fail(f"{path}.{key}", "required semantic field is missing")


def _training_records(root: Path, side: str, config: dict, expected_steps: int,
                      comparison: _Comparison) -> list[dict[str, Any]]:
    path = root / "metrics.jsonl"
    if not comparison.fingerprint(path, side):
        return []
    records = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line, parse_constant=_reject_nonfinite)
                except ValueError as error:
                    comparison.fail(f"{side}/metrics.jsonl:{line_number}", f"invalid JSON: {error}")
                    continue
                if not isinstance(record, dict):
                    comparison.fail(f"{side}/metrics.jsonl:{line_number}", "event must be an object")
                elif record.get("event") == "train":
                    records.append(record)
    except OSError as error:
        comparison.fail(f"{side}/metrics.jsonl", f"cannot read training log: {error}")
    actual_steps = [record.get("step") for record in records]
    comparison.require(actual_steps, list(range(1, expected_steps + 1)), f"{side}/metrics.jsonl.steps",
                       "training steps must be complete, ordered, and unique")
    batch_size = _config_value(config, "batch_episodes")
    for index, record in enumerate(records):
        path = f"{side}/metrics.jsonl.train[{index}]"
        _require_fields(record, TRAIN_FIELDS, path, comparison)
        comparison.require(record.get("episodes"), batch_size, f"{path}.episodes", "incomplete training batch")
    return records


def _checkpoint(root: Path, name: str, side: str, expected_step: int,
                comparison: _Comparison) -> dict[str, torch.Tensor] | None:
    path = root / name
    if not comparison.fingerprint(path, side):
        return None
    try:
        # The old config stores torch.__version__ as this harmless str subtype.
        # Never fall back to unpickling arbitrary checkpoint globals.
        with torch.serialization.safe_globals([TorchVersion]):
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        comparison.fail(f"{side}/{name}", f"safe checkpoint load failed: {type(error).__name__}: {error}")
        return None
    if not isinstance(checkpoint, dict):
        comparison.fail(f"{side}/{name}", "checkpoint envelope must be a mapping")
        return None
    comparison.require(checkpoint.get("step"), expected_step, f"{side}/{name}.step", "checkpoint step differs")
    state = checkpoint.get("writer_state_dict")
    if not isinstance(state, dict) or not state:
        comparison.fail(f"{side}/{name}.writer_state_dict", "nonempty writer tensor mapping is required")
        return None
    digest = hashlib.sha256()
    valid = True
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.Tensor):
            comparison.fail(f"{side}/{name}.writer_state_dict.{key}", "writer state contains a non-tensor")
            valid = False
            continue
        if value.layout != torch.strided:
            comparison.fail(f"{side}/{name}.writer_state_dict.{key}", "writer tensor must have strided layout")
            valid = False
            continue
        if not bool(torch.isfinite(value).all()):
            comparison.fail(f"{side}/{name}.writer_state_dict.{key}", "writer tensor contains non-finite values")
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    comparison.files[side][name].update({"step": checkpoint.get("step"), "tensor_count": len(state),
                                         "tensor_payload_sha256": digest.hexdigest() if valid else None})
    return state if valid else None


def _compare_checkpoints(reference: Path, candidate: Path, name: str, expected_step: int,
                         comparison: _Comparison) -> int:
    left = _checkpoint(reference, name, "reference", expected_step, comparison)
    right = _checkpoint(candidate, name, "candidate", expected_step, comparison)
    if left is None or right is None:
        return 0
    comparison.require(sorted(right), sorted(left), f"{name}.writer_state_dict.keys", "tensor keys differ")
    compared = 0
    for key in sorted(left.keys() & right.keys()):
        before, after = left[key], right[key]
        compared += 1
        if before.dtype != after.dtype or before.shape != after.shape or not torch.equal(before, after):
            comparison.fail(f"{name}.writer_state_dict.{key}", "tensor dtype/shape/torch.equal differs",
                            {"dtype": str(before.dtype), "shape": list(before.shape)},
                            {"dtype": str(after.dtype), "shape": list(after.shape)})
    return compared


def _evaluation_files(root: Path, side: str, expected_schedule: list[int],
                      comparison: _Comparison) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in sorted(root.glob("eval_step_*.json")):
        match = re.fullmatch(r"eval_step_(\d+)\.json", path.name)
        if not match:
            comparison.fail(f"{side}/{path.name}", "invalid evaluation filename")
            continue
        step = int(match.group(1))
        if step in found:
            comparison.fail(f"{side}/{path.name}", "duplicate evaluation step")
        found[step] = path
    comparison.require(sorted(found), expected_schedule, f"{side}/evaluation_steps",
                       "all scheduled evaluations, including initial and final, are required")
    return found


def _check_evaluation(value: dict, side: str, filename: str, step: int, manifest: dict,
                      config: dict, comparison: _Comparison) -> None:
    path = f"{side}/{filename}"
    expected_records = manifest.get("validation_episodes", [])
    count = len(expected_records) if isinstance(expected_records, list) else 0
    _require_fields(value, {"event", "step", "episodes", "requested_episodes", "complete", "scores",
                            "strict_first_line_exact", "records", "write_count", "query_count",
                            "state_tensor_bytes_mean"}, path, comparison)
    for key, expected in (("event", "evaluation"), ("step", step), ("episodes", count),
                          ("requested_episodes", count), ("complete", True)):
        comparison.require(value.get(key), expected, f"{path}.{key}", "evaluation is incomplete or misidentified")
    scores = _mapping(value.get("scores"), f"{path}.scores", comparison)
    strict = _mapping(value.get("strict_first_line_exact"), f"{path}.strict_first_line_exact", comparison)
    _require_fields(scores, BASE_QUERY_KEYS | RETENTION_SCORE_KEYS, f"{path}.scores", comparison)
    _require_fields(strict, BASE_QUERY_KEYS, f"{path}.strict_first_line_exact", comparison)
    records = value.get("records")
    if not isinstance(records, list):
        comparison.fail(f"{path}.records", "complete record list is required")
        return
    comparison.require(len(records), count, f"{path}.records.length", "evaluation records are incomplete")
    openbook_count = _config_value(config, "openbook_episodes")
    if openbook_count is None:
        section = config.get("evaluation", {})
        openbook_count = section.get("openbook_episodes", 0) if isinstance(section, dict) else 0
    for index, item in enumerate(records):
        record_path = f"{path}.records[{index}]"
        record = _mapping(item, record_path, comparison)
        _require_fields(record, RECORD_FIELDS, record_path, comparison)
        if index < len(expected_records):
            expected = expected_records[index]
            metadata = {key: record[key] for key in expected if key in record}
            comparison.compare(expected, metadata, record_path, additions=False)
        keys = BASE_QUERY_KEYS | ({"openbook_q1", "openbook_q2"} if index < openbook_count else set())
        for field in ("predictions", "correct", "strict_first_line_correct"):
            mapping = _mapping(record.get(field), f"{record_path}.{field}", comparison)
            _require_fields(mapping, keys, f"{record_path}.{field}", comparison)
        wrong = _mapping(record.get("wrong_values_follow_corrupted_evidence"),
                         f"{record_path}.wrong_values_follow_corrupted_evidence", comparison)
        _require_fields(wrong, {"q1", "q2"}, f"{record_path}.wrong_values_follow_corrupted_evidence", comparison)


def compare_runs(reference: str | Path, candidate: str | Path, *, expected_steps: int = 500,
                 expected_seed: int = 17, max_mismatches: int = 50) -> dict[str, Any]:
    """Compare all required artifacts and return a machine-readable audit report."""
    if expected_steps < 0 or max_mismatches < 1:
        raise ValueError("expected_steps must be nonnegative and max_mismatches positive")
    reference, candidate = Path(reference).resolve(), Path(candidate).resolve()
    comparison = _Comparison(max_mismatches)
    configs, manifests, summaries, train = {}, {}, {}, {}
    for side, root in (("reference", reference), ("candidate", candidate)):
        configs[side] = _mapping(comparison.json_file(root, "config.json", side), f"{side}/config.json", comparison)
        manifests[side] = _mapping(comparison.json_file(root, "data_manifest.json", side), f"{side}/data_manifest.json", comparison)
        summaries[side] = _mapping(comparison.json_file(root, "summary.json", side), f"{side}/summary.json", comparison)
        config, manifest, summary = configs[side], manifests[side], summaries[side]
        comparison.require(_config_value(config, "seed"), expected_seed, f"{side}/config.seed", "unexpected seed")
        comparison.require(_config_value(config, "steps"), expected_steps, f"{side}/config.steps", "unexpected requested step count")
        episodes = manifest.get("validation_episodes")
        if not isinstance(episodes, list) or not episodes:
            comparison.fail(f"{side}/data_manifest.validation_episodes", "nonempty validation manifest is required")
        else:
            comparison.require(_config_value(config, "eval_episodes"), len(episodes),
                               f"{side}/config.eval_episodes", "config and manifest episode counts differ")
        for key, expected in (("completed_steps", expected_steps), ("requested_steps", expected_steps),
                              ("evaluation_matches_checkpoint", True)):
            comparison.require(summary.get(key), expected, f"{side}/summary.{key}", "run did not complete the required contract")
        last_evaluation = _mapping(summary.get("last_evaluation"), f"{side}/summary.last_evaluation", comparison)
        comparison.require(last_evaluation.get("step"), expected_steps, f"{side}/summary.last_evaluation.step",
                           "summary does not identify the final evaluation")
        comparison.require(last_evaluation.get("complete"), True, f"{side}/summary.last_evaluation.complete",
                           "summary reports an incomplete final evaluation")
        train[side] = _training_records(root, side, config, expected_steps, comparison)
    comparison.compare(manifests["reference"], manifests["candidate"], "data_manifest.json", additions=False)
    for key in ("seed", "steps", "eval_every", "eval_episodes", "batch_episodes"):
        comparison.compare(_config_value(configs["reference"], key), _config_value(configs["candidate"], key),
                           f"config.{key}")
    for index, (left, right) in enumerate(zip(train["reference"], train["candidate"])):
        for field in TRAIN_FIELDS:
            if field in left and field in right:
                comparison.compare(left[field], right[field], f"metrics.train[{index}].{field}")
    tensors = sum(_compare_checkpoints(reference, candidate, name, step, comparison)
                  for name, step in (("writer_init.pt", 0), ("writer.pt", expected_steps)))
    interval = _config_value(configs["reference"], "eval_every")
    if not isinstance(interval, int) or isinstance(interval, bool) or interval < 1:
        comparison.fail("reference/config.eval_every", "positive integer evaluation interval is required", candidate=interval)
        schedule = sorted({0, expected_steps})
    else:
        schedule = sorted({0, expected_steps, *range(interval, expected_steps + 1, interval)})
    evaluations = {side: _evaluation_files(root, side, schedule, comparison)
                   for side, root in (("reference", reference), ("candidate", candidate))}
    compared_evaluations = 0
    for step in sorted(evaluations["reference"].keys() | evaluations["candidate"].keys()):
        values = {}
        for side, root in (("reference", reference), ("candidate", candidate)):
            path = evaluations[side].get(step)
            if path is not None:
                value = _mapping(comparison.json_file(root, path.name, side), f"{side}/{path.name}", comparison)
                _check_evaluation(value, side, path.name, step, manifests[side], configs[side], comparison)
                values[side] = value
        if len(values) == 2:
            comparison.compare(values["reference"], values["candidate"], f"eval_step_{step:06d}.json",
                               exclude=EVAL_EXCLUSIONS)
            compared_evaluations += 1
    manifest_hashes = {side: hashlib.sha256(json.dumps(manifests[side], sort_keys=True).encode()).hexdigest()
                       for side in ("reference", "candidate")}
    comparison.compare(manifest_hashes["reference"], manifest_hashes["candidate"],
                       "data_manifest.json.semantic_sha256")
    for side in ("reference", "candidate"):
        recorded_hash = configs[side].get("manifest_sha256")
        if recorded_hash is not None:
            comparison.require(recorded_hash, manifest_hashes[side], f"{side}/config.manifest_sha256",
                               "recorded manifest hash does not match its artifact")
    return {
        "report_version": 1,
        "passed": comparison.mismatch_count == 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {"expected_seed": expected_seed, "expected_steps": expected_steps,
                     "numeric_tolerance": 0, "tensor_comparison": "dtype + shape + torch.equal",
                     "expected_evaluation_steps": schedule},
        "reference": str(reference), "candidate": str(candidate),
        "checked": {"training_records": {side: len(records) for side, records in train.items()},
                    "evaluations": compared_evaluations, "writer_tensors": tensors,
                    "scalar_values": comparison.compared_values},
        "manifest_semantic_sha256": manifest_hashes,
        "artifacts": comparison.files,
        "provenance": {"python_version": sys.version, "comparator_torch_version": str(torch.__version__),
                       "comparator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "runs": {side: {key: config.get(key) for key in
                                       ("protocol", "torch_version", "resolved_model_commit", "model_dtype")}
                                for side, config in configs.items()}},
        "exclusions": {
            "evaluation_root_fields": sorted(EVAL_EXCLUSIONS),
            "training_log": "Only TRAIN_FIELDS are compared; timings, runtime/memory peaks, and non-train events are excluded.",
            "training_compared_fields": list(TRAIN_FIELDS),
            "checkpoints": "Only step and every writer_state_dict key/tensor are semantic; envelope format, protocol, config, and optimizer metadata are excluded.",
            "config": "Only seed, steps, eval_every, eval_episodes, batch_episodes and completeness requirements are compared; other config/path/runtime metadata is provenance only.",
            "summary": "Only completed/requested steps, evaluation_matches_checkpoint, and final evaluation step/completeness are required.",
            "candidate_additions": "Extra candidate dictionary fields are allowed in evaluation artifacts, including A0/A1/AT and coverage diagnostics; manifest and tensor keys must match exactly.",
        },
        "ignored_candidate_field_count": comparison.ignored_candidate_field_count,
        "ignored_candidate_fields_first_100": comparison.ignored_candidate_fields,
        "mismatch_count": comparison.mismatch_count,
        "mismatches": comparison.mismatches,
        "mismatches_truncated": comparison.mismatch_count > len(comparison.mismatches),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=500, help="default: full 500-step parity; override for fixtures")
    parser.add_argument("--expected-seed", type=int, default=17)
    parser.add_argument("--max-mismatches", type=int, default=50)
    args = parser.parse_args(argv)
    report = compare_runs(args.reference, args.candidate, expected_steps=args.expected_steps,
                          expected_seed=args.expected_seed, max_mismatches=args.max_mismatches)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "mismatch_count": report["mismatch_count"],
                      "report": str(args.out.resolve())}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

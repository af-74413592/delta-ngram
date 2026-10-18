"""Small CPU-only fixtures for exact historical artifact comparison."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest
import torch
from torch.torch_version import TorchVersion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delta_ngram.data import legacy_fact_manifest, validation_episodes
from delta_ngram.parity import BASE_QUERY_KEYS, compare_runs, main


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture(root: Path, *, steps: int = 2):
    root.mkdir()
    manifest = legacy_fact_manifest(17, 2)
    config = {"seed": 17, "steps": steps, "eval_every": 1, "eval_episodes": 2,
              "batch_episodes": 4, "openbook_episodes": 1, "out": str(root),
              "torch_version": str(torch.__version__), "protocol": "old_pilot"}
    _write_json(root / "config.json", config)
    _write_json(root / "data_manifest.json", manifest)
    events = [{"event": "start", "config": config}]
    for step in range(1, steps + 1):
        events.append({"event": "train", "step": step, "episodes": 4,
                       "answer_nll": step / 10, "write1_q1_nll": step / 8,
                       "write2_q2_nll": step / 9, "write2_q1_nll": step / 7,
                       "grad_norm_before_clip": 0.005, "step_seconds": 1.0,
                       "cuda_peak_allocated_bytes": 42})
    for step in range(steps + 1):
        records = []
        for index, episode in enumerate(validation_episodes(manifest)):
            keys = BASE_QUERY_KEYS | ({"openbook_q1", "openbook_q2"} if index == 0 else set())
            predictions = {key: " " + episode.colors[0] + "\n" for key in sorted(keys)}
            records.append({**json.loads(json.dumps(manifest["validation_episodes"][index])),
                            "predictions": predictions,
                            "correct": {key: False for key in predictions},
                            "strict_first_line_correct": {key: False for key in predictions},
                            "wrong_evidence_colors": [episode.evidence_color(i, wrong=True) for i in (0, 1)],
                            "wrong_values_follow_corrupted_evidence": {"q1": False, "q2": False},
                            "state_rows": 10 + step, "state_tensor_bytes": 280})
        keys = sorted({key for record in records for key in record["correct"]})
        scores = {key: 0.0 for key in keys}
        scores.update({"conditional_retention": None, "conditional_retention_denominator": 0,
                       "acquired_and_retained_all_samples": 0.0,
                       "wrong_values_q1_follow_corrupted_evidence": 0.0,
                       "wrong_values_q2_follow_corrupted_evidence": 0.0})
        result = {"event": "evaluation", "step": step, "episodes": 2, "requested_episodes": 2,
                  "complete": True, "scores": scores,
                  "strict_first_line_exact": {key: 0.0 for key in keys},
                  "evaluation_seconds": 3.0, "write_seconds_total": 1.0, "query_seconds_total": 2.0,
                  "write_seconds_mean": 0.125, "query_seconds_mean": 2.0 / 18,
                  "write_count": 8, "query_count": 18, "state_tensor_bytes_mean": 280,
                  "scoring": "greedy free generation", "wrong_state_protocol": "original addresses",
                  "openbook_scope": "capability reference", "records": records}
        _write_json(root / f"eval_step_{step:06d}.json", result)
        events.append({key: value for key, value in result.items() if key != "records"})
    summary = {"completed_steps": steps, "requested_steps": steps, "wall_limit_reached": False,
               "evaluation_matches_checkpoint": True, "last_evaluation": events[-1],
               "elapsed_seconds": 10.0, "checkpoint": "writer.pt"}
    _write_json(root / "summary.json", summary)
    events.append({"event": "complete", **summary})
    (root / "metrics.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    for filename, step in (("writer_init.pt", 0), ("writer.pt", steps)):
        torch.save({"step": step, "protocol": "old_pilot",
                    "config": {**config, "torch_version": TorchVersion(torch.__version__)},
                    "writer_state_dict": {"linear.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) + step,
                                          "linear.bias": torch.tensor([0.5, 1.0])}}, root / filename)


@pytest.fixture
def runs(tmp_path):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    _fixture(reference)
    shutil.copytree(reference, candidate)
    return reference, candidate


def _compare(runs, **kwargs):
    return compare_runs(*runs, expected_steps=2, **kwargs)


def _checkpoint_load(path):
    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def test_exact_fixture_passes_with_full_hashes_and_safe_torchversion_load(runs):
    report = _compare(runs)
    assert report["passed"], report["mismatches"]
    assert report["checked"]["training_records"] == {"reference": 2, "candidate": 2}
    assert report["checked"]["evaluations"] == 3
    assert report["checked"]["writer_tensors"] == 4
    assert report["contract"]["numeric_tolerance"] == 0
    assert report["manifest_semantic_sha256"]["reference"] == report["manifest_semantic_sha256"]["candidate"]
    assert report["artifacts"]["reference"]["writer_init.pt"]["tensor_payload_sha256"]
    assert set(report["artifacts"]["reference"]) == {
        "config.json", "data_manifest.json", "summary.json", "metrics.jsonl", "writer_init.pt", "writer.pt",
        "eval_step_000000.json", "eval_step_000001.json", "eval_step_000002.json"}


def test_candidate_additions_metadata_and_timings_are_explicitly_excluded(runs):
    _, candidate = runs
    config = _read_json(candidate / "config.json")
    config.update({"out": "different/path", "protocol": "delta_ngram_v1", "torch_version": "new-version"})
    _write_json(candidate / "config.json", config)
    for path in candidate.glob("eval_step_*.json"):
        value = _read_json(path)
        value.update({"evaluation_seconds": 999.0, "protocol": "delta_ngram_v1", "A0": 1.0})
        value["scores"].update({"A0": 0.0, "A1": 0.0, "AT": 0.0})
        value["records"][0]["coverage"] = {"written_row_fraction": 0.2}
        _write_json(path, value)
    checkpoint = _checkpoint_load(candidate / "writer.pt")
    checkpoint.update({"format_version": 1, "protocol": "delta_ngram_v1", "config": config})
    torch.save(checkpoint, candidate / "writer.pt")
    events = [json.loads(line) for line in (candidate / "metrics.jsonl").read_text().splitlines()]
    for event in events:
        if event["event"] == "train":
            event.update({"step_seconds": 999.0, "cuda_peak_allocated_bytes": 999, "extra_diagnostic": 100})
    (candidate / "metrics.jsonl").write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    report = _compare(runs)
    assert report["passed"], report["mismatches"]
    assert report["ignored_candidate_field_count"] >= 15
    assert "evaluation_seconds" in report["exclusions"]["evaluation_root_fields"]
    assert report["artifacts"]["reference"]["writer.pt"]["sha256"] != report["artifacts"]["candidate"]["writer.pt"]["sha256"]
    assert report["artifacts"]["reference"]["writer.pt"]["tensor_payload_sha256"] == report["artifacts"]["candidate"]["writer.pt"]["tensor_payload_sha256"]


@pytest.mark.parametrize("mutation", ["prediction", "strict", "correct", "wrong", "rows", "bytes", "score"])
def test_every_legacy_evaluation_semantic_field_is_compared(runs, mutation):
    _, candidate = runs
    path = candidate / "eval_step_000001.json"
    value = _read_json(path)
    record = value["records"][0]
    if mutation == "prediction":
        record["predictions"]["write1_q1"] += " "  # no answer normalization is permitted
    elif mutation == "strict":
        record["strict_first_line_correct"]["write1_q1"] = True
    elif mutation == "correct":
        record["correct"]["write1_q1"] = True
    elif mutation == "wrong":
        record["wrong_values_follow_corrupted_evidence"]["q1"] = True
    elif mutation == "rows":
        record["state_rows"] += 1
    elif mutation == "bytes":
        record["state_tensor_bytes"] += 1
    else:
        value["scores"]["write1_q1"] += 1e-15
    _write_json(path, value)
    report = _compare(runs)
    assert not report["passed"]
    assert any(item["path"].startswith("eval_step_000001.json") for item in report["mismatches"])


@pytest.mark.parametrize("filename,mutation", [("writer_init.pt", "tensor"), ("writer.pt", "tensor"),
                                              ("writer.pt", "key"), ("writer.pt", "dtype"), ("writer.pt", "step")])
def test_initial_and_final_checkpoints_require_exact_keys_dtype_values_and_step(runs, filename, mutation):
    _, candidate = runs
    checkpoint = _checkpoint_load(candidate / filename)
    state = checkpoint["writer_state_dict"]
    if mutation == "tensor":
        state["linear.weight"][0, 0] += 1e-6
    elif mutation == "key":
        state["extra"] = torch.ones(1)
    elif mutation == "dtype":
        state["linear.weight"] = state["linear.weight"].double()
    else:
        checkpoint["step"] = 99
    torch.save(checkpoint, candidate / filename)
    report = _compare(runs)
    assert not report["passed"]
    assert any(filename in item["path"] for item in report["mismatches"])


def test_training_float_comparison_has_no_tolerance(runs):
    _, candidate = runs
    path = candidate / "metrics.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    next(event for event in events if event["event"] == "train")["grad_norm_before_clip"] += 1e-15
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    report = _compare(runs)
    assert not report["passed"]
    assert any("grad_norm_before_clip" in item["path"] for item in report["mismatches"])


@pytest.mark.parametrize("missing", ["initial_checkpoint", "train_step", "evaluation", "record", "strict_field", "complete"])
def test_incomplete_runs_cannot_pass_even_if_remaining_numbers_match(runs, missing):
    _, candidate = runs
    if missing == "initial_checkpoint":
        (candidate / "writer_init.pt").unlink()
    elif missing == "train_step":
        path = candidate / "metrics.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events = [event for event in events if not (event.get("event") == "train" and event.get("step") == 1)]
        path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    elif missing == "evaluation":
        (candidate / "eval_step_000001.json").unlink()
    else:
        path = candidate / "eval_step_000001.json"
        value = _read_json(path)
        if missing == "record":
            value["records"].pop()
        elif missing == "strict_field":
            value.pop("strict_first_line_exact")
        else:
            value["complete"] = False
        _write_json(path, value)
    assert not _compare(runs)["passed"]


def test_matching_but_incomplete_reference_and_candidate_fail(runs):
    for root in runs:
        value = _read_json(root / "eval_step_000001.json")
        value["records"] = []
        value["episodes"] = 0
        value["complete"] = False
        _write_json(root / "eval_step_000001.json", value)
    assert not _compare(runs)["passed"]


def test_manifest_is_exact_and_diagnostics_are_bounded_without_losing_total(runs):
    _, candidate = runs
    manifest = _read_json(candidate / "data_manifest.json")
    manifest.update({f"unexpected_{index}": index for index in range(80)})
    _write_json(candidate / "data_manifest.json", manifest)
    report = _compare(runs)
    assert not report["passed"]
    assert report["mismatch_count"] == 81  # 80 unexpected keys and the semantic manifest hash
    assert len(report["mismatches"]) == 50 and report["mismatches_truncated"]


def test_manifest_formatting_is_ignored_but_canonical_json_hash_is_exact(runs):
    _, candidate = runs
    path = candidate / "data_manifest.json"
    value = _read_json(path)
    path.write_text(json.dumps(dict(reversed(list(value.items()))), separators=(",", ":")), encoding="utf-8")
    assert _compare(runs)["passed"]
    value["training_rng_seed"] = float(value["training_rng_seed"])
    _write_json(path, value)
    report = _compare(runs)
    assert not report["passed"]
    assert any(item["path"] == "data_manifest.json.semantic_sha256" for item in report["mismatches"])


def test_cli_writes_report_and_returns_nonzero_on_default_full_run_contract(runs, tmp_path):
    reference, candidate = runs
    out = tmp_path / "report.json"
    args = ["--reference", str(reference), "--candidate", str(candidate), "--out", str(out)]
    assert main(args + ["--expected-steps", "2"]) == 0
    assert _read_json(out)["passed"]
    assert main(args) == 1
    report = _read_json(out)
    assert not report["passed"] and report["contract"]["expected_steps"] == 500


def test_invalid_json_nan_and_unsafe_checkpoint_fail_as_mismatches(runs):
    _, candidate = runs
    path = candidate / "metrics.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + '{"event": "train", "answer_nll": NaN}\n', encoding="utf-8")
    # A disallowed custom global must not trigger an unrestricted pickle retry.
    torch.save({"step": 2, "writer_state_dict": {"bad": Path("unsafe-envelope")}}, candidate / "writer.pt")
    report = _compare(runs)
    assert not report["passed"]
    assert any("non-finite" in item["reason"] for item in report["mismatches"])
    assert any("safe checkpoint load failed" in item["reason"] for item in report["mismatches"])

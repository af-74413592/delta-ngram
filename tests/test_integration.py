"""Small offline CPU runs through the public CLI and shared evaluator.

These test artifact handoffs and method/data integration, not random-model
answer quality. No model download, external repository, or GPU is required.
"""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_ngram.cli import main
from delta_ngram.config import ExperimentConfig


def write_config(directory, name, **sections):
    config = {
        "version": 1, "seed": 17,
        "model": {"smoke": True, "device": "cpu"},
        "reader": {"buckets": 97}, "writer": {"hidden_dim": 12},
        "data": {"task": "fact", "facts_per_episode": 2, "eval_episodes": 1},
        "train": {"steps": 1, "batch_episodes": 1, "eval_every": 1},
        "evaluation": {"max_new_tokens": 2, "openbook_episodes": 0,
                       "counterfactuals": True, "row_diagnostics": False},
        "run": {"output_dir": name, "max_wall_seconds": 90, "max_evidence_tokens": 512},
    }
    for section, values in sections.items():
        if isinstance(values, dict):
            config.setdefault(section, {}).update(values)
        else:
            config[section] = values
    path = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def training_events(directory):
    return [value for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if (value := json.loads(line))["event"] == "train"]


def test_cli_training_checkpoint_evaluation_and_diagnosis_round_trip(tmp_path, capsys):
    train_config = write_config(tmp_path, "train")
    assert main(["train", "--config", str(train_config)]) == 0
    training = load_json(tmp_path / "train" / "summary.json")
    assert training["completed_steps"] == 1 and training["evaluation_matches_checkpoint"]
    assert training_events(tmp_path / "train")[0]["grad_norm_before_clip"] > 0
    checkpoint = tmp_path / "train" / "writer.pt"
    initial = torch.load(tmp_path / "train" / "writer_init.pt", weights_only=True)["writer_state_dict"]
    trained = torch.load(checkpoint, weights_only=True)["writer_state_dict"]
    assert any(not torch.equal(initial[key], trained[key]) for key in initial)
    original_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected = load_json(tmp_path / "train" / "eval_step_000001.json")

    for command in ("eval", "diagnose"):
        config = write_config(tmp_path, command, run={"checkpoint": "train/writer.pt"},
                              data={"manifest": "train/data_manifest.json"})
        assert main([command, "--config", str(config)]) == 0
        result = load_json(tmp_path / command / "eval_step_000001.json")
        summary = load_json(tmp_path / command / "summary.json")
        assert summary["complete"] and summary["checkpoint_step"] == 1
        assert result["scores"] == expected["scores"]
        assert result["strict_first_line_exact"] == expected["strict_first_line_exact"]
        assert result["records"][0]["predictions"] == expected["records"][0]["predictions"]
        if command == "diagnose":
            diagnostic = result["records"][0]["row_diagnostic"]
            assert result["row_diagnostics"]["episodes"] == 1
            assert diagnostic["old_overlap_rows"]
            assert "final_restore_shared_q1" in diagnostic["predictions"]
            assert "write1_zero_shared_q1" in diagnostic["predictions"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_hash
    wrong_seed = write_config(tmp_path, "wrong-seed", seed=18,
                               run={"checkpoint": "train/writer.pt"})
    with pytest.raises(ValueError, match="smoke seed differs"):
        main(["eval", "--config", str(wrong_seed)])
    # The test drives argparse/dispatch and inspects saved public artifacts.
    assert '"command": "diagnose"' in capsys.readouterr().out


@pytest.mark.parametrize("method", ["context", "read_normalized_sgd"])
def test_shared_cli_evaluator_handles_text_and_online_gradient_methods(tmp_path, method):
    config = write_config(tmp_path, method, method={"name": method, "online_steps": 1, "lr": .01})
    assert main(["eval", "--config", str(config)]) == 0
    summary = load_json(tmp_path / method / "summary.json")
    record = load_json(tmp_path / method / "eval_step_000000.json")["records"][0]
    assert summary["complete"] and summary["episodes"] == 1
    assert summary["method"] == method
    assert record["predictions"]["zero_after_write_q1"] == record["predictions"]["empty_q1"]
    assert "wrong_values_q1" in record["predictions"]
    assert record["persistent_state_bytes"] > 0
    if method == "context":
        assert record["state_rows"] == 0
    else:
        assert record["state_rows"] > 0
        for write in record["writes"]:
            assert len(write["causal_evidence_nll"]) == 1
            assert write["gradient_norm"][0] > 0
            assert "current-block row occurrences" in write["gradient_normalization"]


@pytest.mark.parametrize("task", ["affix", "mixed"])
def test_generated_affix_and_mixed_manifests_train_through_same_pipeline(tmp_path, task):
    config = write_config(tmp_path, task,
                          data={"task": task, "support_examples": 2, "eval_episodes": 2},
                          evaluation={"max_new_tokens": 1, "counterfactuals": False})
    assert main(["train", "--config", str(config)]) == 0
    manifest = load_json(tmp_path / task / "data_manifest.json")
    summary = load_json(tmp_path / task / "summary.json")
    events = training_events(tmp_path / task)
    assert manifest["task"] == task and len(manifest["validation_episodes"]) == 2
    observed_tasks = {record.get("task", "fact") for record in manifest["validation_episodes"]}
    assert observed_tasks == ({"affix", "fact"} if task == "mixed" else {"affix"})
    assert summary["completed_steps"] == 1 and summary["evaluation_matches_checkpoint"]
    assert summary["last_evaluation"]["complete"]
    assert len(events) == 1 and events[0]["grad_norm_before_clip"] > 0
    assert all(torch.isfinite(torch.tensor(events[0][key])) for key in
               ("answer_nll", "write1_q1_nll", "write2_q2_nll", "write2_q1_nll"))


def test_config_rejects_nonfinite_fractional_and_boolean_numeric_fields(tmp_path):
    for name, patch, message in (
        ("nan-lr", {"train": {"lr": float("nan")}}, "train.lr"),
        ("fractional-batch", {"train": {"batch_episodes": 1.5}}, "train.batch_episodes"),
        ("bool-steps", {"train": {"steps": True}}, "train.steps"),
    ):
        path = write_config(tmp_path, name, **patch)
        with pytest.raises(ValueError, match=message):
            ExperimentConfig.load(path)

"""Frozen raw-checkpoint scoring and exact trainer development-sample parity."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("evaluate_raw_checkpoint", ROOT / "scripts/evaluate_raw_checkpoint.py")
evaluate_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate_script
SPEC.loader.exec_module(evaluate_script)

from delta_ngram.config import ExperimentConfig
from delta_ngram.raw_lm import train_raw
from delta_ngram.train import create_runtime


@pytest.fixture
def experiment(tmp_path):
    records = [{"id": f"{source}-{split}-{index}", "group_id": f"{source}-{split}-{index}",
                "source": source, "split": split, "text": ("abc xyz\n" if index % 2 else "xyz abc\n") * 8}
               for source in ("code", "news") for split in ("train", "validation") for index in range(2)]
    dataset = tmp_path / "documents.jsonl"
    dataset.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    values = {"seed": 17, "model": {"smoke": True, "device": "cpu"},
              "reader": {"buckets": 17}, "writer": {"hidden_dim": 12},
              "raw": {"dataset": "documents.jsonl", "block_size": 8, "prefix_tokens": 4,
                      "block_counts": [2, 4], "batch_size": 2, "loss_tokens": 64,
                      "source_weights": {"code": 1., "news": 1.},
                      "source_length_weights": {"code": {4: 1.}, "news": {2: 1.}},
                      "eval_windows": 4, "eval_min_groups_per_source": 2,
                      "eval_every_tokens": 64, "eval_bootstrap_samples": 100},
              "run": {"output_dir": "trained", "max_wall_seconds": 90}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    config = ExperimentConfig.load(path)
    summary = train_raw(config)
    checkpoint = tmp_path / "trained/writer.pt"
    return config, checkpoint, summary


def test_frozen_checkpoint_matches_training_sample_and_scores_without_optimizer(experiment, monkeypatch):
    import delta_ngram.raw_lm as raw_module
    config, checkpoint, summary = experiment
    initial_config = deepcopy(config.values)
    source_manifest = json.loads((checkpoint.parent / "data_manifest.json").read_text(encoding="utf-8"))
    hashes = {path: evaluate_script.file_sha256(path) for path in (checkpoint, config.source,
                                                                 Path(config.values["raw"]["dataset"]))}
    original = raw_module.trajectory_loss
    reads = []

    def checked(method, windows, **kwargs):
        assert not torch.is_grad_enabled()
        assert all(not parameter.requires_grad and parameter.grad is None
                   for module in (method.writer, method.reader.model) for parameter in module.parameters())
        reads.append(kwargs.get("memory_mode", "self"))
        return original(method, windows, **kwargs)

    monkeypatch.setattr(raw_module, "trajectory_loss", checked)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *args, **kwargs: pytest.fail("evaluation constructed an optimizer"))
    result = evaluate_script.evaluate_checkpoint(config, checkpoint)
    assert result["evaluation_only"] and result["optimizer_steps"] == 0
    assert result["conditions"] == ["self", "empty"] and set(reads) == {"self", "empty"}
    for key in ("validation_protocol", "validation_protocol_sha256", "validation_sample", "validation_sample_sha256"):
        assert json.loads(json.dumps(result[key])) == source_manifest[key]
    for key in ("self_nll", "empty_nll", "delta_empty", "groups", "per_source", "source_macro_delta_empty"):
        assert result[key] == summary["last_evaluation"][key]
    assert {(row["source"], row["blocks"]) for row in result["validation_sample"]} == {("code", 4), ("news", 2)}
    assert result["selected_validation_groups_per_source"] == {"code": 2, "news": 2}
    assert result["integrity_checks"]["writer_state_dict_exact_equal"]
    assert result["integrity_checks"]["reader_parameter_objects_and_versions_unchanged"]
    assert result["integrity_checks"]["reader_content_hash_verified"] is False
    assert result["checkpoint"]["training_loss_tokens"] == summary["counters"]["loss_tokens"]
    assert config.values == initial_config
    assert all(evaluate_script.file_sha256(path) == expected for path, expected in hashes.items())


def test_checkpoint_argument_overrides_config_initialization(experiment, tmp_path):
    config, checkpoint, _ = experiment
    config.values["run"]["checkpoint"] = str(tmp_path / "does-not-exist.pt")
    config.values["raw"]["resume"] = str(tmp_path / "does-not-exist-resume.pt")
    result = evaluate_script.evaluate_checkpoint(config, checkpoint)
    assert result["checkpoint"]["path"] == str(checkpoint)
    assert result["checkpoint"]["sha256"] == evaluate_script.file_sha256(checkpoint)


@pytest.mark.parametrize("mutation,match", [
    (lambda payload: payload["config"]["package_config"]["model"].update(revision="b" * 40), "checkpoint model.revision differs"),
    (lambda payload: payload["config"].update(tokenizer_fingerprint="sha256:" + "0" * 64), "tokenizer fingerprint differs"),
    (lambda payload: payload.update(format="not-a-writer"), "unsupported writer checkpoint"),
])
def test_rejects_checkpoint_identity_mismatch(experiment, tmp_path, mutation, match):
    config, checkpoint, _ = experiment
    payload = torch.load(checkpoint, weights_only=True)
    mutation(payload)
    invalid = tmp_path / "invalid.pt"
    torch.save(payload, invalid)
    with pytest.raises(ValueError, match=match):
        evaluate_script.evaluate_checkpoint(config, invalid)


@pytest.mark.parametrize("mutation,match", [
    ("writer_data", "writer state_dict changed"),
    ("reader_parameter", "reader parameters changed"),
    ("reader_grad", "reader parameters changed"),
])
def test_detects_frozen_parameter_mutation(experiment, monkeypatch, mutation, match):
    import delta_ngram.raw_lm as raw_module
    config, checkpoint, _ = experiment
    original = raw_module.evaluate_raw

    def altered(method, *args, **kwargs):
        result = original(method, *args, **kwargs)
        if mutation == "writer_data":
            next(method.writer.parameters()).data.add_(1)
        elif mutation == "reader_parameter":
            with torch.no_grad():
                next(method.reader.model.parameters()).add_(1)
        else:
            parameter = next(method.reader.model.parameters())
            parameter.grad = torch.ones_like(parameter)
        return result

    monkeypatch.setattr(raw_module, "evaluate_raw", altered)
    with pytest.raises(RuntimeError, match=match):
        evaluate_script.evaluate_checkpoint(config, checkpoint)


def test_rejects_missing_per_source_development_groups(experiment):
    config, _, _ = experiment
    config.values["raw"]["eval_min_groups_per_source"] = 3
    config.values["raw"]["eval_windows"] = 6
    method, _ = create_runtime(config)
    with pytest.raises(ValueError, match="has 2 groups; requires 3"):
        evaluate_script.prepare_validation(config, method)


def test_recorded_real_evaluation_rejects_dirty_git_before_model_loading(experiment, monkeypatch):
    import delta_ngram.train as train_module
    config, checkpoint, _ = experiment
    config.values["model"]["smoke"] = False
    monkeypatch.setattr(evaluate_script, "git_identity", lambda: {"commit": "a" * 40, "dirty": True})
    monkeypatch.setattr(train_module, "create_runtime", lambda *args: pytest.fail("dirty real run loaded a model"))
    with pytest.raises(RuntimeError, match="Commit repository changes"):
        evaluate_script.evaluate_checkpoint(config, checkpoint)


def test_cli_deterministic_evaluation_and_output_immutability(experiment, tmp_path):
    config, checkpoint, _ = experiment
    output = tmp_path / "evaluation.json"
    command = [sys.executable, str(ROOT / "scripts/evaluate_raw_checkpoint.py"),
               "--config", str(config.source), "--checkpoint", str(checkpoint),
               "--output", str(output), "--threads", "2"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["complete"] and result["run"]["deterministic_algorithms"]
    assert result["run"]["cublas_workspace_config"] == ":4096:8"
    assert result["run"]["requested_threads"] == result["run"]["threads"] == 2
    before = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert repeated.returncode != 0 and "FileExistsError" in repeated.stderr
    assert output.read_bytes() == before

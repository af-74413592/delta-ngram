"""CPU-only baseline CLI identity, query parity and immutable-output checks."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from delta_ngram.alpha_evaluation import evaluate_revision
from delta_ngram.alpha_training import _json_hash
from delta_ngram.checkpoints import load_checkpoint, save_checkpoint
from test_alpha_evaluation import revision_annotations
from test_recent_news_evaluation import example, prepare

SPEC = importlib.util.spec_from_file_location("evaluate_d1_revision",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_d1_revision.py")
cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cli
SPEC.loader.exec_module(cli)


@pytest.fixture
def inputs(example):
    config, method, _, registry_path, registry_hash, checkpoint = example
    prepared = prepare(example)
    labels = revision_annotations(example, prepared)
    metadata = load_checkpoint(checkpoint)["config"]
    base_identity = deepcopy(metadata["raw_identity"])
    metadata["raw_identity"]["validation"] = {"validation_sample_sha256": "a" * 64}
    metadata["training_loss_tokens"] = 192
    save_checkpoint(checkpoint, method.writer, metadata, 3, protocol="delta_ngram_raw_lm_v1")
    prefix = {"format": "delta_ngram_m1_optimizer_prefix_v1", "config_sha256": cli.file_sha256(config.source),
        "m1_identity": base_identity, "writer_init_sha256": "1" * 64, "m1_checkpoint_sha256": "2" * 64,
        "full_loss_token_budget": config.values["raw"]["loss_tokens"], "stop_step": 3,
        "groups": [{}, {}, {}], "actual_loss_tokens": 192, "nominal_loss_tokens": 192}
    prefix["manifest_sha256"] = _json_hash(prefix)
    prefix_path = checkpoint.parent / "prefix.json"
    prefix_path.write_text(json.dumps(prefix), encoding="utf-8")
    kwargs = {"checkpoint_sha256": cli.file_sha256(checkpoint), "prefix_manifest": prefix_path,
        "registry_path": registry_path, "registry_sha256": registry_hash,
        "annotations_path": labels.path, "annotations_sha256": labels.sha256, "bootstrap_samples": 2}
    return config, method, checkpoint, prepared, labels, kwargs


def test_d1_cli_matches_shared_revision_scorer_and_never_constructs_optimizer(inputs, monkeypatch):
    config, method, checkpoint, prepared, labels, kwargs = inputs
    expected = evaluate_revision(method, prepared, labels, backend=None, bootstrap_samples=2)
    def forbidden(*args, **kwargs):
        pytest.fail("D1 revision constructed optimizer or invoked gradients")
    for name in ("AdamW", "Adam", "SGD"):
        monkeypatch.setattr(torch.optim, name, forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    # These unrelated training/resume locations must never be opened.
    config.values["raw"]["dataset"] = "must-not-open-training-data.jsonl"
    config.values["raw"]["resume"] = "must-not-open-optimizer.pt"
    config.values["run"]["checkpoint"] = "must-not-load-initial-writer.pt"
    result = cli.evaluate_checkpoint(config, checkpoint, **kwargs)
    for key in ("per_family", "retention_curve", "by_revision_type", "control_scope", "revision_causal_limitation"):
        assert result[key] == expected[key]
    assert result["baseline_unit_alpha_D1"]
    assert result["checkpoint"]["checkpoint_tokenizer_fingerprint_verified"]
    assert result["m1_prefix"]["config_bytes_and_complete_prefix_verified"]
    assert result["model"]["matches_registry_reader"]
    assert result["optimizer_steps"] == result["backward_calls"] == 0
    assert result["independent_families"] == 1
    assert set(result["retention_curve"]) == {"0", "1", "4", "8", "16"}


@pytest.mark.parametrize("mutation,match", [
    ("checkpoint_hash", "checkpoint file SHA256"),
    ("config_bytes", "configuration file bytes"),
    ("prefix_hash", "prefix format/hash"),
    ("raw_identity", "prefix identity"),
    ("step", "prefix endpoint"),
    ("tokens", "prefix endpoint"),
    ("alpha", "not an alpha writer"),
    ("registry_hash", "registry file SHA256"),
    ("annotations_hash", "annotations file SHA256"),
])
def test_invalid_identity_rejected_before_runtime_construction(inputs, monkeypatch, mutation, match):
    import delta_ngram.train as training
    config, _, checkpoint, _, _, kwargs = inputs
    if mutation == "checkpoint_hash":
        kwargs["checkpoint_sha256"] = "0" * 64
    elif mutation == "config_bytes":
        # Equivalent YAML with different LF/CRLF bytes remains a different
        # frozen input; this preflight must exercise the actual byte guard.
        original = config.source.read_bytes()
        changed = original.replace(b"\r\n", b"\n")
        changed = changed.replace(b"\n", b"\r\n") if changed == original else changed
        assert changed != original
        config.source.write_bytes(changed)
    elif mutation == "prefix_hash":
        prefix = json.loads(kwargs["prefix_manifest"].read_bytes())
        prefix["manifest_sha256"] = "0" * 64
        kwargs["prefix_manifest"].write_text(json.dumps(prefix), encoding="utf-8")
    elif mutation in ("registry_hash", "annotations_hash"):
        kwargs[mutation.replace("_hash", "_sha256")] = "0" * 64
    else:
        payload = load_checkpoint(checkpoint)
        if mutation == "raw_identity":
            payload["config"]["raw_identity"]["extra"] = "changed"
        elif mutation == "step":
            payload["step"] += 1
        elif mutation == "tokens":
            payload["config"]["training_loss_tokens"] += 1
        else:
            payload["config"]["requires_alpha_payload"] = True
        torch.save(payload, checkpoint)
        kwargs["checkpoint_sha256"] = cli.file_sha256(checkpoint)
    monkeypatch.setattr(training, "create_runtime", lambda *a: pytest.fail("invalid identity constructed a runtime"))
    with pytest.raises(ValueError, match=match):
        cli.evaluate_checkpoint(config, checkpoint, **kwargs)


def test_runtime_tokenizer_and_frozen_parameter_guards(inputs, monkeypatch):
    import delta_ngram.alpha_evaluation as revision
    config, method, checkpoint, _, _, kwargs = inputs
    payload = load_checkpoint(checkpoint)
    payload["config"]["tokenizer_fingerprint"] = "sha256:" + "0" * 64
    torch.save(payload, checkpoint)
    kwargs["checkpoint_sha256"] = cli.file_sha256(checkpoint)
    with pytest.raises(ValueError, match="tokenizer identity"):
        cli.evaluate_checkpoint(config, checkpoint, **kwargs)
    payload["config"]["tokenizer_fingerprint"] = "sha256:" + _json_hash(payload["config"]["raw_identity"]["tokenizer"])
    torch.save(payload, checkpoint)
    kwargs["checkpoint_sha256"] = cli.file_sha256(checkpoint)
    original = revision.score_revision_candidates
    def corrupt(method, *args, **kwargs):
        score = original(method, *args, **kwargs)
        next(method.writer.parameters()).data.add_(1)
        return score
    monkeypatch.setattr(revision, "score_revision_candidates", corrupt)
    with pytest.raises(RuntimeError, match="writer state_dict changed"):
        cli.evaluate_checkpoint(config, checkpoint, **kwargs)


def test_d1_revision_cli_output_is_immutable_and_records_source_hashes(inputs, tmp_path):
    config, _, checkpoint, _, _, kwargs = inputs
    output = tmp_path / "d1-revision.json"
    argv = ["--config", str(config.source), "--checkpoint", str(checkpoint),
            "--output", str(output), "--threads", "1"]
    for key, value in kwargs.items():
        flag = "registry" if key == "registry_path" else "annotations" if key == "annotations_path" else key.replace("_", "-")
        argv.extend(["--" + flag, str(value)])
    assert cli.main(argv) == 0
    result = json.loads(output.read_bytes())
    assert result["run"]["requested_threads"] == 1 and result["run"]["deterministic_algorithms"]
    assert result["run"]["config_sha256"] == cli.file_sha256(config.source)
    assert result["registry_file_sha256"] == kwargs["registry_sha256"]
    assert result["annotation_sha256"] == kwargs["annotations_sha256"]
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        cli.main(argv)
    assert output.read_bytes() == before

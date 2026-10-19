"""Research transfer stays explicit, matched, frozen, and destination-encoded."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_ngram.checkpoints import load_checkpoint, restore_writer, save_checkpoint
from delta_ngram.config import ExperimentConfig
from delta_ngram.data.raw_text import RawDocument, TrajectoryWindow
from delta_ngram.p2_transfer import (compare_transfer_evaluations, evaluate_transfer,
                                    fixed_validation_windows, load_research_transfer_checkpoint)
from delta_ngram.raw_lm import (PROTOCOL, _identity, _json_hash, _validation_sample,
                               _writer_metadata, evaluate_raw, tokenizer_identity)
from delta_ngram.train import create_runtime


def setup_case(tmp_path, *, step=2):
    values = {"seed": 17, "model": {"smoke": True, "device": "cpu", "name_or_path": "research-I", "revision": "I-revision"},
              "reader": {"buckets": 17}, "writer": {"hidden_dim": 12},
              "raw": {"block_size": 4, "prefix_tokens": 2, "block_counts": [2, 4], "batch_size": 2,
                      "eval_windows": 2, "eval_seed": 104729, "eval_bootstrap_samples": 31}}
    destination_path = tmp_path / "destination.yaml"
    destination_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    config = ExperimentConfig.load(destination_path)
    values["model"].update(name_or_path="research-B", revision="B-revision")
    source_path = tmp_path / "source.yaml"
    source_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    source_config = ExperimentConfig.load(source_path)
    method, args = create_runtime(config)
    tokenizer = tokenizer_identity(method)
    fingerprint = "sha256:" + _json_hash(tokenizer)
    docs = [RawDocument(f"doc-{i}", f"group-{i}", "code/python", "validation",
                        tuple(3 + (j * 7 + i) % 83 for j in range(16)), tokenizer_identity=fingerprint)
            for i in range(2)]
    windows = [TrajectoryWindow(doc.id, doc.group_id, doc.source, 0, doc.token_ids, 4) for doc in docs]
    validation = {"name": "group_first_fixed_dev_paired_bootstrap_v1", "selection": "group_first_round_robin",
                  "group_key": "group_id", "seed": 104729, "window_limit": 2,
                  "bootstrap_samples": 31, "minimum_groups": 64,
                  "evaluation_batching": "within_group", "batch_size": 2, "prefix_tokens": 2}
    sample = _validation_sample(windows)
    bindings = {"validation_protocol_sha256": _json_hash(validation), "validation_sample_sha256": _json_hash(sample)}
    manifest = {"dataset_sha256": "a" * 64, "tokenizer": tokenizer, "tokenizer_fingerprint": fingerprint,
                "validation_protocol": validation, "validation_sample": sample, **bindings}
    identities = {}
    checkpoint_paths = {}
    for label, setting in (("destination", config), ("source", source_config)):
        identity = _identity(setting, method, manifest["dataset_sha256"], tokenizer)
        identity["validation"] = bindings
        identities[label] = identity
        metadata = _writer_metadata(setting, method, setting.legacy_args(), identity)
        if step:
            metadata["training_loss_tokens"] = step * 24
        path = tmp_path / f"{label}.pt"
        save_checkpoint(path, method.writer, metadata, step, protocol=PROTOCOL)
        checkpoint_paths[label] = path
    if step:
        payload = load_checkpoint(checkpoint_paths["source"])
        payload["writer_state_dict"]["target_head.2.bias"] = torch.full_like(
            payload["writer_state_dict"]["target_head.2.bias"], .2)
        torch.save(payload, checkpoint_paths["source"])
    return config, method, checkpoint_paths, identities, docs, manifest, windows


def load_transfer(case, **kwargs):
    config, method, paths, identities, *_ = case
    return load_research_transfer_checkpoint(
        method, config, paths["source"], paths["destination"],
        expected_source_identity=identities["source"],
        expected_destination_identity=identities["destination"], **kwargs)


def test_transfer_opt_in_and_original_checkpoint_identity_remain_strict(tmp_path):
    case = setup_case(tmp_path)
    config, method, paths, *_ = case
    with pytest.raises(ValueError, match="research_transfer"):
        load_transfer(case)
    with pytest.raises(ValueError, match="model.name_or_path"):
        restore_writer(method, load_checkpoint(paths["source"]), config)
    metadata = load_transfer(case, research_transfer=True)
    assert metadata["condition"] == "B_to_I"
    assert metadata["source"]["raw_identity"]["model"]["name_or_path"] == "research-B"
    assert metadata["destination"]["raw_identity"]["model"]["name_or_path"] == "research-I"
    assert metadata["source"]["writer_tensor_sha256"] != metadata["destination"]["writer_tensor_sha256"]
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in method.writer.parameters())


def test_step_zero_missing_loss_tokens_is_explicitly_zero(tmp_path):
    case = setup_case(tmp_path, step=0)
    metadata = load_transfer(case, research_transfer=True)
    assert metadata["source"]["step"] == metadata["destination"]["training_loss_tokens"] == 0
    assert metadata["source"]["writer_tensor_sha256"] == metadata["destination"]["writer_tensor_sha256"]


def test_step_zero_requires_identical_initial_weights(tmp_path):
    case = setup_case(tmp_path, step=0)
    payload = load_checkpoint(case[2]["source"])
    payload["writer_state_dict"]["target_head.2.bias"][0] += .2
    torch.save(payload, case[2]["source"])
    with pytest.raises(ValueError, match="step-zero initialization"):
        load_transfer(case, research_transfer=True)


@pytest.mark.parametrize("change,error", [
    ("tokenizer", "tokenizer"), ("reader", "reader"), ("writer", "writer"), ("method", "method"),
    ("raw", "raw"), ("step", "training step"), ("tokens", "actual training"),
    ("missing_tokens", "training_loss_tokens"), ("shape", "shape"), ("dtype", "dtype"),
    ("nonfinite", "finite"), ("width", "width"),
])
def test_incompatible_source_fails_before_writer_mutation(tmp_path, change, error):
    case = setup_case(tmp_path)
    _, method, paths, identities, *_ = case
    before = {name: tensor.clone() for name, tensor in method.writer.state_dict().items()}
    payload = load_checkpoint(paths["source"])
    identity, package = payload["config"]["raw_identity"], payload["config"]["package_config"]
    if change == "tokenizer":
        identity["tokenizer"]["vocab_sha256"] = "b" * 64
        payload["config"]["tokenizer_fingerprint"] = "sha256:" + _json_hash(identity["tokenizer"])
    elif change in ("reader", "writer", "method", "raw"):
        key, value = {"reader": ("layer_index", 0), "writer": ("right_window", 2),
                      "method": ("kappa", .5), "raw": ("prefix_tokens", 1)}[change]
        identity[change][key] = package[change][key] = value
        if change == "reader":
            payload["config"][key] = value
    elif change == "step":
        payload["step"] += 1
    elif change == "tokens":
        payload["config"]["training_loss_tokens"] += 1
    elif change == "missing_tokens":
        del payload["config"]["training_loss_tokens"]
    elif change == "shape":
        payload["writer_state_dict"]["target_head.2.bias"] = torch.zeros(1)
    elif change == "dtype":
        payload["writer_state_dict"]["target_head.2.bias"] = payload["writer_state_dict"]["target_head.2.bias"].double()
    elif change == "nonfinite":
        payload["writer_state_dict"]["target_head.2.bias"][0] = float("nan")
    elif change == "width":
        payload["config"]["width"] += 1
    identities["source"] = deepcopy(identity)
    torch.save(payload, paths["source"])
    with pytest.raises(ValueError, match=error):
        load_transfer(case, research_transfer=True)
    assert all(torch.equal(before[name], value) for name, value in method.writer.state_dict().items())


def test_destination_encoding_readout_exact_targets_and_direct_paired_G(tmp_path, monkeypatch):
    case = setup_case(tmp_path)
    _, method, _, _, docs, manifest, windows = case
    baseline = evaluate_raw(method, windows, prefix_tokens=2, batch_size=2, bootstrap_samples=31)
    baseline.update({key: manifest[key] for key in ("validation_protocol_sha256", "validation_sample_sha256")})
    baseline.update(step=2, training_loss_tokens=48)
    destination_model = method.reader.model
    original_weights = {name: value.clone() for name, value in destination_model.state_dict().items()}
    metadata = load_transfer(case, research_transfer=True)
    reconstructed = fixed_validation_windows(docs, manifest,
                                              expected_identity=metadata["destination"]["raw_identity"], block_size=4)
    assert reconstructed == windows
    evidence_calls = []
    original_encode = method.reader.encode_evidence

    def encode(ids, mask):
        assert method.reader.model is destination_model
        evidence_calls.append(ids.clone())
        return original_encode(ids, mask)

    monkeypatch.setattr(method.reader, "encode_evidence", encode)
    result = evaluate_transfer(method, reconstructed, metadata=metadata,
                               baseline_evaluation=baseline, bootstrap_samples=31)
    assert evidence_calls and len(evidence_calls) == 6
    assert result["comparison"]["G"] == pytest.approx(baseline["self_nll"] - result["transfer_evaluation"]["self_nll"])
    assert result["comparison"]["empty_memory_check"]["max_group_nll_difference"] == 0
    assert result["comparison"]["decision"] == "insufficient_independent_groups"
    assert all(torch.equal(original_weights[name], value) for name, value in destination_model.state_dict().items())
    changed = deepcopy(manifest)
    changed["validation_sample"][0]["token_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="sample digest"):
        fixed_validation_windows(docs, changed, expected_identity=metadata["destination"]["raw_identity"], block_size=4)
    with pytest.raises(ValueError, match="target sample"):
        evaluate_transfer(method, reconstructed[::-1], metadata=metadata, baseline_evaluation=baseline)


def paired_case():
    groups = [{"group_id": f"g-{i}", "document_ids": [f"d-{i}"], "windows": 1,
               "loss_tokens": 10, "self_loss_sum": 20., "empty_loss_sum": 30.} for i in range(64)]
    baseline = {"groups": groups, "group_count": 64, "loss_tokens": 640, "windows": 64,
                "validation_protocol_sha256": "a" * 64, "validation_sample_sha256": "b" * 64}
    transfer = deepcopy(baseline)
    for item in transfer["groups"]:
        item["self_loss_sum"] = 19.
    return baseline, transfer


def test_direct_group_bootstrap_G_has_positive_sign_for_better_base_writer():
    baseline, transfer = paired_case()
    result = compare_transfer_evaluations(baseline, transfer, bootstrap_samples=31)
    assert result["G"] == pytest.approx(.1)
    assert result["paired_group_bootstrap"]["ci_lower"] == pytest.approx(.1)
    assert result["paired_group_bootstrap"]["ci_upper"] == pytest.approx(.1)
    assert result["decision"] == "supports_B_to_I_on_this_seed"
    reverse = compare_transfer_evaluations(transfer, baseline, bootstrap_samples=31)
    assert reverse["G"] == pytest.approx(-.1)
    assert reverse["decision"] == "supports_I_to_I_on_this_seed"


@pytest.mark.parametrize("change", ["sample", "tokens", "group", "empty", "documents", "nonfinite"])
def test_paired_comparison_rejects_unmatched_evaluations(change):
    baseline, transfer = paired_case()
    if change == "sample":
        transfer["validation_sample_sha256"] = "c" * 64
    elif change == "tokens":
        transfer["groups"][0]["loss_tokens"] -= 1
    elif change == "group":
        transfer["groups"][0]["group_id"] = "unmatched"
    elif change == "empty":
        transfer["groups"][0]["empty_loss_sum"] += .1
    elif change == "documents":
        transfer["groups"][0]["document_ids"] = ["other"]
    elif change == "nonfinite":
        transfer["groups"][0]["self_loss_sum"] = float("nan")
    with pytest.raises(ValueError):
        compare_transfer_evaluations(baseline, transfer, bootstrap_samples=31)

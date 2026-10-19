"""Exact initialization and fail-before-mutation tests without model downloads."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_ngram.checkpoints import load_checkpoint, restore_writer, save_checkpoint
from delta_ngram.config import DEFAULTS
from delta_ngram.p2_initialization import (initialize_p2_writer, validate_writer_tensors,
                                            writer_tensor_sha256)
from delta_ngram.raw_lm import _identity
from delta_ngram.writer import ForwardWriter


def fixture(tmp_path):
    values = deepcopy(DEFAULTS)
    values["model"].update(name_or_path="Qwen/Qwen3-0.6B", revision="a" * 40,
                           smoke=False, device="cpu")
    values["reader"].update(buckets=17, orders=[2, 3], layer_index=1)
    values["writer"].update(hidden_dim=8, right_window=4)
    values["raw"]["loss_tokens"] = 5000000
    model = torch.nn.Linear(4, 4)
    model.config = SimpleNamespace(_commit_hash="a" * 40)
    reader = SimpleNamespace(width=4, buckets=17, orders=(2, 3), layer_index=1,
                             backend="external_hash", model=model)
    torch.manual_seed(17)
    writer = ForwardWriter(4, 8)
    method = SimpleNamespace(reader=reader, writer=writer, device=torch.device("cpu"),
                             supports_outer_training=True)
    config = SimpleNamespace(values=values)
    tokenizer = {"class": "Qwen2Tokenizer", "vocab_sha256": "1" * 64,
                 "pipeline_sha256": "2" * 64,
                 "special_tokens": {"eos_token_id": 151645, "pad_token_id": 151643, "bos_token_id": None}}
    source_identity = _identity(config, method, "3" * 64, tokenizer)
    source_identity["validation"] = {"validation_protocol_sha256": "4" * 64,
                                      "validation_sample_sha256": "5" * 64}
    metadata = {"model": values["model"]["name_or_path"], "revision": "a" * 40,
                "resolved_model_commit": "a" * 40, "seed": values["seed"], "smoke": False,
                "buckets": 17, "orders": [2, 3], "layer_index": 1, "width": 4,
                "hidden_dim": 8, "package_config": deepcopy(values),
                "raw_identity": deepcopy(source_identity)}
    path = tmp_path / "writer_init.pt"
    save_checkpoint(path, writer, metadata, 0, protocol="delta_ngram_raw_lm_v1")
    values["model"].update(name_or_path="Qwen/Qwen3-0.6B-Base", revision="b" * 40)
    model.config._commit_hash = "b" * 40
    destination_identity = _identity(config, method, "3" * 64, tokenizer)
    destination_identity["validation"] = deepcopy(source_identity["validation"])
    # Deliberately consume a different RNG stream as a different model load can.
    torch.manual_seed(991)
    method.writer = ForwardWriter(4, 8)
    return path, method, config, destination_identity


def test_exact_initialization_restores_all_tensors_only(tmp_path):
    path, method, config, identity = fixture(tmp_path)
    source = load_checkpoint(path)["writer_state_dict"]
    assert writer_tensor_sha256(source) != writer_tensor_sha256(method.writer.state_dict())
    optimizer = torch.optim.AdamW(method.writer.parameters(), lr=.0003, weight_decay=0)
    rng_before = torch.get_rng_state().clone()
    report = initialize_p2_writer(method, config, path, identity)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert not optimizer.state
    assert all(torch.equal(value, source[name]) for name, value in method.writer.state_dict().items())
    assert report["writer_tensor_sha256"] == writer_tensor_sha256(source)
    assert report["source_file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["source_identity"]["model"]["name_or_path"] == "Qwen/Qwen3-0.6B"
    assert report["destination_identity"]["model"]["name_or_path"] == "Qwen/Qwen3-0.6B-Base"
    assert report["source_step"] == 0


@pytest.mark.parametrize("mutation", ["step", "protocol", "optimizer", "tokens", "tokenizer",
    "dataset", "seed", "address", "operator", "trajectory", "validation", "budget",
    "source_model", "destination_model", "revision", "inconsistent_package", "missing_identity",
    "source_resume", "destination_resume", "width", "runtime_dtype", "tokenizer_incomplete",
    "malformed_hash", "tensor_shape", "tensor_dtype", "tensor_keys", "nan"])
def test_invalid_initialization_cannot_partially_load(tmp_path, mutation):
    path, method, config, identity = fixture(tmp_path)
    payload = load_checkpoint(path)
    source = payload["config"]["raw_identity"]
    if mutation == "step":
        payload["step"] = 1
    elif mutation == "protocol":
        payload["protocol"] = "delta_ngram_pilot_v1"
    elif mutation == "optimizer":
        payload["optimizer_state_dict"] = {}
    elif mutation == "tokens":
        payload["config"]["training_loss_tokens"] = 1
    elif mutation == "tokenizer":
        source["tokenizer"]["vocab_sha256"] = "6" * 64
    elif mutation == "dataset":
        source["dataset_sha256"] = "6" * 64
    elif mutation == "seed":
        source["seed"] = 29
    elif mutation == "address":
        source["reader"]["buckets"] += 1
    elif mutation == "operator":
        source["method"]["operator"] = "D2"
    elif mutation == "trajectory":
        source["raw"]["block_size"] += 1
    elif mutation == "validation":
        source["validation"]["validation_sample_sha256"] = "6" * 64
    elif mutation == "budget":
        config.values["raw"]["loss_tokens"] += 1
    elif mutation == "source_model":
        source["model"]["name_or_path"] = "other"
    elif mutation == "destination_model":
        identity["model"]["name_or_path"] = "other"
    elif mutation == "revision":
        method.reader.model.config._commit_hash = "c" * 40
    elif mutation == "inconsistent_package":
        payload["config"]["package_config"]["writer"]["hidden_dim"] += 1
    elif mutation == "missing_identity":
        del source["tokenizer"]
    elif mutation == "source_resume":
        payload["config"]["package_config"]["raw"]["resume"] = "training.pt"
    elif mutation == "destination_resume":
        config.values["raw"]["resume"] = "training.pt"
    elif mutation == "width":
        payload["config"]["width"] += 1
    elif mutation == "runtime_dtype":
        method.reader.model.double()
    elif mutation == "tokenizer_incomplete":
        del source["tokenizer"]["special_tokens"]["bos_token_id"]
    elif mutation == "malformed_hash":
        source["dataset_sha256"] = "unknown"
    else:
        state = payload["writer_state_dict"]
        key = list(state)[-1]
        # Change the first tensor too: loading before all checks would corrupt it.
        state[next(iter(state))].add_(10)
        if mutation == "tensor_shape":
            state[key] = state[key].reshape(-1).repeat(2)
        elif mutation == "tensor_dtype":
            state[key] = state[key].double()
        elif mutation == "tensor_keys":
            del state[key]
        elif mutation == "nan":
            state[key].fill_(float("nan"))
    torch.save(payload, path)
    before = writer_tensor_sha256(method.writer.state_dict())
    with pytest.raises(ValueError):
        initialize_p2_writer(method, config, path, identity)
    assert writer_tensor_sha256(method.writer.state_dict()) == before


def test_normal_restore_still_rejects_cross_model_transfer(tmp_path):
    path, method, config, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="model.name_or_path differs"):
        restore_writer(method, load_checkpoint(path), config)


def test_explicit_instruction_tokenizer_binding_is_preserved(tmp_path):
    path, method, config, identity = fixture(tmp_path)
    config.values["model"].update(tokenizer_name_or_path="Qwen/Qwen3-0.6B", tokenizer_revision="a" * 40)
    report = initialize_p2_writer(method, config, path, identity)
    binding = report["tokenizer_binding"]
    assert binding["destination_override"] is True
    assert binding["destination"] == binding["source"] == {
        "name_or_path": "Qwen/Qwen3-0.6B", "revision": "a" * 40}


@pytest.mark.parametrize("name,revision", [("other", "a" * 40),
    ("Qwen/Qwen3-0.6B", "c" * 40), ("Qwen/Qwen3-0.6B", None), (None, "a" * 40)])
def test_tokenizer_override_must_explicitly_match_source(tmp_path, name, revision):
    path, method, config, identity = fixture(tmp_path)
    config.values["model"].update(tokenizer_name_or_path=name, tokenizer_revision=revision)
    before = writer_tensor_sha256(method.writer.state_dict())
    with pytest.raises(ValueError, match="tokenizer"):
        initialize_p2_writer(method, config, path, identity)
    assert writer_tensor_sha256(method.writer.state_dict()) == before


def test_tensor_fingerprint_is_order_independent_and_supports_bf16():
    state = {"b": torch.tensor([1., 2.], dtype=torch.bfloat16), "a": torch.tensor(3.)}
    assert writer_tensor_sha256(state) == writer_tensor_sha256(dict(reversed(list(state.items()))))
    changed = {key: value.clone() for key, value in state.items()}
    changed["b"][0] = 4
    assert writer_tensor_sha256(state) != writer_tensor_sha256(changed)


def test_tensor_validation_does_not_cast_dtypes():
    writer = ForwardWriter(4, 8)
    state = {key: value.double() for key, value in writer.state_dict().items()}
    with pytest.raises(ValueError, match="dtype differs"):
        validate_writer_tensors(writer, state)

"""Pinned tokenizer binding must remain explicit and checkpoint-strict."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delta_ngram.config import ExperimentConfig
from delta_ngram.runtime import load_model_and_tokenizer


def test_loader_binds_tokenizer_independently_without_changing_reader(monkeypatch):
    import transformers
    calls = []
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(use_cache=True)
    def tokenizer(name, **kwargs):
        calls.append(("tokenizer", name, kwargs))
        return "tokenizer"
    def weights(name, **kwargs):
        calls.append(("weights", name, kwargs))
        return model
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", weights)
    args = SimpleNamespace(smoke=False, model="base-reader", revision="a" * 40,
                           tokenizer_name_or_path="shared-tokenizer", tokenizer_revision="b" * 40)
    reader, tok = load_model_and_tokenizer(args, torch.device("cpu"))
    assert reader is model and tok == "tokenizer"
    assert calls[0] == ("tokenizer", "shared-tokenizer", {"revision": "b" * 40})
    assert calls[1][1] == "base-reader" and calls[1][2]["revision"] == "a" * 40
    assert all(not p.requires_grad for p in reader.parameters())


@pytest.mark.parametrize("fields", [
    {"tokenizer_name_or_path": "shared"},
    {"tokenizer_revision": "b" * 40},
    {"tokenizer_name_or_path": "shared", "tokenizer_revision": "main"},
    {"tokenizer_name_or_path": "shared", "tokenizer_revision": "b" * 40, "smoke": True},
])
def test_invalid_binding_rejected(tmp_path, fields):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"model": fields}), encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer"):
        ExperimentConfig.load(path)


def test_normal_checkpoint_rejects_changed_tokenizer_binding(tmp_path):
    from delta_ngram.checkpoints import load_checkpoint, restore_writer, save_checkpoint
    from delta_ngram.train import create_runtime
    path = tmp_path / "config.yaml"
    path.write_text("model: {smoke: true, device: cpu}\n", encoding="utf-8")
    config = ExperimentConfig.load(path)
    method, args = create_runtime(config)
    metadata = {**vars(args), "out": str(args.out), "width": method.reader.width,
                "orders": list(method.reader.orders), "package_config": config.values}
    checkpoint = tmp_path / "writer.pt"
    save_checkpoint(checkpoint, method.writer, metadata, 0)
    payload = load_checkpoint(checkpoint)
    # Missing old fields retain their historical model-tokenizer binding.
    payload["config"]["package_config"]["model"].pop("tokenizer_name_or_path")
    payload["config"]["package_config"]["model"].pop("tokenizer_revision")
    restore_writer(method, payload, config)
    changed = deepcopy(config.values)
    changed["model"]["tokenizer_name_or_path"] = "a-different-tokenizer"
    changed["model"]["tokenizer_revision"] = "b" * 40
    with pytest.raises(ValueError, match="tokenizer_name_or_path"):
        restore_writer(method, payload, ExperimentConfig(changed, path))


def test_historical_package_can_still_construct_legacy_args(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model: {smoke: true, device: cpu}\n", encoding="utf-8")
    values = deepcopy(ExperimentConfig.load(path).values)
    values["model"].pop("tokenizer_name_or_path")
    values["model"].pop("tokenizer_revision")
    args = ExperimentConfig(values, path).legacy_args()
    assert args.tokenizer_name_or_path is None and args.tokenizer_revision is None


@pytest.mark.parametrize("field,value", [("tokenizer_name_or_path", "different-tokenizer"),
                                         ("tokenizer_revision", "b" * 40)])
def test_legacy_checkpoint_rejects_changed_tokenizer_binding(tmp_path, field, value):
    from delta_ngram.checkpoints import LEGACY_PROTOCOL, load_checkpoint, restore_writer
    from delta_ngram.train import create_runtime
    path = tmp_path / "config.yaml"
    path.write_text("model: {smoke: true, device: cpu}\n", encoding="utf-8")
    config = ExperimentConfig.load(path)
    method, args = create_runtime(config)
    metadata = {**vars(args), "out": str(args.out), "width": method.reader.width,
                "orders": list(method.reader.orders)}
    checkpoint = tmp_path / "historical.pt"
    torch.save({"protocol": LEGACY_PROTOCOL, "step": 0, "config": metadata,
                "writer_state_dict": method.writer.state_dict()}, checkpoint)
    payload = load_checkpoint(checkpoint, allow_legacy=True)
    restore_writer(method, payload, config)
    changed = deepcopy(config.values)
    changed["model"]["tokenizer_name_or_path"] = changed["model"]["name_or_path"]
    changed["model"]["tokenizer_revision"] = changed["model"]["revision"]
    changed["model"][field] = value
    with pytest.raises(ValueError, match="historical checkpoint tokenizer binding differs"):
        restore_writer(method, payload, ExperimentConfig(changed, path))

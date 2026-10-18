"""Checkpoint identity boundaries without loading a language model."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_ngram.checkpoints import FORMAT, LEGACY_PROTOCOL, VERSION, load_checkpoint, restore_writer


def fixture_objects(*, legacy=False):
    values = {
        "seed": 17,
        "model": {"name_or_path": "fixture/tiny-base", "revision": "revision-a", "smoke": True},
        "reader": {"backend": "external_hash", "buckets": 17, "orders": [2, 3], "layer_index": 1},
        "writer": {"hidden_dim": 8, "right_window": 4},
        "method": {"name": "forward", "operator": "D1"},
    }
    writer = torch.nn.Linear(2, 2)
    writer.operator = SimpleNamespace(name="D1")
    with torch.no_grad():
        writer.weight.zero_()
        writer.bias.zero_()
    reader = SimpleNamespace(width=2, buckets=17, orders=(2, 3), layer_index=1,
                             model=SimpleNamespace(config=SimpleNamespace(_commit_hash=None)))
    method = SimpleNamespace(supports_outer_training=True, writer=writer, reader=reader)
    old = {"width": 2, "buckets": 17, "orders": [2, 3], "layer_index": 1, "hidden_dim": 8,
           "model": "fixture/tiny-base", "revision": "revision-a", "smoke": True, "seed": 17}
    payload = {"step": 3, "config": old,
               "writer_state_dict": {"weight": torch.full((2, 2), .125), "bias": torch.full((2,), .25)}}
    if legacy:
        payload["protocol"] = LEGACY_PROTOCOL
    else:
        payload.update(format=FORMAT, format_version=VERSION, protocol="delta_ngram_pilot_v1")
        old["package_config"] = deepcopy(values)
    return method, payload, SimpleNamespace(values=values)


@pytest.mark.parametrize("empty_config", [False, True])
def test_current_checkpoint_cannot_fall_back_to_legacy_identity(tmp_path, empty_config):
    method, payload, config = fixture_objects()
    if empty_config:
        payload["config"] = {}
    else:
        payload["config"].pop("package_config")
    path = tmp_path / "incomplete-current.pt"
    torch.save(payload, path)
    # Neither the file loader nor direct restoration may bypass current identity.
    with pytest.raises(ValueError, match="identity"):
        load_checkpoint(path, allow_legacy=True)
    with pytest.raises(ValueError, match="identity"):
        restore_writer(method, payload, config)
    assert not method.writer.weight.any() and not method.writer.bias.any()


def test_valid_legacy_checkpoint_requires_explicit_opt_in_and_restores(tmp_path):
    method, payload, config = fixture_objects(legacy=True)
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="unsupported writer checkpoint"):
        load_checkpoint(path)
    loaded = load_checkpoint(path, allow_legacy=True)
    assert loaded["protocol"] == LEGACY_PROTOCOL and loaded["step"] == 3
    restore_writer(method, loaded, config)
    for name, expected in payload["writer_state_dict"].items():
        assert torch.equal(method.writer.state_dict()[name], expected)


@pytest.mark.parametrize("mismatch", ["smoke", "seed", "model", "right_window"])
def test_legacy_restore_rejects_incompatible_identity_before_installing_weights(mismatch):
    method, payload, config = fixture_objects(legacy=True)
    if mismatch == "smoke":
        config.values["model"]["smoke"] = False
    elif mismatch == "seed":
        config.values["seed"] = 18
    elif mismatch == "model":
        config.values["model"]["name_or_path"] = "fixture/different-base"
    else:
        config.values["writer"]["right_window"] = 2
    with pytest.raises(ValueError, match="historical checkpoint"):
        restore_writer(method, payload, config)
    assert not method.writer.weight.any() and not method.writer.bias.any()

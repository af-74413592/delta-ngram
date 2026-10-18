"""Golden tensors captured from the executed 4090 artifact, evaluated on CPU.

The fixture contains no executable legacy code. Its provenance records the
exact original source hash and the CPU torch version used to produce it.
"""
from pathlib import Path

import torch

from delta_ngram.state import SparseMemoryState
from delta_ngram.writer import ForwardWriter


def test_executed_seed17_initialization_rng_keys_two_writes_and_gradients():
    fixture = torch.load(Path(__file__).parent / "fixtures/executed_d1_seed17.pt", weights_only=True)
    torch.set_rng_state(fixture["rng_before"])
    writer = ForwardWriter(**fixture["kwargs"])
    assert torch.equal(torch.get_rng_state(), fixture["rng_after"])
    actual = writer.state_dict()
    assert list(actual) == list(fixture["writer_state"])
    for key, expected in fixture["writer_state"].items():
        assert torch.equal(actual[key], expected), key
    state, snapshots = SparseMemoryState.empty(writer.width), []
    for inputs, expected in zip(fixture["inputs"], fixture["states"], strict=True):
        state = writer(*inputs, state)
        snapshots.append(state)
        for field in ("rows", "values", "counts"):
            assert torch.equal(getattr(state, field), expected[field]), field
    sum(state.values.square().sum() for state in snapshots).backward()
    for name, parameter in writer.named_parameters():
        assert torch.equal(parameter.grad, fixture["gradients"][name]), name

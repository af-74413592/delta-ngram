"""The diagnostic driver cannot accidentally turn into a training entrypoint."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
import qwen38_native_memory_forward_probe as probe


def test_guard_blocks_optimizer_and_backward_then_restores_torch_apis():
    original = (torch.optim.Optimizer.__init__, torch.Tensor.backward,
                torch.autograd.backward, torch.autograd.grad)
    parameter = torch.nn.Parameter(torch.ones(2))
    with probe.forward_only_guard() as blocked:
        assert not torch.is_grad_enabled()
        with pytest.raises(RuntimeError, match="optimizer_construction"):
            torch.optim.AdamW([parameter])
        with pytest.raises(RuntimeError, match="backward"):
            parameter.sum().backward()
        with pytest.raises(RuntimeError, match="grad"):
            torch.autograd.grad(parameter.sum(), parameter)
    assert blocked == ["optimizer_construction", "backward", "grad"]
    assert original == (torch.optim.Optimizer.__init__, torch.Tensor.backward,
                        torch.autograd.backward, torch.autograd.grad)
    assert torch.is_grad_enabled()


def test_select_group_uses_cloned_sampler_and_restores_even_after_plan_failure(monkeypatch):
    class Sampler:
        def __init__(self):
            self.value = {"cursor": 7, "total_loss_tokens": 100}
        def state_dict(self):
            return copy.deepcopy(self.value)
        def load_state_dict(self, value):
            self.value = copy.deepcopy(value)
        def next_batch(self, size):
            self.value["cursor"] += 1
            return [SimpleNamespace(document_id=str(index), group_id="test", source="books",
                                    blocks=8, start=0, loss_tokens=1) for index in range(size)]
    sampler = Sampler()
    before = sampler.state_dict()
    data = SimpleNamespace(sampler=lambda seed: sampler,
                           frames=lambda item: [{"query": None, "evidence": {"input_ids": [1, 2]}}])
    monkeypatch.setattr(probe, "validate_frame_plan", lambda *args: None)
    group, plans, metadata = probe.select_probe_group(data, before)
    assert len(group) == len(plans) == 12
    assert metadata["CE_targets"] == 0 and metadata["source_sampler_restored"]
    assert sampler.state_dict() == before
    def bad(*args):
        raise ValueError("bad frame")
    monkeypatch.setattr(probe, "validate_frame_plan", bad)
    with pytest.raises(ValueError, match="bad frame"):
        probe.select_probe_group(data, before)
    assert sampler.state_dict() == before


def test_protocol_is_forward_only_and_flags_g_without_training_autostart():
    from qwen38_native_memory_forward_schedule import forward_probe_protocol
    protocol = probe.diagnostic_protocol()
    assert protocol["CE_targets"] == protocol["backward_calls"] == protocol["optimizer_updates"] == 0
    assert protocol["relative_epsilons"] == [1e-3, 1e-2]
    assert protocol["matched_native_calls_per_ready_wave"] == 6
    assert protocol["G_p99_above_one"].startswith("risk_flag_only")
    assert protocol["automatic_training_start"] is False
    assert forward_probe_protocol(probe.execution_protocol()) == probe.execution_protocol()
    assert protocol["execution_protocol_sha256"] == probe.canonical_hash(probe.execution_protocol())

"""A forward-only numerical probe must retain analytic forward grad dispatch."""
from pathlib import Path
import sys
import types

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
from qwen38_native_runtime_preflight import schedule_query


@pytest.fixture
def native_schedule(monkeypatch):
    observed = []

    def forward_backward_pipelining_without_interleaving(**kwargs):
        output, loss_fn = kwargs["forward_step_func"](kwargs["data_iterator"], kwargs["model"][0])
        loss, metric = loss_fn(output)
        observed.append((torch.is_grad_enabled(), output.requires_grad, kwargs["forward_only"]))
        if not kwargs["forward_only"]:
            loss.backward()
        return [metric]

    core = types.ModuleType("megatron.core")
    core.parallel_state = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1, get_data_parallel_world_size=lambda: 1,
        is_pipeline_last_stage=lambda: True, get_pipeline_model_parallel_last_rank=lambda: 0,
        get_pipeline_model_parallel_group=lambda: None)
    pipeline = types.ModuleType("megatron.core.pipeline_parallel")
    pipeline.get_forward_backward_func = lambda: forward_backward_pipelining_without_interleaving
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.pipeline_parallel", pipeline)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *args, **kwargs: None)
    return observed


@pytest.mark.parametrize("backward,grad_enabled,expected", [
    (False, None, (False, False, True)),
    (False, True, (True, True, True)),
    (True, None, (True, True, False)),
])
def test_grad_mode_and_backward_schedule_are_separate(native_schedule, backward, grad_enabled, expected):
    leaf = torch.tensor([[2., 3.]], requires_grad=True)
    batch = {"input_ids": torch.ones((1, 2), dtype=torch.long),
             "labels": torch.tensor([[10, 11]])}
    result = schedule_query(
        [lambda **kwargs: leaf.square()], batch, valid_mask=torch.ones((1, 2), dtype=torch.bool),
        owner_ids=torch.tensor([0]), backward=backward, query_grad_enabled=grad_enabled,
        group_total_target_tokens=2)
    assert native_schedule == [expected]
    assert result["normalized_loss"] == 6.5
    if backward:
        torch.testing.assert_close(leaf.grad, leaf.detach())
    else:
        assert leaf.grad is None


def test_backward_cannot_silently_run_under_no_grad(native_schedule):
    with pytest.raises(ValueError, match="grad-enabled"):
        schedule_query([], {}, valid_mask=None, owner_ids=None, backward=True,
                       query_grad_enabled=False, group_total_target_tokens=2)

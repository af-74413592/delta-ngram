import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
from qwen38_native_mode_trace import compare_trace_phases, native_module_trace


class Branch(torch.nn.Module):
    def forward(self, x):
        return x + (.25 if torch.is_grad_enabled() else .5)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.before = torch.nn.Identity()
        self.branch = Branch()
        self.after = torch.nn.Identity()

    def forward(self, x):
        return self.after(self.branch(self.before(x)))


def test_first_same_input_changed_output_is_localized_and_graph_untouched():
    model = Model()
    x = torch.tensor([1., 2.], requires_grad=True)
    with native_module_trace(model, module_names=("before", "branch", "after")) as trace:
        trace.set_phase("grad")
        y = model(x)
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
        trace.set_phase("no_grad")
        with torch.no_grad():
            model(x.detach())
        trace.set_phase(None)
        model(x)
    report = trace.report([("grad", "no_grad")])
    first = report["comparisons"][0]["first_numeric_divergence"]
    assert first["path"] == "branch" and first["event"] == "output"
    assert first["same_numeric_inputs_to_this_call"] is True
    assert report["hooks_removed"]
    assert len(trace.records) == 12
    for module in (model.before, model.branch, model.after):
        assert not module._forward_hooks and not module._forward_pre_hooks


def test_requires_grad_metadata_is_not_a_numeric_divergence():
    model = torch.nn.Sequential(torch.nn.Identity())
    with native_module_trace(model, module_names=("0",)) as trace:
        trace.set_phase("grad")
        model(torch.ones(2, requires_grad=True))
        trace.set_phase("no_grad")
        with torch.no_grad():
            model(torch.ones(2))
    result = compare_trace_phases(trace.records, "grad", "no_grad")
    assert result["all_numeric_events_equal"]


def test_exception_removes_hooks_and_absent_paths_fail_closed():
    model = Model()
    with pytest.raises(RuntimeError, match="diagnostic"):
        with native_module_trace(model, module_names=("branch",)):
            raise RuntimeError("diagnostic")
    assert not model.branch._forward_hooks
    with pytest.raises(ValueError, match="absent"):
        with native_module_trace(model, module_names=("branch", "missing")):
            pass

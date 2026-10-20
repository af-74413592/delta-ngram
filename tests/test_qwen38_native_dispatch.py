import functools
import hashlib
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
from qwen38_native_dispatch import HC_HELPERS, HC_MODULE, _hc_dispatch


def fake_module(tmp_path):
    source = (
        "def _mix_elementwise(x, hc_count):\n"
        "    return x / hc_count\n"
        "def _mix_and_reduce(x, y, hc_count, hidden_size):\n"
        "    return x * y\n")
    path = tmp_path / "hc.py"
    path.write_text(source, encoding="utf8")
    module = types.ModuleType(HC_MODULE)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    for name in HC_HELPERS:
        raw = getattr(module, name)

        @functools.wraps(raw)
        def wrapper(*args, _raw=raw, **kwargs):
            return _raw(*args, **kwargs)

        wrapper._torchdynamo_orig_callable = raw
        setattr(module, name, wrapper)
    return module, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("unwrap", [True, False])
def test_dispatch_preserves_gradients_and_restores_functions(tmp_path, unwrap):
    module, sha = fake_module(tmp_path)
    before = {name: getattr(module, name) for name in HC_HELPERS}
    x = torch.tensor([2., 4.], requires_grad=True)
    with _hc_dispatch(module, unwrap=unwrap, trace=True,
                      expected_source_sha256=sha, torch=torch) as audit:
        audit.set_phase("grad")
        result = module._mix_elementwise(x, 2)
        result.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([.5, .5]))
        audit.set_phase("no_grad")
        with torch.no_grad():
            module._mix_elementwise(x, 2)
        audit.set_phase(None)
        module._mix_elementwise(x, 2)
    assert all(getattr(module, name) is old for name, old in before.items())
    assert audit.report()["restored_original_function_objects"]
    assert len(audit.records) == 2
    assert audit.records[0]["inputs"] == audit.records[1]["inputs"]
    assert audit.records[0]["output"] == audit.records[1]["output"]
    assert audit.records[0]["grad_enabled"] is True
    assert audit.records[1]["grad_enabled"] is False


def test_exception_restores_and_wrong_source_fails_closed(tmp_path):
    module, sha = fake_module(tmp_path)
    before = {name: getattr(module, name) for name in HC_HELPERS}
    with pytest.raises(RuntimeError, match="inside"):
        with _hc_dispatch(module, unwrap=True, trace=False,
                          expected_source_sha256=sha, torch=torch):
            raise RuntimeError("inside")
    assert all(getattr(module, name) is old for name, old in before.items())
    with pytest.raises(ValueError, match="source differs"):
        with _hc_dispatch(module, unwrap=True, trace=False,
                          expected_source_sha256="0" * 64, torch=torch):
            pass
    assert all(getattr(module, name) is old for name, old in before.items())

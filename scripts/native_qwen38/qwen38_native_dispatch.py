"""Scoped dispatch audit for two pinned HC torch.compile helpers only."""
from __future__ import annotations

from contextlib import contextmanager
import functools
import hashlib
import importlib
import inspect
from pathlib import Path


HC_MODULE = "mcore_bridge.model.modules.hyper_connection_gated"
HC_SOURCE_SHA256 = "2d28e50c1570cbf6a20cbe361910f6507b348e82b9c41e05ed15b188679f7b48"
HC_COMMIT = "3a5ab71264ca0c98db9eab7cc4de58aa57e065e3"
HC_HELPERS = ("_mix_elementwise", "_mix_and_reduce")


def _callable_name(value):
    return f"{value.__module__}.{value.__qualname__}"


def _value_signature(value, torch):
    if torch.is_tensor(value):
        with torch.no_grad():
            copy = value.detach().contiguous().cpu()
            return {"shape": list(copy.shape), "dtype": str(copy.dtype),
                    "bytes_sha256": hashlib.sha256(
                        copy.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value, (tuple, list)):
        return [_value_signature(item, torch) for item in value]
    if isinstance(value, dict):
        return {key: _value_signature(item, torch) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"HC trace received unsupported value {type(value).__name__}")


class HCDispatchAudit:
    """Set a phase only for calls that should be traced; no tensors are retained."""

    def __init__(self, identity, trace):
        self.identity = identity
        self.trace = bool(trace)
        self.phase = None
        self.records = []
        self.restored = False

    def set_phase(self, phase):
        if phase is not None and (not isinstance(phase, str) or not phase):
            raise ValueError("trace phase must be a nonempty label or None")
        self.phase = phase

    def report(self):
        return {"identity": dict(self.identity), "trace": list(self.records),
                "restored_original_function_objects": self.restored}


@contextmanager
def _hc_dispatch(module, *, unwrap, trace, expected_source_sha256, torch):
    """Testable internal primitive; public entry below fixes source identity."""
    source = Path(inspect.getfile(module))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != expected_source_sha256:
        raise ValueError("installed HC source differs from the pinned audited commit")
    originals, unwrapped, functions = {}, {}, []
    for name in HC_HELPERS:
        old = getattr(module, name, None)
        raw = getattr(old, "_torchdynamo_orig_callable", None)
        if (not callable(old) or not callable(raw) or old is raw
                or getattr(raw, "__module__", None) != HC_MODULE
                or getattr(raw, "__name__", None) != name):
            raise ValueError(f"{name} is not the expected torch.compile wrapper")
        originals[name], unwrapped[name] = old, raw
        functions.append({"attribute": name, "wrapper": _callable_name(old),
                          "original": _callable_name(raw),
                          "original_from": "_torchdynamo_orig_callable"})
    audit = HCDispatchAudit({
        "protocol": "qwen38_native_hc_dispatch_v1", "module": HC_MODULE,
        "fixed_bridge_commit": HC_COMMIT, "source_sha256": source_sha,
        "torch_version": str(torch.__version__),
        "policy": "unwrap_only_registered_hc_helpers" if unwrap else "native_compiled_hc_helpers",
        "functions": functions, "trace_enabled": bool(trace),
        "mathematical_source_changed": False,
    }, trace)

    def traced(name, target):
        @functools.wraps(target)
        def call(*args, **kwargs):
            phase = audit.phase
            input_signature = None
            if audit.trace and phase is not None:
                input_signature = {"args": _value_signature(args, torch),
                                   "kwargs": _value_signature(kwargs, torch)}
            # The original call executes in its caller's grad mode.
            result = target(*args, **kwargs)
            if input_signature is not None:
                audit.records.append({"phase": phase, "function": name,
                                      "phase_call_index": sum(
                                          row["phase"] == phase for row in audit.records),
                                      "grad_enabled": bool(torch.is_grad_enabled()),
                                      "inputs": input_signature,
                                      "output": _value_signature(result, torch)})
            return result
        return call

    try:
        for name in HC_HELPERS:
            target = unwrapped[name] if unwrap else originals[name]
            setattr(module, name, traced(name, target) if trace else target)
        yield audit
    finally:
        for name, value in originals.items():
            setattr(module, name, value)
        audit.restored = all(getattr(module, name) is value for name, value in originals.items())
        if not audit.restored:
            raise RuntimeError("failed to restore original HC compile wrappers")


@contextmanager
def native_hc_dispatch(*, unwrap=False, trace=False):
    """Optionally unwrap exactly two source-pinned HC functions, then restore.

    ``audit.set_phase('books:delta:grad')`` records detached input/output hashes.
    Set phase=None for repeat or unneeded calls. Caller records this dispatch
    identity separately from any pre-existing gradient acceptance receipt.
    """
    import torch
    module = importlib.import_module(HC_MODULE)
    with _hc_dispatch(module, unwrap=unwrap, trace=trace,
                      expected_source_sha256=HC_SOURCE_SHA256, torch=torch) as audit:
        yield audit

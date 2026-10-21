"""Source-pinned A800 HC dispatch: this bridge implements HC eagerly inline.

The H200 bridge has two torch.compile helpers. The separately audited A800
bridge has neither helper; its module methods already run the eager equations.
Keep these policies separate so an A800 check cannot relax the H200 contract.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path


HC_MODULE = "mcore_bridge.model.modules.hyper_connection_gated"
HC_SOURCE_SHA256 = "8fc7182e2144982c89f65f784366cca4197892c846b765a964726404ee74a706"
HC_COMMIT = "4f2a95c9548bcd588b6f7763074debcdbf6b2d69"
HC_METHODS = (("Qwen4ExpTextGroupedRMSNorm", "_norm"),
              ("Qwen4ExpTextGroupedRMSNorm", "forward"),
              ("Qwen4ExpTextGatedResidual", "forward"))


@contextmanager
def a800_hc_dispatch(module, *, torch):
    """Validate the actual native eager methods without replacing any callable."""
    source = Path(inspect.getfile(module)).resolve()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if module.__name__ != HC_MODULE or source_sha != HC_SOURCE_SHA256:
        raise ValueError("A800 HC module/source differs from its audited native eager implementation")
    if any(hasattr(module, name) for name in ("_mix_elementwise", "_mix_and_reduce")):
        raise ValueError("A800 native eager HC unexpectedly exposes compiled helper attributes")
    originals, methods = [], []
    for class_name, name in HC_METHODS:
        cls = getattr(module, class_name, None)
        function = getattr(cls, name, None)
        if (not inspect.isfunction(function) or function.__module__ != HC_MODULE
                or function.__qualname__ != f"{class_name}.{name}"
                or Path(inspect.getfile(function)).resolve() != source
                or hasattr(function, "_torchdynamo_orig_callable")
                or hasattr(function, "__wrapped__")):
            raise ValueError(f"A800 HC {class_name}.{name} is not the audited native eager callable")
        originals.append((cls, name, function))
        methods.append(f"{HC_MODULE}.{class_name}.{name}")
    identity = {
        "protocol": "qwen38_a800_native_hc_dispatch_v1", "module": HC_MODULE,
        "fixed_bridge_commit": HC_COMMIT, "source_sha256": source_sha,
        "torch_version": str(torch.__version__), "policy": "native_eager_inline_hc",
        "methods": methods, "compiled_helpers_absent": True,
        "callables_replaced": False, "mathematical_source_changed": False,
        "H200_numerical_equivalence_claimed": False,
    }
    try:
        yield identity
    finally:
        if any(getattr(cls, name) is not function for cls, name, function in originals):
            raise RuntimeError("A800 native eager HC callables changed during the session")

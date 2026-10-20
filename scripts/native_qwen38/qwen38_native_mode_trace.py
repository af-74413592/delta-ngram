"""Read-only module boundary traces for native grad/no-grad dispatch diagnosis.

Hooks hash complete tensor bytes and retain no activation or graph. They do
not change forward functions, flags, inputs, outputs, or math. A trace is a
diagnostic: synchronization overhead makes it unsuitable for timing claims.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import hashlib
import inspect
from pathlib import Path
import re


_BOUNDARY_NAMES = re.compile(
    r"linear|norm|attention|gateddelta|gdn|moe|router|expert|qwen4explayer|"
    r"transformerlayer|embedding|gatedresidual|ple|conv", re.IGNORECASE)


def _type_name(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _tensor_signature(value, torch):
    # Requires-grad is separate from the numeric signature. A difference in
    # graph construction must not be mistaken for a numerical divergence.
    metadata = {"shape": list(value.shape), "dtype": str(value.dtype),
                "stride": list(value.stride()), "requires_grad": bool(value.requires_grad),
                "grad_fn": None if value.grad_fn is None else type(value.grad_fn).__name__}
    try:
        with torch.no_grad():
            owned = value.detach().contiguous().cpu()
            metadata["bytes_sha256"] = hashlib.sha256(
                owned.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    except Exception as error:
        metadata["hash_error"] = f"{type(error).__name__}: {error}"
    return metadata


def _tree_signature(value, torch):
    if torch.is_tensor(value):
        return {"tensor": _tensor_signature(value, torch)}
    if isinstance(value, (tuple, list)):
        return {"sequence": [_tree_signature(item, torch) for item in value]}
    if isinstance(value, dict):
        return {"mapping": {str(key): _tree_signature(item, torch) for key, item in value.items()}}
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"scalar": value}
    # Packed sequence/config/inference objects are reported by type. Tensor
    # values supplied directly to the module are all included above.
    return {"opaque_type": _type_name(value)}


def numeric_signature(value):
    """Discard autograd/stride metadata while keeping every captured byte hash."""
    if isinstance(value, list):
        return [numeric_signature(item) for item in value]
    if isinstance(value, dict):
        return {key: numeric_signature(item) for key, item in value.items()
                if key not in ("requires_grad", "grad_fn", "stride")}
    return value


def _has_hash_error(value):
    if isinstance(value, dict):
        return "hash_error" in value or any(_has_hash_error(x) for x in value.values())
    if isinstance(value, list):
        return any(_has_hash_error(x) for x in value)
    return False


def compare_trace_phases(records, reference_phase, observed_phase):
    """Locate first changed numeric boundary, without choosing a later match."""
    left = [row for row in records if row["phase"] == reference_phase]
    right = [row for row in records if row["phase"] == observed_phase]
    keys = ("path", "event", "module_call_index")
    result = {"reference_phase": reference_phase, "observed_phase": observed_phase,
              "reference_events": len(left), "observed_events": len(right),
              "trace_complete": not any(_has_hash_error(row["value"]) for row in left + right),
              "call_sequence_equal": len(left) == len(right),
              "first_numeric_divergence": None, "changed_numeric_events": 0}
    inputs_equal = {}
    for index, (a, b) in enumerate(zip(left, right)):
        if any(a[key] != b[key] for key in keys):
            result["call_sequence_equal"] = False
            result["first_call_sequence_divergence"] = {
                "event_index": index, "reference": {key: a[key] for key in keys},
                "observed": {key: b[key] for key in keys}}
            break
        same = numeric_signature(a["value"]) == numeric_signature(b["value"])
        key = (a["path"], a["module_call_index"])
        if a["event"] == "input":
            inputs_equal[key] = same
        if not same:
            result["changed_numeric_events"] += 1
            if result["first_numeric_divergence"] is None:
                result["first_numeric_divergence"] = {
                    "event_index": index, "path": a["path"], "event": a["event"],
                    "module_call_index": a["module_call_index"], "class": a["class"],
                    "layer_number": a["layer_number"],
                    "same_numeric_inputs_to_this_call": inputs_equal.get(key),
                    "reference_value": a["value"], "observed_value": b["value"]}
    result["all_numeric_events_equal"] = bool(
        left and result["trace_complete"] and result["call_sequence_equal"]
        and result["first_numeric_divergence"] is None)
    return result


class NativeModeTrace:
    def __init__(self, inventory, torch):
        self.inventory = inventory
        self.torch = torch
        self.records = []
        self.phase = None
        self._calls = {}
        self._stacks = {}
        self.restored = False

    def set_phase(self, phase):
        if phase is not None and (not isinstance(phase, str) or not phase):
            raise ValueError("module trace phase must be nonempty or None")
        if any(self._stacks.values()):
            raise RuntimeError("cannot change trace phase during a selected module call")
        self.phase = phase
        self._calls = {}

    def _append(self, item, phase, index, event, value):
        self.records.append({
            "phase": phase, "path": item["path"], "class": item["class"],
            "layer_number": item["layer_number"], "event": event,
            "module_call_index": index, "grad_enabled": bool(self.torch.is_grad_enabled()),
            "value": _tree_signature(value, self.torch)})

    def _pre_hook(self, item):
        def pre(module, args, kwargs):
            phase = self.phase
            if phase is None:
                return
            path = item["path"]
            index = self._calls.get(path, 0)
            self._calls[path] = index + 1
            self._stacks.setdefault(path, []).append((phase, index))
            self._append(item, phase, index, "input", {"args": args, "kwargs": kwargs})
        return pre

    def _post_hook(self, item):
        def post(module, args, kwargs, result):
            stack = self._stacks.get(item["path"], [])
            if not stack:
                return
            phase, index = stack.pop()
            self._append(item, phase, index, "output", result)
        return post

    def report(self, comparisons=()):
        return {
            "protocol": "qwen38_native_module_dispatch_trace_v1",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "torch_version": str(self.torch.__version__), "inventory": self.inventory,
            "records": self.records,
            "comparisons": [compare_trace_phases(self.records, a, b) for a, b in comparisons],
            "hooks_removed": self.restored,
            "limits": [
                "First changed boundary localizes a region; it does not by itself identify a kernel defect.",
                "Only selected module calls are traced; direct .forward or functional calls may lie between boundaries.",
                "Opaque non-tensor objects are represented by type; direct tensor arguments are hashed fully.",
                "Tracing synchronizes device copies. Repeat parity/performance checks with tracing disabled."]}


@contextmanager
def native_module_trace(model, *, module_names=None, layer_numbers=None):
    """Trace selected native boundaries without changing their forward calls.

    Defaults select norm/linear/attention/GDN/MoE/HC/PLE/embedding/conv modules.
    ``module_names`` optionally supplies exact named_modules paths. Otherwise
    ``layer_numbers`` limits decoder-layer descendants by native layer_number,
    while retaining matching non-layer boundaries. Tensor bytes are complete.
    """
    import torch
    if module_names is not None:
        module_names = set(module_names)
    if layer_numbers is not None:
        layer_numbers = set(layer_numbers)
    modules = dict(model.named_modules())
    layer_roots = {
        path: int(module.layer_number) for path, module in modules.items()
        if type(getattr(module, "layer_number", None)) is int
        and ("Layer" in type(module).__name__ or "GDN" in type(module).__name__)
    }
    inventory, selected, source_cache = [], [], {}
    for path, module in modules.items():
        if not path:
            continue
        ancestors = [(root, number) for root, number in layer_roots.items()
                     if path == root or path.startswith(root + ".")]
        number = max(ancestors, key=lambda item: len(item[0]))[1] if ancestors else None
        if module_names is not None:
            if path not in module_names:
                continue
        elif (not _BOUNDARY_NAMES.search(type(module).__name__)
              or (layer_numbers is not None and number is not None and number not in layer_numbers)
              or path.endswith("output_layer")):
            continue
        cls = type(module)
        class_name = f"{cls.__module__}.{cls.__qualname__}"
        if class_name not in source_cache:
            try:
                source_cache[class_name] = hashlib.sha256(
                    Path(inspect.getfile(cls)).read_bytes()).hexdigest()
            except (OSError, TypeError):
                source_cache[class_name] = None
        item = {"path": path, "class": class_name, "layer_number": number,
                "class_source_sha256": source_cache[class_name]}
        inventory.append(item)
        selected.append((item, module))
    if not selected:
        raise ValueError("no native module boundaries matched the registered trace scope")
    if module_names is not None and {x["path"] for x in inventory} != module_names:
        raise ValueError("one or more exact requested trace paths are absent")
    audit = NativeModeTrace(inventory, torch)
    handles = []
    method_bindings = []
    try:
        for item, module in selected:
            handles.append(module.register_forward_pre_hook(audit._pre_hook(item), with_kwargs=True))
            handles.append(module.register_forward_hook(
                audit._post_hook(item), with_kwargs=True, always_call=True))
            dispatcher = getattr(module, "token_dispatcher", None)
            if dispatcher is not None:
                # MoE communication/permutation helpers are plain Python
                # objects, so nn.Module hooks alone do not expose them.
                for method in ("dispatch_preprocess", "token_dispatch", "dispatch_postprocess",
                               "combine_preprocess", "token_combine", "combine_postprocess",
                               "token_permutation", "token_unpermutation"):
                    target = getattr(dispatcher, method, None)
                    if not callable(target):
                        continue
                    method_item = {**item, "path": item["path"] + ".token_dispatcher." + method,
                                   "class": _type_name(dispatcher) + "." + method}
                    try:
                        method_item["class_source_sha256"] = hashlib.sha256(
                            Path(inspect.getfile(type(dispatcher))).read_bytes()).hexdigest()
                    except (OSError, TypeError):
                        method_item["class_source_sha256"] = None
                    inventory.append(method_item)
                    had_instance = method in dispatcher.__dict__
                    instance_value = dispatcher.__dict__.get(method)

                    def wrap_method(bound, boundary):
                        @functools.wraps(bound)
                        def traced(*args, **kwargs):
                            if audit.phase is None:
                                return bound(*args, **kwargs)
                            path, phase = boundary["path"], audit.phase
                            index = audit._calls.get(path, 0)
                            audit._calls[path] = index + 1
                            audit._append(boundary, phase, index, "input",
                                          {"args": args, "kwargs": kwargs})
                            result = bound(*args, **kwargs)
                            audit._append(boundary, phase, index, "output", result)
                            return result
                        return traced

                    method_bindings.append((dispatcher, method, had_instance, instance_value))
                    setattr(dispatcher, method, wrap_method(target, method_item))
        yield audit
    finally:
        for handle in handles:
            handle.remove()
        for obj, name, had_instance, value in reversed(method_bindings):
            if had_instance:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
        audit.restored = True

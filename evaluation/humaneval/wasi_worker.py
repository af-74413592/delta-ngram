"""Trusted WASI host. Candidate Python is interpreted only inside WebAssembly.

No directory, environment, stdin, socket, or host-Python function is granted to
the guest. The process protocol is private to evaluate_humaneval_wasi.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

WASM_SHA256 = "e5dc5a398b07b54ea8fdb503bf68fb583d533f10ec3f930963e02b9505f7a763"
WASMTIME_VERSION = "48.0.0"
FUEL = 10_000_000_000
MEMORY_BYTES = 512 * 1024 * 1024
OUTPUT_BYTES = 64 * 1024


class BoundedOutput:
    def __init__(self):
        self.value = bytearray()
        self.overflow = False

    def write(self, data):
        remaining = OUTPUT_BYTES - len(self.value)
        self.value.extend(data[:remaining])
        self.overflow |= len(data) > remaining
        return len(data)  # Discard excess without unbounded host allocation.


def guest_script(program, nonce):
    # Python exec below is source text for the WASM guest, never a host call.
    # This wrapper reports Python failures separately from host/runtime failure.
    return f'''import os as _os, json as _json, random as _random
_random.seed(0)
def _run():
    _write, _dumps = _os.write, _json.dumps
    try:
        exec({program!r}, {{"__name__": "__main__"}})
        _status = {{"passed": True, "result": "passed"}}
    except BaseException as _exc:
        _status = {{"passed": False, "result": type(_exc).__name__}}
        if isinstance(_exc, (ImportError, NotImplementedError)):
            _status = {{"passed": None, "result": "runtime_compatibility_error",
                       "exception_type": type(_exc).__name__,
                       "module": getattr(_exc, "name", None)}}
    _write(2, ({nonce!r} + _dumps(_status) + "\\n").encode())
_run()
'''


class GuestRunner:
    def __init__(self, runtime):
        import wasmtime
        if importlib.metadata.version("wasmtime") != WASMTIME_VERSION:
            raise RuntimeError(f"requires wasmtime=={WASMTIME_VERSION}")
        if hashlib.sha256(runtime.read_bytes()).hexdigest() != WASM_SHA256:
            raise ValueError("CPython WASM checksum mismatch")
        self.wt = wasmtime
        config = wasmtime.Config()
        config.consume_fuel = True
        self.engine = wasmtime.Engine(config)
        self.module = wasmtime.Module.from_file(self.engine, str(runtime))
        if any(item.module != "wasi_snapshot_preview1" for item in self.module.imports):
            raise RuntimeError("unexpected interpreter host import")
        self.linker = wasmtime.Linker(self.engine)
        self.linker.define_wasi()

    def execute(self, program, nonce):
        started = time.perf_counter()
        output, errors = BoundedOutput(), BoundedOutput()
        store = self.wt.Store(self.engine)
        store.set_limits(memory_size=MEMORY_BYTES, table_elements=100000,
                         instances=1, tables=1, memories=1)
        store.set_fuel(FUEL)
        wasi = self.wt.WasiConfig()
        wasi.argv = ["python", "-I", "-B", "-c", guest_script(program, nonce)]
        # Deliberately no inherit_* and no preopen_dir. Default stdin is EOF;
        # guest environment is empty and there are no filesystem capabilities.
        wasi.stdout_custom = output.write
        wasi.stderr_custom = errors.write
        store.set_wasi(wasi)
        result = None
        try:
            instance = self.linker.instantiate(store, self.module)
            instance.exports(store)["_start"](store)
        except self.wt.ExitTrap as exc:
            result = {"passed": False, "result": "guest_exit", "exit_code": exc.code}
        except self.wt.Trap as exc:
            result = {"passed": False, "result": "fuel_exhausted" if "fuel" in str(exc).lower() else "guest_trap",
                      "diagnostic": str(exc)[:2000]}
        finally:
            remaining = store.get_fuel()
            store.close()
        if result is None:
            lines = errors.value.decode("utf-8", errors="replace").splitlines()
            verdicts = [line[len(nonce):] for line in lines if line.startswith(nonce)]
            if len(verdicts) == 1:
                result = json.loads(verdicts[0])
                if (type(result.get("passed")) is not bool
                        and not (result.get("passed") is None and result.get("result") == "runtime_compatibility_error")):
                    raise ValueError("malformed guest verdict")
            else:
                result = {"passed": False, "result": "guest_missing_verdict"}
        if output.overflow or errors.overflow:
            result = {"passed": False, "result": "output_limit_exceeded"}
        result.update(elapsed_seconds=time.perf_counter()-started,
                      fuel_consumed=FUEL-remaining,
                      stdout_bytes_captured=len(output.value), stderr_bytes_captured=len(errors.value))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    runner = GuestRunner(args.runtime)
    print(json.dumps({"ready": True, "wasmtime": WASMTIME_VERSION, "wasm_sha256": WASM_SHA256}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            result = runner.execute(request["program"], request["nonce"])
            status = "runtime_compatibility_error" if result.get("passed") is None else "completed"
            print(json.dumps({"task_id": request["task_id"], "status": status, **result}), flush=True)
        except Exception as exc:
            print(json.dumps({"task_id": request["task_id"], "status": "infrastructure_error",
                              "passed": None, "error": f"{type(exc).__name__}: {exc}"}), flush=True)


if __name__ == "__main__":
    main()

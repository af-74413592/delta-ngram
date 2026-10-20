"""Run the bounded native TP structural proof with the shared frozen loader."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from qwen38_native_runtime_preflight import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "verification", "delta-repo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    options.tokens = 16
    if not options.execute:
        print(json.dumps({
            "mode": "plan_only", "model_loaded": False, "gpu_initialized": False,
            "torchrun_ranks": 4, "reader_forwards": 1, "reader_backwards": 1,
            "reader_input_tokens": 16, "loss_target_tokens": 11,
            "linear_operator_ce_targets": 0, "optimizer_steps": 0,
            "scope": "native SP/lookup adjoints; not full BF16 CE finite differences",
            "output_must_be_new": str(options.output),
        }, indent=2))
        return
    options.output = options.output.resolve()
    marker = options.output / "PREPARED.json"
    if int(os.environ.get("RANK", "-1")) == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        write_json(marker, {"protocol": "native_tp_structure_only_v1"})
    else:
        deadline = time.monotonic() + 300
        while not marker.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError("rank0 did not prepare structural proof output")
            time.sleep(.1)
    from qwen38_native_frozen_session import run_frozen_session
    from qwen38_native_tp_structure import run_tp_structure_probe

    def probe(session):
        result = run_tp_structure_probe(
            models=session.models, adapter=session.adapter, layout=session.layout,
            device=session.device, tokenizer=session.tokenizer, output=options.output / "tp_structure")
        session.assert_frozen()
        if int(os.environ["RANK"]) == 0:
            write_json(options.output / "COMPLETE.json", {
                "protocol": "native_tp_structure_only_v1", "identity": session.identity,
                "versions": session.versions, "result": result, "optimizer_steps": 0})
        return result

    run_frozen_session(options, probe)


if __name__ == "__main__":
    main()

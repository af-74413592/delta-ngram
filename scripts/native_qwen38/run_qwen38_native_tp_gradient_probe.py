"""Run only the bounded TP finite-difference probe using the shared frozen loader."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from qwen38_native_runtime_preflight import write_json
from qwen38_native_tp_gradient_probe import STEPS


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
            "torchrun_ranks": 4, "tokens_per_forward": 16, "loss_targets_per_forward": 11,
            "planned_forwards": 20, "maximum_loss_target_tokens": 220, "backwards": 1,
            "steps": list(STEPS), "optimizer_steps": 0,
            "same_grad_mode_for_nonempty_delta": True, "output_must_be_new": str(options.output),
        }, indent=2))
        return
    options.output = options.output.resolve()
    marker = options.output / "PREPARED.json"
    if int(os.environ.get("RANK", "-1")) == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        write_json(marker, {"protocol": "native_tp_fd_only_v2", "output": str(options.output)})
    else:
        deadline = time.monotonic() + 300
        while not marker.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError("rank0 did not prepare probe output")
            time.sleep(.1)
    from qwen38_native_frozen_session import run_frozen_session
    from qwen38_native_tp_gradient_probe import run_tp_gradient_probe

    def probe(session):
        result = run_tp_gradient_probe(
            models=session.models, adapter=session.adapter, layout=session.layout,
            device=session.device, tokenizer=session.tokenizer, output=options.output / "tp_fd")
        session.assert_frozen()
        if int(os.environ["RANK"]) == 0:
            write_json(options.output / "COMPLETE.json", {
                "protocol": "native_tp_fd_only_v2", "identity": session.identity,
                "versions": session.versions, "result": result, "optimizer_steps": 0})
        return result

    run_frozen_session(options, probe)


if __name__ == "__main__":
    main()

"""One H200 reader load: bounded integration gate, then two independent 35k chains."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time

import qwen38_h200_seven_task_forward_writer as chain


def main():
    parser = chain.parser()
    parser.add_argument("--main-writer", type=Path, required=True)
    parser.add_argument("--full-writer", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    options = parser.parse_args()
    chain.setup_cl(options.cl_repo)
    from qwen38_h200_writer_timing import SOURCES, WRITERS
    chain.require(all(chain.digest(options.delta_repo / name) == sha for name, sha in SOURCES.items()),
                  "original H200 reader source changed")
    manifest, data = chain.load_encoded(options)
    chain.require(options.execute and options.max_writes == 0 and options.max_tasks == 7
                  and options.query_limit == 0 and options.evidence_batch_size == 1
                  and options.capture_layers, "formal two-chain policy changed")
    for name, sha in manifest["tokenizer_files"].items():
        chain.require(chain.digest(options.model_path / name) == sha, "tokenizer differs")
    prepared = []
    for method, checkpoint in (("ordinary", options.main_writer), ("delta", options.full_writer)):
        arm = copy.copy(options)
        arm.method, arm.writer_checkpoint, arm.writer_sha256 = method, checkpoint, WRITERS[method]["archive"]
        arm.output, arm.resume = options.output / method, None
        chain.writer_source(arm)
        prepared.append(arm)
    import torch
    # Import the callback dependencies on CPU before paying for a model load.
    from delta_ngram.alpha_memory import LazyDecayState
    from delta_ngram.qwen38_native import make_qwen38_writer, write_native_block
    from delta_ngram.qwen38_training import writer_evidence_features
    from delta_ngram.qwen38_probe import tensor_state_hash
    from delta_ngram.qwen38_precision import Qwen38SplitPLEPrecision
    from qwen38_native_h200_forward import native_forward, native_dispatch, sync_written_rows
    from qwen38_native_memory_schedule import preserve_encoding_rng
    from qwen38_mcore_retention_baseline import LayerCapture, reference_batch
    from qwen38_native_dispatch import native_hc_dispatch
    from qwen38_native_router import native_router_dispatch
    from delta_ngram.batching import BatchMemoryState
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        writer, alpha = make_qwen38_writer(alpha=1.)
        hidden = torch.linspace(-1., 1., 3 * 2560).reshape(1, 3, 2560)
        rows = torch.arange(1, 49, dtype=torch.long).reshape(1, 3, 16)
        mask = torch.ones((1, 3), dtype=torch.bool)
        state = LazyDecayState.from_batch(BatchMemoryState.empty(1, 160, 1000))
        expected = write_native_block(writer, state, hidden, rows, mask, alpha)
        actual, statistics = chain.write_with_statistics(writer, state, hidden, rows, mask, alpha)
        chain.require(chain.state_fingerprint(actual) == chain.state_fingerprint(expected)
                      and statistics["event_count"] == 48,
                      "statistics observer changed native D1")
    chain.require(not torch.cuda.is_initialized(), "CPU registration initialized CUDA")
    registration = {"status": "CPU_preflight_passed", "methods": ["ordinary", "delta"],
                    "D1_per_chain": 35000, "total_D1": 70000, "independent_zero_states": True,
                    "encoded_manifest_sha256": options.encoded_manifest_sha256,
                    "query_count": len(data["queries"]), "generation_calls": 0,
                    "integrated_gate": {"D1_per_chain": 2, "query_limit_per_domain": 1},
                    "entry_sha256": chain.digest(__file__)}
    if options.validate_only:
        print(json.dumps(registration), flush=True)
        return
    rank = int(os.environ.get("RANK", "-1"))
    chain.require(rank >= 0, "torchrun required")
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=options.resume_existing)
        chain.write_json(options.output / "campaign_registration.json", registration)
    else:
        deadline = time.monotonic() + 120
        while not options.output.is_dir():
            chain.require(time.monotonic() < deadline, "rank0 output timeout")
            time.sleep(.1)
    options.tokens = 8192
    from qwen38_native_h200_session import run_frozen_session

    def run(session):
        import torch.distributed as dist
        control = dist.new_group(backend="gloo")
        results = {}
        gate_marker = options.output / "INTEGRATION_GATE.json"
        if not gate_marker.exists():
            for arm in prepared:
                smoke = copy.copy(arm)
                smoke.output = options.output / ("integration_gate_" + arm.method)
                smoke.query_limit, smoke.max_writes, smoke.max_tasks = 1, 2, 1
                if rank == 0 and smoke.output.exists():
                    chain.archive_existing(smoke.output, options.output / ("gate_audit_" + str(time.time_ns())))
                dist.barrier(group=control)
                results[arm.method] = chain.execute(smoke, session=session)
                chain.require(results[arm.method]["status"] == "bounded_stop"
                              and results[arm.method]["cursor"]["writes_completed"] == 2,
                              "bounded integration did not complete exactly two D1 updates")
            if rank == 0:
                chain.write_json(gate_marker, {"status": "passed", "results": results,
                                             "entry_sha256": chain.digest(__file__)})
            dist.barrier(group=control)
        else:
            previous = json.loads(gate_marker.read_bytes())
            chain.require(options.resume_existing and previous["status"] == "passed"
                          and previous["entry_sha256"] == chain.digest(__file__), "integration gate identity differs")
        for arm in prepared:
            if arm.output.exists():
                chain.require(options.resume_existing, "formal output already exists")
                latest = json.loads((arm.output / "LATEST.json").read_bytes())
                arm.resume = Path(latest["path"])
                chain.require(chain.digest(arm.resume) == latest["sha256"], "resume checkpoint hash differs")
            if rank == 0:
                chain.write_json(options.output / "ACTIVE_CHAIN.json", {"method": arm.method, "status": "running",
                    "resume": str(arm.resume) if arm.resume else None, "started_unix": time.time()})
            results[arm.method] = chain.execute(arm, session=session)
            chain.require(results[arm.method]["status"] == "completed"
                          and results[arm.method]["cursor"]["writes_completed"] == 35000,
                          "formal chain did not complete its registered population")
        if rank == 0:
            chain.write_json(options.output / "COMPLETE.json", {"status": "completed", "results": results,
                "D1_total": 70000, "metrics": ["K", "ki", "NLL"], "generation_calls": 0})
        return results

    run_frozen_session(options, run)


if __name__ == "__main__":
    main()

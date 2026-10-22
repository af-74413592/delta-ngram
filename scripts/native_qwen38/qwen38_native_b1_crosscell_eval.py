"""Standalone frozen-writer B1 cross-cell DEV64; never constructs an optimizer.

Prepare on CPU, then execute the sealed registration with four torchrun ranks.
One reader load runs main100M/base, main100M/memory and full100M/base. Existing
full100M/memory reports remain separate authenticated reference observations.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from qwen38_native_norm_probe import canonical_hash, file_info, require

PROTOCOL = "qwen38_native_b1_crosscell_dev64_v1"
# The archive hash is deliberately explicit, independently of writer tensor hashes.
MAIN_SHA = "523b951487cee0ec726077db0104a8fc929f4629bdb4ca46604295a4b78fbfe3"
FULL_SHA = "51d5977845e9930f48732c73ae2503511d8976b1201f139a2be46bfab9f57308"
CELLS = (("main100m_base_b1", "main", "base"),
         ("main100m_memory_b1", "main", "memory"),
         ("full100m_base_b1", "full", "base"))
NATIVE_KEYS = ("native", "HC", "router", "precision")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def source_manifest(repo):
    repo = Path(repo).resolve()
    files = sorted([*(repo / "scripts/native_qwen38").glob("*.py"),
                    *(repo / "src/delta_ngram").rglob("*.py")])
    require(bool(files), "empty code inventory")
    return {path.relative_to(repo).as_posix(): file_info(path) for path in files}


def read_writer(path, kind):
    import torch
    from delta_ngram.p2_initialization import writer_tensor_sha256
    from delta_ngram.qwen38_probe import tensor_state_hash
    info = file_info(path)
    require(info["sha256"] == {"main": MAIN_SHA, "full": FULL_SHA}[kind],
            "writer checkpoint archive differs: " + kind)
    source = torch.load(path, map_location="cpu", weights_only=False)
    if kind == "main":
        require(source["protocol"] == "qwen38_native_staged_alpha100_checkpoint_v1",
                "mainline checkpoint protocol differs")
        saved = source["arms"]["alpha100_rmsnorm"]
        require(saved["alpha"] == 1. and saved["optimizer_step"] == 4172,
                "mainline endpoint/alpha differs")
    else:
        require(source["protocol"] == "qwen38_native_memory_fork_checkpoint_v1"
                and source["policy"]["branch"] == "full"
                and source["completed_groups"] == 835
                and source["completed_targets"] == 20015113,
                "full-memory endpoint differs")
        saved = source["branch_state"]
        require(saved["model_optimizer_step"] == 4172,
                "full-memory optimizer endpoint differs")
    require(saved["completed_loss_targets"] == 99997072
            and tensor_state_hash(saved["writer"]) == saved["writer_sha256"],
            "writer tensors or terminal target count differ")
    require(file_info(path) == info, "checkpoint archive changed during reading")
    result = {"path": str(Path(path).resolve()), **info,
              "writer_tensor_state_hash": tensor_state_hash(saved["writer"]),
              "writer_report_hash": writer_tensor_sha256(saved["writer"]),
              "optimizer_step": 4172, "completed_loss_targets": 99997072,
              "alpha": 1., "writer_tensors": len(saved["writer"])}
    return saved["writer"], saved["rng_by_rank"], result


def frozen_writer(state, device="cpu"):
    import torch
    from delta_ngram.qwen38_native import make_qwen38_writer
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(170080)
        writer, alpha = make_qwen38_writer(alpha=1.)
        writer.load_state_dict(state, strict=True)
    return writer.to(device).float().eval(), alpha.to(device).eval()


def prepare(options):
    import torch
    from qwen38_native_dev64_prepare import load_native_dev64_cache
    from qwen38_native_migrate_eval_only import state_digest
    require(not torch.cuda.is_initialized(), "registration must be CPU-only")
    require(not options.output.exists(), "registration output must be new")
    prepared = load_native_dev64_cache(options.dev64_plan)
    sources = {}
    full_rng = None
    for kind, path in (("main", options.main_checkpoint), ("full", options.full_checkpoint)):
        state, rng, sources[kind] = read_writer(path, kind)
        writer, alpha = frozen_writer(state)
        require(alpha.initial_alpha == 1., "fixed alpha differs")
        if kind == "full":
            full_rng = rng
        del writer, alpha, state
    reference = json.loads(options.reference_runtime.read_bytes())
    native = {key: reference[key] for key in NATIVE_KEYS}
    require(native["native"]["topology"] == {"tp": 1, "pp": 1, "ep": 4,
                "etp": 1, "dp": 4, "cp": 1}, "reference native topology differs")
    codes = source_manifest(options.delta_repo)
    manifest = {
        "protocol": PROTOCOL, "created_utc": datetime.now(timezone.utc).isoformat(),
        "delta_repo": str(options.delta_repo.resolve()), "code": codes,
        "code_sha256": canonical_hash(codes), "checkpoints": sources,
        "model_path": str(options.model_path.resolve()),
        "verification": {"path": str(options.verification.resolve()), **file_info(options.verification)},
        "dev64_plan": str(options.dev64_plan.resolve()), "DEV64_identity": prepared["identity"],
        "reference_runtime": {"path": str(options.reference_runtime.resolve()), **file_info(options.reference_runtime)},
        "expected_reader_runtime": native,
        "common_rng": {"source": "full100m terminal checkpoint rng_by_rank",
                       "sha256": state_digest(full_rng), "reset_before_each_cell": True},
        "cells": [{"id": key, "writer": writer, "evidence": evidence,
                   "weights_label": "mainline" if writer == "main" else "memory",
                   "evidence_mode": evidence, "evidence_local_batch": 1,
                   "query_batch_limit": 4}
                  for key, writer, evidence in CELLS],
        "protocol_contract": {"evidence_local_batch": 1, "evidence_padded_sequence": 640,
            "query_batch_limit": 4, "topology": native["native"]["topology"],
            "fresh_memory": "empty independently for every cell/family/self/donor; native evaluator owns states",
            "base_hidden_cache": False, "query_memory_updates": False,
            "direction_permutations": "read views only; writer state never permuted",
            "expected_ce_targets_per_cell": prepared["plan"]["evaluation"]["ce_target_accesses"],
            "optimizer_created": False, "backward_calls": 0,
            "existing_full_memory": "external two-run reference; not rerun or overwritten"},
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    options.output.mkdir(parents=True, exist_ok=False)
    write_json(options.output / "registration.json", manifest)
    write_json(options.output / "REGISTRATION_COMPLETE.json", {
        "protocol": PROTOCOL, "registration": file_info(options.output / "registration.json"),
        "cuda_initialized": torch.cuda.is_initialized(), "zero_model_forwards": True})
    return manifest


def verify_registration(root):
    from qwen38_native_dev64_prepare import load_native_dev64_cache
    root = Path(root)
    seal = json.loads((root / "REGISTRATION_COMPLETE.json").read_bytes())
    require(file_info(root / "registration.json") == seal["registration"], "registration seal differs")
    manifest = json.loads((root / "registration.json").read_bytes())
    require(manifest["protocol"] == PROTOCOL, "registration protocol differs")
    require(source_manifest(manifest["delta_repo"]) == manifest["code"], "sealed source files changed")
    require(load_native_dev64_cache(manifest["dev64_plan"])["identity"] == manifest["DEV64_identity"],
            "DEV64 inputs changed")
    for name in ("verification", "reference_runtime"):
        item = manifest[name]
        require(file_info(item["path"]) == {key: item[key] for key in ("sha256", "bytes")},
                name + " changed")
    return manifest


def execute(options):
    import torch
    from qwen38_native_ep4_session import run_ep4_session
    from qwen38_native_migrate_eval_only import state_digest
    rank = int(os.environ.get("RANK", "-1"))
    require(0 <= rank < 4 and int(os.environ.get("WORLD_SIZE", "0")) == 4,
            "execute requires torchrun world4")
    manifest = verify_registration(options.registration)
    if rank == 0:
        require(not options.output.exists(), "execute output must be a fresh directory")
    states, common_rng = {}, None
    for kind in ("main", "full"):
        states[kind], rng, info = read_writer(manifest["checkpoints"][kind]["path"], kind)
        require(info == manifest["checkpoints"][kind], "checkpoint identity changed")
        if kind == "full":
            common_rng = rng
    require(state_digest(common_rng) == manifest["common_rng"]["sha256"], "common RNG differs")
    marker = options.output / "registration.json"
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        write_json(marker, manifest)
    else:
        deadline = time.monotonic() + 300
        while not marker.exists():
            require(time.monotonic() < deadline, "rank0 did not create output registration")
            time.sleep(.1)
    options.delta_repo = Path(manifest["delta_repo"])
    options.model_path = Path(manifest["model_path"])
    options.verification = Path(manifest["verification"]["path"])
    options.tokens = 640
    ledger = {"protocol": PROTOCOL, "evaluation_ce_targets": 0,
              "encoding_calls": 0, "encoding_real_input_tokens": 0,
              "encoding_padded_input_tokens": 0, "cells": {},
              "optimizer_steps": 0, "backward_calls": 0, "refunds": False}
    def save_ledger():
        if rank == 0:
            write_json(options.output / "budget_ledger.json", ledger)
    save_ledger()

    def run(session):
        import torch.distributed as dist
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        from delta_ngram.p2_initialization import writer_tensor_sha256
        from delta_ngram.qwen38_precision import Qwen38SplitPLEPrecision
        from qwen38_native_dispatch import native_hc_dispatch
        from qwen38_native_router import native_router_dispatch
        from qwen38_native_ep4_session import runtime_identity
        from qwen38_native_train_batched import restore_rng_state
        from qwen38_native_b1_dev64_eval import B1EvidenceWriter
        from qwen38_native_dev64_eval import evaluate_dev64
        with ExitStack() as scope:
            hc = scope.enter_context(native_hc_dispatch(unwrap=True))
            router = scope.enter_context(native_router_dispatch(session.raw))
            precision = scope.enter_context(Qwen38SplitPLEPrecision(session.adapter))
            runtime = {"native": runtime_identity(session), "HC": hc.identity,
                       "router": router["identity"], "precision": precision.identity()}
            require(runtime == manifest["expected_reader_runtime"],
                    "native reader/source/topology/HC/router/precision differs from fourth cell")
            if rank == 0:
                write_json(options.output / "runtime_identity.json", runtime)
            receipts = []
            for cell in manifest["cells"]:
                key = cell["id"]
                if options.cell != "all" and key != options.cell:
                    continue
                writer, alpha = frozen_writer(states[cell["writer"]], session.device)
                checkpoint = manifest["checkpoints"][cell["writer"]]
                require(writer_tensor_sha256(writer.state_dict()) == checkpoint["writer_report_hash"],
                        "loaded GPU writer differs from checkpoint")
                restore_rng_state(next(row for row in common_rng if row["rank"] == rank),
                                  rank=rank, device=session.device,
                                  tensor_parallel_tracker=get_cuda_rng_tracker())
                ledger["cells"][key] = {"evaluation_ce_targets": 0, "encoding_calls": 0,
                                        "encoding_real_input_tokens": 0, "encoding_padded_input_tokens": 0}
                cost = ledger["cells"][key]
                def charge_ce(count):
                    require(type(count) is int and count > 0, "invalid CE charge")
                    cost["evaluation_ce_targets"] += count
                    ledger["evaluation_ce_targets"] += count
                    save_ledger()
                def charge_encoding(row):
                    require(row["ce_targets"] == 0 and row["native_calls"] == 1,
                            "invalid evidence charge")
                    for key2, value in (("encoding_calls", 1),
                        ("encoding_real_input_tokens", row["real_input_tokens"]),
                        ("encoding_padded_input_tokens", row["padded_input_tokens"])):
                        ledger[key2] += value
                        cost[key2] += value
                    save_ledger()
                output = options.output / key
                report = evaluate_dev64(session, writer, alpha,
                    plan_path=manifest["dev64_plan"], arm=key, milestone=99997072,
                    output=output, charge_targets=charge_ce,
                    evidence_writer=B1EvidenceWriter(cell["evidence"], charge_encoding),
                    checkpoint_metadata={"source_checkpoint": checkpoint, "cell": cell,
                        "registration_sha256": file_info(options.registration / "registration.json")["sha256"],
                        "common_rng": manifest["common_rng"], "optimizer_step": 4172,
                        "completed_loss_targets": 99997072, "runtime_identity_sha256": canonical_hash(runtime)})
                require(cost["evaluation_ce_targets"] == report["evaluation_target_accesses"]
                        == manifest["protocol_contract"]["expected_ce_targets_per_cell"], "cell CE cost differs")
                require(writer_tensor_sha256(writer.state_dict()) == checkpoint["writer_report_hash"],
                        "evaluation changed writer")
                receipt = {"cell": cell, "complete": True, "writer_unchanged": True,
                           "report": {"path": str(output / "report.json"), **file_info(output / "report.json")},
                           "budget": cost.copy(), "writer_report_hash": report["writer_sha256"],
                           "actual_evidence_batch_limit": 1, "query_batch_limit": 4}
                if rank == 0:
                    rows = [json.loads(line) for line in (output / "memory_batches.jsonl").read_text().splitlines()]
                    evidence = [row for row in rows if row["kind"] == "evidence"]
                    require(evidence and all(row["native_batch_size"] == 1 for row in evidence),
                            "actual evidence batch log violated B1")
                    receipt["actual_evidence_batches"] = len(evidence)
                    write_json(options.output / (key + "-receipt.json"), receipt)
                receipts.append(receipt)
                del writer, alpha
                session.assert_frozen()
                dist.barrier()
            require(receipts, "no selected cell ran")
            verify_registration(options.registration)
            for kind in ("main", "full"):
                item = manifest["checkpoints"][kind]
                require(file_info(item["path"])["sha256"] == item["sha256"], "source checkpoint mutated")
            if rank == 0:
                write_json(options.output / "COMPLETE.json", {"protocol": PROTOCOL, "complete": True,
                    "cells": receipts, "budget": ledger, "optimizer_created": False,
                    "selection": options.cell, "reader_loads": 1})
            dist.barrier()
    try:
        run_ep4_session(options, run)
    except BaseException as error:
        if rank == 0:
            save_ledger()
            write_json(options.output / "FAILURE.json", {"protocol": PROTOCOL,
                "error_type": type(error).__name__, "message": str(error), "budget": ledger,
                "automatic_retry": False, "refunds": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    preparation = sub.add_parser("prepare")
    for name in ("main-checkpoint", "full-checkpoint", "reference-runtime", "dev64-plan",
                 "delta-repo", "model-path", "verification", "output"):
        preparation.add_argument("--" + name, type=Path, required=True)
    execution = sub.add_parser("execute")
    for name in ("registration", "output"):
        execution.add_argument("--" + name, type=Path, required=True)
    execution.add_argument("--cell", choices=["all", *(row[0] for row in CELLS)], default="all")
    options = parser.parse_args()
    if options.action == "prepare":
        sys.path.insert(0, str(options.delta_repo.resolve() / "src"))
        manifest = prepare(options)
        print(json.dumps({"protocol": PROTOCOL, "prepared": True,
            "code_sha256": manifest["code_sha256"], "checkpoints": manifest["checkpoints"],
            "cuda_initialized": manifest["cuda_initialized"]}, sort_keys=True))
    else:
        manifest = json.loads((options.registration / "registration.json").read_bytes())
        sys.path.insert(0, str(Path(manifest["delta_repo"]) / "src"))
        execute(options)


if __name__ == "__main__":
    main()

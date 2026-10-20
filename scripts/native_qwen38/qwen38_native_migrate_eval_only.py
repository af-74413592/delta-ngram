"""Rebind a pending DEV64 checkpoint after one explicitly recorded evaluator edit.

This CPU-only operation preserves every training-state field. The unchanged
training runner still performs its normal exact code and native runtime checks.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from qwen38_native_norm_probe import canonical_hash, file_info, require
from qwen38_native_train_staged_alternating import (
    CHECKPOINT_PROTOCOL, code_manifest, write_json)

EVALUATOR = "scripts/native_qwen38/qwen38_native_dev64_eval.py"
REBOUND_KEYS = frozenset(("code_manifest_sha256", "runtime_identity_sha256"))


def state_digest(value):
    """Hash tensor/NumPy bytes and typed structure, including RNG state."""
    import numpy as np
    import torch
    hasher = hashlib.sha256()

    def feed(data):
        hasher.update(len(data).to_bytes(8, "big"))
        hasher.update(data)

    def visit(item):
        feed(f"{type(item).__module__}.{type(item).__qualname__}".encode())
        if torch.is_tensor(item):
            require(item.device.type == "cpu", "migration state must remain on CPU")
            feed(repr((str(item.dtype), tuple(item.shape), tuple(item.stride()))).encode())
            feed(item.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            feed(repr((item.dtype.str, item.shape, item.strides)).encode())
            feed(item.tobytes())
        elif isinstance(item, dict):
            feed(str(len(item)).encode())
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            feed(str(len(item)).encode())
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, int, float, bool, bytes, np.generic)):
            feed(repr(item).encode())
        else:
            raise TypeError(f"unsupported checkpoint value: {type(item)!r}")

    visit(value)
    return hasher.hexdigest()


def derive(source, source_manifest, source_runtime, new_codes, ledger, manifest_info):
    require(source["protocol"] == CHECKPOINT_PROTOCOL
            and source["trajectory_boundary"] is True
            and source["memory_reset_at_next_trajectory"] is True,
            "requires a complete alternating checkpoint")
    require(source["registration_sha256"] == manifest_info["sha256"],
            "source registration file differs")
    old_hash = canonical_hash(source_manifest["code"])
    require(old_hash == source_manifest["code_manifest_sha256"]
            == source["code_manifest_sha256"] == source_runtime["code_sha256"],
            "source code identity chain differs")
    require(canonical_hash(source_runtime) == source["runtime_identity_sha256"],
            "source runtime identity differs")
    require(set(new_codes) == set(source_manifest["code"]), "code manifest file set changed")
    changed = sorted(key for key in new_codes if new_codes[key] != source_manifest["code"][key])
    require(changed == [EVALUATOR], "only the DEV64 evaluator file may change")
    require(source["schedule_cursor"]["phase"] == "evaluation_pending",
            "evaluator migration requires a pending evaluation boundary")
    require(source["dev64_plan"] == source_manifest["dev64_plan"] == source_runtime["DEV64_plan"]
            and source["initial_writer"] == source_manifest["initial_writer"],
            "saved evaluation plan or initial writer identity differs")
    for key in ("total_optimizer_target_accesses", "total_diagnostic_target_accesses",
                "total_evaluation_target_accesses"):
        require(type(ledger[key]) is int and ledger[key] >= source["budget"].get(key, 0),
                f"latest ledger would refund {key}")
    require(ledger["total_optimizer_target_accesses"] <= source["shared_optimizer_target_cap"],
            "latest training ledger exceeds the shared cap")
    runtime = copy.deepcopy(source_runtime)
    runtime["code_sha256"] = canonical_hash(new_codes)
    result = copy.deepcopy(source)
    result["code_manifest_sha256"] = runtime["code_sha256"]
    result["runtime_identity_sha256"] = canonical_hash(runtime)
    preserved = {key: value for key, value in source.items() if key not in REBOUND_KEYS}
    require(state_digest(preserved) == state_digest(
        {key: value for key, value in result.items() if key not in REBOUND_KEYS}),
        "migration changed training or historical checkpoint state")
    return result, runtime, state_digest(preserved)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-checkpoint", "source-manifest", "source-runtime",
                 "source-budget-ledger", "delta-repo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    options = parser.parse_args()
    require(not options.output.exists(), "migration output must be new")
    inputs = {name: getattr(options, name) for name in
              ("source_checkpoint", "source_manifest", "source_runtime", "source_budget_ledger")}
    identities = {name: file_info(path) for name, path in inputs.items()}
    source = torch.load(options.source_checkpoint, map_location="cpu", weights_only=False)
    manifest = json.loads(options.source_manifest.read_bytes())
    old_runtime = json.loads(options.source_runtime.read_bytes())
    ledger = json.loads(options.source_budget_ledger.read_bytes())
    codes = code_manifest(options.delta_repo.resolve())
    result, runtime, preserved_digest = derive(
        source, manifest, old_runtime, codes, ledger, identities["source_manifest"])
    options.output.mkdir(parents=True)
    target = options.output / "joint-eval-migrated.pt"
    temporary = target.with_suffix(".pt.partial")
    torch.save(result, temporary)
    restored = torch.load(temporary, map_location="cpu", weights_only=False)
    require(state_digest(restored) == state_digest(result),
            "serialized migrated checkpoint differs")
    require(identities == {name: file_info(path) for name, path in inputs.items()},
            "source changed during migration")
    os.replace(temporary, target)
    write_json(options.output / "runtime_identity.json", runtime)
    write_json(options.output / "code_manifest.json", codes)
    write_json(options.output / "budget_ledger.json", ledger)
    receipt = {
        "protocol": "qwen38_eval_only_checkpoint_migration_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "user requested no_grad DEV64 repair and continuation",
        "source_files": {name: {"path": str(path.resolve()), **identities[name]}
                         for name, path in inputs.items()},
        "tool": {"path": str(Path(__file__).resolve()), **file_info(__file__)},
        "changed_code_files": [EVALUATOR],
        "old_evaluator": manifest["code"][EVALUATOR], "new_evaluator": codes[EVALUATOR],
        "changed_checkpoint_fields": sorted(REBOUND_KEYS),
        "preserved_state_sha256": preserved_digest,
        "source_checkpoint_unchanged": file_info(options.source_checkpoint) == identities["source_checkpoint"],
        "training_state_unchanged": True, "native_runtime_unchanged": True,
        "registration_sha256_retains_original_registration": True,
        "historical_source_fields_retained": True,
        "old_code_manifest_sha256": source["code_manifest_sha256"],
        "new_code_manifest_sha256": result["code_manifest_sha256"],
        "old_runtime_identity_sha256": source["runtime_identity_sha256"],
        "new_runtime_identity_sha256": result["runtime_identity_sha256"],
        "schedule_cursor": result["schedule_cursor"],
        "arms": {name: {key: value[key] for key in
                       ("optimizer_step", "completed_loss_targets", "writer_sha256", "optimizer_sha256")}
                 for name, value in result["arms"].items()},
        "latest_external_budget": ledger,
        "derived_checkpoint": {"path": str(target.resolve()), **file_info(target)},
        "zero_model_forwards": True, "zero_optimizer_steps": True,
    }
    write_json(options.output / "migration_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

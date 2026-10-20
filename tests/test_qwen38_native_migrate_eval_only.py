"""CPU-only evaluator migration preserves training state and fails closed."""
import copy
import importlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"
sys.path.insert(0, str(SCRIPTS))
migration = importlib.import_module("qwen38_native_migrate_eval_only")
TRAINER = "scripts/native_qwen38/qwen38_native_train_staged_alternating.py"


@pytest.fixture(autouse=True)
def forbid_cuda_initialization(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("migration tests must never initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


def assert_identical(actual, expected):
    """Independent typed/byte comparison, without the tool's state_digest."""
    assert type(actual) is type(expected)
    if torch.is_tensor(expected):
        assert actual.device == expected.device == torch.device("cpu")
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        actual_bytes = actual.detach().contiguous().reshape(-1).view(torch.uint8)
        expected_bytes = expected.detach().contiguous().reshape(-1).view(torch.uint8)
        assert torch.equal(actual_bytes, expected_bytes)
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.strides == expected.strides
        assert actual.tobytes() == expected.tobytes()
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_identical(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_identical(left, right)
    else:
        assert actual == expected


def arm_state(name, alpha, steps, targets):
    generator = torch.Generator(device="cpu").manual_seed(17)
    writer = {
        "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
        "target_scale": torch.tensor([0.25, 1.5], dtype=torch.float32),
        "bf16_buffer": torch.tensor([0.0, -0.0, 0.5], dtype=torch.bfloat16),
    }
    optimizer = {
        "state": {0: {
            "step": torch.tensor(float(steps)),
            "exp_avg": torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10,
            "exp_avg_sq": torch.ones((4, 3), dtype=torch.float32) / 100,
        }} if steps else {},
        "param_groups": [{
            "params": [0], "lr": 0.0003, "betas": (0.9, 0.999),
            "eps": 1e-8, "weight_decay": 0.0, "foreach": False, "fused": False,
        }],
    }
    return {
        "arm": name, "alpha": alpha, "optimizer_step": steps,
        "completed_loss_targets": targets, "writer": writer, "optimizer": optimizer,
        "writer_sha256": migration.state_digest(writer),
        "optimizer_sha256": migration.state_digest(optimizer),
        "sampler": {
            "seed": 17, "total_loss_tokens": targets, "cursor": 10008 if steps else 0,
            "python_rng": random.Random(17).getstate(), "draw_order": ("books", "code"),
        },
        "rng_by_rank": [{
            "rank": rank,
            "python": random.Random(17 + rank).getstate(),
            "numpy": np.random.RandomState(17 + rank).get_state(),
            "torch_cpu": generator.get_state().clone(),
            # Checkpoints store CUDA/TP RNG as CPU bytes; no CUDA API is used.
            "torch_cuda": generator.get_state().clone(),
            "tensor_parallel": {"model-parallel-rng": generator.get_state().clone()},
        } for rank in range(4)],
        "stability": {
            "completed_steps": steps, "completed_gradients": [0.25, 1.5] if steps else [],
            "gradient_spike_streak": 0, "beta_saturation_streak": 0,
            "hard_stop_gradient_spikes": False, "hard_stop_beta_saturation": False,
        },
    }


@pytest.fixture
def inputs(tmp_path):
    old_codes = {
        migration.EVALUATOR: {"bytes": 29963, "sha256": "a" * 64},
        TRAINER: {"bytes": 46643, "sha256": "b" * 64},
        "src/delta_ngram/operators.py": {"bytes": 1234, "sha256": "c" * 64},
    }
    manifest = {
        "protocol": "qwen38_native_staged_alternating_training_v1",
        "code": old_codes, "code_manifest_sha256": migration.canonical_hash(old_codes),
        "dev64_plan": {"bytes": 4096, "sha256": "d" * 64},
        "initial_writer": {"bytes": 1024, "sha256": "e" * 64},
    }
    runtime = {
        "protocol": manifest["protocol"], "code_sha256": manifest["code_manifest_sha256"],
        "DEV64_plan": copy.deepcopy(manifest["dev64_plan"]),
        "native": {"topology": {"tp": 1, "pp": 1, "ep": 4, "dp": 4}, "frozen": True},
        "HC": {"unchanged": True}, "router": {"topk_sorted": True},
        "precision": {"math": "D1_FP32"},
        "evidence_batch_limit": 12, "query_batch_limit": 4,
    }
    manifest_path = tmp_path / "source_manifest.json"
    migration.write_json(manifest_path, manifest)
    manifest_info = migration.file_info(manifest_path)
    source = {
        "protocol": migration.CHECKPOINT_PROTOCOL,
        "trajectory_boundary": True, "memory_reset_at_next_trajectory": True,
        "registration_sha256": manifest_info["sha256"],
        "code_manifest_sha256": manifest["code_manifest_sha256"],
        "runtime_identity_sha256": migration.canonical_hash(runtime),
        "dev64_plan": copy.deepcopy(manifest["dev64_plan"]),
        "initial_writer": copy.deepcopy(manifest["initial_writer"]),
        "schedule_cursor": {
            "stage_index": 0, "arm": "alpha100_rmsnorm", "milestone": 20000000,
            "phase": "evaluation_pending", "evaluation_attempt": 2, "completed_stages": [],
            "next_group_cost": 24780,
        },
        "arms": {
            "alpha100_rmsnorm": arm_state("alpha100_rmsnorm", 1.0, 834, 19981748),
            "alpha099_rmsnorm": arm_state("alpha099_rmsnorm", 0.99, 0, 0),
        },
        "budget": {
            "total_optimizer_target_accesses": 21450663,
            "total_diagnostic_target_accesses": 59620,
            "total_evaluation_target_accesses": 143341,
            "evaluation_target_accesses": 143341,
        },
        "shared_optimizer_target_cap": 100000000,
        "cumulative_budget_loss_targets": 21450663,
        "reader_saved": False, "resume_cli_supported": True,
        "source_resume": {"bytes": 123, "sha256": "f" * 64},
        "source_migration": "exact_alternating_resume",
        "source_checkpoint_role": "joint_training_continuation",
        "source_model_state_restored": True, "data_identity": {"corpus": "fixed"},
        "evidence_batch_limit": 12, "query_batch_limit": 4,
    }
    ledger = copy.deepcopy(source["budget"])
    ledger["total_evaluation_target_accesses"] = 286682
    ledger["evaluation_target_accesses"] = 286682
    new_codes = copy.deepcopy(old_codes)
    new_codes[migration.EVALUATOR] = {"bytes": 34676, "sha256": "9" * 64}
    return SimpleNamespace(
        source=source, manifest=manifest, runtime=runtime, new_codes=new_codes,
        ledger=ledger, manifest_info=manifest_info, manifest_path=manifest_path,
    )


def derive(inputs):
    return migration.derive(
        inputs.source, inputs.manifest, inputs.runtime, inputs.new_codes,
        inputs.ledger, inputs.manifest_info)


def preserved(source):
    return {key: value for key, value in source.items() if key not in migration.REBOUND_KEYS}


def test_derive_only_rebinds_two_hashes_and_does_not_mutate_inputs(inputs):
    original = copy.deepcopy(vars(inputs))
    result, runtime, digest = derive(inputs)
    assert result.keys() == inputs.source.keys()
    assert_identical(preserved(result), preserved(inputs.source))
    assert result["code_manifest_sha256"] == migration.canonical_hash(inputs.new_codes)
    assert result["runtime_identity_sha256"] == migration.canonical_hash(runtime)
    assert result["code_manifest_sha256"] != inputs.source["code_manifest_sha256"]
    assert result["runtime_identity_sha256"] != inputs.source["runtime_identity_sha256"]
    assert {key: value for key, value in runtime.items() if key != "code_sha256"} == {
        key: value for key, value in inputs.runtime.items() if key != "code_sha256"}
    assert digest == migration.state_digest(preserved(inputs.source))
    assert_identical(vars(inputs), original)
    # Derived tensors must not alias source storage in memory.
    result["arms"]["alpha100_rmsnorm"]["writer"]["weight"][0, 0] = -99
    assert inputs.source["arms"]["alpha100_rmsnorm"]["writer"]["weight"][0, 0] == 0


def test_cli_roundtrip_preserves_tensor_rng_and_separate_latest_ledger(inputs, tmp_path, monkeypatch, capsys):
    paths = {
        "source-checkpoint": tmp_path / "source.pt",
        "source-manifest": inputs.manifest_path,
        "source-runtime": tmp_path / "runtime.json",
        "source-budget-ledger": tmp_path / "ledger.json",
    }
    torch.save(inputs.source, paths["source-checkpoint"])
    migration.write_json(paths["source-runtime"], inputs.runtime)
    migration.write_json(paths["source-budget-ledger"], inputs.ledger)
    before = {name: path.read_bytes() for name, path in paths.items()}
    output = tmp_path / "migration"
    argv = ["migration"]
    for name, path in {**paths, "delta-repo": tmp_path, "output": output}.items():
        argv.extend(("--" + name, str(path)))
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(migration, "code_manifest", lambda repo: copy.deepcopy(inputs.new_codes))
    migration.main()
    capsys.readouterr()
    actual = torch.load(output / "joint-eval-migrated.pt", map_location="cpu", weights_only=False)
    assert_identical(preserved(actual), preserved(inputs.source))
    assert actual["schedule_cursor"]["evaluation_attempt"] == 2
    assert actual["budget"]["total_evaluation_target_accesses"] == 143341
    assert json.loads((output / "budget_ledger.json").read_bytes()) == inputs.ledger
    runtime = json.loads((output / "runtime_identity.json").read_bytes())
    assert actual["runtime_identity_sha256"] == migration.canonical_hash(runtime)
    receipt = json.loads((output / "migration_receipt.json").read_bytes())
    assert receipt["changed_code_files"] == [migration.EVALUATOR]
    assert set(receipt["changed_checkpoint_fields"]) == migration.REBOUND_KEYS
    assert receipt["preserved_state_sha256"] == migration.state_digest(preserved(actual))
    assert receipt["source_checkpoint_unchanged"] is True
    assert receipt["registration_sha256_retains_original_registration"] is True
    assert receipt["zero_model_forwards"] is True
    assert receipt["zero_optimizer_steps"] is True
    assert receipt["latest_external_budget"]["total_evaluation_target_accesses"] == 286682
    assert {name: path.read_bytes() for name, path in paths.items()} == before
    assert not (output / "joint-eval-migrated.pt.partial").exists()


def test_rejects_additional_training_file_change(inputs):
    inputs.new_codes[TRAINER]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="only the DEV64 evaluator"):
        derive(inputs)


@pytest.mark.parametrize("change", ["add", "remove", "no_edit"])
def test_rejects_manifest_file_set_or_missing_evaluator_change(inputs, change):
    if change == "add":
        inputs.new_codes["src/delta_ngram/unregistered.py"] = {"bytes": 1, "sha256": "0" * 64}
    elif change == "remove":
        del inputs.new_codes[TRAINER]
    else:
        inputs.new_codes = copy.deepcopy(inputs.manifest["code"])
    with pytest.raises(ValueError, match="file set changed|only the DEV64 evaluator"):
        derive(inputs)


@pytest.mark.parametrize("counter", [
    "total_optimizer_target_accesses", "total_diagnostic_target_accesses",
    "total_evaluation_target_accesses",
])
def test_rejects_any_refunded_target_accesses(inputs, counter):
    inputs.ledger[counter] = inputs.source["budget"][counter] - 1
    with pytest.raises(ValueError, match="latest ledger would refund " + counter):
        derive(inputs)


@pytest.mark.parametrize("damage", [
    "registration", "manifest_code", "checkpoint_code", "runtime_code",
    "runtime_content", "checkpoint_runtime", "plan", "initial_writer",
])
def test_rejects_source_identity_mismatch(inputs, damage):
    if damage == "registration":
        inputs.manifest_info["sha256"] = "0" * 64
    elif damage == "manifest_code":
        inputs.manifest["code"][TRAINER]["sha256"] = "0" * 64
    elif damage == "checkpoint_code":
        inputs.source["code_manifest_sha256"] = "0" * 64
    elif damage == "runtime_code":
        inputs.runtime["code_sha256"] = "0" * 64
    elif damage == "runtime_content":
        inputs.runtime["native"]["topology"]["tp"] = 2
    elif damage == "checkpoint_runtime":
        inputs.source["runtime_identity_sha256"] = "0" * 64
    elif damage == "plan":
        inputs.source["dev64_plan"]["sha256"] = "0" * 64
    else:
        inputs.source["initial_writer"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="registration file|identity"):
        derive(inputs)

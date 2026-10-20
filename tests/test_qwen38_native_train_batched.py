"""Continuation preserves optimizer progress, sampler order and charged accesses."""
import copy
import importlib
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

from delta_ngram.data.raw_text import RawWindowSampler, TrajectoryWindow
from delta_ngram.qwen38_probe import tensor_state_hash


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"
sys.path.insert(0, str(SCRIPTS))
runner = importlib.import_module("qwen38_native_train_batched")
BoundedStability = importlib.import_module("qwen38_native_bounded_checks").BoundedStability


def ledger_and_checkpoint():
    source = runner.OptimizationLedger(
        prior_optimizer=10000, prior_diagnostic=14564,
        prior_by_arm={"alpha100_rmsnorm": 10000, "alpha099_rmsnorm": 0})
    source.reserve_query("alpha100_rmsnorm", 4000, 5120)
    source.complete_group("alpha100_rmsnorm", 4000)
    checkpoint = {
        "protocol": "qwen38_native_main_checkpoint_v1",
        "arm": "alpha100_rmsnorm", "alpha": 1., "optimizer_step": 1,
        "completed_loss_targets": 4000, "writer_sha256": "1" * 64,
        "registration_sha256": "2" * 64, "code_manifest_sha256": "3" * 64,
        "runtime_identity_sha256": "4" * 64,
        "trajectory_boundary": True, "memory_reset_at_next_trajectory": True,
        "reader_saved": False, "sampler": {"total_loss_tokens": 4000},
        "stability": {"completed_steps": 1, "completed_gradients": [2.],
                      "gradient_spike_streak": 0, "beta_saturation_streak": 0,
                      "hard_stop_gradient_spikes": False, "hard_stop_beta_saturation": False},
        "rng_by_rank": [{"rank": rank, "python": None, "numpy": None, "torch_cpu": None,
                         "torch_cuda": None, "tensor_parallel": None} for rank in range(4)],
        "budget": source.report(),
    }
    source.reserve_query("alpha100_rmsnorm", 2000, 2560)  # Charged interrupted group.
    final = source.report()
    registration = {
        "resume": runner.resume_metadata(checkpoint),
        "budgets": {"prior_optimizer_target_accesses": 16000,
                    "prior_diagnostic_target_accesses": 14564,
                    "prior_optimizer_targets_by_arm": {"alpha100_rmsnorm": 16000, "alpha099_rmsnorm": 0}},
    }
    return checkpoint, final, registration


def test_resume_keeps_completed_progress_and_charges_replayed_partial_group_again():
    checkpoint, source, registration = ledger_and_checkpoint()
    receipt = runner.validate_resume(checkpoint, source, registration)
    assert receipt["restored_model_loss_targets"] == 4000
    assert receipt["charged_after_checkpoint_targets_by_arm"]["alpha100_rmsnorm"] == 2000
    resumed = runner.OptimizationLedger(
        prior_optimizer=registration["budgets"]["prior_optimizer_target_accesses"],
        prior_diagnostic=14564,
        prior_by_arm=registration["budgets"]["prior_optimizer_targets_by_arm"])
    resumed.reserve_query("alpha100_rmsnorm", 2000, 2560)
    resumed.complete_group("alpha100_rmsnorm", 2000)
    assert resumed.report()["total_optimizer_target_accesses"] == 18000
    assert checkpoint["completed_loss_targets"] + resumed.completed["alpha100_rmsnorm"] == 6000
    assert 2000000 - resumed.prior_by_arm["alpha100_rmsnorm"] - resumed.completed["alpha100_rmsnorm"] == 1982000


@pytest.mark.parametrize("damage", ["refund", "rng_rank", "sampler", "stability"])
def test_resume_rejects_lost_accounting_or_inconsistent_completed_state(damage):
    checkpoint, source, registration = ledger_and_checkpoint()
    if damage == "refund":
        registration["budgets"]["prior_optimizer_target_accesses"] -= 2000
    elif damage == "rng_rank":
        checkpoint["rng_by_rank"][3]["rank"] = 2
    elif damage == "sampler":
        checkpoint["sampler"]["total_loss_tokens"] += 512
    else:
        checkpoint["stability"]["completed_steps"] = 0
    with pytest.raises(ValueError):
        runner.validate_resume(checkpoint, source, registration)


def sampler():
    windows = [TrajectoryWindow(f"doc{i}", f"group{i}", "code", 0, tuple(range(1024)), 2)
               for i in range(7)]
    return RawWindowSampler({2: windows}, seed=17, length_weights={2: 1.},
                            source_weights={"code": 1.})


def step(writer, optimizer, vector):
    optimizer.zero_grad(set_to_none=True)
    writer(vector).square().sum().backward()
    optimizer.step()


def test_restored_adamw_and_sampler_produce_the_exact_next_update():
    torch.manual_seed(123)
    original = torch.nn.Linear(3, 2)
    original_optimizer = torch.optim.AdamW(original.parameters(), lr=.0003, weight_decay=0.,
                                           foreach=False, fused=False)
    original_sampler = sampler()
    original_sampler.next_batch(3)
    step(original, original_optimizer, torch.tensor([[.2, -.7, .3]]))
    original_stability = BoundedStability(hard_stop_gradient_spikes=False, hard_stop_beta_saturation=False)
    original_stability.observe_step(2., 0, 10)
    original_stability.complete_step(2.)
    checkpoint = {
        "writer": copy.deepcopy(original.state_dict()),
        "writer_sha256": tensor_state_hash(original.state_dict()),
        "optimizer": copy.deepcopy(original_optimizer.state_dict()),
        "sampler": original_sampler.state_dict(), "optimizer_step": 1,
        "completed_loss_targets": 1536,
        "stability": {"completed_steps": 1, "completed_gradients": [2.],
                      "gradient_spike_streak": 0, "beta_saturation_streak": 1},
    }
    restored = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.0003, weight_decay=0.,
                                           foreach=False, fused=False)
    restored_sampler = sampler()
    restored_stability = BoundedStability(hard_stop_gradient_spikes=False, hard_stop_beta_saturation=False)
    assert runner.restore_checkpoint_state(
        checkpoint, writer=restored, optimizer=restored_optimizer,
        sampler=restored_sampler, stability=restored_stability) == (1, 1536)
    assert list(restored_stability.completed_gradients) == [2.]
    assert restored_stability.beta_saturation_streak == 1
    assert restored_sampler.next_batch(12) == original_sampler.next_batch(12)
    vector = torch.tensor([[-.6, .9, .1]])
    step(original, original_optimizer, vector)
    step(restored, restored_optimizer, vector)
    assert tensor_state_hash(restored.state_dict()) == tensor_state_hash(original.state_dict())
    for left, right in zip(original_optimizer.state.values(), restored_optimizer.state.values()):
        assert all(torch.equal(left[key], right[key]) for key in left)


def test_restore_all_rng_streams_uses_checkpoint_state_without_seed_reset(monkeypatch):
    random.seed(818)
    np.random.seed(818)
    torch.manual_seed(818)
    value = {"rank": 2, "python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.tensor([8], dtype=torch.uint8),
             "tensor_parallel": {"model-parallel-rng": torch.tensor([9], dtype=torch.uint8)}}
    expected = random.random(), np.random.rand(), torch.rand(3)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    restored = {}
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: restored.update(cuda=(state, device)))
    class Tracker:
        def set_states(self, states):
            restored["tp"] = states
    runner.restore_rng_state(value, rank=2, device="cuda:2", tensor_parallel_tracker=Tracker())
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    assert torch.equal(restored["cuda"][0], value["torch_cuda"]) and restored["cuda"][1] == "cuda:2"
    assert torch.equal(restored["tp"]["model-parallel-rng"], value["tensor_parallel"]["model-parallel-rng"])

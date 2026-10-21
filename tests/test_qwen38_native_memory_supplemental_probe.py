import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
import qwen38_native_memory_supplemental_probe as probe
from qwen38_native_memory_supplemental_schedule import supplemental_probe_protocol


def _plans():
    query = {"input_ids": [1, 2, 3, 4], "encoder_valid_mask": [True] * 4,
             "write_mask": [False] * 4, "position_ids": list(range(4)),
             "labels": [-100, 3, 4, -100], "loss_target_tokens": 2}
    return [[{"evidence": {}, "query": None}] * (2 if i < 6 else 1)
            + [{"evidence": None, "query": copy.deepcopy(query)}] for i in range(12)]


def test_protocol_matches_core_and_explicitly_bounds_final_queries():
    assert supplemental_probe_protocol(probe.execution_protocol()) == probe.execution_protocol()
    policy = probe.diagnostic_protocol()
    assert policy["backward_calls"] == policy["optimizer_updates"] == 0
    assert policy["G_p99_above_one"] == "monitor_only_no_automatic_stop"
    assert policy["automatic_training_start"] is False
    registration = probe.register_final_queries(_plans())
    assert len(registration["records"]) == 6
    assert registration["CE_targets_per_condition"] == 12
    assert registration["CE_targets"] == 48


def test_query_registration_excludes_prior_queries_and_rejects_final_write():
    plans = _plans()
    plans[0][0]["query"] = {"unused": "never scored from the final state"}
    assert probe.register_final_queries(plans)["CE_targets"] == 48
    plans[0][-1]["evidence"] = {}
    with pytest.raises(ValueError, match="never be written"):
        probe.register_final_queries(plans)


def test_precharge_rejects_overrun_before_changing_charged_counts():
    registration = probe.register_final_queries(_plans())
    ledger = {"CE_targets": 0, "final_query_calls": 0, "final_query_real_input_tokens": 0,
              "final_query_padded_input_tokens": 0}
    probe.precharge_score(ledger, registration, kind="memory_ref", ce_targets=12,
                          real_input_tokens=24, padded_input_tokens=640)
    assert ledger["CE_targets"] == 12
    before = copy.deepcopy(ledger)
    with pytest.raises(ValueError, match="reservation exceeded"):
        probe.precharge_score(ledger, registration, kind="memory_ref", ce_targets=1,
                              real_input_tokens=4, padded_input_tokens=640)
    assert ledger == before
    for kind in registration["conditions"][1:]:
        probe.precharge_score(ledger, registration, kind=kind, ce_targets=12,
                              real_input_tokens=24, padded_input_tokens=640)
    assert ledger["CE_targets"] == registration["CE_targets"]


def test_supplemental_guard_allows_forward_but_blocks_training():
    parameter = torch.nn.Parameter(torch.ones(2))
    original = torch.autograd.grad
    with probe.forward_only_guard() as blocked:
        assert not (parameter * 2).requires_grad
        with pytest.raises(RuntimeError, match="optimizer_construction"):
            torch.optim.AdamW([parameter])
        with pytest.raises(RuntimeError, match="backward"):
            parameter.sum().backward()
    assert blocked == ["optimizer_construction", "backward"]
    assert torch.autograd.grad is original

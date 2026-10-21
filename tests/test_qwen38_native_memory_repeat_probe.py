"""CPU receipt/budget checks for standalone repeat-only diagnostics."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
import qwen38_native_memory_repeat_probe as probe
from qwen38_native_memory_repeat_schedule import FINAL_CONDITIONS, repeat_probe_protocol


def plans():
    writes = [1, 1, 1, 1, 7, 1, 15, 15, 1, 4, 7, 7]
    result = []
    for index, count in enumerate(writes):
        targets = 490 if index == 9 else 512
        query = {"input_ids": [1] * (targets + 1), "encoder_valid_mask": [True] * (targets + 1),
                 "write_mask": [False] * (targets + 1), "position_ids": list(range(targets + 1)),
                 "labels": [-100] + [2] * targets, "loss_target_tokens": targets}
        result.append([{"evidence": {}, "query": None}] * count + [{"evidence": None, "query": query}])
    return result


ASSIGNMENTS = [[6, 1, 5], [7, 2, 8], [4, 11, 3], [10, 9, 0]]


def test_exact_repeat_costs_3050_per_condition_and_32_encodings():
    assert repeat_probe_protocol(probe.execution_protocol()) == probe.execution_protocol()
    registration = probe.register_final_queries(plans())
    assert registration["CE_targets_per_condition"] == 3050 and registration["CE_targets"] == 12200
    assert registration["conditions"] == list(FINAL_CONDITIONS)
    assert [row["trajectory_index"] for row in registration["records"]] == [4, 6, 7, 9, 10, 11]
    encoding = probe.register_encoding_calls(plans(), ASSIGNMENTS)
    assert encoding["total_encoding_calls"] == 32 and encoding["H12_base_collective_calls"] == 2
    policy = probe.diagnostic_protocol()
    assert policy["state_perturbations_injected"] == 0 and not policy["training_gate"]
    assert not policy["new_stability_threshold"] and not policy["automatic_training_start"]


def test_prior_receipt_preserves_costs_and_requires_same_source_selection_and_queries(tmp_path):
    registration = probe.register_final_queries(plans())
    manifest = {"source80_checkpoint": {"bytes": 123, "sha256": "source"},
                "selection": {"assignments": ASSIGNMENTS}, "final_query_registration": registration}
    budget = {"CE_targets": 12200, "encoding_calls": 92, "backward_calls": 0, "optimizer_updates": 0,
              "encoding_real_input_tokens": 100, "encoding_padded_input_tokens": 200}
    report = {"protocol": "qwen38_native_memory_supplemental_probe_v2", "selection": manifest["selection"],
              "final_query_registration": registration, "budget": budget}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    complete = {"protocol": report["protocol"], "complete": True, "report": probe.file_info(report_path),
                "source80_checkpoint": manifest["source80_checkpoint"], "budget": budget, "CE_targets": 12200}
    complete_path = tmp_path / "COMPLETE.json"
    complete_path.write_text(json.dumps(complete), encoding="utf-8")
    before = (report_path.read_bytes(), complete_path.read_bytes())
    receipt = probe.prior_supplemental_receipt(tmp_path, manifest)
    assert receipt["budget"] == budget and receipt["preserved_without_refund"]
    assert before == (report_path.read_bytes(), complete_path.read_bytes())
    changed = copy.deepcopy(manifest)
    changed["source80_checkpoint"]["sha256"] = "different"
    with pytest.raises(ValueError, match="immutable source"):
        probe.prior_supplemental_receipt(tmp_path, changed)
    report_path.write_text(json.dumps({**report, "unexpected": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt"):
        probe.prior_supplemental_receipt(tmp_path, manifest)


def test_query_precharge_retains_accesses_and_rejects_extra_or_old_perturbed_label():
    registration = probe.register_final_queries(plans())
    ledger = {"CE_targets": 0, "final_query_calls": 0, "final_query_real_input_tokens": 0,
              "final_query_padded_input_tokens": 0}
    for condition in FINAL_CONDITIONS:
        probe.precharge_score(ledger, registration, kind=condition, ce_targets=3050,
                              real_input_tokens=3840, padded_input_tokens=7680)
    assert ledger["CE_targets"] == 12200
    before = copy.deepcopy(ledger)
    with pytest.raises(ValueError, match="reservation exceeded"):
        probe.precharge_score(ledger, registration, kind="memory_repeat", ce_targets=1,
                              real_input_tokens=640, padded_input_tokens=640)
    assert ledger == before
    with pytest.raises(ValueError, match="invalid global"):
        probe.precharge_score(ledger, registration, kind="memory_eps01", ce_targets=1,
                              real_input_tokens=640, padded_input_tokens=640)


def test_no_optimizer_or_backward_can_enter_repeat_diagnostic():
    parameter = torch.nn.Parameter(torch.ones(1))
    with probe.forward_only_guard():
        assert not torch.is_grad_enabled()
        with pytest.raises(RuntimeError, match="optimizer_construction"):
            torch.optim.AdamW([parameter])
        with pytest.raises(RuntimeError, match="backward"):
            parameter.backward()

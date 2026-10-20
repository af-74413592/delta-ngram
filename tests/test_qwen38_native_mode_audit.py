"""Probability-space checks for the native training/inference audit."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
from qwen38_native_mode_audit import (
    compare_logits, ple_effect, precision_modes, quantization_visibility, query_modes,
    runtime_identity_fields,
)


def test_constant_logit_offset_has_zero_probability_difference():
    logits = torch.tensor([[1., 2., 3.], [-2., 0., 2.]])
    result = compare_logits(logits, logits + 4.)
    assert not result["bitwise_equal"]
    assert result["difference_max_abs"] == 4.
    assert result["top1_changed_tokens"] == 0
    assert abs(result["KL_reference_to_observed_mean"]) < 1e-6
    assert compare_logits(logits, logits)["bitwise_equal"]


def test_changed_prediction_and_nonfinite_output_are_visible():
    result = compare_logits(torch.tensor([[2., 0.]]), torch.tensor([[0., 2.]]))
    assert result["top1_changed_tokens"] == 1
    assert result["KL_reference_to_observed_mean"] > 1
    with pytest.raises(ValueError, match="nonfinite"):
        compare_logits(torch.tensor([[float("nan")]]), torch.ones(1, 1))


def test_visibility_distinguishes_zero_memory_rounding_and_visible_patch():
    base = torch.ones(4, dtype=torch.bfloat16)
    small = quantization_visibility(base, torch.full((4,), .001))
    assert small["all_coordinates_unchanged"]
    assert small["swallowed_delta_energy_fraction"] == 1.
    visible = quantization_visibility(base, torch.full((4,), .01))
    assert visible["changed_fraction_of_nonzero_delta"] == 1.
    zero = quantization_visibility(base, torch.zeros(4))
    assert zero["nonzero_delta_coordinates"] == 0
    assert zero["swallowed_delta_energy_fraction"] is None


def test_optional_dispatch_plan_counts_only_real_ce_passes():
    modes = [query_modes(hc_mode="eager", compare_compiled=True, with_backward=True,
                         source="books", state_case=state)
             for state in ("calibrated_initial_writer", "same_rows_zero_delta")]
    assert len(modes[0]) == 6 and len(modes[1]) == 5
    assert sum(row["backward"] for group in modes for row in group) == 1
    assert len(precision_modes("both")) * sum(map(len, modes)) * 128 == 2816
    assert precision_modes("native") == ("native",)
    with pytest.raises(ValueError, match="requires"):
        query_modes(hc_mode="compiled", compare_compiled=True, with_backward=False,
                    source="books", state_case="calibrated_initial_writer")


def test_full_ple_effect_distinguishes_valid_and_target_positions():
    zero = torch.zeros(4, 1, 2)
    actual = zero.clone()
    actual[0] = 100  # Padding must not contaminate selected-position reports.
    actual[1] = 2
    actual[2] = 4
    valid = torch.tensor([[False, True, True, True]])
    target = torch.tensor([[False, False, True, False]])
    result = ple_effect(actual, zero, valid, target)
    assert result["valid_positions"]["positions"] == 3
    assert result["valid_positions"]["difference_max_abs"] == 4
    assert result["loss_target_positions"]["positions"] == 1
    assert result["loss_target_positions"]["difference_rms"] == 4


def test_quantization_reports_error_and_effect_energy_separately():
    base = torch.ones(1, dtype=torch.bfloat16)
    report = quantization_visibility(base, torch.tensor([.01]))
    assert report["swallowed_delta_energy_fraction"] == 0.
    assert 0 < report["effective_delta_energy_over_requested_energy"] < 1.
    assert report["relative_l2_error"] > 0.
    assert report["requested_effective_cosine"] == pytest.approx(1.)


def test_case_identity_refers_to_primary_grad_even_when_compiled_is_last():
    by_mode = {"grad": "eager", "no_grad": "eager", "grad_repeat": "eager",
               "compiled_grad": "compiled", "compiled_no_grad": "compiled"}
    fields = runtime_identity_fields(by_mode)
    assert fields["runtime_identity_sha256"] == "eager"
    assert fields["runtime_identity_sha256_by_mode"] == by_mode
    by_mode["grad"] = "later mutation"
    assert fields["runtime_identity_sha256_by_mode"]["grad"] == "eager"

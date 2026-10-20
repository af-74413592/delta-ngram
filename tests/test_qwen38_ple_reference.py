from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from delta_ngram.qwen38_ple_reference import (
    _converted, causal_depthwise_conv, dense_delta, directional_fd,
    grouped_rms_norm, overlay_quantization, ple_reference, run_reference_audit,
    signed_sqrt, split_projection_diagnostics, validate_snapshot,
)


def snapshot():
    generator = torch.Generator().manual_seed(219)

    def sample(shape, scale=1.0):
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)

    hidden = sample((1, 5, 8))
    base = sample((1, 5, 6), 0.05)
    rows = torch.arange(1, 16)
    physical = rows.reshape(1, 5, 3)
    delta = torch.randn((15, 2), generator=generator) * 1e-4
    result = {
        "runtime_identity_sha256": "a" * 64, "ple_source_sha256": "b" * 64,
        "hidden_input": hidden, "embedding_output": (base.float() + delta.reshape(1, 5, 6)).bfloat16(),
        "base_lookup": base, "native_output": hidden.clone(),
        "native_output_cotangent": sample((1, 5, 8)),
        "native_embedding_cotangent": base.clone(),
        "rows": rows, "delta": delta, "physical_rows": physical,
        "valid_mask": torch.ones((1, 5), dtype=torch.bool),
        "weights": {
            "key_proj.weight": sample((8, 6), 0.2),
            "value_proj.weight": sample((4, 6), 0.2),
            "norm_key.weight": sample((8,), 0.1),
            "norm_query.weight": sample((8,), 0.1),
            "norm_conv.weight": sample((8,), 0.1),
            "conv1d.weight": sample((8, 1, 3), 0.02),
        },
        "layout": {"hidden_size": 4, "hc_count": 2, "short_conv_state_len": 4},
        "norms": {name: {"eps": 1e-6, "group_size": 4}
                  for name in ("norm_key", "norm_query", "norm_conv")},
        "conv": {"stride": (1,), "padding": (0,), "dilation": (2,), "groups": 8},
    }
    converted = _converted(result)
    embedding = result["embedding_output"].double().requires_grad_(True)
    output = ple_reference(hidden.double(), embedding, converted)
    gradient, = torch.autograd.grad((output * result["native_output_cotangent"].double()).sum(), (embedding,))
    result["native_output"] = output.detach().bfloat16()
    result["native_embedding_cotangent"] = gradient.detach().bfloat16()
    return result


def test_norm_has_one_plus_weight_and_independent_groups():
    x = torch.tensor([[3., 4., 30., 40.]], dtype=torch.float64)
    actual = grouped_rms_norm(x, torch.zeros(4, dtype=torch.float64), 2, 1e-6)
    expected = torch.tensor([[3., 4., 3., 4.]], dtype=torch.float64)
    expected[:, :2] /= (12.5 + 1e-6) ** 0.5
    expected[:, 2:] /= (12.5 + 1e-8) ** 0.5
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(grouped_rms_norm(x, torch.ones(4), 2, 1e-6), 2 * actual)


def test_causal_conv_matches_independent_native_conv_and_cannot_see_future():
    generator = torch.Generator().manual_seed(4)
    x = torch.randn((2, 7, 4), dtype=torch.float64, generator=generator)
    weight = torch.randn((4, 1, 3), dtype=torch.float64, generator=generator)
    actual = causal_depthwise_conv(x, weight, left_pad=4, dilation=2)
    expected = F.conv1d(F.pad(x.transpose(1, 2), (4, 0)), weight, dilation=2, groups=4).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    modified = x.clone()
    modified[:, 5:] += 100
    torch.testing.assert_close(causal_depthwise_conv(modified, weight, left_pad=4, dilation=2)[:, :5], actual[:, :5])


def test_signed_sqrt_preserves_native_nonsmooth_convention():
    x = torch.tensor([-4., -1e-8, 0., 1e-8, 4.], dtype=torch.float64, requires_grad=True)
    y = signed_sqrt(x)
    torch.testing.assert_close(y, torch.tensor([-2., -.001, 0., .001, 2.], dtype=torch.float64))
    gradient, = torch.autograd.grad(y.sum(), (x,))
    torch.testing.assert_close(gradient, torch.tensor([.25, 0., 0., 0., .25], dtype=torch.float64))


def test_all_reference_cutpoints_and_chain_pass_without_promoting_native_acceptance():
    state = torch.random.get_rng_state().clone()
    report = run_reference_audit(snapshot())
    assert report["status"] == "reference_fd_passed"
    assert len(report["stage_finite_differences"]) == 12
    assert len(report["full_chain_finite_differences"]) == 3
    assert report["native_derivative_acceptance"] == "not_established_by_reference_fd"
    assert report["new_ce_target_tokens"] == 0
    assert torch.equal(state, torch.random.get_rng_state())


def test_fd_rejects_wrong_backward_and_does_not_select_favorable_step():
    class WrongBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return 3 * x

        @staticmethod
        def backward(ctx, grad):
            return 6 * grad

    result = directional_fd(WrongBackward.apply, (torch.ones(3, dtype=torch.float64),),
                            torch.ones(3, dtype=torch.float64), seed=4)
    assert not result["passed"]
    assert not any(row["passed"] for row in result["probes"])


def test_quantization_loss_is_measured_on_nonzero_delta_and_reports_energy():
    snap = snapshot()
    snap["base_lookup"].fill_(1)
    snap["delta"].zero_()
    snap["delta"][0] = torch.tensor([1e-4, 0.01])
    snap["embedding_output"] = (snap["base_lookup"].float() + dense_delta(snap)).bfloat16()
    report = overlay_quantization(snap)["overall"]
    assert report["nonzero_delta_components"] == 2
    assert report["swallowed_components"] == 1
    assert report["swallowed_fraction_of_nonzero_delta"] == 0.5
    assert report["changed_fraction_of_nonzero_delta"] == 0.5
    assert report["swallowed_delta_energy_fraction"] == pytest.approx(1e-8 / (1e-8 + 1e-4))
    assert report["effective_delta_energy_fraction"] == pytest.approx(0.0078125 ** 2 / (1e-8 + 1e-4))


def test_dense_delta_respects_padding_and_missing_rows():
    snap = snapshot()
    snap["valid_mask"][0, -1] = False
    snap["physical_rows"][0, 1, 0] = 99
    result = dense_delta(snap).reshape(1, 5, 3, 2)
    assert not bool(result[0, -1].ne(0).any())
    assert not bool(result[0, 1, 0].ne(0).any())
    torch.testing.assert_close(result[0, 0], snap["delta"][:3])


def test_forged_overlay_and_wrong_norm_layout_fail_closed():
    snap = snapshot()
    snap["embedding_output"] = snap["embedding_output"] + 1
    with pytest.raises(ValueError, match="overlay formula"):
        overlay_quantization(snap)
    snap = snapshot()
    snap["norms"]["norm_key"]["group_size"] = 8
    with pytest.raises(ValueError, match="RMSNorm"):
        validate_snapshot(snap)


def test_split_projection_keeps_fp32_signal_but_does_not_claim_empty_identity():
    snap = snapshot()
    snap["delta"] *= .01
    report = split_projection_diagnostics(snap)
    assert report["empty_state_native_identity"].startswith("unverified")
    for row in report["projections"].values():
        before = row["fp32_until_norm_gate"]
        after = row["if_immediately_cast_back_to_bf16"]
        assert before["changed_fraction_of_nonzero_delta"] > .99
        assert before["effective_delta_energy_fraction"] == pytest.approx(1., abs=.003)
        assert after["changed_fraction_of_nonzero_delta"] < .1


def test_snapshot_input_is_not_modified():
    snap = snapshot()
    before = copy.deepcopy(snap)
    run_reference_audit(snap)
    for key in ("hidden_input", "embedding_output", "delta", "native_output"):
        assert torch.equal(snap[key], before[key])
    for key in snap["weights"]:
        assert torch.equal(snap["weights"][key], before["weights"][key])

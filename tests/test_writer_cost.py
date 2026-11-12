import json

import pytest
import torch
from torch.nn import functional as F

from delta_ngram.qwen38_writer import NativeForwardWriter
from delta_ngram.state import SparseMemoryState
from delta_ngram.writer import ForwardWriter
from delta_ngram.writer_cost import (
    WriterDimensions, fp32_training_state_bytes, linear_forward_macs, main,
    parameter_breakdown, preset_dimensions, training_window_cost,
)


@pytest.mark.parametrize("preset", ["m1", "native"])
@pytest.mark.parametrize("hidden", [8, 128, 160, 256])
def test_named_parameter_counts_match_real_writer(preset, hidden):
    dim = preset_dimensions(preset, hidden)
    native = NativeForwardWriter(dim.row_dim, input_dim=dim.input_dim,
                                 hidden_dim=hidden, orders=dim.orders,
                                 heads_per_order=dim.heads_per_order, operator="D1")
    actual = {name: value.numel() for name, value in native.named_parameters()}
    assert parameter_breakdown(dim) == actual
    if preset == "m1":
        original = ForwardWriter(dim.row_dim, hidden_dim=hidden, orders=dim.orders, operator="D1")
        assert actual == {name: value.numel() for name, value in original.named_parameters()}


def test_fp32_bytes_match_materialized_adam_tensors():
    dim = WriterDimensions(7, 5, 11, heads_per_order=3)
    writer = NativeForwardWriter(5, input_dim=7, hidden_dim=11, heads_per_order=3,
                                 target_scale=torch.ones(6))
    optimizer = torch.optim.Adam(writer.parameters())
    for parameter in writer.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    size = lambda tensor: tensor.numel() * tensor.element_size()
    measured = {
        "parameters": sum(size(p) for p in writer.parameters()),
        "gradients": sum(size(p.grad) for p in writer.parameters()),
        "adam_first_moment": sum(size(s["exp_avg"]) for s in optimizer.state.values()),
        "adam_second_moment": sum(size(s["exp_avg_sq"]) for s in optimizer.state.values()),
    }
    measured["total"] = sum(measured.values())
    assert fp32_training_state_bytes(dim) == measured


@pytest.mark.parametrize("heads_per_order", [1, 3])
@pytest.mark.parametrize("empty", [False, True])
def test_macs_match_actual_linear_calls_with_padding(monkeypatch, heads_per_order, empty):
    dim = WriterDimensions(7, 5, 37, heads_per_order=heads_per_order)
    writer = NativeForwardWriter(5, input_dim=7, hidden_dim=37,
                                 heads_per_order=heads_per_order)
    hidden = torch.randn(2, 4, 7)
    rows = torch.ones(2, 4, dim.num_heads, dtype=torch.long)
    rows[0, 0, 0] = 0
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    if empty:
        mask.zero_()
    events = int((mask[:, :, None] & rows.ne(0)).sum())
    observed = []
    linear = F.linear

    def count_linear(inputs, weight, bias=None):
        observed.append((tuple(inputs.shape), weight.shape[0], inputs.numel() * weight.shape[0]))
        return linear(inputs, weight, bias)

    monkeypatch.setattr(F, "linear", count_linear)
    writer.events(hidden, rows, mask, SparseMemoryState.empty(5))
    predicted = linear_forward_macs(dim, padded_positions=8, valid_events=events)
    assert predicted["total"] == sum(macs for _, _, macs in observed)
    if heads_per_order > 1 and not empty:
        assert sum(shape == (events, dim.head_dim) for shape, _, _ in observed) == 2
        assert predicted["target_evidence"] == 8 * 3 * 7 * 37
        assert predicted["target_head_features"] == events * dim.head_dim * 37
    if empty:
        assert not observed


@pytest.mark.parametrize("change", [
    {"input_dim": 0}, {"row_dim": -1}, {"hidden_dim": True},
    {"hidden_dim": 1.5}, {"heads_per_order": 0}, {"orders": ()},
    {"orders": (2, False)}, {"orders": [2, 3]},
])
def test_invalid_dimensions(change):
    with pytest.raises(ValueError):
        WriterDimensions(**({"input_dim": 7, "row_dim": 5, "hidden_dim": 11} | change))


@pytest.mark.parametrize("positions,events", [(-1, 0), (2, -1), (1, 3), (0, 1), (1.5, 1), (1, True)])
def test_invalid_mac_counts(positions, events):
    with pytest.raises(ValueError):
        linear_forward_macs(WriterDimensions(7, 5, 11), padded_positions=positions, valid_events=events)


def test_measured_cost_units_do_not_double_count_overheads():
    result = training_window_cost(target_tokens=2_000_000, elapsed_seconds=3600,
                                  gpu_count=4, price_per_gpu_hour=2.5,
                                  checkpoint_seconds=200, eval_seconds=400,
                                  extra_eval_gpu_billed_seconds=1800)
    assert result["target_tokens_per_second"] == pytest.approx(2_000_000 / 3600)
    assert result["main_pool_gpu_hours"] == 4
    assert result["extra_eval_gpu_hours"] == 0.5
    assert result["total_gpu_hours"] == 4.5
    assert result["gpu_hours_per_million_target_tokens"] == 2.25
    assert result["total_cost"] == 11.25
    assert result["cost_per_million_target_tokens"] == 5.625
    # Overlapping sub-timers do not imply extra billed or serial wall time.
    overlap = training_window_cost(target_tokens=100, elapsed_seconds=10, gpu_count=2,
                                   price_per_gpu_hour=0, checkpoint_seconds=8, eval_seconds=8)
    assert overlap["target_tokens_per_second"] == 10
    assert overlap["main_pool_gpu_hours"] == pytest.approx(20 / 3600)
    assert overlap["total_cost"] == 0


@pytest.mark.parametrize("change", [
    {"target_tokens": 0}, {"gpu_count": True}, {"gpu_count": 1.5},
    {"elapsed_seconds": 0}, {"elapsed_seconds": float("nan")},
    {"price_per_gpu_hour": float("inf")}, {"price_per_gpu_hour": -1},
    {"checkpoint_seconds": 11}, {"eval_seconds": -1},
    {"extra_eval_gpu_billed_seconds": -1},
])
def test_invalid_measurements(change):
    inputs = {"target_tokens": 100, "elapsed_seconds": 10, "gpu_count": 2,
              "price_per_gpu_hour": 1.5} | change
    with pytest.raises(ValueError):
        training_window_cost(**inputs)


def test_price_is_required():
    with pytest.raises(TypeError):
        training_window_cost(target_tokens=100, elapsed_seconds=10, gpu_count=2)


def test_cli_json_and_optional_measurement(tmp_path, capsys):
    measurement = tmp_path / "measurement.json"
    measurement.write_text(json.dumps({"target_tokens": 100, "elapsed_seconds": 10,
                                      "gpu_count": 2, "price_per_gpu_hour": 1.5}), encoding="utf-8")
    assert main(["--preset", "native", "--hidden-dims", "128", "160", "256",
                 "--padded-positions", "8", "--valid-events", "80",
                 "--measurement", str(measurement)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [row["dimensions"]["hidden_dim"] for row in result["comparisons"]] == [128, 160, 256]
    assert result["comparisons"][0]["linear_forward_macs"]["total"] > 0
    assert result["measurement"]["target_tokens_per_second"] == 10
    assert main(["--preset", "m1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["comparisons"][0]["linear_forward_macs"] is None
    assert result["comparisons"][0]["dimensions"]["hidden_dim"] == 128
    with pytest.raises(SystemExit):
        main(["--preset", "native", "--padded-positions", "8"])

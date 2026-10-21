"""CPU analytic checks for label-free memory-writer diagnostic definitions."""
import copy
import math

import pytest
import torch

from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.qwen38_memory_diagnostics import (
    measure_encoding_sensitivity, probe_full_update, summarize_encoding_sensitivity,
)
from delta_ngram.qwen38_memory_writer import NativeMemoryResidualWriter
from delta_ngram.qwen38_writer import NativeForwardWriter, WriteEvents
from delta_ngram.state import SparseMemoryState


def memory(values=None):
    values = torch.tensor([[3., 4.], [0., 0.], [0., 12.], [999., 0.]]) if values is None else values
    return BatchMemoryState(SparseMemoryState(
        torch.arange(1, values.shape[0] + 1), values, torch.ones(values.shape[0])), 1, 8)


def blocks():
    rows = torch.tensor([[[1], [1], [2]]])
    reads = torch.tensor([[3], [1], [1], [2], [3], [0]])
    base = {"physical_rows_new": rows, "physical_rows_read": reads,
            "H_new": torch.tensor([[[0., 1.]]]).expand(1, 3, 2).clone()}
    changed = {"physical_rows_new": rows.clone(), "physical_rows_read": reads.clone(),
               "H_new": torch.tensor([[[1., 0.]]]).expand(1, 3, 2).clone()}
    return changed, base


class KnownEvents:
    """Two independently specified event sets with a physical-row collision."""
    def __init__(self):
        self.states = []

    def events(self, hidden, rows, mask, state):
        self.states.append(state)
        if not rows.numel():
            return WriteEvents(torch.empty(0, dtype=torch.long), torch.empty(0, 2), torch.empty(0))
        if hidden[0, 0, 0] > 0:
            return WriteEvents(rows.flatten(), torch.tensor([[3., 0.], [0., 4.], [0., 5.]]),
                               torch.tensor([.5, .25, 1.]))
        return WriteEvents(rows.flatten(), torch.tensor([[1., 0.], [0., 0.], [0., 1.]]),
                           torch.tensor([1., .5, .5]))


def measured():
    changed, base = blocks()
    return measure_encoding_sensitivity(KnownEvents(), memory(), changed, base, encoding_mode="full")


def test_g_uses_gate_weighted_row_means_unique_prefix_reads_and_fixed_direct_state():
    changed, base = blocks()
    state = memory()
    saved = state.values.clone()
    writer = KnownEvents()
    result = measure_encoding_sensitivity(writer, state, changed, base, encoding_mode="full")
    # Changed row 1 mean is (2,4/3), base is (2/3,0); row 2 differs by (0,4).
    numerator = math.sqrt(176 / 9)
    assert result["target_difference_l2"] == pytest.approx(numerator)
    # Prefix row 3 contributes 12; duplicate read addresses count once and
    # unread row 4 contributes nothing. Written-row norms alone would give 5.
    assert result["read_state_l2"] == 13.
    assert result["block_g"] == pytest.approx(numerator / 13)
    assert result["unique_read_rows"] == 3 and result["written_rows"] == 2
    assert result["eligible_written_rows"] == result["filtered_written_rows"] == 1
    assert result["row_output_gain_values"] == pytest.approx([math.sqrt(32 / 9) / 5])
    assert result["gate_mean_absolute_change"] == .4375
    assert result["gate_max_absolute_change"] == .5
    assert result["normalized_H_relative_l2"] == pytest.approx(math.sqrt(2))
    assert writer.states == [state.memory, state.memory]
    assert torch.equal(state.values, saved)
    assert result["matched_read_addresses_verified"]


def test_lazy_state_denominator_uses_effective_values():
    state = LazyDecayState.from_batch(memory())
    state = LazyDecayState(state.memory, torch.tensor([1]), torch.tensor([math.log(.5)], dtype=torch.float64),
                           state.row_last_written_block, state.row_log_alpha_at_write)
    changed, base = blocks()
    result = measure_encoding_sensitivity(KnownEvents(), state, changed, base, encoding_mode="full")
    assert result["read_state_l2"] == 6.5
    assert result["block_g"] == pytest.approx(measured()["block_g"] * 2)


def test_optional_g_energy_relationship_includes_filtered_new_rows_and_prefix():
    changed, base = blocks()
    result = measure_encoding_sensitivity(KnownEvents(), memory(), changed, base,
                                         encoding_mode="full", include_relationship=True)
    energy = result["denominator_relationship"]
    assert energy["read_state_energy"] == 169.
    assert energy["written_state_energy"] == energy["eligible_written_state_energy"] == 25.
    assert energy["read_only_state_energy"] == 144.
    assert energy["filtered_written_state_energy"] == 0.
    assert energy["eligible_target_difference_energy"] == pytest.approx(32 / 9)
    assert energy["filtered_target_difference_energy"] == 16.
    assert energy["eligible_weighted_row_gain_squared"] == pytest.approx((32 / 9) / 25)
    assert result["block_g"] ** 2 == pytest.approx(
        energy["eligible_contribution_to_block_g_squared"] + energy["filtered_contribution_to_block_g_squared"])
    assert "denominator_relationship" not in measured()


def test_monitoring_g_threshold_never_requests_automatic_stop():
    row = measured()
    row["block_g"] = 5.
    legacy = summarize_encoding_sensitivity([row])
    monitored = summarize_encoding_sensitivity([row], monitor_only=True)
    assert legacy["stop_requested"] and not monitored["stop_requested"]
    assert monitored["block_fraction_above_threshold"] == 1.
    assert monitored["block_g"] == legacy["block_g"]
    assert not summarize_encoding_sensitivity([], monitor_only=True)["stop_requested"]
    with pytest.raises(ValueError, match="boolean"):
        summarize_encoding_sensitivity([row], monitor_only=1)


def test_zero_read_state_and_no_writes_are_explicitly_unassessable():
    changed, base = blocks()
    result = measure_encoding_sensitivity(
        KnownEvents(), memory(torch.zeros(4, 2)), changed, base, encoding_mode="full")
    assert result["block_g"] is None and not result["assessable"]
    assert result["filtered_reason"] == "read_state_l2_below_floor"
    assert result["filtered_written_rows"] == 2 and not result["row_output_gain_values"]
    for block in (changed, base):
        block["physical_rows_new"] = torch.empty(1, 0, 1, dtype=torch.long)
        block["H_new"] = torch.empty(1, 0, 2)
    empty = measure_encoding_sensitivity(KnownEvents(), memory(), changed, base, encoding_mode="full")
    assert empty["filtered_reason"] == "no_write_events" and empty["block_g"] is None
    assert empty["gate_mean_absolute_change"] is None and empty["gate_max_absolute_change"] is None
    summary = summarize_encoding_sensitivity([result, empty])
    assert not summary["assessable"] and not summary["stop_requested"]
    assert summary["filtered_blocks"] == 2 and summary["filtered_written_fraction"] == 1.


@pytest.mark.parametrize("mutation,match", [
    ("out_of_range", "between zero and row_capacity"),
    ("negative", "between zero and row_capacity"),
    ("missing_base_reads", "prefix and new-token"),
    ("mismatched_reads", "identical prefix"),
    ("missing_written", "include every written"),
])
def test_bad_read_address_metadata_cannot_silently_shrink_g_denominator(mutation, match):
    changed, base = blocks()
    if mutation == "out_of_range":
        changed["physical_rows_read"][0] = 9
    elif mutation == "negative":
        changed["physical_rows_read"][0] = -1
    elif mutation == "missing_base_reads":
        del base["physical_rows_read"]
    elif mutation == "mismatched_reads":
        base["physical_rows_read"][0] = 4
    else:
        for block in (changed, base):
            block["physical_rows_read"][3] = 1
    with pytest.raises(ValueError, match=match):
        measure_encoding_sensitivity(KnownEvents(), memory(), changed, base, encoding_mode="full")


def test_zero_gate_denominator_is_rejected_without_a_silent_clamp():
    class Closed(KnownEvents):
        def events(self, *args):
            event = super().events(*args)
            return WriteEvents(event.rows, event.target, torch.zeros_like(event.beta))
    changed, base = blocks()
    with pytest.raises(ValueError, match="nonpositive gate denominator"):
        measure_encoding_sensitivity(Closed(), memory(), changed, base, encoding_mode="full")


def test_zero_initialized_residual_has_zero_encoding_replacement_gain():
    torch.manual_seed(17)
    writer = NativeMemoryResidualWriter(NativeForwardWriter(
        width=2, input_dim=2, hidden_dim=4, orders=(2,), operator="D1"))
    changed, base = blocks()
    # Actual cached base is fixed, even if it differs from the paired diagnostic base.
    changed["H_base_new"] = torch.randn_like(changed["H_new"])
    result = measure_encoding_sensitivity(writer, memory(), changed, base, encoding_mode="residual")
    assert result["block_g"] == 0 and result["row_output_gain_values"] == [0.]
    assert result["gate_mean_absolute_change"] == 0


def test_matched_residual_diagnostic_fixes_h12_and_same_matched_difference_baseline(monkeypatch):
    torch.manual_seed(83)
    writer = NativeMemoryResidualWriter(NativeForwardWriter(
        width=2, input_dim=2, hidden_dim=4, orders=(2,), operator="D1"), difference_baseline="matched_base")
    changed, base = blocks()
    changed["H_base_new"] = torch.randn_like(changed["H_new"])
    observed = []
    original = writer.events
    def events(hidden, *args, **kwargs):
        observed.append((hidden, kwargs["hidden_mem"], kwargs["hidden_base_matched"]))
        return original(hidden, *args, **kwargs)
    monkeypatch.setattr(writer, "events", events)
    result = measure_encoding_sensitivity(writer, memory(), changed, base, encoding_mode="residual")
    assert result["block_g"] == 0.
    assert len(observed) == 2 and observed[0][0] is observed[1][0]
    assert observed[0][2] is observed[1][2] is observed[1][1]
    assert not torch.equal(observed[0][0], observed[0][2])


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf"), True])
def test_floors_and_stop_threshold_must_be_positive_finite(bad):
    changed, base = blocks()
    with pytest.raises(ValueError, match="positive and finite"):
        measure_encoding_sensitivity(KnownEvents(), memory(), changed, base,
                                     encoding_mode="full", denominator_floor=bad)
    with pytest.raises(ValueError, match="positive and finite"):
        summarize_encoding_sensitivity([], stop_p99=bad)
    with pytest.raises(ValueError, match="positive and finite"):
        probe_full_update(lambda state: state, memory(), absolute_floor=bad)


def test_summary_threshold_is_strict_block_p99_not_a_row_gain_or_theorem():
    one = measured()
    one["block_g"] = 1.
    one["row_output_gain_values"] = [100.]
    result = summarize_encoding_sensitivity([one])
    assert result["block_g"]["p99"] == 1. and not result["stop_requested"]
    assert result["row_output_gain"]["p99"] == 100.
    two = copy.deepcopy(one)
    two["block_g"] = 1.5
    result = summarize_encoding_sensitivity([one, two], stop_p99=2.)
    assert result["block_g"]["p99"] == pytest.approx(1.495)
    assert result["block_fraction_above_one"] == .5
    assert result["block_fraction_above_threshold"] == 0 and not result["stop_requested"]
    assert result["written_rows"] == 4  # Row observations, not distinct global IDs.
    assert summarize_encoding_sensitivity([one, two])["stop_requested"]
    assert not summarize_encoding_sensitivity([])["assessable"]
    two["mode"] = "residual"
    with pytest.raises(ValueError, match="one encoding mode"):
        summarize_encoding_sensitivity([one, two])
    one["filtered_written_rows"] = 0
    with pytest.raises(ValueError, match="counts do not match"):
        summarize_encoding_sensitivity([one])


@pytest.mark.parametrize("factor", [0., .5, 1., 2., -2.])
def test_complete_update_directional_gain_matches_known_linear_map(factor):
    state = memory(torch.tensor([[3., 4.], [1., 2.]]))
    saved = state.values.clone()
    rng = torch.get_rng_state().clone()
    result = probe_full_update(lambda snapshot: snapshot.with_values(snapshot.values * factor), state)
    assert result["calls"] == 14 and result["assessable"]
    assert result["relative_epsilons"] == [1e-3, 1e-2]
    assert result["direction_seed"] == 17 and result["structured_row"] == 1
    assert result["repeat_error_l2"] == 0
    for probe in result["probes"]:
        assert probe["directional_amplification"] == pytest.approx(abs(factor), abs=1e-12)
        assert probe["repeat_error_to_signal"] == 0
    assert torch.equal(state.values, saved) and torch.equal(torch.get_rng_state(), rng)


def test_probe_aligns_new_output_rows_and_reports_row_statistics():
    state = memory(torch.tensor([[3., 4.]]))
    def update(snapshot):
        return BatchMemoryState(SparseMemoryState(
            torch.tensor([1, 2]), torch.cat((snapshot.values, 2 * snapshot.values)),
            torch.ones(2)), 1, 8)
    result = probe_full_update(update, state, random_directions=1)
    assert result["calls"] == 10
    for probe in result["probes"]:
        assert probe["directional_amplification"] == pytest.approx(math.sqrt(5))
        assert probe["output_union_rows"] == probe["output_changed_rows"] == 2
        assert probe["output_row_difference_l2"]["maximum"] > 0


def test_repeat_control_exposes_update_noise():
    calls = 0
    def noisy(snapshot):
        nonlocal calls
        calls += 1
        return snapshot.with_values(snapshot.values + calls * .01)
    result = probe_full_update(noisy, memory(torch.tensor([[3., 4.]])), random_directions=1)
    assert result["repeat_error_l2"] > 0
    assert all(probe["repeat_error_to_signal"] > 0 for probe in result["probes"])


@pytest.mark.parametrize("kind,match", [("mutation", "mutated"), ("namespace", "namespace"),
                                        ("counts", "observation counts"), ("nan", "finite")])
def test_probe_rejects_incomparable_or_mutated_update_results(kind, match):
    calls = 0
    def update(snapshot):
        nonlocal calls
        calls += 1
        if kind == "mutation":
            snapshot.values.add_(1.)
            return snapshot
        if kind == "namespace":
            return BatchMemoryState(snapshot.memory, 1, 9)
        if kind == "counts":
            return BatchMemoryState(SparseMemoryState(
                snapshot.rows, snapshot.values, snapshot.counts + calls), 1, 8)
        return snapshot.with_values(torch.full_like(snapshot.values, float("nan")))
    with pytest.raises((ValueError, RuntimeError), match=match):
        probe_full_update(update, memory(torch.tensor([[3., 4.]])))


def test_zero_and_rounded_away_perturbations_are_not_called_stable():
    result = probe_full_update(lambda state: state, memory(torch.zeros(1, 2)))
    assert not result["assessable"] and result["calls"] == 0
    result = probe_full_update(lambda state: state, memory(torch.tensor([[1e20, 1e20]])),
                               relative_epsilons=(1e-40,), random_directions=1)
    assert not result["assessable"] and result["calls"] == 2
    assert all(probe["filtered"] for probe in result["probes"])


def test_probe_rejects_multiowner_states_and_invalid_direction_settings():
    state = memory(torch.tensor([[3., 4.]]))
    multi = BatchMemoryState.from_states([state.memory, state.memory], 8)
    with pytest.raises(ValueError, match="one-owner"):
        probe_full_update(lambda value: value, multi)
    for settings in [{"random_directions": True}, {"random_directions": 1.5},
                     {"relative_epsilons": (float("nan"),)}, {"relative_epsilons": (.01, .01)}]:
        with pytest.raises(ValueError):
            probe_full_update(lambda value: value, state, **settings)

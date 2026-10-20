import copy
import json
import random

import pytest

from delta_ngram.qwen38_probe_analysis import ARMS, SOURCES, analyze_probe_rows


def rows():
    result = []
    for source_index, source in enumerate(SOURCES):
        for family in range(2):
            baseline = 4.0 + source_index / 4 + family / 8
            family_gain = (source_index + family + 1) / 32
            for arm in ARMS:
                for step in (0, 4, 8):
                    gain = family_gain * step / 8
                    if arm == ARMS[1]:
                        gain += .125 * step / 8
                    total = (baseline - gain) * 128
                    empty = (baseline + .25) * 128
                    result.append({
                        "arm": arm, "step": step, "sample_id": f"{source}:{family}",
                        "source": source, "loss_sum": total, "loss_target_tokens": 128,
                        "query_target_tokens": 128, "empty_loss_sum": empty,
                        "delta_empty": (empty - total) / 128,
                    })
    return result


def test_paired_analysis_known_contrast_and_local_random_state():
    data = rows()
    before = random.getstate()
    result = analyze_probe_rows(data)
    assert random.getstate() == before
    assert result["self_probe_target_accesses"] == 9216
    assert result["shared_empty_target_accesses"] == 1536
    contrast = result["paired_alpha_contrast"]["overall"]["8"]
    assert contrast["gain_difference"] == .125
    assert contrast["paired_family_bootstrap_ci95"] == [.125, .125]
    for source in SOURCES:
        value = result["paired_alpha_contrast"]["by_source"][source]["4"]
        assert value["gain_difference"] == .0625
        assert value["paired_family_bootstrap_ci95"] == [.0625, .0625]
    assert not result["automatic_norm_choice"] and not result["automatic_alpha_choice"]
    json.dumps(result, allow_nan=False)
    random.Random(999).shuffle(data)
    assert analyze_probe_rows(data) == result


@pytest.mark.parametrize("corruption", [
    "missing", "duplicate", "different_source", "different_empty",
    "wrong_delta", "nan_loss", "wrong_tokens", "unknown_arm", "unknown_step",
])
def test_rejects_incomplete_or_unpaired_inputs(corruption):
    data = copy.deepcopy(rows())
    if corruption == "missing":
        data.pop()
    elif corruption == "duplicate":
        data[-1] = dict(data[0])
    elif corruption == "different_source":
        data[1]["source"] = "code"
    elif corruption == "different_empty":
        data[1]["empty_loss_sum"] += 1
        data[1]["delta_empty"] += 1 / 128
    elif corruption == "wrong_delta":
        data[0]["delta_empty"] += .01
    elif corruption == "nan_loss":
        data[0]["loss_sum"] = float("nan")
    elif corruption == "wrong_tokens":
        data[0]["loss_target_tokens"] = 127
    elif corruption == "unknown_arm":
        data[0]["arm"] = "alpha100_none"
    elif corruption == "unknown_step":
        data[0]["step"] = 5
    with pytest.raises(ValueError):
        analyze_probe_rows(data)

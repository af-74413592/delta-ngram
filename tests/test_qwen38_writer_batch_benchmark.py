import pytest
import torch

from delta_ngram.qwen38_writer_batch_benchmark import make_jobs, padded_batch, compare_tensor, select_frames


def frames():
    return {task: [{"task": task, "source_row_index": i, "input_ids": list(range(2, 5 + i))}
                   for i in range(12)] for task in ("sme", "nav")}


def test_modes_preserve_d1_order_and_common_padding():
    rows = select_frames(frames())
    canonical, matched, batched = [make_jobs(rows, mode) for mode in
                                  ("b1_canonical", "b1_matched_padding", "b12")]
    assert [len(mode) for mode in (canonical, matched, batched)] == [24, 24, 2]
    assert all([i for job in mode for i in job["indices"]] == list(range(24))
               for mode in (canonical, matched, batched))
    assert matched[0]["sequence_length"] == batched[0]["sequence_length"]
    assert canonical[0]["sequence_length"] < matched[0]["sequence_length"]


def test_padding_keeps_document_positions_and_excludes_all_pad_tokens():
    rows = select_frames(frames())[:2]
    batch = padded_batch(rows, 16, pad_token_id=99)
    assert batch["batch"]["input_ids"].shape == (2, 16)
    assert batch["valid_mask"].sum().item() == 7
    assert batch["batch"]["position_ids"].tolist() == [list(range(16))] * 2
    assert batch["batch"]["input_ids"][~batch["valid_mask"]].eq(99).all()
    assert batch["batch"]["attention_mask"] is None


def test_selection_rejects_changed_support_order():
    data = frames()
    data["sme"][0]["source_row_index"] = 99
    with pytest.raises(ValueError):
        select_frames(data)


def test_padding_rejects_truncation():
    with pytest.raises(ValueError):
        padded_batch(select_frames(frames())[:2], 2, pad_token_id=99)


def test_difference_reports_bf16_rounding_without_accepting_it():
    reference = torch.tensor([1., 2.], dtype=torch.bfloat16)
    actual = torch.tensor([1.0078125, 2.], dtype=torch.bfloat16)
    result = compare_tensor(actual, reference)
    assert not result["bitwise_equal"] and result["unequal_elements"] == 1
    assert result["max_abs"] == .0078125 and result["relative_l2"] > 0


def test_difference_rejects_layout_mismatch():
    with pytest.raises(ValueError):
        compare_tensor(torch.ones(3), torch.ones(2))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_difference_rejects_nonfinite_values(bad):
    with pytest.raises(ValueError, match="nonfinite"):
        compare_tensor(torch.tensor([bad]), torch.ones(1))
    with pytest.raises(ValueError, match="nonfinite"):
        compare_tensor(torch.ones(1), torch.tensor([bad]))

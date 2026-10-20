"""CPU checks for wave binding/scaling; these do not claim a GPU runtime test."""
from __future__ import annotations

import gc
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"
sys.path.insert(0, str(SCRIPTS))
import qwen38_native_batched_schedule as schedule_module


class FakeDist:
    @staticmethod
    def get_world_size():
        return 1

    @staticmethod
    def all_gather_object(output, value):
        output[0] = value

    @staticmethod
    def broadcast(value, **kwargs):
        pass

    @staticmethod
    def all_gather(output, value, **kwargs):
        output[0].copy_(value)
        output[1].copy_(value + 100)


PS = SimpleNamespace(
    is_pipeline_last_stage=lambda: True,
    get_pipeline_model_parallel_last_rank=lambda: 0,
    get_pipeline_model_parallel_group=lambda: "PP",
    get_tensor_model_parallel_group=lambda: "TP",
)


def fake_native_schedule(**kwargs):
    """Emulate only the pinned loss callback CP1/M and metric store contract."""
    count = kwargs["num_microbatches"]
    result = []
    for _ in range(count):
        output, callback = kwargs["forward_step_func"](kwargs["data_iterator"], kwargs["model"][0])
        objective, metric = callback(output)
        objective = objective / count
        if not kwargs["forward_only"]:
            objective.backward()
        result.append(metric)
    return result


def item(ids, labels, *, valid=None):
    tokens = torch.tensor(ids, dtype=torch.long)
    targets = None if labels is None else torch.tensor(labels, dtype=torch.long)
    mask = torch.ones_like(tokens, dtype=torch.bool) if valid is None else torch.tensor(valid, dtype=torch.bool)
    return {
        "batch": {"input_ids": tokens, "labels": targets,
                  "position_ids": torch.arange(tokens.shape[1]).expand(tokens.shape[0], -1).clone(),
                  "attention_mask": None},
        "valid_mask": mask, "owner_ids": torch.arange(tokens.shape[0]),
        "loss_target_tokens": 0 if targets is None else int(targets.ne(-100).sum()),
    }


class QueryModel:
    def __init__(self):
        self.active_state = None

    def __call__(self, input_ids, **kwargs):
        return input_ids.float() * self.active_state.values.sum()


class Adapter:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def run_native(self, call, *, input_ids, valid_mask, owner_ids, mode, state):
        self.calls.append((mode, state))
        self.model.active_state = state
        result = call()
        capture = SimpleNamespace(physical_rows=torch.where(
            valid_mask[..., None], input_ids[..., None].expand(-1, -1, 16) + 1, 0))
        return result, capture


def fake_runtime(monkeypatch):
    monkeypatch.setattr(schedule_module, "_native_runtime",
                        lambda session: (torch, FakeDist, PS, fake_native_schedule))


def test_multiple_microbatches_keep_separate_leaves_and_single_group_denominator(monkeypatch):
    fake_runtime(monkeypatch)
    model = QueryModel()
    adapter = Adapter(model)
    session = SimpleNamespace(models=[model], adapter=adapter)
    batches = [
        item([[2, 3, 4, 5], [7, 8, 9, 10]], [[1, 1, -100, -100], [-100] * 4],
             valid=[[True] * 4, [False] * 4]),
        item([[11, 12, 13, 14], [17, 18, 19, 20]], [[1, -100, -100, -100], [1, 1, 1, -100]]),
        item([[23, 24, 25, 26], [31, 32, 33, 34]], [[1, 1, -100, -100], [1, -100, -100, -100]]),
    ]
    for index, batch in enumerate(batches):
        batch["reader_state"] = SimpleNamespace(
            values=torch.full((2, 3), .1 * (index + 1), requires_grad=True))
    denominator = 37  # Includes other trajectory queries outside this wave.
    result = schedule_module.schedule_query_microbatches(
        session, batches, group_total_target_tokens=denominator)
    assert result["num_microbatches"] == 3 and result["micro_batch_size"] == 2
    assert result["schedule_microbatch_divisor_compensation"] == 3
    for index, (batch, report) in enumerate(zip(batches, result["microbatches"])):
        labels, ids = batch["batch"]["labels"], batch["batch"]["input_ids"]
        selected = labels.ne(-100)
        expected = ids[selected].sum().float() / denominator
        torch.testing.assert_close(batch["reader_state"].values.grad,
                                   torch.full_like(batch["reader_state"].values, expected))
        assert report["loss_target_tokens_by_lane"] == selected.sum(1).tolist()
        expected_lane = torch.where(selected, ids.float(), 0).sum(1) * batch["reader_state"].values.sum().detach()
        torch.testing.assert_close(torch.tensor(report["loss_sum_by_lane"]), expected_lane)
        assert adapter.calls[index] == ("query", batch["reader_state"])
        assert report["reader_loss_divisor"] == denominator
    assert result["query_target_tokens"] == 9


def test_loss_closures_bind_their_own_labels_and_ignore_masked_nonfinite_values():
    first = torch.tensor([[1, -100], [-100, -100]])
    second = torch.tensor([[-100, 2], [3, 4]])
    callbacks = [schedule_module._query_loss_callback(
        labels, index=index, num_microbatches=2, group_total_target_tokens=11)
        for index, labels in enumerate((first, second))]
    first_losses = torch.tensor([[2., float("nan")], [float("nan"), float("nan")]], requires_grad=True)
    second_losses = torch.tensor([[7., 3.], [5., 11.]], requires_grad=True)
    outputs = [callback(losses) for callback, losses in zip(callbacks, (first_losses, second_losses))]
    sum(value[0] / 2 for value in outputs).backward()
    torch.testing.assert_close(first_losses.grad, first.ne(-100).float() / 11)
    torch.testing.assert_close(second_losses.grad, second.ne(-100).float() / 11)
    assert outputs[0][1]["loss_target_tokens_by_lane"].tolist() == [1, 0]
    assert outputs[1][1]["loss_target_tokens_by_lane"].tolist() == [1, 2]


def test_wave_validation_refuses_unscored_queries_and_shape_drift():
    empty = item([[1, 2], [3, 4]], [[-100] * 2, [-100] * 2])
    with pytest.raises(ValueError, match="no supervised"):
        schedule_module._validate_microbatches([empty], query=True, group_total_target_tokens=10)
    a = item([[1, 2], [3, 4]], [[1, -100], [1, -100]])
    b = item([[1, 2, 3, 4], [3, 4, 5, 6]], [[1, -100, -100, -100], [1, -100, -100, -100]])
    with pytest.raises(ValueError, match="share padded shape"):
        schedule_module._validate_microbatches([a, b], query=True, group_total_target_tokens=10)
    with pytest.raises(ValueError, match="exceed"):
        schedule_module._validate_microbatches([a], query=True, group_total_target_tokens=1)


class Head(torch.nn.Module):
    def forward(self, hidden):
        return hidden.float().repeat_interleave(2, dim=0)


class EvidenceModel:
    def __init__(self):
        self.output_layer = Head()
        self.output_refs = []

    def __call__(self, input_ids, **kwargs):
        assert kwargs["labels"] is None and not torch.is_grad_enabled()
        b, s = input_ids.shape
        hidden = input_ids[:, :s // 2].T[..., None].expand(s // 2, b, 2560).to(torch.bfloat16)
        output = self.output_layer(hidden)
        self.output_refs.append(weakref.ref(output))
        return output


def test_evidence_wave_returns_each_lane_H_and_discards_native_head_outputs(monkeypatch):
    fake_runtime(monkeypatch)
    monkeypatch.setattr(schedule_module, "language_model", lambda model: model)
    model = EvidenceModel()
    adapter = Adapter(model)
    session = SimpleNamespace(models=[model], raw=model, adapter=adapter)
    batches = [item([[1, 2, 3, 4], [5, 6, 7, 8]], None),
               item([[11, 12, 13, 14], [15, 16, 17, 18]], None)]
    result = schedule_module.collect_evidence_microbatches(session, batches)
    assert result["num_microbatches"] == 2 and result["loss_target_tokens"] == 0
    assert result["backward"] is False and result["base_only"] is True
    for batch, report in zip(batches, result["microbatches"]):
        first_half = batch["batch"]["input_ids"][:, :2].to(torch.bfloat16)
        expected = torch.cat((first_half, first_half + 100), dim=1)
        torch.testing.assert_close(report["hidden"][..., 0], expected)
        assert report["hidden"].shape == (2, 4, 2560)
        assert report["physical_rows"].shape == (2, 4, 16)
    assert adapter.calls == [("evidence", None), ("evidence", None)]
    assert len(model.output_layer._forward_pre_hooks) == 0
    gc.collect()
    assert all(reference() is None for reference in model.output_refs)

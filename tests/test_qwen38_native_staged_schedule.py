"""Small CPU contracts for the staged direct-call boundary, without a model load."""
from pathlib import Path
from types import SimpleNamespace
import sys
import weakref

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"))
import qwen38_native_ep4_schedule as reference
import qwen38_native_staged_schedule as staged


class FakeDist:
    @staticmethod
    def get_world_size():
        return 4

    @staticmethod
    def all_gather_object(copies, description):
        copies[:] = [description] * 4


def microbatch(size, *, query):
    ids = torch.arange(640).expand(size, -1).clone()
    return {
        "batch": {"input_ids": ids,
                  "labels": torch.full_like(ids, -100) if query else None,
                  "position_ids": ids.clone(), "attention_mask": None},
        "valid_mask": torch.zeros_like(ids, dtype=torch.bool),
        "owner_ids": torch.arange(size),
        "loss_target_tokens": 0,
    }


@pytest.mark.parametrize("size,active", [(1, True), (5, True), (5, False)])
def test_dynamic_batch_and_inactive_local_rank_keep_global_denominator(monkeypatch, size, active):
    leaf = torch.arange(1, size + 1, dtype=torch.float32)[:, None].requires_grad_()
    state = SimpleNamespace(batch_size=size, values=leaf)
    item = microbatch(size, query=True)
    if active:
        item["valid_mask"][:, :2] = True
        item["batch"]["labels"][:, :2] = 10
        item["loss_target_tokens"] = 2 * size
    item["reader_state"] = state

    class Model:
        def __call__(self, **batch):
            coefficients = torch.arange(1, 641, dtype=torch.float32)[None, :]
            values = leaf * coefficients
            # Ignored NaNs must neither enter reported NLL nor alter gradients.
            return torch.where(batch["labels"].ne(-100), values, torch.full_like(values, float("nan")))

    class Adapter:
        def run_native(self, call, **kwargs):
            assert kwargs["state"] is state
            return call(), None

    session = SimpleNamespace(models=[Model()], adapter=Adapter(), device=torch.device("cpu"))
    monkeypatch.setattr(staged, "_runtime", lambda session: (torch, FakeDist))
    result = staged.schedule_query_microbatches(
        session, [item], group_total_target_tokens=37, backward=True)
    expected = torch.full_like(leaf, 3 / 37 if active else 0)
    torch.testing.assert_close(leaf.grad, expected)
    assert result["query_target_tokens"] == (2 * size if active else 0)
    assert result["loss_sum"] == (3 * size * (size + 1) / 2 if active else 0)
    assert result["micro_batch_size"] == size
    assert result["statistics_are_rank_local"] and not result["DP_loss_average"]


def test_evidence_matches_normal_head_input_and_restores_hook(monkeypatch):
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return hidden_states[..., :7].contiguous()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = Head()
            self.ple_completed = False

        def forward(self, input_ids, **kwargs):
            self.ple_completed = True
            hidden = (input_ids.t().to(torch.bfloat16)[:, :, None]
                      + torch.arange(2560, dtype=torch.bfloat16)[None, None, :] % 7)
            self.last_hidden = weakref.ref(hidden)
            return self.output_layer(hidden)

    model = Model()

    class Adapter:
        def __init__(self):
            self.completed_checks = 0

        def run_native(self, call, **kwargs):
            model.ple_completed = False
            output = call()
            assert model.ple_completed
            self.completed_checks += 1
            size, length = kwargs["input_ids"].shape
            capture = SimpleNamespace(physical_rows=torch.ones((size, length, 16), dtype=torch.long))
            return output, capture

    adapter = Adapter()
    session = SimpleNamespace(raw=model, models=[model], adapter=adapter, device=torch.device("cpu"))
    item = microbatch(3, query=False)
    monkeypatch.setattr(reference, "_runtime", lambda session: (torch, FakeDist))
    monkeypatch.setattr(reference, "language_model", lambda raw: raw)
    expected = reference.collect_evidence_microbatches(session, [item])["microbatches"][0]
    assert model.output_layer.calls == 1
    actual = staged._collect_one_evidence(
        session, item, torch=torch, model=model, batch_size=3)
    assert torch.equal(actual["hidden"], expected["hidden"])
    assert torch.equal(actual["physical_rows"], expected["physical_rows"])
    assert model.output_layer.calls == 1  # Staged evidence never called the head.
    assert adapter.completed_checks == 2  # Both exited run_native normally.
    assert not model.output_layer._forward_pre_hooks
    assert model.last_hidden() is None  # No captured exception keeps the decoder frame alive.

    # An unrelated native error must propagate, with the temporary hook removed.
    def fail(**kwargs):
        raise RuntimeError("unrelated native failure")
    session.models = [fail]
    with pytest.raises(RuntimeError, match="unrelated native failure"):
        staged._collect_one_evidence(session, item, torch=torch, model=model, batch_size=3)
    assert not model.output_layer._forward_pre_hooks

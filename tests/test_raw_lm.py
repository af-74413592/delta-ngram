"""Causality, full gradients, and exact CPU resume through the raw LM path."""
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_ngram.checkpoints import load_checkpoint, restore_writer
from delta_ngram.cli import main
from delta_ngram.config import ExperimentConfig
from delta_ngram.data.raw_text import TrajectoryWindow
from delta_ngram.raw_lm import train_raw, trajectory_loss
from delta_ngram.train import create_runtime


def config_for(tmp_path, name="run", *, raw=None, seed=17):
    dataset = tmp_path / "documents.jsonl"
    if not dataset.exists():
        documents = [{"id": f"doc-{index}", "group_id": f"group-{index}", "source": "code",
                      "split": "train" if index < 3 else "validation",
                      "text": ("abc xyz\n" if index % 2 else "xyz abc\n") * 24}
                     for index in range(4)]
        dataset.write_text("\n".join(json.dumps(d) for d in documents) + "\n", encoding="utf-8")
    values = {"seed": seed, "model": {"smoke": True, "device": "cpu"},
              "reader": {"buckets": 17}, "writer": {"hidden_dim": 12},
              "run": {"output_dir": name, "max_wall_seconds": 90},
              "raw": {"dataset": "documents.jsonl", "block_size": 8, "prefix_tokens": 4,
                      "block_counts": [2, 4], "batch_size": 2, "loss_tokens": 128,
                      "eval_windows": 2, "eval_every_tokens": 64, **(raw or {})}}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return ExperimentConfig.load(path)


def window(method, text, name="doc", block_size=8):
    ids = tuple(method.tokenizer.encode(text))
    assert len(ids) % block_size == 0
    return TrajectoryWindow(name, name, "code", 0, ids, len(ids) // block_size)


def grads(method):
    return [parameter.grad.clone() if parameter.grad is not None else None
            for parameter in method.parameters()]


def assert_grads(actual, expected, *, rtol=2e-5, atol=1e-7):
    assert any(g is not None and bool(g.abs().sum() > 0) for g in actual)
    for left, right in zip(actual, expected):
        if left is None or right is None:
            assert left is right
        else:
            torch.testing.assert_close(left, right, rtol=rtol, atol=atol)


def test_future_perturbation_and_read_before_write(tmp_path):
    method, _ = create_runtime(config_for(tmp_path))
    original = window(method, "abc xyz\n" * 4)
    changed = window(method, "abc xyz\n" * 2 + "QQQ ZZZ\n" * 2)

    def run(item):
        events, logits = [], {}

        def observe(event, block, state, ids, mask, labels):
            events.append((event, block))
            if event == "read":
                with torch.no_grad():
                    logits[block] = method.reader(ids, mask, state).logits.detach().clone()
                assert bool(labels[:, :4].eq(-100).all())
                assert bool(labels[:, 4:].eq(ids[:, 4:]).all())

        result = trajectory_loss(method, [item], prefix_tokens=4, observer=observe)
        return result, events, logits

    before, events, before_logits = run(original)
    after, changed_events, after_logits = run(changed)
    assert events == changed_events == [("write", 0), ("read", 1), ("write", 1),
                                        ("read", 2), ("write", 2), ("read", 3)]
    torch.testing.assert_close(before_logits[1], after_logits[1], rtol=0, atol=0)
    torch.testing.assert_close(before.block_loss_sums[0], after.block_loss_sums[0], rtol=0, atol=0)
    # Even for the changed block, logits predicting its first token precede
    # the changed token and therefore remain identical.
    torch.testing.assert_close(before_logits[2][:, :4], after_logits[2][:, :4], rtol=0, atol=0)
    assert before.loss_tokens == 24
    assert before.counters == {"loss_tokens": 24, "raw_tokens_consumed": 32,
                               "writer_new_tokens": 24, "encoder_input_tokens": 24,
                               "reader_input_tokens": 36, "address_prefix_tokens": 4,
                               "write_blocks": 3, "trajectories": 1}


def test_last_loss_reaches_first_state_and_writer_without_backbone_grads(tmp_path):
    method, _ = create_runtime(config_for(tmp_path))
    states = {}

    def capture(event, block, state, *_):
        if event == "read":
            states[block] = state

    result = trajectory_loss(method, [window(method, "abc xyz\n" * 4)],
                             prefix_tokens=4, observer=capture)
    first_values = states[1].memory.values
    gradient, = torch.autograd.grad(result.block_loss_sums[-1], first_values, retain_graph=True)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    result.mean.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in method.parameters())
    assert all(p.grad is None for p in method.reader.model.parameters())
    assert all(state.memory.values.grad_fn is not None for state in states.values())


def test_address_boundary_uses_old_tokens_without_encoding_or_rewriting_them(tmp_path, monkeypatch):
    import delta_ngram.raw_lm as module
    method, _ = create_runtime(config_for(tmp_path))
    item = window(method, "abcd ef\n" * 4)
    actual_rows, encoded_inputs = [], []
    original_write, original_encode = module.write_batch, method.reader.encode_evidence

    def capture_write(writer, state, hidden, rows, mask):
        actual_rows.append(rows.detach().clone())
        assert hidden.shape[1] == mask.shape[1] == 8
        return original_write(writer, state, hidden, rows, mask)

    def capture_encode(ids, mask):
        encoded_inputs.append(ids.detach().clone())
        return original_encode(ids, mask)

    monkeypatch.setattr(module, "write_batch", capture_write)
    monkeypatch.setattr(method.reader, "encode_evidence", capture_encode)
    trajectory_loss(method, [item], prefix_tokens=4)
    whole = torch.tensor([item.token_ids])
    reference = method.reader.adapter.addresses(whole, torch.ones_like(whole))
    assert len(actual_rows) == len(encoded_inputs) == 3
    for block, rows in enumerate(actual_rows):
        torch.testing.assert_close(rows, reference[:, block * 8:(block + 1) * 8])
        torch.testing.assert_close(encoded_inputs[block], whole[:, block * 8:(block + 1) * 8])


@pytest.mark.parametrize("short,long", [(2, 4), (8, 16)])
def test_mixed_length_batch_matches_independent_token_weighted_gradients(tmp_path, short, long):
    method, _ = create_runtime(config_for(tmp_path))
    items = [window(method, "abc xyz\n" * short, "short"), window(method, "xyz abc\n" * long, "long")]
    batch = trajectory_loss(method, items, prefix_tokens=4)
    batch.mean.backward()
    batch_grads = grads(method)
    method.writer.zero_grad(set_to_none=True)
    independent = [trajectory_loss(method, [item], prefix_tokens=4) for item in items]
    independent_mean = sum(item.loss_sum for item in independent) / sum(item.loss_tokens for item in independent)
    independent_mean.backward()
    torch.testing.assert_close(batch.mean, independent_mean)
    assert batch.loss_tokens == (short + long - 2) * 8
    assert_grads(batch_grads, grads(method))
    for packed, single in zip(batch.state.unpack(), independent):
        torch.testing.assert_close(packed.values, single.state.memory.values)
        torch.testing.assert_close(packed.counts, single.state.memory.counts)


def test_reader_recomputation_preserves_snapshot_gradients_and_hook_lifetime(tmp_path):
    method, _ = create_runtime(config_for(tmp_path))
    items = [window(method, "abc xyz\n" * 4), window(method, "xyz abc\n" * 2, "short")]
    layer = method.reader.model.model.layers[method.reader.layer_index]
    initial_hooks = len(layer._forward_pre_hooks)
    ordinary = trajectory_loss(method, items, prefix_tokens=4)
    ordinary.mean.backward()
    expected_grads = grads(method)
    method.writer.zero_grad(set_to_none=True)
    replayed = trajectory_loss(method, items, prefix_tokens=4, checkpoint_reader=True)
    assert len(layer._forward_pre_hooks) == initial_hooks
    replayed.mean.backward()
    assert len(layer._forward_pre_hooks) == initial_hooks
    torch.testing.assert_close(ordinary.mean, replayed.mean, rtol=0, atol=0)
    assert_grads(grads(method), expected_grads, rtol=0, atol=0)
    assert all(p.grad is None for p in method.reader.model.parameters())


def assert_tree_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for value, expected in zip(left, right):
            assert_tree_equal(value, expected)
    else:
        assert left == right


@pytest.mark.parametrize("accumulation", [1, 2])
def test_cli_checkpoint_and_exact_optimizer_boundary_resume(tmp_path, accumulation):
    settings = {"checkpoint_reader": True, "gradient_accumulation_steps": accumulation}
    full = config_for(tmp_path, "full", raw=settings)
    assert main(["train-raw", "--config", str(full.source)]) == 0
    limited = config_for(tmp_path, "partial", raw={**settings, "max_steps": 2})
    partial = train_raw(limited)
    assert not partial["complete"] and partial["stop_reason"] == "step_limit"
    resumed = config_for(tmp_path, "resumed", raw={**settings, "resume": "partial/training.pt"})
    summary = train_raw(resumed)
    assert summary["complete"] and summary["counters"]["loss_tokens"] == 128
    assert not summary["last_evaluation"]["p6_gate_evaluated"]
    expected = torch.load(tmp_path / "full/training.pt", weights_only=True)
    restored = torch.load(tmp_path / "resumed/training.pt", weights_only=True)
    assert_tree_equal(expected, restored)
    payload = load_checkpoint(tmp_path / "resumed/writer.pt")
    method, _ = create_runtime(resumed)
    restore_writer(method, payload, resumed)
    assert_tree_equal(dict(method.writer.state_dict()), payload["writer_state_dict"])


def test_accumulation_matches_one_batch_and_small_final_group(tmp_path, monkeypatch):
    original_clip = torch.nn.utils.clip_grad_norm_
    observed = []

    def capture(parameters, *args, **kwargs):
        parameters = tuple(parameters)
        observed.append([p.grad.detach().clone() if p.grad is not None else None for p in parameters])
        return original_clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture)
    full = train_raw(config_for(tmp_path, "one_batch", raw={"batch_size": 4, "loss_tokens": 56}))
    full_gradients, observed[:] = observed[:], []
    accumulated = train_raw(config_for(tmp_path, "accumulated", raw={"gradient_accumulation_steps": 2,
                                                                    "loss_tokens": 56}))
    assert full["completed_steps"] == accumulated["completed_steps"] == 2
    assert full["counters"] == accumulated["counters"]
    assert len(observed) == len(full_gradients) == 2
    for actual, expected in zip(observed, full_gradients):
        assert_grads(actual, expected, rtol=1e-4, atol=2e-7)


def test_training_requires_explicit_pretokenized_identity(tmp_path):
    config = config_for(tmp_path)
    path = tmp_path / "documents.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record.pop("text")
        record["token_ids"] = [3, 4, 5, 6] * 16
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    with pytest.raises(ValueError, match="must carry tokenizer_identity"):
        train_raw(config)


@pytest.mark.parametrize("change", ["seed", "corpus"])
def test_resume_rejects_changed_identity(tmp_path, change):
    train_raw(config_for(tmp_path, "initial", raw={"max_steps": 1}))
    if change == "corpus":
        path = tmp_path / "documents.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace("abc", "def"), encoding="utf-8")
    config = config_for(tmp_path, "rejected", seed=18 if change == "seed" else 17,
                        raw={"resume": "initial/training.pt"})
    with pytest.raises(ValueError, match="identity differs"):
        train_raw(config)
    assert not (tmp_path / "rejected").exists()


def test_budget_never_overshoots_or_truncates_trajectory(tmp_path):
    summary = train_raw(config_for(tmp_path, raw={"loss_tokens": 19}))
    assert summary["counters"]["loss_tokens"] == 16
    assert summary["unspent_loss_tokens"] == 3
    assert summary["stop_reason"] == "remaining_budget_below_complete_trajectory"
    assert not summary["complete"]


@pytest.mark.parametrize("raw", [{"prefix_tokens": 9}, {"block_counts": [1]},
                                 {"batch_size": True}, {"checkpoint_reader": 1},
                                 {"length_weights": {2: 1}}, {"max_steps": 0}])
def test_raw_config_rejects_invalid_schedule(tmp_path, raw):
    with pytest.raises(ValueError, match="raw\\."):
        config_for(tmp_path, raw=raw)

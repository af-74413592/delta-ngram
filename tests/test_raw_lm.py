"""Causality, full gradients, and exact CPU resume through the raw LM path."""
import json
from pathlib import Path
import random
import sys

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_ngram.checkpoints import load_checkpoint, restore_writer
from delta_ngram.cli import main
from delta_ngram.config import ExperimentConfig
from delta_ngram.data.raw_text import TrajectoryWindow
from delta_ngram.raw_lm import (evaluate_raw, select_validation_windows,
                                summarize_group_losses, train_raw, trajectory_loss)
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


@pytest.mark.parametrize("change", [{"eval_seed": 104730}, {"eval_windows": 3},
                                   {"eval_bootstrap_samples": 2001}])
def test_resume_rejects_changed_fixed_dev_protocol(tmp_path, change):
    train_raw(config_for(tmp_path, "initial", raw={"max_steps": 1}))
    config = config_for(tmp_path, "rejected", raw={"resume": "initial/training.pt", **change})
    with pytest.raises(ValueError, match="identity differs"):
        train_raw(config)
    assert not (tmp_path / "rejected").exists()


def test_budget_never_overshoots_or_truncates_trajectory(tmp_path):
    summary = train_raw(config_for(tmp_path, raw={"loss_tokens": 19}))
    assert summary["counters"]["loss_tokens"] == 16
    assert summary["unspent_loss_tokens"] == 3
    assert summary["stop_reason"] == "remaining_budget_below_complete_trajectory"
    assert not summary["complete"]
    assert summary["budget_endpoint"] and summary["nominal_loss_tokens"] == 19
    assert summary["last_evaluation"]["training_loss_tokens"] == 16
    assert summary["last_evaluation"]["final_nominal_loss_tokens"] == 19
    assert summary["last_evaluation"]["unspent_loss_tokens"] == 3


def test_fixed_dev_selection_covers_groups_before_extra_windows():
    items = [TrajectoryWindow(f"doc-{group}", f"group-{group}", "code", offset,
                              tuple(range(16)), 2)
             for group in range(80) for offset in range(160 if group == 0 else 2)]
    before_python, before_torch = random.getstate(), torch.get_rng_state().clone()
    chosen = select_validation_windows(items, limit=128, seed=104729)
    assert len(chosen) == 128
    assert len({item.group_id for item in chosen[:80]}) == 80
    assert chosen == select_validation_windows(list(reversed(items)), limit=128, seed=104729)
    assert chosen != select_validation_windows(items, limit=128, seed=104730)
    assert random.getstate() == before_python
    assert torch.equal(torch.get_rng_state(), before_torch)


def test_fixed_dev_selection_reserves_independent_groups_for_every_source():
    items = [TrajectoryWindow(f"{source}-doc-{group}", f"{source}-group-{group}", source, offset,
                              tuple(range(16)), 2)
             for source in ("code", "papers", "books", "web", "news")
             for group in range(100 if source == "code" else 64)
             for offset in range(4 if source == "code" else 1)]
    before_python, before_torch = random.getstate(), torch.get_rng_state().clone()
    chosen = select_validation_windows(items, limit=340, seed=104729, min_groups_per_source=64)
    assert len(chosen) == 340
    assert len({(item.group_id, item.start) for item in chosen}) == 340
    for source in {item.source for item in items}:
        assert len({item.group_id for item in chosen[:320] if item.source == source}) == 64
    assert len({item.group_id for item in chosen}) == 340
    assert chosen == select_validation_windows(list(reversed(items)), limit=340, seed=104729,
                                               min_groups_per_source=64)
    assert random.getstate() == before_python
    assert torch.equal(torch.get_rng_state(), before_torch)


@pytest.mark.parametrize("limit,minimum,required,match", [
    (127, 64, ["code", "news"], "window limit"),
    (200, 65, ["code", "news"], "requires 65"),
    (200, 64, ["code", "missing"], "missing.*has 0 groups"),
    (200, True, ["code", "news"], "nonnegative integer"),
])
def test_fixed_dev_rejects_insufficient_source_coverage(limit, minimum, required, match):
    items = [TrajectoryWindow(f"{source}-{group}", f"{source}-{group}", source, 0,
                              tuple(range(16)), 2)
             for source in ("code", "news") for group in range(64)]
    with pytest.raises(ValueError, match=match):
        select_validation_windows(items, limit=limit, seed=17,
                                  min_groups_per_source=minimum, required_sources=required)


def test_per_source_dev_rejects_cross_source_groups():
    items = [TrajectoryWindow(source, "shared-group", source, 0, tuple(range(16)), 2)
             for source in ("code", "news")]
    with pytest.raises(ValueError, match="one source per group_id"):
        select_validation_windows(items, limit=2, seed=17, min_groups_per_source=1)


def test_group_bootstrap_uses_paired_ratio_of_sums_and_private_rng():
    groups = [{"group_id": "short", "self_loss_sum": 1., "empty_loss_sum": 3., "loss_tokens": 1},
              {"group_id": "long", "self_loss_sum": 18., "empty_loss_sum": 9., "loss_tokens": 9}]
    before = random.getstate()
    result = summarize_group_losses(groups, bootstrap_seed=4, bootstrap_samples=2000)
    assert result["delta_empty"] == pytest.approx(-.7)  # Not the unweighted group mean (+.5).
    assert result["paired_group_bootstrap"]["ci_lower"] == -1.
    assert result["paired_group_bootstrap"]["ci_upper"] == 2.
    assert result == summarize_group_losses(list(reversed(groups)), bootstrap_seed=4, bootstrap_samples=2000)
    assert random.getstate() == before
    assert result["group_count"] == 2
    assert result["p1_point_check"]["status"] == "insufficient_groups"
    assert result["p1_point_check"]["passed"] is None
    assert result["p1_gate_passed"] is False and result["p6_gate_evaluated"] is False


def test_positive_group_ci_only_passes_one_checkpoint_not_overall_gate():
    groups = [{"group_id": str(index), "self_loss_sum": 100., "empty_loss_sum": 100.2,
               "loss_tokens": 100} for index in range(64)]
    result = summarize_group_losses(groups)
    assert result["delta_empty"] == pytest.approx(.002)
    assert result["paired_group_bootstrap"]["ci_lower"] > 0
    assert result["p1_point_check"]["passed"] is True
    assert result["p1_gate_passed"] is False
    assert summarize_group_losses(groups[:-1])["p1_point_check"]["passed"] is None


def test_evaluation_retains_paired_group_sums(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import delta_ngram.raw_lm as module
    method, _ = create_runtime(config_for(tmp_path))
    items = [TrajectoryWindow("a", "group-a", "code", 0, tuple(range(16)), 2),
             TrajectoryWindow("b", "group-a", "code", 0, tuple(range(32)), 4),
             TrajectoryWindow("c", "group-c", "code", 0, tuple(range(16)), 2)]

    def fake_trajectory(method, batch, *, prefix_tokens, memory_mode="self"):
        assert len({item.group_id for item in batch}) == 1
        count = sum(item.loss_tokens for item in batch)
        return SimpleNamespace(loss_tokens=count, loss_sum=torch.tensor(float(count * (2 if memory_mode == "empty" else 1))))

    monkeypatch.setattr(module, "trajectory_loss", fake_trajectory)
    result = evaluate_raw(method, items, prefix_tokens=4, batch_size=2)
    assert result["loss_tokens"] == 40 and result["delta_empty"] == 1
    assert result["groups"] == [
        {"group_id": "group-a", "document_ids": ["a", "b"], "windows": 2,
         "self_loss_sum": 32., "empty_loss_sum": 64., "loss_tokens": 32},
        {"group_id": "group-c", "document_ids": ["c"], "windows": 1,
         "self_loss_sum": 8., "empty_loss_sum": 16., "loss_tokens": 8}]


def test_evaluation_reports_each_source_and_keeps_paired_token_weighting(monkeypatch):
    from types import SimpleNamespace
    import delta_ngram.raw_lm as module
    method = SimpleNamespace(eval=lambda: None)
    items = [TrajectoryWindow("a", "code-a", "code", 0, tuple(range(16)), 2),
             TrajectoryWindow("b", "code-a", "code", 16, tuple(range(32)), 4),
             TrajectoryWindow("c", "news-c", "news", 0, tuple(range(16)), 2)]

    def fake_trajectory(method, batch, *, prefix_tokens, memory_mode="self"):
        assert len({item.source for item in batch}) == 1
        count = sum(item.loss_tokens for item in batch)
        own = 1 if batch[0].source == "code" else 3
        return SimpleNamespace(loss_tokens=count,
                               loss_sum=torch.tensor(float(count * (2 if memory_mode == "empty" else own))))

    monkeypatch.setattr(module, "trajectory_loss", fake_trajectory)
    result = evaluate_raw(method, items, prefix_tokens=4, batch_size=2)
    assert result["loss_tokens"] == 40
    assert result["delta_empty"] == pytest.approx(.6)
    assert result["groups"][0]["source"] == "code"
    assert result["per_source"]["code"]["delta_empty"] == 1
    assert result["per_source"]["news"]["delta_empty"] == -1
    assert result["source_macro_delta_empty"] == 0
    for source, tokens, delta in (("code", 32, 1), ("news", 8, -1)):
        summary = result["per_source"][source]
        assert summary["group_count"] == 1 and summary["loss_tokens"] == tokens
        assert summary["paired_group_bootstrap"]["ci_lower"] == delta
        assert summary["paired_group_bootstrap"]["ci_upper"] == delta
    single = evaluate_raw(method, items[:2], prefix_tokens=4, batch_size=2, report_per_source=True)
    assert single["per_source"]["code"]["delta_empty"] == single["delta_empty"]
    crossed = [items[0], TrajectoryWindow("crossed", "code-a", "news", 0, tuple(range(16)), 2)]
    with pytest.raises(ValueError, match="one source per group_id"):
        evaluate_raw(method, crossed, prefix_tokens=4, batch_size=2)


def test_multisource_training_wires_quotas_dev_and_exact_resume(tmp_path):
    documents = [{"id": f"{source}-{split}-{index}", "group_id": f"{source}-{split}-{index}",
                  "source": source, "split": split, "text": "abc xyz\n" * 8}
                 for source in ("code", "news") for split in ("train", "validation") for index in range(2)]
    (tmp_path / "documents.jsonl").write_text(
        "\n".join(json.dumps(record) for record in documents), encoding="utf-8")
    settings = {"source_weights": {"code": 1., "news": 1.},
                "source_length_weights": {"code": {4: 1.}, "news": {2: 1.}},
                "eval_min_groups_per_source": 2, "eval_windows": 4}
    full = train_raw(config_for(tmp_path, "full", raw=settings))
    train_raw(config_for(tmp_path, "partial", raw={**settings, "max_steps": 2}))
    train_raw(config_for(tmp_path, "resumed", raw={**settings, "resume": "partial/training.pt"}))
    expected = torch.load(tmp_path / "full/training.pt", weights_only=True)
    restored = torch.load(tmp_path / "resumed/training.pt", weights_only=True)
    assert_tree_equal(expected, restored)
    assert expected["identity"]["raw"]["source_length_weights"] == settings["source_length_weights"]
    assert expected["identity"]["raw"]["eval_min_groups_per_source"] == 2
    manifest = json.loads((tmp_path / "full/data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_validation_groups_per_source"] == {"code": 2, "news": 2}
    assert {(item["source"], item["blocks"]) for item in manifest["validation_sample"]} == {
        ("code", 4), ("news", 2)}
    assert set(full["last_evaluation"]["per_source"]) == {"code", "news"}
    changed = config_for(tmp_path, "rejected", raw={**settings, "eval_min_groups_per_source": 1,
                                                   "resume": "partial/training.pt"})
    with pytest.raises(ValueError, match="identity differs"):
        train_raw(changed)
    assert not (tmp_path / "rejected").exists()


def test_default_raw_identity_keeps_new_options_absent(tmp_path):
    train_raw(config_for(tmp_path, "legacy", raw={"max_steps": 1}))
    checkpoint = torch.load(tmp_path / "legacy/training.pt", weights_only=True)
    assert "source_length_weights" not in checkpoint["identity"]["raw"]
    assert "eval_min_groups_per_source" not in checkpoint["identity"]["raw"]
    manifest = json.loads((tmp_path / "legacy/data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation_protocol"]["selection"] == "group_first_round_robin"
    assert "minimum_groups_per_source" not in manifest["validation_protocol"]


def test_short_checkpoints_preserve_training_and_milestone_resume(tmp_path, monkeypatch):
    import delta_ngram.raw_lm as module
    original_save = module._save_resume
    saves = []

    def capture(path, method, optimizer, sampler, identity, step, counters):
        saves.append((Path(path).name, step, counters["loss_tokens"]))
        return original_save(path, method, optimizer, sampler, identity, step, counters)

    monkeypatch.setattr(module, "_save_resume", capture)
    settings = {"eval_every_tokens": 48, "checkpoint_every_tokens": 16}
    frequent = train_raw(config_for(tmp_path, "frequent", raw=settings))
    assert any(name == "training.pt" and 0 < count < 48 for name, _, count in saves)
    snapshots = sorted((tmp_path / "frequent").glob("training_step_*_tokens_*.pt"))
    assert len(snapshots) >= 2
    first = torch.load(snapshots[0], weights_only=True)
    count = first["counters"]["loss_tokens"]
    assert 48 <= count < 128
    assert f"tokens_{count:012d}" in snapshots[0].name
    assert snapshots[0].with_name(snapshots[0].name.replace("training_", "writer_", 1)).exists()
    reference = torch.load(tmp_path / "frequent/training.pt", weights_only=True)
    train_raw(config_for(tmp_path, "follow_eval", raw={**settings, "checkpoint_every_tokens": None}))
    assert_tree_equal(reference, torch.load(tmp_path / "follow_eval/training.pt", weights_only=True))
    train_raw(config_for(tmp_path, "resume_snapshot", raw={**settings, "checkpoint_every_tokens": 17,
                                                          "resume": str(snapshots[0])}))
    assert_tree_equal(reference, torch.load(tmp_path / "resume_snapshot/training.pt", weights_only=True))
    manifest = json.loads((tmp_path / "frequent/data_manifest.json").read_text())
    assert manifest["validation_protocol"]["seed"] == 104729
    assert len(manifest["validation_sample_sha256"]) == 64
    assert reference["identity"]["validation"] == {
        key: manifest[key] for key in ("validation_protocol_sha256", "validation_sample_sha256")}
    assert manifest["selected_validation_groups"] == frequent["last_evaluation"]["group_count"]
    assert frequent["last_evaluation"]["validation_sample_sha256"] == manifest["validation_sample_sha256"]


@pytest.mark.parametrize("raw", [{"prefix_tokens": 9}, {"block_counts": [1]},
                                 {"batch_size": True}, {"checkpoint_reader": 1},
                                 {"length_weights": {2: 1}}, {"max_steps": 0},
                                 {"eval_seed": True}, {"eval_bootstrap_samples": 0},
                                 {"checkpoint_every_tokens": 0}])
def test_raw_config_rejects_invalid_schedule(tmp_path, raw):
    with pytest.raises(ValueError, match="raw\\."):
        config_for(tmp_path, raw=raw)

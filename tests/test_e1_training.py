"""E1 ability balancing and unchanged legacy optimization through one loop."""
import importlib
import json
from pathlib import Path
import random
import sys

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_ngram.config import ExperimentConfig
from delta_ngram.data import episode_queries, sample_training_episode
from delta_ngram.runtime import answer_loss
from delta_ngram.train import (create_runtime, e1_balanced_query_loss,
                               e1_query_loss_weights, make_manifest)

training = importlib.import_module("delta_ngram.train")


def write_config(tmp_path, name, *, task="fact_e1", scenario="revision", writes=4,
                 batch=1, styles=("canonical", "named_question")):
    config = {"seed": 17, "model": {"smoke": True, "device": "cpu"},
              "reader": {"buckets": 97}, "writer": {"hidden_dim": 12},
              "data": {"task": task, "e1_scenario": scenario, "facts_per_episode": writes, "eval_episodes": 1},
              "train": {"steps": 1, "batch_episodes": batch, "eval_every": 1},
              "evaluation": {"max_new_tokens": 1, "openbook_episodes": 0, "counterfactuals": False,
                             "query_styles": list(styles)},
              "run": {"output_dir": name, "max_wall_seconds": 90, "max_evidence_tokens": 128}}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return ExperimentConfig.load(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def training_record(directory):
    return next(event for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if (event := json.loads(line))["event"] == "train")


def test_e1_balances_abilities_and_keeps_query_gradients():
    abilities = ("acquisition", "retention", "acquisition", "retention", "revision")
    losses = [torch.tensor(value, requires_grad=True) for value in (1., 9., 3., 15., 27.)]
    loss = e1_balanced_query_loss(losses, abilities)
    assert float(loss.detach()) == pytest.approx((2 + 12 + 27) / 3)
    loss.backward()
    assert [float(value.grad) for value in losses] == pytest.approx(e1_query_loss_weights(abilities))
    assert e1_query_loss_weights(abilities) == pytest.approx((1/6, 1/6, 1/6, 1/6, 1/3))


def test_e1_missing_abilities_do_not_reserve_weight():
    assert e1_query_loss_weights(("acquisition",)) == (1.,)
    abilities = ("acquisition", "acquisition", "acquisition", "retention")
    assert e1_query_loss_weights(abilities) == pytest.approx((1/6, 1/6, 1/6, 1/2))
    assert e1_balanced_query_loss([torch.tensor(7.)], ("revision",)).item() == 7
    with pytest.raises(ValueError, match="nonempty"):
        e1_query_loss_weights(())
    with pytest.raises(ValueError, match="abilities"):
        e1_query_loss_weights(("unknown",))
    with pytest.raises(ValueError, match="align"):
        e1_balanced_query_loss([torch.tensor(1.)], ())


@pytest.mark.parametrize("scenario,writes", [("independent", 3), ("revision", 4)])
def test_e1_smoke_uses_canonical_training_and_balanced_metadata(tmp_path, monkeypatch, scenario, writes):
    config = write_config(tmp_path, scenario, scenario=scenario, writes=writes)
    observed_questions, evaluations = [], []
    original_answer_loss, original_evaluate = training.answer_loss, training.evaluate

    def record_answer_loss(method, state, question, answer):
        observed_questions.append(question)
        return original_answer_loss(method, state, question, answer)

    def record_evaluate(*args, **kwargs):
        evaluations.append(kwargs)
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(training, "answer_loss", record_answer_loss)
    monkeypatch.setattr(training, "evaluate", record_evaluate)
    summary = training.train(config)
    assert summary["complete"] and summary["completed_steps"] == 1
    assert observed_questions and all(question.startswith("Profile\nName:") for question in observed_questions)
    assert all(call["query_styles"] == ("canonical", "named_question") for call in evaluations)
    assert all(call["detailed_rows"] is False for call in evaluations)
    out = Path(config.values["run"]["output_dir"])
    record = training_record(out)
    assert record["grad_norm_before_clip"] > 0
    metadata = read_json(out / "config.json")["query_loss_weights"]
    assert metadata["training_query_style"] == "canonical"
    manifest = make_manifest(config)
    episode = sample_training_episode(manifest, random.Random(manifest["training_rng_seed"]), "train-000001-0")
    schedule = [(f"write{stage}_q{query.index+1}", query.ability)
                for stage in range(1, writes + 1) for query in episode_queries(episode, stage)]
    labels = [ability for _, ability in schedule]
    assert len(observed_questions) == len(schedule)
    assert list(metadata["per_query_weights"].values()) == pytest.approx(e1_query_loss_weights(labels))
    expected = e1_balanced_query_loss([torch.tensor(record[f"{key}_nll"]) for key, _ in schedule], labels)
    assert record["answer_nll"] == float(expected)
    assert sum(metadata["per_query_weights"].values()) == pytest.approx(1)
    if scenario == "revision":
        assert metadata["query_counts"] == {"acquisition": 3, "retention": 4, "revision": 1}
        assert "write3_q1_nll" not in record and "write4_q1_nll" not in record
    else:
        assert "revision" not in metadata["ability_weights"]
    initial = torch.load(out / "writer_init.pt", weights_only=True)["writer_state_dict"]
    trained = torch.load(out / "writer.pt", weights_only=True)["writer_state_dict"]
    assert any(not torch.equal(initial[key], trained[key]) for key in initial)


def test_legacy_training_weights_and_answer_nll_match_independent_historical_replay(tmp_path, monkeypatch):
    config = write_config(tmp_path, "legacy", task="fact", scenario="independent", writes=2,
                          batch=2, styles=("canonical",))
    # A legacy run must never call the E1 objective, even though queries carry abilities.
    monkeypatch.setattr(training, "e1_balanced_query_loss", lambda *args: pytest.fail("legacy used E1 weighting"))
    assert training.train(config)["complete"]
    out = Path(config.values["run"]["output_dir"])
    recorded = training_record(out)
    expected_weights = torch.load(out / "writer.pt", weights_only=True)["writer_state_dict"]
    trained_rng = torch.random.get_rng_state().clone()
    assert read_json(out / "config.json")["query_loss_weights"] == {
        "write1_q1": 1/3, "write2_q2": 1/3, "write2_q1": 1/3}

    # Reconstruct the executed two-write pilot schedule directly. This deliberately
    # does not use the generalized episode_queries schedule or E1 loss helpers.
    method, args = create_runtime(config)
    optimizer = torch.optim.AdamW(method.parameters(), lr=args.lr, weight_decay=0.0)
    manifest = make_manifest(config)
    rng = random.Random(manifest["training_rng_seed"])
    method.train()
    optimizer.zero_grad(set_to_none=True)
    component_rows = []
    for index in range(args.batch_episodes):
        episode = sample_training_episode(manifest, rng, f"train-000001-{index}")
        first = method.write(method.empty_state(), episode.evidence(0)).state
        first_loss = answer_loss(method, first, episode.question(0), episode.answer(0))
        final = method.write(first, episode.evidence(1)).state
        new_loss = answer_loss(method, final, episode.question(1), episode.answer(1))
        old_loss = answer_loss(method, final, episode.question(0), episode.answer(0))
        loss = (first_loss + new_loss + old_loss) / 3
        (loss / args.batch_episodes).backward()
        component_rows.append([float(value.detach()) for value in (first_loss, new_loss, old_loss)])
    norm = torch.nn.utils.clip_grad_norm_(method.parameters(), config.values["train"]["gradient_clip"],
                                         error_if_nonfinite=True)
    optimizer.step()
    assert float(norm) == recorded["grad_norm_before_clip"]
    assert recorded["answer_nll"] == sum(sum(row) for row in component_rows) / (3 * len(component_rows))
    assert all(torch.equal(value, expected_weights[key]) for key, value in method.writer.state_dict().items())
    assert torch.equal(torch.random.get_rng_state(), trained_rng)

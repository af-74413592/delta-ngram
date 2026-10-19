"""CPU-only protocol tests, using explicit fake facts (not actual news claims)."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_ngram.interfaces import WriteResult
from delta_ngram.news_eval import evaluate_news, score_candidates, validate_manifest
from delta_ngram.runtime import CharacterTokenizer
from delta_ngram.state import SparseMemoryState


def manifest_fixture():
    episodes = []
    for name, value, wrong in (("alpha", "red", "blue"), ("beta", "blue", "red")):
        episodes.append({
            "id": name, "event_family": name, "split": "dev", "kind": "synthetic_swap",
            "synthetic": True, "transformation": "Fictional colors for offline protocol testing.",
            "donor_episode_id": "beta" if name == "alpha" else "alpha",
            "sources": [{"id": "s", "url": f"https://example.org/{name}", "title": "Fixture only",
                         "published_at": "2026-10-01", "retrieved_at": "2026-10-02"}],
            "history": [{"id": "h1", "source_ids": ["s"], "available_at": "2026-10-01",
                         "text": f"{name} has color {value}.",
                         "facts": [{"id": "f1", "subject": name, "relation": "color", "value": value}]}],
            "questions": [{"id": "q1", "question": f"Which color belongs to {name}?",
                           "after_history": 1, "asked_at": "2026-10-02", "fact_id": "f1",
                           "candidates": [value, wrong], "answer_index": 0}]})
    return {"protocol": "news_forward_v1", "evaluation_only": True,
            "model": {"name_or_path": "fixture", "revision": "a" * 40,
                      "revision_date": "2026-01-01", "training_cutoff": "unknown"},
            "training_event_families": [], "episodes": episodes}


class UniformReader:
    def __init__(self, events, vocab_size):
        self.model = torch.nn.Linear(1, 1)
        self.events, self.vocab_size = events, vocab_size

    def __call__(self, ids, mask, state):
        self.events.append(("read", float(state.values.sum()), ids.clone()))
        assert not torch.is_grad_enabled()
        return SimpleNamespace(logits=torch.zeros((*ids.shape, self.vocab_size)))


class FakeForward:
    name = "forward"
    supports_outer_training = True

    def __init__(self):
        self.events = []
        self.tokenizer, self.device = CharacterTokenizer(), torch.device("cpu")
        self.reader = UniformReader(self.events, self.tokenizer.vocab_size)
        self.writer = torch.nn.Linear(1, 1)

    def eval(self):
        self.writer.eval()

    def empty_state(self):
        return SparseMemoryState.empty(1)

    def memory(self, state):
        return state

    def write(self, state, evidence):
        self.events.append(("write", evidence))
        assert not torch.is_grad_enabled()
        value = float(state.values.sum()) + 1
        return WriteResult(SparseMemoryState(torch.tensor([1]), torch.tensor([[value]]), torch.tensor([value])))


def test_schema_and_cli_validation_without_model_or_training(tmp_path):
    manifest = manifest_fixture()
    assert validate_manifest(manifest)["by_split"] == {"dev": 2, "final_holdout": 0}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/news_eval.py"),
                             "--manifest", str(path), "--validate-only"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "valid"


@pytest.mark.parametrize("mutation,match", [
    (lambda m: m.update(evaluation_only=False), "evaluation_only"),
    (lambda m: m["episodes"][0].update(split="train"), "evaluation-only"),
    (lambda m: m["training_event_families"].append("alpha"), "training event family"),
    (lambda m: m["episodes"][0].update(donor_episode_id="alpha"), "different event family"),
    (lambda m: m["episodes"][0].update(kind="real"), "synthetic flag"),
    (lambda m: m["model"].update(revision="main"), "immutable"),
    (lambda m: m["episodes"][0]["questions"][0].update(after_history=0), "prior history"),
    (lambda m: m["episodes"][0]["questions"][0].update(asked_at="2026-09-30"), "future history"),
    (lambda m: m["episodes"][0]["questions"][0].update(fact_id="future"), "already be written"),
    (lambda m: m["episodes"][0]["questions"][0].update(answer_index=1), "supported fact"),
    (lambda m: m["episodes"][0]["history"][0].update(text="Which color belongs to alpha? red"), "question text leaked"),
])
def test_rejects_leakage_and_invalid_protocol(mutation, match):
    manifest = manifest_fixture()
    mutation(manifest)
    with pytest.raises(ValueError, match=match):
        validate_manifest(manifest)


def test_rejects_event_family_and_source_url_cross_split():
    manifest = manifest_fixture()
    other = copy.deepcopy(manifest["episodes"][0])
    other.update(id="alpha2", split="final_holdout")
    manifest["episodes"].append(other)
    with pytest.raises(ValueError, match="event family crosses"):
        validate_manifest(manifest)
    other["event_family"] = "different-family"
    with pytest.raises(ValueError, match="source URL crosses"):
        validate_manifest(manifest)


def test_rejects_same_day_future_timestamps_and_future_fact_reference():
    manifest = manifest_fixture()
    episode = manifest["episodes"][0]
    episode["history"][0]["available_at"] = "2026-10-02T12:00:00Z"
    episode["questions"][0]["asked_at"] = "2026-10-02T11:00:00Z"
    with pytest.raises(ValueError, match="future history"):
        validate_manifest(manifest)


def test_fixed_donor_cannot_cross_split():
    manifest = manifest_fixture()
    manifest["episodes"][1]["split"] = "final_holdout"
    with pytest.raises(ValueError, match="donor must stay"):
        validate_manifest(manifest)


@pytest.mark.parametrize("updated", ["2026-09-30", "2026-10-03"])
def test_source_update_must_be_between_publication_and_retrieval(updated):
    manifest = manifest_fixture()
    manifest["episodes"][0]["sources"][0]["updated_at"] = updated
    with pytest.raises(ValueError, match="updated_at must lie between"):
        validate_manifest(manifest)


def test_source_update_cannot_be_used_before_available():
    manifest = manifest_fixture()
    episode = manifest["episodes"][0]
    episode["sources"][0]["updated_at"] = "2026-10-02"
    with pytest.raises(ValueError, match="before its source update"):
        validate_manifest(manifest)
    episode["history"][0]["available_at"] = "2026-10-02"
    assert validate_manifest(manifest)["episodes"] == 2


def add_revision(manifest):
    for episode in manifest["episodes"]:
        block = copy.deepcopy(episode["history"][0])
        block.update(id="h2", available_at="2026-10-02", text=episode["id"] + " now has color green.")
        block["facts"][0].update(id="f2", value="green")
        episode["history"].append(block)
        question = copy.deepcopy(episode["questions"][0])
        question.update(id="q2", after_history=2, fact_id="f2", candidates=["green", "red"])
        episode["questions"].append(question)
        episode.update(kind="synthetic_correction")


def test_rejects_superseded_fact_and_unwritten_future_fact():
    manifest = manifest_fixture()
    add_revision(manifest)
    question = manifest["episodes"][0]["questions"][1]
    question.update(fact_id="f1", candidates=["red", "green"])
    with pytest.raises(ValueError, match="superseded"):
        validate_manifest(manifest)
    question.update(fact_id="f2", after_history=1)
    with pytest.raises(ValueError, match="already be written"):
        validate_manifest(manifest)


def test_evidence_written_before_question_and_all_conditions_pair_prefixes():
    method, manifest = FakeForward(), manifest_fixture()
    add_revision(manifest)
    result = evaluate_news(method, manifest)
    assert method.events[0][:2] == ("write", "alpha has color red.")
    assert method.events[1][:2] == ("write", "beta has color blue.")
    assert method.events[2][:2] == ("read", 1.0)
    writes = [event[1] for event in method.events if event[0] == "write"]
    assert len(writes) == 8 and all("?" not in text and "Answer:" not in text for text in writes)
    for row in result["records"]:
        conditions = row["conditions"]
        assert len({conditions[c]["prefix_token_sha256"] for c in ("self", "empty", "donor")}) == 1
        assert conditions["context"]["prefix_tokens"] > conditions["self"]["prefix_tokens"]
        assert len({conditions[c]["gold_answer_tokens"] for c in conditions}) == 1
    assert result["claims"]["optimizer_steps"] == 0
    assert all(not p.requires_grad and p.grad is None for p in method.writer.parameters())
    # The second query sees both writes, while the first only sees one.
    assert [event[1] for event in method.events if event[0] == "read"][:16] == [1., 1., 0., 0., 1., 1., 0., 0., 2., 2., 0., 0., 2., 2., 0., 0.]


def test_answer_only_loss_and_denominator_ignore_question_and_context_lengths():
    method = FakeForward()
    short = score_candidates(method, method.empty_state(), "Q\nAnswer:", ["red", "long blue"])
    long = score_candidates(method, method.empty_state(), "many words in context " * 9 + "Q\nAnswer:", ["red", "long blue"])
    for left, right in zip(short["candidates"], long["candidates"]):
        assert left["answer_tokens"] == len(" " + left["answer"])
        assert left["answer_tokens"] == right["answer_tokens"]
        assert left["nll"] == pytest.approx(math.log(method.tokenizer.vocab_size), abs=1e-6)
        assert left["nll_sum"] == right["nll_sum"]


def test_all_candidate_lengths_are_checked_before_any_reader_forward():
    method = FakeForward()
    prefix = "Q\nAnswer:"
    # The first candidate fits exactly; only the second candidate exceeds it.
    method.reader.model.config = SimpleNamespace(max_position_embeddings=len(prefix + " red"))
    with pytest.raises(ValueError, match="including candidate.*not truncating"):
        score_candidates(method, method.empty_state(), prefix, ["red", "long blue"])
    assert method.events == []
    result = score_candidates(method, method.empty_state(), prefix, ["red", "x"])
    assert result["model_position_limit"] == len(prefix + " red")


def test_context_full_input_counts_toward_model_capacity():
    method = FakeForward()
    prefix = "Q\nAnswer:"
    method.reader.model.config = SimpleNamespace(max_position_embeddings=len(prefix + " red"))
    with pytest.raises(ValueError, match="max_position_embeddings"):
        score_candidates(method, method.empty_state(), "history\n\n" + prefix, ["red", "x"])
    assert method.events == []


@pytest.mark.parametrize("limit", [None, True, 0, -1, 12.0, "12", 10**30])
def test_invalid_or_sentinel_model_capacity_is_not_treated_as_a_limit(limit):
    method = FakeForward()
    method.reader.model.config = SimpleNamespace(max_position_embeddings=limit)
    result = score_candidates(method, method.empty_state(), "Q\nAnswer:", ["red", "blue"])
    assert result["model_position_limit"] is None


@pytest.mark.parametrize("mutation", ["data", "parameter_replacement", "buffer_data", "new_buffer"])
def test_exact_writer_snapshot_catches_changes_that_parameter_versions_miss(mutation):
    method = FakeForward()
    method.writer.register_buffer("audit_buffer", torch.tensor([1.]))
    old_weight = method.writer.weight
    original_version = old_weight._version

    def mutate(_):
        if mutation == "data":
            method.writer.weight.data.add_(1)
        elif mutation == "parameter_replacement":
            method.writer.weight = torch.nn.Parameter(method.writer.weight.detach() + 1, requires_grad=False)
        elif mutation == "buffer_data":
            method.writer.audit_buffer.data.add_(1)
        else:
            method.writer.register_buffer("new_buffer", torch.tensor([2.]))

    with pytest.raises(RuntimeError, match="writer state_dict changed"):
        evaluate_news(method, manifest_fixture(), progress_callback=mutate)
    assert old_weight._version == original_version


def test_aggregate_denominator_uses_all_answer_tokens_not_question_count():
    result = evaluate_news(FakeForward(), manifest_fixture())
    summary = result["summary_by_kind"]["synthetic_swap"]["conditions"]["self"]
    assert summary["questions"] == 2 and summary["gold_answer_tokens"] == len(" red") + len(" blue")
    assert summary["gold_nll"] == pytest.approx(math.log(CharacterTokenizer().vocab_size), abs=1e-6)


def test_checkpoint_metadata_must_bind_manifest_model():
    with pytest.raises(ValueError, match="bind the manifest model"):
        evaluate_news(FakeForward(), manifest_fixture(), checkpoint_metadata={"model": {"revision": "other"}})


def test_real_forward_writer_and_frozen_reader_cpu_integration(tmp_path):
    from delta_ngram.config import ExperimentConfig
    from delta_ngram.train import create_runtime
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text("model:\n  smoke: true\n  device: cpu\nreader:\n  buckets: 97\nwriter:\n  hidden_dim: 12\n", encoding="utf-8")
    method, _ = create_runtime(ExperimentConfig.load(config_path))
    initial = {key: value.clone() for key, value in method.writer.state_dict().items()}
    result = evaluate_news(method, manifest_fixture())
    assert result["complete"] and len(result["records"]) == 2
    assert result["written_tokens"]["self"] > 0
    assert result["checkpoint_identity_supplied"] is False
    assert result["integrity_checks"]["writer_state_dict_exact_equal"] is True
    assert result["integrity_checks"]["reader_content_hash_verified"] is False
    assert all(torch.equal(initial[key], value) for key, value in method.writer.state_dict().items())
    assert all(math.isfinite(row["conditions"]["self"]["gold_nll_sum"]) for row in result["records"])

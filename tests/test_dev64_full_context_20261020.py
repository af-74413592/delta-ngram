"""CPU checks for exact target matching, causal alignment and length rejection."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

SPEC = importlib.util.spec_from_file_location("dev64_context",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_dev64_full_context_20261020.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def example():
    tokens = {"A": (11, 12, 13), "B": tuple(range(600))}
    tokens.update({"D" + str(i): (1000 + i,) for i in range(16)})
    pair = {"id": "one", "event_family": "family", "domain": "test",
        "earlier_source_id": "A", "later_source_id": "B",
        "distractor_source_ids": ["D" + str(i) for i in range(16)]}
    pair["interference_conditions"] = [{"unrelated_article_count": d,
        "distractor_source_ids": pair["distractor_source_ids"][:d]} for d in runner.DISTANCES]
    return SimpleNamespace(registry={"pairs": [pair]}, tokens=tokens)


def test_full_articles_preserve_target_and_rolling_prefix_identity():
    prepared = example()
    plans = runner.make_plans(prepared, (271,), 40960)
    assert len(plans) == 5
    for plan in plans:
        distance = plan["unrelated_article_count"]
        assert plan["_evidence_ids"][:4] == (11, 12, 13, 271)
        assert plan["context_source_ids"] == ["A"] + ["D" + str(i) for i in range(distance)]
        assert plan["target_tokens"] == 472
        assert [(w["query_context_start_token"], w["target_start_token"], w["target_end_token_exclusive"])
                for w in plan["windows"]] == [(0, 128, 512), (384, 512, 600)]
        assert [w["target_tokens"] for w in plan["windows"]] == [384, 88]
        assert plan["windows"][1]["query_prefix_token_ids_sha256"] == runner.canonical_sha(tuple(range(384, 512)))
        assert plan["windows"][1]["target_token_ids_sha256"] == runner.canonical_sha(tuple(range(512, 600)))
        assert plan["target_token_ids_sha256"] == runner.canonical_sha(tuple(range(128, 600)))


def test_long_context_rejected_without_truncation():
    with pytest.raises(ValueError, match="refusing to truncate"):
        runner.make_plans(example(), (271,), 515)


def test_target_article_cannot_enter_evidence():
    prepared = example()
    prepared.registry["pairs"][0]["earlier_source_id"] = "B"
    with pytest.raises(ValueError, match="Target article"):
        runner.make_plans(prepared, (271,), 40960)


@pytest.fixture
def tiny_raw_model():
    torch.manual_seed(20261020)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=67, hidden_size=32,
        intermediate_size=48, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
        attention_dropout=0.0, use_cache=False))
    return model.eval().requires_grad_(False)


@pytest.mark.parametrize("targets", [1, 3, 9])
def test_selected_tail_logits_equal_masked_full_sequence_ce(tiny_raw_model, targets):
    ids = torch.tensor([[3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]])
    with torch.no_grad():
        logits = tiny_raw_model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                use_cache=False).logits
        labels = torch.full_like(ids, -100)
        labels[:, -targets:] = ids[:, -targets:]
        expected = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                   labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        actual = runner.selected_loss(tiny_raw_model, ids, targets)
    assert actual == pytest.approx(float(expected), abs=1e-6)


def test_future_tokens_cannot_change_earlier_target_prediction(tiny_raw_model):
    original = torch.tensor([[3, 5, 7, 11, 13, 17, 19, 23]])
    modified = original.clone()
    modified[:, 6:] = torch.tensor([[43, 47]])
    with torch.no_grad():
        original_logits = tiny_raw_model(input_ids=original, use_cache=False).logits
        modified_logits = tiny_raw_model(input_ids=modified, use_cache=False).logits
    # Position 4 predicts target at 5, before the perturbed future positions.
    torch.testing.assert_close(original_logits[:, :5], modified_logits[:, :5], rtol=0, atol=1e-7)
    assert not torch.equal(original_logits[:, -1], modified_logits[:, -1])


def test_position_policy_rejects_sliding_attention():
    config = SimpleNamespace(max_position_embeddings=40960, sliding_window=None,
                             use_sliding_window=False, layer_types=["full_attention"])
    assert runner.position_policy(config)["max_position_embeddings"] == 40960
    altered = deepcopy(config)
    altered.sliding_window = 4096
    with pytest.raises(ValueError, match="sliding-window"):
        runner.position_policy(altered)

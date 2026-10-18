"""Offline provenance and intervention checks using actual hash addressing."""
import copy
import importlib
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_ngram import diagnostics
from delta_ngram.data import Episode
from delta_ngram.interfaces import MemoryMethod
from delta_ngram.ngram_lookup import ngram_rows
from delta_ngram.runtime import CharacterTokenizer
from delta_ngram.state import SparseMemoryState

evaluate_module = importlib.import_module("delta_ngram.evaluate")


class FastCharacterTokenizer(CharacterTokenizer):
    is_fast = True

    def __init__(self):
        super().__init__()
        self.offset_calls = 0

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens and return_offsets_mapping
        self.offset_calls += 1
        return {"input_ids": self.encode(text),
                "offset_mapping": [(index, index + 1) for index in range(len(text))]}

    def convert_ids_to_tokens(self, ids):
        return [self.decode([value]) for value in ids]


class HashReader:
    backend = "external_hash"
    orders = (2, 3)
    width = 2

    def __init__(self, buckets):
        self.buckets = buckets

    def _inputs(self, ids, mask):
        return mask, ngram_rows(ids, mask, self.buckets, self.orders)


def setup_case(buckets=97, tokenizer=None, *, zero_values=False):
    episode = Episode("diagnostic", ("Kalibo", "Kalake", "Bobaba"), ("red", "blue", "green"))
    method = MemoryMethod(HashReader(buckets), tokenizer or FastCharacterTokenizer(), "cpu")
    row_sets = [evaluate_module.physical_rows(method, episode.evidence(index)) for index in range(3)]
    first_rows = row_sets[0]
    final_rows = torch.unique(torch.cat(row_sets))
    first_values = torch.stack((first_rows.float() / 10, first_rows.float() / 30), dim=-1)
    final_values = torch.stack((final_rows.float() / 4, final_rows.float() / 7), dim=-1)
    if zero_values:
        first_values.zero_()
        final_values.zero_()
    first = SparseMemoryState(first_rows, first_values, torch.ones_like(first_rows, dtype=torch.float32),
                              first_rows.float() + 1)
    final = SparseMemoryState(final_rows, final_values, torch.full_like(final_rows, 3, dtype=torch.float32),
                              final_rows.float() + 11)
    return method, episode, first, final


def generated_text(state, question):
    # Deterministic decoder surrogate, sensitive to the actual installed values.
    return f"{'red' if int(state.values[:, 0].sum()) % 2 else 'blue'}\n{state.values[:, 0].sum().item():.6f}"


def install_decoder(monkeypatch):
    calls = []

    def generate(method, state, question, max_new_tokens):
        assert max_new_tokens == 5
        calls.append((state, question))
        return generated_text(method.memory(state), question)

    monkeypatch.setattr(diagnostics, "greedy_generate", generate)
    monkeypatch.setattr(evaluate_module, "greedy_generate", generate)
    return calls


def test_actual_offsets_tuples_categories_and_multiple_later_writes(monkeypatch):
    method, episode, first, final = setup_case()
    install_decoder(monkeypatch)
    result = diagnostics.detailed_row_diagnostics(method, episode, first, final, 5)
    assert result["supported"]
    json.dumps(result, allow_nan=False)
    assert method.tokenizer.offset_calls == 5
    assert {row["source"] for row in result["tokenization"]} == {
        "evidence_1", "evidence_2", "evidence_3", "query_1", "query_3"}
    sources = {item["source"]: item for item in result["tokenization"]}
    assert all(item["offset_source"] == "fast_tokenizer_offsets" for item in sources.values())
    expected = set(first.rows.tolist()) & set(evaluate_module.physical_rows(method, episode.question(0)).tolist())
    later = set(torch.cat([evaluate_module.physical_rows(method, episode.evidence(i)) for i in (1, 2)]).tolist())
    assert result["selection"]["rows"] == sorted(expected & later)
    observed_categories = set()
    for row in result["rows"]:
        relevant = []
        for event in row["occurrences"]:
            source = sources[event["source"]]
            ids = torch.tensor([event["token_ids"]])
            address = ngram_rows(ids, torch.ones_like(ids), method.reader.buckets, method.reader.orders)
            assert int(address[0, -1, event["slot"]]) == row["row"]
            assert len(event["token_ids"]) == event["order"]
            assert event["tokens"] == [source["tokens"][i] for i in event["token_positions"]]
            assert event["offsets"] == [source["offsets"][i] for i in event["token_positions"]]
            start, end = event["text_span"]
            assert event["text"] == source["text"][start:end]
            observed_categories.add(event["category"])
            if (event["kind"] == "query" and event["index"] == 0) or (event["kind"] == "evidence" and event["index"] > 0):
                relevant.append(event)
        assert row["categories"] == sorted({event["category"] for event in relevant})
    assert set(diagnostics.CATEGORIES) - {"other"} <= observed_categories
    assert "not additive" in result["scope"]


def test_group_replacements_preserve_counts_precision_and_original_snapshots(monkeypatch):
    method, episode, first, final = setup_case()
    calls = install_decoder(monkeypatch)
    originals = tuple(value.clone() for state in (first, final)
                      for value in (state.rows, state.values, state.counts, state.precision))
    result = diagnostics.detailed_row_diagnostics(method, episode, first, final, 5)
    assert any(not group["reused_identical_group_predictions"] for group in result["groups"].values())
    for group in result["groups"].values():
        chosen = first.rows.new_tensor(group["rows"])
        values = first.lookup(chosen)[0]
        zero = evaluate_module.replace_rows(first, chosen, torch.zeros_like(values))
        restored = evaluate_module.replace_rows(final, chosen, values)
        for key, state, index in (("write1_zero_group_q1", zero, 0),
                                  ("final_restore_group_q1", restored, 0),
                                  ("final_restore_group_new", restored, 2)):
            assert group["predictions"][key]["raw"] == generated_text(state, episode.question(index))
    for state, _ in calls:
        original = first if torch.equal(state.counts, first.counts) else final
        assert torch.equal(state.rows, original.rows)
        assert torch.equal(state.counts, original.counts)
        assert torch.equal(state.precision, original.precision)
    current = tuple(value for state in (first, final)
                    for value in (state.rows, state.values, state.counts, state.precision))
    assert all(torch.equal(before, after) for before, after in zip(originals, current))
    assert all(result["invariants"].values())


def test_literal_sharing_and_hash_collision_can_coexist(monkeypatch):
    method, episode, first, final = setup_case(buckets=1)
    install_decoder(monkeypatch)
    result = diagnostics.detailed_row_diagnostics(method, episode, first, final, 5)
    assert result["selection"]["rows"] == [1, 2]
    for row in result["rows"]:
        assert row["literal_shared"] and row["hash_collision"]
        assert row["literal_shared_tuples"]
        old = {(item["slot"], item["order"], tuple(item["token_ids"])) for item in row["q1_distinct_tuples"]}
        later = {(item["slot"], item["order"], tuple(item["token_ids"])) for item in row["later_evidence_distinct_tuples"]}
        assert old & later and any(left != right for left in old for right in later)
    assert result["groups"]["literal_shared"]["rows"] == result["groups"]["hash_collision"]["rows"]


def test_shared_result_reused_unchanged_and_zero_change_rows_not_filtered(monkeypatch):
    method, episode, first, final = setup_case(zero_values=True)
    calls = install_decoder(monkeypatch)
    shared = evaluate_module.diagnose_rows(method, episode, first, final, 5)
    original = copy.deepcopy(shared)
    previous = len(calls)
    result = diagnostics.detailed_row_diagnostics(method, episode, first, final, 5, shared_diagnostic=shared)
    assert result["shared_group"] is shared and shared == original
    assert len(calls) > previous  # Real proper-subset interventions still ran.
    assert result["selection"]["rows"] == shared["old_overlap_rows"]
    assert result["selection"]["row_count"] > 0
    assert result["selection"]["correctness_filter"] is False
    assert result["selection"]["update_size_filter"] is False
    assert all(group["shared_change_frobenius_norm"] == 0 for group in result["groups"].values())


def test_character_fallback_and_empty_first_memory_are_supported(monkeypatch):
    method, episode, _, final = setup_case(tokenizer=CharacterTokenizer())
    calls = install_decoder(monkeypatch)
    first = SparseMemoryState.empty(2, with_precision=True)
    result = diagnostics.detailed_row_diagnostics(method, episode, first, final, 5)
    assert result["supported"] and result["selection"]["rows"] == []
    assert len(calls) == 6  # Empty category groups reuse the unchanged baseline.
    assert all(item["offset_source"] == "CharacterTokenizer_character_offsets" for item in result["tokenization"])
    assert all(not group["intervention_applied"] for group in result["groups"].values())


def test_real_fast_bpe_offsets_are_used_without_model_download(monkeypatch):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    engine = Tokenizer(models.BPE(unk_token="[UNK]"))
    engine.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    engine.decoder = decoders.ByteLevel()
    episode = Episode("bpe", ("Kalibo", "Kalake", "Bobaba"), ("red", "blue", "green"))
    engine.train_from_iterator([episode.evidence(i) for i in range(3)],
                              trainers.BpeTrainer(vocab_size=70, special_tokens=["[UNK]"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=engine, unk_token="[UNK]")
    method, episode, first, final = setup_case(tokenizer=tokenizer)
    install_decoder(monkeypatch)
    result = diagnostics.detailed_row_diagnostics(method, episode, first, final, 5)
    assert result["supported"] and result["selection"]["row_count"] > 0
    assert any(len(event["text"]) > event["order"] for row in result["rows"] for event in row["occurrences"])
    for source in result["tokenization"]:
        expected = tokenizer(source["text"], add_special_tokens=False, return_offsets_mapping=True)
        assert source["token_ids"] == expected["input_ids"]
        assert source["offsets"] == [list(offset) for offset in expected["offset_mapping"]]
        assert source["offset_source"] == "fast_tokenizer_offsets"


def test_unsupported_method_episode_template_and_tokenizer_are_explicit(monkeypatch):
    method, episode, first, final = setup_case()
    monkeypatch.setattr(diagnostics, "greedy_generate", lambda *args: pytest.fail("unsupported path generated"))
    method.supports_row_diagnostics = False
    assert "method" in diagnostics.detailed_row_diagnostics(method, episode, first, final, 5)["unsupported_reason"]
    method.supports_row_diagnostics = True

    class NoNames:
        evidence_count = 2

    assert "entity names" in diagnostics.detailed_row_diagnostics(method, NoNames(), first, final, 5)["unsupported_reason"]

    class NonTemplate:
        names = ("Al", "Bob")
        evidence_count = 2

        def question(self, index):
            return "Other format"

        def evidence(self, index):
            return "Other format evidence"

    assert "template" in diagnostics.detailed_row_diagnostics(method, NonTemplate(), first, final, 5)["unsupported_reason"]

    class SlowTokenizer:
        def encode(self, text, **kwargs):
            return [1] * len(text)

    method.tokenizer = SlowTokenizer()
    assert "fast offsets" in diagnostics.detailed_row_diagnostics(method, episode, first, final, 5)["unsupported_reason"]


def test_offset_id_mismatch_is_not_silently_classified():
    class BadFast(FastCharacterTokenizer):
        def __call__(self, *args, **kwargs):
            encoded = super().__call__(*args, **kwargs)
            encoded["input_ids"][0] += 1
            return encoded

    method, episode, first, final = setup_case(tokenizer=BadFast())
    result = diagnostics.detailed_row_diagnostics(method, episode, first, final, 5)
    assert not result["supported"] and "differs" in result["unsupported_reason"]


def test_mutating_query_is_detected(monkeypatch):
    method, episode, first, final = setup_case()
    install_decoder(monkeypatch)
    shared = evaluate_module.diagnose_rows(method, episode, first, final, 5)

    def mutate(method, state, *args):
        state.values.add_(1)
        return "red"

    monkeypatch.setattr(diagnostics, "greedy_generate", mutate)
    with pytest.raises(RuntimeError, match="mutated explicit memory"):
        diagnostics.detailed_row_diagnostics(method, episode, first, final, 5, shared_diagnostic=shared)


def test_category_uses_offsets_and_ignores_only_boundary_whitespace():
    text = "Name: Kalibo\nFavorite color: red"
    name = [6, 12]
    answer = [text.index("red"), len(text)]
    assert diagnostics._category(text, [[5, 8], [8, 12]], name, answer) == "name_internal"
    assert diagnostics._category(text, [[0, 5], [5, 8]], name, answer) == "name_boundary"
    assert diagnostics._category(text, [[13, 21]], name, answer) == "template"
    assert diagnostics._category(text, [[22, 28], [29, 32]], name, answer) == "answer"
    assert diagnostics._category(text, [[0, 0], [6, 8]], name, answer) == "other"

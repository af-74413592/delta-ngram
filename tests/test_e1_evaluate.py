"""End-to-end evaluator checks with an evidence-only, deterministic fact memory."""
from copy import deepcopy
import importlib
import re
from types import SimpleNamespace

import pytest

from delta_ngram.data import E1Episode, Episode, FactEvent
from delta_ngram.interfaces import MemoryMethod, WriteResult
from delta_ngram.state import SparseMemoryState


evaluate_module = importlib.import_module("delta_ngram.evaluate")


class EvidenceFactMemory(MemoryMethod):
    """Parse supplied evidence, without receiving an episode or any query labels."""

    supports_row_diagnostics = False
    name = "evidence_fact_mock"

    def __init__(self, *, ignore_updates=False, damage_fruit=False):
        super().__init__(None, None, "cpu")
        self.ignore_updates = ignore_updates
        self.damage_fruit = damage_fruit
        self.evidence_seen = []
        self.snapshots = []

    def empty_state(self):
        return {"facts": {}, "stage": 0}

    def write(self, state, evidence):
        self.evidence_seen.append(evidence)
        match = re.fullmatch(r"(Profile|Update)\nName: ([^\n]+)\nFavorite (color|fruit): ([a-z]+)\n", evidence)
        assert match, "The mock writer accepts only evidence, never queries or labels."
        header, name, attribute, value = match.groups()
        facts = dict(state["facts"])
        if header != "Update" or not self.ignore_updates:
            facts[(name, attribute)] = value
        if header == "Update" and self.damage_fruit:
            facts[(name, "fruit")] = "plum"
        result = {"facts": facts, "stage": state["stage"] + 1}
        self.snapshots.append((result, deepcopy(result)))
        return WriteResult(result)

    def memory(self, state):
        # The mock stores textual facts; the evaluator still exercises its normal
        # state size/report path through an empty tensor table.
        return SparseMemoryState.empty(1)

    def counterfactual(self, reference, alternative):
        return alternative


def _revision_episode():
    return E1Episode("revision-test", (
        FactEvent(("NameA", "favorite_color"), "red", "acquisition"),
        FactEvent(("NameA", "favorite_fruit"), "pear", "acquisition"),
        FactEvent(("NameA", "favorite_color"), "blue", "revision"),
        FactEvent(("NameB", "favorite_color"), "green", "acquisition"),
    ))


def _independent_episode():
    return E1Episode("independent-test", (
        FactEvent(("NameA", "favorite_color"), "red", "acquisition"),
        FactEvent(("NameB", "favorite_color"), "blue", "acquisition"),
        FactEvent(("NameC", "favorite_color"), "green", "acquisition"),
        FactEvent(("NameD", "favorite_color"), "white", "acquisition"),
    ))


def _install_decoder(monkeypatch, *, named_extra_word=False, always_unknown=False):
    calls = []

    def generate(method, state, question, max_new_tokens):
        assert max_new_tokens == 4
        before = deepcopy(state)
        canonical = re.fullmatch(r"Profile\nName: ([^\n]+)\nFavorite (color|fruit):", question)
        named = re.fullmatch(r"What is ([^\n]+)'s favorite (color|fruit)\? Answer with one word:", question)
        match = canonical or named
        assert match, f"Unexpected question template: {question!r}"
        value = "unknown" if always_unknown else state["facts"].get(match.groups(), "unknown")
        if named and named_extra_word and value != "unknown":
            value += " extra"
        calls.append({"stage": state["stage"], "question": question, "raw": value,
                      "style": "canonical" if canonical else "named_question"})
        assert state == before
        return value

    monkeypatch.setattr(evaluate_module, "greedy_generate", generate)
    return calls


def _evaluate(tmp_path, method, episodes, **kwargs):
    args = SimpleNamespace(out=tmp_path, max_new_tokens=4, openbook_episodes=0)
    return evaluate_module.evaluate(method, episodes, args, save=False,
                                    counterfactuals=kwargs.pop("counterfactuals", False), **kwargs)


def test_superseded_endpoint_is_null_and_only_current_versions_are_probed(monkeypatch, tmp_path):
    _install_decoder(monkeypatch)
    episode = _revision_episode()
    method = EvidenceFactMemory()
    result = _evaluate(tmp_path, method, [episode])
    record = result["records"][0]
    endpoints = record["endpoints"]
    assert [row["query_index"] for row in endpoints] == [0, 1, 2, 3]
    assert [row["C0"] for row in endpoints] == [False] * 4
    assert [row["C1"] for row in endpoints] == [True] * 4
    assert [row["CT"] for row in endpoints] == [None, True, True, True]
    assert [row["retention_eligible"] for row in endpoints] == [False, True, True, True]
    assert [row["superseded_at"] for row in endpoints] == [3, None, None, None]
    assert [row["event_kind"] for row in endpoints] == ["acquisition", "acquisition", "revision", "acquisition"]
    assert [tuple(row["fact_key"]) for row in endpoints] == [event.key for event in episode.events]
    assert record["predictions"]["write1_q1"] == "red"
    assert record["predictions"]["write3_q3"] == record["predictions"]["write4_q3"] == "blue"
    assert "write3_q1" not in record["predictions"] and "write4_q1" not in record["predictions"]
    assert record["predictions"]["write3_q2"] == record["predictions"]["write4_q2"] == "pear"
    assert method.evidence_seen == [episode.evidence(index) for index in range(4)]
    assert all(state == snapshot for state, snapshot in method.snapshots)
    assert result["write_count"] == 4 and result["complete"]
    assert result["acquisition_retention"]["A1"] == 1
    assert result["acquisition_retention"]["AT"] is None
    assert result["acquisition_retention"]["conditional_retention"] is None
    assert result["acquisition_retention"]["conditional_retention_denominator"] == 0
    assert result["acquisition_retention"]["superseded_excluded"] == 1


def test_named_and_canonical_queries_reuse_each_evidence_write(monkeypatch, tmp_path):
    calls = _install_decoder(monkeypatch)
    episode = _revision_episode()
    canonical_method, both_method = EvidenceFactMemory(), EvidenceFactMemory()
    canonical = _evaluate(tmp_path, canonical_method, [episode])
    before_both = len(calls)
    both = _evaluate(tmp_path, both_method, [episode], query_styles=("canonical", "named_question"))
    assert canonical_method.evidence_seen == both_method.evidence_seen == [episode.evidence(i) for i in range(4)]
    assert canonical["write_count"] == both["write_count"] == 4
    assert both["query_count"] == 2 * canonical["query_count"]
    assert {call["style"] for call in calls[before_both:]} == {"canonical", "named_question"}
    row = both["records"][0]
    assert row["endpoints"] == row["endpoints_by_style"]["canonical"]
    assert row["endpoints_by_style"]["canonical"] == row["endpoints_by_style"]["named_question"]
    for key, value in canonical["records"][0]["predictions"].items():
        assert row["predictions"][key] == value
        assert row["predictions"][key + "__named_question"] == value
    assert "write4_q1__named_question" not in row["predictions"]
    assert all(state == snapshot for state, snapshot in both_method.snapshots)


def test_revision_chain_marks_each_replaced_version_inactive(monkeypatch, tmp_path):
    _install_decoder(monkeypatch)
    original = _revision_episode()
    episode = E1Episode("revision-chain", (*original.events,
        FactEvent(("NameA", "favorite_color"), "white", "revision")))
    result = _evaluate(tmp_path, EvidenceFactMemory(), [episode])
    row = result["records"][0]
    assert [endpoint["CT"] for endpoint in row["endpoints"]] == [None, True, None, True, True]
    assert [endpoint["superseded_at"] for endpoint in row["endpoints"]] == [3, None, 5, None, None]
    assert "write5_q1" not in row["predictions"] and "write5_q3" not in row["predictions"]
    assert row["predictions"]["write5_q5"] == "white"
    metrics = result["query_style_metrics"]["canonical"]["version_metrics"]
    assert metrics["active_final"]["count"] == 3
    assert metrics["active_final"]["superseded_excluded"] == 2
    assert metrics["revisions"]["count"] == 2
    assert metrics["revisions"]["immediate_success"] == 1
    assert metrics["revisions"]["final_count"] == 1
    assert metrics["revisions"]["final_success"] == 1
    probes = row["revision_probes"]["canonical"]
    assert probes[0]["final_success"] is None and probes[0]["final_still_previous"] is None
    assert probes[1]["previous_version"] == 2


def test_style_scoring_keeps_first_word_and_strict_line_results_separate(monkeypatch, tmp_path):
    _install_decoder(monkeypatch, named_extra_word=True)
    result = _evaluate(tmp_path, EvidenceFactMemory(), [_independent_episode()],
                       query_styles=("canonical", "named_question"))
    row = result["records"][0]
    assert row["correct"]["write4_q1"] is True
    assert row["correct"]["write4_q1__named_question"] is True
    assert row["strict_first_line_correct"]["write4_q1"] is True
    assert row["strict_first_line_correct"]["write4_q1__named_question"] is False
    by_style = result["query_style_metrics"]
    assert by_style["canonical"]["acquisition_retention"]["A1"] == 1
    assert by_style["canonical"]["strict_acquisition_retention"]["A1"] == 1
    assert by_style["named_question"]["acquisition_retention"]["A1"] == 1
    strict = by_style["named_question"]["strict_acquisition_retention"]
    assert strict["A1"] == 0 and strict["AT"] == 0
    assert strict["conditional_retention_denominator"] == 0
    assert strict["conditional_retention"] is None
    first_word_versions = by_style["named_question"]["version_metrics"]["active_final"]
    strict_versions = by_style["named_question"]["strict_version_metrics"]["active_final"]
    assert first_word_versions["C1"] == first_word_versions["CT"] == 1
    assert strict_versions["C1"] == strict_versions["CT"] == 0
    assert strict_versions["conditional_retention_denominator"] == 0
    assert strict_versions["conditional_retention"] is None


def test_legacy_default_and_explicit_canonical_keep_the_original_predictions(monkeypatch, tmp_path):
    _install_decoder(monkeypatch)
    episode = Episode("legacy", ("NameA", "NameB"), ("red", "blue"))
    default = _evaluate(tmp_path, EvidenceFactMemory(), [episode])
    explicit = _evaluate(tmp_path, EvidenceFactMemory(), [episode], query_styles=("canonical",))
    expected = {"empty_q1": "unknown", "empty_q2": "unknown", "write1_q1": "red",
                "write2_q1": "red", "write2_q2": "blue", "zero_after_write_q1": "unknown"}
    assert default["records"][0]["predictions"] == explicit["records"][0]["predictions"] == expected
    assert default["scores"] == explicit["scores"]
    assert default["acquisition_retention"] == explicit["acquisition_retention"]
    assert default["write_count"] == explicit["write_count"] == 2


def test_perfect_revision_metrics_count_only_active_final_versions(monkeypatch, tmp_path):
    _install_decoder(monkeypatch)
    result = _evaluate(tmp_path, EvidenceFactMemory(), [_revision_episode()],
                       query_styles=("canonical", "named_question"))
    for style in ("canonical", "named_question"):
        metric = result["query_style_metrics"][style]["version_metrics"]
        assert metric["active_final"] == {
            "count": 3, "superseded_excluded": 1, "C0": 0, "C1": 1, "CT": 1,
            "conditional_retention": 1, "conditional_retention_denominator": 3,
            "acquired_and_retained_all_samples": 1,
        }
        assert metric["with_later_writes"] == {
            "count": 2, "CT": 1, "conditional_retention": 1, "conditional_retention_denominator": 2,
        }
        assert metric["revisions"] == {
            "count": 1, "immediate_success": 1, "immediate_still_previous": 0,
            "final_count": 1, "final_success": 1, "final_still_previous": 0,
        }
        assert metric["collateral_same_entity"] == {
            "count": 1, "before": 1, "after": 1, "previously_correct_denominator": 1,
            "loss_among_previously_correct": 0,
        }
        assert result["records"][0]["revision_probes"][style] == [{
            "query_index": 2, "previous_version": 0, "immediate_success": True,
            "immediate_still_previous": False, "final_success": True,
            "final_still_previous": False, "collateral": [{"query_index": 1, "before": True, "after": True}],
        }]
        assert result["query_style_metrics"][style]["strict_version_metrics"] == metric


def test_failed_revision_reports_previous_value_without_counting_it_as_old_retention(monkeypatch, tmp_path):
    _install_decoder(monkeypatch)
    result = _evaluate(tmp_path, EvidenceFactMemory(ignore_updates=True), [_revision_episode()])
    row = result["records"][0]
    assert row["predictions"]["write3_q3"] == row["predictions"]["write4_q3"] == "red"
    assert row["endpoints"][0]["CT"] is None
    assert row["endpoints"][2]["C1"] is False and row["endpoints"][2]["CT"] is False
    metric = result["query_style_metrics"]["canonical"]["version_metrics"]
    assert metric["revisions"] == {
        "count": 1, "immediate_success": 0, "immediate_still_previous": 1,
        "final_count": 1, "final_success": 0, "final_still_previous": 1,
    }
    assert metric["active_final"]["count"] == 3
    assert metric["active_final"]["C1"] == metric["active_final"]["CT"] == pytest.approx(2 / 3)
    assert metric["active_final"]["conditional_retention_denominator"] == 2
    assert metric["active_final"]["conditional_retention"] == 1
    assert metric["collateral_same_entity"]["loss_among_previously_correct"] == 0


def test_revision_collateral_measures_other_attribute_before_and_after_write(monkeypatch, tmp_path):
    _install_decoder(monkeypatch)
    result = _evaluate(tmp_path, EvidenceFactMemory(damage_fruit=True), [_revision_episode()])
    row = result["records"][0]
    assert row["predictions"]["write2_q2"] == "pear"
    assert row["predictions"]["write3_q2"] == "plum"
    metric = result["query_style_metrics"]["canonical"]["version_metrics"]
    assert metric["revisions"]["immediate_success"] == 1
    assert metric["collateral_same_entity"] == {
        "count": 1, "before": 1, "after": 0, "previously_correct_denominator": 1,
        "loss_among_previously_correct": 1,
    }
    assert metric["active_final"]["conditional_retention_denominator"] == 3
    assert metric["active_final"]["conditional_retention"] == pytest.approx(2 / 3)
    assert metric["with_later_writes"]["conditional_retention"] == 0.5
    assert row["revision_probes"]["canonical"][0]["collateral"] == [
        {"query_index": 1, "before": True, "after": False}]


def test_failed_acquisition_has_no_retention_or_collateral_success_denominator(monkeypatch, tmp_path):
    _install_decoder(monkeypatch, always_unknown=True)
    result = _evaluate(tmp_path, EvidenceFactMemory(), [_revision_episode()])
    metric = result["query_style_metrics"]["canonical"]["version_metrics"]
    assert metric["active_final"]["C1"] == metric["active_final"]["CT"] == 0
    assert metric["active_final"]["conditional_retention_denominator"] == 0
    assert metric["active_final"]["conditional_retention"] is None
    assert metric["with_later_writes"]["conditional_retention_denominator"] == 0
    assert metric["with_later_writes"]["conditional_retention"] is None
    assert metric["collateral_same_entity"]["count"] == 1
    assert metric["collateral_same_entity"]["previously_correct_denominator"] == 0
    assert metric["collateral_same_entity"]["loss_among_previously_correct"] is None
    assert metric["revisions"]["immediate_success"] == 0
    assert metric["revisions"]["immediate_still_previous"] == 0


def test_counterfactual_queries_follow_only_current_corrupted_versions(monkeypatch, tmp_path):
    _install_decoder(monkeypatch)
    episode = _revision_episode()
    method = EvidenceFactMemory()
    result = _evaluate(tmp_path, method, [episode], counterfactuals=True,
                       query_styles=("canonical", "named_question"))
    assert result["write_count"] == 8
    assert method.evidence_seen == ([episode.evidence(i) for i in range(4)]
                                    + [episode.corrupted_evidence(i) for i in range(4)])
    row = result["records"][0]
    for style in ("canonical", "named_question"):
        suffix = "" if style == "canonical" else "__named_question"
        assert "wrong_values_q1" + suffix not in row["predictions"]
        assert "q1" + suffix not in row["wrong_values_follow_corrupted_evidence"]
        for index in (1, 2, 3):
            key = f"wrong_values_q{index + 1}" + suffix
            assert row["predictions"][key] == episode.corrupted_answer(index).strip()
            assert row["correct"][key] is False
            assert row["wrong_values_follow_corrupted_evidence"][f"q{index + 1}" + suffix] is True


def test_old_version_row_interventions_are_excluded_from_retention_diagnostics(monkeypatch, tmp_path):
    _install_decoder(monkeypatch)
    method = EvidenceFactMemory()
    method.supports_row_diagnostics = True
    monkeypatch.setattr(evaluate_module, "coverage", lambda *args: {"mock": True})
    monkeypatch.setattr(evaluate_module, "diagnose_rows",
                        lambda *args: pytest.fail("Superseded first fact is not an eligible retention intervention."))
    result = _evaluate(tmp_path, method, [_revision_episode()], row_diagnostics=True, detailed_rows=True)
    row = result["records"][0]
    assert row["row_diagnostic"] is None
    assert row["detailed_row_diagnostic"]["supported"] is False
    assert "superseded" in row["detailed_row_diagnostic"]["reason"]
    assert result["row_diagnostics"]["episodes"] == 0

"""E1 configuration and manifest handoffs without model loading."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_ngram.config import ExperimentConfig
from delta_ngram.train import make_manifest


def config_file(tmp_path, *, name="config", **sections):
    raw = {"model": {"smoke": True}, "data": {"task": "fact_e1", "facts_per_episode": 4,
            "eval_episodes": 2}, "run": {"output_dir": f"{name}-out"}}
    for section, fields in sections.items():
        raw.setdefault(section, {}).update(fields)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_e1_defaults_and_legacy_defaults_are_explicit(tmp_path):
    e1 = ExperimentConfig.load(config_file(tmp_path))
    assert e1.values["data"]["e1_scenario"] == "independent"
    assert e1.values["evaluation"]["query_styles"] == ["canonical"]
    assert e1.values["evaluation"]["detailed_rows"] is False
    legacy_path = tmp_path / "legacy.yaml"
    legacy_path.write_text("{}", encoding="utf-8")
    legacy = ExperimentConfig.load(legacy_path)
    assert legacy.values["data"]["task"] == "fact"
    assert legacy.values["data"]["facts_per_episode"] == 2
    assert legacy.values["evaluation"]["query_styles"] == ["canonical"]


@pytest.mark.parametrize("section,patch,message", [
    ("data", {"e1_scenario": "other"}, "e1_scenario"),
    ("data", {"e1_scenario": "revision", "facts_per_episode": 2}, "at least three writes"),
    ("evaluation", {"query_styles": []}, "query_styles"),
    ("evaluation", {"query_styles": "canonical"}, "query_styles"),
    ("evaluation", {"query_styles": ["named_question"]}, "include canonical"),
    ("evaluation", {"query_styles": ["canonical", "canonical"]}, "unique"),
    ("evaluation", {"query_styles": ["canonical", "unknown"]}, "query_styles"),
    ("evaluation", {"query_styles": ["canonical", {}]}, "query_styles"),
    ("evaluation", {"detailed_rows": "true", "row_diagnostics": True}, "boolean"),
    ("evaluation", {"detailed_rows": True}, "requires evaluation.row_diagnostics"),
])
def test_invalid_e1_and_diagnostic_settings_fail_early(tmp_path, section, patch, message):
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.load(config_file(tmp_path, **{section: patch}))


@pytest.mark.parametrize("task", ["affix", "mixed"])
def test_named_question_is_rejected_for_non_fact_tasks(tmp_path, task):
    with pytest.raises(ValueError, match="unsupported for affix or mixed"):
        ExperimentConfig.load(config_file(tmp_path, data={"task": task},
                              evaluation={"query_styles": ["canonical", "named_question"]}))


@pytest.mark.parametrize("task", ["fact", "fact_e1"])
def test_fact_tasks_accept_both_query_styles_and_detailed_rows(tmp_path, task):
    config = ExperimentConfig.load(config_file(tmp_path, data={"task": task},
                                   evaluation={"query_styles": ["named_question", "canonical"],
                                               "row_diagnostics": True, "detailed_rows": True}))
    assert config.values["evaluation"]["query_styles"] == ["named_question", "canonical"]
    assert config.values["evaluation"]["detailed_rows"] is True


def test_make_manifest_passes_e1_scenario_to_creation_and_replay_validation(tmp_path):
    config = ExperimentConfig.load(config_file(tmp_path, data={"e1_scenario": "revision"}))
    manifest = make_manifest(config)
    assert manifest["task"] == "fact_e1" and manifest["manifest_version"] == 2
    assert manifest["generator_config"]["e1_scenario"] == "revision"
    assert manifest["validation_episodes"][0]["events"][2]["kind"] == "revision"
    saved = tmp_path / "manifest.json"
    saved.write_text(json.dumps(manifest), encoding="utf-8")
    replay = ExperimentConfig.load(config_file(tmp_path, name="replay",
        data={"manifest": "manifest.json", "e1_scenario": "revision"}))
    assert make_manifest(replay) == json.loads(json.dumps(manifest))
    mismatch = ExperimentConfig.load(config_file(tmp_path, name="mismatch",
        data={"manifest": "manifest.json", "e1_scenario": "independent"}))
    with pytest.raises(ValueError, match="e1_scenario differs"):
        make_manifest(mismatch)


def test_cli_forwards_query_styles_and_detailed_rows(tmp_path, monkeypatch):
    cli = importlib.import_module("delta_ngram.cli")
    config = ExperimentConfig.load(config_file(tmp_path,
        evaluation={"query_styles": ["canonical", "named_question"],
                    "row_diagnostics": True, "detailed_rows": True}))
    method = SimpleNamespace(supports_outer_training=False, name="empty", device=torch.device("cpu"))
    monkeypatch.setattr(cli, "create_runtime", lambda current: (method, current.legacy_args()))
    monkeypatch.setattr(cli, "run_metadata", lambda *args: {})
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(kwargs)
        return {"complete": True, "scores": {}}

    monkeypatch.setattr(cli, "evaluate", evaluate)
    assert cli.run_evaluation(config)["complete"]
    assert calls[0]["query_styles"] == ("canonical", "named_question")
    assert calls[0]["detailed_rows"] is True and calls[0]["row_diagnostics"] is True

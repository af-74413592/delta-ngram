"""Validated YAML configuration; relative artifact paths use the YAML directory."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
import math
from pathlib import Path
from types import SimpleNamespace

import yaml


DEFAULTS = {
    "version": 1,
    "seed": 17,
    "model": {"name_or_path": "Qwen/Qwen3-0.6B-Base",
              "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd", "device": "auto", "smoke": False},
    "reader": {"backend": "external_hash", "buckets": 16384, "orders": [2, 3], "layer_index": 1},
    "writer": {"hidden_dim": 128, "right_window": 4},
    "method": {"name": "forward", "operator": "D1", "kappa": 1.0, "precision_init": 1.0,
               "frozen_rows": [], "frozen_template_texts": [], "lr": 0.01, "online_steps": 10,
               "top_k": 2, "k1": 1.5, "b": 0.75, "target_steps": 10, "target_lr": 0.05,
               "target_l2": 0.001, "ridge_lambda": 1.0, "reuse_penalty": 1.0},
    "data": {"task": "fact", "facts_per_episode": 2, "support_examples": 3, "e1_scenario": "independent",
             "eval_episodes": 16, "manifest": None},
    "train": {"steps": 500, "batch_episodes": 4, "lr": 0.0003, "eval_every": 100,
              "gradient_clip": 1.0},
    "evaluation": {"max_new_tokens": 3, "openbook_episodes": 4,
                   "counterfactuals": True, "row_diagnostics": False,
                   "query_styles": ["canonical"], "detailed_rows": False},
    "run": {"output_dir": "../../delta-ngram-runs/pilot_seed17", "checkpoint": None,
            "max_wall_seconds": 1500.0, "max_evidence_tokens": 128},
    "raw": {"dataset": None, "block_size": 512, "prefix_tokens": 128,
            "block_counts": [8, 16], "length_weights": None, "source_weights": None,
            "loss_tokens": 5_000_000, "batch_size": 1, "gradient_accumulation_steps": 1, "eval_windows": 8,
            "eval_every_tokens": 1_000_000, "max_steps": None,
            "eval_seed": 104729, "eval_bootstrap_samples": 2000,
            "checkpoint_every_tokens": 250_000,
            "checkpoint_reader": False, "resume": None},
}


def _merge(default, supplied, prefix=""):
    if not isinstance(supplied, dict):
        raise ValueError(f"{prefix or 'config'} must be a mapping")
    unknown = supplied.keys() - default.keys()
    if unknown:
        raise ValueError(f"unknown configuration keys: {prefix}{', '.join(sorted(unknown))}")
    result = deepcopy(default)
    for key, value in supplied.items():
        result[key] = _merge(default[key], value, f"{prefix}{key}.") if isinstance(default[key], dict) else value
    return result


@dataclass(frozen=True)
class ExperimentConfig:
    values: dict
    source: Path

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        values = _merge(DEFAULTS, raw or {})
        if type(values["version"]) is not int or values["version"] != 1:
            raise ValueError("unsupported config version")
        integers = {"seed": (values["seed"], 0)}
        for group, keys, minimum in (
                ("reader", ("buckets",), 1), ("reader", ("layer_index",), 0),
                ("writer", ("hidden_dim",), 1), ("writer", ("right_window",), 0),
                ("data", ("eval_episodes", "facts_per_episode", "support_examples"), 1),
                ("train", ("batch_episodes", "eval_every"), 1), ("train", ("steps",), 0),
                ("evaluation", ("max_new_tokens",), 1), ("evaluation", ("openbook_episodes",), 0),
                ("run", ("max_evidence_tokens",), 1),
                ("method", ("online_steps", "top_k", "target_steps"), 1)):
            integers.update({f"{group}.{key}": (values[group][key], minimum) for key in keys})
        for name, (value, minimum) in integers.items():
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for group, key in (("model", "smoke"), ("evaluation", "counterfactuals"),
                           ("evaluation", "row_diagnostics"), ("evaluation", "detailed_rows")):
            if type(values[group][key]) is not bool:
                raise ValueError(f"{group}.{key} must be a boolean")
        if values["reader"]["backend"] not in ("external_hash", "native"):
            raise ValueError("reader.backend must be external_hash or native")
        if values["model"]["device"] not in ("auto", "cpu", "cuda"):
            raise ValueError("model.device must be auto, cpu, or cuda")
        if values["data"]["task"] not in ("fact", "fact_e1", "affix", "mixed"):
            raise ValueError("data.task must be fact, fact_e1, affix, or mixed")
        if values["data"]["e1_scenario"] not in ("independent", "revision"):
            raise ValueError("data.e1_scenario must be independent or revision")
        if (values["data"]["task"] == "fact_e1" and values["data"]["e1_scenario"] == "revision"
                and values["data"]["facts_per_episode"] < 3):
            raise ValueError("data.e1_scenario revision requires at least three writes in data.facts_per_episode")
        styles = values["evaluation"]["query_styles"]
        if (not isinstance(styles, list) or not styles
                or any(not isinstance(style, str) or style not in ("canonical", "named_question") for style in styles)
                or len(set(styles)) != len(styles) or "canonical" not in styles):
            raise ValueError("evaluation.query_styles must be a unique list of canonical/named_question and include canonical")
        if values["data"]["task"] in ("affix", "mixed") and "named_question" in styles:
            raise ValueError("evaluation.query_styles named_question is unsupported for affix or mixed tasks")
        if values["evaluation"]["detailed_rows"] and not values["evaluation"]["row_diagnostics"]:
            raise ValueError("evaluation.detailed_rows requires evaluation.row_diagnostics")
        if values["method"]["name"] not in (
                "forward", "adam", "sgd", "read_normalized_sgd", "context", "retrieval", "engram_edit_style", "empty"):
            raise ValueError("unknown method.name")
        if str(values["method"]["operator"]).upper() not in ("DIRECT", "OVERWRITE", "C", "D0", "D1", "D2", "D3", "D4"):
            raise ValueError("unknown method.operator")
        positive = (("train", "lr"), ("train", "gradient_clip"), ("run", "max_wall_seconds"),
                    ("method", "lr"), ("method", "precision_init"), ("method", "target_lr"),
                    ("method", "ridge_lambda"), ("method", "k1"))
        for group, key in positive:
            value = values[group][key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{group}.{key} must be positive")
        for key in ("kappa", "target_l2", "reuse_penalty", "b"):
            value = values["method"][key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"method.{key} must be finite and nonnegative")
        if values["method"]["b"] > 1:
            raise ValueError("method.b must be <= 1")
        if not isinstance(values["reader"]["orders"], list) or not values["reader"]["orders"] or any(type(n) is not int or n < 1 for n in values["reader"]["orders"]):
            raise ValueError("reader.orders must be positive integers")
        rows = values["method"]["frozen_rows"]
        if not isinstance(rows, list) or any(type(row) is not int or row < 1 or row > values["reader"]["buckets"] * len(values["reader"]["orders"]) for row in rows):
            raise ValueError("method.frozen_rows must contain valid positive physical row IDs")
        texts = values["method"]["frozen_template_texts"]
        if not isinstance(texts, list) or any(not isinstance(value, str) for value in texts):
            raise ValueError("method.frozen_template_texts must be a list of strings")
        if not isinstance(values["model"]["name_or_path"], str) or not values["model"]["name_or_path"]:
            raise ValueError("model.name_or_path must be a nonempty string")
        if values["model"]["revision"] is not None and not isinstance(values["model"]["revision"], str):
            raise ValueError("model.revision must be a string or null")
        if not isinstance(values["run"]["output_dir"], str) or not values["run"]["output_dir"]:
            raise ValueError("run.output_dir must be a nonempty path string")
        raw = values["raw"]
        for key in ("block_size", "prefix_tokens", "loss_tokens", "batch_size", "gradient_accumulation_steps",
                    "eval_windows", "eval_every_tokens", "eval_bootstrap_samples"):
            if type(raw[key]) is not int or raw[key] < 1:
                raise ValueError(f"raw.{key} must be a positive integer")
        if type(raw["eval_seed"]) is not int or raw["eval_seed"] < 0:
            raise ValueError("raw.eval_seed must be a nonnegative integer")
        if (raw["checkpoint_every_tokens"] is not None
                and (type(raw["checkpoint_every_tokens"]) is not int or raw["checkpoint_every_tokens"] < 1)):
            raise ValueError("raw.checkpoint_every_tokens must be null or a positive integer")
        if raw["prefix_tokens"] > raw["block_size"]:
            raise ValueError("raw.prefix_tokens must not exceed raw.block_size")
        if (not isinstance(raw["block_counts"], list) or not raw["block_counts"]
                or any(type(n) is not int or n < 2 for n in raw["block_counts"])
                or len(set(raw["block_counts"])) != len(raw["block_counts"])):
            raise ValueError("raw.block_counts must contain unique integers >= 2")
        if raw["max_steps"] is not None and (type(raw["max_steps"]) is not int or raw["max_steps"] < 1):
            raise ValueError("raw.max_steps must be null or a positive integer")
        if type(raw["checkpoint_reader"]) is not bool:
            raise ValueError("raw.checkpoint_reader must be boolean")
        for key in ("length_weights", "source_weights"):
            weights = raw[key]
            if weights is not None:
                if (not isinstance(weights, dict) or not weights
                        or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0
                               for w in weights.values()) or sum(weights.values()) <= 0):
                    raise ValueError(f"raw.{key} must be a nonempty mapping of nonnegative finite weights")
                if key == "length_weights":
                    if (any(type(k) is not int for k in weights)
                            or set(weights) != set(raw["block_counts"])):
                        raise ValueError("raw.length_weights must have integer keys matching raw.block_counts")
                elif any(not isinstance(k, str) or not k for k in weights):
                    raise ValueError("raw.source_weights keys must be nonempty strings")
        for key in ("dataset", "resume"):
            if raw[key] is not None and (not isinstance(raw[key], str) or not raw[key]):
                raise ValueError(f"raw.{key} must be null or a nonempty path string")
        for section, key in (("run", "output_dir"), ("run", "checkpoint"), ("data", "manifest"),
                             ("raw", "dataset"), ("raw", "resume")):
            value = values[section][key]
            if value is not None:
                expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
                values[section][key] = str((path.parent / expanded).resolve()) if not expanded.is_absolute() else str(expanded.resolve())
        if values["model"]["smoke"]:
            values["model"]["device"] = "cpu"
        return cls(values, path)

    def legacy_args(self):
        """Explicit migration mapping used by model loading and artifact parity."""
        v = self.values
        return SimpleNamespace(
            model=v["model"]["name_or_path"], revision=v["model"]["revision"],
            device=v["model"]["device"], smoke=v["model"]["smoke"], seed=v["seed"],
            out=Path(v["run"]["output_dir"]), steps=v["train"]["steps"],
            batch_episodes=v["train"]["batch_episodes"], lr=v["train"]["lr"],
            eval_every=v["train"]["eval_every"], eval_episodes=v["data"]["eval_episodes"],
            hidden_dim=v["writer"]["hidden_dim"], max_new_tokens=v["evaluation"]["max_new_tokens"],
            max_wall_seconds=v["run"]["max_wall_seconds"], max_evidence_tokens=v["run"]["max_evidence_tokens"],
            openbook_episodes=v["evaluation"]["openbook_episodes"], buckets=v["reader"]["buckets"],
            layer_index=v["reader"]["layer_index"])

"""Deterministic, offline scoring for the niche continual-learning tasks.

No case folding, accent removal, internal-whitespace collapsing, fuzzy JSON
extraction, or sorting of action arguments is permitted by this protocol.
"""
from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping, Sequence


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_action(text: str) -> dict | None:
    """Accept one JSON object with exactly action and ordered string values."""
    try:
        value = json.loads(text.strip(), object_pairs_hook=_no_duplicate_keys)
    except (ValueError, TypeError):
        return None
    if (not isinstance(value, dict) or set(value) != {"action", "values"}
            or not isinstance(value["action"], str) or not value["action"]
            or not isinstance(value["values"], list)
            or any(not isinstance(item, str) for item in value["values"])):
        return None
    return {"action": nfc(value["action"]),
            "values": [nfc(item) for item in value["values"]]}


def score_prediction(prediction: str, record: Mapping) -> dict:
    """Return per-row scores. Invalid JSON remains in all metric denominators."""
    metric = record["metric"]
    if metric in {"mulms_graph", "smiles_identity"}:
        from .science_niche_metrics import score_science_prediction
        return score_science_prediction(prediction, record)
    target = str(record["target"])
    if metric == "inflection_exact":
        predicted, expected = nfc(prediction.strip()), nfc(target.strip())
        references = record.get("references", [target])
        if (not isinstance(references, list) or not references
                or any(not isinstance(ref, str) or not ref.strip() for ref in references)):
            raise ValueError("Inflection references must be a nonempty list of strings")
        accepted = {nfc(reference.strip()) for reference in references}
        if expected not in accepted:
            raise ValueError("Original gold target is missing from references")
        return {"correct": predicted in accepted,
                "single_reference_correct": predicted == expected,
                "format_valid": bool(predicted) and "\n" not in predicted and "\r" not in predicted,
                "edit_distance": min(_edit_distance(predicted, reference) for reference in accepted),
                "reference_chars": len(expected)}
    if metric == "abcd_action_json":
        expected, predicted = parse_action(target), parse_action(prediction)
        if expected is None:
            raise ValueError("Invalid gold action JSON")
        valid = predicted is not None
        action_correct = valid and predicted["action"] == expected["action"]
        values_correct = valid and predicted["values"] == expected["values"]
        return {"correct": action_correct and values_correct,
                "format_valid": valid, "action_correct": action_correct,
                "values_correct": values_correct,
                "argument_count_correct": valid and len(predicted["values"]) == len(expected["values"])}
    raise ValueError(f"Unknown niche metric: {metric}")


def aggregate_scores(scores: Sequence[Mapping]) -> dict:
    if not scores:
        raise ValueError("Cannot aggregate zero examples")
    keys = set(scores[0])
    if any(set(row) != keys for row in scores):
        raise ValueError("Do not combine different task metric schemas")
    if "entity_tp" in keys or "string_exact" in keys:
        from .science_niche_metrics import aggregate_science_scores
        return aggregate_science_scores(list(scores))
    result = {"n": len(scores), "correct": sum(bool(row["correct"]) for row in scores),
              "accuracy": sum(bool(row["correct"]) for row in scores) / len(scores)}
    for key in sorted(keys - {"correct", "edit_distance", "reference_chars"}):
        result[key + "_rate"] = sum(bool(row[key]) for row in scores) / len(scores)
    if "edit_distance" in keys:
        result["mean_edit_distance"] = sum(row["edit_distance"] for row in scores) / len(scores)
        chars = sum(row["reference_chars"] for row in scores)
        result["character_error_rate"] = sum(row["edit_distance"] for row in scores) / chars if chars else None
    return result

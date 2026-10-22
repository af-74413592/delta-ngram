"""Deterministic structure/chemistry scores; never execute generated programs."""
from __future__ import annotations

import json
import unicodedata

RDKIT_VERSION = "2025.09.3"
ENTITY_TYPES = ("MAT", "NUM", "VALUE", "UNIT", "PROPERTY", "FORM", "MEASUREMENT", "CITE",
                "SAMPLE", "TECHNIQUE", "RANGE", "DEV", "INSTRUMENT")
RELATION_TYPES = ("hasForm", "measuresProperty", "propertyValue", "conditionProperty", "usedAs",
                  "conditionSampleFeatures", "conditionPropertyValue", "usesTechnique", "measuresPropertyValue",
                  "usedTogether", "conditionEnvironment", "conditionInstrument", "usedIn", "takenFrom", "dopedBy")


def _unique_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON keys")
        value[key] = item
    return value


def parse_material_graph(text: str, token_count: int):
    """Parse zero-based, half-open token spans and directed entity-index edges."""
    try:
        graph = json.loads(text.strip(), object_pairs_hook=_unique_keys)
        if not isinstance(graph, dict) or set(graph) != {"entities", "relations"}:
            return None
        entities, relations = graph["entities"], graph["relations"]
        if not isinstance(entities, list) or not isinstance(relations, list):
            return None
        normalized = []
        for entity in entities:
            if (not isinstance(entity, list) or len(entity) != 3
                    or type(entity[0]) is not int or type(entity[1]) is not int
                    or not 0 <= entity[0] < entity[1] <= token_count or entity[2] not in ENTITY_TYPES):
                return None
            normalized.append(tuple(entity))
        if len(set(normalized)) != len(normalized):
            return None
        edges = []
        for relation in relations:
            if (not isinstance(relation, list) or len(relation) != 3
                    or type(relation[0]) is not int or type(relation[2]) is not int
                    or not 0 <= relation[0] < len(normalized) or not 0 <= relation[2] < len(normalized)
                    or relation[1] not in RELATION_TYPES):
                return None
            edges.append((normalized[relation[0]], relation[1], normalized[relation[2]]))
        if len(set(edges)) != len(edges):
            return None
        return set(normalized), set(edges)
    except (TypeError, ValueError):
        return None


def canonical_smiles(smiles: str) -> str | None:
    """Identity under pinned RDKit canonical isomeric SMILES; retain stereo/salts.

    Remove atom-map annotations only. No tautomer, charge, isotope, salt,
    stereochemistry or parent-fragment normalization is performed. Refuse
    trailing molecule names/explanations (RDKit otherwise accepts some).
    """
    from rdkit import Chem, rdBase
    if rdBase.rdkitVersion != RDKIT_VERSION:
        raise RuntimeError(f"RDKit {RDKIT_VERSION} required, found {rdBase.rdkitVersion}")
    value = smiles.strip()
    if not value or any(char.isspace() for char in value):
        return None
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value)
        if molecule is None or molecule.GetNumAtoms() == 0:
            return None
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def score_science_prediction(prediction: str, record: dict) -> dict:
    metric = record["metric"]
    if metric == "mulms_graph":
        gold = parse_material_graph(record["target"], len(record["tokens"]))
        if gold is None:
            raise ValueError("Invalid gold MuLMS graph")
        predicted = parse_material_graph(prediction, len(record["tokens"]))
        scores = {"correct": predicted == gold, "format_valid": predicted is not None,
                  "gold_empty": not gold[0] and not gold[1]}
        actual = predicted or (set(), set())
        for name, expected, output in zip(("entity", "relation"), gold, actual, strict=True):
            scores.update({name + "_tp": len(expected & output), name + "_fp": len(output - expected),
                           name + "_fn": len(expected - output)})
        return scores
    if metric == "smiles_identity":
        gold = canonical_smiles(record["target"])
        if gold is None:
            raise ValueError("Invalid gold product SMILES")
        predicted = canonical_smiles(prediction)
        return {"correct": predicted is not None and predicted == gold, "format_valid": predicted is not None,
                "string_exact": unicodedata.normalize("NFC", prediction.strip()) == unicodedata.normalize("NFC", record["target"].strip())}
    raise ValueError(f"Unknown science metric: {metric}")


def aggregate_science_scores(scores: list[dict]) -> dict:
    if not scores or any(set(row) != set(scores[0]) for row in scores):
        raise ValueError("Expected nonempty scores with one metric schema")
    n = len(scores)
    result = {"n": n, "correct": sum(bool(row["correct"]) for row in scores),
              "accuracy": sum(bool(row["correct"]) for row in scores) / n,
              "format_valid_rate": sum(bool(row["format_valid"]) for row in scores) / n}
    if "entity_tp" in scores[0]:
        for name in ("entity", "relation"):
            tp, fp, fn = (sum(row[name + "_" + key] for row in scores) for key in ("tp", "fp", "fn"))
            result.update({name + "_tp": tp, name + "_fp": fp, name + "_fn": fn,
                           name + "_micro_precision": tp / (tp + fp) if tp + fp else 0.0,
                           name + "_micro_recall": tp / (tp + fn) if tp + fn else 0.0,
                           name + "_micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None})
        empty = [row for row in scores if row["gold_empty"]]
        result["gold_empty_n"] = len(empty)
        result["empty_graph_false_positive_rate"] = (sum(row["entity_fp"] > 0 or not row["format_valid"] for row in empty) / len(empty)) if empty else None
        result["primary_metrics"] = ["entity_micro_f1", "relation_micro_f1"]
    else:
        result["string_exact_rate"] = sum(bool(row["string_exact"]) for row in scores) / n
        result["primary_metrics"] = ["accuracy"]
    return result

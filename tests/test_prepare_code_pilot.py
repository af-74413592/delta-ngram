"""Offline invariants for bounded real-code pilot preparation."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_code_pilot.py"
SPEC = importlib.util.spec_from_file_location("prepare_code_pilot", SCRIPT)
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


def source_record(repo="Owner/Repository", document="blob", language="Python"):
    text = "\n".join(f"value_{index} = transform_{index}(input_{index})" for index in range(500))
    return {"id": document, "source": "stackv2", "text": text, "metadata": {
        "repo_name": repo, "path": "/src/file.py", "revision_id": "abc123",
        "language": language, "license": "MIT", "detected_licenses": ["MIT"],
        "is_generated": False, "is_vendor": False, "provenance": "upstream-shard.gz:17"}}


def candidate(repo, text, index=0, language="Python", count=4096):
    return {"repo": repo, "text": text, "document_id": f"blob-{index}",
            "shard": "stack-edu-0075.json.gz", "line": index + 1,
            "language": language, "token_ids": [2] * count,
            "upstream_source": "stackv2", "upstream_metadata": source_record()["metadata"],
            "shard_sha256": "a" * 64}


@pytest.mark.parametrize("change,reason", [
    ({"repo_name": ""}, "missing_or_invalid_repo"),
    ({"is_generated": True}, "generated_flag"),
    ({"is_vendor": True}, "vendor_flag"),
    ({"path": "/src/node_modules/x.js"}, "vendor_or_generated_path"),
    ({"license": "", "detected_licenses": [], "gha_license_id": "MIT"}, "missing_record_license"),
    ({"language": "Markdown"}, "nonselected_language"),
])
def test_missing_provenance_or_source_quality_is_not_inferred(change, reason):
    record = source_record()
    record["metadata"].update(change)
    assert prepare.eligible_record(record) == reason


def test_source_text_is_unchanged_and_record_license_is_preserved():
    record = source_record()
    original = copy.deepcopy(record)
    assert prepare.eligible_record(record) is None
    assert record == original
    assert prepare.normalize("  x  \r\n y \r\n") == "x\n y"


def test_repository_reservoir_is_bounded_order_independent_and_case_normalized_by_scan():
    records = [candidate(f"r/{index // 3}", "text", index) for index in range(99)]
    a, b = prepare.RepositoryReservoir(7, 17), prepare.RepositoryReservoir(7, 17)
    for item in records:
        a.add(item)
    for item in reversed(records):
        b.add(item)
    assert len(a.selected()) == 7
    assert a.selected() == b.selected()
    assert len({item["repo"] for item in a.selected()}) == 7


def test_exact_duplicate_connects_repositories_before_representative_selection():
    text = "def first(x):\n    return x + 1\n"
    docs = [candidate("a/repo", text, 0), candidate("b/repo", text + "\n", 1),
            candidate("b/repo", "def other(z):\n    return z * 99\n", 2)]
    stats, edges = prepare.form_families(docs)
    assert stats["exact_duplicate_edges"] == 1
    assert stats["repository_edges"] == 1
    assert len({item["group_id"] for item in docs}) == 1
    assert docs[2]["family_repositories"] == ["a/repo", "b/repo"]
    assert edges


def test_near_duplicate_requires_real_shingle_similarity():
    first = "\n".join(f"def function_{i}(argument): return argument + {i}" for i in range(100))
    second = first.replace("function_50", "renamed_once")
    docs = [candidate("a/repo", first, 0), candidate("b/repo", second, 1),
            candidate("c/repo", "different bytes and token vocabulary unrelated", 2)]
    stats, edges = prepare.form_families(docs)
    assert stats["near_duplicate_edges"] == 1
    assert docs[0]["group_id"] == docs[1]["group_id"] != docs[2]["group_id"]
    assert 0.85 <= edges[0][2] < 1


def balanced_documents(per_stratum=40):
    result = []
    for language in prepare.LANGUAGES:
        for long in (False, True):
            for i in range(per_stratum):
                key = f"{language}/{long}/{i}"
                item = candidate(key, key, len(result), language, 8192 if long else 4096)
                item.update(group_id=key, family_repositories=[key], text_sha256=key,
                            normalized_text_sha256=key)
                result.append(item)
    return result


def test_whole_family_stratified_split_has_disjoint_groups_and_both_lengths():
    docs = balanced_documents()
    prepare.assign_splits(docs, seed=17, minimum_validation_groups=12)
    stats = prepare.validate_and_summarize(docs, minimum_validation_groups=12)
    assert stats["splits"]["validation"]["groups"] == 12
    assert stats["splits"]["holdout"]["groups"] == 12
    assert stats["splits"]["train"]["group_fraction"] == 0.95
    assert set(item["split"] for item in docs) == {"train", "validation", "holdout"}
    assert all(value <= 0.4 for value in stats["language_raw_fractions"].values())


def test_repo_alias_in_different_splits_is_rejected():
    docs = balanced_documents()
    prepare.assign_splits(docs, seed=17, minimum_validation_groups=12)
    train = next(item for item in docs if item["split"] == "train")
    validation = next(item for item in docs if item["split"] == "validation")
    validation["family_repositories"].append(train["repo"])
    with pytest.raises(ValueError, match="repo crosses splits"):
        prepare.validate_and_summarize(docs, minimum_validation_groups=12)


def test_export_keeps_current_and_upstream_provenance_and_loader_identity(tmp_path):
    from delta_ngram.data.raw_text import load_documents
    item = candidate("owner/repository", "text", 4)
    item.update(group_id="family", family_repositories=["owner/repository"],
                text_sha256="raw-hash", normalized_text_sha256="normalized-hash", split="train")
    exported = prepare.export_document(item, "sha256:tokenizer-identity")
    meta = exported["metadata"]
    assert meta["license"] == "MIT" and meta["detected_licenses"] == ["MIT"]
    assert meta["provenance"] == "upstream-shard.gz:17"
    assert meta["source_shard"] == "stack-edu-0075.json.gz" and meta["source_line"] == 5
    path = tmp_path / "documents.jsonl"
    path.write_text(json.dumps(exported) + "\n", encoding="utf-8")
    loaded = load_documents(path, tokenizer_identity="sha256:tokenizer-identity", require_record_identity=True)
    assert loaded[0].token_ids == tuple(item["token_ids"])
    with pytest.raises(ValueError, match="tokenizer_identity differs"):
        load_documents(path, tokenizer_identity="wrong-identity", require_record_identity=True)
    exported["split"] = "holdout"
    path.write_text(json.dumps(exported) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split must be"):
        load_documents(path, tokenizer_identity="sha256:tokenizer-identity", require_record_identity=True)


def test_scan_verifies_sha_and_preserves_complete_source(tmp_path):
    path = tmp_path / "stack-edu-0075.json.gz"
    good = source_record()
    rejected = source_record("other/repo", "bad")
    rejected["metadata"].update(license="", detected_licenses=[])
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write(json.dumps(good) + "\n" + json.dumps(rejected) + "\n")
    entry = {"path": path.name, "size": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    parameters = {"candidates_per_language": 2, "seed": 17, "minimum_chars": 8000, "maximum_chars": 100000}
    items, stats = prepare.scan_shard((str(path), entry, parameters))
    assert len(items) == 1 and items[0]["text"] == good["text"]
    assert items[0]["repo"] == "owner/repository"
    assert stats["rejected"]["missing_record_license"] == 1
    with pytest.raises(ValueError, match="shard hash mismatch"):
        prepare.scan_shard((str(path), {**entry, "sha256": "0" * 64}, parameters))

"""Loss-independent cohorts, donor exclusions, and honest after-write coverage."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from delta_ngram.data.raw_text import RawDocument
from delta_ngram.diagnostic_manifest import (build_diagnostic_manifest,
                                              validate_diagnostic_manifest)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def document(name, count=40, *, source="code/python", split="validation", group=None, metadata=None):
    offset = int(hashlib.sha256(name.encode()).hexdigest()[:6], 16)
    return RawDocument(name, group or "family:" + name, source, split,
                       tuple(range(offset, offset + count)),
                       tokenizer_identity="sha256:" + "a" * 64, metadata=metadata or {})


def sample(doc, *, start=0, blocks=8, block_size=4):
    return {"document_id": doc.id, "group_id": doc.group_id, "source": doc.source,
            "start": start, "blocks": blocks, "loss_tokens": (blocks - 1) * block_size,
            "token_sha256": digest(doc.token_ids[start:start + blocks * block_size])}


def build(documents, samples, **settings):
    return build_diagnostic_manifest(documents, samples, block_size=4, prefix_tokens=2,
                                     seed=104729, **settings)


def test_fixed_cohort_hashes_and_matching_are_order_independent_and_rng_free():
    docs = [document("target"), document("donor-a", 32), document("donor-b", 48)]
    samples = [sample(docs[0])]
    before, rng_before = deepcopy(samples), random.getstate()
    result = build(docs, samples, dataset_sha256="b" * 64,
                   baseline_validation_protocol_sha256="c" * 64)
    assert result == build(list(reversed(docs)), samples, dataset_sha256="b" * 64,
                           baseline_validation_protocol_sha256="c" * 64)
    assert random.getstate() == rng_before and samples == before
    assert result["baseline_validation_sample"] == samples
    assert result["baseline_validation_sample_sha256"] == digest(samples)
    assert result["tokenizer_identity"] == docs[0].tokenizer_identity
    assert result["dataset_sha256"] == "b" * 64
    assert result["source_documents_sha256"] != result["dataset_sha256"]
    assert result["manifest_sha256"] == digest({k: v for k, v in result.items() if k != "manifest_sha256"})
    assert "token_ids" not in json.dumps(result)
    validate_diagnostic_manifest(result, docs)


def test_donors_match_language_length_and_distinct_known_families():
    target = document("target", metadata={"family_repositories": ["owner/original"]})
    dev_a = document("dev-a", 32)
    dev_b = document("dev-b", 48)
    alias = document("alias", metadata={"family_repositories": ["OWNER/ORIGINAL"]})
    exact = RawDocument("exact-copy", "other-family", target.source, "validation", target.token_ids,
                        tokenizer_identity=target.tokenizer_identity)
    docs = [target, dev_a, dev_b, alias, exact, document("other-lang", source="code/java"),
            document("too-short", 24), document("too-long", 84),
            document("train-closer", split="train")]
    result = build(docs, [sample(target)])
    window = result["primary_windows"][0]
    assert {item["document_id"] for item in window["donors"]} == {dev_a.id, dev_b.id}
    assert len({item["group_id"] for item in window["donors"]}) == 2
    assert window["all_validation_donors"] and window["donor_train_fallback_count"] == 0
    for donor in window["donors"]:
        doc = next(item for item in docs if item.id == donor["document_id"])
        assert donor["source"] == target.source and donor["split"] == "validation"
        assert donor["history_blocks"] == 7 and donor["start"] % 4 == 0
        assert donor["start"] + 28 <= len(doc.token_ids)
        assert donor["token_sha256"] == digest(doc.token_ids[donor["start"]:donor["start"] + 28])
        assert .5 <= donor["document_length_ratio"] <= 2


def test_only_missing_validation_donor_slots_use_explicit_training_fallback():
    target, dev, train = document("target"), document("dev", 32), document("train", split="train")
    docs = [target, dev, train]
    result = build(docs, [sample(target)])
    window = result["primary_windows"][0]
    assert [item["split"] for item in window["donors"]] == ["validation", "train"]
    assert window["donor_status"] == "matched" and not window["all_validation_donors"]
    assert window["eligible_validation_donor_groups"] == 1
    assert window["donor_train_fallback_count"] == 1
    assert window["donor_fallback_reason"] == "fewer_than_required_eligible_validation_donor_families"
    assert result["coverage"]["primary"]["donor_train_fallback_slots"] == 1
    assert result["coverage"]["all_validation_donor_window_ids"] == []

    strict = build(docs, [sample(target)], donor_pool_policy="validation_only")
    assert len(strict["primary_windows"]) == 1  # No silent cohort reduction.
    unresolved = strict["primary_windows"][0]
    assert unresolved["donor_status"] == "insufficient_donors"
    assert len(unresolved["donors"]) == 1 and unresolved["donors_required"] == 2
    assert strict["coverage"]["primary"]["donor_unmatched_windows"] == 1
    assert strict["coverage"]["primary"]["loss_tokens"] == 28
    assert strict["coverage"]["primary"]["donor_matched_loss_tokens"] == 0


def test_donor_families_are_distinct_from_each_other_including_repository_aliases():
    target = document("target")
    a = document("a", metadata={"family_repositories": ["same/repo"]})
    b = document("b", metadata={"family_repositories": ["same/repo"]})
    c = document("c", 48)
    result = build([target, a, b, c], [sample(target)])
    donors = result["primary_windows"][0]["donors"]
    assert len(donors) == 2
    assert "c" in {item["document_id"] for item in donors}


def test_after_targets_score_next_block_deduplicate_origins_and_keep_missing_coverage():
    target = document("target", 68)
    docs = [target, document("donor-a", 80), document("donor-b", 88)]
    samples = [sample(target, blocks=8), sample(target, blocks=16), sample(target, start=32)]
    result = build(docs, samples)
    assert len(result["primary_windows"]) == 3  # Original overlapping T8/T16 windows stay intact.
    assert len(result["after_targets"]) == 5 and len(result["after_unavailable_targets"]) == 1
    assert len({(item["document_id"], item["start"], item["n_written"])
                for item in result["after_targets"]}) == 5
    point = next(item for item in result["after_targets"] if item["start"] == 0 and item["n_written"] == 16)
    assert len(point["baseline_window_ids"]) == 2
    assert point["target_start"] == 64
    assert point["history_token_sha256"] == digest(target.token_ids[:64])
    assert point["target_token_sha256"] == digest(target.token_ids[64:68])  # Block 17, never block 16.
    missing = result["after_unavailable_targets"][0]
    assert missing["start"] == 32 and missing["n_written"] == 16 and missing["missing_tokens"] == 32
    assert "target_token_sha256" not in missing
    coverage = result["coverage"]["after"]["16"]
    assert coverage["requested_targets"] == 2 and coverage["available_targets"] == 1
    assert coverage["available_groups"] == 1 and coverage["target_coverage_fraction"] == .5
    assert coverage["by_source"]["code/python"]["available_targets"] == 1
    assert set(coverage["by_group"]) == {target.group_id}


@pytest.mark.parametrize("change", [
    {"token_sha256": "0" * 64}, {"group_id": "wrong"}, {"source": "code/java"},
    {"start": 1}, {"blocks": 99}, {"loss_tokens": 99},
])
def test_saved_baseline_cannot_drift_from_the_corpus(change):
    target = document("target")
    saved = sample(target)
    saved.update(change)
    with pytest.raises(ValueError):
        build([target], [saved])


def test_primary_cannot_use_training_targets_or_duplicate_windows():
    target = document("target", split="train")
    with pytest.raises(ValueError, match="split/group/source"):
        build([target], [sample(target)])
    target = document("target")
    with pytest.raises(ValueError, match="duplicate primary"):
        build([target], [sample(target), sample(target)])


def test_document_identity_mismatch_and_holdout_are_rejected():
    from types import SimpleNamespace
    target = document("target")
    other = RawDocument("other", "other", "code/python", "validation", (2, 3),
                        tokenizer_identity="different")
    with pytest.raises(ValueError, match="one tokenizer_identity"):
        build([target, other], [sample(target)])
    no_identity = RawDocument("no-identity", "other", "code/python", "validation", (2, 3))
    with pytest.raises(ValueError, match="explicit tokenizer_identity"):
        build([target, no_identity], [sample(target)])
    holdout = SimpleNamespace(id="holdout", split="holdout")
    with pytest.raises(ValueError, match="never holdout"):
        build([target, holdout], [sample(target)])


def test_replay_rejects_modified_content_even_with_a_recomputed_manifest_hash():
    target = document("target")
    docs = [target, document("a", 48), document("b", 48)]
    result = build(docs, [sample(target)])
    changed = deepcopy(result)
    changed["primary_windows"][0]["donors"][0]["start"] += 4
    with pytest.raises(ValueError, match="content hash"):
        validate_diagnostic_manifest(changed, docs)
    changed["manifest_sha256"] = digest({k: v for k, v in changed.items() if k != "manifest_sha256"})
    with pytest.raises(ValueError, match="fixed pairing protocol"):
        validate_diagnostic_manifest(changed, docs)


@pytest.mark.parametrize("settings", [
    {"donors_per_target": 1}, {"donors_per_target": True},
    {"after_write_blocks": [4, 4]}, {"after_write_blocks": [True]},
    {"max_document_length_ratio": .9}, {"max_document_length_ratio": float("nan")},
    {"donor_pool_policy": "holdout"}, {"dataset_sha256": "not-a-hash"},
])
def test_invalid_pairing_contracts_are_rejected(settings):
    target = document("target")
    with pytest.raises(ValueError):
        build([target], [sample(target)], **settings)

"""Native full-trajectory causality, split binding and unchanged sampler replay."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from delta_ngram.data.event_streams import EventArticle, EventTrajectory
from delta_ngram.data.raw_text import RawWindowSampler, TrajectoryWindow
from delta_ngram.qwen38_train_data import (
    BOS, NativeFullTrainingData, canonical_hash, file_info,
    trajectory_frames, verify_full_cache,
)


def event(lengths=(513, 515, 129)):
    articles = tuple(EventArticle(
        f"article:{index}", tuple(range(1000 * (index + 1), 1000 * (index + 1) + length)), {})
        for index, length in enumerate(lengths))
    chunks = sum((length + 511) // 512 for length in lengths)
    bucket = next(value for value in (4, 8, 16, 32) if value >= chunks)
    return EventTrajectory("event:one", "group:event", "news_event", articles, bucket)


def scored_tokens(frames):
    return [label for step in frames if step["query"] is not None
            for label in step["query"]["labels"] if label != -100]


def written_tokens(frames):
    return [token for step in frames if step["evidence"] is not None
            for token, write in zip(step["evidence"]["input_ids"], step["evidence"]["write_mask"])
            if write]


def test_ordinary_full_targets_shift_and_no_prefix_double_write():
    tokens = tuple(range(4096))
    trajectory = TrajectoryWindow("doc", "group", "code", 8192, tokens, 8)
    frames = trajectory_frames(trajectory)
    assert len(frames) == 8
    assert frames[0]["query"] is None and frames[-1]["evidence"] is None
    assert scored_tokens(frames) == list(tokens[512:])
    assert written_tokens(frames) == list(tokens[:-512])
    query = frames[1]["query"]
    assert query["input_ids"] == list(tokens[384:1024])
    assert query["labels"][:127] == [-100] * 127
    assert query["labels"][127:639] == list(tokens[512:1024])
    assert query["labels"][-1] == -100
    assert sum(query["write_mask"]) == 0
    assert frames[1]["evidence"]["write_mask"] == [False] * 128 + [True] * 512
    assert {step["metadata"]["window_start"] for step in frames} == {8192}
    assert [step["metadata"]["memory_owner_reset"] for step in frames] == [True] + [False] * 7


def test_event_complete_warmup_article_and_local_bos_not_previous_article():
    trajectory = event()
    frames = trajectory_frames(trajectory)
    assert len(frames) == trajectory.actual_blocks == 5
    assert frames[0]["query"] is None and frames[1]["query"] is None
    assert scored_tokens(frames) == list(trajectory.articles[1].token_ids + trajectory.articles[2].token_ids)
    assert written_tokens(frames) == list(trajectory.articles[0].token_ids + trajectory.articles[1].token_ids)
    first_target = frames[2]["query"]
    assert first_target["input_ids"][:2] == [BOS, 2000]
    assert first_target["labels"][0] == 2000
    assert first_target["prefix_tokens"] == 1
    assert first_target["padding_tokens"] == 1
    assert first_target["encoder_valid_mask"][-1] is False
    assert first_target["labels"][-2:] == [-100, -100]
    evidence = frames[2]["evidence"]
    assert evidence["input_ids"][0] == 2000
    assert evidence["prefix_tokens"] == 0
    assert len(evidence["input_ids"]) == 512
    second_target = frames[3]["query"]
    assert second_target["prefix_tokens"] == 128
    assert second_target["input_ids"][:128] == list(trajectory.articles[1].token_ids[384:512])
    assert second_target["loss_target_tokens"] == 3
    assert frames[4]["query"]["input_ids"][:2] == [BOS, 3000]
    assert len({step["metadata"]["sample_id"] for step in frames}) == 1


def test_future_tokens_do_not_change_past_native_inputs_or_targets():
    first = event((512, 1024))
    later = EventTrajectory(
        first.document_id, first.group_id, first.source,
        (first.articles[0], EventArticle(first.articles[1].id,
         first.articles[1].token_ids[:512] + (777,) * 512, {})), first.blocks)
    before, after = trajectory_frames(first), trajectory_frames(later)
    for old_step, new_step in zip(before[:2], after[:2]):
        for kind in ("query", "evidence"):
            old, new = old_step[kind], new_step[kind]
            if old is None:
                assert new is None
            else:
                for field in ("input_ids", "encoder_valid_mask", "write_mask", "labels"):
                    assert old.get(field) == new.get(field)
    assert before[2]["query"]["labels"] != after[2]["query"]["labels"]


@pytest.mark.parametrize("lengths", [(128, 128), (513, 512), (512, 513), (512, 512, 512, 129)])
def test_event_tail_masks_and_real_cost_are_exact(lengths):
    trajectory = event(lengths)
    frames = trajectory_frames(trajectory)
    assert sum(step["metadata"]["loss_target_tokens"] for step in frames) == trajectory.loss_tokens
    assert sum(step["metadata"]["write_tokens"] for step in frames) == trajectory.writer_tokens
    for step in frames:
        for frame in (step["query"], step["evidence"]):
            if frame is None:
                continue
            assert 0 < len(frame["input_ids"]) <= 640
            assert len(frame["input_ids"]) % 2 == 0
            assert all(not write or valid for write, valid in
                       zip(frame["write_mask"], frame["encoder_valid_mask"]))
            if "labels" in frame:
                assert all(label == -100 or valid for label, valid in
                           zip(frame["labels"], frame["encoder_valid_mask"]))
            expected = {key: value for key, value in frame.items() if key != "frame_sha256"}
            assert frame["frame_sha256"] == canonical_hash(expected)


@pytest.fixture
def native_cache(tmp_path):
    """Use the existing full derivation fixture, including real strict loaders."""
    path = Path(__file__).with_name("test_prepare_qwen38_native_data.py")
    spec = importlib.util.spec_from_file_location("native_train_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    source, checksum, _, _ = fixture.corpus(tmp_path)
    output = tmp_path / "native"
    fixture.run(source, checksum, output)
    # A sealed evaluation artifact outside the manifest must never be opened.
    (output / "sealed_FINAL.jsonl").write_bytes(b"NOT JSON")
    return output


def test_full_cache_reuses_exact_sampler_order_and_resume(native_cache):
    data = NativeFullTrainingData(native_cache,
        expected_complete_sha256=file_info(native_cache / "COMPLETE.json")["sha256"],
        expected_manifest_sha256=file_info(native_cache / "data_manifest.json")["sha256"])
    assert data.identity["all_file_hashes_verified"]
    assert len(data.documents) == 15 and len(data.events) == 5
    actual = data.sampler(seed=17)
    original = RawWindowSampler(
        data.pools, seed=17, source_weights=data.raw["source_weights"],
        source_length_weights=data.raw["source_length_weights"])
    assert actual._variable_cost and actual.fingerprint == original.fingerprint
    assert actual.next_batch(12) == original.next_batch(12)
    saved = copy.deepcopy(actual.state_dict())
    tail = actual.next_batch(12, remaining_loss_tokens=50000)
    resumed = data.sampler(seed=17)
    resumed.load_state_dict(saved)
    assert resumed.next_batch(12, remaining_loss_tokens=50000) == tail
    assert resumed.state_dict() == actual.state_dict()
    assert all(item.document_id not in {d.id for d in data.documents if d.split == "validation"}
               for items in data.pools.values() for item in items)
    for item in tail:
        frames = data.frames(item)
        assert sum(step["metadata"]["loss_target_tokens"] for step in frames) == item.loss_tokens


def test_cached_validation_trajectory_is_rejected(native_cache):
    data = NativeFullTrainingData(native_cache)
    document = next(d for d in data.documents if d.split == "validation")
    window = TrajectoryWindow(document.id, document.group_id, document.source,
                              0, document.token_ids[:4096], 8)
    with pytest.raises(ValueError, match="outside the training split"):
        data.frames(window)


def test_hash_mutation_and_calibration_subset_cannot_be_training(native_cache):
    path = native_cache / "tokens.uint32"
    value = bytearray(path.read_bytes())
    value[0] ^= 1
    path.write_bytes(value)
    with pytest.raises(ValueError, match="hash differs"):
        verify_full_cache(native_cache)
    # A worker-side size check is explicitly weaker and never claims hashing.
    _, identity = verify_full_cache(native_cache, verify_files=False)
    assert identity["all_file_hashes_verified"] is False
    seal_path = native_cache / "COMPLETE.json"
    seal = json.loads(seal_path.read_bytes())
    seal["derivation_scope"] = "calibration_subset_only"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(ValueError, match="full native training cache"):
        verify_full_cache(native_cache, verify_files=False)

import copy
import json

import pytest

from delta_ngram.qwen38_forward_recovery import (
    acknowledge_evaluation, advance_writes, evaluation_policy, initial_cursor,
    pending_evaluation, required_evaluations, recover_ledger, validate_cursor,
    validate_evaluation_artifacts, validate_recovery,
)
from delta_ngram.qwen38_seven_forward import digest


def acknowledge(cursor, receipts):
    name, _ = pending_evaluation(cursor)
    receipts[name] = {"complete": True}
    acknowledge_evaluation(cursor, name)
    validate_recovery(cursor, receipts)


def restored(cursor, receipts):
    # Checkpoints cross a serialization boundary, rather than sharing objects.
    value = json.loads(json.dumps({"cursor": cursor, "evaluations": receipts}))
    validate_recovery(value["cursor"], value["evaluations"])
    return value["cursor"], value["evaluations"]


def test_base_and_pre_eval_interruptions_do_not_skip_queries():
    cursor, receipts = initial_cursor(), {}
    cursor, receipts = restored(cursor, receipts)
    assert pending_evaluation(cursor) == ("boundary_00_base", ["math", "code"])
    # A finished on-disk TF artifact is not a cursor acknowledgment. A crash
    # between TF completion and the next checkpoint causes a safe replay.
    saved_pending = copy.deepcopy(cursor)
    acknowledge(cursor, receipts)
    replay, replay_receipts = restored(saved_pending, {})
    assert pending_evaluation(replay)[0] == "boundary_00_base"
    acknowledge(replay, replay_receipts)
    assert pending_evaluation(replay)[0] == "task_01_sme_before"
    replay, replay_receipts = restored(replay, replay_receipts)
    acknowledge(replay, replay_receipts)
    assert replay["phase"] == "write" and replay["writes_completed"] == 0


def test_last_write_becomes_valid_pending_boundary_not_example_5000():
    cursor, receipts = initial_cursor(), {}
    acknowledge(cursor, receipts)
    acknowledge(cursor, receipts)
    advance_writes(cursor, 4999, 9998)
    advance_writes(cursor, 1, 2)
    assert cursor == {"task_index": 1, "example_index": 0, "writes_completed": 5000,
                      "real_evidence_tokens": 10000, "phase": "boundary_eval"}
    cursor, receipts = restored(cursor, receipts)
    assert pending_evaluation(cursor)[0] == "boundary_01_sme"
    assert "boundary_01_sme" not in receipts


def test_final_boundary_must_replay_before_complete_and_all_15_evaluations_exist():
    cursor, receipts = initial_cursor(), {}
    acknowledge(cursor, receipts)
    for stage in range(1, 8):
        acknowledge(cursor, receipts)  # pre-test
        advance_writes(cursor, 5000, 10000)
        cursor, receipts = restored(cursor, receipts)
        assert cursor["phase"] == "boundary_eval"
        assert len(receipts) == stage * 2
        acknowledge(cursor, receipts)
    assert cursor["phase"] == "complete" and cursor["writes_completed"] == 35000
    assert pending_evaluation(cursor) is None and len(required_evaluations(cursor)) == len(receipts) == 15
    del receipts["boundary_07_amh"]
    with pytest.raises(ValueError, match="acknowledgments"):
        validate_recovery(cursor, receipts)


@pytest.mark.parametrize("field,value", [("example_index", 5000), ("phase", "complete"), ("writes_completed", 1)])
def test_inconsistent_cursor_is_rejected(field, value):
    cursor = initial_cursor()
    cursor[field] = value
    with pytest.raises(ValueError):
        validate_cursor(cursor)


def test_bounded_evaluation_and_capture_policy_cannot_equal_full_policy():
    assert evaluation_policy(1, True) != evaluation_policy(0, True)
    assert evaluation_policy(0, False) != evaluation_policy(0, True)


def test_restore_earlier_checkpoint_archives_future_and_torn_ledger(tmp_path):
    path = tmp_path / "write_events.jsonl"
    original = b"".join(json.dumps({"kind": "write", "write_index": i}).encode() + b"\n" for i in range(1, 6))
    original += b'{"kind":"write","write_index":'
    path.write_bytes(original)
    receipt = recover_ledger(path, writes_completed=3, archive_dir=tmp_path / "audit" / "attempt1")
    assert (tmp_path / "audit" / "attempt1" / "write_events.before_resume.jsonl").read_bytes() == original
    assert [json.loads(line)["write_index"] for line in path.read_bytes().splitlines()] == [1, 2, 3]
    with path.open("ab") as stream:
        stream.write(json.dumps({"kind": "write", "write_index": 4}).encode() + b"\n")
    assert [json.loads(line)["write_index"] for line in path.read_bytes().splitlines()] == [1, 2, 3, 4]
    assert receipt["committed_writes"] == 3


@pytest.mark.parametrize("rows", [[1, 1, 2], [1, 3], [1]])
def test_missing_or_duplicate_committed_ledger_writes_are_rejected(tmp_path, rows):
    path = tmp_path / "write_events.jsonl"
    path.write_text("".join(json.dumps({"kind": "write", "write_index": i}) + "\n" for i in rows))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        recover_ledger(path, writes_completed=2, archive_dir=tmp_path / "audit")
    assert path.read_bytes() == before


def artifact_fixture(tmp_path, capture_layers=True):
    name = "boundary_00_base"
    directory = tmp_path / "tf" / name
    directory.mkdir(parents=True)
    records = [{"task_id": "x", "token_count": 2, "prompt_token_count": 1}]
    predictions = directory / "predictions.jsonl"
    predictions.write_text(json.dumps({**records[0], "loss_sum": 3.}) + "\n")
    files = []
    if capture_layers:
        (directory / "representations").mkdir()
        for rank in range(8):
            path = directory / "representations" / f"query-0000-rank-{rank:02d}.pt"
            path.write_bytes(f"rank{rank}".encode())
            files.append({"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size,
                          "sha256": digest(path)})
    summary = {"status": "completed", "identity": {"reader": "locked"}, "cursor": initial_cursor(),
        "capture_layers": capture_layers, "task_count": 1, "completed_tasks": 1,
        "encoded_records": records, "predictions_sha256": digest(predictions), "capture_files": files}
    target = directory / "summary.json"
    target.write_text(json.dumps(summary))
    receipt = {"summary_path": f"tf/{name}/summary.json", "summary_sha256": digest(target), "cursor": initial_cursor()}
    kwargs = {"identity": summary["identity"], "expected_records": records, "capture_layers": capture_layers}
    return name, directory, target, summary, receipt, kwargs


def test_complete_evaluation_artifacts_validate(tmp_path):
    name, _, _, _, receipt, kwargs = artifact_fixture(tmp_path)
    validate_evaluation_artifacts(tmp_path, name, receipt, **kwargs)


def test_missing_rank_capture_prevents_evaluation_completion(tmp_path):
    name, directory, _, _, receipt, kwargs = artifact_fixture(tmp_path)
    (directory / "representations/query-0000-rank-07.pt").unlink()
    with pytest.raises(ValueError, match="capture is missing"):
        validate_evaluation_artifacts(tmp_path, name, receipt, **kwargs)


def test_changed_predictions_prevents_evaluation_completion(tmp_path):
    name, directory, _, _, receipt, kwargs = artifact_fixture(tmp_path)
    (directory / "predictions.jsonl").write_text("")
    with pytest.raises(ValueError, match="predictions"):
        validate_evaluation_artifacts(tmp_path, name, receipt, **kwargs)

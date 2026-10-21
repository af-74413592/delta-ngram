"""CPU-only cursor phases and durable ledger recovery for the seven-task run."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from .qwen38_seven_forward import TASKS, require, digest

RECOVERY_PROTOCOL = "qwen38_forward_pending_evaluation_v1"


def initial_cursor():
    return {"task_index": 0, "example_index": 0, "writes_completed": 0,
            "real_evidence_tokens": 0, "phase": "base_eval"}


def evaluation_policy(query_limit, capture_layers):
    require(type(query_limit) is int and query_limit >= 0 and type(capture_layers) is bool,
            "invalid immutable evaluation policy")
    return {"query_limit_per_domain": query_limit, "capture_layers": capture_layers,
            "reference_population": "locked_encoded_queries", "capture_positions": "all_five_named_positions"}


def evaluation_name(stage, *, before=False):
    require(1 <= stage <= len(TASKS), "invalid evaluation stage")
    return (f"task_{stage:02d}_{TASKS[stage - 1]}_before" if before
            else f"boundary_{stage:02d}_{TASKS[stage - 1]}")


def validate_cursor(cursor):
    t, e, w = (cursor.get(name) for name in ("task_index", "example_index", "writes_completed"))
    phase = cursor.get("phase")
    require(all(type(value) is int for value in (t, e, w, cursor.get("real_evidence_tokens")))
            and 0 <= t <= len(TASKS) and 0 <= e < 5000 and w == t * 5000 + e
            and cursor["real_evidence_tokens"] >= 0, "invalid next-example recovery cursor")
    valid = ((phase == "base_eval" and t == e == 0)
             or (phase == "pre_eval" and t < len(TASKS) and e == 0)
             or (phase == "write" and t < len(TASKS))
             or (phase == "boundary_eval" and 1 <= t <= len(TASKS) and e == 0)
             or (phase == "complete" and t == len(TASKS) and e == 0))
    require(valid, "cursor phase is inconsistent with the task/write boundary")
    return cursor


def required_evaluations(cursor):
    """Names acknowledged before this cursor; the pending evaluation is excluded."""
    validate_cursor(cursor)
    if cursor["phase"] == "base_eval":
        return []
    names = ["boundary_00_base"]
    t = cursor["task_index"]
    for stage in range(1, t + 1):
        names.append(evaluation_name(stage, before=True))
        if not (stage == t and cursor["phase"] == "boundary_eval"):
            names.append(evaluation_name(stage))
    if cursor["phase"] == "write":
        names.append(evaluation_name(t + 1, before=True))
    return names


def validate_recovery(cursor, evaluations):
    required = required_evaluations(cursor)
    require(isinstance(evaluations, dict) and set(evaluations) == set(required),
            "checkpoint is missing required evaluation acknowledgments or contains future ones")
    return required


def pending_evaluation(cursor):
    validate_cursor(cursor)
    t, phase = cursor["task_index"], cursor["phase"]
    if phase == "base_eval":
        return "boundary_00_base", ["math", "code"]
    if phase == "pre_eval":
        return evaluation_name(t + 1, before=True), [TASKS[t]]
    if phase == "boundary_eval":
        return evaluation_name(t), [*TASKS[:t], "math", "code"]
    return None


def acknowledge_evaluation(cursor, name):
    pending = pending_evaluation(cursor)
    require(pending is not None and pending[0] == name, "evaluation acknowledgment is out of order")
    phase = cursor["phase"]
    cursor["phase"] = ("pre_eval" if phase == "base_eval" else "write" if phase == "pre_eval"
                       else "complete" if cursor["task_index"] == len(TASKS) else "pre_eval")
    validate_cursor(cursor)


def advance_writes(cursor, count, tokens):
    validate_cursor(cursor)
    require(cursor["phase"] == "write" and type(count) is int and count > 0
            and cursor["example_index"] + count <= 5000 and type(tokens) is int and tokens > 0,
            "invalid chronological D1 advance")
    cursor["example_index"] += count
    cursor["writes_completed"] += count
    cursor["real_evidence_tokens"] += tokens
    if cursor["example_index"] == 5000:
        cursor.update(task_index=cursor["task_index"] + 1, example_index=0, phase="boundary_eval")
    validate_cursor(cursor)


def recover_ledger(path, *, writes_completed, archive_dir):
    """Preserve the full prior attempt, then retain exactly the committed prefix.

    A torn final line or future events are allowed only after every committed
    write has been accounted for. Missing or duplicate committed writes fail.
    """
    path, archive_dir = Path(path), Path(archive_dir)
    require(type(writes_completed) is int and writes_completed >= 0, "invalid ledger recovery boundary")
    raw = path.read_bytes() if path.exists() else b""
    kept, last_write = [], 0
    for line in raw.splitlines(keepends=True):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            require(last_write == writes_completed, "torn ledger before committed checkpoint cursor")
            break
        require(isinstance(row, dict), "invalid ledger record")
        if row.get("kind") == "write":
            index = row.get("write_index")
            if last_write == writes_completed:
                break
            require(index == last_write + 1, "missing or duplicated committed write in ledger")
            last_write = index
        elif row.get("kind") == "progress":
            if row.get("writes_completed", writes_completed + 1) > writes_completed:
                break
        else:
            raise ValueError("unrecognized ledger event")
        kept.append(line if line.endswith(b"\n") else line + b"\n")
    require(last_write == writes_completed, "ledger does not reach the restored checkpoint cursor")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = archive_dir / "write_events.before_resume.jsonl"
    with archived.open("xb") as stream:
        stream.write(raw)
    temporary = path.with_suffix(".jsonl.recovering")
    temporary.write_bytes(b"".join(kept))
    os.replace(temporary, path)
    return {"archive": str(archived), "archive_sha256": digest(archived),
            "committed_ledger_sha256": digest(path), "committed_writes": writes_completed,
            "archived_bytes": len(raw), "kept_bytes": path.stat().st_size}


def archive_existing(path, archive_dir):
    """Move a stale completion marker or replayed TF directory into the audit attempt."""
    path, archive_dir = Path(path), Path(archive_dir)
    if not path.exists():
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / path.name
    require(not target.exists(), "refuse to replace a prior audit artifact")
    shutil.move(str(path), str(target))
    return str(target)


def query_metadata(row):
    metadata = {key: row[key] for key in ("domain", "task_id", "prompt_sha256", "reference_sha256",
        "prompt_token_ids_sha256", "target_token_ids_sha256", "input_token_ids_sha256")}
    metadata.update(token_count=row["target_token_count"], prompt_token_count=row["prompt_length"])
    metadata.update({key: row[key] for key in
                    ("task", "training_labels_sha256", "excluded_suffix_token_ids") if key in row})
    return metadata


def select_queries(rows, domains, query_limit):
    selected, counts = [], {}
    for row in rows:
        domain = row["domain"]
        if domain not in domains:
            continue
        counts[domain] = counts.get(domain, 0) + 1
        if not query_limit or counts[domain] <= query_limit:
            selected.append(row)
    require(set(counts) == set(domains), "evaluation domain has no encoded queries")
    return selected


def validate_evaluation_artifacts(output, name, receipt, *, identity, expected_records, capture_layers):
    """A phase can be acknowledged only with its complete, checksum-bound artifacts."""
    output = Path(output).resolve()
    require(receipt.get("summary_path") == f"tf/{name}/summary.json", "evaluation receipt path differs")
    summary_path = output / receipt["summary_path"]
    require(summary_path.is_file() and digest(summary_path) == receipt.get("summary_sha256"),
            "acknowledged evaluation summary is missing or changed")
    summary = json.loads(summary_path.read_bytes())
    require(summary.get("status") == "completed" and summary.get("identity") == identity
            and summary.get("cursor") == receipt.get("cursor")
            and summary.get("capture_layers") is capture_layers
            and summary.get("task_count") == summary.get("completed_tasks") == len(expected_records)
            and summary.get("encoded_records") == expected_records,
            "acknowledged evaluation population/policy/identity is incomplete or changed")
    predictions = summary_path.parent / "predictions.jsonl"
    require(predictions.is_file() and digest(predictions) == summary.get("predictions_sha256"),
            "acknowledged evaluation predictions are missing or changed")
    scored = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    require(len(scored) == len(expected_records) and all(
        all(row.get(key) == value for key, value in metadata.items())
        for row, metadata in zip(scored, expected_records)), "scored query population differs from capture manifest")
    files = summary.get("capture_files")
    require(isinstance(files, list), "evaluation capture file manifest is missing")
    topology = summary.get("topology")
    topology_keys = ("tp", "pp", "ep", "cp", "dp")
    require(isinstance(topology, dict) and set(topology) == set(topology_keys)
            and all(type(topology[key]) is int for key in topology_keys)
            and tuple(topology[key] for key in topology_keys) in ((2, 2, 2, 1, 1), (2, 4, 2, 1, 1)),
            "evaluation topology is missing or is not a registered native layout")
    # EP partitions experts within the model's existing parallel groups; it is
    # not an additional multiplier of the ranks writing capture files.
    world_size = topology["tp"] * topology["pp"] * topology["cp"] * topology["dp"]
    expected = ({f"representations/query-{index:04d}-rank-{rank:02d}.pt"
                 for index in range(len(expected_records)) for rank in range(world_size)} if capture_layers else set())
    require(len(files) == len(expected) and {row.get("path") for row in files} == expected,
            "evaluation capture files do not match the registered topology and query population")
    for row in files:
        path = summary_path.parent / row["path"]
        require(path.is_file() and path.stat().st_size == row.get("bytes") and digest(path) == row.get("sha256"),
                "acknowledged evaluation capture is missing or changed")
    return summary

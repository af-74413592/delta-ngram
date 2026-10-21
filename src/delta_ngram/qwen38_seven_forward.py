"""CPU-testable seven-task forward-only memory protocol and resume state.

No model, tokenizer or distributed runtime is loaded by this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch

from .alpha_memory import LazyDecayState
from .batching import BatchMemoryState
from .state import SparseMemoryState

TASKS = ("sme", "nav", "hye", "afb", "sqi", "mkd", "amh")
PROTOCOL = "qwen38_seven_task_frozen_writer_alpha1_v1"
STATE_FIELDS = ("rows", "values", "counts", "block_clock", "cumulative_log_alpha",
                "row_last_written_block", "row_log_alpha_at_write")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def state_payload(state):
    require(isinstance(state, LazyDecayState) and state.batch_size == 1
            and state.precision is None, "one-owner full lazy state is required")
    require(not bool(state.cumulative_log_alpha.ne(0).any())
            and not bool(state.row_log_alpha_at_write.ne(0).any()), "only alpha=1 is registered")
    return {"format": PROTOCOL + "/full-state", "row_capacity": state.row_capacity,
            "width": state.width, "batch_size": 1,
            **{key: getattr(state, key).detach().cpu().clone() for key in STATE_FIELDS}}


def restore_state(payload, *, device="cpu"):
    require(payload.get("format") == PROTOCOL + "/full-state" and payload.get("batch_size") == 1,
            "not a complete one-owner forward-writer resume state")
    values = {key: payload[key].detach().to(device=device, copy=True) for key in STATE_FIELDS}
    require(values["values"].dtype == torch.float32 and values["values"].shape[-1] == payload["width"],
            "resume values must remain full FP32 rows")
    sparse = SparseMemoryState(values["rows"], values["values"], values["counts"])
    memory = BatchMemoryState(sparse, 1, payload["row_capacity"])
    result = LazyDecayState(memory, *(values[key] for key in STATE_FIELDS[3:]))
    require(not bool(result.cumulative_log_alpha.ne(0).any())
            and not bool(result.row_log_alpha_at_write.ne(0).any()), "resume alpha differs from one")
    return result


def state_fingerprint(state):
    result = hashlib.sha256()
    for key in STATE_FIELDS:
        value = getattr(state, key).detach().contiguous().cpu()
        result.update(key.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def immutable_guard(state):
    return tuple((getattr(state, key), getattr(state, key)._version) for key in STATE_FIELDS)


def assert_unchanged(guard):
    require(all(value._version == version for value, version in guard),
            "read-only query changed memory rows, values, counts or clocks")


def evidence_batch(rows, *, pad_token_id, device="cpu"):
    """Independent examples; causal attention/positions restart in every lane."""
    require(bool(rows), "empty evidence batch")
    lengths = [len(row["input_ids"]) for row in rows]
    require(all(0 < length <= 128 for length in lengths), "evidence exceeds the locked 128-token ceiling")
    length = max(2, (max(lengths) + 1) // 2 * 2)
    ids = torch.full((len(rows), length), pad_token_id, dtype=torch.long, device=device)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for lane, (row, size) in enumerate(zip(rows, lengths)):
        ids[lane, :size] = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        valid[lane, :size] = True
    return {"batch": {"input_ids": ids, "position_ids": torch.arange(length, device=device)
                      .expand(len(rows), -1).clone(), "attention_mask": None, "labels": None},
            "valid_mask": valid, "owner_ids": torch.arange(len(rows), device=device),
            "lengths": lengths, "real_tokens": sum(lengths), "padded_tokens": ids.numel()}


def validate_encoded_record(row, *, evidence):
    ids = row.get("input_ids")
    require(isinstance(ids, list) and bool(ids) and all(type(token) is int and 0 <= token < 248320 for token in ids),
            "invalid encoded token sequence")
    require(row.get("input_token_ids_sha256") == canonical_hash(ids), "encoded token sequence hash mismatch")
    if evidence:
        require(row.get("task") in TASKS and len(ids) <= 128 and row.get("write_events") == 1,
                "each support example must be one independent short D1 event")
    else:
        p, targets = row.get("prompt_length"), row.get("target_ids")
        require(type(p) is int and p > 0 and isinstance(targets, list) and bool(targets)
                and ids[p:p + len(targets)] == targets and row.get("target_token_count") == len(targets),
                "TF answer boundary is invalid")
    return row

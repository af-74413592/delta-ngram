"""Strict CPU contracts for the bounded native memory-aware acceptance run.

This is an acceptance protocol, not the seven-task experiment. Its fixed budget
includes one extra write on a restored branch and two long-context baselines.
"""
from __future__ import annotations

import hashlib

import torch

from .qwen38_seven_forward import PROTOCOL, STATE_FIELDS, require

SHORTCHAIN_PROTOCOL = "qwen38_native_memory_shortchain_alpha1_v1"


def shortchain_plan(tasks=("sme", "nav"), frames_per_task=32, query_limit=1, restore_after=16):
    require(tuple(tasks) == ("sme", "nav"), "shortchain task order must be sme,nav")
    require(type(frames_per_task) is int and frames_per_task == 32,
            "shortchain requires exactly 32 writes per task")
    require(type(query_limit) is int and query_limit == 1,
            "shortchain requires one query per domain and node")
    require(type(restore_after) is int and restore_after == 16,
            "shortchain restore boundary must be write 16")
    nodes = [
        {"name": "base", "writes_completed": 0, "domains": ["math", "code"]},
        {"name": "task_01_sme_before", "writes_completed": 0, "domains": ["sme"]},
        {"name": "after_task_01_sme", "writes_completed": 32, "domains": ["sme", "math", "code"]},
        {"name": "task_02_nav_before", "writes_completed": 32, "domains": ["nav"]},
        {"name": "after_task_02_nav", "writes_completed": 64, "domains": ["sme", "nav", "math", "code"]},
    ]
    pairs = [
        {"metric": "K", "before": "base", "after": "after_task_01_sme", "domains": ["math", "code"]},
        {"metric": "K", "before": "base", "after": "after_task_02_nav", "domains": ["math", "code"]},
        {"metric": "ki", "before": "after_task_01_sme", "after": "after_task_02_nav", "domains": ["sme"]},
    ]
    return {"protocol": SHORTCHAIN_PROTOCOL, "task_order": list(tasks),
            "frames_per_task": frames_per_task, "query_limit": query_limit,
            "restore_after": restore_after, "restore_branch_extra_writes": 1,
            "total_writes": 64, "evaluation_nodes": nodes, "capture_pairs": pairs,
            "paired_node_queries": 11,
            "extra_evaluations": [{"name": "long_context_base", "domains": ["math", "code"],
                "selection": "longest_input_per_domain", "writes_completed": 0}],
            "extra_baseline_queries": 2, "total_queries": 13, "total_capture_files": 104}


def validate_shortchain_cursor(cursor, plan):
    require(isinstance(plan, dict) and plan == shortchain_plan(), "unknown shortchain plan")
    require(isinstance(cursor, dict), "shortchain cursor must be a mapping")
    for field in ("writes_completed", "task_index", "example_index"):
        require(type(cursor.get(field)) is int and cursor[field] >= 0,
                f"invalid shortchain cursor: {field}")
    writes = cursor["writes_completed"]
    require(writes <= plan["total_writes"] and
            (cursor["task_index"], cursor["example_index"]) == divmod(writes, plan["frames_per_task"]),
            "shortchain cursor does not describe a complete D1 event boundary")
    if "real_evidence_tokens" in cursor:
        require(type(cursor["real_evidence_tokens"]) is int and cursor["real_evidence_tokens"] >= 0,
                "invalid shortchain evidence token count")
    return cursor


def _tensor_bytes(value):
    return value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def validate_state_payload(payload, *, expected_writes=None):
    """Validate the complete serialized CPU state, including lazy alpha clocks."""
    require(isinstance(payload, dict), "full state payload must be a mapping")
    require(payload.get("format") == PROTOCOL + "/full-state"
            and type(payload.get("batch_size")) is int and payload["batch_size"] == 1,
            "full state must use the one-owner native alpha=1 format")
    require(type(payload.get("row_capacity")) is int and payload["row_capacity"] > 1
            and type(payload.get("width")) is int and payload["width"] == 160,
            "full state capacity or native row width differs")
    for name in STATE_FIELDS:
        value = payload.get(name)
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.layout == torch.strided,
                f"full state requires CPU tensor: {name}")
    rows = payload["rows"]
    require(rows.ndim == 1, "full state rows must be one-dimensional")
    size = rows.numel()
    specs = {"rows": (torch.int64, (size,)), "values": (torch.float32, (size, 160)),
             "counts": (torch.float32, (size,)), "block_clock": (torch.int64, (1,)),
             "cumulative_log_alpha": (torch.float64, (1,)),
             "row_last_written_block": (torch.int64, (size,)),
             "row_log_alpha_at_write": (torch.float64, (size,))}
    for name, (dtype, shape) in specs.items():
        value = payload[name]
        require(value.dtype == dtype and tuple(value.shape) == shape,
                f"full state tensor shape/dtype differs: {name}; FP32 values and counts are required")
        require(bool(torch.isfinite(value).all()), f"full state contains nonfinite {name}")
    require(bool((rows > 0).all()) and bool((rows <= payload["row_capacity"]).all())
            and (size < 2 or bool((rows[1:] > rows[:-1]).all())),
            "full state physical rows must be sorted unique non-padding addresses")
    require(bool((payload["counts"] >= 0).all()), "full state counts must be nonnegative")
    clock = int(payload["block_clock"][0])
    require(clock >= 0 and bool((payload["row_last_written_block"] >= 0).all())
            and bool((payload["row_last_written_block"] <= clock).all()),
            "full state row clocks exceed the block clock")
    require(not bool(payload["cumulative_log_alpha"].ne(0).any())
            and not bool(payload["row_log_alpha_at_write"].ne(0).any()),
            "full state alpha must remain exactly one")
    if expected_writes is not None:
        require(type(expected_writes) is int and expected_writes >= 0 and clock == expected_writes,
                "full state block clock differs from completed writes")
    digest = hashlib.sha256()
    for name in STATE_FIELDS:
        value = payload[name]
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(_tensor_bytes(value))
    return {"fingerprint": digest.hexdigest(), "rows": size, "writes_completed": clock,
            "row_capacity": payload["row_capacity"], "width": payload["width"],
            "values_l2": float(torch.linalg.vector_norm(payload["values"].double())),
            "values_bytes": payload["values"].numel() * payload["values"].element_size(),
            "state_bytes": sum(payload[name].numel() * payload[name].element_size() for name in STATE_FIELDS)}


def assert_state_payload_equal(left, right):
    """Require bytewise equality of all seven fields, including signed zeros."""
    left_stats, right_stats = validate_state_payload(left), validate_state_payload(right)
    for field in ("format", "batch_size", "row_capacity", "width"):
        require(left[field] == right[field], f"restored full state metadata differs: {field}")
    for field in STATE_FIELDS:
        require(left[field].dtype == right[field].dtype and left[field].shape == right[field].shape
                and _tensor_bytes(left[field]) == _tensor_bytes(right[field]),
                f"restored full state is not bitwise equal: {field}")
    require(left_stats["fingerprint"] == right_stats["fingerprint"], "restored full state fingerprint differs")
    return left_stats["fingerprint"]

"""CPU contracts for an engineering capacity probe, never a learned endpoint."""
from __future__ import annotations

import torch

from .qwen38_seven_forward import TASKS, STATE_FIELDS, require
from .qwen38_shortchain_protocol import validate_state_payload

CAPACITY_PROTOCOL = "qwen38_native_zero_padded_capacity_probe_v1"


def select_capacity_frames(data):
    """Sixteen distinct real supports with all seven longest frames included."""
    selected = []
    for task in TASKS:
        rows = sorted(data[task], key=lambda row: (len(row["input_ids"]), row["source_row_index"]))
        require(len(rows) == 5000, "capacity probe needs the sealed 5000-row supports")
        selected.extend((rows[len(rows)//2], rows[-1]))
    seen = {(row["task"], row["source_row_index"]) for row in selected}
    for task in ("sme", "nav"):
        row = next(row for row in data[task] if row["source_row_index"] >= 32
                   and (task, row["source_row_index"]) not in seen)
        selected.append(row)
        seen.add((task, row["source_row_index"]))
    require(len(selected) == len(seen) == 16, "capacity frame selection must be unique")
    require(max(len(row["input_ids"]) for row in selected) ==
            max(len(row["input_ids"]) for task in TASKS for row in data[task]),
            "capacity frames must include the longest support")
    return selected


def expand_zero_capacity(seed, union_rows, *, expected_rows=666747):
    """Preserve all seed rows exactly; allocate zero values/counts in other rows.

    The clock remains at the seed's real update count. Extra addresses do not
    represent prior writes and cannot be reported as a 35000-write result.
    """
    source = validate_state_payload(seed)
    require(isinstance(union_rows, torch.Tensor) and union_rows.device.type == "cpu"
            and union_rows.dtype == torch.int64 and union_rows.ndim == 1,
            "capacity universe must be a CPU int64 vector")
    require(union_rows.numel() == expected_rows and expected_rows > 0
            and bool((union_rows > 0).all()) and bool((union_rows <= seed["row_capacity"]).all())
            and bool((union_rows[1:] > union_rows[:-1]).all()),
            "capacity universe must have the exact sorted unique native addresses")
    slots = torch.searchsorted(union_rows, seed["rows"])
    require(bool((slots < expected_rows).all()) and torch.equal(union_rows[slots], seed["rows"]),
            "capacity universe omitted a real seed address")
    result = {key: seed[key] for key in ("format", "row_capacity", "width", "batch_size")}
    result["rows"] = union_rows.clone()
    for name in ("values", "counts", "row_last_written_block", "row_log_alpha_at_write"):
        value = seed[name]
        result[name] = torch.zeros((expected_rows, *value.shape[1:]), dtype=value.dtype)
        result[name][slots] = value
        require(torch.equal(result[name][slots].reshape(-1).view(torch.uint8),
                            value.contiguous().reshape(-1).view(torch.uint8)),
                f"capacity padding changed existing {name} bytes")
    for name in ("block_clock", "cumulative_log_alpha"):
        result[name] = seed[name].clone()
    stats = validate_state_payload(result, expected_writes=source["writes_completed"])
    require(torch.equal(result["rows"][slots], seed["rows"]), "capacity padding changed seed addresses")
    return result, {"protocol": CAPACITY_PROTOCOL, "scientific_endpoint": False,
                    "seed_rows": source["rows"], "zero_placeholder_rows": expected_rows-source["rows"],
                    "seed_writes": source["writes_completed"], "capacity": stats,
                    "semantics": "real shortchain values retained; other rows have zero values/counts and no write history"}

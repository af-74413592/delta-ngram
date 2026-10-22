"""Bounded ordinary-writer batching probe helpers; no model or CUDA setup."""
from __future__ import annotations

import torch

from .qwen38_seven_forward import evidence_batch, require

BENCHMARK_PROTOCOL = "qwen38_ordinary_writer_batch_benchmark_v1"
MODES = ("b1_canonical", "b1_matched_padding", "b12")


def select_frames(data):
    rows = [row for task in ("sme", "nav") for row in data[task][:12]]
    require(len(rows) == 24 and [(r["task"], r["source_row_index"]) for r in rows]
            == [(task, i) for task in ("sme", "nav") for i in range(12)],
            "benchmark requires the first twelve independent SME and NAV supports")
    return rows


def make_jobs(rows, mode):
    require(mode in MODES and len(rows) == 24, "invalid bounded benchmark plan")
    jobs = []
    for start in (0, 12):
        group = rows[start:start + 12]
        matched = max(2, (max(len(row["input_ids"]) for row in group) + 1) // 2 * 2)
        width = 12 if mode == "b12" else 1
        for offset in range(0, 12, width):
            current = group[offset:offset + width]
            padded = (max(2, (len(current[0]["input_ids"]) + 1) // 2 * 2)
                      if mode == "b1_canonical" else matched)
            jobs.append({"indices": list(range(start + offset, start + offset + width)),
                         "rows": current, "sequence_length": padded})
    require([i for job in jobs for i in job["indices"]] == list(range(24)),
            "batching changed chronological D1 order")
    return jobs


def padded_batch(rows, sequence_length, *, pad_token_id, device="cpu"):
    batch = evidence_batch(rows, pad_token_id=pad_token_id, device=device)
    current = batch["batch"]["input_ids"].shape[1]
    require(current <= sequence_length <= 128 and sequence_length % 2 == 0,
            "invalid common evidence padding")
    if current != sequence_length:
        width = len(rows)
        ids = torch.full((width, sequence_length), pad_token_id, dtype=torch.long, device=device)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        ids[:, :current] = batch["batch"]["input_ids"]
        valid[:, :current] = batch["valid_mask"]
        batch["batch"]["input_ids"] = ids
        batch["batch"]["position_ids"] = torch.arange(sequence_length, device=device).expand(width, -1).clone()
        batch["valid_mask"] = valid
        batch["padded_tokens"] = ids.numel()
    require(batch["valid_mask"].sum().item() == batch["real_tokens"], "pad became evidence")
    return batch


def compare_tensor(actual, reference):
    require(actual.shape == reference.shape and actual.dtype == reference.dtype,
            "tensor comparison layout differs")
    require(bool(actual.isfinite().all()) and bool(reference.isfinite().all()),
            "tensor comparison contains nonfinite values")
    unequal = int(actual.ne(reference).sum())
    difference = actual.double() - reference.double()
    norm = float(reference.double().norm())
    absolute_l2 = float(difference.norm())
    return {"shape": list(actual.shape), "dtype": str(actual.dtype), "numel": actual.numel(),
            "bitwise_equal": torch.equal(actual.contiguous().view(torch.uint8),
                                         reference.contiguous().view(torch.uint8)),
            "unequal_elements": unequal, "max_abs": float(difference.abs().max()) if actual.numel() else 0.,
            "absolute_l2": absolute_l2, "relative_l2": absolute_l2 / max(norm, 1e-30)}

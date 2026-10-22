"""Matched local-B1 evidence encoders for the frozen DEV64 evaluator."""
from __future__ import annotations

from qwen38_native_memory_dev64_eval import MemoryEvidenceWriter


class B1EvidenceWriter(MemoryEvidenceWriter):
    """Only the evidence reader's memory input differs between the two cells.

    Both paths rebuild each self/donor trajectory and call the same native
    encoder at B1, including inactive-rank padding. There is no H12 cache.
    """

    def __init__(self, evidence_mode, charge_encoding=None):
        if evidence_mode not in ("base", "memory"):
            raise ValueError("evidence mode must be base or memory")
        super().__init__("full", charge_encoding)
        self.evidence_mode = evidence_mode
        self.cache_base_evidence = False

    def metadata(self):
        return {
            **super().metadata(), "protocol": "qwen38_matched_b1_evidence_v1",
            "mode": self.evidence_mode, "base_cache_batch_limit": None,
            "actual_evidence_batch_limit": 1, "query_batch_limit": 4,
            "base_cache": False, "writer_hidden": "rmsnorm(H_new), eps=1e-6",
            "base_evidence": "frozen T0 without memory, same B1 native framing",
            "memory_evidence": "detached pre-write own state, same B1 native framing",
            "training_gradient": "evaluation only; no_grad; zero backward",
            "legacy_manifest_evidence_batch_limit": "12 is the disabled cache limit; actual B1 logged per call",
        }

    def write_article(self, session, writer, alpha, state, frames, *, base_blocks,
                      gather, batches, execution, begin_batch, finish_batch,
                      condition, distance):
        import torch
        from delta_ngram.qwen38_memory_batching import collate_memory_encoding_frames
        from delta_ngram.qwen38_memory_writer import write_memory_aware_native_block
        from delta_ngram.qwen38_train_batching import crop_evidence_lanes
        from delta_ngram.qwen38_training import writer_evidence_features
        from qwen38_native_memory_schedule import collect_memory_encoding_microbatches

        if len(frames) != len(base_blocks) or any(item is not None for item in base_blocks):
            raise ValueError("matched B1 evidence must not receive cached hidden blocks")
        for index, size in batches(len(frames), 1):
            if size != 1:
                raise ValueError("matched evidence requires local B1")
            started = begin_batch()
            active = index < len(frames)
            frame = frames[index] if active else None
            state_args = ([state if active else None],) if self.evidence_mode == "memory" else ()
            mb = collate_memory_encoding_frames(
                [frame], *state_args, device=session.device,
                row_capacity=session.layout.row_capacity, width=session.layout.row_width)
            cost = self._charge(mb, gather, self.evidence_mode)
            execution["native_evidence_calls"] += 1
            execution["evidence_input_tokens"] += cost["real_input_tokens"]
            captured = collect_memory_encoding_microbatches(
                session, [mb], encoding=self.evidence_mode)["microbatches"][0]
            block = crop_evidence_lanes(mb, captured["hidden"], captured["physical_rows"])[0]
            if active:
                if block is None:
                    raise RuntimeError("B1 encoder dropped an arrived DEV64 block")
                hidden = writer_evidence_features(
                    block["H_new"].to(session.device), mode="rmsnorm", eps=1e-6)
                state = write_memory_aware_native_block(
                    writer, state, hidden, block["physical_rows_new"].to(session.device),
                    torch.ones(hidden.shape[:2], dtype=torch.bool, device=session.device), alpha,
                    hidden_base=None)
                if not bool(state.values.isfinite().all()):
                    raise RuntimeError("nonfinite matched B1 DEV64 writer state")
            del captured, block, mb
            finish_batch("evidence", started, batch_size=1,
                         local_active_frames=int(active), condition=condition, distance=distance)
        return state

"""Registered memory-aware writing for the unchanged DEV64 scoring protocol.

Every self/donor state encodes its own arrived evidence. Inactive EP ranks
participate with padding; direction permutations are exclusively scoring views.
"""
from __future__ import annotations

from qwen38_native_dev64_eval import evaluate_dev64


class MemoryEvidenceWriter:
    def __init__(self, encoding_mode, charge_encoding=None, *, difference_baseline="actual_base"):
        if encoding_mode not in ("full", "residual"):
            raise ValueError("memory encoding mode must be full or residual")
        if charge_encoding is not None and not callable(charge_encoding):
            raise TypeError("charge_encoding must be callable")
        if (difference_baseline not in ("actual_base", "matched_base")
                or encoding_mode == "full" and difference_baseline != "actual_base"):
            raise ValueError("matched difference baseline requires residual encoding")
        self.encoding_mode = encoding_mode
        self.difference_baseline = difference_baseline
        self.cache_base_evidence = encoding_mode == "residual"
        self.charge_encoding = charge_encoding

    def metadata(self):
        return {
            "protocol": "qwen38_memory_encoding_v1", "mode": self.encoding_mode,
            "memory_encoding": "detached pre-write effective state; prefix readable; new tokens writable",
            "memory_encoding_batch_limit": 1, "base_cache_batch_limit": 12,
            "base_cache": self.cache_base_evidence,
            "residual_difference_baseline": self.difference_baseline,
            "matched_base_batch_limit": 1 if self.difference_baseline == "matched_base" else None,
            "state_specific": "self and donor encode separately; no H_mem reuse across states",
            "direction_control": "read-only scoring permutation; later writes remain unpermuted",
            "encoding_ce_targets": 0, "encoding_rng": "save and restore around each encoding call",
            "training_gradient": "stop gradient through H_mem; retain writer state graph",
        }

    def _charge(self, mb, gather, kind):
        counts = gather((mb["real_input_tokens"], mb["padded_input_tokens"]))
        record = {
            "phase": "evaluation", "kind": kind, "native_calls": 1,
            "real_input_tokens": sum(row[0] for row in counts),
            "padded_input_tokens": sum(row[1] for row in counts), "ce_targets": 0,
        }
        if self.charge_encoding is not None:
            self.charge_encoding(record)
        return record

    def charge_base_encoding(self, mb, gather):
        return self._charge(mb, gather, "base")

    def collect_base_encoding(self, session, mb):
        """Retain legacy H12 base inputs while isolating its encoding RNG."""
        from qwen38_native_memory_schedule import preserve_encoding_rng
        from qwen38_native_staged_schedule import collect_evidence_microbatches
        with preserve_encoding_rng(session.device):
            return collect_evidence_microbatches(session, [mb])["microbatches"][0]

    def write_article(self, session, writer, alpha, state, frames, *, base_blocks,
                      gather, batches, execution, begin_batch, finish_batch,
                      condition, distance):
        import torch
        from delta_ngram.qwen38_memory_batching import collate_memory_encoding_frames
        from delta_ngram.qwen38_memory_writer import write_memory_aware_native_block
        from delta_ngram.qwen38_train_batching import crop_evidence_lanes
        from delta_ngram.qwen38_training import writer_evidence_features
        from qwen38_native_memory_schedule import collect_memory_encoding_microbatches

        if len(frames) != len(base_blocks):
            raise ValueError("base blocks and arrived article frames must align")
        if getattr(writer, "difference_baseline", "actual_base") != self.difference_baseline:
            raise ValueError("DEV64 residual difference baseline differs from writer")
        for index, size in batches(len(frames), 1):
            if size != 1:
                raise ValueError("causal article encoding requires one ready block per state")
            started = begin_batch()
            active = index < len(frames)
            frame = frames[index] if active else None
            mb = collate_memory_encoding_frames(
                [frame], [state if active else None], device=session.device,
                row_capacity=session.layout.row_capacity, width=session.layout.row_width)
            cost = self._charge(mb, gather, "memory")
            execution["native_evidence_calls"] += 1
            execution["evidence_input_tokens"] += cost["real_input_tokens"]
            captured = collect_memory_encoding_microbatches(
                session, [mb], encoding="memory")["microbatches"][0]
            block = crop_evidence_lanes(mb, captured["hidden"], captured["physical_rows"])[0]
            matched_block = None
            if self.difference_baseline == "matched_base":
                matched_mb = collate_memory_encoding_frames(
                    [frame], device=session.device,
                    row_capacity=session.layout.row_capacity, width=session.layout.row_width)
                cost = self._charge(matched_mb, gather, "base_matched")
                execution["native_evidence_calls"] += 1
                execution["evidence_input_tokens"] += cost["real_input_tokens"]
                matched_capture = collect_memory_encoding_microbatches(
                    session, [matched_mb], encoding="base")["microbatches"][0]
                matched_block = crop_evidence_lanes(
                    matched_mb, matched_capture["hidden"], matched_capture["physical_rows"])[0]
                del matched_capture, matched_mb
            if active:
                if block is None:
                    raise RuntimeError("memory encoder dropped an arrived DEV64 block")
                hidden = writer_evidence_features(
                    block["H_new"].to(session.device), mode="rmsnorm", eps=1e-6)
                base_hidden = None
                if self.cache_base_evidence:
                    base = base_blocks[index]
                    if base is None or not torch.equal(
                            base["physical_rows_new"], block["physical_rows_new"]):
                        raise RuntimeError("base and memory encoders used different write addresses")
                    base_hidden = writer_evidence_features(
                        base["H_new"].to(session.device), mode="rmsnorm", eps=1e-6)
                write_kwargs = {"hidden_base": base_hidden}
                if self.difference_baseline == "matched_base":
                    if matched_block is None or not torch.equal(
                            matched_block["physical_rows_new"], block["physical_rows_new"]):
                        raise RuntimeError("matched base and memory encoders used different write addresses")
                    write_kwargs["hidden_base_matched"] = writer_evidence_features(
                        matched_block["H_new"].to(session.device), mode="rmsnorm", eps=1e-6)
                state = write_memory_aware_native_block(
                    writer, state, hidden, block["physical_rows_new"].to(session.device),
                    torch.ones(hidden.shape[:2], dtype=torch.bool, device=session.device),
                    alpha, **write_kwargs)
                if not bool(state.values.isfinite().all()):
                    raise RuntimeError("nonfinite memory-aware DEV64 writer state")
            del captured, block, matched_block, mb
            finish_batch("evidence", started, batch_size=1,
                         local_active_frames=int(active), condition=condition, distance=distance)
        return state


def evaluate_memory_dev64(session, writer, alpha, *, encoding_mode,
                          charge_encoding=None, **kwargs):
    """Same result schema, fixed families, scoring masks, costs and bootstrap."""
    provider = MemoryEvidenceWriter(encoding_mode, charge_encoding,
        difference_baseline=getattr(writer, "difference_baseline", "actual_base"))
    return evaluate_dev64(session, writer, alpha, evidence_writer=provider, **kwargs)

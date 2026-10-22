from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/native_qwen38"))
from qwen38_native_b1_dev64_eval import B1EvidenceWriter
from qwen38_native_memory_dev64_eval import MemoryEvidenceWriter
from delta_ngram.alpha_memory import BlockAlpha, LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.qwen38_train_data import _frame
from delta_ngram.qwen38_writer import NativeForwardWriter
from delta_ngram.state import SparseMemoryState


def exercise(provider, *, padded=True):
    torch.manual_seed(7)
    writer = NativeForwardWriter(width=160, input_dim=2560, hidden_dim=5,
        orders=(2, 3), heads_per_order=8, right_window=2, operator="D1",
        target_scale=torch.ones(16))
    alpha = BlockAlpha(width=2560, kind="fixed", initial_alpha=1.)
    session = SimpleNamespace(device=torch.device("cpu"),
        layout=SimpleNamespace(row_capacity=4096, row_width=160))
    state = LazyDecayState.from_batch(BatchMemoryState.from_states(
        [SparseMemoryState.empty(160, "cpu")], 4096))
    frames = [_frame({"document_id": "x", "source": "test", "article_id": "x"},
                     [3, 7, 5, 8, 9], start, stop, query=False)
              for start, stop in ((0, 3), (3, 5))]
    calls, finishes, charges = [], [], []
    provider.charge_encoding = charges.append
    def capture(_session, items, *, encoding):
        mb = items[0]
        calls.append((encoding, mb["reader_state"] if "reader_state" in mb else None,
                      mb["batch"]["input_ids"].clone(), mb["write_mask"].clone()))
        ids = mb["batch"]["input_ids"]
        rows = torch.stack([ids % 61 + head * 128 for head in range(16)], -1)
        hidden = torch.arange(2560).float().reshape(1, 1, 2560).expand(1, 640, 2560).clone()
        if encoding == "memory":
            hidden += float(mb["reader_state"].values.sum())
        return {"microbatches": [{"hidden": hidden, "physical_rows": rows}]}
    execution = {"native_evidence_calls": 0, "evidence_input_tokens": 0}
    def batches(count, limit):
        assert limit == 1
        return [(i, 1) for i in range(count + int(padded))]
    with patch("qwen38_native_memory_schedule.collect_memory_encoding_microbatches", capture), torch.no_grad():
        state = provider.write_article(session, writer, alpha, state, frames,
            base_blocks=[None] * len(frames), gather=lambda row: [row],
            batches=batches, execution=execution, begin_batch=lambda: 0.,
            finish_batch=lambda *args, **kwargs: finishes.append(kwargs),
            condition="self", distance=0)
    return state, calls, finishes, charges, execution


def test_memory_provider_matches_existing_fourth_cell_exactly():
    old = exercise(MemoryEvidenceWriter("full"))
    new = exercise(B1EvidenceWriter("memory"))
    for attribute in ("values", "counts", "rows", "block_clock", "cumulative_log_alpha",
                      "row_last_written_block", "row_log_alpha_at_write"):
        left, right = getattr(old[0], attribute), getattr(new[0], attribute)
        if torch.is_tensor(left):
            assert torch.equal(left, right), attribute
        else:
            assert left == right, attribute
    assert old[2:] == new[2:]
    assert len(new[1]) == 3
    assert all(row[0] == "memory" and not row[1].values.requires_grad for row in new[1])
    for a, b in zip(old[1], new[1]):
        assert torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])
        for attribute in ("rows", "values", "counts"):
            assert torch.equal(getattr(a[1], attribute), getattr(b[1], attribute))


def test_base_b1_has_no_memory_or_cache_and_still_pads_inactive_rank():
    state, calls, finishes, charges, execution = exercise(B1EvidenceWriter("base"))
    assert all(row[0] == "base" and row[1] is None for row in calls)
    assert all(row[2].shape == (1, 640) for row in calls)
    assert [row["local_active_frames"] for row in finishes] == [1, 1, 0]
    assert all(row["batch_size"] == 1 for row in finishes)
    assert execution["native_evidence_calls"] == 3
    assert all(row["ce_targets"] == 0 for row in charges)
    assert bool(state.values.isfinite().all())


def test_reject_unknown_mode():
    with pytest.raises(ValueError, match="base or memory"):
        B1EvidenceWriter("legacy_h12")


def test_contract_keeps_queries_b4_and_disables_legacy_cache():
    for mode in ("base", "memory"):
        provider = B1EvidenceWriter(mode)
        metadata = provider.metadata()
        assert not provider.cache_base_evidence
        assert metadata["actual_evidence_batch_limit"] == 1
        assert metadata["query_batch_limit"] == 4
        assert metadata["base_cache_batch_limit"] is None

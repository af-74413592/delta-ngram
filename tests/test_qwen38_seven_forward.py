import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.state import SparseMemoryState
from delta_ngram.qwen38_seven_forward import (
    state_payload, restore_state, state_fingerprint, immutable_guard, assert_unchanged,
    evidence_batch, validate_encoded_record, canonical_hash,
)


def state_fixture():
    rows = torch.tensor([1, 7, 91])
    memory = BatchMemoryState(SparseMemoryState(rows, torch.randn(3, 160), torch.tensor([3., 1., 2.])), 1, 100)
    return LazyDecayState(memory, torch.tensor([5]), torch.zeros(1, dtype=torch.float64),
                          torch.tensor([5, 2, 4]), torch.zeros(3, dtype=torch.float64))


def test_resume_retains_all_clocks_counts_and_fp32_values(tmp_path):
    state = state_fixture()
    path = tmp_path / "state.pt"
    torch.save(state_payload(state), path)
    restored = restore_state(torch.load(path, weights_only=True))
    assert state_fingerprint(restored) == state_fingerprint(state)
    assert restored.values.data_ptr() != state.values.data_ptr()
    assert restored.block_clock.tolist() == [5]
    assert restored.row_last_written_block.tolist() == [5, 2, 4]
    assert restored.counts.tolist() == [3., 1., 2.]


def test_resume_rejects_non_alpha1_and_bf16():
    payload = state_payload(state_fixture())
    payload["values"] = payload["values"].bfloat16()
    with pytest.raises(ValueError, match="FP32"):
        restore_state(payload)
    payload = state_payload(state_fixture())
    payload["cumulative_log_alpha"][0] = -.1
    with pytest.raises(ValueError, match="alpha"):
        restore_state(payload)


def test_lookup_is_read_only_and_guard_detects_count_mutation():
    state = state_fixture()
    guard = immutable_guard(state)
    values, counts = state.lookup(torch.tensor([[1, 7, 50]]))
    assert counts.tolist() == [[3., 1., 0.]]
    assert_unchanged(guard)
    state.counts.add_(1)
    with pytest.raises(ValueError, match="read-only"):
        assert_unchanged(guard)


def test_evidence_preserves_independent_lanes_and_masks_right_padding():
    batch = evidence_batch([{"input_ids": [4, 5, 6]}, {"input_ids": [7]}], pad_token_id=248044)
    assert batch["batch"]["input_ids"].tolist() == [[4, 5, 6, 248044], [7, 248044, 248044, 248044]]
    assert batch["valid_mask"].tolist() == [[True, True, True, False], [True, False, False, False]]
    assert batch["batch"]["position_ids"].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert batch["owner_ids"].tolist() == [0, 1]
    assert batch["real_tokens"] == 4
    assert batch["padded_tokens"] == 8
    assert batch["batch"]["labels"] is None
    with pytest.raises(ValueError, match="128"):
        evidence_batch([{"input_ids": list(range(129))}], pad_token_id=0)


def test_encoded_answer_boundary_and_support_token_hash():
    ids = [10, 11, 12, 13, 248046]
    record = {"input_ids": ids, "input_token_ids_sha256": canonical_hash(ids),
              "prompt_length": 2, "target_ids": [12, 13], "target_token_count": 2}
    validate_encoded_record(record, evidence=False)
    with pytest.raises(ValueError, match="boundary"):
        validate_encoded_record({**record, "prompt_length": 1}, evidence=False)
    with pytest.raises(ValueError, match="hash"):
        validate_encoded_record({**record, "input_ids": [11, *ids[1:]]}, evidence=False)
    support = {"input_ids": ids, "input_token_ids_sha256": canonical_hash(ids), "task": "sme", "write_events": 1}
    validate_encoded_record(support, evidence=True)
    with pytest.raises(ValueError, match="one independent"):
        validate_encoded_record({**support, "write_events": 40}, evidence=True)


def test_a800_cli_uses_four_pipeline_stages_twelve_layers_and_explicit_template():
    scripts = Path(__file__).resolve().parents[1] / "scripts" / "native_qwen38"
    sys.path.insert(0, str(scripts))
    from qwen38_native_a800_session import a800_arguments
    from types import SimpleNamespace
    args = a800_arguments(SimpleNamespace(model_path=Path("model"), output=Path("run"), tokens=8192), Path("empty"))
    pairs = dict(zip(args[::2], args[1::2]))
    assert pairs["--tensor_model_parallel_size"] == "2"
    assert pairs["--pipeline_model_parallel_size"] == "4"
    assert pairs["--expert_model_parallel_size"] == "2"
    assert pairs["--decoder_first_pipeline_num_layers"] == "12"
    assert pairs["--template"] == "qwen3_8"

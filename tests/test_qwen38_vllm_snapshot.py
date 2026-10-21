"""CPU boundary tests for the fixed-Delta vLLM snapshot, without vLLM imports."""
from dataclasses import FrozenInstanceError, asdict
import hashlib
import math

import pytest
import torch

from delta_ngram.alpha_memory import LazyDecayState
from delta_ngram.batching import BatchMemoryState
from delta_ngram.qwen38_native import Qwen38PLELayout
from delta_ngram.qwen38_vllm_snapshot import load_vllm_snapshot, save_vllm_snapshot
from delta_ngram.state import SparseMemoryState


@pytest.fixture
def layout():
    return Qwen38PLELayout(tuple(range(0, 64, 4)), (4,) * 16)


@pytest.fixture
def identity():
    return {"base_model": {"config_sha256": "config", "index_sha256": "index"},
            "tokenizer": {"tokenizer.json": "tokenizer-hash"},
            "addressing": "native-order-major-head-regions",
            "writer": "synthetic-test-fixture", "operator": "D1",
            "trajectory": {"owner": 0, "evidence_end": 3}}


def _state(layout, rows=(1, 6, 64), values=None):
    rows = torch.tensor(rows, dtype=torch.int64)
    if values is None:
        values = torch.arange(rows.numel() * 160, dtype=torch.float32).reshape(-1, 160) / 1000
    return BatchMemoryState(SparseMemoryState(rows, values, torch.ones(rows.shape)),
                            1, layout.row_capacity)


def _save_load(tmp_path, state, layout, identity):
    path = tmp_path / "delta.pt"
    receipt = save_vllm_snapshot(path, state=state, layout=layout, identity=identity)
    snapshot = load_vllm_snapshot(path, expected_sha256=receipt["sha256"],
                                  expected_identity=identity, expected_layout=layout)
    return path, receipt, snapshot


def _ids(layout):
    result = torch.tensor([layout.head_offsets, layout.head_offsets])
    result[1, 1] += 1
    result[1, -1] += 3
    return result


def test_native_zero_misses_and_head_major_flattening(tmp_path, layout, identity):
    state = _state(layout)
    path, receipt, snapshot = _save_load(tmp_path, state, layout, identity)
    ids = _ids(layout)
    expected = torch.zeros((2, 16, 160), dtype=torch.float32)
    expected[:, 0] = state.values[0]
    expected[1, 1] = state.values[1]
    expected[1, -1] = state.values[2]
    actual = snapshot.lookup(ids)
    torch.testing.assert_close(actual, expected.reshape(2, 2560), rtol=0, atol=0)
    assert actual.dtype == torch.float32
    assert not actual.requires_grad
    assert snapshot.row_count == 3
    assert snapshot.device == torch.device("cpu")
    assert snapshot.layout == layout
    assert snapshot.identity == identity
    assert snapshot.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt["size_bytes"] == path.stat().st_size
    assert receipt["snapshot_sha256"] == receipt["sha256"]
    assert receipt["expected_snapshot_identity"] == identity
    assert receipt["schema_version"] == 1
    assert snapshot.to("cpu") is snapshot
    # int32 native IDs and noncontiguous views preserve the same semantics.
    torch.testing.assert_close(snapshot.lookup(ids.int()), actual, rtol=0, atol=0)
    padded = torch.zeros(2, 32, dtype=torch.long)
    padded[:, ::2] = ids
    torch.testing.assert_close(snapshot.lookup(padded[:, ::2]), actual, rtol=0, atol=0)
    assert snapshot.lookup(ids[:0]).shape == (0, 2560)


def test_lazy_export_materializes_exact_effective_rows_without_pruning(tmp_path, layout, identity):
    memory = _state(layout, rows=(1, 6, 64),
                    values=torch.stack((torch.full((160,), 2.0), torch.zeros(160),
                                        torch.full((160,), 1e-12))))
    lazy = LazyDecayState(memory, torch.tensor([7]),
                          torch.tensor([7 * math.log(0.99)], dtype=torch.float64),
                          torch.tensor([2, 7, 0]),
                          torch.tensor([2 * math.log(0.99), 7 * math.log(0.99), 0.0],
                                       dtype=torch.float64))
    before_values = lazy.values.clone()
    before_clocks = lazy.cumulative_log_alpha.clone()
    path, receipt, snapshot = _save_load(tmp_path, lazy, layout, identity)
    ids = _ids(layout)
    expected, _ = lazy.lookup((ids + 1)[None])
    torch.testing.assert_close(snapshot.lookup(ids), expected.reshape(2, 2560), rtol=0, atol=0)
    assert snapshot.row_count == 3  # includes stored zero and very small rows
    assert receipt["source_kind"] == "lazy_decay"
    assert receipt["decay_materialized"] is True
    payload = torch.load(path, weights_only=True)
    assert payload["values"][0, 0] != before_values[0, 0]
    assert payload["values"][1].count_nonzero() == 0
    assert payload["values"][2, 0] > 0
    torch.testing.assert_close(lazy.values, before_values, rtol=0, atol=0)
    torch.testing.assert_close(lazy.cumulative_log_alpha, before_clocks, rtol=0, atol=0)


@pytest.mark.parametrize("lazy", [False, True])
def test_empty_delta_is_fp32_and_still_validates_ids(tmp_path, layout, identity, lazy):
    state = layout.empty_state(1)
    if lazy:
        state = LazyDecayState.from_batch(state)
    _, _, snapshot = _save_load(tmp_path, state, layout, identity)
    result = snapshot.lookup(_ids(layout))
    assert result.dtype == torch.float32
    assert result.shape == (2, 2560)
    assert result.count_nonzero() == 0
    assert snapshot.lookup(torch.empty(0, 16, dtype=torch.long)).shape == (0, 2560)
    with pytest.raises(ValueError, match="head region"):
        snapshot.lookup(torch.zeros(1, 16, dtype=torch.long))


def test_independent_snapshots_have_no_owner_or_storage_leak(tmp_path, layout, identity):
    state = _state(layout, rows=(1,), values=torch.full((1, 160), 1.001))
    _, _, first = _save_load(tmp_path, state, layout, identity)
    other_identity = {**identity, "trajectory": {"owner": 0, "evidence_end": 9}}
    state.values.fill_(9.0)
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    _, _, second = _save_load(second_dir, state, layout, other_identity)
    ids = _ids(layout)
    assert first.lookup(ids)[0, 0] == torch.tensor(1.001)
    assert second.lookup(ids)[0, 0] == 9
    first.lookup(ids).fill_(-1)
    metadata = first.identity
    metadata["trajectory"]["evidence_end"] = -1
    assert first.lookup(ids)[0, 0] == torch.tensor(1.001)
    assert first.identity == identity
    with pytest.raises(FrozenInstanceError):
        first.owner_id = 1
    with pytest.raises(ValueError, match="owner_id"):
        first.lookup(ids, owner_id=1)


@pytest.mark.parametrize("owner", [1, -1, True, "0", None])
def test_wrong_owner_rejected_at_every_boundary(tmp_path, layout, identity, owner):
    state = _state(layout)
    path, receipt, snapshot = _save_load(tmp_path, state, layout, identity)
    with pytest.raises(ValueError, match="owner_id"):
        save_vllm_snapshot(tmp_path / "wrong.pt", state=state, layout=layout,
                           identity=identity, owner_id=owner)
    with pytest.raises(ValueError, match="owner_id"):
        load_vllm_snapshot(path, expected_sha256=receipt["sha256"],
                           expected_identity=identity, expected_layout=layout,
                           expected_owner_id=owner)
    with pytest.raises(ValueError, match="owner_id"):
        snapshot.lookup(_ids(layout), owner_id=owner)


def test_multi_owner_export_rejected_then_unpacked_owner_isolated(tmp_path, layout, identity):
    first = _state(layout, rows=(1,), values=torch.full((1, 160), 2.0)).memory
    second = _state(layout, rows=(1,), values=torch.full((1, 160), 7.0)).memory
    batch = BatchMemoryState.from_states((first, second), layout.row_capacity)
    with pytest.raises(ValueError, match="exactly one owner"):
        save_vllm_snapshot(tmp_path / "wrong.pt", state=batch, layout=layout, identity=identity)
    isolated = BatchMemoryState.from_states((batch.unpack()[1],), layout.row_capacity)
    _, _, snapshot = _save_load(tmp_path, isolated, layout, identity)
    assert snapshot.lookup(_ids(layout))[0, 0] == 7


def test_inference_mode_load_and_private_mutation_guards(tmp_path, layout, identity):
    path, receipt, _ = _save_load(tmp_path, _state(layout), layout, identity)
    with torch.inference_mode():
        snapshot = load_vllm_snapshot(path, expected_sha256=receipt["sha256"],
                                      expected_identity=identity, expected_layout=layout)
        assert snapshot.lookup(_ids(layout)).shape == (2, 2560)
    snapshot._values.add_(1)
    with pytest.raises(RuntimeError, match="mutated"):
        snapshot.lookup(_ids(layout))
    with pytest.raises(RuntimeError, match="mutated"):
        snapshot.to("cpu")


@pytest.mark.parametrize("change", [
    lambda ids: ids.float(),
    lambda ids: ids[:, :15],
    lambda ids: ids[None],
    lambda ids: ids.fill_(-1),
    lambda ids: ids.add_(4),
    lambda ids: ids.flip(1),
    lambda ids: ids.to("meta"),
])
def test_invalid_lookup_rejected(tmp_path, layout, identity, change):
    _, _, snapshot = _save_load(tmp_path, _state(layout), layout, identity)
    with pytest.raises(ValueError):
        snapshot.lookup(change(_ids(layout)))


def test_hash_identity_and_layout_are_pinned(tmp_path, layout, identity):
    path, receipt, _ = _save_load(tmp_path, _state(layout), layout, identity)
    kwargs = dict(expected_sha256=receipt["sha256"], expected_identity=identity,
                  expected_layout=layout)
    wrong_identity = {**identity, "writer": "different-writer"}
    with pytest.raises(ValueError, match="identity or layout"):
        load_vllm_snapshot(path, **{**kwargs, "expected_identity": wrong_identity})
    wrong_layout = Qwen38PLELayout(tuple(range(0, 80, 5)), (5,) * 16)
    with pytest.raises(ValueError, match="identity or layout"):
        load_vllm_snapshot(path, **{**kwargs, "expected_layout": wrong_layout})
    for digest in ("", "abc", "g" * 64, "0" * 64):
        with pytest.raises(ValueError, match="SHA256"):
            load_vllm_snapshot(path, **{**kwargs, "expected_sha256": digest})
    # asdict/JSON layout and a case-insensitive hex digest are accepted.
    loaded = load_vllm_snapshot(path, **{**kwargs, "expected_layout": asdict(layout),
                                        "expected_sha256": receipt["sha256"].upper()})
    assert loaded.layout == layout
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        load_vllm_snapshot(path, **kwargs)


def _corrupt_payload(payload, kind):
    if kind == "format":
        payload["format"] = "unknown"
    elif kind == "schema":
        payload["schema_version"] = True
    elif kind == "extra":
        payload["counts"] = torch.ones(3)
    elif kind == "missing":
        del payload["decay_materialized"]
    elif kind == "owner":
        payload["owner_id"] = True
    elif kind == "batch":
        payload["batch_size"] = 2
    elif kind == "materialized":
        payload["decay_materialized"] = False
    elif kind == "row_format":
        payload["row_format"] = "native_zero_based"
    elif kind == "source":
        payload["source_kind"] = "unknown"
    elif kind == "row_dtype":
        payload["rows"] = payload["rows"].int()
    elif kind == "value_dtype":
        payload["values"] = payload["values"].bfloat16()
    elif kind == "shape":
        payload["values"] = payload["values"][:, :159]
    elif kind == "nan":
        payload["values"][0, 0] = float("nan")
    elif kind == "inf":
        payload["values"][0, 0] = float("inf")
    elif kind == "zero":
        payload["rows"][0] = 0
    elif kind == "range":
        payload["rows"][-1] = 65
    elif kind == "duplicate":
        payload["rows"][1] = payload["rows"][0]
    elif kind == "order":
        payload["rows"] = payload["rows"].flip(0)
    elif kind == "layout":
        payload["layout"]["input_dim"] = True
    elif kind == "identity":
        payload["identity"]["operator"] = "D2"
    elif kind == "identity_type":
        # Python mapping equality conflates False and 0; the pinned JSON must not.
        payload["identity"]["trajectory"]["owner"] = False
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize("kind", [
    "format", "schema", "extra", "missing", "owner", "batch", "materialized",
    "row_format", "source", "row_dtype", "value_dtype", "shape", "nan", "inf",
    "zero", "range", "duplicate", "order", "layout", "identity", "identity_type",
])
def test_malformed_payload_rejected_even_with_matching_file_hash(tmp_path, layout, identity, kind):
    path, _, _ = _save_load(tmp_path, _state(layout), layout, identity)
    payload = torch.load(path, weights_only=True)
    _corrupt_payload(payload, kind)
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_vllm_snapshot(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           expected_identity=identity, expected_layout=layout)


def test_export_rejects_bad_source_and_unsupported_device(tmp_path, layout, identity):
    state = _state(layout)
    kwargs = dict(state=state, layout=layout, identity=identity)
    state.values[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        save_vllm_snapshot(tmp_path / "bad.pt", **kwargs)
    state.values[0, 0] = 0
    state.counts[0] = 0.5
    with pytest.raises(ValueError, match="observation counts"):
        save_vllm_snapshot(tmp_path / "bad.pt", **kwargs)
    with pytest.raises(ValueError, match="capacity or width"):
        save_vllm_snapshot(tmp_path / "bad.pt", **{**kwargs, "layout":
                           Qwen38PLELayout(tuple(range(0, 80, 5)), (5,) * 16)})
    state.counts[0] = 1
    path, receipt, snapshot = _save_load(tmp_path, state, layout, identity)
    with pytest.raises(ValueError, match="CPU or CUDA"):
        snapshot.to("meta")
    with pytest.raises(ValueError, match="CPU or CUDA"):
        load_vllm_snapshot(path, expected_sha256=receipt["sha256"],
                           expected_identity=identity, expected_layout=layout, device="meta")

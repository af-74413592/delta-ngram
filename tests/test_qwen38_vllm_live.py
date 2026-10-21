"""CPU contracts for request-frozen, replicated live Delta state."""
import copy

import pytest
import torch

from delta_ngram.alpha_memory import BlockAlpha
from delta_ngram.qwen38_native import Qwen38PLELayout, write_native_block
from delta_ngram.qwen38_training import writer_evidence_features
from delta_ngram.qwen38_writer import NativeForwardWriter
from delta_ngram import qwen38_vllm_live as live


@pytest.fixture(scope="module")
def evidence():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(817)
        layout = Qwen38PLELayout(tuple(4 * i for i in range(16)), (4,) * 16)
        writer = NativeForwardWriter(width=160, input_dim=2560, hidden_dim=8,
            orders=(2, 3), heads_per_order=8, right_window=4, operator="D1",
            target_scale=torch.ones(16)).eval().requires_grad_(False)
        policy = BlockAlpha(width=2560, kind="fixed", initial_alpha=1.0)
        hidden = torch.randn(1, 3, 2560)
    ids = torch.tensor(layout.head_offsets).expand(3, 16).clone()
    return layout, writer, policy, hidden, ids


def new_store(evidence):
    store = live.LiveMemoryStore(evidence[0], "cpu")
    store.initialized = True
    return store


def begin_support(store, arm="delta", request="support-0"):
    store.begin(arm, "support", request, store.epochs[arm], [12, 13, 14])


def update_for(store, evidence, arm="delta", request="support-0", offset=0):
    layout, writer, policy, hidden, ids = evidence
    begin_support(store, arm, request)
    store.finish()
    physical = layout.physical_rows((ids + offset)[None], torch.ones((1, 3), dtype=torch.bool))
    before = store.states[arm]
    with torch.no_grad():
        reference = write_native_block(writer, before, writer_evidence_features(hidden, mode="rmsnorm", eps=1e-6),
                                       physical, torch.ones((1, 3), dtype=torch.bool), policy)
    update = store.produce_update(writer, policy, hidden, physical)
    return update, reference


def assert_state_equal(left, right):
    for field in live.FIELDS:
        assert torch.equal(getattr(left, field), getattr(right, field)), field
    assert left.row_capacity == right.row_capacity
    assert live.state_hash(left) == live.state_hash(right)


def test_two_ranks_commit_identical_complete_state_and_preserve_untouched_rows(evidence):
    stores = [new_store(evidence), new_store(evidence)]
    previous = None
    for index, offset in enumerate((0, 1, 0)):
        request = f"support-{index}"
        update, reference = update_for(stores[0], evidence, request=request, offset=offset)
        begin_support(stores[1], request=request)
        stores[1].finish()
        receipts = [store.commit(copy.deepcopy(update)) for store in stores]
        assert receipts[0] == receipts[1]
        for store in stores:
            assert_state_equal(store.states["delta"], reference)
            assert store.epochs["delta"] == index + 1
            assert store.states["ordinary"].rows.numel() == 0
            assert store.epochs["ordinary"] == 0
        if previous is not None:
            untouched = previous.rows[~torch.isin(previous.rows, update["rows"])]
            old_slots = torch.searchsorted(previous.rows, untouched)
            new_slots = torch.searchsorted(reference.rows, untouched)
            for field in ("values", "counts", "row_last_written_block", "row_log_alpha_at_write"):
                assert torch.equal(getattr(previous, field)[old_slots], getattr(reference, field)[new_slots])
        previous = reference
    assert not torch.cuda.is_initialized()


def test_ordinary_support_reads_t0_while_queries_and_delta_support_read_own_state(evidence):
    store = new_store(evidence)
    for arm in live.ARMS:
        update, _ = update_for(store, evidence, arm=arm, request=f"seed-{arm}")
        store.commit(update)
    ids = evidence[-1]
    for arm, kind in (("ordinary", "support"), ("ordinary", "query"), ("delta", "support"), ("delta", "query")):
        before = {key: live.state_hash(value) for key, value in store.states.items()}
        store.begin(arm, kind, f"read-{arm}-{kind}", store.epochs[arm], [12, 13, 14])
        actual = store.lookup(ids)
        if (arm, kind) == ("ordinary", "support"):
            assert torch.count_nonzero(actual) == 0
        else:
            rows = evidence[0].physical_rows(ids[None], torch.ones((1, 3), dtype=torch.bool))
            expected = store.states[arm].lookup(rows)[0][0].flatten(-2)
            assert torch.equal(actual, expected) and actual.abs().sum() > 0
        receipt = store.finish()
        assert receipt["input_sha256"] == live.canonical_hash([12, 13, 14])
        assert before == {key: live.state_hash(value) for key, value in store.states.items()}
        if kind == "support":
            assert store.pending is not None
            # Each read case is independent; no test-only production abort API is assumed.
            store.pending = None
        else:
            assert store.pending is None


@pytest.mark.parametrize("field", live.FIELDS)
def test_request_freeze_detects_mutation_in_every_state_tensor(evidence, field):
    store = new_store(evidence)
    update, _ = update_for(store, evidence)
    store.commit(update)
    store.begin("delta", "query", "query", 1, [12, 13, 14])
    getattr(store.states["delta"], field).add_(0)
    with pytest.raises(ValueError, match="mutated live state"):
        store.lookup(evidence[-1])
    with pytest.raises(ValueError, match="mutated live state"):
        store.finish()


def test_no_overlap_stale_epoch_or_writes_from_queries(evidence):
    store = new_store(evidence)
    with pytest.raises(ValueError, match="arm/kind/epoch"):
        store.begin("delta", "query", "future", 1, [12])
    store.begin("delta", "query", "read", 0, [12])
    with pytest.raises(ValueError, match="not idle"):
        begin_support(store)
    store.finish()
    with pytest.raises(ValueError, match="pending support"):
        store.commit({})
    with pytest.raises(ValueError, match="completed support"):
        store.produce_update(evidence[1], evidence[2], evidence[3], torch.ones((1, 3, 16), dtype=torch.long))
    update, _ = update_for(store, evidence)
    with pytest.raises(ValueError, match="not idle"):
        begin_support(store, request="overlap")
    store.commit(update)
    with pytest.raises(ValueError, match="pending support"):
        store.commit(update)


@pytest.mark.parametrize("field,value", [
    ("epoch", 0), ("epoch", 2), ("arm", "ordinary"), ("request_id", "other"),
    ("previous_identity", "0" * 64), ("format", "other")])
def test_stale_or_mismatched_patch_is_atomic_and_can_be_retried(evidence, field, value):
    store = new_store(evidence)
    update, reference = update_for(store, evidence)
    before = live.state_hash(store.states["delta"])
    bad = copy.deepcopy(update)
    bad[field] = value
    with pytest.raises(ValueError, match="stale or mismatched"):
        store.commit(bad)
    assert live.state_hash(store.states["delta"]) == before and store.epochs["delta"] == 0
    assert store.pending is not None
    store.commit(update)
    assert_state_equal(store.states["delta"], reference)


@pytest.mark.parametrize("corruption", ["clock_skip", "float_values", "nan_values", "inf_counts", "future_row",
                                        "wrong_alpha", "duplicate_row", "zero_row", "outside_capacity"])
def test_invalid_tensor_patch_does_not_mutate_state(evidence, corruption):
    store = new_store(evidence)
    update, _ = update_for(store, evidence)
    before = live.state_hash(store.states["delta"])
    bad = copy.deepcopy(update)
    if corruption == "clock_skip": bad["block_clock"] += 1
    elif corruption == "float_values": bad["values"] = bad["values"].double()
    elif corruption == "nan_values": bad["values"][0, 0] = float("nan")
    elif corruption == "inf_counts": bad["counts"][0] = float("inf")
    elif corruption == "future_row": bad["row_last_written_block"][0] += 1
    elif corruption == "wrong_alpha": bad["cumulative_log_alpha"][0] = -.01
    elif corruption == "duplicate_row": bad["rows"][1] = bad["rows"][0]
    elif corruption == "zero_row": bad["rows"][0] = 0
    elif corruption == "outside_capacity": bad["rows"][-1] = store.layout.row_capacity + 1
    with pytest.raises(ValueError):
        store.commit(bad)
    assert live.state_hash(store.states["delta"]) == before and store.pending is not None


@pytest.mark.parametrize("corruption", ["missing_increment", "negative_count", "stale_row_clock"])
def test_patch_requires_one_count_increment_and_current_write_clock(evidence, corruption):
    store = new_store(evidence)
    update, _ = update_for(store, evidence)
    bad = copy.deepcopy(update)
    if corruption == "missing_increment": bad["counts"][0] = 0
    elif corruption == "negative_count": bad["counts"][0] = -1
    else: bad["row_last_written_block"][0] = 0
    with pytest.raises(ValueError):
        store.commit(bad)


def test_restore_keeps_all_seven_fields_and_owns_tensors(evidence):
    store = new_store(evidence)
    for index in range(2):
        update, _ = update_for(store, evidence, request=f"seed-{index}", offset=index)
        store.commit(update)
    source = store.states["delta"]
    payload = live.state_payload(source)
    restored = live.restore_state(payload, "cpu")
    assert_state_equal(source, restored)
    for field in live.FIELDS:
        assert getattr(source, field).data_ptr() != payload[field].data_ptr()
        assert getattr(restored, field).data_ptr() != payload[field].data_ptr()
    payload["values"].add_(1)
    assert_state_equal(source, restored)


@pytest.mark.parametrize("corruption", ["missing_field", "format", "clock_dtype", "clock_negative", "nan_values",
                                        "wrong_width", "count_dtype", "nonzero_alpha", "missing_clock_shape"])
def test_restore_rejects_incomplete_or_invalid_state(evidence, corruption):
    store = new_store(evidence)
    update, _ = update_for(store, evidence)
    store.commit(update)
    payload = live.state_payload(store.states["delta"])
    if corruption == "missing_field": del payload["row_last_written_block"]
    elif corruption == "format": payload["format"] = "unknown"
    elif corruption == "clock_dtype": payload["block_clock"] = payload["block_clock"].float()
    elif corruption == "clock_negative": payload["block_clock"][0] = -1
    elif corruption == "nan_values": payload["values"][0, 0] = float("nan")
    elif corruption == "wrong_width": payload["values"] = payload["values"][:, :-1]
    elif corruption == "count_dtype": payload["counts"] = payload["counts"].double()
    elif corruption == "nonzero_alpha": payload["cumulative_log_alpha"][0] = -.01
    elif corruption == "missing_clock_shape": payload["block_clock"] = payload["block_clock"][0]
    with pytest.raises((ValueError, KeyError)):
        live.restore_state(payload, "cpu")

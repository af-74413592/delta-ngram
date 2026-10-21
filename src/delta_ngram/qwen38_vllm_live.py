"""Request-bound FP32 PLE state for the separately registered live vLLM fork.

The two writer trajectories own complete lazy states. A request borrows one
unchanging state; only a completed support may commit one D1 update. This
module never modifies the frozen model or the existing fixed-snapshot API.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

import torch

from .alpha_memory import LazyDecayState
from .batching import BatchMemoryState
from .state import SparseMemoryState
from .qwen38_native import Qwen38PLELayout, write_native_block
from .qwen38_training import writer_evidence_features
from .qwen38_vllm_ple import _validate_ple_layout, split_cached_body


PROTOCOL = "qwen38_vllm_live_two_chain_fp32_v1"
ENV = "DELTA_NGRAM_VLLM_LIVE_BINDING"
ENV_SHA = ENV + "_SHA256"
FIELDS = ("rows", "values", "counts", "block_clock", "cumulative_log_alpha",
          "row_last_written_block", "row_log_alpha_at_write")
ARMS = ("ordinary", "delta")
_binding = None
_environment = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tensor_hash(value):
    value = value.detach().contiguous().cpu()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def state_hash(state):
    return canonical_hash({key: {"shape": list(getattr(state, key).shape),
        "dtype": str(getattr(state, key).dtype), "sha256": tensor_hash(getattr(state, key))} for key in FIELDS})


def state_payload(state):
    return {"format": PROTOCOL + "/state", "row_capacity": state.row_capacity,
            **{key: getattr(state, key).detach().cpu().clone() for key in FIELDS}}


def restore_state(payload, device):
    require(payload.get("format") == PROTOCOL + "/state", "unknown live state format")
    values = {key: payload[key].detach().to(device=device, copy=True) for key in FIELDS}
    state = LazyDecayState(BatchMemoryState(SparseMemoryState(*(values[k] for k in FIELDS[:3])),
        1, payload["row_capacity"]), *(values[k] for k in FIELDS[3:]))
    validate_state(state)
    return state


def validate_state(state):
    require(isinstance(state, LazyDecayState) and state.batch_size == 1 and state.precision is None,
            "live writer requires a one-owner lazy state")
    require(state.values.dtype == state.counts.dtype == torch.float32 and state.width == 160,
            "live state values/counts must remain FP32 width160")
    require(not bool(state.cumulative_log_alpha.ne(0).any())
            and not bool(state.row_log_alpha_at_write.ne(0).any()), "live protocol fixes alpha=1")
    require(bool(state.values.isfinite().all()) and bool(state.counts.isfinite().all()), "nonfinite state")
    require(bool(state.counts.ge(0).all()) and bool(state.block_clock.ge(0).all())
            and bool(state.row_last_written_block.ge(0).all())
            and bool(state.row_last_written_block.le(state.block_clock[0]).all()), "invalid state clocks/counts")


def versions(state):
    return tuple((getattr(state, key), getattr(state, key)._version) for key in FIELDS)


def unchanged(guard):
    require(all(value._version == version for value, version in guard), "request mutated live state")


def make_update(before, after, touched, *, arm, epoch, request_id, previous_identity):
    validate_state(after)
    require(torch.equal(after.block_clock, before.block_clock + 1), "one support must advance one clock")
    touched = torch.unique(touched[touched.ne(0)], sorted=True)
    slots = torch.searchsorted(after.rows, touched)
    require(torch.equal(after.rows[slots], touched), "D1 omitted a touched address")
    return {"format": PROTOCOL + "/row-update", "arm": arm, "epoch": epoch + 1,
            "request_id": request_id, "previous_identity": previous_identity,
            "rows": touched.detach().clone(), "values": after.values[slots].detach().clone(),
            "counts": after.counts[slots].detach().clone(),
            "row_last_written_block": after.row_last_written_block[slots].detach().clone(),
            "row_log_alpha_at_write": after.row_log_alpha_at_write[slots].detach().clone(),
            "block_clock": after.block_clock.detach().clone(),
            "cumulative_log_alpha": after.cumulative_log_alpha.detach().clone()}


def update_hash(update):
    return canonical_hash({key: ({"shape": list(value.shape), "dtype": str(value.dtype),
        "sha256": tensor_hash(value)} if isinstance(value, torch.Tensor) else value)
        for key, value in update.items()})


def apply_update(state, update):
    rows = update["rows"]
    require(rows.dtype == torch.int64 and rows.ndim == 1 and rows.numel() > 0
            and bool((rows[1:] > rows[:-1]).all()) and bool((rows > 0).all())
            and bool((rows <= state.row_capacity).all()), "invalid row update addresses")
    n = rows.numel()
    require(update["values"].shape == (n, state.width) and update["values"].dtype == torch.float32
            and update["counts"].shape == (n,) and update["counts"].dtype == torch.float32
            and update["row_last_written_block"].shape == (n,)
            and update["row_last_written_block"].dtype == torch.int64
            and update["row_log_alpha_at_write"].shape == (n,)
            and update["row_log_alpha_at_write"].dtype == torch.float64,
            "row update dtype/shape changed")
    require(update["block_clock"].dtype == torch.int64 and update["block_clock"].shape == (1,)
            and update["cumulative_log_alpha"].dtype == torch.float64
            and update["cumulative_log_alpha"].shape == (1,)
            and torch.equal(update["block_clock"], state.block_clock + 1)
            and not bool(update["cumulative_log_alpha"].ne(0).any())
            and not bool(update["row_log_alpha_at_write"].ne(0).any()), "row update clock/alpha changed")
    _, prior_counts = state.memory.memory.lookup(rows)
    require(torch.equal(update["counts"], prior_counts + 1)
            and bool(update["row_last_written_block"].eq(update["block_clock"][0]).all()),
            "D1 must increment touched counts once and stamp the new block clock")
    require(bool(update["values"].isfinite().all()) and bool(update["counts"].isfinite().all()),
            "nonfinite row update")
    union = torch.unique(torch.cat((state.rows, rows)), sorted=True)
    old_values, old_counts = state.memory.memory.lookup(union)
    old_slots, slots = torch.searchsorted(union, state.rows), torch.searchsorted(union, rows)
    last = torch.zeros_like(union).index_copy(0, old_slots, state.row_last_written_block)
    log = torch.zeros(union.numel(), dtype=torch.float64, device=union.device).index_copy(
        0, old_slots, state.row_log_alpha_at_write)
    memory = BatchMemoryState(SparseMemoryState(union,
        old_values.index_copy(0, slots, update["values"]), old_counts.index_copy(0, slots, update["counts"])),
        1, state.row_capacity)
    return LazyDecayState(memory, update["block_clock"].clone(), update["cumulative_log_alpha"].clone(),
        last.index_copy(0, slots, update["row_last_written_block"]),
        log.index_copy(0, slots, update["row_log_alpha_at_write"]))


class LiveMemoryStore:
    """Pure state machine; the worker enforces scheduler quiescence around it."""

    def __init__(self, layout, device):
        self.layout, self.device = layout, torch.device(device)
        with torch.inference_mode(False), torch.no_grad():
            self.states = {arm: LazyDecayState.from_batch(layout.empty_state(1, device)) for arm in ARMS}
        self.epochs = dict.fromkeys(ARMS, 0)
        self.identities = {arm: canonical_hash({"protocol": PROTOCOL, "arm": arm, "epoch": 0}) for arm in ARMS}
        self.active = self.pending = None
        self.initialized = False

    def begin(self, arm, kind, request_id, expected_epoch, input_ids):
        require(self.initialized and self.active is None and self.pending is None, "live store is not idle")
        require(arm in ARMS and kind in ("support", "query") and expected_epoch == self.epochs[arm],
                "request arm/kind/epoch differs")
        require(isinstance(request_id, str) and bool(request_id) and isinstance(input_ids, list)
                and bool(input_ids) and all(type(x) is int and 0 <= x < 248320 for x in input_ids),
                "invalid request identity or tokens")
        self.active = {"arm": arm, "kind": kind, "request_id": request_id, "epoch": expected_epoch,
            "input_ids": list(input_ids), "input_sha256": canonical_hash(input_ids),
            "state": self.states[arm], "guard": versions(self.states[arm])}

    def lookup(self, native_ids):
        require(native_ids.dtype == torch.long and native_ids.ndim == 2 and native_ids.shape[-1] == 16,
                "native PLE IDs must be [tokens,16]")
        if not self.initialized:
            return torch.zeros((native_ids.shape[0], 2560), dtype=torch.float32, device=native_ids.device)
        require(self.active is not None, "model forward has no registered live request")
        current = self.active; unchanged(current["guard"])
        if current["kind"] == "support" and current["arm"] == "ordinary":
            return torch.zeros((native_ids.shape[0], 2560), dtype=torch.float32, device=native_ids.device)
        valid = torch.ones((1, native_ids.shape[0]), dtype=torch.bool, device=native_ids.device)
        physical = self.layout.physical_rows(native_ids[None], valid)
        value, _ = current["state"].lookup(physical)
        return value[0].flatten(-2).float()

    def finish(self):
        require(self.active is not None, "no active request")
        current = self.active; unchanged(current["guard"])
        require(self.states[current["arm"]] is current["state"]
                and self.epochs[current["arm"]] == current["epoch"], "request state changed")
        self.active = None
        if current["kind"] == "support":
            self.pending = current
        return {key: current[key] for key in ("arm", "kind", "request_id", "epoch", "input_sha256")}

    def produce_update(self, writer, alpha, hidden, physical):
        require(self.active is None and self.pending is not None, "D1 needs one completed support")
        current = self.pending; arm = current["arm"]
        require(hidden.shape == (1, len(current["input_ids"]), 2560)
                and physical.shape == (*hidden.shape[:2], 16), "support H/addresses differ")
        mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        with torch.inference_mode(False), torch.no_grad():
            features = writer_evidence_features(hidden, mode="rmsnorm", eps=1e-6)
            candidate = write_native_block(writer, self.states[arm], features, physical, mask, alpha)
            return make_update(self.states[arm], candidate, physical.flatten(), arm=arm,
                epoch=self.epochs[arm], request_id=current["request_id"], previous_identity=self.identities[arm])

    def commit(self, update):
        require(self.active is None and self.pending is not None, "commit has no pending support")
        current = self.pending; arm = current["arm"]
        require(update.get("format") == PROTOCOL + "/row-update" and update["arm"] == arm
                and update["epoch"] == self.epochs[arm] + 1
                and update["request_id"] == current["request_id"]
                and update["previous_identity"] == self.identities[arm], "stale or mismatched update")
        unchanged(current["guard"])
        digest = update_hash(update)
        with torch.inference_mode(False), torch.no_grad():
            result = apply_update(self.states[arm], update)
        self.states[arm], self.epochs[arm], self.identities[arm] = result, update["epoch"], digest
        self.pending = None
        return {"arm": arm, "epoch": update["epoch"], "identity": digest, "rows": result.rows.numel(),
                "block_clock": result.block_clock.cpu().tolist(), "update_rows": update["rows"].numel()}


def split_enabled():
    global _binding, _environment
    current = (os.environ.get(ENV), os.environ.get(ENV_SHA))
    if _environment is None:
        if current == (None, None):
            _environment = current
            return False
        require(all(current), "live binding path and SHA must be provided together")
        raw = Path(current[0]).read_bytes()
        require(re.fullmatch(r"[0-9a-f]{64}", current[1]) is not None
                and hashlib.sha256(raw).hexdigest() == current[1], "live binding SHA differs")
        data = json.loads(raw)
        require(data.get("protocol") == PROTOCOL and data.get("schema_version") == 1,
                "live binding protocol differs")
        fields = dict(data["layout"])
        for key in ("orders", "head_offsets", "head_vocab_sizes"):
            fields[key] = tuple(fields[key])
        data["_layout"] = Qwen38PLELayout(**fields)
        _binding, _environment = data, current
    require(current == _environment, "live binding environment changed")
    return _binding is not None


@dataclass
class LiveRuntime:
    store: LiveMemoryStore
    fp32_weight: torch.Tensor
    parameters: tuple
    fp32_version: int
    observer: object = None

    def check(self, ple):
        require(split_enabled(), "live binding disabled")
        require(tuple(id(p) for p in ple.parameters()) == tuple(id(p) for p, _ in self.parameters)
                and all(p._version == version and not p.requires_grad and p.grad is None
                        for p, version in self.parameters), "frozen live PLE changed")
        require(self.fp32_weight._version == self.fp32_version, "FP32 projection copy changed")


def runtime(ple):
    require(split_enabled(), "live reader is not enabled")
    current = getattr(ple, "_delta_ngram_live_runtime", None)
    if current is None:
        layout = _binding["_layout"]; _validate_ple_layout(ple, layout)
        weight = ple.kv_proj.weight
        require(weight.dtype == torch.bfloat16 and tuple(weight.shape) == (12800, 2560),
                "live PLE requires original merged BF16 projection")
        require(tuple(ple.kv_proj.output_sizes) == (10240, 2560), "PLE projection order changed")
        with torch.inference_mode(False), torch.no_grad():
            copied = weight.detach().float().clone()
            store = LiveMemoryStore(layout, weight.device)
        current = LiveRuntime(store, copied, tuple((p, p._version) for p in ple.parameters()), copied._version)
        ple._delta_ngram_live_runtime = current
    current.check(ple)
    return current


@torch.no_grad()
def split_ple_forward(ple, hidden_states, input_ids, query_start_loc, ngram_context):
    live = runtime(ple)
    ids = input_ids.reshape(-1)
    require(ids.shape[0] == hidden_states.shape[0], "PLE token count differs")
    base = ple._dequantize_embeddings(ple.ple_embedding(hidden_states, ids, query_start_loc, ngram_context),
                                     hidden_states.dtype)
    native_ids = ple.ple_embedding.compute_ngram_ids(ids, query_start_loc, ngram_context)
    if live.observer is not None:
        live.observer(ple, native_ids, query_start_loc, ngram_context)
    delta = live.store.lookup(native_ids)
    return split_cached_body(ple, hidden_states, base, delta, live.fp32_weight)

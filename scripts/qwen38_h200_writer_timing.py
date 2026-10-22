"""Bounded H200 TP2/PP2 timing using the original frozen native reader.

No A800 reader, host-table replacement, HC override, TF, or generation path is
imported. Ordinary/main100M and memory/full100M have separate chronological
states. Optional zero-filled capacity storage is engineering evidence only.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time

PROTOCOL = "qwen38_h200_two_writer_B1_timing_v1"
FIELDS = ("rows", "values", "counts", "block_clock", "cumulative_log_alpha",
          "row_last_written_block", "row_log_alpha_at_write")
SOURCES = {
    "scripts/native_qwen38/qwen38_native_frozen_session.py": "0373b66255dbd7d6b876ba91090fe5af156b76f2cb24fb3e13c9ba5dd710e802",
    "scripts/native_qwen38/qwen38_native_runtime_preflight.py": "9e47920b432e4c7b7659bd0069b385227c3c3c50e319398407b37430856254c2",
    "scripts/native_qwen38/qwen38_native_dispatch.py": "81eee204e7466817706bcd49710439a1ffa99d916763561a94007c48188d1e7d",
    "scripts/native_qwen38/qwen38_native_router.py": "90086da11a565dab1157a6f8719db7a127f6297d930beed1ef5e05d0d0d5faea",
    "src/delta_ngram/qwen38_adapter.py": "ebef7c6353da1b5cc2e9cfd0c8874242f11af821cc39cc5c49352cf2fefb00a6",
    "src/delta_ngram/qwen38_precision.py": "6a2a1a01dd23649fc15c31b9782f6bc5d7a9c8cd23608f675830595939c69c80",
    "src/delta_ngram/qwen38_native.py": "c98eabd958f49ff1143743b6cfab75c134c790abd4b0f7f34dcbaf8acfd0d33a",
}
WRITERS = {
    "ordinary": {"archive": "523b951487cee0ec726077db0104a8fc929f4629bdb4ca46604295a4b78fbfe3",
                 "tensor": "d83d2ba750913ec5c9e1a3fc43385a7c16fbe683a106934d7f8f21f853fdb460"},
    "delta": {"archive": "51d5977845e9930f48732c73ae2503511d8976b1201f139a2be46bfab9f57308",
              "tensor": "95e9766c2d90805cede77d03ade5af97b850e9cb3bcc3c55ea103d5cf0e0cb9c"},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def payload(state):
    return {"format": PROTOCOL + "/full-state", "row_capacity": state.row_capacity,
            "width": state.width, "batch_size": 1,
            **{key: getattr(state, key).detach().cpu().clone() for key in FIELDS}}


def restore(values, device):
    from delta_ngram.alpha_memory import LazyDecayState
    from delta_ngram.batching import BatchMemoryState
    from delta_ngram.state import SparseMemoryState
    fields = {key: values[key].detach().to(device=device, copy=True) for key in FIELDS}
    return LazyDecayState(BatchMemoryState(SparseMemoryState(*(fields[k] for k in FIELDS[:3])),
                                          1, values["row_capacity"]), *(fields[k] for k in FIELDS[3:]))


def clone_state(state):
    from delta_ngram.alpha_memory import LazyDecayState
    from delta_ngram.batching import BatchMemoryState
    from delta_ngram.state import SparseMemoryState
    copied = {key: getattr(state, key).detach().clone() for key in FIELDS}
    return LazyDecayState(BatchMemoryState(SparseMemoryState(*(copied[k] for k in FIELDS[:3])),
                                          1, state.row_capacity), *(copied[k] for k in FIELDS[3:]))


def fingerprint(state):
    import torch
    result = hashlib.sha256()
    for key in FIELDS:
        value = getattr(state, key).detach().cpu().contiguous()
        result.update(key.encode()); result.update(str(value.dtype).encode()); result.update(str(tuple(value.shape)).encode())
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def expand_capacity(state, union):
    import torch
    require(union.dtype == torch.int64 and union.ndim == 1 and bool((union[1:] > union[:-1]).all())
            and bool((union > 0).all()) and bool((union <= state.row_capacity).all()), "invalid capacity rows")
    old = payload(state)
    union_cpu = union.cpu()
    slots = torch.searchsorted(union_cpu, old["rows"])
    require(bool((slots < union_cpu.numel()).all()) and torch.equal(union_cpu[slots], old["rows"]),
            "capacity universe omits a real written row")
    padded = dict(old, rows=union_cpu.clone())
    for key in ("values", "counts", "row_last_written_block", "row_log_alpha_at_write"):
        padded[key] = torch.zeros((union_cpu.numel(), *old[key].shape[1:]), dtype=old[key].dtype)
        padded[key][slots] = old[key]
        require(torch.equal(padded[key][slots], old[key]), "capacity expansion changed written values")
    return restore(padded, state.values.device)


def evidence_batch(row, pad_token_id, device):
    import torch
    length = len(row["input_ids"])
    require(2 <= length <= 128, "support length outside registered ceiling")
    size = (length + 1) // 2 * 2
    ids = torch.full((1, size), pad_token_id, dtype=torch.long, device=device)
    ids[0, :length] = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
    return {"input_ids": ids, "position_ids": torch.arange(size, device=device)[None],
            "attention_mask": None, "labels": None}, torch.arange(size, device=device)[None] < length


@contextmanager
def preserve_rng(device):
    import numpy as np
    import torch
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    py, np_state = random.getstate(), np.random.get_state()
    tracker = get_cuda_rng_tracker(); tp_state = copy.deepcopy(tracker.get_states())
    try:
        with torch.random.fork_rng(devices=[device.index]):
            yield
    finally:
        random.setstate(py); np.random.set_state(np_state); tracker.set_states(tp_state)


class _HeadCaptured(Exception):
    pass


def encode(session, batch, valid, read_state, memory):
    """Original H200 model/schedule, stopping at its real pre-vocabulary H."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from qwen38_native_runtime_preflight import language_model
    require((ps.get_tensor_model_parallel_world_size(), ps.get_pipeline_model_parallel_world_size(),
             ps.get_expert_model_parallel_world_size()) == (2, 2, 2), "H200 topology changed")
    require(memory or read_state is None, "static T0 encoding cannot receive delta")
    length = batch["input_ids"].shape[1]
    captured, native_capture = [], None
    handle = None
    if ps.is_pipeline_last_stage():
        def before_head(module, args, kwargs):
            value = args[0] if args else kwargs.get("hidden_states", kwargs.get("input_"))
            require(isinstance(value, torch.Tensor) and value.shape == (length // 2, 1, 2560)
                    and value.dtype == torch.bfloat16 and not captured, "native H boundary changed")
            captured.append(value.detach().clone())
            raise _HeadCaptured()
        handle = language_model(session.raw).output_layer.register_forward_pre_hook(before_head, with_kwargs=True)

    def step(iterator, model):
        nonlocal native_capture
        def call():
            try:
                return model(**batch)
            except _HeadCaptured:
                require(ps.is_pipeline_last_stage() and len(captured) == 1, "unexpected H boundary")
                return captured[0].new_zeros((1, length), dtype=torch.float32)
        if session.adapter is None:
            output = call()
        else:
            output, native_capture = session.adapter.run_native(call, input_ids=batch["input_ids"],
                valid_mask=valid, owner_ids=torch.zeros(1, dtype=torch.long, device=session.device),
                mode="query" if memory else "evidence", state=read_state)
        return output, lambda value: (value.new_zeros((), dtype=torch.float32), {"evidence_only": True})

    try:
        with preserve_rng(session.device), torch.no_grad():
            schedule = get_forward_backward_func()
            require(schedule.__name__ == "forward_backward_pipelining_without_interleaving", "pipeline schedule changed")
            schedule(forward_step_func=step, data_iterator=None, model=session.models,
                     num_microbatches=1, seq_length=length, micro_batch_size=1, forward_only=True)
    finally:
        if handle is not None:
            handle.remove()
    hidden = torch.empty((length, 1, 2560), dtype=torch.bfloat16, device=session.device)
    if ps.is_pipeline_last_stage():
        require(len(captured) == 1, "native H not captured")
        pieces = [torch.empty_like(captured[0]) for _ in range(2)]
        dist.all_gather(pieces, captured[0], group=ps.get_tensor_model_parallel_group())
        hidden.copy_(torch.cat(pieces, dim=0))
    dist.broadcast(hidden, src=ps.get_pipeline_model_parallel_last_rank(), group=ps.get_pipeline_model_parallel_group())
    return hidden.transpose(0, 1).contiguous(), native_capture.physical_rows if native_capture is not None else None


def sync_rows(state, touched, device):
    """Only first-stage TP ranks participate; TP0 is the sole D1 writer."""
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from delta_ngram.alpha_memory import LazyDecayState
    from delta_ngram.batching import BatchMemoryState
    from delta_ngram.state import SparseMemoryState
    group, source = ps.get_tensor_model_parallel_group(), ps.get_tensor_model_parallel_src_rank()
    root = ps.get_tensor_model_parallel_rank() == 0
    count = torch.tensor([touched.numel() if root else 0], dtype=torch.long, device=device)
    dist.broadcast(count, src=source, group=group)
    rows = touched if root else torch.empty(int(count), dtype=torch.long, device=device)
    dist.broadcast(rows, src=source, group=group)
    if root:
        slots = torch.searchsorted(state.rows, rows)
        values, counts, last = state.values[slots].contiguous(), state.counts[slots].contiguous(), state.row_last_written_block[slots].contiguous()
    else:
        values = torch.empty((rows.numel(), state.width), dtype=torch.float32, device=device)
        counts = torch.empty(rows.numel(), dtype=torch.float32, device=device)
        last = torch.empty_like(rows)
    clock = state.block_clock.clone()
    for value in (values, counts, last, clock):
        dist.broadcast(value, src=source, group=group)
    if root:
        return state
    union = torch.unique(torch.cat((state.rows, rows)), sorted=True)
    old_values, old_counts = state.memory.memory.lookup(union)
    slots, old_slots = torch.searchsorted(union, rows), torch.searchsorted(union, state.rows)
    old_values, old_counts = old_values.index_copy(0, slots, values), old_counts.index_copy(0, slots, counts)
    old_last = torch.zeros_like(union).index_copy(0, old_slots, state.row_last_written_block).index_copy(0, slots, last)
    memory = BatchMemoryState(SparseMemoryState(union, old_values, old_counts), 1, state.row_capacity)
    return LazyDecayState(memory, clock, torch.zeros_like(state.cumulative_log_alpha), old_last,
                          torch.zeros(union.numel(), dtype=torch.float64, device=device))


def load_writers(options):
    import torch
    from delta_ngram.qwen38_probe import tensor_state_hash
    tensors, identities = {}, {}
    for mode, path in (("ordinary", options.main_writer), ("delta", options.full_writer)):
        expected = WRITERS[mode]
        require(digest(path) == expected["archive"], "writer archive differs: " + mode)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if mode == "ordinary":
            selected = [x for x in saved["arms"].values() if x["alpha"] == 1.]
            require(len(selected) == 1, "main alpha=1 arm missing")
            branch = selected[0]
            step = branch["optimizer_step"]
        else:
            require(saved["protocol"] == "qwen38_native_memory_fork_checkpoint_v1"
                    and saved["policy"]["branch"] == "full" and saved["completed_groups"] == 835
                    and saved["completed_targets"] == 20015113, "full fork endpoint changed")
            branch = saved["branch_state"]
            step = branch["model_optimizer_step"]
        require(step == 4172 and branch["completed_loss_targets"] == 99997072
                and tensor_state_hash(branch["writer"]) == expected["tensor"], "writer tensor endpoint differs")
        tensors[mode] = branch["writer"]
        identities[mode] = {**expected, "optimizer_step": step, "loss_targets": 99997072,
                           "target_scale_sha256": tensor_state_hash({k: v for k, v in branch["writer"].items() if "target_scale" in k})}
    return tensors, identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("delta-repo", "model-path", "verification", "inputs", "main-writer", "full-writer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--union-rows", type=Path)
    parser.add_argument("--union-sha256")
    parser.add_argument("--validate-only", action="store_true")
    options = parser.parse_args()
    options.delta_repo = options.delta_repo.resolve()
    sys.path[:0] = [str(options.delta_repo / "scripts/native_qwen38"), str(options.delta_repo / "src")]
    require(all(digest(options.delta_repo / name) == sha for name, sha in SOURCES.items()), "original H200 source pin changed")
    writer_source_sha256 = {name: digest(options.delta_repo / "src/delta_ngram" / name)
                           for name in ("qwen38_writer.py", "qwen38_training.py", "alpha_memory.py",
                                        "operators.py", "state.py", "batching.py", "qwen38_probe.py")}
    require(digest(options.inputs) == options.inputs_sha256, "timing inputs changed")
    pack = json.loads(options.inputs.read_bytes()); rows = pack["rows"]
    require(len(rows) == 16 and sum(len(r["input_ids"]) for r in rows) == 1354
            and all(canonical_hash(r["input_ids"]) == r["input_token_ids_sha256"] for r in rows), "support population changed")
    for name, sha in pack["tokenizer_files"].items():
        require(digest(options.model_path / name) == sha, "native tokenizer changed")
    import numpy as np
    import torch
    tensors, writer_identity = load_writers(options)
    union = None
    if options.union_rows is not None:
        require(digest(options.union_rows) == options.union_sha256 ==
                "e5610e9a1ef41d320136ce4c4bb6209c38490e5131fce9aad951a3fb50e40169", "capacity universe changed")
        union = torch.from_numpy(np.load(options.union_rows, allow_pickle=False))
        require(union.shape == (666747,) and union.dtype == torch.int64, "capacity universe shape differs")
    require(not torch.cuda.is_initialized(), "CPU preflight initialized CUDA")
    if options.validate_only:
        print(json.dumps({"status": "CPU_preflight_passed", "writer": writer_identity,
                          "writer_source_sha256": writer_source_sha256, "frames": 16,
                          "real_tokens": 1354, "capacity_rows": None if union is None else union.numel()})); return
    rank = int(os.environ.get("RANK", "-1"))
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
    else:
        deadline = time.monotonic() + 120
        while not options.output.is_dir():
            require(time.monotonic() < deadline, "rank0 output timeout"); time.sleep(.1)
    options.tokens = 128
    from qwen38_native_frozen_session import run_frozen_session

    def run(session):
        import torch.distributed as dist
        from megatron.core import parallel_state as ps
        from delta_ngram.alpha_memory import LazyDecayState
        from delta_ngram.qwen38_native import make_qwen38_writer, write_native_block
        from delta_ngram.qwen38_training import writer_evidence_features
        from delta_ngram.qwen38_probe import tensor_state_hash
        from delta_ngram.qwen38_precision import Qwen38SplitPLEPrecision
        from qwen38_native_dispatch import native_hc_dispatch
        from qwen38_native_router import native_router_dispatch
        control, device = dist.new_group(backend="gloo"), session.device
        first = ps.is_pipeline_first_stage()
        started = time.monotonic(); reports, seeds, warmups, hidden_records = {}, {}, [], {}
        calls = 0
        flags = {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                 "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                 "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            flags["fp32_precision"] = torch.backends.cuda.matmul.fp32_precision
        with ExitStack() as stack:
            hc = stack.enter_context(native_hc_dispatch(unwrap=True, trace=False))
            router = stack.enter_context(native_router_dispatch(session.raw))
            precision = stack.enter_context(Qwen38SplitPLEPrecision(session.adapter)) if first else None
            identity = {"base": session.identity, "runtime": session.versions, "HC": hc.identity,
                        "router": router["identity"], "precision": precision.identity() if first else None,
                        "topology": {"tp": 2, "pp": 2, "ep": 2}, "writer": writer_identity, "fp32_flags": flags,
                        "source_pins": SOURCES, "writer_source_sha256": writer_source_sha256,
                        "runner_sha256": digest(__file__), "inputs_sha256": options.inputs_sha256}
            if rank == 0:
                write_json(options.output / "registration.json", identity)
            for phase in ("small", "capacity") if union is not None else ("small",):
                for mode in ("ordinary", "delta"):
                    state = LazyDecayState.from_batch(session.layout.empty_state(1, device)) if first else None
                    setup_tick = time.monotonic()
                    if phase == "capacity" and first:
                        state = expand_capacity(restore(seeds[mode], device), union)
                        require(state.rows.numel() == 666747 and int(state.block_clock[0]) == 16, "capacity seed changed")
                    setup_seconds = time.monotonic() - setup_tick
                    writer = alpha = None
                    if rank == 0:
                        writer, alpha = make_qwen38_writer(alpha=1.)
                        writer.load_state_dict(tensors[mode], strict=True)
                        writer = writer.to(device).float().eval().requires_grad_(False)
                        alpha = alpha.to(device).eval().requires_grad_(False)
                    versions = [(p, p._version) for p in writer.parameters()] if writer is not None else []
                    dist.barrier(group=control)
                    records, saved_h = [], []
                    seen = set()
                    active_rows = rows if phase == "small" else rows[:8]
                    torch.cuda.reset_peak_memory_stats(device)
                    for index, row in enumerate(active_rows):
                        key = ((len(row["input_ids"]) + 1) // 2 * 2, mode == "delta" and (phase == "capacity" or index > 0))
                        if key not in seen:
                            seen.add(key)
                            tick = time.monotonic()
                            batch, valid = evidence_batch(row, int(session.tokenizer.pad_token_id), device)
                            read = clone_state(state) if first and mode == "delta" else None
                            guard = [(getattr(read, k), getattr(read, k)._version) for k in FIELDS] if read is not None else []
                            encode(session, batch, valid, read, mode == "delta"); calls += 1
                            require(all(v._version == ver for v, ver in guard), "warmup mutated read state")
                            torch.cuda.synchronize(device); dist.barrier(group=control)
                            warmups.append({"phase": phase, "mode": mode, "shape": key, "seconds": time.monotonic() - tick, "D1": 0})
                            del read, guard, batch, valid
                        dist.barrier(group=control); torch.cuda.synchronize(device)
                        tick = time.monotonic()
                        batch, valid = evidence_batch(row, int(session.tokenizer.pad_token_id), device)
                        torch.cuda.synchronize(device); data_at = time.monotonic()
                        read = clone_state(state) if first and mode == "delta" else None
                        guard = [(getattr(read, k), getattr(read, k)._version) for k in FIELDS] if read is not None else []
                        torch.cuda.synchronize(device); snapshot_at = time.monotonic()
                        h, physical = encode(session, batch, valid, read, mode == "delta"); calls += 1
                        torch.cuda.synchronize(device); encoded_at = time.monotonic()
                        require(all(v._version == ver for v, ver in guard), "encoding mutated detached read state")
                        length = len(row["input_ids"]); h = h[:, :length]
                        phase_times = {"data": data_at - tick, "state_snapshot": snapshot_at - data_at,
                                       "encoding_H_transport": encoded_at - snapshot_at,
                                       "features_readstats": 0., "D1": 0., "state_checks": 0., "TP_sync": 0.}
                        touched, read_nonzero = None, None
                        if rank == 0:
                            physical = physical[:, :length]
                            require(bool(h.isfinite().all()), "nonfinite H")
                            prior, _ = state.lookup(physical)
                            read_nonzero = int(prior.ne(0).any(-1).sum()) if mode == "delta" else 0
                            features = writer_evidence_features(h, mode="rmsnorm", eps=1e-6)
                            torch.cuda.synchronize(device); features_at = time.monotonic()
                            state = write_native_block(writer, state, features, physical,
                                torch.ones((1, length), dtype=torch.bool, device=device), alpha)
                            torch.cuda.synchronize(device); written_at = time.monotonic()
                            require(int(state.block_clock[0]) == (0 if phase == "small" else 16) + index + 1
                                    and bool(state.values.isfinite().all())
                                    and not bool(state.cumulative_log_alpha.any()) and not bool(state.row_log_alpha_at_write.any()), "D1 state invalid")
                            if phase == "capacity": require(state.rows.numel() == 666747, "capacity universe expanded")
                            touched = torch.unique(physical[physical.ne(0)], sorted=True)
                            torch.cuda.synchronize(device); checked_at = time.monotonic()
                            phase_times.update(features_readstats=features_at - encoded_at,
                                               D1=written_at - features_at, state_checks=checked_at - written_at)
                        sync_at = time.monotonic()
                        if first: state = sync_rows(state, touched, device)
                        torch.cuda.synchronize(device); synced_at = time.monotonic()
                        phase_times["TP_sync"] = synced_at - sync_at
                        dist.barrier(group=control); finished = time.monotonic()
                        phase_times["control_barrier"], phase_times["total"] = finished - synced_at, finished - tick
                        copies = [None] * dist.get_world_size()
                        dist.all_gather_object(copies, phase_times, group=control)
                        if rank == 0:
                            record = {"phase": phase, "mode": mode, "frame": index, "task": row["task"],
                                "source_row_index": row["source_row_index"], "true_tokens": length, "padded_tokens": batch["input_ids"].numel(),
                                "state_rows": state.rows.numel(), "block_clock": int(state.block_clock[0]), "read_nonzero_positions": read_nonzero,
                                "seconds_rank0": phase_times, "seconds_max_rank": {k: max(x[k] for x in copies) for k in phase_times}}
                            records.append(record)
                            saved_h.append(h.detach().cpu().clone())
                            print(json.dumps(record), flush=True)
                            with (options.output / "events.jsonl").open("a") as stream: stream.write(json.dumps(record) + "\n")
                        del read, guard, h, batch, valid
                    fingerprint_tick = time.monotonic()
                    signature = fingerprint(state) if first else None
                    fingerprint_seconds = time.monotonic() - fingerprint_tick
                    signatures = [None] * dist.get_world_size()
                    dist.all_gather_object(signatures, signature, group=control)
                    require(signatures[0] == signatures[1] and signatures[0] is not None
                            and signatures[2:] == [None, None], "seven-field TP state fingerprints differ")
                    if phase == "small" and first: seeds[mode] = payload(state)
                    peaks = [None] * dist.get_world_size()
                    dist.all_gather_object(peaks, {"rank": rank, "allocated": torch.cuda.max_memory_allocated(device),
                        "reserved": torch.cuda.max_memory_reserved(device), "state_setup_seconds": setup_seconds,
                        "state_fingerprint_seconds": fingerprint_seconds}, group=control)
                    if rank == 0:
                        require(all(p._version == v and p.grad is None for p, v in versions)
                                and tensor_state_hash(writer.state_dict()) == WRITERS[mode]["tensor"], "writer changed")
                        checkpoint = None
                        if phase == "small" or mode == "delta":
                            copy_tick = time.monotonic(); cpu = payload(state); copy_seconds = time.monotonic() - copy_tick
                            path = options.output / (phase + "_" + mode + ".state.pt")
                            save_tick = time.monotonic()
                            with path.open("xb") as stream:
                                torch.save({"state": cpu, "fingerprint": signature, "identity": identity}, stream)
                                stream.flush(); os.fsync(stream.fileno())
                            save_seconds = time.monotonic() - save_tick
                            hash_tick = time.monotonic(); sha = digest(path); hash_seconds = time.monotonic() - hash_tick
                            reload_tick = time.monotonic(); reloaded = torch.load(path, map_location="cpu", weights_only=True)
                            require(fingerprint(restore(reloaded["state"], "cpu")) == signature, "saved full state failed round trip")
                            checkpoint = {"path": path.name, "bytes": path.stat().st_size, "sha256": sha,
                                "GPU_to_CPU_copy_seconds": copy_seconds, "serialize_flush_fsync_seconds": save_seconds,
                                "file_SHA_seconds": hash_seconds, "readback_and_state_verify_seconds": time.monotonic() - reload_tick}
                            del cpu, reloaded
                        seconds = math.fsum(r["seconds_max_rank"]["total"] for r in records)
                        name = phase + "_" + mode
                        reports[name] = {"phase": phase, "mode": mode, "D1_writes": len(records), "real_tokens": sum(r["true_tokens"] for r in records),
                            "write_seconds": seconds, "seconds_per_frame": seconds / len(records),
                            "records": records, "state_fingerprint": signature, "gpu_peaks": peaks, "checkpoint": checkpoint,
                            "phase_seconds_rank0": {k: math.fsum(r["seconds_rank0"][k] for r in records) for k in records[0]["seconds_rank0"]}}
                        hidden_records[name] = saved_h
                        write_json(options.output / (name + ".json"), reports[name])
                        if phase == "small" and mode == "delta":
                            require(torch.equal(hidden_records["small_ordinary"][0], saved_h[0]), "zero-state H differs between writer arms")
                            require(any(r["read_nonzero_positions"] > 0 for r in records[1:]), "memory arm did not read nonzero current state")
                            write_json(options.output / "main_16_summary.json", {"status": "completed", "reports": reports,
                                "scope": "same sixteen supports, separate empty states, no TF or generation"})
                    del state, writer, alpha
                    dist.barrier(group=control)
            session.assert_frozen()
            if rank == 0:
                require(calls <= 100, "bounded encoding budget exceeded")
                torch.save(hidden_records, options.output / "evidence_H.pt")
                write_json(options.output / "COMPLETE.json", {"protocol": PROTOCOL, "status": "completed", "identity": identity,
                    "reports": reports, "warmups": warmups, "encoding_calls": calls, "D1_writes": 48 if union is not None else 32,
                    "TF_queries": 0, "generation_calls": 0, "backward_calls": 0, "optimizer_steps": 0,
                    "callback_seconds": time.monotonic() - started, "capacity_is_scientific_terminal_state": False,
                    "timing_notes": ["Measured writes include data collation, full memory read snapshot, native H transport, features, D1, state checks, TP sync and control barrier.",
                        "Rank0 phase durations are non-overlapping. Per-phase rank maxima must not be summed: TP1 synchronization includes waiting for TP0 D1.",
                        "Warmups, CPU H capture, telemetry/event IO, model loading, checkpoints and CPU fingerprints are excluded and/or recorded separately.",
                        "Capacity expands each arm's genuine sixteen-write state with zero placeholders, then performs eight engineering writes; it is not a 35000-write endpoint.",
                        "Small sample timing does not measure long-chain accuracy, evaluation, generation or a complete campaign."],
                    "H_artifact_sha256": digest(options.output / "evidence_H.pt")})
            dist.barrier(group=control)
    run_frozen_session(options, run)


if __name__ == "__main__":
    main()

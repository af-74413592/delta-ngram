"""Calibration-only native H/T0/writer observation, with zero optimizer updates.

Use the first two hash-ranked training families per source in calibration24.
Ordinary evidence is its first 512 tokens. Event evidence is the complete
first article, in <=512-token chunks with <=128 same-article prefix tokens.
The original native path sees prefix+new; only new tokens enter H statistics,
T0 statistics, writer pooling, or right context. No labels, CE, logits export,
DEV, FINAL, memory writes, optimizer, or trained checkpoint are produced.

Default is a CPU-only plan/cache validation. --execute requires torchrun 4.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time

from qwen38_native_runtime_preflight import (
    MODEL_REPO, MODEL_REVISION, SWIFT_COMMIT, BRIDGE_COMMIT, digest,
    distribution_commit, language_model, source_snapshot, swift_arguments,
    verified_model, write_json,
)


SOURCES = ("books", "code", "papers", "web", "news", "news_event")
PREFIX = 128
BLOCK = 512


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()


def file_info(path):
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": hasher.hexdigest()}


def read_tokens(path, entry):
    require(type(entry["offset_tokens"]) is int and entry["offset_tokens"] >= 0
            and type(entry["token_count"]) is int and entry["token_count"] > 0,
            "invalid cache offset or token count")
    with Path(path).open("rb") as stream:
        stream.seek(entry["offset_tokens"] * 4)
        payload = stream.read(entry["token_count"] * 4)
    require(len(payload) == entry["token_count"] * 4
            and hashlib.sha256(payload).hexdigest() == entry["token_bytes_sha256"],
            "cached token bytes differ from verified index")
    values = array("I")
    require(values.itemsize == 4, "uint32 platform width mismatch")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    ids = values.tolist()
    require(canonical_hash(ids) == entry["token_sha256"]
            and all(0 <= item < 248320 for item in ids), "invalid native token IDs")
    return ids


def verify_cache(cache_root):
    root = Path(cache_root).resolve()
    complete = json.loads((root / "COMPLETE.json").read_bytes())
    manifest = json.loads((root / "data_manifest.json").read_bytes())
    require(complete.get("format") == "qwen38_native_derived_seal_v1"
            and complete.get("ready_for_calibration") is True
            and complete.get("derivation_scope") == "calibration_subset_only"
            and complete.get("data_manifest") == file_info(root / "data_manifest.json"),
            "requires a sealed calibration-only cache")
    require(manifest.get("format") == "qwen38_native_derived_data_v1"
            and manifest.get("ready_for_calibration") is True
            and manifest.get("ready_for_training") is False
            and manifest.get("derivation_scope") == "calibration_subset_only"
            and manifest.get("model") == {"name": MODEL_REPO, "revision": MODEL_REVISION},
            "native cache model/scope/readiness mismatch")
    require(complete["files"] == manifest["files"], "cache seal file list differs")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and path.is_file()
                and file_info(path) == expected, f"cache file hash mismatch: {name}")
    selection = json.loads((root / "calibration_selection.json").read_bytes())
    require(selection.get("ready") is True and not selection.get("deficits")
            and selection.get("split") == "train" and selection.get("seed") == 17
            and selection.get("groups_per_source_requested") == 4
            and selection.get("source_manifest_sha256") == manifest["source_manifest_sha256"]
            and selection.get("tokenizer_fingerprint") == manifest["tokenizer"]["fingerprint"],
            "requires the registered seed17, four-families-per-source calibration24")
    samples = selection["samples"]
    require(len(samples) == 24
            and Counter(row["source"] for row in samples) == Counter({s: 4 for s in SOURCES})
            and all(row["split"] == "train" and row["eligible"] for row in samples),
            "calibration24 membership mismatch")
    require(len({row["sample_id"] for row in samples}) == 24
            and len({row["group_id"] for row in samples}) == 24,
            "calibration24 must contain distinct training families")
    selected, held_out = [], []
    for source in SOURCES:
        ordered = sorted((row for row in samples if row["source"] == source),
                         key=lambda row: (row["selection_hash"], row["group_id"]))
        selected.extend(ordered[:2])
        held_out.extend(ordered[2:])
    identity = {
        "cache_root": str(root), "complete": file_info(root / "COMPLETE.json"),
        "data_manifest": file_info(root / "data_manifest.json"),
        "calibration_selection": file_info(root / "calibration_selection.json"),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "tokenizer_fingerprint": manifest["tokenizer"]["fingerprint"],
        "selected_calibration_samples": selected,
        "held_out_training_probe_sample_ids": [row["sample_id"] for row in held_out],
        "selection_rule": "Per source sort by (selection_hash,group_id), use first two of four.",
    }
    return root, identity, selected


def read_index(path):
    entries = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                require(row["id"] not in entries, "duplicate native index ID")
                entries[row["id"]] = row
    return entries


def observation_frames(root, samples):
    """Derive labels-free evidence frames; never read the held-out probe tokens."""
    documents = read_index(root / "document_index.jsonl")
    events = read_index(root / "event_index.jsonl")
    frames = []
    for owner, sample in enumerate(samples):
        if sample["kind"] == "ordinary":
            entry = documents[sample["id"]]
            require(entry["group_id"] == sample["group_id"]
                    and entry["split"] == "train"
                    and entry["text_sha256"] == sample["text_sha256"]
                    and entry["token_sha256"] == sample["token_sha256"],
                    "ordinary family identity mismatch")
            ids = read_tokens(root / "tokens.uint32", entry)
            require(sample["start"] == 0 and len(ids) >= BLOCK + 128,
                    "ordinary mechanism evidence/query capacity mismatch")
            ids = ids[:BLOCK]
            article_id = entry["id"]
        elif sample["kind"] == "event":
            event = events[sample["id"]]
            require(event["group_id"] == sample["group_id"]
                    and event["split"] == "train" and event["eligible"]
                    and [a["id"] for a in event["articles"]] == sample["article_ids"]
                    and len(event["articles"]) >= 2
                    and event["articles"][1]["token_count"] >= 128,
                    "event mechanism evidence/query capacity mismatch")
            entry = event["articles"][0]
            require(entry["text_sha256"] == sample["article_text_sha256"][0]
                    and entry["token_sha256"] == sample["article_token_sha256"][0],
                    "event history article identity mismatch")
            ids = read_tokens(root / "tokens.uint32", entry)
            article_id = entry["id"]
        else:
            raise ValueError("unknown calibration sample kind")
        for start in range(0, len(ids), BLOCK):
            end = min(start + BLOCK, len(ids))
            prefix_start = max(0, start - PREFIX)
            prefix_count = start - prefix_start
            sequence = ids[prefix_start:end]
            unpadded = len(sequence)
            pad_count = unpadded % 2
            sequence = sequence + [248044] * pad_count
            encoder_valid = [True] * unpadded + [False] * pad_count
            write_mask = [False] * prefix_count + [True] * (end - start) + [False] * pad_count
            frame = {
                "frame_index": len(frames), "owner_id": owner,
                "sample_id": sample["sample_id"], "group_id": sample["group_id"],
                "source": sample["source"], "kind": sample["kind"], "article_id": article_id,
                "article_token_start": start, "article_token_end": end,
                "prefix_token_start": prefix_start, "prefix_tokens": prefix_count,
                "new_tokens": end - start, "padding_tokens": pad_count,
                "input_ids": sequence, "encoder_valid_mask": encoder_valid,
                "write_mask": write_mask,
                "position_ids": list(range(len(sequence))),
                "article_initial_chunk": start == 0,
                "loss_labels": None, "loss_target_tokens": 0,
            }
            require(len(sequence) <= BLOCK + PREFIX and len(sequence) % 2 == 0
                    and sum(write_mask) == end - start and end > start,
                    "invalid native observation frame")
            frame["frame_sha256"] = canonical_hash(frame)
            frames.append(frame)
    return frames


def plan(options, identity=None, frames=None):
    return {
        "protocol": "qwen38_native_zero_update_norm_observation_v1",
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "scope": "first two of four hash-ranked training families per six sources",
        "frame": {
            "ordinary": "first512 only",
            "news_event": "all chunks of first/history article only",
            "block_size": BLOCK, "evidence_prefix_tokens": PREFIX,
            "article_start": "natural start; no synthetic BOS/EOS",
            "padding": "right EOS248044 only to multiple of TP2; excluded from every statistic",
            "H": "actual output_layer input after HC contraction, crop to new evidence",
            "writer_context": "new evidence H only for attention pool and right window",
            "address_capture_mask": "write_mask for evidence-only adapter; prefix contributes native hash",
            "query_read_valid_mask": "not executed here; must include valid prefix in later query runner",
        },
        "writer": {"seed": 17, "modes": ["none", "rmsnorm"], "rmsnorm_eps": options.rmsnorm_eps,
                   "rmsnorm_dtype": "float32", "learned_norm_parameters": 0,
                   "target_scale": "ones16; report raw target amplitude",
                   "initial_memory": "empty for event diagnostics; no state write/decay"},
        "optimizer_steps": 0, "loss_target_tokens": 0, "labels_constructed": False,
        "logits_saved": False, "DEV_access": False, "FINAL_access": False,
        "additional_TP_gradient_probe": {
            "enabled": options.with_tp_gradient_probe,
            "registered_max_loss_target_tokens": 1000,
            "separate_from_labels_free_observation": True,
        },
        "cache": identity,
        "frame_count": len(frames) if frames is not None else None,
        "new_evidence_tokens": sum(row["new_tokens"] for row in frames) if frames is not None else None,
        "forward_input_tokens_including_prefix_and_pad": sum(len(row["input_ids"]) for row in frames)
        if frames is not None else None,
        "no_norm_decision": "Raw scale/gate observation only; no trained or held-out comparison.",
    }


def quantiles(values, torch):
    values = torch.cat([value.detach().float().reshape(-1).cpu() for value in values])
    require(bool(values.isfinite().all()) and values.numel() > 0, "nonfinite or empty statistics")
    probabilities = torch.tensor([0., .01, .05, .5, .95, .99, 1.])
    return {"count": values.numel(), "mean": float(values.mean()),
            "rms": float(values.square().mean().sqrt()),
            **{name: float(value) for name, value in zip(
                ("min", "p01", "p05", "median", "p95", "p99", "max"),
                torch.quantile(values, probabilities))}}


def execute(options, cache_identity, frames):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                "MODELSCOPE_OFFLINE", "USE_HF"):
        os.environ[key] = "1"
    os.environ["PLE_CPU_OFFLOAD"] = "1"
    os.environ["PLE_FUSED_KERNEL"] = "0"
    require(int(os.environ.get("WORLD_SIZE", "0")) == 4, "--execute requires torchrun 4 ranks")
    rank = int(os.environ["RANK"])
    identity = verified_model(options.model_path, options.verification)
    versions = {"ms-swift": distribution_commit("ms-swift", SWIFT_COMMIT),
                "mcore-bridge": distribution_commit("mcore-bridge", BRIDGE_COMMIT)}
    for name in ("torch", "megatron-core", "transformers", "transformer-engine",
                 "flash-attn", "flash-linear-attention", "tokenizers"):
        versions[name] = importlib.metadata.version(name)
    require(versions["megatron-core"] == "0.18.0" and versions["transformers"] == "5.16.0",
            "unregistered native runtime")
    sys.path.insert(0, str(options.delta_repo / "src"))
    from delta_ngram.qwen38_adapter import Qwen38NativePLEAdapter
    from delta_ngram.qwen38_native import Qwen38PLELayout, make_qwen38_writer
    from delta_ngram.qwen38_writer import NativeForwardWriter
    from delta_ngram.qwen38_training import writer_evidence_features
    from delta_ngram.state import SparseMemoryState
    for name, module in (("adapter", Qwen38NativePLEAdapter), ("writer", make_qwen38_writer),
                         ("writer_implementation", NativeForwardWriter),
                         ("features", writer_evidence_features)):
        path = Path(inspect.getfile(module)).resolve()
        require(path.is_relative_to(options.delta_repo.resolve()), "delta source outside --delta-repo")
        identity[name + "_source_sha256"] = digest(path)
    identity["observation_script_sha256"] = digest(__file__)
    identity["runtime_helper_sha256"] = digest(inspect.getfile(verified_model))
    registration = plan(options, cache_identity, frames)
    empty_path = options.output / "unused_empty_dataset.jsonl"
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        (options.output / "blocks").mkdir()
        # CLI compatibility only. The overridden run never prepares a dataset.
        empty_path.write_bytes(b"")
        write_json(options.output / "registration.json", registration)
        write_json(options.output / "frames.json", frames)
        write_json(options.output / "INCOMPLETE.json", {"status": "loading_frozen_reader"})
    else:
        deadline = time.monotonic() + 300
        while not empty_path.is_file():
            require(time.monotonic() < deadline, "rank0 output preparation timeout")
            time.sleep(.1)

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from swift.megatron.model import get_mcore_model
    from swift.megatron.pipelines.train.sft import MegatronSft
    from swift.megatron.utils import forward_step_helper, wrap_model

    class NativeObservation(MegatronSft):
        def run(self):
            started = time.monotonic()
            device = torch.device("cuda", torch.cuda.current_device())
            agreements = [None] * dist.get_world_size()
            dist.all_gather_object(agreements, canonical_hash(
                {"identity": identity, "registration": registration, "frames": frames}))
            require(len(set(agreements)) == 1, "ranks disagree on observation inputs")
            require((ps.get_tensor_model_parallel_world_size(),
                     ps.get_pipeline_model_parallel_world_size(),
                     ps.get_expert_model_parallel_world_size(),
                     ps.get_data_parallel_world_size(),
                     ps.get_context_parallel_world_size()) == (2, 2, 2, 1, 1),
                    "requires TP2 PP2 EP2 DP1 CP1")
            sources = source_snapshot(options.output) if rank == 0 else None
            helper_source = inspect.getsource(forward_step_helper)
            if rank == 0:
                (options.output / "runtime_sources" / "forward_step_helper.py").write_text(
                    helper_source, encoding="utf-8")
            raw_models = get_mcore_model(self.args, self.template.config)
            require(len(raw_models) == 1, "requires one non-interleaved local PP chunk")
            raw, config = raw_models[0], raw_models[0].config
            config.recompute_granularity = None
            config.recompute_method = None
            config.recompute_modules = []
            config.deallocate_pipeline_outputs = False
            config.overlap_p2p_comm = False
            config.variable_seq_lengths = True
            config.calculate_per_token_loss = False
            for name in ("grad_scale_func", "no_sync_func", "grad_sync_func",
                         "param_sync_func", "finalize_model_grads_func"):
                setattr(config, name, None)
            config.bridge.load_weights(raw_models, self.args.model_dir)
            raw.requires_grad_(False).eval()
            models = wrap_model(self.args, raw_models, wrap_with_ddp=False)
            require(len(models) == 1, "unexpected model wrapper chunks")
            model = models[0]
            model.requires_grad_(False).eval()
            parameters = list(raw.named_parameters())
            before = {name: (id(value), value._version) for name, value in parameters}
            require(not any(value.requires_grad or value.grad is not None
                            for _, value in parameters), "reader must be frozen")
            native = [module for module in raw.modules()
                      if type(module).__name__ == "Qwen4ExpTextPLELayer"]
            require(len(native) == (1 if ps.is_pipeline_first_stage() else 0),
                    "native PLE placement mismatch")
            adapter, embedding, gather_module, gather_original, host_identity = (None,) * 5
            layout = None
            hook = None
            head = None
            current_frame = None
            writer = None
            t0_square_sum = torch.zeros(16, dtype=torch.float64)
            t0_token_count = 0
            summaries, artifacts, hidden_rms = [], [], []
            mode_data = {mode: {"input_rms": [], "targets": [[] for _ in range(16)],
                                "beta": [[] for _ in range(16)]}
                         for mode in ("none", "rmsnorm")}
            writer_signature = None
            try:
                if native:
                    ple = native[0]
                    embedding = ple.ple_embedding
                    require(embedding.cpu_offload and embedding.host_table.device.type == "cpu"
                            and embedding.host_table.dtype == torch.bfloat16,
                            "native T0 must remain frozen CPU BF16")
                    host_identity = (id(embedding.host_table), embedding.host_table._version)
                    import importlib
                    gather_module = importlib.import_module(type(ple).__module__)
                    gather_original = getattr(gather_module, "gather_ple_rows", None)
                    require(callable(gather_original), "missing audited native host lookup switch")
                    gather_module.gather_ple_rows = lambda *args, **kwargs: None
                    layout = Qwen38PLELayout.from_native(
                        json.loads((options.model_path / "config.json").read_bytes()),
                        embedding.ngram_heads_offsets, embedding.ngram_heads_vocab_sizes)
                    adapter = Qwen38NativePLEAdapter(ple, layout)
                if ps.is_pipeline_last_stage():
                    lm = language_model(raw)
                    require(type(lm.decoder.final_layernorm).__name__ == "IdentityOp",
                            "native final norm/HC path differs")

                    def capture_head(module, args, kwargs):
                        nonlocal head
                        value = args[0] if args else kwargs.get(
                            "hidden_states", kwargs.get("input_", kwargs.get("input")))
                        require(value is not None and head is None
                                and tuple(value.shape) == (len(current_frame["input_ids"]) // 2, 1, 2560),
                                "actual LM-head input must be unique SP [T/2,1,2560]")
                        head = value.detach().clone()

                    hook = lm.output_layer.register_forward_pre_hook(capture_head, with_kwargs=True)
                if rank == 0:
                    require(ps.is_pipeline_first_stage() and ps.get_tensor_model_parallel_rank() == 0,
                            "rank0 must own first-stage TP0 statistics")
                    with torch.random.fork_rng(devices=[device.index]):
                        torch.manual_seed(17)
                        writer, _ = make_qwen38_writer(alpha=1.0)
                    writer = writer.to(device).float().eval().requires_grad_(False)
                    require(torch.equal(writer.target_scale, torch.ones(16, device=device)),
                            "observation must use raw target_scale=1")
                    writer_signature = {name: (id(value), value._version)
                                        for name, value in writer.named_parameters()}
                    path = options.output / "writer_initial_state.pt"
                    torch.save({name: value.detach().cpu() for name, value in writer.state_dict().items()}, path)
                    artifacts.append({"path": path.name, **file_info(path)})
                for frame in frames:
                    current_frame = frame
                    head = None
                    frame_started = time.monotonic()
                    ids = torch.tensor([frame["input_ids"]], dtype=torch.long, device=device)
                    write_mask = torch.tensor([frame["write_mask"]], dtype=torch.bool, device=device)
                    owners = torch.tensor([frame["owner_id"]], dtype=torch.long, device=device)
                    inputs = {"input_ids": ids, "position_ids": torch.arange(
                        ids.shape[1], device=device).unsqueeze(0), "attention_mask": None,
                        "labels": None}
                    with torch.no_grad():
                        if adapter is not None:
                            output, capture = adapter.run_native(
                                lambda: forward_step_helper(model, inputs),
                                input_ids=ids, valid_mask=write_mask, owner_ids=owners,
                                mode="evidence", state=None)
                        else:
                            output = forward_step_helper(model, inputs)
                            capture = None
                        del output  # Full logits are neither retained nor inspected.
                        full_head = torch.empty((ids.shape[1], 1, 2560),
                                                device=device, dtype=torch.bfloat16)
                        if ps.is_pipeline_last_stage():
                            require(head is not None and head.dtype == torch.bfloat16,
                                    "missing BF16 actual head capture")
                            parts = [torch.empty_like(head) for _ in range(2)]
                            dist.all_gather(parts, head.contiguous(),
                                            group=ps.get_tensor_model_parallel_group())
                            full_head.copy_(torch.cat(parts, dim=0))
                        dist.broadcast(full_head, src=ps.get_pipeline_model_parallel_last_rank(),
                                       group=ps.get_pipeline_model_parallel_group())
                        if rank == 0:
                            require(capture is not None and capture.lookup_backend == "host",
                                    "missing native host PLE capture")
                            begin, count = frame["prefix_tokens"], frame["new_tokens"]
                            new_h = full_head[begin:begin + count].transpose(0, 1).contiguous()
                            new_rows = capture.physical_rows[:, begin:begin + count].contiguous()
                            mask = torch.ones((1, count), dtype=torch.bool, device=device)
                            require(bool(new_rows.ne(0).all()) and bool(new_h.isfinite().all()),
                                    "invalid new-evidence H or physical row")
                            hrms = new_h.float().square().mean(-1).sqrt().cpu()
                            hidden_rms.append(hrms)
                            base_rms = capture.base_head_rms.detach().double().cpu()
                            require(base_rms.shape == (16,) and bool(base_rms.isfinite().all())
                                    and bool(base_rms.gt(0).all()), "invalid native T0 RMS")
                            t0_square_sum += base_rms.square() * count
                            t0_token_count += count
                            report = {"frame_index": frame["frame_index"],
                                      "frame_sha256": frame["frame_sha256"],
                                      "sample_id": frame["sample_id"], "source": frame["source"],
                                      "new_tokens": count, "prefix_tokens": begin,
                                      "H_token_rms": quantiles([hrms], torch),
                                      "T0_head_rms": base_rms.tolist(), "modes": {}}
                            head_ids = (mask[:, :, None] & new_rows.ne(0)).nonzero()[:, 2]
                            for mode in ("none", "rmsnorm"):
                                features = writer_evidence_features(
                                    new_h, mode=mode, eps=options.rmsnorm_eps)
                                events = writer.events(features, new_rows, mask,
                                                       SparseMemoryState.empty(160, device))
                                input_rms = features.square().mean(-1).sqrt().cpu()
                                target_rms = events.target.square().mean(-1).sqrt()
                                require(bool(events.target.isfinite().all())
                                        and bool(events.beta.isfinite().all()), "nonfinite raw writer observation")
                                mode_data[mode]["input_rms"].append(input_rms)
                                for h in range(16):
                                    select = head_ids == h
                                    mode_data[mode]["targets"][h].append(target_rms[select].cpu())
                                    mode_data[mode]["beta"][h].append(events.beta[select].cpu())
                                report["modes"][mode] = {
                                    "input_token_rms": quantiles([input_rms], torch),
                                    "raw_target_event_rms": quantiles([target_rms], torch),
                                    "beta": quantiles([events.beta], torch),
                                    "raw_target_max_abs": float(events.target.abs().max()),
                                    "event_count": events.rows.numel(),
                                }
                            block_path = options.output / "blocks" / f'{frame["frame_index"]:04d}.pt'
                            torch.save({"frame": frame, "H_new": new_h.cpu(),
                                        "physical_rows_new": new_rows.cpu(),
                                        "native_ids": capture.native_ids.cpu(),
                                        "T0_head_rms_new": base_rms,
                                        "H_site": "actual output_layer input after native HC contraction",
                                        "reader_delta": None, "loss_labels": None}, block_path)
                            report["artifact"] = {"path": block_path.relative_to(options.output).as_posix(),
                                                  **file_info(block_path)}
                            artifacts.append(report["artifact"])
                            report["seconds"] = time.monotonic() - frame_started
                            write_json(block_path.with_suffix(".json"), report)
                            summaries.append(report)
                            write_json(options.output / "progress.json", {
                                "status": "incomplete", "completed_frames": len(summaries),
                                "total_frames": len(frames), "new_evidence_tokens": t0_token_count,
                                "optimizer_steps": 0, "loss_target_tokens": 0, "last_frame": report})
                            print(json.dumps({"event": "observation_frame_complete",
                                              "frame": frame["frame_index"],
                                              "frames": len(frames), "sample_id": frame["sample_id"],
                                              "new_tokens": count, "seconds": report["seconds"]}), flush=True)
                    dist.barrier()
                if hook is not None:
                    hook.remove()
                    hook = None
                tp_probe = None
                if options.with_tp_gradient_probe:
                    from qwen38_native_tp_gradient_probe import run_tp_gradient_probe
                    tp_probe = run_tp_gradient_probe(
                        models=models, adapter=adapter, layout=layout, device=device,
                        tokenizer=self.template.tokenizer, output=options.output / "tp_fd")
                require(all(not value.requires_grad and value.grad is None
                            and (id(value), value._version) == before[name]
                            for name, value in parameters), "frozen reader changed")
                if embedding is not None:
                    require((id(embedding.host_table), embedding.host_table._version) == host_identity,
                            "native host table changed")
                dist.barrier()
                if rank == 0:
                    require(all(not value.requires_grad and value.grad is None
                                and (id(value), value._version) == writer_signature[name]
                                for name, value in writer.named_parameters()), "initial writer changed")
                    modes = {}
                    for mode, values in mode_data.items():
                        modes[mode] = {
                            "input_token_rms": quantiles(values["input_rms"], torch),
                            "raw_target_event_rms_by_head": [
                                quantiles(parts, torch) for parts in values["targets"]],
                            "beta_by_head": [quantiles(parts, torch) for parts in values["beta"]],
                        }
                    result = {
                        "status": "complete", "protocol": registration["protocol"],
                        "identity": identity, "versions": versions, "registration": registration,
                        "runtime_sources": sources,
                        "forward_helper_source_sha256": hashlib.sha256(helper_source.encode()).hexdigest(),
                        "H_token_rms": quantiles(hidden_rms, torch),
                        "T0_head_rms_new": (t0_square_sum / t0_token_count).sqrt().tolist(),
                        "T0_weighting": "each new token and each of160 coordinates once; TP replicas excluded",
                        "writer_modes": modes, "artifacts": artifacts,
                        "frames_completed": len(summaries), "new_evidence_tokens": t0_token_count,
                        "optimizer_steps": 0, "loss_target_tokens": 0,
                        "additional_TP_gradient_probe": tp_probe,
                        "frozen_reader_unchanged": True, "native_host_table_unchanged": True,
                        "initial_writer_unchanged": True, "seconds": time.monotonic() - started,
                        "conclusion": "Scale/gate observation only; no normalization selection or training result.",
                    }
                    write_json(options.output / "COMPLETE.json", result)
                    print(json.dumps({"event": "observation_complete", "frames": len(summaries),
                                      "new_tokens": t0_token_count, "loss_target_tokens": 0,
                                      "optimizer_steps": 0}), flush=True)
                dist.barrier()
            finally:
                if hook is not None:
                    hook.remove()
                if adapter is not None:
                    adapter.close()
                if gather_module is not None:
                    gather_module.gather_ple_rows = gather_original
            return {"zero_update_observation": True}

    options.tokens = BLOCK + PREFIX
    runner = NativeObservation(swift_arguments(options, empty_path))
    runner.main()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--delta-repo", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rmsnorm-eps", type=float, default=1e-6)
    parser.add_argument("--with-tp-gradient-probe", action="store_true")
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    require(options.rmsnorm_eps == 1e-6, "registered observation epsilon is exactly1e-6")
    cache_identity = frames = None
    if options.cache_root is not None:
        root, cache_identity, selected = verify_cache(options.cache_root)
        frames = observation_frames(root, selected)
    if not options.execute:
        print(json.dumps(plan(options, cache_identity, frames), indent=2, allow_nan=False))
        return
    require(all(getattr(options, name) is not None for name in (
        "model_path", "verification", "delta_repo", "cache_root", "output")),
        "--execute requires model, verification, delta repo, calibration cache, and output")
    execute(options, cache_identity, frames)


if __name__ == "__main__":
    main()

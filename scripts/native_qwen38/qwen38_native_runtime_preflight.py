"""Bounded native PLE runtime acceptance on the pinned four-H200 stack.

Default is a CPU-only plan. --execute must be launched with torchrun (4 ranks).
Only new synthetic 16/32-token inputs are used; no dataset/DEV/FINAL is read.
MegatronSft supplies distributed initialization/model metadata only. Its run()
is overridden: no trainer, DDP, optimizer, update, or checkpoint is created.

The three one-microbatch passes are original base, empty native delta, and a
nonzero leaf delta with a real pipeline backward. This accepts reader-to-delta
autograd, NOT writer BPTT, writer TP reduction, resume, or training convergence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time


MODEL_REPO = "Qwen/Qwen3.8-Flash-Next"
MODEL_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
SWIFT_COMMIT = "21ada46205e0abfc6ab33ef8b6dd260cc4eccf8d"
BRIDGE_COMMIT = "3a5ab71264ca0c98db9eab7cc4de58aa57e065e3"
TEXT = "Native memory runtime check. The small red box contains three blue stones. "


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checked_tokens(tokenizer, length):
    """Explicit EOS-reset pair plus right padding, with no chat template."""
    if length not in (16, 32):
        raise ValueError("only the registered 16/32-token synthetic probes are allowed")
    candidates = tokenizer.encode(TEXT * 4, add_special_tokens=False)
    candidates = [int(token) for token in candidates if 0 <= int(token) < 248044]
    if len(candidates) < length or len(set(candidates[:2])) < 2:
        raise ValueError("synthetic source did not encode to two distinct ordinary tokens")
    # The native hasher's EOS is config.eos_token_id=248044, independently
    # of the chat tokenizer's eos_token=<|im_end|>.
    valid_length = length - 4
    ids = [candidates[0], candidates[1], 248044, candidates[0], candidates[1]]
    ids.extend(candidates[2:2 + valid_length - len(ids)])
    ids.extend([248044] * (length - len(ids)))
    labels = ids[1:valid_length] + [-100] * (length - valid_length + 1)
    mask = [position < valid_length for position in range(length)]
    assert len(ids) == len(labels) == len(mask) == length
    return {"input_ids": ids, "labels": labels, "valid_mask": mask,
            "valid_tokens": valid_length, "loss_tokens": valid_length - 1,
            "eos_reset_compare_positions": [[0, 1], [3, 4]],
            "native_eos_token_id": 248044, "synthetic_source": TEXT}


def verified_model(model_path, verification):
    """Accept a completed prior hash audit, then check shard file signatures.

    Historical receipts preserve exact file mtimes. The new HF download
    receipt did not save per-file mtimes: for that format, require unchanged
    sizes and mtimes no later than completion, and rehash small config/tokenizer
    files. Report this weaker continuity boundary explicitly.
    """
    root = Path(model_path).resolve()
    report = json.loads(Path(verification).read_bytes())
    index = json.loads((root / "model.safetensors.index.json").read_bytes())
    shards = set(index["weight_map"].values())
    if report.get("repo") != MODEL_REPO or report.get("revision") != MODEL_REVISION or len(shards) != 131:
        raise ValueError("requires the fixed revision and exactly 131 indexed weight shards")
    if report.get("all_file_hashes_verified") is True and isinstance(report.get("files"), list):
        files = {entry["path"]: entry for entry in report["files"]}
        if (len(files) != len(report["files"])
                or {name for name in files if name.endswith(".safetensors")} != shards):
            raise ValueError("download receipt and checkpoint index disagree")
        finished = datetime.fromisoformat(report["finished_at"])
        if finished.tzinfo is None:
            raise ValueError("download completion timestamp must include a timezone")
        finished_ns = int(finished.timestamp() * 1_000_000_000)
        for name, entry in files.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"invalid downloaded file: {name}")
            stat = path.stat()
            if (entry.get("verified") is not True or stat.st_size != entry["size"]
                    or stat.st_mtime_ns > finished_ns or len(entry.get("sha256", "")) != 64
                    or (entry.get("lfs_sha256") and entry["sha256"] != entry["lfs_sha256"])):
                raise ValueError(f"downloaded file is not within the completed hash audit boundary: {name}")
        for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                     "tokenizer_config.json"):
            if name not in files or digest(root / name) != files[name]["sha256"]:
                raise ValueError(f"small runtime input changed since download verification: {name}")
        continuity = ("Prior full HF content-hash audit plus current file sizes and mtime<=finished_at; "
                      "per-file original mtime was not recorded. Config/index/tokenizer hashes rechecked.")
    else:
        if (report.get("state") != "complete" or report.get("counts") != {"passed": 131}
                or Path(report.get("root", "")).resolve() != root):
            raise ValueError("unsupported or incomplete full-hash verification receipt")
        files = report["files"]
        if set(files) != shards:
            raise ValueError("historical hash receipt and checkpoint index disagree")
        for name, entry in files.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError("shard escapes model root")
            signature, expected, stat = entry["signature"], entry["expected"], path.stat()
            if (entry.get("status") != "passed" or entry.get("sha256") != expected.get("sha256")
                    or len(entry["sha256"]) != 64 or stat.st_size != expected["bytes"]
                    or stat.st_mtime_ns != signature["mtime_ns"]):
                raise ValueError(f"shard identity changed after full hash verification: {name}")
        continuity = "Historical full-shard SHA audit plus exact current size/mtime signature checks."
    config = json.loads((root / "config.json").read_bytes())
    text = config["text_config"]
    expected_config = {"vocab_size": 248320, "hidden_size": 2560, "hc_count": 4,
                       "eos_token_id": 248044, "ple_layer_ids": [2],
                       "num_hidden_layers": 48, "heads_per_ngram": 8, "ngram_size": 3}
    if config.get("model_type") != "qwen4_exp" or any(
            text.get(key) != value for key, value in expected_config.items()):
        raise ValueError("model config differs from the registered native architecture")
    return {"repo": MODEL_REPO, "revision": MODEL_REVISION,
            "verification_sha256": digest(verification),
            "config_sha256": digest(root / "config.json"),
            "index_sha256": digest(root / "model.safetensors.index.json"),
            "tokenizer_sha256": digest(root / "tokenizer.json"),
            "prior_full_hash_audit_continuity": continuity,
            "full_weight_hashes_recomputed_by_preflight": False}


def distribution_commit(name, expected):
    """Use VCS provenance or a clean pinned editable checkout."""
    from urllib.parse import unquote, urlparse
    import subprocess
    distribution = importlib.metadata.distribution(name)
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = provenance.get("vcs_info", {}).get("commit_id")
    if commit is None and provenance.get("dir_info", {}).get("editable"):
        url = urlparse(provenance["url"])
        if url.scheme != "file":
            raise ValueError("editable source must be local")
        source = Path(unquote(url.path))
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        changed = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            text=True).strip()
        if changed:
            raise ValueError(f"{name} pinned source has tracked modifications")
    if commit != expected:
        raise ValueError(f"{name} does not resolve to the accepted commit")
    return {"version": distribution.version, "commit": commit}


def swift_arguments(options, synthetic_path):
    return [
        "--model", str(options.model_path), "--model_type", "qwen4_exp",
        "--bridge_backend", "mcore-bridge", "--check_model", "false",
        "--dataset", str(synthetic_path), "--train_iters", "1", "--eval_iters", "0",
        "--output_dir", str(options.output / "framework"), "--add_version", "false",
        "--language_model_only", "true", "--freeze_vit", "true", "--freeze_aligner", "true",
        "--tuner_type", "full", "--linear_decoupled_in_proj", "false",
        "--mtp_num_layers", "0", "--gradient_accumulation_fusion", "false",
        "--recompute_granularity", "none",
        "--tensor_model_parallel_size", "2", "--pipeline_model_parallel_size", "2",
        "--expert_model_parallel_size", "2", "--expert_tensor_parallel_size", "1",
        "--decoder_first_pipeline_num_layers", "24", "--context_parallel_size", "1",
        "--sequence_parallel", "true", "--moe_grouped_gemm", "true",
        "--moe_permute_fusion", "true", "--use_cpu_initialization", "true",
        "--padding_free", "false", "--packing", "false", "--padding_side", "right",
        "--attention_backend", "flash", "--torch_dtype", "bfloat16",
        "--max_length", str(options.tokens), "--micro_batch_size", "1",
        "--global_batch_size", "1", "--cross_entropy_loss_fusion", "false",
        "--use_distributed_optimizer", "false", "--enable_thinking", "false",
        "--strict", "true", "--lazy_tokenize", "false", "--seed", "42",
        "--dataset_num_proc", "1", "--dataloader_num_workers", "0",
    ]


def first_tensor(value, torch):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise ValueError("native output hook did not receive a tensor or (tensor, bias)")


def language_model(raw):
    seen = set()
    while not (hasattr(raw, "output_layer") and hasattr(raw, "decoder")):
        if id(raw) in seen:
            raise ValueError("cycle resolving the native language model")
        seen.add(id(raw))
        raw = getattr(raw, "module", getattr(raw, "language_model", None))
        if raw is None:
            raise ValueError("cannot locate the actual native language-model head")
    return raw


def tensor_summary(tensor):
    floating = tensor.detach().float()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "finite": bool(floating.isfinite().all()),
            "rms": float(floating.square().mean().sqrt()) if tensor.numel() else 0.,
            "max_abs": float(floating.abs().max()) if tensor.numel() else 0.}


def head_summary(hidden, valid, tp_rank):
    """Keep SP token ownership; omit right padding from scale diagnostics."""
    local_start = tp_rank * hidden.shape[0]
    local_valid = valid[0, local_start:local_start + hidden.shape[0]].cpu()
    selected = hidden[local_valid]
    rms = selected.float().square().mean(-1).sqrt().flatten()
    return {
        "all_positions": tensor_summary(hidden),
        "valid_positions": tensor_summary(selected),
        "valid_global_positions": [local_start + i for i, keep in
                                   enumerate(local_valid.tolist()) if keep],
        "valid_token_rms": rms.tolist(),
        "valid_token_rms_min": float(rms.min()) if rms.numel() else None,
        "valid_token_rms_max": float(rms.max()) if rms.numel() else None,
    }


def source_snapshot(output):
    """Save the actual installed autograd/parallel source, not just versions."""
    import importlib
    modules = (
        "megatron.core.tensor_parallel.mappings",
        "megatron.core.extensions.transformer_engine",
        "megatron.core.pipeline_parallel.schedules",
        "transformer_engine.pytorch.module.linear",
        "mcore_bridge.model.modules.ple",
    )
    target = Path(output) / "runtime_sources"
    target.mkdir()
    records = []
    for name in modules:
        module = importlib.import_module(name)
        path = Path(inspect.getfile(module))
        data = path.read_bytes()
        saved = target / (name.replace(".", "_") + ".py")
        saved.write_bytes(data)
        records.append({"module": name, "installed_path": str(path),
                        "saved_path": saved.relative_to(output).as_posix(),
                        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return records


def schedule_query(models, batch, *, valid_mask, owner_ids, adapter=None,
                   reader_state=None, backward=False, mode="query",
                   group_total_target_tokens, query_grad_enabled=None):
    """One native CP1/DP1/PP2 microbatch; no optimizer and no hidden loss mean.

    ``batch["labels"]`` must ALREADY be shifted to prediction positions, with
    -100 for excluded targets. Differentiated loss is query_sum / the explicit
    group denominator, which can cover a whole writer trajectory. MCore's
    remaining CP multiplier and microbatch divisor are both exactly one.

    reader_state may be a leaf copy of a writer snapshot. Its .values.grad
    remains visible after this call; the caller owns the separate writer VJP.
    Full last-H collection is via a hook on the actual model output_layer.

    query_grad_enabled defaults to backward. Set it True for finite-difference
    forward-only calls, with the SAME requires-grad delta boundary as analytic
    backward. This preserves grad-sensitive native forward dispatch while the
    schedule's forward_only flag controls whether backward is executed.
    """
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.pipeline_parallel import get_forward_backward_func
    if query_grad_enabled is None:
        query_grad_enabled = backward
    if type(query_grad_enabled) is not bool or (backward and not query_grad_enabled):
        raise ValueError("backward requires an explicit grad-enabled query forward")
    if type(group_total_target_tokens) is not int or group_total_target_tokens < 1:
        raise ValueError("an explicit positive group target-token denominator is required")
    if (len(models) != 1 or ps.get_context_parallel_world_size() != 1
            or ps.get_data_parallel_world_size() != 1):
        raise ValueError("initial schedule adapter supports one chunk, CP1 and DP1 only")
    labels = batch["labels"]
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("initial schedule adapter requires one sequence")
    count = int(labels.ne(-100).sum())
    if count < 1 or count > group_total_target_tokens:
        raise ValueError("invalid query target count or group denominator")
    captured = None

    def loss_func(losses):
        selected = losses.float()[labels != -100]
        if selected.numel() != count or not bool(selected.isfinite().all()):
            raise ValueError("invalid native loss positions or nonfinite loss")
        total = selected.sum()
        statistic = torch.stack((total.detach(), total.new_tensor(count)))
        return total / group_total_target_tokens, {"loss_sum_and_tokens": statistic}

    def forward_step(data_iterator, model):
        nonlocal captured
        current = dict(next(data_iterator))
        if not ps.is_pipeline_last_stage():
            current["labels"] = None
        if adapter is None:
            output = model(**current)
        else:
            output, captured = adapter.run_native(
                lambda: model(**current), input_ids=batch["input_ids"], valid_mask=valid_mask,
                owner_ids=owner_ids, mode=mode, state=reader_state)
        return output, loss_func

    schedule = get_forward_backward_func()
    if schedule.__name__ != "forward_backward_pipelining_without_interleaving":
        raise ValueError(f"unexpected pipeline schedule: {schedule.__name__}")
    with torch.enable_grad() if query_grad_enabled else torch.no_grad():
        metrics = schedule(
            forward_step_func=forward_step, data_iterator=iter([batch]), model=models,
            num_microbatches=1, seq_length=labels.shape[1], micro_batch_size=1,
            forward_only=not backward)
    statistic = torch.zeros(2, dtype=torch.float32, device=labels.device)
    if ps.is_pipeline_last_stage():
        if len(metrics) != 1:
            raise ValueError("one microbatch must return one final-stage metric")
        statistic.copy_(metrics[0]["loss_sum_and_tokens"])
    dist.broadcast(statistic, src=ps.get_pipeline_model_parallel_last_rank(),
                   group=ps.get_pipeline_model_parallel_group())
    if int(statistic[1]) != count:
        raise ValueError("pipeline metric target count mismatch")
    return {"metrics": metrics, "capture": captured,
            "loss_sum": float(statistic[0]), "query_target_tokens": count,
            "group_total_target_tokens": group_total_target_tokens,
            "normalized_loss": float(statistic[0]) / group_total_target_tokens,
            "query_grad_enabled": query_grad_enabled,
            "schedule": schedule.__name__}


def execute(options):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                "MODELSCOPE_OFFLINE", "USE_HF"):
        os.environ[key] = "1"
    os.environ["PLE_CPU_OFFLOAD"] = "1"
    os.environ["PLE_FUSED_KERNEL"] = "0"
    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise ValueError("--execute requires torchrun --nproc_per_node=4")
    rank = int(os.environ["RANK"])
    identity = verified_model(options.model_path, options.verification)
    versions = {
        "ms-swift": distribution_commit("ms-swift", SWIFT_COMMIT),
        "mcore-bridge": distribution_commit("mcore-bridge", BRIDGE_COMMIT),
    }
    for name in ("torch", "megatron-core", "transformers", "transformer-engine",
                 "flash-attn", "flash-linear-attention", "tokenizers"):
        versions[name] = importlib.metadata.version(name)
    if versions["megatron-core"] != "0.18.0" or versions["transformers"] != "5.16.0":
        raise ValueError("requires the audited Megatron-Core 0.18.0 / Transformers 5.16.0 stack")
    sys.path.insert(0, str(options.delta_repo / "src"))
    from delta_ngram.qwen38_adapter import Qwen38NativePLEAdapter
    from delta_ngram.qwen38_native import Qwen38PLELayout
    from delta_ngram.batching import BatchMemoryState
    from delta_ngram.state import SparseMemoryState
    adapter_source = Path(inspect.getfile(Qwen38NativePLEAdapter)).resolve()
    if not adapter_source.is_relative_to(options.delta_repo.resolve()):
        raise ValueError("delta adapter resolved outside --delta-repo")
    identity["adapter_sha256"] = digest(adapter_source)
    identity["native_boundary_sha256"] = digest(inspect.getfile(Qwen38PLELayout))
    identity["preflight_sha256"] = digest(__file__)
    synthetic_path = options.output / "synthetic_only.jsonl"
    if rank == 0:
        options.output.mkdir(parents=True, exist_ok=False)
        synthetic_path.write_text(json.dumps({"messages": [
            {"role": "user", "content": TEXT}, {"role": "assistant", "content": "Three."}
        ]}) + "\n", encoding="utf-8")
    else:
        deadline = time.monotonic() + 300
        while not synthetic_path.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError("rank 0 did not prepare the new output directory")
            time.sleep(.1)

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from swift.megatron.model import get_mcore_model
    from swift.megatron.pipelines.train.sft import MegatronSft
    from swift.megatron.utils import wrap_model

    class NativePreflight(MegatronSft):
        def run(self):
            started = time.monotonic()
            device = torch.device("cuda", torch.cuda.current_device())
            case = checked_tokens(self.template.tokenizer, options.tokens)
            agreements = [None] * dist.get_world_size()
            protocol = {"identity": identity, "tokens": case}
            dist.all_gather_object(agreements, hashlib.sha256(
                json.dumps(protocol, sort_keys=True).encode()).hexdigest())
            if len(set(agreements)) != 1:
                raise ValueError("ranks disagree on model, adapter or synthetic tokens")
            runtime_sources = source_snapshot(options.output) if rank == 0 else None
            if (ps.get_tensor_model_parallel_world_size(), ps.get_pipeline_model_parallel_world_size(),
                    ps.get_expert_model_parallel_world_size(), ps.get_data_parallel_world_size()) != (2, 2, 2, 1):
                raise ValueError("unexpected TP/PP/EP/DP topology")
            raw_models = get_mcore_model(self.args, self.template.config)
            if len(raw_models) != 1:
                raise ValueError("one local non-interleaved pipeline chunk is required")
            raw, config = raw_models[0], raw_models[0].config
            if config.padded_vocab_size != 248320 or config.cross_entropy_loss_fusion:
                raise ValueError("requires exact 248320 vocabulary and unfused FP32 CE")
            # No PLE replay is permitted outside adapter.run_native. The tiny
            # probe has enough memory to retain all downstream activations.
            config.recompute_granularity = None
            config.recompute_method = None
            config.recompute_modules = []
            config.deallocate_pipeline_outputs = False
            config.overlap_p2p_comm = False
            config.variable_seq_lengths = True  # PP residual width is hc_count * hidden_size.
            config.calculate_per_token_loss = False
            for name in ("grad_scale_func", "no_sync_func", "grad_sync_func",
                         "param_sync_func", "finalize_model_grads_func"):
                setattr(config, name, None)
            config.bridge.load_weights(raw_models, self.args.model_dir)
            raw.requires_grad_(False).eval()
            models = wrap_model(self.args, raw_models, wrap_with_ddp=False)
            for model in models:
                model.requires_grad_(False).eval()
            if len(models) != 1:
                raise ValueError("unexpected wrapper chunk count")
            named_parameters = list(raw.named_parameters())
            before = {name: (id(param), param._version) for name, param in named_parameters}
            if any(param.requires_grad or param.grad is not None for _, param in named_parameters):
                raise ValueError("reader is not fully frozen")
            native = [module for module in raw.modules()
                      if type(module).__name__ == "Qwen4ExpTextPLELayer"]
            if len(native) != (1 if ps.is_pipeline_first_stage() else 0):
                raise ValueError("native PLE must exist once on each TP rank of the first stage")
            ple = native[0] if native else None
            host_identity = None
            host_gather_module = None
            host_gather_original = None
            layout = None
            if ple is not None:
                embedding = ple.ple_embedding
                if not embedding.cpu_offload or embedding.host_table.device.type != "cpu":
                    raise ValueError("native table must remain on the CPU")
                if embedding.host_table.dtype != torch.bfloat16:
                    raise ValueError("native host table was not dequantized to BF16")
                host_identity = (id(embedding.host_table), embedding.host_table._version)
                # Select the existing torch-gather/H2D fallback in _host_lookup,
                # without replacing the native hash, values, scale, or TP sum.
                # This process-local switch is restored in finally.
                import importlib
                host_gather_module = importlib.import_module(type(ple).__module__)
                host_gather_original = getattr(host_gather_module, "gather_ple_rows", None)
                if not callable(host_gather_original):
                    raise ValueError("cannot select the audited native host-gather fallback")
                host_gather_module.gather_ple_rows = lambda *unused_args, **unused_kwargs: None
                layout = Qwen38PLELayout.from_native(
                    json.loads((options.model_path / "config.json").read_bytes()),
                    embedding.ngram_heads_offsets, embedding.ngram_heads_vocab_sizes)
            batch = {
                "input_ids": torch.tensor([case["input_ids"]], dtype=torch.long, device=device),
                "labels": torch.tensor([case["labels"]], dtype=torch.long, device=device),
                "position_ids": torch.arange(options.tokens, device=device).unsqueeze(0),
                "attention_mask": None,
            }
            valid = torch.tensor([case["valid_mask"]], dtype=torch.bool, device=device)
            owner = torch.zeros(1, dtype=torch.long, device=device)
            captures = {}
            phase = None
            hook_handles = []
            if ps.is_pipeline_last_stage():
                lm = language_model(raw)
                if type(lm.decoder.final_layernorm).__name__ != "IdentityOp":
                    raise ValueError("expected the native identity final norm and explicit HC contraction")

                def head_input(module, args, kwargs):
                    value = args[0] if args else kwargs.get(
                        "hidden_states", kwargs.get("input_", kwargs.get("input")))
                    if value is None or tuple(value.shape) != (options.tokens // 2, 1, 2560):
                        raise ValueError("actual LM head input is not [local_sequence,1,2560]")
                    if "head" in captures[phase]:
                        raise ValueError("head ran twice; internal recomputation is not allowed")
                    captures[phase]["head"] = value.detach().cpu()

                def head_output(module, args, output):
                    value = first_tensor(output, torch)
                    if value.ndim != 3 or value.shape[-1] != 124160:
                        raise ValueError("expected TP2 local-vocabulary logits with width 124160")
                    captures[phase]["logits"] = value.detach().cpu()

                hook_handles.extend((
                    lm.output_layer.register_forward_pre_hook(head_input, with_kwargs=True),
                    lm.output_layer.register_forward_hook(head_output),
                ))
            adapter = None
            native_capture = None
            delta = None
            state = None
            metrics_by_phase = {}
            schedule = get_forward_backward_func()
            if schedule.__name__ != "forward_backward_pipelining_without_interleaving":
                raise ValueError(f"unexpected pipeline schedule: {schedule.__name__}")

            def run_phase(name, *, backward, use_adapter):
                nonlocal phase, native_capture
                phase = name
                captures[name] = {}
                result = schedule_query(
                    models, batch, valid_mask=valid, owner_ids=owner,
                    adapter=adapter if use_adapter else None, reader_state=state,
                    backward=backward, group_total_target_tokens=case["loss_tokens"])
                native_capture = result["capture"]
                torch.cuda.synchronize()
                metrics_by_phase[name] = {key: result[key] for key in (
                    "loss_sum", "query_target_tokens", "group_total_target_tokens", "normalized_loss")}
                partial = {
                    "completed_phase": name, "overall_acceptance_complete": False,
                    "rank": rank, "tp_rank": ps.get_tensor_model_parallel_rank(),
                    "pp_rank": ps.get_pipeline_model_parallel_rank(),
                    "metrics": metrics_by_phase[name],
                }
                if "head" in captures[name]:
                    hidden = captures[name]["head"]
                    partial["head_input"] = head_summary(hidden, valid, partial["tp_rank"])
                    partial["head_input"]["scope"] = "synthetic runtime probe, not training-data calibration"
                    head_path = options.output / f"phase-{name}-rank-{rank}-H.pt"
                    torch.save(hidden, head_path)
                    partial["head_artifact"] = {"path": head_path.name, "sha256": digest(head_path)}
                write_json(options.output / f"phase-{name}-rank-{rank}.json", partial)
                dist.barrier()

            try:
                run_phase("original", backward=False, use_adapter=False)
                if ple is not None:
                    adapter = Qwen38NativePLEAdapter(ple, layout)
                    state = layout.empty_state(1, device=device)
                run_phase("empty", backward=False, use_adapter=True)
                local_checks = {}
                if ps.is_pipeline_last_stage():
                    for kind in ("logits", "head"):
                        original, empty = captures["original"][kind], captures["empty"][kind]
                        if original.shape != empty.shape or not torch.equal(original, empty):
                            difference = (original.float() - empty.float()).abs().max().item()
                            raise ValueError(f"empty {kind} differs from the native base: {difference}")
                    local_checks["empty_logits_bitwise_equal"] = True
                    local_checks["empty_head_bitwise_equal"] = True
                address_record = None
                if ple is not None:
                    if native_capture is None:
                        raise ValueError("missing native address capture")
                    ids, physical = native_capture.native_ids, native_capture.physical_rows
                    if not torch.equal(ids[0, :2], ids[0, 3:5]):
                        raise ValueError("native hash did not reset after EOS=248044")
                    if bool(physical[~valid].ne(0).any()):
                        raise ValueError("padding read positions are not mapped to physical row zero")
                    local_checks["native_eos_hash_reset"] = True
                    local_checks["padding_rows_zero"] = True
                    rows = torch.unique(physical[valid].reshape(-1), sorted=True)
                    rows = rows[rows > 0]
                    # Deterministic identical complete delta on both TP ranks;
                    # no RNG or original table values are changed.
                    grid = torch.arange(rows.numel() * 160, device=device, dtype=torch.float32)
                    delta = (grid.mul(.017).sin().reshape(-1, 160) * .01).requires_grad_()
                    state = BatchMemoryState.from_states([
                        SparseMemoryState(rows, delta, torch.ones(rows.numel(), device=device))
                    ], layout.row_capacity)
                    address_record = {
                        "native_ids": ids.cpu().tolist(), "physical_rows": physical.cpu().tolist(),
                        "base_head_rms": native_capture.base_head_rms.cpu().tolist(),
                        "lookup_backend": native_capture.lookup_backend,
                        "state_rows": rows.cpu().tolist(), "state_values_shape": list(delta.shape),
                        "state_values_rms": float(delta.detach().square().mean().sqrt()),
                    }
                run_phase("delta_backward", backward=True, use_adapter=True)
                gradient = None
                if delta is not None:
                    if delta.grad is None or not bool(delta.grad.isfinite().all()):
                        raise ValueError("native frozen-reader backward did not produce finite delta gradients")
                    summed = delta.grad.detach().clone()
                    dist.all_reduce(summed, op=dist.ReduceOp.SUM, group=ps.get_tensor_model_parallel_group())
                    gradient = {
                        "local": tensor_summary(delta.grad), "tp_sum": tensor_summary(summed),
                        "local_norm": float(delta.grad.norm()), "tp_sum_norm": float(summed.norm()),
                        "meaning": "raw replicated-delta contributions and TP sum; writer reduction/scaling not accepted",
                    }
                    if gradient["local_norm"] <= 0 or gradient["tp_sum_norm"] <= 0:
                        raise ValueError("zero delta gradient")
                    local_checks["finite_nonzero_delta_gradient"] = True
                tensor_artifact = options.output / f"rank-{rank}-captures.pt"
                torch.save({
                    "rank": rank,
                    "tp_rank": ps.get_tensor_model_parallel_rank(),
                    "pp_rank": ps.get_pipeline_model_parallel_rank(),
                    "head_input": {name: values["head"] for name, values in captures.items()
                                   if "head" in values},
                    "delta_values": delta.detach().cpu() if delta is not None else None,
                    "delta_grad": delta.grad.detach().cpu() if delta is not None else None,
                    "native_ids": native_capture.native_ids.cpu() if native_capture is not None else None,
                    "physical_rows": native_capture.physical_rows.cpu() if native_capture is not None else None,
                }, tensor_artifact)
                for name, parameter in named_parameters:
                    if (parameter.requires_grad or parameter.grad is not None
                            or (id(parameter), parameter._version) != before[name]):
                        raise ValueError(f"frozen reader parameter changed or received gradients: {name}")
                if ple is not None and (
                        id(ple.ple_embedding.host_table), ple.ple_embedding.host_table._version) != host_identity:
                    raise ValueError("native host table changed during the preflight")
                local_checks["reader_frozen_no_grad_unchanged"] = True
                head_summaries = {}
                for name, values in captures.items():
                    if "head" not in values:
                        continue
                    hidden = values["head"]
                    head_summaries[name] = head_summary(hidden, valid, ps.get_tensor_model_parallel_rank())
                report = {
                    "rank": rank, "tp_rank": ps.get_tensor_model_parallel_rank(),
                    "pp_rank": ps.get_pipeline_model_parallel_rank(), "checks": local_checks,
                    "reader_parameter_tensors": len(named_parameters),
                    "reader_parameter_elements": sum(p.numel() for _, p in named_parameters),
                    "groups": {
                        "tp": dist.get_process_group_ranks(ps.get_tensor_model_parallel_group()),
                        "pp": dist.get_process_group_ranks(ps.get_pipeline_model_parallel_group()),
                        "ep": dist.get_process_group_ranks(ps.get_expert_model_parallel_group()),
                        "dp": dist.get_process_group_ranks(ps.get_data_parallel_group()),
                        "matching_last_stage_rank": ps.get_pipeline_model_parallel_last_rank(),
                        "ple_owner": ple is not None,
                    },
                    "tensor_artifact": {"path": tensor_artifact.name, "sha256": digest(tensor_artifact)},
                    "native_addresses": address_record, "delta_gradient": gradient,
                    "head_input": head_summaries,
                    "logits": {name: tensor_summary(values["logits"]) for name, values in captures.items()
                               if "logits" in values},
                    "metrics": metrics_by_phase,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
                write_json(options.output / f"rank-{rank}.json", report)
                all_reports = [None] * dist.get_world_size()
                dist.all_gather_object(all_reports, report)
                if rank == 0:
                    first_stage = [r for r in all_reports if r["pp_rank"] == 0]
                    last_stage = [r for r in all_reports if r["pp_rank"] == 1]
                    if len(first_stage) != 2 or len(last_stage) != 2:
                        raise ValueError("incomplete distributed receipt")
                    if first_stage[0]["native_addresses"] != first_stage[1]["native_addresses"]:
                        raise ValueError("TP ranks disagree on native addresses or complete delta")
                    result = {
                        "protocol": "qwen38_native_runtime_preflight_20261020_v1",
                        "complete": True, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "elapsed_seconds": time.monotonic() - started,
                        "model": identity, "versions": versions, "runtime_sources": runtime_sources,
                        "synthetic_input": case,
                        "topology": {"tp": 2, "pp": 2, "ep": 2, "etp": 1, "cp": 1,
                                     "dp": 1, "sequence_parallel": True},
                        "schedule": schedule.__name__, "microbatches_per_pass": 1,
                        "forward_passes": 3, "backward_passes": 1,
                        "logical_forward_input_tokens": options.tokens * 3,
                        "logical_backward_loss_tokens": case["loss_tokens"],
                        "optimizer_constructed": False, "optimizer_steps": 0,
                        "training_or_dev_or_final_run": False, "reader_trainable_parameters": 0,
                        "empty_logits_bitwise_equal_all_tp": True,
                        "native_delta_gradient_nonzero_all_ple_tp": True,
                        "reader_no_grad_unchanged_all_ranks": True,
                        "ple_cpu_offload": True, "ple_eager": True,
                        "native_host_lookup": "existing torch gather/H2D fallback selected process-locally",
                        "loss_scalar": "sum of shifted synthetic CE / valid target tokens; one microbatch, CP1",
                        "logit_sharding": "each last-stage TP rank owns contiguous 124160 vocabulary logits; no logits average",
                        "internal_activation_recompute": False,
                        "last_H_site": "actual output_layer input after native HC contraction",
                        "head_reconstruction": "SP token slices by tp_rank; do not average hidden channels",
                        "limits": [
                            "No writer parameter, writer update, BPTT, TP writer reduction, or resume was tested.",
                            "Nonzero leaf-delta gradients do not establish a writer gradient scaling rule.",
                            "Tiny synthetic inputs do not establish full-length memory or throughput capacity.",
                        ],
                        "ranks": all_reports,
                    }
                    write_json(options.output / "COMPLETE.json", result)
                    print(json.dumps({"native_runtime_preflight": "passed",
                                      "receipt": str(options.output / "COMPLETE.json")}), flush=True)
                dist.barrier()
                return {"complete": True}
            finally:
                if adapter is not None:
                    adapter.close()
                if host_gather_module is not None:
                    host_gather_module.gather_ple_rows = host_gather_original
                for handle in hook_handles:
                    handle.remove()

    runner = NativePreflight(swift_arguments(options, synthetic_path))
    runner.main()
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--delta-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, choices=(16, 32), default=16)
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    for name in ("model_path", "verification", "delta_repo", "output"):
        setattr(options, name, getattr(options, name).resolve())
    if not options.execute:
        print(json.dumps({
            "mode": "plan_only", "model_repo": MODEL_REPO, "revision": MODEL_REVISION,
            "requires": "completed fixed-revision 131-shard verification and pinned Swift environment",
            "torchrun_ranks": 4, "tokens_per_pass": options.tokens, "forward_passes": 3,
            "backward_passes": 1, "optimizer_steps": 0,
            "output_must_be_new": str(options.output),
            "gpu_initialized": False, "model_loaded": False,
        }, indent=2))
        return
    if sys.platform != "linux":
        raise SystemExit("GPU runtime acceptance must run on the audited Linux host")
    execute(options)


if __name__ == "__main__":
    main()

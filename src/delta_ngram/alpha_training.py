"""Prefix-locked alpha experiments with independent optimizer/resume artifacts.

Build the manifest from M1's actual first-milestone checkpoint and train log.
The full M1 budget remains unchanged when replaying its sampler; only the
number of completed optimizer groups is limited by the manifest. No FINAL
data, validation tuning, or online optimizer is used here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch

from .alpha_lm import alpha_trajectory_loss
from .alpha_memory import make_alpha_policy
from .checkpoints import load_checkpoint, restore_writer, save_checkpoint
from .data.raw_text import RawWindowSampler
from .raw_corpora import load_corpora, trajectory_pools
from .raw_lm import RESUME_FORMAT as M1_RESUME_FORMAT, _identity, _json_hash as _raw_json_hash, _writer_metadata, tokenizer_identity
from .runtime import log_event, save_json, synchronize
from .train import create_runtime


PROTOCOL = "delta_ngram_alpha_lm_v1"
PREFIX_FORMAT = "delta_ngram_m1_optimizer_prefix_v1"
RESUME_FORMAT = "delta_ngram_alpha_resume_v1"
ALPHA_FORMAT = "delta_ngram_alpha_policy_v1"


def _json_hash(value):
    # Frozen YAML uses integer length keys (4/8/16/32); JSON reloads them as
    # strings. Normalize before sorting so 16 versus 4 never changes a digest.
    return _raw_json_hash(json.loads(json.dumps(value, allow_nan=False)))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def _execution_identity():
    source = Path(__file__).parent
    files = ("alpha_training.py", "alpha_memory.py", "alpha_lm.py", "raw_lm.py", "event_lm.py",
             "batching.py", "state.py", "operators.py", "writer.py", "reader.py",
             "ngram_lookup.py", "raw_corpora.py", "data/raw_text.py", "data/event_streams.py")
    return {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "threads": torch.get_num_threads(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "source_sha256": {name: file_sha256(source / name) for name in files}}


def _save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _setup(config):
    raw = config.values["raw"]
    if (config.values["method"]["name"] != "forward" or config.values["method"]["operator"] != "D1"
            or raw.get("resume")):
        raise ValueError("alpha requires the unchanged initial M1 D1 config, without raw.resume")
    # The final M1 config may name P1's step-zero writer in run.checkpoint.
    # Alpha always restores the explicitly supplied, identity-checked M1
    # writer_init export below; it must never silently restore that path here.
    method, args = create_runtime(config)
    tokens = tokenizer_identity(method)
    documents, events = load_corpora(raw, method.tokenizer, "sha256:" + _json_hash(tokens),
        vocabulary_size=method.reader.model.get_input_embeddings().weight.shape[0])
    options = {} if raw.get("source_length_weights") is None else {
        "source_length_weights": raw["source_length_weights"]}
    sampler = RawWindowSampler(trajectory_pools(raw, documents, events), seed=config.values["seed"],
        length_weights=raw["length_weights"], source_weights=raw["source_weights"], **options)
    identity = _identity(config, method, file_sha256(raw["dataset"]), tokens)
    return method, args, sampler, identity


def window_identity(window):
    result = {"document_id": window.document_id, "group_id": window.group_id,
        "source": window.source, "start": window.start, "blocks": window.blocks,
        "block_size": window.block_size, "loss_tokens": window.loss_tokens,
        "variable_cost": bool(getattr(window, "variable_cost", False))}
    if result["variable_cost"]:
        result["articles"] = [{"id": a.id, "tokens": len(a.token_ids),
                               "token_sha256": _json_hash(a.token_ids)} for a in window.articles]
    else:
        result["token_sha256"] = _json_hash(window.token_ids)
    return result


def _next(sampler, raw, consumed):
    return sampler.next_batch(raw["batch_size"] * raw["gradient_accumulation_steps"],
                              remaining_loss_tokens=raw["loss_tokens"] - consumed)


def _group(batch, step, consumed, sampler):
    windows = [window_identity(w) for w in batch]
    return {"step": step, "windows": windows, "batch_sha256": _json_hash(windows),
            "batch_loss_tokens": sum(w.loss_tokens for w in batch),
            "cumulative_loss_tokens": consumed + sum(w.loss_tokens for w in batch),
            "sampler_after_sha256": _json_hash(sampler.state_dict())}


def _check_init(config, method, writer_init, identity):
    payload = load_checkpoint(writer_init)
    if payload["step"] != 0 or payload["config"].get("training_loss_tokens", 0) != 0:
        raise ValueError("alpha needs M1's step-zero writer_init, not a trained writer")
    baseline = dict(payload["config"].get("raw_identity", {}))
    baseline.pop("validation", None)
    if _json_hash(baseline) != _json_hash(identity):
        raise ValueError("M1 writer_init and frozen configuration/corpus identity differ")
    if payload["config"]["package_config"]["raw"]["loss_tokens"] != config.values["raw"]["loss_tokens"]:
        raise ValueError("keep M1's full raw.loss_tokens budget; do not replace it with the prefix budget")
    restore_writer(method, payload, config)
    return payload


def build_m1_prefix(config, *, m1_checkpoint, m1_metrics, writer_init, output,
                    nominal_loss_tokens=5_000_000):
    """Verify M1's first crossing of the nominal budget, then freeze each batch."""
    if type(nominal_loss_tokens) is not int or nominal_loss_tokens < 1:
        raise ValueError("nominal_loss_tokens must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"prefix manifest already exists: {output}")
    method, _, sampler, identity = _setup(config)
    _check_init(config, method, writer_init, identity)
    baseline = _load(m1_checkpoint)
    old_identity = dict(baseline.get("identity", {}))
    old_identity.pop("validation", None)
    if baseline.get("format") != M1_RESUME_FORMAT or _json_hash(old_identity) != _json_hash(identity):
        raise ValueError("M1 checkpoint identity differs from frozen config/corpus")
    step = baseline.get("step")
    if type(step) is not int or step < 1:
        raise ValueError("M1 milestone checkpoint must have positive step")
    records = {}
    for line in Path(m1_metrics).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("event") == "train" and row.get("step", step + 1) <= step:
            if row["step"] in records:
                raise ValueError("M1 metrics contains duplicate train steps; supply one complete chronological log")
            records[row["step"]] = row
    if set(records) != set(range(1, step + 1)):
        raise ValueError("M1 metrics must cover every optimizer step from zero to the milestone")
    groups, consumed, previous = [], 0, 0
    for index in range(1, step + 1):
        batch = _next(sampler, config.values["raw"], consumed)
        if not batch:
            raise ValueError("M1 prefix exceeds complete trajectory budget")
        group = _group(batch, index, consumed, sampler)
        light = [{k: w[k] for k in ("document_id", "start", "blocks")} for w in group["windows"]]
        observed = records[index]
        if (observed.get("windows") != light or observed.get("batch_loss_tokens") != group["batch_loss_tokens"]
                or observed.get("counters", {}).get("loss_tokens") != group["cumulative_loss_tokens"]):
            raise ValueError(f"M1 sampler replay differs from actual logged step {index}")
        previous, consumed = consumed, group["cumulative_loss_tokens"]
        groups.append(group)
    if not previous < nominal_loss_tokens <= consumed:
        raise ValueError("checkpoint is not M1's first complete optimizer group crossing the nominal prefix")
    if (baseline.get("counters", {}).get("loss_tokens") != consumed
            or baseline.get("sampler_state") != sampler.state_dict()):
        raise ValueError("M1 checkpoint final counters/sampler disagree with the full replay")
    result = {"format": PREFIX_FORMAT, "config_sha256": file_sha256(config.source),
        "m1_identity": identity, "writer_init_sha256": file_sha256(writer_init),
        "m1_checkpoint_sha256": file_sha256(m1_checkpoint), "m1_metrics_sha256": file_sha256(m1_metrics),
        "full_loss_token_budget": config.values["raw"]["loss_tokens"],
        "nominal_loss_tokens": nominal_loss_tokens, "actual_loss_tokens": consumed, "stop_step": step,
        "groups": groups, "sampler_final": sampler.state_dict(),
        "baseline_counters": baseline["counters"],
        "rule": "M1 actual complete optimizer prefix, replaying its unchanged full token budget"}
    result["manifest_sha256"] = _json_hash(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)
    return result


def _prefix(config, path, writer_init, identity):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = manifest.pop("manifest_sha256", None)
    if manifest.get("format") != PREFIX_FORMAT or digest != _json_hash(manifest):
        raise ValueError("invalid prefix manifest format/hash")
    manifest["manifest_sha256"] = digest
    if (manifest["config_sha256"] != file_sha256(config.source)
            or manifest["writer_init_sha256"] != file_sha256(writer_init)
            or _json_hash(manifest["m1_identity"]) != _json_hash(identity)
            or manifest["full_loss_token_budget"] != config.values["raw"]["loss_tokens"]):
        raise ValueError("prefix manifest differs from frozen config, full budget, corpus, or M1 writer_init")
    if manifest["stop_step"] != len(manifest["groups"]):
        raise ValueError("prefix step count differs from group count")
    return manifest


def _resume_payload(method, alpha, optimizer, sampler, identity, step, counters):
    return {"format": RESUME_FORMAT, "identity": identity, "step": step, "counters": counters,
        "writer_state_dict": method.writer.state_dict(), "alpha_state_dict": alpha.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "sampler_state": sampler.state_dict(),
        "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if method.device.type == "cuda" else []}


def _restore(path, method, alpha, optimizer, sampler, identity, prefix):
    payload = _load(path)
    if payload.get("format") != RESUME_FORMAT or payload.get("identity") != identity:
        raise ValueError("alpha resume identity differs (variant, prefix, initial writer, or config)")
    step, counters = payload.get("step"), payload.get("counters")
    if type(step) is not int or not 0 <= step <= prefix["stop_step"] or not isinstance(counters, dict):
        raise ValueError("invalid alpha resume step/counters")
    if any(type(v) is not int or v < 0 for v in counters.values()):
        raise ValueError("invalid alpha resume token counters")
    sampler.load_state_dict(payload["sampler_state"])
    expected = 0 if step == 0 else prefix["groups"][step - 1]["cumulative_loss_tokens"]
    if counters.get("loss_tokens") != expected or sampler.total_loss_tokens != expected:
        raise ValueError("alpha resume loss tokens differ from the exact M1 prefix")
    if step and _json_hash(sampler.state_dict()) != prefix["groups"][step - 1]["sampler_after_sha256"]:
        raise ValueError("alpha resume sampler is not at the exact M1 prefix boundary")
    method.writer.load_state_dict(payload["writer_state_dict"], strict=True)
    alpha.load_state_dict(payload["alpha_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["python_rng"])
    if method.device.type == "cuda":
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return step, counters


def freeze_alpha_online(method, alpha):
    """Explicit online boundary: both learned modules become inference-only."""
    method.writer.eval().requires_grad_(False)
    alpha.eval().requires_grad_(False)


def train_alpha(config, *, variant, writer_init, prefix_manifest, output, resume=None,
                 stop_after_steps=None):
    """Train only the frozen M1 optimizer prefix with fresh AdamW plus alpha."""
    if stop_after_steps is not None and (type(stop_after_steps) is not int or stop_after_steps < 0):
        raise ValueError("stop_after_steps must be a nonnegative absolute optimizer step")
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"use an empty output directory: {out}")
    method, args, sampler, base_identity = _setup(config)
    _check_init(config, method, writer_init, base_identity)
    prefix = _prefix(config, prefix_manifest, writer_init, base_identity)
    alpha = make_alpha_policy(variant, method.reader.width).to(method.device)
    parameters = list(method.parameters()) + list(alpha.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.values["train"]["lr"], weight_decay=0.0)
    identity = {"protocol": PROTOCOL, "variant": variant, "alpha": alpha.metadata(),
        "config_sha256": prefix["config_sha256"], "prefix_sha256": prefix["manifest_sha256"],
        "writer_init_sha256": prefix["writer_init_sha256"], "raw_identity_sha256": _json_hash(base_identity),
        "execution": _execution_identity()}
    counters = dict.fromkeys(("loss_tokens", "raw_tokens_consumed", "writer_new_tokens", "encoder_input_tokens",
                             "reader_input_tokens", "address_prefix_tokens", "write_blocks", "trajectories"), 0)
    step = 0
    if resume:
        step, counters = _restore(resume, method, alpha, optimizer, sampler, identity, prefix)
    out.mkdir(parents=True, exist_ok=True)
    args.out = out
    metadata = _writer_metadata(config, method, args, base_identity)
    metadata.update(protocol=PROTOCOL, alpha_identity=identity, requires_alpha_payload=True,
                    alpha_payload="alpha.pt", m1_prefix_manifest_sha256=prefix["manifest_sha256"])
    save_json(out / "config.json", metadata)
    raw = config.values["raw"]
    limit = prefix["stop_step"] if stop_after_steps is None else min(stop_after_steps, prefix["stop_step"])
    started = time.monotonic()
    interval = raw["checkpoint_every_tokens"] or raw["eval_every_tokens"]
    next_checkpoint = (counters["loss_tokens"] // interval + 1) * interval
    _save(out / "training.pt", _resume_payload(method, alpha, optimizer, sampler, identity, step, counters))
    while step < limit:
        if time.monotonic() - started >= args.max_wall_seconds:
            break
        batch = _next(sampler, raw, counters["loss_tokens"])
        actual = _group(batch, step + 1, counters["loss_tokens"], sampler)
        if actual != prefix["groups"][step]:
            raise ValueError(f"alpha batch differs from frozen M1 prefix at step {step + 1}")
        method.train()
        alpha.train()
        optimizer.zero_grad(set_to_none=True)
        group_tokens, loss_sum = actual["batch_loss_tokens"], 0.0
        group_counters = dict.fromkeys(counters, 0)
        before = {name: p.detach().clone() for name, p in alpha.named_parameters()}
        for start in range(0, len(batch), raw["batch_size"]):
            def observe(row, start=start):
                log_event(out / "alpha_blocks.jsonl", {"event": "alpha_block", "step": step + 1,
                    "microbatch": start // raw["batch_size"], "seed": config.values["seed"],
                    "optimizer_group_loss_tokens_before": counters["loss_tokens"], **row})
            result = alpha_trajectory_loss(method, batch[start:start + raw["batch_size"]], alpha,
                prefix_tokens=raw["prefix_tokens"], checkpoint_reader=raw["checkpoint_reader"], alpha_observer=observe)
            loss = result.loss_sum / group_tokens
            if not loss.requires_grad or not bool(torch.isfinite(loss)):
                raise RuntimeError("alpha loss is disconnected or nonfinite")
            loss.backward()
            loss_sum += float(result.loss_sum.detach())
            for key, value in result.counters.items():
                group_counters[key] += value
            del result, loss
        if any(p.grad is not None for p in method.reader.model.parameters()):
            raise RuntimeError("frozen reader accumulated gradients")
        alpha_grad = {name: None if p.grad is None else float(p.grad.double().norm())
                      for name, p in alpha.named_parameters()}
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.values["train"]["gradient_clip"], error_if_nonfinite=True)
        if float(norm) == 0:
            raise RuntimeError("alpha/writer gradient is zero")
        optimizer.step()
        step += 1
        for key, value in group_counters.items():
            counters[key] += value
        if counters["loss_tokens"] != actual["cumulative_loss_tokens"]:
            raise RuntimeError("alpha trajectory target counts differ from M1 prefix")
        log_event(out / "metrics.jsonl", {"event": "train", "step": step, "counters": counters.copy(),
            "nll": loss_sum / group_tokens, "batch_sha256": actual["batch_sha256"],
            "batch_loss_tokens": group_tokens, "grad_norm_before_clip": float(norm),
            "alpha_gradient_l2_before_clip": alpha_grad,
            "alpha_parameter_update_l2": {name: float((p.detach() - before[name]).double().norm())
                                          for name, p in alpha.named_parameters()},
            "alpha_parameter_l2": {name: float(p.detach().double().norm()) for name, p in alpha.named_parameters()}})
        if counters["loss_tokens"] >= next_checkpoint:
            _save(out / "training.pt", _resume_payload(method, alpha, optimizer, sampler, identity, step, counters))
            next_checkpoint = (counters["loss_tokens"] // interval + 1) * interval
    complete = step == prefix["stop_step"]
    if complete and (sampler.state_dict() != prefix["sampler_final"] or counters != prefix["baseline_counters"]):
        raise RuntimeError("completed alpha run does not match M1 prefix counters and final sampler")
    _save(out / "training.pt", _resume_payload(method, alpha, optimizer, sampler, identity, step, counters))
    freeze_alpha_online(method, alpha)
    save_checkpoint(out / "writer.pt", method.writer, {**metadata, "training_loss_tokens": counters["loss_tokens"]},
                    step, protocol=PROTOCOL)
    _save(out / "alpha.pt", {"format": ALPHA_FORMAT, "identity": identity, "step": step,
        "policy": alpha.metadata(), "variant": variant, "writer_sha256": file_sha256(out / "writer.pt"),
        "state_dict": {name: value.detach().cpu() for name, value in alpha.state_dict().items()},
        "online_requires_frozen_writer_and_alpha": True})
    synchronize(method.device)
    summary = {"protocol": PROTOCOL, "variant": variant, "complete": complete, "completed_steps": step,
        "target_steps": prefix["stop_step"], "counters": counters, "prefix_sha256": prefix["manifest_sha256"],
        "nominal_prefix_loss_tokens": prefix["nominal_loss_tokens"], "actual_prefix_loss_tokens": prefix["actual_loss_tokens"],
        "unchanged_full_loss_token_budget": raw["loss_tokens"], "elapsed_seconds": time.monotonic() - started,
        "evaluation_performed": False, "writer_checkpoint": "writer.pt", "alpha_checkpoint": "alpha.pt",
        "resume_checkpoint": "training.pt", "online_parameters_frozen": True}
    save_json(out / "summary.json", summary)
    return summary

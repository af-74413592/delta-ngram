"""The only outer training loop; operator choice never creates another loop."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time

import torch

from .checkpoints import save_checkpoint
from .data import (COLORS, build_manifest, episode_count, episode_queries, legacy_fact_manifest,
                   manifest_sha256, sample_training_episode, validate_manifest, validation_episodes)
from .evaluate import evaluate
from .methods import make_method
from .reader import FrozenMemoryReader
from .runtime import (answer_batch, answer_loss, load_model_and_tokenizer, log_event,
                      mean, save_json, synchronize, timed)

PROTOCOL = "delta_ngram_pilot_v1"


def create_runtime(config):
    args = config.legacy_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.smoke:
        torch.set_num_threads(2)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer = load_model_and_tokenizer(args, device)
    spec = config.values["reader"]
    reader = FrozenMemoryReader(model, buckets=spec["buckets"], orders=tuple(spec["orders"]),
                                layer_index=spec["layer_index"], backend=spec["backend"])
    method = make_method(config, reader, tokenizer, device)
    return method, args


def make_manifest(config):
    data = config.values["data"]
    if data["manifest"]:
        manifest = json.loads(Path(data["manifest"]).read_text(encoding="utf-8"))
        validate_manifest(manifest, expected_task=data["task"], expected_eval_episodes=data["eval_episodes"],
                          expected_facts_per_episode=data["facts_per_episode"],
                          expected_support_examples=data["support_examples"])
        return manifest
    if data["task"] == "fact" and data["facts_per_episode"] == 2:
        return legacy_fact_manifest(config.values["seed"], data["eval_episodes"])
    return build_manifest(task=data["task"], seed=config.values["seed"],
                          eval_episodes=data["eval_episodes"], facts_per_episode=data["facts_per_episode"],
                          support_examples=data["support_examples"])


def initialize_output(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {args.out}")


def run_metadata(config, method, args, manifest):
    metadata = {**vars(args), "out": str(args.out), "device": str(method.device), "protocol": PROTOCOL,
                "package_config": config.values, "config_source": str(config.source),
                "interface": "external zero-base hash rows and fixed residual injection; NOT native pretrained N-gram",
                "orders": list(method.reader.orders), "evaluation_is_validation": True, "final_seven_tasks_used": False,
                "deadline_policy": "soft limit checked between complete training/evaluation episodes",
                "width": method.reader.width, "writer_parameters": sum(p.numel() for p in method.parameters()),
                "manifest_sha256": manifest_sha256(manifest),
                "model_dtype": str(next(method.reader.model.parameters()).dtype),
                "resolved_model_commit": getattr(method.reader.model.config, "_commit_hash", None),
                "model_name_or_path": getattr(method.reader.model.config, "_name_or_path", None),
                "torch_version": str(torch.__version__), "method": method.name,
                "query_loss_weights": {"write1_q1": 1/3, "write2_q2": 1/3, "write2_q1": 1/3}
                if config.values["data"]["facts_per_episode"] == 2 else "equal weight per scheduled query"}
    package_root = Path(__file__).parent
    metadata["source_sha256"] = {str(path.relative_to(package_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(package_root.rglob("*.py"))}
    return metadata


def train(config):
    started = time.monotonic()
    args = config.legacy_args()
    deadline = started + args.max_wall_seconds
    initialize_output(args)
    method, args = create_runtime(config)
    if not method.supports_outer_training:
        raise ValueError("outer train requires a learned forward writer; run eval for online baselines")
    optimizer = torch.optim.AdamW(method.parameters(), lr=args.lr, weight_decay=0.0)
    manifest = make_manifest(config)
    validation = validation_episodes(manifest)
    if not validation:
        raise ValueError("validation manifest must not be empty")
    metadata = run_metadata(config, method, args, manifest)
    save_json(args.out / "config.json", metadata)
    save_json(args.out / "data_manifest.json", manifest)
    log_event(args.out / "metrics.jsonl", {"event": "start", "config": metadata})
    # Keep the original pilot's exact prefix-boundary check and RNG ordering.
    for color in COLORS:
        answer_batch(method.tokenizer, validation[0].question(0), " " + color, method.device)
    save_checkpoint(args.out / "writer_init.pt", method.writer, metadata, 0)

    def validate(step):
        return evaluate(method, validation, args, step=step, deadline=deadline,
                        counterfactuals=config.values["evaluation"]["counterfactuals"],
                        row_diagnostics=config.values["evaluation"]["row_diagnostics"])

    last_evaluation = validate(0)
    training_rng = random.Random(manifest["training_rng_seed"])
    completed_steps = 0
    for step in range(1, args.steps + 1):
        if time.monotonic() >= deadline:
            break
        method.train()
        synchronize(method.device)
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses, write_times, query_times = [], [], []
        component_losses = {}
        for index in range(args.batch_episodes):
            episode = sample_training_episode(manifest, training_rng, f"train-{step:06d}-{index}")
            state = method.empty_state()
            query_losses = []
            for write_index in range(episode_count(episode)):
                result, duration = timed(lambda: method.write(state, episode.evidence(write_index)), method.device)
                state = result.state
                write_times.append(duration)
                for query in episode_queries(episode, write_index + 1):
                    loss, duration = timed(lambda: answer_loss(method, state, query.question, query.answer), method.device)
                    query_times.append(duration)
                    query_losses.append(loss)
                    component_losses.setdefault(f"write{write_index+1}_q{query.index+1}_nll", []).append(float(loss.detach()))
            # Left-associated addition matches the historical (first + new + old)/3.
            loss = query_losses[0]
            for value in query_losses[1:]:
                loss = loss + value
            loss = loss / len(query_losses)
            if not loss.requires_grad or not torch.isfinite(loss):
                raise RuntimeError("Answer loss is disconnected from writer or non-finite")
            (loss / args.batch_episodes).backward()
            losses.append([float(value.detach()) for value in query_losses])
            del loss, state, result, query_losses
        if any(parameter.grad is not None for parameter in method.reader.model.parameters()):
            raise RuntimeError("Frozen backbone accumulated gradients")
        grad_norm = torch.nn.utils.clip_grad_norm_(method.parameters(), config.values["train"]["gradient_clip"], error_if_nonfinite=True)
        if float(grad_norm) == 0:
            raise RuntimeError("Writer gradient is zero; inspect read/write address coverage")
        optimizer.step()
        synchronize(method.device)
        completed_steps = step
        log_event(args.out / "metrics.jsonl", {"event": "train", "step": step, "episodes": args.batch_episodes,
                  "answer_nll": (sum(sum(row) for row in losses) / (len(losses[0]) * len(losses))
                                 if len({len(row) for row in losses}) == 1 else mean([sum(row)/len(row) for row in losses])),
                  **{key: mean(values) for key, values in component_losses.items()},
                  "grad_norm_before_clip": float(grad_norm), "step_seconds": time.perf_counter()-step_started,
                  "write_seconds_mean": mean(write_times), "query_forward_seconds_mean": mean(query_times),
                  "elapsed_seconds": time.monotonic()-started,
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(method.device) if method.device.type == "cuda" else 0})
        if step % args.eval_every == 0 or step == args.steps:
            save_checkpoint(args.out / "writer.pt", method.writer, metadata, step, optimizer)
            if time.monotonic() < deadline:
                last_evaluation = validate(step)
    save_checkpoint(args.out / "writer.pt", method.writer, metadata, completed_steps, optimizer)
    if last_evaluation["step"] != completed_steps and time.monotonic() < deadline:
        last_evaluation = validate(completed_steps)
    summary = {"protocol": PROTOCOL, "completed_steps": completed_steps, "requested_steps": args.steps,
               "complete": completed_steps == args.steps and last_evaluation["step"] == completed_steps and last_evaluation["complete"],
               "wall_limit_reached": time.monotonic() >= deadline, "elapsed_seconds": time.monotonic()-started,
               "checkpoint": "writer.pt", "last_evaluation": last_evaluation,
               "evaluation_matches_checkpoint": last_evaluation["step"] == completed_steps,
               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(method.device) if method.device.type == "cuda" else 0,
               "scope": "CPU plumbing only" if args.smoke else "frozen-reader external-memory fact pilot; validation only"}
    save_json(args.out / "summary.json", summary)
    log_event(args.out / "metrics.jsonl", {"event": "complete", **summary})
    return summary

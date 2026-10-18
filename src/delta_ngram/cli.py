"""The installed ``delta-ngram`` command."""
from __future__ import annotations

import argparse
import json
import time

import torch

from . import __version__
from .checkpoints import load_checkpoint, restore_writer
from .config import ExperimentConfig
from .data import validation_episodes
from .evaluate import evaluate
from .runtime import save_json
from .train import create_runtime, initialize_output, make_manifest, run_metadata, train


def run_evaluation(config, *, diagnose=False):
    started = time.monotonic()
    args = config.legacy_args()
    initialize_output(args)
    method, args = create_runtime(config)
    checkpoint = config.values["run"]["checkpoint"]
    step = 0
    if checkpoint:
        payload = load_checkpoint(checkpoint)
        restore_writer(method, payload, config)
        step = payload["step"]
    elif method.supports_outer_training:
        raise ValueError("eval/diagnose for a learned writer requires run.checkpoint")
    if method.supports_outer_training:
        method.writer.requires_grad_(False)
    manifest = make_manifest(config)
    metadata = run_metadata(config, method, args, manifest)
    save_json(args.out / "config.json", metadata)
    save_json(args.out / "data_manifest.json", manifest)
    result = evaluate(method, validation_episodes(manifest), args, step=step,
                      deadline=started+args.max_wall_seconds,
                      counterfactuals=config.values["evaluation"]["counterfactuals"],
                      row_diagnostics=diagnose or config.values["evaluation"]["row_diagnostics"],
                      query_styles=tuple(config.values["evaluation"]["query_styles"]),
                      detailed_rows=config.values["evaluation"]["detailed_rows"])
    summary = {"protocol": "delta_ngram_evaluation_v1", "method": method.name,
               "checkpoint_step": step, "elapsed_seconds": time.monotonic()-started,
               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(method.device) if method.device.type == "cuda" else 0,
               **result}
    save_json(args.out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog="delta-ngram", description="Sparse memory for frozen language models")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("train", "Train a forward writer using the shared outer loop"),
                            ("eval", "Evaluate a writer or evidence-only baseline"),
                            ("diagnose", "Evaluate with counterfactuals and physical-row interventions")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--config", required=True, help="YAML configuration file")
    chat = commands.add_parser("chat", help="Query a writer interactively with optional evidence writes")
    chat.add_argument("--config", required=True, help="YAML configuration file")
    chat.add_argument("--checkpoint", required=True, help="Saved writer checkpoint")
    args = parser.parse_args(argv)
    config = ExperimentConfig.load(args.config)
    if args.command == "chat":
        from .chat import run_chat
        run_chat(config, args.checkpoint)
        return 0
    result = train(config) if args.command == "train" else run_evaluation(config, diagnose=args.command == "diagnose")
    print(json.dumps({"command": args.command, "output_dir": config.values["run"]["output_dir"],
                      "complete": result.get("complete", result.get("completed_steps") == result.get("requested_steps")),
                      "scores": result.get("scores", result.get("last_evaluation", {}).get("scores"))}, allow_nan=False))
    return 0 if result.get("complete", result.get("completed_steps") == result.get("requested_steps")) else 2


if __name__ == "__main__":
    raise SystemExit(main())

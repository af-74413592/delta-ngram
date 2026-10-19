"""Evaluate an explicit B->I writer transfer at the matching archived I budget."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate(args):
    if not args.research_transfer:
        raise ValueError("Cross-model evaluation requires --research-transfer")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch
    from delta_ngram.config import ExperimentConfig
    from delta_ngram.data.raw_text import load_documents
    from delta_ngram.p2_transfer import (PROTOCOL, evaluate_transfer, file_sha256,
                                        fixed_validation_windows, load_research_transfer_checkpoint)
    from delta_ngram.train import create_runtime

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Commit source changes before a recorded P2 transfer evaluation")
    config = ExperimentConfig.load(args.destination_config)
    source_run, destination_run = Path(args.source_run).resolve(), Path(args.destination_run).resolve()
    source_config = read_json(source_run / "config.json")
    destination_config = read_json(destination_run / "config.json")
    source_identity, destination_identity = source_config["raw_identity"], destination_config["raw_identity"]
    dataset = Path(config.values["raw"]["dataset"])
    dataset_hash = file_sha256(dataset)
    if dataset_hash != source_identity["dataset_sha256"] or dataset_hash != destination_identity["dataset_sha256"]:
        raise ValueError("P2 dataset bytes differ from the source/destination training corpus")
    source_checkpoint = Path(args.source_checkpoint)
    destination_checkpoint = Path(args.destination_checkpoint)
    if not source_checkpoint.is_absolute():
        source_checkpoint = source_run / source_checkpoint
    if not destination_checkpoint.is_absolute():
        destination_checkpoint = destination_run / destination_checkpoint
    baseline_path = Path(args.baseline_evaluation)
    if not baseline_path.is_absolute():
        baseline_path = destination_run / baseline_path
    baseline = read_json(baseline_path)
    manifest_path = destination_run / "data_manifest.json"
    manifest = read_json(manifest_path)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    input_files = {"destination_config": Path(args.destination_config).resolve(),
                   "source_run_config": source_run / "config.json",
                   "destination_run_config": destination_run / "config.json",
                   "source_checkpoint": source_checkpoint.resolve(),
                   "destination_checkpoint": destination_checkpoint.resolve(),
                   "baseline_evaluation": baseline_path.resolve(), "validation_manifest": manifest_path}
    launch = {"protocol": PROTOCOL, "created_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_git_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
              "inputs": {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in input_files.items()},
              "dataset_sha256": dataset_hash, "threads": args.threads, "deterministic_algorithms": True,
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "source_sha256": {str(path.relative_to(repo)): file_sha256(path)
                                for path in sorted((repo / "src/delta_ngram").rglob("*.py"))},
              "evaluation_source_sha256": file_sha256(__file__), "research_transfer": True,
              "evidence_encoder": "destination", "readout": "destination"}
    save_json(output / "launch.json", launch)
    started = time.monotonic()
    print(json.dumps({"event": "launch", "output": str(output), "condition": "B_to_I"}), flush=True)
    try:
        method, _ = create_runtime(config)
        metadata = load_research_transfer_checkpoint(
            method, config, source_checkpoint, destination_checkpoint,
            expected_source_identity=source_identity, expected_destination_identity=destination_identity,
            research_transfer=True)
        for role in ("source", "destination"):
            if metadata[role]["checkpoint_sha256"] != launch["inputs"][f"{role}_checkpoint"]["sha256"]:
                raise ValueError(f"P2 {role} checkpoint changed after launch provenance was recorded")
        documents = load_documents(dataset, method.tokenizer,
                                   tokenizer_identity=manifest["tokenizer_fingerprint"], require_record_identity=True)
        windows = fixed_validation_windows(documents, manifest, expected_identity=destination_identity,
                                           block_size=config.values["raw"]["block_size"])
        vocabulary_size = method.reader.model.get_input_embeddings().weight.shape[0]
        if any(max(window.token_ids) >= vocabulary_size for window in windows):
            raise ValueError("P2 target token IDs exceed destination vocabulary")
        print(json.dumps({"event": "compatibility_passed", "step": metadata["source"]["step"],
                          "training_loss_tokens": metadata["source"]["training_loss_tokens"],
                          "windows": len(windows)}), flush=True)
        result = evaluate_transfer(method, windows, metadata=metadata, baseline_evaluation=baseline,
                                   bootstrap_seed=config.values["raw"]["eval_seed"],
                                   bootstrap_samples=config.values["raw"]["eval_bootstrap_samples"])
        result["run_provenance"] = launch
        result["elapsed_seconds"] = time.monotonic() - started
        if method.device.type == "cuda":
            result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(method.device)
        save_json(output / "results.json", result)
        print(json.dumps({"event": "finished", "results": str(output / "results.json"),
                          "G": result["comparison"]["G"],
                          "decision": result["comparison"]["decision"]}, allow_nan=False), flush=True)
    except BaseException as error:
        save_json(output / "failure.json", {"complete": False, "type": type(error).__name__,
                                           "message": str(error), "elapsed_seconds": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination-config", required=True, help="Original I training YAML; never rewrite its model identity")
    parser.add_argument("--source-run", required=True, help="Base training run, containing config.json")
    parser.add_argument("--source-checkpoint", default="writer.pt", help="Absolute path or path relative to source run")
    parser.add_argument("--destination-run", required=True, help="Original I run, containing config.json and data_manifest.json")
    parser.add_argument("--destination-checkpoint", default="writer.pt", help="Absolute path or path relative to destination run")
    parser.add_argument("--baseline-evaluation", required=True, help="Matching I eval_step_*.json, relative to destination run or absolute")
    parser.add_argument("--research-transfer", action="store_true", help="Explicitly enable validated research-only writer transfer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if not args.research_transfer:
        parser.error("--research-transfer is required for cross-model evaluation")
    evaluate(args)


if __name__ == "__main__":
    main()

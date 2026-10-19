"""Evaluate a complete prefix-locked writer/alpha pair without any optimizer.

The full suite runs the M1 fixed historical sample, DEV continuation with all
registered distances/directions/pruning thresholds, and annotated revision
candidate queries. Outputs contain applied per-block alpha traces. Partial
suites are explicitly labeled and never stand in for the full experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))


def git_identity():
    return {"commit": subprocess.check_output(
        ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(
            ["git", "-C", str(REPOSITORY), "status", "--porcelain"], text=True).strip())}


def verify_prefix(config, runtime, path):
    from delta_ngram.alpha_evaluation import _canonical, file_sha256
    from delta_ngram.raw_lm import _json_hash
    path = Path(path).resolve()
    digest = file_sha256(path)
    prefix = json.loads(path.read_bytes())
    declared = prefix.pop("manifest_sha256", None)
    if (prefix.get("format") != "delta_ngram_m1_optimizer_prefix_v1"
            or declared != _json_hash(_canonical(prefix))):
        raise ValueError("invalid frozen M1 optimizer prefix format/hash")
    identity = runtime.manifest["identity"]
    if (declared != identity.get("prefix_sha256")
            or prefix["config_sha256"] != identity.get("config_sha256")
            or prefix["writer_init_sha256"] != identity.get("writer_init_sha256")
            or _json_hash(_canonical(prefix["m1_identity"])) != identity.get("raw_identity_sha256")
            or _canonical(prefix["m1_identity"]) != _canonical(runtime.manifest["raw_identity"])
            or prefix["full_loss_token_budget"] != config.values["raw"]["loss_tokens"]):
        raise ValueError("alpha checkpoint differs from the exact M1 initialization/configuration/prefix")
    if (runtime.manifest["step"] != prefix["stop_step"]
            or runtime.manifest["training_loss_tokens"] != prefix["actual_loss_tokens"]
            or prefix["stop_step"] != len(prefix["groups"])):
        raise ValueError("alpha checkpoint has not completed the exact full M1 prefix")
    if not config.values["model"]["smoke"] and prefix["nominal_loss_tokens"] != 5_000_000:
        raise ValueError("registered alpha evaluation requires the nominal first 5M M1 prefix")
    return {"path": str(path), "file_sha256": digest, "manifest_sha256": declared,
        "m1_checkpoint_sha256": prefix["m1_checkpoint_sha256"],
        "writer_init_sha256": prefix["writer_init_sha256"], "stop_step": prefix["stop_step"],
        "nominal_loss_tokens": prefix["nominal_loss_tokens"], "actual_loss_tokens": prefix["actual_loss_tokens"],
        "full_loss_token_budget": prefix["full_loss_token_budget"], "exact_complete_prefix_verified": True}


def evaluate_checkpoint(config, checkpoint_path, alpha_path, *, prefix_manifest, suite="all",
        m1_validation_manifest=None, registry_path=None, registry_sha256=None,
        annotations_path=None, annotations_sha256=None, bootstrap_seed=20261019,
        bootstrap_samples=10000, require_clean_git=True, progress_callback=None):
    import torch
    import delta_ngram.alpha_evaluation as evaluation
    import delta_ngram.recent_news_evaluation as recent
    from delta_ngram.raw_lm import _json_hash

    if suite not in ("all", "historical", "continuation", "revision"):
        raise ValueError("invalid evaluation suite")
    if type(bootstrap_samples) is not int or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    values = config.values
    smoke = values["model"]["smoke"]
    if (values["method"]["name"] != "forward" or values["method"]["operator"] != "D1"
            or values["reader"]["backend"] != "external_hash"):
        raise ValueError("alpha evaluation requires the frozen forward D1 external_hash configuration")
    if not smoke and not re.fullmatch(r"[a-f0-9]{40}", values["model"]["revision"] or ""):
        raise ValueError("non-smoke evaluation requires an immutable model commit")
    historical = suite in ("all", "historical")
    news = suite in ("all", "continuation", "revision")
    if historical and (m1_validation_manifest is None or not values["raw"].get("eval_donor")):
        raise ValueError("historical alpha evaluation requires M1 validation manifest and configured fixed donors")
    if news:
        if registry_path is None or not re.fullmatch(r"[a-f0-9]{64}", registry_sha256 or ""):
            raise ValueError("DEV registry and its explicit file SHA256 are required")
        if annotations_path is None or not re.fullmatch(r"[a-f0-9]{64}", annotations_sha256 or ""):
            raise ValueError("frozen DEV annotations and their explicit file SHA256 are required")
        if values["raw"]["block_size"] != 512 or values["raw"]["prefix_tokens"] != 128:
            raise ValueError("news evaluation requires the registered 512-token blocks and 128-token prefix")
    recorded = require_clean_git and not smoke
    git_before = git_identity()
    if recorded and git_before["dirty"]:
        raise RuntimeError("Commit repository changes before a recorded alpha evaluation")
    started = time.monotonic()
    runtime = evaluation.load_joint_alpha(config, checkpoint_path, alpha_path)
    method = runtime.method
    guard = evaluation.FrozenGuard(method, runtime.alpha)
    prefix = verify_prefix(config, runtime, prefix_manifest)
    files = [(Path(prefix["path"]), prefix["file_sha256"])]
    results, traces = {}, []

    def observe(row):
        traces.append({"sequence": len(traces), **row})

    if historical:
        from evaluate_raw_checkpoint import prepare_validation
        raw = values["raw"]
        data = Path(raw["dataset"]).resolve()
        data_hash = evaluation.file_sha256(data)
        identity = runtime.manifest["raw_identity"]
        if data_hash != identity["dataset_sha256"]:
            raise ValueError("historical corpus differs from frozen M1/alpha training identity")
        files.append((data, data_hash))
        if raw.get("event_dataset"):
            events = Path(raw["event_dataset"]).resolve()
            event_hash = evaluation.file_sha256(events)
            if event_hash != identity["raw"]["event_dataset_sha256"]:
                raise ValueError("historical event corpus differs from frozen M1/alpha identity")
            files.append((events, event_hash))
        reference_path = Path(m1_validation_manifest).resolve()
        reference_hash = evaluation.file_sha256(reference_path)
        reference = json.loads(reference_path.read_bytes())
        files.append((reference_path, reference_hash))
        if reference.get("dataset_sha256") != data_hash:
            raise ValueError("M1 validation manifest has a different corpus")
        windows, validation, donors = prepare_validation(config, method, include_donors=True)
        for key in ("validation_protocol", "validation_protocol_sha256", "validation_sample", "validation_sample_sha256"):
            if evaluation._canonical(reference.get(key)) != evaluation._canonical(validation[key]):
                raise ValueError(f"historical sample/protocol differs from M1: {key}")
        del reference  # The original manifest also contains a large document inventory.
        if progress_callback is not None:
            progress_callback({"stage": "historical_prepared", "windows": len(windows),
                "validation_sample_sha256": validation["validation_sample_sha256"]})
        backend = evaluation.AlphaEvaluationBackend(runtime.alpha, alpha_observer=observe)
        result = evaluation.evaluate_alpha_historical(method, windows, backend,
            prefix_tokens=raw["prefix_tokens"], batch_size=raw["batch_size"],
            bootstrap_seed=raw["eval_seed"], bootstrap_samples=raw["eval_bootstrap_samples"],
            report_per_source=True, report_donor=True, macro_source_groups=raw.get("macro_source_groups"),
            fixed_donors=donors)
        results["historical"] = {**result, **validation,
            "m1_validation_manifest": {"path": str(reference_path), "sha256": reference_hash},
            "m1_validation_sample_and_protocol_exact_match": True,
            "pruning_measured": False, "pruning_location": "continuation self/donor query snapshots"}
    if news:
        prepared = recent.prepare_continuation(registry_path, method,
            tokenizer_source=values["model"].get("tokenizer_name_or_path") or values["model"]["name_or_path"],
            tokenizer_revision=values["model"].get("tokenizer_revision") or values["model"]["revision"],
            expected_sha256=registry_sha256)
        labels = recent.prepare_prefix_annotations(annotations_path, prepared, method, expected_sha256=annotations_sha256)
        if suite in ("all", "continuation"):
            backend = evaluation.AlphaEvaluationBackend(runtime.alpha, pruning=True, alpha_observer=observe)
            results["continuation"] = recent.evaluate_continuation(method, prepared,
                bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples,
                prefix_annotations=labels, progress_callback=progress_callback, memory_backend=backend)
        if suite in ("all", "revision"):
            backend = evaluation.AlphaEvaluationBackend(runtime.alpha, alpha_observer=observe)
            results["revision"] = evaluation.evaluate_revision(method, prepared, labels, backend,
                bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples,
                progress_callback=progress_callback)
        prepared.verify_files()
        labels.verify_file()
    guard.verify()
    runtime.verify_files()
    for path, digest in files:
        if evaluation.file_sha256(path) != digest:
            raise RuntimeError(f"alpha evaluation input changed: {path}")
    git_after = git_identity()
    if recorded and (git_after["dirty"] or git_after["commit"] != git_before["commit"]):
        raise RuntimeError("repository changed during recorded alpha evaluation")
    result = {"protocol": evaluation.PROTOCOL, "complete": True, "suite": suite,
        "all_registered_suites_executed": suite == "all", "evaluation_only": True,
        "optimizer_steps": 0, "backward_calls": 0, "checkpoint": runtime.manifest,
        "m1_prefix": prefix, "results": results, "alpha_trace": traces,
        "alpha_trace_sha256": _json_hash(traces),
        "alpha_trace_scope": "Actual applied block policy; owner clocks advance for every active write segment, including donor article-boundary segments.",
        "integrity_checks": {"joint_payload_and_config_files_unchanged": True,
            "writer_and_alpha_state_dict_exact_equal": True, "parameters_frozen_no_gradients": True,
            "reader_parameter_objects_and_versions_unchanged": True, "reader_weight_content_hash_verified": False},
        "run": {"created_utc": datetime.now(timezone.utc).isoformat(), "config": values,
            "config_path": str(config.source), "config_sha256": evaluation.file_sha256(config.source),
            "resolved_config_sha256": _json_hash(evaluation._canonical(values)),
            "git_before": git_before, "git_after": git_after, "clean_git_enforced": recorded,
            "script_sha256": evaluation.file_sha256(__file__),
            "evaluator_sha256": evaluation.file_sha256(evaluation.__file__),
            "torch_version": str(torch.__version__), "device": str(method.device),
            "threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "elapsed_seconds": time.monotonic() - started},
        "limitations": ["DEV results only; no FINAL access or checkpoint selection by final answers.",
            "Pruning measures read snapshots and actual archive size, not live GPU memory reclamation.",
            "Revision candidate ranking is separate from original-news continuation and free generation.",
            "Reader parameter version checks do not detect every .data or buffer mutation."]}
    json.dumps(result, allow_nan=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Complete prefix alpha writer.pt")
    parser.add_argument("--alpha", required=True, help="Paired alpha.pt")
    parser.add_argument("--prefix-manifest", required=True)
    parser.add_argument("--suite", choices=("all", "historical", "continuation", "revision"), default="all")
    parser.add_argument("--m1-validation-manifest", help="Original M1 data_manifest.json; required for historical/all")
    parser.add_argument("--registry", help="Frozen DEV continuation registry; required for news suites")
    parser.add_argument("--registry-sha256")
    parser.add_argument("--annotations", help="Frozen DEV prefix/revision annotations")
    parser.add_argument("--annotations-sha256")
    parser.add_argument("--bootstrap-seed", type=int, default=20261019)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", required=True, help="New immutable JSON output")
    args = parser.parse_args(argv)
    if args.threads < 1 or args.bootstrap_samples < 1:
        parser.error("threads and bootstrap-samples must be positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Keep prior evaluation results immutable: {output}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import torch
    from delta_ngram.config import ExperimentConfig
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    result = evaluate_checkpoint(ExperimentConfig.load(args.config), args.checkpoint, args.alpha,
        prefix_manifest=args.prefix_manifest, suite=args.suite, m1_validation_manifest=args.m1_validation_manifest,
        registry_path=args.registry, registry_sha256=args.registry_sha256,
        annotations_path=args.annotations, annotations_sha256=args.annotations_sha256,
        bootstrap_seed=args.bootstrap_seed, bootstrap_samples=args.bootstrap_samples,
        progress_callback=lambda row: print(json.dumps(row), flush=True))
    result["run"]["requested_threads"] = args.threads
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "complete": True, "suite": args.suite,
        "variant": result["checkpoint"]["variant"], "alpha_trace_sha256": result["alpha_trace_sha256"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a preliminary/final M1 reproducibility ZIP without weights or PDFs.

Keep the repository and run directories as siblings so the CPU analysis scripts
retain working defaults after extraction. Historical numerical analysis is
self-contained. Capacity and weight-parameter audits have explicit external
dependencies; this package does not claim to reproduce every check by itself.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
RUNS = WORKSPACE / "delta-ngram-runs"
OPS = RUNS / "multidomain_launch_20261019"
REPORT = RUNS / "m1_analysis_20261020"
FROZEN_REPO = OPS / "m1_frozen_source_20261020/delta-ngram"
FROZEN_COMMIT = "167d24783161be7842bbb9aec52f2c7804191ba9"
FIXED_ZIP_TIME = (2026, 10, 20, 0, 0, 0)
ALPHAS = ("fixed_099", "fixed_095", "learned_global_099", "evidence_block_099")
FORBIDDEN_SUFFIXES = {".pdf", ".pt", ".pth", ".safetensors", ".bin", ".zip", ".gz", ".tar"}
SOURCE_FILES = (
    "raw_evaluation.py", "raw_lm.py", "event_lm.py", "batching.py", "state.py",
    "writer.py", "operators.py", "ngram_lookup.py", "block_monitor.py",
    "recent_news_evaluation.py", "direction_control.py",
)
SCRIPT_FILES = (
    "analyze_m1_results_20261020.py", "diagnose_m1_endpoint_20261020.py",
    "audit_m1_capacity_20261020.py", "build_m1_analysis_report_20261020.py",
    "package_m1_analysis_20261020.py", "evaluate_recent_news_continuation.py",
    "verify_m1_analysis_portable_20261020.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


VERIFY_SCRIPT = '''"""Verify every payload file against package_manifest.json; standard library only."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "package_manifest.json").read_bytes())
for item in manifest["members"]:
    path = (root / item["member"]).resolve()
    assert path.is_relative_to(root), "unsafe manifest member"
    assert path.is_file(), item["member"]
    assert path.stat().st_size == item["bytes"], item["member"]
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    assert actual == item["sha256"], item["member"]
print("Verified", len(manifest["members"]), "payload files;", manifest["stage"])
'''


def readme(stage, completed):
    return f"""M1 analysis reproducibility package - 2026-10-20
Package stage: {stage.upper()}
Completed new DEV64 result checkpoints included: {len(completed)} / 21

START HERE
  delta-ngram-runs/m1_analysis_20261020/M1_analysis_20261020.md
The Markdown report is the main deliverable. Figures are embedded by relative
paths. No PDF, model weights, tokenizer/model cache, source article corpus,
sealed FINAL dataset, or unfinished DEV64 result is included.

CHECKPOINT AND EVALUATION SCOPE
  100M endpoint: primary.
  65M originally selected by historical donor gap: secondary.
  New DEV64 curve: 0M plus 5M, 10M, ... 100M, 21 checkpoints.
  Other curve points are diagnostic; they cannot relabel the primary checkpoint.
  FINAL is not run here and is restricted to the locked 100M weights once.
  Original policy and later curve addenda are preserved as separate records.
  The old first-5M baseline stopped at 40/64 and has no complete score.
Only completed DEV64 result JSONs pass the result inclusion gate. Queue status,
registrations and inventories are execution metadata, not result scores.
Completion is determined by the package stage and included result count above;
an incomplete snapshot must not be described as a complete 21-point result.

DIRECTORY LAYOUT
  delta-ngram/scripts/                  CPU analysis and plotting source
  delta-ngram/src/delta_ngram/           statistics helper and inspection source
  delta-ngram/docs/                     checkpoint policy and curve addenda
  delta-ngram-runs/m1_analysis_20261020/ Markdown, figures, CSV/JSON, diagnostics
  delta-ngram-runs/multidomain_launch_20261019/
    final_backup_20261020/              21 historical evaluations + summary
    alpha_evaluations/                 4 historical alpha evaluations + metadata
    diagnostic_logs/metrics.jsonl      complete 4173-step training metrics
    dev64_curve_results_20261020/      completed DEV64 results if available

FROZEN EVALUATOR SOURCE
The src/delta_ngram files and evaluate_recent_news_continuation.py come from
the scored evaluator commit {FROZEN_COMMIT}, with exact Git-blob bytes.
They are placed under delta-ngram/ so the included analysis scripts retain
their normal relative imports. The separate source snapshot manifest records
their SHA-256 hashes and original local snapshot locations:
  delta-ngram-runs/multidomain_launch_20261019/m1_frozen_source_manifest_20261020.json
Later native-writer implementation changes are outside this M1 analysis bundle.
Analysis, plotting, transfer and packaging scripts are the versions used to
prepare this report; their individual hashes are in package_manifest.json.

VERIFY THE EXTRACTED PAYLOAD
Requires Python 3.11 or newer; the original numerical run used Python 3.12.
Open a terminal at the extracted package root, then:
  python verify_bundle.py
package_manifest.json lists size and SHA-256 for every payload member, including
this README and verify_bundle.py. The manifest cannot hash itself; the ZIP SHA
sidecar covers the archive as a whole. Existing source manifests retain their
original absolute provenance paths. Use package_manifest.json to verify the
relocated package, rather than treating those old absolute paths as dependencies.

RECOMPUTE MAIN NUMERICAL RESULTS (SELF-CONTAINED, CPU, STANDARD LIBRARY)
Run from the extracted package root:
  python delta-ngram/scripts/analyze_m1_results_20261020.py
  python delta-ngram/scripts/diagnose_m1_endpoint_20261020.py
The defaults resolve the sibling delta-ngram-runs directory after extraction.
These commands recompute the historical 21-point table, four same-prefix alpha
comparisons, paired bootstrap intervals, source and first-block decompositions,
training-token-weighted intervals, and writer-risk mathematical audit. They do
not execute models or read weights. They overwrite derived output tables in the
extracted copy; raw input evaluations remain unchanged. Numerical results should
match; path-containing manifest fields naturally change after relocation.
The original random seed is 20261020 with 10000 paired bootstrap draws.

Equivalent commands on the original machine, from the repository directory:
  python scripts/analyze_m1_results_20261020.py
  python scripts/diagnose_m1_endpoint_20261020.py

REMAKE THE FIGURES (OPTIONAL)
Pre-rendered figures are included. Re-rendering needs matplotlib, numpy and a
Chinese-capable TTF/TTC font supplied by the local machine:
  python delta-ngram/scripts/build_m1_analysis_report_20261020.py --font "<font path>"
On the original Windows machine the font is C:/Windows/Fonts/msyh.ttc.
Fonts are not redistributed. Rendering software/font differences can change PNG
bytes without changing the numerical results. The builder does not make PDFs.

CAPACITY AUDIT: TWO EXTERNAL RAW INPUTS REQUIRED
The completed capacity outputs and CPU audit script are included, but its large
raw block log and full data manifest are intentionally external. Exact original
paths, byte sizes, SHA-256 and required destinations are recorded under
external_inputs in package_manifest.json:
  1. diagnostic_logs/block_metrics.jsonl.gz (about 86 MiB compressed)
  2. final_backup_20261020/data_manifest.json (about 128 MiB)
After providing and hash-checking those files, run:
  python delta-ngram/scripts/audit_m1_capacity_20261020.py --log "<block log .jsonl.gz>" --data-manifest "<data_manifest.json>"
The included original config, historical evaluations and training summary meet
the other defaults. This command verifies the uncompressed block-log hash too.
Without the two raw inputs, capacity results can be read but cannot be fully
independently recomputed from this ZIP alone.

PARAMETER-DISPLACEMENT AUDIT: NOT SELF-CONTAINED
diagnostics/writer_objective_audit.py and its saved JSON are retained for evidence.
That original script imports torch, uses CPU weight loading, and contains original
machine paths. Re-running it needs the omitted 65M and 100M writer/training
weights, PyTorch, and deliberate path adjustment. Their identities are listed as
external inputs. Do not run it as part of the standard-library reproduction.
The included mathematical_checks_summary.json is an editorial synthesis of
separate audits, not an additional independently observed measurement.

NEW DEV64 RESULTS
Any complete results included here are preserved observations, not freshly run
inference. This package does not contain the model and source-article archive
needed to repeat those forward passes. Partial progress is not a score.
The completed curve statistics can be recomputed on CPU with numpy:
  python delta-ngram/scripts/analyze_m1_dev64_curve_20261020.py --no-figures
Omit --no-figures to recreate the three DEV64 plots (matplotlib required).
The address-capacity figure can be redrawn from the included CSV alone:
  python delta-ngram/scripts/build_m1_dev64_capacity_figure_20261020.py
This verifies checkpoint and data identities, per-family target and write
budgets, exact empty-arm equality, and the saved bootstrap means/intervals.
DEV64 averages the 64 event families equally; it is not the five-category
historical macro average. The raw bootstrap seed is 20261019; added paired
checkpoint comparisons use 20261020, both with 10000 draws.

The DEV64 address-capacity audit is a separate CPU replay:
  python delta-ngram/scripts/audit_m1_dev64_capacity_20261020.py --result "<complete DEV64 JSON>" --registry "<frozen DEV64 registry JSON>"
It requires the omitted frozen source-article text files, matching local
tokenizer, torch and transformers. No model weights or GPU are needed.
Its saved per-family counts and source hashes are included; the ZIP alone
does not reproduce the tokenizer/address replay.

REBUILD THE PACKAGE
On the original machine:
  python scripts/package_m1_analysis_20261020.py
After all 21 new DEV64 results are complete and the Markdown is updated:
  python scripts/package_m1_analysis_20261020.py --stage final
The builder rereads updated Markdown, figures, tables, policies and completed
DEV64 results; it never includes its own ZIP, PDFs or weights. Final mode requires
exactly the 21 expected historical checkpoint steps as complete DEV64 outputs.
It performs no training, GPU inference, FINAL evaluation or git commit.

STATISTICAL LIMITS
One training seed; fixed validation families and donors. Added intervals are
exploratory, pointwise, and unadjusted for selecting checkpoints or multiple
comparisons. Historical held-out performance is not a pretraining-novelty or QA
accuracy claim. Absence of observed overall overfitting does not guarantee every
future update has positive benefit.
"""


def collect_sources():
    files = {}
    def add(path, role, required=True, archive_relative=None):
        path = path.resolve()
        if not path.is_file():
            require(not required, f"missing package input: {path}")
            return
        require(path.is_relative_to(WORKSPACE), f"input outside workspace: {path}")
        require(path.suffix.lower() not in FORBIDDEN_SUFFIXES, f"forbidden payload: {path}")
        require(not any("final_holdout" in part.lower() for part in path.parts),
                "sealed FINAL paths are forbidden")
        name = archive_relative or path.relative_to(WORKSPACE).as_posix()
        require((WORKSPACE / name).resolve().is_relative_to(WORKSPACE)
                and "\\" not in name and not Path(name).is_absolute(),
                "unsafe archive destination")
        if name in files:
            return
        files[name] = {"path": path, "role": role}

    frozen_manifest = json.loads((OPS / "m1_frozen_source_manifest_20261020.json").read_bytes())
    require(frozen_manifest["git_commit"] == FROZEN_COMMIT, "Wrong frozen evaluator commit")
    frozen_files = {item["source_relative_path"]: item for item in frozen_manifest["frozen_files"]}
    def add_frozen(relative):
        path = FROZEN_REPO / relative
        require(relative in frozen_files and path.is_file()
                and digest(path) == frozen_files[relative]["sha256"],
                "Frozen evaluator source differs: " + relative)
        add(path, "frozen_m1_evaluator_source", archive_relative="delta-ngram/" + relative)

    add(REPORT / "M1_analysis_20261020.md", "main_markdown_report")
    for filename in ("training_curve.png", "endpoint_sources.png",
                     "alpha_paired_contrasts.png", "mathematical_checks.png"):
        add(REPORT / "figures" / filename, "figure")
    for folder, suffixes in (
        ("data", {".csv", ".json", ".txt"}),
        ("dev64_data", {".csv", ".json", ".txt", ".md"}),
        ("diagnostics", {".csv", ".json", ".txt", ".py"}),
        ("figures", {".png", ".svg"}),
    ):
        for path in sorted((REPORT / folder).rglob("*")):
            if (path.is_file() and path.suffix.lower() in suffixes
                    and "__pycache__" not in path.parts
                    and "preview" not in path.parts
                    and not path.name.startswith("dev64_preview_")):
                add(path, "derived_" + folder)
    for filename in SCRIPT_FILES:
        if filename == "evaluate_recent_news_continuation.py":
            add_frozen("scripts/" + filename)
        else:
            add(REPO / "scripts" / filename, "analysis_source")
    for path in sorted((REPO / "scripts").glob("*m1*dev64*.py")):
        add(path, "dev64_analysis_or_execution_source")
    for filename in SOURCE_FILES:
        add_frozen("src/delta_ngram/" + filename)
    add(REPO / "configs/m1_event_mixture_seed17.yaml", "training_config")
    for path in sorted((REPO / "docs").glob("m1*20261020*.md")):
        add(path, "policy_document")
    historical = sorted((OPS / "final_backup_20261020").glob("eval_step_*.json"))
    require(len(historical) == 21, "expected 21 historical evaluation files")
    steps = []
    for path in historical:
        steps.append(json.loads(path.read_bytes())["step"])
        add(path, "raw_historical_evaluation")
    add(OPS / "final_backup_20261020/summary.json", "raw_training_summary")
    add(OPS / "first_m1_checkpoint/config.json", "original_capacity_config")
    for variant in ALPHAS:
        add(OPS / "alpha_evaluations" / f"{variant}_historical.json", "raw_alpha_historical_evaluation")
        add(OPS / "alpha_evaluations" / f"{variant}_training_summary.json", "alpha_training_summary")
    for name in ("m1_first_5m_prefix.json", "parallel_user_override_20261019.json",
                 "user_stop_small_alpha_20261019.json"):
        add(OPS / "alpha_evaluations" / name, "alpha_prefix_or_execution_policy")
    add(OPS / "diagnostic_logs/metrics.jsonl", "raw_training_metrics")
    add(OPS / "run_m1_dev64_curve_20261020.py", "frozen_curve_execution_source")
    add(OPS / "evaluation_cuda_entry.py", "frozen_evaluation_entry_source", required=False)
    add(OPS / "dev64_curve_monitor_20261020.jsonl", "read_only_gpu_and_progress_monitor", required=False)
    for pattern in ("m1_*_20261020.json", "*registration*.json"):
        for path in sorted(OPS.glob(pattern)):
            add(path, "registration_inventory_or_status")

    completed, skipped = [], []
    result_dir = OPS / "dev64_curve_results_20261020"
    if result_dir.is_dir():
        for path in sorted(result_dir.rglob("*.json")):
            data = json.loads(path.read_bytes())
            if not isinstance(data, dict):
                skipped.append({"file": str(path), "reason": "not a metadata/result object"})
                continue
            is_result = ("per_family" in data or
                         data.get("protocol") == "delta_ngram_recent_news_continuation_eval_v1")
            if is_result:
                if not (data.get("complete") is True and data.get("split") == "dev"
                        and data.get("evaluation_only") is True
                        and data.get("independent_families") == 64
                        and len(data.get("per_family", [])) == 320
                        and data.get("optimizer_steps") == 0 and data.get("backward_calls") == 0):
                    skipped.append({"file": str(path), "reason": "not a complete read-only DEV64 result"})
                    continue
                checkpoint = data["checkpoint"]
                require(checkpoint["step"] in steps, "unexpected DEV64 checkpoint step")
                completed.append({"file": path.relative_to(WORKSPACE).as_posix(),
                                  "step": checkpoint["step"],
                                  "checkpoint_sha256": checkpoint["sha256"]})
                add(path, "complete_dev64_evaluation")
            elif any(word in path.name.lower() for word in (
                "registration", "inventory", "policy", "status", "verification", "manifest",
                "runtime", "completion")):
                add(path, "dev64_execution_metadata")
            else:
                skipped.append({"file": str(path), "reason": "not a recognized complete result or metadata"})
    return files, sorted(steps), completed, skipped


def external_inputs():
    definitions = [
        (OPS / "diagnostic_logs/block_metrics.jsonl.gz", "capacity_raw_block_log",
         "Required to recompute capacity observations; excluded for size."),
        (OPS / "final_backup_20261020/data_manifest.json", "capacity_full_data_manifest",
         "Required to recompute frozen validation length bounds; excluded for size."),
        (OPS / "selected_65m_backup/writer_step_002713_tokens_000065009696.pt", "selected_writer_weights",
         "Required only for the parameter-displacement audit; weights intentionally excluded."),
        (OPS / "selected_65m_backup/training_step_002713_tokens_000065009696.pt", "selected_training_weights",
         "Required only for the parameter-displacement audit; weights intentionally excluded."),
        (OPS / "final_backup_20261020/writer.pt", "endpoint_writer_weights",
         "Required only for the parameter-displacement audit; weights intentionally excluded."),
        (OPS / "final_backup_20261020/training.pt", "endpoint_training_weights",
         "Required only for the parameter-displacement audit; weights intentionally excluded."),
    ]
    result = []
    for path, role, reason in definitions:
        require(path.is_file(), f"external identity cannot be recorded: {path}")
        result.append({
            "original_path": str(path.resolve()), "role": role, "bytes": path.stat().st_size,
            "sha256": digest(path), "reason_not_bundled": reason,
            "expected_relative_path_if_supplied": path.relative_to(WORKSPACE).as_posix(),
        })
    return result


def zip_info(name):
    item = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    item.compress_type = zipfile.ZIP_DEFLATED
    item.external_attr = 0o100644 << 16
    return item


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preliminary", "final"), default="preliminary")
    parser.add_argument("--output", type=Path,
                        default=REPORT / "m1_analysis_bundle_20261020.zip")
    args = parser.parse_args(argv)
    files, expected_steps, completed, skipped = collect_sources()
    complete_steps = sorted(row["step"] for row in completed)
    require(len(complete_steps) == len(set(complete_steps)), "duplicate completed DEV64 checkpoint results")
    if args.stage == "final":
        require(complete_steps == expected_steps, "final package requires all 21 complete DEV64 checkpoints")
        curve_path = REPORT / "dev64_data/dev64_summary.json"
        require(curve_path.is_file(), "final package requires the validated DEV64 curve analysis")
        curve = json.loads(curve_path.read_bytes())
        require(curve.get("complete_curve") is True and curve.get("preview_only") is False
                and curve.get("completed_checkpoints") == 21
                and curve.get("checkpoint_selection_performed") is False,
                "final package requires complete diagnostics with unchanged checkpoint roles")
        require(curve.get("validation") and all(curve["validation"].values()),
                "DEV64 analysis validation failed")
        capacity = json.loads((REPORT / "dev64_data/capacity_scope_note.json").read_bytes())
        require(capacity["verification"]["all_21_checkpoints_verified"] is True
                and capacity["verification"]["equivalent_completed_checkpoints_verified"] == 21,
                "final package requires capacity identity coverage for all 21 checkpoints")
        source_manifest = json.loads((REPORT / "dev64_data/dev64_source_manifest.json").read_bytes())
        for item in source_manifest["files"]:
            source_path = Path(item["path"])
            require(source_path.is_file() and source_path.stat().st_size == item["bytes"]
                    and digest(source_path) == item["sha256"],
                    f"DEV64 analysis input changed after validation: {source_path}")
        report_text = (REPORT / "M1_analysis_20261020.md").read_text(encoding="utf-8")
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", report_text):
            if link.startswith(("https://", "http://", "#")):
                continue
            linked_file = (REPORT / link.split("#", 1)[0].strip("<>")).resolve()
            require(linked_file.is_relative_to(WORKSPACE),
                    f"Report link is outside the portable package: {link}")
            member_name = linked_file.relative_to(WORKSPACE).as_posix()
            require(member_name in files, f"Report link is missing from the final package: {link}")
    output = args.output.resolve()
    require(output.suffix == ".zip" and output.parent == REPORT.resolve(),
            "package output must be a ZIP directly in the intended analysis directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    external = external_inputs()
    members = []
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, item in sorted(files.items()):
            path = item["path"]
            before = path.stat()
            content = path.read_bytes()
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    f"input changed during packaging: {path}")
            archive.writestr(zip_info(name), content)
            members.append({
                "member": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                "role": item["role"], "original_path": str(path),
            })
        for name, content, role in (
            ("README.txt", readme(args.stage, completed).encode("utf-8"), "reproduction_instructions"),
            ("verify_bundle.py", VERIFY_SCRIPT.encode("utf-8"), "payload_verification_script"),
        ):
            archive.writestr(zip_info(name), content)
            members.append({"member": name, "bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(), "role": role})
        manifest = {
            "protocol": "m1_reproducibility_bundle_20261020_v1", "stage": args.stage,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "expected_dev64_checkpoints": 21, "completed_dev64_checkpoints_included": len(completed),
            "expected_dev64_steps": expected_steps, "complete_dev64_results": completed,
            "skipped_dev64_files": skipped, "primary": "100M", "secondary": "65M",
            "other_dev64_checkpoints": "diagnostic only",
            "members": sorted(members, key=lambda item: item["member"]),
            "external_inputs": external,
            "self_hash_note": "The manifest excludes itself to avoid recursive hashing. "
            "All other archive members are hashed here; the external ZIP SHA-256 covers the entire archive.",
            "reproduction_scope": {
                "historical_main_statistics": "self-contained; Python standard library",
                "endpoint_and_training_loss_diagnosis": "self-contained; Python standard library",
                "figures": "data included; matplotlib, numpy and an external Chinese font required",
                "capacity": "requires the two declared external raw inputs",
                "new_dev64_statistics": "self-contained; numpy required, matplotlib optional",
                "new_dev64_address_capacity": "requires omitted frozen source texts and tokenizer; CPU torch/transformers",
                "writer_parameter_displacement": "requires omitted weights, PyTorch and original path adjustment",
                "new_dev64_inference": "not self-contained; source articles and model weights intentionally omitted",
            },
            "contains_pdf": False, "contains_model_weights": False,
            "sealed_final_accessed": False, "python_version": sys.version,
        }
        manifest_content = json_bytes(manifest)
        archive.writestr(zip_info("package_manifest.json"), manifest_content)
    with zipfile.ZipFile(temporary) as archive:
        require(archive.testzip() is None, "ZIP CRC verification failed")
        require(set(archive.namelist()) == {row["member"] for row in members} | {"package_manifest.json"},
                "unexpected ZIP member")
        for row in members:
            content = archive.read(row["member"])
            require(len(content) == row["bytes"] and hashlib.sha256(content).hexdigest() == row["sha256"],
                    f"archived payload hash mismatch: {row['member']}")
    temporary.replace(output)
    (output.parent / "package_manifest.json").write_bytes(manifest_content)
    archive_sha256 = digest(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{archive_sha256}  {output.name}\n", encoding="ascii", newline="\n")
    print(json.dumps({
        "output": str(output), "stage": args.stage, "payload_files": len(members),
        "archive_bytes": output.stat().st_size, "archive_sha256": archive_sha256,
        "completed_dev64_checkpoints": len(completed), "external_inputs": len(external),
        "payload_and_crc_verification": "passed", "published": False,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

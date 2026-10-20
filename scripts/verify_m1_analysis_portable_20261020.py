"""Extract the final M1 bundle and reproduce its CPU numerical results."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
REPORT_RELATIVE = Path("delta-ngram-runs/m1_analysis_20261020")
REPORT = WORKSPACE / REPORT_RELATIVE


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize(value, workspace):
    if isinstance(value, str):
        return value.replace(str(workspace), "<WORKSPACE>").replace(
            workspace.as_posix(), "<WORKSPACE>")
    if isinstance(value, list):
        return [normalize(item, workspace) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item, workspace) for key, item in value.items()}
    return value


def equivalent(left, right, location):
    if isinstance(left, bool) or isinstance(right, bool):
        require(type(left) is type(right) and left == right, location)
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        require(math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), location)
    elif isinstance(left, dict) and isinstance(right, dict):
        require(left.keys() == right.keys(), location + ": keys differ")
        for key in left:
            equivalent(left[key], right[key], location + "." + str(key))
    elif isinstance(left, list) and isinstance(right, list):
        require(len(left) == len(right), location + ": lengths differ")
        for index, (a, b) in enumerate(zip(left, right)):
            equivalent(a, b, location + "[" + str(index) + "]")
    else:
        require(left == right, location + ": values differ")


def csv_rows(path, workspace):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                row[key] = normalize(value, workspace)
    return rows


def numerical_summary(path, workspace):
    value = normalize(json.loads(path.read_bytes()), workspace)
    if path.name == "dev64_summary.json":
        for key in ("analysis_created_at_utc", "figures"):
            value.pop(key, None)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=REPORT / "m1_analysis_bundle_20261020.zip")
    args = parser.parse_args(argv)
    archive_path = args.archive.resolve()
    require(archive_path.is_file(), "Final bundle is missing")
    archive_sha = sha(archive_path)
    target = Path(tempfile.mkdtemp(prefix="bundle_validation_final_", dir=REPORT)).resolve()
    require(target.parent == REPORT.resolve(), "Validation directory escaped report folder")
    with zipfile.ZipFile(archive_path) as archive:
        require(archive.testzip() is None, "ZIP CRC failed")
        entries = archive.infolist()
        require(len({item.filename for item in entries}) == len(entries),
                "Duplicate archive entries")
        for item in entries:
            name = PurePosixPath(item.filename)
            destination = (target / item.filename).resolve()
            require(not name.is_absolute() and ".." not in name.parts
                    and "\\" not in item.filename and destination.is_relative_to(target)
                    and not item.is_dir() and (item.external_attr >> 16) & 0o170000 != 0o120000,
                    "Unsafe archive entry: " + item.filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(archive.read(item))
    manifest = json.loads((target / "package_manifest.json").read_bytes())
    require(manifest["stage"] == "final"
            and manifest["completed_dev64_checkpoints_included"] == 21,
            "Portable check requires a complete final 21-checkpoint bundle")
    extracted_report = target / REPORT_RELATIVE
    csv_files = [
        path for folder in ("data", "diagnostics", "dev64_data")
        for path in sorted((extracted_report / folder).glob("*.csv"))
    ]
    expected_csv = {path.relative_to(extracted_report).as_posix(): csv_rows(path, WORKSPACE)
                    for path in csv_files}
    csv_initial_mtimes = {path.relative_to(extracted_report).as_posix(): path.stat().st_mtime_ns
                          for path in csv_files}
    summary_names = ("data/report_summary.json", "diagnostics/mathematical_audit.json",
                     "dev64_data/dev64_summary.json")
    expected_json = {name: numerical_summary(extracted_report / name, WORKSPACE)
                     for name in summary_names}
    commands = [
        ["verify_bundle.py"],
        ["delta-ngram/scripts/analyze_m1_results_20261020.py"],
        ["delta-ngram/scripts/diagnose_m1_endpoint_20261020.py"],
        ["delta-ngram/scripts/analyze_m1_dev64_curve_20261020.py", "--no-figures"],
    ]
    executions = []
    for command in commands:
        result = subprocess.run([sys.executable, *command], cwd=target, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=300)
        executions.append({"command": command, "returncode": result.returncode,
                           "stdout": result.stdout, "stderr": result.stderr})
        require(result.returncode == 0, "Portable command failed: " + " ".join(command)
                + "\n" + result.stderr[-4000:])
    csv_cells, recomputed_csv, preserved_csv = 0, [], []
    for name, expected in expected_csv.items():
        csv_path = extracted_report / name
        actual = csv_rows(csv_path, target)
        # Capacity and editorial CSV files are preserved observations, not
        # outputs of the three commands above. Normalize their original paths.
        actual = normalize(actual, WORKSPACE)
        equivalent(expected, actual, name)
        csv_cells += sum(len(row) for row in actual)
        (recomputed_csv if csv_path.stat().st_mtime_ns != csv_initial_mtimes[name]
         else preserved_csv).append(name)
    for name, expected in expected_json.items():
        equivalent(expected, numerical_summary(extracted_report / name, target), name)
    raw_count = 0
    for item in manifest["members"]:
        if item["role"].startswith("raw_") or item["role"] == "complete_dev64_evaluation":
            require(sha(target / item["member"]) == item["sha256"],
                    "Raw input changed during numerical reproduction")
            raw_count += 1
    require(sha(archive_path) == archive_sha, "Archive changed during portable verification")
    receipt = {
        "protocol": "m1_final_portable_reproduction_20261020_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive_path), "archive_sha256": archive_sha,
        "stage": "final", "relocated_to": str(target), "verification": "passed",
        "verified_payload_files": len(manifest["members"]),
        "completed_dev64_checkpoints": 21, "compared_csv_files": len(expected_csv),
        "compared_csv_cells": csv_cells, "recomputed_csv_files": recomputed_csv,
        "preserved_csv_files": preserved_csv, "compared_summary_json": list(summary_names),
        "unchanged_raw_inputs": raw_count, "numerical_tolerance": 1e-12,
        "excluded_from_comparison": ["relocated absolute path prefix",
                                     "DEV64 analysis timestamp", "DEV64 plot paths (--no-figures)"],
        "model_or_gpu_used": False, "pdf_generated": False,
        "external_capacity_or_weight_audits_rerun": False, "commands": executions,
    }
    receipt_path = REPORT / "bundle_reproduction_check_final.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False)
                            + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "commands"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

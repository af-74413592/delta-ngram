"""Create a new incremental result archive; reuse previously archived inputs."""
from pathlib import Path
import hashlib
import json
import zipfile
from datetime import datetime, timezone

BASE = Path("C:/Users/shijingzi/Documents/ChatGPT")
REPO = BASE / "delta-ngram"
OPS = BASE / "delta-ngram-runs/multidomain_launch_20261019"
RAW = OPS / "dev64_context_results_20261020"
ANALYSIS = OPS / "dev64_context_analysis_20261020"
REPORT = REPO / "docs/reports/m1_20261020"
ARCHIVE = OPS / "m1_dev64_context_addon_20261020.zip"
CORE = "b2://clclcl/cl-research/delta-ngram/20261020-rtx4090/"
files = {}


def include(path, name):
    assert path.is_file() and not path.is_symlink(), path
    assert name not in files and not name.startswith("/") and ".." not in Path(name).parts
    files[name] = path


for name in ("context.json", "preflight.json", "context.json.empty_sanity.json", "deployment_receipt.json"):
    include(RAW / name, "raw/" + name)
for path in ANALYSIS.iterdir():
    if path.is_file():
        include(path, "analysis/" + path.name)
for name in ("evaluate_dev64_full_context_20261020.py", "analyze_dev64_context_recovery_20261020.py",
             "build_dev64_context_figures_20261020.py"):
    include(REPO / "scripts" / name, "delta-ngram/scripts/" + name)
for name in ("test_dev64_full_context_20261020.py", "test_dev64_context_recovery_analysis.py"):
    include(REPO / "tests" / name, "delta-ngram/tests/" + name)
include(REPO / "docs/m1_dev64_context_registration_20261020.md",
        "delta-ngram/docs/m1_dev64_context_registration_20261020.md")
report_path = "delta-ngram/docs/reports/m1_20261020/"
include(REPORT / "context_reference.md", report_path + "context_reference.md")
include(REPORT / "archive_manifest.json", report_path + "archive_manifest.json")
for path in (REPORT / "context_data").iterdir():
    if path.is_file():
        include(path, report_path + "context_data/" + path.name)
for ext in ("png", "svg"):
    include(REPORT / "figures" / ("dev64_context_reference_100m." + ext),
            report_path + "figures/dev64_context_reference_100m." + ext)
include(Path(__file__), "provenance/package_dev64_context_addon_20261020.py")
include(OPS / "deploy_dev64_context_20261020.py", "provenance/deploy_dev64_context_20261020.py")
for name in ("download_verification.json", "download_manifest.json"):
    include(RAW / name, "provenance/" + name)
include(RAW / "logs/delta_ngram_dev64_context_20261020.log", "provenance/runtime.log")
for name in ("supervisor_wrapper.sh", "supervisor_config.conf"):
    include(RAW / "execution" / name, "provenance/" + name)
members = [
    {"path": name, "bytes": path.stat().st_size,
     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for name, path in sorted(files.items())
]
manifest = {
    "protocol": "dev64_context_incremental_archive_20261020_v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "payload_count": len(members), "payload_bytes": sum(r["bytes"] for r in members),
    "members": members,
    "existing_inputs_not_duplicated": {
        "original_21_results": CORE + "runs/m1_dev64_curve_20261020/",
        "checkpoint_inventory": CORE + "operations/multidomain/m1_dev64_curve_inventory_20261020.json",
        "complete_core_manifest": CORE + "metadata/member_manifest.jsonl",
        "original_reader_and_tokenizer": CORE + "model_dependencies/Qwen--Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca/",
    },
    "scope": "New raw-weight DEV64 context evaluation and its CPU analyses only; no training or FINAL.",
}
readme = """# DEV64 original-model full-context add-on

This archive adds one original-weight context evaluation to the previously
verified core archive. It contains no duplicated old checkpoints or old 21-point
evaluation files. The external inputs are listed in manifest.json.

## CPU reproduction

Extract to a new directory and verify every manifest member before use. With the
previously verified core available as $CORE, run from this archive's root:

```bash
python delta-ngram/scripts/analyze_dev64_context_recovery_20261020.py \\
  --context raw/context.json --preflight raw/preflight.json \\
  --curve-dir "$CORE/runs/m1_dev64_curve_20261020" \\
  --inventory "$CORE/operations/multidomain/m1_dev64_curve_inventory_20261020.json" \\
  --output-dir recomputed
python delta-ngram/scripts/build_dev64_context_figures_20261020.py \\
  --analysis-dir recomputed --output-dir recomputed_figures
```

Use Python 3.11+, NumPy, and Matplotlib for plotting. The statistical analysis
does not load the model, source articles or writer weights. Training and FINAL
are not launched by these commands. The original 21-result inputs are also
available inside the previously verified complete analysis ZIP.

The report's links to other reports belong to the Git repository. See the
current GitHub report index for those documents and the final cloud receipts.
The immutable Git code delta is stored alongside this archive in B2.
"""
assert not ARCHIVE.exists(), ARCHIVE
with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for name, path in sorted(files.items()):
        archive.write(path, name)
    archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    archive.writestr("README.md", readme)
with zipfile.ZipFile(ARCHIVE) as archive:
    assert archive.testzip() is None
    for member in members:
        data = archive.read(member["path"])
        assert len(data) == member["bytes"]
        assert hashlib.sha256(data).hexdigest() == member["sha256"]
receipt = {
    "archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size,
    "sha256": hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
    "payload_count": len(members), "archive_members": len(members) + 2,
    "all_payload_hashes_verified": True,
}
receipt_path = OPS / "dev64_context_addon_package_receipt_20261020.json"
with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))

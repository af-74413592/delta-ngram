# Historical M1 and DEV64 launch sources

These are byte-for-byte archival copies of the deployed M1 and alpha launch
sources, plus the completed 21-checkpoint DEV64 queue. They are evidence of the
historical run and are not a new execution plan.

The original operations directory was
`/workspace/delta-ngram-transfer-20261019/multidomain`. Each copied Python file
was checked against a read-only SHA256 query of that remote directory on
2026-10-20. The DEV64 queue source additionally matches the runner SHA256 recorded
at queue startup. A current remote-file digest alone does not prove that every
helper was unchanged at every earlier moment of execution.

## Contents

- `run_m1.py`, `install_m1.py`, `preflight_m1.py`: M1 preparation, training,
  deployment and the recorded preflight.
- `run_m1_dev64_curve_20261020.py`, `evaluation_cuda_entry.py`: completed frozen
  21-checkpoint DEV64 queue and its allocator wrapper.
- `run_alpha_parallel.py`, `run_alpha_parallel_resume.py`, `alpha_cuda_entry.py`,
  `install_alpha_parallel.py`, `install_alpha_parallel_resume.py`: historical
  four-arm alpha training orchestration. Archival inclusion does not restart
  the cancelled supplemental evaluation queue.
- `remote_source_verification.json`: dated read-only remote hash receipt.
- `provenance_manifest.json`: SHA256, sizes, local origins, remote paths and
  evidence for the archived sources and registration records.

The scripts retain their original absolute paths and side effects. In particular,
the installer files can start supervisor jobs when executed or imported.
Inspect these sources as historical records; do not execute or import them as
part of restoring the code archive. No archived training, installer, GPU
preflight, model inference or FINAL evaluation was run during this archival step.

The superseded two-checkpoint DEV64 launcher is omitted; the completed
21-checkpoint launcher above is the executed final scope. The unexecuted
`scripts/evaluate_d1_revision.py` preparation and independent Qwen3.8 prototype
remain outside this historical launch archive.

## Registration and verification

Exact policy, inventory, registration and selected runtime records are in
[the report registration directory](../../../docs/reports/m1_20261020/registration/README.md).
The executed evaluator source remains identified by commit
`167d24783161be7842bbb9aec52f2c7804191ba9`; no source import or checkout is
implicitly substituted by this archive.

From the repository root, verify all archived payloads without executing them:

```bash
python scripts/archive/m1_20261020/verify_archive.py
```

The manifest excludes itself and its SHA256 sidecar to avoid recursive hashing.
The sidecar covers the manifest. Directory-local `.gitattributes` files disable
line-ending conversion so Git can preserve the exact original bytes.

Large evaluation outputs, model and writer weights, source corpora, credentials,
and caches are not copied here. The 2.7 MB first-5M prefix manifest is identified
as an omitted input and remains in the already verified analysis bundle.

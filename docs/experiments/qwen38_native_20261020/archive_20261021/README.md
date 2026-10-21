# Qwen3.8 native experiment archive — 2026-10-21

This archive preserves the completed mainline evaluation and the startup evidence for the ongoing memory-encoding fork. It is a snapshot, not the final result of the fork.

## Results and controls

- [Completed 100M evaluation](qwen38_native_100m_evaluation_report_20261021.md): alpha=1 completed 99,997,072 training targets and 4,172 updates; alpha=.99 remains at its independently saved 39,998,107-target / 1,669-update control. The report retains both 20M and 40M equal-budget comparisons.
- [Memory encoding diagnostics and fork startup](qwen38_memory_forward_diagnostic_report_20261021.md): full starts from the 79,981,959-target checkpoint, replays 835 groups, inherits AdamW at 3e-4, and schedules two independent terminal DEV64 evaluations. The residual run was stopped by the user after 77 groups and is excluded from endpoint method comparisons. The repeat-only NLL control was still queued at this snapshot.
- [Mechanism and stability discussion](memory_aware_writer_stability_response_20261021.md) and its [earlier derivation](memory_aware_writer_stability_20261021.md) preserve the development record; the diagnostic report contains the subsequent corrections and measured execution results.
- [File provenance](provenance.json) records copied-file hashes and the 89-file match against the active frozen training manifest. Markdown links alone were made portable.

## Data and reproduction

The `data/` directory contains the seven-node DEV64 per-family extract, saved statistical analysis, integrity receipt, CPU analysis/rendering scripts, and small immutable diagnostic/startup bundles. The four ZIPs total less than 1 MB and are deliberately curated evidence. Unpack each ZIP into `data/<zip-stem>/` to recover the original receipt tree. Selected linked JSON receipts are also expanded for browsing.

Run `python data/analyze_qwen38_native_100m_20261021.py` with NumPy to recompute the saved bootstrap statistics. The render scripts additionally require Matplotlib. They write outputs beside their archived inputs; use a disposable copy when reproducing the original receipt hashes. These are CPU result-processing tools, not GPU jobs.

The [archived launch recipes](../../../../scripts/archive/qwen38_memory_20261021/README.md) preserve the executed full supervisor and queued repeat control. The runtime manifests are authoritative for exact source identity; Git can normalize line endings.

## Verified cloud backup

The completed native experiment snapshot is stored in bucket `clclcl`, prefix `cl-research/delta-ngram/20261021-qwen38-native/`. **All 2,469 files / 4,760,098,462 bytes were fully downloaded to a separate directory on the remote host and matched by file set, size and SHA256.** The independent seal check additionally verified all 2,466 payload files against the original preregistered staging manifest and its three metadata hashes.

- [Final transfer status](cloud_backup/status.json): payload complete at 07:12:33 UTC; six verification artifacts were uploaded and read back successfully by 07:12:44 UTC.
- [Independent staging-seal receipt](cloud_backup/seal_verification.json): passed at 07:16:13 UTC, including exact file set and safe paths.
- [Checkpoint identity](cloud_backup/checkpoint_identity.json): confirms the 100M mainline and 40M alpha=.99 optimizer/target identities, full continuation state, and file hashes.
- [Local receipt-package verification](cloud_backup/local_receipt_verification.json): checks the downloaded evidence ZIP, 27 manifest members and all six verification-metadata copies. **Only the evidence package was downloaded to the local computer; the 4.76 GB payload readback occurred on the remote host.**
- [Complete evidence ZIP](cloud_backup/qwen38_native_b2_backup_receipts_20261021.zip): 581,084 bytes, SHA256 `0a1fee3f4fba20253770b734fffeeda9249426a4d0547e6d30b15667d6e84bd4`; contains original manifests, logs, status and verification receipts. The [receipt member manifest](cloud_backup/RECEIPT_MANIFEST.json) describes its 27 evidence members plus the manifest itself.

The original `COMPLETE.json` certifies payload verification and predates metadata upload; use the final status and independent seal receipts above for the completed audit. The evidence bundle retains the original code-backup receipt at Git commit `24aa69c`; this later evidence-archive commit is separate and does not rewrite that uploaded receipt.

Priority checkpoints are complete joint bundles, including optimizer and continuation state:

- `native_main_training_08/joint-policy-stage-2-alpha100_rmsnorm-model_milestone-100000000.pt`
- `native_main_training_08/joint-policy-stage-3-final.pt`
- `native_main_training_07/joint-stage-3-alpha099_rmsnorm-nominal-40000000.pt`

This cloud snapshot covers the completed native mainline/control reports, generated datasets, and their frozen source snapshots. The public base-model cache and previously archived restored raw materials were not uploaded again. The actively training memory-fork directory and its 90M checkpoint are outside this completed backup scope, following the user's priority; its implementation and startup evidence are preserved separately in this Git archive.

[Backup and independent verification helpers](../../../../scripts/archive/qwen38_native_backup_20261021/README.md) are archived with source hashes and the verifier's 11 adversarial fixture results.

## Verification scope

The active frozen implementation had already passed its registered CPU validation and training startup checks. This backup task verified UTF-8 decoding, Python syntax in memory, obvious credential patterns, implementation Git whitespace, portable report links, archive member safety, and the 89-file frozen-source match. It did not launch or repeat GPU tests and did not change the running training checkout.

The archived Matplotlib SVGs retain their original path-line trailing spaces, and the copied 100M report retains its original final blank line. These producer-format warnings are excluded from the whitespace check; original archive bytes and recorded transformations are preserved.

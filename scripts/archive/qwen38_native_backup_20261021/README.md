# Native experiment B2 backup helpers — 2026-10-21

The upload and staging Python files are byte-identical copies of the executed backup helpers. The independent verifier was added after read-only audit. Their hashes are recorded in `provenance.json`. No credentials are stored here.

- `qwen38_native_backup_prepare_20261021.py` inventories the completed native runs and generated datasets, scans selected text for obvious credentials, creates a new hardlink/copy snapshot, and seals file hashes in `metadata/member_manifest.jsonl`, `metadata/member_sha256.txt`, and `metadata/STAGED.json`. The selected source files must remain immutable while hardlinks are in use.
- `qwen38_native_b2_backup_20261021.py` uploads the priority 100M and alpha=.99 40M joint checkpoints plus evaluations first, copies the remaining snapshot, downloads the entire prefix into a new directory, and compares every downloaded file and staged source against its SHA256 inventory. The destination is explicitly restricted to `clclcl/cl-research/delta-ngram/20261021-qwen38-native/`.

The uploader uses `rclone copy --immutable`, never cloud synchronization/deletion. Its authentication file is supplied outside this archive; the account/key become subprocess environment variables rather than command-line arguments, and the helper removes that authentication file in `finally`. Use a dedicated temporary authentication file, not an original credential store.

## Completion evidence

The running helper writes `COMPLETE.json` after the payload readback check, then uploads and reads back verification metadata. Its final `status.json` is authoritative for `verification_metadata_uploaded_and_readback_verified`. A payload COMPLETE receipt alone does not certify that later metadata step.

The helper also inventories the stage afresh before upload. To preserve the stronger staging-seal claim, the final audit independently compares restored payload files with `metadata/member_manifest.jsonl` and checks the seal hashes in `metadata/STAGED.json`. Final upload/readback and independent-seal receipts are preserved in the [completed cloud backup archive](../../../docs/experiments/qwen38_native_20261020/archive_20261021/README.md#verified-cloud-backup). All 2,469 files and 4,760,098,462 bytes passed remote full readback; the local machine holds the verified evidence package.

## Restoration

Restore the exact cloud prefix to a new directory and validate both inventories and their recorded seals before resuming any experiment. Restore generated `data/`, reports/checkpoints, frozen source snapshots, and operational metadata with their recorded relative paths. Joint checkpoints contain the writer plus optimizer, sampler, RNG, budget and other continuation state. Restore previously archived raw-material dependencies only if required by the selected recipe; they were not duplicated in this incremental upload. Public base-model weights must match the model identity recorded by the native verification manifests.

The source recipes contain historical absolute `/workspace` paths and are preserved as evidence. Reconstruct and review the intended paths, code identities, dependencies, and resume state before launching training. These backup helpers do not authorize or initiate new GPU jobs.

## Independent staging-seal check

`qwen38_native_b2_verify_seal_20261021.py` embeds the preregistered three metadata SHA256 values and the 2,466-payload count. Run it with `--restore <original-readback-directory> --output <separate-ops-directory>/seal_verification.json`. It checks the original seal, each payload size and SHA256, checksum text order/content, safe paths, no symlinks, and the exact payload-plus-three-metadata file set. It refuses an extra `verification/` directory and never opens original `source_path` entries. The output must be a new file outside the restored tree. Eleven adversarial fixture checks are recorded in `qwen38_native_b2_verify_seal_tests_20261021.json`.

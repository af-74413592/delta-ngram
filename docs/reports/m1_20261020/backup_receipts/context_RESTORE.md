# DEV64 context add-on recovery

This independent increment is stored in bucket `clclcl` under
`cl-research/delta-ngram/20261020-dev64-context/`. It contains only new context
evaluation results, analyses, operation sources and an incremental Git bundle.
It does not duplicate the old model weights or the original 21-result curve.

## Download and verify

Use an existing authorized B2 remote and a new empty destination:

```bash
export ADDON=/workspace/restored-dev64-context-20261020
rclone copy existingRemote:clclcl/cl-research/delta-ngram/20261020-dev64-context/ "$ADDON"
```

Before executing the verifier, check that
`metadata/verify_context_backup_20261020.py` has SHA256
`495da3ca1fc7b809c70951b00d7dc3a587d9e7049f14d1bafe0db96f0e189960`.
Obtain the expected COMPLETE hash from the independent finalization/restore
receipt linked in the current Git report index. Do not derive the expected
hash solely from the file being tested.

```bash
python "$ADDON/metadata/verify_context_backup_20261020.py" \
  --root "$ADDON" \
  --expected-complete-sha256 "<external receipt's COMPLETE SHA256>" \
  --receipt /workspace/context_addon_recovery_receipt.json
```

The receipt must be outside the restored archive. If the original stage is
available, also pass `--source-stage` to verify that all files have independent
inodes. This validates eight objects, their byte counts, SHA256, exact file set
and stability during verification.

## Recompute statistics

Extract `m1_dev64_context_addon_20261020.zip` into a new directory and check its
`manifest.json` payload hashes. The archive's README gives CPU commands using
the existing core archive's `runs/m1_dev64_curve_20261020/` and
`operations/multidomain/m1_dev64_curve_inventory_20261020.json`.

The existing core is at
`b2://clclcl/cl-research/delta-ngram/20261020-rtx4090/`. Its `supplement/`
subdirectory must be excluded when checking the exact 1,733-object core set.
The separate add-on prefix above is outside that core prefix.

## Recover code

The incremental bundle requires base commit
`fcd056414fd148425940d76c845b39e0ec14e381`, which is included in the existing
full Git bundle:

`b2://clclcl/cl-research/delta-ngram/20261020-rtx4090/supplement/code/delta_ngram_code_fcd0564_20261020.bundle`

The full bundle's SHA256 is
`bf5f2728b311938ec9830ebc980c3f36bb3db38fc5ac77827709dde1373ec550`.
After checking both archives, clone that full bundle into a new repository,
then fetch the add-on bundle and check out its HEAD:

```bash
git clone /path/to/delta_ngram_code_fcd0564_20261020.bundle /workspace/restored-delta-ngram
git -C /workspace/restored-delta-ngram fetch "$ADDON/dev64_context_delta_20261020.bundle" HEAD
git -C /workspace/restored-delta-ngram checkout --detach FETCH_HEAD
```

The resulting commit must equal `git_tip` in the verified COMPLETE file.
Cloud receipts generated after sealing are saved in a subsequent Git commit.
No command here starts training, FINAL or model inference.

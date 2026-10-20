# Frozen registration and execution records

These files are exact byte copies from the completed local M1 evidence set.
Their source paths, sizes and SHA256 values are recorded in
[`provenance_manifest.json`](../../../../scripts/archive/m1_20261020/provenance_manifest.json).
No registration was edited, reissued or applied during archival.

## Scope

- `m1_checkpoint_policy_20261020.json`: original 100M primary / 65M secondary
  policy. Its initial two-point scope is retained as a historical record.
- `m1_dev64_curve_registration_20261020.json` and
  `m1_dev64_curve_inventory_20261020.json`: subsequent 21-point scope and exact
  checkpoint identities. The registration's policy and inventory digests are
  checked against the copied files.
- `m1_frozen_source_manifest_20261020.json`: original frozen evaluator source
  snapshot identities, including its historical local paths.
- `alpha_parallel_registration.json`, `parallel_user_override_20261019.json`,
  `user_stop_small_alpha_20261019.json`: alpha orchestration decisions and the
  user-directed stop. Their presence does not authorize resuming that queue.
- `m1_preflight.json`: recorded M1 preflight and environment evidence.
- `dev64_runtime.json`, `dev64_completion.json`: original runtime and completion
  records, renamed only to distinguish them in this directory. Contents are
  unchanged; the completed queue contains 21 results and no failures.

These records refer to historical absolute paths. They do not assert that model
weights, full datasets or object-storage uploads are available at those paths
on another machine. See the report's archive index for recovery status.

The first-5M prefix manifest is retained in the verified analysis bundle rather
than duplicated here; its path, size and SHA256 are listed as an omitted input
in the provenance manifest.

From the repository root, check copied payloads with:

```bash
python scripts/archive/m1_20261020/verify_archive.py
```

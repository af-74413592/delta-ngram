# Archived full-only memory-fork launch recipes

These are the exact operational scripts used for the 2026-10-21 full-only fork and its queued independent-repeat NLL control. They are preserved for inspection and reconstruction, not immediate execution from this directory.

- `full_supervisor.py`: runs the registered full branch once and verifies 835 groups, the 90M checkpoint, immutable terminal checkpoint, and both DEV64 receipts.
- `repeat_after_full_queue.py`: waits for successful full training plus both terminal DEV64 runs, then executes the registered repeat-only control. No automatic retry is enabled.
- `full_inputs/`: copied full spec, training decision, validation wrapper, and registration/launch script.
- `repeat_inputs/spec.json`: copied repeat-control spec.

The original deployment names were `native_memory_full_inputs_03/full_supervisor.py` and `native_memory_repeat_inputs_02/queue.py`. The scripts resolve their runtime directory from `__file__` and reference frozen remote roots. Reproduction requires recreating the corresponding directory layout, checkpoints, data, validation receipts, and code manifests, then explicitly reviewing and registering any changed paths. Do not run a copied launcher against a live run or assume an old `claim`/status file authorizes a restart.

The [experiment archive](../../../docs/experiments/qwen38_native_20261020/archive_20261021/README.md) contains the original small startup bundle and file provenance. Existing spec paths and source hashes have not been rewritten. Cloud frozen-source backups preserve exact runtime bytes; Git may normalize text line endings.

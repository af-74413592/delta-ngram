# Qwen3.8 native runtime

This directory holds the GPU runtime and diagnostics for the separately
registered Qwen3.8 experiment. Reusable writer and memory code lives in
`src/delta_ngram`; the M1 command-line runner retains its own backend.

## Code boundaries

| Location | Responsibility |
| --- | --- |
| `src/delta_ngram/qwen38_writer.py` | Native writer dimensions, targets, and gates |
| `src/delta_ngram/qwen38_native.py` | Head layout, native row mapping, fixed alpha, and target scaling |
| `src/delta_ngram/qwen38_adapter.py` | Overlay at the real native PLE lookup boundary |
| `src/delta_ngram/qwen38_training.py` | Query leaf gradients and exact full-trajectory writer backpropagation |
| `src/delta_ngram/qwen38_ple_reference.py` | CPU reference of native PLE arithmetic and precision diagnostics |
| `scripts/prepare_qwen38_native_data.py` | Frozen-text retokenization with identity and split checks |
| `scripts/native_qwen38/` | Pinned Megatron loader, runtime checks, and experiment entry points |
| `scripts/native_qwen38/ops/` | Environment bootstrap, verified model download, and frozen-data restore |
| `docs/experiments/qwen38_native_20261020/` | Experiment registration and numerical contracts |

Run outputs belong outside the checkout, for example in the sibling
`delta-ngram-runs` directory. Checkpoints, model weights, restored corpora,
credential files, and full logs must not be committed.

## Runtime contract

The native model is `Qwen/Qwen3.8-Flash-Next`, revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`. The launch uses four H200 GPUs
with TP=2, PP=2, EP=2, ETP=1, CP=1, and DP=1. The reader and native base
table stay frozen. The base table is host-resident; its values in this
checkpoint are BF16. The initial gradient baseline uses the eager PLE path.

The isolated environment bootstrap pins SWIFT and mcore-bridge source commits
and checks the preinstalled CUDA packages before installing its Python overlay.
It does not authorize replacing a different image's CUDA stack.

## Validation status

On 2026-10-20, the real four-GPU preflight passed empty-memory identity,
native address and EOS checks, and a finite nonzero gradient through the
native lookup overlay. A separate evidence-only observation completed
18 frames and 8,403 new tokens with no optimizer update.

The native TP scatter/gather adjoint check passed: both reconstruction
errors were zero, and each TP rank holds a complete replica gradient.
The gradient must not be summed a second time. The earlier full-reader
BF16 finite differences remain inconclusive and are retained as such.

Using captured native inputs, weights, and output cotangents, all twelve
local PLE reference checks and three full-chain directions passed FP64
finite differences. The captured native BF16 input gradient had 0.006946
relative L2 error and 0.999976 cosine similarity against this reference.
This establishes reference consistency, not exact native-kernel derivatives.

Grad-enabled replays are bitwise repeatable, but no-grad queries differ,
including with zero delta. Unwrapping the two compiled HC helpers did not
remove that difference. PLE fusion was disabled throughout these checks.
The provisional evaluation contract uses grad-enabled forwards without a
backward pass; its equivalence to the actual training forward is checked
separately. It retains autograd activation costs.

Small initial writes are also distorted by the BF16 lookup cast. This
precision issue is independent of dispatch parity. A split-projection
repair, refreshed evidence caches and scales, the writer adjoint check,
and a training-mode directional check are required before optimizer steps.
The paired probe entry point remains an unexecuted draft; a registration
or CPU test result alone does not authorize it to train.

CPU tests cover row ownership, valid native row zero, prefix masking,
full-trajectory gradient propagation, and frozen-data identity. These tests
do not replace the real-runtime numerical check or establish training quality.

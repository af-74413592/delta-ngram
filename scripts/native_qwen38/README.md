# Qwen3.8 native runtime

This directory holds the GPU runtime and diagnostics for the separately
registered Qwen3.8 experiment. Reusable writer and memory code lives in
`src/delta_ngram`; the M1 command-line runner retains its own backend.

## Current alternating training

`qwen38_native_train_staged_alternating.py --restart-from-initial --execute`
starts both fixed-alpha arms from the same seed17 initial writer and calibrated
target scales, with independent AdamW, sampler, RNG and model-token counters.
The supplied historical checkpoint provides calibration and corpus identity
only in this mode. Without a migration flag, the runner resumes its exact
joint checkpoint and saved schedule cursor.

The current topology is TP1/PP1/EP4/ETP1/DP4/CP1. Each optimizer group contains
12 complete trajectories, three assigned to each rank. The four phases collect
base-only H in batches of up to 12, construct causal writer histories, read
independent snapshots in batches of up to four, then apply the complete writer
VJP and one DP SUM before clipping and AdamW.

Each completed group atomically saves both arms in `joint-latest.pt`. Each
arm's 20M, 40M and later milestones use the last whole group at or below the
threshold; the unscheduled next group is retained for its next stage. The
runner archives the checkpoint, evaluates the frozen native-tokenizer DEV64
plan, then switches arms. Evaluation failures preserve the pending node for
recovery. The shared 100M optimizer-access cap includes previous consumption;
DEV64 accesses have their own durable ledger and do not update training state.

`qwen38_native_dev64_prepare.py` derives the fixed 64 families from their frozen
UTF-8 originals. `qwen38_native_dev64_eval.py` runs the registered continuation
and direction controls at stage boundaries, with grad-enabled native query
forwards and no backward. The native plan has 53,593 later-article targets;
each complete evaluation charges 2,197,313 target accesses.

The user explicitly adopted the new schedule without further numerical launch
validation. The previous paired gradient thresholds failed and remain failed;
neither numerical equivalence nor harmless gradient noise is claimed.
See the [adoption record](../../docs/experiments/qwen38_native_20261020/qwen38_native_staged_adoption_20261020.json)
and [results](../../docs/experiments/qwen38_native_20261020/qwen38_native_staged_schedule_20261020.md).

## Code boundaries

| Location | Responsibility |
| --- | --- |
| `src/delta_ngram/qwen38_writer.py` | Native writer dimensions, targets, and gates |
| `src/delta_ngram/qwen38_native.py` | Head layout, native row mapping, fixed alpha, and target scaling |
| `src/delta_ngram/qwen38_adapter.py` | Overlay at the real native PLE lookup boundary |
| `src/delta_ngram/qwen38_training.py` | Query leaf gradients and exact full-trajectory writer backpropagation |
| `src/delta_ngram/qwen38_train_batching.py` | Independent trajectory lanes, padding, and packed query cotangents |
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
`de4b8e4d43b917e7706784d8bb445c9af86a3540`. The initial preflight used four H200 GPUs
with TP=2, PP=2, EP=2, ETP=1, CP=1, and DP=1; current training uses the EP4
topology above. The reader and native base
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

The original no-grad discrepancy was localized to the MoE router. MCore
selected a different top-k sorting order according to gradient mode,
changing the order of floating-point score normalization. A scoped native
top-k callback now fixes the training order in both modes. On the audited
books family, all four native/split and initial/zero-memory cases had
bitwise-equal grad/no-grad logits, as did the actual training forwards.
HC compiled and eager paths still differ numerically; the selected runtime
uses eager HC consistently. PLE fusion was disabled throughout.

The split PLE path preserves the native BF16 base projections and computes
the delta branch, addition, normalization, gating and convolution in FP32.
Its projection-addition cosine exceeds 0.99999997 on the audited sample.
The final BF16 output cast still distorts the perturbation substantially;
this repair does not make the full reader loss smooth or lossless.
Fresh evidence and scales were collected with the selected runtime.
The user subsequently canceled the separate eight-step stage and explicitly
authorized direct training without the remaining checks. The writer
fixed-cotangent finite difference and end-to-end directional check remain
unverified; their acceptance is not claimed.

`qwen38_native_train.py` is the original B1 full-pool training entry point. It uses the
existing token-weighted sampler, causal query-before-write execution and
full-trajectory writer BPTT, with at most 2M loss targets per alpha arm.
The two arms share one frozen reader. Small TP gradient differences are
checked against the existing FP32 tolerance before selecting TP0's copy;
replica gradients are not added together. The main process has been
dispatched with the explicit authorization and immutable source hashes.
Checkpoints include writer, optimizer, sampler and per-rank RNG states.
Each completed optimizer group
atomically saves the latest checkpoint. At the user's request, gradient
spikes and gate saturation are monitoring signals only; neither stops
training. Gradient clipping remains 1.0.

`qwen38_native_train_batched.py` continues a completed alpha=1 checkpoint
with B4 (or B2) reader microbatches. Each block wave compacts active independent
trajectories, then runs all query microbatches through one native PP2 schedule,
followed by all base-only evidence microbatches. The initial 12-trajectory wave
provides three B4 microbatches. Every trajectory keeps its own causal memory
and full writer history. Padding creates neither targets nor writes.
The loss callback compensates MCore's microbatch averaging, leaving exactly
one division by the whole optimizer group's actual target count.

Continuation restores writer, AdamW, sampler, all four rank RNG states and
stability history. Its source checkpoint and final source ledger are hashed
in a new registration. Any dispatched unfinished work stays charged to the
budget; it is not treated as completed model training. The new runner supports
a stop-after-group file and saves a latest checkpoint after every update.

CPU tests cover row ownership, valid native row zero, prefix masking,
full-trajectory gradient propagation, and frozen-data identity. These tests
do not replace the real-runtime numerical check or establish training quality.

See the [numerical runtime audit](../../docs/experiments/qwen38_native_20261020/numerical_runtime_audit_20261020.md)
for measured differences, remaining quantization, coverage and receipt hashes.

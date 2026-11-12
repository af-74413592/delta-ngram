# Writer capacity and measured training cost

This analysis separates the native memory interface from the size of the
network that writes it. The released Qwen3.8 model fixes the native row width
at **160**. Its writer's **256** hidden units are an architectural choice.
Neither this analysis nor the accompanying calculator selects a new training
configuration or starts a model run.

The current 8B V7.5 experiment is a separate implementation. Its four source
features, next-token projection, readout, and fixed EMA are not represented by
the historical native writer cost formula below.

## 1. What the existing checkpoints actually contain

Let `D` denote one backbone feature's width, `d` one table row's width, `h` the
writer hidden width, `K` the number of address heads, and `p` the concatenated
event feature width. These are different dimensions.

| Released architecture | D | d | K | h | Target MLP | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Qwen3-0.6B M1 | 1024 | 1024 | 2 | 128 | 3104 → 128 → 1024 | 1,190,465 |
| Qwen3.8 native plain/delta | 2560 | 160 | 16 | 256 | 7712 → 256 → 160 | 4,075,681 |

The prototype adds an external zero table. Its two order vectors are summed
and divided by sqrt(2), producing a 1024-dimensional injection. Qwen3.8 instead
has eight native bigram heads and eight native trigram heads. Their sixteen
160-dimensional rows are concatenated into a 2560-dimensional vector, then
processed by the pretrained PLE projections, gates, and convolution. The
160-dimensional row was inherited from the backbone; it was not a compression
decision made by the writer experiment. See the paper's
[native interface and prototype descriptions](https://arxiv.org/html/2610.31125v1#S3.SS1).

Source definitions:

- [Prototype writer](../../src/delta_ngram/writer.py) and
  [prototype reader](../../src/delta_ngram/reader.py).
- [Native writer](../../src/delta_ngram/qwen38_writer.py) and
  [fixed native layout and factory](../../src/delta_ngram/qwen38_native.py).
- [Actual split-precision native readout](../../src/delta_ngram/qwen38_precision.py).

Plain and delta checkpoints have the same native tensor shapes but different
encoding and training histories. Shape compatibility does not establish
experimental equivalence. The 0.6B M1 weights cannot be directly loaded into
the different native input, row, and head dimensions.

Both released layouts use orders `(2, 3)`. Adding a unigram branch requires
an explicit address layout and reader interface. Appending another head to
the concatenation would change the input size of the frozen native
projections.

For the proposed unigram branch, direct compatibility with token embeddings
is a design constraint: its row width must equal the token embedding width.
The pinned Qwen3.8 checkpoint
(`de4b8e4d43b917e7706784d8bb445c9af86a3540`) contains
`model.language_model.embed_tokens.weight` and `lm_head.weight`, both with
shape `[248320, 2560]`. The tables have separate weights. Thus this branch's
row width is **2560**, independently verified from the embedding/head tensor
headers rather than inferred from the native PLE concatenation. The current
8B model instead uses a **4096**-dimensional token embedding interface.

Equal width is necessary for direct vector compatibility; it does not by
itself align input and output embedding semantics or determine how the writer
learns its target. A smaller row plus a learned projection would be a separate
architecture, outside this direct-compatibility design. The unigram reader
must still be specified separately from the existing 160-dimensional native
bigram/trigram heads.

## 2. What the hidden-width bound does and does not prove

For a fixed writer checkpoint, write the unscaled target as

\[
v(x)=W_2\,\mathrm{SiLU}(W_1x+b_1)+b_2,
\qquad W_2\in\mathbb R^{d\times h}.
\]

Raw targets for a fixed head lie in the affine space
`b2 + col(W2)`, of dimension at most `min(h,d)`. Their linear span can include
the bias direction and have dimension up to `min(h+1,d)`. A scalar head scale
preserves the common linear span, although differently scaled heads need not
share one affine plane.

For the existing additive-patch D1 rule, starting from zero and holding the
writer parameters fixed, scalar gates, scalar decay, and block means keep
each patch row in `span(W2,b2)`. Each row has independent coefficients; this
does not limit the whole memory to `h` scalar degrees of freedom. A radial
cap also preserves this linear span, but generally does not preserve the
original affine plane. The released native target scale is a calibration
buffer, not a cap.

If `h >= d`, the output matrix can have full row rank. This removes one
explicit output bottleneck. It does not prove that the feature distribution
can reach all targets, that training finds them, or that the reader uses them
well. Conversely, a lower-dimensional subspace may be enough for a particular
task. There is no derivation here that proves 128, 160, or 256 is optimal.

The bounds concern varying evidence at fixed parameters. They are not bounds
on the parameter Jacobian: the derivative with respect to the output bias is
the identity. They also do not make the state recurrence Jacobian low rank:
for `T_next = alpha*T + (1-alpha)*T_hat`, holding the target fixed leaves the
direct retention derivative `alpha * I`. If the target depends on the old
state, the full forward Jacobian is
`alpha * I + (1-alpha) * d(T_hat)/dT`. Detaching the target or its encoding
restricts the backpropagation graph; it does not remove that dependence from
the forward map.

For a fixed query, a row perturbation basis `B` reaches the reader through
`J_reader B`. Its local rank cannot exceed the rank of `B`. Multiple rows,
different query Jacobians, and nonlinear finite perturbations must be analyzed
separately; the single-row bound is not an answer-space dimension bound.

## 3. The pretrained row changes the update semantics

The released native writer emits a **patch target** `v`, not a complete
pretrained row target. Its D1 update is

\[
\Delta_j^+=\Delta_j+
\frac1{m_j}\sum_{a:j(a)=j}\beta_a(v_a-\Delta_j),
\qquad T_j=T_j^0+\Delta_j.
\]

Its optional fixed alpha first decays the patch on an evidence-block clock;
learned-beta D1 still follows. Setting that alpha to 0.99 does not implement
a fixed 0.99/0.01 complete-row EMA.

A proposed complete-row target has a different equation:

\[
T_{j,n}=\alpha T_{j,n-1}+(1-\alpha)\widehat T_{j,n},
\quad
\Delta_{j,n}=\alpha\Delta_{j,n-1}
 +(1-\alpha)(\widehat T_{j,n}-T_j^0).
\]

For uncapped `T_hat = W2 z + b2`, fixed alpha, and a fixed checkpoint,

\[
T_{j,n}=\alpha^nT_j^0+W_2q_{j,n}+(1-\alpha^n)b_2,
\quad
\Delta_{j,n}=W_2q_{j,n}+(1-\alpha^n)(b_2-T_j^0).
\]

For a fixed row, the patch lies in `span(W2,b2-T0_j)`. Across different rows,
the different `T0_j` terms can make the patch matrix full rank; that rank is
not evidence that the writer learned additional directions. At fixed row
and write count, evidence-dependent changes still pass through `W2`.

The prior weight is `alpha**n`, so it decays rather than being preserved
forever. Bias-correcting the residual mixture to total weight one would
cancel this initial-prior contribution in the linear case. Capping the full
target, the residual target, or the final patch gives different updates and
different subspace statements. Those choices must be specified before
implementing a migration. The current native writer does not receive `T0`
in its event construction and cannot implement `T_hat-T0` by changing a
configuration value alone.

## 4. Exact size and limited compute accounting

The calculator covers the current D1 architecture, where target and gate use
the same hidden width. With `e=min(32,h)` and `p=3D+e`,

\[
P_{target}=h(p+d+1)+d,
\qquad P_{gate}=h(p+2d+3)+1,
\]
\[
P_{total}=P_{target}+P_{gate}+D+Ke.
\]

The last terms are the pool scorer and head embeddings. Fixed target-scale
buffers are not trainable parameters.

| Native comparison candidate | Total parameters | FP32 weights, gradients and two Adam moments |
| --- | ---: | ---: |
| h = 128 | 2,039,457 | 31.12 MiB |
| h = 160 | 2,548,513 | 38.89 MiB |
| h = 256, existing baseline | 4,075,681 | 62.19 MiB |

These are comparison candidates, not selected configurations. The last
column is `16 * parameter_count` bytes, per unsharded replica with all four
tensors allocated. It excludes activations, sparse rows, historical write
graphs, backbone, buffers, workspaces, and optimizer bookkeeping. Reducing
the hidden width does not reduce the explicit 640-byte FP32 payload of a
160-dimensional row.

The native multi-head implementation already computes each large evidence
projection once per token. Let `P` be padded positions and `E` valid write
events. For `E>0`, the current linear-layer forward MAC count, including the
pool scorer, is

\[
C_{shared}=P(6Dh+D)+Eh(2e+3d+2).
\]

The head-feature projections still execute per event; they are not cached
once per unique head. The single-head-per-order legacy path uses dense event
features instead:

\[
C_{dense}=PD+Eh(6D+2e+3d+2).
\]

An empty event set exits before these linears. These counts exclude biases,
activation functions, attention pooling reductions, indexing, aggregation,
backbone calls, and backward. They cannot be reported as measured speedup.

If the writer occupies fraction `f` of measured training time and its own
execution speeds up by a measured factor `r`, the idealized overall speedup
is `1 / ((1-f) + f/r)`. Halving parameter count does not establish `r=2`, and
cannot halve time spent in the frozen backbone.

## 5. Existing engineering work to reuse

| Concern | Existing implementation | What remains to verify for a new training recipe |
| --- | --- | --- |
| Sparse tensor storage | `SparseMemoryState` has sorted row IDs and an `[active_rows,d]` value tensor | Profile copying, search/unique/index operations and graph storage; do not allocate the full native table per world |
| World isolation | `BatchMemoryState` namespaces row IDs by owner | Preserve owner IDs through encoding, reads, writes, and padding |
| Cross-world source encoding | `collate_memory_encoding_frames` and `build_causal_writer_histories` batch independent trajectories at causal waves | Preserve within-world order, pre-write state, loss normalization, and the logical optimizer batch |
| Shared writer projection | `NativeForwardWriter.event_components` factors evidence/head/state projections | Count the existing implementation, not an expanded reference computation |
| Bounded reader graphs | `NativeTrajectoryGradients` returns query cotangents to the original writer history | Preserve its native adjoint and distributed reduction contracts |

Relevant modules are [state](../../src/delta_ngram/state.py),
[batching](../../src/delta_ngram/batching.py),
[memory encoding](../../src/delta_ngram/qwen38_memory_batching.py),
[causal schedule](../../scripts/native_qwen38/qwen38_native_memory_schedule.py),
and [split reader/writer gradients](../../src/delta_ngram/qwen38_training.py).
Their existence does not establish that a new QA/EMA/next-token recipe is
already integrated, nor that it will meet a particular H200 throughput.

## 6. Measure cost before choosing a configuration

Run the CPU-only calculator from an installed checkout:

```sh
python -m delta_ngram.writer_cost --preset native --hidden-dims 128 160 256
python -m delta_ngram.writer_cost --preset m1 --hidden-dims 128
python -m delta_ngram.writer_cost --preset native --hidden-dims 256 --padded-positions 128 --valid-events 2048
```

Without an explicit batch layout it reports MAC coefficients rather than
inventing a batch. Timing input and prices must be supplied by the caller.
The measured wall-clock window includes checkpoint and synchronous
evaluation time; those subdurations must not be billed a second time.
Additional evaluation resources contribute their actual billed GPU-seconds,
including paid idle time. Parallel worker durations do not add to total job
wall-clock as if they ran sequentially.

For a new recipe:

1. Fix native versus added-row interfaces, feature inputs, target semantics,
   cap placement, and loss/update accounting before a size sweep.
2. Retain 256 as the existing baseline; use 160 to test removal of the explicit
   output-rank bottleneck and 128 as a smaller constrained alternative. A
   target-only width study should hold gate width fixed; that requires an
   explicit architecture change because the current constructor couples them.
3. Compare identical logical data groups and completed target-token budgets.
   Record forward/gradient differences numerically and verify isolation and
   update accounting. Do not silently enlarge the effective batch.
4. On the target H200 topology, collect a bounded representative training
   window, separated by source length and active-row load. Report completed
   target tokens/s, GPU-hours per million targets, billed cost, and peak
   memory. Keep one-time loading/warmup separate and visible.
5. Choose the smallest candidate meeting a prespecified task-quality and
   retention criterion. Raw/capped target spectra and reader-gradient
   projections are diagnostics, not proofs of optimal hidden width.

Asynchronous validation needs an immutable checkpoint, the same evaluation
identity, and explicit handling of delayed checkpoint selection or stopping
decisions. The existing native runtime uses multiple ranks; a single extra
evaluation GPU is not automatically a supported substitute. It may reduce
training idle time without reducing total billed GPU-hours. No asynchronous
evaluation worker or new GPU run is enabled by this analysis.

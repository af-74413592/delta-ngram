# Qwen3.8 A800 bounded forward validation — 2026-10-21

## Status and scope

The independent A800 TP2/PP4/EP2 frozen reader has passed bounded, real-model
memory encoding, sequential D1, teacher-forced NLL, representation capture,
checkpoint recovery, and a full-address-capacity probe. This is not approval
for the complete seven-task 35,000-write experiment or cached generation.
Cross-host A800/H200 numerical equivalence has **failed** the registered screen.
The screen remains unchanged.

The user's subsequent execution decision is to run the formal experiment on
H200 only: one ordinary-writer chain and one delta-writer chain, 70,000 writes
in total. The previously proposed second delta-writer repetition is cancelled.
This A800 report remains a record of bounded engineering tests.

The native base table stays on CPU; memory values and split delta projection
remain FP32. The model and writer are frozen. Query forwards must preserve all
seven memory-state tensors. There is no training or optimizer in these probes.

## Accepted bounded evidence

| Probe | Actual computation | Result |
|---|---|---|
| `native_smoke_v4` | Two D1 writes, four TF queries, empty and nonzero memory | Passed native forward, read-only state, TP join, and 50 capture-site checks |
| `native_shortchain_v5` | 32 SME then 32 NAV writes, 13 TF queries, 104 rank capture files | Passed; memory-aware H and downstream D1 used the current detached delta |
| Recovery inside v5 | Save after write 16, restore, replay write 17 | All seven state tensors, next H, and next D1 state reproduced byte for byte |
| Independent model loads | v4 and v5 empty-memory math, code, SME queries | NLL and 3,146 representation checks were identical |
| `native_capacity_v2` | 666,747 real address-universe rows, 16 further writes | Passed capacity and read-only checks; placeholder rows do not represent a trained terminal state |

The v5 callback took 72.26 seconds; process wall time including loading was
127.24 seconds. Its 63 ordinary timed writes (excluding one extra feedback
diagnostic) took 34.19 seconds, or 0.5427 seconds per frame. Real frame lengths
were 69–86 tokens. Sampled device usage reached 36,025 MiB on the fullest GPU.
The largest registered math query had 946 input tokens and 884 target tokens.
It was a TF probe, not a generated 884-token response.

CPU float64 metric consumers processed 2,420 basic rows, 1,250 detailed rows,
and seven paired NLL records from these captures. The few queries are pipeline
checks, not estimates of seven-task accuracy or long-chain stability.

The capacity probe preserved the 5,224 genuinely written rows from v5 and
added zero-valued placeholders for the remaining addresses. Its seven state
tensors occupied 445,387,012 bytes. Sixteen 74–101-token writes averaged
0.6046 seconds, with sampled device usage up to 37,053 MiB. A cold 946-token
forward took 25.52 seconds; this is not steady-state query throughput.

## Ordinary-writer batching measurement

`native_writer_benchmark_v2` used the ordinary main100M alpha=1 writer,
not the full delta-writer checkpoint. Its checkpoint SHA is
`523b951487cee0ec726077db0104a8fc929f4629bdb4ca46604295a4b78fbfe3`,
and writer tensor SHA is
`d83d2ba750913ec5c9e1a3fc43385a7c16fbe683a106934d7f8f21f853fdb460`.
Each of three independent empty-state runs consumed the first twelve SME
supports followed by the first twelve NAV supports: 24 frames and 1,841 real
tokens. All three used static-T0 evidence and the same chronological sequence
of 24 individual FP32 D1 updates. Only true-token H and native addresses entered
D1; batch padding never became evidence.

| Encoding layout | Measured seconds / frame | Real tokens / second | Speed relative to canonical B1 |
|---|---:|---:|---:|
| Canonical B1, minimum even padding per frame | 0.52825 | 145.21 | 1.00x |
| B1, padded to the corresponding twelve-frame group's maximum | 0.52328 | 146.59 | 1.01x |
| B12, same group padding | 0.05339 | 1,436.69 | 9.89x |

These times include native encoding and H transport, RMS features, individual
D1 updates, TP row synchronization, and a final control barrier. Seven shape
warmups, model loading, CPU captures and hashes, telemetry, and serialization
are excluded. The run performed 57 encoding forwards, 72 total D1 updates,
and no TF or generation. Process wall time was 133.56 seconds; maximum sampled
device usage was 35,741 MiB. This small empty-state measurement does not establish
throughput at 666,747 stored rows or over a complete task chain.

The ordinary canonical B1 result, 0.528 seconds per frame, is close to the
observed delta-writer range of roughly 0.54--0.60 seconds per frame. This is a
comparison between different cases, writer checkpoints, support populations,
and state sizes, not a paired estimate of the cost of memory-aware encoding.

The observed B12 speedup is **not numerical-equivalence acceptance**. The
following relative L2 differences were recomputed on CPU from all saved H
tensors and final FP32 states:

| Candidate versus reference | H relative L2 | Final delta-values relative L2 |
|---|---:|---:|
| Matched-padding B1 versus canonical B1 | 6.0858% | 3.7114% |
| B12 versus matched-padding B1 | 6.8448% | 4.7687% |
| B12 versus canonical B1 | 6.8270% | 4.7295% |

Native addresses, counts, block clocks, and both lazy-alpha clock fields were
identical in every comparison, and all values were finite. Both padding and
batch shape changed H and the resulting delta values. B12 therefore remains a
distinct numerical execution layout; this probe does not establish that its
errors are acceptable or that it can replace canonical B1 in a comparison.
Eight CPU helper tests and independent verification of all 72 H captures and
all seven tensors of each final state passed. Raw numerical evidence is retained
in `native_writer_benchmark_v2/numerical_evidence.pt`; its SHA is
`37cc4e0fd47913c8f87ef350feceb877a4512e5b94f933215706ee21d7a54a50`.

## Pipeline communication correction

The pinned Swift implementation sends activation P2P traffic through the
default NCCL world group, while sending shapes through pipeline groups.
An otherwise unused default communicator could first initialize in only
the upstream stages while downstream stages waited for shapes. Capture
setup had previously hidden this by incidentally running a world collective.

`initialize_default_nccl_world` now performs one all-rank int32 sum before
pipeline execution and confirms the expected result on every rank through
the Gloo control group. Trace v3 and capacity v2 succeeded with this explicit
bootstrap. Earlier stalled trace outputs are retained as failed diagnostics.

## Cross-host precision result

The comparison holds source weights, token IDs, probe masks, capture positions,
and the full fixed delta state constant. Both hosts reproduce their own
outputs. The registered limits include absolute NLL difference <= 0.001,
representation relative L2 <= 0.001, and 1-cosine <= 0.000001.

| Fixed query | A800 NLL | H200 NLL |
|---|---:|---:|
| Math, empty memory | 1.123663193 | 1.125322782 |
| Code, empty memory | 0.189374002 | 0.191847006 |
| SME, empty memory | 3.413286845 | 3.443820953 |
| SME, fixed nonzero memory | 2.913557053 | 2.864527067 |

The SME memory-induced NLL changes differ by 0.0795641. Later representation
differences amplify the initial discrepancies. These results cannot be called
numerically interchangeable or used to silently change acceptance thresholds.

A first-layer HC projection supplies a concrete isolated reproducer: both
hosts have the same complete BF16 input `[41,1,320]` and loaded BF16 weight
`[10240,320]`. Replaying this projection independently on each host reproduces
that host's captured output samples, including five differing sampled values
between hosts. This establishes a matrix-operation numerical discrepancy;
GPU architecture and software libraries differ together, so it does not
isolate hardware as the sole cause or explain every later discrepancy.

Replacing H200 HC TE linear calls with PyTorch linear calls and changing the
global FP32 matmul setting to IEEE left the whole-model comparison unchanged.
In the separate standalone GEMM probe, disabling BF16 reduced-precision
reduction left its 52 cross-host differences out of 419,840 elements unchanged;
this switch was not tested in a whole-model comparison. CPU exact rational
dot products at the five originally captured differing
coordinates favor different hosts at different coordinates. H200 is not a
universal numerical oracle.

The later standalone `hc_reduction_probe_v2` recorded actual CUDA kernel names:
`ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x4_tn` on A800 and
`nvjet_sm90_tst_128x48_64x9_4x1_v_bz_TNT` on H200. Explicit cuBLASLt selection,
disabling reduced-precision reduction, and disabling split-K within cuBLASLt
all left the same 52 cross-host BF16 differences. A direct split-K switch
under the original cuBLAS backend was unsupported; the failed v1 attempts
are retained. The complete FP64 GPU outputs matched byte for byte across
hosts and differed from the CPU FP64 reference by at most 7.11e-15. These are
single-projection results, not a repair or a full-model precision pass.

## Final execution decision (2026-10-21 evening)

The user selected two independent experiments. H200 uses its native MCore
TP2/PP2/EP2 path to write its own ordinary and delta-writer chains and measure
K, ki and answer NLL. A800 uses TP8/PP1 vLLM to write its own two chains and
measure greedy generation accuracy. Each host performs 70,000 chronological
D1 updates; both hosts together perform 140,000. Each chain starts with empty
memory. Both writers use B1 evidence encoding, with distinct registered
main100M/full100M checkpoints. Only the delta writer reads current memory
when encoding H; both D1 gates read their own current memory.

H200 is authorized to start as soon as its local runner passes preflight,
without waiting for A800. A800 still requires a request-boundary live-delta
protocol, full-token pre-head H capture, and full-model short-run checks.
The previous immutable fixed-delta API must not be silently mutated. All TP
workers must acknowledge the new FP32 delta before the next evidence request;
evaluation requests hold the endpoint fixed and never write test tokens.

H200 has 7,168 planned TF queries plus zero-memory baselines and bounded
checks. A800 has 4,480 planned generation queries before any verified reuse.
These metrics describe separately evolved states. Host, method, endpoint and
state hashes must accompany results; H200 NLL and A800 accuracy are not joint
measurements of one shared delta.

Recovery artifact validation now accepts exactly the registered native
TP2/PP2/EP2 and TP2/PP4/EP2 layouts (CP=DP=1). It expects one capture per query
per rank, with rank count TP*PP*CP*DP; EP is not an extra rank multiplier.
The focused 27-test suite covers both layouts, missing ranks, incompatible
manifests, invalid topology metadata, and the existing durable cursor rules.

## H200 formal runner acceptance

The H200 runner uses one model load for an isolated two-write/four-query gate
for each method, then resets each method to an independent empty state for
its formal 35,000-write chain. The gate passed for both methods. Formal
ordinary-writer evaluation began at 2026-10-21 12:22:33 UTC under supervisor
229046, with artifacts in
`/workspace/qwen38-h200-formal-20261021-v1/run_v2`. This records startup and
gate acceptance; it is not a completed long-chain result.

The full math/code baseline has 128/64 queries. Main K uses last-prompt
representations, with four additional answer positions retained. Every task
endpoint repeats these probes and the relevant task queries; no query
updates memory. Each chain checkpoints every 100 writes and at evaluation
boundaries with all seven memory fields, cursor and evaluation receipts.

The runner requires the companion CL workspace. In particular,
`cl/scripts/qwen38_h200_writer_timing.py` supplies registered writer and
original-H200 source hashes; its SHA256 is
`c24674672c48e79b4ff3f0eba16d6c01bf0f87d5e41d07ed1ecd2a7189241bba`.
CL's original LayerCapture and retention helpers are also required. The
actual deployed source package contains all 153 files, including the older
H200 core ABI, and is preserved as `code_v2_source.tar.gz` with SHA256
`5b42b841f3c0171b7c0f836d73a178d7fe3ed011b28658ff7a68c661b67f4600`
in the companion workspace's `h200_formal_v1` transfer receipts.

The earlier v1 attempt exited with zero TF queries and zero D1 writes because
a newer statistics helper required a type absent from the old H200 writer.
V2 observes the original writer's event method once and calls the original
D1 implementation. Actual CPU imports and a seven-field byte-equality check
passed before launch; writer/native/operator sources retain their original
hashes. V1 artifacts are preserved.

The bounded paired B1 timing measured 0.487011 seconds per ordinary write and
0.475749 seconds per delta write on the same 16 supports. Engineering runs
with 666,747 allocated rows measured 0.481556 and 0.469635 seconds per write.
This suggests about 9.25 hours for 70,000 writes alone, excluding TF captures,
checkpointing, loading and transfer. The expanded capacity state is not a
scientific 35,000-write endpoint.

The separate CPU drift consumer supports the four-rank H200 layout and keeps
the historical float64 metric formulas. Its 33 tests passed. Reading the
actual two-method gate produced 2,904 basic and 1,500 detailed metric rows,
plus checksum-bound paired NLL, without initializing CUDA. Ordinary and delta
identities cannot be paired as if they were one trajectory.

## Remaining gates

- Integrate the validated current-memory encoding and full checkpoint loader
  into the complete seven-task runner, with independent state and ledgers for
  the ordinary writer and the single delta-writer run on H200.
- Keep the registered B1 ordinary-writer layout. The bounded B12 probe measured
  a speedup but failed exact H/state equality; it is not the formal protocol.
  The ordinary trained writer checkpoint differs from the delta-writer checkpoint.
- Finish fixed-FP32-delta cached generation acceptance, including multi-token
  decode, request isolation, and same-stack uncached comparisons.
- Resolve or explicitly account for cross-stack numerical differences before
  comparing new tables with historical baselines. New within-run drift
  references must use the same backend, precision path, and batch layout.
- Do not infer support for inputs above 2,048 tokens from the 946-token probe.
  The A800 QSA sequence-parallel gather path above its budget requires separate
  repair and acceptance.
- Measure checkpoint I/O and generation before estimating total experiment
  time. At observed serial write rates, one 35,000-write delta chain alone
  is roughly 5.3–5.9 wall-clock hours on eight GPUs. This is an extrapolation
  of writes, excluding evaluation, loading, transfer, and checkpoint costs.

## Local evidence and code checks

The companion `cl` workspace holds immutable run receipts and SHA-verified
captures under `artifacts/qwen38_a800_forward_20261021/` and
`artifacts/qwen38_native_parity_transfer_20261021/`. Important receipts:

- `native_shortchain_v5.acceptance.json`: COMPLETE SHA
  `b129dd59f4fbad725f5a2e92eb7f3f6406175495d8218c4594b0b0dc6fb85e3d`.
- `native_shortchain_v5_metrics/report.json`: SHA
  `1d363fcbd8ec177476d505e0fbbdada9a2e86c407b21daab8634a7f8b10a920e`.
- `cpu_comparison_v2/report.json`: SHA
  `1ea84bf4378624ae62a792de9a1ad6a65cdcd9166de5bb1434f344e8cb78e51c`.
- Full delta-writer checkpoint file SHA
  `51d5977845e9930f48732c73ae2503511d8976b1201f139a2be46bfab9f57308`.
- `native_writer_benchmark_v2/COMPLETE.json`: SHA
  `2a5a3ff4dbc199da4c9e64e55c736594efbab395eea5a7604d6b968e64e175dd`.
- `native_writer_benchmark_v2.acceptance.json`: independent CPU evidence
  validation only; it does not declare numerical equivalence.

Local CPU validation: 113 tests passed across A800 host transport, memory
encoding, recovery, seven-task protocol, shortchain protocol, FP32 snapshot
export, and capacity-probe modules. CPU tests complement the real GPU probes;
they do not establish numerical equality across GPU stacks.

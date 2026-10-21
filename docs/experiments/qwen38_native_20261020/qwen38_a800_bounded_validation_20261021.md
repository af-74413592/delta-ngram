# Qwen3.8 A800 bounded forward validation — 2026-10-21

## Status and scope

The independent A800 TP2/PP4/EP2 frozen reader has passed bounded, real-model
memory encoding, sequential D1, teacher-forced NLL, representation capture,
checkpoint recovery, and a full-address-capacity probe. This is not approval
for the complete seven-task 35,000-write experiment or cached generation.
Cross-host A800/H200 numerical equivalence has **failed** the registered screen.
The screen remains unchanged.

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

## Remaining gates

- Integrate the validated current-memory encoding and full checkpoint loader
  into the complete seven-task runner, with independent state and ledgers for
  the ordinary writer and both delta-writer repetitions.
- Validate ordinary-writer batched static-T0 encoding separately. Its trained
  writer checkpoint differs from the delta-writer checkpoint. No batching
  speedup or equivalence is accepted by this report.
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

Local CPU validation: 113 tests passed across A800 host transport, memory
encoding, recovery, seven-task protocol, shortchain protocol, FP32 snapshot
export, and capacity-probe modules. CPU tests complement the real GPU probes;
they do not establish numerical equality across GPU stacks.

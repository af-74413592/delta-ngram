# P2 and the first news evaluation

Registered on 2026-10-19, after the P1 diagnostics and before P2 optimization.

## Matched Base experiment

- Train one Base arm, seed 17, with a nominal ceiling of 5,000,000 loss tokens.
  Reuse P1's I arm; do not extend P1 or add another P2 seed to seek significance.
- Copy every tensor from the archived P1 `writer_init.pt`, step zero. Initialize
  a fresh optimizer and sampler; never copy a trained writer or optimizer state.
- Keep dataset bytes, validation sample, window order, batch/accumulation,
  D1 operator, 512/128 block/prefix, 8/16 trajectory mixture and optimizer fixed.
- Base weights: `Qwen/Qwen3-0.6B-Base` at
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.
- I weights: `Qwen/Qwen3-0.6B` at
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- **Both arms explicitly use the I tokenizer at the I revision.** The pinned
  models have identical vocabulary and tokenizer pipeline hashes, but native
  Base EOS is 151643 and I EOS is 151645. Native tokenizer identities therefore
  fail an equality check. This predeclared common-tokenizer binding preserves
  P1's complete tokenizer identity and pre-encoded data; no fingerprint check
  is weakened. Base's own validation is labelled Base weights + I tokenizer.
- The I tokenizer does not change Base model weights or its config. Raw training
  uses the archived token IDs without adding EOS and does not perform generation.
  Model artifact hashes and both native tokenizer identities are retained in the
  launch archive, including the initially failed native-tokenizer check.

The implementation exposes the exact initialization only through an explicit
`--p2-initialization` argument. It rejects a trained source checkpoint, identity,
shape, dtype or nonfinite mismatch, and combinations with resume/ordinary load.
Normal checkpoint loading continues to reject cross-model identities.

Evaluate at 0, about 2.5M and the bounded 5M endpoint, pairing each with the
archived I checkpoint at the same step and actual loss-token count. The original
P1 endpoints are 2,502,656 and 4,997,632 tokens; the nominal budget tail remains
explicit. Training continues only while engineering invariants hold. Stop for
nonfinite tensors, causality/state isolation failures, mismatch or budget error.

The primary endpoint uses **I evidence encoding and I readout** for both writers:

`G = L_self(I-trained writer on I) - L_self(Base-trained writer on I)`.

Compute G directly with the fixed document-family paired bootstrap, 2,000 draws,
seed 104729, weighted by scored tokens. Match the validation sample/protocol,
per-group token counts and empty-reader losses. Also retain Base's own validation
as a trainability diagnostic. Final self/empty/donor/context and recurrence
breakdowns use the same existing diagnostic protocol after the primary run.
No Base evidence encoder is silently retained in the B-to-I condition.

At 5M, a positive G interval supports a benefit in this one matched experiment;
a negative interval favors I; overlap with zero is inconclusive and defaults to
I. No extra training is authorized by a nonsignificant endpoint.

## New components and writer generalization

Record a new memory component, a new training source and a new online document
as different interventions. A document updates memory state; it does not update
frozen writer parameters.

For each future component, keep the same dev manifest, reader, data budget and
question protocol, and log:

1. Frozen-writer inference intervention: immediate state/use benefit.
2. Writer retrained with that component, with matched initialization/order/budget:
   learned benefit, separately reported from the frozen intervention.
3. Held-out document families and domains, changed question wording, corrections
   and intervening writes: generalization and retention, not only training loss.
4. Gold answer NLL, candidate/free-generation accuracy as separate metrics,
   self-versus-empty and self-versus-donor, writing cost, memory size and latency.

Use isolated one-component comparisons for attribution. A cumulative sequence
alone cannot attribute a gain to one component because training time and other
changes also accumulate. These comparisons are registered for later stages;
this P2 run keeps D1 fixed and does not launch a module search.

## News sequence

The first frozen-writer news pilot follows P2. The current two-event development
fixture verifies sourced input and the execution path only; it is not a powered
news benchmark. Expand and freeze an event-family dev cohort before making
generalization claims. Keep the final unseen-event cohort separate and create it
only after the deployment checkpoint and scoring rules are frozen.

The [news protocol](news_evaluation_protocol.md) specifies evidence-only writes,
answer-only loss, fixed donor/context controls and explicit synthetic labels.
Current news is evaluation-only. A source date after the model revision alone
does not prove a fact was unforeseeable; reported actual outcomes, synthetic
fact swaps and correction probes must be distinguished. P4 tracks the same dev
cohort at the planned 25M/50M/100M checkpoints.

The prior proposed value-permutation diagnostics, query-recurrence/collision
cross-buckets and >=64-family long-document cohort remain separate pending
evaluation work. They are not silently included in this implementation or
added as P2 admission gates.

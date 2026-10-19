# P1 checkpoint diagnostics

This protocol follows the requested pause after the second P1 seed. Do not enter
P2 or P3 until the requested diagnostics have been reported. Both writers use
their completed nominal 5M checkpoints; no additional writer training or
checkpoint selection is part of this experiment.

## Fixed population and pairs

- Targets retain the first pilot's exact development sample: 111 windows from
  76 document groups. The manifest binds the dataset, tokenizer, sample and
  original validation protocol identities. The separate holdout is not used.
- Both seeds use one immutable donor manifest prepared before diagnostic loss
  results are observed. Each target has two distinct donor families, both
  different from the target family and in the same programming-language source.
- Donor document lengths must be within a factor of two of the target document.
  Prefer the closest logarithmic length ratio. A fixed seed resolves ties and
  selects a valid contiguous start. Each donor supplies the exact same number
  of complete historical blocks as its target; files are never concatenated.
- Prefer development donors. Two TypeScript 16-block target windows have only
  one eligible development donor each, so their second donor comes from the
  training split. Record these two slots explicitly and also report the subset
  whose donors both belong to development. This sensitivity analysis does not
  change the primary target population.
- Different families mean the repository/near-duplicate families detected by
  the bounded corpus builder, not a claim of exhaustive global fork detection.

## Conditions and causal boundary

All model parameters are frozen. Independent windows reset their online state.
At every target block, use identical target tokens and causal labels:

| Condition | State | Reader input before current target block |
|---|---|---|
| self | Written only from earlier target-document blocks | Last 128 target-document tokens |
| empty | Empty delta state | Last 128 target-document tokens |
| donor 1 / donor 2 | Written from the matched donor's earlier blocks | Last 128 **target-document** tokens |
| context | Empty delta state | All earlier target-window tokens |

Score the block before writing it. Donor state advances by donor blocks, not by
target blocks. Context is a reference condition, not a theoretical upper bound.
No KV cache or cross-window state may supply another history channel.

Use batch size one for every diagnostic condition. The original P1 evaluation
batched within groups with a maximum of four. Report the recomputed self/empty
values and their differences from the recorded P1 endpoint; preserve exact
target identities rather than claiming bitwise equality across batch shapes.

## Outcomes and resampling

Report loss in natural-log units per target token. Average donor 1 and donor 2
loss sums with equal weight before computing `delta_donor = donor_mean - self`;
also report both donors individually. `delta_empty = empty - self` and the
context loss are reported on the same targets.

For each outcome and bucket, accumulate paired loss sums and target counts per
document group. Resample groups with replacement, carrying every condition
together, then divide each draw's summed loss difference by its summed target
count. Use 2,000 fixed-seed percentile replicates and a 95% interval. Groups, not
tokens or windows, are the sampling units. Report subgroup sizes; fewer than 64
groups is limited evidence under the original plan, even when an interval is
numerically positive. These diagnostics do not retroactively alter the P1 gate.

## Delta-empty buckets

1. **Historical token recurrence:** a target token's ID either occurs or does
   not occur among already written tokens earlier than the current block's
   directly visible 128-token prefix. The boundary is fixed at the block start.
   Future blocks and current-block tokens cannot enter this historical set.
   This is literal tokenizer-ID recurrence, not a semantic novelty label.
2. **Programming language:** the six existing `code/<language>` sources.
3. **After 4/8/16 writes:** score block 5/9/17 respectively, using the natural
   next block from the same development document. Deduplicate `(document,
   start, writes)` across original 8/16-block windows. Never score a block that
   has already been written, truncate a target, or join another document.

The fixed 95 distinct starting points support 95/76 groups after four writes,
79/65 groups after eight, and 14/13 groups after sixteen. Report the unavailable
starts and language coverage. The sixteen-write result covers a small long-file
subset and must not be presented as representative of all 76 groups.

## Block monitoring and future training

Capture the actual writer events once, without recomputing gates or retaining
autograd graphs in logs:

- Beta count, sum, squared sum, fixed histogram, quantiles and range.
- Unique touched rows per block, new rows and cumulative written rows.
- Frobenius norm of the patch change, touched-row change norms, and total patch
  norm. Newly allocated rows have a zero prior value.
- Query-address hit counts and denominators by head and overall, plus the share
  of scored tokens with any written address. A hit means that the address has
  been written previously, including a row whose current delta happens to be zero.

Query addresses must align with the logit that predicts each target: addresses
at position `j` are paired with the label at `j+1`, including the last prefix
position that predicts the current block's first token. Ignore padding and
zero addresses. Compute query statistics before writing the block.

Future raw training records these metrics in `block_metrics.jsonl`, annotated
with optimizer step, microbatch, stream, document and block. Logging must not
consume RNG, change D1 updates, detach the training state, or run again during
reader checkpoint replay. Monitoring-on/off loss, state and gradient parity
is required. Existing completed runs retain their original logs.

## Artifacts and cost

Save the fixed pairing manifest, training/checkpoint identities, evaluation
source commit, per-block records, per-group outcome sums, coverage and final
results. Keep partial outputs clearly incomplete after failure or a wall limit.
Record scored targets, reader input tokens, evidence tokens, write blocks and
elapsed time separately by condition; full context costs more input computation.

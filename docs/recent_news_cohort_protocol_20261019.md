# Recent-news development cohort — frozen 2026-10-19

## Purpose and status

The development cohort measures whether a trained forward writer can retain earlier news and help a fixed reader answer questions or predict a later report. Online adaptation consists exclusively of forward memory writes and reads with all learned parameters fixed. There is no online optimizer, backward pass, gradient update, or optimization of memory against an answer. Offline training may use backpropagation through time.

Two evaluation datasets are frozen:

| Dataset | Independent event families | Queries or pairs | Status |
| --- | ---: | ---: | --- |
| Recent-news QA | 64 | 256 two-candidate questions | Development diagnostic |
| Original-news continuation | 16 | 16 earlier/later article pairs | Engineering validation; formal continuation conclusions require at least 64 independent pairs |

QA alone does not establish the complete intended capability. The continuation task tests prediction of a later original report after forward writing an earlier report. Neither dataset is a training corpus. This document does not report model performance, and its preparation did not read the sealed final holdout.

## Reader, dates, and claim boundary

- Reader/tokenizer: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Observed revision date: `2025-07-26T03:46:27Z`; the actual pretraining cutoff is unknown.
- Scored-source availability window: 2025-08-01 through 2025-12-31, inclusive.
- Sources were verified and retrieved on 2026-10-19. The archived response is not a certified historical snapshot of the page in 2025.
- Use the maximum observed publication/update timestamp as availability. Missing modification metadata means unknown modification history. A publication date alone does not prove the retrieved body existed unchanged on that date.
- These are writer-held-out news events relative to the intended 2017–2018 CCNews training split, not certified facts unknown to the base model. The QA manifest's empty `training_event_families` list is not an exhaustive training-corpus audit.
- Exclude previously used IMAP/Carruthers/SWFO launch material and the entire Sentinel-1D launch family. Keep repeated reports of one championship, award, discovery, or project in one event family.

Actual announcements or results are required; scheduled future dates do not count as completed events. Source updates outside the window were excluded where discovered. The archived registry records additional excluded candidates and reasons. Results should include an empty-memory baseline because base-model familiarity remains possible.

## QA cohort

The canonical manifest is `examples/news_recent_dev_20261019.json`, using `news_forward_v1`. It contains 64 real, nonsynthetic episodes and exactly four questions per episode. Each evidence paragraph is an authored compact paraphrase of an official primary source, with source URL, dates, fact identifiers, and hashes. Each gold value appears verbatim in its evidence; it is absent from its question. There are exactly two candidates, and each family has two correct answers at position 0 and two at position 1, giving 128 at each position overall.

| Domain | Families |
| --- | ---: |
| Science | 13 |
| Public infrastructure | 11 |
| Technology | 11 |
| Sport | 15 |
| Culture | 8 |
| Weather and environment | 6 |

Evidence contains 42–78 tokens per episode, 3,886 tokens in total, using the pinned tokenizer without special tokens. The evidence ceiling is 128 tokens. This short-evidence diagnostic is distinct from the full-body, 512-token write-block continuation protocol below.

For every question, keep its reader prefix, candidate strings, tokenization, and scoring span identical across:

1. **Self:** forward write the question's own evidence.
2. **Empty:** use the exact empty memory state and perform no write.
3. **Donor:** forward write a different event family's evidence.

Donors are a fixed within-domain cyclic permutation after sorting by evidence token length. Each episode is used once as a donor; no answer label affects matching. Questions are asked at collection time, after all donor sources are available. A separate context condition may expose evidence directly in the reader prefix, but is an extra-information diagnostic and must not be mixed into the same-prefix contrasts.

Report candidate accuracy, gold-answer mean token NLL, and paired self-versus-empty and self-versus-donor differences. Candidate selection minimizes mean candidate-token NLL. First average the four questions within each family, then macro-average families. Questions within a family are not independent observations. Optional answer generation requires separate, fixed decoding and answer-normalization settings and is not interchangeable with two-candidate accuracy.

### QA provenance limitations

All evidence has an `evidence_sha256`. Each source has a `verification_record_sha256` whose declared scope is the canonical compact verification record, **not raw page bytes**. Available raw-response hashes are kept separately. Do not relabel verification-record hashes as original-page hashes.

The collection audit flags nine records whose verification relied partly or entirely on official-primary search text: Asia Cup, Tokyo athletics pole vault, EuroBasket, UCI road, Nobel chemistry, FIFA U20, Blender 5, Davis Cup, and FIFA U17. Some originals were subsequently fetched for continuation. The QA evidence and access-method labels remain frozen; these cases should be visible in a sensitivity analysis rather than silently described as uniformly archived original pages.

## Original-news continuation

The local registry is `delta-ngram-runs/recent_news_sources/recent_news_continuation_20261019.json` (`recent_news_continuation_v1`). It contains 16 existing QA event families, 32 original article bodies, and 17 background replay articles. The 16 pair domains are science (3), culture (2), technology (3), and sport (8). The collected bodies range from 128 to 1,332 words. All 32 paired originals returned HTTP 200 and were extracted from publisher pages, rather than generated summaries.

For each pair, require `earlier.available_at < later.published_at`. Both scored articles fall within the August–December window. Source metadata, the exact retrieved response, extracted original text, their separate SHA-256 hashes, and extraction method are retained locally. Body extraction removes navigation, related-story cards, and identified repeated video captions; the scored object is the frozen extracted body. An HTML response may contain page furniture that is absent from that body.

Pairs include software release/maintenance, award shortlist/result, sports semifinal/final, and mission progress/report sequences. These relations offer prior information without guaranteeing that a later result is inferable. A sports semifinal does not logically determine a final's winner. Report per-family outcomes and avoid interpreting generic topic or entity reuse as proof of forecasting a genuinely unpredictable result.

### Forward-only execution

For each pair and interference condition, start from a fresh empty memory. Write the complete earlier article using consecutive blocks of at most **512 tokens**. Retain only the designated forward memory across article boundaries; reset the ordinary local reader/KV context at those boundaries. Do not truncate an article to one block. Padding must not update memory.

After the anchor write, replay exactly 0, 1, 4, 8, or 16 unrelated articles in the frozen nested order. These July 2025 background publications were already available before each query. Their original dates are preserved. Replaying them after the anchor tests interference from previously available articles; it does not claim that they were first published between the paired reports. They are unscored background, outside the August–December scored-source window.

Run the following conditions independently:

| Condition | Anchor | Subsequent writes |
| --- | --- | --- |
| Self | Earlier same-family article | Fixed unrelated replay prefix |
| Empty | None | None; exact empty state |
| Donor | Earlier available different-family article | Same unrelated replay prefix as self |

Donor selection prefers the same domain, then closest complete-article token length, then a deterministic SHA-256 tie break. It uses no target labels. Every donor is available before query time and belongs to another event family. Reuse is allowed; a background donor is used when an earlier paired donor is unavailable. A selected background donor is excluded from that pair's distractors.

**Score the later report before any possible write of that report. This evaluation never writes it.** Its prefix, target tokens, generated continuation, questions, and answer labels must never enter the writer. Hold the selected memory fixed throughout later-article scoring. Standard teacher-forced *past* tokens may appear in causal reader context; no future token may be visible to an earlier prediction.

### Read windows and metrics

- Tokenize the full later body without special tokens using the pinned tokenizer.
- Fix the initial prefix at **128 tokens** for every condition.
- Score every token from index 128 through the body end, exactly once, using read windows of at most 512 tokens. Each window supplies the preceding 128 tokens as ordinary reader context and scores at most 384 new target tokens. These reader-context tokens are never memory-write inputs.
- There are **13,200 scored target tokens per interference distance** across the 16 pairs.
- Primary continuation outcome: mean target-token NLL within each family, then an unweighted macro-average across families. Report token-weighted pooled NLL separately if desired.
- Positive improvement is `NLL(empty) - NLL(self)` and `NLL(donor) - NLL(self)`. Report both at every interference count.
- Secondary outcome: NLL on the first 384 target tokens, or the shorter remaining body. Auxiliary free continuation uses the same initial 128-token prefix, greedy decoding (`do_sample=false`), at most 128 new tokens, and the model's fixed EOS termination; it must not affect writer state or checkpoint selection.
- Bootstrap event families, not questions or windows: 10,000 resamples, seed 20261019, percentile 95% confidence intervals. Preserve all five distances and all conditions when resampling a family. Report all 16 family outcomes and uncertainty. This engineering subset is too small and compositionally narrow to stand in for the planned 64-pair formal cohort.

Report both unrelated **article count** and actual written **block count**. At 16 distractors, self writes 17 articles in total and 38–41 actual 512-token blocks across these pairs. The registry separately records token, article, and block totals for self, donor, and empty. Article count and block count are not interchangeable.

### Oracle separation

Any supervised optimization of memory using later targets, answer labels, gradients, or repeated loss-driven updates is an oracle diagnostic. It must be clearly labeled, excluded from the online forward-only claim, and excluded from checkpoint/hyperparameter selection and primary results. Keep offline writer training, genuine forward evaluation, and oracle diagnostics in separate run records.

## Runtime field map and portability

Resolve every `*_relpath` against the continuation registry's directory. Absolute `*_path` values document the original collection location and are fallback only. Transfer the registry together with its `continuation_raw/` files; preserve bytes and verify hashes before running elsewhere.

| Registry field | Runtime use |
| --- | --- |
| `model`, `tokenization` | Pin tokenizer/revision and 512-token write blocks |
| `sources`, `background_sources` | Resolve original bodies, dates, hashes, token counts |
| `pairs[].earlier_source_id` | Self anchor write source |
| `pairs[].later_source_id` | Query-only later body; never writer input |
| `pairs[].donor_source_id` | Fixed alternative anchor |
| `pairs[].scoring` | Prefix length and complete target-token interval |
| `pairs[].interference_conditions` | Frozen distractor order plus expected write totals |
| `online_invariants` | Enforce fixed parameters and prohibited target writes |

The registry is independent data for the continuation runner; support by the QA CLI alone does not imply that continuation execution is implemented. A runtime should fail on missing sources, a hash mismatch, incompatible tokenizer/block size, noncausal source order, or unexpected write totals.

## Frozen artifact identities and checks

| Artifact | SHA-256 |
| --- | --- |
| QA file bytes | `8dae87e3df740261bd1cebe88f588c574157841f074fbb8114eee79ba342c59b` |
| QA validator canonical manifest | `f52a153a366f3cb449989c1102434e3f3328bd85c62e379011e7db26a64f0247` |
| Continuation registry file bytes | `708e2ff104ff4e0541ac6c0092026b5921a2632e04c049422f54866a890cd667` |

The canonical-manifest and file-byte hashes intentionally differ. Collection scripts, component source registries, `dev_audit.json`, `continuation_audit.json`, and `continuation_integrity_audit.json` reside in the local `delta-ngram-runs/recent_news_sources/` directory. The integrity audit checks original/raw file hashes, pinned-tokenizer token counts, causal ordering, family membership, nested distractors, write totals, and complete nonoverlapping scoring coverage. Dataset checks are not model-performance results.

Before formal continuation claims, expand to at least 64 independent pairs under a separately frozen registry and preserve a sealed final evaluation. Do not select that expansion based on per-family gains from the engineering set. Development data may guide implementation; it must never be relabeled as untouched final evidence.

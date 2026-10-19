# Old-news training source and temporal isolation

2026-10-19. This document registers the dated news corpus for the new mixed-domain
run. The earlier two-event recent-news smoke test was an execution check, not a
complete news training or evaluation program.

## Source selection and immutable identity

Use [vblagoje/cc_news](https://huggingface.co/datasets/vblagoje/cc_news), revision
`81eb2ce0d2a9dad6ad16b68ef750ec290880fa36`. The source contains
708,241 English articles; usable date coverage must be measured from the records,
not assumed from the advertised corpus era. Its actual
Parquet schema contains `date`, `url`, `domain`, `title`, `text`, `description`, and
`image_url`; **`date` is present and documented as publication date**. There is no
need to infer publication dates from URLs. The card identifies `news-please`
extraction, followed by English-language filtering.

The five immutable Parquet files total **1,122,805,586 compressed bytes**. The
collector hardcodes their upstream LFS SHA256 values and sizes, downloaded from
the revision-specific Hugging Face endpoint. It checks both before publication
and again before scanning. The default compressed download cap is 1.2 GB.

The [Common Crawl source announcement](https://commoncrawl.org/blog/news-dataset-available)
describes the underlying daily news WARC collection. The compact derived dataset
does not retain per-record WARC paths, crawl times, or update times. Its extracted
publication date is source metadata, not an independently verified timestamp.
The dataset card reports `license: unknown` and supplies no article-level license;
the export preserves this uncertainty and does not infer reuse rights.

[stanford-oval/ccnews](https://huggingface.co/datasets/stanford-oval/ccnews) is a
larger alternative with publication date, crawl date, language score, URLs and
WARC provenance. Its year configurations are **crawl years**, so they cannot
substitute for filtering publication dates. The smaller, explicitly dated
English corpus is sufficient to attempt the first bounded collection.

## Fixed time partitions after the complete raw-date audit

The initial quality-filtered scan under the earlier 2017-2019 registration found
581,109 eligible records: 581,107 in 2017/2018 and just one in each old 2019
evaluation partition. Both 2019 candidates failed the 1,024-token filter. The
original 2019 split therefore failed collection readiness.

The completed [raw-date audit](news_date_audit_20261019.json) independently read
the date column of **all 708,241 records**, before text or quality filtering. The
strict parser and an independent `dateutil` parser both accepted 630,643 dates
and produced exactly the same monthly counts. The remaining 77,598 values were
missing or blank. There were zero nonblank strict-parser failures, zero alternate
parser recoveries and zero parser disagreements. The raw year-pattern check
also found only two records mentioning 2019. Thus the missing 2019 coverage is
in the source, not a strict-parser loss. The two records occur in April and
August 2019; August-December 2018 contain only nine records in total.

| Raw date-parser outcome | Records |
|---|---:|
| Total rows audited | 708,241 |
| Strict-parser success | 630,643 |
| Independent-parser success | 630,643 |
| Missing or blank date | 77,598 |
| Nonblank strict-parser failures | 0 |
| Alternate-parser recoveries / disagreements | 0 / 0 |

Monthly counts below are **raw records with valid dates**, not token-qualified
articles. Zero means no dated record in that month.

| Month | 2017 | 2018 | 2019 |
|---|---:|---:|---:|
| January | 30,761 | 2,344 | 0 |
| February | 43,695 | 59,364 | 0 |
| March | 9,075 | 47,367 | 0 |
| April | 58,615 | 51,324 | 1 |
| May | 4,927 | 62,930 | 0 |
| June | 40,276 | 1,429 | 0 |
| July | 1,479 | 46,227 | 0 |
| August | 62,353 | 2 | 1 |
| September | 1,824 | 1 | 0 |
| October | 49,811 | 3 | 0 |
| November | 1,986 | 1 | 0 |
| December | 54,845 | 2 | 0 |

The following replacement is now frozen from that audit. This change uses data
availability, not checkpoint performance. The failed build and raw downloads
are retained unchanged. The audit JSON records the pinned shard identities,
original parser hash, audit-script hash, per-shard counts, and counts after the
original quality filters. Diagnostic parsing did not supply inferred timestamps
to the training corpus.

| Partition | Explicit publication dates | Raw dated records | Use |
|---|---|---:|---|
| Train | 2017-01-01 through 2017-12-31 | 359,647 | Writer updates in the mixed run |
| Validation | 2018-01-01 through 2018-04-30 | 160,399 | Fixed loss diagnostics/checkpoint selection |
| Old-news holdout | 2018-05-01 through 2018-07-31 | 110,586 | Reserved corpus-era evaluation |
| Recent-event final test | Separately curated 2025/2026 event families | Separately curated | Frozen writer, new evidence and scored questions |

Malformed, missing or out-of-range publication dates are rejected. A recent
article cannot enter any old-corpus partition. The original full date string is
retained alongside its parsed calendar date. Timezone-free dates remain
timezone-free; the collector does not invent a timezone. The 2018 old-news
holdout does not replace the separate recent-event final test.

## Article lengths: explicitly a long-news training subset

A direct read of row group 0 (the first 1,000 records) of pinned shard 0 was
tokenized with `Qwen/Qwen3-0.6B` revision
`c1899de289a04d12100db370d81485cdf75e47ca`, with no special tokens or truncation.
This is an unfiltered first-row-group measurement, **not a random corpus sample**.
Median length was 453 tokens; p75 was 837.75, p90 was 1,175.7, p95 was 1,436.5,
and the maximum was 7,096.

| Minimum tokens | Articles out of 1,000 | Retention |
|---:|---:|---:|
| 512 | 442 | 44.2% |
| 1,024 | 155 | 15.5% |
| 2,048 | 14 | 1.4% |
| 4,096 | 5 | 0.5% |
| 8,192 | 0 | 0.0% |

Keeping the original 4,096-token minimum would exclude almost all this sample.
The new news training source instead uses **512-token blocks and 2/4-block
trajectories**, with target-token weights 0.9/0.1. The collector retains complete
extracted articles of 1,024 through 16,384 tokens. It neither truncates an article
to the cap nor joins unrelated articles to meet a length minimum. Articles over
the cap are rejected as whole units. Training windows use existing token IDs.

This is still a substantial bias toward long news: 2-block trajectories require
at least 1,024 tokens. The source and actual selected-pool retention are saved in
the manifest. The later recent-news QA test should include ordinary short news
to measure transfer explicitly. A future smaller-block experiment would be a
separate controlled configuration; it is not silently introduced here.

## Grouping and temporal leakage checks

Before token-length filtering or selecting representatives, the builder joins:

- Canonical article URLs, removing fragments and known tracking parameters while
  preserving identity query parameters.
- Exact normalized full-text hashes (the existing mixed-corpus hash convention).
- Exact headlines containing at least five lexical words, conservatively tying
  syndicated articles or updates.
- Confirmed full-article 5-lexical-token-shingle Jaccard similarity >= 0.85;
  candidates are retrieved with 64 MinHash values in 16 bands of four.

Grouping uses connected components: a URL edge followed by a text or near-text
edge binds the whole family. **Every detected component touching more than one
time partition is excluded in full.** Only one complete article is selected per
remaining family. The final audit checks group/URL/raw-hash/normalized-hash split
disjointness, publication boundaries, at least 64 validation and holdout groups,
and available 2/4-block windows in every partition.

By default the scan keeps deterministic bounded pools of 400,000 training and
20,000 articles per evaluation partition. For the main collection, use candidate
caps of 708,241 for every partition so all quality-eligible records in this fixed
708,241-row source participate in duplicate grouping. Date/text quality filters
still apply. Approximate MinHash retrieval can miss near duplicates, and lexical
or headline similarity does not prove semantic event identity. A detected
duplicate family must not be described as a fully annotated event family. The
recent-event final test therefore requires separate curated event families.

## Token budget and readiness

The first mixed pilot needs at least 10M qualified news raw tokens. The requested
100M-loss-token main run motivates a larger news pool; the builder's main target
is **50M training raw tokens**, plus at least **5M raw tokens in each 2018 evaluation
partition** and at least 64 independent groups each. Larger evaluation pools are
needed to recover enough multi-report families while preserving 64 single-news
groups after event families are removed. These are **collection
targets**, not completed corpus counts. Availability must be measured after
dates, quality, duplicate quarantine and token-length filtering. If the target
cannot be met, the script fails with counts rather than marking the pool ready.

The manifest reports raw tokens, available window/loss tokens for 2/4/8/16 blocks,
unused tails for each length, publisher token shares, date ranges, rejection
reasons and source-file hashes. Raw tokens are not the same as unique usable
target tokens, and the sampler may revisit windows; the mixed-run budget must
use these actual counts. A 10M pilot collection can be explicitly requested with
`--target-train-tokens 10000000`; it must not be relabeled a 50M pool.

## Commands and outputs

`prepare_news_corpus.py` requires `pyarrow`, `numpy`, `transformers` and the local
package. All source text is treated as data, never as code. On the GPU host,
execute long operations through its required Supervisor workflow after reading
the host guide; the commands below describe the worker payload only.

```text
python scripts/prepare_news_corpus.py --download-dir /workspace/delta-ngram-data/multidomain_20261019/downloads/news --download-only
python scripts/prepare_news_corpus.py --download-manifest /workspace/delta-ngram-data/multidomain_20261019/downloads/news/download_manifest.json --output-dir /workspace/delta-ngram-data/multidomain_20261019/news --train-candidates 708241 --eval-candidates 708241 --target-train-tokens 50000000 --target-eval-tokens 5000000
```

Published outputs are `train_validation.jsonl`, `holdout.jsonl`,
`selected_texts.jsonl.gz`, and `data_manifest.json`, plus scan/candidate audit
manifests. Holdout is never present in the trainer file. Each record has source
`news`, the pinned tokenizer fingerprint, original publication metadata, URL,
dataset revision, shard and row provenance, source/text/token hashes and detected
family membership. Existing exported corpora are never overwritten.

Two reusable stages are saved **before quota selection**, including when a later
selection fails: `families_candidates.jsonl.gz` after grouping/quarantine and
`tokenized_candidates.jsonl.gz` after the registered token-length filter. Each
has a corresponding `*_cache_manifest.json` with record count, file bytes/SHA256,
statistics and an input fingerprint. Complete extracted text, source metadata,
original membership and (at the tokenized stage) token IDs are retained. Cache
files belong in the data output directory, not the Git repository.

The family fingerprint pins source shards, time windows, code/primitives hashes,
sampling seed/caps, quality bounds and grouping threshold. The tokenized cache
also pins the family-cache hash, tokenizer identity and token-length bounds.
Changing these inputs requires a new cache directory. Selection targets and
minimum selected group counts do not enter the cache fingerprint, so a failed
quota can be retried without rescanning, regrouping or retokenizing. The original
source shards are still verified when caches are reused. `--cache-dir` defaults
to `--output-dir`; explicitly point a new output build at an existing verified
cache to preserve an earlier failed output directory. A partial or corrupt cache
fails closed. The tokenized cache contains length-eligible candidates; changing
the token-length bounds needs a new tokenized cache.

Offline verification is in `tests/test_prepare_news_corpus.py`, covering date
boundaries, invalid/recent-date rejection, download integrity and size bounds,
URL identity, deterministic sampling, transitive cross-time quarantine, exact
and near duplicates, retained short-article statistics, full-article encoding,
raw-loader compatibility, token targets and final audit failures.

## Additional whole-article news streams (2026-10-19 revision)

The single-article corpus above remains a separate stratum. The new
`scripts/prepare_news_event_streams.py` builds chronological multi-article
streams from its **completed** selection. It does not modify or restart the
running old-news collector, and does not rerun MinHash.

### Recovery and membership

`selected_texts.jsonl.gz` retains `metadata.family_members` from before the old
token-length filter. Its rows do not contain top-level `group_id` or `split`.
The new builder reconstructs the original group ID exactly as
`news-family:` plus the canonical digest of sorted member IDs. It reconstructs
each original document ID from the pinned revision, shard path, shard checksum
and one-based row while rescanning the five verified Parquet files. Every member
must be found exactly once; source representative hashes and family-wide date
ranges must match. Every recovered member is checked against the original time
partition **before any deduplication, length filter or article selection**.

Only exact normalized full-body duplicates are removed. A matching headline,
URL or near-identical body is inherited grouping evidence, never a reason to
remove a changed article. These are **automatic reporting families**, not
human-verified events or confirmed correction examples. Lexical grouping can
miss related reporting and join semantically different stories. The separate
recent-news semantic tests require human source verification.

The new family pool is restricted to families represented in the original
selection. It recovers their shorter members, but does not undo the original
requirement that a selected representative had at least 1,024 tokens. The
manifest records this selection bias and all subsequent exclusions.

### Preregistered stream selection

- Exactly one stream per inherited family; 2 through 4 distinct whole articles.
- Article lengths: 128 through 16,384 tokens, with no special tokens and no
  truncation. Short or long articles are excluded whole and counted.
- Every selected article must follow every earlier selected article. Two
  timezone-aware timestamps are ordered by strict UTC time, including when
  offsets reverse calendar-date order. Other pairs require different increasing
  explicit source calendar dates. Same-day naive/date-only clocks cannot order
  two articles; equal UTC instants cannot order them either.
- Every pair spans at most 30 source calendar days and, when both timestamps
  have explicit timezones, at most 30 UTC days. Full original family date ranges
  remain visible even when a narrower stream is chosen.
- Choose by deterministic earliest-feasible traversal from each possible start;
  prefer more articles and then the seeded chain hash. This is conservative,
  not a claim of a maximum-length chain for mixed timezone metadata.
- The whole stream contains at most 16,384 real tokens **and** at most 32
  article-local 512-token chunks. These are separate limits: two articles of
  8,193 and 8,191 tokens total 16,384 tokens but require 33 chunks, so that pair
  is rejected. A feasible shorter chain may still be retained without truncation.
- Report actual target tokens in buckets 4/8/16/32, where the bucket is the next
  supported count at least as large as the sum of per-article rounded-up chunks.

The upstream timestamps remain extraction metadata with unknown original crawl
and update times. Calendar order with unknown timezones is not proof of UTC or
publisher availability order. Timestamp precision and timezone status are
exported; no timezone is invented.

### Export contract and budget audit

Each JSONL row has `id`, inherited `group_id`, `source: "news_event"`, `split`,
`tokenizer_identity`, and `articles`. Each article keeps its own `id`, complete
`token_ids`, and original date/URL/shard/row/text-hash/token-hash metadata. There
is **no top-level flattened token list**. Cumulative article boundaries are audit
indices, not a joined model input. Metadata includes all original family member
IDs, the chosen ordered IDs, raw-token and chunk counts, selected calendar span,
and explicit `human_verified_event: false`.

The runtime writes the first article without loss, then scores every real token
of later articles with article-local prefix and n-gram state. A local BOS/EOS
input enables scoring the first token of each new article; BOS/EOS and padding
are not targets. The builder reports the sum of complete later-article lengths,
5% and 10% shares of the registered 100M target-token budget, and whether either
share fits at most three average uses of the available targets. Runtime must
enforce actual reuse and masks; a capacity estimate is not permission to exceed
three uses.

All selected stream family IDs are saved in `event_stream_families.json`,
including diagnostic builds that fail readiness. The integration must remove
these entire families from the single-news stratum in each corresponding split.
The builder verifies the original single-news JSONL checksums and reports the
remaining independent groups, raw tokens and separate 2/4/8/16-block capacities.
Those window cells overlap and must not be summed as unique target capacity.

Structural readiness requires at least 64 independent stream families in
**each** of train, validation and holdout, plus at least 64 remaining single-news
families in each evaluation split after removal. If any gate fails, the builder
writes its candidate audit and family list, raises an error, and does not publish
training JSONL or a ready final manifest. A structurally ready corpus still needs
the intended 5% event / 15% single-news training budget and runtime reuse checks
to pass before training starts; budget booleans are separate from structural
readiness.

```text
python scripts/prepare_news_event_streams.py --selected-texts /workspace/delta-ngram-data/multidomain_20261019/news/selected_texts.jsonl.gz --download-manifest /workspace/delta-ngram-data/multidomain_20261019/downloads/news/download_manifest.json --output-dir /workspace/delta-ngram-data/multidomain_20261019/news_event --maximum-stream-tokens 16384 --maximum-stream-chunks 32 --total-training-tokens 100000000 --maximum-reuse 3
```

The source manifest defaults to `data_manifest.json` beside the selected texts.
Outputs are `train_validation_event_streams.jsonl`,
`holdout_event_streams.jsonl`, `event_stream_families.json`,
`selected_event_texts.jsonl.gz`, `event_stream_candidate_manifest.json`, and (only on structural success)
`event_stream_manifest.json`. Existing builds are never overwritten. The worker
is CPU data preparation; launch it through the GPU host's required Supervisor
workflow, separately from the running old collector. Offline checks are in
`tests/test_prepare_news_event_streams.py`.

`selected_event_texts.jsonl.gz` exports every chosen stream article once as
`{id, text, metadata}`, preserving its full extracted body and original article
metadata. Its bytes and SHA256 are recorded in the final stream manifest. The
mixed finalizer must audit **all** these articles for cross-source leakage, not
only the original single-news family representative; auditing must preserve
changed reporting within an accepted stream.

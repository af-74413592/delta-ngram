# Old-news training source and temporal isolation

2026-10-19. This document registers the dated news corpus for the new mixed-domain
run. The earlier two-event recent-news smoke test was an execution check, not a
complete news training or evaluation program.

## Source selection and immutable identity

Use [vblagoje/cc_news](https://huggingface.co/datasets/vblagoje/cc_news), revision
`81eb2ce0d2a9dad6ad16b68ef750ec290880fa36`. Its maintained dataset card documents
708,241 English articles published January 2017 through December 2019. Its actual
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

## Fixed time partitions

| Partition | Explicit publication dates | Use |
|---|---|---|
| Train | 2017-01-01 through 2018-12-31 | Writer updates in the mixed run |
| Validation | 2019-01-01 through 2019-06-30 | Fixed loss diagnostics/checkpoint selection |
| Old-news holdout | 2019-07-01 through 2019-12-31 | Reserved corpus-era evaluation |
| Recent-event final test | Separately curated 2025/2026 event families | Frozen writer, new evidence and scored questions |

Malformed, missing or out-of-range publication dates are rejected. A recent
article cannot enter any old-corpus partition. The original full date string is
retained alongside its parsed calendar date. Timezone-free dates remain
timezone-free; the collector does not invent a timezone. The 2019 old-news
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
is **50M training raw tokens**, plus at least 250K raw tokens in each 2019
partition and at least 64 independent groups each. These are **collection
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
python scripts/prepare_news_corpus.py --download-manifest /workspace/delta-ngram-data/multidomain_20261019/downloads/news/download_manifest.json --output-dir /workspace/delta-ngram-data/multidomain_20261019/news --train-candidates 708241 --eval-candidates 708241 --target-train-tokens 50000000
```

Published outputs are `train_validation.jsonl`, `holdout.jsonl`,
`selected_texts.jsonl.gz`, and `data_manifest.json`, plus scan/candidate audit
manifests. Holdout is never present in the trainer file. Each record has source
`news`, the pinned tokenizer fingerprint, original publication metadata, URL,
dataset revision, shard and row provenance, source/text/token hashes and detected
family membership. Existing exported corpora are never overwritten.

Offline verification is in `tests/test_prepare_news_corpus.py`, covering date
boundaries, invalid/recent-date rejection, download integrity and size bounds,
URL identity, deterministic sampling, transitive cross-time quarantine, exact
and near duplicates, retained short-article statistics, full-article encoding,
raw-loader compatibility, token targets and final audit failures.

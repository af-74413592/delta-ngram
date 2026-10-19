# Frozen-writer news evaluation

`news_forward_v1` tests whether a fixed writer can use new, sourced news evidence
at inference time. It supplements raw-document development NLL. It does not
change training or certify that the base model never saw a fact.

## Execution and controls

For every episode, start with independent empty states. Write each history block
through the existing `ForwardMemoryMethod.write`, then ask questions scheduled
after that block. The writer receives only the block's `text`; source metadata,
questions, candidate answers and labels are never writer inputs. A query never
updates the memory. A subsequent query reuses the appropriate written state.
Both the reader and writer are frozen, in evaluation mode, with gradients
disabled; there is no optimizer or backward call. Source files are never training
inputs to this runner. Do not add the manifest or its current-news text to any
training dataset.

Each query has four paired conditions:

| Condition | Memory | Visible input |
| --- | --- | --- |
| `self` | Correct history, through `after_history` | Question and `Answer:` suffix |
| `empty` | Empty | Identical question prefix |
| `donor` | Fixed different event family's history, same number of writes | Identical question prefix |
| `context` | Empty | Correct prior history followed by the same question suffix |

The manifest fixes the donor before scoring. Donors stay in the same split and
real/synthetic kind and must supply enough blocks. Their token lengths need not
match; the result reports actual written token totals. Do not claim a matched
token-budget comparison unless the data construction separately enforces it.
Context is a separate capability reference; it has a longer visible prefix and
is not a memory-only comparison or guaranteed upper bound. Context and evidence
are never silently truncated. Evidence exceeding `max_evidence_tokens` fails.
Before any candidate is scored, every full scoring sequence (context, question
and candidate together) is checked against the model's `max_position_embeddings`
when that metadata is a positive integer no greater than `2**31 - 1`. Oversized
inputs fail; there is no automatic shortening. Missing, malformed or sentinel
metadata produces a recorded null limit, not a verified capacity guarantee.

## Candidate scoring

The prefix is exactly `question + "\nAnswer:"`. Each candidate completion is
exactly one space followed by the candidate text. The existing `answer_batch`
requires `tokenize(prefix + completion)` to begin with `tokenize(prefix)` exactly;
otherwise scoring fails. Every prefix label is masked. Logit `t` predicts token
`t+1`; every completion token, including the leading space when separate, counts
in the answer denominator. No question/context token and no implicit EOS token
enters the loss. All conditions verify identical candidate answer tokens; the
three memory conditions additionally verify identical visible-prefix tokens.

The primary candidate prediction minimizes **mean answer-token NLL**, with first
index breaking exact ties. Candidate sum NLL and length are also saved. Gold NLL
is the sum of gold-answer losses divided by the total gold-answer token count,
not the question count or full-sequence length. Candidate accuracy is the
fraction of questions whose minimum-NLL candidate is the supported answer.
This is candidate ranking, not free-generation accuracy. Balance candidate
length/type and shuffle correct-answer positions during dataset construction;
inspect exact ties and do not use a constant answer position in real evaluations.

Report `self - empty` and `self - donor` gold NLL (negative is better), candidate
accuracy for all conditions, and counts. Results separate real news,
synthetic swaps, and synthetic corrections. Per-event-family results support
later paired family-level uncertainty estimates. This minimal runner does not
compute confidence intervals or a significance gate; do not infer a general
improvement from a tiny pilot or a single fact.

## Split, provenance, and boundaries

- Split by **event family**, including follow-up articles, revised reports,
  translated/mirrored reports, and derived synthetic versions. A single family
  must stay wholly in `dev` or wholly in `final_holdout`.
- Source URLs (ignoring URL fragments) and normalized identical evidence cannot
  cross splits. Human review must still detect URL aliases, paraphrases and
  wrongly assigned event-family labels; lexical checks cannot establish semantic
  independence.
- Register known training event families in `training_event_families`; any
  overlap is rejected. This is an explicit dataset-owner declaration, not an
  exhaustive audit of the base model's pretraining corpus. Current news remains
  evaluation-only even if a family was absent from this declaration.
- Keep a fixed model ID and immutable commit hash, the verified date of that
  revision, and its documented training cutoff or `"unknown"`. The CLI binds
  this identity to the experiment config, checkpoint and resolved model commit.
  Preserve the evidence publication, retrieval and availability dates. Preserve
  the article title, URL, fact subject/relation/value and question date.
- ISO dates have date granularity and mean midnight UTC for comparisons. When
  within-day chronology matters, use complete timezone-aware ISO timestamps;
  offsets are normalized to UTC. Future history, unavailable facts, and queries
  for a superseded fact value are rejected. An optional source `updated_at` must
  lie between publication and retrieval, and evidence using that source cannot
  become available before the update.
- `kind: real` requires `synthetic: false`. Altered facts, swapped values and
  manufactured correction sequences require `synthetic: true`, either
  `synthetic_swap` or `synthetic_correction`, and a `transformation` description.
  Keep the original sources as provenance, but never claim they support altered
  facts. Summarize synthetic results separately from actual news.

Freeze the manifest and its SHA256 before comparing checkpoints/modules. Use
only `dev` to choose changes. Reserve `final_holdout` until the implementation,
candidate construction, checkpoint selection and scoring rule are fixed. The
CLI defaults to `dev`; final evaluation needs the explicit `--split
final_holdout`. This flag records intent; it is not a security boundary and does
not prevent a person from reading a file containing both splits. Store final
materials separately if blindness is required.

## Manifest shape

The following episode is an explicitly fictional schema illustration. A valid
manifest needs a matching different-family donor episode with the same split
and kind. Replace the model identity and source material with verified records;
do not present this example as real news.

```json
{
  "protocol": "news_forward_v1",
  "evaluation_only": true,
  "model": {
    "name_or_path": "model/identifier",
    "revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "revision_date": "2026-01-01",
    "training_cutoff": "unknown"
  },
  "training_event_families": [],
  "episodes": [
    {
      "id": "alpha",
      "event_family": "alpha-event",
      "split": "dev",
      "kind": "synthetic_swap",
      "synthetic": true,
      "transformation": "Fictional color substitution for schema illustration.",
      "donor_episode_id": "beta",
      "sources": [{
        "id": "s1",
        "url": "https://example.org/alpha",
        "title": "Fictional example",
        "published_at": "2026-10-01",
        "retrieved_at": "2026-10-02"
      }],
      "history": [{
        "id": "h1",
        "source_ids": ["s1"],
        "available_at": "2026-10-01",
        "text": "Alpha has color red.",
        "facts": [{"id": "f1", "subject": "Alpha", "relation": "color", "value": "red"}]
      }],
      "questions": [{
        "id": "q1",
        "question": "Which color belongs to Alpha?",
        "after_history": 1,
        "asked_at": "2026-10-02",
        "fact_id": "f1",
        "candidates": ["red", "blue"],
        "answer_index": 0
      }]
    }
  ]
}
```

For a correction, append a later history block with a **new fact ID** and the
same subject/relation. Schedule the old-fact query before the correction and the
new-fact query after it. Gold values must appear verbatim in the corresponding
evidence and equal the selected candidate. This is a deliberately narrow
extractive-fact pilot; source support still requires human verification.

## Commands and artifacts

Use the repository's installed Python environment (`torch`, `transformers`,
`PyYAML`; `pytest` for tests). Validation and unit tests are CPU-only and do not
download a model. Actual evaluation loads the model and fixed writer checkpoint
specified by the existing YAML; it uses that YAML's device.

```text
python scripts/news_eval.py --manifest PATH/news.json --validate-only
python scripts/news_eval.py --manifest PATH/news.json --config PATH/experiment.yaml --checkpoint PATH/writer.pt --split dev --output PATH/news_dev_result.json
python -m pytest tests/test_news_eval.py -q
```

The runner refuses to overwrite a result. It saves the manifest hash, checkpoint
hash/step/training-token metadata, model boundary, config, code revision/dirty
flag, device, per-question candidate scores and denominators, actual write-token
counts, and separated summaries. Preserve the exact manifest beside the result;
its content must match the recorded hash. The checkpoint hash is checked again
after scoring. The writer's complete `state_dict`, including persistent buffers,
is cloned before replay and compared afterward with exact tensor equality
(keys, shapes, dtypes and devices must also match). This catches `.data` changes
and replacement parameters with different values that version counters miss.
Writer and reader parameter versions, frozen flags and absent gradients are
also checked. Reader weights are not copied or content-hashed; its version check
plus the frozen, optimizer-free execution path is a narrower assurance and does
not detect every possible `.data`, buffer or parameter-replacement mutation.
The result records these guarantees separately under `integrity_checks`.
Evaluation never saves changed weights or modifies the existing training code.

The Python API accepts an already-created `ForwardMemoryMethod` for controlled
tests. Calling it without checkpoint metadata is explicitly recorded; only the
CLI performs checkpoint loading and runtime revision verification. Random-model
CPU smoke results verify wiring only and are not evidence of memory quality.

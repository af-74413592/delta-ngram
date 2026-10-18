# delta-ngram

Persistent sparse memory for frozen language models. A learned writer converts
arrived evidence into row targets and gates; a pure update operator changes the
explicit memory. Queries read snapshots through the frozen model.

**Current scope:** an external n-gram hash table with an implicit zero base and
fixed residual injection. The native pretrained-table adapter is **not
implemented**; `reader.backend: native` raises `NotImplementedError`.

## Install and run

Python 3.11 or newer is required. Install a PyTorch build appropriate for the
execution machine, then install the package and development tools:

```sh
python -m pip install -e ".[dev]"
delta-ngram --help
python -m pytest
```

The offline smoke configuration uses a tiny random model on CPU and does not
download pretrained weights. It checks execution, not task quality:

```sh
delta-ngram train --config configs/smoke.yaml
```

The same entry point handles learned writers and baselines:

```sh
delta-ngram train --config configs/pilot_seed17.yaml
delta-ngram eval --config configs/eval_pilot.yaml
delta-ngram diagnose --config configs/diagnose.yaml
delta-ngram eval --config configs/adam.yaml
delta-ngram eval --config configs/sgd.yaml
delta-ngram eval --config configs/read_normalized_sgd.yaml
delta-ngram eval --config configs/context.yaml
delta-ngram eval --config configs/retrieval.yaml
delta-ngram eval --config configs/engram_edit_style.yaml
```

`python -m delta_ngram` accepts the same commands. `scripts/run.py` is only a
thin launcher. Learned-writer `eval` and `diagnose` require `run.checkpoint`;
online baselines use `eval` and perform their evidence-only updates during that
same evaluation loop. `train` starts a fresh outer training run; automatic
training resume is not implemented.

### Interactive memory debugging

```sh
delta-ngram chat --config configs/pilot_seed17.yaml --checkpoint ../delta-ngram-runs/pilot_seed17/writer.pt
```

Each input line is a fresh query with no conversation history. The writer is
frozen, and the answer is generated from the current memory before any write.
`/write off` is the default. `/write on` writes the user input after answering;
`/write all` then also writes the raw answer as a separate block prefixed with
`Assistant said: `. Each write reports the number of touched physical rows and
new rows. `/reset` clears memory; `/quit` exits.

The existing `evaluation.max_new_tokens` controls greedy answer length (3 in the
pilot preset), and `run.max_evidence_tokens` still limits each write. The chat
does not create an experiment output directory. Checkpoint compatibility uses
the same strict checks as evaluation.

## Repository structure

```text
configs/                         YAML examples, with no model weights or results
scripts/run.py                   thin CLI launcher
src/delta_ngram/
  state.py                       sparse state and versioned snapshot persistence
  operators.py                   pure C and D0-D4 row-update operators
  writer.py                      evidence-conditioned targets and gates
  reader.py                      frozen reader and address/read adapter boundary
  interfaces.py                  MemoryMethod and WriteResult contracts
  methods.py                     method construction and shared write entry point
  runtime.py                     tokenization, generation, scoring, and timing
  train.py                       one outer training loop
  evaluate.py                    one evaluator and shared row interventions
  cli.py                         train / eval / diagnose
  checkpoints.py                 writer checkpoint schema and legacy migration
  parity.py                      exact historical-artifact comparison
  data/                          fact/affix episodes, splits, and manifests
  baselines/                     online optimizers, context, BM25, and ridge
tests/                           CPU correctness and parity checks
README.md
MANIFEST.in
pyproject.toml
```

The address/read adapter lives in `reader.py`. It is independent of the write
operator; changing an operator does not create another trainer or evaluator.

## State and read/write boundaries

`SparseMemoryState` stores sorted positive physical row IDs, FP32 row patches
`values`, FP32 block observation `counts`, and optional FP32 `precision` for D3.
Row 0 denotes padding or an incomplete n-gram and is never stored. Unseen rows
read as zero. Updates return new tensors and preserve earlier readable snapshots.

Only arrived evidence enters `write(state, evidence)`. For learned forward
writers, the frozen model encodes that evidence without dynamic patches. The
writer receives actual LM-head input
features, a within-evidence right-context mean, a learned evidence pool, and a
head embedding. Targets depend on evidence; D1-D4 gates also inspect the old row,
target residual, and `log(1 + count)`. D0 gates receive evidence features only.

The external reader adds the sum of the addressed row patches, divided by
`sqrt(number_of_orders)`, before the configured decoder layer. Queries use fresh
full-prefix forwards with no KV cache and never write memory. Activation
checkpointing is disabled to keep query graphs tied to their own state snapshots.

Only writer parameters are optimized in outer training. Parameters stay fixed
through each evidence trajectory; an outer update occurs after the batch of
independent trajectories. The current loop uses full backpropagation through the
trajectory; truncated backpropagation is not implemented. At evaluation, the
learned writer is frozen. Text-context baselines instead retain evidence text and
materialize their context only in `render_prompt`.

## Pure update operators

For the allowed events addressing physical row $j$ in one evidence block, define

$$
n_j=|\mathcal A_j|,\qquad
A_j=\frac{1}{n_j}\sum_{a\in\mathcal A_j}\beta_a v_a,\qquad
w_j=\frac{1}{n_j}\sum_{a\in\mathcal A_j}\beta_a.
$$

All events use the same old snapshot. The denominator counts every allowed
event, including events with a zero gate; it is **not** the sum of gate weights.
For every touched allowed row, $c_j^+=c_j+1$ once per block, including when all
gates are zero. Unobserved rows remain unchanged. The formulas use the old $c_j$.

| Operator | Row update | Distinction |
| --- | --- | --- |
| **C**, `DIRECT`, `OVERWRITE` | $\Delta_j^+=\frac{1}{n_j}\sum_a v_a$ | Unweighted target-mean overwrite; ignores $\beta$. |
| **D0** | $\Delta_j^+=\Delta_j+A_j-w_j\Delta_j$ | Gate uses evidence only. |
| **D1** | $\Delta_j^+=\Delta_j+A_j-w_j\Delta_j$ | Gate also uses old row, target residual, and count. |
| **D2** | $\Delta_j^+=\Delta_j+\frac{A_j-w_j\Delta_j}{1+\kappa c_j}$ | Observation-count decay, $\kappa\ge0$. |
| **D3** | $q_j^+=q_j+w_j$, $\Delta_j^+=\Delta_j+\frac{A_j-w_j\Delta_j}{q_j^+}$ | Scalar RLS, unseen-row prior $q_j=\lambda_q>0$. |
| **D4** | D1 on $j\notin\mathcal F$; unchanged on $j\in\mathcal F$ | Excludes fixed physical rows, including their counts and precision. |

D3 adds the **block mean gate** $w_j$ to precision, not the sum of all gates.
`method.precision_init` sets $\lambda_q$. D4's frozen set is fixed before a run,
using positive `frozen_rows` and/or addresses of `frozen_template_texts` under the
configured tokenizer and hash. Hash collisions can therefore freeze additional
content sharing those physical rows. It does not select rows using query labels,
answer correctness, gate size, or measured update magnitude.

These are operator ablations: each learned variant should be trained under its
own operator. Switching formulas on an existing D1 checkpoint is not a separately
trained method comparison, and checkpoint compatibility checks reject mismatches.

## Baselines and their exact scope

All online gradient methods use the same mean causal next-token loss on the
current evidence block $E=(x_1,\ldots,x_T)$:

$$
L_E=-\frac{1}{T-1}\sum_{t=1}^{T-1}\log p(x_{t+1}\mid x_{\le t};\Delta).
$$

No future query or scoring answer enters the updater. The full arrived evidence,
including its supplied fact or support answer, is legitimate input.

| `method.name` | Update or query behavior |
| --- | --- |
| `empty` | No writes; empty explicit memory. |
| `adam` | Real PyTorch `SparseAdam`, restricted to current-evidence rows. Moments and scalar step survive row expansion and later blocks in one trajectory. |
| `sgd` | Momentum-free SGD on $G_j=\partial L_E/\partial\Delta_j$: $\Delta_j\leftarrow\Delta_j-\eta G_j$. |
| `read_normalized_sgd` | $\Delta_j\leftarrow\Delta_j-\eta G_j/n_j$, where $n_j$ counts current-block physical-row occurrences. The token mean in $L_E$ is **not** undone. |
| `context` | Concatenate the complete evidence history before the query. |
| `retrieval` | BM25 over stored evidence blocks, retrieve top $k$, then render those blocks in arrival order. Ties use the original arrival index. |
| `engram_edit_style` | Evidence-only optimized-residual target followed by a reuse-penalized ridge solve, described below. |

`method.online_steps` repeats the same evidence loss/update loop. Observation
counts advance once for the evidence block, not once per optimizer step. Adam
masks both moment and value updates to observed rows, including observed rows
whose current gradient is zero. All optimizer state resets between independent
trajectories. Old gradient-method endpoints remain readable; writing an old or
diagnostic snapshot is rejected instead of silently branching shared moments.

### EngramEdit-style variant

This implementation is **not an exact reproduction of EngramEdit**. It first
optimizes a zero-initialized per-token FP32 extra residual $R$ at the same decoder
injection point, with the old row state fixed:

$$
R^*\approx\arg\min_R L_E(\Delta,R)
+\lambda_R\operatorname{mean}(R^2).
$$

This uses a fresh Adam optimizer per evidence block with `target_steps`,
`target_lr`, and `target_l2`; it includes online gradient computation. It then
constructs $X$ from the actual current-evidence row reads, coalescing repeated
physical rows with the same `1/sqrt(number_of_orders)` read coefficients, and
solves for row increments:

$$
D^*=\arg\min_D\|XD-R^*\|_F^2
+\lambda\sum_j(1+\rho c_j)\|D_j\|_2^2,\qquad
\Delta_j^+=\Delta_j+D_j^*.
$$

Here `ridge_lambda` is $\lambda>0$, `reuse_penalty` is $\rho\ge0$, and $c_j$ is
the old block count. Unobserved rows remain unchanged; touched rows advance their
count once after solving. Target-optimization and solve/update times are reported
separately. Finite target optimization and ridge fitting do not guarantee that
the resulting table attains the optimized token-residual loss.

## Configurations and data

YAML configurations use schema `version: 1` and merge with defaults in
`config.py`; unknown keys are rejected. There is no `extends` key. Model revision,
tokenizer/addressing settings, operator settings, and data manifest must stay
consistent when evaluating a checkpoint.

| Example | Command | Purpose |
| --- | --- | --- |
| `smoke.yaml` | `train` | Offline CPU plumbing check. |
| `pilot_seed17.yaml` | `train` | Two-fact D1 run, seed 17, 500 outer steps. |
| `eval_pilot.yaml` | `eval` | Evaluate that saved writer. |
| `diagnose.yaml` | `diagnose` | Saved D1 writer with fixed-address row interventions. |
| `direct.yaml`, `d0.yaml`, `d2.yaml`, `d3.yaml`, `d4.yaml` | `train` | Separately train each operator ablation. |
| `adam.yaml`, `sgd.yaml`, `read_normalized_sgd.yaml` | `eval` | Evidence-only online gradient baselines. |
| `context.yaml`, `retrieval.yaml`, `engram_edit_style.yaml` | `eval` | Text baselines and the scoped ridge variant. |
| `affix.yaml`, `mixed.yaml` | `train` | Synthetic affix tasks or episode-level fact/affix mixing. |

Facts use disjoint training/validation entity pools and newly sampled worlds.
Affix episodes split rule-family and stem pools before generation. Mixed data
chooses one task type for an entire episode, with the current generator's fixed
default fact probability of 0.5; the YAML schema does not expose a mixing-weight
key. `facts_per_episode` sets the number of facts or rule-bearing blocks, and
`support_examples` applies to affix blocks.

Set `data.manifest` to replay a saved validation manifest. Its task, episode count,
facts per episode, and applicable support count must match the YAML. Loading also
checks split disjointness and each validation record's membership and task type.
The manifest records split pools, seeds, and validation
worlds; metadata is not passed to the writer. The two-fact compatibility path
retains the historical manifest representation and random-number ordering.

## Evaluation and diagnostics

The shared evaluator reports actual cache-free greedy generation, raw outputs,
first-word correctness, strict first-line correctness, immediate acquisition,
final correctness, conditional retention with its denominator, and the
all-sample acquired-and-retained fraction. A short answer training target does
not supervise its eventual continuation or stopping format, so the two scoring
rules should be read separately.

Controls clear explicit state and substitute wrong-evidence values on the
original row addresses/counts. For table methods, `diagnose` selects

$$
\mathcal S=\text{first-state rows}\cap\text{old-query prefix rows}
\cap\text{later-evidence candidate rows}.
$$

It zeros this fixed group at the first endpoint, and restores its first-endpoint
values at the final endpoint, then queries both old and new items. Selection
never filters by nonzero updates or successful answers. Group-level effects do
not establish a causal effect for every individual shared row. These row
interventions do not apply to text-context methods.

Retention comparisons must report new-task acquisition alongside retention;
lower forgetting with weaker acquisition is not an acquisition-matched advantage.
Repeatedly inspected validation worlds remain development evidence.

## Versioned state and checkpoints

Table snapshots use format `delta-ngram-state`, `schema_version: 1`. Saving
detaches and copies tensors without cutting the live training graph. Loading
requires exact agreement with a caller-supplied identity containing all six
nonempty fields:

| Identity field | Record |
| --- | --- |
| `base_model` | Model identifier and pinned revision. |
| `tokenizer` | Tokenizer identifier/revision and relevant configuration. |
| `addressing` | Backend, hash version, buckets, orders, and injection location. |
| `writer` | Writer/checkpoint identity or baseline configuration. |
| `operator` | Operator name and all update settings. |
| `trajectory` | Trajectory identity and endpoint. |

```python
from delta_ngram.state import SparseMemoryState

# identity is the complete, JSON-compatible provenance of this state.
state.save("../delta-ngram-runs/session-state.pt", identity=identity)
restored = SparseMemoryState.load(
    "../delta-ngram-runs/session-state.pt",
    expected_identity=identity,
    device="cpu",
)
```

The caller supplies provenance; the state loader checks it rather than inferring
it from a model object. These snapshots contain row values, counts, and optional
precision, **not** writer weights or online optimizer moments. They are not a
complete optimizer-resume checkpoint.

Writer checkpoints separately use format `delta_ngram_writer`,
`format_version: 1`, with writer parameters, step, configuration, and an optional
outer optimizer state. The Python loader requires `allow_legacy=True` to accept
the historical D1 format; the CLI accepts the versioned package format. Legacy
migration checks the model, seed for random models, addresses, dimensions, and
the historical fixed right-context window; it does not reinterpret it as another
operator or a native-table backend.

## Output locations and accounting

Keep generated artifacts outside this repository. The example paths resolve to
the sibling `delta-ngram-runs/` directory. Relative `run.output_dir`,
`run.checkpoint`, and `data.manifest` paths resolve against the **YAML file's
directory**, with environment-variable and home-directory expansion. A run
requires a new or empty output directory; change its name before repeating it.

Training writes `config.json`, `data_manifest.json`, `metrics.jsonl`,
`writer_init.pt`, `writer.pt`, endpoint `eval_step_*.json`, and `summary.json`.
Evaluation/diagnosis writes metadata, manifest, evaluation records, metrics, and
a summary. Weights, downloaded models, historical results, and generated reports
are not source-package contents. The source distribution includes small golden
tensor and data fixtures with recorded provenance, not trained-model checkpoints.

Table and optimizer tensor bytes are reported separately. Gradient-method
endpoint copies are disclosed separately from the historical table-plus-optimizer
accounting. Context/retrieval report retained UTF-8 text bytes; a zero table-byte
count does not mean a text baseline uses no storage. Offline outer training,
online evidence updates, and query costs must also remain separate.

## E1 fact versions and question transfer

`data.task: fact_e1` uses manifest schema 2. Each event has an audit-only
`(entity, attribute)` key, value, and acquisition/revision kind. Only rendered
evidence text enters the writer. `data.e1_scenario: independent` introduces
distinct entities; `revision` starts with A.color, A.fruit, then an explicit
update of A.color, followed by other entities. `facts_per_episode` counts writes.
Legacy fact/affix manifests and their RNG order remain unchanged.

Use `evaluation.query_styles: [canonical, named_question]` to probe the same
state with the original completion and a question retaining the entity name.
Evidence is written once per stage regardless of the number of query styles.
Generation, coverage, and first-word/strict scores are recorded separately for
each style. Query changes never consume generator RNG or enter writer training.

E1 queries include the newest event and all currently active older fact versions.
An explicitly superseded version has `CT: null` and is excluded from retention;
its old answer is not a valid target after revision. `query_style_metrics` reports:

- First-fact endpoints, with the number of superseded first facts excluded.
- All final active versions, plus a separate subset with later intervening writes.
- Revision immediate/final success and persistence of the previous value.
- Same-entity other-attribute accuracy immediately before/after revision, including
  the loss rate among previously correct attributes and its denominator.

Outer training uses the existing loop. For E1 only, it averages query losses
within acquisition, retention, and revision, then equally averages the nonempty
ability groups. Legacy training retains its original arithmetic order. No
truncated backpropagation is implemented yet; the 64/256-write generator tests
do not establish feasible training or scientific performance at those lengths.

The `e1_transfer_*.yaml` presets freeze the two-write D1 checkpoint and test
paraphrases, 4/16 writes, and revision. The 16-write pilot uses four worlds to
bound this exploratory run; other transfer presets use sixteen. They are
distribution-transfer probes, not writers trained on E1 or independent final
tests. `e1_train_revision4.yaml` is a separate example for E1 outer training.
An evaluation checkpoint must match its original model/reader/writer/method
identity; configure the exact model path used during training.

Set both `evaluation.row_diagnostics: true` and `detailed_rows: true` for token
provenance and grouped interventions on the original fact template. Selection is
the first written rows intersecting the old query prefix and later evidence;
correctness and update magnitude never select rows. Groups distinguish template,
entity-internal, entity-boundary, answer, literal shared n-grams, and hash
collisions between distinct n-grams. Groups can overlap; effects are not additive.
For a superseded first fact, a retention intervention is explicitly unsupported.
The detailed token classifier currently supports the original fact template;
E1 still records actual read/write coverage and supported shared-row diagnostics.

Full E1 overlap strata, entity-token-length balancing, indirect references,
long-trajectory truncated training, and native Qwen replication remain future
experiments. The transfer pilot cannot establish those claims.

## Build distributions

Build a wheel and source distribution into a sibling artifact directory:

```sh
python -m build --sdist --wheel --outdir ../delta-ngram-dist
```

For an environment with its build dependencies already installed, add
`--no-isolation`. The wheel contains the installable Python package and CLI;
the source distribution additionally includes README, example YAML files, the
thin launcher, and tests through `MANIFEST.in`. Top-level example configs are
not installed as wheel package resources. Obtain them from the source checkout
or source distribution.

To check a built wheel in a separate environment, install its concrete filename
and run `delta-ngram --help`. Building artifacts does not upload them to a package
index. Version, dependency constraints, and entry-point metadata are maintained
in `pyproject.toml`.

## Validation status

Compare a completed historical pilot and the refactored run with:

```sh
python -m delta_ngram.parity --reference ../historical-pilot --candidate ../delta-ngram-runs/pilot_seed17 --out ../delta-ngram-runs/parity.json
```

The default contract requires seed 17, all 500 training steps, six complete
evaluations, the entire manifest, and every initial/final writer tensor. It uses
exact equality with no numeric tolerance. It compares raw generated answers and
all original scores, losses, and gradient norms. Runtime durations, allocator
peaks, paths, envelope metadata, and additional new diagnostics are excluded and
listed in the report. Missing or unequal required artifacts return a nonzero exit.

CPU tests cover state invariants and persistence, operator equations, evidence
boundaries, shared evaluation behavior, optimizer migration, and numerical parity.
The online Adam refactor has also been compared against the archived executed
implementation on CPU: two blocks of ten updates, at learning rates 0.01 and
0.05, with bitwise agreement in tables, counts, losses, and Adam state.

**Passed on 2026-10-18:** the full seed-17 pilot was rerun from scratch on the
same RTX 4090, Qwen3-0.6B-Base weights, BF16 reader, and PyTorch 2.14.1+cu132.
The refactor at commit `250957b` matched the archived execution exactly:

- 500 training records (2,000 sampled training episodes).
- All six evaluations at steps 0/100/200/300/400/500, including raw predictions.
- All 20 initial/final writer tensors and 9,037 compared scalar/string fields.
- Identical manifest; **zero mismatches and zero numeric tolerance**.

The machine-readable report is `parity-250957b.json` in the external acceptance
bundle. Later changes at `b937be9` tighten checkpoint and manifest input checks;
the outer loop, runtime construction, checkpoint writer, and data sampling
functions retain identical syntax trees. Package and fixture changes are separate
from that numerical implementation.

The later code also loaded the step-500 checkpoint through the strict loader and
replayed the same manifest with `diagnose`: all original evaluation fields still
matched exactly, and row diagnostics completed for all 16 worlds. The associated
report is `release-verification-b937be9.json`; tracked runtime sources were checked
byte-for-byte against their committed archives.

All **148 CPU tests** also pass after extracting the source distribution into an
independent directory, without the old repository or skipped tests. The wheel
installs and executes the CLI from a separate environment. This validation covers
the refactor; native-table replication and broader multi-seed scientific claims
remain outside its scope.

### Development comparison and E1 transfer (2026-10-18)

The same sixteen two-write development worlds were evaluated with the same
Qwen3-0.6B-Base reader. Counts below use the primary first-word score. `First`
is Q1 after E1, `Retained` is Q1 after E2, and `New` is Q2 after E2.

| Method | First / 16 | Retained / 16 | New / 16 |
| --- | ---: | ---: | ---: |
| D1, step 500 | 16 | 16 | 15 |
| Online SparseAdam, 0.01 x 10 | 16 | 2 | 16 |
| Online SparseAdam, 0.05 x 10 | 16 | 4 | 16 |
| Full evidence context | 16 | 16 | 14 |
| BM25 retrieval, top 2 | 16 | 16 | 14 |
| D4, step 500 | 6 | 6 | 9 |

D4 was independently trained from the identical initialization for the same
500-step budget, with ten template rows frozen before training. Its strict
first-line counts at step 500 are only 2/2/2; both color errors and generation
format drift occur. Its primary counts were 16/16/16 at step 400, so the full
curve is required to show its instability. The final checkpoint was fixed at
500 rather than selected from these development results. With only two evidence
blocks, top-2 retrieval retains both, and all 136 recorded predictions exactly
match full context.

The earlier D1 `1.0 -> 0.875` retention result is at **step 400**: 16 first
answers correct, then 14 retained. A recovered run matches the original first
400 training records and five evaluations exactly. Restoring the ten literal
template rows at its second write rescues both lost old answers (14 -> 16),
while harming new answers (12 -> 2). Restoring only the name-boundary rows
rescues neither loss. No name-internal rows or distinct-ngram hash collisions
were selected in these sixteen worlds. These are interventions on row groups;
they do not establish the necessity of every individual row.

The E1 implementation is covered by **245 passing CPU tests**, also run from
an independently extracted source distribution. Its canonical diagnostic
replays match **9,873 historical non-timing fields** across D1 step 400/500,
D4 step 500, and both Adam settings, with zero differences. The newer tests
extend the historical 148-test refactor validation above.

The frozen step-500 writers were then transferred to E1 without retraining.
Final active-fact counts below are `canonical / named_question`; each cell has
64 facts. Four-write probes use sixteen worlds; sixteen-write probes use four.

| Method | 4 writes, final / 64 | 16 writes, final / 64 |
| --- | ---: | ---: |
| D1 | 60 / 3 | 19 / 5 |
| D4 | 26 / 5 | 25 / 6 |
| Full evidence context | 61 / 61 | 62 / 47 |
| BM25 retrieval, top 2 | 64 / 64 | 60 / 64 |

On the original two-write worlds, D1's first-fact accuracy under the named
question stays at its empty-state 2/16 before and after writing, versus 16/16
under the training template. In the sixteen-write probe, canonical D1 acquires
46/64 facts immediately but answers only 19/64 at the end. Actual post-write
named-question prefixes have about 6% written-row coverage, with no zero-coverage
queries; reduced overlap alone is not a causal explanation of the failure.

In sixteen revision worlds, canonical D1 follows the revised value immediately
in 5/16 cases. All superseded versions are excluded from retention (48 active
facts remain). Full-context named questions follow 15/16 revisions immediately
but only 5/16 after the last unrelated write. Retrieval named questions retain
15/16, while canonical retrieval selects the old color and fruit evidence,
excluding the update in all sixteen worlds. Its query-style comparison changes
the retrieved evidence as well as the wording. Full prompt/selection audits are
preserved with the results.

All 22 campaign jobs completed, including a separate two-step E1 training
integration check with nonzero writer gradients. That check is not an E1-trained
performance result. These one-seed development probes demonstrate template and
trajectory-length limits; they do not establish broad or native-table memory
performance. Executed source snapshots, manifests, checkpoints, raw predictions,
learning curves, and validation reports are kept in the external experiment
archive rather than tracked as repository weights.


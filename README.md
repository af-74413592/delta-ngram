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

**Pending acceptance:** the standalone package's full seed-17, 500-step GPU run
and its exact artifact comparison with the historical run. The historical pilot
does not by itself certify this refactor. Do not treat a short CPU smoke run,
successful package build, or earlier pilot metrics as completion of that check.
Native-table replication and broader multi-seed scientific comparisons are also
outside the current implementation validation.


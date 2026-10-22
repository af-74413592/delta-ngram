# Runtime and source map

The [assessment](research/qwen38_delta_writer_assessment_20261022.md) records
the seven-task chains, H200 representation measurements, seen-question recall,
and LoCoMo generation and reference-answer losses. Linked JSON files retain
their original run identities and absolute paths. The publication manifest
records each source file and its published copy.

## Source locations

| Component | Repository location |
|---|---|
| Native PLE, writers and state handling | `src/delta_ngram/` |
| Native training, preparation and compatibility helpers | `scripts/native_qwen38/` |
| A800 live memory worker and two-chain runner | `scripts/qwen38_vllm_live_worker.py`, `scripts/qwen38_vllm_live_two_chain.py` |
| Generated-answer scoring | `scripts/qwen38_score_generated.py`, `scripts/qwen38_accuracy_eval.py` and their metric helpers |
| LoCoMo short prompts and memory states | `scripts/qwen38_locomo_native.py`, `scripts/qwen38_locomo_delta*.py` |
| LoCoMo full-context reader | `scripts/qwen38_locomo_full_context_v3.py`, `scripts/qwen38_locomo_prefetch_worker.py` |
| LoCoMo reference-answer NLL | `scripts/qwen38_locomo_nll*.py` |
| Explicit NLL method amendment | `scripts/launch_amended_nll.py`, `docs/experiments/qwen38_native_20261020/locomo_nll_protocol_amendment_20261022.json` |
| H200 writer timing runner | `scripts/qwen38_h200_writer_timing.py` |
| Two modified vLLM files and application instructions | [third_party/vllm](../../../third_party/vllm/README.md) |

The staged runtime sources match their recorded bytes. Existing native package
sources and the native preflight/frozen-session helpers are reused from this
repository. The source mapping records their SHA256 identities.

## Environment and path setup

The recorded vLLM environment uses **vLLM 0.30.0**. Check the two original source
hashes before applying the two replacements; the version string alone is not
sufficient. The vLLM guide identifies the corresponding adapters and binding
variables. GPU drivers, compiled vLLM dependencies, model weights and tokenizer
assets must be supplied by the execution environment.

For the Linux runners, make all three source locations importable:

```sh
export PYTHONPATH="/path/to/delta-ngram/scripts:/path/to/delta-ngram/scripts/native_qwen38:/path/to/delta-ngram/src${PYTHONPATH:+:$PYTHONPATH}"
```

Where supported by a runner, point `--delta-repo` at this repository and
`--formal-scripts` at its `scripts/` directory. Consult that runner's saved launch
receipt for the actual arguments and environment.

Original run bindings deliberately check absolute paths, source hashes,
checkpoint identities and protocol names. A source checkout alone does not
recreate those paths or the recorded experiment. Restore the files to the
recorded locations to use the historical commands. A relocated experiment
requires its own binding and registration; preserve the original receipts.

The amended NLL launcher also checks its own location. Its published filename
is a source reference: the recorded launch uses `launch_amended_nll.py` and
`protocol_amendment.json` under
`/mnt/disk2/wuhao/qwen38_locomo_nll_20261022_amended_v1`, together with the
unchanged original runtime and preserved failed diagnostic under
`/mnt/disk2/wuhao/qwen38_locomo_nll_20261022_v3`.

## Scoring dependencies and scope

The HumanEval WASI route uses
`evaluation/humaneval/requirements-wasi.txt` and `wasi_worker.py`; the pinned
CPython WASM URL and SHA256 are in `scripts/evaluate_humaneval_wasi.py`.
The binary and benchmark inputs are external assets.

The exported math and HumanEval helper files retain some historical
model-loading CLI branches. Those branches were outside the recorded scoring
path and their legacy `cl_research` training dependencies are not included in
this source closure. The selected metrics import individual modules from
`src/cl_research`; the scientific relative-import branch in `niche_metrics`
requires a package import context and was not used by the recorded
`inflection_exact` scorer.

## Interpretation and preservation

The NLL amendment preserves the failed cross-shape diagnostic and changes the
admission decision, with explicit user authorization. It does not change the
formal loss computation or claim numerical equivalence to fresh-prefix
generation. Read the assessment and amendment together.

The Git report package contains the sources, figures and selected evidence
linked by the reports. Large state tensors, raw captures and full result
collections are addressed by the B2 backup inventory and restore records.
Cloud verification is established by the backup's completion receipts.

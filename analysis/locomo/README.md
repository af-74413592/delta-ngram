# Recompute the LoCoMo reports

These eight Python files are byte-identical copies of the recorded scoring and
analysis sources. `source_manifest.json` records their identities and original
run receipts. The three analysis entry points use NumPy; plotting also requires
Matplotlib. They read saved results and do not run the language model.

Restore the full LoCoMo result directory from the B2 inventory to supply the per-question
generation scores and raw NLL rows. In the commands below, `R` is the restored
`artifacts/qwen38_locomo_20261022` directory and `A` is this analysis directory.
Use fresh output directories. Run these commands in a POSIX shell:

```sh
R=/path/to/restored/artifacts/qwen38_locomo_20261022
A=/path/to/delta-ngram/analysis/locomo

python "$A/render_generation_report_v2.py" \
  --source "$R/four_condition_results_v2/report.json" \
  --source-sha256 d539835cb8f9cfd2a5ff60c3989e58b1721114ed2c4568428c3c08c22a1ce234 \
  --per-question "$R/four_condition_results_v2/official_scores/per_question.jsonl" \
  --per-question-sha256 10a92b2f9d366277e3b54ab3d41a3aecfdc91e0c53b70e8985df0e132f550f78 \
  --output ./generation_recomputed

python "$A/render_nll_report.py" \
  --summary "$R/nll_amended_v1/essential_v1/run_v1/summary.json" \
  --summary-sha256 6d6ebc4d65cc53cdc4d6ceb48566a46ed17f3cfcf84103015962d39eeab13133 \
  --rows "$R/nll_amended_v1/essential_v1/run_v1/nll_results.jsonl" \
  --rows-sha256 5179e5fbbfa02a4d6ec75410cec69df5d4688d6c1b457057680960724d8437ed \
  --output ./nll_recomputed

python "$A/analyze_nll_positions.py" \
  --summary "$R/nll_amended_v1/essential_v1/run_v1/summary.json" \
  --summary-sha256 6d6ebc4d65cc53cdc4d6ceb48566a46ed17f3cfcf84103015962d39eeab13133 \
  --rows "$R/nll_amended_v1/essential_v1/run_v1/nll_results.jsonl" \
  --rows-sha256 5179e5fbbfa02a4d6ec75410cec69df5d4688d6c1b457057680960724d8437ed \
  --protocol "$R/nll_position_analysis_v1/protocol.json" \
  --output ./nll_positions_recomputed
```

The generation renderer independently checks 42 intervals. The NLL renderer
checks raw losses and 12 total-loss intervals; the position analysis adds 24
first/continuation intervals using the same paired conversation draws.
Plot image bytes can vary with plotting-library versions; compare the audited
numeric values and input hashes.

## Original CPU scorer

`scoring/assemble_and_score.py` is the recorded assembly and provenance-checking
wrapper. `scoring/scorer.py` computes the category-aware F1/refusal scores and
paired bootstrap. Its `upstream_v1/` directory contains exactly the three
pinned source files used for AST extraction: the selected official scoring
functions and two NLTK 3.8.1 classes. Keep that directory beside `scorer.py`.

The recorded scorer used NumPy 2.4.6 and regex 2026.9.29. It does not import
the full upstream evaluation module, BERTScore or a complete NLTK package.
The original assembly also requires the restored model/tokenizer assets,
sealed input bindings and runtime source locations described in
[native runtime guide](../../scripts/native_qwen38/README.md). Its original absolute-path and hash checks remain
intact. Consult the saved execution receipts before replaying assembly.

# delta-ngram

A research package for persistent sparse memory in frozen language models.
Only the forward writer is trained in the outer loop. Evidence updates explicit
memory; queries read snapshots without retaining ordinary context or KV caches.

The current reader uses an external, implicitly zero n-gram table and fixed
residual injection. Native pretrained Qwen table integration is a future backend.

## Install

```sh
python -m pip install -e '.[dev]'
delta-ngram --help
```

## Layout

- `state.py`: immutable sparse states and versioned persistence.
- `operators.py`: pure overwrite and D0–D4 updates.
- `writer.py`, `reader.py`: trainable write proposals and frozen read adapters.
- `data/`, `baselines/`: episode generators and interchangeable methods.
- `train.py`, `evaluate.py`: the shared training and evaluation implementations.
- `cli.py`: YAML-configured `train`, `eval`, and `diagnose` commands.

Runtime output belongs outside the repository or under ignored output folders.
The historical pilot and its artifacts remain outside this repository.


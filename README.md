# Delta N-gram

Forward-only persistent memory for frozen language models.

## Method

The model reads a frozen pretrained N-gram table plus a sparse patch: **T = T⁰ + Δ**.
A small writer turns each evidence block into row targets and scalar gates.
D1 averages writes to the same row, keeping each update a convex combination
of the old patch and its targets.

The writer is trained offline with next-token prediction. At use time, writing
requires only forward passes. The **plain writer** encodes evidence without
memory; the **delta writer** encodes it with the current memory.

## Models and data

- [Writer checkpoints](https://huggingface.co/haowu11/Delta-Ngram):
  Qwen3.8 plain/delta writers and the Qwen3-0.6B prototype.
- [Training and evaluation data](https://huggingface.co/datasets/haowu11/Delta-Ngram):
  five-source corpus, DEV64 and the seven-task benchmark.

Both repositories currently require Hugging Face access permission.
Checkpoints contain writer weights; download the matching base model separately.

## Run

Python 3.11+ and PyTorch are required. From this checkout:

```sh
python -m pip install -e .
delta-ngram train-raw --config configs/raw_smoke.yaml
```

The smoke run uses a tiny CPU model and needs no pretrained weights.

| Task | Entry point |
| --- | --- |
| 0.6B raw-text training | `delta-ngram train-raw --config configs/m1_five_source_seed17.yaml` |
| Prototype evaluation | `delta-ngram eval --config configs/eval_pilot.yaml` |
| Qwen3.8 training and inference | [Native runtime guide](scripts/native_qwen38/README.md) |
| vLLM reader setup | [vLLM modifications](third_party/vllm/README.md) |
| LoCoMo result analysis | [Analysis commands](analysis/locomo/README.md) |

Set model, data and output paths in the chosen configuration; evaluation also
requires `run.checkpoint`. The native Qwen3.8 path uses a separate GPU runtime.
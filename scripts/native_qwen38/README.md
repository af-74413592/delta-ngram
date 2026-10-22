# Qwen3.8 native runtime

## Environment

- Base model: `Qwen/Qwen3.8-Flash-Next`, revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Training: the pinned SWIFT/Megatron environment in [ops](ops/).
  The four-GPU training layout is TP1/PP1/EP4/ETP1/DP4/CP1.
- vLLM inference: [vLLM 0.30.0 setup and source replacements](../../third_party/vllm/README.md).

Install the package from the repository root and make the native helpers
importable on Linux:

```sh
python -m pip install -e ".[data]"
export PYTHONPATH="$PWD/scripts:$PWD/scripts/native_qwen38:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## Entry points

| Task | Script |
| --- | --- |
| Prepare native-tokenizer data | [prepare_qwen38_native_data.py](../prepare_qwen38_native_data.py) |
| Train the plain writer from initialization | [qwen38_native_train_staged_alternating.py](qwen38_native_train_staged_alternating.py) |
| Continue the plain writer to 100M tokens | [qwen38_native_train_staged_alpha100.py](qwen38_native_train_staged_alpha100.py) |
| Train with memory-aware encoding | [qwen38_native_train_memory_fork.py](qwen38_native_train_memory_fork.py) |
| Seven-task forward writing | [qwen38_seven_task_forward_writer.py](qwen38_seven_task_forward_writer.py) |
| DEV64 evaluation | [qwen38_native_dev64_eval.py](qwen38_native_dev64_eval.py) |
| vLLM memory worker | [qwen38_vllm_live_worker.py](../qwen38_vllm_live_worker.py) |
| LoCoMo reference-answer NLL | [qwen38_locomo_nll.py](../qwen38_locomo_nll.py) |

Training launchers require their checkpoint/input manifests, runtime identities
and output paths. Use each launcher's `--help` for its arguments. Existing
experiment launchers retain their original path and checksum bindings.

The Hugging Face export contains `writer.safetensors`, not the resumable joint
training checkpoint expected by the training and seven-task launchers.
Its `runtime/load_native_writer.py` reconstructs a writer on CPU; full model
inference also needs the matching reader, tokenizer and memory adapter.

## Implementation

- [qwen38_writer.py](../../src/delta_ngram/qwen38_writer.py): targets and scalar gates.
- [qwen38_adapter.py](../../src/delta_ngram/qwen38_adapter.py): sparse patch at native lookup.
- [qwen38_training.py](../../src/delta_ngram/qwen38_training.py): writer gradients through the write chain.
- [qwen38_precision.py](../../src/delta_ngram/qwen38_precision.py): split BF16 base / FP32 patch readout.

Store checkpoints, data and run outputs outside the source checkout.
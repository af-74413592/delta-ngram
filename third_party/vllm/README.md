# vLLM 0.30.0: two modified Python files

This directory contains the two source replacements used by the recorded
`live_v3` Qwen3.8 runtime, plus their provenance. The installed vLLM distribution
and its compiled dependencies are supplied by the execution environment.

## Compatible inputs and exact outputs

Use **vLLM 0.30.0** and check both original file hashes below. A matching version
string alone does not identify the required source bytes. Paths are relative to
the installed `vllm/` package directory.

| File | Original SHA256 | Modified SHA256 |
|---|---|---|
| `models/qwen4_exp/nvidia/model.py` | `2d9d98054dfa4e2c2c434526ef6fbdceede06b35c38a8ea6dccb42161a666dfb` | `5c5d4bbe41222c846f50faf9075f45eebb84880e45eac835634948928bd260f1` |
| `models/qwen4_exp/nvidia/ple_layer.py` | `b81f9372eca168157c7f3cf5646d61eb0f139b64c9420588d8288672f18a3fd2` | `5cde5ec75e554553079882c8d0f72244eaef9cd5cedf73b1f37bd65ba9a384c6` |

The modified files total 60,461 bytes. [provenance.json](provenance.json) records
the frozen source-manifest identity, input/output sizes and project support
source hashes.

## Changes

- `model.py`: when live memory is enabled, report an FP32 PLE convolution-state
  dtype instead of the upstream state-dtype calculation.
- `ple_layer.py`: in the same branch, freeze PLE parameter gradients, report the
  FP32 state dtype, and delegate PLE forward to the project's
  `qwen38_vllm_live.split_ple_forward` adapter.
- Both files validate the live-binding environment captured at import time.
  Changing it inside an existing process is rejected. If both binding variables
  are absent, these files take their upstream branches.

The Delta state, FP32 split computation and request lifecycle live in the
project's [delta_ngram package](../../src/delta_ngram/), especially
[qwen38_vllm_live.py](../../src/delta_ngram/qwen38_vllm_live.py),
[qwen38_vllm_ple.py](../../src/delta_ngram/qwen38_vllm_ple.py) and
[qwen38_vllm_snapshot.py](../../src/delta_ngram/qwen38_vllm_snapshot.py).
Use the corresponding project revision and its dependencies; the two vLLM
replacements do not supply those adapters or a state binding.

## Apply to the intended local environment

1. Stop Python/vLLM workers using the target environment. Locate that
   environment's installed `vllm/` directory and confirm distribution version
   `0.30.0`.
2. Compute SHA256 for the two installed files and compare with the **Original**
   column. Preserve their original copies. If either input differs, resolve the
   source-version mismatch before replacing it.
3. Copy the two files from this directory to the same relative paths under the
   installed `vllm/` directory, then verify the **Modified** hashes. Retain the
   source bytes and line endings as provided.
4. Make this repository's `src/` importable in the runtime, for example through
   the project's existing installation or `PYTHONPATH`. Use the existing
   registered runners and engine settings for the intended experiment.
5. Before starting a **new Python process**, set both
   `DELTA_NGRAM_VLLM_LIVE_BINDING` to the absolute path of the intended binding
   JSON and `DELTA_NGRAM_VLLM_LIVE_BINDING_SHA256` to that file's lowercase
   SHA256. The binding must use protocol
   `qwen38_vllm_live_two_chain_fp32_v1`, schema version 1. Worker processes must
   inherit the same environment. Do not change either variable after import.

To use the upstream branches, leave **both** binding variables unset before
starting the new process. Setting only one is invalid.

## Reproduction and notices

The existing [prepare_vllm_local.py](../../scripts/native_qwen38/prepare_vllm_local.py)
and [prepare_vllm_live.py](../../scripts/native_qwen38/prepare_vllm_live.py) define
the changes. Applying their patch logic to the two pinned original files
reproduces both files here byte for byte. The historical `prepare_fork` workflow
copies an installed package into a local runtime directory on the execution
machine; that package copy is not part of this Git delivery.

Both source files retain the upstream Apache-2.0 SPDX and copyright notices.
The matching installed distribution declares `License-Expression: Apache-2.0`
and supplies the included [LICENSE](LICENSE). Its 11,357 bytes and SHA256
`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
match the same distribution's recorded license file.

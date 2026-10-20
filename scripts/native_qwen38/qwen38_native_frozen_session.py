"""One accepted native reader lifetime, shared by bounded experiment runners.

The caller prepares options.output and its experiment registration beforehand.
options supplies model_path, verification, delta_repo, output (Path), and an even
tokens ceiling <=640. callback(session) runs on every rank and owns sampling,
writer/optimizer state, loss budgets, checkpointing, and result receipts.

No dataset is prepared: the empty file only satisfies the audited Swift CLI.
There is no import-time CUDA/model initialization and no standalone CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

from qwen38_native_runtime_preflight import (
    SWIFT_COMMIT, BRIDGE_COMMIT, digest, distribution_commit,
    swift_arguments, verified_model,
)


@dataclass(frozen=True)
class FrozenNativeSession:
    models: list
    raw: Any
    adapter: Any
    layout: Any
    device: Any
    tokenizer: Any
    identity: dict
    versions: dict
    assert_frozen: Callable[[], None]


def run_frozen_session(options, callback):
    """Initialize TP2/PP2/EP2, run callback once, then restore hooks and groups."""
    if sys.platform != "linux" or int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise ValueError("native frozen session requires Linux torchrun with four ranks")
    if type(options.tokens) is not int or not 2 <= options.tokens <= 640 or options.tokens % 2:
        raise ValueError("register an even native sequence ceiling in [2,640]")
    if not callable(callback) or not Path(options.output).is_dir():
        raise ValueError("callback and an already prepared shared output directory are required")
    for name in ("model_path", "verification", "delta_repo", "output"):
        setattr(options, name, Path(getattr(options, name)).resolve())
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                 "MODELSCOPE_OFFLINE", "USE_HF"):
        os.environ[name] = "1"
    os.environ["PLE_CPU_OFFLOAD"], os.environ["PLE_FUSED_KERNEL"] = "1", "0"
    identity = verified_model(options.model_path, options.verification)
    versions = {"ms-swift": distribution_commit("ms-swift", SWIFT_COMMIT),
                "mcore-bridge": distribution_commit("mcore-bridge", BRIDGE_COMMIT)}
    for name in ("torch", "megatron-core", "transformers", "transformer-engine",
                 "flash-attn", "flash-linear-attention", "tokenizers"):
        versions[name] = importlib.metadata.version(name)
    if versions["megatron-core"] != "0.18.0" or versions["transformers"] != "5.16.0":
        raise ValueError("requires the audited MCore 0.18.0 / Transformers 5.16.0 stack")
    sys.path.insert(0, str(options.delta_repo / "src"))
    from delta_ngram.qwen38_adapter import Qwen38NativePLEAdapter
    from delta_ngram.qwen38_native import Qwen38PLELayout
    for name, symbol in (("adapter", Qwen38NativePLEAdapter), ("native_boundary", Qwen38PLELayout)):
        path = Path(inspect.getfile(symbol)).resolve()
        if not path.is_relative_to(options.delta_repo):
            raise ValueError("native delta source resolved outside --delta-repo")
        identity[name + "_sha256"] = digest(path)
    identity["frozen_session_sha256"] = digest(__file__)
    identity["runtime_helper_sha256"] = digest(inspect.getfile(verified_model))
    empty = options.output / "unused_native_session_dataset.jsonl"
    if int(os.environ["RANK"]) == 0:
        # A previous failed session may have created this zero-byte CLI marker.
        if empty.exists():
            if empty.read_bytes() != b"":
                raise ValueError("unused dataset marker must remain empty")
        else:
            with empty.open("xb"):
                pass
    else:
        deadline = time.monotonic() + 300
        while not empty.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError("rank0 did not create the session CLI marker")
            time.sleep(.1)

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from swift.megatron.model import get_mcore_model
    from swift.megatron.pipelines.train.sft import MegatronSft
    from swift.megatron.utils import wrap_model

    callback_result = None

    class NativeSessionRunner(MegatronSft):
        def run(self):
            nonlocal callback_result
            device = torch.device("cuda", torch.cuda.current_device())
            agreements = [None] * dist.get_world_size()
            dist.all_gather_object(agreements, {"identity": identity, "versions": versions})
            if any(value != agreements[0] for value in agreements):
                raise ValueError("ranks disagree on the frozen reader identity")
            if (ps.get_tensor_model_parallel_world_size(), ps.get_pipeline_model_parallel_world_size(),
                    ps.get_expert_model_parallel_world_size(), ps.get_expert_tensor_parallel_world_size(),
                    ps.get_data_parallel_world_size(), ps.get_context_parallel_world_size()) != (2, 2, 2, 1, 1, 1):
                raise ValueError("requires TP2 PP2 EP2 ETP1 DP1 CP1")
            if get_forward_backward_func().__name__ != "forward_backward_pipelining_without_interleaving":
                raise ValueError("requires the accepted non-interleaved pipeline schedule")
            raw_models = get_mcore_model(self.args, self.template.config)
            if len(raw_models) != 1:
                raise ValueError("requires one local model chunk")
            raw, config = raw_models[0], raw_models[0].config
            if config.padded_vocab_size != 248320 or config.cross_entropy_loss_fusion:
                raise ValueError("requires native vocabulary and unfused FP32 cross entropy")
            config.recompute_granularity = config.recompute_method = None
            config.recompute_modules = []
            config.deallocate_pipeline_outputs = config.overlap_p2p_comm = False
            config.variable_seq_lengths = True
            config.calculate_per_token_loss = False
            for name in ("grad_scale_func", "no_sync_func", "grad_sync_func",
                         "param_sync_func", "finalize_model_grads_func"):
                setattr(config, name, None)
            config.bridge.load_weights(raw_models, self.args.model_dir)
            raw.requires_grad_(False).eval()
            models = wrap_model(self.args, raw_models, wrap_with_ddp=False)
            if len(models) != 1:
                raise ValueError("requires one wrapped model chunk")
            models[0].requires_grad_(False).eval()
            original = {name: (value, value._version) for name, value in raw.named_parameters()}
            native = [value for value in raw.modules() if type(value).__name__ == "Qwen4ExpTextPLELayer"]
            if len(native) != (1 if ps.is_pipeline_first_stage() else 0):
                raise ValueError("native PLE must occur once on each first-stage TP replica")
            embedding = native[0].ple_embedding if native else None
            adapter = layout = gather_module = gather_original = host_identity = None

            def assert_frozen():
                current = dict(raw.named_parameters())
                if current.keys() != original.keys() or any(
                        current[name] is not value or value._version != version
                        or value.requires_grad or value.grad is not None
                        for name, (value, version) in original.items()):
                    raise ValueError("frozen reader parameters changed or acquired gradients")
                if raw.training or models[0].training:
                    raise ValueError("frozen reader left evaluation mode")
                if embedding is not None and host_identity is not None and (
                        id(embedding.host_table), embedding.host_table._version) != host_identity:
                    raise ValueError("native host table changed")

            error = None
            try:
                if embedding is not None:
                    if (not embedding.cpu_offload or embedding.host_table.device.type != "cpu"
                            or embedding.host_table.dtype != torch.bfloat16):
                        raise ValueError("requires frozen dequantized BF16 host T0")
                    host_identity = (id(embedding.host_table), embedding.host_table._version)
                    gather_module = importlib.import_module(type(native[0]).__module__)
                    gather_original = getattr(gather_module, "gather_ple_rows", None)
                    if not callable(gather_original):
                        raise ValueError("native host lookup fallback selector is missing")
                    gather_module.gather_ple_rows = lambda *args, **kwargs: None
                    layout = Qwen38PLELayout.from_native(
                        json.loads((options.model_path / "config.json").read_bytes()),
                        embedding.ngram_heads_offsets, embedding.ngram_heads_vocab_sizes)
                    adapter = Qwen38NativePLEAdapter(native[0], layout)
                assert_frozen()
                session = FrozenNativeSession(
                    models=models, raw=raw, adapter=adapter, layout=layout, device=device,
                    tokenizer=self.template.tokenizer, identity=identity, versions=versions,
                    assert_frozen=assert_frozen)
                callback_result = callback(session)
                assert_frozen()
                return callback_result
            except BaseException as failure:
                error = failure
                raise
            finally:
                try:
                    assert_frozen()
                except BaseException as failure:
                    if error is not None:
                        error.add_note(f"Frozen reader exit validation also failed: {failure}")
                    else:
                        raise
                finally:
                    if adapter is not None:
                        adapter.close()
                    if gather_module is not None and callable(gather_original):
                        gather_module.gather_ple_rows = gather_original

    try:
        NativeSessionRunner(swift_arguments(options, empty)).main()
        return callback_result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

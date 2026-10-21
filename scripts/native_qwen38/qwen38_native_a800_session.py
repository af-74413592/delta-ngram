"""Independent A800 TP2/PP4/EP2 frozen reader; requires its own GPU acceptance.

The caller prepares options.output and its experiment registration beforehand.
options supplies model_path, verification, delta_repo, output (Path), and an even
tokens ceiling <=8192. callback(session) runs on every rank and owns sampling,
writer/optimizer state, loss budgets, checkpointing, and result receipts.

No dataset is prepared: the empty file only satisfies the audited Swift CLI.
There is no import-time CUDA/model initialization and no standalone CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import faulthandler
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
    swift_arguments as original_swift_arguments, verified_model,
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
    control_group: Any
    host_transport: dict | None
    world_communication_preflight: dict | None = None


def runtime_phase(name):
    print(json.dumps({"kind": "runtime_phase", "rank": int(os.environ.get("RANK", "-1")),
                      "phase": name, "unix_time": time.time()}), flush=True)


def initialize_default_nccl_world(device, control_group):
    """Bootstrap Swift's world P2P communicator with every rank present.

    The pinned Swift patch routes activation sends through group=None, while
    shape sends use pipeline groups. Without an earlier world collective,
    downstream stages can wait for shapes while upstream stages initialize the
    world communicator. A caller's capture hooks must not supply this implicitly.
    """
    import torch
    import torch.distributed as dist
    if device.type != "cuda" or dist.get_backend() != "nccl" or dist.get_backend(control_group) != "gloo":
        raise ValueError("A800 communication preflight requires CUDA, NCCL world, and Gloo control")
    size = dist.get_world_size()
    token = torch.ones(1, dtype=torch.int32, device=device)
    dist.all_reduce(token)
    results = [None] * size
    dist.all_gather_object(results, int(token.item()), group=control_group)
    if results != [size] * size:
        raise ValueError("default NCCL world collective was not confirmed by every rank")
    return {"status": "passed", "operation": "NCCL_world_all_reduce_sum_int32_one",
            "world_size": size, "gloo_confirmed_results": results,
            "purpose": "initialize default activation P2P communicator before pipeline subsets"}


def a800_arguments(options, synthetic_path):
    values = original_swift_arguments(options, synthetic_path)
    for name, value in (("--pipeline_model_parallel_size", "4"),
                        ("--decoder_first_pipeline_num_layers", "12")):
        values[values.index(name) + 1] = value
    values += ["--template", "qwen3_8"]
    return values


def a800_package_lock():
    """Pin the separately inventoried A800 stack; not an H200 equivalence claim."""
    expected = {"torch": "2.12.0+cu132", "megatron-core": "0.18.2+571370c82",
                "ms-swift": "4.5.2", "mcore-bridge": "1.7.0.dev0",
                "transformers": "5.14.1", "transformer-engine": "2.18.0+a2736167",
                "flash-attn": "2.8.4", "flash-linear-attention": "0.5.2"}
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise ValueError(f"A800 package lock changed: {actual}")
    sources = {
        "megatron/core/pipeline_parallel/schedules.py": "946a71d2fd69dfebaceab461bd8e01a37cc4797221651d64943c2c5c9c4e6f0a",
        "megatron/core/transformer/moe/moe_utils.py": "629c48538f2d69a2e110ec5f5b5925d20ccf541a7e960adff43ebb61d4528713",
        "megatron/core/tensor_parallel/layers.py": "fd9c4191a59d6b9a181eec1e208de56eae4678536f290c6d6aff8a353992e638",
        "mcore_bridge/model/gpts/qwen4_exp.py": "7a27d2b5b633366e016d28676c312831124c79ed707cb9c969c9c5171552f8c3",
        "mcore_bridge/model/modules/ple.py": "94d709eba089228ed07b7325dea75427189fe7fd395d82f6e79c0614eeaa2a71",
        "mcore_bridge/model/modules/hyper_connection_gated.py": "8fc7182e2144982c89f65f784366cca4197892c846b765a964726404ee74a706",
        "mcore_bridge/model/modules/topk_router.py": "0dc1f53b547d6810a3cc318d6dc5c556edcc6587686ab1d43bfc77261e599797",
    }
    for relative, expected_hash in sources.items():
        package = "mcore-bridge" if relative.startswith("mcore_bridge") else "megatron-core"
        path = importlib.metadata.distribution(package).locate_file(relative)
        if digest(path) != expected_hash:
            raise ValueError(f"A800 source lock changed: {relative}")
    return {"packages": actual, "sources": sources, "H200_numerical_equivalence_claimed": False}


def run_a800_session(options, callback):
    """Initialize TP2/PP4/EP2, run callback once, then restore hooks and groups."""
    if sys.platform != "linux" or int(os.environ.get("WORLD_SIZE", "0")) != 8:
        raise ValueError("native frozen session requires Linux torchrun with eight ranks")
    if type(options.tokens) is not int or not 2 <= options.tokens <= 8192 or options.tokens % 2:
        raise ValueError("register an even native sequence ceiling in [2,8192]")
    if not callable(callback) or not Path(options.output).is_dir():
        raise ValueError("callback and an already prepared shared output directory are required")
    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=True)
    runtime_phase("session_start")
    for name in ("model_path", "verification", "delta_repo", "output"):
        setattr(options, name, Path(getattr(options, name)).resolve())
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                 "MODELSCOPE_OFFLINE", "USE_HF"):
        os.environ[name] = "1"
    os.environ["PLE_CPU_OFFLOAD"], os.environ["PLE_FUSED_KERNEL"] = "1", "0"
    identity = verified_model(options.model_path, options.verification)
    versions = a800_package_lock()
    sys.path.insert(0, str(Path(options.cl_repo).resolve() / "scripts"))
    sys.path.insert(0, str(Path(options.cl_repo).resolve() / "src"))
    from qwen38_mcore_retention_baseline import register_qwen4_exp_swift_compat
    register_qwen4_exp_swift_compat()
    sys.path.insert(0, str(options.delta_repo / "src"))
    from delta_ngram.qwen38_adapter import Qwen38NativePLEAdapter
    from delta_ngram.qwen38_native import Qwen38PLELayout
    from qwen38_native_a800_host import install_a800_host_table
    for name, symbol in (("adapter", Qwen38NativePLEAdapter), ("native_boundary", Qwen38PLELayout)):
        path = Path(inspect.getfile(symbol)).resolve()
        if not path.is_relative_to(options.delta_repo):
            raise ValueError("native delta source resolved outside --delta-repo")
        identity[name + "_sha256"] = digest(path)
    identity["frozen_session_sha256"] = digest(__file__)
    identity["runtime_helper_sha256"] = digest(inspect.getfile(verified_model))
    identity["a800_host_transport_sha256"] = digest(inspect.getfile(install_a800_host_table))
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
            runtime_phase("control_group_create")
            control_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=5))
            agreements = [None] * dist.get_world_size()
            dist.all_gather_object(agreements, {"identity": identity, "versions": versions}, group=control_group)
            if any(value != agreements[0] for value in agreements):
                raise ValueError("ranks disagree on the frozen reader identity")
            if (ps.get_tensor_model_parallel_world_size(), ps.get_pipeline_model_parallel_world_size(),
                    ps.get_expert_model_parallel_world_size(), ps.get_expert_tensor_parallel_world_size(),
                    ps.get_data_parallel_world_size(), ps.get_context_parallel_world_size()) != (2, 4, 2, 1, 1, 1):
                raise ValueError("requires TP2 PP4 EP2 ETP1 DP1 CP1")
            if get_forward_backward_func().__name__ != "forward_backward_pipelining_without_interleaving":
                raise ValueError("requires the accepted non-interleaved pipeline schedule")
            runtime_phase("model_construct_begin")
            raw_models = get_mcore_model(self.args, self.template.config)
            runtime_phase("model_construct_complete")
            dist.barrier(group=control_group)
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
            runtime_phase("weights_load_begin")
            config.bridge.load_weights(raw_models, self.args.model_dir)
            runtime_phase("weights_load_complete")
            dist.barrier(group=control_group)
            raw.requires_grad_(False).eval()
            native = [value for value in raw.modules() if type(value).__name__ == "Qwen4ExpTextPLELayer"]
            if len(native) != (1 if ps.is_pipeline_first_stage() else 0):
                raise ValueError("native PLE must occur once on each first-stage TP replica")
            runtime_phase("host_table_bind_begin")
            host_binding = install_a800_host_table(native[0]) if native else None
            if host_binding is not None:
                host_binding.assert_frozen()
                host = host_binding.replacement.host_table
                if any(value.data_ptr() == host.data_ptr() for value in list(raw.parameters()) + list(raw.buffers())):
                    raise ValueError("CPU T0 is still registered and would move during CUDA wrapping")
            runtime_phase("host_table_bind_complete")
            dist.barrier(group=control_group)
            runtime_phase("model_wrap_cuda_begin")
            models = wrap_model(self.args, raw_models, wrap_with_ddp=False)
            if host_binding is not None:
                host_binding.assert_frozen()
            runtime_phase("model_wrap_cuda_complete")
            dist.barrier(group=control_group)
            if len(models) != 1:
                raise ValueError("requires one wrapped model chunk")
            models[0].requires_grad_(False).eval()
            original = {name: (value, value._version) for name, value in raw.named_parameters()}
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
                if host_binding is not None:
                    host_binding.assert_frozen()

            error = None
            try:
                runtime_phase("adapter_setup_begin")
                if embedding is not None:
                    layout = Qwen38PLELayout.from_native(
                        json.loads((options.model_path / "config.json").read_bytes()),
                        embedding.ngram_heads_offsets, embedding.ngram_heads_vocab_sizes)
                    adapter = Qwen38NativePLEAdapter(native[0], layout)
                runtime_phase("adapter_setup_complete")
                assert_frozen()
                dist.barrier(group=control_group)
                runtime_phase("default_nccl_world_bootstrap_begin")
                communication_preflight = initialize_default_nccl_world(device, control_group)
                if dist.get_rank() == 0:
                    print(json.dumps({"kind": "world_communication_preflight", **communication_preflight}), flush=True)
                runtime_phase("default_nccl_world_bootstrap_complete")
                session = FrozenNativeSession(
                    models=models, raw=raw, adapter=adapter, layout=layout, device=device,
                    tokenizer=self.template.tokenizer, identity=identity, versions=versions,
                    assert_frozen=assert_frozen, control_group=control_group,
                    host_transport=host_binding.identity if host_binding is not None else None,
                    world_communication_preflight=communication_preflight)
                runtime_phase("callback_begin")
                callback_result = callback(session)
                runtime_phase("callback_complete")
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
                    if host_binding is not None:
                        host_binding.restore()

    try:
        NativeSessionRunner(a800_arguments(options, empty)).main()
        return callback_result
    finally:
        faulthandler.cancel_dump_traceback_later()
        if dist.is_initialized():
            dist.destroy_process_group()

"""Read-only base and native prefetch storage audit for LoCoMo."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path


def host_weight_records(offloader):
    """Canonical weights live on the CPU; shared GPU slots are transient copies."""
    import torch
    if (type(offloader).__name__ != "PrefetchOffloader" or offloader.group_size != 3
            or offloader.num_in_group != 1 or offloader.prefetch_step != 1
            or offloader.offload_params != {"experts"} or offloader.mode != "cpu"
            or len(offloader.module_offloaders) != 16 or offloader.buffer_pool is None
            or offloader.buffer_pool.slot_capacity != 1):
        raise ValueError("native prefetch configuration differs")
    records = []
    for index, module in enumerate(offloader.module_offloaders):
        for name, parameter in sorted(module._param_offloaders.items()):
            value = parameter._cpu_storage
            if ("experts" not in name.split(".") or value is None or value.device.type != "cpu"
                    or value.dtype != torch.bfloat16 or value.requires_grad or value.grad is not None
                    or parameter._gpu_buffer is None):
                raise ValueError("offloaded canonical weight is not a frozen BF16 CPU expert")
            records.append((index, name, value))
    if not records:
        raise ValueError("prefetch did not offload any weights")
    return tuple(records)


def host_weight_sha256(records):
    """Stream tensor bytes without allocating a second full host weight copy."""
    import torch
    result = hashlib.sha256()
    for index, name, value in records:
        metadata = [index, name, str(value.dtype), list(value.shape), list(value.stride())]
        result.update(json.dumps(metadata, separators=(",", ":"), ensure_ascii=True).encode())
        # Processed weights may have non-contiguous strides; hash logical values.
        raw = value.detach().contiguous().view(torch.uint8).reshape(-1).numpy()
        raw_view = memoryview(raw)
        for start in range(0, raw.nbytes, 8 << 20):
            result.update(raw_view[start:start + (8 << 20)])
    return result.hexdigest()


def assert_host_weights_unchanged(before, after, expected_sha):
    if len(before) != len(after) or not all(
            index == old_index and name == old_name and tensor is old_tensor
            and tensor._version == version and not tensor.requires_grad and tensor.grad is None
            for (old_index, old_name, old_tensor, version), (index, name, tensor) in zip(before, after)):
        raise ValueError("canonical CPU weight identity/version changed")
    actual = host_weight_sha256(after)
    if actual != expected_sha:
        raise ValueError("canonical CPU weight bytes changed")
    return actual


class LoCoMoPrefetchAuditWorker:
    def locomo_base_status(self):
        runner = self.model_runner
        return {"rank": self.rank, "quiescent": runner.req_states.num_reqs == 0 and runner.execute_model_state is None}

    def locomo_base_initialize(self):
        from delta_ngram import qwen38_vllm_live as live
        from vllm.model_executor.offloader import get_offloader
        if any(value for key, value in os.environ.items() if key.startswith("DELTA_NGRAM_VLLM")) or live.split_enabled():
            raise ValueError("full-context base evaluation cannot enable Delta bindings")
        model = self.get_model()
        model.requires_grad_(False).eval()
        self._locomo_parameters = tuple((p, p._version) for p in model.parameters())
        records = host_weight_records(get_offloader())
        self._locomo_host_guard = tuple((index, name, tensor, tensor._version) for index, name, tensor in records)
        self._locomo_host_sha = host_weight_sha256(records)
        return self.locomo_base_resources(reset_peak=True)

    def locomo_base_resources(self, reset_peak=False):
        import resource
        import torch
        from vllm.model_executor.offloader import get_offloader
        from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
        model = self.get_model()
        parameters = tuple(model.parameters())
        if model.training or len(parameters) != len(self._locomo_parameters) or not all(
            p is old and p._version == version and not p.requires_grad and p.grad is None
            for p, (old, version) in zip(parameters, self._locomo_parameters)):
            raise ValueError("base model parameters changed")
        runner = self.model_runner
        if runner.req_states.num_reqs != 0 or runner.execute_model_state is not None:
            raise ValueError("resource observation requires a released request")
        offloader = get_offloader()
        records = host_weight_records(offloader)
        cpu_sha = assert_host_weights_unchanged(self._locomo_host_guard, records, self._locomo_host_sha)
        host_bytes = sum(t.numel() * t.element_size() for _, _, t in records)
        if host_bytes != offloader.total_offloaded_bytes:
            raise ValueError("native offload byte accounting differs")
        cache = runner.kv_cache_config
        capacity, concurrency = get_kv_cache_capacity(runner.vllm_config, cache)
        free, total = torch.cuda.mem_get_info(self.device)
        groups = [{"type": type(group.kv_cache_spec).__name__, "layers": len(group.layer_names),
            "block_size": group.kv_cache_spec.block_size, "page_size_bytes": group.kv_cache_spec.page_size_bytes,
            "max_request_bytes": group.kv_cache_spec.max_memory_usage_bytes(runner.vllm_config)}
            for group in cache.kv_cache_groups]
        ples = [module for module in model.modules() if type(module).__name__ == "Qwen4ExpPLELayer"]
        result = {"rank": self.rank, "base_parameters_unchanged": True, "Delta_enabled": False,
            "canonical_cpu_weights_unchanged": True, "canonical_cpu_weights_sha256": cpu_sha,
            "offload_backend": type(offloader).__name__, "offload_module_count": len(offloader.module_offloaders),
            "offload_host_bytes": host_bytes, "offload_gpu_pool_bytes": offloader.buffer_pool.total_bytes,
            "offload_net_saved_bytes": host_bytes - offloader.buffer_pool.total_bytes,
            "offload_host_dtypes": sorted({str(t.dtype) for _, _, t in records}),
            "offload_host_all_pinned": all(t.is_pinned() for _, _, t in records),
            "offload_group_size": offloader.group_size, "offload_num_in_group": offloader.num_in_group,
            "offload_prefetch_step": offloader.prefetch_step, "offload_params": sorted(offloader.offload_params),
            "quiescent": True, "kv_capacity_tokens": capacity, "kv_max_concurrency": concurrency,
            "kv_num_blocks": cache.num_blocks, "kv_groups": groups,
            "allocated_bytes": torch.cuda.memory_allocated(self.device), "reserved_bytes": torch.cuda.memory_reserved(self.device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device), "free_bytes": free, "total_bytes": total,
            "cpu_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "PLE_cache_dtypes": [str(module.kv_cache[0].dtype) for module in ples],
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(self.device)
        return result

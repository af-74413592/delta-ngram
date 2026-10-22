"""Named live-state RPCs. One B1 request, then one canonical FP32 D1 commit."""
from __future__ import annotations
import hashlib
import inspect
import json
import os
from pathlib import Path
import types


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


class DeltaNGramLiveWorkerExtension:
    def _live_idle(self):
        from delta_ngram.qwen38_vllm_live import require
        runner = self.model_runner
        require(hasattr(runner, "req_states") and hasattr(runner, "execute_model_state"),
                "live protocol requires audited v2 model runner")
        return runner.req_states.num_reqs == 0 and runner.execute_model_state is None

    def delta_live_status(self):
        return {"rank": self.rank, "quiescent": self._live_idle()}

    def _live_check(self):
        from delta_ngram.qwen38_vllm_live import require
        model = self.get_model()
        current = tuple(model.parameters())
        require(len(current) == len(self._live_parameters) and not model.training
            and all(p is prior and p._version == version and not p.requires_grad and p.grad is None
                    for p, (prior, version) in zip(current, self._live_parameters)), "frozen reader changed")
        self._live_runtime.check(self._live_ple)
        for writer, alpha, guard in self._live_writers.values():
            require(all(p._version == version and not p.requires_grad and p.grad is None
                        for p, version in guard), "frozen writer changed")

    def delta_live_initialize(self, writer_specs):
        import torch
        import torch.distributed as dist
        from vllm.distributed import get_tp_group
        from delta_ngram import qwen38_vllm_live as live
        from delta_ngram.qwen38_native import make_qwen38_writer
        from delta_ngram.qwen38_probe import tensor_state_hash
        live.require(not hasattr(self, "_live_runtime") and self._live_idle(), "live worker already active")
        model = self.get_model(); model.requires_grad_(False).eval()
        ples = [m for m in model.modules() if type(m).__name__ == "Qwen4ExpPLELayer"]
        backs = [m for m in model.modules() if type(m).__name__ == "Qwen4ExpModel"]
        live.require(len(ples) == len(backs) == 1, "requires one complete TP-only Qwen backbone")
        self._live_ple, self._live_backbone = ples[0], backs[0]
        self._live_runtime = live.runtime(self._live_ple)
        live.require(not self._live_runtime.store.initialized, "live state was initialized early")
        self._live_runtime.store.initialized = True
        self._live_parameters = tuple((p, p._version) for p in model.parameters())
        self._live_tp = get_tp_group()
        live.require(self._live_tp.world_size == 8 and dist.get_world_size() == 8,
                     "live protocol requires exactly eight TP workers")
        live.require(getattr(self.model_runner, "batch_sharder", None) is None,
                     "live B1 protocol does not allow batch-sharded sampling")
        self._live_control = dist.new_group(backend="gloo")
        self._live_writers = {}
        if self.rank == 0:
            for arm, spec in writer_specs.items():
                live.require(digest(spec["path"]) == spec["sha256"], "writer archive changed")
                saved = torch.load(spec["path"], map_location="cpu", weights_only=False)
                if arm == "ordinary":
                    choices = [x for x in saved["arms"].values() if x.get("alpha") == 1.]
                    live.require(len(choices) == 1, "ordinary alpha=1 writer missing")
                    source = choices[0]; step = source["optimizer_step"]
                else:
                    live.require(arm == "delta" and saved["protocol"] == "qwen38_native_memory_fork_checkpoint_v1"
                        and saved["policy"]["branch"] == "full" and saved["completed_groups"] == 835
                        and saved["completed_targets"] == 20015113, "full writer endpoint differs")
                    source = saved["branch_state"]; step = source["model_optimizer_step"]
                live.require(step == 4172 and source["completed_loss_targets"] == 99997072
                    and tensor_state_hash(source["writer"]) == spec["tensor_sha256"], "writer tensors differ")
                with torch.inference_mode(False), torch.no_grad():
                    writer, alpha = make_qwen38_writer(alpha=1.)
                    writer.load_state_dict(source["writer"], strict=True)
                    writer = writer.to(self.device).float().eval().requires_grad_(False)
                    alpha = alpha.to(self.device).eval().requires_grad_(False)
                self._live_writers[arm] = (writer, alpha, tuple((p, p._version) for p in writer.parameters()))
        self._live_calls = []; self._live_handles = []
        self._live_diagnostics = False
        signature = inspect.signature(self._live_backbone.forward)

        def before(module, args, kwargs):
            active = self._live_runtime.store.active
            live.require(active is not None, "unregistered live model forward")
            bound = signature.bind(*args, **kwargs).arguments
            row = {key: bound[key].detach().clone() for key in
                   ("input_ids", "positions", "query_start_loc", "ngram_context")}
            self._live_calls.append(row)

        def after(module, args, kwargs, output):
            active = self._live_runtime.store.active
            if active["kind"] == "support":
                live.require(isinstance(output, torch.Tensor) and output.ndim == 2
                    and output.shape[1] == 2560 and output.dtype == torch.bfloat16, "pre-head H differs")
                self._live_calls[-1]["hidden"] = output.detach().clone()

        def observe(ple, native_ids, qsl, context):
            from vllm.forward_context import get_forward_context
            row = self._live_calls[-1]
            row["native_ids"] = native_ids.detach().clone()
            metadata = get_forward_context().attn_metadata[ple.prefix]
            row["metadata"] = {name: (value.detach().clone() if isinstance(value, torch.Tensor) else value)
                for name in ("state_indices_tensor", "has_initial_states_p", "has_initial_states_d", "num_prefills",
                             "num_prefill_tokens", "num_decodes", "num_decode_tokens")
                for value in (getattr(metadata, name, None),)}

        self._live_handles.append(self._live_backbone.register_forward_pre_hook(before, with_kwargs=True))
        self._live_handles.append(self._live_backbone.register_forward_hook(after, with_kwargs=True))
        self._live_runtime.observer = observe
        original_conv = self._live_ple._short_conv
        def conv(module, inputs, residual, outer_residual):
            if not self._live_diagnostics:
                return original_conv(inputs, residual, outer_residual)
            from vllm.forward_context import get_forward_context
            from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
            metadata = get_forward_context().attn_metadata[module.prefix]
            indices = metadata.state_indices_tensor.flatten().to(dtype=torch.long)
            cache = module.kv_cache[0]
            if not is_conv_state_dim_first(): cache = cache.transpose(-1, -2)
            safe = indices.clamp(0, cache.shape[0] - 1)
            record = {"input": inputs.detach().clone(), "before": cache.index_select(0, safe).detach().clone(),
                "initial": tuple(value.detach().clone() for name in ("has_initial_states_d", "has_initial_states_p")
                    for value in (getattr(metadata, name, None),) if isinstance(value, torch.Tensor))}
            result = original_conv(inputs, residual, outer_residual)
            record["after"] = cache.index_select(0, safe).detach().clone()
            self._live_calls[-1]["conv"] = record
            return result
        self._live_ple._short_conv = types.MethodType(conv, self._live_ple)
        original_logits = model.compute_logits
        def logits(module, *args, **kwargs):
            result = original_logits(*args, **kwargs)
            if self._live_diagnostics and self.rank == 0 and result is not None:
                live.require(result.ndim == 2 and result.shape[0] == 1, "diagnostic logits require B1")
                self._live_calls[-1]["logits"] = result.detach().clone()
            return result
        model.compute_logits = types.MethodType(logits, model)
        self._live_check()
        return {"rank": self.rank, "binding_sha256": os.environ[live.ENV_SHA], "protocol": live.PROTOCOL,
                "worker_sha256": digest(__file__), "helper_sha256": digest(live.__file__),
                "cache_dtype": str(self._live_ple.kv_cache[0].dtype), "quiescent": self._live_idle()}

    def delta_live_begin(self, arm, kind, request_id, epoch, input_ids, diagnostics=False, diagnostics_path=None):
        from delta_ngram.qwen38_vllm_live import require
        self._live_check()
        require(self._live_idle() and not self._live_calls, "previous request/cache ownership is still active")
        require(type(diagnostics) is bool, "diagnostics flag must be boolean")
        self._live_diagnostics = diagnostics
        require(not diagnostics or (isinstance(diagnostics_path, str) and Path(diagnostics_path).is_absolute()
            and Path(diagnostics_path).parent.is_dir() and not Path(diagnostics_path).exists()),
            "diagnostic logits require a fresh file in an existing directory")
        self._live_diagnostics_path = diagnostics_path
        self._live_runtime.store.begin(arm, kind, request_id, epoch, input_ids)
        return {"rank": self.rank, "epoch": epoch, "request_id": request_id}

    def delta_live_finish(self):
        import torch
        from delta_ngram import qwen38_vllm_live as live
        self._live_check(); live.require(self._live_idle(), "scheduler has not released the completed request")
        store = self._live_runtime.store; active = store.active; rows = self._live_calls
        live.require(active is not None and rows, "request produced no model call")
        row = rows[0]; n = len(active["input_ids"])
        ids = row["input_ids"].reshape(-1)
        live.require(ids.tolist() == active["input_ids"] and row["positions"].tolist() == list(range(n))
            and row["query_start_loc"].tolist() == [0, n], "request did not start with one complete fresh prefill")
        live.require(bool(row["ngram_context"].eq(int(self._live_ple.ple_embedding.eos_token_id)).all()),
                     "fresh request inherited text ngram context")
        meta = row["metadata"]
        live.require(meta["num_prefills"] == 1 and meta["num_prefill_tokens"] == n
            and meta["num_decodes"] == meta["num_decode_tokens"] == 0
            and isinstance(meta["has_initial_states_p"], torch.Tensor)
            and not bool(meta["has_initial_states_p"].any()), "fresh request reused convolution state")
        result = {key: active[key] for key in ("arm", "kind", "request_id", "epoch", "input_sha256")}
        result.update(rank=self.rank, forward_calls=len(rows), fresh_context=True,
            physical_cache_slots=meta["state_indices_tensor"].reshape(-1).cpu().tolist(),
            native_ids_sha256=live.tensor_hash(row["native_ids"]),
            history_sha256=live.canonical_hash([{key: live.tensor_hash(call[key]) for key in
                ("input_ids", "positions", "query_start_loc", "ngram_context", "native_ids")} for call in rows]))
        if self._live_diagnostics:
            history = {}; diagnostics = []; expected_position = 0
            for call_index, call in enumerate(rows):
                positions = call["positions"].cpu().tolist()
                current_ids = call["input_ids"].reshape(-1)
                live.require(positions == list(range(expected_position, expected_position + current_ids.numel()))
                    and call["query_start_loc"].cpu().tolist() == [0, current_ids.numel()],
                    "diagnostic token positions/boundaries are not consecutive")
                expected_position += current_ids.numel()
                metadata = call["metadata"]
                if call_index:
                    live.require(current_ids.numel() == 1 and metadata["num_decodes"] == metadata["num_decode_tokens"] == 1
                        and metadata["num_prefills"] == metadata["num_prefill_tokens"] == 0
                        and isinstance(metadata["has_initial_states_d"], torch.Tensor)
                        and bool(metadata["has_initial_states_d"].all()), "decode lacks initialized request state")
                conv = call["conv"]; inputs = conv["input"].cpu()
                before, after = conv["before"][0].cpu(), conv["after"][0].cpu()
                width, window = after.shape
                def expected(last):
                    return torch.stack([history.get(i, torch.zeros(width, dtype=inputs.dtype))
                                        for i in range(last - window + 1, last + 1)], dim=-1)
                initial = any(bool(x.any()) for x in conv["initial"])
                live.require(initial == bool(call_index), "convolution initial-state flags differ from request phase")
                before_equal = torch.equal(before, expected(positions[0] - 1)) if initial else None
                live.require(before_equal is not False, "PLE cache differs from its own prior input history")
                for position, value in zip(positions, inputs): history[position] = value
                after_equal = torch.equal(after, expected(positions[-1]))
                live.require(after_equal and bool(inputs.isfinite().all()), "PLE cache differs from its own new input history")
                item = {"positions": positions, "input_ids": call["input_ids"].reshape(-1).cpu().tolist(),
                    "native_ids": call["native_ids"].cpu().tolist(),
                    "state_before_matches_own_history": before_equal, "state_after_matches_own_history": after_equal,
                    "conv_input_sha256": live.tensor_hash(inputs)}
                if self.rank == 0:
                    value = call["logits"][0].cpu()
                    live.require(value.numel() == 248320 and bool(value.isfinite().all()), "invalid diagnostic logits")
                    scores, tokens = value.float().topk(16)
                    item["top_logits"] = [{"token_id": int(t), "logit": float(s)} for s, t in zip(scores, tokens)]
                    item["argmax"] = int(value.argmax())
                    item["top_margin"] = float(scores[0] - scores[1])
                diagnostics.append(item)
            result["diagnostics"] = diagnostics
            if self.rank == 0:
                target = Path(self._live_diagnostics_path)
                torch.save({"format": live.PROTOCOL + "/query-logits", "request": result,
                    "logits": [call["logits"].detach().cpu() for call in rows]}, target)
                result["logits_file"] = {"path": str(target), "sha256": digest(target)}
        self._live_diagnostics = False
        if active["kind"] == "support":
            live.require(len(rows) == 1 and row["hidden"].shape == (n, 2560),
                         "support must have exactly one complete prefill and no decode")
            self._live_hidden = row["hidden"][None]
            mask = torch.ones((1, n), dtype=torch.bool, device=self.device)
            self._live_physical = store.layout.physical_rows(row["native_ids"][None], mask)
            result["hidden_sha256"] = live.tensor_hash(self._live_hidden)
            result["physical_rows_sha256"] = live.tensor_hash(self._live_physical)
        store.finish()
        self._live_calls = []
        return result

    def delta_live_pad_capacity(self, target_rows):
        """Engineering-only zero storage; caller restores the zero checkpoint."""
        import torch
        from delta_ngram import qwen38_vllm_live as live
        store = self._live_runtime.store
        self._live_check()
        live.require(self._live_idle() and store.active is None and store.pending is None
            and type(target_rows) is int and target_rows == 666747
            and all(store.epochs[arm] == 0 and state.rows.numel() == 0 for arm, state in store.states.items()),
            "capacity probe requires two empty states and the registered row bound")
        with torch.inference_mode(False), torch.no_grad():
            for arm in live.ARMS:
                payload = live.state_payload(store.states[arm])
                payload.update(rows=torch.arange(1, target_rows + 1, dtype=torch.long),
                    values=torch.zeros((target_rows, 160), dtype=torch.float32),
                    counts=torch.zeros(target_rows, dtype=torch.float32),
                    row_last_written_block=torch.zeros(target_rows, dtype=torch.long),
                    row_log_alpha_at_write=torch.zeros(target_rows, dtype=torch.float64))
                store.states[arm] = live.restore_state(payload, self.device)
                store.identities[arm] = live.canonical_hash({"engineering_capacity": target_rows, "arm": arm})
        torch.cuda.reset_peak_memory_stats(self.device)
        return {"rank": self.rank, "engineering_only": True, "rows_per_arm": target_rows,
                "state_bytes_per_rank": sum(getattr(s, key).numel() * getattr(s, key).element_size()
                    for s in store.states.values() for key in live.FIELDS),
                "cuda_free_bytes": torch.cuda.mem_get_info(self.device)[0]}

    def delta_live_capacity_result(self):
        import torch
        from delta_ngram import qwen38_vllm_live as live
        store = self._live_runtime.store
        self._live_check()
        return {"rank": self.rank, "rows_per_arm": {arm: state.rows.numel() for arm, state in store.states.items()},
                "epochs": dict(store.epochs), "state_hashes": {arm: live.state_hash(state) for arm, state in store.states.items()},
                "max_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
                "max_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
                "cuda_free_bytes": torch.cuda.mem_get_info(self.device)[0]}

    def delta_live_commit(self):
        import torch
        import torch.distributed as dist
        from delta_ngram import qwen38_vllm_live as live
        self._live_check(); live.require(self._live_idle(), "D1 while scheduler active")
        store = self._live_runtime.store
        update = None
        if self.rank == 0:
            arm = store.pending["arm"]; writer, alpha, _ = self._live_writers[arm]
            update = store.produce_update(writer, alpha, self._live_hidden, self._live_physical)
        tensor_keys = live.FIELDS
        header = [{key: ({"shape": list(value.shape), "dtype": str(value.dtype)} if isinstance(value, torch.Tensor)
                        else value) for key, value in update.items()} if self.rank == 0 else None]
        dist.broadcast_object_list(header, src=0, group=self._live_control)
        if self.rank != 0:
            update = dict(header[0])
            for key in tensor_keys:
                spec = header[0][key]
                update[key] = torch.empty(spec["shape"], dtype=getattr(torch, spec["dtype"].split(".")[-1]), device=self.device)
        for key in tensor_keys:
            dist.broadcast(update[key], src=0, group=self._live_tp.device_group)
        expected_rows = torch.unique(self._live_physical[self._live_physical.ne(0)], sorted=True)
        live.require(torch.equal(update["rows"], expected_rows), "D1 patch omitted or added observed addresses")
        result = store.commit(update)
        hashes = [None] * 8
        dist.all_gather_object(hashes, result, group=self._live_control)
        live.require(all(x == hashes[0] for x in hashes), "TP replicas disagree after D1")
        del self._live_hidden, self._live_physical
        self._live_check()
        return dict(result, rank=self.rank)

    def delta_live_checkpoint(self, path, replace=False):
        import torch
        import torch.distributed as dist
        from delta_ngram import qwen38_vllm_live as live
        self._live_check(); store = self._live_runtime.store
        live.require(self._live_idle() and store.active is None and store.pending is None, "checkpoint is not at a D1 boundary")
        hashes = {arm: live.state_hash(state) for arm, state in store.states.items()}
        reports = [None] * 8; dist.all_gather_object(reports, hashes, group=self._live_control)
        live.require(all(x == reports[0] for x in reports), "full seven-field TP state mismatch")
        receipt = {"state_hashes": hashes, "epochs": store.epochs, "identities": store.identities}
        if self.rank == 0:
            target = Path(path); live.require(replace or not target.exists(), "checkpoint already exists")
            temporary = target.with_suffix(target.suffix + ".partial")
            torch.save({"format": live.PROTOCOL, "binding_sha256": os.environ[live.ENV_SHA], **receipt,
                        "states": {arm: live.state_payload(state) for arm, state in store.states.items()}}, temporary)
            with temporary.open("rb") as stream: os.fsync(stream.fileno())
            os.replace(temporary, target)
            receipt.update(path=str(target), sha256=digest(target))
        return dict(receipt, rank=self.rank)

    def delta_live_restore(self, path, sha256):
        import torch
        from delta_ngram import qwen38_vllm_live as live
        self._live_check(); store = self._live_runtime.store
        live.require(self._live_idle() and store.active is None and store.pending is None, "restore during request")
        live.require(digest(path) == sha256, "checkpoint checksum changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        live.require(saved["format"] == live.PROTOCOL and saved["binding_sha256"] == os.environ[live.ENV_SHA],
                     "checkpoint protocol/binding differs")
        with torch.inference_mode(False), torch.no_grad():
            states = {arm: live.restore_state(saved["states"][arm], self.device) for arm in live.ARMS}
        for arm, state in states.items():
            live.require(live.state_hash(state) == saved["state_hashes"][arm]
                and int(state.block_clock[0]) == saved["epochs"][arm]
                and state.row_capacity == store.layout.row_capacity, "restored state/epoch/layout differs")
        store.states, store.epochs, store.identities = states, dict(saved["epochs"]), dict(saved["identities"])
        return {"rank": self.rank, "epochs": store.epochs, "identities": store.identities,
                "state_hashes": saved["state_hashes"]}

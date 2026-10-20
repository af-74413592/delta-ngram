"""Independent TP-scale evidence at native sequence scatter and sparse lookup.

A uses the installed scatter with a known FP64 linear functional (zero CE).
B uses one real native query backward. Its comparisons are manual gather and
FP32 index-add, not a second autograd invocation of the adapter.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
from pathlib import Path

from qwen38_native_runtime_preflight import checked_tokens, digest, schedule_query, write_json
from qwen38_native_tp_gradient_probe import canonical_hash, live_runtime_identity, tensor_hash


@contextmanager
def native_cutpoint_gradients(adapter):
    """Capture gradients without replacing native PLE arithmetic or its graph."""
    captures = {}
    handles = []
    ple = adapter.ple_layer
    original = ple.compute
    had_instance = "compute" in ple.__dict__
    previous_instance = ple.__dict__.get("compute")
    host_original = getattr(adapter, "_host_original", None)
    base_lookup = []

    def capture(name, value):
        if name in captures or not value.requires_grad:
            raise ValueError(f"expected one differentiable native {name} cutpoint")
        captures[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}

        def save(gradient):
            if "gradient" in captures[name]:
                raise ValueError(f"native {name} backward ran more than once")
            captures[name]["gradient"] = gradient.detach().clone()

        value.register_hook(save)

    def compute(*args, **kwargs):
        result = original(*args, **kwargs)
        capture("full_compute", result)
        captures["full_compute"]["input"] = (
            args[0] if args else kwargs["hidden_states"]).detach().clone()
        captures["full_compute"]["value"] = result.detach().clone()
        return result

    def local_hook(module, args, result):
        capture("local_sp", result)

    def embedding_hook(module, args, result):
        capture("embedding", result)
        captures["embedding"]["value"] = result.detach().clone()
        if base_lookup:
            captures["embedding"]["base_lookup"] = base_lookup[0]

    def host_lookup(*args, **kwargs):
        result = host_original(*args, **kwargs)
        base_lookup.append(result.detach().clone())
        return result

    try:
        ple.compute = compute
        if callable(host_original):
            adapter._host_original = host_lookup
        handles.append(ple.register_forward_hook(local_hook))
        handles.append(adapter.embedding.register_forward_hook(embedding_hook))
        yield captures
    finally:
        for handle in handles:
            handle.remove()
        if callable(host_original):
            adapter._host_original = host_original
        if had_instance:
            ple.compute = previous_instance
        else:
            delattr(ple, "compute")


def native_addresses_only(adapter, input_ids):
    """Run the native hashing code, without a table lookup or reader forward."""
    import torch
    embedding = adapter.embedding
    previous = embedding._host_lookup
    had_instance = "_host_lookup" in embedding.__dict__
    previous_instance = embedding.__dict__.get("_host_lookup")
    seen = []

    def capture(ids):
        seen.append(ids.detach().clone())
        return torch.zeros((*ids.shape[:2], 2560), dtype=torch.bfloat16, device=ids.device)

    try:
        embedding._host_lookup = capture
        with torch.no_grad():
            # Direct method call intentionally skips Module hooks. Only the
            # existing native hash and the temporary zero lookup execute.
            embedding.forward(input_ids)
    finally:
        if had_instance:
            embedding._host_lookup = previous_instance
        else:
            delattr(embedding, "_host_lookup")
    if len(seen) != 1 or embedding._host_lookup != previous:
        raise ValueError("native address capture did not restore the existing adapter")
    return seen[0]


def manual_lookup_adjoint(rows, physical_rows, embedding_gradient, valid_mask):
    """dL/d stored delta using only row matching and FP32 index accumulation."""
    import torch
    if rows.ndim != 1 or rows.dtype != torch.long or not bool(rows.gt(0).all()):
        raise ValueError("manual adjoint requires positive sorted sparse rows")
    if rows.numel() > 1 and not bool((rows[1:] > rows[:-1]).all()):
        raise ValueError("manual adjoint rows must be strictly sorted")
    width = embedding_gradient.shape[-1] // physical_rows.shape[-1]
    values = embedding_gradient.float().reshape(*physical_rows.shape, width)
    active = valid_mask.unsqueeze(-1).expand_as(physical_rows) & physical_rows.ne(0)
    selected = physical_rows[active]
    indices = torch.searchsorted(rows, selected)
    if bool((indices >= rows.numel()).any()) or not torch.equal(rows[indices], selected):
        raise ValueError("native query touched an unregistered sparse row")
    result = torch.zeros((rows.numel(), width), dtype=torch.float32, device=rows.device)
    result.index_add_(0, indices, values[active])
    return result


def run_linear_sp_proof(scatter, *, device, group, tp_rank):
    """Use installed SP autograd and independently known global linear losses."""
    import torch
    import torch.distributed as dist
    shape = (8, 1, 5)
    grid = torch.arange(40, device=device, dtype=torch.float64).reshape(shape)
    base = .15 + grid / 77
    direction = (grid + 1) / 41
    full_v = .2 + grid / 59
    local_v = full_v.chunk(2, dim=0)[tp_rank].contiguous()
    leaf = base.clone().requires_grad_()
    local_loss = (scatter(leaf, group=group) * local_v).sum()
    global_loss = local_loss.detach().clone()
    dist.all_reduce(global_loss, group=group)
    local_loss.backward()
    reference_loss = (base * full_v).sum()
    if not torch.allclose(global_loss, reference_loss, rtol=1e-12, atol=1e-12):
        raise ValueError("native scatter fails the independent bilinear identity")
    if not torch.equal(leaf.grad, full_v):
        raise ValueError("native scatter backward is not the full concatenated cotangent")
    derivative = float((leaf.grad * direction).sum())
    derivatives = [None, None]
    dist.all_gather_object(derivatives, derivative, group=group)
    probes = []
    for step in (.001, .002):
        losses = []
        for sign in (1, -1):
            with torch.no_grad():
                local = (scatter(base + sign * step * direction, group=group) * local_v).sum()
            dist.all_reduce(local, group=group)
            losses.append(float(local))
        probes.append({"step": step, "finite_difference": (losses[0] - losses[1]) / (2 * step),
                       "rank_directional_gradients": derivatives, "plus_loss": losses[0],
                       "minus_loss": losses[1]})
    return {"probes": probes, "direction_id": tensor_hash(direction),
            "global_linear_loss": float(global_loss), "known_linear_loss": float(reference_loss),
            "known_directional_gradient": float((full_v * direction).sum()),
            "adjoint_max_abs_error": float((leaf.grad - full_v).abs().max()),
            "operator_source_sha256": digest(inspect.getfile(scatter)),
            "ce_targets": 0}


def run_tp_structure_probe(*, models, adapter, layout, device, tokenizer, output):
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region
    from delta_ngram.batching import BatchMemoryState
    from delta_ngram.state import SparseMemoryState
    from delta_ngram.qwen38_training import GradientScaleProbe, NativeTPStructurePolicy

    rank = dist.get_rank()
    first = ps.is_pipeline_first_stage()
    tp_rank = ps.get_tensor_model_parallel_rank()
    group = ps.get_tensor_model_parallel_group()
    output = Path(output)

    def gather(value):
        values = [None] * dist.get_world_size()
        dist.all_gather_object(values, value)
        return values

    def check(condition, message):
        failures = gather(None if condition else f"rank {rank}: {message}")
        if any(failures):
            raise ValueError("; ".join(x for x in failures if x))

    check((dist.get_world_size(), ps.get_tensor_model_parallel_world_size(),
           ps.get_pipeline_model_parallel_world_size(), ps.get_context_parallel_world_size(),
           ps.get_data_parallel_world_size()) == (4, 2, 2, 1, 1),
          "structure proof requires TP2 PP2 CP1 DP1")
    config = models[0].config
    check(len(models) == 1 and bool(config.sequence_parallel)
          and config.recompute_granularity is None and not config.deallocate_pipeline_outputs
          and not models[0].training and (adapter is not None) == first,
          "structure proof requires the pinned frozen native SP schedule")
    parameters = [(p, p._version) for p in models[0].parameters()]
    check(all(not p.requires_grad and p.grad is None for p, _ in parameters),
          "reader parameters must be frozen")
    identity = live_runtime_identity(models)
    identity_sha = canonical_hash(identity)
    check(len(set(gather(identity_sha))) == 1, "runtime identities disagree")
    case = checked_tokens(tokenizer, 16)
    check(case["loss_tokens"] == 11, "expected eleven query targets")
    registration = {
        "protocol": "qwen38_native_tp_structure_probe_v1", "runtime_identity": identity,
        "runtime_identity_sha256": identity_sha, "proof_source_sha256": digest(__file__),
        "case": case, "operator_steps": [.001, .002], "operator_dtype": "float64",
        "operator_rtol": 1e-8, "operator_atol": 1e-9,
        "lookup_rtol": 1e-6, "lookup_atol": 1e-7,
        "planned_reader_forwards": 1, "planned_reader_backwards": 1,
        "planned_loss_target_tokens": 11, "optimizer_steps": 0,
        "scope": "SP/lookup adjoints and TP scaling; not full BF16 CE finite differences",
    }
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "registration.json", registration)
    dist.barrier()
    linear = run_linear_sp_proof(scatter_to_sequence_parallel_region, device=device,
                                group=group, tp_rank=tp_rank) if first else None
    linear_ranks = [x for x in gather(linear) if x is not None]
    check(len(linear_ranks) == 2 and linear_ranks[0] == linear_ranks[1],
          "native SP linear observations disagree across replicas")
    batch = {"input_ids": torch.tensor([case["input_ids"]], dtype=torch.long, device=device),
             "labels": torch.tensor([case["labels"]], dtype=torch.long, device=device),
             "position_ids": torch.arange(16, device=device).unsqueeze(0), "attention_mask": None}
    valid = torch.tensor([case["valid_mask"]], dtype=torch.bool, device=device)
    owner = torch.zeros(1, dtype=torch.long, device=device)
    state = leaf = rows = physical = None
    readout_metadata = readout_sha = None
    if first:
        physical = layout.physical_rows(native_addresses_only(adapter, batch["input_ids"]), valid)
        rows = torch.unique(physical[valid].reshape(-1), sorted=True)
        rows = rows[rows > 0]
        grid = torch.arange(rows.numel() * 160, dtype=torch.float32, device=device)
        leaf = (grid.mul(.017).sin().reshape(-1, 160) * .01).requires_grad_()
        state = BatchMemoryState(SparseMemoryState(rows, leaf, torch.ones(
            rows.numel(), device=device)), 1, layout.row_capacity)
        readout_metadata = {}
        for name in ("key_proj", "value_proj"):
            layer = getattr(adapter.ple_layer, name)
            readout_metadata[name] = {
                "tp_size": getattr(layer, "tp_size", None),
                "parallel_mode": getattr(layer, "parallel_mode", None),
                "sequence_parallel": getattr(layer, "sequence_parallel", None)}
        readout_sha = canonical_hash({
            name: tensor_hash(value) for name, value in adapter.ple_layer.named_parameters()
            if not name.startswith("ple_embedding.")})
    check(not first or all(value == {"tp_size": 1, "parallel_mode": None,
                                     "sequence_parallel": False}
                           for value in readout_metadata.values()),
          "native PLE readout must be duplicated and contain no TP collectives")
    readout_hashes = [x for x in gather(readout_sha) if x]
    check(len(readout_hashes) == 2 and len(set(readout_hashes)) == 1,
          "native PLE readout weights differ across TP replicas")
    input_hash = canonical_hash({"rows": tensor_hash(rows), "delta": tensor_hash(leaf)}) if first else None
    check(len(set(x for x in gather(input_hash) if x)) == 1, "native delta inputs disagree")
    if first:
        with native_cutpoint_gradients(adapter) as captures:
            query = schedule_query(models, batch, valid_mask=valid, owner_ids=owner,
                                   adapter=adapter, reader_state=state, backward=True,
                                   query_grad_enabled=True, group_total_target_tokens=11)
    else:
        captures = None
        query = schedule_query(models, batch, valid_mask=valid, owner_ids=owner,
                               adapter=None, reader_state=None, backward=True,
                               query_grad_enabled=True, group_total_target_tokens=11)
    torch.cuda.synchronize(device)
    check(not first or (set(captures) == {"full_compute", "local_sp", "embedding"}
          and all("gradient" in x for x in captures.values()) and leaf.grad is not None),
          "native backward did not reach all registered cutpoints")
    local = None
    if first:
        if not torch.equal(physical, query["capture"].physical_rows):
            raise ValueError("native full query addresses differ from native hash-only capture")
        local_grad = captures["local_sp"]["gradient"].contiguous()
        gathered = [torch.empty_like(local_grad) for _ in range(2)]
        dist.all_gather(gathered, local_grad, group=group)
        full_reference = torch.cat(gathered, dim=0).transpose(0, 1).contiguous()
        full_grad = captures["full_compute"]["gradient"]
        manual = manual_lookup_adjoint(rows, physical, captures["embedding"]["gradient"], valid)
        if full_reference.shape != full_grad.shape or manual.shape != leaf.grad.shape:
            raise ValueError("native cutpoint shapes differ from the registered topology")
        local = {
            "rank": rank, "tp_rank": tp_rank,
            "sp_adjoint_max_abs_error": float((full_grad - full_reference).float().abs().max()),
            "overlay_adjoint_max_abs_error": float((leaf.grad - manual).abs().max()),
            "overlay_reference_max_abs": float(manual.abs().max()),
            "raw_gradient_sha256": tensor_hash(leaf.grad),
            "readout_sha256": readout_sha, "readout_metadata": readout_metadata,
            "cutpoints": {key: {k: v for k, v in value.items() if not torch.is_tensor(v)}
                          for key, value in captures.items()},
            "raw_gradient_l2": float(leaf.grad.norm()), "unique_rows": rows.numel(),
            "loss_sum": query["loss_sum"], "normalized_loss": query["normalized_loss"],
        }
        torch.save({"rows": rows.cpu(), "delta": leaf.detach().cpu(), "physical_rows": physical.cpu(),
                    "valid_mask": valid.cpu(), "raw_delta_gradient": leaf.grad.cpu(),
                    "manual_delta_gradient": manual.cpu(), "full_reference": full_reference.cpu(),
                    "cutpoint_gradients": {k: v["gradient"].cpu() for k, v in captures.items()}},
                   output / f"native-adjoints-rank-{rank}.pt")
        if tp_rank == 0:
            ple = adapter.ple_layer
            torch.save({
                "runtime_identity_sha256": identity_sha,
                "ple_source_sha256": identity["sources"]["mcore_bridge.model.modules.ple"],
                "hidden_input": captures["full_compute"]["input"].cpu(),
                "embedding_output": captures["embedding"]["value"].cpu(),
                "base_lookup": captures["embedding"].get("base_lookup").cpu(),
                "native_output": captures["full_compute"]["value"].cpu(),
                "native_output_cotangent": full_grad.cpu(),
                "native_embedding_cotangent": captures["embedding"]["gradient"].cpu(),
                "rows": rows.cpu(), "delta": leaf.detach().cpu(),
                "physical_rows": physical.cpu(), "valid_mask": valid.cpu(),
                "weights": {name: value.detach().cpu() for name, value in ple.named_parameters()
                            if not name.startswith("ple_embedding.")},
                "layout": {"hidden_size": ple.hidden_size, "hc_count": ple.hc_count,
                           "short_conv_state_len": ple.short_conv_state_len},
                "norms": {name: {"eps": float(getattr(ple, name).eps),
                                "group_size": int(getattr(ple, name).group_size)}
                          for name in ("norm_key", "norm_query", "norm_conv")},
                "conv": {name: getattr(ple.conv1d, name)
                         for name in ("stride", "padding", "dilation", "groups")},
            }, output / "native-ple-reference-inputs.pt")
    observations = [x for x in gather(local) if x is not None]
    losses = gather(query["normalized_loss"])
    check(max(losses) - min(losses) <= 1e-7, "logical CE differs across TP/PP copies")
    check(all(p._version == version and p.grad is None and not p.requires_grad
              for p, version in parameters), "reader changed during native structure proof")
    policy = None
    rejection = None
    try:
        policy = NativeTPStructurePolicy(
            reduction="replica", world_size=2, runtime_identity_sha256=identity_sha,
            operator_source_sha256=linear_ranks[0]["operator_source_sha256"],
            proof_source_sha256=digest(__file__), direction_id=linear_ranks[0]["direction_id"],
            operator_probes=tuple(GradientScaleProbe(
                x["step"], x["finite_difference"], tuple(x["rank_directional_gradients"]))
                for x in linear_ranks[0]["probes"]),
            sp_adjoint_max_abs_errors=tuple(x["sp_adjoint_max_abs_error"] for x in observations),
            overlay_adjoint_max_abs_errors=tuple(x["overlay_adjoint_max_abs_error"] for x in observations),
            overlay_reference_max_abs=tuple(x["overlay_reference_max_abs"] for x in observations),
            replicated_gradient_sha256=tuple(x["raw_gradient_sha256"] for x in observations),
            duplicated_readout_sha256=tuple(x["readout_sha256"] for x in observations))
    except ValueError as error:
        rejection = str(error)
    result = {
        "protocol": registration["protocol"], "complete": True,
        "status": "accepted" if policy else "inconclusive",
        "runtime_identity_sha256": identity_sha, "registration_sha256": canonical_hash(registration),
        "selected_reduction": "replica" if policy else None,
        "selected_policy_receipt": policy.receipt() if policy else None, "reason": rejection,
        "linear_sp_proof": linear_ranks, "native_adjoints": observations,
        "forward_passes": 1, "backward_passes": 1, "loss_target_tokens": 11,
        "operator_ce_targets": 0, "optimizer_steps": 0, "reader_unchanged": True,
        "limits": ["Validates TP scaling at native SP and lookup boundaries.",
                   "Does not validate full BF16 CE smoothness or native local-kernel derivatives.",
                   "Previous inconclusive full-reader finite differences remain inconclusive."]}
    if rank == 0:
        write_json(output / "COMPLETE.json", result)
    dist.barrier()
    return result

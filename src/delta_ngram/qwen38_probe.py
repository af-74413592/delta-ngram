"""Small reusable contracts for the registered native fixed-RMS stability test.

No model loader, process-group selection, optimizer step, or data discovery
lives here. The CLI supplies verified frames, frozen scales, and native runtime.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Callable

import torch

from .alpha_memory import LazyDecayState
from .qwen38_native import write_native_block
from .qwen38_training import (
    GradientScaleProbe, NativeTPStructurePolicy, TPGradientPolicy, writer_evidence_features,
)


SOURCES = ("books", "code", "papers", "web", "news", "news_event")
ARMS = (("alpha100_rmsnorm", 1., "rmsnorm"), ("alpha099_rmsnorm", .99, "rmsnorm"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode()).hexdigest()


def query_frame(sample, token_ids):
    """Already-shifted labels for exactly 128 registered targets."""
    if sample["kind"] == "ordinary":
        require(len(token_ids) >= 640, "ordinary family has fewer than 640 tokens")
        ids = list(token_ids[384:640])
        labels = [-100] * 256
        labels[127:255] = token_ids[512:640]
        valid = [True] * 256
        prefix = 128
    elif sample["kind"] == "event":
        require(len(token_ids) >= 128, "second article has fewer than 128 tokens")
        ids = [248044] + list(token_ids[:128]) + [248044]
        labels = list(token_ids[:128]) + [-100, -100]
        valid = [True] * 129 + [False]
        prefix = 1
    else:
        raise ValueError("unknown registered sample kind")
    require(len(ids) == len(labels) == len(valid) and len(ids) % 2 == 0
            and sum(value != -100 for value in labels) == 128,
            "query token/mask alignment failed")
    result = {"sample_id": sample["sample_id"], "source": sample["source"],
              "kind": sample["kind"], "input_ids": ids, "labels": labels,
              "encoder_valid_mask": valid, "write_mask": [False] * len(ids),
              "prefix_tokens": prefix, "loss_target_tokens": 128,
              "position_ids": list(range(len(ids))), "owner_id": 0}
    result["frame_sha256"] = canonical_hash(result)
    return result


def policy_from_receipt(receipt):
    require(receipt.get("accepted") is True, "requires accepted native TP evidence")
    common = {name: receipt[name] for name in (
        "reduction", "world_size", "runtime_identity_sha256", "direction_id", "rtol", "atol")}

    def probes(values):
        return tuple(GradientScaleProbe(
            step=p["step"], finite_difference=p["finite_difference"],
            rank_directional_gradients=tuple(p["rank_directional_gradients"])) for p in values)

    if receipt.get("protocol") == "qwen38_tp_gradient_scale_v1":
        policy = TPGradientPolicy(**common, probes=probes(receipt["probes"]))
    elif receipt.get("protocol") == "qwen38_native_tp_structure_v1":
        policy = NativeTPStructurePolicy(
            **common, operator_source_sha256=receipt["operator_source_sha256"],
            proof_source_sha256=receipt["proof_source_sha256"],
            operator_probes=probes(receipt["operator_probes"]),
            **{name: tuple(receipt[name]) for name in (
                "sp_adjoint_max_abs_errors", "overlay_adjoint_max_abs_errors",
                "overlay_reference_max_abs", "replicated_gradient_sha256",
                "duplicated_readout_sha256")})
    else:
        raise ValueError("unrecognized TP scaling proof")
    require(canonical_hash(policy.receipt()) == canonical_hash(receipt),
            "TP policy receipt content/hash differs")
    return policy


def validate_numeric_registration(registration):
    require(registration.get("schema") == "qwen38_native_rms_stability_registration_v2",
            "wrong short-test registration schema")
    require([(arm["id"], arm["alpha"], arm["norm_mode"]) for arm in registration["arms"]]
            == list(ARMS), "registered arms or their order changed")
    opt = registration["optimizer"]
    expected = {
        "name": "AdamW", "learning_rate": 3e-4, "weight_decay": 0.,
        "betas": [.9, .999], "epsilon": 1e-8, "amsgrad": False, "maximize": False,
        "gradient_clip_l2": 1., "error_if_nonfinite": True,
        "learning_rate_scheduler": "none", "loss_denominator_per_arm_step": 1536,
        "steps_per_arm": 8, "target_tokens_per_sample": 128,
        "backend_flags": {"foreach": False, "fused": False, "capturable": False,
                          "differentiable": False},
    }
    require(all(opt.get(k) == v for k, v in expected.items()),
            "optimizer differs from the fixed eight-step protocol")
    require(registration["writer_input"]["rmsnorm_epsilon"] == 1e-6
            and registration["writer_input"]["rmsnorm_has_parameters"] is False
            and registration["writer"]["initial_parameter_seed"] == 17
            and registration["probe"]["optimizer_steps"] == [0, 4, 8],
            "normalization, initialization or probe cadence changed")
    require(registration["budgets"]["planned_target_accesses"] == 37632
            and registration["budgets"]["normshort_target_access_cap"] == 75000
            and registration["budgets"]["all_native_diagnostic_target_access_cap"] == 100000,
            "registered target budgets changed")
    return validate_sample_registration(registration)


def validate_sample_registration(registration):
    """Frame-only callers may use a preserved earlier registration's same IDs."""
    samples = registration["data"]["frozen_samples"]
    require(len(samples) == 24 and len({s["group_id"] for s in samples}) == 24
            and len({s["sample_id"] for s in samples}) == 24, "family membership mismatch")
    for source in SOURCES:
        subset = [sample for sample in samples if sample["source"] == source]
        require([sample["source_family_rank_zero_based"] for sample in subset] == [0, 1, 2, 3],
                "family rank/order differs")
        for sample in subset:
            number = sample["source_family_rank_zero_based"]
            role = "training_and_scale_calibration" if number < 2 else "fixed_train_side_probe"
            require(sample["role"] == role and sample["split"] == "train"
                    and sample["eligible"] is True
                    and sample["selection_hash"] == canonical_hash([17, source, sample["group_id"]])
                    and sample["sample_id"] == "calibration:" + canonical_hash(
                        [17, source, sample["group_id"], sample["id"]]),
                    "frozen sample role or selection identity mismatch")
    return samples


class NativeProbeLedger:
    """Reserve each dispatch durably before the reader sees its loss targets."""

    def __init__(self, *, prior_diagnostic: int, prior_optimizer: int, prior_normshort: int,
                 save: Callable[[dict], None]):
        require(type(prior_diagnostic) is int and prior_diagnostic >= 0
                and type(prior_optimizer) is int and prior_optimizer >= 0
                and type(prior_normshort) is int and 0 <= prior_normshort <= prior_diagnostic,
                "explicit prior global budget counts required")
        self.prior_diagnostic, self.prior_optimizer = prior_diagnostic, prior_optimizer
        self.prior_normshort = prior_normshort
        self.save = save
        self.calls = []
        self.tags = set()
        self.targets = self.optimizer_targets = self.forward_input_tokens = 0

    def reserve(self, tag, *, targets, inputs, optimizer=False):
        require(isinstance(tag, str) and tag and tag not in self.tags,
                "dispatch tag must be unique; retries require new registered consumption")
        require(type(targets) is int and targets >= 0
                and type(inputs) is int and inputs > 0, "invalid dispatch token counts")
        opt = targets if optimizer else 0
        require(self.prior_normshort + self.targets + targets <= 75000
                and self.prior_diagnostic + self.targets + targets <= 100000
                and self.prior_optimizer + self.optimizer_targets + opt <= 100000000,
                "dispatch would exceed a registered target budget")
        self.tags.add(tag)
        self.targets += targets
        self.optimizer_targets += opt
        self.forward_input_tokens += inputs
        self.calls.append({"tag": tag, "loss_target_tokens": targets,
                           "writer_optimizer_target_tokens": opt,
                           "forward_input_tokens": inputs, "status": "dispatched"})
        self.save(self.report())

    def report(self):
        return {"protocol": "qwen38_native_dispatch_ledger_v1",
                "prior_diagnostic_target_accesses": self.prior_diagnostic,
                "prior_writer_optimizer_target_accesses": self.prior_optimizer,
                "prior_normshort_target_accesses": self.prior_normshort,
                "new_normshort_target_accesses": self.targets,
                "normshort_target_accesses": self.prior_normshort + self.targets,
                "writer_optimizer_target_accesses": self.optimizer_targets,
                "global_diagnostic_target_accesses": self.prior_diagnostic + self.targets,
                "global_writer_optimizer_target_accesses": self.prior_optimizer + self.optimizer_targets,
                "forward_input_tokens": self.forward_input_tokens, "calls": self.calls,
                "failure_accounting": "all dispatched targets remain charged"}


def build_history_memory(writer, alpha_policy, blocks, *, norm_mode, row_capacity, device,
                         diagnostics=None):
    """Keep all writes in one differentiable graph; callers choose grad context."""
    require(bool(blocks), "registered history cannot be empty")
    state = LazyDecayState.empty(1, 160, row_capacity, device)
    raw, gates = [], []
    handles = []
    if diagnostics is not None:
        handles.append(writer.target_head[2].register_forward_hook(
            lambda module, args, output: raw.append(output.detach().float().cpu())))
        handles.append(writer.gate_head[2].register_forward_hook(
            lambda module, args, output: gates.append(output.detach().float().sigmoid().cpu())))
    try:
        for block in blocks:
            hidden = block["H_new"].to(device)
            rows = block["physical_rows_new"].to(device)
            require(hidden.ndim == 3 and hidden.shape[0] == 1
                    and hidden.shape[2] == 2560 and rows.shape == (*hidden.shape[:2], 16)
                    and bool(rows.ne(0).all()), "cached native evidence shape mismatch")
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=device)
            features = writer_evidence_features(hidden, mode=norm_mode, eps=1e-6)
            state = write_native_block(writer, state, features, rows, mask, alpha_policy)
    finally:
        for handle in handles:
            handle.remove()
    require(int(state.block_clock[0]) == len(blocks), "writer clock differs from evidence blocks")
    if diagnostics is not None:
        targets = torch.cat(raw).reshape(-1, 16, 160)
        beta = torch.cat(gates).reshape(-1, 16)
        scale = writer.target_scale.detach().cpu().float()
        delta_square, base_square, token_count = torch.zeros(16), torch.zeros(16), 0
        with torch.no_grad():
            for block in blocks:
                rows = block["physical_rows_new"].to(device)
                count = rows.shape[1]
                effective = state.lookup(rows)[0].float()
                delta_square += effective.square().mean((0, 1, 3)).cpu() * count
                base_square += block["T0_head_rms_new"].float().cpu().square() * count
                token_count += count
        diagnostics.update(
            raw_target_rms_by_head=targets.square().mean((0, 2)).sqrt().tolist(),
            scaled_target_rms_by_head=(targets * scale[None, :, None]).square().mean((0, 2)).sqrt().tolist(),
            beta=scalar_distribution(beta),
            beta_below_001_fraction=float((beta < .01).float().mean()),
            beta_above_099_fraction=float((beta > .99).float().mean()),
            write_blocks=len(blocks), write_events=beta.numel(),
            effective_delta_rms=float(state.lookup(state.rows[None])[0].detach().float().square().mean().sqrt()),
            final_effective_delta_rms_by_head_on_history_events=(delta_square / token_count).sqrt().tolist(),
            T0_rms_by_head_on_history_events=(base_square / token_count).sqrt().tolist(),
        )
    return state


def tensor_state_hash(state_dict):
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parameter_change_norm(writer, initial):
    return math.sqrt(sum(float((value.detach().double().cpu()
                                - initial[name].double()).square().sum())
                         for name, value in writer.named_parameters()))


def scalar_distribution(value):
    value = value.detach().float().flatten().cpu()
    require(bool(value.numel()) and bool(value.isfinite().all()), "invalid diagnostic values")
    quantiles = torch.quantile(value, torch.tensor([.01, .5, .99]))
    return {"count": value.numel(), "mean": float(value.mean()),
            "rms": float(value.square().mean().sqrt()), "min": float(value.min()),
            "p01": float(quantiles[0]), "median": float(quantiles[1]),
            "p99": float(quantiles[2]), "max": float(value.max())}

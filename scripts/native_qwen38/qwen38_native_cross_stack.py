"""Short immutable native greedy probes inside an already loaded session.

Call on every rank while the caller holds native_dispatch(session). No model
load, state restore, writer update, optimizer, or alternate arithmetic occurs.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time

import torch
from delta_ngram.qwen38_seven_forward import (
    require, digest, canonical_hash, immutable_guard, assert_unchanged, state_fingerprint,
)
from qwen38_native_a800_forward import native_forward, runtime

FORMAT = "qwen38_cross_stack_engineering_v1"
VOCAB = 248320


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def prompt_batch(token_ids, *, pad_token_id, device):
    require(isinstance(token_ids, list) and 0 < len(token_ids) <= 8192
            and all(type(t) is int and 0 <= t < VOCAB for t in token_ids), "invalid native probe prefix")
    length = len(token_ids)
    padded = max(2, (length + 1) // 2 * 2)
    tokens = torch.full((1, padded), pad_token_id, dtype=torch.long, device=device)
    tokens[0, :length] = torch.tensor(token_ids, dtype=torch.long, device=device)
    valid = torch.arange(padded, device=device)[None] < length
    return {"input_ids": tokens, "position_ids": torch.arange(padded, device=device)[None],
            "labels": None, "attention_mask": None}, valid


def logit_summary(logits):
    require(logits.shape == (VOCAB,) and logits.dtype == torch.float32
            and bool(logits.isfinite().all()), "invalid full native logits")
    best = logits.topk(2).values
    return {"argmax_token_id": int(logits.argmax()), "top1_score": float(best[0]),
            "top1_top2_margin": float(best[0] - best[1])}


def verify_snapshot_state(state, layout, reference):
    """Prove the actual live query state equals the exported vLLM snapshot."""
    from delta_ngram.qwen38_vllm_snapshot import load_vllm_snapshot
    contract = json.loads(Path(reference).read_text(encoding="utf-8"))
    snapshot = Path(contract["path"]).resolve()
    frozen = load_vllm_snapshot(snapshot, expected_sha256=contract["snapshot_sha256"],
        expected_identity=contract["expected_snapshot_identity"], expected_layout=layout,
        expected_owner_id=0, device="cpu")
    payload = torch.load(snapshot, map_location="cpu", weights_only=True)
    effective, _ = state._lookup_packed(state.rows)
    require(torch.equal(state.rows.detach().cpu(), payload["rows"])
            and torch.equal(effective.detach().cpu(), payload["values"]),
            "live native query state differs from the exported fixed Delta")
    return {"contract_path": str(Path(reference).resolve()), "contract_sha256": digest(reference),
            "snapshot_sha256": frozen.sha256, "snapshot_identity": frozen.identity,
            "actual_live_state_matches_snapshot": True}


@torch.no_grad()
def run_probe(session, output, state, prompts, max_new_tokens=8, snapshot_reference=None,
              *, eos_token_ids=None, max_context_tokens=8192, prompt_manifest_path=None):
    """All-rank callback; output is a new shared directory, prompts have token IDs.

    snapshot_reference is the exported delta_contract.json path. It is required
    so vLLM can later compare exactly this live state, including the empty state.
    """
    dist, ps, _ = runtime(session)
    require(snapshot_reference is not None, "register the exported snapshot contract before this probe")
    require(type(max_new_tokens) is int and 1 <= max_new_tokens <= 8, "probe ceiling must be 1..8 new tokens")
    require(2 <= max_context_tokens <= 8192 and max_context_tokens % 2 == 0, "invalid native context ceiling")
    require(bool(prompts) and len({row["id"] for row in prompts}) == len(prompts), "probe IDs must be unique")
    pad = session.tokenizer.pad_token_id
    if pad is None:
        pad = session.tokenizer.eos_token_id
    require(type(pad) is int, "native tokenizer needs one padding token")
    for row in prompts:
        prompt_batch(row["prompt_token_ids"], pad_token_id=pad, device="cpu")
        require(len(row["prompt_token_ids"]) + max_new_tokens <= max_context_tokens,
                "probe context would overflow; no truncation is allowed")
    eos = eos_token_ids if eos_token_ids is not None else session.tokenizer.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    require(all(type(t) is int and 0 <= t < VOCAB for t in eos), "invalid EOS configuration")
    group = session.control_group
    rank = dist.get_rank()
    output = Path(output).resolve()
    creation = [None]
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=False)
        except BaseException as error:
            creation[0] = str(error)
    dist.broadcast_object_list(creation, src=0, group=group)
    require(creation[0] is None, "cannot create fresh native probe output: " + str(creation[0]))
    guard = immutable_guard(state) if state is not None else None
    before = state_fingerprint(state) if state is not None else None
    snapshot = verify_snapshot_state(state, session.layout, snapshot_reference) if session.adapter is not None else None
    refs = [None] * dist.get_world_size()
    dist.all_gather_object(refs, snapshot, group=group)
    actual_refs = [r for r in refs if r is not None]
    require(len(actual_refs) == 2 and actual_refs[0] == actual_refs[1], "TP readers disagree on snapshot identity")
    snapshot = actual_refs[0]
    records, vectors = [], {}
    started = time.perf_counter()
    for row in prompts:
        prefix, generated, steps, probes = list(row["prompt_token_ids"]), [], [], []
        for step in range(max_new_tokens):
            batch, valid = prompt_batch(prefix, pad_token_id=pad, device=session.device)
            logits = native_forward(session, batch, valid, state=state, return_logits=True)["next_logits"][0]
            summary = logit_summary(logits)
            chosen = summary["argmax_token_id"]
            signature = {"prefix_sha256": canonical_hash(prefix), "logits_sha256": tensor_hash(logits),
                         "chosen_token_id": chosen}
            signatures = [None] * dist.get_world_size()
            dist.all_gather_object(signatures, signature, group=group)
            require(all(item == signatures[0] for item in signatures), "native ranks disagree on logits or argmax")
            steps.append(dict(step=step, prefix_tokens=len(prefix), **signature, **summary))
            if step == 0 or chosen in eos or step + 1 == max_new_tokens:
                key = canonical_hash(prefix)
                if rank == 0:
                    vectors[key] = logits.detach().cpu().clone()
                probes.append({"kind": "prompt" if step == 0 else "final_decision",
                               "prefix_token_ids": list(prefix), "logits_key": key})
            generated.append(chosen)
            prefix.append(chosen)
            if rank == 0:
                print(json.dumps({"kind": "native_probe_token", "id": row["id"], "step": step,
                                  "token_id": chosen}), flush=True)
            if chosen in eos:
                break
        records.append({"id": row["id"], "prompt_token_ids": row["prompt_token_ids"],
            "generated_token_ids": generated, "steps": steps, "probes": probes,
            "text": session.tokenizer.decode(generated, skip_special_tokens=True),
            "finish_reason": "eos" if generated[-1] in eos else "max_new_tokens"})
    if state is not None:
        assert_unchanged(guard)
        require(state_fingerprint(state) == before, "native probe changed fixed state bytes")
    session.assert_frozen()
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, before, group=group)
    first_stage = [value for value in states if value is not None]
    require(len(first_stage) == 2 and first_stage[0] == first_stage[1], "native fixed-state replicas disagree")
    report = {"format": FORMAT, "backend": "native_mcore_full_prefix", "status": "complete",
        "diagnostic_only": True, "same_loaded_session": True, "bitwise_equivalence_required": False,
        "model_identity": session.identity, "versions": session.versions, **snapshot,
        "prompt_manifest_sha256": digest(prompt_manifest_path) if prompt_manifest_path else None,
        "registered_prompt_tokens_sha256": canonical_hash(prompts), "records": records,
        "max_new_tokens": max_new_tokens, "eos_token_ids": eos, "max_context_tokens": max_context_tokens,
        "per_step_all_rank_logits_agreement": True, "state_unchanged": True,
        "state_hashes_by_rank": states, "optimizer_steps": 0, "generator_sha256": digest(__file__),
        "native_forward_sha256": digest(inspect.getfile(native_forward)),
        "elapsed_seconds": time.perf_counter() - started}
    if rank == 0:
        path = output / "logits.pt"
        torch.save({"format": FORMAT, "vectors": vectors}, path)
        report["logits"] = {"file": path.name, "sha256": digest(path), "dtype": "float32", "vocab_size": VOCAB}
        (output / "native_report.json").write_text(json.dumps(report, indent=2, sort_keys=True,
            allow_nan=False) + "\n", encoding="utf-8")
    dist.barrier(group=group)
    return report

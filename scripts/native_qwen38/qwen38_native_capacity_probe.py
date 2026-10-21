"""Bounded engineering capacity/timing probe using zero-padded address storage.

This is not a scientific 35000-write endpoint. It preserves a real 64-write
seed, adds zero placeholders for the complete known support address universe,
then measures sixteen real memory-aware writes and one longest-math TF query.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))

from delta_ngram.qwen38_capacity_probe import CAPACITY_PROTOCOL, expand_zero_capacity, select_capacity_frames
from delta_ngram.qwen38_seven_forward import (
    require,digest,canonical_hash,evidence_batch,restore_state,state_fingerprint,immutable_guard,assert_unchanged,
)
from delta_ngram.qwen38_shortchain_protocol import validate_state_payload
from qwen38_seven_task_forward_writer import load_encoded,setup_cl,write_json
from qwen38_native_memory_shortchain import load_full_writer


def execute(options):
    import numpy as np
    import torch
    from qwen38_native_a800_session import run_a800_session
    manifest,data = load_encoded(options)
    selected = select_capacity_frames(data)
    longest = max((r for r in data['queries'] if r['domain'] == 'math'),key=lambda r:len(r['input_ids']))
    require(len(longest['input_ids']) == 946 and max(len(r['input_ids']) for r in selected) == 101,
            'fixed longest support/query lengths differ')
    require(digest(options.seed_state) == options.seed_state_sha256,'seed-state archive changed')
    saved = torch.load(options.seed_state,map_location='cpu',weights_only=True)
    seed_stats = validate_state_payload(saved['state'],expected_writes=64)
    require(seed_stats['rows'] == 5224 and seed_stats['fingerprint'] == saved['state_fingerprint'],
            'requires exact 64-write shortchain seed')
    require(digest(options.union_rows) == options.union_rows_sha256,'capacity universe file changed')
    union = torch.from_numpy(np.load(options.union_rows,allow_pickle=False))
    require(union.dtype == torch.int64 and tuple(union.shape) == (666747,),'native address universe differs')
    tensors,writer_source = load_full_writer(options)
    require(saved['identity']['writer']['tensor_sha256'] == writer_source['writer_tensor_state_hash'],
            'capacity seed and frozen full100M writer differ')
    rank = int(os.environ.get('RANK','0'))
    if rank == 0:
        options.output.mkdir(parents=True,exist_ok=False)
    else:
        deadline = time.monotonic()+120
        while not options.output.is_dir():
            require(time.monotonic() < deadline,'rank0 did not create output')
            time.sleep(.1)
    setup_cl(options.cl_repo)

    def run(session):
        import torch.distributed as dist
        from megatron.core import parallel_state as ps
        from delta_ngram.qwen38_native import make_qwen38_writer
        from delta_ngram.qwen38_memory_writer import write_memory_aware_native_block
        from delta_ngram.qwen38_training import writer_evidence_features
        from delta_ngram.qwen38_probe import tensor_state_hash
        from qwen38_native_a800_forward import native_forward,native_dispatch,sync_written_rows
        from qwen38_mcore_retention_baseline import reference_batch
        rank,device,control = dist.get_rank(),session.device,session.control_group
        require(session.world_communication_preflight['status'] == 'passed',
                'A800 session must initialize default NCCL world before a capture-free query')
        first = ps.is_pipeline_first_stage()
        state = writer = alpha = None
        expanded_receipt = None
        state_setup_started = time.monotonic()
        if first:
            payload,expanded_receipt = expand_zero_capacity(saved['state'],union)
            state = restore_state(payload,device=device)
            require(state_fingerprint(state) == expanded_receipt['capacity']['fingerprint'],
                    'capacity GPU state differs from CPU-expanded state')
            del payload
        state_setup_seconds = time.monotonic()-state_setup_started
        if rank == 0:
            writer,alpha = make_qwen38_writer(alpha=1.)
            writer.load_state_dict(tensors,strict=True)
            writer = writer.to(device).float().eval().requires_grad_(False)
            alpha = alpha.to(device).eval().requires_grad_(False)
        writer_versions = [(p,p._version) for p in writer.parameters()] if writer is not None else []
        dist.barrier(group=control)
        torch.cuda.synchronize(device)
        setup_peaks = {'rank':rank,'allocated_bytes':torch.cuda.max_memory_allocated(device),
                       'reserved_bytes':torch.cuda.max_memory_reserved(device)}
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        records = []
        with native_dispatch(session),torch.no_grad():
            batch,_ = reference_batch(longest,pad_token_id=int(session.tokenizer.pad_token_id),pad_multiple=2,device=device)
            valid = torch.arange(batch['input_ids'].shape[1],device=device)[None] < len(longest['input_ids'])
            guard = immutable_guard(state) if first else None
            tick = time.monotonic()
            diagnostic_tf = native_forward(session,batch,valid,state=state)
            torch.cuda.synchronize(device)
            query_seconds = time.monotonic()-tick
            if guard is not None:
                assert_unchanged(guard)
                fingerprint_started = time.monotonic()
                require(state_fingerprint(state) == expanded_receipt['capacity']['fingerprint'],
                        'capacity query mutated full state')
                query_fingerprint_seconds = time.monotonic()-fingerprint_started
            else:
                query_fingerprint_seconds = 0.
            del guard
            require(diagnostic_tf['token_count'] == longest['target_token_count'],'longest query targets differ')
            dist.barrier(group=control)
            for offset,row in enumerate(selected,1):
                torch.cuda.synchronize(device)
                start = time.monotonic()
                mb = evidence_batch([row],pad_token_id=int(session.tokenizer.pad_token_id),device=device)
                result = native_forward(session,mb['batch'],mb['valid_mask'],state=state,
                                        evidence=True,encoding_kind='memory')
                torch.cuda.synchronize(device)
                encoded_at = time.monotonic()
                length = mb['lengths'][0]
                h = result['hidden'][:,:length].detach()
                require(bool(h.isfinite().all()),'nonfinite capacity evidence H')
                h_hash = tensor_state_hash({'H':h})
                hashed_at = time.monotonic()
                touched = None
                stats = {}
                phase_seconds = {'encoding':encoded_at-start,'H_finite_and_CPU_hash':hashed_at-encoded_at,
                                 'readstats_features':0.,'D1':0.,'full_state_validation':0.,'TP_sync':0.}
                if rank == 0:
                    physical = result['physical_rows'][:,:length]
                    prior,_ = state.lookup(physical)
                    stats = {'read_nonzero_row_positions':int(prior.ne(0).any(dim=-1).sum())}
                    features = writer_evidence_features(h,mode='rmsnorm',eps=1e-6)
                    torch.cuda.synchronize(device)
                    stats_at = time.monotonic()
                    phase_seconds['readstats_features'] = stats_at-hashed_at
                    state = write_memory_aware_native_block(writer,state,features,physical,
                        torch.ones(features.shape[:2],dtype=torch.bool,device=device),alpha)
                    torch.cuda.synchronize(device)
                    written_at = time.monotonic()
                    phase_seconds['D1'] = written_at-stats_at
                    require(state.rows.numel() == 666747 and int(state.block_clock[0]) == 64+offset,
                            'capacity D1 changed address universe or write count')
                    require(bool(state.values.isfinite().all()),'capacity D1 produced nonfinite state')
                    touched = torch.unique(physical[physical.ne(0)],sorted=True)
                    torch.cuda.synchronize(device)
                    validated_at = time.monotonic()
                    phase_seconds['full_state_validation'] = validated_at-written_at
                sync_started = time.monotonic()
                if first:
                    state = sync_written_rows(state,touched,device=device)
                torch.cuda.synchronize(device)
                synced_at = time.monotonic()
                phase_seconds['TP_sync'] = synced_at-sync_started
                dist.barrier(group=control)
                done = time.monotonic()
                phase_seconds['control_barrier'] = done-synced_at
                measured_phases = [None]*dist.get_world_size()
                dist.all_gather_object(measured_phases,phase_seconds,group=control)
                telemetry_seconds = time.monotonic()-done
                if rank == 0:
                    record = {'engineering_write':offset,'task':row['task'],'source_row_index':row['source_row_index'],
                              'input_tokens':length,'padded_tokens':mb['padded_tokens'],'H_sha256':h_hash,
                              'input_sha256':row['input_token_ids_sha256'],'state_rows':666747,'block_clock':64+offset,
                              'phase_seconds_rank0':phase_seconds,
                              'phase_seconds_max_across_ranks':{k:max(x[k] for x in measured_phases) for k in phase_seconds},
                              'telemetry_allgather_seconds':telemetry_seconds,
                              'total_seconds_before_telemetry_and_io':done-start,**stats}
                    records.append(record)
                    io_started = time.monotonic()
                    with (options.output/'events.jsonl').open('a') as stream:
                        stream.write(json.dumps(record,allow_nan=False)+'\n')
                    print(json.dumps(record),flush=True)
                    record['event_io_seconds'] = time.monotonic()-io_started
        fingerprint_started = time.monotonic()
        final_fingerprint = state_fingerprint(state) if first else None
        final_fingerprint_seconds = time.monotonic()-fingerprint_started
        fingerprints = [None]*dist.get_world_size()
        dist.all_gather_object(fingerprints,final_fingerprint,group=control)
        require(fingerprints[0] == fingerprints[1] and fingerprints[0] is not None
                and all(x is None for x in fingerprints[2:]),'final capacity TP replicas differ')
        peak = {'rank':rank,'allocated_bytes':torch.cuda.max_memory_allocated(device),
                'reserved_bytes':torch.cuda.max_memory_reserved(device),'setup_inclusive_peak':setup_peaks,
                'state_setup_seconds':state_setup_seconds,'query_full_state_CPU_fingerprint_seconds':query_fingerprint_seconds,
                'final_full_state_CPU_fingerprint_seconds':final_fingerprint_seconds}
        peaks = [None]*dist.get_world_size()
        dist.all_gather_object(peaks,peak,group=control)
        if rank == 0:
            require(len(records) == 16 and all(p._version == version and p.grad is None for p,version in writer_versions),
                    'capacity budget incomplete or frozen writer changed')
            require(tensor_state_hash(writer.state_dict()) == writer_source['writer_tensor_state_hash'],
                    'capacity writer tensor bytes changed')
            write_json(options.output/'COMPLETE.json',{
                'protocol':CAPACITY_PROTOCOL,'status':'completed','scientific_endpoint':False,
                'purpose':'capacity and runtime only; not evidence of a correct 35000-write trajectory',
                'seed_state_sha256':options.seed_state_sha256,'seed_state_fingerprint':saved['state_fingerprint'],
                'union_rows_sha256':options.union_rows_sha256,'expanded_state':expanded_receipt,
                'initial_real_writes':64,'engineering_writes':16,'encoding_forward_calls':16,'TF_queries':1,
                'longest_query':{'task_id':longest['task_id'],'input_tokens':946,'target_tokens':longest['target_token_count'],
                                 'seconds_including_first_forward_warmup':query_seconds,
                                 'diagnostic_only_nll':diagnostic_tf['answer_nll']},
                'records':records,'write_seconds_sum':math.fsum(r['total_seconds_before_telemetry_and_io'] for r in records),
                'event_io_seconds_sum':math.fsum(r['event_io_seconds'] for r in records),
                'telemetry_allgather_seconds_sum':math.fsum(r['telemetry_allgather_seconds'] for r in records),
                'timing_notes':{'encoding':'includes detached whole-state GPU clone and frozen reader forward',
                    'write_total':'includes per-frame H hash/readstats/full-state finite guard/TP sync/barrier; excludes event disk IO and telemetry allgather',
                    'CPU_state_fingerprints':'query/final fingerprints separately timed outside write loops',
                    'capture':'no representation hooks or tensor captures in capacity probe',
                    'scope':'writing and one TF only; 105000 intended chain writes, 10752 post-state TF queries plus baselines, and 6720 generation queries remain unexecuted'},
                'real_support_tokens':sum(r['input_tokens'] for r in records),'final_state_fingerprint':fingerprints[0],
                'gpu_peaks':peaks,'callback_seconds':time.monotonic()-started,
                'runtime':session.versions,'base_identity':session.identity,'writer_source':writer_source,
                'world_communication_preflight':session.world_communication_preflight,
                'backward_calls':0,'optimizer_steps':0,'generation_calls':0,'H200_numerical_equivalence_claimed':False,
                'code_sha256':{str(Path(__file__)):digest(__file__),
                    'capacity_protocol':digest(REPO/'src/delta_ngram/qwen38_capacity_probe.py')}})
    return run_a800_session(options,run)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cl-repo','delta-repo','model-path','verification','encoded-dir','writer-checkpoint','output',
                 'seed-state','union-rows'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('encoded-manifest-sha256','writer-sha256','seed-state-sha256','union-rows-sha256'):
        p.add_argument('--'+name,required=True)
    options = p.parse_args()
    options.tokens = 8192
    execute(options)

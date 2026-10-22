"""Explicit delta-writer amendment over the unchanged prepared LoCoMo inputs."""
from __future__ import annotations
import json
from pathlib import Path
from qwen38_locomo_native import (CONDITIONS,PILOT,PLAN_SHA,PREPARED_SHA,PROTOCOL_SHA,
    TOKENIZER_FILES,canonical,digest,load_runtime,require,validate_frame,write_json)

PROTOCOL='qwen38_locomo_a800_delta_readonly_v1'
WRITER_SHA='51d5977845e9930f48732c73ae2503511d8976b1201f139a2be46bfab9f57308'
WRITER_TENSOR_SHA='95e9766c2d90805cede77d03ade5af97b850e9cb3bcc3c55ea103d5cf0e0cb9c'
HASH_ALGORITHM='delta_ngram.qwen38_probe.tensor_state_hash'
NEW_TOKENS=32
# A 640-token evidence prefill uses one unused sample to collect H.
ENGINE_CONTEXT=641

def amendment_definition():
    return {
      'protocol':PROTOCOL,'parent_protocol_sha256':PROTOCOL_SHA,'parent_execution_plan_sha256':PLAN_SHA,
      'prepared_manifest_sha256':PREPARED_SHA,
      'authorization':'User explicitly selected the original formal seven-task delta writer and all four LoCoMo conditions on A800.',
      'writer':{'method':'delta','checkpoint_archive_sha256':WRITER_SHA,'tensor_sha256':WRITER_TENSOR_SHA,
        'tensor_hash_algorithm':HASH_ALGORITHM,'checkpoint_protocol':'qwen38_native_memory_fork_checkpoint_v1',
        'branch':'full','completed_groups':835,'branch_completed_targets':20015113,
        'model_optimizer_step':4172,'model_completed_loss_targets':99997072,'alpha':1.0,
        'initial_memory':'New empty state for each of the ten conversations. No seven-task final memory is loaded.'},
      'evidence':{'encoder':'Read the current conversation Delta through the unchanged live_v3 FP32 split PLE path.',
        'new_tokens_per_frame':512,'old_prefix_tokens_max':128,'chronological':True,'write_final_chunk':True,
        'writer_input':'Crop both pre-head H and physical rows to the new-token span before RMSNorm, right-context features, pooling and D1.',
        'D1_commits_per_frame':1,'D1_commits_total':426,'new_tokens_total':215278,
        'padding_transport':'Prepared SP-even padding is masked non-evidence. vLLM receives exactly the valid prefix+new tokens, omitting only the trailing masked pad; valid token IDs and positions are unchanged.',
        'state_fields':['rows','values','counts','block_clock','cumulative_log_alpha','row_last_written_block','row_log_alpha_at_write'],
        'state_clock':'Exactly the actual evidence-frame count of that conversation, never 35000.'},
      'generation':{'conditions':['self','empty','donor'],'short_prompt_ids':'Unchanged prepared_v1 IDs, identical across the three conditions.',
        'algorithm':'Complete-prefix greedy recomputation; each next-token request is a fresh B1 live_v3 FP32 split read.',
        'max_new_tokens':32,'eos_token_ids':[248044,248046],'temperature':0.0,'memory_updates':0,
        'self':'Own final conversation snapshot.','empty':'Fresh empty state through the same FP32 split path.',
        'donor':'Fixed prepared_v1 partner final snapshot.','query_count':5958,
        'decode':'Continuation token IDs only; pinned tokenizer decode(skip_special_tokens=True), no answer cleanup.'},
      'full_context':'Separate unchanged full-history base condition on A800; no truncation, owned by its existing runner.',
      'unchanged':['All prepared files, valid input tokens, conversation/question population, donor pairs, prompts, EOS, 32-token limit, official scorer and bootstrap.'],
      'old_protocol_modified':False,'new_training':False,'new_gate':False,'answers_loaded_by_runtime':False,
      'pilot':{'sample_id':PILOT,'donor':'conv-47','questions':239,'short_requests':717,'selection_uses_scores':False}}

def load_amended_runtime(prepared,execution_plan,amendment):
    value=json.loads(Path(amendment).read_bytes())
    require(value==amendment_definition(),'delta-writer amendment differs from the explicit fixed method')
    data=load_runtime(prepared,execution_plan)
    data['amendment']=value;data['amendment_sha256']=digest(amendment)
    require(data['protocol']['generation']['max_new_tokens']==32
        and data['protocol']['generation']['stop_token_ids']==[248044,248046]
        and data['protocol']['generation']['temperature']==0.,'frozen decode contract differs')
    return data

def transport_frame_ids(frame):
    prefix,new,pad=frame['prefix_tokens'],frame['new_tokens'],frame['padding_tokens']
    require(type(prefix) is int and type(new) is int and type(pad) is int
        and 0<=prefix<=128 and 1<=new<=512 and pad in (0,1),'invalid LoCoMo frame lengths')
    n=prefix+new
    require(len(frame['input_ids'])==n+pad
        and frame['encoder_valid_mask']==[True]*n+[False]*pad
        and frame['write_mask']==[False]*prefix+[True]*new+[False]*pad
        and frame['position_ids']==list(range(n+pad))
        and (not pad or frame['input_ids'][-1]==248044),'frame mask/pad transport differs')
    ids=list(frame['input_ids'][:n])
    require(all(type(t) is int and 0<=t<248320 for t in ids),'invalid evidence token')
    return ids

def crop_writer_inputs(frame,hidden,physical):
    n=len(transport_frame_ids(frame));start=frame['prefix_tokens'];end=start+frame['new_tokens']
    require(hidden.shape==(1,n,2560) and physical.shape==(1,n,16),'uncropped evidence H/physical layout differs')
    return hidden[:,start:end],physical[:,start:end]


def validate_conversation_provenance(provenance,sample_id,data):
    """One exact provenance contract for every normal or orphan snapshot load."""
    frames=data['frames'][sample_id];conversation=data['conversations'][sample_id]
    require(provenance=={'protocol':PROTOCOL,'sample_id':sample_id,'frames':len(frames),
        'frame_sequence_sha256':canonical([f['frame_sha256'] for f in frames]),
        'written_tokens':conversation['token_count'],'writer_tensor_sha256':WRITER_TENSOR_SHA,
        'conversation_token_sha256':conversation['token_ids_sha256']},
        'snapshot conversation/frame coverage differs')

def ordered_conversations(data):
    return [PILOT,data['donors'][PILOT]]+[sid for sid in data['conversations'] if sid not in (PILOT,data['donors'][PILOT])]

def selected_questions(data,mode):
    require(mode in ('pilot','rest','formal'),'invalid execution mode')
    questions=data['questions']
    if mode=='pilot':return [q for q in questions if q['sample_id']==PILOT]
    if mode=='rest':return [q for q in questions if q['sample_id']!=PILOT]
    return [q for q in questions if q['sample_id']==PILOT]+[q for q in questions if q['sample_id']!=PILOT]

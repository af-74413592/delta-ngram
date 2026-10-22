"""Independent, preregistered LoCoMo teacher-forced NLL contract (CPU-safe)."""
from __future__ import annotations
import math
from qwen38_locomo_native import canonical, digest, require, write_json

PROTOCOL = 'qwen38_locomo_nll_v1'
CONDITIONS = ('self', 'empty', 'donor')
PARENT_SHA = '37826dd71c6fd3798fa5aa4ddf68031571461d6fcb376023ac2e0e3155b575d7'
VOCAB = 248320
HARDWARE_INVENTORY_SHA = '76754e851586fcc9986e45b0f33b5c1f48e216abd4a8a0388e8d3ca5c42446a2'
GPU_UUIDS = (
    'GPU-94d3fe7c-d91a-3dfc-7276-572c79968ad8','GPU-ee02e5c8-ab00-d911-4fca-1c3177ce7647',
    'GPU-cc14360a-2621-f819-b156-edd831967d86','GPU-407d9998-9019-74d5-be99-028635583c02',
    'GPU-a85c753d-d66a-f5ec-cf73-b5ff5bbc30a1','GPU-cdaa34f5-f5a8-d590-9b6e-67f774d7dd52',
    'GPU-547cc9c7-534d-dff4-f9f6-09f6c0335f14','GPU-f516f158-5bdb-2236-0380-585b94703dbd')
FORK_SOURCES = {
    'v1/worker/gpu/model_runner.py':'174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce',
    'v1/worker/gpu/sample/prompt_logprob.py':'4cf22d390e7bf59e44c458cff7180268db63e95d06af2f1e14d720a2d77c9663',
    'v1/worker/gpu/sample/logprob.py':'f47479a818192acb6b297c79fa67a3b2f9820e3492a394d5adda777e9b7c832e',
    'v1/engine/logprobs.py':'14bbe78d92a2c451d79b3d9dd9d5268bc57e16bad5079e44ce9ed2cd805e873e',
    'logprobs.py':'3cc2ed6cdd30572d29f9b293ddbc14ad318c6b5d5dc7fdd36bee2f8af259b373',
}
STATE_FIELDS = ('sample_id','selected_source','query_arm','phase','epochs','identities',
                'written_tokens','state_hashes','state_fields','writer_sha256',
                'frozen_reader_writer','binding_sha256')

def definition():
    return {
        'protocol': PROTOCOL, 'parent_binding_sha256': PARENT_SHA,
        'population': {'categories': [1,2,3,4], 'questions':1540, 'conditions':list(CONDITIONS),
                       'target_evaluations':4620, 'excluded_category5':446,
                       'exclusion_reason':'Category 5 uses refusal-substring accuracy; adversarial_answer is an incorrect lure, not a gold answer.'},
        'reference': {'count_per_question':1, 'canonical':'str(answer), except category 3 uses split(";")[0].strip().',
                      'category1':'Keep the complete answer string, including every comma-separated component.',
                      'container_types':'Reject list/dict/bool/null instead of guessing interchangeable references.',
                      'selection':'Never select a reference using model probabilities, scores, or predictions.'},
        'tokens': {'prompt':'Exact frozen short_prompt_ids; never re-tokenize the prompt or joined strings.',
                   'target':'tokenizer.encode(canonical_reference, add_special_tokens=False), without added whitespace, BOS, EOS, or chat terminator.',
                   'hashes':'Separate canonical SHA256 for prompt IDs and target IDs, plus concatenated input IDs.',
                   'loss_positions':'response.prompt_logprobs[P:P+A], where P=len(prompt_ids), A=len(target_ids).',
                   'discarded_sample':'One engine-required generated sample is discarded and excluded from targets and loss.',
                   'length_gate':'P+A+1 <= max_model_len and P+A <= max_num_batched_tokens; empty targets or overflow fail the complete preparation and report every affected qid. No truncation or silent dropping.'},
        'engine': {'route':'One native causal backbone prefill on prompt_ids + target_ids; reuse its hidden states for prompt-logprob LM heads.',
                   'hardware_inventory_sha256':HARDWARE_INVENTORY_SHA,
                   'hardware':'Eight UUID-bound NVIDIA A800-SXM4-40GB; observed nvidia-smi total is 40960 MiB per device, independent of CUDA allocatable capacity.',
                   'prompt_logprobs':0, 'logprobs_mode':'raw_logprobs', 'flat_logprobs':False, 'max_tokens':1,
                   'gpu_memory_utilization':0.80, 'max_model_len':641, 'max_num_batched_tokens':641,
                   'prefix_caching':False, 'chunked_prefill':False, 'head_chunk_positions':1024,
                   'normalization':'Full 248320-token vocabulary; FP32 exponential sum, log and gold subtraction after max selection. Only the gold probability is returned. LM-head GEMM retains model dtype.',
                   'penalties':'No presence/frequency/repetition penalty or custom logits processors; raw model probabilities.',
                   'states':'Reuse the same ten final snapshots and fixed donor mapping; same FP32 split PLE read path.',
                   'D1_commits':0, 'optimizer_steps':0, 'backward_calls':0,
                   'readonly':'All seven state fields and frozen identities agree on eight TP ranks before/after each conversation-condition block; existing per-request state guards remain active.'},
        'metrics': {'unit':'nats', 'query_mean_nll':'Mean over 1540 questions of each target token-mean NLL.',
                    'token_pooled_nll':'Sum of all target-token NLL divided by total target tokens.',
                    'paired_differences':['self-empty','donor-empty','self-donor'], 'negative_difference':'First condition is better.',
                    'bootstrap': {'cluster':'conversation','cluster_count':10,'replicates':10000,'seed':20261022,
                                  'paired':True,'percentile':[2.5,97.5],
                                  'weighting':'Resample complete conversations; ratio of resampled sums to question counts or token counts.'}},
        'firewalls': {'annotations':'Open only after all 7944 original four-condition predictions and final exits are sealed and audited.',
                      'GPU':'Full-context remaining 1747 completes first; CPU review and explicit root authorization precede a separate numerical/read-only/memory gate.',
                      'formal':'A successful technical gate and explicit root authorization are required; no automatic retry, fallback, prompt change, or state rewrite.'},
        'technical_gate': {'targets':'First frozen qid in conv-48; artificial target [42,43,44,45], plus repeated pattern filling input to 640 tokens. No benchmark reference selection.',
                           'conditions':['empty','self'],
                           'comparison_positions':'All four short target positions; fixed first/middle/last target positions at full input 640.',
                           'expected_comparisons':14, 'expected_native_forward_calls':18,
                           'comparison':'Full-prefill gold logprobs against independent fresh-prefix diagnostic full-vocabulary log_softmax at the preregistered target positions.',
                           'absolute_tolerance_nats':0.005, 'tolerance_status':'Upper bound only, not bitwise equivalence; record actual maximum and discuss effects near its magnitude.',
                           'checks':['P:P+A alignment','one backbone call','expected LM-head row counts','finite full-vocab probabilities','all eight-rank state hashes unchanged','allocator peak and fresh GPU exit']},
        'new_prompt_or_training':False, 'original_generation_untouched':True,
    }

def canonical_reference(annotation):
    category = annotation['category']
    require(type(category) is int and category in range(1,6), 'unknown annotation category')
    if category == 5:
        return None
    answer = annotation.get('answer')
    require(type(answer) in (str,int,float) and (not isinstance(answer,float) or math.isfinite(answer)),
            'answer must be a scalar canonical reference; alternative/reference containers are not inferred')
    text = str(answer)
    if category == 3:
        text = text.split(';')[0].strip()
    require(bool(text), 'empty canonical reference')
    return text

def valid_ids(ids, label):
    require(type(ids) is list and len(ids)>0 and all(type(x) is int and 0<=x<VOCAB for x in ids), label+' token IDs are invalid')

def target_record(question, annotation, tokenizer, *, max_model_len=641, max_num_batched_tokens=641):
    require(annotation['qid']==question['qid'] and annotation['sample_id']==question['sample_id'], 'annotation/question identity differs')
    reference = canonical_reference(annotation)
    if reference is None:
        return None
    prompt = list(question['short_prompt_ids']); valid_ids(prompt, 'prompt')
    require(canonical(prompt)==question['short_prompt_sha256'], 'frozen prompt changed')
    target = list(tokenizer.encode(reference, add_special_tokens=False)); valid_ids(target, 'target')
    require(len(prompt)+len(target)+1<=max_model_len and len(prompt)+len(target)<=max_num_batched_tokens,
            'reference exceeds the exact frozen engine length limits')
    return {'qid':question['qid'],'sample_id':question['sample_id'],'category':annotation['category'],
            'reference':reference,'reference_sha256':canonical(reference),'prompt_ids':prompt,'target_ids':target,
            'prompt_sha256':canonical(prompt),'target_sha256':canonical(target),'input_sha256':canonical(prompt+target),
            'prompt_tokens':len(prompt),'target_tokens':len(target)}

def build_targets(questions, annotations, tokenizer, *, expected_questions=1986,
                  expected_main=1540, expected_excluded=446, max_model_len=641, max_num_batched_tokens=641):
    lookup={a['qid']:a for a in annotations}
    require(len(lookup)==len(annotations)==len(questions)==expected_questions
            and len({q['qid'] for q in questions})==expected_questions
            and set(lookup)=={q['qid'] for q in questions},'reference/question population differs')
    targets=[];excluded=[];failures=[]
    for question in questions:
        annotation=lookup[question['qid']]
        try:
            row=target_record(question,annotation,tokenizer,max_model_len=max_model_len,
                              max_num_batched_tokens=max_num_batched_tokens)
            if row is None: excluded.append(question['qid'])
            else: targets.append(row)
        except (ValueError,TypeError,KeyError) as error:
            failures.append({'qid':question['qid'],'error_type':type(error).__name__,'reason':str(error)})
    report={'expected_main_questions':expected_main,'target_evaluations':expected_main*3,
            'prepared_main_questions':len(targets),'excluded_category5_qids':excluded,
            'failures':failures,'pending_evaluations':len(failures)*3,'truncated_targets':0,
            'status':'passed' if not failures and len(targets)==expected_main and len(excluded)==expected_excluded else 'failed'}
    return targets,report

def gold_logprobs(response, prompt_ids, target_ids):
    valid_ids(prompt_ids,'prompt');valid_ids(target_ids,'target')
    whole = prompt_ids+target_ids
    require(response.finished and list(response.prompt_token_ids)==whole and len(response.outputs)==1
            and len(response.outputs[0].token_ids)==1, 'engine returned a different NLL request')
    rows = response.prompt_logprobs
    require(type(rows) is list and len(rows)==len(whole) and rows[0] is None, 'prompt-logprob boundary/leading None differs')
    values=[]
    for token, row in zip(target_ids,rows[len(prompt_ids):]):
        require(isinstance(row,dict) and set(row)=={token}, 'gold token missing or unexpected top-k output')
        value=float(row[token].logprob)
        require(math.isfinite(value) and value<=1e-6, 'gold log probability is not finite/nonpositive')
        values.append(value)
    require(len(values)==len(target_ids), 'target loss positions were dropped')
    return values

def validate_head_rows(rows, input_tokens, *, diagnostics=False):
    # The sealed fork computes N head rows and drops the final unused probability
    # after the head; only N-1 entries are exposed behind the leading None.
    expected=[1] if diagnostics else [1]+[min(1024,input_tokens-i) for i in range(0,input_tokens,1024)]
    require(rows==expected, 'LM-head call rows differ from one sample plus complete prompt-logprob chunks')

def check_readonly(before, after):
    require(all(before[k]==after[k] for k in STATE_FIELDS), 'NLL changed a seven-field state or frozen identity')

def result_row(target, condition, source, state_hash, epoch, values, observation, binding_sha):
    require(condition in CONDITIONS and len(values)==target['target_tokens'], 'NLL result population differs')
    require(all(math.isfinite(v) and v<=1e-6 for v in values), 'invalid NLL values')
    loss=-math.fsum(values)
    return {'protocol':PROTOCOL,'qid':target['qid'],'sample_id':target['sample_id'],'category':target['category'],
            'condition':condition,'source_memory_sample_id':source,'prompt_sha256':target['prompt_sha256'],
            'target_sha256':target['target_sha256'],'reference_sha256':target['reference_sha256'],
            'prompt_tokens':target['prompt_tokens'],'target_tokens':target['target_tokens'],
            'token_logprobs':values,'nll_sum':loss,'nll_token_mean':loss/len(values),'unit':'nats',
            'state_fingerprint_before':state_hash,'state_fingerprint_after':state_hash,'state_epoch':epoch,
            'native_forward_calls':observation['forward_calls'],'head_rows':observation['nll_head_rows'],
            'parent_binding_sha256':PARENT_SHA,'nll_binding_sha256':binding_sha,
            'memory_updates':0,'teacher_forced':True,'discarded_sample_tokens':1,'eos_loss_added':False}

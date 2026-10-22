"""Independent CPU preparation, numerical gate and readonly LoCoMo NLL runner.

Annotations stay behind the completed original four-condition generation seal.
The original generation directories, sources and bindings are never modified.
GPU gate/formal actions require separate explicit root authorization.
"""
from __future__ import annotations
import argparse,copy,datetime,json,math,os,sys,time
from pathlib import Path
from qwen38_locomo_nll_contract import (PROTOCOL,PARENT_SHA,CONDITIONS,STATE_FIELDS,definition,
    canonical,digest,require,write_json,build_targets,check_readonly,result_row,FORK_SOURCES,GPU_UUIDS)
from qwen38_locomo_native import TOKENIZER_FILES

OWN=('qwen38_locomo_nll.py','qwen38_locomo_nll_contract.py','qwen38_locomo_nll_engine.py','qwen38_locomo_nll_worker.py','qwen38_locomo_nll_summary.py','qwen38_locomo_nll_supervisor.py')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_bytes())
def jsonl(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def source_hashes():return {name:digest(Path(__file__).with_name(name)) for name in OWN}

def verify_nll_dependencies(parent):
    model=Path(parent['base_model']['model_path']);fork=Path(parent['fork']['package_root'])
    for name,sha in TOKENIZER_FILES.items():require(digest(model/name)==sha,'original tokenizer changed: '+name)
    for name,sha in FORK_SOURCES.items():require(digest(fork/name)==sha,'audited prompt-logprob source changed: '+name)
    return {'tokenizer_files':dict(TOKENIZER_FILES),'fork_sources':dict(FORK_SOURCES)}

def generation_firewall(provenance, provenance_sha, execution, execution_sha):
    """Validate only provenance/hashes/exits before any annotation read."""
    provenance=Path(provenance);execution=Path(execution)
    require(digest(provenance)==provenance_sha and digest(execution)==execution_sha,'original generation seal changed')
    seal=read(provenance);exit_receipt=read(execution)
    require(seal['protocol']=='qwen38_locomo_four_condition_assembly_v2' and seal['status']=='validated'
            and seal['population']=={'unique_pairs':7944,'questions':1986,'conversations':10,
                'per_condition':{'self':1986,'empty':1986,'donor':1986,'full_context':1986}},
            'all original four-condition generations must be sealed before references open')
    require(exit_receipt['returncode']==0 and exit_receipt['CPU_only'] is True
            and isinstance(exit_receipt['finished'],(int,float)), 'original assembly has not completed successfully')
    combined=provenance.parent/'combined_predictions.jsonl'
    require(digest(combined)==seal['combined_predictions_sha256'],'sealed original combined generations changed')
    require(set(seal['source_predictions'])=={'delta','full_context'},'original four-condition source receipts missing')
    for item in seal['source_predictions'].values():
        require(digest(item['path'])==item['sha256'],'original source generations changed')
    receipts=seal['completion_receipts'];require(len(receipts)==4,'original generation completion/exit receipts missing')
    for path,sha in receipts.items():require(digest(path)==sha,'original generation completion/exit receipt changed')
    return {'provenance':{'path':str(provenance.resolve()),'sha256':provenance_sha},
            'execution':{'path':str(execution.resolve()),'sha256':execution_sha},
            'combined_predictions':{'path':str(combined.resolve()),'sha256':seal['combined_predictions_sha256']},
            'sealed_original_predictions':7944,'annotations_were_opened_before_seal':False}

def prepare(options):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    require(not options.output.exists(),'NLL preparation directory must be new')
    firewall=generation_firewall(options.generation_provenance,options.generation_provenance_sha256,
                                options.generation_execution,options.generation_execution_sha256)
    from qwen38_locomo_delta import verify_binding
    parent,data=verify_binding(options.parent_binding)
    require(digest(options.parent_binding/'binding.json')==PARENT_SHA,'NLL requires the completed v2 original delta binding')
    parent_root=Path(parent['locomo']['execution_root']);completion=read(parent_root/'COMPLETE.json')
    require(completion['status']=='completed' and completion['predictions']==5958
            and completion['binding_sha256']==PARENT_SHA,'original short generations are incomplete')
    index=read(parent_root/'state_index.json');require(len(index['states'])==10,'ten final conversation states are required')
    for sid,item in index['states'].items():
        require(Path(item['checkpoint']['path']).resolve()==(parent_root/'states'/(sid+'.state.pt')).resolve()
                and digest(item['checkpoint']['path'])==item['checkpoint']['sha256'],'original final snapshot changed')
    import torch
    from transformers import AutoTokenizer
    require(not torch.cuda.is_initialized(),'CPU NLL preparation initialized CUDA')
    dependencies=verify_nll_dependencies(parent)
    tokenizer=AutoTokenizer.from_pretrained(parent['base_model']['model_path'],local_files_only=True)
    # This is the only actual annotation read. The original generation firewall
    # above has completed before execution can reach this point.
    annotations_path=Path(parent['locomo']['prepared'])/'scoring'/'annotations.jsonl'
    annotation_entry=read(Path(parent['locomo']['prepared'])/'manifest.json')['files']['scoring/annotations.jsonl']
    require(digest(annotations_path)==annotation_entry['sha256'] and annotations_path.stat().st_size==annotation_entry['bytes'],
            'original prepared annotation file changed')
    annotations=jsonl(annotations_path)
    targets,report=build_targets(data['questions'],annotations,tokenizer)
    require(verify_nll_dependencies(parent)==dependencies,'tokenizer/fork changed during target preparation')
    options.output.mkdir(parents=True,exist_ok=False)
    write_json(options.output/'target_preflight.json',dict(report,utc=utc(),annotations_sha256=digest(annotations_path),firewall=firewall))
    require(report['status']=='passed','NLL target preparation failed; all pending qids are preserved in target_preflight.json')
    target_path=options.output/'targets.jsonl'
    target_path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in targets),encoding='utf-8')
    protocol_path=options.output/'protocol.json';write_json(protocol_path,definition())
    binding={'protocol':PROTOCOL,'created_utc':utc(),'parent_binding':str(options.parent_binding.resolve()),
        'parent_binding_sha256':PARENT_SHA,'generation_firewall':firewall,'source_sha256':source_hashes(),
        'protocol_sha256':digest(protocol_path),'targets_path':str(target_path.resolve()),'targets_sha256':digest(target_path),
        'target_questions':1540,'target_evaluations':4620,'excluded_category5':446,
        'annotations':{'path':str(annotations_path),'sha256':digest(annotations_path)},
        'state_index':{'path':str((parent_root/'state_index.json').resolve()),'sha256':digest(parent_root/'state_index.json')},
        'parent_complete':{'path':str((parent_root/'COMPLETE.json').resolve()),'sha256':digest(parent_root/'COMPLETE.json')},
        'capacity':{'gpu_memory_utilization':.80,'max_model_len':641,'max_num_batched_tokens':641},
        'nll_dependencies':dependencies,
        'target_token_count':sum(r['target_tokens'] for r in targets),'max_input_tokens':max(r['prompt_tokens']+r['target_tokens'] for r in targets),
        'CPU_only':True,'cuda_initialized':False,'GPU_fit_verified':False}
    write_json(options.output/'binding.json',binding)
    write_json(options.output/'BINDING_COMPLETE.json',{'protocol':PROTOCOL,'binding_sha256':digest(options.output/'binding.json'),
        'CPU_only':True,'cuda_initialized':False,'target_questions':1540,'target_evaluations':4620})
    return {'status':'CPU_prepared','binding_sha256':digest(options.output/'binding.json'),'target_questions':1540,
            'target_tokens':binding['target_token_count'],'max_input_tokens':binding['max_input_tokens']}

def verify(options):
    root=options.binding;seal=read(root/'BINDING_COMPLETE.json');binding=read(root/'binding.json')
    require(seal['protocol']==PROTOCOL and digest(root/'binding.json')==seal['binding_sha256'],'NLL binding seal differs')
    require(binding['protocol']==PROTOCOL and binding['source_sha256']==source_hashes()
            and read(root/'protocol.json')==definition() and digest(root/'protocol.json')==binding['protocol_sha256'],'NLL sources/protocol changed')
    fw=binding['generation_firewall']
    generation_firewall(fw['provenance']['path'],fw['provenance']['sha256'],fw['execution']['path'],fw['execution']['sha256'])
    for key in ('annotations','state_index','parent_complete'):
        require(digest(binding[key]['path'])==binding[key]['sha256'],key+' changed')
    require(digest(binding['targets_path'])==binding['targets_sha256'],'NLL targets changed')
    from qwen38_locomo_delta import verify_binding
    parent,data=verify_binding(binding['parent_binding'])
    require(digest(Path(binding['parent_binding'])/'binding.json')==PARENT_SHA,'parent binding changed')
    require(verify_nll_dependencies(parent)==binding['nll_dependencies'],'NLL tokenizer/fork dependencies changed')
    for item in read(binding['state_index']['path'])['states'].values():
        require(digest(item['checkpoint']['path'])==item['checkpoint']['sha256'],'frozen NLL source snapshot changed')
    targets=jsonl(binding['targets_path']);require(len(targets)==1540 and len({r['qid'] for r in targets})==1540,'NLL target population differs')
    questions={q['qid']:q for q in data['questions']}
    for row in targets:
        require(row['prompt_ids']==questions[row['qid']]['short_prompt_ids']
            and canonical(row['prompt_ids'])==row['prompt_sha256'] and canonical(row['target_ids'])==row['target_sha256']
            and canonical(row['prompt_ids']+row['target_ids'])==row['input_sha256']
            and row['target_tokens']==len(row['target_ids'])>0 and row['prompt_tokens']==len(row['prompt_ids'])
            and row['prompt_tokens']+row['target_tokens']<=640,'NLL target token boundary/capacity differs')
    return binding,parent,data,targets

def initialize(options,binding,parent):
    import torch
    require(sys.platform=='linux' and not torch.cuda.is_initialized(),'NLL requires a fresh Linux process')
    from qwen38_locomo_nll_supervisor import gpu_status
    _,_,idle=gpu_status();require(idle,'NLL initialization requires the exact eight UUID-bound idle A800 GPUs')
    import qwen38_vllm_accuracy as accuracy
    from qwen38_vllm_delta_accuracy import verify_fork
    from qwen38_native_runtime_preflight import verified_model
    from delta_ngram import qwen38_vllm_live as live
    current=accuracy.hf_model_identity(Path(parent['base_model']['model_path']))
    current['base_hash_audit']=verified_model(Path(parent['base_model']['model_path']),Path(parent['base_verification']['path']))
    current['verification_sha256']=current['base_hash_audit']['verification_sha256']
    require(current==parent['base_model'],'NLL base model changed')
    os.environ.update({live.ENV:str(Path(binding['parent_binding'])/'binding.json'),live.ENV_SHA:PARENT_SHA,
        'VLLM_WORKER_MULTIPROC_METHOD':'spawn','VLLM_ENABLE_V1_MULTIPROCESSING':'1','VLLM_USE_V2_MODEL_RUNNER':'1',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'})
    import vllm,vllm.envs as envs
    from vllm import LLM,SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from qwen38_vllm_live_two_chain import LiveEngine,agree
    require(verify_fork(parent['vllm_fork_manifest']['path'],vllm.__file__)==parent['fork'],'NLL imported another vLLM fork')
    kwargs=copy.deepcopy(parent['requested_engine_settings']);kwargs.update(binding['capacity'])
    kwargs.update(logprobs_mode='raw_logprobs',worker_extension_cls='qwen38_locomo_nll_worker.LoCoMoNLLWorker')
    accuracy.configure_ple_cpu_offload(kwargs,EngineArgs,envs,requested=True)
    planned=EngineArgs(**copy.deepcopy(kwargs)).create_engine_config()
    require(planned.use_v2_model_runner and planned.model_config.max_model_len==641
        and planned.model_config.logprobs_mode=='raw_logprobs' and not planned.cache_config.enable_prefix_caching
        and not planned.scheduler_config.enable_chunked_prefill and planned.scheduler_config.max_num_batched_tokens==641
        and planned.speculative_config is None,'NLL engine configuration differs')
    write_json(options.output/'engine_configuration.json',{'kwargs':kwargs,'execution':accuracy.execution_config(kwargs,planned,envs,ple_cpu_offload_requested=True)})
    llm=LLM(**copy.deepcopy(kwargs))
    require(accuracy.execution_config(kwargs,llm.llm_engine.vllm_config,envs,ple_cpu_offload_requested=True)
        ==accuracy.execution_config(kwargs,planned,envs,ple_cpu_offload_requested=True),'resolved NLL engine changed')
    engine=LiveEngine(llm,SamplingParams,options.output)
    ready=agree(engine.rpc('delta_live_initialize',parent['writers'],timeout=600),
        ['binding_sha256','cache_dtype','quiescent','adapter_protocol','adapter_sha256','state_adapter_sha256',
         'empty_state_hashes','nll_readonly_worker','nll_worker_sha256'],label='NLL initialize')
    require(ready['binding_sha256']==PARENT_SHA and ready['cache_dtype']=='torch.float32' and ready['quiescent']
        and ready['nll_readonly_worker'] and ready['nll_worker_sha256']==binding['source_sha256']['qwen38_locomo_nll_worker.py'],
        'NLL initialization source/state differs')
    return engine,ready

def select(engine,index,sid,condition,donors,empty_hash):
    from qwen38_vllm_live_two_chain import agree
    source=None if condition=='empty' else sid if condition=='self' else donors[sid]
    receipt=None if source is None else index['states'][source]['checkpoint']
    before=agree(engine.rpc('locomo_select_snapshot',source,receipt),list(STATE_FIELDS),label='NLL snapshot select')
    arm='ordinary' if source is None else 'delta'
    expected=empty_hash if source is None else index['states'][source]['state_hashes']['delta']
    require(before['selected_source']==source and before['query_arm']==arm and before['binding_sha256']==PARENT_SHA
        and before['phase']==('empty' if source is None else 'frozen') and before['state_hashes'][arm]==expected,
        'NLL selected a different conversation/empty state')
    engine.epochs=dict(before['epochs']);return source,before

def readonly(engine,before,path):
    from qwen38_vllm_live_two_chain import agree
    ranks=engine.rpc('locomo_state_status');after=agree(ranks,list(STATE_FIELDS),label='NLL seven-field readonly audit')
    check_readonly(before,after)
    write_json(path,{'protocol':PROTOCOL,'before':before,'after':after,'resources':ranks,'all_seven_fields_identical':True})
    return ranks

def synthetic_gate(options,binding,parent,data,engine,ready,index):
    import torch
    from qwen38_locomo_nll_engine import NLLRequestEngine
    nll=NLLRequestEngine(engine)
    # Fixed original pilot conversation and first frozen qid; no reference or
    # probability is used to select any gate input or comparison position.
    q=next(q for q in data['questions'] if q['sample_id']=='conv-48');prompt=q['short_prompt_ids']
    patterns={'short':[42,43,44,45],'full_input_640':([42,43,44,45]*160)[:640-len(prompt)]}
    comparisons=[];resources=[];request_count=0
    for condition in ('empty','self'):
        source,before=select(engine,index,'conv-48',condition,data['donors'],ready['empty_state_hashes']['ordinary']);arm=before['query_arm']
        for case,target_ids in patterns.items():
            target={'prompt_ids':prompt,'target_ids':target_ids,'input_sha256':canonical(prompt+target_ids)}
            values,observation=nll.request(arm,target,request_id=f'nll-gate:{condition}:{case}')
            request_count+=1
            positions=list(range(4)) if case=='short' else [0,len(target_ids)//2,len(target_ids)-1]
            for position in positions:
                prefix=prompt+target_ids[:position]
                params=engine.params_cls(n=1,temperature=0.,top_p=1.,top_k=-1,seed=17,max_tokens=1,
                    ignore_eos=True,stop_token_ids=[],detokenize=False,
                    presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.)
                _,obs=engine.request(arm,'query',prefix,params,request_id=f'nll-gate-fresh:{condition}:{case}:{position}',diagnostics=True)
                request_count+=1
                receipt=obs['logits_file'];require(digest(receipt['path'])==receipt['sha256'],'diagnostic full logits changed')
                logits=torch.load(receipt['path'],map_location='cpu',weights_only=True)['logits'][0][0]
                require(logits.numel()==248320 and bool(logits.isfinite().all()),'diagnostic full vocabulary differs')
                fresh=float(torch.log_softmax(logits.float(),dim=-1)[target_ids[position]])
                comparisons.append({'condition':condition,'case':case,'target_position':position,'target_token':target_ids[position],
                    'full_prefill_logprob':values[position],'fresh_prefix_logprob':fresh,'abs_difference_nats':abs(values[position]-fresh),
                    'prefix_sha256':canonical(prefix),'logits_file':receipt,'nll_observation':observation,
                    'fresh_forward_calls':obs['forward_calls'],'captured_logits_dtype':str(logits.dtype)})
        resources.append(readonly(engine,before,options.output/f'READONLY_gate_{condition}.json'))
    maximum=max(r['abs_difference_nats'] for r in comparisons)
    require(len(comparisons)==14 and request_count==18
        and all(math.isfinite(r['abs_difference_nats']) for r in comparisons), 'synthetic gate coverage/nonfinite values differ')
    result={'protocol':PROTOCOL,'status':'passed' if maximum<=.005 else 'failed','nll_binding_sha256':digest(options.binding/'binding.json'),
        'fixed_question_qid':q['qid'],'synthetic_targets_only':True,'actual_max_abs_difference_nats':maximum,
        'absolute_tolerance_nats':.005,'bitwise_equivalence_claimed':False,'comparisons':comparisons,'resources':resources,
        'native_forward_calls':request_count,'D1_commits':0,'optimizer_steps':0,'backward_calls':0,
        'numerical_effect_caveat':'Effects near the observed numerical difference require sensitivity discussion.',
        'full_input_stress_tokens':640,'finished_utc':utc()}
    write_json(options.output/'GATE.json',result)
    require(result['status']=='passed','NLL numerical gate failed; no automatic fallback or formal execution')
    return result

def formal(options,binding,parent,data,targets,engine,ready,index):
    from qwen38_locomo_nll_engine import NLLRequestEngine
    from qwen38_locomo_delta import append_event
    nll=NLLRequestEngine(engine);rows=[];resources=[]
    (options.output/'rows').mkdir();events=options.output/'execution_events.jsonl'
    for sid in dict.fromkeys(r['sample_id'] for r in targets):
        selected=[r for r in targets if r['sample_id']==sid]
        for condition in CONDITIONS:
            source,before=select(engine,index,sid,condition,data['donors'],ready['empty_state_hashes']['ordinary']);arm=before['query_arm']
            for target in selected:
                append_event(events,{'phase':'before_nll','qid':target['qid'],'condition':condition,'input_sha256':target['input_sha256'],
                    'target_sha256':target['target_sha256'],'input_tokens':target['prompt_tokens']+target['target_tokens']})
                values,obs=nll.request(arm,target,request_id=f"nll:{target['qid']}:{condition}")
                row=result_row(target,condition,source,before['state_hashes'][arm],before['epochs'][arm],values,obs,digest(options.binding/'binding.json'))
                path=options.output/'rows'/(target['qid'].replace(':','__')+'--'+condition+'.json');write_json(path,row)
                append_event(events,{'phase':'nll_complete','qid':target['qid'],'condition':condition,'row_sha256':digest(path),'observation':obs})
                rows.append(row)
            resources.append(readonly(engine,before,options.output/f'READONLY_{sid}_{condition}.json'))
    require(len(rows)==4620 and len({(r['qid'],r['condition']) for r in rows})==4620,'NLL result matrix is incomplete')
    path=options.output/'nll_results.jsonl';path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
    from qwen38_locomo_nll_summary import summarize
    summary_path=options.output/'summary.json';write_json(summary_path,summarize(rows))
    verify(options)
    result={'protocol':PROTOCOL,'status':'completed','nll_binding_sha256':digest(options.binding/'binding.json'),
        'questions':1540,'evaluations':4620,'excluded_category5':446,'D1_commits':0,'optimizer_steps':0,'backward_calls':0,
        'native_forward_calls':4620,'result_path':str(path.resolve()),'result_sha256':digest(path),
        'summary_path':str(summary_path.resolve()),'summary_sha256':digest(summary_path),
        'resources':resources,'finished_utc':utc()}
    write_json(options.output/'COMPLETE.json',result);return result

def execute(options):
    require(options.root_authorized,'GPU execution needs explicit root authorization after full-context completion')
    require(not options.output.exists(),'NLL GPU output directory must be new; no automatic resume/retry')
    binding,parent,data,targets=verify(options)
    if options.action=='formal':
        require(options.gate is not None and options.gate_sha256 is not None,'formal requires a sealed successful NLL gate')
        require(digest(options.gate)==options.gate_sha256,'NLL gate changed')
        gate=read(options.gate);require(gate['status']=='passed' and gate['nll_binding_sha256']==digest(options.binding/'binding.json'),
                                      'NLL gate belongs to another binding or failed')
        require(options.gate_execution is not None and options.gate_execution_sha256 is not None,
                'formal requires successful gate process exit and released GPUs')
        validate_gate_exit(options.gate,options.gate_sha256,options.gate_execution,options.gate_execution_sha256,
                           digest(options.binding/'binding.json'))
    options.output.mkdir(parents=True,exist_ok=False)
    write_json(options.output/'STARTED.json',{'protocol':PROTOCOL,'action':options.action,'utc':utc(),
        'nll_binding_sha256':digest(options.binding/'binding.json'),'automatic_retry':False})
    try:
        engine,ready=initialize(options,binding,parent);index=read(binding['state_index']['path'])
        result=synthetic_gate(options,binding,parent,data,engine,ready,index) if options.action=='gate' else formal(options,binding,parent,data,targets,engine,ready,index)
        verify(options)
        return {'status':result['status'],'output':str(options.output.resolve())}
    except BaseException as error:
        write_json(options.output/'FAILED.json',{'protocol':PROTOCOL,'error_type':type(error).__name__,'message':str(error),'utc':utc(),'automatic_retry':False})
        raise

def validate_gate_exit(gate_path,gate_sha,exit_path,exit_sha,binding_sha):
    require(digest(exit_path)==exit_sha,'NLL gate process exit receipt changed')
    receipt=read(exit_path);gate=read(gate_path)
    require(receipt['protocol']==PROTOCOL and receipt['action']=='gate' and type(receipt['returncode']) is int and receipt['returncode']==0
        and receipt['error'] is None
        and receipt['nll_binding_sha256']==binding_sha and receipt['gate_sha256']==gate_sha
        and Path(receipt['gate_path']).resolve()==Path(gate_path).resolve()
        and receipt['gpu_idle_after_exit'] is True and receipt['gpu_compute_apps_after_exit'].strip()==''
        and len(receipt['gpu_memory_after_exit'])==8
        and sorted(r['index'] for r in receipt['gpu_memory_after_exit'])==list(range(8))
        and all(r['memory_used_mib']<1000 and r['uuid']==GPU_UUIDS[r['index']]
                and r['name']=='NVIDIA A800-SXM4-40GB' and r['memory_total_mib']==40960
                for r in receipt['gpu_memory_after_exit'])
        and receipt['finished_epoch']>=receipt['started_epoch'],
        'NLL gate did not finish successfully and release all eight GPUs')
    gate_time=datetime.datetime.fromisoformat(gate['finished_utc']).timestamp()
    require(receipt['started_epoch']<=gate_time<=receipt['finished_epoch'],'NLL gate timestamp lies outside its process execution')
    return receipt

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    for name in ('parent-binding','generation-provenance','generation-execution','output'):prep.add_argument('--'+name,type=Path,required=True)
    for name in ('generation-provenance-sha256','generation-execution-sha256'):prep.add_argument('--'+name,required=True)
    check=sub.add_parser('verify');check.add_argument('--binding',type=Path,required=True)
    for action in ('gate','formal'):
        p=sub.add_parser(action);p.add_argument('--binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        p.add_argument('--root-authorized',action='store_true');p.add_argument('--gate',type=Path);p.add_argument('--gate-sha256')
        p.add_argument('--gate-execution',type=Path);p.add_argument('--gate-execution-sha256')
    options=parser.parse_args()
    if options.action=='verify':
        os.environ['CUDA_VISIBLE_DEVICES']='';binding,_,_,targets=verify(options)
        import torch
        require(not torch.cuda.is_initialized(),'CPU verification initialized CUDA')
        result={'status':'CPU_verified','nll_binding_sha256':digest(options.binding/'binding.json'),
                'target_questions':len(targets),'cuda_initialized':False}
    else:result=prepare(options) if options.action=='prepare' else execute(options)
    print(json.dumps(result,allow_nan=False))

if __name__=='__main__':main()

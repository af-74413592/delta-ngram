"""Frozen A800 LoCoMo delta-writer self/empty/donor evaluation.

CPU preparation binds the original live_v3 sources, checkpoint, data and this
independent adapter. GPU execution is an explicit separate action. Each
conversation starts empty, writes each new evidence span once, and exports a
final seven-field snapshot. Queries only borrow those snapshots.
"""
from __future__ import annotations
import argparse,copy,datetime,importlib.metadata,json,os,sys,time
from contextlib import contextmanager
from pathlib import Path
from qwen38_locomo_delta_contract import (PROTOCOL,WRITER_SHA,WRITER_TENSOR_SHA,HASH_ALGORITHM,
    ENGINE_CONTEXT,NEW_TOKENS,CONDITIONS,PILOT,PREPARED_SHA,PROTOCOL_SHA,TOKENIZER_FILES,
    canonical,digest,load_amended_runtime,ordered_conversations,selected_questions,
    transport_frame_ids,validate_conversation_provenance,require,write_json)

OWN_NAMES=('qwen38_locomo_delta.py','qwen38_locomo_delta_contract.py','qwen38_locomo_delta_state.py',
           'qwen38_locomo_delta_worker.py','qwen38_locomo_native.py')

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def configure_paths(delta_repo,formal_scripts):
    for path in (Path(delta_repo)/'src',Path(delta_repo)/'scripts/native_qwen38',Path(formal_scripts)):
        sys.path.insert(0,str(path.resolve()))
    sys.path.insert(0,str(Path(__file__).resolve().parent))

def read_writer_checkpoint(path):
    import torch
    from delta_ngram.qwen38_probe import tensor_state_hash
    from delta_ngram.qwen38_native import make_qwen38_writer
    path=Path(path);require(digest(path)==WRITER_SHA,'delta writer checkpoint archive differs')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    require(saved['protocol']=='qwen38_native_memory_fork_checkpoint_v1' and saved['policy']['branch']=='full'
        and saved['completed_groups']==835 and saved['completed_targets']==20015113,'delta branch endpoint differs')
    source=saved['branch_state']
    require(source['model_optimizer_step']==4172 and source['completed_loss_targets']==99997072
        and tensor_state_hash(source['writer'])==WRITER_TENSOR_SHA,'delta writer tensor identity/endpoint differs')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17);writer,alpha=make_qwen38_writer(alpha=1.)
        writer.load_state_dict(source['writer'],strict=True);writer.requires_grad_(False).eval();alpha.requires_grad_(False).eval()
    require(tensor_state_hash(writer.state_dict())==WRITER_TENSOR_SHA,'loaded CPU delta writer differs')
    require(all(torch.isfinite(value).all() for value in writer.state_dict().values()),'nonfinite writer')
    require(digest(path)==WRITER_SHA,'checkpoint changed while loading')
    return {'path':str(path.resolve()),'sha256':WRITER_SHA,'tensor_sha256':WRITER_TENSOR_SHA,
        'tensor_hash_algorithm':HASH_ALGORITHM,'model_optimizer_step':4172,'model_completed_loss_targets':99997072,
        'branch_completed_groups':835,'branch_completed_targets':20015113,'checkpoint_memory_loaded_as_initial_state':False}

def engine_settings(model):
    import qwen38_vllm_accuracy as accuracy
    kwargs=accuracy.engine_kwargs(str(model),enforce_eager=True,gpu_memory_utilization=.85,tensor_parallel_size=8,moe_backend='triton')
    kwargs.update(max_model_len=ENGINE_CONTEXT,max_num_seqs=1,max_num_batched_tokens=ENGINE_CONTEXT,
        enable_prefix_caching=False,enable_chunked_prefill=False,mamba_cache_mode='none',speculative_config=None,
        worker_extension_cls='qwen38_locomo_delta_worker.LoCoMoDeltaWorker')
    return kwargs


def runtime_versions():
    import torch
    # The wheel metadata omits CUDA's local suffix; the frozen tokenizer receipt
    # records the actual imported torch build (for example 2.13.0+cu130).
    return {'torch':torch.__version__,**{name:importlib.metadata.version(name)
        for name in ('vllm','transformers','tokenizers')}}

def source_identity(delta_repo,formal_scripts):
    import qwen38_vllm_live_worker,qwen38_vllm_live_two_chain,qwen38_vllm_accuracy,qwen38_vllm_delta_accuracy,qwen38_greedy_decode
    modules=(qwen38_vllm_live_worker,qwen38_vllm_live_two_chain,qwen38_vllm_accuracy,qwen38_vllm_delta_accuracy,qwen38_greedy_decode)
    require(all(Path(m.__file__).resolve().parent==Path(formal_scripts).resolve() for m in modules),
            'original live modules must resolve inside the sealed live_v3 scripts directory')
    native=Path(delta_repo).resolve()
    paths=[*(native/'src/delta_ngram').rglob('*.py'),*(native/'scripts/native_qwen38').glob('*.py')]
    paths += [Path(m.__file__).resolve() for m in modules]
    paths += [Path(__file__).with_name(name).resolve() for name in OWN_NAMES]
    return {str(path):digest(path) for path in sorted(set(paths))}

def build_binding(options):
    import torch
    import qwen38_vllm_accuracy as accuracy
    from qwen38_native_runtime_preflight import verified_model
    from qwen38_vllm_delta_accuracy import verify_fork
    from delta_ngram import qwen38_vllm_live as live
    torch.set_num_threads(2)
    require(not any(value for key,value in os.environ.items() if key.startswith('DELTA_NGRAM_VLLM')),
            'CPU binding requires a clean process without an active Delta binding')
    data=load_amended_runtime(options.prepared,options.execution_plan,options.amendment)
    reference=json.loads(options.reference_binding.read_bytes())
    require(reference['protocol']==live.PROTOCOL and reference['writers']['delta']['sha256']==WRITER_SHA
        and reference['writers']['delta']['tensor_sha256']==WRITER_TENSOR_SHA,'reference is not the original formal delta binding')
    for path,expected in reference['source_sha256'].items():
        require(digest(path)==expected,'sealed original live_v3 source changed: '+path)
    writer=read_writer_checkpoint(options.writer_checkpoint)
    require(writer['path']==reference['writers']['delta']['path'],'delta writer path differs from the original formal writer')
    identity=accuracy.hf_model_identity(options.model)
    identity['base_hash_audit']=verified_model(options.model,options.base_verification)
    identity['verification_sha256']=identity['base_hash_audit']['verification_sha256']
    require(identity==reference['base_model'],'base model differs from the original formal A800 reader')
    for name,expected in TOKENIZER_FILES.items():require(digest(options.model/name)==expected,'tokenizer/generation configuration differs')
    manifest=json.loads(options.vllm_fork_manifest.read_bytes())
    fork=verify_fork(options.vllm_fork_manifest,Path(manifest['package_root'])/'__init__.py')
    require(manifest['protocol']==live.PROTOCOL and manifest['live_helper']['sha256']==digest(live.__file__)
        and Path(manifest['live_helper']['path']).resolve()==Path(live.__file__).resolve(),'live helper provenance differs')
    versions=runtime_versions()
    require(versions['vllm']=='0.30.0' and all(versions[k]==v for k,v in data['protocol']['dependencies'].items()),
            'runtime libraries differ from the pinned native tokenizer/live stack')
    root=options.run_root.resolve()
    require(not root.exists() and not root.is_relative_to(options.delta_repo.resolve().parent),
            'LoCoMo execution root must be fresh and outside immutable live_v3')
    sources=source_identity(options.delta_repo,options.formal_scripts)
    binding={'schema_version':1,'protocol':live.PROTOCOL,'adapter_protocol':PROTOCOL,'layout':reference['layout'],
        'base_model':identity,'writers':{'delta':{k:writer[k] for k in ('path','sha256','tensor_sha256')}},
        'writer_audit':writer,'source_sha256':sources,'runtime_versions':versions,'fork':fork,
        'vllm_fork_manifest':{'path':str(options.vllm_fork_manifest.resolve()),'sha256':digest(options.vllm_fork_manifest)},
        'base_verification':{'path':str(options.base_verification.resolve()),'sha256':digest(options.base_verification)},
        'reference_binding':{'path':str(options.reference_binding.resolve()),'sha256':digest(options.reference_binding)},
        'delta_repo':str(options.delta_repo.resolve()),'formal_scripts':str(options.formal_scripts.resolve()),
        'requested_engine_settings':engine_settings(options.model.resolve()),
        'locomo':{'prepared':str(options.prepared.resolve()),'parent_execution_plan':str(options.execution_plan.resolve()),
            'amendment':str(options.amendment.resolve()),'amendment_sha256':data['amendment_sha256'],
            'runtime_hashes':data['runtime_hashes'],'prepared_manifest_sha256':PREPARED_SHA,'parent_protocol_sha256':PROTOCOL_SHA,
            'execution_root':str(root),'method':data['amendment'],'conversation_order':ordered_conversations(data),
            'question_order':[q['qid'] for q in selected_questions(data,'formal')],
            'query_prompt_bindings':{q['qid']:q['short_prompt_sha256'] for q in data['questions']}},
        'cpu_only':True,'cuda_initialized':torch.cuda.is_initialized(),'model_forwards':0,'D1_commits':0,
        'optimizer_created':False,'backward_calls':0,'GPU_fit_verified':False}
    require(not binding['cuda_initialized'],'CPU binding initialized CUDA')
    return binding

def verify_binding(path):
    path=Path(path);seal=json.loads((path/'BINDING_COMPLETE.json').read_bytes())
    require(seal['protocol']==PROTOCOL and digest(path/'binding.json')==seal['binding_sha256'],'LoCoMo binding seal differs')
    binding=json.loads((path/'binding.json').read_bytes())
    configure_paths(binding['delta_repo'],binding['formal_scripts'])
    require(binding['adapter_protocol']==PROTOCOL,'wrong adapter binding')
    require(source_identity(binding['delta_repo'],binding['formal_scripts'])==binding['source_sha256'],'a bound source changed')
    for key in ('vllm_fork_manifest','reference_binding','base_verification'):
        require(digest(binding[key]['path'])==binding[key]['sha256'],key+' changed')
    require(digest(binding['writers']['delta']['path'])==WRITER_SHA,'writer archive changed')
    contract=binding['locomo'];data=load_amended_runtime(contract['prepared'],contract['parent_execution_plan'],contract['amendment'])
    require(data['amendment_sha256']==contract['amendment_sha256'] and data['runtime_hashes']==contract['runtime_hashes'],
            'LoCoMo runtime inputs changed')
    require(runtime_versions()==binding['runtime_versions'],
            'runtime library versions changed')
    return binding,data

@contextmanager
def run_lock(root):
    import fcntl
    root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a+') as handle:
        try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('another LoCoMo coordinator is active') from error
        try:yield
        finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)

def append_event(path,event):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(event,utc=utc()),ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())

def prediction_path(root,qid,condition):return root/'predictions'/(qid.replace(':','__')+'--'+condition+'.json')

def validate_prediction(row,q,condition,source,fp,binding_sha):
    require(row['qid']==q['qid'] and row['sample_id']==q['sample_id'] and row['condition']==condition
        and row['source_memory_sample_id']==source and row['prompt_sha256']==q['short_prompt_sha256']
        and row['binding_sha256']==binding_sha and row['state_fingerprint_before']==row['state_fingerprint_after']==fp,
        'saved prediction prompt/state/binding differs')
    tokens=row['generated_token_ids']
    require(isinstance(tokens,list) and 0<len(tokens)<=32 and all(type(t) is int and 0<=t<248320 for t in tokens)
        and canonical(tokens)==row['generated_token_ids_sha256'] and isinstance(row['text'],str)
        and row['memory_updates']==0 and row['per_request_state_guard_passed'] is True
        and row['native_forward_calls']==len(tokens) and row['writer_tensor_sha256']==WRITER_TENSOR_SHA,
        'saved continuation is invalid')
    eos={248044,248046}
    require(not any(t in eos for t in tokens[:-1]) and
        ((tokens[-1] in eos and row['stop_reason']=='eos') or
         (tokens[-1] not in eos and len(tokens)==32 and row['stop_reason']=='max_new_tokens'))
        and row['max_token_limit_reached']==(len(tokens)==32),'saved EOS/length contract differs')


def recover_snapshot(path,sid,binding_sha,data):
    """Recover an atomic final snapshot if the coordinator died before indexing it."""
    import torch
    from delta_ngram import qwen38_vllm_live as live
    path=Path(path);saved=torch.load(path,map_location='cpu',weights_only=True)
    frames=data['frames'][sid]
    require(saved['format']==live.PROTOCOL and saved['binding_sha256']==binding_sha
        and saved['epochs']=={'ordinary':0,'delta':len(frames)}
        and set(saved['states'])==set(live.ARMS),'unindexed snapshot provenance/clock differs')
    validate_conversation_provenance(saved.get('locomo'),sid,data)
    states={arm:live.restore_state(saved['states'][arm],'cpu') for arm in live.ARMS}
    require({arm:live.state_hash(s) for arm,s in states.items()}==saved['state_hashes']
        and states['ordinary'].rows.numel()==0
        and all(int(s.block_clock[0])==saved['epochs'][arm] for arm,s in states.items()),
        'unindexed snapshot seven-field payload differs')
    return {'sample_id':sid,'frames':len(frames),'written_tokens':data['conversations'][sid]['token_count'],
        'state_hashes':saved['state_hashes'],'epochs':saved['epochs'],
        'checkpoint':{'path':str(path.resolve()),'sha256':digest(path),
            **{k:saved[k] for k in ('state_hashes','epochs','identities')}},
        'write_seconds':None,'recovered_atomic_final_snapshot':True}


def write_combined(path,records):
    """Resume publication of an already-complete prediction matrix without inference."""
    path=Path(path)
    payload=''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in records)
    if path.exists():
        require(path.read_text(encoding='utf-8')==payload,'existing combined predictions differ')
        return
    temporary=path.with_suffix(path.suffix+'.partial')
    with temporary.open('w',encoding='utf-8') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)

def execute(options):
    binding,data=verify_binding(options.binding)
    root=Path(binding['locomo']['execution_root']);binding_sha=digest(options.binding/'binding.json')
    with run_lock(root):
        require(not (root/'COMPLETE.json').exists(),'LoCoMo is complete; completed evaluations cannot rerun')
        attempts=root/'attempts';attempts.mkdir(exist_ok=True)
        existing=sorted(attempts.glob('attempt_*.json'))
        require(not existing or options.resume,'existing execution requires explicit --resume; previous costs remain recorded')
        if options.mode=='rest':require((root/'PILOT.json').is_file(),'rest requires the completed fixed pilot')
        attempt_path=attempts/f'attempt_{len(existing)+1:03d}.json'
        attempt={'protocol':PROTOCOL,'binding_sha256':binding_sha,'mode':options.mode,'status':'starting','started_utc':utc()}
        write_json(attempt_path,attempt)
        try:
            result=execute_session(binding,data,root,binding_sha,options)
            attempt.update(status='completed',finished_utc=utc(),result=result);write_json(attempt_path,attempt)
        except BaseException as error:
            attempt.update(status='failed',finished_utc=utc(),error_type=type(error).__name__,message=str(error),automatic_retry=False)
            write_json(attempt_path,attempt);raise

def execute_session(binding,data,root,binding_sha,options):
    import torch
    import qwen38_vllm_accuracy as accuracy
    from qwen38_vllm_delta_accuracy import verify_fork
    from qwen38_native_runtime_preflight import verified_model
    from qwen38_vllm_live_two_chain import LiveEngine,agree
    from qwen38_greedy_decode import greedy_decode
    from delta_ngram import qwen38_vllm_live as live
    require(sys.platform=='linux' and not torch.cuda.is_initialized(),'start native LoCoMo in a fresh Linux process')
    current=accuracy.hf_model_identity(Path(binding['base_model']['model_path']))
    current['base_hash_audit']=verified_model(Path(binding['base_model']['model_path']),Path(binding['base_verification']['path']))
    current['verification_sha256']=current['base_hash_audit']['verification_sha256']
    require(current==binding['base_model'],'frozen base identity changed since CPU preparation')
    os.environ.update({live.ENV:str((options.binding/'binding.json').resolve()),live.ENV_SHA:binding_sha,
        'VLLM_WORKER_MULTIPROC_METHOD':'spawn','VLLM_ENABLE_V1_MULTIPROCESSING':'1','VLLM_USE_V2_MODEL_RUNNER':'1',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'})
    import vllm
    import vllm.envs as envs
    from vllm import LLM,SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from transformers import AutoTokenizer
    require(verify_fork(binding['vllm_fork_manifest']['path'],vllm.__file__)==binding['fork'],'vLLM imported another fork')
    kwargs=copy.deepcopy(binding['requested_engine_settings'])
    accuracy.configure_ple_cpu_offload(kwargs,EngineArgs,envs,requested=True)
    planned=EngineArgs(**copy.deepcopy(kwargs)).create_engine_config()
    require(planned.use_v2_model_runner and planned.model_config.max_model_len==ENGINE_CONTEXT
        and not planned.cache_config.enable_prefix_caching and not planned.scheduler_config.enable_chunked_prefill
        and planned.speculative_config is None,'resolved evidence/read request policy differs')
    execution=accuracy.execution_config(kwargs,planned,envs,ple_cpu_offload_requested=True)
    started=time.monotonic();write_json(root/'STATUS.json',{'status':'loading','mode':options.mode,'binding_sha256':binding_sha,'execution':execution})
    llm=LLM(**copy.deepcopy(kwargs))
    require(accuracy.execution_config(kwargs,llm.llm_engine.vllm_config,envs,ple_cpu_offload_requested=True)==execution,
            'live engine differs from registered settings')
    engine=LiveEngine(llm,SamplingParams,root)
    workers=engine.rpc('delta_live_initialize',binding['writers'],timeout=600)
    ready=agree(workers,['binding_sha256','protocol','worker_sha256','helper_sha256','cache_dtype','quiescent',
        'adapter_protocol','adapter_sha256','state_adapter_sha256','empty_state_hashes'],label='LoCoMo initialize')
    require(ready['binding_sha256']==binding_sha and ready['adapter_protocol']==PROTOCOL and ready['cache_dtype']=='torch.float32'
        and ready['quiescent'] and ready['adapter_sha256']==digest(Path(__file__).with_name('qwen38_locomo_delta_worker.py'))
        and ready['state_adapter_sha256']==digest(Path(__file__).with_name('qwen38_locomo_delta_state.py')),
        'worker adapter/FP32 state initialization differs')
    model_load_seconds=time.monotonic()-started
    tokenizer=AutoTokenizer.from_pretrained(binding['base_model']['model_path'],local_files_only=True)
    (root/'states').mkdir(exist_ok=True);(root/'predictions').mkdir(exist_ok=True)
    index_path=root/'state_index.json'
    index=json.loads(index_path.read_bytes()) if index_path.exists() else {'protocol':PROTOCOL,'binding_sha256':binding_sha,'states':{}}
    require(index['protocol']==PROTOCOL and index['binding_sha256']==binding_sha,'saved conversation index belongs to another run')
    event_path=root/'execution_events.jsonl'
    support_params=SamplingParams(n=1,temperature=0.,top_p=1.,top_k=-1,seed=17,max_tokens=1,
        ignore_eos=True,stop_token_ids=[],detokenize=False)
    query_params=SamplingParams(n=1,temperature=0.,top_p=1.,top_k=-1,seed=17,max_tokens=1,
        ignore_eos=True,stop_token_ids=[],detokenize=False)
    state_fields=['sample_id','selected_source','query_arm','phase','epochs','identities','written_tokens','state_hashes','state_fields','writer_sha256','frozen_reader_writer','binding_sha256']

    def status():return agree(engine.rpc('locomo_state_status'),state_fields,label='seven-field state audit')

    def ensure_state(sid):
        orphan=root/'states'/(sid+'.state.pt')
        if sid not in index['states'] and orphan.exists():
            index['states'][sid]=recover_snapshot(orphan,sid,binding_sha,data);write_json(index_path,index)
            append_event(event_path,{'phase':'recovered_atomic_final_snapshot','sample_id':sid,
                'checkpoint_sha256':index['states'][sid]['checkpoint']['sha256'],'new_model_forwards':0})
        if sid in index['states']:
            record=index['states'][sid];path=Path(record['checkpoint']['path'])
            require(path==(root/'states'/(sid+'.state.pt')).resolve() and digest(path)==record['checkpoint']['sha256']
                and record['frames']==len(data['frames'][sid]) and record['written_tokens']==data['conversations'][sid]['token_count'],
                'saved conversation state/index differs')
            return
        fresh=agree(engine.rpc('locomo_reset_conversation',sid),state_fields,label='new empty conversation')
        require(fresh['phase']=='writing' and fresh['epochs']=={'ordinary':0,'delta':0}
            and fresh['state_hashes']==ready['empty_state_hashes'],'conversation did not begin empty')
        engine.epochs=dict(fresh['epochs']);tick=time.monotonic()
        for frame in data['frames'][sid]:
            armed=agree(engine.rpc('locomo_arm_frame',sid,frame['frame_index'],frame['frame_sha256']),
                ['sample_id','frame_index','frame_sha256','new_tokens','prefix_tokens'],label='frame arm')
            ids=transport_frame_ids(frame)
            append_event(event_path,{'phase':'before_evidence_forward','sample_id':sid,'frame_index':frame['frame_index'],
                'frame_sha256':frame['frame_sha256'],'encoder_tokens':len(ids),'new_tokens':frame['new_tokens'],'current_delta_epoch':engine.epochs['delta']})
            _,observation=engine.request('delta','support',ids,support_params,request_id=f"delta:support:{sid}:{frame['frame_index']}")
            require(observation['frame_sha256']==frame['frame_sha256'] and observation['new_tokens']==frame['new_tokens']
                and observation['prefix_tokens_not_written']==frame['prefix_tokens'] and observation['current_delta_encoding'],
                'writer did not use exactly the registered new-token crop')
            append_event(event_path,{'phase':'before_D1','sample_id':sid,'frame_index':frame['frame_index'],'new_tokens':frame['new_tokens']})
            committed=agree(engine.rpc('delta_live_commit'),['arm','epoch','identity','rows','block_clock','update_rows'],label='cropped D1')
            require(committed['epoch']==engine.epochs['delta']+1 and committed['block_clock']==[committed['epoch']],
                    'D1 must advance exactly one evidence frame')
            engine.epochs['delta']=committed['epoch']
            append_event(event_path,{'phase':'D1_complete','sample_id':sid,'frame_index':frame['frame_index'],'state':committed,'evidence':observation})
        final=agree(engine.rpc('locomo_finalize_conversation'),['sample_id','frames','written_tokens','state_hashes','epochs','phase','adapter_protocol'],label='final conversation')
        require(final['frames']==len(data['frames'][sid]) and final['written_tokens']==data['conversations'][sid]['token_count'],
                'final evidence coverage differs')
        receipt=engine.checkpoint(root/'states'/(sid+'.state.pt'))
        require(receipt['state_hashes']==final['state_hashes'] and receipt['epochs']==final['epochs'],'export changed final state')
        record={key:final[key] for key in ('sample_id','frames','written_tokens','state_hashes','epochs')}
        record.update(checkpoint={k:receipt[k] for k in ('path','sha256','state_hashes','epochs','identities')},write_seconds=time.monotonic()-tick)
        index['states'][sid]=record;write_json(index_path,index)

    write_ids=ordered_conversations(data)[:2] if options.mode=='pilot' else ordered_conversations(data)
    for sid in write_ids:ensure_state(sid)
    generation_seconds={};questions=selected_questions(data,options.mode)
    for sid in dict.fromkeys(q['sample_id'] for q in questions):
        selected=[q for q in questions if q['sample_id']==sid]
        for condition in CONDITIONS:
            source=None if condition=='empty' else sid if condition=='self' else data['donors'][sid]
            receipt=None if source is None else index['states'][source]['checkpoint']
            before=agree(engine.rpc('locomo_select_snapshot',source,receipt),state_fields,label='select readonly state')
            engine.epochs=dict(before['epochs']);arm=before['query_arm'];fp=before['state_hashes'][arm]
            if source is not None:require(fp==index['states'][source]['state_hashes']['delta'],'query selected a different final state')
            tick=time.monotonic()
            for question in selected:
                target=prediction_path(root,question['qid'],condition)
                if target.exists():
                    validate_prediction(json.loads(target.read_bytes()),question,condition,source,fp,binding_sha);continue
                calls=0
                append_event(event_path,{'phase':'before_question','qid':question['qid'],'condition':condition,
                    'source_memory_sample_id':source,'state_hash':fp,'maximum_forward_calls':32})
                def next_token(prefix):
                    nonlocal calls
                    append_event(event_path,{'phase':'before_query_forward','qid':question['qid'],'condition':condition,
                        'step':calls,'prefix_tokens':len(prefix),'prefix_sha256':canonical(prefix)})
                    response,observation=engine.request(arm,'query',prefix,query_params,
                        request_id=f"locomo:{question['qid']}:{condition}:step{calls:02d}")
                    ids=list(response.outputs[0].token_ids)
                    require(len(ids)==1 and observation['epoch']==before['epochs'][arm]
                        and observation['forward_calls']==1,'readonly next-token request differs')
                    calls+=1;return ids[0]
                result=greedy_decode(question['short_prompt_ids'],forward_next=next_token,
                    decode=lambda ids:tokenizer.decode(ids,skip_special_tokens=True),eos_token_ids=[248044,248046],
                    max_new_tokens=32,max_context_tokens=ENGINE_CONTEXT)
                row={'qid':question['qid'],'sample_id':sid,'condition':condition,'prompt_sha256':question['short_prompt_sha256'],
                    'prompt_tokens':question['short_prompt_tokens'],'source_memory_sample_id':source,
                    'state_fingerprint_before':fp,'state_fingerprint_after':fp,'state_epoch':before['epochs'][arm],
                    'generated_token_ids':result['generated_token_ids'],'generated_token_ids_sha256':canonical(result['generated_token_ids']),
                    'text':result['raw_completion'],'stop_reason':result['finish_reason'],'max_token_limit_reached':result['max_token_hit'],
                    'elapsed_seconds':result['elapsed_seconds'],'native_forward_calls':calls,'memory_updates':0,
                    'per_request_state_guard_passed':True,'binding_sha256':binding_sha,'writer_tensor_sha256':WRITER_TENSOR_SHA}
                validate_prediction(row,question,condition,source,fp,binding_sha)
                write_json(target,row)
                append_event(event_path,{'phase':'question_complete','qid':question['qid'],'condition':condition,
                    'prediction_sha256':digest(target),'actual_forward_calls':calls,'generated_tokens':len(result['generated_token_ids'])})
            after=status()
            require(all(after[key]==before[key] for key in state_fields),'a query block changed a seven-field memory state or frozen identity')
            generation_seconds[sid+'/'+condition]=time.monotonic()-tick
            write_json(root/f'READONLY_{sid}_{condition}.json',{'protocol':PROTOCOL,'binding_sha256':binding_sha,
                'before':before,'after':after,'questions':len(selected),'all_seven_fields_identical':True})
    resources=engine.rpc('locomo_state_status');agree(resources,state_fields,label='final readonly/frozen check')
    verify_binding(options.binding)
    require(digest(binding['writers']['delta']['path'])==WRITER_SHA,'source writer changed during evaluation')
    wanted=selected_questions(data,'pilot' if options.mode=='pilot' else 'formal')
    records=[]
    for question in wanted:
        for condition in CONDITIONS:
            source=None if condition=='empty' else question['sample_id'] if condition=='self' else data['donors'][question['sample_id']]
            fp=ready['empty_state_hashes']['ordinary'] if source is None else index['states'][source]['state_hashes']['delta']
            row=json.loads(prediction_path(root,question['qid'],condition).read_bytes())
            validate_prediction(row,question,condition,source,fp,binding_sha);records.append(row)
    expected=717 if options.mode=='pilot' else 5958
    require(len(records)==expected and len({(r['qid'],r['condition']) for r in records})==expected,'prediction matrix is incomplete/duplicated')
    if options.mode!='pilot':require(len(index['states'])==10 and sum(r['frames'] for r in index['states'].values())==426,'final state coverage differs')
    result={'protocol':PROTOCOL,'status':'completed','mode':options.mode,'binding_sha256':binding_sha,
        'predictions':len(records),'states':index['states'],'model_load_seconds':model_load_seconds,
        'session_seconds':time.monotonic()-started,'generation_seconds':generation_seconds,'resources':resources,
        'writer_tensor_sha256':WRITER_TENSOR_SHA,'writer_archive_sha256':WRITER_SHA,'source_code_reverified':True,
        'query_memory_updates':0,'optimizer_steps':0,'backward_calls':0,'annotations_opened':False,
        'execution_events_sha256':digest(event_path),'prior_attempt_costs_preserved':True}
    if options.mode!='pilot':
        combined=root/'delta_predictions.jsonl'
        write_combined(combined,records)
        result.update(combined_predictions_path=str(combined),combined_predictions_sha256=digest(combined))
    write_json(root/('PILOT.json' if options.mode=='pilot' else 'COMPLETE.json'),result)
    write_json(root/'STATUS.json',{'status':'completed','mode':options.mode,'predictions':len(records),'binding_sha256':binding_sha})
    return {'predictions':len(records),'session_seconds':result['session_seconds']}

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    for name in ('prepared','execution-plan','amendment','reference-binding','delta-repo','formal-scripts','model',
                 'writer-checkpoint','base-verification','vllm-fork-manifest','output','run-root'):
        prep.add_argument('--'+name,type=Path,required=True)
    run=sub.add_parser('execute');run.add_argument('--binding',type=Path,required=True)
    run.add_argument('--mode',choices=('pilot','rest','formal'),required=True);run.add_argument('--resume',action='store_true')
    options=parser.parse_args()
    if options.action=='prepare':
        os.environ['CUDA_VISIBLE_DEVICES']='';configure_paths(options.delta_repo,options.formal_scripts)
        require(not options.output.exists(),'CPU binding directory must be fresh')
        binding=build_binding(options);options.output.mkdir(parents=True,exist_ok=False)
        write_json(options.output/'binding.json',binding)
        write_json(options.output/'BINDING_COMPLETE.json',{'protocol':PROTOCOL,'binding_sha256':digest(options.output/'binding.json'),
            'cpu_only':True,'cuda_initialized':False,'model_forwards':0,'D1_commits':0})
        print(json.dumps({'status':'CPU_binding_passed','binding_sha256':digest(options.output/'binding.json'),
            'writer':binding['writer_audit'],'evidence_frames':426,'new_evidence_tokens':215278,'short_queries':5958,
            'cuda_initialized':False,'GPU_fit_verified':False},indent=2))
    else:execute(options)

if __name__=='__main__':main()

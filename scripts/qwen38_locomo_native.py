"""Frozen ordinary-writer LoCoMo self/empty/donor generation on native H200.

CPU --prepare validates the exact prepared runtime inputs and writer. GPU
--execute-pilot writes conv-48 and its fixed donor once and generates all 717
pilot requests. --execute-rest reuses those states/predictions, writes the other
eight conversations once, and completes the remaining 5241 requests. No scorer,
answer annotation, optimizer, backward, training or prompt tuning is imported.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace

PROTOCOL='qwen38_locomo_native_readonly_v1'
PLAN_SHA='19c4757e20d0e98de7edd8d6c606f4db0d273bd8e90e0e90225295e8566a2246'
PREPARED_SHA='3a5b27f86738fe6cfb1135d8bd1ef862e4aff82d07cc0f4c4dbadf1cd9250bd4'
PROTOCOL_SHA='866045ef6e8afa937b3424da2c232c20e57a4a687a1cb35c50613127bdaf791c'
TOKENIZER_FILES={'tokenizer.json':'0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3',
 'tokenizer_config.json':'b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27',
 'generation_config.json':'e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e'}
PILOT='conv-48'
CONDITIONS=('self','empty','donor')

def require(value,message):
 if not value:raise ValueError(message)

def canonical(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(8<<20),b''):h.update(part)
 return h.hexdigest()

def write_json(path,value):
 path=Path(path);temporary=path.with_suffix(path.suffix+'.partial')
 with temporary.open('w',encoding='utf-8') as f:
  json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(temporary,path)

def read_rows(path):
 return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def validate_frame(frame,tokens):
 """Reconstruct the chronological 512-new/128-old native frame exactly."""
 require(canonical({k:v for k,v in frame.items() if k!='frame_sha256'})==frame['frame_sha256'],'frame checksum differs')
 start,end=frame['article_token_start'],frame['article_token_end'];prefix_start=max(0,start-128)
 require(start==512*frame['frame_index'] and end==min(start+512,len(tokens)) and start<end,'frame chronology differs')
 require(frame['prefix_token_start']==prefix_start and frame['prefix_tokens']==start-prefix_start and frame['new_tokens']==end-start,'frame spans differ')
 ids=tokens[prefix_start:end];padding=len(ids)%2
 require(frame['input_ids']==ids+[248044]*padding,'frame token prefix/padding differs')
 require(frame['encoder_valid_mask']==[True]*len(ids)+[False]*padding,'encoder valid mask differs')
 require(frame['write_mask']==[False]*(start-prefix_start)+[True]*(end-start)+[False]*padding,'write mask includes prefix or padding')
 require(frame['position_ids']==list(range(len(ids)+padding)) and frame['padding_tokens']==padding,'frame positions differ')
 require(frame['loss_labels'] is None and frame['loss_target_tokens']==0 and frame['evaluation_only'] is True and frame['owner_id']==0,'frame is not one-owner inference only')
 require(frame['article_initial_chunk']==(start==0),'frame initial flag differs')
 return start-prefix_start,end-start

def load_runtime(prepared,execution_plan):
 prepared=Path(prepared)
 require(digest(execution_plan)==PLAN_SHA,'execution plan differs')
 plan=json.loads(Path(execution_plan).read_bytes())
 require(digest(prepared/'manifest.json')==PREPARED_SHA and digest(prepared/'protocol.json')==PROTOCOL_SHA,'prepared protocol/manifest differs')
 manifest=json.loads((prepared/'manifest.json').read_bytes())
 # Only runtime input files are opened. Scoring annotations remain outside this runner.
 names=['protocol.json','donor_registry.json','runtime_data/conversations.jsonl','runtime_data/questions.jsonl','runtime_data/writer_frames.jsonl']
 for name in names:
  path=prepared/name;entry=manifest['files'][name]
  require(path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256'],'prepared runtime file differs: '+name)
 protocol=json.loads((prepared/'protocol.json').read_bytes())
 donors=json.loads((prepared/'donor_registry.json').read_bytes())
 require(canonical(donors['mapping'])==donors['mapping_sha256']==plan['runtime_inputs']['donors'].split('mapping_sha256=')[1],'donor mapping differs')
 conversations=read_rows(prepared/'runtime_data/conversations.jsonl')
 questions=read_rows(prepared/'runtime_data/questions.jsonl')
 frames=read_rows(prepared/'runtime_data/writer_frames.jsonl')
 require(len(conversations)==10 and len(questions)==1986 and len(frames)==426,'runtime population differs')
 byid={x['sample_id']:x for x in conversations};require(len(byid)==10,'duplicate conversation')
 require(set(donors['mapping'])==set(byid) and all(v!=k and donors['mapping'][v]==k for k,v in donors['mapping'].items()),'donor pairs not disjoint symmetric')
 grouped=collections.defaultdict(list)
 for frame in frames:
  sid=frame['document_id'];require(frame['article_id']==sid and sid in byid,'frame conversation differs')
  validate_frame(frame,byid[sid]['token_ids']);grouped[sid].append(frame)
 for sid,c in byid.items():
  require(c['token_count']==len(c['token_ids']) and canonical(c['token_ids'])==c['token_ids_sha256'],'conversation tokens differ')
  selected=grouped[sid]
  require([f['frame_index'] for f in selected]==list(range(len(selected))) and len(selected)==c['write_frames'],'frame ordering/count differs')
  require(sum(f['new_tokens'] for f in selected)==c['token_count'],'conversation evidence coverage differs')
 ids=set()
 for q in questions:
  require(set(q)=={'qid','sample_id','question','short_prompt_ids','short_prompt_sha256','short_prompt_tokens','full_context_prefix_id','full_context_suffix_ids','full_context_prompt_sha256','full_context_prompt_tokens','donor_sample_id'},'question contains an unregistered field')
  require(q['qid'] not in ids and q['sample_id'] in byid,'question identity differs');ids.add(q['qid'])
  require(canonical(q['short_prompt_ids'])==q['short_prompt_sha256'] and len(q['short_prompt_ids'])==q['short_prompt_tokens'],'short prompt differs')
  require(q['donor_sample_id']==donors['mapping'][q['sample_id']],'question donor differs')
  require(52<=len(q['short_prompt_ids'])<=80 and all(type(t) is int and 0<=t<248320 for t in q['short_prompt_ids']),'query token range/length differs')
 require(sum(c['token_count'] for c in conversations)==215278 and sum(q['sample_id']==PILOT for q in questions)==239,'evidence/pilot population differs')
 return {'plan':plan,'protocol':protocol,'manifest':manifest,'conversations':byid,'questions':questions,'frames':dict(grouped),'donors':donors['mapping'],'runtime_hashes':{n:manifest['files'][n]['sha256'] for n in names}}

def runtime_setup(options):
 for path in [options.delta_repo/'src',options.delta_repo/'scripts/native_qwen38',options.cl_repo/'scripts']:
  sys.path.insert(0,str(path.resolve()))
 sys.path.insert(0,str(Path(__file__).resolve().parent))

def code_hashes(options):
 paths=[Path(__file__),Path(__file__).with_name('qwen38_greedy_decode.py')]
 paths += [options.delta_repo/'scripts/native_qwen38'/name for name in ['qwen38_native_h200_session.py','qwen38_native_h200_forward.py','qwen38_h200_seven_task_forward_writer.py','qwen38_native_runtime_preflight.py','qwen38_native_memory_schedule.py','qwen38_native_dispatch.py','qwen38_native_router.py']]
 paths += [options.delta_repo/'src/delta_ngram'/name for name in ['qwen38_seven_forward.py','qwen38_forward_recovery.py','qwen38_native.py','qwen38_adapter.py','qwen38_precision.py','qwen38_writer.py','qwen38_training.py','qwen38_probe.py','alpha_memory.py','batching.py','state.py','operators.py']]
 paths += [options.cl_repo/'scripts/qwen38_h200_writer_timing.py']
 return {str(p.resolve()):digest(p) for p in paths}

def writer_options(options):
 return SimpleNamespace(method='ordinary',writer_checkpoint=options.writer_checkpoint,writer_sha256='523b951487cee0ec726077db0104a8fc929f4629bdb4ca46604295a4b78fbfe3')

def prepare(options):
 import torch
 torch.set_num_threads(2);require(not torch.cuda.is_initialized(),'CPU preparation initialized CUDA')
 data=load_runtime(options.prepared,options.execution_plan)
 from qwen38_h200_seven_task_forward_writer import writer_source
 source=writer_source(writer_options(options))
 formal=json.loads(options.formal_base_summary.read_bytes())
 for name,expected in TOKENIZER_FILES.items():require(digest(options.model_path/name)==expected,'tokenizer/generation config differs')
 for name,expected in formal['identity']['code_sha256'].items():
  require(digest(options.delta_repo/name)==expected,'native source differs from formal reader/writer: '+name)
 options.output.mkdir(parents=True,exist_ok=False)
 receipt={'protocol':PROTOCOL,'status':'prepared','plan_sha256':PLAN_SHA,'prepared_manifest_sha256':PREPARED_SHA,'protocol_sha256':PROTOCOL_SHA,'runtime_hashes':data['runtime_hashes'],'code_sha256':code_hashes(options),'model_identity':formal['identity']['base_model'],'native_versions':formal['identity']['native_runtime'],'formal_base_summary_sha256':digest(options.formal_base_summary),'writer_sha256':source['writer_sha256'],'writer_checkpoint_sha256':digest(options.writer_checkpoint),'writer_optimizer_step':4172,'writer_completed_loss_targets':99997072,'pilot':{'sample_id':PILOT,'donor':data['donors'][PILOT],'questions':239,'requests':717},'requests':5958,'frames':426,'history_tokens':215278,'max_context_ceiling':640,'cuda_initialized':torch.cuda.is_initialized(),'writes':0,'backward_calls':0,'optimizer_steps':0,'annotation_file_opened':False}
 write_json(options.output/'preflight.json',receipt);print(json.dumps(receipt,indent=2))

def token_batch(ids,device):
 import torch
 require(bool(ids) and len(ids)<=640,'native prefix/frame outside registered ceiling')
 n=max(2,(len(ids)+1)//2*2)
 tokens=torch.full((1,n),248044,dtype=torch.long,device=device);tokens[0,:len(ids)]=torch.tensor(ids,dtype=torch.long,device=device)
 return {'input_ids':tokens,'position_ids':torch.arange(n,device=device)[None],'attention_mask':None,'labels':None},torch.arange(n,device=device)[None]<len(ids)

def crop_new(frame,hidden,physical_rows):
 prefix,n=frame['prefix_tokens'],frame['new_tokens'];total=len(frame['input_ids'])
 require(hidden.shape[:2]==physical_rows.shape[:2]==(1,total),'native evidence layout differs')
 require(frame['write_mask']==[False]*prefix+[True]*n+[False]*frame['padding_tokens'],'write mask differs from new-token crop')
 return hidden[:,prefix:prefix+n],physical_rows[:,prefix:prefix+n]

def execute(options):
 import torch
 from qwen38_native_h200_session import run_frozen_session
 from qwen38_h200_seven_task_forward_writer import writer_source
 preflight=json.loads((options.output/'preflight.json').read_bytes());data=load_runtime(options.prepared,options.execution_plan)
 require(preflight['status']=='prepared' and preflight['cuda_initialized'] is False and preflight['code_sha256']==code_hashes(options),'missing/changed CPU preflight')
 require(preflight['runtime_hashes']==data['runtime_hashes'],'runtime inputs changed after preparation')
 require(digest(options.formal_base_summary)==preflight['formal_base_summary_sha256'],'native identity reference changed')
 for name,expected in TOKENIZER_FILES.items():require(digest(options.model_path/name)==expected,'tokenizer changed')
 formal=json.loads(options.formal_completion.read_bytes())
 require(formal['status']=='completed' and all(formal['results'][a]['status']=='completed' and formal['results'][a]['cursor']['writes_completed']==35000 for a in ['ordinary','delta']),'finish both formal seven-task chains before this GPU session')
 mode='pilot' if options.execute_pilot else 'rest'
 if mode=='rest':
  require((options.output/'PILOT.json').is_file(),'complete fixed pilot before remaining requests')
  pilot=json.loads((options.output/'PILOT.json').read_bytes())
  require(pilot['protocol']==PROTOCOL and pilot['status']=='completed' and pilot['mode']=='pilot' and pilot['plan_sha256']==PLAN_SHA and pilot['code_sha256']==preflight['code_sha256'] and pilot['model_identity']==preflight['model_identity'],'pilot protocol/plan/code/model differs')
 marker=options.output/('EXECUTING_'+mode+'.json')
 if int(os.environ.get('RANK','0'))==0:
  with marker.open('x',encoding='utf-8') as f:json.dump({'mode':mode,'started_unix':time.time(),'pid':os.getpid()},f)
 source=writer_source(writer_options(options));started=time.monotonic()

 def callback(session):
  import torch.distributed as dist
  from megatron.core import parallel_state as ps
  from delta_ngram.alpha_memory import LazyDecayState
  from delta_ngram.qwen38_native import make_qwen38_writer
  from delta_ngram.qwen38_training import writer_evidence_features
  from delta_ngram.qwen38_seven_forward import state_payload,restore_state,state_fingerprint,immutable_guard,assert_unchanged
  from delta_ngram.qwen38_probe import tensor_state_hash
  from qwen38_native_h200_forward import native_forward,native_dispatch,sync_written_rows
  from qwen38_h200_seven_task_forward_writer import write_with_statistics
  from qwen38_greedy_decode import greedy_decode
  rank=dist.get_rank();first=ps.is_pipeline_first_stage();device=session.device;control=dist.new_group(backend='gloo')
  require(session.identity==preflight['model_identity'] and session.versions==preflight['native_versions'],'native reader identity/runtime differs')
  require((rank==0)==(first and ps.get_tensor_model_parallel_rank()==0),'canonical writer rank differs')
  load_seconds=time.monotonic()-started;writer=alpha=None
  if rank==0:
   writer,alpha=make_qwen38_writer(alpha=1.);writer.load_state_dict(source['writer'],strict=True)
   writer=writer.to(device).float().eval().requires_grad_(False);alpha=alpha.to(device).eval().requires_grad_(False)
  versions=[(x,x._version) for module in [writer,alpha] if module is not None for x in module.parameters()]
  state_dir=options.output/'states';pred_dir=options.output/'predictions'
  if rank==0:state_dir.mkdir(exist_ok=True);pred_dir.mkdir(exist_ok=True)
  dist.barrier(group=control)
  index_path=options.output/'state_index.json'
  index=json.loads(index_path.read_bytes()) if index_path.exists() else {'protocol':PROTOCOL,'states':{}}
  states={};fingerprints={};events=0;write_tokens=0;write_seconds={};generation_seconds={};prediction_files=[]
  empty=LazyDecayState.from_batch(session.layout.empty_state(1,device)) if first else None

  def signature(state):
   local=state_fingerprint(state) if first else None;gathered=[None]*4
   dist.all_gather_object(gathered,local,group=control)
   require(gathered[0] is not None and gathered[0]==gathered[1] and gathered[2:]==[None,None],'TP state copies differ')
   return gathered[0]

  empty_fp=signature(empty)
  def obtain_state(sid):
   nonlocal events,write_tokens
   if sid in states:return
   if sid in index['states']:
    record=index['states'][sid];path=Path(record['path']);require(digest(path)==record['sha256'],'saved conversation state file changed')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    require(saved['protocol']==PROTOCOL and saved['sample_id']==sid and saved['prepared_manifest_sha256']==PREPARED_SHA,'saved state provenance differs')
    require(saved['writer_tensor_sha256']==source['writer_sha256'] and saved['frames']==record['frames']==len(data['frames'][sid]) and saved['evidence_tokens']==record['evidence_tokens']==data['conversations'][sid]['token_count'],'saved state writer/evidence provenance differs')
    state=restore_state(saved['state'],device=device) if first else None
    if first:require(int(state.block_clock[0])==len(data['frames'][sid]) and state.row_capacity==session.layout.row_capacity and state.width==160,'saved state epoch/layout differs')
    require(signature(state)==record['state_fingerprint']==saved['state_fingerprint'],'saved state tensors changed')
    states[sid]=state;fingerprints[sid]=record['state_fingerprint'];return
   state=LazyDecayState.from_batch(session.layout.empty_state(1,device)) if first else None
   require(signature(state)==empty_fp,'new conversation did not start empty')
   tick=time.monotonic();frames=data['frames'][sid]
   for frame in frames:
    batch,valid=token_batch(frame['input_ids'],device);valid=torch.tensor([frame['encoder_valid_mask']],dtype=torch.bool,device=device)
    result=native_forward(session,batch,valid,state=None,evidence=True,encoding_kind='base')
    touched=None
    if rank==0:
     prefix,n=frame['prefix_tokens'],frame['new_tokens']
     h_new,physical=crop_new(frame,result['hidden'],result['physical_rows'])
     hidden=writer_evidence_features(h_new,mode='rmsnorm',eps=1e-6)
     state,observation=write_with_statistics(writer,state,hidden,physical,torch.ones(hidden.shape[:2],dtype=torch.bool,device=device),alpha)
     require(bool(state.values.isfinite().all()) and int(state.block_clock[0])==frame['frame_index']+1,'native state clock/value differs after write')
     touched=torch.unique(physical[physical.ne(0)],sorted=True)
     with (options.output/'write_events.jsonl').open('a',encoding='utf-8') as f:
      f.write(json.dumps({'sample_id':sid,'frame_index':frame['frame_index'],'frame_sha256':frame['frame_sha256'],'new_tokens':n,'prefix_tokens':prefix,'state_epoch':int(state.block_clock[0]),'native_T0_encoding':True,**observation},allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    if first:state=sync_written_rows(state,touched,device=device)
    events+=1;write_tokens+=frame['new_tokens'];del result
    dist.barrier(group=control)
   fp=signature(state);write_seconds[sid]=time.monotonic()-tick
   if rank==0:
    path=state_dir/(sid+'.state.pt');require(not path.exists(),'refuse to overwrite a conversation state')
    saved={'protocol':PROTOCOL,'sample_id':sid,'prepared_manifest_sha256':PREPARED_SHA,'writer_tensor_sha256':source['writer_sha256'],'state_fingerprint':fp,'frames':len(frames),'evidence_tokens':data['conversations'][sid]['token_count'],'state':state_payload(state)}
    with path.open('xb') as f:torch.save(saved,f);f.flush();os.fsync(f.fileno())
    index['states'][sid]={'path':str(path),'sha256':digest(path),'state_fingerprint':fp,'frames':len(frames),'evidence_tokens':saved['evidence_tokens'],'bytes':path.stat().st_size,'write_seconds':write_seconds[sid]};write_json(index_path,index)
   shared=[index];dist.broadcast_object_list(shared,src=0,group=control);index.clear();index.update(shared[0] if rank!=0 else json.loads(index_path.read_bytes()))
   states[sid]=state;fingerprints[sid]=fp

  def generate(sid,condition):
   key=None if condition=='empty' else (sid if condition=='self' else data['donors'][sid])
   state=empty if key is None else states[key];fp=empty_fp if key is None else fingerprints[key]
   require(signature(state)==fp,'query block begins from changed state')
   guard=immutable_guard(state) if first else None
   rows=[q for q in data['questions'] if q['sample_id']==sid];path=pred_dir/(sid+'.'+condition+'.jsonl')
   if rank==0:
    with path.open('x',encoding='utf-8'):pass
   dist.barrier(group=control);tick=time.monotonic();count=0
   def next_token(prefix):
    batch,valid=token_batch(prefix,device)
    logits=native_forward(session,batch,valid,state=state,evidence=False,return_logits=True)['next_logits']
    require(logits.shape==(1,248320),'full native vocabulary not returned')
    token=int(logits[0].argmax());agreement=torch.tensor([token,-token],dtype=torch.long,device=device)
    dist.all_reduce(agreement,op=dist.ReduceOp.MAX)
    require(int(agreement[0])==-int(agreement[1]),'ranks selected different greedy token')
    return token
   for q in rows:
    if first:assert_unchanged(guard)
    result=greedy_decode(q['short_prompt_ids'],forward_next=next_token,decode=lambda ids:session.tokenizer.decode(ids,skip_special_tokens=True),eos_token_ids=[248044,248046],max_new_tokens=32,stop_strings=(),max_context_tokens=640)
    if first:assert_unchanged(guard)
    row={'qid':q['qid'],'sample_id':sid,'condition':condition,'prompt_sha256':q['short_prompt_sha256'],'source_memory_sample_id':key,'state_fingerprint_before':fp,'state_fingerprint_after':fp,'state_fingerprint_basis':'full hash at block boundaries plus seven-field version guard before/after every query','generated_token_ids':result['generated_token_ids'],'text':result['raw_completion'],'stop_reason':result['finish_reason'],'max_token_limit_reached':result['finish_reason']=='max_new_tokens','elapsed_seconds':result['elapsed_seconds'],'memory_updates':0,'teacher_forced':False}
    if rank==0:
     with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
     count+=1
     if count%16==0:write_json(options.output/'PROGRESS.json',{'mode':mode,'sample_id':sid,'condition':condition,'completed_questions':count,'condition_questions':len(rows),'writes_during_generation':0})
   require(signature(state)==fp,'question answering changed a state')
   generation_seconds[sid+'.'+condition]=time.monotonic()-tick
   if rank==0:
    receipt={'sample_id':sid,'condition':condition,'path':str(path),'sha256':digest(path),'rows':len(rows),'state_fingerprint_before':fp,'state_fingerprint_after':fp,'new_write_events':0,'seconds':generation_seconds[sid+'.'+condition]}
    write_json(path.with_suffix('.receipt.json'),receipt);prediction_files.append(receipt)
   dist.barrier(group=control)

  with native_dispatch(session),torch.no_grad():
   ids=[PILOT] if mode=='pilot' else [s for s in data['conversations'] if s!=PILOT]
   if mode=='pilot':
    obtain_state(PILOT);obtain_state(data['donors'][PILOT])
   else:
    for sid in data['conversations']:obtain_state(sid)
   for sid in ids:
    for condition in CONDITIONS:generate(sid,condition)
   session.assert_frozen();require(all(p._version==v and not p.requires_grad and p.grad is None for p,v in versions),'writer parameters changed/acquired gradients')
   if rank==0:require(tensor_state_hash(writer.state_dict())==source['writer_sha256'],'writer final tensor hash changed')
   memory={'rank':rank,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(device)}
   import resource
   memory['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
   resources=[None]*4;dist.all_gather_object(resources,memory,group=control)
   if rank==0:
    if mode=='rest':
     pilot=json.loads((options.output/'PILOT.json').read_bytes())
     require(pilot['protocol']==PROTOCOL and pilot['plan_sha256']==PLAN_SHA and pilot['code_sha256']==preflight['code_sha256'] and pilot['model_identity']==preflight['model_identity'],'pilot identity changed during remaining session')
     for item in pilot['prediction_files']:require(digest(item['path'])==item['sha256'],'pilot predictions changed')
     prediction_files[:0]=pilot['prediction_files']
    records=[r for item in prediction_files for r in read_rows(item['path'])]
    require(len(records)==(717 if mode=='pilot' else 5958) and len({(r['qid'],r['condition']) for r in records})==len(records),'prediction matrix incomplete/duplicated')
    if mode=='rest':
     expected={(q['qid'],c) for q in data['questions'] for c in CONDITIONS};require({(r['qid'],r['condition']) for r in records}==expected,'prediction matrix differs from full request set')
     require(len(index['states'])==10 and sum(r['frames'] for r in index['states'].values())==426,'conversation states incomplete')
     combined=options.output/'native_predictions.jsonl'
     with combined.open('x',encoding='utf-8') as f:
      for row in records:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    receipt={'protocol':PROTOCOL,'status':'completed','mode':mode,'plan_sha256':PLAN_SHA,'code_sha256':preflight['code_sha256'],'model_identity':session.identity,'writer_tensor_sha256':source['writer_sha256'],'model_load_seconds':load_seconds,'session_seconds':time.monotonic()-started,'new_write_events_this_session':events,'new_written_tokens_this_session':write_tokens,'write_seconds':write_seconds,'generation_seconds':generation_seconds,'prediction_files':prediction_files,'predictions':len(records),'states':index['states'],'state_fingerprints_verified':True,'memory_updates_during_generation':0,'backward_calls':0,'optimizer_steps':0,'resources':resources,'annotation_file_opened':False}
    if mode=='rest':receipt.update(combined_predictions_path=str(combined),combined_predictions_sha256=digest(combined))
    write_json(options.output/('PILOT.json' if mode=='pilot' else 'COMPLETE.json'),receipt)
   dist.barrier(group=control)
 return run_frozen_session(options,callback)

def main():
 p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
 m.add_argument('--prepare',action='store_true');m.add_argument('--execute-pilot',action='store_true');m.add_argument('--execute-rest',action='store_true')
 for name in ['prepared','execution-plan','cl-repo','delta-repo','model-path','verification','writer-checkpoint','formal-base-summary','formal-completion','output']:p.add_argument('--'+name,type=Path,required=True)
 options=p.parse_args();options.tokens=640;runtime_setup(options)
 prepare(options) if options.prepare else execute(options)

if __name__=='__main__':main()

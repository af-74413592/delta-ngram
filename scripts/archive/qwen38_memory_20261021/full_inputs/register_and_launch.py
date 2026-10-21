from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,time,os,traceback
r=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
def write(name,value):
 p=r/name;t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(value,indent=2));os.replace(t,p)
try:
 write('registration_status.json',{'status':'waiting_for_CPU_validation','time':now()})
 for _ in range(120):
  if (r/'full_validation.exitcode').exists():break
  time.sleep(5)
 else:raise RuntimeError('CPU validation timeout; no launch')
 assert (r/'full_validation.exitcode').read_text().strip()=='0'
 v=json.loads((r/'full_validation.log').read_text());assert v['status']=='memory_fork_CPU_valid'
 m=v['manifest'];assert m['policy']['paired_loss']['action']=='monitor' and m['policy']['paired_loss']['automatic_stop'] is False
 stopped=json.loads((r.parent/'native_memory_residual_01/USER_STOPPED.json').read_text());assert stopped['status']=='stopped_early_by_user_decision'
 decision={'protocol':'qwen38_memory_full_only_registration_v1','status':'registered_for_execution','registered_utc':now(),
 'user_authorization':'Train full only; residual terminated by explicit user authorization; paired loss warns only and stopping requires user decision',
 'order':['full'],'GPU_count':4,'complete_groups':835,'additional_targets':20015113,'optimizer':'inherited_AdamW','learning_rate':0.0003,
 'endpoint_DEV64_runs':2,'node_checkpoint':m['node_checkpoint_boundary'],'paired_loss':m['policy']['paired_loss'],
 'diagnostic_G_action':'monitor_only_never_automatic_stop','diagnostic_twins':'actual_training_reference_and_independent_repeat; no added CE; monitor-only',
 'first_group_effect':'compare pre-update Hmem writer group loss to authenticated mainline step3338',
 'residual_status':'stopped_early_by_user_decision_not_an_experimental_result','residual_completed_groups':stopped['completed_groups'],
 'residual_completed_targets':stopped['completed_targets'],'residual_receipt':str(r.parent/'native_memory_residual_01/USER_STOPPED.json'),
 'supplemental_output':'/workspace/delta-ngram-memory-fork-20261021/reports/native_memory_supplemental_probe_01',
 'supplemental_report_sha256':'031f55fb23e6a616ca8a2e6cd0b3b6a5b076a0e064c9bb8c45ab361d832561ab',
 'training_code_manifest_sha256':m['code_manifest_sha256'],'stop_after_group_file':m['stop_after_group_file'],
 'automatic_retry':False,'FINAL_used':False}
 write('training_decision.json',decision)
 with (r/'supervisor_bootstrap.claim').open('x') as f:f.write(now())
 with (r/'full_supervisor.log').open('w') as f:p=subprocess.Popen(['/workspace/venvs/qwen38-swift/bin/python',str(r/'full_supervisor.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 write('registration_status.json',{'status':'supervisor_started','time':now(),'pid':p.pid})
except BaseException as e:
 write('registration_status.json',{'status':'failed','time':now(),'error':str(e),'traceback':traceback.format_exc()});raise

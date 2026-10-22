"""One explicit NLL process; CPU verification and exclusive GPU handoff guards."""
from __future__ import annotations
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
from qwen38_locomo_nll_contract import PROTOCOL,digest,require,write_json,GPU_UUIDS

def gpu_status():
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in memory.splitlines():
        parts=[p.strip() for p in line.split(',')]
        require(len(parts)==5,'unexpected GPU status row')
        rows.append({'index':int(parts[0]),'uuid':parts[1],'name':parts[2],'memory_total_mib':int(parts[3]),'memory_used_mib':int(parts[4])})
    idle=not apps.strip() and len(rows)==8 and sorted(r['index'] for r in rows)==list(range(8))
    idle=idle and all(r['uuid']==GPU_UUIDS[r['index']] and r['name']=='NVIDIA A800-SXM4-40GB'
        and r['memory_total_mib']==40960 and r['memory_used_mib']<1000 for r in rows)
    return apps,rows,idle

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('gate','formal'),required=True)
    for name in ('binding','output','launch-dir'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--root-authorized',action='store_true',required=True)
    parser.add_argument('--gate',type=Path);parser.add_argument('--gate-sha256')
    parser.add_argument('--gate-execution',type=Path);parser.add_argument('--gate-execution-sha256')
    options=parser.parse_args()
    require(sys.platform=='linux' and options.root_authorized,'NLL launch requires Linux and root authorization')
    require(not options.launch_dir.exists() and not options.output.exists(),'NLL launch/output must be fresh; no automatic retry')
    options.launch_dir.mkdir(parents=True,exist_ok=False)
    started=time.time();binding_sha=digest(options.binding/'binding.json');entry=Path(__file__).with_name('qwen38_locomo_nll.py')
    argv=[sys.executable,'-B',str(entry),options.action,'--binding',str(options.binding.resolve()),
          '--output',str(options.output.resolve()),'--root-authorized']
    if options.action=='formal':
        require(all((options.gate,options.gate_sha256,options.gate_execution,options.gate_execution_sha256)),
                'formal launch requires the exact passed gate and process exit seals')
        argv.extend(['--gate',str(options.gate.resolve()),'--gate-sha256',options.gate_sha256,
            '--gate-execution',str(options.gate_execution.resolve()),'--gate-execution-sha256',options.gate_execution_sha256])
    write_json(options.launch_dir/'launch.json',{'protocol':PROTOCOL,'action':options.action,'argv':argv,
        'nll_binding_sha256':binding_sha,'source_supervisor_sha256':digest(__file__),'started_epoch':started,
        'environment':{k:os.environ[k] for k in ('PYTHONPATH','CUDA_VISIBLE_DEVICES','VLLM_USE_V2_MODEL_RUNNER','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE') if k in os.environ},
        'automatic_retry':False})
    code=-1;error=None;child_pid=None
    try:
        cpu_env=dict(os.environ);cpu_env['CUDA_VISIBLE_DEVICES']=''
        verification=subprocess.run([sys.executable,'-B',str(entry),'verify','--binding',str(options.binding.resolve())],
            env=cpu_env,capture_output=True,text=True,check=False)
        write_json(options.launch_dir/'cpu_verification.json',{'returncode':verification.returncode,'stdout':verification.stdout,'stderr':verification.stderr})
        require(verification.returncode==0,'NLL CPU prelaunch verification failed')
        apps,memory,idle=gpu_status();require(idle,'all eight A800 GPUs must be idle and released before NLL launch')
        write_json(options.launch_dir/'gpu_before.json',{'compute_apps':apps,'memory':memory,'idle':idle})
        with (options.launch_dir/'process.log').open('ab',buffering=0) as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            child_pid=child.pid;write_json(options.launch_dir/'child.json',{'pid':child.pid,'started_epoch':time.time(),'nll_binding_sha256':binding_sha})
            code=child.wait()
    except BaseException as failure:
        error={'type':type(failure).__name__,'message':str(failure)}
    apps='unavailable';memory=[];idle=False
    try:
        for attempt in range(10):
            apps,memory,idle=gpu_status()
            if idle:break
            if attempt<9:time.sleep(2)
    except BaseException as failure:
        error=error or {'type':type(failure).__name__,'message':str(failure)}
    gate_path=options.output/'GATE.json' if options.action=='gate' else options.gate
    receipt={'protocol':PROTOCOL,'action':options.action,'returncode':code,'nll_binding_sha256':binding_sha,
        'started_epoch':started,'finished_epoch':time.time(),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'child_pid':child_pid,'output_path':str(options.output.resolve()),'error':error,
        'gate_path':str(gate_path.resolve()) if gate_path is not None else None,
        'gate_sha256':digest(gate_path) if gate_path is not None and gate_path.is_file() else None,
        'gpu_compute_apps_after_exit':apps,'gpu_memory_after_exit':memory,'gpu_idle_after_exit':idle,
        'automatic_retry':False}
    write_json(options.launch_dir/'execution.json',receipt)
    print(json.dumps({'returncode':code,'gpu_idle_after_exit':idle,'execution_path':str((options.launch_dir/'execution.json').resolve()),'error':error}))
    raise SystemExit(0 if code==0 and idle and error is None else 1)

if __name__=='__main__':main()

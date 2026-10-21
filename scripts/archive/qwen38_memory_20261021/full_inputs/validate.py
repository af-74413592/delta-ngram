import pathlib,json,subprocess,os,sys
r=pathlib.Path(__file__).parent;branch=sys.argv[1];s=json.loads((r/(branch+'_spec.json')).read_text());env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=s['repo']+'/src:'+s['repo']+'/scripts/native_qwen38')
with (r/(branch+'_validation.log')).open('w') as f:p=subprocess.run([s['python'],*s['args'],'--validate'],env=env,cwd=s['repo'],stdout=f,stderr=subprocess.STDOUT)
(r/(branch+'_validation.exitcode')).write_text(str(p.returncode))

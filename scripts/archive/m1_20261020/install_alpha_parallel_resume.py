"""Replace only the idle resource gate with the authorized parallel resume."""
import json
from pathlib import Path
import subprocess

ops = Path('/workspace/delta-ngram-transfer-20261019/multidomain')
name = 'delta_ngram_alpha_parallel_resume'
wrapper = Path('/opt/supervisor-scripts') / (name + '.sh')
conf = Path('/etc/supervisor/conf.d') / (name + '.conf')
assert not wrapper.exists() and not conf.exists()
status = ops / 'alpha_parallel_status.json'
previous = json.loads(status.read_text())
assert previous['stage'] == 'parallel_probe_did_not_pass_waiting_for_m1_endpoint'
assert previous.get('pid') is None
subprocess.run(['supervisorctl', 'stop', 'delta_ngram_alpha_parallel'], check=True)
with (ops / 'alpha_parallel_status_before_user_resume.json').open('x') as stream:
    json.dump(json.loads(status.read_text()), stream, indent=2)
wrapper.write_text('''#!/bin/bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
export HF_HOME=/workspace/.hf_home
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/workspace/delta-ngram-alpha/src
cd /workspace/delta-ngram-alpha
exec /venv/main/bin/python -u /workspace/delta-ngram-transfer-20261019/multidomain/run_alpha_parallel_resume.py
''')
wrapper.chmod(0o755)
subprocess.run(['bash', '-n', str(wrapper)], check=True)
conf.write_text(f'''[program:{name}]
environment=PROC_NAME="{name}"
command={wrapper}
directory=/workspace/delta-ngram-alpha
autostart=false
autorestart=false
exitcodes=0
startsecs=3
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=30
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_events_enabled=true
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
''')
subprocess.run(['supervisorctl', 'reread'], check=True)
subprocess.run(['supervisorctl', 'update', name], check=True)
subprocess.run(['supervisorctl', 'start', name], check=True)

from pathlib import Path
import subprocess

name = 'delta_ngram_m1'
wrapper = Path('/opt/supervisor-scripts') / (name + '.sh')
conf = Path('/etc/supervisor/conf.d') / (name + '.conf')
assert not wrapper.exists() and not conf.exists()
wrapper.write_text('''#!/bin/bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
export HF_HOME=/workspace/.hf_home
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export PYTHONPATH=/workspace/delta-ngram-m1/src
cd /workspace/delta-ngram-m1
exec /venv/main/bin/python -u /workspace/delta-ngram-transfer-20261019/multidomain/run_m1.py
''', encoding='utf-8')
wrapper.chmod(0o755)
subprocess.run(['bash', '-n', str(wrapper)], check=True)
conf.write_text(f'''[program:{name}]
environment=PROC_NAME="{name}"
command={wrapper}
directory=/workspace/delta-ngram-m1
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
''', encoding='utf-8')
subprocess.run(['supervisorctl', 'reread'], check=True)
subprocess.run(['supervisorctl', 'update', name], check=True)
subprocess.run(['supervisorctl', 'start', name], check=True)

"""Install one registered read-only DEV64 context job; no training or FINAL."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

BUNDLE = Path("/workspace/dev64_context_code_20261020.bundle")
EXPECTED_BUNDLE = "ad2bb75e414d2b8f8da53fcf7ff56de8145c17853ce43be59325b157492f124e"
COMMIT = "e1df5725fd2a4d6c9f523ad024ade3b7eba7a8cc"
ROOT = Path("/workspace/delta-ngram-context-evaluation-20261020")
OUT = Path("/workspace/delta-ngram-runs/m1_dev64_context_20261020")
NEWS = Path("/workspace/delta-ngram-data/recent_news_dev64_20261019")
NAME = "delta_ngram_dev64_context_20261020"
assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == EXPECTED_BUNDLE
assert not ROOT.exists() and not OUT.exists(), "Independent target already exists"
subprocess.run(["git", "clone", str(BUNDLE), str(ROOT)], check=True)
subprocess.run(["git", "-C", str(ROOT), "checkout", "--detach", COMMIT], check=True)
assert subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip() == COMMIT
assert not subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()
runner = ROOT / "scripts/evaluate_dev64_full_context_20261020.py"
assert hashlib.sha256(runner.read_bytes()).hexdigest() == "0c0c5902452351f702c004dc07682247d1f4ee3152732bc20c861a0e03768130"
OUT.mkdir()
args = [
    "/venv/main/bin/python", "-u", str(runner),
    "--registry", str(NEWS / "recent_news_continuation64_20261019.json"),
    "--annotations", str(NEWS / "recent_news_prefix_annotations64_20261019.json"),
    "--reference", "/workspace/delta-ngram-runs/m1_dev64_curve_20261020/m1_100m_continuation.json",
]
env = dict(os.environ, HF_HOME="/workspace/.hf_home", HF_HUB_OFFLINE="1",
           TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
subprocess.run(args + ["--output", str(OUT / "preflight.json"), "--preflight-only"],
               env=env, check=True)
preflight = json.loads((OUT / "preflight.json").read_bytes())
assert preflight["registration_sha256"] == "9fcb452ba4a6276fa9029e01e6b22abf27f20792ecebb135109852e5c2cdc9b5"
assert preflight["model_weights_loaded"] is False
assert preflight["registration"]["maximum_context_tokens"] == 17866
assert subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                               text=True).strip() == "", "GPU is in use"
import shlex
command = shlex.join(args + ["--output", str(OUT / "context.json"),
                            "--preflight", str(OUT / "preflight.json")])
wrapper = Path("/opt/supervisor-scripts") / (NAME + ".sh")
conf = Path("/etc/supervisor/conf.d") / (NAME + ".conf")
assert not wrapper.exists() and not conf.exists()
with wrapper.open("x") as f:
    f.write("""#!/bin/bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
""" + "cd " + str(ROOT) + "\nexec " + command + "\n")
wrapper.chmod(0o755)
with conf.open("x") as f:
    f.write(f"""[program:{NAME}]
environment=PROC_NAME="{NAME}"
command={wrapper}
directory={ROOT}
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
""")
receipt = {
    "source_commit": COMMIT, "code_bundle_sha256": EXPECTED_BUNDLE,
    "registration_sha256": preflight["registration_sha256"],
    "preflight_sha256": hashlib.sha256((OUT / "preflight.json").read_bytes()).hexdigest(),
    "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
    "supervisor_wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
    "supervisor_config_sha256": hashlib.sha256(conf.read_bytes()).hexdigest(),
    "command": command, "job": NAME,
    "training_or_final": False, "writer_loaded": False,
}
with (OUT / "deployment_receipt.json").open("x") as f:
    json.dump(receipt, f, indent=2)
    f.write("\n")
subprocess.run(["supervisorctl", "reread"], check=True)
subprocess.run(["supervisorctl", "update", NAME], check=True)
subprocess.run(["supervisorctl", "start", NAME], check=True)
print(json.dumps(receipt), flush=True)

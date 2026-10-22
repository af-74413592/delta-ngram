"""Explicitly amended formal-only launcher; the six v3 runtime files are immutable.

The user authorized the original 4620 one-prefill evaluations after seeing the
failed cross-shape diagnostic. This entry point records that amendment and calls
the original verify, initialize, and formal functions without monkeypatching.
It has no gate, retry, resume, threshold adjustment, or alternate scoring path.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

OLD_ROOT = Path('/mnt/disk2/wuhao/qwen38_locomo_nll_20261022_v3')
NEW_ROOT = Path('/mnt/disk2/wuhao/qwen38_locomo_nll_20261022_amended_v1')
PROTOCOL = 'qwen38_locomo_nll_explicit_method_amendment_v1'
BINDING_SHA = '9ee5da0bfbd58531ce733e49c5e98ab5e373e961af99ce39c2bbdbf8ce8541a8'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def evidence_checks(root, amendment):
    """Check the existing failure exactly; do not reinterpret it as a passed gate."""
    for name, item in amendment['preserved_v3_files'].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()) and path != root.resolve(), 'evidence path escapes root')
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'v3 evidence changed: ' + name)
    gate = read(root / 'gate_v1/GATE.json')
    exit_record = read(root / 'gate_launch_v1/execution.json')
    failure = read(root / 'gate_v1/FAILED.json')
    require(gate['status'] == 'failed' and gate['absolute_tolerance_nats'] == 0.005
            and gate['native_forward_calls'] == 18 and len(gate['comparisons']) == 14
            and gate['D1_commits'] == gate['optimizer_steps'] == gate['backward_calls'] == 0,
            'the exact original failed diagnostic is required')
    differences = [r['abs_difference_nats'] for r in gate['comparisons']]
    require(all(math.isfinite(x) and x >= 0 for x in differences)
            and max(differences) == gate['actual_max_abs_difference_nats']
            and max(differences) == amendment['sensitivity']['observed_max_abs_difference_nats']
            and sum(x > 0.005 for x in differences) == 9, 'diagnostic values differ')
    require(exit_record['returncode'] == 1 and exit_record['error'] is None
            and exit_record['automatic_retry'] is False and exit_record['gpu_idle_after_exit'] is True
            and exit_record['gpu_compute_apps_after_exit'].strip() == ''
            and exit_record['nll_binding_sha256'] == amendment['binding_sha256']
            and exit_record['gate_sha256'] == sha(root / 'gate_v1/GATE.json'), 'original exit 1/release changed')
    memory = exit_record['gpu_memory_after_exit']
    require(len(memory) == 8 and sorted(x['index'] for x in memory) == list(range(8))
            and all(x['uuid'] == amendment['gpu_uuids'][x['index']] and x['name'] == 'NVIDIA A800-SXM4-40GB'
                    and x['memory_total_mib'] == 40960 and 0 <= x['memory_used_mib'] < 1000 for x in memory),
            'original gate did not release the eight recorded GPUs')
    require(failure['error_type'] == 'ValueError'
            and failure['message'] == 'NLL numerical gate failed; no automatic fallback or formal execution'
            and failure['automatic_retry'] is False, 'original failure was not the numerical comparison')
    fields = amendment['readonly_fields']
    for condition in ('empty', 'self'):
        block = read(root / ('gate_v1/READONLY_gate_' + condition + '.json'))
        before, after, ranks = block['before'], block['after'], block['resources']
        require(block['all_seven_fields_identical'] is True and all(before[k] == after[k] for k in fields)
                and len(ranks) == 8 and sorted(r['rank'] for r in ranks) == list(range(8))
                and all(all(r[k] == before[k] for k in fields) for r in ranks), 'original readonly evidence differs')
    require(gate['resources'] == [read(root / ('gate_v1/READONLY_gate_' + c + '.json'))['resources']
                                  for c in ('empty', 'self')], 'original resource coverage differs')
    return {'structural_pass': True, 'cross_shape_parity_failed': True,
            'original_gate_status': 'failed', 'original_gate_returncode': 1,
            'numerical_gate_pass_claimed': False, 'new_GPU_gate_performed': False}


def identity(options):
    require(sys.platform == 'linux' and Path(__file__).resolve() == NEW_ROOT / 'launch_amended_nll.py',
            'launch must use the reviewed new Linux entry point')
    require(sha(__file__) == options.launcher_sha256 and sha(NEW_ROOT / 'protocol_amendment.json') == options.amendment_sha256,
            'amended launcher/protocol identity differs')
    amendment = read(NEW_ROOT / 'protocol_amendment.json')
    require(amendment['protocol'] == PROTOCOL and amendment['authorization']['formal_evaluations'] == 4620
            and amendment['authorization']['additional_GPU_gate_authorized'] is False
            and amendment['original_runtime_modified'] is False, 'method amendment differs')
    evidence = evidence_checks(OLD_ROOT, amendment)
    require(sha(OLD_ROOT / 'binding_v1/binding.json') == amendment['binding_sha256'] == BINDING_SHA, 'original binding changed')
    sys.path.insert(0, str(OLD_ROOT / 'scripts'))
    import qwen38_locomo_nll as nll
    require(Path(nll.__file__).resolve() == OLD_ROOT / 'scripts/qwen38_locomo_nll.py'
            and nll.source_hashes() == amendment['runtime_source_sha256'], 'imported runtime changed')
    runtime_options = argparse.Namespace(binding=OLD_ROOT / 'binding_v1', output=NEW_ROOT / 'run_v1')
    binding, parent, data, targets = nll.verify(runtime_options)
    index = read(binding['state_index']['path'])
    require(len(index['states']) == 10 and len(targets) == 1540
            and binding['target_evaluations'] == 4620 and binding['target_token_count'] == 10804
            and binding['targets_sha256'] == amendment['targets_sha256']
            and binding['state_index']['sha256'] == amendment['state_index_sha256'], 'original population/states changed')
    return amendment, evidence, nll, runtime_options, binding, parent, data, targets, index


def admission(options, amendment, evidence):
    return {'protocol': PROTOCOL, 'utc': utc(), **evidence,
            'admission_basis': 'explicit_user_authorized_method_amendment_after_failed_cross_shape_diagnostic',
            'amendment_sha256': options.amendment_sha256, 'launcher_sha256': options.launcher_sha256,
            'nll_binding_sha256': amendment['binding_sha256'], 'runtime_source_sha256': amendment['runtime_source_sha256'],
            'target_questions': 1540, 'target_evaluations': 4620, 'original_runtime_modified': False,
            'automatic_retry': False}


def verify_cpu(options):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    amendment, evidence, *_ = identity(options)
    import torch
    require(not torch.cuda.is_initialized(), 'CPU identity verification initialized CUDA')
    return dict(admission(options, amendment, evidence), status='CPU_verified', cuda_initialized=False, GPU_launched=False)


def run(options):
    require(options.root_authorized and not (NEW_ROOT / 'run_v1').exists(), 'formal authorization/fresh output required')
    amendment, evidence, nll, runtime_options, binding, parent, data, targets, index = identity(options)
    output = runtime_options.output
    output.mkdir(exist_ok=False)
    entry = admission(options, amendment, evidence)
    write(output / 'ADMISSION.json', entry)
    write(output / 'STARTED.json', dict(entry, action='formal_amended', computational_protocol=nll.PROTOCOL))
    try:
        engine, ready = nll.initialize(runtime_options, binding, parent)
        result = nll.formal(runtime_options, binding, parent, data, targets, engine, ready, index)
        identity(options)
        require(result['status'] == 'completed' and result['evaluations'] == result['native_forward_calls'] == 4620,
                'formal output is incomplete')
        write(output / 'AMENDED_COMPLETE.json', dict(entry, status='completed', finished_utc=utc(),
              original_complete_sha256=sha(output / 'COMPLETE.json'), result_sha256=result['result_sha256'],
              summary_sha256=result['summary_sha256'], native_forward_calls=4620))
        return {'status': 'completed', 'output': str(output), 'evaluations': 4620}
    except BaseException as error:
        write(output / 'FAILED.json', dict(entry, error_type=type(error).__name__, message=str(error), failed_utc=utc()))
        raise


def supervise(options):
    require(sys.platform == 'linux' and options.root_authorized, 'Linux and explicit formal authorization required')
    require(not (NEW_ROOT / 'launch_v1').exists() and not (NEW_ROOT / 'run_v1').exists(), 'no automatic retry/resume')
    folder = NEW_ROOT / 'launch_v1'
    folder.mkdir(exist_ok=False)
    started = time.time()
    base = [sys.executable, '-B', str(NEW_ROOT / 'launch_amended_nll.py')]
    seals = ['--amendment-sha256', options.amendment_sha256, '--launcher-sha256', options.launcher_sha256]
    argv = base + ['run'] + seals + ['--root-authorized']
    write(folder / 'launch.json', {'protocol': PROTOCOL, 'action': 'formal_amended', 'argv': argv,
          'started_epoch': started, 'launcher_sha256': options.launcher_sha256,
          'amendment_sha256': options.amendment_sha256, 'automatic_retry': False,
          'environment': {k: os.environ[k] for k in ('PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'VLLM_USE_V2_MODEL_RUNNER',
                         'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE') if k in os.environ}})
    code, error, child_pid, admission_sha = -1, None, None, None
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        checked = subprocess.run(base + ['verify'] + seals, env=env, capture_output=True, text=True)
        write(folder / 'cpu_verification.json', {'argv': base + ['verify'] + seals, 'returncode': checked.returncode,
              'stdout': checked.stdout, 'stderr': checked.stderr})
        require(checked.returncode == 0, 'CPU identity verification failed')
        verified = json.loads(checked.stdout.splitlines()[-1])
        require(verified['status'] == 'CPU_verified' and verified['cuda_initialized'] is False, 'CPU result differs')
        write(folder / 'ADMISSION.json', verified)
        admission_sha = sha(folder / 'ADMISSION.json')
        sys.path.insert(0, str(OLD_ROOT / 'scripts'))
        from qwen38_locomo_nll_supervisor import gpu_status
        apps, memory, idle = gpu_status()
        write(folder / 'gpu_before.json', {'compute_apps': apps, 'memory': memory, 'idle': idle})
        require(idle, 'all eight original A800 GPUs must be idle')
        with (folder / 'process.log').open('xb', buffering=0) as log:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
            child_pid = child.pid
            write(folder / 'child.json', {'pid': child.pid, 'started_epoch': time.time(), 'argv': argv})
            code = child.wait()
    except BaseException as failure:
        error = {'type': type(failure).__name__, 'message': str(failure)}
    apps, memory, idle = 'unavailable', [], False
    try:
        sys.path.insert(0, str(OLD_ROOT / 'scripts'))
        from qwen38_locomo_nll_supervisor import gpu_status
        for attempt in range(10):
            apps, memory, idle = gpu_status()
            if idle:
                break
            if attempt < 9:
                time.sleep(2)
    except BaseException as failure:
        error = error or {'type': type(failure).__name__, 'message': str(failure)}
    record = {'protocol': PROTOCOL, 'action': 'formal_amended', 'returncode': code, 'error': error,
              'nll_binding_sha256': BINDING_SHA,
              'started_epoch': started, 'finished_epoch': time.time(), 'finished_utc': utc(), 'child_pid': child_pid,
              'output_path': str(NEW_ROOT / 'run_v1'), 'amendment_sha256': options.amendment_sha256,
              'launcher_sha256': options.launcher_sha256, 'admission_sha256': admission_sha,
              'gpu_compute_apps_after_exit': apps, 'gpu_memory_after_exit': memory, 'gpu_idle_after_exit': idle,
              'automatic_retry': False}
    write(folder / 'execution.json', record)
    print(json.dumps(record, allow_nan=False))
    raise SystemExit(0 if code == 0 and idle and error is None else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'run', 'supervise'))
    parser.add_argument('--amendment-sha256', required=True)
    parser.add_argument('--launcher-sha256', required=True)
    parser.add_argument('--root-authorized', action='store_true')
    options = parser.parse_args()
    if options.action == 'supervise':
        supervise(options)
    else:
        print(json.dumps(verify_cpu(options) if options.action == 'verify' else run(options), allow_nan=False))


if __name__ == '__main__':
    main()

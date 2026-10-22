"""Run the official HumanEval checker only inside a pinned Docker+gVisor image."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import uuid

from humaneval_probe import REVISION, TASK_IDS, PROBE_IDS, file_hash, load_locked_data, project_tasks, scope_ids, validate_tasks, write_json


def wilson_interval(passed, total=164, z=1.959963984540054):
    p = passed / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0.0, center-half), min(1.0, center+half)]


def docker_command(docker, image, directory, container_name=None):
    if re.fullmatch(r"(?:[^\s]+@)?sha256:[0-9a-f]{64}", image) is None:
        raise ValueError("execution image must be pinned by a full sha256 image ID/digest")
    return [docker, "run", "--rm", *(["--name", container_name] if container_name else []), "--pull=never", "--runtime=runsc", "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=128", "--memory=2g",
            "--memory-swap=2g", "--cpus=2", "--user=65534:65534", "--tmpfs=/tmp:rw,noexec,nosuid,size=256m",
            "--mount", f"type=bind,src={directory.resolve()},dst=/input,readonly", image]


def transition_counts(previous, current, expected_ids=TASK_IDS):
    before = {row['task_id']: row['passed'] for row in project_tasks(previous, expected_ids)}
    after = {row['task_id']: row['passed'] for row in project_tasks(current, expected_ids)}
    lost = [task for task in expected_ids if before[task] and not after[task]]
    gained = [task for task in expected_ids if not before[task] and after[task]]
    return {'lost_task_ids': lost, 'gained_task_ids': gained, 'lost': len(lost), 'gained': len(gained),
            'n': len(expected_ids), 'task_ids': list(expected_ids),
            'pass_at_1_change': (len(gained)-len(lost))/len(expected_ids), 'reference_zero_floor': not any(before.values())}


def result_metrics(results, expected_ids):
    results = project_tasks(results, expected_ids)
    if any(type(row.get('passed')) is not bool for row in results):
        raise ValueError('all declared tasks require completed boolean correctness results')
    passed = sum(row['passed'] for row in results)
    return {'n': len(expected_ids), 'passed': passed, 'pass_at_1': passed/len(expected_ids),
            'wilson_95_interval': wilson_interval(passed, len(expected_ids))}


def run(args):
    problems, manifest = load_locked_data(args.data_dir, "problems.json")
    source = json.loads((args.samples_dir / "summary.json").read_text(encoding="utf-8"))
    if source['status'] != 'completed' or file_hash(args.samples_dir / 'humaneval_completions.json') != source['completions_sha256']:
        raise ValueError("generation is incomplete or samples changed")
    if source['data_manifest'] != manifest:
        raise ValueError("generation and execution datasets differ")
    source_ids = scope_ids(source['scope'])
    if source['task_ids'] != list(source_ids) or source['protocol']['version'] != 2:
        raise ValueError('generation scope or decoding protocol is not the locked version')
    scope = getattr(args, 'scope', 'probe')
    selected_ids = scope_ids(scope)
    samples = validate_tasks(json.loads((args.samples_dir / "humaneval_completions.json").read_text(encoding="utf-8")), source_ids)
    samples = project_tasks(samples, selected_ids)
    problems = project_tasks(problems, selected_ids)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("use a fresh execution output directory")
    summary = {"execution_status": "pending_execution", "pass_at_1": None,
               "source_generation": str(args.samples_dir), "source_summary_sha256": file_hash(args.samples_dir / 'summary.json'),
               "protocol_sha256": source['protocol_sha256'], "harness_revision": REVISION,
               "model_identity": source['model_identity'], "source_checkpoint": source['source_checkpoint'],
               "scope": scope, "source_scope": source['scope'], "task_ids": list(selected_ids), "task_count": len(selected_ids),
               "elapsed_seconds": None, "seconds_per_task": None,
               "generation_timing": source.get('timing'), "generation_total_elapsed_seconds": source.get('elapsed_seconds'),
               "isolation": "Docker + gVisor runsc, no network, nonroot, readonly"}
    docker = shutil.which("docker")
    if not docker or not args.image:
        summary['reason'] = 'A pinned evaluator image and Docker with gVisor runsc are required; nothing executed.'
        write_json(args.output / 'summary.json', summary)
        return summary
    try:
        info = subprocess.run([docker, "info", "--format", "{{json .Runtimes}}"], capture_output=True, text=True, timeout=20, check=True)
        if 'runsc' not in json.loads(info.stdout):
            raise RuntimeError('gVisor runsc runtime is not configured')
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as exc:
        summary['reason'] = f'{type(exc).__name__}: isolation unavailable; nothing executed.'
        write_json(args.output / 'summary.json', summary)
        return summary
    with tempfile.TemporaryDirectory(prefix='humaneval-input-') as folder:
        directory = Path(folder)
        write_json(directory / 'problems.json', problems)
        write_json(directory / 'samples.json', [{k: row[k] for k in ('task_id', 'completion')} for row in samples])
        os.chmod(directory, 0o755)
        for path in directory.iterdir():
            os.chmod(path, 0o444)
        container_name = 'humaneval-eval-' + uuid.uuid4().hex
        command = docker_command(docker, args.image, directory, container_name)
        execution_started = time.perf_counter()
        timeout = 60 + 8 * math.ceil(len(selected_ids)/2)
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
            payload = json.loads(completed.stdout)
            results = validate_tasks(payload['results'], selected_ids)
            if payload['harness_revision'] != REVISION or any(type(row.get('passed')) is not bool for row in results):
                raise ValueError('invalid isolated evaluation output')
            write_json(args.output / 'per_question.json', results)
            summary.update(execution_status='completed', **result_metrics(results, selected_ids), image=args.image,
                           probe32=result_metrics(results, PROBE_IDS), workers=payload['workers'],
                           isolated_elapsed_seconds=payload['elapsed_seconds'],
                           python_version=payload['python_version'], per_question_sha256=file_hash(args.output / 'per_question.json'))
            if getattr(args, 'reference', None):
                reference = json.loads((args.reference / 'summary.json').read_text(encoding='utf-8'))
                if (reference['execution_status'] != 'completed' or reference['protocol_sha256'] != summary['protocol_sha256']
                        or reference['model_identity'] != summary['model_identity']
                        or file_hash(args.reference / 'per_question.json') != reference['per_question_sha256']):
                    raise ValueError('reference result protocol, model, or result checksum mismatch')
                previous = validate_tasks(json.loads((args.reference / 'per_question.json').read_text()), scope_ids(reference['scope']))
                comparison_ids = selected_ids if reference['scope'] == scope else PROBE_IDS
                summary['transitions'] = transition_counts(previous, results, comparison_ids)
                summary['probe32_transitions'] = transition_counts(previous, results, PROBE_IDS)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            summary.update(execution_status='failed', pass_at_1=None,
                           error=f'{type(exc).__name__}: isolated evaluator did not return all declared results')
            write_json(args.output / 'summary.json', summary)
            raise
        finally:
            summary['elapsed_seconds'] = time.perf_counter() - execution_started
            summary['seconds_per_task'] = summary['elapsed_seconds']/len(selected_ids) if summary['execution_status'] == 'completed' else None
            summary['execution_timeout_seconds'] = timeout
            write_json(args.output / 'summary.json', summary)
            # Kill only this evaluator's named container if a client timeout left it running.
            try:
                subprocess.run([docker, 'rm', '--force', container_name], capture_output=True, timeout=15)
            except (OSError, subprocess.SubprocessError):
                pass
    write_json(args.output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--samples-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', help='Immutable sha256 image ID/digest, prepared on an isolated evaluation host')
    parser.add_argument('--scope', choices=('probe', 'full'), default='probe', help='May project full generation to the fixed probe32')
    parser.add_argument('--reference', type=Path, help='Previous/base completed execution directory for paired lost/gained counts')
    print(json.dumps(run(parser.parse_args()), indent=2))

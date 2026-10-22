"""CPU-only, fail-closed four-condition assembly; score only with --score.

No model is constructed. No annotation is parsed by this module. The pinned
original scorer is invoked only after both full runs exited zero and all
population, identity, state, prompt, trace, and pilot-reuse checks pass.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

PREPARED_SHA = '3a5b27f86738fe6cfb1135d8bd1ef862e4aff82d07cc0f4c4dbadf1cd9250bd4'
SCORER_SHA = '5f4cb98ab27d5bec6d028f929a85f49aaba7e6313f876474e8158c4d175e1d26'
FULL_CONTRACT_SHA = '16298a251b388559678f108deed05eee1cec06698618e9ac742e4fcd4dd47768'
FULL_PILOT_MANIFEST_SHA = '66ab4181e69e66d27f16d4c6042e86b3b93baa285904f606e7ecfc804aa0da12'
DELTA_BINDING_SHA = '37826dd71c6fd3798fa5aa4ddf68031571461d6fcb376023ac2e0e3155b575d7'
DELTA_PILOT_MANIFEST_SHA = '77f4bbe8589fccfebd68269f8bfeff997ababacaa32b65d4cbce5658481b44e0'
AMENDMENT_SHA = 'fab0e8f5c24a45bb7d29da5177fef0da180cbc3d65f944b0fdd3e379f63e452f'
WRITER_SHA = '51d5977845e9930f48732c73ae2503511d8976b1201f139a2be46bfab9f57308'
WRITER_TENSOR_SHA = '95e9766c2d90805cede77d03ade5af97b850e9cb3bcc3c55ea103d5cf0e0cb9c'
DELTA_PROTOCOL = 'qwen38_locomo_a800_delta_readonly_v1'
CONDITIONS = ('self', 'empty', 'donor', 'full_context')
FIELDS = ['rows', 'values', 'counts', 'block_clock', 'cumulative_log_alpha',
          'row_last_written_block', 'row_log_alpha_at_write']
GUARD_FIELDS = ('sample_id', 'selected_source', 'query_arm', 'phase', 'epochs',
                'identities', 'written_tokens', 'state_hashes', 'state_fields',
                'writer_sha256', 'frozen_reader_writer', 'binding_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def require_completions(full, full_exit, delta, delta_exit):
    require(type(full_exit.get('returncode')) is int and full_exit['returncode'] == 0
        and type(delta_exit.get('exit_code')) is int and delta_exit['exit_code'] == 0,
            'both full-run processes must have exited zero')
    require(full.get('status') == 'completed' and full.get('mode') == 'formal'
        and full.get('generated_questions') == full.get('total_questions') == 1986
        and full.get('contract_sha256') == FULL_CONTRACT_SHA, 'full_context is incomplete or unbound')
    require(delta.get('protocol') == DELTA_PROTOCOL and delta.get('status') == 'completed'
        and delta.get('mode') == 'rest' and delta.get('predictions') == 5958
        and delta.get('binding_sha256') == delta_exit.get('binding_sha256') == DELTA_BINDING_SHA,
        'delta is incomplete or unbound')
    for key in ('memory_updates', 'optimizer_steps', 'backward_calls'):
        require(type(full.get(key)) is int and full[key] == 0, 'full_context changed model or memory')
    for key in ('query_memory_updates', 'optimizer_steps', 'backward_calls'):
        require(type(delta.get(key)) is int and delta[key] == 0, 'delta query changed model or memory')
    require(delta.get('annotations_opened') is False and delta.get('source_code_reverified') is True
        and delta.get('writer_tensor_sha256') == WRITER_TENSOR_SHA
        and delta.get('writer_archive_sha256') == WRITER_SHA, 'delta frozen writer/annotation contract differs')


def argv_value(argv, option):
    require(isinstance(argv, list) and all(isinstance(v, str) for v in argv)
        and argv.count(option) == 1 and argv.index(option) + 1 < len(argv), 'missing or duplicate launch option: ' + option)
    return argv[argv.index(option) + 1]


def validate_launch_identity(options, full_exit, delta_exit):
    full_root, delta_root = options.full_context_run.parent, options.delta_run.parent
    full_exit_path = full_root / 'formal_execution.json'
    delta_launch_root = delta_root / 'rest_launch_v1'
    require(options.full_context_execution_receipt.resolve() == full_exit_path.resolve()
        and options.delta_execution_receipt.resolve() == (delta_launch_root / 'execution.json').resolve(),
        'exit receipt is not the registered formal/rest receipt')
    full_launch = read_json(full_root / 'formal_launch.json')
    config_path = full_root / 'formal_launch_config.json'
    cfg = read_json(config_path)
    require(digest(config_path) == full_launch['launch_config_sha256']
        and cfg['argv'] == full_launch['argv'] and cfg['argv'][-3:] == ['--mode', 'formal', '--resume']
        and Path(argv_value(cfg['argv'], '--output')).resolve() == options.full_context_run.resolve()
        and cfg['resume_contract_sha256'] == full_launch['contract_sha256'] == FULL_CONTRACT_SHA
        and cfg['pilot_archive_manifest_sha256'] == FULL_PILOT_MANIFEST_SHA
        and cfg['GPU_launch_authorized'] is True and cfg['pilot_predictions_reused'] == 239
        and full_launch['completed_questions_before_launch'] == 239 and full_launch['remaining_questions'] == 1747,
        'full_context exit receipt belongs to another launch')
    full_child = read_json(full_root / 'formal_child.json')
    full_modified = (options.full_context_run / 'COMPLETE.json').stat().st_mtime
    require(type(full_child['pid']) is int and full_child['pid'] > 0
        and full_child['started'] <= full_modified <= full_exit['finished'],
        'full_context completion is outside the formal process lifetime')
    launch = read_json(delta_launch_root / 'launch.json')
    delta_child = read_json(delta_launch_root / 'child.json')
    argv = launch['argv']
    require(launch['binding_sha256'] == delta_child['binding_sha256'] == delta_exit['binding_sha256'] == DELTA_BINDING_SHA
        and argv_value(argv, '--mode') == 'rest' and argv.count('--resume') == 1
        and Path(argv_value(argv, '--binding')).resolve() == options.delta_binding.resolve()
        and argv.count('execute') == 1 and Path(argv[2]).name == 'qwen38_locomo_delta.py',
        'delta exit receipt belongs to another launch')
    delta_modified = (options.delta_run / 'COMPLETE.json').stat().st_mtime
    require(type(delta_child['pid']) is int and delta_child['pid'] > 0
        and datetime.fromisoformat(delta_child['started_utc']).timestamp() <= delta_modified
        <= datetime.fromisoformat(delta_exit['finished_utc']).timestamp(),
        'delta completion is outside the rest process lifetime')
    paths = [full_root / 'formal_launch.json', config_path, full_root / 'formal_child.json',
             delta_launch_root / 'launch.json', delta_launch_root / 'child.json']
    return {str(path): digest(path) for path in paths}


def expected_source(question, condition, donors):
    return question['sample_id'] if condition == 'self' else donors[question['sample_id']] if condition == 'donor' else None


def validate_guard(guard, sid, condition, count, states, donors):
    require(guard.get('protocol') == DELTA_PROTOCOL and guard.get('binding_sha256') == DELTA_BINDING_SHA
        and guard.get('questions') == count and guard.get('all_seven_fields_identical') is True,
        'readonly receipt header differs')
    before, after = guard['before'], guard['after']
    require(all(before[k] == after[k] for k in GUARD_FIELDS), 'query changed a seven-field state or frozen identity')
    source = expected_source({'sample_id': sid}, condition, donors)
    require(before['selected_source'] == before['sample_id'] == source
        and before['state_fields'] == FIELDS and before['frozen_reader_writer'] is True
        and before['writer_sha256'] == WRITER_TENSOR_SHA and before['binding_sha256'] == DELTA_BINDING_SHA,
        'readonly receipt has a wrong source, writer, binding, or state layout')
    empty_hashes = {record['state_hashes']['ordinary'] for record in states.values()}
    require(len(empty_hashes) == 1 and before['state_hashes']['ordinary'] in empty_hashes,
        'empty memory hash differs across conversations')
    if source is None:
        require(before['query_arm'] == 'ordinary' and before['phase'] == 'empty'
            and before['epochs'] == {'ordinary': 0, 'delta': 0} and before['written_tokens'] == 0
            and set(before['state_hashes'].values()) == empty_hashes, 'empty condition contains evidence')
    else:
        record = states[source]
        require(before['query_arm'] == 'delta' and before['phase'] == 'frozen'
            and before['state_hashes'] == record['state_hashes'] and before['epochs'] == record['epochs']
            and before['identities'] == record['checkpoint']['identities']
            and before['written_tokens'] == record['written_tokens'], 'query did not borrow the selected final state')
    return before['state_hashes'][before['query_arm']], before['epochs'][before['query_arm']]


def validate_matrix(questions, full_rows, delta_rows, donors, guards, delta_validator):
    require(len(questions) == len({q['qid'] for q in questions}) == 1986, 'question population differs')
    registry = {q['qid']: q for q in questions}
    require(len(full_rows) == 1986 and len(delta_rows) == 5958, 'incomplete four-condition matrix')
    seen = {}
    for row_number, row in enumerate(full_rows + delta_rows):
        key = row['qid'], row['condition']
        require(key not in seen and row['qid'] in registry and row['condition'] in CONDITIONS,
            'duplicate or unknown prediction')
        q = registry[row['qid']]
        require(row['sample_id'] == q['sample_id'] and isinstance(row['text'], str), 'prediction identity/text differs')
        tokens = row['generated_token_ids']
        require(isinstance(tokens, list) and 0 < len(tokens) <= 32
            and all(type(t) is int and 0 <= t < 248320 for t in tokens)
            and canonical(tokens) == row['generated_token_ids_sha256'], 'invalid continuation token trace')
        require(type(row['memory_updates']) is int and row['memory_updates'] == 0
            and isinstance(row['elapsed_seconds'], (int, float)) and math.isfinite(row['elapsed_seconds'])
            and row['elapsed_seconds'] >= 0, 'invalid timing or query writes')
        if row['condition'] == 'full_context':
            require(row_number < len(full_rows) and row['contract_sha256'] == FULL_CONTRACT_SHA
                and row['prompt_sha256'] == q['full_context_prompt_sha256']
                and row['prompt_tokens'] == q['full_context_prompt_tokens']
                and row['source_memory_sample_id'] is None and row['state_fingerprint_before'] is None
                and row['state_fingerprint_after'] is None and row['Delta_enabled'] is False,
                'full_context prompt/source/contract differs')
            require(row['finish_reason'] in ('stop', 'length')
                and row['max_token_limit_reached'] == (row['finish_reason'] == 'length' or len(tokens) >= 32)
                and (row['finish_reason'] != 'length' or len(tokens) == 32), 'full_context stopping contract differs')
        else:
            require(row_number >= len(full_rows), 'condition was published by the wrong runner')
            fp, epoch = guards[(q['sample_id'], row['condition'])]
            delta_validator(row, q, row['condition'], expected_source(q, row['condition'], donors), fp, DELTA_BINDING_SHA)
            require(row['state_epoch'] == epoch and row['prompt_tokens'] == q['short_prompt_tokens'],
                'delta prompt length/state clock differs')
        seen[key] = row
    require(set(seen) == {(qid, c) for qid in registry for c in CONDITIONS}, 'incomplete four-condition matrix')
    require([r['qid'] for r in full_rows] == [q['qid'] for q in questions], 'full_context order differs')
    require([(r['qid'], r['condition']) for r in delta_rows]
        == [(q['qid'], c) for q in questions for c in CONDITIONS[:3]], 'delta order differs')
    return seen


def verify_archive(manifest_path, expected_sha):
    require(digest(manifest_path) == expected_sha, 'pilot archive manifest SHA differs')
    manifest = read_json(manifest_path)
    files = manifest.get('files', manifest)
    require(isinstance(files, dict) and bool(files), 'pilot archive manifest has no files')
    for name, item in files.items():
        path = Path(item['archive_path'])
        require(path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['sha256'],
            'immutable pilot archive changed: ' + name)
    return files


def validate_pilot_rows(pilot_rows, final_lookup, expected_count, conditions):
    require(len(pilot_rows) == expected_count
        and len({(row['qid'], row['condition']) for row in pilot_rows}) == expected_count
        and all(row['sample_id'] == 'conv-48' and row['condition'] in conditions for row in pilot_rows),
        'pilot population differs')
    for row in pilot_rows:
        require(final_lookup[(row['qid'], row['condition'])] == row, 'a preserved pilot prediction was replaced')


def verify_journal_prefix(final_path, pilot_path, pilot_sha):
    prefix = Path(pilot_path).read_bytes()
    require(hashlib.sha256(prefix).hexdigest() == pilot_sha, 'pilot journal prefix SHA differs')
    with Path(final_path).open('rb') as stream:
        require(stream.read(len(prefix)) == prefix, 'final journal does not preserve the sealed pilot prefix')
    return len(prefix)


def validate_event_trace(events, data, lookup, prediction_hashes):
    frames = {(sid, f['frame_index']): f for sid, frames in data['frames'].items() for f in frames}
    question_registry = {q['qid']: q for q in data['questions']}
    expected_queries = {key for key in lookup if key[1] != 'full_context'}
    counters = {name: Counter() for name in ('before_evidence_forward', 'before_D1', 'D1_complete',
        'before_question', 'question_complete', 'before_query_forward')}
    for event in events:
        phase = event['phase']
        if phase == 'recovered_atomic_final_snapshot':
            require(event['new_model_forwards'] == 0, 'snapshot recovery performed inference')
            continue
        require(phase in counters, 'unregistered event phase')
        if phase in ('before_evidence_forward', 'before_D1', 'D1_complete'):
            key = event['sample_id'], event['frame_index']
            require(key in frames, 'unregistered evidence frame')
            f = frames[key]
            if phase == 'before_evidence_forward':
                require(event['frame_sha256'] == f['frame_sha256'] and event['new_tokens'] == f['new_tokens']
                    and event['encoder_tokens'] == f['prefix_tokens'] + f['new_tokens']
                    and event['current_delta_epoch'] == f['frame_index'], 'evidence prefill differs')
            elif phase == 'before_D1':
                require(event['new_tokens'] == f['new_tokens'], 'D1 new-token crop differs')
            else:
                evidence, state = event['evidence'], event['state']
                require(evidence['frame_sha256'] == f['frame_sha256'] and evidence['new_tokens'] == f['new_tokens']
                    and evidence['prefix_tokens_not_written'] == f['prefix_tokens']
                    and evidence['current_delta_encoding'] is True
                    and state['epoch'] == f['frame_index'] + 1 and state['block_clock'] == [f['frame_index'] + 1],
                    'D1 commit provenance differs')
        else:
            key = event['qid'], event['condition']
            require(key in expected_queries, 'unregistered query event')
            row = lookup[key]
            if phase == 'before_question':
                require(event['source_memory_sample_id'] == row['source_memory_sample_id']
                    and event['state_hash'] == row['state_fingerprint_before']
                    and event['maximum_forward_calls'] == 32, 'query event source/state differs')
            elif phase == 'question_complete':
                require(event['prediction_sha256'] == prediction_hashes[key]
                    and event['actual_forward_calls'] == event['generated_tokens'] == len(row['generated_token_ids']),
                    'completed query event differs')
            else:
                step = event['step']
                require(type(step) is int and 0 <= step < len(row['generated_token_ids']), 'query step out of range')
                prefix = question_registry[key[0]]['short_prompt_ids'] + row['generated_token_ids'][:step]
                require(event['prefix_tokens'] == len(prefix) and event['prefix_sha256'] == canonical(prefix),
                    'complete-prefix forward trace differs')
                key = (*key, step)
        counters[phase][key] += 1
    for phase in ('before_evidence_forward', 'before_D1', 'D1_complete'):
        require(counters[phase] == Counter({key: 1 for key in frames}), 'evidence was missing or written twice')
    for phase in ('before_question', 'question_complete'):
        require(counters[phase] == Counter({key: 1 for key in expected_queries}), 'question was missing or regenerated')
    expected_steps = Counter({(*key, step): 1 for key in expected_queries
        for step in range(len(lookup[key]['generated_token_ids']))})
    require(counters['before_query_forward'] == expected_steps, 'native forward trace missing or duplicated')
    return {phase: sum(values.values()) for phase, values in counters.items()}


def load_bound_module(name, path, source_hashes):
    path = Path(path)
    require(str(path) in source_hashes and digest(path) == source_hashes[str(path)], 'adapter module is unbound')
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_path(sources, name):
    candidates = [path for path in sources if Path(path).name == name]
    require(len(candidates) == 1, 'ambiguous adapter source: ' + name)
    return candidates[0]


def audit_full_resources(run, complete, pilot_files):
    gate = read_json(run / 'gate.json')
    require(gate['status'] == 'passed' and gate['max_prompt_tokens'] == 25194
        and gate['longest_admitted_without_truncation'] is True and gate['repeat_same_tokens_after_other_prompt'] is True
        and gate['observations'][0]['generated_token_ids'] == gate['observations'][2]['generated_token_ids']
        and digest(run / 'gate.json') == complete['gate_sha256'], 'full_context fit/repeat gate differs')
    pilot = read_json(pilot_files['run/PILOT_COMPLETE.json']['archive_path'])
    sets = [pilot['initial_resources'], pilot['final_resources'], gate['resources_after'],
            complete['initial_resources'], complete['final_resources']]
    sets += [read_json(path)['workers'] for path in sorted(run.glob('resources_after_*.json'))]
    require((run / 'resources_after_1986.json').is_file(), 'missing final resource boundary')
    initial = {row['rank']: row['canonical_cpu_weights_sha256'] for row in sets[0]}
    for workers in sets:
        require(sorted(row['rank'] for row in workers) == list(range(8)), 'missing TP resource report')
        for row in workers:
            require(row['canonical_cpu_weights_sha256'] == initial[row['rank']]
                and row['canonical_cpu_weights_unchanged'] is True and row['base_parameters_unchanged'] is True
                and row['Delta_enabled'] is False and row['offload_module_count'] == 16
                and row['offload_host_dtypes'] == ['torch.bfloat16']
                and row['offload_host_bytes'] == 10066329600 and row['offload_gpu_pool_bytes'] == 629145600
                and row['offload_net_saved_bytes'] == 9437184000 and row['kv_capacity_tokens'] >= 25226,
                'full_context canonical CPU/base weight or capacity audit differs')
    return initial


def output_diagnostics(rows):
    """Report repetitive generations transparently after the full matrix is sealed."""
    result = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row['condition'] == condition]
        traces = Counter(tuple(row['generated_token_ids']) for row in selected)
        common, count = traces.most_common(1)[0]
        exemplar = next(row for row in selected if tuple(row['generated_token_ids']) == common)
        pilot = [row for row in selected if row['sample_id'] == 'conv-48']
        result[condition] = {'questions': len(selected), 'distinct_generated_token_sequences': len(traces),
            'most_common_sequence_count': count, 'most_common_generated_token_ids': list(common),
            'most_common_decoded_text': exemplar['text'],
            'fixed_pilot_questions': len(pilot),
            'fixed_pilot_distinct_generated_token_sequences': len({tuple(row['generated_token_ids']) for row in pilot}),
            'interpretation': 'Generation diversity only; does not establish answer correctness or memory retrieval.'}
    return result


def validate_scored_populations(scored):
    require(scored['metrics']['main_f1']['question_count'] == 1540
        and scored['metrics']['adversarial']['question_count'] == 446,
        'main F1 and adversarial populations must remain separate at 1540 and 446')
    require(scored['primary_population'] == 'categories 1-4 only; adversarial separate',
        'scorer mixed the primary and adversarial populations')


def assemble(options):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'run assembly with CUDA_VISIBLE_DEVICES empty')
    require(not options.output.exists(), 'assembly output must be fresh')
    # This gate intentionally precedes opening prediction or annotation files.
    full = read_json(options.full_context_run / 'COMPLETE.json')
    delta = read_json(options.delta_run / 'COMPLETE.json')
    full_exit, delta_exit = read_json(options.full_context_execution_receipt), read_json(options.delta_execution_receipt)
    require_completions(full, full_exit, delta, delta_exit)
    launch_identities = validate_launch_identity(options, full_exit, delta_exit)
    require(read_json(options.full_context_run / 'STATUS.json') == full, 'full_context final status differs')
    final_delta_status = read_json(options.delta_run / 'STATUS.json')
    require(final_delta_status == {'status': 'completed', 'mode': delta['mode'], 'predictions': 5958,
        'binding_sha256': DELTA_BINDING_SHA}, 'delta final status differs')
    require(digest(options.prepared / 'manifest.json') == PREPARED_SHA
        and digest(options.scorer) == SCORER_SHA, 'original manifest or scorer changed')
    contract = read_json(options.full_context_run / 'contract.json')
    require(canonical(contract) == FULL_CONTRACT_SHA, 'full_context contract differs')
    binding_path = options.delta_binding / 'binding.json'
    require(digest(binding_path) == DELTA_BINDING_SHA, 'delta binding differs')
    binding = read_json(binding_path)
    require(binding['locomo']['amendment_sha256'] == AMENDMENT_SHA
        and digest(binding['locomo']['amendment']) == AMENDMENT_SHA, 'delta amendment differs')
    require(Path(binding['locomo']['execution_root']).resolve() == options.delta_run.resolve()
        and Path(binding['locomo']['prepared']).resolve() == options.prepared.resolve(), 'bound runtime paths differ')
    for sources in (contract['source_sha256'], binding['source_sha256']):
        for path, sha in sources.items():
            require(digest(path) == sha, 'bound runtime source changed: ' + path)
    full_runner = load_bound_module('locomo_assembly_full_runner',
        source_path(contract['source_sha256'], 'qwen38_locomo_full_context_v3.py'), contract['source_sha256'])
    delta_runner = load_bound_module('locomo_assembly_delta_runner',
        source_path(binding['source_sha256'], 'qwen38_locomo_delta.py'), binding['source_sha256'])
    verified_binding, data = delta_runner.verify_binding(options.delta_binding)
    require(verified_binding == binding, 'binding changed during verification')
    writer = delta_runner.read_writer_checkpoint(binding['writers']['delta']['path'])
    require(writer == binding['writer_audit'], 'original formal delta writer tensor/endpoint differs')
    _, questions, _, _ = full_runner.load_population(options.prepared)
    require({q['qid']: q for q in questions} == {q['qid']: q for q in data['questions']}, 'runner question populations differ')
    require(contract['question_order'] == [q['qid'] for q in questions]
        and binding['locomo']['question_order'] == [q['qid'] for q in questions]
        and binding['locomo']['query_prompt_bindings'] == {q['qid']: q['short_prompt_sha256'] for q in questions}
        and contract['prompt_bindings'] == [{'qid': q['qid'], 'sha256': q['full_context_prompt_sha256'],
            'tokens': q['full_context_prompt_tokens']} for q in questions], 'frozen prompt binding differs')
    # Native prefetch source identities are part of the fixed full_context contract.
    fork = read_json(binding['vllm_fork_manifest']['path'])
    for name, sha in contract['native_offload_sources'].items():
        require(digest(Path(fork['package_root']) / name) == sha, 'native offload source changed')
    full_path, delta_path = options.full_context_run / 'predictions.jsonl', options.delta_run / 'delta_predictions.jsonl'
    require(Path(delta['combined_predictions_path']).resolve() == delta_path.resolve(), 'delta publication path differs')
    require(digest(full_path) == full['predictions_sha256'] and digest(delta_path) == delta['combined_predictions_sha256'],
        'completed prediction SHA differs')
    full_archive = verify_archive(options.full_context_pilot_archive / 'manifest.json', FULL_PILOT_MANIFEST_SHA)
    require(options.delta_pilot_manifest_sha256 == DELTA_PILOT_MANIFEST_SHA, 'registered delta pilot archive SHA differs')
    delta_archive = verify_archive(options.delta_pilot_manifest, options.delta_pilot_manifest_sha256)
    index = read_json(options.delta_run / 'state_index.json')
    require(index['protocol'] == DELTA_PROTOCOL and index['binding_sha256'] == DELTA_BINDING_SHA
        and index['states'] == delta['states'] and set(index['states']) == set(data['conversations']),
        'final ten-state index differs')
    states = index['states']
    for sid, record in states.items():
        require(Path(record['checkpoint']['path']).resolve() == (options.delta_run / 'states' / (sid + '.state.pt')).resolve(),
            'snapshot path differs from its conversation')
        recovered = delta_runner.recover_snapshot(record['checkpoint']['path'], sid, DELTA_BINDING_SHA, data)
        require(all(recovered[k] == record[k] for k in ('sample_id', 'frames', 'written_tokens', 'state_hashes', 'epochs', 'checkpoint')),
            'seven-field snapshot payload/provenance differs')
    require(sum(r['frames'] for r in states.values()) == 426 and sum(r['written_tokens'] for r in states.values()) == 215278,
        'final evidence coverage differs')
    guards, guard_hashes = {}, {}
    counts = Counter(q['sample_id'] for q in questions)
    for sid, count in counts.items():
        for condition in CONDITIONS[:3]:
            path = options.delta_run / f'READONLY_{sid}_{condition}.json'
            guards[sid, condition] = validate_guard(read_json(path), sid, condition, count, states, data['donors'])
            guard_hashes[path.name] = digest(path)
    final_workers = delta['resources']
    require(sorted(row['rank'] for row in final_workers) == list(range(8)), 'missing final delta TP report')
    require(all(all(row[k] == final_workers[0][k] for k in GUARD_FIELDS) for row in final_workers),
        'final delta TP state disagreement')
    require(all(row['frozen_reader_writer'] is True and row['writer_sha256'] == WRITER_TENSOR_SHA
        and row['binding_sha256'] == DELTA_BINDING_SHA and row['state_fields'] == FIELDS for row in final_workers),
        'final delta model or state layout differs')
    full_rows, delta_rows = read_rows(full_path), read_rows(delta_path)
    lookup = validate_matrix(questions, full_rows, delta_rows, data['donors'], guards, delta_runner.validate_prediction)
    require(full_runner.resume_rows(options.full_context_run, contract, questions) == full_rows, 'full checkpoint/publication differs')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(binding['base_model']['model_path'], local_files_only=True)
    require(all(row['text'] == tokenizer.decode(row['generated_token_ids'], skip_special_tokens=True)
        for row in full_rows + delta_rows), 'continuation text differs from the pinned tokenizer')
    prediction_hashes = {}
    for row in delta_rows:
        path = delta_runner.prediction_path(options.delta_run, row['qid'], row['condition'])
        require(read_json(path) == row, 'delta combined publication differs from durable prediction')
        prediction_hashes[row['qid'], row['condition']] = digest(path)
    require(len(list((options.delta_run / 'predictions').glob('*.json'))) == 5958, 'extra durable delta predictions')
    pilot_full = read_rows(full_archive['run/predictions.jsonl']['archive_path'])
    validate_pilot_rows(pilot_full, lookup, 239, ('full_context',))
    pilot_delta = []
    for name, item in delta_archive.items():
        if name.startswith('predictions/') and name.endswith('.json'):
            row = read_json(item['archive_path'])
            require(digest(options.delta_run / name) == item['sha256'], 'pilot durable prediction bytes changed')
            pilot_delta.append(row)
    validate_pilot_rows(pilot_delta, lookup, 717, CONDITIONS[:3])
    pilot = read_json(delta_archive['PILOT.json']['archive_path'])
    require(pilot['protocol'] == DELTA_PROTOCOL and pilot['status'] == 'completed' and pilot['predictions'] == 717
        and pilot['binding_sha256'] == DELTA_BINDING_SHA, 'delta pilot completion differs')
    require(read_json(delta_archive['execution.json']['archive_path'])['exit_code'] == 0, 'delta pilot did not exit zero')
    for condition in CONDITIONS[:3]:
        name = f'READONLY_conv-48_{condition}.json'
        require(digest(options.delta_run / name) == delta_archive[name]['sha256'], 'pilot readonly receipt changed')
    for sid, record in pilot['states'].items():
        require(record == states[sid], 'pilot final snapshot/index was replaced')
    event_path = options.delta_run / 'execution_events.jsonl'
    require(digest(event_path) == delta['execution_events_sha256'], 'final event journal changed')
    verify_journal_prefix(event_path, delta_archive['execution_events.jsonl']['archive_path'], pilot['execution_events_sha256'])
    events = validate_event_trace(read_rows(event_path), data, lookup, prediction_hashes)
    cpu_weight_hashes = audit_full_resources(options.full_context_run, full, full_archive)
    import torch
    require(not torch.cuda.is_initialized(), 'CPU assembly initialized CUDA')
    require(digest(full_path) == full['predictions_sha256'] and digest(delta_path) == delta['combined_predictions_sha256']
        and digest(options.scorer) == SCORER_SHA and digest(binding_path) == DELTA_BINDING_SHA,
        'input changed during assembly validation')
    statistics = data['protocol']['statistics']
    require(statistics['primary_questions'] == 1540 and statistics['adversarial_questions'] == 446,
        'frozen category population differs')
    receipt = {'protocol': 'qwen38_locomo_four_condition_assembly_v2', 'status': 'validated',
        'experimental_method': {'memory_conditions': 'Original frozen DELTA writer checkpoint used for the seven-task evaluation; A800 live_v3 FP32 split PLE; independent empty state per conversation; 512 new tokens with at most 128 old prefix tokens; crop before writer features; one D1 per evidence frame; alpha=1; final seven-field state borrowed read-only.',
            'full_context': 'A800 BF16 full original history, no truncation, native vLLM prefetch offload; Delta disabled.',
            'parent_protocol_role': 'Unchanged prepared-data and scoring contract. Memory execution follows the separately sealed delta amendment.',
            'memory_writer_method': 'delta', 'device_for_all_four_conditions': 'A800',
            'fixed_instruction': data['protocol']['generation']['instruction'],
            'zero_shot_transfer': 'Frozen offline-trained delta writer checkpoint also used for the seven-task forward-write evaluation; LoCoMo starts each conversation from empty memory, without using the seven-task terminal Delta or any LoCoMo training/prompt tuning/score-based adaptation.',
            'writer_training_regime': 'offline', 'writer_updated_during_seven_task_forward_write_evaluation': False,
            'seven_task_terminal_memory_loaded': False,
            'limitations': ['The fixed instruction explicitly requests No information available when the history does not answer the question; adversarial refusal accuracy must not substitute for retrieval F1.',
                'Memory conditions use the original FP32 split PLE read path; full_context uses the BF16 base path with exact full history. The comparison includes this execution-path difference.',
                'Images are represented only by the original supplied text captions.'],
            'binding_sha256': DELTA_BINDING_SHA, 'amendment_sha256': AMENDMENT_SHA,
            'writer_archive_sha256': WRITER_SHA, 'writer_tensor_sha256': WRITER_TENSOR_SHA,
            'full_context_contract_sha256': FULL_CONTRACT_SHA},
        'population': {'unique_pairs': len(lookup), 'questions': 1986, 'conversations': 10,
            'per_condition': dict(Counter(row['condition'] for row in lookup.values()))},
        'reporting_policy': {'primary': {'metric': 'official category-aware token F1', 'categories': [1, 2, 3, 4], 'questions': 1540},
            'adversarial_separate': {'metric': 'official refusal-substring accuracy', 'category': 5, 'questions': 446},
            'combined_main_and_adversarial_score': False,
            'generation_diversity_must_accompany_scores': True,
            'fixed_prompt_and_method_unchanged_after_pilot': True},
        'output_diagnostics': output_diagnostics(full_rows + delta_rows),
        'prepared_manifest_sha256': PREPARED_SHA, 'original_scorer_sha256': SCORER_SHA,
        'source_predictions': {'full_context': {'path': str(full_path), 'sha256': full['predictions_sha256']},
            'delta': {'path': str(delta_path), 'sha256': delta['combined_predictions_sha256']}},
        'completion_receipts': {str(path): digest(path) for path in (options.full_context_run / 'COMPLETE.json',
            options.full_context_execution_receipt, options.delta_run / 'COMPLETE.json', options.delta_execution_receipt)},
        'formal_launch_identity_sha256': launch_identities,
        'pilot_reuse': {'full_context_rows': 239, 'delta_rows': 717, 'no_regeneration': True,
            'full_context_archive_manifest_sha256': FULL_PILOT_MANIFEST_SHA,
            'delta_archive_manifest_sha256': options.delta_pilot_manifest_sha256, 'journal_prefix_preserved': True},
        'all_prompts_bound': True, 'all_continuations_match_pinned_tokenizer': True,
        'all_ten_snapshot_payloads_and_conversation_provenance_verified': True,
        'all_thirty_seven_field_readonly_receipts_verified': True, 'readonly_receipt_sha256': guard_hashes,
        'event_counts': events, 'canonical_CPU_weight_SHA256_by_rank': cpu_weight_hashes,
        'query_memory_updates': 0, 'optimizer_steps': 0, 'backward_calls': 0,
        'cpu_only': True, 'cuda_initialized': False, 'annotations_parsed_by_validator': False,
        'scores_computed': False, 'assembly_source_sha256': digest(__file__)}
    options.output.mkdir(parents=True, exist_ok=False)
    combined = options.output / 'combined_predictions.jsonl'
    with combined.open('x', encoding='utf-8', newline='\n') as stream:
        for q in questions:
            for condition in CONDITIONS:
                stream.write(json.dumps(lookup[q['qid'], condition], ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
    receipt['combined_predictions_sha256'] = digest(combined)
    write_json(options.output / 'method_and_provenance.json', receipt)
    if options.score:
        # Original scoring code and its own pinned upstream dependencies remain unchanged.
        command = [sys.executable, '-B', str(options.scorer), '--prepared', str(options.prepared),
                   '--predictions', str(combined), '--output', str(options.output / 'official_scores')]
        result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True)
        write_json(options.output / 'scoring_execution.json', {'argv': command, 'returncode': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr, 'original_scorer_sha256': digest(options.scorer)})
        require(result.returncode == 0 and digest(options.scorer) == SCORER_SHA, 'original scorer failed or changed')
        scored = read_json(options.output / 'official_scores/report.json')
        require(scored['predictions_sha256'] == digest(combined)
            and scored['prepared_manifest_sha256'] == PREPARED_SHA, 'scorer report provenance differs')
        validate_scored_populations(scored)
        write_json(options.output / 'report.json', {'experimental_method': receipt['experimental_method'],
            'reporting_policy': receipt['reporting_policy'], 'output_diagnostics': receipt['output_diagnostics'],
            'main_results_1540_questions': scored['metrics']['main_f1'],
            'adversarial_results_446_questions': scored['metrics']['adversarial'],
            'method_and_provenance_sha256': digest(options.output / 'method_and_provenance.json'),
            'original_scorer_report_sha256': digest(options.output / 'official_scores/report.json'),
            'scores': scored})
    return {'status': 'scored' if options.score else 'assembled_not_scored', 'output': str(options.output),
            'unique_pairs': len(lookup), 'combined_predictions_sha256': digest(combined), 'cuda_initialized': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'full-context-run', 'full-context-execution-receipt', 'full-context-pilot-archive',
                 'delta-run', 'delta-binding', 'delta-execution-receipt', 'delta-pilot-manifest', 'scorer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--delta-pilot-manifest-sha256', required=True)
    parser.add_argument('--score', action='store_true', help='Invoke the unchanged scorer only after every validation passes.')
    print(json.dumps(assemble(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()

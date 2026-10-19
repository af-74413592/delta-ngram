"""Fixed donor controls and source-balanced checkpoint selection statistics."""
from collections import defaultdict
import hashlib
import json
import math
import random


def window_key(window):
    return window.source, window.document_id, window.start, window.blocks


def select_donors(windows, seed=104729, *, pool=None, allow_missing=False):
    """Deterministic same-source/length, different-family development donors.

    Every scored target must have a donor; no target is silently discarded and
    there is no fallback to training documents or another source.
    """
    pools = defaultdict(list)
    for window in windows if pool is None else pool:
        pools[window.source, window.blocks, window.block_size].append(window)
    result = {}
    for target in windows:
        candidates = [item for item in pools[target.source, target.blocks, target.block_size]
                      if item.group_id != target.group_id]
        if hasattr(target, 'articles'):
            from .data.event_streams import event_donor_compatible
            candidates = [item for item in candidates if event_donor_compatible(target, item)]
        if not candidates:
            if allow_missing:
                continue
            raise ValueError(f'no matched independent development donor for {window_key(target)}')
        def rank(donor):
            value = [seed, window_key(target), window_key(donor)]
            return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()
        result[window_key(target)] = min(candidates, key=rank)
    return result


def donor_supported_pool(windows, seed):
    """Freeze eligibility before sample selection, using all development donors."""
    donors = select_donors(windows, seed, allow_missing=True)
    supported = [window for window in windows if window_key(window) in donors]
    excluded = [list(window_key(window)) for window in windows if window_key(window) not in donors]
    return supported, donors, excluded


def donor_manifest(windows, donors):
    from .raw_lm import _json_hash
    return [{'target': list(window_key(target)), 'target_group_id': target.group_id,
             'donor': list(window_key(donors[window_key(target)])),
             'donor_group_id': donors[window_key(target)].group_id,
             'donor_token_sha256': _json_hash(donors[window_key(target)].token_ids)}
            for target in windows]


def donor_statistics(records, *, bootstrap_seed, bootstrap_samples):
    from .raw_lm import summarize_group_losses
    comparison = summarize_group_losses([
        {**record, 'empty_loss_sum': record['donor_loss_sum']} for record in records],
        bootstrap_seed=bootstrap_seed, bootstrap_samples=bootstrap_samples)
    return {'donor_nll': comparison['empty_nll'], 'delta_donor': comparison['delta_empty'],
            'donor_paired_group_bootstrap': comparison['paired_group_bootstrap']}


def macro_statistics(records, source_groups, *, bootstrap_seed, bootstrap_samples, donor):
    """Equal semantic-source macro, with family clusters shared across strata.

    news/news_event are one source for this macro. Within a source all fixed
    development targets contribute with their recorded target-token weights.
    """
    sources = {record['source'] for record in records}
    mapping = source_groups or {source: source for source in sources}
    if set(mapping) != sources:
        raise ValueError('macro_source_groups must cover exactly evaluated sources')
    grouped = defaultdict(dict)
    for record in records:
        category = mapping[record['source']]
        family = record.get('family_id', record['group_id'])
        item = grouped[category].setdefault(family, {'loss_tokens': 0, 'empty': 0., 'donor': 0.})
        item['loss_tokens'] += record['loss_tokens']
        item['empty'] += record['empty_loss_sum'] - record['self_loss_sum']
        if donor:
            item['donor'] += record['donor_loss_sum'] - record['self_loss_sum']
    conditions = ('empty', 'donor') if donor else ('empty',)
    rng = random.Random(bootstrap_seed)
    categories = {name: list(items.values()) for name, items in sorted(grouped.items())}
    def ratio(items, condition):
        return math.fsum(item[condition] for item in items) / sum(item['loss_tokens'] for item in items)
    point = {condition: math.fsum(ratio(items, condition) for items in categories.values()) / len(categories)
             for condition in conditions}
    draws = {condition: [] for condition in conditions}
    for _ in range(bootstrap_samples):
        samples = [[items[rng.randrange(len(items))] for _ in items] for items in categories.values()]
        for condition in conditions:
            draws[condition].append(math.fsum(ratio(items, condition) for items in samples) / len(samples))
    result = {'macro_source_groups': mapping, 'macro_categories': sorted(categories),
              'macro_groups_per_category': {name: len(items) for name, items in categories.items()}}
    for condition in conditions:
        values = sorted(draws[condition])
        def percentile(p):
            position = (len(values) - 1) * p
            lower = int(position)
            return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)
        result['source_macro_delta_' + condition] = point[condition]
        result['source_macro_' + condition + '_bootstrap'] = {
            'method': 'paired_family_bootstrap_within_source_equal_source_macro',
            'seed': bootstrap_seed, 'replicates': bootstrap_samples, 'confidence': .95,
            'ci_lower': percentile(.025), 'ci_upper': percentile(.975)}
    if donor:
        enough = all(len(items) >= 64 for items in categories.values())
        result['checkpoint_selection_eligible'] = (enough
            and result['source_macro_empty_bootstrap']['ci_lower'] > 0
            and result['source_macro_donor_bootstrap']['ci_lower'] > 0)
        result['checkpoint_selection_rule'] = {
            'primary': 'maximize source_macro_delta_donor among eligible registered milestones',
            'tie_break': 'earliest training_loss_tokens',
            'eligibility': 'at least 64 families/category; both macro paired CI lower bounds > 0',
            'if_none': 'no qualifying checkpoint; endpoint remains a reported negative or uncertain result',
            'oracle_used': False}
    return result

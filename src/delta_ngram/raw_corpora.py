"""Load disjoint single-document and complete event-stream training strata."""
from .data.raw_text import build_windows, load_documents


def load_corpora(raw, tokenizer, fingerprint, *, vocabulary_size=None):
    documents = load_documents(raw['dataset'], tokenizer, tokenizer_identity=fingerprint,
                               require_record_identity=True)
    events = []
    if raw.get('event_dataset'):
        from .data.event_streams import load_event_documents
        events = load_event_documents(raw['event_dataset'], tokenizer_identity=fingerprint)
        old_ids = {value for document in documents
                   for value in (document.id, document.metadata.get('original_document_id')) if value}
        old_groups = {value for document in documents
                      for value in (document.group_id, document.metadata.get('original_group_id'),
                                    document.metadata.get('audit_component_id')) if value}
        old_members = {member for document in documents for member in document.metadata.get('family_members', [])}
        if any(event.id in old_ids or event.group_id in old_groups
               or event.metadata.get('audit_component_id') in old_groups
               or any(article.id in old_ids for article in event.articles)
               or old_members.intersection(event.metadata.get('family_members', [])) for event in events):
            raise ValueError('event families must be excluded from the single-document corpus')
    if vocabulary_size is not None:
        for document in documents:
            if max(document.token_ids) >= vocabulary_size:
                raise ValueError(f'document {document.id} has token IDs outside the reader vocabulary')
        for event in events:
            if any(max(article.token_ids) >= vocabulary_size for article in event.articles):
                raise ValueError(f'event {event.id} has token IDs outside the reader vocabulary')
    return documents, events


def event_pool(raw, events, split):
    from .data.event_streams import build_event_trajectories
    conditional = raw.get('source_length_weights')
    counts = raw['block_counts'] if conditional is None else [
        count for count, weight in conditional.get('news_event', {}).items() if weight > 0]
    if events and not counts:
        raise ValueError('event dataset requires nonzero news_event length cells')
    if not events:
        return {}
    return build_event_trajectories(events, raw['block_size'], split=split, block_counts=counts)


def trajectory_pools(raw, documents, events, split='train'):
    conditional = raw.get('source_length_weights')
    options = {} if conditional is None else {'source_block_counts': {
        source: [count for count, weight in weights.items() if weight > 0]
        for source, weights in conditional.items()}}
    windows = build_windows(documents, raw['block_size'], tuple(raw['block_counts']), split=split, **options)
    for count, items in event_pool(raw, events, split).items():
        windows[count].extend(items)
    return windows


def event_document_metadata(event):
    from .data.raw_text import _digest
    return {'id': event.id, 'group_id': event.group_id, 'source': event.source, 'split': event.split,
            'tokens': sum(len(a.token_ids) for a in event.articles),
            'target_tokens': sum(len(a.token_ids) for a in event.articles[1:]),
            'articles': [{'id': a.id, 'tokens': len(a.token_ids), 'token_sha256': _digest(a.token_ids),
                          'metadata': a.metadata} for a in event.articles], 'metadata': event.metadata}

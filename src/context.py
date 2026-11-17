"""Repack all distinct query rankings; exactly one bounded context per request."""
import json

from .retrieval import rrf


def pack_context(searches: list[dict], budget: int) -> dict:
    if type(budget) is not int or budget < 1:
        raise ValueError('budget must be positive')
    unique = {}
    for search in searches:
        key = (search['query'], json.dumps(search.get('filter', {}), sort_keys=True))
        unique[key] = search['results']
    ranked = rrf(list(unique.values()))
    blocks, ids = [], []
    for chunk in ranked:
        block = f"[{chunk['id']} | {chunk['source']} | PDF page {chunk['page']}]\n{chunk['text']}"
        if len('\n\n'.join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(chunk['id'])
    return {'text': '\n\n'.join(blocks), 'ids': ids, 'budget_chars': budget,
            'chars': len('\n\n'.join(blocks)), 'selection_policy': 'rrf_over_distinct_queries_then_id',
            'chunks': [chunk for chunk in ranked if chunk['id'] in ids]}

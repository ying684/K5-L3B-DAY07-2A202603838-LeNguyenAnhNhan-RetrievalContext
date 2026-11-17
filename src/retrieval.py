"""Three methods over identical Day 6 chunks: dense, TF-IDF, hybrid RRF."""
from __future__ import annotations

import time

from starter.baseline import TfidfIndex

from .contracts import nonempty, validate_filters


def rrf(rankings: list[list[dict]], constant: int = 60) -> list[dict]:
    merged: dict[str, dict] = {}
    if constant < 1:
        raise ValueError('RRF constant must be positive')
    for branch, results in enumerate(rankings):
        if len({row['id'] for row in results}) != len(results):
            raise ValueError('each ranking must have unique IDs')
        for rank, row in enumerate(results, 1):
            item = merged.setdefault(row['id'], {**row, 'score': 0.0, 'component_ranks': {}})
            if item['text'] != row['text'] or item['source_id'] != row['source_id'] or item['page'] != row['page']:
                raise ValueError('same chunk ID has inconsistent content/provenance')
            item['score'] += 1 / (constant + rank)
            item['component_ranks'][str(branch)] = {'rank': rank, 'score': row.get('score')}
    return [{**row, 'rank': rank, 'method': 'hybrid'} for rank, row in enumerate(
        sorted(merged.values(), key=lambda row: (-row['score'], row['id'])), 1)]


class Retriever:
    def __init__(self, chunks: list[dict], config: dict, *, store=None, embedder=None):
        self.chunks, self.config, self.store, self.embedder = chunks, config, store, embedder
        self.lexical = TfidfIndex(chunks)

    def search(self, query: str, *, filters: dict | None = None) -> dict:
        nonempty(query, 'search query')
        filters = validate_filters(filters)
        started = time.perf_counter()
        config = self.config
        branches, usage = {}, []
        if config['method'] in ('dense', 'hybrid'):
            if self.store is None or self.embedder is None:
                raise RuntimeError('dense retrieval needs a real vector store and embedder')
            usage_start = len(self.embedder.usage)
            vector = self.embedder.embed_query(query)
            branches['dense'] = self.store.search(vector, filters=filters, top_k=config['candidate_k'])
            usage = self.embedder.usage[usage_start:]
        if config['method'] in ('lexical', 'hybrid'):
            rows = self.lexical.search(query, top_k=len(self.chunks))
            rows = [row for row in rows if all(row['metadata'].get(k) == v for k, v in filters.items())]
            branches['lexical'] = [{**row, 'rank': rank, 'method': 'lexical'}
                                   for rank, row in enumerate(rows[:config['candidate_k']], 1)]
        if config['method'] == 'hybrid':
            ranked = rrf([branches['dense'], branches['lexical']], config['rrf_constant'])
        else:
            ranked = branches[config['method']]
        return {'query': query, 'filter': filters, 'filter_provenance': 'explicit_runtime_input' if filters else 'none',
                'method': config['method'], 'candidate_k': config['candidate_k'],
                'return_k': config['return_k'], 'results': ranked[:config['return_k']],
                'original_top3': ranked[:3], 'candidate_rankings': branches,
                'embedding_usage': usage, 'elapsed_ms': (time.perf_counter() - started) * 1000}

"""Shared validation. Benchmark labels never belong to a runtime query."""
from __future__ import annotations

import math
from typing import Any


def nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    return value


def validate_filters(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) - {'chapter', 'language'}:
        raise ValueError('filters only support chapter and language')
    if 'chapter' in value and (type(value['chapter']) is not int or value['chapter'] < 1):
        raise ValueError('chapter must be a positive integer')
    if 'language' in value:
        nonempty(value['language'], 'language')
    return dict(value)


def runtime_query(case: dict[str, Any]) -> dict[str, Any]:
    """Construct a whitelist, never pass the original benchmark dict onward."""
    return {'question': nonempty(case.get('question'), 'question'),
            'filters': validate_filters(case.get('filter', case.get('filters')))}


def normalize_chunk(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError('chunk must be an object')
    out = {'id': row.get('id', row.get('chunk_id')),
           'source_id': row.get('source_id', row.get('document_id')),
           'source': row.get('source'), 'page': row.get('page'),
           'text': row.get('text'), 'metadata': row.get('metadata')}
    for key in ('id', 'source_id', 'source', 'text'):
        nonempty(out[key], key)
    if type(out['page']) is not int or out['page'] < 1:
        raise ValueError('page must be a positive physical PDF page')
    if not isinstance(out['metadata'], dict):
        raise ValueError('metadata must be an object')
    return out


def validate_vector(vector: Any, dimensions: int = 1024) -> list[float]:
    if (not isinstance(vector, list) or len(vector) != dimensions
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)
            or not any(vector)):
        raise ValueError(f'embedding must contain {dimensions} finite values and have nonzero norm')
    return [float(v) for v in vector]


def validate_cases(cases: Any, *, split: str | None = None) -> list[dict[str, Any]]:
    if not isinstance(cases, list) or not cases:
        raise ValueError('cases must be a nonempty array')
    seen: set[str] = set()
    for case in cases:
        cid = nonempty(case.get('id'), 'case id')
        if cid in seen:
            raise ValueError(f'duplicate case: {cid}')
        seen.add(cid)
        runtime_query(case)
        if case.get('split') not in ('dev', 'test') or (split and case['split'] != split):
            raise ValueError(f'{cid}: incorrect split')
        if type(case.get('answerable')) is not bool:
            raise ValueError(f'{cid}: answerable must be boolean')
        if case['answerable']:
            nonempty(case.get('expected_answer'), 'expected_answer')
            if not isinstance(case.get('gold'), list) or not case['gold']:
                raise ValueError(f'{cid}: gold alternatives required')
            for gold in case['gold']:
                nonempty(gold.get('source_id'), 'gold source_id')
                nonempty(gold.get('evidence'), 'gold evidence')
                if type(gold.get('page')) is not int or gold['page'] < 1:
                    raise ValueError(f'{cid}: invalid evidence page')
        else:
            nonempty(case.get('absence_reason'), 'absence_reason')
            if case.get('expected_behavior') != 'search_and_abstain':
                raise ValueError(f'{cid}: unanswerable case must expect search_and_abstain')
    return cases

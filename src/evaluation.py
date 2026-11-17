"""Evidence metrics and explicit reviews; no automatic semantic grading."""
from __future__ import annotations

import statistics

from starter.agent_metrics import aggregate

from .contracts import validate_cases


def evidence_hit(chunk: dict, gold: list[dict]) -> bool:
    return any(chunk['source_id'] == g['source_id'] and chunk['page'] == g['page']
               and g['evidence'] in chunk['text'] for g in gold)


def validate_benchmark(cases: list[dict], chunks: list[dict], *, split: str,
                       require_minimum: bool = True) -> dict:
    validate_cases(cases, split=split)
    for case in cases:
        if case['answerable']:
            for gold in case['gold']:
                if not any(evidence_hit(chunk, [gold]) for chunk in chunks):
                    raise ValueError(f"{case['id']}: evidence alternative does not fit its source/page/chunk")
    answerable = sum(c['answerable'] for c in cases)
    unanswerable = len(cases) - answerable
    if require_minimum and (answerable < (15 if split == 'dev' else 5) or unanswerable < 2):
        raise ValueError('benchmark does not meet split minimum')
    return {'split': split, 'cases': len(cases), 'answerable': answerable, 'unanswerable': unanswerable}


def fraction(n: int, denominator: int) -> dict:
    return {'n': n, 'N': denominator, 'rate': n / denominator if denominator else None}


def retrieval_summary(cases: list[dict], traces: list[dict]) -> dict:
    by_id = {trace['query_id']: trace for trace in traces}
    if len(by_id) != len(traces) or set(by_id) != {c['id'] for c in cases}:
        raise ValueError('trace IDs must match all benchmark cases exactly')
    counts = {'hit1': 0, 'hit3': 0, 'context_evidence': 0}
    n_answerable = sum(case['answerable'] for case in cases)
    errors = 0
    for case in cases:
        trace = by_id[case['id']]
        if trace['status'] == 'error':
            errors += 1
            continue
        if not case['answerable']:
            continue
        results = trace.get('original_top3', [])
        counts['hit1'] += bool(results and evidence_hit(results[0], case['gold']))
        counts['hit3'] += any(evidence_hit(chunk, case['gold']) for chunk in results[:3])
        counts['context_evidence'] += any(evidence_hit(chunk, case['gold'])
                                           for chunk in trace['context']['chunks'])
    times = [trace['elapsed_ms'] for trace in traces]
    return {'cases': len(cases), 'answerable': n_answerable, 'errors': errors,
            'completion': fraction(len(cases) - errors, len(cases)),
            **{key: fraction(value, n_answerable) for key, value in counts.items()},
            'context_metric': 'source/page/exact alternative evidence in selected chunk; semantic review separate',
            'elapsed_ms_total': sum(times), 'elapsed_ms_median': statistics.median(times) if times else None}


def agent_summary(cases: list[dict], traces: list[dict], reviews: list[dict]) -> dict:
    if set(t['query_id'] for t in traces) != {c['id'] for c in cases}:
        raise ValueError('agent trace IDs must match all cases')
    if any(t['status'] not in ('answered', 'abstain', 'error') for t in traces):
        raise ValueError('invalid agent status')
    by_review = {review['query_id']: review for review in reviews}
    for case in cases:
        trace = next(t for t in traces if t['query_id'] == case['id'])
        if trace['status'] == 'error':
            continue
        review = by_review.get(case['id'])
        if review is None or not isinstance(review.get('rationale'), str) or not review['rationale'].strip():
            raise ValueError(f"{case['id']}: review rationale required")
        if not review.get('reviewer'):
            raise ValueError(f"{case['id']}: reviewer identity/type required")
        if case['answerable'] and (type(review.get('context_grade')) is not int or review['context_grade'] not in (0, 1, 2)):
            raise ValueError(f"{case['id']}: context grade 0/1/2 required")
        details = review.get('citation_reviews')
        support = review.get('citation_support')
        if not isinstance(details, list) or len(details) != len(trace['citations']):
            raise ValueError(f"{case['id']}: review every citation")
        if not isinstance(support, list) or len(support) != len(details):
            raise ValueError(f"{case['id']}: citation support labels required")
        for cid, detail, label in zip(trace['citations'], details, support):
            if detail.get('id') != cid or not detail.get('rationale') or type(detail.get('supports_claim')) is not bool:
                raise ValueError(f"{case['id']}: incomplete citation review")
            in_context = cid in trace['context']['ids']
            if label is not (in_context and detail['supports_claim']):
                raise ValueError(f"{case['id']}: citation label conflicts with provenance/support")
    summary = aggregate(cases, traces, reviews)
    summary['completion'] = fraction(len(cases) - summary['provider_errors'], len(cases))
    grades = [by_review[c['id']]['context_grade'] for c in cases if c['answerable']
              and next(t for t in traces if t['query_id'] == c['id'])['status'] != 'error']
    summary['context_grades'] = {str(grade): grades.count(grade) for grade in (0, 1, 2)}
    summary['context_grades']['N'] = len(grades)
    summary['search_call_distribution'] = {}
    for trace in traces:
        key = str(sum(e.get('event') == 'search' for e in trace['events']))
        summary['search_call_distribution'][key] = summary['search_call_distribution'].get(key, 0) + 1
    return summary


def review_template(cases: list[dict], traces: list[dict]) -> list[dict]:
    by_case = {case['id']: case for case in cases}
    rows = []
    for trace in traces:
        if trace['status'] == 'error':
            continue
        row = {'query_id': trace['query_id'], 'reviewer': '', 'rationale': '',
               'citation_support': [None for _ in trace['citations']],
               'citation_reviews': [{'id': cid, 'supports_claim': None, 'rationale': ''} for cid in trace['citations']]}
        if by_case[trace['query_id']]['answerable']:
            row.update({'answer_grade': None, 'context_grade': None})
        else:
            row['abstention_correct'] = None
        rows.append(row)
    return rows

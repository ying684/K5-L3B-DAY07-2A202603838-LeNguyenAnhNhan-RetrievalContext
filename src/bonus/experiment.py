"""Frozen split runner and strict reviewed metrics, isolated from basic experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

from ..config import Settings
from .index import INDEX, PageIndex, build_index, sha, write
from .runtime import PROMPT, question_only, run
from .transport import PACER
from .providers import settings_for

BENCH = Path('data/bonus/benchmark')
ART = Path('artifacts/bonus')


def now() -> str:
    return datetime.now(timezone(timedelta(hours=7))).isoformat()


def snapshot() -> dict:
    paths = [*sorted(Path('src/bonus').glob('*.py')), *sorted(Path('configs/bonus').glob('*.json')),
             Path('src/config.py'), Path('src/http_client.py'), Path('requirements.txt'),
             *sorted(BENCH.glob('*.json')),
             INDEX / 'pages.jsonl', INDEX / 'manifest.json']
    if (ART/'provider_selection.json').exists():
        paths.append(ART/'provider_selection.json')
    hashes = {p.as_posix(): sha(p) for p in paths}
    return {'files': hashes, 'fingerprint': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def validate_benchmark() -> dict:
    counts, seen, questions = {}, set(), set()
    for split, total, answerable in [('dev', 18, 15), ('test', 30, 24)]:
        qs = json.loads((BENCH / f'{split}_questions.json').read_text(encoding='utf-8'))
        gold = json.loads((BENCH / f'{split}_gold.json').read_text(encoding='utf-8'))
        if len(qs) < total or len(qs) != len(gold) or [q['id'] for q in qs] != [g['id'] for g in gold]:
            raise ValueError('split cardinality/ID mismatch')
        groups = Counter()
        for q, g in zip(qs, gold):
            if set(q) != {'id', 'question'} or q['id'] in seen or q['question'] in questions:
                raise ValueError('runtime question schema/duplicate')
            seen.add(q['id']); questions.add(q['question'])
            if g['answerable']:
                groups[g['group']] += 1
                if not g['expected_answer'] or not g['evidence']:
                    raise ValueError('missing gold/evidence')
                if any(not 1 <= e['page'] <= 400 or not e['region'] for e in g['evidence']):
                    raise ValueError('invalid evidence')
                if g['group'] == 'multi_hop' and len({e['page'] for e in g['evidence']}) < 2:
                    raise ValueError('multi-hop requires two pages')
            elif not g.get('absence_check'):
                raise ValueError('unanswerable needs absence check')
        if sum(groups.values()) < answerable or len(gold)-sum(groups.values()) < total-answerable:
            raise ValueError('insufficient answerable/unanswerable')
        if split == 'test' and any(groups[k] < v for k, v in {'table':6,'diagram':5,'chart':5,'multi_hop':4,'text':4}.items()):
            raise ValueError('test groups below required minima')
        counts[split] = {'N': len(qs), 'answerable': sum(groups.values()),
                         'unanswerable': len(gold)-sum(groups.values()), 'groups': dict(groups)}
    return counts


def protocol() -> None:
    if (ART / 'protocol.json').exists():
        raise FileExistsError('preserve existing protocol')
    write(ART / 'protocol.json', {'created_at': now(), 'benchmark': validate_benchmark(),
          'selection_order': ['dev passed count', 'correct abstention', 'fewer errors', 'known cost', 'latency', 'config ID'],
          'passing': 'Answer correct and complete, every claim supported by correct original page/region; correct abstention for absent evidence. Errors/missing remain failures.',
          'threshold': {'numerator': 27, 'denominator': 30, 'percent': 90},
          'snapshot_at_protocol': snapshot(), 'authorship': 'Same Codex assistant authors and source-reviews; not an independent human audit.'})


def assert_test_allowed(config: dict, settings: Settings) -> None:
    path = ART / 'selection.json'
    if not path.exists():
        raise RuntimeError('test locked: freeze reviewed dev first')
    selection = json.loads(path.read_text(encoding='utf-8'))
    if selection['snapshot'] != snapshot():
        raise RuntimeError('test locked: code/config/benchmark/index changed after selection')
    if selection.get('protocol_sha256') != sha(ART / 'protocol.json'):
        raise RuntimeError('test locked: protocol changed after selection')
    if config['id'] != selection['selected_config'] or config != selection['config']:
        raise RuntimeError('test config is not frozen selection')
    if settings.chat_model != selection['model'] or settings.chat_base_url != selection['endpoint']:
        raise RuntimeError('test model/endpoint differs from selection')


def run_split(config_path: Path, split: str, workers: int = 2) -> Path:
    config = json.loads(config_path.read_text(encoding='utf-8'))
    settings=settings_for(config)
    if not settings.allow_external or not settings.chat_key:
        raise RuntimeError('authorized provider settings unavailable')
    if config.get('provider','openrouter')!='openrouter':
        verification=json.loads((ART/'provider_selection.json').read_text(encoding='utf-8'))
        if (not verification.get('end_to_end_smoke_passed') or verification['model']!=settings.chat_model
                or verification['endpoint']!=settings.chat_base_url or verification['provider']!=config['provider']):
            raise RuntimeError('direct provider preflight does not match configuration')
    if config.get('case_batch_size') and workers!=1:
        raise ValueError('configured batch cooldown requires exactly one worker')
    if not (ART / 'protocol.json').exists():
        raise RuntimeError('protocol must precede dev/test')
    frozen_protocol = json.loads((ART / 'protocol.json').read_text(encoding='utf-8'))
    for name, expected in frozen_protocol['snapshot_at_protocol']['files'].items():
        if name.startswith(BENCH.as_posix() + '/') and sha(Path(name)) != expected:
            raise RuntimeError('benchmark changed after protocol; create a new documented split')
    if split == 'test':
        assert_test_allowed(config, settings)
    # Deliberately never load evaluator gold in the runner.
    cases = json.loads((BENCH / f'{split}_questions.json').read_text(encoding='utf-8'))
    index = PageIndex()
    verification = index.verify()
    if not verification['continuous_400'] or verification['image_hash_failures']:
        raise RuntimeError('index/image verification failed')
    tag = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder = Path('results/bonus/runs') / f'{tag}_{config["id"]}_{split}'
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write(folder / 'manifest.json', {'started_at': now(), 'split': split, 'config': config,
          'model': settings.chat_model, 'endpoint': settings.chat_base_url, 'N': len(cases),
          'provider':config.get('provider','openrouter'),
          'snapshot': snapshot(), 'protocol_sha256':sha(ART / 'protocol.json'),
          'prompt': PROMPT, 'workers': workers, 'index_verification': verification})
    records = {}
    def execute_case(q, ordinal):
        batch=config.get('case_batch_size',0)
        if batch and ordinal and ordinal%batch==0:
            PACER.defer(config['batch_cooldown_seconds'])
        return run(question_only(q),config,index,settings)
    with (folder / 'traces.jsonl').open('x', encoding='utf-8') as stream:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(execute_case,q,i):q['id'] for i,q in enumerate(cases)}
            for future in as_completed(futures):
                query_id = futures[future]
                try:
                    trace = future.result()
                except Exception as exc:
                    trace = {'status':'error', 'error':{'type':type(exc).__name__,'message':str(exc)}, 'model_calls':[], 'events':[]}
                trace['id'] = query_id
                records[query_id] = trace
                stream.write(json.dumps(trace, ensure_ascii=False) + '\n'); stream.flush()
                print(query_id, trace['status'], trace.get('elapsed_seconds'), flush=True)
    write(folder / 'results.json', [records[q['id']] for q in cases])
    write(folder / 'review.json', [{'id':q['id'], 'passed':None, 'answer_correct_complete':None,
          'citation_support':None, 'rationale':'', 'source_pages_checked':[], 'reviewer':'Codex source comparison'} for q in cases])
    write(folder / 'execution.json', {'completed_at':now(), 'N':len(cases), 'wall_seconds':time.perf_counter()-started,
          'status_counts':dict(Counter(r['status'] for r in records.values()))})
    print(folder.as_posix(), flush=True)
    return folder


def summarize(folder: Path, gold_path: Path | None = None) -> dict:
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    split = manifest['split']
    gold_path = gold_path or BENCH / f'{split}_gold.json'
    expected_gold = manifest.get('snapshot',{}).get('files',{}).get((BENCH / f'{split}_gold.json').as_posix())
    if expected_gold and sha(gold_path) != expected_gold:
        raise ValueError('review gold differs from run snapshot; pass archived --gold for historical runs')
    cases = json.loads(gold_path.read_text(encoding='utf-8'))
    result_rows = json.loads((folder / 'results.json').read_text(encoding='utf-8'))
    review_rows = json.loads((folder / 'review.json').read_text(encoding='utf-8'))
    results = {r['id']:r for r in result_rows}
    reviews = {r['id']:r for r in review_rows}
    if len(results) != len(result_rows) or len(reviews) != len(review_rows):
        raise ValueError('duplicate result/review IDs')
    expected = {g['id'] for g in cases}
    if set(results) - expected or set(reviews) != expected:
        raise ValueError('review/results ID mismatch')
    groups, passed, errors, final_errors, costs, unknown = {}, 0, 0, 0, [], 0
    provider_error_cases, failed_requests, recovered_cases=0,0,0
    for g in cases:
        r, v = results.get(g['id'], {}), reviews[g['id']]
        if type(v['passed']) is not bool or not v['rationale'].strip():
            raise ValueError('every case needs a boolean review and rationale')
        if v['passed']:
            if any('error' in c for c in r.get('model_calls',[])):
                raise ValueError('conservative score: even recovered provider errors cannot pass')
            if r.get('status') not in ('answered','abstain'):
                raise ValueError('errors/missing cannot pass')
            if g['answerable'] and (r['status'] != 'answered' or v['answer_correct_complete'] is not True or v['citation_support'] is not True or not v['source_pages_checked']):
                raise ValueError('answerable pass needs answer/citation/source verification')
            if not g['answerable'] and r['status'] != 'abstain':
                raise ValueError('unanswerable pass requires abstention')
        group = g['group'] if g['answerable'] else 'unanswerable'
        entry = groups.setdefault(group, {'passed':0,'N':0})
        entry['N'] += 1; entry['passed'] += int(v['passed']); passed += int(v['passed'])
        final_error=r.get('status') in ('error',None)
        failed=sum('error' in c for c in r.get('model_calls',[]))
        final_errors+=final_error
        errors+=bool(final_error or failed)
        provider_error_cases+=bool(failed)
        failed_requests+=failed
        recovered_cases+=bool(failed and not final_error)
        for call in r.get('model_calls', []):
            value = call.get('usage', {}).get('cost')
            if isinstance(value, (int,float)):
                costs.append(value)
            else:
                unknown += 1
    for entry in groups.values():
        entry['rate'] = entry['passed']/entry['N']
    elapsed = [r.get('elapsed_seconds',0) for r in results.values()]
    output = {'split':split,'config_id':manifest['config']['id'], 'passed':passed,'N':len(cases),
              'rate':passed/len(cases), 'at_least_90_percent':passed*10 >= len(cases)*9,
              'groups':groups,'errors_or_missing':errors,'known_cost_usd':sum(costs),
              'final_errors_or_missing':final_errors,'provider_error_cases':provider_error_cases,
              'failed_request_attempts':failed_requests,'recovered_error_cases_still_scored_fail':recovered_cases,
              'unknown_cost_records':unknown, 'total_cost_usd':sum(costs) if not unknown and not errors else None,
              'sum_case_seconds':sum(elapsed),'median_case_seconds':statistics.median(elapsed),
              'model_requests':sum(len(r.get('model_calls',[])) for r in results.values()),
              'reviewer':'Codex source comparison, not independent human audit', 'gold_path':gold_path.as_posix()}
    write(folder / 'summary.json', output)
    return output


def freeze(folders: list[Path]) -> dict:
    if (ART / 'selection.json').exists():
        raise FileExistsError('selection already exists; preserve it')
    candidates = []
    for folder in folders:
        summary = summarize(folder)
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if summary['split'] != 'dev' or manifest['snapshot'] != snapshot():
            raise ValueError('selection needs dev run of current snapshot')
        if summary.get('final_errors_or_missing',summary['errors_or_missing']):
            raise ValueError('selection needs complete dev without system errors; keep failed runs and rerun dev')
        s = summary
        key = (-s['passed'], -s['groups'].get('unanswerable',{}).get('passed',0), s['errors_or_missing'],
               s['known_cost_usd'] if s['total_cost_usd'] is not None else float('inf'),
               s['sum_case_seconds'], s['config_id'])
        candidates.append((key, folder, manifest, summary))
    _, folder, m, s = min(candidates, key=lambda row:row[0])
    output = {'frozen_at':now(),'selected_config':s['config_id'],'config':m['config'],
              'provider':m.get('provider','openrouter'),
              'model':m['model'],'endpoint':m['endpoint'],'snapshot':snapshot(),
              'protocol_sha256':sha(ART / 'protocol.json'),
              'selected_dev_run':folder.as_posix(), 'dev_summaries':[c[3] for c in candidates],
              'dev_review_sha256':{c[1].as_posix():sha(c[1] / 'review.json') for c in candidates}}
    write(ART / 'selection.json', output)
    return output


def export_reviewable(folder: Path) -> Path:
    """Keep full copyrighted page text local; publish hash-addressed trace metadata."""
    def redact(value):
        if isinstance(value,list):
            return [redact(v) for v in value]
        if isinstance(value,dict):
            output = {}
            for k,v in value.items():
                if k == 'text' and isinstance(v,str):
                    output['text_sha256'] = hashlib.sha256(v.encode('utf-8')).hexdigest()
                    output['text_characters'] = len(v)
                else:
                    output[k] = redact(v)
            return output
        return value
    target = ART / 'runs' / folder.name
    target.mkdir(parents=True,exist_ok=True)
    for name in ['manifest.json','results.json','review.json','summary.json','execution.json']:
        path = folder / name
        if path.exists():
            write(target / name,redact(json.loads(path.read_text(encoding='utf-8'))))
    write(target / 'context_policy.json',{'full_trace_local':folder.as_posix(),
          'policy':'Full source text remains in ignored local results. Export keeps exact text SHA-256/length, page IDs, rank/score, image hashes, queries, tool calls, output, usage and errors.',
          'reconstruction':'Obtain the book legally, verify source hash, rebuild the pinned index. Search text is the page slice source_text_start_char:source_text_end_char (legacy: first2400chars); read_pages text is complete. Apply context_updates before the indicated modelstep; compacted previews become metadata while opened pages remain. Verify exported text SHA-256.'})
    return target


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('index'); sub.add_parser('protocol'); sub.add_parser('validate')
    p = sub.add_parser('run'); p.add_argument('--split', choices=['dev','test'], required=True)
    p.add_argument('--config',type=Path, required=True); p.add_argument('--allow-api',action='store_true')
    p.add_argument('--workers',type=int,choices=range(1,5),default=2)
    p = sub.add_parser('summarize'); p.add_argument('folder',type=Path); p.add_argument('--gold',type=Path)
    p = sub.add_parser('export'); p.add_argument('folder',type=Path)
    p = sub.add_parser('freeze'); p.add_argument('folders',nargs='+',type=Path)
    args = parser.parse_args()
    if args.command == 'index':
        result = build_index()
    elif args.command == 'protocol':
        protocol(); result = {'protocol_created':True}
    elif args.command == 'validate':
        result = {'benchmark':validate_benchmark(),'index':PageIndex().verify()}
    elif args.command == 'run':
        if not args.allow_api:
            parser.error('--allow-api required for real model calls')
        result = {'run':str(run_split(args.config,args.split,args.workers))}
    elif args.command == 'summarize':
        result = summarize(args.folder,args.gold)
    elif args.command == 'export':
        result = {'export':str(export_reviewable(args.folder))}
    else:
        result = freeze(args.folders)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

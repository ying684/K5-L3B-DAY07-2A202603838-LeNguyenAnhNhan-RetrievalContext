"""Split-isolated runner. Test evaluation requires a matching frozen selection."""
from __future__ import annotations

import hashlib
import json
import platform
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .agent import run_agent
from .config import Settings, load_config
from .context import pack_context
from .contracts import runtime_query
from .controller import Controller, SYSTEM_PROMPT
from .day6 import load_chunks, load_jsonl, sha256_file, write_json
from .evaluation import agent_summary, retrieval_summary, review_template, validate_benchmark


def timestamp() -> str:
    return datetime.now(timezone(timedelta(hours=7))).isoformat()


def runtime_fingerprint(root: Path = Path('.')) -> str:
    paths = [*sorted((root / 'src').glob('*.py')), *sorted((root / 'configs').glob('*.json')),
             root / 'requirements.txt', root / 'artifacts/protocol.json']
    records = [(p.relative_to(root).as_posix(), sha256_file(p)) for p in paths]
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def source_fingerprint(root: Path = Path('.')) -> str:
    paths = [*sorted((root / 'src').glob('*.py')), root / 'requirements.txt']
    return hashlib.sha256(json.dumps([(p.relative_to(root).as_posix(), sha256_file(p)) for p in paths]).encode()).hexdigest()


def check_protocol(split: str, config_path: Path, *, protocol_path: Path = Path('artifacts/protocol.json'),
                   selection_path: Path = Path('artifacts/selection.json'), data_dir: Path = Path('data/day6')) -> dict:
    if split == 'test' and not selection_path.exists():
        raise RuntimeError('test is locked: freeze the dev selection before evaluating test')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    # Read hashes, never the other split's labels.
    for name, expected in protocol['benchmark_sha256'].items():
        if sha256_file(Path('data/benchmark') / (name + '.json')) != expected:
            raise ValueError('benchmark differs from the protocol snapshot')
    manifest = json.loads(Path('artifacts/data_manifest.json').read_text(encoding='utf-8'))
    if manifest['vector_fingerprint'] != protocol['vector_fingerprint']:
        raise ValueError('index differs from the protocol snapshot')
    for name, expected in manifest['input_hashes'].items():
        if sha256_file(data_dir / name) != expected:
            raise ValueError('copied Day 6 data changed after snapshot validation')
    if split == 'test':
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        if selection['runtime_sha256'] != runtime_fingerprint():
            raise RuntimeError('test is locked: runtime/config/protocol changed after selection')
        if sha256_file(config_path) not in selection['allowed_config_sha256']:
            raise RuntimeError('test only permits the frozen baseline and selected config')
    return protocol


def usage_summary(traces: list[dict]) -> dict:
    model, embedding = [], []
    for trace in traces:
        model.extend(trace.get('model_calls', []))
        original = trace.get('original_search') or {}
        embedding.extend(original.get('embedding_usage', []))
        for event in trace.get('events', []):
            embedding.extend(event.get('embedding_usage', []))
    rows = model + embedding
    costs = [row.get('service_cost_usd') for row in rows]
    failed_cases = sum(t['status'] == 'error' for t in traces)
    return {'model_requests': len(model), 'embedding_requests': sum(not r.get('cache_hit', False) for r in embedding),
            'embedding_cache_hits': sum(r.get('cache_hit', False) for r in embedding),
            'known_service_cost_usd': sum(c for c in costs if isinstance(c, (int, float))),
            'cost_records_unknown': sum(c is None for c in costs),
            'failed_cases_cost_not_assumed_zero': failed_cases,
            'total_cost_usd': sum(costs) if not failed_cases and all(type(c) in (int, float) for c in costs) else None,
            'prompt_tokens': sum(r.get('usage', {}).get('prompt_tokens', 0) for r in model),
            'completion_tokens': sum(r.get('usage', {}).get('completion_tokens', 0) for r in model),
            'note': 'all requests in run, including original-query evaluation; unknown costs are not zero'}


def benchmark(settings: Settings, config_path: Path, *, split: str, kind: str,
              allow_api: bool, build_retriever) -> Path:
    protocol = check_protocol(split, config_path, data_dir=settings.data_dir)
    config = load_config(config_path)
    cases_path = Path('data/benchmark') / (split + '.json')
    cases = json.loads(cases_path.read_text(encoding='utf-8'))
    validate_benchmark(cases, load_chunks(settings.data_dir), split=split)
    if split == 'test':
        selection = json.loads(Path('artifacts/selection.json').read_text(encoding='utf-8'))
        for field in ('chat_model', 'embedding_model', 'dimensions', 'chat_base_url', 'embedding_base_url'):
            if selection[field] != getattr(settings, field):
                raise RuntimeError('test is locked: model/endpoint differs from frozen selection')
    if kind == 'agent' or config['method'] != 'lexical':
        verification = Path('artifacts/api_verification.json')
        if not verification.exists():
            raise RuntimeError('real API preflight has not passed; run smoke-api first')
        api = json.loads(verification.read_text(encoding='utf-8'))
        if not api.get('embedding_verified') or (kind == 'agent' and not api.get('chat_verified')):
            raise RuntimeError('real API preflight is incomplete')
        if (api.get('embedding_model') != settings.embedding_model
                or api.get('dimensions') != settings.dimensions
                or api.get('embedding_base_url') != settings.embedding_base_url
                or (kind == 'agent' and (api.get('chat_model') != settings.chat_model
                                         or api.get('chat_base_url') != settings.chat_base_url))):
            raise RuntimeError('configured model differs from API preflight')
    retriever = build_retriever(settings, config, allow_api)
    controller = Controller(settings, allow_api=allow_api) if kind == 'agent' else None
    run_id = datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f')
    output = Path('results/runs') / f'{run_id}_{config["id"]}_{split}_{kind}'
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'run_id': run_id, 'started_at': timestamp(), 'kind': kind, 'split': split,
                'config': config, 'config_path': config_path.as_posix(), 'cases_path': cases_path.as_posix(),
                'benchmark_sha256': protocol['benchmark_sha256'][split], 'runtime_sha256': runtime_fingerprint(),
                'source_sha256': source_fingerprint(), 'config_sha256': sha256_file(config_path),
                'vector_fingerprint': protocol['vector_fingerprint'], 'python_version': platform.python_version(),
                'embedding_model': settings.embedding_model, 'dimensions': settings.dimensions,
                'embedding_base_url': settings.embedding_base_url, 'chat_base_url': settings.chat_base_url,
                'chat_model': settings.chat_model if kind == 'agent' else None,
                'system_prompt': SYSTEM_PROMPT if kind == 'agent' else None, 'filters': 'none unless explicitly in input'}
    write_json(output / 'manifest.json', manifest)
    traces = []
    with (output / 'traces.jsonl').open('x', encoding='utf-8', newline='\n') as handle:
        for case in cases:
            start = time.perf_counter()
            runtime = runtime_query(case)
            original = None
            try:
                original = retriever.search(runtime['question'], filters=runtime['filters'])
                if kind == 'agent':
                    trace = run_agent(runtime, controller, retriever, config)
                else:
                    trace = {'status': 'retrieved', 'events': [], 'citations': [],
                             'context': pack_context([original], config['budget_chars']),
                             'elapsed_ms': (time.perf_counter() - start) * 1000}
                trace['original_top3'] = original['original_top3']
            except Exception as exc:
                trace = {'status': 'error', 'events': [], 'citations': [],
                         'context': pack_context([], config['budget_chars']), 'original_top3': [],
                         'error': {'type': type(exc).__name__, 'message': str(exc)},
                         'elapsed_ms': (time.perf_counter() - start) * 1000}
            trace.update({'query_id': case['id'], 'original_question': runtime['question'],
                          'original_search': original, 'query_total_including_evaluation_ms': (time.perf_counter() - start) * 1000})
            traces.append(trace)
            handle.write(json.dumps(trace, ensure_ascii=False) + '\n')
            handle.flush()
            print(json.dumps({'config': config['id'], 'split': split, 'query_id': case['id'], 'status': trace['status']}), flush=True)
    summary = {'completed_at': timestamp(), 'kind': kind, 'split': split, 'config_id': config['id'],
               'retrieval_context': retrieval_summary(cases, traces), 'usage': usage_summary(traces),
               'agent_reviews_pending': kind == 'agent'}
    write_json(output / 'summary.json', summary)
    if kind == 'agent':
        write_json(output / 'reviews.json', review_template(cases, traces))
    return output


def summarize_reviewed_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    cases_path = Path(manifest['cases_path'])
    if sha256_file(cases_path) != manifest['benchmark_sha256']:
        raise ValueError('benchmark changed since this run')
    cases = json.loads(cases_path.read_text(encoding='utf-8'))
    traces = load_jsonl(run_dir / 'traces.jsonl')
    reviews = json.loads((run_dir / 'reviews.json').read_text(encoding='utf-8'))
    reviewed = agent_summary(cases, traces, reviews)
    return {'run': run_dir.as_posix(), 'config_id': manifest['config']['id'], 'split': manifest['split'],
            'agent': reviewed, 'retrieval_context': retrieval_summary(cases, traces),
            'usage': usage_summary(traces), 'median_agent_ms': statistics.median(t['elapsed_ms'] for t in traces)}


def freeze_selection(run_dirs: list[Path]) -> dict:
    """No model calls. Selection is legal only with completed source-reviewed dev."""
    protocol = json.loads(Path('artifacts/protocol.json').read_text(encoding='utf-8'))
    if not Path('artifacts/dev_error_analysis.md').exists():
        raise ValueError('record at least two actual dev errors and improvement hypotheses first')
    records, manifests = [], []
    for run in run_dirs:
        manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['split'] != 'dev' or manifest['kind'] != 'agent':
            raise ValueError('only real agent dev runs may select the final pipeline')
        if manifest.get('source_sha256') != source_fingerprint():
            raise ValueError('runtime source changed since dev; re-evaluate affected configs')
        if manifest['config_sha256'] != sha256_file(Path(manifest['config_path'])):
            raise ValueError('config changed since its dev run')
        if manifest['vector_fingerprint'] != protocol['vector_fingerprint']:
            raise ValueError('dev runs used different index snapshots')
        reviewed = summarize_reviewed_run(run)
        if reviewed['agent']['completion']['rate'] != 1.0:
            raise ValueError('all dev cases must complete before configuration selection')
        records.append(reviewed)
        manifests.append(manifest)
    ids = [m['config']['id'] for m in manifests]
    if len(ids) != len(set(ids)) or not {'D0', 'L0', 'H0', 'H1', 'H2'} <= set(ids):
        raise ValueError('supply one complete dev agent run for D0/L0/H0/H1/H2')
    for field in ('chat_model', 'embedding_model', 'dimensions', 'chat_base_url', 'embedding_base_url'):
        if len({m[field] for m in manifests}) != 1:
            raise ValueError('comparison requires the same models and endpoints')
    use_cost = all(r['usage']['total_cost_usd'] is not None for r in records)
    def key(record):
        agent = record['agent']; metrics = record['retrieval_context']
        abstention = agent['abstention']['n'] / agent['abstention']['N']
        return (-agent['grounded_answer']['mean'], -abstention,
                -metrics['context_evidence']['rate'], -metrics['hit3']['rate'],
                record['usage']['total_cost_usd'] if use_cost else 0,
                record['median_agent_ms'], record['config_id'])
    selected = min(records, key=key)
    allowed = [m for m in manifests if m['config']['id'] in ('D0', selected['config_id'])]
    selection = {'selected_at': timestamp(), 'selected_config_id': selected['config_id'], 'baseline_id': 'D0',
                 'selected_run': selected['run'], 'selection_order': protocol['selection_order'],
                 'cost_tiebreak_used': use_cost, 'runtime_sha256': runtime_fingerprint(),
                 'vector_fingerprint': protocol['vector_fingerprint'],
                 'benchmark_sha256': protocol['benchmark_sha256'],
                 'allowed_config_sha256': [m['config_sha256'] for m in allowed],
                 **{field: manifests[0][field] for field in ('chat_model', 'embedding_model', 'dimensions',
                                                            'chat_base_url', 'embedding_base_url')},
                 'dev_results': records}
    write_json(Path('artifacts/selection.json'), selection)
    return selection

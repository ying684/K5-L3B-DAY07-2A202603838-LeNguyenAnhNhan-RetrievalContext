"""Run from repo root: python -X utf8 -m src.run --help."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .agent import run_agent
from .config import Settings, load_config
from .context import pack_context
from .controller import Controller
from .day6 import cached_vector, inspect_snapshot, load_chunks, write_json
from .embeddings import Embedder
from .experiments import benchmark, freeze_selection, summarize_reviewed_run
from .retrieval import Retriever
from .store import PgVectorStore


def run_id() -> str:
    return datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d_%H%M%S_%f')


def build_retriever(settings: Settings, config: dict, allow_api: bool) -> Retriever:
    store = PgVectorStore(settings.database_url, settings.dimensions, settings.strategy) if config['method'] != 'lexical' else None
    embedder = Embedder(settings, allow_api=allow_api) if config['method'] != 'lexical' else None
    return Retriever(load_chunks(settings.data_dir), config, store=store, embedder=embedder)


def main() -> int:
    parser = argparse.ArgumentParser(description='Day 7 RAG: real adapters, separate evaluation')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare', help='Validate copied Day 6 snapshot and cached vectors')
    sub.add_parser('verify-db', help='Read-only DB comparison against the copied snapshot')
    sub.add_parser('readiness', help='Show configuration presence, never credential values')
    smoke = sub.add_parser('smoke-api', help='Verify real query embedding and tool-calling model outside benchmark')
    smoke.add_argument('--allow-api', action='store_true')
    bench = sub.add_parser('benchmark')
    bench.add_argument('--config', type=Path, required=True)
    bench.add_argument('--split', choices=('dev', 'test'), required=True)
    bench.add_argument('--kind', choices=('retrieval', 'agent'), required=True)
    bench.add_argument('--allow-api', action='store_true')
    summary = sub.add_parser('summarize')
    summary.add_argument('--run', required=True, type=Path)
    select = sub.add_parser('select', help='Freeze pipeline after complete source-reviewed dev comparison')
    select.add_argument('--runs', nargs='+', type=Path, required=True)
    for name in ('search', 'ask'):
        command = sub.add_parser(name)
        command.add_argument('--question', required=True)
        command.add_argument('--config', required=True, type=Path)
        command.add_argument('--allow-api', action='store_true')
        command.add_argument('--chapter', type=int)
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.command == 'select':
        selection = freeze_selection(args.runs)
        print(json.dumps({'selected_config_id': selection['selected_config_id'], 'test_unlocked': True}))
        return 0
    if args.command == 'summarize':
        result = summarize_reviewed_run(args.run)
        write_json(args.run / 'reviewed_summary.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == 'benchmark':
        path = benchmark(settings, args.config, split=args.split, kind=args.kind,
                         allow_api=args.allow_api, build_retriever=build_retriever)
        print(json.dumps({'output': path.as_posix()}))
        return 0
    if args.command == 'smoke-api':
        question = 'Tài liệu mô tả khái niệm completion và hệ thống khuyến nghị như thế nào?'
        config = load_config(Path('configs/hybrid.json'))
        retriever = build_retriever(settings, config, args.allow_api)
        verification = {'checked_at': datetime.now(timezone(timedelta(hours=7))).isoformat(),
                        'embedding_model': settings.embedding_model, 'dimensions': settings.dimensions,
                        'embedding_base_url': settings.embedding_base_url, 'chat_base_url': settings.chat_base_url,
                        'chat_model': settings.chat_model, 'embedding_verified': False,
                        'chat_verified': False, 'unscored_smoke': True, 'stage': 'embedding', 'error': None}
        try:
            retriever.embedder.embed_query(question)
            verification['embedding_verified'] = True
            verification['stage'] = 'chat'
            result = run_agent({'question': question}, Controller(settings, allow_api=args.allow_api), retriever, config)
            verification.update({'agent_trace': result, 'error': result.get('error'),
                                 'chat_verified': result['status'] in ('answered', 'abstain') and bool(result['model_calls'])})
        except Exception as exc:
            verification['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        verification['embedding_usage'] = retriever.embedder.usage
        path = Path('results/preflight') / ('api_' + run_id() + '.json')
        write_json(path, verification)
        if verification['chat_verified']:
            write_json(Path('artifacts/api_verification.json'), verification, overwrite=True)
        print(json.dumps({'output': path.as_posix(), 'embedding_verified': verification['embedding_verified'],
                          'chat_verified': verification['chat_verified'], 'error': verification['error']}, ensure_ascii=False))
        return 0 if verification['chat_verified'] else 1
    if args.command == 'readiness':
        print(json.dumps(settings.readiness(), indent=2))
        return 0
    if args.command == 'prepare':
        manifest = inspect_snapshot(settings.data_dir, settings.embedding_model, settings.dimensions)
        path = Path('results/preflight') / ('snapshot_' + run_id() + '.json')
        write_json(path, manifest)
        write_json(Path('artifacts/data_manifest.json'), manifest, overwrite=True)
        print(json.dumps({'output': path.as_posix(), 'chunks': manifest['chunks'],
                          'pages': manifest['pages'], 'vectors': manifest['valid_vectors']}, ensure_ascii=False))
        return 0
    if args.command == 'verify-db':
        chunks = load_chunks(settings.data_dir)
        vectors = {c['id']: cached_vector(settings.data_dir, c['text'], settings.embedding_model, settings.dimensions) for c in chunks}
        result = PgVectorStore(settings.database_url, settings.dimensions, settings.strategy).verify(chunks, vectors)
        path = Path('results/preflight') / ('db_' + run_id() + '.json')
        write_json(path, result)
        write_json(Path('artifacts/db_verification.json'), result, overwrite=True)
        manifest_path = Path('artifacts/data_manifest.json')
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest.update({'db_verified': True, 'db_verification_path': path.as_posix()})
            write_json(manifest_path, manifest, overwrite=True)
        print(json.dumps({'output': path.as_posix(), **result}))
        return 0
    config = load_config(args.config)
    retriever = build_retriever(settings, config, args.allow_api)
    query = {'question': args.question, 'filters': {'chapter': args.chapter} if args.chapter else {}}
    if args.command == 'search':
        search = retriever.search(args.question, filters=query['filters'])
        result = {**search, 'context': pack_context([search], config['budget_chars']), 'unscored_smoke': True}
    else:
        result = run_agent(query, Controller(settings, allow_api=args.allow_api), retriever, config)
        result['unscored_smoke'] = True
    path = Path('results/preflight') / (args.command + '_' + config['id'] + '_' + run_id() + '.json')
    write_json(path, result)
    print(json.dumps({'output': path.as_posix(), 'status': result.get('status', 'retrieved'),
                      'context_ids': result['context']['ids'], 'error': result.get('error')}, ensure_ascii=False))
    return 1 if result.get('status') == 'error' else 0


if __name__ == '__main__':
    raise SystemExit(main())

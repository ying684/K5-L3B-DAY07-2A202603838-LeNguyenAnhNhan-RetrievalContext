"""Read the copied Day 6 snapshot and prove every vector's provenance."""
from __future__ import annotations

import hashlib
import json
import math
import platform
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .contracts import normalize_chunk, validate_vector


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path.name}:{line_number}: invalid JSON') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path.name}:{line_number}: expected object')
            rows.append(row)
    if not rows:
        raise ValueError(f'{path.name}: no rows')
    return rows


def load_chunks(data_dir: Path) -> list[dict[str, Any]]:
    chunks = [normalize_chunk(row) for row in load_jsonl(data_dir / 'prepared/chunks_B.jsonl')]
    if len({c['id'] for c in chunks}) != len(chunks):
        raise ValueError('duplicate chunk IDs')
    return chunks


def cache_key(text: str, model: str, dimensions: int) -> str:
    return hashlib.sha256(f'{model}|{dimensions}|{text}'.encode('utf-8')).hexdigest()


def cached_vector(data_dir: Path, text: str, model: str, dimensions: int) -> list[float]:
    path = data_dir / 'embeddings' / (cache_key(text, model, dimensions) + '.json')
    if not path.exists():
        raise FileNotFoundError(f'missing Day 6 embedding cache: {path.name}')
    return validate_vector(json.loads(path.read_text(encoding='utf-8')), dimensions)


def inspect_snapshot(data_dir: Path, model: str = 'baai/bge-m3', dimensions: int = 1024) -> dict:
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
    plan = json.loads((data_dir / 'prepared/plan.json').read_text(encoding='utf-8'))
    snapshot = plan['snapshot']
    if snapshot['model'] != model or snapshot['dimensions'] != dimensions:
        raise ValueError('configured embedding model/dimensions differ from Day 6 snapshot')
    chunks = load_chunks(data_dir)
    pages = load_jsonl(data_dir / 'prepared/pages.jsonl')
    if len({(p['document_id'], p['page']) for p in pages}) != len(pages):
        raise ValueError('duplicate source pages')
    page_map = {(p['document_id'], p['page']): p for p in pages}
    norms, vector_hashes = [], []
    for chunk in chunks:
        page = page_map.get((chunk['source_id'], chunk['page']))
        if page is None or chunk['text'] not in page['text']:
            raise ValueError(f"{chunk['id']}: text not found on its source page")
        if page['source'] != chunk['source']:
            raise ValueError(f"{chunk['id']}: inconsistent source path")
        vector = cached_vector(data_dir, chunk['text'], model, dimensions)
        norms.append(math.sqrt(sum(v * v for v in vector)))
        vector_hashes.append({'id': chunk['id'], 'vector_sha256': hashlib.sha256(
            json.dumps(vector, separators=(',', ':')).encode()).hexdigest()})
    pdf = data_dir / 'raw/book_ML_color.pdf'
    pdf_hash = sha256_file(pdf)
    if pdf_hash != manifest['sha256'] or pdf_hash != snapshot['corpus_sha256']:
        raise ValueError('PDF does not match Day 6 manifest and snapshot')
    expected_chunks = plan['stats']['B']['chunk_count']
    if len(chunks) != expected_chunks or len(pages) != manifest['counted_content_pages']:
        raise ValueError('chunk/page counts do not match Day 6 provenance')
    tracked_inputs = ['manifest.json', 'prepared/chunks_B.jsonl', 'prepared/pages.jsonl',
                      'prepared/plan.json', 'provenance/dev_selection.json', 'provenance/protocol.json']
    input_hashes = {p: sha256_file(data_dir / p) for p in tracked_inputs}
    vector_fingerprint = hashlib.sha256(json.dumps(vector_hashes, sort_keys=True).encode()).hexdigest()
    return {'checked_at': datetime.now(timezone(timedelta(hours=7))).isoformat(),
            'python_version': platform.python_version(), 'source_id': manifest['source_id'],
            'snapshot': manifest['snapshot'], 'chunk_strategy': 'B', 'chunks': len(chunks),
            'pages': len(pages), 'physical_page_range': [min(p['page'] for p in pages), max(p['page'] for p in pages)],
            'model': model, 'dimensions': dimensions, 'valid_vectors': len(norms),
            'vector_norm_range': [min(norms), max(norms)], 'vector_fingerprint': vector_fingerprint,
            'pdf_sha256': pdf_hash, 'input_hashes': input_hashes,
            'source_path_mapping': {manifest['source_path_or_url']: pdf.as_posix()},
            'permission_record': {key: manifest.get(key) for key in ('license_or_permission', 'permission_evidence')},
            'db_verified': False, 'query_embedding_verified': False, 'chat_model_verified': False}


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if overwrite else 'x'
    with path.open(mode, encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')

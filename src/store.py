"""Read-only adapter to the original Day 6 pgvector table."""
from __future__ import annotations

import json
import math

import psycopg
from psycopg.types.json import Jsonb

from .contracts import normalize_chunk, validate_filters, validate_vector


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int = 1024, strategy: str = 'B'):
        if not database_url:
            raise ValueError('DATABASE_URL is missing')
        self.database_url, self.dimensions, self.strategy = database_url, dimensions, strategy

    def connect(self):
        # Every transaction is read-only, including verification calls.
        connection = psycopg.connect(self.database_url, connect_timeout=5,
                                     options='-c default_transaction_read_only=on')
        return connection

    def search(self, vector: list[float], *, filters: dict | None = None, top_k: int = 10) -> list[dict]:
        filters = validate_filters(filters)
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError('top_k must be between 1 and 100')
        literal = json.dumps(validate_vector(vector, self.dimensions), separators=(',', ':'))
        with self.connect() as connection:
            rows = connection.execute('''
                SELECT chunk_id, document_id, source, page, text, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM day6_chunks WHERE strategy=%s AND metadata @> %s::jsonb
                ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s
                ''', (literal, self.strategy, Jsonb(filters), literal, top_k)).fetchall()
        out = []
        for rank, row in enumerate(rows, 1):
            chunk = normalize_chunk(dict(zip(('chunk_id', 'document_id', 'source', 'page', 'text', 'metadata'), row[:6])))
            out.append({**chunk, 'score': float(row[6]), 'rank': rank, 'method': 'dense'})
        return out

    def verify(self, chunks: list[dict], vectors: dict[str, list[float]]) -> dict:
        with self.connect() as connection:
            readonly = connection.execute('SHOW transaction_read_only').fetchone()[0]
            pg = connection.execute('SHOW server_version').fetchone()[0]
            ext = connection.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
            rows = connection.execute('''SELECT chunk_id, document_id, source, page, text,
                                      metadata, embedding::text FROM day6_chunks
                                      WHERE strategy=%s ORDER BY chunk_id''', (self.strategy,)).fetchall()
        expected = {chunk['id']: chunk for chunk in chunks}
        if {row[0] for row in rows} != set(expected):
            raise ValueError('DB chunk IDs differ from copied Day 6 B snapshot')
        max_delta = 0.0
        for cid, source_id, source, page, text, metadata, raw_vector in rows:
            chunk = expected[cid]
            if (source_id, source, page, text) != (chunk['source_id'], chunk['source'], chunk['page'], chunk['text']):
                raise ValueError(f'{cid}: DB content differs from copied snapshot')
            if any(metadata.get(key) != value for key, value in chunk['metadata'].items()):
                raise ValueError(f'{cid}: DB metadata differs from copied snapshot')
            vector = validate_vector(json.loads(raw_vector), self.dimensions)
            delta = max(abs(a - b) for a, b in zip(vector, vectors[cid]))
            max_delta = max(max_delta, delta)
            if not math.isfinite(delta) or delta > 1e-6:
                raise ValueError(f'{cid}: DB vector differs from cached embedding')
        if readonly != 'on' or not ext:
            raise ValueError('read-only pgvector verification failed')
        return {'read_only': True, 'postgres_version': pg, 'pgvector_version': ext[0],
                'strategy': self.strategy, 'matched_chunks': len(rows),
                'matched_vectors': len(rows), 'max_vector_absolute_difference': max_delta}

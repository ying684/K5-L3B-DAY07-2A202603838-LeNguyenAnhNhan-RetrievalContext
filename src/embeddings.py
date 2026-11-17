"""Real Day 6-compatible embeddings, with separate query cache and usage."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import Settings
from .contracts import nonempty, validate_vector
from .day6 import cache_key
from .http_client import post_json


class Embedder:
    def __init__(self, settings: Settings, *, allow_api: bool = False,
                 cache_dir: Path = Path('.cache/query_embeddings')):
        self.settings, self.allow_api, self.cache_dir = settings, allow_api, cache_dir
        self.usage: list[dict] = []

    def embed_query(self, text: str) -> list[float]:
        nonempty(text, 'embedding input')
        settings = self.settings
        key = cache_key(text, settings.embedding_model, settings.dimensions)
        # Reuse original real query caches when the exact input already exists.
        for path in (settings.data_dir / 'embeddings' / (key + '.json'), self.cache_dir / (key + '.json')):
            if path.exists():
                vector = validate_vector(json.loads(path.read_text(encoding='utf-8')), settings.dimensions)
                self.usage.append({'model': settings.embedding_model, 'cache_hit': True,
                                   'elapsed_ms': 0, 'service_cost_usd': 0, 'usage': {}})
                return vector
        if not self.allow_api:
            raise RuntimeError('query embedding is not cached; pass --allow-api for a real request')
        started = time.perf_counter()
        data = post_json(settings.embedding_base_url + '/embeddings',
                         {'model': settings.embedding_model, 'input': [text], 'encoding_format': 'float',
                          'provider': {'allow_fallbacks': False}}, settings.embedding_key,
                         timeout=settings.timeout, allow_external=settings.allow_external)
        rows = data.get('data', [])
        if len(rows) != 1 or rows[0].get('index') != 0:
            raise ValueError('embedding API returned incomplete/invalid indices')
        accepted = {settings.embedding_model.casefold()}
        if settings.embedding_model.casefold() == 'baai/bge-m3':
            accepted.add('parasail-bge-m3')
        returned = data.get('model')
        if not isinstance(returned, str) or returned.casefold() not in accepted:
            raise ValueError('embedding response model differs from configured Day 6 model')
        vector = validate_vector(rows[0]['embedding'], settings.dimensions)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / (key + '.json')).write_text(json.dumps(vector), encoding='utf-8')
        usage = data.get('usage') or {}
        self.usage.append({'model': settings.embedding_model, 'returned_model': returned,
                           'provider': data.get('provider'), 'request_id': data.get('id'),
                           'cache_hit': False, 'usage': usage,
                           'service_cost_usd': usage.get('cost'),
                           'elapsed_ms': (time.perf_counter() - started) * 1000})
        return vector

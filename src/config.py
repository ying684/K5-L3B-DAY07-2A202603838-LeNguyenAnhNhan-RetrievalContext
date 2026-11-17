"""Configuration without exposing credential values in reports or errors."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def load_environment(path: Path = Path('.env')) -> None:
    if path.exists():
        from dotenv import load_dotenv
        load_dotenv(path, override=False, interpolate=False)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_url: str
    read_only: bool
    strategy: str
    embedding_model: str
    dimensions: int
    embedding_base_url: str
    embedding_key: str
    chat_base_url: str
    chat_model: str
    chat_key: str
    allow_external: bool
    timeout: int

    @classmethod
    def from_env(cls) -> 'Settings':
        load_environment()
        return cls(Path(os.getenv('DAY6_DATA_DIR', 'data/day6')),
                   os.getenv('DATABASE_URL', ''), os.getenv('DAY6_READ_ONLY', '1') != '0',
                   os.getenv('DAY6_STRATEGY', 'B'), os.getenv('EMBEDDING_MODEL', 'baai/bge-m3'),
                   int(os.getenv('EMBEDDING_DIMENSIONS', '1024')),
                   os.getenv('EMBEDDING_BASE_URL', 'https://openrouter.ai/api/v1').rstrip('/'),
                   os.getenv('OPENROUTER_API_KEY', ''),
                   os.getenv('CHAT_BASE_URL', 'https://openrouter.ai/api/v1').rstrip('/'),
                   os.getenv('CHAT_MODEL', ''),
                   os.getenv('CHAT_API_KEY') or os.getenv('OPENROUTER_API_KEY', ''),
                   os.getenv('ALLOW_EXTERNAL_PROCESSING', '').lower() == 'true',
                   int(os.getenv('REQUEST_TIMEOUT_SECONDS', '60')))

    def readiness(self) -> dict[str, bool]:
        return {'database_configured': bool(self.database_url),
                'embedding_key_configured': bool(self.embedding_key),
                'chat_model_configured': bool(self.chat_model),
                'chat_key_configured': bool(self.chat_key),
                'external_processing_allowed': self.allow_external}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding='utf-8'))
    if config.get('method') not in ('dense', 'lexical', 'hybrid'):
        raise ValueError('method must be dense, lexical or hybrid')
    for name in ('candidate_k', 'return_k', 'budget_chars', 'max_searches', 'rrf_constant'):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if config['candidate_k'] < config['return_k'] or config['candidate_k'] > 100:
        raise ValueError('candidate_k must be between return_k and 100')
    return config

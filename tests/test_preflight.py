import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.run import main


class PreflightTests(unittest.TestCase):
    def test_auth_failure_is_saved_and_never_unlocks_benchmark(self):
        settings = SimpleNamespace(embedding_model='baai/bge-m3', dimensions=1024,
                                   embedding_base_url='https://openrouter.ai/api/v1',
                                   chat_base_url='https://openrouter.ai/api/v1', chat_model='openai/gpt-4.1-mini')
        embedder = Mock(usage=[])
        embedder.embed_query.side_effect = RuntimeError('provider HTTP 401; no automatic retry')
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)
                with patch('sys.argv', ['run', 'smoke-api', '--allow-api']), \
                     patch('src.run.Settings.from_env', return_value=settings), \
                     patch('src.run.load_config', return_value={}), \
                     patch('src.run.build_retriever', return_value=SimpleNamespace(embedder=embedder)), \
                     patch('src.run.Controller') as controller, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 1)
                    controller.assert_not_called()
                logs = list(Path('results/preflight').glob('api_*.json'))
                self.assertEqual(len(logs), 1)
                record = json.loads(logs[0].read_text(encoding='utf-8'))
                self.assertFalse(record['embedding_verified'])
                self.assertFalse(record['chat_verified'])
                self.assertEqual(record['stage'], 'embedding')
                self.assertIn('401', record['error']['message'])
                self.assertFalse(Path('artifacts/api_verification.json').exists())
                self.assertEqual(embedder.embed_query.call_count, 1)
            finally:
                os.chdir(original)

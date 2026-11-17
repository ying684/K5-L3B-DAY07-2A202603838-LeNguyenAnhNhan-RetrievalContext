import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.config import Settings
from src.controller import Controller
from src.embeddings import Embedder
from src.http_client import post_json


def settings(data_dir):
    return Settings(Path(data_dir), '', True, 'B', 'baai/bge-m3', 2,
                    'https://openrouter.ai/api/v1', 'test-only-dummy',
                    'https://openrouter.ai/api/v1', 'openai/gpt-4.1-mini',
                    'test-only-dummy', True, 5)


class HttpAdapterTests(unittest.TestCase):
    def test_external_permission_and_missing_key_fail_before_transport(self):
        with patch('src.http_client.urlopen') as transport:
            for key, allowed in (('test-only-dummy', False), ('', True)):
                with self.assertRaises(RuntimeError):
                    post_json('https://example.com/api', {}, key, timeout=5, allow_external=allowed)
            transport.assert_not_called()

    def test_timeout_is_reported_without_retry_or_secret(self):
        with patch('src.http_client.urlopen', side_effect=TimeoutError('transport detail')) as transport:
            with self.assertRaisesRegex(RuntimeError, 'network/timeout') as error:
                post_json('https://example.com/api', {}, 'test-only-dummy', timeout=5, allow_external=True)
            self.assertEqual(transport.call_count, 1)
            self.assertNotIn('test-only-dummy', str(error.exception))

    def test_wrong_embedding_model_not_cached_and_api_flag_required(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / 'query_cache'
            embedder = Embedder(settings(temp), cache_dir=cache)
            with patch('src.embeddings.post_json') as provider:
                with self.assertRaises(RuntimeError):
                    embedder.embed_query('uncached')
                provider.assert_not_called()
            embedder.allow_api = True
            with patch('src.embeddings.post_json', return_value={
                    'model': 'different-model', 'data': [{'index': 0, 'embedding': [1, 0]}]}):
                with self.assertRaises(ValueError):
                    embedder.embed_query('uncached')
            self.assertFalse(cache.exists())

    def test_controller_forces_initial_search_and_removes_search_at_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            controller = Controller(settings(temp), allow_api=True)
            reply = {'choices': [{'message': {'role': 'assistant', 'tool_calls': []}}]}
            with patch('src.controller.post_json', return_value=reply) as provider:
                controller.decide([{'role': 'user', 'content': 'q'}], first=True, can_search=True)
                first = provider.call_args.args[1]
                self.assertEqual(first['tool_choice']['function']['name'], 'search_corpus')
                controller.decide([{'role': 'user', 'content': 'q'}], first=False, can_search=False)
                final = provider.call_args.args[1]
                self.assertNotIn('search_corpus', [tool['function']['name'] for tool in final['tools']])
                self.assertNotIn('test-only-dummy', json.dumps(final))
                for tool in final['tools']:
                    self.assertTrue(tool['function']['strict'])
                    schema = tool['function']['parameters']
                    self.assertEqual(set(schema['required']), set(schema['properties']))
                    self.assertFalse(schema['additionalProperties'])

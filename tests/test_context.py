import unittest

from src.context import pack_context
from tests_helper import sample_chunk


class ContextTests(unittest.TestCase):
    def test_repeated_query_does_not_inflate_old_chunk_rank(self):
        a, b = sample_chunk('a'), sample_chunk('b')
        result = pack_context([{'query': 'old', 'results': [b]}, {'query': 'old', 'results': [b]},
                               {'query': 'new', 'results': [a]}], 1000)
        self.assertEqual(result['ids'], ['a', 'b'])

    def test_oversized_chunk_does_not_block_new_evidence(self):
        result = pack_context([{'query': 'one', 'results': [sample_chunk('a', 'x' * 500)]},
                               {'query': 'two', 'results': [sample_chunk('b', 'new evidence')]}], 100)
        self.assertEqual(result['ids'], ['b'])
        self.assertLessEqual(result['chars'], 100)

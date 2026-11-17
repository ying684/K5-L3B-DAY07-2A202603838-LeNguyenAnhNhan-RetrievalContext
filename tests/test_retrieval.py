import unittest

from src.retrieval import Retriever, rrf


def chunk(cid, text='term', chapter=5, score=0.5):
    return {'id': cid, 'text': text, 'source_id': 'doc', 'source': 'doc.pdf',
            'page': 1, 'metadata': {'chapter': chapter}, 'score': score}


class RetrievalTests(unittest.TestCase):
    def test_rrf_uses_rank_not_incompatible_scores(self):
        a, b, c = chunk('a', score=-50), chunk('b', score=0.99), chunk('c', score=9999)
        results = rrf([[a, b], [c, b]])
        self.assertEqual(results[0]['id'], 'b')
        self.assertEqual([r['id'] for r in results[1:]], ['a', 'c'])
        self.assertEqual(results[0]['component_ranks']['0']['rank'], 2)

    def test_rrf_rejects_duplicate_ids_and_conflicting_provenance(self):
        with self.assertRaises(ValueError):
            rrf([[chunk('a'), chunk('a')]])
        with self.assertRaises(ValueError):
            rrf([[chunk('a')], [{**chunk('a'), 'page': 2}]])

    def test_lexical_filter_is_applied_before_cutoff(self):
        config = {'method': 'lexical', 'candidate_k': 1, 'return_k': 1, 'rrf_constant': 60}
        retriever = Retriever([chunk('a', 'term', 5), chunk('b', 'term other', 6)], config)
        results = retriever.search('term', filters={'chapter': 6})['results']
        self.assertEqual([r['id'] for r in results], ['b'])

    def test_dense_does_not_fallback_to_lexical(self):
        config = {'method': 'dense', 'candidate_k': 3, 'return_k': 3, 'rrf_constant': 60}
        with self.assertRaises(RuntimeError):
            Retriever([chunk('a')], config).search('term')

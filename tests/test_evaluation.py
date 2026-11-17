import unittest

from src.evaluation import agent_summary, fraction, retrieval_summary, review_template, validate_benchmark
from tests_helper import sample_chunk


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.case = {'id': 'a', 'split': 'dev', 'question': 'q', 'answerable': True,
                     'expected_answer': '7 ngày', 'gold': [{'source_id': 'doc', 'page': 1, 'evidence': '7 ngày'}]}
        self.unanswerable = {'id': 'u', 'split': 'dev', 'question': 'other', 'answerable': False,
                             'expected_behavior': 'search_and_abstain', 'absence_reason': 'source reviewed'}
        self.trace = {'query_id': 'a', 'status': 'answered', 'answer': '7 ngày [c1]', 'citations': ['c1'],
                      'events': [{'event': 'search'}], 'original_top3': [sample_chunk()],
                      'context': {'ids': ['c1'], 'chunks': [sample_chunk()]}, 'elapsed_ms': 10}

    def test_unanswerable_not_in_retrieval_denominator_and_errors_stay_visible(self):
        traces = [self.trace, {**self.trace, 'query_id': 'u', 'status': 'error'}]
        summary = retrieval_summary([self.case, self.unanswerable], traces)
        self.assertEqual(summary['hit3'], {'n': 1, 'N': 1, 'rate': 1.0})
        self.assertEqual(summary['completion']['rate'], 0.5)
        self.assertIsNone(fraction(0, 0)['rate'])

    def test_valid_id_does_not_fill_semantic_review(self):
        templates = review_template([self.case], [self.trace])
        self.assertIsNone(templates[0]['answer_grade'])
        self.assertEqual(templates[0]['citation_support'], [None])
        with self.assertRaises(ValueError):
            agent_summary([self.case], [self.trace], templates)

    def test_gold_wrong_page_and_insufficient_split_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_benchmark([{**self.case, 'gold': [{**self.case['gold'][0], 'page': 2}]}],
                               [sample_chunk()], split='dev', require_minimum=False)
        with self.assertRaises(ValueError):
            validate_benchmark([self.case], [sample_chunk()], split='dev')

    def test_semantic_support_false_even_when_citation_id_exists(self):
        review = {'query_id': 'a', 'reviewer': 'test source reviewer', 'rationale': 'claim unsupported',
                  'answer_grade': 0, 'context_grade': 2, 'citation_support': [False],
                  'citation_reviews': [{'id': 'c1', 'supports_claim': False, 'rationale': 'wrong claim'}]}
        result = agent_summary([self.case], [self.trace], [review])
        self.assertEqual(result['citation_validity'], {'n': 0, 'N': 1})

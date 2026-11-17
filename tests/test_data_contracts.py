import json
import math
import tempfile
import unittest
from pathlib import Path

from src.contracts import normalize_chunk, runtime_query, validate_cases, validate_filters, validate_vector
from src.day6 import cache_key, load_jsonl


class DataContractTests(unittest.TestCase):
    def test_runtime_input_is_whitelisted_and_filter_has_no_gold(self):
        case = {'question': 'q', 'filter': {'chapter': 5}, 'gold': ['secret'],
                'expected_answer': 'secret', 'answerable': True, 'absence_reason': 'secret'}
        self.assertEqual(runtime_query(case), {'question': 'q', 'filters': {'chapter': 5}})
        with self.assertRaises(ValueError):
            runtime_query({'question': 'q', 'filter': {'source_id': 'gold-source'}})

    def test_jsonl_reads_unicode_and_reports_malformed_line(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'rows.jsonl'
            path.write_text(json.dumps({'text': 'bằng chứng'}, ensure_ascii=False) + '\n\n', encoding='utf-8')
            self.assertEqual(load_jsonl(path), [{'text': 'bằng chứng'}])
            path.write_text('{}\nnot-json', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, ':2:'):
                load_jsonl(path)

    def test_day6_mapping_preserves_text_and_metadata(self):
        raw = {'chunk_id': 'a', 'document_id': 'doc', 'source': 'x.pdf', 'page': 78,
               'text': ' exact text ', 'metadata': {'printed_page': 66}}
        row = normalize_chunk(raw)
        self.assertEqual((row['id'], row['source_id'], row['text']), ('a', 'doc', ' exact text '))
        self.assertEqual(row['metadata']['printed_page'], 66)
        self.assertNotEqual(cache_key('x', 'm', 2), cache_key('x ', 'm', 2))

    def test_rejects_invalid_vectors_and_filter_types(self):
        self.assertEqual(validate_vector([1, 0], 2), [1.0, 0.0])
        for vector in ([0, 0], [1], [True, 1], [1, math.nan], [1, math.inf]):
            with self.assertRaises(ValueError):
                validate_vector(vector, 2)
        with self.assertRaises(ValueError):
            validate_filters({'chapter': '5'})

    def test_unanswerable_requires_reason_but_not_gold(self):
        case = {'id': 'u', 'question': 'q', 'split': 'dev', 'answerable': False,
                'expected_behavior': 'search_and_abstain', 'absence_reason': 'reviewed corpus'}
        self.assertEqual(validate_cases([case], split='dev'), [case])
        with self.assertRaises(ValueError):
            validate_cases([case, case])
        with self.assertRaises(ValueError):
            validate_cases([{**case, 'absence_reason': ''}])


if __name__ == '__main__':
    unittest.main()

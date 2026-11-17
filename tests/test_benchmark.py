import json
import tempfile
import unittest
from pathlib import Path

from src.day6 import load_chunks
from src.evaluation import validate_benchmark
from src.experiments import check_protocol


class BenchmarkTests(unittest.TestCase):
    def test_dev_contract_matches_real_copied_evidence(self):
        if not Path('data/day6/prepared/chunks_B.jsonl').exists():
            self.skipTest('local Day 6 data not supplied')
        dev = json.loads(Path('data/benchmark/dev.json').read_text(encoding='utf-8'))
        result = validate_benchmark(dev, load_chunks(Path('data/day6')), split='dev')
        self.assertEqual((result['answerable'], result['unanswerable']), (15, 2))

    def test_test_is_locked_before_dev_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, 'test is locked'):
                check_protocol('test', Path('configs/dense.json'),
                               selection_path=Path(temp)/'missing-selection.json')

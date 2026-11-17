"""Full-page visual index and deterministic lexical/reference retrieval."""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

INDEX = Path('data/bonus/index')
SOURCE = Path('data/day6/raw/book_ML_color.pdf')
EXPECTED_PDF = 'f40e8c07dbdba41b1b2e2fbec9e5c7432752fcc288aa706fa33358a84d23b543'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFD', text.casefold()).replace('đ', 'd')
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')


def tokens(text: str) -> list[str]:
    return re.findall(r'[a-z0-9]+', normalize(text))


def references(text: str) -> set[str]:
    aliases = {'figure':'hinh','table':'bang'}
    return {f'{aliases.get(kind,kind)}:{number}' for kind, number in
            re.findall(r'\b(hinh|bang|figure|table)\s*(\d+\.\d+)(?:[a-z])?\b', normalize(text))}


def build_index(source: Path = SOURCE, target: Path = INDEX) -> dict:
    import pymupdf
    if sha(source) != EXPECTED_PDF:
        raise ValueError('source differs from the documented book snapshot')
    target.mkdir(parents=True, exist_ok=True)
    if (target / 'pages.jsonl').exists():
        raise FileExistsError('index already exists; preserve it and choose a new target')
    images = target / 'pages'
    images.mkdir(exist_ok=True)
    doc = pymupdf.open(source)
    records, errors = [], []
    for i, page in enumerate(doc):
        try:
            text = page.get_text(sort=True)
            blocks = [{'bbox_points': [round(v, 2) for v in b[:4]], 'text': b[4]}
                      for b in page.get_text('blocks') if b[6] == 0]
            path = images / f'p{i + 1:03d}.jpg'
            page.get_pixmap(dpi=150, alpha=False).save(path, jpg_quality=92)
            records.append({'id': f'mlcb-p{i+1:03d}', 'page': i+1,
                            'source_id': 'mlcb-20180327-full', 'text': text, 'blocks': blocks,
                            'width_points': page.rect.width, 'height_points': page.rect.height,
                            'image': path.as_posix(), 'image_sha256': sha(path),
                            'references': sorted(references(text)),
                            'embedded_images': len(page.get_images()),
                            'drawing_objects': len(page.get_drawings())})
        except Exception as exc:
            errors.append({'page': i+1, 'error_type': type(exc).__name__, 'error': str(exc)})
    path = target / 'pages.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    manifest = {'source': source.as_posix(), 'source_sha256': sha(source),
                'source_url': 'https://machinelearningcoban.com/ebook/',
                'title': 'Machine Learning cơ bản', 'author': 'Vũ Hữu Tiệp', 'edition': '2018-03-27',
                'physical_pages': len(doc), 'indexed_pages': len(records), 'render_dpi': 150,
                'page_ids': [r['page'] for r in records], 'index_sha256': sha(path),
                'low_text_pages': [r['page'] for r in records if len(r['text'].strip()) < 100],
                'errors': errors, 'unprocessed_pages': [e['page'] for e in errors],
                'coverage_policy': 'Every physical page, complete native text/layout and full-page visual raster; no benchmark-based exclusion.',
                'limitations': ['No OCR or vision-generated captions in discovery index.',
                               'Graphics without searchable text/caption can be opened by page but lexical discovery may fail.',
                               'Native math extraction/reading order can be wrong; visual source is authoritative.',
                               '150 dpi can obscure very small labels; report uncertain values instead of guessing.'],
                'rights': 'Copyright Vũ Hữu Tiệp, physical PDF p9. Existing user permission for corpus processing; no redistribution inferred. Full images/index/PDF stay local.',
                'reference_parser_revision':2,
                'reference_revision_reason':'Dev queries Hinh16.2b and Hinh3.4a need panel-suffix normalization; no test result used. Text and 400 image bytes unchanged.'}
    write(target / 'manifest.json', manifest)
    write(Path('artifacts/bonus/index_manifest.json'), manifest)
    if errors or len(records) != len(doc):
        raise RuntimeError('incomplete index; see manifest, not eligible for bonus')
    return manifest


class PageIndex:
    def __init__(self, path: Path = INDEX / 'pages.jsonl'):
        self.path = path
        self.rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        self.by_page = {r['page']: r for r in self.rows}
        if len(self.by_page) != len(self.rows):
            raise ValueError('duplicate pages in index')
        self.tf = [Counter(tokens(r['text'])) for r in self.rows]
        self.df = Counter(t for row in self.tf for t in row)
        self.lengths = [sum(row.values()) for row in self.tf]
        self.avg = sum(self.lengths) / max(1, len(self.lengths))

    def search(self, query: str, k: int = 8) -> list[dict]:
        query_terms = set(tokens(query))
        refs = references(query)
        scored = []
        n = len(self.rows)
        for row, counts, length in zip(self.rows, self.tf, self.lengths):
            score = sum(math.log(1 + (n-self.df[t]+0.5)/(self.df[t]+0.5)) *
                        counts[t] * 2.5 / (counts[t] + 1.5*(0.25+0.75*length/max(1,self.avg)))
                        for t in query_terms if counts[t])
            matched = refs & set(row['references'])
            if matched:
                score += 35 * len(matched)
                # A caption at the start of a native line is stronger than a cross-reference.
                for kind, number in (r.split(':') for r in matched):
                    if re.search(r'(?m)^\s*' + kind + r'\s*' + re.escape(number) + r'\s*:', normalize(row['text'])):
                        score += 60
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda x: (-x[0], x[1]['page']))
        return [{'rank': i+1, 'score': round(s, 6), 'page': r['page'], 'id': r['id'],
                 'references': r['references'], 'text': r['text'][:2400]}
                for i, (s, r) in enumerate(scored[:k])]

    def verify(self) -> dict:
        missing = [r['page'] for r in self.rows if not Path(r['image']).exists() or
                   sha(Path(r['image'])) != r['image_sha256']]
        return {'pages': len(self.rows), 'image_hash_failures': missing,
                'continuous_400': sorted(self.by_page) == list(range(1, 401))}

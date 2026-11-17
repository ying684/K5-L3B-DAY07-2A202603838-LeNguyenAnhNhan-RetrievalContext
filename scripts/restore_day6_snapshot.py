"""Restore copied real vectors into an EMPTY, separate local day7_lab database.

No upsert/delete/truncate. Refuse Day 6 ports/names or an existing chunk table.
The application's retrieval adapter remains read-only after restore.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import load_environment
from src.day6 import cached_vector, inspect_snapshot, load_chunks


def validate_target(url):
    target=urlsplit(url)
    if (target.scheme not in ('postgres','postgresql')
            or target.hostname not in ('127.0.0.1','localhost')
            or target.port!=55434 or target.path!='/day7_lab'):
        raise ValueError('restore target must be separate local day7_lab at port 55434; Day 6 is never a target')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path('data/day6'))
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(); load_environment()
    snapshot=inspect_snapshot(args.data_dir)
    chunks=load_chunks(args.data_dir)
    vectors={c['id']:cached_vector(args.data_dir,c['text'],'baai/bge-m3',1024) for c in chunks}
    if args.dry_run:
        print(json.dumps({'dry_run':True,'validated_chunks':len(chunks),'validated_vectors':len(vectors),
                          'vector_fingerprint':snapshot['vector_fingerprint'],'database_writes':0}))
        return
    target=os.getenv('DAY7_RESTORE_DATABASE_URL','')
    validate_target(target)
    import psycopg
    from psycopg.types.json import Jsonb
    with psycopg.connect(target,connect_timeout=5) as connection:
        if connection.execute('SELECT current_database()').fetchone()[0]!='day7_lab':
            raise ValueError('wrong target database')
        if connection.execute("SELECT to_regclass('public.day6_chunks')").fetchone()[0] is not None:
            raise ValueError('target chunk table already exists; refusing to overwrite it')
        connection.execute('CREATE EXTENSION IF NOT EXISTS vector')
        connection.execute('''CREATE TABLE day6_chunks (
            strategy text NOT NULL, chunk_id text NOT NULL, document_id text NOT NULL,
            source text NOT NULL, page integer NOT NULL, text text NOT NULL,
            metadata jsonb NOT NULL, embedding vector(1024) NOT NULL,
            PRIMARY KEY(strategy,chunk_id))''')
        records=[('B',c['id'],c['source_id'],c['source'],c['page'],c['text'],
                  Jsonb(c['metadata']),json.dumps(vectors[c['id']])) for c in chunks]
        with connection.cursor() as cursor:
            cursor.executemany('INSERT INTO day6_chunks VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)',records)
        count=connection.execute('SELECT count(*) FROM day6_chunks').fetchone()[0]
        if count!=len(chunks):
            raise ValueError('restored chunk count does not match the snapshot')
    print(json.dumps({'restored_chunks':count,'strategy':'B','target':'local day7_lab:55434',
                      'source_database_modified':False}))


if __name__=='__main__':
    main()

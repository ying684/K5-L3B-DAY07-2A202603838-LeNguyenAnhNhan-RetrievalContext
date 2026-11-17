"""Offline submission audit. No API calls and no automatic semantic grading."""
from pathlib import Path
from collections import Counter
from importlib.metadata import version
from datetime import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
from src.config import load_environment
from src.experiments import runtime_fingerprint
from src.bonus.experiment import snapshot, validate_benchmark
from src.bonus.index import PageIndex

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(data): return hashlib.sha256(data).hexdigest()
errors=[]
def check(ok, reason):
    if not ok: errors.append(reason)
def ignored(p):
    return subprocess.run(['git','check-ignore','--no-index','-q',p],capture_output=True).returncode==0

with zipfile.ZipFile('artifacts/frozen_runtime.zip') as z:
    inventory=json.loads(z.read('INVENTORY.json'))
    check(z.testzip() is None and all(sha(z.read(n))==h for n,h in inventory.items()), 'Frozen archive integrity')
    check(not any(n.endswith(('.pdf','.env','.key','.pem')) for n in inventory), 'Private file in frozen archive')
    basic_paths=sorted(n for n in inventory if re.fullmatch(r'src/[^/]+\.py',n))
    basic_paths+=sorted(n for n in inventory if re.fullmatch(r'configs/[^/]+\.json',n))
    basic_paths+=['requirements.lock.txt','artifacts/protocol.json']
    frozen_basic=sha(json.dumps([(n,inventory[n]) for n in basic_paths],sort_keys=True).encode())
    check(frozen_basic==read('artifacts/selection.json')['runtime_sha256'], 'Measured basic fingerprint')
    frozen_bonus=read('artifacts/bonus/selection.json')['snapshot']
    for n,h in frozen_bonus['files'].items():
        actual=inventory.get(n) if n in inventory else sha(Path(n).read_bytes())
        check(actual==h,'Measured bonus input: '+n)
    for n,h in inventory.items():
        if n.startswith(('src/','configs/','data/')) and n not in ('src/experiments.py','src/bonus/experiment.py'):
            check(Path(n).is_file() and sha(Path(n).read_bytes())==h,'Unplanned runtime change: '+n)
    for name in ['src/experiments.py','src/bonus/experiment.py']:
        original=z.read(name).decode('utf-8').replace('\r\n','\n')
        if name=='src/experiments.py':
            migrated=original.replace("root / 'requirements.lock.txt'","root / 'requirements.txt'")
        else:
            migrated=original.replace("Path('requirements.lock.txt')","Path('requirements.txt')").replace("Path('requirements-bonus.txt'), ",'')
        check(Path(name).read_text(encoding='utf-8')==migrated,'Only dependency-path migration permitted: '+name)

deps={}
for line in Path('requirements.txt').read_text(encoding='utf-8').splitlines():
    if not line.strip() or line.startswith('#'): continue
    package,pin=line.split('=='); package=package.split('[')[0]
    deps[package]=version(package)
    check(deps[package]==pin,'Installed dependency mismatch: '+package)
check(version('psycopg-binary')=='3.3.8','Binary adapter version')
check(not Path('requirements.lock.txt').exists() and not Path('requirements-bonus.txt').exists(),'Requirements not consolidated')

basic=[]
for p in sorted(Path('artifacts/basic_runs').glob('*.json')):
    x=read(p); m=x['manifest']; reviews=x['reviews']; traces=x['traces']; s=x['summary']
    expected=17 if m['split']=='dev' else 7
    check(len(traces)==len(reviews)==s['agent']['cases']==expected,'Basic counts: '+p.name)
    check({r['query_id'] for r in reviews}=={t['query_id'] for t in traces},'Basic IDs: '+p.name)
    check(all(r.get('rationale') and r.get('reviewer') for r in reviews),'Basic rationale: '+p.name)
    check(m['runtime_sha256']==frozen_basic,'Basic measured runtime: '+p.name)
    check(sha(Path(m['cases_path']).read_bytes())==m['benchmark_sha256'],'Basic benchmark: '+p.name)
    # No full retrieved source text or message body should survive the public export.
    def source_strings(v):
        if isinstance(v,list): return any(source_strings(i) for i in v)
        if isinstance(v,dict): return any((k in ('text','content') and isinstance(a,str)) or source_strings(a) for k,a in v.items())
        return False
    check(not source_strings(traces),'Unredacted text in export: '+p.name)
    raw=Path(x['context_policy']['raw_local'])/'traces.jsonl'
    if raw.exists(): check(sha(raw.read_bytes())==x['context_policy']['raw_trace_sha256'],'Raw trace changed: '+p.name)
    basic.append({'config':m['config']['id'],'split':m['split'],'N':expected})
check(len(basic)==7 and sum(r['N'] for r in basic)==99,'Seven basic runs / 99 records')

bonus=[]
for p in sorted(Path('artifacts/bonus/runs').iterdir()):
    if not p.is_dir(): continue
    m,s,rs,rv=[read(p/n) for n in ['manifest.json','summary.json','results.json','review.json']]
    gold=read('data/bonus/benchmark/'+m['split']+'_gold.json')
    check({g['id'] for g in gold}=={r['id'] for r in rs}=={r['id'] for r in rv},'Bonus IDs: '+p.name)
    check(len(rs)==len(rv)==len(gold)==s['N'] and sum(r['passed'] for r in rv)==s['passed'],'Bonus counts: '+p.name)
    check(all(r.get('rationale') and r.get('reviewer') for r in rv),'Bonus rationale: '+p.name)
    check(sum(r['status']=='error' for r in rs)==s['final_errors_or_missing'],'Bonus error count: '+p.name)
    check(m['snapshot']==frozen_bonus,'Bonus measured runtime: '+p.name)
    for group,counts in s['groups'].items():
        ids={g['id'] for g in gold if ('unanswerable' if not g['answerable'] else g['group'])==group}
        check(len(ids)==counts['N'] and sum(r['passed'] for r in rv if r['id'] in ids)==counts['passed'],'Bonus group: '+p.name+'/'+group)
    bonus.append({'config':s['config_id'],'split':s['split'],'N':s['N'],'passed':s['passed']})
check(len(bonus)==4 and [r['passed'] for r in bonus if r['split']=='test']==[21],'Four bonus runs / 21 test passes')
coverage=PageIndex().verify()
check(coverage['continuous_400'] and not coverage['image_hash_failures'],'Local 400-page index')
benchmark=validate_benchmark()

log=Path('artifacts/code_tests.log').read_text(encoding='utf-8-sig')
count=re.search(r'Ran (\d+) tests?',log)
check(bool(count) and int(count.group(1))==65 and bool(re.search(r'^OK\s*$',log,re.M)) and 'skipped=' not in log,'Offline test log')

private=['.env','data/day6/raw/book_ML_color.pdf','data/bonus/index/pages.jsonl','results/runs/example/traces.jsonl',
         '.archive/example.zip','artifacts/new_private.json','artifacts/bonus/new_probe.json','artifacts/bonus/runs/new_test/results.json',
         'data/unapproved.json','specs/example/plan.md']
ignore_results={p:ignored(p) for p in private}
check(all(ignore_results.values()),'Private/generated paths not ignored')
public=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],capture_output=True,check=True).stdout.decode('utf-8').split('\0')
public=[n for n in public if n and Path(n).is_file()]
check(all(not n.startswith(('results/','.archive/','data/day6/','data/bonus/index/')) and n!='.env' for n in public),'Private file already tracked')
load_environment()
keys=[v.encode() for k,v in os.environ.items() if (k.endswith(('_API_KEY','_TOKEN','_SECRET','_PASSWORD')) or k=='DATABASE_URL') and len(v)>15]
suspects=[]
def scan(name,data):
    if any(k in data for k in keys) or re.search(rb'sk-(?:or-v1-|proj-)?[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{30,}',data): suspects.append(name)
for name in public:
    scan(name,Path(name).read_bytes())
    if name=='artifacts/frozen_runtime.zip':
        with zipfile.ZipFile(name) as z:
            for member in z.namelist(): scan(name+'!'+member,z.read(member))
check(not suspects,'Potential credential in publishable files')

out=Path('artifacts/verification.json')
if not out.exists(): out.write_text('{}\n',encoding='utf-8')
broken=[]
markdown=[p for n in public if (p:=Path(n)).suffix=='.md']
for p in markdown:
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
        if not target.startswith(('http://','https://','#')) and not (p.parent/target.split('#')[0]).exists(): broken.append({'file':str(p),'target':target})
check(not broken,'Broken local links')
diff=subprocess.run(['git','diff','--check'],capture_output=True)
check(diff.returncode==0,'git diff --check')
report=Path('report/individual/2A202603838_LeNguyenAnhNhan.md')
check(not any(Path(n).exists() for n in ['AGENTS.md','specs','.specify','docs/NEXT_CHAT_HANDOFF.md']),'Unneeded planning/checkpoint files remain')
result={'checked_at':datetime.now().astimezone().isoformat(),'status':'passed' if not errors else 'failed',
        'scope':'Offline packaging/evidence checks, no API or semantic rescoring','tests_passed':65,
        'dependencies':deps,'basic_runs':basic,'bonus_runs':bonus,'bonus_test':'21/30; not qualified',
        'frozen_runtime_zip_sha256':sha(Path('artifacts/frozen_runtime.zip').read_bytes()),
        'measured_basic_fingerprint':frozen_basic,'measured_bonus_fingerprint':frozen_bonus['fingerprint'],
        'current_basic_packaging_fingerprint':runtime_fingerprint(),'current_bonus_packaging_fingerprint':snapshot()['fingerprint'],
        'packaging_changed':True,'model_rerun':False,'historical_selection_labels_unchanged':True,
        'index':coverage,'benchmark':benchmark,'ignore_checks':ignore_results,'public_files_scanned':len(public),
        'potential_secret_files':suspects,'broken_local_links':broken,'git_diff_check_ok':diff.returncode==0,
        'single_report':str(report),'report_bytes':report.stat().st_size,
        'remaining':['Learner source review','Grader repository/data access','Push and submit link','Bonus threshold not met'],
        'errors':errors}
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['status','tests_passed','public_files_scanned','report_bytes','potential_secret_files','broken_local_links','errors']},ensure_ascii=False,indent=2))
raise SystemExit(bool(errors))

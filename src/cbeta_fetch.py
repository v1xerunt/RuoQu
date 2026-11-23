"""CBETA work metadata (cbdata.dila.edu.tw API: byline, time_dynasty, time_from, time_to) for the undated Buddhist
texts, to tell texts that CBETA dates (external check of the model) from texts nobody dates (our contribution).
Kanripo source field -> CBETA work id: 'Taisho Tripitaka Vol. 14, No. 499' -> T0499, Xuzangjing -> X, 藏外 -> ZW,
Fangshan -> F, Jiaxing -> J, Zhonghua -> C, Koreana -> K, Songzang yizhen -> S, Fojiao Canon -> B.
One request per second. Output: raw/cbeta/works.jsonl, data/cbeta_dates.tsv
Usage: python cbeta_fetch.py
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, re, json, time, urllib.request
import pandas as pd

D = (WORK + '/data'); OUT = (WORK + '/raw/cbeta')
P = (PROJECT + '/data')
CANON = {'Taisho Tripitaka': 'T', 'Xuzangjing': 'X', 'not contained': 'ZW', 'Fangshan': 'F', 'Jiaxing': 'J', 'Zhonghua': 'C',
         'Koreana': 'K', 'Songzang yizhen': 'S', 'Fojiao Canon': 'B'}


def work_ids(src):
    src = str(src)
    c = next((v for k, v in CANON.items() if k in src), None)
    m = re.search(r'No\. ([A-Za-z]?)(\d+)([a-z]?)', src)
    if not c or not m: return []
    pre, num, suf = m.groups()
    return [f'{c}{pre}{num.zfill(4 - len(pre))}{suf}', f'{c}{pre}{num.zfill(3)}{suf}', f'{c}{pre}{num}{suf}']


def get(work):
    url = f'https://cbdata.dila.edu.tw/stable/works?work={work}'
    for k in range(3):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=60).read().decode('utf8'))
        except Exception:
            time.sleep(5 * (k + 1))
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    u = pd.read_csv(f'{D}/corpus3_f0/undated.tsv', sep='\t')
    u = u[u.cat.str.startswith('KR6')]
    src = pd.read_csv(f'{D}/meta.tsv', sep='\t').set_index('id').source
    path = f'{OUT}/works.jsonl'
    done = {json.loads(l)['id'] for l in open(path)} if os.path.exists(path) else set()
    with open(path, 'a') as f:
        for n, i in enumerate(u.id):
            if i in done: continue
            rec = None
            for w in dict.fromkeys(work_ids(src.get(i))):
                r = get(w); time.sleep(1.0)
                if r and r.get('num_found'):
                    rec = r['results'][0]; break
            f.write(json.dumps(dict(id=i, work=rec and rec.get('work'), rec=rec), ensure_ascii=False) + '\n'); f.flush()
            if n % 100 == 0: print(n, i, rec and rec.get('work'), rec and rec.get('time_dynasty'), flush=True)
    rows = []
    for l in open(path):
        r = json.loads(l); x = r['rec'] or {}
        rows.append(dict(id=r['id'], work=r['work'], byline=x.get('byline'), dynasty=x.get('time_dynasty'),
                         c_from=x.get('time_from'), c_to=x.get('time_to'), creators=x.get('creators')))
    df = pd.DataFrame(rows)
    df.to_csv(f'{P}/cbeta_dates.tsv', sep='\t', index=False)
    print(len(df), 'texts;', int(df.work.notna().sum()), 'found in CBETA;', int(df.c_from.notna().sum()), 'with years;',
          int((df.dynasty == 'unknown').sum()), 'dynasty unknown')
    print('done')


if __name__ == '__main__':
    main()

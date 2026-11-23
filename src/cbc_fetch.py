"""Polite download of the Chinese Buddhist Canon database (Radich & Norrish, dazangthings.nz/cbc) entries for the
Taishō texts in our corpus: the attribution assertions serve as external labels for the systematic attribution test.
One request every 1.5 s. Raw HTML only (parsed later by cbc_parse.py). Usage: python cbc_fetch.py
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, re, time, json, urllib.request
import pandas as pd

OUT = (WORK + '/raw/cbc_dl')
D = (WORK + '/data')
UA = {'User-Agent': 'research (classical Chinese dating project; low-rate fetch)'}


def get(url):
    for k in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode('utf8')
        except Exception as e:
            time.sleep(5 * (k + 1))
    return None


def main():
    os.makedirs(f'{OUT}/text', exist_ok=True)
    idx_path = f'{OUT}/index.json'
    if os.path.exists(idx_path):
        index = json.load(open(idx_path))
    else:
        index = {}
        for page in range(1, 20):
            h = get(f'https://dazangthings.nz/cbc/text/?results_per_page=500&page={page}')
            if not h: break
            links = re.findall(r'href="/cbc/text/(\d+)/">([^<]*)</a>', h)
            if not links: break
            for i, lab in links:
                m = re.match(r'\s*(T\d{4}[A-Za-z]?)', lab)
                if m: index.setdefault(m.group(1), i)
            print('page', page, len(links), 'links;', len(index), 'Taisho ids', flush=True)
            time.sleep(1.5)
        json.dump(index, open(idx_path, 'w'))
    tmap = dict(re.findall(r':KR_ID: (KR6\w+)\n:CUSTOM_ID: (T\d+n\d+\w?)',
                           open((WORK + '/raw/KR-Catalog/buddhist/taisho.org'), encoding='utf8').read()))
    ours = set(pd.concat([pd.read_csv(f'{D}/corpus3_f{f}/heldout.tsv', sep='\t', usecols=['id']) for f in range(5)]).id)
    want = []
    for kr, t in tmap.items():
        if kr not in ours: continue
        tn = re.sub(r'^T\d+n', 'T', t)
        tn = 'T' + tn[1:].zfill(4) if re.fullmatch(r'T\d+', tn) else 'T' + re.sub(r'\D', '', tn).zfill(4) + re.sub(r'[\d]', '', tn[1:])
        if tn in index: want.append((kr, tn, index[tn]))
    print(len(want), 'corpus texts with a CBC entry', flush=True)
    json.dump(want, open(f'{OUT}/want.json', 'w'))
    for n, (kr, tn, cid) in enumerate(want):
        p = f'{OUT}/text/{cid}.html'
        if os.path.exists(p): continue
        h = get(f'https://dazangthings.nz/cbc/text/{cid}/')
        if h: open(p, 'w').write(h)
        if n % 100 == 0: print(n, kr, tn, flush=True)
        time.sleep(1.5)
    print('done')


if __name__ == '__main__':
    main()

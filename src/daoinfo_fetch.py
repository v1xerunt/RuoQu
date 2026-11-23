"""Polite download of the 道教文化中心資料庫 (zh.daoinfo.org) entry for every Daoist-canon text in the corpus (KR5),
to obtain scholarly date statements (largely after 任繼愈《道藏提要》) as external labels for dating the undated
Daoist texts. One request every 1.5 s; raw first paragraphs saved to raw/daoinfo/entries.jsonl.
Usage: python daoinfo_fetch.py
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, re, json, time, html, urllib.parse, urllib.request
import pandas as pd

OUT = (WORK + '/raw/daoinfo')
D = (WORK + '/data')
UA = {'User-Agent': 'research (classical Chinese dating project; low-rate fetch)'}


def get(title):
    url = 'https://zh.daoinfo.org/index.php?title=' + urllib.parse.quote(title) + '&variant=zh-hant'
    for k in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode('utf8')
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(5 * (k + 1))
        except Exception:
            time.sleep(5 * (k + 1))
    return None


def first_para(h):
    body = h[h.find('mw-parser-output'):] if 'mw-parser-output' in h else h
    ps = re.findall(r'<p>(.*?)</p>', body, re.S)
    txt = [re.sub(r'\s+', '', html.unescape(re.sub(r'<[^>]+>', '', p))) for p in ps]
    return ' '.join(t for t in txt[:3] if t)


def main():
    os.makedirs(OUT, exist_ok=True)
    m = pd.read_csv(f'{D}/meta.tsv', sep='\t')
    m = m[m.cat.str.startswith('KR5')]
    done = set()
    path = f'{OUT}/entries.jsonl'
    if os.path.exists(path):
        done = {json.loads(l)['id'] for l in open(path)}
    with open(path, 'a') as f:
        for n, r in enumerate(m.itertuples()):
            if r.id in done: continue
            t = re.sub(r'\(.*?\)|（.*?）', '', str(r.title)).strip()
            h = get(t)
            txt = first_para(h) if h and '沒有找到' not in h[:5000] and 'noarticletext' not in h else ''
            f.write(json.dumps(dict(id=r.id, title=r.title, query=t, text=txt[:1500]), ensure_ascii=False) + '\n'); f.flush()
            if n % 100 == 0: print(n, r.id, t, len(txt), flush=True)
            time.sleep(1.5)
    print('done')


if __name__ == '__main__':
    main()

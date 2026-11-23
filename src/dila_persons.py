"""Life dates of the CBETA creators of the undated texts (DILA person authority, authority.dila.edu.tw), so that a text
whose author is datable counts as externally dated, and serves as one more validation set.
Modern editors (byline segment ending in 整理, 校錄, 錄文, 點校, 校注, 編輯, 譯註; or born after 1850) are skipped: they date the
edition, not the text. Indian authors (名 with 菩薩, or a segment ending in 造 by a person without DILA dates) are skipped too:
their dates do not date the Chinese text.
One request per second. Output: raw/cbeta/persons.jsonl, data/creator_dates.tsv (id, creator, born, died, a_lo, a_hi)
a_lo/a_hi: born + 20 .. died (or the reported floruit when only one end is known: ±30 y).
Usage: python dila_persons.py
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, re, json, time, urllib.request
import pandas as pd

OUT = (WORK + '/raw/cbeta'); P = (PROJECT + '/data')
EDITOR = re.compile(r'整理|校錄|錄校|錄文|點校|校注|校點|編輯|譯註|校勘|輯校')


def year(s):
    m = re.match(r'([+-])(\d{4})', str(s or ''))
    return (int(m.group(2)) * (1 if m.group(1) == '+' else -1)) if m else None


def get(pid):
    url = f'https://authority.dila.edu.tw/webwidget/getAuthorityData.php?type=person&id={pid}&jsoncallback='
    for k in range(3):
        try:
            t = urllib.request.urlopen(url, timeout=60).read().decode('utf8').strip()
            t = t[t.find('{'):t.rfind('}') + 1]
            return json.loads(t).get('data1')
        except Exception:
            time.sleep(5 * (k + 1))
    return None


def main():
    works = [json.loads(l) for l in open(f'{OUT}/works.jsonl')]
    need = {}
    for w in works:
        x = w['rec'] or {}
        cw, bl = x.get('creators_with_id') or '', x.get('byline') or ''
        for c in cw.split(';'):
            m = re.match(r'(.+?)\((A\d+)\)', c)
            if not m: continue
            name, pid = m.groups()
            seg = next((g for g in re.split(r'[\s　]+', bl) if name in g), '')  # '大內文雄．齊藤隆信整理' -> role of the whole segment
            if EDITOR.search(seg[-4:]) or '菩薩' in name or '菩薩' in seg: continue
            need.setdefault(pid, name); w.setdefault('people', []).append((name, pid))
    path = f'{OUT}/persons.jsonl'
    done = {json.loads(l)['pid'] for l in open(path)} if os.path.exists(path) else set()
    with open(path, 'a') as f:
        for k, pid in enumerate(need):
            if pid in done: continue
            d = get(pid); time.sleep(1.0)
            f.write(json.dumps(dict(pid=pid, data=d), ensure_ascii=False) + '\n'); f.flush()
            if k % 100 == 0: print(k, pid, need[pid], flush=True)
    per = {}
    for l in open(path):
        r = json.loads(l); d = r['data'] or {}
        per[r['pid']] = (year(d.get('bornDateBegin')), year(d.get('diedDateEnd') or d.get('diedDateBegin')))
    rows = []
    for w in works:
        for name, pid in w.get('people', []):
            b, dd = per.get(pid, (None, None))
            if b is not None and b > 1850: continue  # modern editor or translator
            if b is not None and dd is not None: lo, hi = b + 20, dd
            elif dd is not None: lo, hi = dd - 30, dd
            elif b is not None: lo, hi = b + 20, b + 70
            else: lo = hi = None
            rows.append(dict(id=w['id'], creator=name, pid=pid, born=b, died=dd, a_lo=lo, a_hi=hi))
    df = pd.DataFrame(rows)
    df.to_csv(f'{P}/creator_dates.tsv', sep='\t', index=False)
    print(len(df), 'creator rows;', df.id.nunique(), 'texts;', int(df.a_lo.notna().sum()), 'rows with life dates')
    print('done')


if __name__ == '__main__':
    main()

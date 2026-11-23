"""Chapter-level segmentation of composite classics (stratigraphy experiment).

Headings: org headings in TLS/CHANT editions ('** ...'), otherwise short lines '<title>第N' (one per line;
table-of-contents lines listing several titles are skipped), plus custom patterns (Shanhaijing).
Usage: python chapters.py list            -> print chapter titles per work
       python chapters.py build            -> write corpus2_f{k}/chapters.{bin,tsv} (fold where the work is held out)
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, re, sys, json, zipfile
import numpy as np
import pandas as pd
from extract import clean_juan, juan_key

RAW = (WORK + '/raw')
D = (WORK + '/data')
V = os.environ.get('LETTER_V', '2')
WORKS = {  # id: custom heading regex (None = generic)
    'KR3e0001': None,   # 黃帝內經素問
    'KR1a0001': None,   # 周易 (tls; split into 經/傳 by markers below)
    'KR5c0051': None,   # 莊子 (南華真經)
    'KR3j0002': None,   # 墨子
    'KR3c0001': None,   # 管子
    'KR1h0004': None,   # 論語
    'KR3a0002': None,   # 荀子
    'KR3c0004': None,   # 商君書
    'KR1d0052': None,   # 禮記
    'KR1d0001': None,   # 周禮
    'KR3l0090': r'^(?:[南西北東中](?:次[一二三四五六七八九十]+)?山經|海(?:外|內)[南西北東]經|大荒[南西北東]經|海內經)',  # 山海經
    'KR2d0001': None,   # 逸周書
    'KR3c0005': None,   # 韓非子
    'KR5c0125': None,   # 列子 (blind)
}
NUM = '一二三四五六七八九十百'
GEN = re.compile(rf'^[　 ]*([一-鿿]{{1,14}}?)(?:篇)?第[{NUM}]+')


def raw_lines(zid):
    z = zipfile.ZipFile(f'{RAW}/krzip/{zid}.zip')
    names = sorted([n for n in z.namelist() if re.search(r'_\d+[a-z]?\.txt$', n)], key=juan_key)
    for n in names:
        t = z.read(n).decode('utf8', 'replace')
        mj = re.search(r'#\+PROPERTY: JUAN\s+(.*)', t)
        if juan_key(n)[0] == 0 or (mj and '提要' in mj.group(1)): continue
        for l in t.split('\n'):
            yield l


def segment(zid, pat=None):
    chapters, cur_title, buf = [], None, []
    cust = re.compile(pat) if pat else None
    for l in raw_lines(zid):
        s = re.sub(r'<pb:[^>]*>|¶', '', l).strip()
        title = None
        if l.startswith('**') and not l.startswith('***'):
            title = re.sub(r'^\*+\s*\d*\s*', '', s).strip('《》 ')
        elif not l.startswith('#') and not l.startswith('*'):
            core = re.sub(r'\([^)]*\)', '', s)
            if cust is not None:
                m = cust.match(core.strip('　 '))
                if m and len(core) < 20: title = m.group(0)
            elif len(core) < 40 and core.count('第') == 1 and '卷' not in core:
                m = GEN.match(core)
                if m: title = core.strip('　 ')
        if title:
            if cur_title is not None and buf: chapters.append((cur_title, '\n'.join(buf)))
            cur_title, buf = title, []
        elif cur_title is not None:
            buf.append(l)
    if cur_title is not None and buf: chapters.append((cur_title, '\n'.join(buf)))
    out = []
    for t, b in chapters:
        txt = clean_juan(b)
        if len(txt) >= 100: out.append((t, txt))
    return out


def zhouyi_split(chs):
    """周易: split each hexagram chapter into 經 (卦爻辭) and 傳 (彖/象/文言); 繫辭等 are 傳."""
    out = []
    for t, txt in chs:
        if re.search(r'繫辭|說卦|序卦|雜卦', t):
            out.append((t, '傳', txt)); continue
        # positions of 彖曰/象曰/文言曰 start commentary spans; 經 = text outside them up to the next 爻 line
        parts = re.split(r'(彖曰|象曰|文言曰)', txt)
        jing, zhuan = [parts[0]], []
        for k in range(1, len(parts), 2):
            seg = parts[k] + parts[k + 1]
            # commentary runs until the next line that starts a line statement (初/九/六/上/用)
            m = re.search(r'(?<=[也矣焉])(?=(?:初[九六]|[九六][二三四五]|上[九六]|用[九六]))', seg)
            if m and parts[k] != '文言曰':
                zhuan.append(seg[:m.start()]); jing.append(seg[m.start():])
            else:
                zhuan.append(seg)
        out.append((t, '經', ''.join(jing))); out.append((t, '傳', ''.join(zhuan)))
    return out


def build():
    vm = json.load(open(f'{D}/varmap{V}.json'))
    folds = {}
    for f in range(5):
        h = pd.read_csv(f'{D}/corpus{V}_f{f}/heldout.tsv', sep='\t')
        for i in h.id: folds[i] = f
    vocab = json.load(open(f'{D}/corpus{V}_f0/vocab.json')); c2i = {c: k for k, c in enumerate(vocab)}
    meta = pd.read_csv(f'{D}/meta.tsv', sep='\t').set_index('id')
    per_fold = {f: [] for f in range(5)}
    for zid, pat in WORKS.items():
        f = folds.get(zid, 0)  # blind works (excluded from all training) are scored by fold 0
        chs = segment(zid, pat)
        rows = zhouyi_split(chs) if zid == 'KR1a0001' else [(t, '', x) for t, x in chs]
        for k, (t, part, x) in enumerate(rows):
            tok = np.array([c2i.get(vm.get(c, c), 2) for c in x], np.uint16)
            per_fold[f].append(dict(id=f'{zid}#{k:03d}', work=zid, title=meta.loc[zid, 'title'], chapter=t, part=part,
                                    n_raw=len(tok), lo=np.nan, hi=np.nan, genre=0, tok=tok))
        print(zid, meta.loc[zid, 'title'], 'fold', f, len(rows), 'chapters', flush=True)
    for f, rows in per_fold.items():
        out = f'{D}/corpus{V}_f{f}'
        arrs, off = [], 0
        for r in rows:
            t = np.concatenate([[1], r.pop('tok')]).astype(np.uint16)
            r['offset'] = off; r['length'] = len(t); off += len(t); arrs.append(t)
        if rows:
            np.concatenate(arrs).tofile(f'{out}/chapters.bin')
            pd.DataFrame(rows).to_csv(f'{out}/chapters.tsv', sep='\t', index=False)


if __name__ == '__main__':
    if sys.argv[1] == 'list':
        for zid, pat in WORKS.items():
            chs = segment(zid, pat)
            print(zid, len(chs), ' | '.join(f'{k}:{t[:10]}({len(x)})' for k, (t, x) in enumerate(chs)))
            print()
    else:
        build()

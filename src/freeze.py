"""Freezing of written Classical Chinese: datability and local rate of change by period and register.

For each out-of-fold held-out text with a narrow label (<= 80 years):
  distinctiveness d = [ll(t) - (ll(t-100) + ll(t+100)) / 2] / n_chars   (per character, at the label midpoint t)
  i.e. how much better the text is explained by its own century than by its neighbours -> local rate of change.
Registers: history (KR2), masters (KR3 ex. fiction), fiction/notes (KR3l), literary collections (KR4b-g),
song lyrics & drama (KR4j), Chan records (KR6q), Buddhist translations, Daoist canon (KR5).
Usage: python freeze.py <prefix> [corpus prefix, default corpus_f]
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, sys
import numpy as np
import pandas as pd

R = (WORK + '/runs')
D = (WORK + '/data')
GRID = np.arange(-1000, 1901, 25).astype(float)


def register(cat, genre):
    if cat == 'KR3l': return '小說筆記'
    if cat == 'KR4j': return '詞曲'
    if cat == 'KR6q': return '禪宗語錄'
    if cat.startswith('KR6'): return '佛典譯經' if genre == 5 else '佛教撰述'
    if cat.startswith('KR5'): return '道教'
    if cat.startswith('KR4'): return '別集'
    if cat.startswith('KR3'): return '子部'
    if cat.startswith('KR2'): return '史部'
    if cat.startswith('KR1'): return '經部'
    return 'other'


def main(prefix, cprefix='corpus_f'):
    rows = []
    for f in range(5):
        run = prefix if f == 0 else f'{prefix}_f{f}'
        p = f'{R}/{run}/scores_heldout_unk.npz'
        if not os.path.exists(p): continue
        z = np.load(p); idx = pd.read_csv(f'{D}/{cprefix}{f}/heldout.tsv', sep='\t').set_index('id')
        meta = pd.DataFrame({'id': z['ids'], 'nt': z['nt']}); ll = z['ll'][:, :len(GRID)].astype(np.float64)
        for i, ii in meta.groupby('id').indices.items():
            r = idx.loc[i]
            if r.hi - r.lo > 80: continue
            t = (r.lo + r.hi) / 2
            if t < -400 or t > 1800: continue
            k = int(np.argmin(np.abs(GRID - t)))
            n = meta.nt.values[ii].sum(); s = ll[ii].sum(0) / max(n, 1)
            d = s[k] - (s[k - 4] + s[k + 4]) / 2
            rows.append(dict(id=i, fold=f, year=t, register=register(r['cat'], r['genre']), d=d, n=n))
    df = pd.DataFrame(rows)
    df['period'] = pd.cut(df.year, [-400, -221, 25, 220, 589, 907, 1127, 1279, 1368, 1644, 1800],
                          labels=['戰國', '秦西漢', '東漢', '魏晉南北朝', '隋唐', '五代北宋', '南宋', '元', '明', '清'])
    pd.set_option('display.width', 220)
    tab = df.pivot_table(index='period', columns='register', values='d', aggfunc='median', observed=True)
    cnt = df.pivot_table(index='period', columns='register', values='d', aggfunc='size', observed=True)
    print('per-character distinctiveness of the own century vs +-100 years (x1000; higher = faster local change)')
    print((tab * 1000).round(1).to_string())
    print('counts'); print(cnt.fillna(0).astype(int).to_string())
    df.to_csv(f'{R}/{prefix}_freeze.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main(*sys.argv[1:])


def offsets(oof_tsv, max_width=100):
    """apparent-date offset (OOF median - label midpoint) by register x period; negative = conservative"""
    t = pd.read_csv(oof_tsv, sep='\t')
    t = t[(t.hi - t.lo) <= max_width].copy()
    t['mid'] = (t.lo + t.hi) / 2; t['offset'] = t['median'] - t.mid
    t['register'] = [register(c, g) for c, g in zip(t.cat, t.genre)]
    # (lo, hi] intervals; the labels name exactly these intervals
    t['period'] = pd.cut(t.mid, [-600, -221, 25, 220, 589, 907, 1127, 1279, 1368, 1644, 1950],
                         labels=['戰國', '秦西漢', '東漢', '魏晉南北朝', '隋唐', '五代北宋', '南宋', '元', '明', '清'])
    off = t.pivot_table(index='period', columns='register', values='offset', aggfunc='median', observed=True)
    err = t.assign(ae=t.offset.abs()).pivot_table(index='period', columns='register', values='ae', aggfunc='median', observed=True)
    n = t.pivot_table(index='period', columns='register', values='offset', aggfunc='size', observed=True)
    pd.set_option('display.width', 250)
    print('median apparent-date offset (years; negative = reads older than it is)')
    print(off.round(0).to_string())
    print('median absolute error (datability)'); print(err.round(0).to_string())
    print('n'); print(n.fillna(0).astype(int).to_string())
    return t

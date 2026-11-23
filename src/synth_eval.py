"""Evaluate the controlled anachronism test (E8a).

Span score at position i (length n) = sum_{j in span} [ max_{c >= t+300} lp_j(c) - lp_j(t) ],
t = true (label-midpoint) date of the passage. Report how often the planted span is the top-ranked
span of its length in the passage (hit@1, hit@3) and its percentile; null = the same position in
the unmodified passage.
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import sys, json, pickle
import numpy as np
import pandas as pd

R = (WORK + '/runs')
GRID = np.arange(-1000, 1901, 25).astype(float)


def main(name):
    cfg = json.load(open(f'{R}/{name}/config.json'))
    C = cfg['corpus']
    T = pickle.load(open(f'{R}/{name}/scores_synth_unk_tok.pkl', 'rb'))
    meta = {m['id']: m for m in json.load(open(f'{C}/synth_meta.json'))}
    idx = pd.read_csv(f'{C}/synth.tsv', sep='\t').set_index('id')
    rows = []
    for e in T:
        m = meta[e['id']]; base = e['id'].rsplit('#', 1)[0]
        lo, hi = idx.loc[e['id'], 'lo'], idx.loc[e['id'], 'hi']
        t = (lo + hi) / 2
        lp = e['lp'][:len(GRID)].astype(np.float64)
        it = int(np.argmin(np.abs(GRID - t)))
        later = GRID >= t + 300
        if not later.any(): continue
        gain = lp[later].max(0) - lp[it]  # per token
        rows.append(dict(id=e['id'], base=base, kind=m['kind'], at=m['at'], n=m['n'], word=m['word'], gain=gain))
    df = pd.DataFrame(rows)
    # reference positions for 'orig': use the positions of the planted late word of the same base passage
    pos = {(r.base, k): (r.at, r.n) for r in df.itertuples() for k in [r.kind] if r.at >= 0}
    out = []
    for r in df.itertuples():
        for kind in (['late', 'ctrl', 'late_func', 'ctrl_func'] if r.kind == 'orig' else [r.kind]):
            if (r.base, kind) not in pos: continue
            at, n = pos[(r.base, kind)]
            g = r.gain
            cs = np.concatenate([[0], np.cumsum(g)]); s = cs[n:] - cs[:-n]
            sc = s[at]
            rank = int((s > sc).sum())
            out.append(dict(kind=kind, planted=r.kind != 'orig', rank=rank, pct=(s <= sc).mean(), score=sc))
    res = pd.DataFrame(out)
    summ = res.groupby(['kind', 'planted']).agg(n=('rank', 'size'), hit1=('rank', lambda x: (x == 0).mean()),
                                                hit3=('rank', lambda x: (x < 3).mean()), pct=('pct', 'mean'),
                                                score=('score', 'median'))
    print(summ.round(3).to_string())
    # AUC planted vs null for each kind
    for kind in ['late', 'late_func', 'ctrl', 'ctrl_func']:
        a = res[(res.kind == kind) & res.planted].score.values; b = res[(res.kind == kind) & ~res.planted].score.values
        if len(a) and len(b):
            auc = (a[:, None] > b[None]).mean() + 0.5 * (a[:, None] == b[None]).mean()
            print(f'{kind}: AUC planted vs unmodified (same position) = {auc:.3f}')
    return res


if __name__ == '__main__':
    main(sys.argv[1])

"""Character-level evidence for the texts flagged by the cross-fitted audit (audit.py, flags fixed beforehand).

For a text flagged 'later than the catalogue', each character gets
    llr_i = log p(x_i | context, c) - log p(x_i | context, a)
with a = catalogue date (label midpoint, floor -600) and c = the model's median date, both on the 100-year token grid.
Reported per text: the top non-overlapping 6-character windows by summed llr (with context), the share of the total
positive evidence carried by those windows, and the function-word share of the evidence. Flagged 'earlier' texts
are treated symmetrically (c = model median < a). Scores: score.py --tokens 4 --tag _audit (each text by its fold model).
Usage: python audit_evidence.py [prefix=l3] [late|early] [max texts]
"""
from paths import WORK, PROJECT  # configurable data locations, see paths.py
import os, sys, json, pickle
import numpy as np
import pandas as pd

R = (WORK + '/runs')
D = (WORK + '/data')
YEARS = np.arange(-1000, 1901, 25)[::4].astype(float)
W, TOP, CTX = 6, 6, 5


def yi(y):
    return int(np.argmin(np.abs(YEARS - max(y, -600))))


def main(prefix='l3', side='late', maxn=200):
    sys.path.insert(0, os.path.dirname(__file__))
    from views import FUNC
    vocab = json.load(open(f'{D}/corpus3_f0/vocab.json')) + ['X']
    isf = np.array([c in FUNC for c in vocab])
    a = pd.read_csv(f'{R}/{prefix}_audit.tsv', sep='\t').set_index('id')
    a = a[a.flag_late] if side == 'late' else a[a.flag_early]
    a = a.sort_values('mass_later' if side == 'late' else 'mass_earlier', ascending=False).head(int(maxn))
    tok = {}
    for f in range(5):
        run = prefix if f == 0 else f'{prefix}_f{f}'
        p = f'{R}/{run}/scores_heldout_unk_audit_tok.pkl'
        if os.path.exists(p):
            for e in pickle.load(open(p, 'rb')):
                if e['id'] in a.index: tok.setdefault(e['id'], []).append(e)
    rows = []
    for i, r in a.iterrows():
        if i not in tok: continue
        ia, ic = yi((r.lo + r.hi) / 2), yi(r['median'])
        spans, tot, tf = [], 0.0, 0.0
        for e in sorted(tok[i], key=lambda e: e['pi']):
            t = e['tok'].astype(int); lp = e['lp'][:len(YEARS)].astype(np.float64)
            m = t > 1; llr = np.where(m, lp[ic] - lp[ia], 0.0)
            pos = np.clip(llr, 0, None); tot += pos.sum(); tf += pos[isf[t]].sum()
            if len(t) >= W:
                cs = np.convolve(llr, np.ones(W), 'valid')
                for s in np.argsort(-cs)[:3 * TOP]:
                    spans.append((cs[s], e['pi'], s, t))
        spans.sort(key=lambda x: -x[0])
        chosen, used = [], set()
        for sc, pi, s, t in spans:
            key = [(pi, k) for k in range(s - W, s + W)]
            if any(k in used for k in key): continue
            used.update((pi, k) for k in range(s, s + W))
            dec = lambda u, v: ''.join(vocab[x] for x in t[max(u, 0):v] if x > 1)
            chosen.append((sc, f'{dec(s - CTX, s)}【{dec(s, s + W)}】{dec(s + W, s + W + CTX)}'))
            if len(chosen) == TOP: break
        rows.append(dict(id=i, title=r.title, author=r.author, lo=r.lo, hi=r.hi, median=r['median'], q10=r.q10,
                         n_pass=r.n_pass, func_share=tf / max(tot, 1e-9),
                         top_share=sum(c[0] for c in chosen) / max(tot, 1e-9), spans=' ｜ '.join(c[1] for c in chosen)))
    df = pd.DataFrame(rows)
    df.to_csv(f'{R}/{prefix}_audit_evidence_{side}.tsv', sep='\t', index=False)
    for r in df.itertuples():
        print(f'## {r.id} {r.title}（{r.author}，目录 {r.lo:.0f}~{r.hi:.0f}）模型 {r.median:.0f}（q10 {r.q10:.0f}）'
              f' 虚词占证据 {r.func_share:.2f}，前 {TOP} 段占 {r.top_share:.2f}')
        print('   ' + r.spans)


if __name__ == '__main__':
    main(*sys.argv[1:])
